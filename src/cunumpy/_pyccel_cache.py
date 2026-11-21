"""Compile Pyccel kernel modules once and reuse the builds across processes.

:class:`PyccelCache` builds a module with the ``pyccel compile`` command line
interface, in a subprocess, once. It keys a build on the module source, the
target language, the OpenMP setting, the Pyccel, NumPy and Python versions and
the platform, and stores it under a per-package cache directory. Later
processes, and other MPI ranks, import the stored extension directly; only the
first run after an edit compiles. A new build is imported once in a subprocess
before it is stored, so a build this interpreter cannot load is an error, not a
crash of the process that uses it.

A kernel may import functions from other Python modules, also of other packages
(``from mpiparticles.kernels.boris_pyccel import boris_rotate``): those modules
are built first, with the kernel, and their sources are part of its key (see
:meth:`PyccelCache.dependencies`).

cunumpy needs no Pyccel to import this module: Pyccel is looked up when a
module is compiled, and a missing Pyccel is an :class:`ImportError` there, which
the host-kernel dispatch records like any other failed compilation.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import scope_profiler as sp

from cunumpy._profile_tags import _PROFILE_TAGS

#: Bumped when the layout of a cached build changes (it was an ``epyccel``
#: folder before the ``pyccel compile`` command line was used). Part of the key.
_LAYOUT = "pyccel-cli-1"

#: Top-level modules whose imports are never built with a kernel: the standard
#: library, NumPy and Pyccel (Pyccel knows them).
_NOT_DEPENDENCIES = frozenset(sys.stdlib_module_names) | {"numpy", "pyccel"}

_FALSE = ("0", "false", "no", "off")


def _uses_openmp_runtime(source: str) -> bool:
    """Whether the module calls the OpenMP runtime (thread ids, thread counts).

    Such a module only compiles with OpenMP, so it is built with it even when
    OpenMP is turned off; ``OMP_NUM_THREADS=1`` then runs it on one thread.
    """
    return "pyccel.stdlib.internal.openmp" in source


def _source_path(module: ModuleType) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise FileNotFoundError(f"No source file for {module.__name__}")
    return Path(source)


def _module_file(name: str) -> Path | None:
    """Return the Python source of module `name`, or None if it has none."""
    if name.split(".", 1)[0] in _NOT_DEPENDENCIES:
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _flat_name(name: str) -> str:
    """Return the module name of a dependency in the staging folder (no dots).

    Pyccel inlines an ``@inline`` function only from a top-level module, so every
    dependency is built as one: ``a.b.c`` becomes ``a__b__c``.
    """
    return name.replace(".", "__")


def _dependency_imports(source: str, package: str) -> list[tuple[ast.ImportFrom, str]]:
    """Return the ``from <module> import ...`` statements of `source` that are dependencies.

    Returns each statement with the absolute name of its module; `package` resolves
    relative imports. A dependency is a module with a Python source outside the
    standard library, NumPy and Pyccel.
    """
    found = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        name = node.module
        if node.level:
            try:
                name = importlib.util.resolve_name("." * node.level + name, package)
            except ImportError:
                continue
        if _module_file(name) is not None:
            found.append((node, name))
    return found


def _rewrite_imports(source: str, package: str) -> str:
    """Return `source` with its dependency imports from their flat module names.

    The line numbers do not change, so Pyccel's messages point at the original lines.
    """
    lines = source.splitlines(keepends=True)
    for node, name in reversed(_dependency_imports(source, package)):
        names = ", ".join(
            alias.name + (f" as {alias.asname}" if alias.asname else "")
            for alias in node.names
        )
        first, last = node.lineno - 1, node.end_lineno - 1
        indent = lines[first][: node.col_offset]
        statement = f"{indent}from {_flat_name(name)} import {names}\n"
        lines[first : last + 1] = [statement] + ["\n"] * (last - first)
    return "".join(lines)


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def _extension_path(directory: Path, stem: str) -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _load_extension(directory: Path, stem: str) -> ModuleType | None:
    path = _extension_path(directory, stem)
    if path is not None:
        spec = importlib.util.spec_from_file_location(stem, path)
        if spec is not None and spec.loader is not None:
            extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
            return extension
    return None


#: Imports the extension at ``argv[1]`` as module ``argv[2]``.
_IMPORT_CHECK = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])\n"
    "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
)


def _check_importable(directory: Path, stem: str) -> None:
    """Import a new build in a subprocess; raise if this interpreter cannot.

    A build that crashes the interpreter on import (an extension linked to
    ``libpython`` loaded into a Python with ``libpython`` built in, as with some
    standalone CPython builds, runs two Pythons in one process) would otherwise
    crash every process that loads it from the cache.
    """
    path = _extension_path(directory, stem)
    if path is None:
        raise ImportError(f"pyccel compile wrote no extension module for {stem}")
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK, str(path), stem],
        capture_output=True,
        text=True,
        check=False,
        cwd=directory,
    )
    if result.returncode == 0:
        return
    how = (
        f"crashed with signal {-result.returncode}"
        if result.returncode < 0
        else f"failed ({result.returncode})"
    )
    hint = (
        "\nAn extension linked to libpython crashes a Python that has libpython "
        "built in; `otool -L` (macOS) or `ldd` shows what it links."
        if result.returncode < 0
        else ""
    )
    raise RuntimeError(
        f"importing the new build of {stem} with {sys.executable} {how}; it is "
        f"not stored:\n{result.stdout}{result.stderr}{hint}"
    )


def _numpy_version() -> str:
    try:
        return importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        return "none"


def _pyccel_version() -> str:
    try:
        return importlib.metadata.version("pyccel")
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("pyccel is not installed") from error


def _pyccel_command() -> list[str]:
    """Return the ``pyccel`` executable of the running environment."""
    beside_python = Path(sys.executable).parent / "pyccel"
    found = str(beside_python) if beside_python.is_file() else shutil.which("pyccel")
    if found is None:
        raise ImportError("the pyccel command line tool is not installed")
    return [found]


class PyccelCache:
    """Compile Pyccel modules on first use and keep the builds on disk.

    One instance per package: builds go to ``~/.cache/<name>/kernels``, or to
    the directory in the environment variable ``<PREFIX>_KERNEL_CACHE``. If that
    directory is not writable (a CI container whose home belongs to another
    user), builds go to ``<tmp>/<name>-kernels`` instead, so the code does not
    fall back to slow uncompiled kernels.

    Modules are built with OpenMP (``--openmp``), so loops marked
    ``#$ omp parallel for`` run on all cores; limit them with
    ``OMP_NUM_THREADS`` (under MPI, the cores per rank). ``<PREFIX>_OPENMP=0``
    builds them serial, except modules that call the OpenMP runtime. The setting
    is part of the key, so serial and OpenMP builds live side by side.

    ``<PREFIX>_FLAGS`` passes extra flags to the C compiler (Pyccel's ``--flags``), such
    as ``-march=native`` for the machine at hand; they are part of the key too. None
    by default: such builds may not run on another CPU, and flags like ``-ffast-math``
    change the rounding.

    Builds are published with an atomic rename: processes that compile the
    same module at the same time (MPI ranks on a cold cache) never see a
    partial build, and the first one to finish wins.

    The instance is callable, so it can be passed as ``compile_host`` to
    :meth:`Kernel.from_folder` and :meth:`KernelCatalog.from_package`.

    Parameters
    ----------
    name : str
        The package name, used for the cache directories.
    env_prefix : str, optional
        Prefix of the environment variables. By default `name` in upper case
        with ``-`` and ``.`` replaced by ``_`` (``"tree-d-3d"`` gives
        ``TREE_D_3D_KERNEL_CACHE``, ``TREE_D_3D_OPENMP`` and ``TREE_D_3D_FLAGS``).
    language : str, optional
        Pyccel's target language. Default ``"c"``.

    Examples
    --------
    >>> cache = xp.kernels.PyccelCache("my-sim")
    >>> cache.cache_env, cache.openmp_env, cache.flags_env
    ('MY_SIM_KERNEL_CACHE', 'MY_SIM_OPENMP', 'MY_SIM_FLAGS')

    In the package's kernel folder::

        kernel = xp.kernels.Kernel.from_folder(__name__, compile_host=cache)
    """

    def __init__(
        self, name: str, *, env_prefix: str | None = None, language: str = "c"
    ) -> None:
        if not name:
            raise ValueError("PyccelCache needs a package name")
        self.name = name
        self.env_prefix = (
            env_prefix
            if env_prefix is not None
            else re.sub(r"[^0-9A-Za-z]", "_", name).upper()
        )
        self.language = language

    @property
    def cache_env(self) -> str:
        """Name of the environment variable that moves the cache directory."""
        return f"{self.env_prefix}_KERNEL_CACHE"

    @property
    def openmp_env(self) -> str:
        """Name of the environment variable that turns OpenMP off (``0``)."""
        return f"{self.env_prefix}_OPENMP"

    @property
    def flags_env(self) -> str:
        """Name of the environment variable with extra C compiler flags."""
        return f"{self.env_prefix}_FLAGS"

    def flags(self) -> str:
        """Return the extra compiler flags of the builds (``<PREFIX>_FLAGS``, empty by default)."""
        return " ".join(os.environ.get(self.flags_env, "").split())

    def cache_root(self) -> Path:
        """Directory holding the builds (``~/.cache/<name>/kernels`` by default)."""
        default = Path.home() / ".cache" / self.name / "kernels"
        return Path(os.environ.get(self.cache_env, default))

    def fallback_root(self) -> Path:
        """Directory for builds when :meth:`cache_root` is not writable."""
        return Path(tempfile.gettempdir()) / f"{self.name}-kernels"

    def openmp_enabled(self) -> bool:
        """Whether modules are built with OpenMP (``<PREFIX>_OPENMP=0`` turns it off)."""
        return os.environ.get(self.openmp_env, "1").lower() not in _FALSE

    def _openmp_for(self, module: ModuleType) -> bool:
        return self.openmp_enabled() or any(
            _uses_openmp_runtime(text)
            for text in (
                _source_path(module).read_text(),
                *self.dependencies(module).values(),
            )
        )

    def dependencies(self, module: ModuleType) -> dict[str, str]:
        """Return the modules `module` imports that are built with it, and their sources.

        A dependency is a module imported with ``from <module> import ...`` (also
        relative, and in other packages) that has a Python source outside the standard
        library, NumPy and Pyccel; its own dependencies count too. They come in build
        order, every module after the ones it imports. The kernel's build contains them
        (Pyccel links them in, and inlines their ``@inline`` functions), and their
        sources are part of its key, so editing one rebuilds the kernel.

        Parameters
        ----------
        module : module
            The kernel module.

        Returns
        -------
        dict of str to str
            The absolute module names and their sources, in build order.

        Raises
        ------
        ImportError
            If the modules import each other in a cycle.
        """
        package = module.__package__ or module.__name__.rpartition(".")[0]
        found: dict[str, str] = {}
        visiting: list[str] = []

        def visit(source: str, package: str) -> None:
            for _, name in _dependency_imports(source, package):
                if name in found:
                    continue
                if name in visiting:
                    cycle = " -> ".join([*visiting, name])
                    raise ImportError(f"kernel modules import each other: {cycle}")
                path = _module_file(name)
                assert path is not None  # a dependency has a source
                text = path.read_text()
                visiting.append(name)
                visit(text, name.rpartition(".")[0])
                visiting.pop()
                found[name] = text

        visit(_source_path(module).read_text(), package)
        return found

    def build_key(self, module: ModuleType, language: str | None = None) -> str:
        """Return the key of `module`'s build, which changes with anything that changes it.

        The module's own source, the language, the OpenMP setting, the extra
        compiler flags (:meth:`flags`), the Pyccel,
        NumPy and Python versions and the platform, and the names and sources of
        the modules built with it (:meth:`dependencies`), so editing one of them
        rebuilds the kernel.

        Parameters
        ----------
        module : module
            The module to build.
        language : str, optional
            Pyccel's target language; by default the instance's.

        Returns
        -------
        str
            20 hexadecimal digits, the suffix of the build's directory name.

        Raises
        ------
        ImportError
            If Pyccel is not installed.
        """
        digest = hashlib.sha256()
        for name, source in self.dependencies(module).items():
            digest.update(f"dependency {name}\n{source}".encode())
        for part in (
            inspect.getsource(module),
            language or self.language,
            _LAYOUT,
            f"openmp={self._openmp_for(module)}",
            f"flags={self.flags()}",
            _pyccel_version(),
            f"numpy={_numpy_version()}",
            sys.version,
            platform.platform(),
        ):
            digest.update(part.encode())
        return digest.hexdigest()[:20]

    @staticmethod
    def _run_pyccel(
        source: Path,
        language: str,
        output: Path,
        openmp: bool,
        staging: Path,
        flags: str = "",
    ) -> None:
        """Compile `source` into `output` with ``pyccel compile``; raise on failure."""
        command = [
            *_pyccel_command(),
            "compile",
            str(source),
            "--language",
            language,
            "--output",
            str(output),
        ]
        if openmp:
            command.append("--openmp")
        if flags:
            command.append(f"--flags={flags}")
        # Pyccel writes lock files (``.lock_acquisition.lock``, ``stc.lock``) to
        # the directory it runs in: run it in the staging folder, removed after
        # the build, so they never land in the caller's working directory.
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, cwd=staging
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(command)} failed ({result.returncode}):\n"
                f"{result.stdout}{result.stderr}"
            )

    def _build(self, module: ModuleType, language: str, staging: Path) -> Path:
        """Run ``pyccel compile`` on copies of the module and its dependencies.

        The dependencies are copied under flat names (:func:`_flat_name`), with the
        imports rewritten to them, and built first into the same output folder, where
        Pyccel finds their headers; the kernel's extension contains their code, so
        only it is kept. Returns the build folder.
        """
        stem = module.__name__.rsplit(".", 1)[-1]
        package = module.__package__ or module.__name__.rpartition(".")[0]
        sources = staging / "src"
        sources.mkdir()
        output = staging / "build"
        openmp = self._openmp_for(module)
        flags = self.flags()
        dependencies = self.dependencies(module)
        for name, source in dependencies.items():
            flat = _flat_name(name)
            path = sources / f"{flat}.py"
            path.write_text(_rewrite_imports(source, name.rpartition(".")[0]))
            self._run_pyccel(path, language, output, openmp, staging, flags)
        path = sources / f"{stem}.py"
        path.write_text(_rewrite_imports(_source_path(module).read_text(), package))
        self._run_pyccel(path, language, output, openmp, staging, flags)
        for name in dependencies:
            extension = _extension_path(output, _flat_name(name))
            if extension is not None:
                extension.unlink()
        return output

    @sp.profile("cunumpy.PyccelCache.compile", tags=_PROFILE_TAGS)
    def compile(self, module: ModuleType, *, language: str | None = None) -> ModuleType:
        """Return the compiled form of `module`, building it on first use.

        Parameters
        ----------
        module : module
            A module Pyccel can compile (its functions keep their names).
        language : str, optional
            Overrides the instance's language for this build.

        Returns
        -------
        module
            The compiled extension module.

        Raises
        ------
        ImportError
            If Pyccel is not installed.
        RuntimeError
            With the output of ``pyccel compile`` if the build fails, or of
            the import if this interpreter cannot import the new build (it is
            then not stored); callers decide on a fallback.
        PermissionError
            If neither the cache nor the fallback directory is writable.
        """
        language = language or self.language
        stem = module.__name__.rsplit(".", 1)[-1]
        name = f"{stem}-{self.build_key(module, language)}"
        roots = [self.cache_root(), self.fallback_root()]
        for root in roots:
            if (root / name).is_dir():
                extension = _load_extension(root / name, stem)
                if extension is not None:
                    return extension
        target = next((root / name for root in roots if _writable(root)), None)
        if target is None:
            raise PermissionError(f"No writable kernel cache among {roots}")
        # absolute: Pyccel runs inside it, so a relative cache path must not move
        staging = Path(
            tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent)
        ).absolute()
        try:
            build = self._build(module, language, staging)
            _check_importable(build, stem)
            try:
                # Atomic, so concurrent builders never see a partial directory.
                os.rename(build, target)
            except OSError:
                pass  # Another process published the same build first.
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        extension = _load_extension(target, stem)
        if extension is None:
            raise ImportError(f"Compiled module for {module.__name__} not found")
        return extension

    def __call__(self, module: ModuleType) -> ModuleType:
        """:meth:`compile`, so the instance can be a ``compile_host``."""
        return self.compile(module)

    def __repr__(self) -> str:
        return f"PyccelCache({self.name!r}, env_prefix={self.env_prefix!r})"
