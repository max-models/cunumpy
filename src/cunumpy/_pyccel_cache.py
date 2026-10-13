"""Compile Pyccel kernel modules once and reuse the builds across processes.

:class:`PyccelCache` builds a module with the ``pyccel compile`` command line
interface, in a subprocess, once. It keys a build on the module source, the
target language, the OpenMP setting, the Pyccel and Python versions and the
platform, and stores it under a per-package cache directory. Later processes,
and other MPI ranks, import the stored extension directly; only the first run
after an edit compiles.

cunumpy needs no Pyccel to import this module: Pyccel is looked up when a
module is compiled, and a missing Pyccel is an :class:`ImportError` there, which
the host-kernel dispatch records like any other failed compilation.
"""

from __future__ import annotations

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

#: Bumped when the layout of a cached build changes (it was an ``epyccel``
#: folder before the ``pyccel compile`` command line was used). Part of the key.
_LAYOUT = "pyccel-cli-1"

_FALSE = ("0", "false", "no", "off")


def _uses_openmp_runtime(source: Path) -> bool:
    """Whether the module calls the OpenMP runtime (thread ids, thread counts).

    Such a module only compiles with OpenMP, so it is built with it even when
    OpenMP is turned off; ``OMP_NUM_THREADS=1`` then runs it on one thread.
    """
    return "pyccel.stdlib.internal.openmp" in source.read_text()


def _source_path(module: ModuleType) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise FileNotFoundError(f"No source file for {module.__name__}")
    return Path(source)


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def _load_extension(directory: Path, stem: str) -> ModuleType | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(stem, path)
        if spec is None or spec.loader is None:
            continue
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        return extension
    return None


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
        ``TREE_D_3D_KERNEL_CACHE`` and ``TREE_D_3D_OPENMP``).
    language : str, optional
        Pyccel's target language. Default ``"c"``.

    Examples
    --------
    >>> cache = xp.kernels.PyccelCache("my-sim")
    >>> cache.cache_env, cache.openmp_env
    ('MY_SIM_KERNEL_CACHE', 'MY_SIM_OPENMP')

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

    def __repr__(self) -> str:
        return f"PyccelCache({self.name!r}, env_prefix={self.env_prefix!r})"

    @property
    def cache_env(self) -> str:
        """Name of the environment variable that moves the cache directory."""
        return f"{self.env_prefix}_KERNEL_CACHE"

    @property
    def openmp_env(self) -> str:
        """Name of the environment variable that turns OpenMP off (``0``)."""
        return f"{self.env_prefix}_OPENMP"

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
        return self.openmp_enabled() or _uses_openmp_runtime(_source_path(module))

    def build_key(self, module: ModuleType, language: str | None = None) -> str:
        """Return the key of `module`'s build, which changes with anything that changes it.

        The module's own source, the language, the OpenMP setting, the Pyccel
        and Python versions and the platform. Modules the kernel imports are not
        part of it: edit the kernel (or clear the cache) after changing one.
        """
        digest = hashlib.sha256()
        for part in (
            inspect.getsource(module),
            language or self.language,
            _LAYOUT,
            f"openmp={self._openmp_for(module)}",
            _pyccel_version(),
            sys.version,
            platform.platform(),
        ):
            digest.update(part.encode())
        return digest.hexdigest()[:20]

    def _build(self, module: ModuleType, language: str, staging: Path) -> Path:
        """Run ``pyccel compile`` on a copy of the module's file; return the build folder."""
        source = _source_path(module)
        stem = module.__name__.rsplit(".", 1)[-1]
        sources = staging / "src"
        sources.mkdir()
        shutil.copy(source, sources / f"{stem}.py")
        output = staging / "build"
        command = [
            *_pyccel_command(),
            "compile",
            str(sources / f"{stem}.py"),
            "--language",
            language,
            "--output",
            str(output),
        ]
        if self._openmp_for(module):
            command.append("--openmp")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(command)} failed ({result.returncode}):\n"
                f"{result.stdout}{result.stderr}"
            )
        return output

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
            With the output of ``pyccel compile`` if the build fails; callers
            decide on a fallback.
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
        staging = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent))
        try:
            build = self._build(module, language, staging)
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
