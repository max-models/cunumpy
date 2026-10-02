"""Compile Pyccel kernel modules once, cache them on disk, use them as host kernels.

``pyccel.epyccel`` recompiles on every call and gives each build a random
module name. :func:`compile_cached` keys a build on the module source, the
language, the Pyccel and Python versions and the platform, and stores it under
``CUNUMPY_KERNEL_CACHE`` (default ``~/.cache/cunumpy/kernels``); later processes
import the stored extension directly. A build is moved into the cache in one
atomic rename, so concurrent processes (e.g. MPI ranks starting together) never
see a partial build. If the cache is not writable (a CI container whose home
belongs to another user), the build goes to the system temporary directory.

:class:`CompiledHostKernel` is a host kernel compiled on its first call, with a
fallback when compilation is not possible; it is what
:meth:`KernelCatalog.from_package(..., compile_host=compile_cached)
<cunumpy.KernelCatalog.from_package>` uses for the host side of each kernel.

Pyccel is imported only when something is compiled; it is an optional
dependency of cunumpy.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import os
import platform
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["CACHE_ENV_VAR", "CompiledHostKernel", "cache_root", "compile_cached"]

#: Environment variable with the directory of the compiled kernel cache.
CACHE_ENV_VAR = "CUNUMPY_KERNEL_CACHE"


def cache_root() -> Path:
    """Directory holding compiled kernel builds (``CUNUMPY_KERNEL_CACHE``)."""
    default = Path.home() / ".cache" / "cunumpy" / "kernels"
    return Path(os.environ.get(CACHE_ENV_VAR, default))


def _fallback_root() -> Path:
    return Path(tempfile.gettempdir()) / "cunumpy-kernels"


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def _build_key(module: ModuleType, language: str) -> str:
    digest = hashlib.sha256()
    for part in (
        inspect.getsource(module),
        language,
        importlib.metadata.version("pyccel"),
        sys.version,
        platform.platform(),
    ):
        digest.update(part.encode())
    return digest.hexdigest()[:20]


def _load_extension(directory: Path, stem: str) -> ModuleType | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for path in sorted(directory.glob(f"{stem}_*{suffix}")):
            name = path.name[: -len(suffix)]
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                continue
            extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
            return extension
    return None


def compile_cached(module: ModuleType, *, language: str = "c") -> ModuleType:
    """The Pyccel-compiled form of `module`, built on first use and cached on disk.

    Parameters
    ----------
    module : ModuleType
        A module written in Pyccel's subset of Python.
    language : str
        Pyccel's target language, ``"c"`` or ``"fortran"``.

    Returns
    -------
    ModuleType
        The compiled extension module, with the same functions.

    Raises
    ------
    ImportError
        If Pyccel is not installed, or the build produced no extension.
    PermissionError
        If neither the cache nor the temporary directory is writable.
    Exception
        Whatever Pyccel raises if the module does not compile; callers decide
        on a fallback (see :class:`CompiledHostKernel`).
    """
    stem = module.__name__.rsplit(".", 1)[-1]
    name = f"{stem}-{_build_key(module, language)}"
    roots = [cache_root(), _fallback_root()]
    for root in roots:
        if (root / name).is_dir():
            extension = _load_extension(root / name, stem)
            if extension is not None:
                return extension
    from pyccel import epyccel

    target = next((root / name for root in roots if _writable(root)), None)
    if target is None:
        raise PermissionError(f"No writable kernel cache among {roots}")
    staging = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent))
    try:
        epyccel(module, language=language, folder=str(staging))
        try:
            # atomic, so concurrent builders never see a partial directory
            os.rename(staging / "__epyccel__", target)
        except OSError:
            pass  # another process published the same build first
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    extension = _load_extension(target, stem)
    if extension is None:
        raise ImportError(f"Compiled kernel for {module.__name__} not found")
    return extension


class CompiledHostKernel:
    """A host kernel compiled on its first call, with a fallback.

    Parameters
    ----------
    module : str | ModuleType
        The module that defines the kernel (written for Pyccel), or its name.
    name : str
        Name of the kernel function in `module`.
    compiler : Callable[[ModuleType], ModuleType]
        Compiles the module, e.g. :func:`compile_cached`.
    fallback : Callable | None
        Called instead when compilation fails (e.g. a vectorized NumPy
        implementation with the same arguments). Without one, the uncompiled
        Python function is called, with a warning: correct, but slow.

    Notes
    -----
    :attr:`compiled` builds the kernel and reports whether that worked, so
    that callers can choose another code path; :attr:`error` keeps the
    exception of a failed build.
    """

    def __init__(
        self,
        module: str | ModuleType,
        name: str,
        compiler: Callable[[ModuleType], ModuleType] = compile_cached,
        fallback: Callable[..., Any] | None = None,
    ) -> None:
        self._module = module
        self.__name__ = name
        self._compiler = compiler
        self._fallback = fallback
        self._compiled: Callable[..., Any] | None = None
        self._built = False
        self.error: BaseException | None = None
        self._warned = False

    def __repr__(self) -> str:
        state = (
            "not built"
            if not self._built
            else ("compiled" if self._compiled else "failed")
        )
        return f"CompiledHostKernel({self.__name__!r}, {state})"

    @property
    def module(self) -> ModuleType:
        """The (uncompiled) module that defines the kernel."""
        if isinstance(self._module, str):
            self._module = importlib.import_module(self._module)
        return self._module

    @property
    def python(self) -> Callable[..., Any]:
        """The uncompiled Python function (for signatures and reference results)."""
        return getattr(self.module, self.__name__)

    def build(self) -> Callable[..., Any] | None:
        """Compile now (once); the compiled function, or None if that failed."""
        if not self._built:
            self._built = True
            try:
                self._compiled = getattr(self._compiler(self.module), self.__name__)
            except Exception as error:  # noqa: BLE001 -- no Pyccel or no compiler
                self.error = error
        return self._compiled

    @property
    def compiled(self) -> bool:
        """Whether the compiled version is available (compiles on first access)."""
        return self.build() is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kernel = self.build()
        if kernel is None:
            if self._fallback is not None:
                kernel = self._fallback
            else:
                if not self._warned:
                    warnings.warn(
                        f"Kernel {self.__name__!r} could not be compiled "
                        f"({self.error!r}); running its uncompiled Python version.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned = True
                kernel = self.python
        return kernel(*args, **kwargs)
