from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import array_api_compat
import array_api_compat.numpy as np

from ._transfers import _ACTIVE as _COUNTERS
from ._transfers import _describe, _record

if os.environ.get("CUNUMPY_FAKE_CUPY", "").strip().lower() in ("1", "true", "yes"):
    # tests without a GPU: a strict host stand-in for CuPy, see cunumpy._fake_cupy
    from ._fake_cupy import install as _install_fake_cupy

    _install_fake_cupy()

BackendType = Literal["numpy", "cupy"]

_logger = logging.getLogger(__name__)


_CUPY_AVAILABLE_CACHE = None


def cupy_available() -> bool:
    """Check if CuPy is available and functional."""
    global _CUPY_AVAILABLE_CACHE
    if _CUPY_AVAILABLE_CACHE is not None:
        return _CUPY_AVAILABLE_CACHE

    try:
        import cupy as cp

        # Check if a GPU is available
        _CUPY_AVAILABLE_CACHE = cp.is_available()
        return _CUPY_AVAILABLE_CACHE
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        _CUPY_AVAILABLE_CACHE = False
        return False


class ArrayBackend:
    """Holds the process-wide active backend (NumPy or CuPy).

    Not thread-safe: `set_backend`/`use_backend` mutate this single shared
    instance in place, so concurrent code (threads, async tasks) switching
    backends independently will race. Safe for the typical single-threaded
    script/notebook usage this library targets.
    """

    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        if backend.lower() not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")

        self._backend: BackendType = "cupy" if backend.lower() == "cupy" else "numpy"
        self._xp: ModuleType = np  # Placeholder
        self._listeners: list[Callable[[ModuleType], None]] = []

        # Import numpy/cupy
        self._xp = self._load_backend(self._backend, verbose)

    def _load_backend(self, backend: BackendType, verbose: bool = False) -> ModuleType:
        if backend == "cupy":
            if cupy_available():
                import array_api_compat.cupy as cp

                self._backend = "cupy"
                return cp
            else:
                if verbose:
                    print(
                        "CuPy not available or not functional. Falling back to NumPy."
                    )
                self._backend = "numpy"
                return np
        self._backend = "numpy"
        return np

    def __repr__(self) -> str:
        return f"ArrayBackend(backend={self._backend!r}, module={self._xp.__name__!r})"

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        return self._xp

    def add_listener(self, listener: Callable[[ModuleType], None]) -> None:
        """Call `listener(module)` now and whenever the backend module changes."""
        self._listeners.append(listener)
        listener(self._xp)

    def _set(self, backend: BackendType, module: ModuleType) -> None:
        changed = module is not self._xp
        self._backend = backend
        self._xp = module
        if changed:
            for listener in self._listeners:
                listener(module)

    def set(self, backend: BackendType) -> None:
        """Select `backend` (falls back to NumPy if CuPy is not functional)."""
        if backend not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")
        module = self._load_backend(backend)  # sets self._backend to the effective one
        self._set(self._backend, module)

    @contextmanager
    def use_backend(self, backend: BackendType) -> Generator[None, None, None]:
        """Temporarily change the backend."""
        old_backend = self._backend
        old_xp = self._xp
        self.set(backend)
        try:
            yield
        finally:
            self._set(old_backend, old_xp)


array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)


def use_backend(backend: BackendType) -> Generator[None, None, None]:
    """Temporarily change the backend."""
    return array_backend.use_backend(backend)


def set_backend(backend: BackendType) -> None:
    """Set the backend globally."""
    array_backend.set(backend)


def get_backend() -> BackendType:
    """Return the currently active global backend name."""
    return array_backend.backend


def _cupy_backend() -> bool:
    """Check if the active global backend is CuPy."""
    return array_backend.backend == "cupy"


def _numpy_backend() -> bool:
    """Check if the active global backend is NumPy."""
    return array_backend.backend == "numpy"


def default_float_dtype() -> Any:
    """Return the active backend's `float64` dtype object.

    NumPy and CuPy resolve Python `int`/`float` literals and the bare
    `dtype=float` spelling to a platform- or backend-dependent default
    (e.g. NumPy's default integer width differs between Windows and
    Linux/macOS). Use ``dtype=xp.default_float_dtype()`` instead of
    ``dtype=float`` when a specific, portable precision matters.
    """
    return array_backend.xp.float64


def synchronize() -> None:
    """Wait for all kernels in all streams on current device to complete."""
    if array_backend.backend == "cupy":
        try:
            import cupy as cp

            cp.cuda.Device().synchronize()
        except ImportError:
            pass
        except AttributeError as e:
            warnings.warn(
                f"CuPy synchronize() failed unexpectedly, this may indicate a "
                f"CuPy API mismatch: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def _to_numpy(array: Any) -> np.ndarray:
    """`to_numpy` without transfer counting, for internal use."""
    if get_array_backend(array) == "cupy":
        return array.get()

    return np.asarray(array)


def _to_cupy(array: Any) -> Any:
    """`to_cupy` without transfer counting, for internal use."""
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import array_api_compat.cupy as cp

    return cp.asarray(array)


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array to a NumPy array.

    A CuPy array is copied to the host, which `count_transfers()` counts as a
    ``to_host`` transfer; anything else is passed through `numpy.asarray`.
    """
    if _COUNTERS and get_array_backend(array) == "cupy":
        _record("to_host", f"to_numpy({_describe(array)})")
    return _to_numpy(array)


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array.

    Anything that is not a CuPy array already is copied to the device, which
    `count_transfers()` counts as a ``to_device`` transfer.
    """
    if _COUNTERS and get_array_backend(array) != "cupy":
        _record("to_device", f"to_cupy({_describe(array)})")
    return _to_cupy(array)


def as_device_array(
    value: Any,
    dtype: Any = None,
    ndim: int | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Reference `value` on the device, or make one device copy of it.

    The "reference or copy once" rule for building CUDA argument objects
    (`CudaArguments` subclasses, `CudaStruct` values): call it once when the
    argument object is built, never per kernel call. A CuPy array that already
    has the requested `dtype` (any dtype if `dtype` is None) and is C-contiguous
    is returned unchanged, the same object without a copy, so kernels write
    into the caller's array. Anything else is converted with one device copy,
    ``cupy.ascontiguousarray(cupy.asarray(value, dtype))``: a tuple or list
    (e.g. ``degree = (3, 3, 3)``), a host NumPy array (one explicit transfer
    at build time), a device array of another dtype, or a non-contiguous view.
    The result passes the pointer checks of `CudaKernel` and `CudaStruct`.

    Raises on the NumPy backend: device argument objects are only built when
    running on CuPy, and host data is never copied to the device implicitly.

    Parameters
    ----------
    value : array-like
        A CuPy array, a NumPy array, or a sequence of numbers.
    dtype : dtype-like, optional
        The dtype the kernel expects, e.g. the pointed-to type of the
        parameter. None keeps the dtype of `value`.
    ndim : int, optional
        The expected number of dimensions of the result.
    name : str, optional
        Name of the argument, used in error messages.

    Returns
    -------
    cupy.ndarray
        `value` itself, or a C-contiguous device copy with dtype `dtype`.

    Raises
    ------
    RuntimeError
        The active backend is not CuPy.
    ValueError
        `ndim` is given and the array has another number of dimensions.
    """
    what = f"device argument {name!r}" if name is not None else "device argument"
    if array_backend.backend != "cupy":
        raise RuntimeError(
            f"{what}: the active backend is {array_backend.backend!r}; device "
            "arguments are only built on the CuPy backend, and host data is never "
            "copied to the device implicitly (build host arguments instead)"
        )

    import cupy as cp

    if (
        isinstance(value, cp.ndarray)
        and value.flags.c_contiguous
        and (dtype is None or value.dtype == np.dtype(dtype))
    ):
        result = value
    else:
        result = cp.ascontiguousarray(cp.asarray(value, dtype=dtype))
    if ndim is not None and result.ndim != ndim:
        raise ValueError(
            f"{what} must have {ndim} dimension(s), got {result.ndim} "
            f"(shape {result.shape})"
        )
    return result


def to_cunumpy(array: Any) -> Any:
    """Convert an array to the currently active backend.

    Delegates to `to_cupy()` or `to_numpy()`, so an actual copy is counted by
    `count_transfers()` as a ``to_device`` or ``to_host`` transfer.
    """
    if array_backend.backend == "cupy" and cupy_available():
        return to_cupy(array)
    return to_numpy(array)


def get_array_backend(array: Any) -> BackendType:
    """Return 'cupy' or 'numpy' depending on the array's type."""
    return "cupy" if array_api_compat.is_cupy_array(array) else "numpy"


def get_array_module(array: Any) -> ModuleType:
    """Return the array-api-compat module matching `array`'s own backend.

    Unlike `xp.xp`, which reflects the process-wide active backend, this
    dispatches on the array itself -- useful for writing functions that
    operate correctly regardless of what `set_backend`/`use_backend` last
    selected. Mirrors `cupy.get_array_module`, but also works in Pyodide
    (where `cupy` cannot be imported) and returns array-api-compat modules
    for standard-conformant behavior, consistent with `xp.xp`.
    """
    if get_array_backend(array) == "cupy":
        import array_api_compat.cupy as cp

        return cp
    return np


def is_gpu(array: Any) -> bool:
    """Check if the array is stored on a GPU (CuPy)."""
    return get_array_backend(array) == "cupy"


def is_cpu(array: Any) -> bool:
    """Check if the array is stored on a CPU (NumPy)."""
    return get_array_backend(array) == "numpy"


def same_backend(*arrays: Any) -> bool:
    """Return True if all given arrays live on the same backend.

    Trivially True for zero or one array.
    """
    if len(arrays) <= 1:
        return True
    backends = {get_array_backend(array) for array in arrays}
    return len(backends) == 1


def assert_same_backend(*arrays: Any) -> None:
    """Raise TypeError if the given arrays don't all live on the same backend.

    Mixing NumPy and CuPy arrays in an operation typically fails with a
    confusing, backend-internal error (e.g. a CuPy kernel dispatch error
    complaining about an "unsupported type"). Call this upfront to fail
    with a clear message instead. Use `to_cunumpy()`/`to_numpy()`/`to_cupy()`
    to align mismatched arrays onto one backend first.
    """
    if not same_backend(*arrays):
        backends = [get_array_backend(array) for array in arrays]
        raise TypeError(
            f"Arrays are on mismatched backends: {backends}. Use "
            "xp.to_cunumpy()/xp.to_numpy()/xp.to_cupy() to align them first."
        )


# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    import numpy as xp  # noqa: F401 - type-checker-only alias for autocompletion
else:
    # Use module-level __getattr__ for dynamic xp (Python 3.7+)
    def __getattr__(name):
        if name == "xp":
            return array_backend.xp
        if name == "numpy_backend":
            return _numpy_backend()
        if name == "cupy_backend":
            return _cupy_backend()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
