from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import array_api_compat
import array_api_compat.numpy as np

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import (
    _describe,
    _device_pointer,
    _is_device_copy,
    _nbytes,
    _record,
)

if os.environ.get("CUNUMPY_FAKE_CUPY", "").strip().lower() in ("1", "true", "yes"):
    # tests without a GPU: a strict host stand-in for CuPy, see cunumpy._fake_cupy
    from cunumpy._fake_cupy import install as _install_fake_cupy

    _install_fake_cupy()

BackendType = Literal["numpy", "cupy"]

_logger = logging.getLogger(__name__)


_CUPY_AVAILABLE_CACHE = None
_CUPY_UNAVAILABLE_REASON: str | None = None


def cupy_available() -> bool:
    """Check if CuPy is available and functional."""
    global _CUPY_AVAILABLE_CACHE, _CUPY_UNAVAILABLE_REASON
    if _CUPY_AVAILABLE_CACHE is not None:
        return _CUPY_AVAILABLE_CACHE

    try:
        import cupy as cp

        # Check if a GPU is available
        _CUPY_AVAILABLE_CACHE = cp.is_available()
        _CUPY_UNAVAILABLE_REASON = (
            None if _CUPY_AVAILABLE_CACHE else "CuPy reports no usable CUDA device"
        )
        return _CUPY_AVAILABLE_CACHE
    except Exception as error:  # noqa: BLE001 - tolerate driver/runtime failure
        _CUPY_AVAILABLE_CACHE = False
        _CUPY_UNAVAILABLE_REASON = f"{type(error).__name__}: {error}"
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
                        "CuPy not available or not functional. Falling back to NumPy.",
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

    def set(self, backend: BackendType, *, strict: bool = False) -> None:
        """Select `backend` (falls back to NumPy if CuPy is not functional)."""
        if backend not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")
        if strict and backend == "cupy" and not cupy_available():
            raise RuntimeError(
                "Cannot select the CuPy backend: "
                + (_CUPY_UNAVAILABLE_REASON or "CuPy/CUDA is unavailable"),
            )
        module = self._load_backend(backend)  # sets self._backend to the effective one
        self._set(self._backend, module)

    @contextmanager
    def use_backend(
        self,
        backend: BackendType,
        *,
        strict: bool = False,
    ) -> Generator[None, None, None]:
        """Temporarily change the backend."""
        old_backend = self._backend
        old_xp = self._xp
        self.set(backend, strict=strict)
        try:
            yield
        finally:
            self._set(old_backend, old_xp)


array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("CUNUMPY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)


def use_backend(
    backend: BackendType,
    *,
    strict: bool = False,
) -> Generator[None, None, None]:
    """Temporarily change the backend."""
    return array_backend.use_backend(backend, strict=strict)


def set_backend(backend: BackendType, *, strict: bool = False) -> None:
    """Select a backend; with `strict`, unavailable CUDA raises without switching."""
    array_backend.set(backend, strict=strict)


def backend_info() -> dict[str, Any]:
    """Return JSON-compatible backend, dependency, and CUDA diagnostics.

    Does not change the backend or import MPI. CUDA availability is cached, as
    in :func:`cupy_available`. Device inspection errors are reported in the
    result rather than hiding the otherwise useful CPU/dependency information.
    """
    versions = {}
    for package in ("cunumpy", "numpy", "array-api-compat"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    available = bool(cupy_available())
    versions["cupy"] = getattr(sys.modules.get("cupy"), "__version__", None)
    info: dict[str, Any] = {
        "backend": get_backend(),
        "cupy_available": available,
        "cuda_unavailable_reason": None if available else _CUPY_UNAVAILABLE_REASON,
        "versions": versions,
        "device": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if available:
        try:
            import cupy as cp

            versions["cupy"] = getattr(cp, "__version__", None)
            dev = cp.cuda.Device()
            properties = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = properties["name"]
            info["device"] = {
                "id": int(dev.id),
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "compute_capability": str(dev.compute_capability),
                "visible_devices": int(cp.cuda.runtime.getDeviceCount()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
                "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            }
        except Exception as error:  # noqa: BLE001 - diagnostics must remain usable
            info["cuda_inspection_error"] = f"{type(error).__name__}: {error}"
    return info


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
    result = _to_numpy(array)
    if _COUNTERS and get_array_backend(array) == "cupy":
        _record("to_host", f"to_numpy({_describe(array)})", nbytes=_nbytes(result))
    return result


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array.

    Host input is copied to the device and counted as a ``to_device`` transfer.
    CUDA-array-interface inputs may be referenced without a copy; device-only
    copies are recorded separately as ``device_copy``.
    """
    result = _to_cupy(array)
    if _COUNTERS:
        device = get_array_backend(array) == "cupy" or hasattr(
            array, "__cuda_array_interface__"
        )
        if not device:
            _record("to_device", f"to_cupy({_describe(array)})", nbytes=_nbytes(result))
        elif _device_pointer(array) is not None and _is_device_copy(array, result):
            _record(
                "device_copy", f"to_cupy({_describe(array)})", nbytes=_nbytes(result)
            )
    return result


def as_device_array(
    value: Any,
    dtype: Any = None,
    ndim: int | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Reference `value` on the device, or convert it to a contiguous device array.

    The "reference or copy once" rule for building CUDA argument objects
    (`CudaArguments` subclasses, `CudaStruct` values): call it once when the
    argument object is built, never per kernel call. A CuPy array that already
    has the requested `dtype` (any dtype if `dtype` is None) and is C-contiguous
    is returned unchanged, the same object without a copy, so kernels write
    into the caller's array. Anything else is converted with
    ``cupy.ascontiguousarray(cupy.asarray(value, dtype))``: a tuple or list
    (e.g. ``degree = (3, 3, 3)``), a host NumPy array (one explicit transfer
    at build time), a device array of another dtype, or a non-contiguous view.
    The result passes the pointer checks of `CudaKernel` and `CudaStruct`.
    Dtype and layout conversion can require separate device copies; each copy
    through this helper is visible in transfer accounting.

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
            "copied to the device implicitly (build host arguments instead)",
        )

    import cupy as cp

    if (
        isinstance(value, cp.ndarray)
        and value.flags.c_contiguous
        and (dtype is None or value.dtype == np.dtype(dtype))
    ):
        result = value
    else:
        converted = cp.asarray(value, dtype=dtype)
        if _COUNTERS:
            device_only = isinstance(value, cp.ndarray) or hasattr(
                value,
                "__cuda_array_interface__",
            )
            if not device_only or _is_device_copy(value, converted):
                _record(
                    "device_copy" if device_only else "to_device",
                    f"as_device_array({_describe(value)})",
                    nbytes=_nbytes(converted),
                )
        result = cp.ascontiguousarray(converted)
        if _COUNTERS and _is_device_copy(converted, result):
            _record(
                "device_copy",
                f"as_device_array({_describe(converted)}) layout",
                nbytes=_nbytes(result),
            )
    if ndim is not None and result.ndim != ndim:
        raise ValueError(
            f"{what} must have {ndim} dimension(s), got {result.ndim} "
            f"(shape {result.shape})",
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
            "xp.to_cunumpy()/xp.to_numpy()/xp.to_cupy() to align them first.",
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
