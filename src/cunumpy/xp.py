"""Backend selection, array inspection and conversion.

The functions here are re-exported at the top level of :mod:`cunumpy`
(``xp.set_backend``, ``xp.to_numpy``, ...); use them from there. The attribute
``xp.xp`` is the active ``array-api-compat`` NumPy or CuPy module. See
:doc:`/guides/backends` and :doc:`/guides/data-movement`.
"""

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
    _record_sync,
)

if os.environ.get("CUNUMPY_FAKE_CUPY", "").strip().lower() in ("1", "true", "yes"):
    # tests without a GPU: a strict host stand-in for CuPy, see cunumpy._fake_cupy
    from cunumpy._fake_cupy import install as _install_fake_cupy

    _install_fake_cupy()

#: The backend names: ``"numpy"`` or ``"cupy"``.
BackendType = Literal["numpy", "cupy"]

_logger = logging.getLogger(__name__)


_CUPY_AVAILABLE_CACHE = None
_CUPY_UNAVAILABLE_REASON: str | None = None


def cupy_available() -> bool:
    """Return whether CuPy can be imported and reports a usable CUDA device.

    The result is cached for the process. It does not guarantee that every
    GPU operation succeeds later; :func:`backend_info` gives the reason when
    CuPy is unavailable.

    Returns
    -------
    bool
        True if CuPy is importable and ``cupy.is_available()`` is true.

    Examples
    --------
    >>> xp.cupy_available()  # doctest: +SKIP
    False
    """
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
    """The process-wide active backend (NumPy or CuPy).

    One instance, ``array_backend``, backs :func:`cunumpy.set_backend` and
    :func:`cunumpy.use_backend`. Not thread-safe: threads or async tasks that
    switch backends independently race on this shared state.

    Parameters
    ----------
    backend : {"numpy", "cupy"}, optional
        The requested backend; ``"cupy"`` falls back to NumPy if CuPy is not
        functional.
    verbose : bool, optional
        Print a message when falling back to NumPy.

    Raises
    ------
    ValueError
        If `backend` is neither ``"numpy"`` nor ``"cupy"``.
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
        """The name of the active backend."""
        return self._backend

    @property
    def xp(self) -> ModuleType:
        """The active ``array-api-compat`` NumPy or CuPy module."""
        return self._xp

    def add_listener(self, listener: Callable[[ModuleType], None]) -> None:
        """Call ``listener(module)`` now and whenever the backend module changes.

        Parameters
        ----------
        listener : callable
            Called with the new backend module.
        """
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
        """Select `backend`; see :func:`cunumpy.set_backend`.

        Parameters
        ----------
        backend : {"numpy", "cupy"}
            The requested backend.
        strict : bool, optional
            Raise instead of falling back to NumPy when CuPy is unavailable.

        Raises
        ------
        ValueError
            If `backend` is neither ``"numpy"`` nor ``"cupy"``.
        RuntimeError
            If `strict` is true and CuPy is unavailable.
        """
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
        """Select `backend` inside a ``with`` block; see :func:`cunumpy.use_backend`.

        Parameters
        ----------
        backend : {"numpy", "cupy"}
            The requested backend.
        strict : bool, optional
            Raise instead of falling back to NumPy when CuPy is unavailable.

        Yields
        ------
        None
            The previous backend is restored on exit, also on an exception.
        """
        old_backend = self._backend
        old_xp = self._xp
        self.set(backend, strict=strict)
        try:
            yield
        finally:
            self._set(old_backend, old_xp)


#: The process-wide backend state; NumPy unless ``CUNUMPY_BACKEND=cupy`` was set
#: at import.
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
    """Select a backend inside a ``with`` block and restore the previous one on exit.

    The previous backend is restored also when the block raises. Backend state
    is process-wide, so this is meant for sequential use, not for threads.

    Parameters
    ----------
    backend : {"numpy", "cupy"}
        The requested backend; ``"cupy"`` falls back to NumPy if CuPy is not
        functional.
    strict : bool, optional
        Raise ``RuntimeError`` instead of falling back, keeping the current
        backend.

    Returns
    -------
    context manager
        Selects `backend` on entry.

    Raises
    ------
    RuntimeError
        If `strict` is true and CuPy is unavailable.

    Examples
    --------
    >>> with xp.use_backend("numpy"):
    ...     reference = xp.zeros(3)
    >>> xp.get_array_backend(reference)
    'numpy'
    """
    return array_backend.use_backend(backend, strict=strict)


def set_backend(backend: BackendType, *, strict: bool = False) -> None:
    """Select the process-wide backend for new array operations.

    Requesting ``"cupy"`` selects it only if CuPy and CUDA are functional and
    otherwise falls back to NumPy silently; read :func:`get_backend` to see the
    effect. Existing arrays are not moved. Not thread-safe.

    Parameters
    ----------
    backend : {"numpy", "cupy"}
        The requested backend.
    strict : bool, optional
        Raise ``RuntimeError`` with the reason instead of falling back, keeping
        the current backend.

    Raises
    ------
    ValueError
        If `backend` is neither ``"numpy"`` nor ``"cupy"``.
    RuntimeError
        If `strict` is true and CuPy is unavailable.

    See Also
    --------
    use_backend : Select a backend for a ``with`` block.

    Examples
    --------
    >>> xp.set_backend("numpy")
    >>> xp.get_backend()
    'numpy'
    >>> xp.set_backend("cupy", strict=True)  # doctest: +SKIP
    """
    array_backend.set(backend, strict=strict)


def backend_info() -> dict[str, Any]:
    """Return JSON-compatible backend, dependency and CUDA diagnostics.

    Does not change the backend or import MPI. CUDA availability is cached as
    in :func:`cupy_available`; device inspection errors are reported in the
    result instead of raised.

    Returns
    -------
    dict
        Keys ``backend``, ``cupy_available``, ``cuda_unavailable_reason``,
        ``versions``, ``device`` (None without CUDA) and
        ``cuda_visible_devices``, plus ``cuda_inspection_error`` on failure.

    Examples
    --------
    >>> xp.backend_info()["backend"]
    'numpy'
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
    """Return the name of the active process-wide backend.

    Returns
    -------
    {"numpy", "cupy"}
        The effective backend, after any fallback of :func:`set_backend`.

    Examples
    --------
    >>> xp.get_backend()
    'numpy'
    """
    return array_backend.backend


def _cupy_backend() -> bool:
    """Check if the active global backend is CuPy."""
    return array_backend.backend == "cupy"


def _numpy_backend() -> bool:
    """Check if the active global backend is NumPy."""
    return array_backend.backend == "numpy"


def default_float_dtype() -> Any:
    """Return the active backend's ``float64`` dtype.

    Use ``dtype=xp.default_float_dtype()`` instead of ``dtype=float`` when an
    explicit, portable precision matters: default dtypes can depend on the
    platform and backend.

    Returns
    -------
    dtype
        ``float64`` of the active backend module.

    Examples
    --------
    >>> xp.asarray([0.1, 0.2], dtype=xp.default_float_dtype()).dtype
    dtype('float64')
    """
    return array_backend.xp.float64


def synchronize() -> None:
    """Wait for all work in all streams on the current CUDA device to finish.

    Call it before host code reads results computed asynchronously, or before
    timing. A no-op on the NumPy backend.

    Examples
    --------
    >>> xp.synchronize()
    """
    if array_backend.backend == "cupy":
        if _COUNTERS:
            _record_sync("synchronize()")
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
        return array.get(order="A")

    return np.asarray(array)


def _to_cupy(array: Any) -> Any:
    """`to_cupy` without transfer counting, for internal use."""
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import array_api_compat.cupy as cp

    return cp.asarray(array)


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array to a NumPy array on the host.

    A CuPy array is copied to the host (counted as a ``to_host`` transfer by
    :func:`~cunumpy.profiling.count_transfers`), keeping F order if it is
    F-contiguous and using C order otherwise. Anything else goes through
    ``numpy.asarray``, so a NumPy array is returned without a copy. See
    :doc:`/guides/array-ordering`.

    Parameters
    ----------
    array : array-like
        A NumPy or CuPy array, or anything ``numpy.asarray`` accepts.

    Returns
    -------
    numpy.ndarray
        The host array.

    Examples
    --------
    >>> xp.to_numpy([1, 2])
    array([1, 2])
    """
    result = _to_numpy(array)
    if _COUNTERS and get_array_backend(array) == "cupy":
        _record("to_host", f"to_numpy({_describe(array)})", nbytes=_nbytes(result))
    return result


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array on the device.

    Host input is copied to the device (counted as a ``to_device`` transfer).
    A CuPy array or an object with ``__cuda_array_interface__`` may be
    referenced without a copy; device-to-device copies are counted as
    ``device_copy``. The source is not modified.

    Parameters
    ----------
    array : array-like
        A NumPy or CuPy array, or anything ``cupy.asarray`` accepts.

    Returns
    -------
    cupy.ndarray
        The device array.

    Raises
    ------
    ImportError
        If CuPy or CUDA is unavailable or not functional.

    Examples
    --------
    >>> xp.to_cupy([1, 2])  # doctest: +SKIP
    array([1, 2])
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
    """Return `value` as a C-contiguous device array, copying only if needed.

    For building CUDA argument objects (:class:`~cunumpy.arguments.CudaArguments`
    subclasses, :class:`~cunumpy.arguments.CudaStruct` values): call it once
    when the object is built, never per kernel call. A C-contiguous CuPy array
    of the requested `dtype` is returned as the same object, so kernels write
    into the caller's array. Anything else (a tuple, a host array, another
    dtype, a non-contiguous view) is copied with
    ``cupy.ascontiguousarray(cupy.asarray(value, dtype))``; each copy is
    counted by :func:`~cunumpy.profiling.count_transfers`.

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
        If the active backend is not CuPy (host data is never copied to the
        device implicitly).
    ValueError
        If `ndim` is given and the array has another number of dimensions.

    Examples
    --------
    >>> degree = xp.as_device_array((3, 3, 3), "int32", 1, name="degree")  # doctest: +SKIP
    >>> degree.dtype, degree.shape  # doctest: +SKIP
    (dtype('int32'), (3,))
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
    """Convert an array to the active backend.

    Delegates to :func:`to_cupy` or :func:`to_numpy`, so an actual copy is
    counted as a ``to_device`` or ``to_host`` transfer. Neither the backend
    nor the source is changed.

    Parameters
    ----------
    array : array-like
        A NumPy or CuPy array, or another array-like.

    Returns
    -------
    array
        An array of the active backend.

    Examples
    --------
    >>> xp.to_cunumpy([1.0, 2.0])
    array([1., 2.])
    """
    if array_backend.backend == "cupy" and cupy_available():
        return to_cupy(array)
    return to_numpy(array)


def get_array_backend(array: Any) -> BackendType:
    """Return the backend of `array`, independent of the active backend.

    Parameters
    ----------
    array : object
        Any object; only CuPy arrays count as ``"cupy"``.

    Returns
    -------
    {"numpy", "cupy"}
        ``"cupy"`` for a CuPy array, ``"numpy"`` for anything else.

    Examples
    --------
    >>> xp.get_array_backend(np.zeros(3))
    'numpy'
    """
    return "cupy" if array_api_compat.is_cupy_array(array) else "numpy"


def get_array_module(array: Any) -> ModuleType:
    """Return the ``array-api-compat`` module of `array`'s own backend.

    Dispatches on the array, not on the active backend, for functions that
    must follow their input. Like ``cupy.get_array_module``, but it works
    without CuPy installed and returns ``array-api-compat`` modules, not raw
    ``numpy``/``cupy``.

    Parameters
    ----------
    array : object
        A NumPy or CuPy array.

    Returns
    -------
    module
        ``array_api_compat.cupy`` for a CuPy array, ``array_api_compat.numpy``
        otherwise.

    Examples
    --------
    >>> xp.get_array_module(np.zeros(3)).__name__
    'array_api_compat.numpy'
    """
    if get_array_backend(array) == "cupy":
        import array_api_compat.cupy as cp

        return cp
    return np


def is_gpu(array: Any) -> bool:
    """Return whether `array` is a CuPy (GPU) array.

    Parameters
    ----------
    array : object
        Any object.

    Returns
    -------
    bool
        True if ``get_array_backend(array) == "cupy"``.

    Examples
    --------
    >>> xp.is_gpu(np.zeros(3))
    False
    """
    return get_array_backend(array) == "cupy"


def is_cpu(array: Any) -> bool:
    """Return whether `array` is not a CuPy array (NumPy, CPU).

    Parameters
    ----------
    array : object
        Any object.

    Returns
    -------
    bool
        True if ``get_array_backend(array) == "numpy"``.

    Examples
    --------
    >>> xp.is_cpu(np.zeros(3))
    True
    """
    return get_array_backend(array) == "numpy"


def same_backend(*arrays: Any) -> bool:
    """Return whether all `arrays` live on the same backend.

    Parameters
    ----------
    *arrays : object
        Arrays to compare.

    Returns
    -------
    bool
        True if they share a backend; always True for zero or one array.

    Examples
    --------
    >>> xp.same_backend(np.zeros(2), np.ones(3))
    True
    """
    if len(arrays) <= 1:
        return True
    backends = {get_array_backend(array) for array in arrays}
    return len(backends) == 1


def assert_same_backend(*arrays: Any) -> None:
    """Raise ``TypeError`` if `arrays` do not all live on the same backend.

    Use it at API boundaries: mixing NumPy and CuPy arrays otherwise fails
    later with a confusing backend-internal error. Align arrays with
    :func:`to_cunumpy`, :func:`to_numpy` or :func:`to_cupy`.

    Parameters
    ----------
    *arrays : object
        Arrays to compare.

    Raises
    ------
    TypeError
        If the arrays are on different backends; the message lists them.

    Examples
    --------
    >>> xp.assert_same_backend(np.zeros(2), np.ones(2))
    >>> xp.assert_same_backend(np.zeros(2), xp.to_cupy([1.0]))  # doctest: +SKIP
    Traceback (most recent call last):
    TypeError: Arrays are on mismatched backends: ['numpy', 'cupy']. ...
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
