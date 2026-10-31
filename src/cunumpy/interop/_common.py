"""Shared implementation for optional framework adapters.

The adapters share three rules here. Memory: an array lives on the host or on
a device, as its ``__dlpack_device__`` says (Apple GPUs share the host's
memory and count as host). Copies: a conversion shares memory when it can,
``copy=False`` raises where it cannot, ``copy=True`` always copies, and the
transfer profiler records a copy as ``to_host``, ``to_device`` or
``device_copy`` (a copy that stays in one memory) and a shared conversion not
at all. Dtypes: a conversion never changes a dtype that was not asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np

from cunumpy._transfers import _ACTIVE, _describe, _nbytes, _record

HOST = (1, 0)
"""The DLPack device of host memory, ``(kDLCPU, 0)``."""

# kDLCPU, kDLCUDAHost, kDLMetal (Apple's unified memory) and kDLROCMHost
_HOST_DEVICE_TYPES = {1, 3, 8, 11}
_DEVICE_NAMES = {
    2: "CUDA",
    4: "OpenCL",
    7: "Vulkan",
    10: "ROCm",
    13: "CUDA managed",
    14: "oneAPI",
}
_DLPACK_ERRORS = (AttributeError, BufferError, RuntimeError, TypeError, ValueError)


def require(module: str, *, extra: str | None = None) -> Any:
    """Import an optional framework and give a consistent installation error.

    Parameters
    ----------
    module : str
        Import name of the optional framework.
    extra : str or None, optional
        CuNumpy extra name, inferred from `module` when omitted.

    Returns
    -------
    module
        Imported framework module.
    """
    try:
        return import_module(module)
    except ImportError as error:
        extra = module.split(".", 1)[0] if extra is None else extra
        raise ImportError(
            f"{module} interoperability needs {module.split('.', 1)[0]!r}; "
            f"install cunumpy[{extra}]"
        ) from error


def target_backend(backend: str) -> Any:
    """Load a NumPy or CuPy target module.

    Parameters
    ----------
    backend : {"numpy", "cupy"}
        Name of the target array library.

    Returns
    -------
    module
        The requested array module.
    """
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy
        except ImportError as error:
            raise ImportError(
                "CuPy interoperability needs CuPy; install a compatible CuPy release"
            ) from error
        return cupy
    raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")


def cast(array: Any, dtype: Any) -> Any:
    """Cast an array only when a dtype was requested.

    Parameters
    ----------
    array : object
        Array to cast.
    dtype : object or None
        Requested dtype, or None to preserve the input.

    Returns
    -------
    object
        The original or cast array.
    """
    return array if dtype is None else array.astype(dtype)


def check_copy(copy: bool | None) -> None:
    """Reject a `copy` argument other than None, True or False.

    Parameters
    ----------
    copy : bool or None
        The `copy` argument of a conversion.
    """
    if copy is not None and copy is not True and copy is not False:
        raise ValueError(f"copy must be None, True or False, got {copy!r}")


def memory(array: Any) -> tuple[int, int]:
    """Return the DLPack device of the memory that holds `array`.

    Parameters
    ----------
    array : object
        Any array; one without ``__dlpack_device__`` counts as host memory.

    Returns
    -------
    tuple of int
        ``(device_type, device_id)``; :data:`HOST` for host memory, also for
        CUDA/ROCm pinned host memory and Apple's unified memory.
    """
    get = getattr(array, "__dlpack_device__", None)
    if get is None:
        return HOST
    try:
        device_type, device_id = get()
    except Exception:  # noqa: BLE001 - e.g. a JAX tracer has no device
        return HOST
    device_type = int(device_type)
    if device_type in _HOST_DEVICE_TYPES:
        return HOST
    return (device_type, int(device_id))


def describe_memory(device: tuple[int, int]) -> str:
    """Name a DLPack device for error messages.

    Parameters
    ----------
    device : tuple of int
        ``(device_type, device_id)`` as returned by :func:`memory`.

    Returns
    -------
    str
        E.g. ``"the host"`` or ``"CUDA device 0"``.
    """
    if device == HOST:
        return "the host"
    name = _DEVICE_NAMES.get(device[0], f"DLPack device type {device[0]}")
    return f"{name} device {device[1]}"


def pointer(array: Any) -> int | None:
    """Return the address of the first element of `array`, or None if unknown.

    Parameters
    ----------
    array : object
        A NumPy, CuPy, JAX, PyTorch, MLX or Julia (juliacall) array.

    Returns
    -------
    int or None
        The address, or None when the array does not expose it.
    """
    getters = (
        lambda: array.data_ptr(),  # PyTorch
        lambda: array.unsafe_buffer_pointer(),  # JAX
        lambda: array.__cuda_array_interface__["data"][0],  # CuPy
        lambda: array.__array_interface__["data"][0],  # NumPy, juliacall
        lambda: np.asarray(memoryview(array)).ctypes.data,  # MLX
    )
    for get in getters:
        try:
            return int(get())
        except Exception:  # noqa: BLE001, S112 - try the next protocol
            continue
    return None


def shares_memory(source: Any, result: Any) -> bool:
    """Whether a conversion returned the memory of its source.

    Parameters
    ----------
    source, result : object
        The array that was converted and the converted array.

    Returns
    -------
    bool
        True for the same object, an empty array or the same first-element
        address; False otherwise, also when an address is unknown.
    """
    if result is source or getattr(result, "size", None) == 0:
        return True
    address = pointer(source)
    return address is not None and address == pointer(result)


def copy_error(source: Any, target: str, reason: str) -> ValueError:
    """Return the error for a conversion that needs a copy under ``copy=False``.

    Parameters
    ----------
    source : object
        The array that was converted.
    target : str
        The framework or array library converted to.
    reason : str
        Why a copy is needed.

    Returns
    -------
    ValueError
        The error to raise.
    """
    return ValueError(
        f"converting {type(source).__name__} ({_describe(source)}) to {target} needs "
        f"a copy ({reason}), but copy=False; pass copy=None to allow it"
    )


def device_error(source: Any, target: str, error: Exception) -> ValueError:
    """Return the error for a framework that cannot take an array's device.

    Parameters
    ----------
    source : object
        The array that was converted.
    target : str
        The framework converted to.
    error : Exception
        The framework's own error.

    Returns
    -------
    ValueError
        The error to raise.
    """
    return ValueError(
        f"{target} cannot take a {type(source).__name__} on "
        f"{describe_memory(memory(source))} ({error}); move it to a device "
        f"{target} supports first, e.g. to the host with xp.to_numpy"
    )


def dtype_name(dtype: Any) -> str:
    """Return a framework-neutral dtype name, e.g. ``"float64"``.

    Parameters
    ----------
    dtype : object
        A NumPy, CuPy, JAX, PyTorch or MLX dtype, or a dtype name.

    Returns
    -------
    str
        The name, e.g. ``"float32"`` for ``torch.float32`` and
        ``mlx.core.float32``.
    """
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype).rsplit(".", 1)[-1]


def check_dtype(source: Any, result: Any, dtype: Any, target: str, advice: str) -> None:
    """Raise when a conversion changed a dtype that was not asked for.

    Parameters
    ----------
    source, result : object
        The array that was converted and the converted array.
    dtype : object or None
        The requested dtype, or None to keep the source's.
    target : str
        The framework converted to.
    advice : str
        How to keep the dtype or ask for the cast.
    """
    expected = dtype if dtype is not None else getattr(source, "dtype", None)
    if expected is None:
        return
    wanted, got = dtype_name(expected), dtype_name(result.dtype)
    if wanted != got:
        raise TypeError(f"{target} would turn {wanted} into {got}; {advice}")


def _size(array: Any) -> int | None:
    """Payload bytes of an array of any framework, or None if unknown."""
    nbytes = getattr(array, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    try:
        return _nbytes(array)
    except AttributeError:
        return None


def record(source: Any, result: Any) -> None:
    """Record an adapter copy when transfer profiling is active.

    A copy from host to device is a ``to_device`` event, from device to host a
    ``to_host`` event, and one within a memory a ``device_copy`` event.

    Parameters
    ----------
    source, result : object
        The array that was converted and its copy.
    """
    if not _ACTIVE:
        return
    before, after = memory(source) == HOST, memory(result) == HOST
    if before == after:
        kind = "device_copy"
    else:
        kind = "to_device" if before else "to_host"
    _record(
        kind,
        f"interop {type(source).__name__} -> {type(result).__name__} "
        f"({_describe(result)})",
        nbytes=_size(result),
    )


def finish(
    source: Any,
    result: Any,
    *,
    copy: bool | None,
    target: str,
    duplicate: Callable[[Any], Any],
) -> Any:
    """Apply the `copy` rule to a converted array and record the copy.

    Parameters
    ----------
    source, result : object
        The array that was converted and the converted array.
    copy : bool or None
        None: share memory when the conversion did; False: raise if it did
        not; True: copy if it did.
    target : str
        The framework or array library converted to, for the error message.
    duplicate : callable
        Copies a result in its own framework.

    Returns
    -------
    object
        The result, copied for ``copy=True``.
    """
    shared = shares_memory(source, result)
    if copy is False and not shared:
        raise copy_error(source, target, "the conversion could not share memory")
    if copy and shared:
        result = duplicate(result)
        shared = False
    if not shared:
        record(source, result)
    return result


def to_backend(
    source: Any,
    backend: str,
    *,
    dtype: Any,
    copy: bool | None,
    host_array: Callable[[Any], Any],
) -> Any:
    """Convert a framework array to NumPy or CuPy (the adapters' ``from_framework``).

    DLPack is used when the source and the target share a memory; between
    host and device the array is copied (``copy=False`` raises).

    Parameters
    ----------
    source : object
        Framework array to convert.
    backend : {"numpy", "cupy"}
        Array library to construct.
    dtype : object or None
        Dtype for the result.
    copy : bool or None
        See :func:`finish`.
    host_array : callable
        Returns `source` as a NumPy array (a view where the framework allows),
        used when DLPack cannot.

    Returns
    -------
    object
        NumPy or CuPy array.
    """
    check_copy(copy)
    target = target_backend(backend)
    on_host = memory(source) == HOST
    if backend == "numpy":
        if not on_host:
            if copy is False:
                raise copy_error(source, "NumPy", "device to host")
            result = np.asarray(host_array(source))
        else:
            try:
                result = np.from_dlpack(source)
            except _DLPACK_ERRORS:
                result = np.asarray(host_array(source))
    elif on_host:
        if copy is False:
            raise copy_error(source, "CuPy", "host to device")
        result = target.asarray(host_array(source))
    else:
        try:
            result = target.from_dlpack(source)
        except _DLPACK_ERRORS as error:
            raise device_error(source, "CuPy", error) from error
    result = cast(result, dtype)
    return finish(
        source, result, copy=copy, target=backend, duplicate=lambda array: array.copy()
    )
