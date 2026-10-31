"""Optional interoperability with PyTorch tensors."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import (
    HOST,
    check_copy,
    check_dtype,
    device_error,
    finish,
    memory,
    require,
    to_backend,
)

_MPS_ADVICE = (
    "Apple GPUs (MPS) have no float64; ask for the cast with dtype=torch.float32"
)


def supports_float64(device: Any = None) -> bool:
    """Whether PyTorch keeps float64 tensors on `device`.

    Parameters
    ----------
    device : torch.device, str or None, optional
        The device to check; the CPU when None.

    Returns
    -------
    bool
        False on an Apple GPU (``"mps"``), True otherwise.
    """
    torch = require("torch")
    return torch.device("cpu" if device is None else device).type != "mps"


def to_framework(
    array: Any, *, device: Any = None, dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert a NumPy or CuPy array to a PyTorch tensor.

    DLPack shares the memory when PyTorch can; a host array it cannot share is
    copied. `device` and `dtype` are explicit conversions. A float64 array
    for an Apple GPU raises instead of becoming float32.

    Parameters
    ----------
    array : object
        Array to convert.
    device : object or None, optional
        PyTorch device on which to place the result.
    dtype : object or None, optional
        Dtype for the result.
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy.

    Returns
    -------
    object
        PyTorch tensor.
    """
    torch = require("torch")
    check_copy(copy)
    if isinstance(array, torch.Tensor):
        result = array
    elif hasattr(array, "__dlpack__"):
        try:
            result = torch.from_dlpack(array)
        except Exception as error:
            if memory(array) != HOST:
                raise device_error(array, "PyTorch", error) from error
            result = torch.as_tensor(array)
    else:
        result = torch.as_tensor(array)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        if result.dtype == torch.float64 and not supports_float64(device):
            raise TypeError(f"PyTorch would turn float64 into float32; {_MPS_ADVICE}")
        result = result.to(device=device)
    check_dtype(array, result, dtype, "PyTorch", _MPS_ADVICE)
    return finish(
        array,
        result,
        copy=copy,
        target="PyTorch",
        duplicate=lambda value: value.clone(),
    )


def from_framework(
    array: Any, *, backend: str = "numpy", dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert a PyTorch tensor to a NumPy or CuPy array.

    DLPack shares the memory when the tensor is in the target's memory;
    between host and device (and from an Apple GPU) the tensor is copied. The
    result is detached from autograd.

    Parameters
    ----------
    array : object
        PyTorch tensor to convert.
    backend : {"numpy", "cupy"}, default="numpy"
        Array library to construct.
    dtype : object or None, optional
        Dtype for the result.
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy.

    Returns
    -------
    object
        NumPy or CuPy array.
    """
    detach = getattr(array, "detach", None)
    source = array if detach is None else detach()
    return to_backend(
        source,
        backend,
        dtype=dtype,
        copy=copy,
        host_array=lambda tensor: tensor.cpu().numpy(),
    )


to_torch = to_framework
from_torch = from_framework

__all__ = [
    "from_framework",
    "from_torch",
    "supports_float64",
    "to_framework",
    "to_torch",
]
