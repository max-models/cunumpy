"""Optional interoperability with PyTorch tensors."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import cast, record, require, target_backend


def to_framework(array: Any, *, device: Any = None, dtype: Any = None) -> Any:
    """Convert a NumPy or CuPy array to a PyTorch tensor.

    Parameters
    ----------
    array : object
        Array to convert.
    device : object or None, optional
        PyTorch device on which to place the result.
    dtype : object or None, optional
        Dtype for the result.

    Returns
    -------
    object
        PyTorch tensor.
    """
    torch = require("torch")
    if type(array).__module__.startswith("torch"):
        result = array
    elif hasattr(torch, "from_dlpack") and hasattr(array, "__dlpack__"):
        result = torch.from_dlpack(array)
    else:
        result = torch.as_tensor(array)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    record(array, result)
    return result


def from_framework(array: Any, *, backend: str = "numpy", dtype: Any = None) -> Any:
    """Convert a PyTorch tensor to a NumPy or CuPy array.

    Parameters
    ----------
    array : object
        PyTorch tensor to convert.
    backend : {"numpy", "cupy"}, default="numpy"
        Array library to construct.
    dtype : object or None, optional
        Dtype for the result.

    Returns
    -------
    object
        NumPy or CuPy array.
    """
    target = target_backend(backend)
    if hasattr(target, "from_dlpack") and hasattr(array, "__dlpack__"):
        result = target.from_dlpack(array)
    else:
        result = target.asarray(array.detach().cpu().numpy())
    result = cast(result, dtype)
    record(array, result)
    return result


to_torch = to_framework
from_torch = from_framework

__all__ = ["from_framework", "from_torch", "to_framework", "to_torch"]
