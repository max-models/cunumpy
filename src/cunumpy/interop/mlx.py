"""Optional interoperability with Apple's MLX arrays."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import cast, record, require, target_backend


def to_framework(array: Any, *, device: Any = None, dtype: Any = None) -> Any:
    """Convert a NumPy or CuPy array to an MLX array.

    Parameters
    ----------
    array : object
        Array to convert.
    device : object or None, optional
        MLX device on which to place the result.
    dtype : object or None, optional
        Dtype for the result.

    Returns
    -------
    object
        MLX array.
    """
    mlx = require("mlx.core", extra="mlx")
    if type(array).__module__.startswith("mlx"):
        result = array
    elif hasattr(mlx, "from_dlpack") and hasattr(array, "__dlpack__"):
        result = mlx.from_dlpack(array)
    else:
        result = mlx.array(array)
    result = cast(result, dtype)
    if device is not None:
        result = mlx.array(result, device=device)
    record(array, result)
    return result


def from_framework(array: Any, *, backend: str = "numpy", dtype: Any = None) -> Any:
    """Convert an MLX array to a NumPy or CuPy array.

    Parameters
    ----------
    array : object
        MLX array to convert.
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
    if hasattr(array, "__dlpack__") and hasattr(target, "from_dlpack"):
        result = target.from_dlpack(array)
    else:
        result = target.asarray(array)
    result = cast(result, dtype)
    record(array, result)
    return result


to_mlx = to_framework
from_mlx = from_framework

__all__ = ["from_framework", "from_mlx", "to_framework", "to_mlx"]
