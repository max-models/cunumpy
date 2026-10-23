"""Optional interoperability with JAX arrays."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import cast, record, require, target_backend


def to_framework(array: Any, *, device: Any = None, dtype: Any = None) -> Any:
    """Convert a NumPy or CuPy array to a JAX array.

    DLPack is used when available; `device` and `dtype` are explicit optional
    conversions.

    Parameters
    ----------
    array : object
        Array to convert.
    device : object or None, optional
        JAX device on which to place the result.
    dtype : object or None, optional
        Dtype for the result.

    Returns
    -------
    object
        JAX array.
    """
    jax = require("jax")
    import jax.numpy as jnp

    if type(array).__module__.startswith("jax"):
        result = array
    elif hasattr(array, "__dlpack__"):
        result = jax.dlpack.from_dlpack(array)
    else:
        result = jnp.asarray(array)
    result = cast(result, dtype)
    if device is not None:
        result = jax.device_put(result, device)
    record(array, result)
    return result


def from_framework(array: Any, *, backend: str = "numpy", dtype: Any = None) -> Any:
    """Convert a JAX array to a NumPy or CuPy array.

    Parameters
    ----------
    array : object
        JAX array to convert.
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
        result = target.asarray(array)
    result = cast(result, dtype)
    record(array, result)
    return result


to_jax = to_framework
from_jax = from_framework

__all__ = ["from_framework", "from_jax", "to_framework", "to_jax"]
