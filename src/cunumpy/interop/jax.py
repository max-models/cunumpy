"""Optional interoperability with JAX arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy.interop._common import (
    HOST,
    cast,
    check_copy,
    check_dtype,
    device_error,
    finish,
    memory,
    require,
    to_backend,
)

_X64_ADVICE = (
    "enable float64 with cunumpy.interop.jax.enable_x64() before creating JAX arrays, "
    "or ask for the cast with dtype=..."
)


def enable_x64(enabled: bool = True) -> None:
    """Turn JAX's 64-bit types (``jax_enable_x64``) on or off.

    Without them JAX turns float64 into float32 and int64 into int32, which
    :func:`to_framework` refuses unless the cast is asked for. Call it before
    creating JAX arrays.

    Parameters
    ----------
    enabled : bool, default=True
        Whether JAX keeps 64-bit types.
    """
    jax = require("jax")
    jax.config.update("jax_enable_x64", enabled)


def supports_float64(device: Any = None) -> bool:
    """Whether JAX keeps float64 arrays on `device`.

    Parameters
    ----------
    device : jax.Device or None, optional
        The device to check; the default device when None.

    Returns
    -------
    bool
        True when ``jax_enable_x64`` is on and the device is not an Apple GPU
        (jax-metal has no float64).
    """
    jax = require("jax")
    if not jax.config.read("jax_enable_x64"):
        return False
    device = jax.devices()[0] if device is None else device
    return device.platform.lower() != "metal"


def to_framework(
    array: Any, *, device: Any = None, dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert a NumPy or CuPy array to a JAX array.

    DLPack shares the memory when JAX can (a contiguous array on a device JAX
    has; on the CPU, XLA also needs the buffer aligned, which small NumPy
    arrays may not be); a host array it cannot share is copied. `device` and `dtype` are
    explicit conversions. A float64 array without ``jax_enable_x64`` raises
    instead of becoming float32 (see :func:`enable_x64`).

    Parameters
    ----------
    array : object
        Array to convert.
    device : object or None, optional
        JAX device on which to place the result.
    dtype : object or None, optional
        Dtype for the result.
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy.

    Returns
    -------
    object
        JAX array.
    """
    jax = require("jax")
    import jax.numpy as jnp

    check_copy(copy)
    if type(array).__module__.startswith("jax"):
        result = array
    elif hasattr(array, "__dlpack__"):
        try:
            result = jax.dlpack.from_dlpack(array)
        except Exception as error:
            if memory(array) != HOST:
                raise device_error(array, "JAX", error) from error
            result = jnp.asarray(array)
    else:
        result = jnp.asarray(array)
    result = cast(result, dtype)
    if device is not None:
        result = jax.device_put(result, device)
    check_dtype(array, result, dtype, "JAX", _X64_ADVICE)
    return finish(
        array,
        result,
        copy=copy,
        target="JAX",
        duplicate=lambda value: jnp.array(value, copy=True),
    )


def from_framework(
    array: Any, *, backend: str = "numpy", dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert a JAX array to a NumPy or CuPy array.

    DLPack shares the memory when the JAX array is in the target's memory;
    between host and device the array is copied. A traced value (inside
    ``jax.jit``, ``jax.grad`` or ``jax.vmap``) has no values and raises.

    Parameters
    ----------
    array : object
        JAX array to convert.
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
    jax = require("jax")
    if isinstance(array, jax.core.Tracer):
        raise TypeError(
            "cannot convert a traced JAX value (inside jax.jit, jax.grad or jax.vmap): "
            "it has no concrete values; convert outside the transformation, or call "
            "the cunumpy code through jax.pure_callback"
        )
    return to_backend(array, backend, dtype=dtype, copy=copy, host_array=np.asarray)


to_jax = to_framework
from_jax = from_framework

__all__ = [
    "enable_x64",
    "from_framework",
    "from_jax",
    "supports_float64",
    "to_framework",
    "to_jax",
]
