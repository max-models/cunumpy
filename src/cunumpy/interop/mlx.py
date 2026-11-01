"""Optional interoperability with Apple's MLX arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy.interop._common import (
    HOST,
    check_copy,
    check_dtype,
    copy_error,
    device_error,
    dtype_name,
    finish,
    is_array,
    memory,
    require,
    scalar_value,
    to_backend,
    tree_map,
)

_GPU_ADVICE = (
    "MLX has float64 on the CPU only: pass device=mlx.core.cpu to keep float64, "
    "or ask for the cast with dtype=mlx.core.float32"
)


def supports_float64(device: Any = None) -> bool:
    """Whether MLX can compute with float64 arrays on `device`.

    Parameters
    ----------
    device : mlx.core.Device or None, optional
        The device to check; MLX's default device when None.

    Returns
    -------
    bool
        True on the CPU, False on the GPU.
    """
    mx = require("mlx.core", extra="mlx")
    device = mx.default_device() if device is None else device
    return device == mx.cpu


def to_framework(
    array: Any, *, device: Any = None, dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert a NumPy array to an MLX array.

    MLX always copies into its own memory, so ``copy=False`` raises. MLX
    arrays have no device; `device` is the device they will be used on, which
    decides whether float64 is kept (CPU) or raises (GPU) instead of
    becoming float32, as ``mlx.core.array`` does.

    Parameters
    ----------
    array : object
        Host array to convert.
    device : object or None, optional
        MLX device the array is meant for; MLX's default device when None.
    dtype : object or None, optional
        Dtype for the result.
    copy : bool or None, optional
        None or True: copy (MLX cannot share); False: raise.

    Returns
    -------
    object
        MLX array.
    """
    mx = require("mlx.core", extra="mlx")
    check_copy(copy)
    if type(array).__module__.startswith("mlx"):
        result = array if dtype is None else array.astype(dtype)
    else:
        if memory(array) != HOST:
            raise device_error(array, "MLX", TypeError("MLX takes host arrays"))
        if copy is False:
            raise copy_error(array, "MLX", "MLX copies into its own memory")
        source_dtype = getattr(array, "dtype", None)
        if (
            dtype is None
            and source_dtype is not None
            and dtype_name(source_dtype) == "float64"
            and supports_float64(device)
        ):
            dtype = mx.float64
        result = mx.array(array, dtype=dtype)
    check_dtype(array, result, dtype, "MLX", _GPU_ADVICE)
    return finish(
        array,
        result,
        copy=copy,
        target="MLX",
        duplicate=lambda value: mx.array(np.array(value)),
    )


def from_framework(
    array: Any, *, backend: str = "numpy", dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert an MLX array to a NumPy or CuPy array.

    A NumPy result shares the MLX array's memory (Apple's unified memory); a
    CuPy result is a copy.

    Parameters
    ----------
    array : object
        MLX array to convert.
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
    return to_backend(
        array,
        backend,
        dtype=dtype,
        copy=copy,
        host_array=lambda value: np.asarray(memoryview(value)),
    )


def scalar(value: Any, *, device: Any = None, dtype: Any = None) -> Any:
    """Return a scalar objective or diagnostic as a 0-d MLX array.

    A Python number becomes a NumPy scalar first (float64 for a float), so
    the dtype rules of :func:`to_framework` apply to it as well.

    Parameters
    ----------
    value : object
        A Python number, a NumPy or CuPy scalar, or an array of size one.
    device : object or None, optional
        As for :func:`to_framework`.
    dtype : object or None, optional
        Dtype for the result.

    Returns
    -------
    object
        0-d MLX array.
    """
    value = scalar_value(value)
    if not is_array(value):
        value = np.asarray(value)
    return to_framework(value, device=device, dtype=dtype)


def tree_to_framework(
    tree: Any, *, device: Any = None, dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert the arrays in nested dicts, lists and tuples to MLX arrays.

    The structure is kept and leaves that are not arrays (numbers, strings,
    None) are returned unchanged; each array goes through
    :func:`to_framework` with the same arguments.

    Parameters
    ----------
    tree : object
        An array, or dicts, lists and tuples of arrays.
    device, dtype, copy : object, optional
        As for :func:`to_framework`.

    Returns
    -------
    object
        The same structure with MLX arrays.
    """

    def convert(array: Any) -> Any:
        return to_framework(array, device=device, dtype=dtype, copy=copy)

    return tree_map(convert, tree, is_array)


def tree_from_framework(
    tree: Any, *, backend: str = "numpy", dtype: Any = None, copy: bool | None = None
) -> Any:
    """Convert the MLX arrays in nested dicts, lists and tuples to NumPy or CuPy.

    The structure is kept and other leaves are returned unchanged; each MLX
    array goes through :func:`from_framework` with the same arguments.

    Parameters
    ----------
    tree : object
        A MLX array, or dicts, lists and tuples of them.
    backend, dtype, copy : object, optional
        As for :func:`from_framework`.

    Returns
    -------
    object
        The same structure with NumPy or CuPy arrays.
    """
    mx = require("mlx.core", extra="mlx")

    def convert(array: Any) -> Any:
        return from_framework(array, backend=backend, dtype=dtype, copy=copy)

    return tree_map(convert, tree, lambda value: isinstance(value, mx.array))


to_mlx = to_framework
from_mlx = from_framework
tree_to_mlx = tree_to_framework
tree_from_mlx = tree_from_framework

__all__ = [
    "from_framework",
    "from_mlx",
    "scalar",
    "supports_float64",
    "to_framework",
    "to_mlx",
    "tree_from_framework",
    "tree_from_mlx",
    "tree_to_framework",
    "tree_to_mlx",
]
