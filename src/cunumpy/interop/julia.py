"""Optional interoperability with Julia arrays through juliacall.

A NumPy array becomes a Julia ``Array`` that shares its memory, so Julia code
(e.g. a KernelAbstractions.jl kernel on the CPU backend) reads and writes the
NumPy data in place. Julia is column-major: a C-ordered array arrives with its
dimensions reversed (NumPy ``a[i, j]`` is Julia ``A[j+1, i+1]``), an
F-ordered one with the same dimensions. A cunumpy component-major array of
shape ``(3, n)`` is therefore a Julia ``n x 3`` matrix with contiguous columns.
:func:`from_framework` reverses the dimensions back (``order="C"``) or keeps
them (``order="F"``).

Only host arrays so far; CuPy arrays to CUDA.jl ``CuArray`` through DLPack
are planned.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from cunumpy.interop._common import (
    HOST,
    check_copy,
    copy_error,
    device_error,
    finish,
    memory,
    record,
    require,
    to_backend,
)

_JULIA_TYPES = {
    "bool": "Bool",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "float16": "Float16",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "ComplexF32",
    "complex128": "ComplexF64",
}

# The Julia array keeps the NumPy array alive through its finalizer's closure.
_HELPERS = """
module CunumpyInterop
release(owner) = nothing
function wrap(::Type{T}, pointer::Integer, dims::Tuple, owner) where {T}
    array = unsafe_wrap(Array, Ptr{T}(UInt(pointer)), Tuple(Int.(dims)); own=false)
    finalizer(_ -> release(owner), array)
    return array
end
end
"""


@functools.cache
def _julia() -> Any:
    """Start Julia (once) and return ``juliacall.Main`` with the helpers defined."""
    juliacall = require("juliacall", extra="julia")
    juliacall.Main.seval(_HELPERS)
    return juliacall.Main


def _is_julia(array: Any) -> bool:
    return type(array).__module__.startswith("juliacall")


def to_framework(array: Any, *, dtype: Any = None, copy: bool | None = None) -> Any:
    """Convert a NumPy array to a Julia ``Array``.

    The Julia array shares the memory of a C- or F-contiguous, writeable
    array (see the module docstring for the order of the dimensions) and
    keeps it alive; any other array is first copied to C order.

    Parameters
    ----------
    array : object
        Host array to convert.
    dtype : object or None, optional
        NumPy dtype for the result (a copy when it differs).
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy.

    Returns
    -------
    juliacall.ArrayValue
        The Julia array.
    """
    jl = _julia()
    check_copy(copy)
    if _is_julia(array):
        if dtype is not None:
            raise TypeError("cast Julia arrays in Julia, e.g. Float32.(A)")
        return finish(array, array, copy=copy, target="Julia", duplicate=jl.copy)
    if memory(array) != HOST:
        raise device_error(
            array, "Julia", NotImplementedError("only host arrays so far")
        )
    host = np.asarray(array)
    if dtype is not None:
        host = host.astype(dtype, copy=False)
    name = _JULIA_TYPES.get(host.dtype.name)
    if name is None:
        raise TypeError(f"no Julia element type for dtype {host.dtype}")
    if not host.flags.writeable or not (
        host.flags.c_contiguous or host.flags.f_contiguous
    ):
        if copy is False:
            raise copy_error(array, "Julia", "not contiguous or read-only")
        host = np.array(host, order="C")
    dims = host.shape[::-1] if host.flags.c_contiguous else host.shape
    result = jl.CunumpyInterop.wrap(jl.seval(name), host.ctypes.data, dims, host)
    return finish(array, result, copy=copy, target="Julia", duplicate=jl.copy)


def from_framework(
    array: Any,
    *,
    backend: str = "numpy",
    dtype: Any = None,
    copy: bool | None = None,
    order: str = "C",
) -> Any:
    """Convert a Julia array to a NumPy or CuPy array.

    A NumPy result is a view of a Julia ``Array`` and keeps it alive; other
    Julia arrays (views, ranges) are first collected into an ``Array``.

    Parameters
    ----------
    array : juliacall.ArrayValue
        Julia array to convert.
    backend : {"numpy", "cupy"}, default="numpy"
        Array library to construct.
    dtype : object or None, optional
        Dtype for the result.
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy.
    order : {"C", "F"}, default="C"
        "C": reverse the dimensions, giving a C-ordered array (the inverse of
        :func:`to_framework` for a C-ordered array); "F": keep them, giving
        an F-ordered array.

    Returns
    -------
    object
        NumPy or CuPy array.
    """
    jl = _julia()
    if order not in ("C", "F"):
        raise ValueError(f"order must be 'C' or 'F', got {order!r}")
    if not _is_julia(array) or not jl.isa(array, jl.AbstractArray):
        raise TypeError(f"expected a Julia array, got {type(array).__name__}")
    source = array
    if not jl.isa(array, jl.Array):
        if copy is False:
            raise copy_error(array, backend, "not a Julia Array")
        source = jl.collect(array)
        record(array, source)

    def host_array(value: Any) -> np.ndarray:
        view = np.asarray(value)
        return view.T if order == "C" else view

    return to_backend(source, backend, dtype=dtype, copy=copy, host_array=host_array)


to_julia = to_framework
from_julia = from_framework

__all__ = ["from_framework", "from_julia", "to_framework", "to_julia"]
