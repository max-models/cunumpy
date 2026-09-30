"""CUDA kernels (``cupy.RawKernel``) called like their NumPy/Pyccel counterparts.

:class:`CudaKernel` wraps a CUDA C kernel so that it can be called with the same
arguments as the host kernel it mirrors:

* argument objects that implement the :class:`CudaArguments` protocol
  (a ``__cuda_args__()`` method) are flattened into their device arrays and
  scalars, so an object holding several arrays can be passed as one argument;
* C structs can be passed by value: :class:`CudaStruct` defines the struct once,
  generates its C declaration and packs its values;
* the ``__global__`` signature is parsed once, and every call is checked against
  it: the number of arguments, the dtype of every array, every struct, and
  every scalar. Python scalars are cast to the declared C type; a scalar that
  does not fit the declared type (a ``float`` for an ``int``, an integer out of
  range, a NumPy scalar that would lose precision) raises instead of reaching
  the kernel as a silently wrong value, which is what ``cupy.RawKernel`` would
  do;
* arrays are never converted or copied: they must already be CuPy arrays;
* strided array views: a parameter or struct field of type ``Array2D<double>``
  (from the shipped header ``cunumpy/array_view.cuh``, see
  :func:`cuda_include_dir`) takes a 2D CuPy array, contiguous or not, and
  receives its pointer, shape and strides, so that kernels index ``a(i, j)``
  like the pyccel kernels they are ported from;
* C++ function templates are instantiated with ``template_args``, and generated
  kernels (one source per variant) are compiled once per variant by
  :class:`CudaKernelVariants`.

The launch shape is given at each call, either as the number of threads
(``n_threads``, in 1 to 3 dimensions) or as an explicit ``grid``.

Argument structs can be generated from the annotations of a Python class
(:meth:`CudaStruct.from_signature`) and written to a header
(:meth:`CudaStruct.to_header`, :func:`write_cuda_header`), so that the Python
class is the one definition of the arguments.

This module imports CuPy only when a kernel is compiled, so it can be imported
(and signatures parsed) without CuPy.
"""

from __future__ import annotations

import inspect
import math
import re
import typing
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructValue",
    "ctype_of",
    "cuda_include_dir",
    "parse_cuda_signature",
    "write_cuda_header",
]

# CUDA limit on the number of threads per block
_MAX_THREADS_PER_BLOCK = 1024

# Headers shipped with cunumpy: #include <cunumpy/array_view.cuh> etc.
_CUDA_INCLUDE_DIR = Path(__file__).resolve().parent / "cuda" / "include"
_ARRAY_VIEW_INCLUDE = '#include "cunumpy/array_view.cuh"'


def cuda_include_dir() -> str:
    """The directory of the CUDA headers shipped with cunumpy.

    :class:`CudaKernel` adds it to the include path automatically, so kernels
    can ``#include "cunumpy/array_view.cuh"`` (strided ``Array1D<T>``,
    ``Array2D<T>``, ``Array3D<T>`` views passed by value) and
    ``#include "cunumpy/index.cuh"`` (thread-index and grid-stride macros such
    as ``CUNUMPY_THREAD_1D(i, n)``). Pass it as ``-I`` to other compilers.
    """
    return str(_CUDA_INCLUDE_DIR)


class CudaArguments:
    """Base class for objects passed to a :class:`CudaKernel` as one argument.

    A :class:`CudaKernel` replaces every argument that has a ``__cuda_args__()``
    method by the values it returns, in order. Subclassing this class is
    optional: any object implementing ``__cuda_args__()`` is flattened.

    Parameters
    ----------
    *values
        The CUDA kernel arguments this object stands for: CuPy arrays and
        scalars, in the order of the kernel signature.

    Examples
    --------
    >>> class Particles(CudaArguments):
    ...     def __init__(self, positions, velocities):
    ...         self.positions = positions
    ...         super().__init__(positions, velocities, positions.shape[0])
    >>> kernel(dt, Particles(x, v), n_threads=x.shape[0])  # doctest: +SKIP
    """

    def __init__(self, *values: Any) -> None:
        self._cuda_args = tuple(values)

    def __cuda_args__(self) -> tuple[Any, ...]:
        """The CUDA kernel arguments this object stands for."""
        return self._cuda_args


class CudaParameter(NamedTuple):
    """One parameter of a CUDA kernel signature (or one field of a struct).

    Attributes
    ----------
    name : str
        Parameter name.
    ctype : str
        Normalized C type without qualifiers or ``*``, e.g. ``"double"`` or
        ``"Array2D<double>"``.
    dtype : numpy.dtype | None
        NumPy dtype of the value (or of the pointed-to elements, or of the
        elements of an array view; the structured dtype for a struct);
        ``None`` for ``void*``.
    pointer : bool
        Whether the parameter is a pointer (a device array).
    struct : CudaStruct | None
        The struct type, for a struct passed by value.
    view_ndim : int | None
        The number of dimensions, for an array view (``Array1D<T>`` to
        ``Array3D<T>``, see :func:`cuda_include_dir`) passed by value.
    """

    name: str
    ctype: str
    dtype: np.dtype | None
    pointer: bool
    struct: CudaStruct | None = None
    view_ndim: int | None = None


# C types (after removing qualifiers) and their NumPy dtypes. ``long`` is 64 bit,
# as on Linux (LP64), the platform CUDA runs on in practice.
_CTYPES = {
    "bool": np.bool_,
    "char": np.int8,
    "signed char": np.int8,
    "unsigned char": np.uint8,
    "short": np.int16,
    "short int": np.int16,
    "unsigned short": np.uint16,
    "unsigned short int": np.uint16,
    "int": np.int32,
    "signed": np.int32,
    "signed int": np.int32,
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "long": np.int64,
    "long int": np.int64,
    "long long": np.int64,
    "long long int": np.int64,
    "unsigned long": np.uint64,
    "unsigned long int": np.uint64,
    "unsigned long long": np.uint64,
    "unsigned long long int": np.uint64,
    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "uint8_t": np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,
    "size_t": np.uint64,
    "ptrdiff_t": np.int64,
    "ssize_t": np.int64,
    "float": np.float32,
    "double": np.float64,
    "complex<float>": np.complex64,
    "complex<double>": np.complex128,
}

# The C type used for each NumPy dtype, see ctype_of()
_CTYPE_OF = {
    np.dtype(np.bool_): "bool",
    np.dtype(np.int8): "signed char",
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "unsigned int",
    np.dtype(np.int64): "long long",
    np.dtype(np.uint64): "unsigned long long",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "complex<float>",
    np.dtype(np.complex128): "complex<double>",
}

_QUALIFIERS = {"const", "volatile", "__restrict__", "__restrict", "restrict"}

_COMPLEX = re.compile(r"(?:(?:thrust|cuda::std)::)?complex\s*<\s*(float|double)\s*>")
# Array1D<T> to Array3D<T> (cunumpy/array_view.cuh), T a scalar type of _CTYPES
_VIEW = re.compile(r"\bArray([123])D\s*<((?:[^<>]|complex<[^<>]*>)+?)>")
_TOKEN = re.compile(
    r"Array[123]D<[^<>]*(?:<[^<>]*>[^<>]*)?>|complex<(?:float|double)>"
    r"|[A-Za-z_]\w*|\*|\[\s*\]"
)


def _normalize_view(match: re.Match) -> str:
    """``Array2D< const double >`` -> ``Array2D<double>``."""
    words = [w for w in match.group(2).split() if w not in _QUALIFIERS]
    return f"Array{match.group(1)}D<{' '.join(words)}>"


def _view_dtype(ndim: int) -> np.dtype:
    """The structured dtype with the C layout of ``Array<ndim>D<T>``."""
    return np.dtype(
        [
            ("data", np.uint64),
            ("shape", np.int64, (ndim,)),
            ("strides", np.int64, (ndim,)),
        ],
        align=True,
    )


def ctype_of(dtype: Any) -> str:
    """The C type of a NumPy dtype, e.g. ``ctype_of(np.float64) == "double"``.

    Useful to generate CUDA source or template arguments for a given dtype.
    Complex dtypes map to ``complex<float>``/``complex<double>`` (include
    ``<cupy/complex.cuh>`` in the source).
    """
    try:
        return _CTYPE_OF[np.dtype(dtype)]
    except (KeyError, TypeError):
        raise ValueError(f"no C type for dtype {dtype!r}") from None


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


def _parse_parameter(
    text: str, structs: dict[str, CudaStruct] | None = None
) -> CudaParameter:
    text = _COMPLEX.sub(lambda m: f"complex<{m.group(1)}>", text)
    text = _VIEW.sub(_normalize_view, text)
    tokens = _TOKEN.findall(text)
    pointers = sum(1 for t in tokens if t == "*" or t.startswith("["))
    words = [t for t in tokens if t != "*" and not t.startswith("[")]
    words = [t for t in words if t not in _QUALIFIERS]
    if words[:1] == ["struct"]:
        words = words[1:]
    if len(words) < 2:
        raise ValueError(f"cannot parse the kernel parameter {text.strip()!r}")
    name, ctype = words[-1], " ".join(words[:-1])

    if structs and ctype in structs:
        if pointers:
            raise ValueError(
                f"cannot check the kernel parameter {text.strip()!r}: structs can "
                "only be passed by value"
            )
        struct = structs[ctype]
        return CudaParameter(name, ctype, struct.dtype, False, struct)
    view = _VIEW.fullmatch(ctype)
    if view is not None:
        element = view.group(2)
        if pointers or element not in _CTYPES:
            raise ValueError(
                f"cannot check the kernel parameter {text.strip()!r}: array views "
                f"take a scalar element type and are passed by value"
            )
        ndim = int(view.group(1))
        return CudaParameter(name, ctype, np.dtype(_CTYPES[element]), False, None, ndim)
    if ctype == "void" and pointers == 1:
        return CudaParameter(name, ctype, None, True)
    if pointers > 1 or ctype not in _CTYPES:
        raise ValueError(
            f"cannot check the kernel parameter {text.strip()!r}: unsupported type "
            f"{ctype + '*' * pointers!r}"
        )
    return CudaParameter(name, ctype, np.dtype(_CTYPES[ctype]), pointers == 1)


def _split_top_level(text: str) -> list[str]:
    """Split at commas that are not inside ``<...>`` (e.g. ``complex<double>``)."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _template_arg(value: Any) -> str:
    """A template argument as C++ source: a C type for dtypes, else a literal."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return ctype_of(value)


def parse_cuda_signature(
    source: str,
    name: str,
    *,
    structs: Iterable[CudaStruct] = (),
    template_args: Sequence[Any] | None = None,
) -> tuple[CudaParameter, ...]:
    """Parse the parameters of the ``__global__`` function `name` in `source`.

    Parameters
    ----------
    source : str
        CUDA C source code.
    name : str
        Name of the ``__global__`` function.
    structs : Iterable[CudaStruct]
        Struct types that may appear as parameters (passed by value). If the
        source defines a struct of the same name, its fields must match.
    template_args : Sequence | None
        Template arguments, if `name` is a function template: C types (or NumPy
        dtypes, see :func:`ctype_of`) for type parameters, integers or bools for
        non-type parameters. They are substituted into the parameter list.

    Returns
    -------
    tuple[CudaParameter, ...]
        The parameters, in order.

    Raises
    ------
    ValueError
        If there is no such function, a template is used without (the right
        number of) `template_args`, a struct definition in the source does not
        match its :class:`CudaStruct`, or a parameter has a type that cannot be
        checked (e.g. a macro or a pointer to pointer).
    """
    code = _strip_comments(source)
    structs = {s.name: s for s in structs}
    for struct in structs.values():
        struct.check_source(code)

    pattern = r"(?:template\s*<(?P<template>[^{};]*?)>\s*)?__global__\s+void\s+"
    match = re.search(pattern + re.escape(name) + r"\s*\(", code)
    if match is None:
        raise ValueError(f"no __global__ function {name!r} found in the CUDA source")

    depth, start = 1, match.end()
    for pos in range(start, len(code)):
        if code[pos] == "(":
            depth += 1
        elif code[pos] == ")":
            depth -= 1
            if depth == 0:
                break
    else:
        raise ValueError(f"unbalanced parentheses in the signature of {name!r}")
    params = code[start:pos].strip()

    template = match.group("template")
    if template is not None:
        template_params = [p.split() for p in _split_top_level(template) if p.strip()]
        if template_args is None or len(template_args) != len(template_params):
            raise ValueError(
                f"{name!r} is a template with {len(template_params)} parameters; "
                f"pass them as template_args"
            )
        for words, value in zip(template_params, template_args):
            if words[0] in ("typename", "class"):
                params = re.sub(
                    r"\b" + re.escape(words[-1]) + r"\b", _template_arg(value), params
                )
    elif template_args:
        raise ValueError(f"{name!r} is not a template, but template_args were given")

    if params in ("", "void"):
        return ()
    return tuple(_parse_parameter(p, structs) for p in _split_top_level(params))


def _describe(param: CudaParameter, index: int) -> str:
    ctype = param.ctype + ("*" if param.pointer else "")
    return f"argument {index} ({ctype} {param.name})"


def _check_device_array(param: CudaParameter, index: int, value: Any) -> None:
    """Raise unless `value` is a device array of the declared dtype."""
    # checked on the class: on the instance, CuPy builds the whole interface dict
    if not hasattr(type(value), "__cuda_array_interface__") and not hasattr(
        value, "__cuda_array_interface__"
    ):
        raise TypeError(
            f"{_describe(param, index)} must be a CuPy array, got "
            f"{type(value).__name__}; arrays are never copied to the device"
        )
    if param.dtype is not None and value.dtype != param.dtype:
        raise TypeError(
            f"{_describe(param, index)} must have dtype {param.dtype}, got "
            f"{value.dtype}"
        )


def _pointer_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for a pointer parameter: a device array with the right dtype."""

    def check(value: Any) -> Any:
        _check_device_array(param, index, value)
        return value

    return check


def _view_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for an array view: packs (pointer, shape, strides) of a device array.

    Strides are converted from bytes to elements; the array need not be
    contiguous.
    """
    ndim = param.view_ndim
    dtype = _view_dtype(ndim)

    def check(value: Any) -> Any:
        _check_device_array(param, index, value)
        if value.ndim != ndim:
            raise TypeError(
                f"{_describe(param, index)} must be a {ndim}D array, got "
                f"{value.ndim}D"
            )
        itemsize = value.dtype.itemsize
        strides = [s // itemsize for s in value.strides]
        if any(s * itemsize != stride for s, stride in zip(strides, value.strides)):
            raise TypeError(
                f"{_describe(param, index)}: strides {tuple(value.strides)} are not "
                f"multiples of the element size {itemsize}"
            )
        packed = np.zeros((), dtype=dtype)
        packed["data"] = value.data.ptr
        packed["shape"] = value.shape
        packed["strides"] = strides
        return packed[()]

    return check


def _scalar_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for a scalar parameter: casts to the declared type, or raises."""
    dtype = param.dtype
    kind = dtype.kind
    scalar_type = dtype.type
    if kind in "iu":
        info = np.iinfo(dtype)
        low, high = int(info.min), int(info.max)

    def cast_int(value: int) -> Any:
        if kind in "iu":
            if not low <= value <= high:
                raise OverflowError(
                    f"{_describe(param, index)}: {value} is out of range "
                    f"[{low}, {high}]"
                )
            return scalar_type(value)
        if kind in "fc":
            return scalar_type(value)
        raise TypeError(
            f"{_describe(param, index)} cannot take a value of type "
            f"{type(value).__name__}"
        )

    def check(value: Any) -> Any:
        value_type = type(value)
        # fast paths for the common cases
        if value_type is scalar_type:
            return value
        if value_type is int:
            return cast_int(value)
        if value_type is float and kind in "fc":
            return scalar_type(value)

        if isinstance(value, np.generic):
            if value.dtype == dtype:
                return value
            if np.can_cast(value.dtype, dtype, casting="safe"):
                return scalar_type(value)
            raise TypeError(
                f"{_describe(param, index)} cannot take a {value.dtype} scalar "
                f"without losing information"
            )
        # bool is a subclass of int, but must not be cast like one
        is_bool = isinstance(value, bool)
        if is_bool and kind in "biu":
            return scalar_type(value)
        if isinstance(value, int) and not is_bool:
            return cast_int(int(value))
        if isinstance(value, float) and kind in "fc":
            return scalar_type(value)
        if isinstance(value, complex) and kind == "c":
            return scalar_type(value)
        raise TypeError(
            f"{_describe(param, index)} cannot take a value of type "
            f"{value_type.__name__}"
        )

    return check


def _struct_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for a struct parameter: a packed value of that struct type."""
    dtype = param.dtype

    def check(value: Any) -> Any:
        if isinstance(value, np.void) and value.dtype == dtype:
            return value
        raise TypeError(
            f"{_describe(param, index)} must be a value of struct {param.ctype} "
            f"(created with that CudaStruct), got {type(value).__name__}"
        )

    return check


def _checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    if param.struct is not None:
        return _struct_checker(param, index)
    if param.view_ndim is not None:
        return _view_checker(param, index)
    if param.pointer:
        return _pointer_checker(param, index)
    return _scalar_checker(param, index)


def _field_dtype(field: CudaParameter) -> np.dtype:
    """The dtype of a struct field: pointers are stored as device addresses."""
    if field.pointer:
        return np.dtype(np.uint64)
    if field.view_ndim is not None:
        return _view_dtype(field.view_ndim)
    return field.dtype


# pyccel scalar annotations and their C types; ``int`` is set by ``int_type``
_PYCCEL_SCALARS = {
    "float": "double",
    "float64": "double",
    "float32": "float",
    "bool": "bool",
    "int64": "long long",
    "int32": "int",
    "int16": "short",
    "int8": "signed char",
    "complex": "complex<double>",
    "complex128": "complex<double>",
    "complex64": "complex<float>",
}
_PYCCEL_ANNOTATION = re.compile(
    r"^(?:(?:typing\.)?Final\s*\[\s*)?(?:const\s+)?(?P<scalar>\w+)\s*"
    r"(?:\[(?P<dims>[\s:,]*)\])?\s*\]?$"
)


def _pyccel_ctype(annotation: Any, scalars: Mapping[str, str], what: str) -> str:
    """The C type for a pyccel-style annotation, e.g. ``"float[:, :]"``."""
    if annotation is inspect.Parameter.empty:
        raise ValueError(f"{what} has no type annotation")
    if typing.get_origin(annotation) is typing.Final:
        (annotation,) = typing.get_args(annotation)
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        match = _PYCCEL_ANNOTATION.match(annotation.strip())
        if match is None:
            raise ValueError(f"{what}: cannot parse the annotation {annotation!r}")
        scalar, dims = match.group("scalar"), match.group("dims")
        ndim = 0 if dims is None else len(dims.split(","))
    elif annotation in (int, float, bool, complex):
        scalar, ndim = annotation.__name__, 0
    elif isinstance(annotation, type) and issubclass(annotation, np.generic):
        scalar, ndim = np.dtype(annotation).name, 0
    else:
        raise ValueError(f"{what}: unsupported annotation {annotation!r}")
    if scalar not in scalars:
        raise ValueError(
            f"{what}: unsupported scalar type {scalar!r} in {annotation!r}"
        )
    ctype = scalars[scalar]
    if ndim == 0:
        return ctype
    if ndim > 3:
        raise ValueError(f"{what}: arrays have at most 3 dimensions, got {ndim}")
    return f"Array{ndim}D<{ctype}>"


def _header_guard(name: str) -> str:
    guard = re.sub(r"\W", "_", name).upper().strip("_")
    return guard if re.match(r"[A-Z_]", guard) else f"_{guard}"


def _header_source(
    structs: Sequence[CudaStruct], guard: str, includes: Iterable[str]
) -> str:
    includes = list(includes)
    if any(f.view_ndim is not None for s in structs for f in s.fields):
        includes.insert(0, _ARRAY_VIEW_INCLUDE)
    lines = [
        "// Generated by cunumpy.CudaStruct from the Python definition; do not edit.",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
    ]
    if includes:
        lines += [
            inc if inc.startswith("#include") else f'#include "{inc}"'
            for inc in includes
        ] + [""]
    for struct in structs:
        lines += [struct.declaration]
    lines += [f"#endif  // {guard}", ""]
    return "\n".join(lines)


def write_cuda_header(
    path: str | Path,
    structs: Iterable[CudaStruct],
    guard: str | None = None,
    *,
    includes: Iterable[str] = (),
) -> str:
    """Write the declarations of several structs to one header file.

    The header has an include guard, ``#include "cunumpy/array_view.cuh"`` if
    a struct has array view fields, then the struct definitions in order.
    Generate the header at build or test time from the Python definitions,
    and commit it next to the kernels; a test can regenerate it and compare
    (see :meth:`CudaStruct.to_header`).

    Parameters
    ----------
    path : str | Path
        File to write.
    structs : Iterable[CudaStruct]
        The structs, in the order they are declared.
    guard : str | None
        Include guard macro; by default from the file name
        (``pusher_args.cuh`` -> ``PUSHER_ARGS_CUH``).
    includes : Iterable[str]
        Additional headers to include, as file names or ``#include`` lines.

    Returns
    -------
    str
        The header source that was written.
    """
    path = Path(path)
    source = _header_source(tuple(structs), guard or _header_guard(path.name), includes)
    path.write_text(source)
    return source


class CudaStruct:
    """A C struct type passed to CUDA kernels by value.

    Defines the struct once: :attr:`declaration` is its C definition, to put in
    the CUDA source (or a header), and calling the struct packs values into it.
    The packed value has the memory layout of the C struct (the NumPy structured
    dtype with C alignment), pointers are stored as device addresses. Kernels
    that are told about the struct (``CudaKernel(..., structs=[...])``) check
    that they get a value of the right struct, and that a definition of the
    struct in their source matches.

    Instead of a long flat parameter list, a group of arguments (e.g. all arrays
    describing particles) becomes one struct parameter; adding a field then
    changes one definition instead of every kernel signature.

    Parameters
    ----------
    name : str
        Name of the struct type in C.
    fields : Sequence[tuple[str, str]]
        ``(field name, C type)`` pairs, in order, e.g. ``("x", "double*")`` or
        ``("n", "int")``. Scalar fields, pointers to the scalar types of
        :func:`ctype_of` (or ``void*``), and array views ``Array1D<T>`` to
        ``Array3D<T>`` of those scalar types (from ``cunumpy/array_view.cuh``,
        packed as pointer, shape and strides in elements) are supported.

    Examples
    --------
    >>> Vec = CudaStruct("Vec", [("data", "double*"), ("n", "int")])
    >>> source = Vec.declaration + r'''
    ... extern "C" __global__ void scale(Vec v, double a) {
    ...     int i = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (i < v.n) v.data[i] *= a;
    ... }'''
    >>> scale = CudaKernel(source, "scale", structs=[Vec])
    >>> scale(Vec(data=x, n=x.size), 2.0, n_threads=x.size)  # doctest: +SKIP

    With array views, the kernel indexes like the pyccel kernel it mirrors:

    >>> Markers = CudaStruct("Markers", [("markers", "Array2D<double>"), ("n", "int")])
    >>> source = '#include "cunumpy/array_view.cuh"\n' + Markers.declaration + r'''
    ... extern "C" __global__ void push(Markers m, double dt) {
    ...     int ip = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (ip < m.n) m.markers(ip, 0) += dt * m.markers(ip, 3);
    ... }'''
    >>> push = CudaKernel(source, "push", structs=[Markers])
    >>> push(Markers(markers=markers, n=markers.shape[0]), 0.1, n_threads=markers.shape[0])  # doctest: +SKIP
    """

    def __init__(self, name: str, fields: Sequence[tuple[str, str]]) -> None:
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"invalid struct name {name!r}")
        parsed = [_parse_parameter(f"{ctype} {fname}") for fname, ctype in fields]
        names = [f.name for f in parsed]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in struct {name!r}: {names}")
        self._name = name
        self._fields = tuple(parsed)
        self._dtype = np.dtype(
            [(f.name, _field_dtype(f)) for f in parsed],
            align=True,
        )
        self._checkers = {f.name: _checker(f, i) for i, f in enumerate(parsed)}

    @classmethod
    def from_signature(
        cls,
        func: Callable[..., Any],
        name: str,
        *,
        int_type: str = "long long",
        scalar_names: Mapping[str, str] | None = None,
    ) -> CudaStruct:
        """Build the struct from the annotated parameters of a Python function.

        One field per parameter of `func` (``self`` is skipped), in order,
        with the C type given by the parameter's annotation, written in the
        pyccel style: ``"float"`` -> ``double``, ``"int"`` -> `int_type`,
        ``"bool"`` -> ``bool``, and an array ``"float[:, :]"`` ->
        ``Array2D<double>`` (``Final[...]`` and ``const`` are ignored). The
        annotations may also be the real types ``int``, ``float``, ``bool``
        (or NumPy scalar types such as ``np.float32``). Typically `func` is the
        ``__init__`` of an argument class, so that the class is the one
        definition of the arguments on the host and on the device.

        Parameters
        ----------
        func : Callable
            The function whose parameters define the fields.
        name : str
            Name of the struct type in C.
        int_type : str
            C type for ``int`` annotations: pyccel integers are 64 bit, so
            ``"long long"`` by default; ``"int"`` for 32 bit.
        scalar_names : Mapping[str, str] | None
            Additional (or changed) mappings from annotation scalar names to
            C types, e.g. ``{"float": "float"}`` for single precision.

        Raises
        ------
        ValueError
            A parameter without annotation, or with an annotation that cannot
            be mapped (an unknown scalar, more than 3 dimensions).

        Examples
        --------
        >>> class MarkerArguments:
        ...     def __init__(self, markers: "float[:, :]", n_markers: "int", valid: "bool[:]"):
        ...         ...
        >>> MarkerArgs = CudaStruct.from_signature(MarkerArguments.__init__, "MarkerArgs")
        >>> print(MarkerArgs.declaration)
        struct MarkerArgs {
            Array2D<double> markers;
            long long n_markers;
            Array1D<bool> valid;
        };
        """
        scalars = dict(_PYCCEL_SCALARS, int=int_type)
        if scalar_names:
            scalars.update(scalar_names)
        fields = []
        for param in inspect.signature(func).parameters.values():
            if param.name == "self" or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            what = f"parameter {param.name!r} of {getattr(func, '__qualname__', func)}"
            fields.append((param.name, _pyccel_ctype(param.annotation, scalars, what)))
        return cls(name, fields)

    def __repr__(self) -> str:
        return f"CudaStruct({self._name!r}, {len(self._fields)} fields)"

    @property
    def name(self) -> str:
        """Name of the struct type in C."""
        return self._name

    @property
    def fields(self) -> tuple[CudaParameter, ...]:
        """The fields, in order."""
        return self._fields

    @property
    def dtype(self) -> np.dtype:
        """The NumPy structured dtype with the C memory layout of the struct."""
        return self._dtype

    @property
    def has_views(self) -> bool:
        """Whether a field is an array view (``cunumpy/array_view.cuh`` is needed)."""
        return any(f.view_ndim is not None for f in self._fields)

    @property
    def declaration(self) -> str:
        """The C definition of the struct, to include in the CUDA source.

        A struct with array view fields needs
        ``#include "cunumpy/array_view.cuh"`` before the definition (see
        :attr:`has_views`); :meth:`to_header` adds it.
        """
        lines = [
            f"    {f.ctype}{'*' if f.pointer else ''} {f.name};" for f in self._fields
        ]
        return f"struct {self._name} {{\n" + "\n".join(lines) + "\n};\n"

    def to_header(
        self,
        path: str | Path | None = None,
        *,
        guard: str | None = None,
        includes: Iterable[str] = (),
    ) -> str:
        """The struct definition as a header file, with include guard.

        The header includes ``cunumpy/array_view.cuh`` if the struct has array
        view fields, then any `includes`, then the definition. A generated
        header committed next to the kernels stays in sync with the Python
        definition through a test::

            def test_header_is_up_to_date():
                assert Path("pusher_args.cuh").read_text() == MarkerArgs.to_header()

        Parameters
        ----------
        path : str | Path | None
            If given, the header is also written to this file.
        guard : str | None
            Include guard macro; by default ``<NAME>_CUH`` from the struct
            name.
        includes : Iterable[str]
            Additional headers to include, as file names or ``#include`` lines.

        Returns
        -------
        str
            The header source.
        """
        source = _header_source(
            (self,), guard or _header_guard(f"{self._name}_cuh"), includes
        )
        if path is not None:
            Path(path).write_text(source)
        return source

    def check_source(self, source: str) -> None:
        """Check a definition of this struct in `source` against its fields.

        Does nothing if `source` does not define the struct (e.g. because it is
        in a header).

        Raises
        ------
        ValueError
            If the definition in `source` has other fields or types.
        """
        code = _strip_comments(source)
        match = re.search(
            r"struct\s+" + re.escape(self._name) + r"\s*\{(.*?)\}", code, re.DOTALL
        )
        if match is None:
            return
        members = [m for m in match.group(1).split(";") if m.strip()]
        try:
            found = [_parse_parameter(m) for m in members]
        except ValueError as exc:
            raise ValueError(
                f"cannot compare the definition of struct {self._name!r}: {exc}"
            ) from None
        key = [(f.name, f.ctype, f.pointer) for f in self._fields]
        if [(f.name, f.ctype, f.pointer) for f in found] != key:
            raise ValueError(
                f"the definition of struct {self._name!r} in the CUDA source does not "
                f"match its CudaStruct:\n{self.declaration}"
            )

    def __call__(self, **values: Any) -> CudaStructValue:
        """Pack values into the struct.

        Pointer fields take CuPy arrays of the declared dtype (never copied),
        array view fields take CuPy arrays of the declared dtype and number of
        dimensions (contiguous or not; their pointer, shape and strides are
        packed), scalar fields are checked and cast like scalar kernel
        arguments.

        Raises
        ------
        TypeError
            A missing or unknown field, or a value of the wrong type.
        """
        missing = [f.name for f in self._fields if f.name not in values]
        unknown = [k for k in values if k not in self._checkers]
        if missing or unknown:
            raise TypeError(
                f"struct {self._name}: missing fields {missing}, unknown fields "
                f"{unknown}"
            )
        packed = np.zeros((), dtype=self._dtype)
        for field in self._fields:
            value = self._checkers[field.name](values[field.name])
            packed[field.name] = value.data.ptr if field.pointer else value
        return CudaStructValue(self, packed[()], values)


class CudaStructValue:
    """A value of a :class:`CudaStruct`, passed to kernels as one argument.

    Holds the packed struct and references to the arrays it points to (the
    struct itself only stores device addresses), and flattens into the packed
    struct through ``__cuda_args__()``. Field values are available as
    ``value["name"]``.
    """

    def __init__(self, struct: CudaStruct, packed: np.void, values: dict) -> None:
        self._struct = struct
        self._packed = packed
        self._values = dict(values)

    def __repr__(self) -> str:
        return f"<{self._struct.name} value>"

    @property
    def struct(self) -> CudaStruct:
        """The struct type."""
        return self._struct

    @property
    def packed(self) -> np.void:
        """The packed struct, with the memory layout of the C struct."""
        return self._packed

    def __getitem__(self, field: str) -> Any:
        return self._values[field]

    def __cuda_args__(self) -> tuple[np.void]:
        return (self._packed,)


def _as_shape(value: int | Sequence[int], what: str) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"{what} must have 1 to 3 dimensions, got {shape}")
    return tuple(int(n) for n in shape)


class CudaKernel:
    """A CUDA C kernel, compiled with NVRTC through CuPy.

    Parameters
    ----------
    source : str
        CUDA C source code containing the ``__global__`` function `name`
        (``extern "C"`` unless it is a template).
    name : str
        Name of the kernel function in `source`.
    block_size : int | Sequence[int]
        Threads per block: an integer for 1D launches, or 1 to 3 integers
        (at most 1024 threads in total).
    options : Sequence[str]
        Additional NVRTC compiler options, e.g. ``("-std=c++17",)``.
    include_dirs : Sequence[str | Path]
        Directories searched for ``#include`` files (passed as ``-I<dir>``).
        The headers shipped with cunumpy (:func:`cuda_include_dir`) are always
        found: ``#include "cunumpy/array_view.cuh"`` gives the ``Array1D<T>``
        to ``Array3D<T>`` views, ``#include "cunumpy/index.cuh"`` the
        thread-index macros.
    structs : Iterable[CudaStruct]
        Struct types passed to the kernel by value.
    template_args : Sequence | None
        Template arguments if `name` is a function template, e.g.
        ``("double", 3)`` or ``(np.float64, 3)``; the kernel is then the
        instantiation ``name<double, 3>``.
    check_signature : bool
        Parse the kernel signature and check (and cast) every call against it.
        Raises ``ValueError`` at construction if the signature cannot be parsed
        (e.g. macros in the parameter list); pass False to launch with the
        arguments as they are, like ``cupy.RawKernel``.

    Examples
    --------
    >>> axpy = CudaKernel(r'''
    ... extern "C" __global__
    ... void axpy(double a, const double* x, double* y, int n) {
    ...     int i = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (i < n) y[i] += a * x[i];
    ... }''', "axpy")
    >>> axpy(2.0, x, y, x.size, n_threads=x.size)  # doctest: +SKIP
    """

    def __init__(
        self,
        source: str,
        name: str,
        *,
        block_size: int | Sequence[int] = 128,
        options: Sequence[str] = (),
        include_dirs: Sequence[str | Path] = (),
        structs: Iterable[CudaStruct] = (),
        template_args: Sequence[Any] | None = None,
        check_signature: bool = True,
    ) -> None:
        self._block = self._check_block(_as_shape(block_size, "block_size"))
        self._source = source
        self._name = name
        self._options = (
            tuple(options)
            + tuple(f"-I{d}" for d in include_dirs)
            + (f"-I{cuda_include_dir()}",)
        )
        self._structs = tuple(structs)
        self._template_args = None if template_args is None else tuple(template_args)
        self._signature = (
            parse_cuda_signature(
                source, name, structs=self._structs, template_args=self._template_args
            )
            if check_signature
            else None
        )
        # one checker per parameter, built once so that calls stay cheap
        self._checkers = (
            None
            if self._signature is None
            else [_checker(p, i) for i, p in enumerate(self._signature)]
        )
        self._raw_kernel = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        name: str | None = None,
        *,
        suffix: str = "_cuda.cu",
        **kwargs: Any,
    ) -> CudaKernel:
        """Load the CUDA source from a file.

        Parameters
        ----------
        path : str | Path
            Path of the CUDA source file.
        name : str | None
            Name of the kernel function; by default the file name without
            `suffix` (``axpy_cuda.cu`` -> ``axpy``).
        suffix : str
            File name suffix stripped to get the default kernel name.
        **kwargs
            Passed on to :class:`CudaKernel`. The directory of the file is
            always added to ``include_dirs``.
        """
        path = Path(path)
        if name is None:
            if not path.name.endswith(suffix):
                raise ValueError(
                    f"{path.name} does not end with {suffix!r}; pass the kernel name"
                )
            name = path.name[: -len(suffix)]
        include_dirs = (path.parent, *kwargs.pop("include_dirs", ()))
        return cls(path.read_text(), name, include_dirs=include_dirs, **kwargs)

    @staticmethod
    def _check_block(block: tuple[int, ...]) -> tuple[int, ...]:
        if any(n <= 0 for n in block):
            raise ValueError(f"block sizes must be positive, got {block}")
        if math.prod(block) > _MAX_THREADS_PER_BLOCK:
            raise ValueError(
                f"a block has at most {_MAX_THREADS_PER_BLOCK} threads, got "
                f"{block} = {math.prod(block)}"
            )
        return block

    def __repr__(self) -> str:
        return f"CudaKernel(name={self.expression!r}, block_size={self.block_size})"

    @property
    def name(self) -> str:
        """Name of the kernel function."""
        return self._name

    @property
    def expression(self) -> str:
        """The compiled function: `name`, or its template instantiation."""
        if self._template_args is None:
            return self._name
        args = ", ".join(_template_arg(a) for a in self._template_args)
        return f"{self._name}<{args}>"

    @property
    def source(self) -> str:
        """CUDA C source code."""
        return self._source

    @property
    def block_size(self) -> int | tuple[int, ...]:
        """Threads per block: an integer for 1D blocks, else a tuple."""
        return self._block[0] if len(self._block) == 1 else self._block

    @property
    def options(self) -> tuple[str, ...]:
        """NVRTC compiler options, including ``-I`` include directories.

        The last one is cunumpy's own include directory (:func:`cuda_include_dir`).
        """
        return self._options

    @property
    def structs(self) -> tuple[CudaStruct, ...]:
        """Struct types passed to the kernel by value."""
        return self._structs

    @property
    def template_args(self) -> tuple[Any, ...] | None:
        """Template arguments, or None if the kernel is not a template."""
        return self._template_args

    @property
    def signature(self) -> tuple[CudaParameter, ...] | None:
        """The parsed kernel parameters, or None if calls are not checked."""
        return self._signature

    @property
    def is_compiled(self) -> bool:
        """Whether the kernel has been compiled in this process."""
        return self._raw_kernel is not None

    def compile(self) -> Any:
        """Compile the kernel now (it is otherwise compiled on the first call).

        Returns
        -------
        cupy.RawKernel
            The compiled kernel; compiled once and cached (also on disk by CuPy).

        Raises
        ------
        RuntimeError
            If CuPy or a GPU is not available.
        """
        if self._raw_kernel is None:
            from .xp import cupy_available

            if not cupy_available():
                raise RuntimeError(
                    f"cannot compile CUDA kernel {self.expression!r}: "
                    "CuPy is not installed or no GPU is available"
                )
            import cupy as cp

            if self._template_args is None:
                self._raw_kernel = cp.RawKernel(
                    self._source, self._name, options=self._options
                )
            else:
                module = cp.RawModule(
                    code=self._source,
                    options=self._options,
                    name_expressions=[self.expression],
                )
                self._raw_kernel = module.get_function(self.expression)
        return self._raw_kernel

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """The arguments as passed to ``cupy.RawKernel``: flattened and checked.

        Argument objects with ``__cuda_args__()`` (including struct values) are
        flattened. If the signature is checked, the number of arguments, the
        dtype of every array, every struct and every scalar are checked, Python
        scalars are cast to the declared C types, and arrays for array view
        parameters (``Array2D<double>``) are packed into (pointer, shape,
        strides).

        Raises
        ------
        TypeError
            Wrong number of arguments, a host array or an array of the wrong
            dtype for a pointer parameter, an array of the wrong dtype or
            number of dimensions for an array view, a value of the wrong
            struct, or a scalar of an incompatible type.
        OverflowError
            A Python integer out of range of the declared integer type.
        """
        values: list[Any] = []
        for arg in args:
            cuda_args = getattr(arg, "__cuda_args__", None)
            if cuda_args is not None:
                values.extend(cuda_args())
            else:
                values.append(arg)
        if self._signature is None:
            return tuple(values)

        if len(values) != len(self._signature):
            raise TypeError(
                f"{self._name}() takes {len(self._signature)} arguments after "
                f"flattening argument objects, got {len(values)}"
            )
        return tuple([check(v) for check, v in zip(self._checkers, values)])

    def launch_shape(
        self,
        n_threads: int | Sequence[int] | None = None,
        *,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The ``(grid, block)`` a call with these launch arguments uses.

        See :meth:`__call__`. A grid with a zero dimension launches nothing.
        """
        block_shape = (
            self._block
            if block is None
            else self._check_block(_as_shape(block, "block"))
        )
        if (n_threads is None) == (grid is None):
            raise TypeError("pass exactly one of n_threads and grid")

        if grid is not None:
            grid_shape = _as_shape(grid, "grid")
            if any(n < 0 for n in grid_shape):
                raise ValueError(f"grid must be non-negative, got {grid_shape}")
            return grid_shape, block_shape

        threads = _as_shape(n_threads, "n_threads")
        if any(n < 0 for n in threads):
            raise ValueError(f"n_threads must be non-negative, got {threads}")
        if len(block_shape) != len(threads):
            if len(block_shape) != 1:
                raise ValueError(
                    f"block {block_shape} and n_threads {threads} have different "
                    "numbers of dimensions"
                )
            block_shape = block_shape + (1,) * (len(threads) - 1)
        grid_shape = tuple(math.ceil(n / b) for n, b in zip(threads, block_shape))
        return grid_shape, block_shape

    def __call__(
        self,
        *args: Any,
        n_threads: int | Sequence[int] | None = None,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
        shared_mem: int = 0,
        stream: Any = None,
    ) -> None:
        """Launch the kernel.

        The launch shape is given either by `n_threads` (the grid is the number
        of threads divided by the block size, rounded up, per dimension) or by
        an explicit `grid`.

        Parameters
        ----------
        *args
            Kernel arguments: CuPy arrays, scalars and argument objects with
            ``__cuda_args__()``, see :meth:`prepare_args`.
        n_threads : int | Sequence[int] | None
            Number of threads in 1 to 3 dimensions, e.g. ``n`` or ``(nx, ny)``.
            With a 1D block size, the block is ``(block_size, 1, ...)``.
        grid : int | Sequence[int] | None
            Number of blocks in 1 to 3 dimensions, instead of `n_threads`.
        block : int | Sequence[int] | None
            Block shape for this call, instead of the kernel's `block_size`.
        shared_mem : int
            Dynamic shared memory per block, in bytes (``extern __shared__``).
        stream : cupy.cuda.Stream | None
            Stream to launch on; the current stream if None.
        """
        grid_shape, block_shape = self.launch_shape(n_threads, grid=grid, block=block)
        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        values = self.prepare_args(*args)
        if 0 in grid_shape:
            return

        kernel = self.compile()
        with stream if stream is not None else nullcontext():
            kernel(grid_shape, block_shape, values, shared_mem=shared_mem)


class CudaKernelVariants:
    """Kernels generated per variant (e.g. per dtype and dimension), compiled once.

    For kernels whose source is generated for each variant, e.g. a stencil
    product for ``ndim`` in 1 to 3 and several dtypes. The `factory` is called
    the first time a variant is requested; the kernel is cached per key.

    Parameters
    ----------
    factory : Callable[..., CudaKernel]
        Creates the kernel for a variant from its key, e.g.
        ``lambda ndim, dtype: CudaKernel(make_source(ndim, ctype_of(dtype)), "f")``.

    Examples
    --------
    >>> matvec = CudaKernelVariants(
    ...     lambda ndim, dtype: CudaKernel(source(ndim, ctype_of(dtype)), "matvec")
    ... )
    >>> matvec.get(3, np.float64)(mat, x, out, n_threads=out.size)  # doctest: +SKIP
    """

    def __init__(self, factory: Callable[..., CudaKernel]) -> None:
        self._factory = factory
        self._kernels: dict[tuple[Hashable, ...], CudaKernel] = {}

    def __repr__(self) -> str:
        return f"CudaKernelVariants({len(self._kernels)} variants)"

    def get(self, *key: Hashable) -> CudaKernel:
        """The kernel for the variant `key`, created on first use."""
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = self._factory(*key)
            if not isinstance(kernel, CudaKernel):
                raise TypeError(
                    f"the factory must return a CudaKernel, got {type(kernel).__name__}"
                )
            self._kernels[key] = kernel
        return kernel

    def __len__(self) -> int:
        return len(self._kernels)

    def __iter__(self) -> Iterator[tuple[Hashable, ...]]:
        """Iterate over the keys of the variants created so far."""
        return iter(list(self._kernels))

    def keys(self) -> list[tuple[Hashable, ...]]:
        """The keys of the variants created so far."""
        return list(self._kernels)

    def compile_all(self, keys: Iterable[Sequence[Hashable]] = ()) -> None:
        """Compile the given variants (created if needed) and all existing ones.

        Parameters
        ----------
        keys : Iterable[Sequence]
            Keys of variants to create and compile now, e.g. ``[(3, np.float64)]``.
        """
        for key in keys:
            self.get(*key)
        for kernel in self._kernels.values():
            kernel.compile()
