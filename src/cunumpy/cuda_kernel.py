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
* C++ function templates are instantiated with ``template_args``, and generated
  kernels (one source per variant) are compiled once per variant by
  :class:`CudaKernelVariants`.

The launch shape is given at each call, either as the number of threads
(``n_threads``, in 1 to 3 dimensions) or as an explicit ``grid``.

In debug mode (``debug=True``, ``xp.set_cuda_debug(True)`` or the environment
variable ``CUNUMPY_CUDA_DEBUG=1``) kernels are compiled with ``-lineinfo`` and
``-DCUNUMPY_BOUNDS_CHECK``, and every launch is synchronized so that an
asynchronous CUDA error is raised, as a ``RuntimeError`` naming the kernel, at
the launch that caused it.

This module imports CuPy only when a kernel is compiled, so it can be imported
(and signatures parsed) without CuPy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "DEBUG_OPTIONS",
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructValue",
    "ctype_of",
    "parse_cuda_signature",
]

# CUDA limit on the number of threads per block
_MAX_THREADS_PER_BLOCK = 1024

#: NVRTC options added in debug mode: source line information for
#: ``compute-sanitizer``/``nsys``, and bounds checks in the array views.
#: (``-G`` is not among them: NVRTC does not support it.)
DEBUG_OPTIONS = ("-lineinfo", "-DCUNUMPY_BOUNDS_CHECK")


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
        Normalized C type without qualifiers or ``*``, e.g. ``"double"``.
    dtype : numpy.dtype | None
        NumPy dtype of the value (or of the pointed-to elements; the structured
        dtype for a struct); ``None`` for ``void*``.
    pointer : bool
        Whether the parameter is a pointer (a device array).
    struct : CudaStruct | None
        The struct type, for a struct passed by value.
    """

    name: str
    ctype: str
    dtype: np.dtype | None
    pointer: bool
    struct: CudaStruct | None = None


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
_TOKEN = re.compile(r"complex<(?:float|double)>|[A-Za-z_]\w*|\*|\[\s*\]")


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


def _pointer_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for a pointer parameter: a device array with the right dtype."""
    dtype = param.dtype

    def check(value: Any) -> Any:
        # checked on the class: on the instance, CuPy builds the whole interface dict
        if not hasattr(type(value), "__cuda_array_interface__") and not hasattr(
            value, "__cuda_array_interface__"
        ):
            raise TypeError(
                f"{_describe(param, index)} must be a CuPy array, got "
                f"{type(value).__name__}; arrays are never copied to the device"
            )
        if dtype is not None and value.dtype != dtype:
            raise TypeError(
                f"{_describe(param, index)} must have dtype {dtype}, got {value.dtype}"
            )
        return value

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
    if param.pointer:
        return _pointer_checker(param, index)
    return _scalar_checker(param, index)


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
        ``("n", "int")``. Scalar fields and pointers to the scalar types of
        :func:`ctype_of` (or ``void*``) are supported.

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
            [(f.name, np.uint64 if f.pointer else f.dtype) for f in parsed],
            align=True,
        )
        self._checkers = {
            f.name: _pointer_checker(f, i) if f.pointer else _scalar_checker(f, i)
            for i, f in enumerate(parsed)
        }

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
    def declaration(self) -> str:
        """The C definition of the struct, to include in the CUDA source."""
        lines = [
            f"    {f.ctype}{'*' if f.pointer else ''} {f.name};" for f in self._fields
        ]
        return f"struct {self._name} {{\n" + "\n".join(lines) + "\n};\n"

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
        scalar fields are checked and cast like scalar kernel arguments.

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
    debug : bool | None
        Debug mode: compile with :data:`DEBUG_OPTIONS` (``-lineinfo`` and
        ``-DCUNUMPY_BOUNDS_CHECK``) and synchronize after every launch, so
        that an asynchronous CUDA error is raised as a ``RuntimeError`` naming
        this kernel. None (the default) follows the global setting
        (:func:`cunumpy.set_cuda_debug`, ``CUNUMPY_CUDA_DEBUG``) at every
        launch; True or False fix it for this kernel. The compile options are
        fixed when the kernel is compiled.

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
        debug: bool | None = None,
    ) -> None:
        self._block = self._check_block(_as_shape(block_size, "block_size"))
        self._debug = None if debug is None else bool(debug)
        self._source = source
        self._name = name
        self._options = tuple(options) + tuple(f"-I{d}" for d in include_dirs)
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
        """NVRTC compiler options as given, including ``-I`` include directories.

        See :meth:`compile_options` for the options a compilation uses.
        """
        return self._options

    @property
    def debug(self) -> bool | None:
        """The kernel's debug setting: True, False, or None for the global one."""
        return self._debug

    def debug_active(self) -> bool:
        """Whether debug mode applies to this kernel now.

        The kernel's own setting if it was created with ``debug=True`` or
        ``debug=False``, else the global setting (:func:`cunumpy.get_cuda_debug`),
        read at the time of the call.
        """
        if self._debug is not None:
            return self._debug
        from .xp import get_cuda_debug

        return get_cuda_debug()

    def compile_options(self) -> tuple[str, ...]:
        """The NVRTC options a compilation now would use.

        :attr:`options`, followed by :data:`DEBUG_OPTIONS` (``-lineinfo`` and
        ``-DCUNUMPY_BOUNDS_CHECK``) if :meth:`debug_active` and they are not
        already among the options. ``-G`` is not added: NVRTC does not support
        it.
        """
        options = self._options
        if self.debug_active():
            options += tuple(o for o in DEBUG_OPTIONS if o not in options)
        return options

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

        The options are :meth:`compile_options`, evaluated now: a kernel
        compiled before debug mode was enabled keeps its options.

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

            options = self.compile_options()
            if self._template_args is None:
                self._raw_kernel = cp.RawKernel(
                    self._source, self._name, options=options
                )
            else:
                module = cp.RawModule(
                    code=self._source,
                    options=options,
                    name_expressions=[self.expression],
                )
                self._raw_kernel = module.get_function(self.expression)
        return self._raw_kernel

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """The arguments as passed to ``cupy.RawKernel``: flattened and checked.

        Argument objects with ``__cuda_args__()`` (including struct values) are
        flattened. If the signature is checked, the number of arguments, the
        dtype of every array, every struct and every scalar are checked, and
        Python scalars are cast to the declared C types.

        Raises
        ------
        TypeError
            Wrong number of arguments, a host array or an array of the wrong
            dtype for a pointer parameter, a value of the wrong struct, or a
            scalar of an incompatible type.
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

        Raises
        ------
        RuntimeError
            In debug mode (see :meth:`debug_active`), an asynchronous CUDA
            error found when synchronizing the stream after the launch, e.g. an
            illegal memory access; the CuPy error is chained. Without debug
            mode, such an error surfaces at a later synchronization (a
            ``.get()``, an MPI call, ...), not necessarily in this kernel.
        """
        grid_shape, block_shape = self.launch_shape(n_threads, grid=grid, block=block)
        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        values = self.prepare_args(*args)
        if 0 in grid_shape:
            return

        kernel = self.compile()
        debug = self.debug_active()
        with stream if stream is not None else nullcontext():
            kernel(grid_shape, block_shape, values, shared_mem=shared_mem)
            if debug:
                self._synchronize_after_launch(stream, grid_shape, block_shape)

    def _synchronize_after_launch(
        self, stream: Any, grid: tuple[int, ...], block: tuple[int, ...]
    ) -> None:
        """Wait for the launch and re-raise a CUDA error naming this kernel."""
        import cupy as cp

        try:
            if stream is None:
                stream = cp.cuda.get_current_stream()
            stream.synchronize()
        except (
            cp.cuda.runtime.CUDARuntimeError,
            cp.cuda.driver.CUDADriverError,
        ) as error:
            raise RuntimeError(
                f"CUDA error after launching kernel {self.expression!r} with "
                f"grid {grid} and block {block}: {error}"
            ) from error


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
