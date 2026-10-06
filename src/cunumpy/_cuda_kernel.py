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
* arrays are never converted or copied: they must already be C-contiguous
  CuPy arrays (build them once with :func:`cunumpy.as_device_array`);
* strided array views: a parameter or struct field of type ``Array2D<double>``
  (from the shipped header ``cunumpy/array_view.cuh``, see
  :func:`cuda_include_dir`) takes a 2D CuPy array, contiguous or not, and
  receives its pointer, shape and strides, so that kernels index ``a(i, j)``
  like the pyccel kernels they are ported from; ``CArray2D<double>`` is the
  C-contiguous variant (shape only, ``a(i, j)`` is ``data[i * shape[1] + j]``),
  which takes C-contiguous arrays only and raises for other views;
* C++ function templates are instantiated with ``template_args``, and generated
  kernels (one source per variant) are compiled once per variant by
  :class:`CudaKernelVariants`.

The launch shape is given at each call, either as the number of threads
(``n_threads``, in 1 to 3 dimensions) or as an explicit ``grid``.

Argument structs can be generated from the annotations of a Python class
(:meth:`CudaStruct.from_signature`) and written to a header
(:meth:`CudaStruct.to_header`, :func:`write_cuda_header`), so that the Python
class is the one definition of the arguments.

In debug mode (``debug=True``, ``xp.cuda.set_cuda_debug(True)`` or the environment
variable ``CUNUMPY_CUDA_DEBUG=1``) kernels are compiled with ``-lineinfo`` and
``-DCUNUMPY_BOUNDS_CHECK``, and every launch is synchronized so that an
asynchronous CUDA error is raised, as a ``RuntimeError`` naming the kernel, at
the launch that caused it.

This module imports CuPy only when a kernel is compiled, so it can be imported
(and signatures parsed) without CuPy.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import operator
import os
import re
import sys
import typing
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
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
    "CudaStructArguments",
    "CudaStructValue",
    "ctype_of",
    "cuda_include_dir",
    "cuda_kernel_names",
    "include_hash",
    "parse_cuda_signature",
    "resolve_includes",
    "write_cuda_header",
]

# CUDA limit on the number of threads per block
_MAX_THREADS_PER_BLOCK = 1024

# Runtime queries are setup work, never repeated for a prepared launch.
_DEVICE_LIMITS: dict[int, dict[str, Any]] = {}


def _current_device() -> int:
    import cupy as cp

    return cp.cuda.runtime.getDevice()


def _device_limits(device: int) -> dict[str, Any]:
    if device not in _DEVICE_LIMITS:
        import cupy as cp

        _DEVICE_LIMITS[device] = cp.cuda.runtime.getDeviceProperties(device)
    return _DEVICE_LIMITS[device]


@dataclass
class _CompiledKernel:
    raw: Any
    limits: dict[str, Any]
    attributes: dict[str, int]
    dynamic_shared: int


# Headers shipped with cunumpy: #include <cunumpy/array_view.cuh> etc.
_CUDA_INCLUDE_DIR = Path(__file__).resolve().parent / "cuda" / "include"
_ARRAY_VIEW_INCLUDE = '#include "cunumpy/array_view.cuh"'


def cuda_include_dir() -> str:
    """The directory of the CUDA headers shipped with cunumpy.

    :class:`CudaKernel` adds it to the include path automatically, so kernels
    can ``#include "cunumpy/array_view.cuh"`` (strided ``Array1D<T>``
    to ``Array4D<T>`` views passed by value) and
    ``#include "cunumpy/index.cuh"`` (thread-index and grid-stride macros such
    as ``CUNUMPY_THREAD_1D(i, n)``), ``#include "cunumpy/atomic.cuh"`` (atomic
    adds) and ``#include "cunumpy/reduce.cuh"`` (warp and block reductions).
    Pass it as ``-I`` to other compilers.
    """
    return str(_CUDA_INCLUDE_DIR)


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
        ``Array4D<T>`` or ``CArray1D<T>`` to ``CArray4D<T>``, see
        :func:`cuda_include_dir`) passed by value.
    contiguous : bool
        Whether the array view is C-contiguous (``CArray2D<T>``): packed
        without strides, and only C-contiguous arrays are accepted.
    """

    name: str
    ctype: str
    dtype: np.dtype | None
    pointer: bool
    struct: CudaStruct | None = None
    view_ndim: int | None = None
    contiguous: bool = False


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
# Array1D<T> to Array4D<T> and CArray1D<T> to CArray4D<T> (cunumpy/array_view.cuh),
# T a scalar type of _CTYPES
_VIEW = re.compile(r"\b(C?)Array([1234])D\s*<((?:[^<>]|complex<[^<>]*>)+?)>")
_TOKEN = re.compile(
    r"C?Array[1234]D<[^<>]*(?:<[^<>]*>[^<>]*)?>|complex<(?:float|double)>"
    r"|[A-Za-z_]\w*|\*|\[\s*\]",
)


def _normalize_view(match: re.Match) -> str:
    """``Array2D< const double >`` -> ``Array2D<double>``."""
    words = [w for w in match.group(3).split() if w not in _QUALIFIERS]
    return f"{match.group(1)}Array{match.group(2)}D<{' '.join(words)}>"


def _view_dtype(ndim: int, contiguous: bool = False) -> np.dtype:
    """The structured dtype with the C layout of ``Array<ndim>D<T>``.

    ``CArray<ndim>D<T>`` (`contiguous`) has no strides.
    """
    fields = [("data", np.uint64), ("shape", np.int64, (ndim,))]
    if not contiguous:
        fields.append(("strides", np.int64, (ndim,)))
    return np.dtype(fields, align=True)


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


# ``#include "name"``: quoted includes are the project's own headers. Angle
# bracket includes are system headers and are not tracked, except in the
# directories given as ``angle_dirs`` (cunumpy's shipped headers).
_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*(?:"([^"\n]+)"|<([^>\n]+)>)',
    re.MULTILINE,
)


def _includes(source: str) -> list[tuple[str, bool]]:
    """``(name, quoted)`` of every ``#include`` in `source`, in order."""
    return [
        (quoted or angle, bool(quoted))
        for quoted, angle in _INCLUDE.findall(_strip_comments(source))
    ]


def _quoted_includes(source: str) -> list[str]:
    return [name for name, quoted in _includes(source) if quoted]


def resolve_includes(
    source: str,
    include_dirs: Iterable[str | Path] = (),
    *,
    base_dir: str | Path | None = None,
    angle_dirs: Iterable[str | Path] = (),
) -> list[Path]:
    """The header files a CUDA source includes, recursively.

    Scans `source` (comments removed) for ``#include "name"`` and resolves each
    name like NVRTC does: relative to `base_dir` (the directory of the
    including file), then in `include_dirs`, then in `angle_dirs`, in order.
    Found headers are scanned in turn, relative to their own directory.
    Includes in angle brackets (``#include <name>``) are system headers and
    ignored, unless they are found in `angle_dirs`. Includes that cannot be
    found are ignored; NVRTC reports them when the kernel is compiled.

    Parameters
    ----------
    source : str
        CUDA C source code.
    include_dirs : Iterable[str | Path]
        Directories searched for included files, in order (the ``-I`` options).
    base_dir : str | Path | None
        Directory of the file `source` was read from, searched first; None if
        the source is not from a file.
    angle_dirs : Iterable[str | Path]
        Directories whose headers are tracked also when included in angle
        brackets, searched last; :class:`CudaKernel` passes
        :func:`cuda_include_dir`, so that ``#include <cunumpy/reduce.cuh>``
        is tracked.

    Returns
    -------
    list[Path]
        The resolved header files, each once, in order of first inclusion
        (depth first). Empty if the source has no includes to track; the file
        system is not touched in that case.
    """
    angle = tuple(Path(d) for d in angle_dirs)
    dirs = tuple(Path(d) for d in include_dirs)
    dirs += tuple(d for d in angle if d not in dirs)
    found: list[Path] = []
    seen: set[Path] = set()

    def visit(code: str, directory: Path | None) -> None:
        for name, quoted in _includes(code):
            if quoted:
                candidates = [directory / name] if directory is not None else []
                candidates += [d / name for d in dirs]
            else:
                candidates = [d / name for d in angle]
            for candidate in candidates:
                if candidate.is_file():
                    path = candidate.resolve()
                    if path not in seen:
                        seen.add(path)
                        found.append(candidate)
                        visit(path.read_text(errors="replace"), path.parent)
                    break

    visit(source, None if base_dir is None else Path(base_dir))
    return found


def include_hash(paths: Iterable[str | Path]) -> str:
    """A short hex digest of the contents of `paths`, in order.

    Only the file contents count, not their locations: moving a header does not
    change the hash, editing it does. Used to make CuPy's kernel cache key
    depend on the included headers, see :meth:`CudaKernel.compile_options`.

    Parameters
    ----------
    paths : Iterable[str | Path]
        Files to hash, e.g. from :func:`resolve_includes`.

    Returns
    -------
    str
        The first 16 hex digits of the SHA-256 digest.
    """
    digest = hashlib.sha256()
    for path in paths:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()[:16]


_GLOBAL_FUNCTION = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(")


def cuda_kernel_names(source: str) -> list[str]:
    """The names of the ``__global__`` functions defined in `source`, in order.

    Comments are ignored. Templates are included; a function declared more
    than once (e.g. a forward declaration) is listed once.

    Parameters
    ----------
    source : str
        CUDA C source code.

    Returns
    -------
    list[str]
        The kernel names, in the order of their first appearance.
    """
    return list(dict.fromkeys(_GLOBAL_FUNCTION.findall(_strip_comments(source))))


def _compile_in_threads(
    compilers: Mapping[Hashable, Callable[[], Any]],
    jobs: int | None,
) -> list[Hashable]:
    """Run the `compilers` (name -> compile function), `jobs` at a time.

    With ``jobs=1`` they run one after the other in the calling thread; with
    ``jobs=None`` as many threads as CPUs are used. All compilers are run even
    if one fails; the first exception (in the order of `compilers`) is raised
    afterwards.

    Returns
    -------
    list
        The names whose compiler succeeded, in the order of `compilers`.
    """
    if jobs is None:
        jobs = os.cpu_count() or 1
    if jobs < 1:
        raise ValueError(f"jobs must be positive or None, got {jobs}")
    if jobs == 1 or len(compilers) <= 1:
        for compile in compilers.values():
            compile()
        return list(compilers)

    from cunumpy.xp import cupy_available

    device = _current_device() if cupy_available() else None

    def on_device(compiler: Callable[[], Any]) -> Any:
        if device is None:
            return compiler()
        import cupy as cp

        with cp.cuda.Device(device):
            return compiler()

    with ThreadPoolExecutor(max_workers=min(jobs, len(compilers))) as pool:
        futures = {
            name: pool.submit(on_device, compile) for name, compile in compilers.items()
        }
    compiled, error = [], None
    for name, future in futures.items():
        exc = future.exception()
        if exc is None:
            compiled.append(name)
        elif error is None:
            error = exc
    if error is not None:
        raise error
    return compiled


def _parse_parameter(
    text: str,
    structs: dict[str, CudaStruct] | None = None,
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
                "only be passed by value",
            )
        struct = structs[ctype]
        return CudaParameter(name, ctype, struct.dtype, False, struct)
    view = _VIEW.fullmatch(ctype)
    if view is not None:
        element = view.group(3)
        if pointers or element not in _CTYPES:
            raise ValueError(
                f"cannot check the kernel parameter {text.strip()!r}: array views "
                f"take a scalar element type and are passed by value",
            )
        ndim = int(view.group(2))
        dtype = np.dtype(_CTYPES[element])
        return CudaParameter(name, ctype, dtype, False, None, ndim, bool(view.group(1)))
    if ctype == "void" and pointers == 1:
        return CudaParameter(name, ctype, None, True)
    if pointers > 1 or ctype not in _CTYPES:
        raise ValueError(
            f"cannot check the kernel parameter {text.strip()!r}: unsupported type "
            f"{ctype + '*' * pointers!r}",
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
                f"pass them as template_args",
            )
        for words, value in zip(template_params, template_args):
            if words[0] in ("typename", "class"):
                params = re.sub(
                    r"\b" + re.escape(words[-1]) + r"\b",
                    _template_arg(value),
                    params,
                )
    elif template_args:
        raise ValueError(f"{name!r} is not a template, but template_args were given")

    if params in ("", "void"):
        return ()
    return tuple(_parse_parameter(p, structs) for p in _split_top_level(params))


def _describe(param: CudaParameter, index: int) -> str:
    ctype = param.ctype + ("*" if param.pointer else "")
    return f"argument {index} ({ctype} {param.name})"


def _current_device_id() -> int | None:
    """The id of the current CUDA device, or None if CuPy has not been imported.

    CuPy is looked up in ``sys.modules`` and never imported here: without it
    there are no device arrays whose device could be checked.
    """
    cp = sys.modules.get("cupy")
    if cp is None:
        return None
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception:  # noqa: BLE001 - no device check without a working runtime
        return None


def _check_device_array(param: CudaParameter, index: int, value: Any) -> None:
    """Raise unless `value` is a device array of the declared dtype on the current device."""
    # checked on the class: on the instance, CuPy builds the whole interface dict
    if not hasattr(type(value), "__cuda_array_interface__") and not hasattr(
        value,
        "__cuda_array_interface__",
    ):
        raise TypeError(
            f"{_describe(param, index)} must be a CuPy array, got "
            f"{type(value).__name__}; arrays are never copied to the device",
        )
    if param.dtype is not None and value.dtype != param.dtype:
        raise TypeError(
            f"{_describe(param, index)} must have dtype {param.dtype}, got "
            f"{value.dtype}",
        )
    # an array on another GPU: the kernel would read a foreign address, which
    # neither cupy.RawKernel nor the struct packing notices
    device_id = getattr(getattr(value, "device", None), "id", None)
    if device_id is not None:
        current = _current_device_id()
        if current is not None and device_id != current:
            raise ValueError(
                f"{_describe(param, index)} is on CUDA device {device_id}, but the "
                f"current device is {current}; kernels only take arrays of the "
                "current device (see cunumpy.cuda.bind_local_device)",
            )


def _pointer_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for a pointer parameter: a C-contiguous device array of the dtype."""

    def check(value: Any) -> Any:
        _check_device_array(param, index, value)
        # the kernel reads the pointer as a flat buffer: a non-contiguous view
        # (e.g. a[:, 0:3]) would give silently wrong results
        flags = getattr(value, "flags", None)
        if flags is not None and not flags.c_contiguous:
            raise TypeError(
                f"{_describe(param, index)} must be C-contiguous: a non-contiguous "
                f"view (e.g. a[:, 0:3]) would be read as a flat buffer; use "
                f"cupy.ascontiguousarray or cunumpy.as_device_array",
            )
        return value

    return check


def _view_checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    """Checker for an array view: packs (pointer, shape, strides) of a device array.

    Strides are converted from bytes to elements; the array need not be
    contiguous. A contiguous view (``CArray2D<T>``) packs (pointer, shape) and
    takes C-contiguous arrays only: it is never copied, since what the kernel
    writes into a copy would be lost.
    """
    ndim = param.view_ndim
    contiguous = param.contiguous
    dtype = _view_dtype(ndim, contiguous)

    def check(value: Any) -> Any:
        _check_device_array(param, index, value)
        if value.ndim != ndim:
            raise TypeError(
                f"{_describe(param, index)} must be a {ndim}D array, got {value.ndim}D",
            )
        if contiguous:
            flags = getattr(value, "flags", None)
            if flags is not None and not flags.c_contiguous:
                raise TypeError(
                    f"{_describe(param, index)} must be C-contiguous: got a view "
                    f"with strides {tuple(value.strides)}; declare the parameter "
                    f"as Array{ndim}D to take strided views, or pass "
                    f"cupy.ascontiguousarray(...) and copy the result back",
                )
            packed = np.zeros((), dtype=dtype)
            packed["data"] = value.data.ptr
            packed["shape"] = value.shape
            return packed[()]
        itemsize = value.dtype.itemsize
        strides = [s // itemsize for s in value.strides]
        if any(s * itemsize != stride for s, stride in zip(strides, value.strides)):
            raise TypeError(
                f"{_describe(param, index)}: strides {tuple(value.strides)} are not "
                f"multiples of the element size {itemsize}",
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
                    f"[{low}, {high}]",
                )
            return scalar_type(value)
        if kind in "fc":
            return scalar_type(value)
        raise TypeError(
            f"{_describe(param, index)} cannot take a value of type "
            f"{type(value).__name__}",
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
            if isinstance(value, np.integer) and kind in "iuf":
                # by value, like a Python int: np.int64(5) fits an int parameter
                return cast_int(int(value))
            if np.can_cast(value.dtype, dtype, casting="safe"):
                return scalar_type(value)
            raise TypeError(
                f"{_describe(param, index)} cannot take a {value.dtype} scalar "
                f"without losing information",
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
            f"{value_type.__name__}",
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
            f"(created with that CudaStruct), got {type(value).__name__}",
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
        return _view_dtype(field.view_ndim, field.contiguous)
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
    r"(?:\[(?P<dims>[\s:,]*)\])?\s*\]?$",
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
            f"{what}: unsupported scalar type {scalar!r} in {annotation!r}",
        )
    ctype = scalars[scalar]
    if ndim == 0:
        return ctype
    if ndim > 4:
        raise ValueError(f"{what}: arrays have at most 4 dimensions, got {ndim}")
    return f"Array{ndim}D<{ctype}>"


def _apply_contiguous(
    fields: list[tuple[str, str]],
    contiguous: bool | Iterable[str],
    what: str,
) -> list[tuple[str, str]]:
    """Make array fields ``CArray<n>D<T>``: all of them, or those named."""
    if contiguous is True:
        names = {f for f, ctype in fields if ctype.startswith("Array")}
    elif contiguous is False:
        return fields
    else:
        names = {contiguous} if isinstance(contiguous, str) else set(contiguous)
        arrays = {f for f, ctype in fields if ctype.startswith("Array")}
        if names - arrays:
            raise ValueError(
                f"{what}: contiguous names {sorted(names - arrays)} that are not "
                f"array fields (array fields: {sorted(arrays)})",
            )
    return [(f, f"C{ctype}" if f in names else ctype) for f, ctype in fields]


def _header_guard(name: str) -> str:
    guard = re.sub(r"\W", "_", name).upper().strip("_")
    return guard if re.match(r"[A-Z_]", guard) else f"_{guard}"


def _header_source(
    structs: Sequence[CudaStruct],
    guard: str,
    includes: Iterable[str],
) -> str:
    includes = list(includes)
    if any(f.view_ndim is not None for s in structs for f in s.fields):
        includes.insert(0, _ARRAY_VIEW_INCLUDE)
    lines = [
        "// Generated by cunumpy.cuda.CudaStruct from the Python definition; do not edit.",
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


def _read_python_source(source: str | Path) -> tuple[str, str]:
    """`(text, description)` of a Python source given as a path or as code."""
    if isinstance(source, Path) or (
        "\n" not in source and source.strip().endswith(".py")
    ):
        path = Path(source)
        return path.read_text(), str(path)
    return str(source), "<source>"


def _find_init(tree: ast.Module, class_name: str, where: str) -> ast.FunctionDef:
    """The ``__init__`` function of the class `class_name` in a parsed module."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
            raise ValueError(f"class {class_name!r} in {where} has no __init__")
    raise ValueError(f"no class {class_name!r} in {where}")


def _stored_parameters(init: ast.FunctionDef) -> dict[str, str]:
    """``{parameter: attribute}`` for the ``self.<attribute> = <parameter>`` statements."""
    stored: dict[str, str] = {}
    for node in ast.walk(init):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Name):
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                stored.setdefault(value.id, target.attr)
    return stored


def _annotation_text(annotation: ast.expr) -> str:
    """The pyccel-style annotation string of a parsed annotation."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    text = ast.unparse(annotation)
    # Final['float[:]'] -> float[:] (the quotes come from the string annotation)
    return text.replace("'", "").replace('"', "")


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
        ``Array4D<T>`` of those scalar types (from ``cunumpy/array_view.cuh``,
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
        contiguous: bool | Iterable[str] = False,
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
        contiguous : bool | Iterable[str]
            Array fields that become C-contiguous views (``CArray2D<double>``
            instead of ``Array2D<double>``): ``True`` for all of them, or their
            names. Packing such a field raises for a non-contiguous array.

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
        what = f"{getattr(func, '__qualname__', func)}"
        return cls(name, _apply_contiguous(fields, contiguous, what))

    @classmethod
    def from_pyccel_class(
        cls,
        source: str | Path,
        class_name: str,
        name: str | None = None,
        *,
        int_type: str = "long long",
        scalar_names: Mapping[str, str] | None = None,
        exclude: Sequence[str] = (),
        attribute_names: bool = True,
        contiguous: bool | Iterable[str] = False,
    ) -> CudaStruct:
        """Build the struct from the ``__init__`` of a class in a Python source file.

        Like :meth:`from_signature`, but for a class whose module is compiled
        by pyccel: importing such a module gives the compiled class, whose
        ``__init__`` has no Python signature. The ``.py`` source is parsed
        with :mod:`ast` instead and is never imported or executed.

        One field per parameter of ``__init__`` (``self`` skipped), in order.
        A parameter that ``__init__`` stores as ``self.<attribute> = <parameter>``
        gives a field named after the attribute (`attribute_names`), so that
        the struct members are the attribute names the host kernels use, also
        when the constructor parameter is called differently.

        Parameters
        ----------
        source : str | Path
            Path of the ``.py`` file, or the source code itself (a string that
            contains a newline or does not end with ``.py``).
        class_name : str
            Name of the class in the source.
        name : str | None
            Name of the struct type in C; `class_name` by default.
        int_type, scalar_names, contiguous
            As for :meth:`from_signature` (`contiguous` takes field names, i.e.
            attribute names when `attribute_names` is set).
        exclude : Sequence[str]
            Parameter or attribute names that do not become fields, e.g.
            scratch arrays the host class allocates for itself.
        attribute_names : bool
            Name the fields after the attributes the parameters are stored in
            (default); False keeps the parameter names.

        Raises
        ------
        ValueError
            The class or its ``__init__`` is not found, or an annotation
            cannot be mapped (see :meth:`from_signature`).

        Examples
        --------
        >>> MarkerArgs = CudaStruct.from_pyccel_class(
        ...     "kernel_arguments/pusher_args_kernels.py", "MarkerArguments", "MarkerArgs"
        ... )  # doctest: +SKIP
        """
        text, where = _read_python_source(source)
        init = _find_init(ast.parse(text, filename=where), class_name, where)
        stored = _stored_parameters(init) if attribute_names else {}
        scalars = dict(_PYCCEL_SCALARS, int=int_type)
        if scalar_names:
            scalars.update(scalar_names)
        excluded = set(exclude)
        fields = []
        for arg in init.args.posonlyargs + init.args.args + init.args.kwonlyargs:
            if arg.arg == "self":
                continue
            field = stored.get(arg.arg, arg.arg)
            if arg.arg in excluded or field in excluded:
                continue
            what = f"parameter {arg.arg!r} of {class_name}.__init__ in {where}"
            if arg.annotation is None:
                raise ValueError(f"{what} has no type annotation")
            annotation = _annotation_text(arg.annotation)
            fields.append((field, _pyccel_ctype(annotation, scalars, what)))
        fields = _apply_contiguous(fields, contiguous, f"{class_name} in {where}")
        return cls(class_name if name is None else name, fields)

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
            (self,),
            guard or _header_guard(f"{self._name}_cuh"),
            includes,
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
            r"struct\s+" + re.escape(self._name) + r"\s*\{(.*?)\}",
            code,
            re.DOTALL,
        )
        if match is None:
            return
        members = [m for m in match.group(1).split(";") if m.strip()]
        try:
            found = [_parse_parameter(m) for m in members]
        except ValueError as exc:
            raise ValueError(
                f"cannot compare the definition of struct {self._name!r}: {exc}",
            ) from None
        key = [(f.name, f.ctype, f.pointer) for f in self._fields]
        if [(f.name, f.ctype, f.pointer) for f in found] != key:
            raise ValueError(
                f"the definition of struct {self._name!r} in the CUDA source does not "
                f"match its CudaStruct:\n{self.declaration}",
            )

    def layout_source(self, include: str | None = None) -> str:
        """The CUDA source of the kernel used by :meth:`verify_layout`.

        The kernel ``cunumpy_layout_<name>(unsigned long long* out)`` writes
        ``sizeof``, ``alignof`` and the offset of every field (in field order)
        of the struct, as the compiler lays it out.

        Parameters
        ----------
        include : str | None
            Header that defines the struct, as a file name or an ``#include``
            line; by default the struct is defined by :attr:`declaration`.
        """
        if include is None:
            lines = ['#include "cunumpy/array_view.cuh"'] if self.has_views else []
            lines.append(self.declaration)
        else:
            line = include.strip()
            lines = [line if line.startswith("#") else f'#include "{line}"']
        offsets = "\n".join(
            f"    out[{i + 2}] = (unsigned long long)((const char*)&s.{f.name}"
            " - (const char*)&s);"
            for i, f in enumerate(self._fields)
        )
        lines.append(
            f'extern "C" __global__ void cunumpy_layout_{self._name}('
            "unsigned long long* out) {\n"
            f"    {self._name} s;\n"
            f"    out[0] = sizeof({self._name});\n"
            f"    out[1] = alignof({self._name});\n"
            f"{offsets}\n"
            "}\n",
        )
        return "\n".join(lines)

    def verify_layout(
        self,
        include: str | None = None,
        *,
        include_dirs: Sequence[str | Path] = (),
        options: Sequence[str] = (),
    ) -> dict[str, int]:
        """Check the struct layout of the CUDA compiler against :attr:`dtype`.

        Compiles and runs a one-thread kernel (:meth:`layout_source`) that
        reports the size, alignment and field offsets of the struct, and
        compares them with the NumPy dtype that values are packed into. A
        difference means every kernel taking the struct reads some fields at
        the wrong place, without any error. Run it once per struct in a GPU
        test, especially with a hand-written or generated header (`include`)
        and on a new compiler or platform (e.g. ROCm).

        Parameters
        ----------
        include : str | None
            Header that defines the struct (file name or ``#include`` line),
            found in `include_dirs`; by default :attr:`declaration` is compiled.
        include_dirs : Sequence[str | Path]
            Directories searched for `include`.
        options : Sequence[str]
            Additional compiler options.

        Returns
        -------
        dict[str, int]
            ``"sizeof"``, ``"alignof"`` and the offset of each field, by name.

        Raises
        ------
        ValueError
            If the compiled layout differs from :attr:`dtype` (the message lists
            each difference).
        RuntimeError
            If CuPy is not available.
        """
        from cunumpy.xp import cupy_available, to_numpy

        if not cupy_available():
            raise RuntimeError("verify_layout() compiles a CUDA kernel and needs CuPy")
        import cupy as cp

        kernel = CudaKernel(
            self.layout_source(include),
            f"cunumpy_layout_{self._name}",
            include_dirs=include_dirs,
            options=options,
            structs=[self],
            block_size=1,
        )
        out = cp.zeros(len(self._fields) + 2, dtype=cp.uint64)
        kernel(out, n_threads=1)
        measured = [int(v) for v in to_numpy(out)]
        layout = {"sizeof": measured[0], "alignof": measured[1]}
        layout.update({f.name: n for f, n in zip(self._fields, measured[2:])})

        expected = {"sizeof": self._dtype.itemsize, "alignof": self._dtype.alignment}
        expected.update({f.name: self._dtype.fields[f.name][1] for f in self._fields})
        differences = [
            f"{key}: compiler {layout[key]}, dtype {expected[key]}"
            for key in expected
            if layout[key] != expected[key]
        ]
        if differences:
            raise ValueError(
                f"the compiled layout of struct {self._name!r} differs from its "
                "CudaStruct dtype:\n  " + "\n  ".join(differences),
            )
        return layout

    def __call__(self, **values: Any) -> CudaStructValue:
        """Pack values into the struct.

        Pointer fields take C-contiguous CuPy arrays of the declared dtype
        (never copied), array view fields take CuPy arrays of the declared
        dtype and number of dimensions (contiguous or not; their pointer, shape
        and strides are packed), scalar fields are checked and cast like scalar kernel
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
                f"{unknown}",
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


class CudaStructArguments(CudaArguments):
    """Base class for argument objects that are passed to CUDA kernels as one C struct.

    The class form of :class:`CudaStruct`: a subclass names the struct
    (:attr:`struct_name`) and lists its fields (:attr:`fields`), stores every
    field as an attribute of the same name, and calls :meth:`pack` at the end
    of its constructor. The object is then passed to a :class:`CudaKernel` as
    it is and arrives as the packed struct.

    The :class:`CudaStruct` is built once per subclass, when the class is
    defined, and is available as :attr:`struct` (for ``CudaKernel(...,
    structs=[...])``, :attr:`CudaStruct.declaration` and
    :meth:`CudaStruct.to_header`). Packing checks every field like a kernel
    argument: pointers need C-contiguous CuPy arrays of the declared dtype
    (never copied), scalars are range-checked and cast.

    The packed struct always reflects the current field attributes: at every
    use (:attr:`packed`, :meth:`__cuda_args__`, so at every kernel launch) the
    device address, shape and strides of every array field and the value of
    every scalar field are compared with those that were packed, and the
    struct is packed again if any changed. A field may therefore be a
    property that reads the owner's current array, so that resizing the
    owner's arrays never leaves the struct pointing at freed device memory
    (see the second example). Copies (``copy.copy``, ``copy.deepcopy``) and
    unpickled objects are packed again from their own arrays.

    A subclass that sets neither :attr:`struct_name` nor :attr:`fields` is an
    intermediate base class; a subclass that sets only one of them raises
    ``TypeError``.

    Attributes
    ----------
    struct_name : str
        Name of the C struct type.
    fields : Sequence[tuple[str, str]]
        ``(field name, C type)`` pairs, in declaration order, as for
        :class:`CudaStruct`.
    struct : CudaStruct
        The struct type, built from :attr:`struct_name` and :attr:`fields`.

    Examples
    --------
    >>> class MarkerArguments(CudaStructArguments):
    ...     struct_name = "MarkerArgs"
    ...     fields = (
    ...         ("markers", "Array2D<double>"),
    ...         ("valid", "bool*"),
    ...         ("n_markers", "int"),
    ...     )
    ...
    ...     def __init__(self, markers, valid):
    ...         self.markers = markers
    ...         self.valid = valid
    ...         self.n_markers = markers.shape[0]
    ...         self.pack()
    >>> print(MarkerArguments.struct.declaration)
    struct MarkerArgs {
        Array2D<double> markers;
        bool* valid;
        int n_markers;
    };
    >>> push = CudaKernel(source, "push", structs=[MarkerArguments.struct])  # doctest: +SKIP
    >>> push(MarkerArguments(markers, valid), 0.1, n_threads=markers.shape[0])  # doctest: +SKIP

    Fields as properties follow the arrays of an owner object, also after the
    owner replaced them (e.g. when it resized its marker array):

    >>> class ParticleArguments(CudaStructArguments):
    ...     struct_name = "ParticleArgs"
    ...     fields = (("markers", "Array2D<double>"), ("n_markers", "int"))
    ...
    ...     def __init__(self, particles):
    ...         self._particles = particles
    ...         self.pack()
    ...
    ...     @property
    ...     def markers(self):
    ...         return self._particles.markers
    ...
    ...     @property
    ...     def n_markers(self):
    ...         return self._particles.markers.shape[0]
    """

    struct_name: str
    fields: Sequence[tuple[str, str]]
    struct: CudaStruct

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        has_name = "struct_name" in cls.__dict__
        has_fields = "fields" in cls.__dict__
        if not has_name and not has_fields:
            return  # an intermediate base class, or a subclass of a complete one
        if not (has_name and has_fields):
            raise TypeError(
                f"{cls.__qualname__} must define both struct_name and fields",
            )
        cls.struct = CudaStruct(cls.struct_name, cls.fields)

    def pack(self) -> None:
        """Pack the field attributes into the struct.

        Called at the end of the constructor, so that invalid field values
        raise there. Afterwards the struct is packed again automatically when
        a field changes (see the class documentation); calling this method
        again is never needed, but harmless.

        Raises
        ------
        TypeError
            If the class does not define a struct, or a field value has the
            wrong type (see :meth:`CudaStruct.__call__`).
        AttributeError
            If a field has no attribute of the same name.
        """
        struct = getattr(type(self), "struct", None)
        if struct is None:
            raise TypeError(
                f"{type(self).__qualname__} does not define struct_name and fields",
            )
        values = self._field_values(struct)
        self._struct_value = struct(**values)
        self._packed_state = _field_state(struct, values)

    def _field_values(self, struct: CudaStruct) -> dict[str, Any]:
        """The current value of every field attribute, by field name."""
        values = {}
        for field in struct.fields:
            try:
                values[field.name] = getattr(self, field.name)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__qualname__} has no attribute {field.name!r} "
                    f"for the field of struct {struct.name}",
                ) from None
        return values

    @property
    def packed(self) -> np.void:
        """The packed struct, with the memory layout of the C struct.

        Packed on first use, and again whenever a field attribute changed
        since the last packing (a different array, or a different scalar).
        """
        value = self.__dict__.get("_struct_value")
        if value is None:
            self.pack()
            return self._struct_value.packed
        struct = value.struct
        values = self._field_values(struct)
        if _field_state(struct, values) != self._packed_state:
            self._struct_value = struct(**values)
            self._packed_state = _field_state(struct, values)
        return self._struct_value.packed

    def __cuda_args__(self) -> tuple[np.void]:
        """The packed struct, as the one kernel argument this object stands for."""
        return (self.packed,)

    def __getstate__(self) -> dict[str, Any]:
        # the packed struct holds the device addresses of the original arrays,
        # which a copy (or another process) does not share
        state = self.__dict__.copy()
        state.pop("_struct_value", None)
        state.pop("_packed_state", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.pack()


def _field_state(struct: CudaStruct, values: Mapping[str, Any]) -> tuple[Any, ...]:
    """What the packed struct depends on, to detect changed field attributes.

    For an array field the device address, shape and strides (the address
    alone for a pointer field); for a scalar field its value. A value that is
    neither (e.g. a host array in a pointer field) is identified by its id,
    so replacing it triggers a repack, which then raises the type error.
    """
    state = []
    for field in struct.fields:
        value = values[field.name]
        if field.pointer or field.view_ndim is not None:
            ptr = getattr(getattr(value, "data", None), "ptr", None)
            if ptr is None:
                state.append(("id", id(value)))
            elif field.pointer:
                state.append(ptr)
            else:
                state.append((ptr, tuple(value.shape), tuple(value.strides)))
        elif isinstance(value, (bool, int, float, complex, np.generic)):
            state.append((type(value), value))
        else:
            state.append(("id", id(value)))
    return tuple(state)


def _is_device_array(value: Any) -> bool:
    return hasattr(type(value), "__cuda_array_interface__") or hasattr(
        value,
        "__cuda_array_interface__",
    )


def _array_shapes_in(args: Sequence[Any]) -> Iterator[tuple[int, ...]]:
    """Array shapes in argument order, including supported argument objects."""
    for arg in args:
        shape = getattr(arg, "shape", None)
        if shape is not None and hasattr(arg, "dtype"):
            if len(shape) > 0:
                yield tuple(shape)
        elif isinstance(arg, CudaStructArguments) and hasattr(type(arg), "struct"):
            yield from _array_shapes_in(
                [getattr(arg, field.name, None) for field in arg.struct.fields]
            )
        elif isinstance(arg, CudaStructValue):
            yield from _array_shapes_in(
                [arg[field.name] for field in arg.struct.fields]
            )
        elif hasattr(arg, "__cuda_args__"):
            yield from _array_shapes_in(arg.__cuda_args__())


def _first_array_shape(args: Sequence[Any], dimensions: int) -> int | tuple[int, ...]:
    shape = next(_array_shapes_in(args), None)
    if shape is None:
        raise TypeError(
            "inferring n_threads needs an array argument; pass n_threads or grid, "
            "or set n_threads_from to a callable",
        )
    if len(shape) < dimensions:
        raise ValueError(
            f"cannot infer {dimensions}D n_threads from the first array's shape "
            f"{shape}; pass n_threads or grid, or set n_threads_from to a callable",
        )
    return int(shape[0]) if dimensions == 1 else shape[:dimensions]


def _first_array_length(args: tuple[Any, ...]) -> int:
    """``n_threads_from="first_array"``: the first axis of the first array argument."""
    return typing.cast(int, _first_array_shape(args, 1))


def _as_shape(value: int | Sequence[int], what: str) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"{what} must have 1 to 3 dimensions, got {shape}")
    return tuple(int(n) for n in shape)


def _device_arrays_in(args: Sequence[Any]) -> Iterator[tuple[str, Any]]:
    """``(label, array)`` for every device array among kernel arguments.

    Arrays are found directly, in the fields of :class:`CudaStructArguments`
    objects and struct values, and in the values of other ``__cuda_args__()``
    objects.
    """
    for index, arg in enumerate(args):
        label = f"argument {index}"
        if _is_device_array(arg):
            yield label, arg
        elif isinstance(arg, CudaStructArguments) and hasattr(type(arg), "struct"):
            for field in arg.struct.fields:
                value = getattr(arg, field.name, None)
                if _is_device_array(value):
                    yield f"{label}.{field.name}", value
        elif isinstance(arg, CudaStructValue):
            for field in arg.struct.fields:
                value = arg[field.name]
                if _is_device_array(value):
                    yield f"{label}.{field.name}", value
        elif hasattr(arg, "__cuda_args__"):
            for j, value in enumerate(arg.__cuda_args__()):
                if _is_device_array(value):
                    yield f"{label}[{j}]", value


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
        found at compile time (see :meth:`compile_options`):
        ``#include "cunumpy/array_view.cuh"`` gives the ``Array1D<T>`` to
        ``Array4D<T>`` views, ``#include "cunumpy/index.cuh"`` the thread-index
        macros, ``#include "cunumpy/atomic.cuh"`` atomic adds,
        ``#include "cunumpy/reduce.cuh"`` warp and block reductions.
    source_dir : str | Path | None
        Directory the source was read from (set by :meth:`from_file`), where
        ``#include "..."`` files are looked up first.
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
        (:func:`cunumpy.cuda.set_cuda_debug`, ``CUNUMPY_CUDA_DEBUG``) at every
        launch; True or False fix it for this kernel. The compile options are
        fixed when the kernel is compiled.
    n_threads_from : {"auto", "first_array"} | callable | None
        Default "auto" infers thread counts from the first array's leading
        shape axes, matching the block dimensionality (1D: one thread per row).
        Arrays in supported argument objects are included. "first_array"
        always uses the first axis. A callable receives the positional argument
        tuple; None requires an explicit launch size. Explicit `n_threads` or
        `grid` overrides inference.

    Notes
    -----
    CuPy caches compiled kernels on disk, keyed on the source and the compiler
    options, but not on the files pulled in by ``#include "..."``. At compile
    time the headers are resolved (:attr:`included_headers`) and a define with
    the hash of their contents is added to the options
    (:meth:`compile_options`), so editing a header recompiles the kernel; this
    includes cunumpy's own headers (``#include <cunumpy/reduce.cuh>``), so an
    upgrade that changes them recompiles too.

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
        source_dir: str | Path | None = None,
        structs: Iterable[CudaStruct] = (),
        template_args: Sequence[Any] | None = None,
        check_signature: bool = True,
        debug: bool | None = None,
        n_threads_from: Callable[[tuple[Any, ...]], Any] | str | None = "auto",
        check_finite: bool = False,
    ) -> None:
        self._block = self._check_block(_as_shape(block_size, "block_size"))
        self.n_threads_from = n_threads_from
        self._check_finite = bool(check_finite)
        self._debug = None if debug is None else bool(debug)
        self._source = source
        self._name = name
        self._include_dirs = tuple(Path(d) for d in include_dirs)
        self._source_dir = None if source_dir is None else Path(source_dir)
        self._options = tuple(options) + tuple(f"-I{d}" for d in self._include_dirs)
        self._structs = tuple(structs)
        self._template_args = None if template_args is None else tuple(template_args)
        self._signature = (
            parse_cuda_signature(
                source,
                name,
                structs=self._structs,
                template_args=self._template_args,
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
        self._compiled: dict[int, _CompiledKernel] = {}

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
            always added to ``include_dirs`` and is the ``source_dir``.
        """
        path = Path(path)
        if name is None:
            if not path.name.endswith(suffix):
                raise ValueError(
                    f"{path.name} does not end with {suffix!r}; pass the kernel name",
                )
            name = path.name[: -len(suffix)]
        include_dirs = (path.parent, *kwargs.pop("include_dirs", ()))
        kwargs.setdefault("source_dir", path.parent)
        return cls(path.read_text(), name, include_dirs=include_dirs, **kwargs)

    @classmethod
    def all_from_file(cls, path: str | Path, **kwargs: Any) -> dict[str, CudaKernel]:
        """Load every ``__global__`` function of a file as a kernel.

        For files that group several small kernels. The kernels share the
        source (and the compile options), so CuPy compiles the file once and
        the kernels are functions of the same compiled module.

        Parameters
        ----------
        path : str | Path
            Path of the CUDA source file.
        **kwargs
            Passed on to :class:`CudaKernel` for every kernel. The directory of
            the file is always added to ``include_dirs``.

        Returns
        -------
        dict[str, CudaKernel]
            One kernel per ``__global__`` function, by name, in the order of
            the source (see :func:`cuda_kernel_names`).

        Raises
        ------
        ValueError
            If the file defines no ``__global__`` function.
        """
        path = Path(path)
        source = path.read_text()
        names = cuda_kernel_names(source)
        if not names:
            raise ValueError(f"no __global__ function found in {path}")
        include_dirs = (path.parent, *kwargs.pop("include_dirs", ()))
        return {
            name: cls(source, name, include_dirs=include_dirs, **kwargs)
            for name in names
        }

    @staticmethod
    def _check_block(block: tuple[int, ...]) -> tuple[int, ...]:
        if any(n <= 0 for n in block):
            raise ValueError(f"block sizes must be positive, got {block}")
        if math.prod(block) > _MAX_THREADS_PER_BLOCK:
            raise ValueError(
                f"a block has at most {_MAX_THREADS_PER_BLOCK} threads, got "
                f"{block} = {math.prod(block)}",
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

        The debug options and the header hash define are not part of them;
        they are added at compile time, see :meth:`compile_options`.
        """
        return self._options

    @property
    def debug(self) -> bool | None:
        """The kernel's debug setting: True, False, or None for the global one."""
        return self._debug

    def debug_active(self) -> bool:
        """Whether debug mode applies to this kernel now.

        The kernel's own setting if it was created with ``debug=True`` or
        ``debug=False``, else the global setting (:func:`cunumpy.cuda.get_cuda_debug`),
        read at the time of the call.
        """
        if self._debug is not None:
            return self._debug
        from cunumpy._device import get_cuda_debug

        return get_cuda_debug()

    @property
    def include_dirs(self) -> tuple[Path, ...]:
        """Directories searched for ``#include`` files."""
        return self._include_dirs

    @property
    def source_dir(self) -> Path | None:
        """Directory the source was read from, if known."""
        return self._source_dir

    @property
    def included_headers(self) -> tuple[Path, ...]:
        """The header files the source includes, recursively.

        Quoted includes (``#include "..."``) are resolved in `source_dir`,
        `include_dirs` and cunumpy's header directory
        (:func:`cuda_include_dir`); cunumpy's shipped headers are tracked
        also when included in angle brackets (``#include <cunumpy/...>``).
        Resolved at every access (see :func:`resolve_includes`), so the result
        follows the files on disk. Empty if the source includes no project or
        cunumpy headers.
        """
        return tuple(
            resolve_includes(
                self._source,
                self._include_dirs,
                base_dir=self._source_dir,
                angle_dirs=(cuda_include_dir(),),
            ),
        )

    def compile_options(self) -> tuple[str, ...]:
        """The NVRTC options a compilation now would use.

        :attr:`options`, followed by ``-I`` for cunumpy's own header directory
        (:func:`cunumpy.cuda.cuda_include_dir`) unless already present, then
        :data:`DEBUG_OPTIONS` (``-lineinfo`` and
        ``-DCUNUMPY_BOUNDS_CHECK``) if :meth:`debug_active` and they are not
        already among the options (``-G`` is not added: NVRTC does not support
        it), and, if the source includes header files,
        ``-DCUNUMPY_INCLUDE_HASH=0x<hash>`` with the hash of the contents of
        :attr:`included_headers` (see :func:`include_hash`). CuPy keys its
        kernel cache on the options, so a changed header means a recompile,
        also for a header shipped with cunumpy that changed in an upgrade.
        """
        options = self._options
        # cunumpy's own headers (<cunumpy/atomic.cuh>, ...) are always found
        cunumpy_include = f"-I{cuda_include_dir()}"
        if cunumpy_include not in options:
            options += (cunumpy_include,)
        if self.debug_active():
            options += tuple(o for o in DEBUG_OPTIONS if o not in options)
        headers = self.included_headers
        if headers:
            options += (f"-DCUNUMPY_INCLUDE_HASH=0x{include_hash(headers)}",)
        return options

    @property
    def structs(self) -> tuple[CudaStruct, ...]:
        """Struct types passed to the kernel by value."""
        return self._structs

    @property
    def n_threads_from(self) -> Callable[[tuple[Any, ...]], Any] | None:
        """Default launch size: a function of the positional arguments, or None.

        Called with the tuple of arguments of a launch that gives neither
        `n_threads` nor `grid`, and returns `n_threads` (an integer or a
        tuple), e.g. ``lambda args: args[2].n_markers`` for a kernel whose
        third argument is a struct argument object with the marker count.
        Default ``"auto"`` uses the first array's leading axes, matching the
        launch block dimensions; 1D launches use its first axis, one thread per
        row. Supported argument objects are searched in field/argument order.
        ``"first_array"`` always uses its first axis. None disables inference.
        Settable, also on the ``cuda_kernel`` of a :class:`~cunumpy.kernels.Kernel`.
        """
        return self._n_threads_from

    @n_threads_from.setter
    def n_threads_from(
        self,
        value: Callable[[tuple[Any, ...]], Any] | str | None,
    ) -> None:
        automatic = isinstance(value, str) and value == "auto"
        if automatic:
            value = self._default_n_threads
        elif isinstance(value, str) and value == "first_array":
            value = _first_array_length
        if value is not None and not callable(value):
            raise TypeError(
                "n_threads_from must be callable, 'auto', 'first_array' or None"
            )
        self._automatic_threads = automatic
        self._n_threads_from = value

    def _default_n_threads(self, args: tuple[Any, ...]) -> int | tuple[int, ...]:
        return _first_array_shape(args, len(self._block))

    @property
    def check_finite(self) -> bool:
        """Whether every launch checks the floating-point arrays for NaN or inf.

        After the launch (synchronized), every floating-point or complex array
        among the arguments, including the array fields of struct argument
        objects, is scanned, and a non-finite value raises ``RuntimeError``
        naming the kernel and the argument. Costs a synchronization and one
        pass over the arrays per launch; for debugging (e.g. a pusher writing
        NaN velocities), not for production. Settable.
        """
        return self._check_finite

    @check_finite.setter
    def check_finite(self, value: bool) -> None:
        self._check_finite = bool(value)

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
        """Whether compilation succeeded on the current CUDA device."""
        return bool(self._compiled) and _current_device() in self._compiled

    def compile(self, *, log_stream: Any = None) -> Any:
        """Compile the kernel now (it is otherwise compiled on the first call).

        The options are :meth:`compile_options`, evaluated now: a kernel
        compiled before debug mode was enabled keeps its options. Call
        :meth:`recompile` to refresh headers/options explicitly. Compilation
        and launch resources are cached separately on each CUDA device;
        failed compilation remains retryable. ``log_stream`` receives compiler
        output (a writable file object, or None).

        Returns
        -------
        cupy.RawKernel
            The compiled kernel; compiled once and cached (also on disk by CuPy,
            keyed on the source and :meth:`compile_options`). No launch is needed
            to surface compiler errors.

        Raises
        ------
        RuntimeError
            If CuPy or a GPU is not available.
        """
        from cunumpy.xp import cupy_available

        if not cupy_available():
            raise RuntimeError(
                f"cannot compile CUDA kernel {self.expression!r}: "
                "CuPy is not installed or no GPU is available",
            )
        import cupy as cp

        device = _current_device()
        if device not in self._compiled:
            options = self.compile_options()
            if self._template_args is None:
                raw = cp.RawKernel(
                    self._source,
                    self._name,
                    options=options,
                )
                raw.compile(log_stream=log_stream)
            else:
                module = cp.RawModule(
                    code=self._source,
                    options=options,
                    name_expressions=[self.expression],
                )
                module.compile(log_stream=log_stream)
                raw = module.get_function(self.expression)
            limits = _device_limits(device)
            attributes = dict(raw.attributes)
            static = max(0, attributes.get("shared_size_bytes", 0))
            dynamic = attributes.get("max_dynamic_shared_size_bytes", -1)
            if dynamic < 0:
                dynamic = max(0, int(limits["sharedMemPerBlock"]) - static)
            dynamic = min(dynamic, max(0, int(limits["sharedMemPerBlock"]) - static))
            # Do not publish failed compilation or incomplete setup as compiled.
            self._compiled[device] = _CompiledKernel(raw, limits, attributes, dynamic)
        return self._compiled[device].raw

    def recompile(self, *, log_stream: Any = None) -> Any:
        """Recompile on the current device using current headers/debug options.

        Other devices keep their compiled kernels. A failed rebuild remains
        uncompiled and can be retried; in-flight launches must finish first.
        """
        from cunumpy.xp import cupy_available

        if cupy_available():
            self._compiled.pop(_current_device(), None)
        return self.compile(log_stream=log_stream)

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """The arguments as passed to ``cupy.RawKernel``: flattened and checked.

        Argument objects with ``__cuda_args__()`` (including struct values) are
        flattened. If the signature is checked, the number of arguments, the
        dtype and C-contiguity of every array, every struct and every scalar
        are checked, Python scalars are cast to the declared C types, and arrays
        for array view parameters (``Array2D<double>``) are packed into
        (pointer, shape, strides).

        Raises
        ------
        TypeError
            Wrong number of arguments, a host array, an array of the wrong
            dtype or a non-contiguous array for a pointer parameter, an array of
            the wrong dtype or number of dimensions for an array view, a value
            of the wrong struct, or a scalar of an incompatible type.
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
                f"flattening argument objects, got {len(values)}",
            )
        return tuple([check(v) for check, v in zip(self._checkers, values)])

    def launch_shape(
        self,
        n_threads: int | Sequence[int] | None = None,
        *,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
        args: Sequence[Any] | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The ``(grid, block)`` a call with these launch arguments uses.

        Pass `args` to use default thread-count inference when neither
        `n_threads` nor `grid` is given. See :meth:`__call__`. A grid with a
        zero dimension launches nothing.
        """
        block_shape = (
            self._block
            if block is None
            else self._check_block(_as_shape(block, "block"))
        )
        if n_threads is None and grid is None and args is not None:
            if self._automatic_threads:
                n_threads = _first_array_shape(args, len(block_shape))
            elif self._n_threads_from is not None:
                n_threads = self._n_threads_from(tuple(args))
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
                    "numbers of dimensions",
                )
            block_shape = block_shape + (1,) * (len(threads) - 1)
        grid_shape = tuple((n + b - 1) // b for n, b in zip(threads, block_shape))
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
        an explicit `grid`. With neither, it is inferred from the first array's
        leading axes (or the configured `n_threads_from` callback).

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
        ValueError
            Dimensions, threads or static plus dynamic shared memory exceed
            this device/kernel's limits, or arrays/stream belong to another device.
        """
        grid_shape, block_shape = self.launch_shape(
            n_threads, grid=grid, block=block, args=args
        )
        shared_mem = operator.index(shared_mem)
        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        values = self.prepare_args(*args)
        if 0 in grid_shape:
            return

        kernel = self.compile()
        device = _current_device()
        state = self._compiled[device]
        self._validate_launch(state, device, grid_shape, block_shape, shared_mem)
        if stream is not None and getattr(stream, "device_id", device) not in (
            device,
            -1,
        ):
            raise ValueError(
                f"kernel {self.expression!r}: stream belongs to another device"
            )
        for label, array in _device_arrays_in(args):
            array_device = getattr(getattr(array, "device", None), "id", None)
            if array_device is not None and array_device != device:
                raise ValueError(
                    f"kernel {self.expression!r} on CUDA device {device}: "
                    f"{label} belongs to CUDA device {array_device}",
                )
        debug = self.debug_active()
        with stream if stream is not None else nullcontext():
            kernel(grid_shape, block_shape, values, shared_mem=shared_mem)
            if debug or self._check_finite:
                self._synchronize_after_launch(stream, grid_shape, block_shape)
            if self._check_finite:
                self._check_finite_arrays(args)

    def _validate_launch(
        self,
        state: _CompiledKernel,
        device: int,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        shared_mem: int,
    ) -> None:
        limits, attributes = state.limits, state.attributes
        name = limits.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        context = f"kernel {self.expression!r} on CUDA device {device} ({name})"
        for label, shape, maximum in (
            ("block", block, limits["maxThreadsDim"]),
            ("grid", grid, limits["maxGridSize"]),
        ):
            for axis, (requested, limit) in enumerate(zip(shape, maximum)):
                if requested > limit:
                    raise ValueError(
                        f"{context}: {label}[{axis}]={requested} exceeds the {limit} limit",
                    )
        threads_limit = int(limits["maxThreadsPerBlock"])
        kernel_limit = attributes.get("max_threads_per_block", -1)
        if kernel_limit > 0:
            threads_limit = min(threads_limit, kernel_limit)
        if math.prod(block) > threads_limit:
            raise ValueError(
                f"{context}: block {block} has {math.prod(block)} threads, "
                f"exceeding the {threads_limit} device/kernel limit",
            )
        static = max(0, attributes.get("shared_size_bytes", 0))
        maximum_shared = max(
            int(limits["sharedMemPerBlock"]),
            int(limits.get("sharedMemPerBlockOptin", 0)),
        )
        if shared_mem + static > maximum_shared:
            raise ValueError(
                f"{context}: static shared memory {static} + shared_mem={shared_mem} "
                f"exceeds the {maximum_shared} bytes a block may use on this device",
            )
        if shared_mem > state.dynamic_shared:
            state.raw.max_dynamic_shared_size_bytes = shared_mem
            state.dynamic_shared = shared_mem

    def _check_finite_arrays(self, args: tuple[Any, ...]) -> None:
        """Raise if a floating-point array among `args` holds NaN or inf."""
        import cupy as cp

        for label, array in _device_arrays_in(args):
            kind = getattr(array.dtype, "kind", "")
            if kind not in "fc":
                continue
            if not bool(cp.isfinite(array).all()):
                raise RuntimeError(
                    f"kernel {self.expression!r} left a NaN or inf in {label} "
                    f"(dtype {array.dtype}, shape {tuple(array.shape)})",
                )

    def _synchronize_after_launch(
        self,
        stream: Any,
        grid: tuple[int, ...],
        block: tuple[int, ...],
    ) -> None:
        """Wait for the launch and re-raise a CUDA error naming this kernel.

        Skipped while the stream is being captured into a CUDA graph: the
        launch is only recorded then, and synchronizing would invalidate the
        capture. Errors of the captured kernels surface when the graph is
        launched (in debug mode, synchronize after ``graph.launch()``).
        """
        if stream is None:
            import cupy as cp

            stream = cp.cuda.get_current_stream()
        if _is_capturing(stream):
            return
        try:
            stream.synchronize()
        except Exception as error:
            import cupy as cp

            if not isinstance(
                error,
                (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError),
            ):
                raise
            raise RuntimeError(
                f"CUDA error after launching kernel {self.expression!r} with "
                f"grid {grid} and block {block}: {error}",
            ) from error


def _is_capturing(stream: Any) -> bool:
    """Whether `stream` is being captured into a CUDA graph."""
    is_capturing = getattr(stream, "is_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except Exception:  # noqa: BLE001 - e.g. the legacy null stream cannot capture
        return False


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
                    f"the factory must return a CudaKernel, got {type(kernel).__name__}",
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

    def compile_all(
        self,
        keys: Iterable[Sequence[Hashable]] = (),
        *,
        jobs: int | None = 1,
    ) -> None:
        """Compile the given variants (created if needed) and all existing ones.

        Parameters
        ----------
        keys : Iterable[Sequence]
            Keys of variants to create and compile now, e.g. ``[(3, np.float64)]``.
        jobs : int | None
            Number of variants compiled at a time, in threads (NVRTC releases
            the GIL); None for the number of CPUs. See
            :meth:`KernelCatalog.compile_all <cunumpy.kernels.KernelCatalog.compile_all>`.
        """
        for key in keys:
            self.get(*key)
        _compile_in_threads(
            {key: kernel.compile for key, kernel in self._kernels.items()},
            jobs,
        )
