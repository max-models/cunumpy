"""CUDA kernels (``cupy.RawKernel``) called like their NumPy/Pyccel counterparts.

:class:`~cunumpy.kernels.CudaKernel` parses the ``__global__`` signature once
and checks every call against it: argument count, array dtypes and
contiguity, structs, and scalars (cast to the declared C type, or raising
instead of reaching the kernel as a wrong value). Arrays are never copied to
the device. Argument objects (:class:`~cunumpy.arguments.CudaArguments`) are
flattened, C structs (:class:`~cunumpy.arguments.CudaStruct`) are passed by
value, and ``Array2D<T>`` parameters take strided CuPy arrays.
:class:`~cunumpy.kernels.CudaKernelVariants` caches generated kernels per
variant.

CuPy is imported only when a kernel is compiled, so kernels can be created and
signatures parsed without CuPy. See :doc:`/kernels/cuda-kernel`,
:doc:`/kernels/arguments` and :doc:`/kernels/debugging`.
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

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _record_sync

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
    """Return the directory of the CUDA headers shipped with cunumpy.

    :class:`~cunumpy.kernels.CudaKernel` always finds these headers:
    ``cunumpy/array_view.cuh`` (strided ``Array1D<T>`` to ``Array16D<T>``
    views), ``cunumpy/index.cuh`` (thread-index macros such as
    ``CUNUMPY_THREAD_1D(i, n)``), ``cunumpy/atomic.cuh`` (atomic adds),
    ``cunumpy/reduce.cuh`` and ``cunumpy/scan.cuh`` (warp and block
    reductions and prefix sums), ``cunumpy/random.cuh`` (Philox random
    numbers) and ``cunumpy/morton.cuh`` (Morton keys). See
    :doc:`/kernels/cuda-headers`.

    Returns
    -------
    str
        The directory, to pass as ``-I<dir>`` to other compilers.

    Examples
    --------
    >>> import os
    >>> os.path.isfile(os.path.join(xp.cuda.cuda_include_dir(), "cunumpy", "atomic.cuh"))
    True
    """
    return str(_CUDA_INCLUDE_DIR)


#: NVRTC options added in debug mode: line information for
#: ``compute-sanitizer``/``nsys`` and bounds checks in the array views
#: (no ``-G``: NVRTC does not support it).
DEBUG_OPTIONS = ("-lineinfo", "-DCUNUMPY_BOUNDS_CHECK")


class CudaArguments:
    """Base class for objects passed to a CUDA kernel as several arguments.

    A :class:`~cunumpy.kernels.CudaKernel` replaces every argument that has a
    ``__cuda_args__()`` method by the values it returns, in order. Subclassing
    is optional: any object with ``__cuda_args__()`` is flattened. See
    :doc:`/kernels/arguments`.

    Parameters
    ----------
    *values
        The kernel arguments this object stands for (CuPy arrays and
        scalars), in the order of the kernel signature.

    Examples
    --------
    >>> class Particles(xp.arguments.CudaArguments):
    ...     def __init__(self, positions, velocities):
    ...         super().__init__(positions, velocities, positions.shape[0])
    >>> Particles(np.zeros(3), np.ones(3)).__cuda_args__()
    (array([0., 0., 0.]), array([1., 1., 1.]), 3)
    >>> kernel(dt, Particles(x, v))  # doctest: +SKIP
    """

    def __init__(self, *values: Any) -> None:
        self._cuda_args = tuple(values)

    def __cuda_args__(self) -> tuple[Any, ...]:
        """Return the kernel arguments this object stands for."""
        return self._cuda_args


class CudaParameter(NamedTuple):
    """One parameter of a CUDA kernel signature, or one field of a struct.

    Returned by :func:`~cunumpy.cuda.parse_cuda_signature`.

    Attributes
    ----------
    name : str
        Parameter name.
    ctype : str
        C type without qualifiers or ``*``, e.g. ``"double"`` or
        ``"Array2D<double>"``.
    dtype : numpy.dtype or None
        Dtype of the value, of the pointed-to or viewed elements, or the
        structured dtype of a struct; None for ``void*``.
    pointer : bool
        Whether the parameter is a pointer (a device array).
    struct : CudaStruct or None
        The struct type, for a struct passed by value.
    view_ndim : int or None
        Number of dimensions of an array view (``Array1D<T>`` to
        ``Array16D<T>``, ``CArray1D<T>`` to ``CArray16D<T>``).
    contiguous : bool
        Whether the array view is C-contiguous (``CArray2D<T>``): packed
        without strides, takes C-contiguous arrays only.

    Examples
    --------
    >>> source = 'extern "C" __global__ void f(const double* x, int n) {}'
    >>> x, n = xp.cuda.parse_cuda_signature(source, "f")
    >>> x.ctype, x.dtype, x.pointer
    ('double', dtype('float64'), True)
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
# Array1D<T> to Array16D<T> and CArray1D<T> to CArray16D<T> (cunumpy/array_view.cuh),
# T a scalar type of _CTYPES
_VIEW = re.compile(r"\b(C?)Array([1-9]|1[0-6])D\s*<((?:[^<>]|complex<[^<>]*>)+?)>")
_TOKEN = re.compile(
    r"C?Array(?:[1-9]|1[0-6])D<[^<>]*(?:<[^<>]*>[^<>]*)?>|complex<(?:float|double)>"
    r"|[A-Za-z_]\w*|\*|\[\s*\]",
)


def _normalize_view(match: re.Match) -> str:
    """Normalize a view type: ``Array2D< const double >`` -> ``Array2D<double>``."""
    words = [w for w in match.group(3).split() if w not in _QUALIFIERS]
    return f"{match.group(1)}Array{match.group(2)}D<{' '.join(words)}>"


def _view_dtype(ndim: int, contiguous: bool = False) -> np.dtype:
    """Return the structured dtype of ``Array<ndim>D<T>`` (no strides if `contiguous`)."""
    fields = [("data", np.uint64), ("shape", np.int64, (ndim,))]
    if not contiguous:
        fields.append(("strides", np.int64, (ndim,)))
    return np.dtype(fields, align=True)


def ctype_of(dtype: Any) -> str:
    """Return the C type of a NumPy dtype.

    Useful to generate CUDA source or template arguments for a dtype. Complex
    dtypes map to ``complex<float>``/``complex<double>`` (include
    ``<cupy/complex.cuh>`` in the source).

    Parameters
    ----------
    dtype : dtype-like
        A boolean, integer, floating-point or complex dtype.

    Returns
    -------
    str
        The C type, e.g. ``"double"`` or ``"long long"``.

    Raises
    ------
    ValueError
        If the dtype has no C type.

    Examples
    --------
    >>> xp.cuda.ctype_of(np.float64), xp.cuda.ctype_of(np.int64)
    ('double', 'long long')
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
    """Return ``(name, quoted)`` of every ``#include`` in `source`, in order."""
    return [
        (quoted or angle, bool(quoted))
        for quoted, angle in _INCLUDE.findall(_strip_comments(source))
    ]


def resolve_includes(
    source: str,
    include_dirs: Iterable[str | Path] = (),
    *,
    base_dir: str | Path | None = None,
    angle_dirs: Iterable[str | Path] = (),
) -> list[Path]:
    r"""Return the header files a CUDA source includes, recursively.

    Each ``#include "name"`` (comments ignored) is resolved like NVRTC does:
    in `base_dir`, then `include_dirs`, then `angle_dirs`; found headers are
    scanned in turn, relative to their own directory. ``#include <name>`` is
    ignored unless found in `angle_dirs`, and so are includes that cannot be
    found (NVRTC reports them at compile time).

    Parameters
    ----------
    source : str
        CUDA C source code.
    include_dirs : iterable of str or Path, optional
        Directories searched in order (the ``-I`` options).
    base_dir : str or Path, optional
        Directory of the file `source` was read from, searched first.
    angle_dirs : iterable of str or Path, optional
        Directories searched last, also for ``#include <name>``;
        :class:`~cunumpy.kernels.CudaKernel` passes
        :func:`~cunumpy.cuda.cuda_include_dir`.

    Returns
    -------
    list of Path
        The headers, each once, in depth-first order of first inclusion.
        Empty, without touching the file system, if there is nothing to track.

    Examples
    --------
    >>> source = '#include <cunumpy/index.cuh>\n#include <cstdio>'
    >>> [p.name for p in xp.cuda.resolve_includes(source, angle_dirs=[xp.cuda.cuda_include_dir()])]
    ['index.cuh']
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
    """Return a short hex digest of the contents of files, in order.

    Only the contents count: moving a header keeps the hash, editing it
    changes it. Makes CuPy's kernel cache key depend on the included headers,
    see :meth:`CudaKernel.compile_options
    <cunumpy.kernels.CudaKernel.compile_options>`.

    Parameters
    ----------
    paths : iterable of str or Path
        Files to hash, e.g. from :func:`~cunumpy.cuda.resolve_includes`.

    Returns
    -------
    str
        The first 16 hex digits of the SHA-256 digest.

    Examples
    --------
    >>> import os
    >>> header = os.path.join(xp.cuda.cuda_include_dir(), "cunumpy", "index.cuh")
    >>> len(xp.cuda.include_hash([header]))
    16
    """
    digest = hashlib.sha256()
    for path in paths:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()[:16]


_GLOBAL_FUNCTION = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(")


def cuda_kernel_names(source: str) -> list[str]:
    """Return the names of the ``__global__`` functions in a CUDA source.

    Comments are ignored, templates included; a function declared more than
    once is listed once.

    Parameters
    ----------
    source : str
        CUDA C source code.

    Returns
    -------
    list of str
        The kernel names, in order of first appearance.

    Examples
    --------
    >>> source = '''
    ... extern "C" __global__ void scale(double* x) {}
    ... // __global__ void old(double* x) {}
    ... extern "C" __global__ void shift(double* x) {}'''
    >>> xp.cuda.cuda_kernel_names(source)
    ['scale', 'shift']
    """
    return list(dict.fromkeys(_GLOBAL_FUNCTION.findall(_strip_comments(source))))


def _compile_in_threads(
    compilers: Mapping[Hashable, Callable[[], Any]],
    jobs: int | None,
) -> list[Hashable]:
    """Run the `compilers` `jobs` at a time; raise the first error after all ran."""
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
    """Return a template argument as C++ source: a C type for dtypes, else a literal."""
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
    """Parse the parameters of a ``__global__`` function.

    C types map to dtypes as on Linux (LP64): ``int`` is int32, ``long`` and
    ``long long`` are int64.

    Parameters
    ----------
    source : str
        CUDA C source code.
    name : str
        Name of the ``__global__`` function.
    structs : iterable of CudaStruct, optional
        Struct types that may appear as parameters (by value). A definition
        of the struct in `source` must match.
    template_args : sequence, optional
        Template arguments if `name` is a function template: C types or
        dtypes (see :func:`~cunumpy.cuda.ctype_of`) for type parameters,
        ints or bools for non-type parameters.

    Returns
    -------
    tuple of CudaParameter
        The parameters, in order.

    Raises
    ------
    ValueError
        If there is no such function, `template_args` do not fit, a struct
        definition does not match its :class:`~cunumpy.arguments.CudaStruct`,
        or a parameter type cannot be checked (e.g. a macro or ``double**``).

    Examples
    --------
    >>> source = '''
    ... template <typename T>
    ... __global__ void scale(T* x, T factor, int n) {}'''
    >>> [(p.name, p.ctype, p.pointer) for p in xp.cuda.parse_cuda_signature(
    ...     source, "scale", template_args=[np.float32])]
    [('x', 'float', True), ('factor', 'float', False), ('n', 'int', False)]
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
    """Return the current CUDA device id, or None if CuPy was never imported."""
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
    """Checker for an array view: packs pointer, shape and strides (in elements)."""
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


def _max_view_ndim(params: Iterable[CudaParameter]) -> int:
    """Return the largest `view_ndim` among `params` and their struct fields (0 if none)."""
    best = 0
    for p in params:
        if p.view_ndim is not None:
            best = max(best, p.view_ndim)
        if p.struct is not None:
            best = max(best, _max_view_ndim(p.struct.fields))
    return best


def _checker(param: CudaParameter, index: int) -> Callable[[Any], Any]:
    if param.struct is not None:
        return _struct_checker(param, index)
    if param.view_ndim is not None:
        return _view_checker(param, index)
    if param.pointer:
        return _pointer_checker(param, index)
    return _scalar_checker(param, index)


def _field_dtype(field: CudaParameter) -> np.dtype:
    """Return the dtype of a struct field (pointers are stored as addresses)."""
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
    """Return the C type of a pyccel-style annotation, e.g. ``"float[:, :]"``."""
    if annotation is inspect.Parameter.empty:
        raise ValueError(f"{what} has no type annotation")
    if typing.get_origin(annotation) is typing.Final:
        (annotation,) = typing.get_args(annotation)
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        # with `from __future__ import annotations`, 'float[:]' arrives quoted:
        # "'float[:]'" (and Final['float[:]'] as one string), as in _annotation_text
        text = annotation.replace("'", "").replace('"', "")
        match = _PYCCEL_ANNOTATION.match(text.strip())
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
    if ndim > 16:
        raise ValueError(f"{what}: arrays have at most 16 dimensions, got {ndim}")
    return f"Array{ndim}D<{ctype}>"


def _apply_contiguous(
    fields: list[tuple[str, str]],
    contiguous: bool | Iterable[str],
    what: str,
) -> list[tuple[str, str]]:
    """Make all array fields, or the named ones, ``CArray<n>D<T>``."""
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
        "// Generated by cunumpy.arguments.CudaStruct from the Python definition; do not edit.",
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
    """Write the definitions of several structs to one header file.

    The header has an include guard, ``#include "cunumpy/array_view.cuh"`` if
    a struct has array view fields, any `includes`, then the structs in
    order. Commit it next to the kernels and let a test regenerate and
    compare it (see :meth:`CudaStruct.to_header`).

    Parameters
    ----------
    path : str or Path
        File to write.
    structs : iterable of CudaStruct
        The structs, in declaration order.
    guard : str, optional
        Include guard macro; by default from the file name
        (``pusher_args.cuh`` -> ``PUSHER_ARGS_CUH``).
    includes : iterable of str, optional
        Additional headers, as file names or ``#include`` lines.

    Returns
    -------
    str
        The header source that was written.

    Examples
    --------
    >>> import os, tempfile
    >>> Vec = xp.arguments.CudaStruct("Vec", [("data", "double*"), ("n", "int")])
    >>> path = os.path.join(tempfile.mkdtemp(), "vec.cuh")
    >>> print(xp.arguments.write_cuda_header(path, [Vec]), end="")
    // Generated by cunumpy.arguments.CudaStruct from the Python definition; do not edit.
    #ifndef VEC_CUH
    #define VEC_CUH
    <BLANKLINE>
    struct Vec {
        double* data;
        int n;
    };
    <BLANKLINE>
    #endif  // VEC_CUH
    """
    path = Path(path)
    source = _header_source(tuple(structs), guard or _header_guard(path.name), includes)
    path.write_text(source)
    return source


def _read_python_source(source: str | Path) -> tuple[str, str]:
    """Return ``(text, description)`` of a Python source given as a path or as code."""
    if isinstance(source, Path) or (
        "\n" not in source and source.strip().endswith(".py")
    ):
        path = Path(source)
        return path.read_text(), str(path)
    return str(source), "<source>"


def _find_init(tree: ast.Module, class_name: str, where: str) -> ast.FunctionDef:
    """Return the ``__init__`` of the class `class_name` in a parsed module."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
            raise ValueError(f"class {class_name!r} in {where} has no __init__")
    raise ValueError(f"no class {class_name!r} in {where}")


def _stored_parameters(init: ast.FunctionDef) -> dict[str, str]:
    """Return ``{parameter: attribute}`` for ``self.<attribute> = <parameter>``."""
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
    """Return the pyccel-style annotation string of a parsed annotation."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    text = ast.unparse(annotation)
    # Final['float[:]'] -> float[:] (the quotes come from the string annotation)
    return text.replace("'", "").replace('"', "")


class CudaStruct:
    """A C struct type passed to CUDA kernels by value.

    Defines the struct once: :attr:`declaration` is its C definition, and
    calling the struct packs values into a :class:`CudaStructValue` with the C
    memory layout (pointers stored as device addresses). Kernels created with
    ``structs=[...]`` check that they get a value of this struct and that a
    definition in their source matches. See :doc:`/kernels/arguments`.

    Parameters
    ----------
    name : str
        Name of the struct type in C.
    fields : sequence of (str, str)
        ``(field name, C type)`` pairs in order, e.g. ``("x", "double*")``.
        Scalars, pointers to scalars (or ``void*``), and array views
        ``Array1D<T>`` to ``Array16D<T>`` (packed as pointer, shape and
        strides in elements) or C-contiguous ``CArray1D<T>`` to
        ``CArray16D<T>`` are supported.

    Raises
    ------
    ValueError
        If the name is not a C identifier, a field type is not supported or
        field names repeat.

    Examples
    --------
    >>> Vec = xp.arguments.CudaStruct("Vec", [("data", "double*"), ("n", "int")])
    >>> print(Vec.declaration, end="")
    struct Vec {
        double* data;
        int n;
    };
    >>> source = Vec.declaration + r'''
    ... extern "C" __global__ void scale(Vec v, double a) {
    ...     int i = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (i < v.n) v.data[i] *= a;
    ... }'''
    >>> scale = xp.kernels.CudaKernel(source, "scale", structs=[Vec])
    >>> scale(Vec(data=x, n=x.size), 2.0)  # doctest: +SKIP
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

        One field per parameter of `func` (``self`` skipped), in order, typed
        by its pyccel-style annotation: ``float`` -> ``double``, ``int`` ->
        `int_type`, ``bool`` -> ``bool``, ``float[:, :]`` ->
        ``Array2D<double>`` (``Final[...]`` and ``const`` ignored). Real
        types (``int``, ``np.float32``, ...) work too. Typically `func` is the
        ``__init__`` of the host argument class, so that one class defines
        the arguments on host and device.

        Parameters
        ----------
        func : callable
            The function whose parameters define the fields.
        name : str
            Name of the struct type in C.
        int_type : str, optional
            C type for ``int``: ``"long long"`` (default, pyccel integers are
            64 bit) or ``"int"``.
        scalar_names : mapping of str to str, optional
            Additional or changed scalar mappings, e.g. ``{"float": "float"}``
            for single precision.
        contiguous : bool or iterable of str, optional
            Array fields that become C-contiguous views (``CArray2D<double>``):
            True for all, or their names. Packing them raises for a
            non-contiguous array.

        Returns
        -------
        CudaStruct
            The struct.

        Raises
        ------
        ValueError
            If a parameter has no annotation, or one that cannot be mapped
            (an unknown scalar, more than 16 dimensions).

        Examples
        --------
        >>> class MarkerArguments:
        ...     def __init__(self, markers: "float[:, :]", n_markers: int): ...
        >>> MarkerArgs = xp.arguments.CudaStruct.from_signature(
        ...     MarkerArguments.__init__, "MarkerArgs"
        ... )
        >>> print(MarkerArgs.declaration, end="")
        struct MarkerArgs {
            Array2D<double> markers;
            long long n_markers;
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
        """Build the struct from the ``__init__`` of a class in a Python source.

        Like :meth:`from_signature`, for a class whose module pyccel compiles
        (the compiled ``__init__`` has no Python signature): the source is
        parsed with :mod:`ast`, never imported or executed. A parameter stored
        as ``self.<attribute> = <parameter>`` gives a field named after the
        attribute, as the host kernels use it.

        Parameters
        ----------
        source : str or Path
            Path of the ``.py`` file, or the source code (a string with a
            newline or not ending in ``.py``).
        class_name : str
            Name of the class in the source.
        name : str, optional
            Name of the struct type in C; `class_name` by default.
        int_type : str, optional
            As for :meth:`from_signature`.
        scalar_names : mapping of str to str, optional
            As for :meth:`from_signature`.
        exclude : sequence of str, optional
            Parameter or attribute names that do not become fields, e.g.
            scratch arrays of the host class.
        attribute_names : bool, optional
            Name the fields after the attributes (default), else after the
            parameters.
        contiguous : bool or iterable of str, optional
            As for :meth:`from_signature`, with field names.

        Returns
        -------
        CudaStruct
            The struct.

        Raises
        ------
        ValueError
            If the class or its ``__init__`` is not found, or an annotation
            cannot be mapped.

        Examples
        --------
        >>> source = '''
        ... class MarkerArguments:
        ...     def __init__(self, markers: 'float[:, :]', n: 'int'):
        ...         self.markers = markers
        ...         self.n_markers = n
        ... '''
        >>> MarkerArgs = xp.arguments.CudaStruct.from_pyccel_class(source, "MarkerArguments")
        >>> [f.name for f in MarkerArgs.fields]
        ['markers', 'n_markers']
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
        """The fields, as :class:`~cunumpy.cuda.CudaParameter` tuples in order."""
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
        """The C definition, to put after ``cunumpy/array_view.cuh`` if :attr:`has_views`."""
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
        """Return the struct definition as a header, with include guard.

        The header includes ``cunumpy/array_view.cuh`` if the struct has array
        view fields, then any `includes`, then the definition. A test keeps a
        committed header in sync with the Python definition::

            def test_header_is_up_to_date():
                assert Path("pusher_args.cuh").read_text() == MarkerArgs.to_header()

        Parameters
        ----------
        path : str or Path, optional
            If given, the header is also written to this file.
        guard : str, optional
            Include guard macro; ``<NAME>_CUH`` by default.
        includes : iterable of str, optional
            Additional headers, as file names or ``#include`` lines.

        Returns
        -------
        str
            The header source.

        See Also
        --------
        cunumpy.arguments.write_cuda_header : Several structs in one header.
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
        """Check a definition of this struct in a CUDA source against its fields.

        Does nothing if `source` does not define the struct (e.g. it is in a
        header).

        Parameters
        ----------
        source : str
            CUDA C source code.

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
        """Return the CUDA source of the kernel used by :meth:`verify_layout`.

        The kernel ``cunumpy_layout_<name>(unsigned long long* out)`` writes
        ``sizeof``, ``alignof`` and the offset of every field, as the compiler
        lays out the struct.

        Parameters
        ----------
        include : str, optional
            Header that defines the struct, as a file name or ``#include``
            line; by default :attr:`declaration` is used.

        Returns
        -------
        str
            The CUDA source.
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
        """Check the CUDA compiler's struct layout against :attr:`dtype`.

        Compiles and runs a one-thread kernel (:meth:`layout_source`) and
        compares size, alignment and field offsets with the dtype values are
        packed into; a difference would make kernels silently read fields at
        the wrong place. Run it once per struct in a GPU test, especially with
        a hand-written header or a new compiler or platform (e.g. ROCm).

        Parameters
        ----------
        include : str, optional
            Header that defines the struct (file name or ``#include`` line),
            found in `include_dirs`; by default :attr:`declaration`.
        include_dirs : sequence of str or Path, optional
            Directories searched for `include`.
        options : sequence of str, optional
            Additional compiler options.

        Returns
        -------
        dict of str to int
            ``"sizeof"``, ``"alignof"`` and the offset of each field, by name.

        Raises
        ------
        ValueError
            If the compiled layout differs from :attr:`dtype` (the message
            lists each difference).
        RuntimeError
            If CuPy is not available.

        Examples
        --------
        >>> Vec.verify_layout()  # doctest: +SKIP
        {'sizeof': 16, 'alignof': 8, 'data': 0, 'n': 8}
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
        """Pack field values, given by name, into a struct value.

        Pointer fields take C-contiguous CuPy arrays of the declared dtype
        (never copied), array view fields CuPy arrays of the declared dtype
        and ndim (contiguous or not), and scalars are checked and cast like
        scalar kernel arguments.

        Parameters
        ----------
        **values
            One value per field.

        Returns
        -------
        CudaStructValue
            The packed value.

        Raises
        ------
        TypeError
            If a field is missing or unknown, or a value has the wrong type.
        OverflowError
            If an integer is out of range of its field type.
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

    Created by calling the struct. Keeps the arrays it points to alive (the
    packed struct only stores device addresses) and flattens into the packed
    struct through ``__cuda_args__()``. Field values are ``value["name"]``.

    Parameters
    ----------
    struct : CudaStruct
        The struct type.
    packed : numpy.void
        The packed struct.
    values : dict
        The field values, by name.

    Examples
    --------
    >>> Params = xp.arguments.CudaStruct("Params", [("dt", "double"), ("n", "int")])
    >>> value = Params(dt=0.1, n=10)
    >>> value["n"], value.packed["n"]
    (10, np.int32(10))
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
        """Return the packed struct, the one kernel argument this value stands for."""
        return (self._packed,)


class CudaStructArguments(CudaArguments):
    """Base class for argument objects passed to CUDA kernels as one C struct.

    The class form of :class:`CudaStruct`: a subclass sets ``struct_name``
    and ``fields``, stores every field as an attribute of the same name
    and calls :meth:`pack` at the end of its constructor. The object is
    passed to a :class:`~cunumpy.kernels.CudaKernel` as is and arrives as the
    packed struct. See :doc:`/kernels/arguments`.

    The struct is repacked whenever a field changed (another array, shape,
    strides or scalar value), checked at every use, so a field may be a
    property reading an owner's current array (second example). Copies and
    unpickled objects are packed from their own arrays. A subclass setting
    neither ``struct_name`` nor ``fields`` is an intermediate base
    class; setting only one raises ``TypeError``.

    Attributes
    ----------
    struct_name : str
        Name of the C struct type.
    fields : sequence of (str, str)
        ``(field name, C type)`` pairs, as for :class:`CudaStruct`.
    struct : CudaStruct
        The struct type, built once when the subclass is defined.

    Examples
    --------
    >>> class MarkerArguments(xp.arguments.CudaStructArguments):
    ...     struct_name = "MarkerArgs"
    ...     fields = (("markers", "Array2D<double>"), ("n_markers", "int"))
    ...
    ...     def __init__(self, markers):
    ...         self.markers = markers
    ...         self.n_markers = markers.shape[0]
    ...         self.pack()
    >>> print(MarkerArguments.struct.declaration, end="")
    struct MarkerArgs {
        Array2D<double> markers;
        int n_markers;
    };
    >>> push = xp.kernels.CudaKernel(source, "push", structs=[MarkerArguments.struct])  # doctest: +SKIP
    >>> push(MarkerArguments(markers), 0.1)  # doctest: +SKIP

    Fields as properties follow the arrays of an owner object, also after the
    owner replaced them (e.g. resized its marker array):

    >>> class ParticleArguments(xp.arguments.CudaStructArguments):
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

        Call it at the end of the constructor, so that invalid values raise
        there; later changes are repacked automatically.

        Raises
        ------
        TypeError
            If the class defines no struct, or a field value has the wrong
            type (see :meth:`CudaStruct.__call__ <cunumpy.arguments.CudaStruct.__call__>`).
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
        """Return the current value of every field attribute, by field name."""
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
        """The packed struct, repacked first if a field attribute changed."""
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
        """Return the packed struct, the one kernel argument this object stands for."""
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
    """Return what the packed struct depends on, to detect changed fields."""
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
    """Yield array shapes in argument order, including those in argument objects."""
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
    """Return the first axis of the first array (``n_threads_from="first_array"``)."""
    return typing.cast(int, _first_array_shape(args, 1))


def _last_axis_length(args: tuple[Any, ...]) -> int:
    """Return the last axis of the first array (``n_threads_from="last_axis"``)."""
    shape = next(_array_shapes_in(args), None)
    if shape is None:
        raise TypeError(
            "inferring n_threads needs an array argument; pass n_threads or grid, "
            "or set n_threads_from to a callable",
        )
    return int(shape[-1])


def _as_shape(value: int | Sequence[int], what: str) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"{what} must have 1 to 3 dimensions, got {shape}")
    return tuple(int(n) for n in shape)


def _device_arrays_in(args: Sequence[Any]) -> Iterator[tuple[str, Any]]:
    """Yield ``(label, array)`` for every device array among the arguments."""
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

    The ``__global__`` signature is parsed at construction (no GPU needed)
    and every call is checked against it: arrays must be C-contiguous CuPy
    arrays of the declared dtype (never copied), scalars are cast to the
    declared C type or raise, argument objects with ``__cuda_args__()`` are
    flattened. The kernel is compiled on the first call (or by
    :meth:`compile`) and cached per device, also on disk by CuPy. See
    :doc:`/kernels/cuda-kernel`.

    Parameters
    ----------
    source : str
        CUDA C source with the ``__global__`` function `name` (``extern "C"``
        unless it is a template).
    name : str
        Name of the kernel function.
    block_size : int or sequence of int, optional
        Threads per block, 1 to 3 dimensions, at most 1024 in total; 128 by
        default.
    options : sequence of str, optional
        Additional NVRTC options, e.g. ``("-std=c++17",)``.
    include_dirs : sequence of str or Path, optional
        Directories for ``#include`` (``-I<dir>``). cunumpy's own headers
        (:func:`~cunumpy.cuda.cuda_include_dir`) are always found.
    source_dir : str or Path, optional
        Directory the source was read from, searched first for
        ``#include "..."`` (set by :meth:`from_file`).
    structs : iterable of CudaStruct, optional
        Struct types the kernel takes by value.
    template_args : sequence, optional
        Template arguments if `name` is a function template, e.g.
        ``(np.float64, 3)`` for ``name<double, 3>``.
    check_signature : bool, optional
        Check and cast every call against the signature (default). False
        launches with the arguments as they are, like ``cupy.RawKernel``.
    debug : bool, optional
        Debug mode: compile with :data:`~cunumpy.cuda.DEBUG_OPTIONS` and
        synchronize after every launch, raising a ``RuntimeError`` naming
        this kernel. None (default) follows
        :func:`~cunumpy.cuda.set_cuda_debug` at every launch. See
        :doc:`/kernels/debugging`.
    n_threads_from : {"auto", "first_array", "last_axis"}, callable or None, optional
        Launch size when a call gives neither `n_threads` nor `grid`. "auto"
        (default): the leading axes of the first array argument (also inside
        argument objects), one per block dimension; "first_array": its first
        axis; "last_axis": its last axis (component-major ``(ncomp, N)``); a
        callable of the argument tuple, e.g. ``lambda args: args[0].size``;
        None: no inference. Settable later as :attr:`n_threads_from`.
    check_finite : bool, optional
        After every launch, synchronize and raise ``RuntimeError`` if a
        floating-point or complex array argument (struct fields included)
        holds NaN or inf. Costs a pass over the arrays: for debugging only.

    Raises
    ------
    ValueError
        If the signature cannot be parsed (e.g. macros in the parameter
        list) while `check_signature` is set, or `block_size` is invalid.

    Notes
    -----
    CuPy's disk cache is keyed on the source and options, not on included
    files. :meth:`compile_options` therefore adds a define with the hash of
    the :attr:`included_headers`, so editing a header (also one of cunumpy's)
    recompiles the kernel.

    Examples
    --------
    >>> axpy = xp.kernels.CudaKernel(r'''
    ... extern "C" __global__
    ... void axpy(double a, const double* x, double* y, int n) {
    ...     int i = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (i < n) y[i] += a * x[i];
    ... }''', "axpy")
    >>> axpy.launch_shape(1000)
    ((8,), (128,))
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
        """Create a kernel from a CUDA source file.

        Parameters
        ----------
        path : str or Path
            Path of the CUDA source file.
        name : str, optional
            Name of the kernel function; by default the file name without
            `suffix` (``axpy_cuda.cu`` -> ``axpy``).
        suffix : str, optional
            File name suffix stripped to get the default name.
        **kwargs
            Passed on to :class:`CudaKernel`. The file's directory is added
            to `include_dirs` and is the `source_dir`.

        Returns
        -------
        CudaKernel
            The kernel.

        Raises
        ------
        ValueError
            If `name` is not given and the file name does not end in `suffix`.

        Examples
        --------
        >>> push = xp.kernels.CudaKernel.from_file("push/push_cuda.cu")  # doctest: +SKIP
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
        """Create one kernel per ``__global__`` function of a file.

        The kernels share the source and options, so CuPy compiles the file
        once.

        Parameters
        ----------
        path : str or Path
            Path of the CUDA source file.
        **kwargs
            Passed on to :class:`CudaKernel` for every kernel. The file's
            directory is added to `include_dirs`.

        Returns
        -------
        dict of str to CudaKernel
            The kernels by name, in source order (see
            :func:`~cunumpy.cuda.cuda_kernel_names`).

        Raises
        ------
        ValueError
            If the file defines no ``__global__`` function.

        Examples
        --------
        >>> kernels = xp.kernels.CudaKernel.all_from_file("small_kernels.cu")  # doctest: +SKIP
        >>> kernels["scale"](x, 2.0, x.size)  # doctest: +SKIP
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
        """The compiled function: `name`, or its instantiation such as ``"scale<double, 3>"``."""
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
        """The threads per block: an integer for 1D blocks, else a tuple."""
        return self._block[0] if len(self._block) == 1 else self._block

    @property
    def options(self) -> tuple[str, ...]:
        """NVRTC options as given, with ``-I`` for `include_dirs` (see :meth:`compile_options`)."""
        return self._options

    @property
    def debug(self) -> bool | None:
        """The kernel's debug setting: True, False, or None for the global one."""
        return self._debug

    def debug_active(self) -> bool:
        """Return whether debug mode applies to this kernel now.

        Returns
        -------
        bool
            The kernel's own `debug` setting, else the current global one
            (:func:`~cunumpy.cuda.get_cuda_debug`).
        """
        if self._debug is not None:
            return self._debug
        from cunumpy._device import get_cuda_debug

        return get_cuda_debug()

    @property
    def include_dirs(self) -> tuple[Path, ...]:
        """The directories searched for ``#include`` files."""
        return self._include_dirs

    @property
    def source_dir(self) -> Path | None:
        """Directory the source was read from, if known."""
        return self._source_dir

    @property
    def included_headers(self) -> tuple[Path, ...]:
        """The header files the source includes, resolved at every access (see :func:`~cunumpy.cuda.resolve_includes`)."""
        return tuple(
            resolve_includes(
                self._source,
                self._include_dirs,
                base_dir=self._source_dir,
                angle_dirs=(cuda_include_dir(),),
            ),
        )

    def compile_options(self) -> tuple[str, ...]:
        """Return the compiler options a compilation now would use.

        These are :attr:`options`, ``-I`` for cunumpy's header directory,
        :data:`~cunumpy.cuda.DEBUG_OPTIONS` if :meth:`debug_active` (minus
        ``-lineinfo`` on a HIP/ROCm build, an NVRTC-only flag HIPRTC does not
        accept), and, if the source includes headers,
        ``-DCUNUMPY_INCLUDE_HASH=0x<hash>`` of their contents
        (:func:`~cunumpy.cuda.include_hash`). CuPy keys its kernel cache on
        the options, so a changed header means a recompile.

        Returns
        -------
        tuple of str
            The options.
        """
        options = self._options
        # cunumpy's own headers (<cunumpy/atomic.cuh>, ...) are always found
        cunumpy_include = f"-I{cuda_include_dir()}"
        if cunumpy_include not in options:
            options += (cunumpy_include,)
        if self.debug_active():
            from cunumpy._device import is_hip

            debug_options = DEBUG_OPTIONS
            if is_hip():
                debug_options = tuple(o for o in debug_options if o != "-lineinfo")
            options += tuple(o for o in debug_options if o not in options)
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
        """The launch size inference: a callable of the argument tuple, or None; settable as in the constructor."""
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
        elif isinstance(value, str) and value == "last_axis":
            value = _last_axis_length
        if value is not None and not callable(value):
            raise TypeError(
                "n_threads_from must be callable, 'auto', 'first_array', "
                "'last_axis' or None"
            )
        self._automatic_threads = automatic
        self._n_threads_from = value

    def _default_n_threads(self, args: tuple[Any, ...]) -> int | tuple[int, ...]:
        return _first_array_shape(args, len(self._block))

    @property
    def check_finite(self) -> bool:
        """Whether every launch synchronizes and raises on NaN or inf in its float arrays (debugging; settable)."""
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
        """Compile the kernel on the current device now, instead of at the first call.

        Compiled once per device and cached (also on disk by CuPy) with the
        :meth:`compile_options` of now: a kernel compiled before debug mode
        was enabled keeps its options (see :meth:`recompile`). A failed
        compilation can be retried.

        Parameters
        ----------
        log_stream : file-like, optional
            Writable file object that receives the compiler output.

        Returns
        -------
        cupy.RawKernel
            The compiled kernel.

        Raises
        ------
        RuntimeError
            If CuPy or a GPU is not available.
        NotImplementedError
            If the source includes ``cunumpy/reduce.cuh`` or
            ``cunumpy/scan.cuh`` and the active CuPy build targets HIP/ROCm:
            their warp-level shuffles assume a 32-lane warp and CUDA's
            ``_sync`` intrinsics, neither of which hold on AMD wavefronts
            (commonly 64 lanes on CDNA GPUs), so compiling them as-is would
            silently reduce or scan the wrong set of threads. Also raised, on
            a HIP/ROCm build, for a kernel using an ``Array5D``/``CArray5D``
            view or higher: launching one reliably corrupts the device there
            (observed as ``hipErrorIllegalState`` on a later launch), even for
            an in-bounds access; ``Array1D`` to ``Array4D`` are unaffected.
            Not yet root-caused (looks like a HIP code-generation or
            by-value-struct-argument issue in the generic, variadic-template
            ``ArrayView<T, N>`` that implements dimensions 5 and up, see
            ``cunumpy/array_view.cuh``). Also raised, on a HIP/ROCm build,
            for a kernel that includes ``cunumpy/array_view.cuh`` and
            compiles with ``-DCUNUMPY_BOUNDS_CHECK`` (directly or via debug
            mode): merely compiling the resulting ``printf()`` and trap
            reliably corrupts the device on any launch of that kernel, even
            one that never takes an out-of-bounds index; also not yet
            root-caused.
        """
        from cunumpy.xp import cupy_available

        if not cupy_available():
            raise RuntimeError(
                f"cannot compile CUDA kernel {self.expression!r}: "
                "CuPy is not installed or no GPU is available",
            )
        import cupy as cp

        from cunumpy._device import is_hip

        device = _current_device()
        if device not in self._compiled:
            if is_hip():
                unported = {"reduce.cuh", "scan.cuh"} & {
                    h.name for h in self.included_headers
                }
                if unported:
                    names = ", ".join(sorted(f"cunumpy/{n}" for n in unported))
                    raise NotImplementedError(
                        f"CUDA kernel {self.expression!r} includes {names}, which are "
                        "not yet ported to HIP/ROCm: their warp-level shuffles assume "
                        "a 32-lane warp and CUDA's `_sync` intrinsics, neither of "
                        "which hold on AMD wavefronts (commonly 64 lanes on CDNA "
                        "GPUs). Compiling them as-is would silently reduce/scan the "
                        "wrong set of threads rather than fail loudly.",
                    )
                if self._signature is not None:
                    max_view = _max_view_ndim(self._signature)
                    if max_view >= 5:
                        raise NotImplementedError(
                            f"CUDA kernel {self.expression!r} uses an "
                            f"Array{max_view}D/CArray{max_view}D view: on this "
                            "HIP/ROCm build, launching a kernel with a 5+ "
                            "dimensional array view reliably corrupts the device "
                            "(hipErrorIllegalState on a later launch), even for an "
                            "in-bounds access. Array1D to Array4D are unaffected; "
                            "this is not yet root-caused (likely a HIP code "
                            "generation or by-value struct argument issue in the "
                            "generic ArrayView<T, N> template, not something "
                            "cunumpy can safely paper over).",
                        )
            options = self.compile_options()
            if (
                is_hip()
                and "-DCUNUMPY_BOUNDS_CHECK" in options
                and "array_view.cuh" in {h.name for h in self.included_headers}
            ):
                raise NotImplementedError(
                    f"CUDA kernel {self.expression!r} includes cunumpy/array_view.cuh "
                    "and compiles with -DCUNUMPY_BOUNDS_CHECK (directly, or via debug "
                    "mode) on this HIP/ROCm build: merely compiling the resulting "
                    "printf() + trap in its bounds check reliably corrupts the "
                    "device on any launch of that kernel, even one that never takes "
                    "an out-of-bounds index. Not yet root-caused (a HIP "
                    "device-printf or trap-instruction issue is suspected); compile "
                    "without CUNUMPY_BOUNDS_CHECK (and without debug mode, which adds "
                    "it) for a kernel using array views on this build.",
                )
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
        """Recompile on the current device with the current headers and options.

        Other devices keep their kernels; in-flight launches must finish first.

        Parameters
        ----------
        log_stream : file-like, optional
            Writable file object that receives the compiler output.

        Returns
        -------
        cupy.RawKernel
            The compiled kernel.
        """
        from cunumpy.xp import cupy_available

        if cupy_available():
            self._compiled.pop(_current_device(), None)
        return self.compile(log_stream=log_stream)

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """Return the arguments as passed to ``cupy.RawKernel``: flattened and checked.

        Argument objects with ``__cuda_args__()`` are flattened. With a
        checked signature, the argument count, every array's dtype and
        contiguity, structs and scalars are checked, scalars are cast to the
        declared C types and arrays for view parameters (``Array2D<double>``)
        are packed into pointer, shape and strides.

        Parameters
        ----------
        *args
            The kernel arguments.

        Returns
        -------
        tuple
            The values to launch with.

        Raises
        ------
        TypeError
            If the argument count is wrong, an array is a host array, has the
            wrong dtype, contiguity or ndim, a struct value is of another
            struct, or a scalar has an incompatible type.
        OverflowError
            If an integer is out of range of the declared type.

        Examples
        --------
        >>> scale = xp.kernels.CudaKernel(
        ...     'extern "C" __global__ void scale(double a, int n) {}', "scale")
        >>> scale.prepare_args(2, 10)
        (np.float64(2.0), np.int32(10))
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
        """Return the ``(grid, block)`` a call with these launch arguments uses.

        Parameters
        ----------
        n_threads : int or sequence of int, optional
            Number of threads, as for :meth:`~cunumpy.kernels.CudaKernel.__call__`.
        grid : int or sequence of int, optional
            Number of blocks, instead of `n_threads`.
        block : int or sequence of int, optional
            Block shape, instead of :attr:`block_size`.
        args : sequence, optional
            Call arguments, to infer the launch size (:attr:`n_threads_from`)
            when neither `n_threads` nor `grid` is given.

        Returns
        -------
        tuple of (tuple of int, tuple of int)
            The grid and block shapes. A grid with a zero dimension launches
            nothing.

        Raises
        ------
        TypeError
            If not exactly one of `n_threads` and `grid` is given or inferred.

        Examples
        --------
        >>> kernel = xp.kernels.CudaKernel(
        ...     'extern "C" __global__ void f(double* a) {}', "f", block_size=(16, 16))
        >>> kernel.launch_shape((100, 50))
        ((7, 4), (16, 16))
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
        """Launch the kernel (asynchronously, unless in debug mode).

        The grid is `n_threads` divided by the block, rounded up, or an
        explicit `grid`; with neither, the size is inferred
        (:attr:`n_threads_from`).

        Parameters
        ----------
        *args
            Kernel arguments: CuPy arrays, scalars and argument objects, see
            :meth:`prepare_args`.
        n_threads : int or sequence of int, optional
            Number of threads in 1 to 3 dimensions, e.g. ``(nx, ny)``. With a
            1D block size the block is ``(block_size, 1, ...)``.
        grid : int or sequence of int, optional
            Number of blocks in 1 to 3 dimensions, instead of `n_threads`.
        block : int or sequence of int, optional
            Block shape for this call, instead of :attr:`block_size`.
        shared_mem : int, optional
            Dynamic shared memory per block in bytes (``extern __shared__``).
        stream : cupy.cuda.Stream, optional
            Stream to launch on; the current stream by default.

        Raises
        ------
        RuntimeError
            In debug mode, a CUDA error found by synchronizing after the
            launch (CuPy's error chained); otherwise such errors surface at a
            later synchronization. Also a NaN or inf with :attr:`check_finite`.
        ValueError
            If the launch exceeds the device or kernel limits (dimensions,
            threads, shared memory), or arrays or stream are on another device.
        """
        grid_shape, block_shape = self.launch_shape(
            n_threads, grid=grid, block=block, args=args
        )
        shared_mem = operator.index(shared_mem)
        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        values = self.prepare_args(*args)
        # RawKernel accepts size-one NumPy arrays for structs passed by value,
        # but not structured NumPy scalars (np.void). Keep their packed bytes
        # and alignment intact, including array-view pointers and strides.
        values = tuple(np.asarray(v) if isinstance(v, np.void) else v for v in values)
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
        """Wait for the launch and re-raise a CUDA error naming this kernel (not while graph-capturing)."""
        if stream is None:
            import cupy as cp

            stream = cp.cuda.get_current_stream()
        if _is_capturing(stream):
            return
        if _COUNTERS:
            _record_sync(f"debug synchronization after kernel {self.expression!r}")
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
    """Kernels generated per variant (e.g. per dtype and dimension), created once.

    For kernels whose source is generated per variant. The `factory` is called
    the first time a key is requested, and the kernel is cached per key.

    Parameters
    ----------
    factory : callable
        Returns the :class:`CudaKernel` for a variant, called with the key.

    Examples
    --------
    >>> def make_scale(dtype):
    ...     ctype = xp.cuda.ctype_of(dtype)
    ...     return xp.kernels.CudaKernel(
    ...         f'extern "C" __global__ void scale({ctype}* x, {ctype} a) {{}}', "scale")
    >>> scale = xp.kernels.CudaKernelVariants(make_scale)
    >>> scale.get(np.float32).signature[1].ctype
    'float'
    >>> scale.compile_all([(np.float64,)])  # doctest: +SKIP
    """

    def __init__(self, factory: Callable[..., CudaKernel]) -> None:
        self._factory = factory
        self._kernels: dict[tuple[Hashable, ...], CudaKernel] = {}

    def __repr__(self) -> str:
        return f"CudaKernelVariants({len(self._kernels)} variants)"

    def get(self, *key: Hashable) -> CudaKernel:
        """Return the kernel for a variant, created on first use.

        Parameters
        ----------
        *key
            The variant key, passed to the factory.

        Returns
        -------
        CudaKernel
            The kernel.

        Raises
        ------
        TypeError
            If the factory does not return a :class:`CudaKernel`.
        """
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
        """Return the keys of the variants created so far.

        Returns
        -------
        list of tuple
            The keys.
        """
        return list(self._kernels)

    def compile_all(
        self,
        keys: Iterable[Sequence[Hashable]] = (),
        *,
        jobs: int | None = 1,
    ) -> None:
        """Compile the given variants (created if needed) and all existing ones.

        All are compiled even if one fails; the first error is raised after.

        Parameters
        ----------
        keys : iterable of sequence, optional
            Keys of variants to create now, e.g. ``[(3, np.float64)]``.
        jobs : int or None, optional
            Number of variants compiled at a time, in threads (NVRTC releases
            the GIL); None for the number of CPUs. 1 by default.
        """
        for key in keys:
            self.get(*key)
        _compile_in_threads(
            {key: kernel.compile for key, kernel in self._kernels.items()},
            jobs,
        )
