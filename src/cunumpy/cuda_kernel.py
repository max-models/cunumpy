"""CUDA kernels (``cupy.RawKernel``) called like their NumPy/Pyccel counterparts.

:class:`CudaKernel` wraps a CUDA C kernel so that it can be called with the same
arguments as the host kernel it mirrors:

* argument objects that implement the :class:`CudaArguments` protocol
  (a ``__cuda_args__()`` method) are flattened into their device arrays and
  scalars, so an object holding several arrays can be passed as one argument;
* the ``extern "C" __global__`` signature is parsed once, and every call is
  checked against it: the number of arguments, the dtype of every array, and
  every scalar. Python scalars are cast to the declared C type; a scalar that
  does not fit the declared type (a ``float`` for an ``int``, an integer out of
  range, a NumPy scalar that would lose precision) raises instead of reaching
  the kernel as a silently wrong value, which is what ``cupy.RawKernel`` would
  do;
* arrays are never converted or copied: they must already be CuPy arrays.

The number of threads is given at each call (``n_threads``); the kernel is
launched on ``ceil(n_threads / block_size)`` blocks.

This module imports CuPy only when a kernel is compiled, so it can be imported
(and signatures parsed) without CuPy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "CudaArguments",
    "CudaKernel",
    "CudaParameter",
    "parse_cuda_signature",
]


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
    """One parameter of a CUDA kernel signature.

    Attributes
    ----------
    name : str
        Parameter name.
    ctype : str
        Normalized C type without qualifiers or ``*``, e.g. ``"double"``.
    dtype : numpy.dtype | None
        NumPy dtype of the value (or of the pointed-to elements); ``None`` for
        ``void*``.
    pointer : bool
        Whether the parameter is a pointer (a device array).
    """

    name: str
    ctype: str
    dtype: np.dtype | None
    pointer: bool


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

_QUALIFIERS = {"const", "volatile", "__restrict__", "__restrict", "restrict"}

_COMPLEX = re.compile(r"(?:(?:thrust|cuda::std)::)?complex\s*<\s*(float|double)\s*>")
_TOKEN = re.compile(r"complex<(?:float|double)>|[A-Za-z_]\w*|\*|\[\s*\]")


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


def _parse_parameter(text: str) -> CudaParameter:
    text = _COMPLEX.sub(lambda m: f"complex<{m.group(1)}>", text)
    tokens = _TOKEN.findall(text)
    pointers = sum(1 for t in tokens if t == "*" or t.startswith("["))
    words = [t for t in tokens if t != "*" and not t.startswith("[")]
    words = [t for t in words if t not in _QUALIFIERS]
    if len(words) < 2:
        raise ValueError(f"cannot parse the kernel parameter {text.strip()!r}")
    name, ctype = words[-1], " ".join(words[:-1])

    if ctype == "void" and pointers == 1:
        return CudaParameter(name, ctype, None, True)
    if pointers > 1 or ctype not in _CTYPES:
        raise ValueError(
            f"cannot check the kernel parameter {text.strip()!r}: unsupported type "
            f"{ctype + '*' * pointers!r}"
        )
    return CudaParameter(name, ctype, np.dtype(_CTYPES[ctype]), pointers == 1)


def parse_cuda_signature(source: str, name: str) -> tuple[CudaParameter, ...]:
    """Parse the parameters of the ``__global__`` function `name` in `source`.

    Parameters
    ----------
    source : str
        CUDA C source code.
    name : str
        Name of the ``__global__`` function.

    Returns
    -------
    tuple[CudaParameter, ...]
        The parameters, in order.

    Raises
    ------
    ValueError
        If there is no such function, or a parameter has a type that cannot be
        checked (e.g. a template parameter, a macro or a pointer to pointer).
    """
    code = _strip_comments(source)
    match = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(", code)
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
    if params in ("", "void"):
        return ()
    return tuple(_parse_parameter(p) for p in params.split(","))


def _flatten(args: Sequence[Any]) -> list[Any]:
    values: list[Any] = []
    for arg in args:
        cuda_args = getattr(arg, "__cuda_args__", None)
        if cuda_args is not None:
            values.extend(cuda_args())
        else:
            values.append(arg)
    return values


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
        if isinstance(value, bool):
            if kind in "biu":
                return scalar_type(value)
        elif isinstance(value, int):
            return cast_int(int(value))
        elif isinstance(value, float):
            if kind in "fc":
                return scalar_type(value)
        elif isinstance(value, complex):
            if kind == "c":
                return scalar_type(value)
        raise TypeError(
            f"{_describe(param, index)} cannot take a value of type "
            f"{value_type.__name__}"
        )

    return check


class CudaKernel:
    """A CUDA C kernel, compiled with NVRTC through ``cupy.RawKernel``.

    Parameters
    ----------
    source : str
        CUDA C source code containing the ``extern "C" __global__`` function
        `name`.
    name : str
        Name of the kernel function in `source`.
    block_size : int
        Number of threads per block.
    options : Sequence[str]
        Additional NVRTC compiler options, e.g. ``("-std=c++17",)``.
    include_dirs : Sequence[str | Path]
        Directories searched for ``#include`` files (passed as ``-I<dir>``).
    check_signature : bool
        Parse the kernel signature and check (and cast) every call against it.
        Raises ``ValueError`` at construction if the signature cannot be parsed
        (e.g. templates or macros in the parameter list); pass False to launch
        with the arguments as they are, like ``cupy.RawKernel``.

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
        block_size: int = 128,
        options: Sequence[str] = (),
        include_dirs: Sequence[str | Path] = (),
        check_signature: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._source = source
        self._name = name
        self._block_size = int(block_size)
        self._options = tuple(options) + tuple(f"-I{d}" for d in include_dirs)
        self._signature = (
            parse_cuda_signature(source, name) if check_signature else None
        )
        # one checker per parameter, built once so that calls stay cheap
        self._checkers = (
            None
            if self._signature is None
            else [
                _pointer_checker(p, i) if p.pointer else _scalar_checker(p, i)
                for i, p in enumerate(self._signature)
            ]
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

    def __repr__(self) -> str:
        return f"CudaKernel(name={self._name!r}, block_size={self._block_size})"

    @property
    def name(self) -> str:
        """Name of the kernel function."""
        return self._name

    @property
    def source(self) -> str:
        """CUDA C source code."""
        return self._source

    @property
    def block_size(self) -> int:
        """Number of threads per block."""
        return self._block_size

    @property
    def options(self) -> tuple[str, ...]:
        """NVRTC compiler options, including ``-I`` include directories."""
        return self._options

    @property
    def signature(self) -> tuple[CudaParameter, ...] | None:
        """The parsed kernel parameters, or None if calls are not checked."""
        return self._signature

    def compile(self) -> Any:
        """Compile the kernel now (it is otherwise compiled on the first call).

        Returns
        -------
        cupy.RawKernel
            The compiled kernel; compiled once and cached (also on disk by CuPy).
        """
        if self._raw_kernel is None:
            from .xp import cupy_available

            if not cupy_available():
                raise RuntimeError(
                    f"cannot compile CUDA kernel {self._name!r}: "
                    "CuPy is not installed or no GPU is available"
                )
            import cupy as cp

            self._raw_kernel = cp.RawKernel(
                self._source, self._name, options=self._options
            )
        return self._raw_kernel

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """The arguments as passed to ``cupy.RawKernel``: flattened and checked.

        Argument objects with ``__cuda_args__()`` are flattened. If the signature
        is checked, the number of arguments, the dtype of every array and every
        scalar are checked, and Python scalars are cast to the declared C types.

        Raises
        ------
        TypeError
            Wrong number of arguments, a host array or an array of the wrong
            dtype for a pointer parameter, or a scalar of an incompatible type.
        OverflowError
            A Python integer out of range of the declared integer type.
        """
        values = _flatten(args)
        if self._signature is None:
            return tuple(values)

        if len(values) != len(self._signature):
            raise TypeError(
                f"{self._name}() takes {len(self._signature)} arguments after "
                f"flattening argument objects, got {len(values)}"
            )
        return tuple([check(v) for check, v in zip(self._checkers, values)])

    def __call__(
        self,
        *args: Any,
        n_threads: int,
        shared_mem: int = 0,
        stream: Any = None,
    ) -> None:
        """Launch the kernel.

        Parameters
        ----------
        *args
            Kernel arguments: CuPy arrays, scalars and argument objects with
            ``__cuda_args__()``, see :meth:`prepare_args`.
        n_threads : int
            Number of threads to launch, rounded up to a multiple of the block
            size; nothing is launched for 0.
        shared_mem : int
            Dynamic shared memory per block, in bytes.
        stream : cupy.cuda.Stream | None
            Stream to launch on; the current stream if None.
        """
        if n_threads < 0:
            raise ValueError(f"n_threads must be non-negative, got {n_threads}")
        values = self.prepare_args(*args)
        if n_threads == 0:
            return

        kernel = self.compile()
        grid = (math.ceil(n_threads / self._block_size),)
        with stream if stream is not None else nullcontext():
            kernel(grid, (self._block_size,), values, shared_mem=shared_mem)
