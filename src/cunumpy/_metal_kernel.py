"""Metal kernels for the GPU of Apple silicon Macs, run through MLX."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _describe, _nbytes, _record

_FLOAT64 = np.dtype(np.float64)
_UNSUPPORTED = (np.dtype(np.complex128),)


def _mlx() -> Any:
    """Import ``mlx.core``, with an actionable error if it is not usable."""
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as error:
        raise ImportError(
            "MetalKernel needs MLX on an Apple silicon Mac: pip install 'cunumpy[metal]'",
        ) from error
    if not mx.metal.is_available():
        raise RuntimeError("MetalKernel needs a Metal GPU, but MLX reports none")
    return mx


def metal_available() -> bool:
    """Whether `MetalKernel` can run here (MLX installed and a Metal GPU present)."""
    try:
        _mlx()
    except (ImportError, RuntimeError):
        return False
    return True


class MetalKernel:
    """A Metal Shading Language kernel for the GPU of Apple silicon, run with MLX.

    The arrays are NumPy arrays: they are copied to MLX arrays for the launch
    and the results are written back into the output arrays you pass, like a
    :class:`~cunumpy.kernels.PyccelKernel` writes into its output arguments. No
    backend switch is needed. The copies are counted by
    :func:`~cunumpy.profiling.count_transfers`.

    Parameters
    ----------
    source : str
        Body of the kernel function (MSL). MLX generates the signature from
        `inputs` and `outputs`: every name is a pointer to the flat, row-major
        data of that array (``const device T*`` for inputs, ``device T*`` for
        outputs), and ``thread_position_in_grid`` and the other Metal
        attributes used in the body are added to the signature automatically.
        Names from `template` are available as compile-time constants.
    inputs : Sequence[str]
        Names of the input arrays, in the order they are passed to the call.
    outputs : Sequence[str]
        Names of the output arrays, in the order they are passed as `out`.
    name : str
        Name of the kernel; part of the compiled function name.
    header : str
        Source placed before the kernel function: includes, ``#define``\\ s and
        helper functions.
    threadgroup : int | Sequence[int]
        Threads per threadgroup: an integer, or 1 to 3 integers.
    float64 : {"error", "cast"}
        The GPU has no float64. With "error" (the default) a float64 input or
        output raises ``TypeError``. With "cast" float64 inputs are computed
        in float32 and float64 outputs are filled from float32 results, which
        is what to pick when float32 precision is enough.
    atomic_outputs : bool
        Declare the outputs as ``device atomic<T>*`` for atomic updates.
    init_value : float | None
        Value the outputs are filled with before the launch. Outputs are
        otherwise uninitialized: write every element, or pass the old array as
        an input as well to read its values.

    Notes
    -----
    The kernel is compiled by MLX at its first call and cached. All inputs
    and outputs are made C-contiguous, so ``a[i]`` in the source is the flat
    index of the NumPy array.

    Examples
    --------
    >>> scale = xp.kernels.MetalKernel(
    ...     "uint i = thread_position_in_grid.x; y[i] = a[0] * x[i];",
    ...     inputs=["x", "a"],
    ...     outputs=["y"],
    ... )
    >>> y = np.empty(n, dtype=np.float32)
    >>> scale(x, np.float32([2.0]), out=y)
    """

    def __init__(
        self,
        source: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        name: str = "cunumpy_kernel",
        header: str = "",
        threadgroup: int | Sequence[int] = 256,
        float64: str = "error",
        atomic_outputs: bool = False,
        init_value: float | None = None,
    ) -> None:
        if float64 not in ("error", "cast"):
            raise ValueError(f"float64 must be 'error' or 'cast', not {float64!r}")
        if not outputs:
            raise ValueError("a MetalKernel needs at least one output")
        names = [*inputs, *outputs]
        if len(set(names)) != len(names):
            raise ValueError(f"input and output names must be distinct: {names}")
        self.source = source
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.name = name
        self.header = header
        self.threadgroup = self._as_triple(threadgroup, "threadgroup", default=1)
        self.float64 = float64
        self.atomic_outputs = atomic_outputs
        self.init_value = init_value
        self._kernel: Any = None

    def __repr__(self) -> str:
        return (
            f"MetalKernel({self.name!r}, inputs={self.inputs}, outputs={self.outputs})"
        )

    @staticmethod
    def _as_triple(value: int | Sequence[int], what: str, default: int) -> tuple:
        values = (int(value),) if isinstance(value, (int, np.integer)) else tuple(value)
        if not 1 <= len(values) <= 3 or any(v < 1 for v in values):
            raise ValueError(f"{what} must be 1 to 3 positive integers, got {value!r}")
        return values + (default,) * (3 - len(values))

    @staticmethod
    def _as_input(value: Any) -> np.ndarray:
        """A NumPy array; Python scalars become float32 or int32 arrays of shape (1,)."""
        if isinstance(value, (bool, int)):
            return np.array([value], dtype=np.int32)
        if isinstance(value, float):
            return np.array([value], dtype=np.float32)
        array = np.asarray(value)
        return array.reshape(1) if array.ndim == 0 else array

    def _compiled(self, mx: Any) -> Any:
        if self._kernel is None:
            self._kernel = mx.fast.metal_kernel(
                name=self.name,
                input_names=list(self.inputs),
                output_names=list(self.outputs),
                source=self.source,
                header=self.header,
                ensure_row_contiguous=True,
                atomic_outputs=self.atomic_outputs,
            )
        return self._kernel

    def _device_dtype(self, dtype: np.dtype, role: str, name: str) -> np.dtype:
        if dtype == _FLOAT64 and self.float64 == "cast":
            return np.dtype(np.float32)
        if dtype == _FLOAT64 or dtype in _UNSUPPORTED:
            raise TypeError(
                f"{role} {name!r} of MetalKernel {self.name!r} has dtype {dtype}, "
                "which the Apple GPU does not support; use float32, or "
                "float64='cast' to compute in float32",
            )
        return dtype

    def __call__(
        self,
        *args: Any,
        out: Any,
        n_threads: int | Sequence[int] | None = None,
        template: Mapping[str, Any] | None = None,
    ) -> Any:
        """Launch the kernel.

        Parameters
        ----------
        *args : numpy.ndarray or scalar
            The inputs, in the order of `inputs`. Python scalars become
            1-element float32 or int32 arrays; read them as ``a[0]`` in the source.
        out : numpy.ndarray or Sequence[numpy.ndarray]
            The output arrays, in the order of `outputs`; their shapes and
            dtypes define the outputs and they are filled in place.
        n_threads : int | Sequence[int] | None
            Total number of threads (1 to 3 dimensions), not threadgroups. The
            default is the first axis of the first output.
        template : Mapping[str, int | bool | numpy.dtype] | None
            Compile-time constants; each name is a constant in the source. A
            different value compiles another variant.

        Returns
        -------
        The output array, or a tuple of them if there are several.
        """
        mx = _mlx()
        if len(args) != len(self.inputs):
            raise TypeError(
                f"MetalKernel {self.name!r} takes {len(self.inputs)} input(s) "
                f"{self.inputs}, got {len(args)}",
            )
        outs = (out,) if isinstance(out, np.ndarray) else tuple(out or ())
        if len(outs) != len(self.outputs) or not all(
            isinstance(o, np.ndarray) for o in outs
        ):
            raise TypeError(
                f"out must be {len(self.outputs)} NumPy array(s) for {self.outputs}",
            )

        host_inputs = [self._as_input(a) for a in args]
        for name, array in zip(self.inputs, host_inputs):
            self._device_dtype(array.dtype, "input", name)
        device_dtypes = [
            self._device_dtype(o.dtype, "output", name)
            for name, o in zip(self.outputs, outs)
        ]
        if n_threads is None:
            n_threads = outs[0].shape[0] if outs[0].ndim else 1
        grid = self._as_triple(n_threads, "n_threads", default=1)

        mlx_inputs = []
        for array in host_inputs:
            if array.dtype == _FLOAT64:
                array = array.astype(np.float32)
            mlx_inputs.append(mx.array(np.ascontiguousarray(array)))
            if _COUNTERS:
                _record(
                    "to_device",
                    f"MetalKernel {self.name!r} input ({_describe(array)})",
                    nbytes=_nbytes(array),
                )

        results = self._compiled(mx)(
            inputs=mlx_inputs,
            template=list((template or {}).items()),
            grid=grid,
            threadgroup=self.threadgroup,
            output_shapes=[o.shape for o in outs],
            output_dtypes=[getattr(mx, dtype.name) for dtype in device_dtypes],
            init_value=self.init_value,
        )
        mx.eval(*results)
        for result, host in zip(results, outs):
            np.copyto(host, np.asarray(result), casting="same_kind")
            if _COUNTERS:
                _record(
                    "to_host",
                    f"MetalKernel {self.name!r} output ({_describe(host)})",
                    nbytes=_nbytes(host),
                )
        return outs[0] if len(outs) == 1 else outs
