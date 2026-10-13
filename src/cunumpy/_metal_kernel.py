"""Metal kernels for the GPU of Apple silicon Macs, run through MLX.

See :doc:`/kernels/metal-kernel`.
"""

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
    """Check whether a :class:`~cunumpy.kernels.MetalKernel` can run here.

    Returns
    -------
    bool
        True if MLX is installed and reports a Metal GPU.

    Examples
    --------
    >>> xp.kernels.metal_available()  # doctest: +SKIP
    True
    """
    try:
        _mlx()
    except (ImportError, RuntimeError):
        return False
    return True


class MetalKernel:
    """A Metal Shading Language kernel for the GPU of Apple silicon, run with MLX.

    Takes NumPy arrays, no backend switch needed: every call copies the inputs
    to MLX arrays and the results back into the output arrays you pass,
    counted by :func:`~cunumpy.profiling.count_transfers`. The kernel is
    compiled by MLX at its first call and cached. Needs ``cunumpy[metal]``;
    see :doc:`/kernels/metal-kernel`.

    Parameters
    ----------
    source : str
        Body of the kernel function. MLX generates the signature: each name
        in `inputs` and `outputs` is a pointer to the flat, row-major data of
        that array, and ``thread_position_in_grid`` and the other Metal
        attributes used in the body are added automatically.
    inputs : sequence of str
        Names of the input arrays, in call order.
    outputs : sequence of str
        Names of the output arrays, in the order passed as `out`.
    name : str, optional
        Name of the kernel; part of the compiled function name.
    header : str, optional
        Source before the kernel function: includes, defines, helper functions.
    threadgroup : int or sequence of int, optional
        Threads per threadgroup, 1 to 3 integers.
    float64 : {"error", "cast"}, optional
        The Apple GPU has no float64. ``"error"`` (default) raises
        ``TypeError`` for a float64 input or output; ``"cast"`` computes
        float64 data in float32.
    atomic_outputs : bool, optional
        Declare the outputs as ``device atomic<T>*`` for atomic updates.
    init_value : float, optional
        Value the outputs are filled with before the launch. Otherwise they
        are uninitialized: write every element, or pass the old array as an
        input too to read it.

    Raises
    ------
    ValueError
        If `float64` is unknown, there is no output, or names repeat.

    Examples
    --------
    >>> scale = xp.kernels.MetalKernel(
    ...     "uint i = thread_position_in_grid.x; y[i] = a[0] * x[i];",
    ...     inputs=["x", "a"],
    ...     outputs=["y"],
    ... )
    >>> y = np.empty(4, dtype=np.float32)
    >>> scale(np.arange(4, dtype=np.float32), 2.0, out=y)  # doctest: +SKIP
    array([0., 2., 4., 6.], dtype=float32)
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
        """Return `value` as a NumPy array; Python scalars become 1-element arrays."""
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
        """Launch the kernel; blocks until the results are copied back.

        Parameters
        ----------
        *args : numpy.ndarray or scalar
            The inputs, in the order of `inputs`. Python scalars become
            1-element float32 or int32 arrays; read them as ``a[0]``.
        out : numpy.ndarray or sequence of numpy.ndarray
            The output arrays, in the order of `outputs`; their shapes and
            dtypes define the outputs, and they are filled in place.
        n_threads : int or sequence of int, optional
            Total number of threads (1 to 3 dimensions), not threadgroups; by
            default the first axis of the first output.
        template : mapping, optional
            Compile-time constants (int, bool or dtype) by name, e.g.
            ``{"NSTEPS": 200}``; a new value compiles another variant.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The output array, or a tuple of them if there are several.

        Raises
        ------
        TypeError
            If the number of inputs or outputs is wrong, or a dtype is not
            supported on the GPU.
        ImportError
            If MLX is not installed.
        RuntimeError
            If MLX reports no Metal GPU.
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
