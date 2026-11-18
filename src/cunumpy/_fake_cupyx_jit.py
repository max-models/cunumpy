"""The ``cupyx.jit`` of the fake CuPy: raw kernels run on the CPU, for tests only.

A ``@jit.rawkernel()`` kernel is called once per thread, one thread after
another, with the thread and block indices set (``jit.threadIdx``,
``jit.blockIdx``, ``jit.blockDim``, ``jit.gridDim``, ``jit.grid``,
``jit.gridsize``) and the fake device arrays replaced by their host buffers, so
a :class:`~cunumpy.kernels.CupyJitKernel` can be checked against its host
kernel without a GPU. Atomics are plain operations (threads never overlap).

Not emulated: ``jit.syncthreads``, ``jit.shared_memory`` and warp functions
raise ``NotImplementedError``. Arithmetic follows NumPy/Python, not CUDA (e.g.
integer division of negative numbers), so compare with a tolerance.
"""

from __future__ import annotations

import builtins
import itertools
import threading
import types
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["make_module"]

_STATE = threading.local()


class _Dim3:
    """``threadIdx`` and friends: ``x``, ``y``, ``z`` of the running thread."""

    def __init__(self, key: str) -> None:
        self._key = key

    def _get(self) -> tuple[int, int, int]:
        value = getattr(_STATE, self._key, None)
        if value is None:
            raise RuntimeError(
                f"jit.{self._key} is only defined inside an emulated kernel launch"
            )
        return value

    @property
    def x(self) -> int:
        return self._get()[0]

    @property
    def y(self) -> int:
        return self._get()[1]

    @property
    def z(self) -> int:
        return self._get()[2]


def _as_dim3(value: int | Sequence[int]) -> tuple[int, int, int]:
    shape = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"grid and block have 1 to 3 dimensions, got {shape}")
    return tuple(int(n) for n in shape) + (1,) * (3 - len(shape))  # type: ignore[return-value]


def _grid(ndim: int) -> int | tuple[int, ...]:
    thread, block, dim = _STATE.threadIdx, _STATE.blockIdx, _STATE.blockDim
    index = tuple(block[a] * dim[a] + thread[a] for a in range(ndim))
    return index[0] if ndim == 1 else index


def _gridsize(ndim: int) -> int | tuple[int, ...]:
    grid, dim = _STATE.gridDim, _STATE.blockDim
    size = tuple(grid[a] * dim[a] for a in range(ndim))
    return size[0] if ndim == 1 else size


def _atomic(update: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def atomic(array: Any, index: Any, value: Any, *rest: Any) -> Any:
        old = array[index].copy() if hasattr(array[index], "copy") else array[index]
        array[index] = update(old, value, *rest)
        return old

    return atomic


def _not_emulated(name: str) -> Callable[..., Any]:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"jit.{name} is not emulated by the fake CuPy (threads run one after "
            "another); test this kernel on a GPU",
        )

    return refuse


class _EmulatedRawKernel:
    """A ``cupyx.jit.rawkernel`` that runs on the CPU, thread by thread."""

    __module__ = "cupyx.jit._interface"

    def __init__(self, func: Callable[..., Any], cupy: Any, device: bool) -> None:
        self._func = func
        self._cupy = cupy
        self._device = device
        self.__name__ = getattr(func, "__name__", "kernel")

    def _host(self, value: Any) -> Any:
        from cunumpy import _fake_cupy

        if isinstance(value, self._cupy.ndarray):
            return _fake_cupy.host_buffer(value)
        if isinstance(value, np.ndarray):
            raise TypeError(
                f"kernel {self.__name__!r}: NumPy arrays cannot be passed to a CuPy "
                "kernel; pass a cupy.ndarray"
            )
        return value

    def __call__(
        self,
        grid: int | Sequence[int],
        block: int | Sequence[int],
        args: Sequence[Any],
        shared_mem: int = 0,
        stream: Any = None,
    ) -> None:
        if self._device:
            raise TypeError("a device function cannot be launched")
        grid3, block3 = _as_dim3(grid), _as_dim3(block)
        values = [self._host(a) for a in args]
        saved = dict(vars(_STATE))
        _STATE.gridDim, _STATE.blockDim = grid3, block3
        try:
            for bz, by, bx in itertools.product(
                *(builtins.range(n) for n in grid3[::-1])
            ):
                _STATE.blockIdx = (bx, by, bz)
                for tz, ty, tx in itertools.product(
                    *(builtins.range(n) for n in block3[::-1])
                ):
                    _STATE.threadIdx = (tx, ty, tz)
                    self._func(*values)
        finally:
            vars(_STATE).clear()
            vars(_STATE).update(saved)

    def __getitem__(self, launch: tuple[Any, Any]) -> Callable[..., None]:
        grid, block = launch
        return lambda *args: self(grid, block, args)


def make_module(cupy: Any) -> types.ModuleType:
    """Return the fake ``cupyx.jit`` module for the fake `cupy`."""
    jit = types.ModuleType("cupyx.jit")
    jit.__cunumpy_fake__ = True
    jit.__doc__ = __doc__

    def rawkernel(*, mode: str = "cuda", device: bool = False) -> Callable[..., Any]:
        return lambda func: _EmulatedRawKernel(func, cupy, device)

    jit.rawkernel = rawkernel
    jit.threadIdx = _Dim3("threadIdx")
    jit.blockIdx = _Dim3("blockIdx")
    jit.blockDim = _Dim3("blockDim")
    jit.gridDim = _Dim3("gridDim")
    jit.grid = _grid
    jit.gridsize = _gridsize
    jit.range = builtins.range
    jit.atomic_add = _atomic(lambda old, value: old + value)
    jit.atomic_sub = _atomic(lambda old, value: old - value)
    jit.atomic_max = _atomic(max)
    jit.atomic_min = _atomic(min)
    jit.atomic_exch = _atomic(lambda old, value: value)
    jit.atomic_cas = _atomic(
        lambda old, compare, value: value if old == compare else old
    )
    for name in ("syncthreads", "syncwarp", "shared_memory", "shfl_sync", "laneid"):
        setattr(jit, name, _not_emulated(name))
    return jit
