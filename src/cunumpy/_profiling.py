"""NVTX ranges and timed regions (see :mod:`cunumpy.profiling`)."""

from __future__ import annotations

import importlib
import time
from collections.abc import Generator
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from types import ModuleType

from .xp import array_backend, synchronize


def _nvtx_module() -> ModuleType | None:
    """Return ``cupy.cuda.nvtx`` on the CuPy backend, else None.

    None is also returned when NVTX is not available in this CuPy build, so
    callers can degrade to a no-op instead of failing.
    """
    if array_backend.backend != "cupy":
        return None
    try:
        return importlib.import_module("cupy.cuda.nvtx")
    except Exception:  # noqa: BLE001 - tolerate any missing/broken NVTX
        return None


class nvtx_range(ContextDecorator):
    """Mark a code region as an NVTX range, visible in ``nsys`` and Nsight.

    On the CuPy backend the block is wrapped in ``cupy.cuda.nvtx.RangePush``
    / ``RangePop``, so the region appears on the profiler timeline next to
    the kernels it launches. On the NumPy backend, or if NVTX is not
    available, it is a no-op. The range is popped when the block raises.

    It can also be used as a decorator, and the same instance can be nested
    or re-entered (e.g. on a recursive function).

    Parameters
    ----------
    name : str
        Name shown in the profiler.
    color : int, optional
        Index into NVTX's colour table (``id_color`` of ``RangePush``).
        ``None`` uses the default colour.

    Examples
    --------
    >>> with xp.profiling.nvtx_range("push markers"):
    ...     kernel(markers, dt, n_threads=n)

    >>> @xp.profiling.nvtx_range("accumulate")
    ... def accumulate(...):
    ...     ...
    """

    def __init__(self, name: str, color: int | None = None) -> None:
        self.name = str(name)
        self.color = color
        self._stack: list[ModuleType | None] = []

    def __enter__(self):
        nvtx = _nvtx_module()
        if nvtx is not None:
            if self.color is None:
                nvtx.RangePush(self.name)
            else:
                nvtx.RangePush(self.name, id_color=int(self.color))
        self._stack.append(nvtx)
        return self

    def __exit__(self, *exc_info: object) -> None:
        nvtx = self._stack.pop()
        if nvtx is not None:
            nvtx.RangePop()

    def __repr__(self) -> str:
        return f"nvtx_range(name={self.name!r}, color={self.color!r})"


@dataclass
class Timing:
    """Result of a `timed_region()` block.

    Attributes
    ----------
    name : str
        Name of the region.
    elapsed : float | None
        Wall-clock seconds spent in the block; ``None`` until the block
        exits.
    synced : bool
        Whether the device was synchronized before the clock was read, i.e.
        whether `elapsed` includes the queued device work.
    """

    name: str
    elapsed: float | None = None
    synced: bool = False


@contextmanager
def timed_region(name: str, *, sync: bool = True) -> Generator[Timing, None, None]:
    """Time a code region, including the device work it queues.

    CUDA kernels run asynchronously: a wall-clock timer around a launch
    measures the launch, not the kernel. On the CuPy backend this context
    manager synchronizes the device on entry (so earlier queued work is not
    charged to the region) and, if `sync` is true, again on exit before the
    clock is read, so the measured time includes the kernels launched in
    the block. It also pushes an `nvtx_range()` of the same name, so the
    region shows in ``nsys``. On the NumPy backend it only times the block.
    The time is recorded when the block raises as well.

    Parameters
    ----------
    name : str
        Name of the region (also the NVTX range name).
    sync : bool, default True
        Synchronize the device before reading the clock on exit. With
        ``False`` the time is the host time only, as for a plain timer.

    Yields
    ------
    Timing
        `elapsed` is set (in seconds, from ``time.perf_counter``) when the
        block exits; `synced` tells whether the device was synchronized.

    Examples
    --------
    >>> with xp.profiling.timed_region("push markers") as timing:
    ...     kernel(markers, dt, n_threads=n)
    >>> print(f"{timing.name}: {timing.elapsed:.3f} s (synced={timing.synced})")
    """
    timing = Timing(name=str(name))
    on_gpu = array_backend.backend == "cupy"
    with nvtx_range(timing.name):
        if sync and on_gpu:
            synchronize()
        start = time.perf_counter()
        try:
            yield timing
        finally:
            if sync and on_gpu:
                synchronize()
                timing.synced = True
            timing.elapsed = time.perf_counter() - start
