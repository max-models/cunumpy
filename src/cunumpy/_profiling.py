"""NVTX ranges and timed regions (see :mod:`cunumpy.profiling`)."""

from __future__ import annotations

import importlib
import time
from collections.abc import Generator
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from types import ModuleType

from cunumpy.xp import array_backend, synchronize


def _nvtx_module() -> ModuleType | None:
    """Return ``cupy.cuda.nvtx`` on the CuPy backend with NVTX, else None."""
    if array_backend.backend != "cupy":
        return None
    try:
        return importlib.import_module("cupy.cuda.nvtx")
    except Exception:  # noqa: BLE001 - tolerate any missing/broken NVTX
        return None


class nvtx_range(ContextDecorator):
    """Mark a code region as an NVTX range, visible in ``nsys`` and Nsight.

    A context manager and decorator. On the CuPy backend it calls
    ``cupy.cuda.nvtx.RangePush`` / ``RangePop`` (also when the block raises),
    so the region shows on the timeline above the kernels it launches; it
    does not synchronize. On NumPy, or without NVTX, it does nothing. One
    instance may be nested or re-entered (e.g. on a recursive function). See
    :doc:`/guides/profiling`.

    Parameters
    ----------
    name : str
        Name shown in the profiler.
    color : int, optional
        Index into NVTX's colour table (``id_color`` of ``RangePush``); None
        uses the default colour.

    See Also
    --------
    timed_region : Time a region (also pushes an NVTX range).

    Examples
    --------
    >>> with xp.profiling.nvtx_range("push markers"):
    ...     x = xp.ones(3) * 2.0
    >>> @xp.profiling.nvtx_range("step")
    ... def step(x):
    ...     return 2.0 * x
    >>> step(xp.ones(2))
    array([2., 2.])
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
    """The result of a :func:`timed_region` block.

    Attributes
    ----------
    name : str
        Name of the region.
    elapsed : float or None
        Wall-clock seconds spent in the block; None until the block exits.
    synced : bool
        Whether the device was synchronized before the clock was read, i.e.
        whether `elapsed` includes the queued device work.

    Examples
    --------
    >>> with xp.profiling.timed_region("solve") as timing:
    ...     pass
    >>> timing.name, timing.synced
    ('solve', False)
    """

    name: str
    elapsed: float | None = None
    synced: bool = False


@contextmanager
def timed_region(name: str, *, sync: bool = True) -> Generator[Timing, None, None]:
    """Time a code region, including the device work it queues.

    CUDA kernels run asynchronously, so a plain timer measures only the
    launch. On the CuPy backend the device is synchronized on entry (earlier
    work is not charged to the region) and, if `sync` is true, on exit before
    the clock is read. The region is also an :class:`nvtx_range` of the same
    name. On NumPy it is a plain timer. The time is recorded when the block
    raises as well. See :doc:`/guides/profiling`.

    Parameters
    ----------
    name : str
        Name of the region (also the NVTX range name).
    sync : bool, optional
        Synchronize the device on exit before reading the clock (default).
        With False only the host time (the launch overhead) is measured.

    Yields
    ------
    Timing
        `elapsed` (seconds, from ``time.perf_counter``) is set when the block
        exits; `synced` tells whether the device was synchronized.

    Examples
    --------
    >>> with xp.profiling.timed_region("solve") as timing:
    ...     x = xp.linalg.solve(xp.eye(3), xp.ones(3))
    >>> timing.elapsed >= 0.0, timing.synced
    (True, False)
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
