"""Timing, NVTX ranges and counting host/device transfers, on either backend.

:func:`timed_region` times a block (synchronizing the device first and last),
:class:`nvtx_range` marks it for Nsight (a no-op without NVTX), and
:func:`count_transfers` / :func:`assert_no_transfers` count the copies between
host and device that cunumpy makes inside a block::

    import cunumpy as xp

    with xp.profiling.timed_region("push") as t, xp.profiling.nvtx_range("push"):
        push(positions, velocities, dt)
    with xp.profiling.assert_no_transfers():
        step()
"""

from ._profiling import Timing, nvtx_range, timed_region
from ._transfers import (TransferCounter, TransferEvent, assert_no_transfers,
                         count_transfers)

__all__ = [
    "Timing",
    "TransferCounter",
    "TransferEvent",
    "assert_no_transfers",
    "count_transfers",
    "nvtx_range",
    "timed_region",
]
