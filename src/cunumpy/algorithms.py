"""Array algorithms missing from NumPy/CuPy, on either backend.

Morton (Z-order) keys (the same as ``cunumpy/morton.cuh`` computes in a
kernel, at most ``MAX_MORTON_LEVELS`` bits per axis), a stable sort of several
arrays by one key, sums per key, and the compaction of the live rows of
particle arrays::

    keys = xp.algorithms.morton_keys(positions, lower, upper, levels)
    keys, order, positions = xp.algorithms.sort_by_key(keys, positions)
    charge = xp.algorithms.segment_sum(q, cell, n_cells)
    n = xp.algorithms.compact_by_mask(alive, positions, charges)

See :doc:`/guides/particle-codes`.
"""

from cunumpy._algorithms import (
    SegmentPlan,
    cell_offsets,
    compact_by_mask,
    segment_boundaries,
    segment_sum,
    sort_by_key,
)
from cunumpy._morton import (
    MAX_MORTON_LEVELS,
    morton_decode,
    morton_encode,
    morton_keys,
    morton_scales,
)

__all__ = [
    "MAX_MORTON_LEVELS",
    "SegmentPlan",
    "cell_offsets",
    "compact_by_mask",
    "morton_decode",
    "morton_encode",
    "morton_keys",
    "morton_scales",
    "segment_boundaries",
    "segment_sum",
    "sort_by_key",
]
