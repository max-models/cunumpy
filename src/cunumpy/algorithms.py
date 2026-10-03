"""Array algorithms missing from NumPy/CuPy, on either backend.

Morton (Z-order) keys (the same as ``cunumpy/morton.cuh`` computes in a
kernel), a stable sort of several arrays by one key, and sums per key::

    import cunumpy as xp

    keys = xp.algorithms.morton_keys(positions, lower, upper, levels)
    keys, order, positions = xp.algorithms.sort_by_key(keys, positions)
    charge = xp.algorithms.segment_sum(q, cell, n_cells)
"""

from .morton import (
    MAX_MORTON_LEVELS,
    morton_decode,
    morton_encode,
    morton_keys,
    morton_scales,
)
from .xp import segment_sum, sort_by_key

__all__ = [
    "MAX_MORTON_LEVELS",
    "morton_decode",
    "morton_encode",
    "morton_keys",
    "morton_scales",
    "segment_sum",
    "sort_by_key",
]
