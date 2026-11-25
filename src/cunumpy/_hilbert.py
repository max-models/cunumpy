"""Hilbert keys: the position of integer cells along a Hilbert curve.

Like a Morton key (:mod:`cunumpy._morton`), a Hilbert key orders cells so that
cells close on the curve are close in space, and the cells of every quadtree
(2D) or octree (3D) node are a contiguous range. The Hilbert curve has no jumps:
consecutive keys are neighbouring cells, which keeps the pieces of a curve cut
into equal parts more compact than the Morton curve does::

    keys = xp.algorithms.hilbert_encode(i, j, levels=10)

The keys follow Skilling's construction (J. Skilling, "Programming the Hilbert
curve", AIP Conf. Proc. 707, 381 (2004)): the coordinates are transformed in
place and their bits interleaved, axis 0 the highest bit of every group.
"""

from __future__ import annotations

from typing import Any

from cunumpy._morton import _check_levels, morton_encode
from cunumpy._philox import _module

__all__ = ["hilbert_encode"]


def hilbert_encode(*cells: Any, levels: int) -> Any:
    """Return the Hilbert keys of integer cell coordinates.

    Parameters
    ----------
    *cells : array of int
        One array per axis (2 or 3 of them), broadcast against each other; in
        ``[0, 2**levels)``.
    levels : int
        Bits per axis: 1 to 32 in 2D, 1 to 21 in 3D.

    Returns
    -------
    array of uint64
        The position of every cell along the Hilbert curve through the
        ``2**levels`` cells per axis, on the backend of `cells`: a permutation of
        ``0 .. 2**(ndim * levels) - 1`` over all cells, and cells with
        consecutive keys share a face.

    Raises
    ------
    ValueError
        If there are not 2 or 3 arrays, or `levels` is out of range.

    See Also
    --------
    morton_encode : The Z-order curve, with jumps between its quadrants.

    Examples
    --------
    >>> xp.algorithms.hilbert_encode(
    ...     xp.asarray([0, 0, 1, 1]), xp.asarray([0, 1, 1, 0]), levels=1
    ... )
    array([0, 1, 2, 3], dtype=uint64)
    """
    ndim = len(cells)
    _check_levels(ndim, levels)
    xpm = _module(*cells)
    x = [
        a.copy()
        for a in xpm.broadcast_arrays(
            *(xpm.asarray(c).astype(xpm.uint64) for c in cells)
        )
    ]
    one = xpm.uint64(1)
    zero = xpm.uint64(0)
    top = one << xpm.uint64(levels - 1)
    # inverse undo of the excess work (Skilling's AxestoTranspose)
    q = top
    while q > one:
        p = q - one
        for i in range(ndim):
            high = (x[i] & q) != zero
            # where the bit is set, flip the low bits of axis 0; else swap them
            swap = (x[0] ^ x[i]) & p
            x[0] = xpm.where(high, x[0] ^ p, x[0] ^ swap)
            x[i] = xpm.where(high, x[i], x[i] ^ swap)
        q = q >> one
    # Gray encode
    for i in range(1, ndim):
        x[i] = x[i] ^ x[i - 1]
    t = xpm.zeros(x[0].shape, dtype=xpm.uint64)
    q = top
    while q > one:
        t = xpm.where((x[ndim - 1] & q) != zero, t ^ (q - one), t)
        q = q >> one
    x = [axis ^ t for axis in x]
    # the transposed bits, axis 0 highest in every group
    return morton_encode(*reversed(x))
