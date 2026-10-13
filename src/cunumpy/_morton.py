"""Morton (Z-order) keys, the same on the host and in ``cunumpy/morton.cuh``.

A Morton key interleaves the bits of the integer cell coordinates of a point.
Sorting by it orders points along a Z-shaped curve: points close in space end
up close in memory, and the points of every quadtree (2D) or octree (3D) node
on the box are a contiguous range::

    keys = xp.algorithms.morton_keys(positions, lower, upper, levels)
    keys, order, positions = xp.algorithms.sort_by_key(keys, positions)
    node = keys >> np.uint64(ndim * (levels - level))  # node index at `level`

With ``levels`` bits per axis the key has ``ndim * levels`` bits; axis 0 is
the lowest bit of every group of ``ndim`` bits, and the top group is the child
of the root. In a kernel, ``cunumpy_morton_key2`` / ``cunumpy_morton_key3``
with the :func:`morton_scales` of the host return the keys of
:func:`morton_keys`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cunumpy._philox import _module

__all__ = [
    "MAX_MORTON_LEVELS",
    "morton_decode",
    "morton_encode",
    "morton_keys",
    "morton_scales",
]

#: Largest number of bits per axis that fits a uint64 key, by dimension.
MAX_MORTON_LEVELS = {2: 32, 3: 21}

_SPREAD = {
    2: (
        (16, 0x0000FFFF0000FFFF),
        (8, 0x00FF00FF00FF00FF),
        (4, 0x0F0F0F0F0F0F0F0F),
        (2, 0x3333333333333333),
        (1, 0x5555555555555555),
    ),
    3: (
        (32, 0x001F00000000FFFF),
        (16, 0x001F0000FF0000FF),
        (8, 0x100F00F00F00F00F),
        (4, 0x10C30C30C30C30C3),
        (2, 0x1249249249249249),
    ),
}
_LOW_BITS = {2: 0xFFFFFFFF, 3: 0x1FFFFF}


def _check_levels(ndim: int, levels: int) -> None:
    if ndim not in MAX_MORTON_LEVELS:
        raise ValueError(f"Morton keys support 2 or 3 dimensions, got {ndim}")
    if not 1 <= levels <= MAX_MORTON_LEVELS[ndim]:
        raise ValueError(
            f"levels must be between 1 and {MAX_MORTON_LEVELS[ndim]} in {ndim}D, "
            f"got {levels}",
        )


def _spread(xpm: Any, value: Any, ndim: int) -> Any:
    value = value & xpm.uint64(_LOW_BITS[ndim])
    for shift, mask in _SPREAD[ndim]:
        value = (value | (value << xpm.uint64(shift))) & xpm.uint64(mask)
    return value


def _compact(xpm: Any, value: Any, ndim: int) -> Any:
    steps = _SPREAD[ndim]
    value = value & xpm.uint64(steps[-1][1])
    masks = [mask for _, mask in steps[:-1]][::-1] + [_LOW_BITS[ndim]]
    shifts = [shift for shift, _ in steps][::-1]
    for shift, mask in zip(shifts, masks):
        value = (value ^ (value >> xpm.uint64(shift))) & xpm.uint64(mask)
    return value


def morton_encode(*cells: Any) -> Any:
    """Interleave the integer cell coordinates of every point into a uint64 key.

    Parameters
    ----------
    *cells : array of int
        One array per axis (2 or 3 of them), broadcast against each other;
        non-negative, fitting in 32 bits (2D) or 21 bits (3D).

    Returns
    -------
    array of uint64
        The keys, bit ``ndim * b + a`` holding bit ``b`` of axis ``a``, on the
        backend of `cells`.

    Raises
    ------
    ValueError
        If there are not 2 or 3 arrays.

    See Also
    --------
    morton_decode : The inverse.

    Examples
    --------
    >>> xp.algorithms.morton_encode(xp.asarray([1, 0, 3]), xp.asarray([0, 1, 3]))
    array([ 1,  2, 15], dtype=uint64)
    """
    ndim = len(cells)
    _check_levels(ndim, 1)
    xpm = _module(*cells)
    cells = xpm.broadcast_arrays(*(xpm.asarray(c).astype(xpm.uint64) for c in cells))
    key = xpm.zeros(cells[0].shape, dtype=xpm.uint64)
    for axis, cell in enumerate(cells):
        key |= _spread(xpm, cell, ndim) << xpm.uint64(axis)
    return key


def morton_decode(keys: Any, ndim: int) -> tuple[Any, ...]:
    """Return the integer cell coordinates of Morton keys.

    The inverse of :func:`morton_encode`.

    Parameters
    ----------
    keys : array of uint64
        Morton keys.
    ndim : int
        Number of axes, 2 or 3.

    Returns
    -------
    tuple of arrays of uint64
        One array per axis, shaped like `keys`, on its backend.

    Raises
    ------
    ValueError
        If `ndim` is not 2 or 3.

    Examples
    --------
    >>> xp.algorithms.morton_decode(xp.asarray([1, 2, 15], dtype=np.uint64), 2)
    (array([1, 0, 3], dtype=uint64), array([0, 1, 3], dtype=uint64))
    """
    _check_levels(ndim, 1)
    xpm = _module(keys)
    keys = xpm.asarray(keys).astype(xpm.uint64)
    return tuple(_compact(xpm, keys >> xpm.uint64(axis), ndim) for axis in range(ndim))


def morton_scales(lower: Sequence[float], upper: Sequence[float], levels: int) -> Any:
    """Return the cells per unit length of every axis, ``2**levels / (upper - lower)``.

    Pass them to ``cunumpy_morton_key2`` / ``cunumpy_morton_key3`` in a kernel so
    that it computes the keys of :func:`morton_keys`.

    Parameters
    ----------
    lower, upper : sequence of float
        Corners of the box, one value per axis (2 or 3).
    levels : int
        Bits per axis: 1 to 32 in 2D, 1 to 21 in 3D.

    Returns
    -------
    numpy.ndarray of float64
        One scale per axis, always on the host.

    Raises
    ------
    ValueError
        If the corners have different lengths, a dimension other than 2 or 3, or
        are equal on an axis, or if `levels` is out of range.

    Examples
    --------
    >>> xp.algorithms.morton_scales([0.0, 0.0], [1.0, 2.0], 4)
    array([16.,  8.])
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError(
            f"lower and upper must be sequences of equal length, got shapes "
            f"{lower.shape} and {upper.shape}",
        )
    _check_levels(lower.shape[0], levels)
    if np.any(upper == lower):
        raise ValueError("upper and lower must differ on every axis")
    return float(2**levels) / (upper - lower)


def morton_keys(
    positions: Any,
    lower: Sequence[float],
    upper: Sequence[float],
    levels: int,
) -> Any:
    """Return the Morton keys of points in the box ``[lower, upper]``.

    The cell of a point along an axis is ``floor((x - lower) * scale)`` with the
    :func:`morton_scales`, clipped to ``[0, 2**levels - 1]``: points on or outside
    the box get the cell at the nearest face, and a point on a cell boundary
    belongs to the upper cell. ``lower > upper`` on an axis reverses that axis.

    Parameters
    ----------
    positions : array of float, shape (n, ndim)
        The points, ``ndim`` 2 or 3; must be finite.
    lower, upper : sequence of float
        Corners of the box, one value per axis.
    levels : int
        Bits per axis: 1 to 32 in 2D, 1 to 21 in 3D.

    Returns
    -------
    array of uint64, shape (n,)
        The keys, on the backend of `positions`.

    Raises
    ------
    ValueError
        If the shapes do not match or `levels` is out of range.

    See Also
    --------
    sort_by_key : Sort points by their keys.

    Examples
    --------
    >>> positions = xp.asarray([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]])
    >>> xp.algorithms.morton_keys(positions, [0.0, 0.0], [1.0, 1.0], levels=2)
    array([ 0,  5, 15], dtype=uint64)
    """
    xpm = _module(positions)
    positions = xpm.asarray(positions)
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape (n, ndim), got {positions.shape}")
    ndim = positions.shape[1]
    scales = morton_scales(lower, upper, levels)
    if scales.shape[0] != ndim:
        raise ValueError(f"lower and upper need {ndim} values, got {scales.shape[0]}")
    top = float(2**levels - 1)
    cells = []
    for axis in range(ndim):
        cell = xpm.floor(
            (positions[:, axis] - float(lower[axis])) * float(scales[axis]),
        )
        cells.append(xpm.clip(cell, 0.0, top).astype(xpm.uint64))
    return morton_encode(*cells)
