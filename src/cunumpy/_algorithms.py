"""Sums per key and sorting by key on either backend (see :mod:`cunumpy.algorithms`)."""

from __future__ import annotations

from typing import Any

import array_api_compat.numpy as np

from .xp import get_array_backend, get_array_module


def segment_sum(values: Any, keys: Any, n_segments: int) -> Any:
    """Sum `values` per key: ``out[k] = sum(values[i] for keys[i] == k)``.

    The reduction step of a sort-then-reduce accumulation (particles binned to
    cells, contributions summed per cell), on either backend, with
    ``bincount`` under the hood. For a 2D `values` the columns are summed
    separately (one bincount per column).

    Parameters
    ----------
    values : array
        Shape ``(n,)`` or ``(n, m)``, on the backend of `keys`.
    keys : array
        Integer segment of every value, shape ``(n,)``; a negative key drops
        the value (e.g. a particle outside the grid).
    n_segments : int
        Number of segments; keys must be smaller than it.

    Returns
    -------
    array
        Shape ``(n_segments,)`` or ``(n_segments, m)``, dtype of `values` for
        floating-point and complex values, ``float64`` otherwise.
    """
    xpm = get_array_module(keys)
    keys = xpm.asarray(keys)
    values = xpm.asarray(values)
    if keys.ndim != 1 or values.shape[:1] != keys.shape:
        raise ValueError(
            f"keys must be 1D with one entry per value, got keys {keys.shape} and "
            f"values {values.shape}"
        )
    if values.ndim not in (1, 2):
        raise ValueError(f"values must be 1D or 2D, got shape {values.shape}")
    if bool((keys >= n_segments).any()):
        raise ValueError(f"keys must be smaller than n_segments={n_segments}")
    valid = keys >= 0
    if not bool(valid.all()):
        keys = keys[valid]
        values = values[valid]
    out_dtype = values.dtype if values.dtype.kind in "fc" else np.dtype(np.float64)
    if values.ndim == 1:
        if values.dtype.kind == "c":
            real = xpm.bincount(keys, weights=values.real, minlength=n_segments)
            imag = xpm.bincount(keys, weights=values.imag, minlength=n_segments)
            return (real + 1j * imag).astype(out_dtype, copy=False)
        return xpm.bincount(keys, weights=values, minlength=n_segments).astype(
            out_dtype, copy=False
        )
    out = xpm.empty((n_segments, values.shape[1]), dtype=out_dtype)
    for j in range(values.shape[1]):
        out[:, j] = segment_sum(values[:, j], keys, n_segments)
    return out


def sort_by_key(keys: Any, *arrays: Any) -> tuple[Any, ...]:
    """Sort `keys` and reorder every array the same way, in one stable argsort.

    The usual first step of a particle code on the GPU: sort the particles by
    cell index or Morton key (:func:`cunumpy.algorithms.morton_keys`), then work on
    contiguous ranges. The sort is stable, so equal keys keep their order and
    the result is reproducible::

        keys, order, positions, charges = xp.algorithms.sort_by_key(keys, positions, charges)

    Parameters
    ----------
    keys : array, shape (n,)
        The sort keys.
    *arrays : arrays
        Arrays with ``n`` rows, on the backend of `keys`, reordered along
        axis 0.

    Returns
    -------
    tuple
        ``(sorted_keys, order, *sorted_arrays)``: ``order`` (int64) is the
        permutation, ``sorted_keys = keys[order]``, and each sorted array is
        ``array[order]`` (a new array).
    """
    if get_array_backend(keys) == "cupy":
        import cupy as xpm  # its argsort is a stable radix sort
    else:
        xpm = np
    keys = xpm.asarray(keys)
    if keys.ndim != 1:
        raise ValueError(f"keys must be 1D, got shape {keys.shape}")
    for array in arrays:
        if array.shape[:1] != keys.shape:
            raise ValueError(
                f"every array needs {keys.shape[0]} rows, got shape {array.shape}"
            )
    order = xpm.argsort(keys, kind="stable").astype(xpm.int64, copy=False)
    return (keys[order], order, *(array[order] for array in arrays))
