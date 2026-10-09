"""Sums per key and sorting by key on either backend (see :mod:`cunumpy.algorithms`)."""

from __future__ import annotations

import math
import operator
from typing import Any

import array_api_compat.numpy as np

from cunumpy.xp import assert_same_backend, get_array_backend, get_array_module


def _integer_keys(keys: Any) -> tuple[Any, Any]:
    xpm = get_array_module(keys)
    keys = xpm.asarray(keys)
    if keys.ndim != 1:
        raise ValueError(f"keys must be 1D, got shape {keys.shape}")
    if keys.dtype.kind not in "iu":
        raise TypeError("keys must have an integer dtype")
    return xpm, keys


def _segment_count(n_segments: int) -> int:
    n_segments = operator.index(n_segments)
    if n_segments < 0:
        raise ValueError("n_segments must be non-negative")
    return n_segments


def _sorted_keys(keys: Any) -> tuple[Any, Any]:
    xpm, keys = _integer_keys(keys)
    if bool((keys[1:] < keys[:-1]).any()):
        raise ValueError("keys must be sorted in nondecreasing order")
    return xpm, keys


def segment_boundaries(sorted_keys: Any) -> tuple[Any, Any, Any]:
    """Return the runs of equal keys in sorted integer keys.

    Sparse, negative and uint64 Morton keys are supported. The results stay on
    the backend of `sorted_keys`. Validation and the variable-length output may
    synchronize on CUDA: compute the boundaries once and reuse them.

    Parameters
    ----------
    sorted_keys : array of int, shape (n,)
        Keys sorted in nondecreasing order.

    Returns
    -------
    unique_keys : array of int
        The distinct keys.
    starts, stops : array of int64
        Run ``i`` is the half-open slice ``[starts[i], stops[i])`` of the input.

    Raises
    ------
    ValueError
        If `sorted_keys` is not 1D or not sorted.
    TypeError
        If `sorted_keys` is not an integer array.

    See Also
    --------
    cell_offsets : Dense offsets for every cell, also the empty ones.

    Examples
    --------
    >>> xp.algorithms.segment_boundaries(xp.asarray([3, 3, 5, 9, 9, 9]))
    (array([3, 5, 9]), array([0, 2, 3]), array([2, 3, 6]))
    """
    xpm, keys = _sorted_keys(sorted_keys)
    if keys.size == 0:
        return keys.copy(), xpm.empty(0, dtype=np.int64), xpm.empty(0, dtype=np.int64)
    starts = xpm.concatenate(
        (xpm.zeros(1, dtype=np.int64), xpm.nonzero(keys[1:] != keys[:-1])[0] + 1),
    ).astype(np.int64, copy=False)
    stops = xpm.concatenate((starts[1:], xpm.full(1, keys.size, dtype=np.int64)))
    return keys[starts], starts, stops


def cell_offsets(sorted_cells: Any, n_cells: int) -> Any:
    """Return the offsets of each cell into sorted cell IDs.

    Cell ``k`` occupies ``[offsets[k], offsets[k + 1])``; empty cells have equal
    offsets. The result stays on the backend of `sorted_cells`; validation may
    synchronize on CUDA. Filter out negative (invalid) cell IDs first.

    Parameters
    ----------
    sorted_cells : array of int
        Cell IDs in ``[0, n_cells)``, sorted in nondecreasing order.
    n_cells : int
        Number of cells.

    Returns
    -------
    array of int64
        ``n_cells + 1`` offsets.

    Raises
    ------
    ValueError
        If `sorted_cells` is not sorted or has an ID outside ``[0, n_cells)``.

    See Also
    --------
    segment_boundaries : The runs of sorted keys that are present.

    Examples
    --------
    >>> xp.algorithms.cell_offsets(xp.asarray([0, 0, 2]), 3)
    array([0, 2, 2, 3])
    """
    n_cells = _segment_count(n_cells)
    xpm, cells = _sorted_keys(sorted_cells)
    if bool(((cells < 0) | (cells >= n_cells)).any()):
        raise ValueError("cell IDs must be in [0, n_cells)")
    return xpm.searchsorted(
        cells,
        xpm.arange(n_cells + 1, dtype=np.int64),
        side="left",
    ).astype(np.int64, copy=False)


_SUM_KERNELS: dict[str, Any] = {}


def _sum_kernel(dtype: Any) -> Any:
    name = np.dtype(dtype).name
    if name not in _SUM_KERNELS:
        from cunumpy._cuda_kernel import CudaKernel

        ctype = "float" if name == "float32" else "double"
        _SUM_KERNELS[name] = CudaKernel(
            "#include <cunumpy/atomic.cuh>\n"
            'extern "C" __global__ void segment_sum_components('
            f"const long long* keys, const {ctype}* values, {ctype}* out, "
            "long long n, long long width) {\n"
            "long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;\n"
            "if (i < n * width) { long long key = keys[i / width];\n"
            "if (key >= 0) cunumpy_atomic_add(out + key * width + i % width, values[i]); }\n"
            "}",
            "segment_sum_components",
        )
    return _SUM_KERNELS[name]


class SegmentPlan:
    """Validated segment keys, reused by repeated :meth:`sum` calls.

    The keys are copied, so later changes to the caller's array do not affect the
    plan. Setup may synchronize on CUDA; repeated sums do not validate the keys
    again or read GPU values back to the host. A GPU plan is bound to its CUDA
    device, which must be current when it is created and used.

    Parameters
    ----------
    keys : array of int, shape (n,)
        Segment of every row; a negative key drops the row.
    n_segments : int
        Number of segments; every key must be smaller.

    Raises
    ------
    ValueError
        If `keys` is not 1D, a key is ``>= n_segments``, or the keys' CUDA device
        is not current.
    TypeError
        If `keys` is not an integer array.

    See Also
    --------
    segment_sum : The same for a single use of the keys.

    Examples
    --------
    >>> plan = xp.algorithms.SegmentPlan(xp.asarray([0, 1, 0, -1]), 2)
    >>> plan.sum(xp.asarray([1.0, 2.0, 3.0, 4.0]))
    array([4., 2.])
    """

    def __init__(self, keys: Any, n_segments: int) -> None:
        self._n_segments = _segment_count(n_segments)
        self._xp, keys = _integer_keys(keys)
        if bool((keys >= self.n_segments).any()):
            raise ValueError(f"keys must be smaller than n_segments={n_segments}")
        self._backend = get_array_backend(keys)
        self._device = keys.device.id if self._backend == "cupy" else None
        if self._device is not None:
            import cupy as cp

            if cp.cuda.Device().id != self._device:
                raise ValueError("segment keys require their CUDA device current")
        self._keys = self._xp.array(keys, dtype=np.int64, copy=True)
        self._valid = None if self._device is not None else self._keys >= 0

    @property
    def n_segments(self) -> int:
        """Number of output segments, fixed when the plan is created."""
        return self._n_segments

    def sum(self, values: Any, *, out: Any = None) -> Any:
        """Sum the rows of `values` per segment.

        Floating-point and complex dtypes are kept; integer and bool values give
        float64. CUDA supports float16/32/64 and complex64/128 (float16 accumulates
        in float32) and sums all components in one launch with atomics, so the
        floating-point order is not deterministic.

        Parameters
        ----------
        values : array, shape (n, ...)
            One row per key, on the backend of the keys.
        out : array, optional
            Output to overwrite: exact shape and dtype, writable, C-contiguous, and
            not aliasing `values`.

        Returns
        -------
        array, shape (n_segments, ...)
            The sums, `out` if given.

        Raises
        ------
        ValueError
            If the shapes do not match, `out` is unsuitable, or the CUDA device is
            not the plan's.
        TypeError
            If `values` is not numeric, or its dtype is not supported on CUDA.
        """
        assert_same_backend(self._keys, values)
        xpm = self._xp
        values = xpm.asarray(values)
        if values.ndim < 1 or values.shape[0] != self._keys.size:
            raise ValueError("keys must have one entry per value row")
        if values.dtype.kind not in "biufc":
            raise TypeError("values must have a numeric dtype")
        if self._device is not None:
            import cupy as cp

            if values.device.id != self._device or cp.cuda.Device().id != self._device:
                raise ValueError(
                    "segment plan and values require their CUDA device current",
                )
        dtype = values.dtype if values.dtype.kind in "fc" else np.dtype(np.float64)
        if self._device is not None and np.dtype(dtype).name not in (
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ):
            raise TypeError("CUDA segment sums require float16/32/64 or complex64/128")
        shape = (self.n_segments, *values.shape[1:])
        if out is None:
            out = xpm.empty(shape, dtype=dtype)
        else:
            assert_same_backend(self._keys, out)
            if out.shape != shape or out.dtype != dtype or not out.flags.c_contiguous:
                raise ValueError(
                    "out must have the exact shape/dtype and be C-contiguous",
                )
            if not getattr(out.flags, "writeable", True):
                raise ValueError("out must be writable")
            if self._device is not None and out.device.id != self._device:
                raise ValueError("out must be on the segment plan's CUDA device")
            if xpm.may_share_memory(out, values):
                raise ValueError("out must not alias values")
        out.fill(0)
        if self._device is None:
            np.add.at(
                out,
                self._keys[self._valid],
                values[self._valid].astype(dtype, copy=False),
            )
        elif values.size and self.n_segments:
            half = np.dtype(dtype) == np.dtype(np.float16)
            packed = xpm.ascontiguousarray(values, dtype=np.float32 if half else dtype)
            if np.dtype(dtype).kind == "c":
                real_dtype = np.float32 if np.dtype(dtype).itemsize == 8 else np.float64
                packed, target = packed.view(real_dtype), out.view(real_dtype)
            else:
                target = xpm.zeros(shape, dtype=np.float32) if half else out
            width = math.prod(values.shape[1:]) * (
                2 if np.dtype(dtype).kind == "c" else 1
            )
            _sum_kernel(packed.dtype)(
                self._keys,
                packed,
                target,
                self._keys.size,
                width,
                n_threads=self._keys.size * width,
            )
            if half:
                out[...] = target
        return out


def segment_sum(values: Any, keys: Any, n_segments: int, *, out: Any = None) -> Any:
    """Sum the rows of `values` per integer key.

    ``out[k]`` is the sum of the rows ``i`` with ``keys[i] == k``; negative keys
    drop their row. The result is on the backend of `keys`; mixed backends raise.
    Use :class:`SegmentPlan` when the keys are reused, to avoid repeating the
    validation and its GPU synchronization.

    Parameters
    ----------
    values : array, shape (n, ...)
        Values with any trailing dimensions.
    keys : array of int, shape (n,)
        Segment of every row.
    n_segments : int
        Number of segments; every key must be smaller.
    out : array, optional
        Output to overwrite, see :meth:`SegmentPlan.sum`.

    Returns
    -------
    array, shape (n_segments, ...)
        The sums, float64 for integer or bool values.

    Examples
    --------
    >>> keys = xp.asarray([0, 1, 0, -1])
    >>> xp.algorithms.segment_sum(xp.asarray([1.0, 2.0, 3.0, 4.0]), keys, 2)
    array([4., 2.])
    """
    return SegmentPlan(keys, n_segments).sum(values, out=out)


#: Integer keys of more entries than this are sorted by :func:`_radix_argsort` on NumPy.
_RADIX_MIN_SIZE = 4096


def _radix_argsort(keys: np.ndarray) -> np.ndarray:
    """Argsort integer `keys` stably, by an LSD radix sort on 16-bit digits."""
    low = keys.min()
    span = int(keys.max()) - int(low)
    if keys.dtype.kind == "i":
        # the span of int64 keys fits uint64; shift them to start at zero
        shifted = keys.astype(np.int64, copy=False).view(np.uint64) - np.uint64(
            np.int64(low).view(np.uint64)
        )
    else:
        shifted = keys.astype(np.uint64, copy=False) - np.uint64(low)
    order = None
    shift = 0
    while True:
        digits = shifted if order is None else shifted[order]
        digits = (digits >> np.uint64(shift)).astype(np.uint16)
        step = np.argsort(digits, kind="stable")
        order = step if order is None else order[step]
        shift += 16
        if span >> shift == 0:
            break
    return order.astype(np.int64, copy=False)


def sort_by_key(keys: Any, *arrays: Any, axis: int = 0) -> tuple[Any, ...]:
    """Sort `keys` and reorder every array the same way, in one stable argsort.

    The usual first step of a particle code: sort the particles by cell index or
    Morton key (:func:`morton_keys`), then work on contiguous ranges. Equal keys
    keep their order, so the result is reproducible. On CuPy the argsort is a
    radix sort; on NumPy, integer keys of more than 4096 entries get an LSD
    radix sort too (one stable ``uint16`` argsort per 16-bit digit the key range
    needs), about ten times faster than NumPy's stable sort of 64-bit integers.

    Parameters
    ----------
    keys : array, shape (n,)
        The sort keys.
    *arrays : array
        Arrays with ``n`` entries along `axis`, on the backend of `keys`.
    axis : int, optional
        The axis of every array that `keys` indexes, 0 by default; ``-1`` is the
        last axis of each array, for component-major ``(ncomp, n)`` markers next
        to ``(n,)`` arrays.

    Returns
    -------
    tuple of arrays
        ``(keys[order], order, *sorted_arrays)``, with the int64 permutation
        ``order`` and each array taken along `axis` (new arrays).

    Raises
    ------
    ValueError
        If `keys` is not 1D or an array does not have ``n`` entries along `axis`.

    Examples
    --------
    >>> keys = xp.asarray([2, 0, 1, 0])
    >>> xp.algorithms.sort_by_key(keys, xp.asarray([10.0, 11.0, 12.0, 13.0]))
    (array([0, 0, 1, 2]), array([1, 3, 2, 0]), array([11., 13., 12., 10.]))
    """
    if get_array_backend(keys) == "cupy":
        import cupy as xpm  # its argsort is a stable radix sort
    else:
        xpm = np
    keys = xpm.asarray(keys)
    if keys.ndim != 1:
        raise ValueError(f"keys must be 1D, got shape {keys.shape}")
    axes = []
    for i, array in enumerate(arrays):
        if not -array.ndim <= axis < array.ndim:
            raise ValueError(
                f"array {i} has {array.ndim} axes, so it has no axis {axis}",
            )
        array_axis = axis % array.ndim
        if array.shape[array_axis] != keys.shape[0]:
            raise ValueError(
                f"array {i} has shape {array.shape}, expected {keys.shape[0]} "
                f"entries along axis {axis}",
            )
        axes.append(array_axis)
    if xpm is np and keys.dtype.kind in "iu" and keys.size > _RADIX_MIN_SIZE:
        order = _radix_argsort(keys)
    else:
        order = xpm.argsort(keys, kind="stable").astype(xpm.int64, copy=False)
    return (
        keys[order],
        order,
        *(
            xpm.take(array, order, axis=array_axis)
            for array, array_axis in zip(arrays, axes, strict=True)
        ),
    )


def compact_by_mask(mask: Any, *arrays: Any, axis: int = 0) -> int:
    """Move the entries where `mask` is True to the front of every array, in place.

    The usual step after particles left the domain: keep the live ones at the
    front, in their original order, and continue with ``markers[:n]``. The
    entries after the first ``n`` are unspecified. The count is needed on the
    host, so on CuPy each call synchronizes once (seen by
    :func:`~cunumpy.profiling.count_transfers`).

    Parameters
    ----------
    mask : array of bool, shape (n,)
        True for the entries to keep.
    *arrays : array
        Arrays with ``n`` entries along `axis`, on the backend of `mask`;
        modified in place.
    axis : int, optional
        The axis of every array that `mask` indexes, 0 by default; ``-1`` is the
        last axis of each array, for component-major ``(ncomp, n)`` markers next
        to ``(n,)`` arrays.

    Returns
    -------
    int
        The number of kept entries.

    Raises
    ------
    TypeError
        If `mask` is not a 1D boolean array.
    ValueError
        If an array does not have ``n`` entries along `axis`.

    Examples
    --------
    >>> x = xp.asarray([1.0, 2.0, 3.0, 4.0])
    >>> n = xp.algorithms.compact_by_mask(xp.asarray([True, False, True, False]), x)
    >>> x[:n]
    array([1., 3.])
    """
    assert_same_backend(mask, *arrays)
    xpm = get_array_module(mask)
    mask = xpm.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise TypeError(
            f"mask must be a 1D boolean array, got dtype {mask.dtype}, {mask.ndim}D",
        )
    axes = []
    for i, array in enumerate(arrays):
        if not -array.ndim <= axis < array.ndim:
            raise ValueError(
                f"array {i} has {array.ndim} axes, so it has no axis {axis}",
            )
        array_axis = axis % array.ndim
        if array.shape[array_axis] != mask.shape[0]:
            raise ValueError(
                f"array {i} has shape {array.shape}, expected {mask.shape[0]} "
                f"entries along axis {axis}",
            )
        axes.append(array_axis)
    rows = xpm.nonzero(mask)[0]
    n_kept = int(rows.size)
    for array, array_axis in zip(arrays, axes, strict=True):
        front = (slice(None),) * array_axis + (slice(0, n_kept),)
        # the right side is a copy: no overlap problem
        array[front] = xpm.take(array, rows, axis=array_axis)
    return n_kept
