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
    """Return ``(unique_keys, starts, stops)`` for runs of sorted integer keys.

    Starts/stops are int64 indices and describe half-open slices of the input.
    Sparse, negative, and uint64 Morton keys are supported. All results stay on
    the input backend. Validation and variable-length GPU output may synchronize;
    prepare once and reuse the boundaries in repeated operations.
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
    """Dense int64 offsets into sorted cell IDs in ``[0, n_cells)``.

    Cell k occupies ``[offsets[k], offsets[k+1])``; empty cells have equal
    offsets. Returns n_cells+1 entries on the input backend. Filter negative
    (invalid) cell IDs before calling. Validation may synchronize on CUDA.
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
    """Snapshot and validate segment keys once, then reuse :meth:`sum`.

    A negative key drops its row; other keys must be below `n_segments`.
    Keys are copied so later caller mutation cannot invalidate the plan. A GPU
    plan is bound to its CUDA device. Setup may synchronize, but repeated sums
    do not read GPU reductions back into Python or validate keys per column.
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
        """Number of output segments, fixed when the plan is prepared."""
        return self._n_segments

    def sum(self, values: Any, *, out: Any = None) -> Any:
        """Sum rows into ``(n_segments, *values.shape[1:])``; overwrite `out`.

        Floating/complex dtypes are preserved; integer/bool inputs produce
        float64. CUDA supports float16/32/64 and complex64/128 outputs; float16
        accumulates in a float32 workspace. Other CUDA accumulation
        uses atomics and its floating-point order is not deterministic. `out`
        must have the exact shape/dtype, be C-contiguous, and not alias values.
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
    """Sum rows per integer key, dropping negatives; optionally overwrite `out`.

    Accepts arbitrary trailing value dimensions. Keys are validated once per
    call and CUDA reduces all components in one launch, without per-column host
    checks. Use ``SegmentPlan(keys, n_segments).sum(values, out=out)`` when the
    keys are reused, to avoid repeating setup and its GPU synchronization.
    """
    return SegmentPlan(keys, n_segments).sum(values, out=out)


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
                f"every array needs {keys.shape[0]} rows, got shape {array.shape}",
            )
    order = xpm.argsort(keys, kind="stable").astype(xpm.int64, copy=False)
    return (keys[order], order, *(array[order] for array in arrays))


def compact_by_mask(mask: Any, *arrays: Any, axis: int = 0) -> int:
    """Move the entries where `mask` is True to the front of every array, in place.

    The usual step after particles left the domain or were absorbed: keep the
    live ones at the front of the marker array (and of the arrays that go with
    it) and continue with ``markers[:n]``. The order of the kept entries is
    preserved, so the result is reproducible::

        n = xp.algorithms.compact_by_mask(alive, markers, weights)
        markers, weights = markers[:n], weights[:n]

    Component-major marker arrays, ``(ncomp, N)`` next to ``(N,)`` scalars, keep
    the markers along the last axis of each, so compact that one::

        n = xp.algorithms.compact_by_mask(alive, positions, weights, axis=-1)
        positions, weights = positions[:, :n], weights[:n]

    Parameters
    ----------
    mask : array of bool, shape (n,)
        True for the entries to keep.
    *arrays : arrays
        Arrays with ``n`` entries along `axis` (any other axes), on the backend
        of `mask`. Entries ``[:count]`` along `axis` hold the kept ones
        afterwards; the entries after them are unspecified, so ignore them (or
        overwrite them).
    axis : int
        The axis of every array that `mask` indexes, the first by default;
        ``-1`` is the last axis of each array, whatever its number of axes.

    Returns
    -------
    int
        The number of kept entries. Its value is needed on the host, so on CuPy
        the call synchronizes once per call (counted by
        :func:`~cunumpy.profiling.count_transfers` where it can be seen).
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
