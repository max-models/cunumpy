"""Grouping and reusable reductions on CPU and (when available) CUDA."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.algorithms import (
    SegmentPlan,
    cell_offsets,
    segment_boundaries,
    segment_sum,
)
from cunumpy.kernel_testing import BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_dense_and_sparse_boundaries(backend):
    with xp.use_backend(backend, strict=True):
        cells = xp.asarray([0, 0, 2, 4, 4], dtype=xp.int64)
        np.testing.assert_array_equal(
            xp.to_numpy(cell_offsets(cells, 6)),
            [0, 2, 2, 3, 3, 5, 5],
        )
        unique, starts, stops = segment_boundaries(cells)
        for got, expected in zip(
            (unique, starts, stops),
            ([0, 2, 4], [0, 2, 3], [2, 3, 5]),
        ):
            assert xp.get_array_backend(got) == backend
            np.testing.assert_array_equal(xp.to_numpy(got), expected)
        large = xp.asarray([2**63 + 1, 2**63 + 1, 2**64 - 1], dtype=xp.uint64)
        unique, starts, stops = segment_boundaries(large)
        np.testing.assert_array_equal(
            xp.to_numpy(unique),
            np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64),
        )
        np.testing.assert_array_equal(xp.to_numpy(starts), [0, 2])
        np.testing.assert_array_equal(xp.to_numpy(stops), [2, 3])


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_invalid_grouping(backend):
    with xp.use_backend(backend, strict=True):
        empty = xp.empty(0, dtype=xp.int64)
        np.testing.assert_array_equal(xp.to_numpy(cell_offsets(empty, 3)), [0, 0, 0, 0])
        assert all(a.size == 0 for a in segment_boundaries(empty))
        np.testing.assert_array_equal(xp.to_numpy(cell_offsets(empty, 0)), [0])
        for bad in ([2, 0], [-1, 0], [0, 3]):
            with pytest.raises(ValueError):
                cell_offsets(xp.asarray(bad), 3)
        with pytest.raises(TypeError, match="integer"):
            segment_boundaries(xp.asarray([1.5]))
        with pytest.raises(ValueError, match="1D"):
            segment_boundaries(xp.zeros((2, 2), dtype=xp.int64))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int64,
        np.bool_,
    ],
)
def test_prepared_reductions_and_reusable_output(backend, dtype):
    keys_host = np.array([0, 2, 0, -1, 2, 0])
    values_host = np.arange(36).reshape(6, 2, 3).astype(dtype)
    if np.dtype(dtype).kind == "c":
        values_host += 1j * values_host
    output_dtype = dtype if np.dtype(dtype).kind in "fc" else np.float64
    expected = np.zeros((4, 2, 3), dtype=output_dtype)
    np.add.at(expected, keys_host[keys_host >= 0], values_host[keys_host >= 0])
    with xp.use_backend(backend, strict=True):
        keys = xp.array(keys_host, copy=True)
        plan = SegmentPlan(keys, 4)
        keys[:] = 99  # the plan owns a validated snapshot
        values = xp.asarray(values_host)
        out = xp.full(expected.shape, -1, dtype=output_dtype)
        assert plan.sum(values, out=out) is out
        np.testing.assert_allclose(xp.to_numpy(out), expected, rtol=1e-5)
        plan.sum(values, out=out)  # overwrite, rather than accumulate prior sums
        np.testing.assert_allclose(xp.to_numpy(out), expected, rtol=1e-5)
        # Non-contiguous trailing values are accepted.
        got = segment_sum(values[:, :, ::2], xp.asarray(keys_host), 4)
        np.testing.assert_allclose(xp.to_numpy(got), expected[:, :, ::2], rtol=1e-5)
        with pytest.raises(AttributeError):
            plan.n_segments = 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_dropped_rows_and_output_validation(backend):
    with xp.use_backend(backend, strict=True):
        got = segment_sum(xp.empty((0, 2)), xp.empty(0, dtype=xp.int64), 3)
        np.testing.assert_array_equal(xp.to_numpy(got), np.zeros((3, 2)))
        got = segment_sum(xp.asarray([np.nan, np.inf]), xp.asarray([-1, -2]), 0)
        assert got.shape == (0,)
        plan = SegmentPlan(xp.asarray([0, 1]), 2)
        values = xp.ones(2)
        for out in (
            values,
            xp.zeros(3),
            xp.zeros(2, dtype=xp.float32),
            xp.zeros(4)[::2],
        ):
            with pytest.raises(ValueError):
                plan.sum(values, out=out)
        with pytest.raises(ValueError, match="one entry"):
            plan.sum(xp.ones(3))
        with pytest.raises(ValueError, match="non-negative"):
            SegmentPlan(xp.asarray([0]), -1)
        with pytest.raises(ValueError, match="smaller"):
            SegmentPlan(xp.asarray([2]), 2)


def test_cpu_readonly_out_is_rejected():
    out = np.zeros(2)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        SegmentPlan(np.array([0, 1]), 2).sum(np.ones(2), out=out)


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
def test_gpu_plan_avoids_scalar_sync_and_rejects_host_values():
    import cupy as cp

    plan = SegmentPlan(cp.array([0, -1, 1]), 2)
    values = cp.arange(12.0).reshape(3, 4)
    with pytest.raises(TypeError, match="mismatched"):
        plan.sum(np.ones((3, 4)))
    # A scalar-read-free sum can be captured; setup/compilation is outside capture.
    out = cp.zeros((2, 4))
    plan.sum(values, out=out)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.Device().synchronize()
    with stream:
        stream.begin_capture()
        plan.sum(values, out=out)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(out), [[0, 1, 2, 3], [8, 9, 10, 11]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_sum_kernel_matches_reference_in_cpu_emulation(dtype):
    from cunumpy._algorithms import _sum_kernel
    from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

    if emulation_compiler() is None:
        pytest.skip("requires a C++ compiler")
    keys = np.array([0, 2, -1, 0, 2], dtype=np.int64)
    values = np.arange(30, dtype=dtype).reshape(5, 6)
    out = np.zeros((3, 6), dtype=dtype)
    emulate_cuda_kernel(_sum_kernel(dtype), keys, values, out, 5, 6, n_threads=30)
    expected = np.zeros_like(out)
    np.add.at(expected, keys[keys >= 0], values[keys >= 0])
    np.testing.assert_array_equal(out, expected)
