"""Hardware tests for scans/masked reductions, plus CPU-emulated integer atomics."""

import subprocess

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.cuda import CudaKernel
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

requires_cuda = pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")

TYPED_SOURCE = r"""
#include <cunumpy/scan.cuh>
template <typename T>
__global__ void typed_scan(const T* x, T* inclusive, T* exclusive, T* total) {
    int t = cunumpy_block_thread();
    int i = blockIdx.x * cunumpy_block_threads() + t;
    inclusive[i] = cunumpy_block_inclusive_sum(x[i]);
    exclusive[i] = cunumpy_block_exclusive_sum(x[i]);
    cunumpy_block_sum_to(total, x[i]);
}
"""


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64],
)
def test_scans_and_block_atomic_totals_for_shuffle_types(dtype):
    import cupy as cp

    host = np.arange(70, dtype=dtype)
    data = cp.asarray(host)
    inc, exc, total = cp.zeros_like(data), cp.zeros_like(data), cp.zeros(1, dtype=dtype)
    kernel = CudaKernel(
        TYPED_SOURCE,
        "typed_scan",
        template_args=(dtype,),
        block_size=35,
    )
    kernel(data, inc, exc, total, grid=2)
    expected = np.cumsum(host.reshape(2, 35), axis=1)
    np.testing.assert_array_equal(cp.asnumpy(inc).reshape(2, 35), expected)
    np.testing.assert_array_equal(
        cp.asnumpy(exc).reshape(2, 35),
        np.concatenate((np.zeros((2, 1)), expected[:, :-1]), axis=1),
    )
    np.testing.assert_array_equal(cp.asnumpy(total), [host.sum()])


BLOCK_SOURCE = r"""
#include <cunumpy/scan.cuh>
extern "C" __global__ void collectives(const double* x, double* inclusive,
    double* exclusive, double* again, double* sums, double* mins, double* maxs) {
    int t = cunumpy_block_thread();
    int count = cunumpy_block_threads();
    int i = blockIdx.x * count + t;
    double v = x[i];
    inclusive[i] = cunumpy_block_inclusive_sum(v);
    exclusive[i] = cunumpy_block_exclusive_sum(v);
    again[i] = cunumpy_block_inclusive_sum(v * 2);
    double sum = cunumpy_block_sum(v);
    double lo = cunumpy_block_min(v);
    double hi = cunumpy_block_max(v);
    if (t == 0) { sums[blockIdx.x] = sum; mins[blockIdx.x] = lo; maxs[blockIdx.x] = hi; }
}
"""


@requires_cuda
@pytest.mark.parametrize(
    "block",
    [1, 17, 31, 32, 33, 64, 127, 128, 1024, (7, 5), (3, 4, 3)],
)
def test_partial_warps_multidimensional_blocks_and_repeated_collectives(block):
    import cupy as cp

    count = int(np.prod(block))
    host = np.arange(3 * count, dtype=float) % 11 - 5
    data = cp.asarray(host)
    inc, exc, again = (cp.empty_like(data) for _ in range(3))
    sums, lo, hi = (cp.empty(3) for _ in range(3))
    kernel = CudaKernel(
        BLOCK_SOURCE, "collectives", block_size=block, options=("-lineinfo",)
    )
    kernel(data, inc, exc, again, sums, lo, hi, grid=3)
    rows = host.reshape(3, count)
    expected_inc = np.cumsum(rows, axis=1)
    expected_exc = np.concatenate((np.zeros((3, 1)), expected_inc[:, :-1]), axis=1)
    np.testing.assert_array_equal(cp.asnumpy(inc).reshape(3, count), expected_inc)
    np.testing.assert_array_equal(cp.asnumpy(exc).reshape(3, count), expected_exc)
    np.testing.assert_array_equal(cp.asnumpy(again).reshape(3, count), expected_inc * 2)
    for got, expected in ((sums, rows.sum(1)), (lo, rows.min(1)), (hi, rows.max(1))):
        np.testing.assert_array_equal(cp.asnumpy(got), expected)


REPEATED_EXTREMA_SOURCE = r"""
#include <cunumpy/reduce.cuh>
template <typename T>
__global__ void repeated_extrema(const T* x, T* minima, T* maxima, int rounds) {
    int t = cunumpy_block_thread();
    int count = cunumpy_block_threads();
    int i = blockIdx.x * count + t;
    int stride = gridDim.x * count;
    for (int r = 0; r < rounds; ++r) {
        T v = ((r & 1) ? -x[i] : x[i]) + T(r);
        T lo = cunumpy_block_min(v);
        T hi = cunumpy_block_max(v);
        minima[r * stride + i] = lo;
        maxima[r * stride + i] = hi;
    }
}
"""


@requires_cuda
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("block", [17, 32, 33, 128, 1024, (7, 5)])
def test_repeated_extrema_are_returned_to_every_thread(dtype, block):
    import cupy as cp

    count, blocks, rounds = int(np.prod(block)), 3, 8
    host = np.random.default_rng(123).integers(-100, 100, blocks * count).astype(dtype)
    data = cp.asarray(host)
    minima, maxima = (cp.empty((rounds, data.size), dtype=dtype) for _ in range(2))
    kernel = CudaKernel(
        REPEATED_EXTREMA_SOURCE,
        "repeated_extrema",
        block_size=block,
        template_args=(dtype,),
        options=("-lineinfo",),
    )
    kernel(data, minima, maxima, rounds, grid=blocks)
    rows = host.reshape(blocks, count)
    values = np.stack([(rows if r % 2 == 0 else -rows) + r for r in range(rounds)])
    for actual, expected in (
        (minima, values.min(axis=2)),
        (maxima, values.max(axis=2)),
    ):
        np.testing.assert_array_equal(
            cp.asnumpy(actual), np.repeat(expected, count, axis=1)
        )


MASKED_SOURCE = r"""
#include <cunumpy/scan.cuh>
extern "C" __global__ void masked(const double* x, unsigned int mask,
    double* sums, double* mins, double* maxs, double* inclusive, double* exclusive) {
    int lane = threadIdx.x;
    if (mask & (1u << lane)) {
        double v = x[lane];
        sums[lane] = cunumpy_warp_sum(v, mask);
        mins[lane] = cunumpy_warp_min(v, mask);
        maxs[lane] = cunumpy_warp_max(v, mask);
        inclusive[lane] = cunumpy_warp_inclusive_sum(v, mask);
        exclusive[lane] = cunumpy_warp_exclusive_sum(v, mask);
    }
}
"""


@requires_cuda
@pytest.mark.parametrize(
    "mask",
    [1, 1 << 31, 0x80000001, 0x55555555, 0xAAAAAAAA, 0x17, 0x7FFFFFFF, 0xFFFFFFFF],
)
def test_arbitrary_sparse_masks(mask):
    import cupy as cp

    host = np.arange(32, dtype=float) - 10
    active = np.array([i for i in range(32) if mask & (1 << i)])
    out = [cp.full(32, -999.0) for _ in range(5)]
    kernel = CudaKernel(MASKED_SOURCE, "masked", block_size=32)
    kernel(cp.asarray(host), mask, *out, n_threads=32)
    selected = host[active]
    expected = [
        selected.sum(),
        selected.min(),
        selected.max(),
        selected.cumsum(),
        np.concatenate(([0.0], selected.cumsum()[:-1])),
    ]
    inactive = np.setdiff1d(np.arange(32), active)
    for got, want in zip(out, expected):
        result = cp.asnumpy(got)
        np.testing.assert_array_equal(
            result[active],
            np.broadcast_to(want, active.shape),
        )
        np.testing.assert_array_equal(result[inactive], -999)


@requires_cuda
@pytest.mark.parametrize("mask", [0xFFFFFFFF, 0x80000001])
def test_exclusive_scan_preserves_small_previous_values(mask):
    import cupy as cp

    host = np.zeros(32)
    active = [i for i in range(32) if mask & (1 << i)]
    host[active[0]], host[active[1]] = 1.0, 1e20
    out = [cp.zeros(32) for _ in range(5)]
    CudaKernel(MASKED_SOURCE, "masked", block_size=32)(
        cp.asarray(host),
        mask,
        *out,
        n_threads=32,
    )
    assert float(out[-1][active[1]]) == 1.0


ATOMICS_SOURCE = r"""
#include <cunumpy/atomic.cuh>
extern "C" __global__ void integer_counts(int* a, unsigned int* b,
    long long* c, unsigned long long* d, long long* previous, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cunumpy_atomic_add_2d(a, 3, i % 2, i % 3, 1);
        cunumpy_atomic_add_3d(b, 2, 3, 0, i % 2, i % 3, 1u);
        previous[i] = cunumpy_atomic_add(c, -1ll);
        cunumpy_atomic_add(d, 1ull);
    }
}
"""


@pytest.mark.parametrize("backend", ["emulation", "cuda"])
def test_integer_atomic_counts_and_returned_old_values(backend):
    n = 71
    if backend == "emulation":
        if emulation_compiler() is None:
            pytest.skip("requires a C++ compiler")
        arrays = [
            np.zeros((2, 3), np.int32),
            np.zeros((1, 2, 3), np.uint32),
            np.zeros(1, np.int64),
            np.zeros(1, np.uint64),
            np.zeros(n, np.int64),
        ]
        kernel = CudaKernel(ATOMICS_SOURCE, "integer_counts")
        emulate_cuda_kernel(kernel, *arrays, n, n_threads=n)
    else:
        if not xp.cupy_available():
            pytest.skip("requires CUDA")
        import cupy as cp

        arrays = [
            cp.zeros((2, 3), cp.int32),
            cp.zeros((1, 2, 3), cp.uint32),
            cp.zeros(1, cp.int64),
            cp.zeros(1, cp.uint64),
            cp.zeros(n, cp.int64),
        ]
        CudaKernel(ATOMICS_SOURCE, "integer_counts")(*arrays, n, n_threads=n)
        arrays = [cp.asnumpy(a) for a in arrays]
    expected = np.zeros((2, 3), dtype=int)
    for i in range(n):
        expected[i % 2, i % 3] += 1
    np.testing.assert_array_equal(arrays[0], expected)
    np.testing.assert_array_equal(arrays[1][0], expected)
    assert arrays[2][0] == -n and arrays[3][0] == n
    np.testing.assert_array_equal(np.sort(arrays[4]), np.sort(-np.arange(n)))


def test_scan_header_is_resolved_and_emulation_refuses_warp_intrinsics():
    kernel = CudaKernel(BLOCK_SOURCE, "collectives")
    assert [p.name for p in kernel.included_headers] == [
        "scan.cuh",
        "reduce.cuh",
        "atomic.cuh",
    ]
    if emulation_compiler() is None:
        pytest.skip("requires a C++ compiler")
    with pytest.raises(NotImplementedError, match="warp (shuffles|synchronization)"):
        emulate_cuda_kernel(kernel, *(np.zeros(32) for _ in range(7)), n_threads=32)


def test_collective_headers_compile_with_all_supported_arithmetic_types():
    # A C++ syntax/instantiation check only. Hardware tests above establish warp
    # semantics; identity stubs must never be used as a correctness emulator.
    from cunumpy._emulation import _STUBS
    from cunumpy.cuda import cuda_include_dir

    compiler = emulation_compiler()
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    stubs = r"""
inline int __ffs(unsigned v) { return v ? __builtin_ctz(v) + 1 : 0; }
inline int __clz(unsigned v) { return __builtin_clz(v); }
inline void __syncthreads() {}
inline void __syncwarp(unsigned) {}
template<class T> T __shfl_sync(unsigned, T v, int) { return v; }
template<class T> T __shfl_up_sync(unsigned, T v, int) { return v; }
template<class T> T __shfl_xor_sync(unsigned, T v, int) { return v; }
"""
    instantiate = r"""
template<class T> void check_type(T v) {
    cunumpy_block_inclusive_sum(v);
    cunumpy_block_exclusive_sum(v);
    cunumpy_warp_exclusive_sum(v, 0x55555555u);
    cunumpy_block_sum(v); cunumpy_block_min(v); cunumpy_block_max(v);
    T out = T(0); cunumpy_block_sum_to(&out, v);
}
void check_all() {
    check_type(1); check_type(1u); check_type(1ll); check_type(1ull);
    check_type(1.0f); check_type(1.0);
}
"""
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I" + cuda_include_dir(),
            "-",
        ],
        input=_STUBS
        + stubs
        + BLOCK_SOURCE
        + REPEATED_EXTREMA_SOURCE
        + MASKED_SOURCE
        + ATOMICS_SOURCE
        + instantiate,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
