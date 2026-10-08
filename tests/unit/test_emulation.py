"""Tests for `cunumpy.kernel_testing.emulate_cuda_kernel`: CUDA kernels run on the CPU."""

import weakref

import numpy as np
import pytest

from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from cunumpy.kernels import CudaKernel

pytestmark = pytest.mark.skipif(
    emulation_compiler() is None,
    reason="no C++ compiler for the emulation",
)

AXPY = r"""
extern "C" __global__
void axpy(double a, const double* __restrict__ x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""


def test_axpy():
    rng = np.random.default_rng(0)
    x, y = rng.random(1000), rng.random(1000)
    expected = y + 2.5 * x
    emulate_cuda_kernel(CudaKernel(AXPY, "axpy"), 2.5, x, y, 1000)
    # the compiler may fuse y + a * x into one FMA, as NVRTC does by default
    np.testing.assert_allclose(y, expected, rtol=1e-15, atol=0)
    exact = rng.random(1000)
    exact_expected = exact + 2.5 * x
    emulate_cuda_kernel(
        CudaKernel(AXPY, "axpy"),
        2.5,
        x,
        exact,
        1000,
        n_threads=1000,
        options=("-ffp-contract=off",),
    )
    np.testing.assert_array_equal(exact, exact_expected)  # no FMA: NumPy's rounding


COLUMN = r"""
#include "cunumpy/array_view.cuh"
#include <cunumpy/index.cuh>
extern "C" __global__
void scale_column(Array2D<double> a, long long column, double factor) {
    CUNUMPY_GRID_STRIDE_1D(i, a.shape[0]) {
        a(i, column) *= factor;
    }
}
"""


def test_strided_view_written_back_into_the_callers_array():
    markers = np.arange(24.0).reshape(8, 3)
    every_second_row = markers[::2]  # a non-contiguous view
    emulate_cuda_kernel(
        CudaKernel(COLUMN, "scale_column"),
        every_second_row,
        1,
        10.0,
        grid=1,
        block=2,
    )
    expected = np.arange(24.0).reshape(8, 3)
    expected[::2, 1] *= 10.0
    np.testing.assert_array_equal(markers, expected)


CONTIGUOUS = r"""
#include "cunumpy/array_view.cuh"
#include <cunumpy/index.cuh>
__device__ double sum_row(Array2D<double> a, long long i) {
    double total = 0.0;
    for (long long j = 0; j < a.shape[1]; ++j) total += a(i, j);
    return total;
}
extern "C" __global__
void row_sums(CArray2D<double> a, CArray3D<double> b, CArray1D<double> out) {
    CUNUMPY_GRID_STRIDE_1D(i, a.shape[0]) {
        out(i) = sum_row(a, i) + b(i, 1, 2);  // CArray2D converts to Array2D
    }
}
"""


def test_contiguous_views():
    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(4 * 2 * 3.0).reshape(4, 2, 3)
    out = np.zeros(4)
    kernel = CudaKernel(CONTIGUOUS, "row_sums")
    emulate_cuda_kernel(kernel, a, b, out, grid=1, block=2)
    np.testing.assert_array_equal(out, a.sum(axis=1) + b[:, 1, 2])

    with pytest.raises(TypeError, match="must be C-contiguous"):
        emulate_cuda_kernel(
            kernel, a[:, :2].copy()[::2], b[::2], out[:2], grid=1, block=2
        )


TEMPLATE = r"""
template <typename T, int K>
__global__ void power(T* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { T v = x[i]; for (int k = 1; k < K; ++k) x[i] *= v; }
}
"""


def test_template_kernel():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    kernel = CudaKernel(TEMPLATE, "power", template_args=(np.float32, 3))
    emulate_cuda_kernel(kernel, x, 3, n_threads=3)
    assert x.tolist() == [1.0, 8.0, 27.0]


GRID_2D = r"""
#include "cunumpy/array_view.cuh"
extern "C" __global__ void fill(Array2D<long long> a) {
    long long i = blockIdx.y * (long long)blockDim.y + threadIdx.y;
    long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < a.shape[0] && j < a.shape[1]) a(i, j) = 10 * i + j;
}
"""


def test_2d_launch():
    a = np.zeros((5, 7), dtype=np.int64)
    kernel = CudaKernel(GRID_2D, "fill", block_size=(4, 2))
    emulate_cuda_kernel(kernel, a, n_threads=(7, 5))
    np.testing.assert_array_equal(a, 10 * np.arange(5)[:, None] + np.arange(7))


def test_inferred_2d_launch_updates_every_element():
    source = r"""
    #include <cunumpy/array_view.cuh>
    extern "C" __global__ void fill(Array2D<long long> a) {
        long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
        long long j = blockIdx.y * (long long)blockDim.y + threadIdx.y;
        if (i < a.shape[0] && j < a.shape[1]) a(i, j) = 10 * i + j;
    }
    """
    array = np.zeros((5, 7), dtype=np.int64)
    emulate_cuda_kernel(CudaKernel(source, "fill", block_size=(2, 4)), array)
    np.testing.assert_array_equal(array, 10 * np.arange(5)[:, None] + np.arange(7))


HISTOGRAM = r"""
#include "cunumpy/atomic.cuh"
extern "C" __global__
void histogram(const double* x, int n, double* counts, double lower, double width, int bins) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int b = (int)floor((x[i] - lower) / width);
    if (b >= 0 && b < bins) cunumpy_atomic_add(&counts[b], 1.0);
}
"""


def test_atomics_and_math_functions():
    x = np.random.default_rng(1).normal(size=5000)
    counts = np.zeros(10)
    emulate_cuda_kernel(
        CudaKernel(HISTOGRAM, "histogram"),
        x,
        x.size,
        counts,
        -2.5,
        0.5,
        10,
        n_threads=x.size,
    )
    np.testing.assert_array_equal(counts, np.histogram(x, 10, (-2.5, 2.5))[0])


def test_scalars_are_checked_like_a_launch():
    kernel = CudaKernel(AXPY, "axpy")
    x, y = np.ones(3), np.zeros(3)
    emulate_cuda_kernel(kernel, np.int64(2), x, y, np.int64(3), n_threads=3)
    assert y.tolist() == [2.0] * 3
    with pytest.raises(TypeError):
        emulate_cuda_kernel(kernel, 1.0, x, y, 2.5, n_threads=3)
    with pytest.raises(OverflowError):
        emulate_cuda_kernel(kernel, 1.0, x, y, 2**40, n_threads=3)


def test_arrays_are_checked():
    kernel = CudaKernel(AXPY, "axpy")
    with pytest.raises(TypeError, match="dtype float64"):
        emulate_cuda_kernel(
            kernel,
            1.0,
            np.ones(3, np.float32),
            np.ones(3),
            3,
            n_threads=3,
        )
    with pytest.raises(TypeError, match="NumPy array"):
        emulate_cuda_kernel(kernel, 1.0, [1.0], np.ones(3), 3, n_threads=3)
    with pytest.raises(TypeError, match="takes 4 arguments"):
        emulate_cuda_kernel(kernel, 1.0, np.ones(3), n_threads=3)
    view = CudaKernel(COLUMN, "scale_column")
    with pytest.raises(TypeError, match="2D array"):
        emulate_cuda_kernel(view, np.ones(3), 0, 1.0, n_threads=3)


@pytest.mark.parametrize(
    ("body", "what"),
    [
        ("__syncwarp();", "__syncwarp"),
        ("double v = __shfl_xor_sync(0xffffffff, 1.0, 1);", "warp shuffles"),
        ("int v = __ballot_sync(0xffffffff, n > 0);", "warp votes"),
    ],
)
def test_unsupported_constructs_are_refused(body, what):
    kernel = CudaKernel(f'extern "C" __global__ void k(int n) {{ {body} }}', "k")
    with pytest.raises(NotImplementedError, match=what):
        emulate_cuda_kernel(kernel, 1, n_threads=1)


def test_warp_shuffles_in_an_included_header_are_refused():
    kernel = CudaKernel(
        '#include "cunumpy/reduce.cuh"\n'
        'extern "C" __global__ void k(double* out, int n) {'
        "  double s = cunumpy_block_sum(1.0); if (threadIdx.x == 0) out[0] = s; }",
        "k",
    )
    with pytest.raises(NotImplementedError, match="warp (shuffles|synchronization)"):
        emulate_cuda_kernel(kernel, np.zeros(1), 1, n_threads=32)


TREE_SUM = r"""
extern "C" __global__ void block_sums(const double* x, long long n, double* sums) {
    __shared__ double partial[256];
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    partial[threadIdx.x] = i < n ? x[i] : 0.0;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = partial[0];
}
"""


def test_shared_memory_tree_reduction_with_barriers():
    x = np.random.default_rng(2).random(1000)
    sums = np.zeros(4)
    kernel = CudaKernel(TREE_SUM, "block_sums", block_size=256)
    emulate_cuda_kernel(kernel, x, x.size, sums, n_threads=x.size)
    expected = [x[b * 256 : (b + 1) * 256].sum() for b in range(4)]
    np.testing.assert_allclose(sums, expected, rtol=1e-12)


BLOCK_DEPOSIT = r"""
#include "cunumpy/atomic.cuh"
extern "C" __global__
void deposit(const double* positions, long long n, double* field, int nx) {
    extern __shared__ double block_field[];
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    for (int k = threadIdx.x; k < nx; k += blockDim.x) block_field[k] = 0.0;
    __syncthreads();
    if (i < n) {
        int cell = (int)floor(positions[i] * nx);
        if (cell < 0) cell = 0;
        if (cell > nx - 1) cell = nx - 1;
        cunumpy_atomic_add(&block_field[cell], 1.0);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < nx; k += blockDim.x)
        cunumpy_atomic_add(&field[k], block_field[k]);
}
"""


def test_dynamic_shared_memory_per_block_deposit():
    positions = np.random.default_rng(3).random(5000)
    field = np.zeros(37)
    kernel = CudaKernel(BLOCK_DEPOSIT, "deposit", block_size=128)
    emulate_cuda_kernel(
        kernel,
        positions,
        positions.size,
        field,
        37,
        n_threads=positions.size,
        shared_mem=37 * 8,
    )
    np.testing.assert_array_equal(field, np.histogram(positions, 37, (0, 1))[0])


REVERSE = r"""
extern "C" __global__ void reverse_blocks(double* x) {
    __shared__ double tile[64];
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    tile[threadIdx.x] = x[i];
    __syncthreads();  // every thread must have written before anyone reads
    x[i] = tile[blockDim.x - 1 - threadIdx.x];
}
"""


def test_barrier_orders_writes_before_reads():
    x = np.arange(128.0)
    emulate_cuda_kernel(
        CudaKernel(REVERSE, "reverse_blocks", block_size=64),
        x,
        n_threads=128,
    )
    expected = np.concatenate([np.arange(64.0)[::-1], np.arange(64.0, 128.0)[::-1]])
    np.testing.assert_array_equal(x, expected)


EARLY_EXIT = r"""
extern "C" __global__ void early(double* x, int n) {
    __shared__ double s[32];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // leaves before the barrier
    s[threadIdx.x] = 2.0 * x[i];
    __syncthreads();
    x[i] = s[threadIdx.x];
}
"""


def test_threads_leaving_before_a_barrier_do_not_hang():
    x = np.ones(40)
    emulate_cuda_kernel(
        CudaKernel(EARLY_EXIT, "early", block_size=32),
        x,
        40,
        n_threads=40,
    )
    assert x.tolist() == [2.0] * 40


TRANSPOSE_TILE = r"""
#include "cunumpy/array_view.cuh"
extern "C" __global__ void transpose(Array2D<double> a, Array2D<double> out) {
    __shared__ double tile[4][4];
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = a(i, j);
    __syncthreads();
    const int ti = blockIdx.x * blockDim.x + threadIdx.y;
    const int tj = blockIdx.y * blockDim.y + threadIdx.x;
    out(ti, tj) = tile[threadIdx.x][threadIdx.y];
}
"""


def test_2d_blocks_with_shared_tiles():
    a = np.arange(64.0).reshape(8, 8)
    out = np.zeros((8, 8))
    kernel = CudaKernel(TRANSPOSE_TILE, "transpose", block_size=(4, 4))
    emulate_cuda_kernel(kernel, a, out, n_threads=(8, 8))
    np.testing.assert_array_equal(out, a.T)


def test_compile_errors_and_crashes_are_reported():
    broken = CudaKernel(
        'extern "C" __global__ void k(int n) { undefined_call(n); }',
        "k",
    )
    with pytest.raises(RuntimeError, match="does not compile"):
        emulate_cuda_kernel(broken, 1, n_threads=1)
    trap = CudaKernel(
        'extern "C" __global__ void k(int n) { if (n > 0) __trap(); }',
        "k",
    )
    with pytest.raises(RuntimeError, match="crashed"):
        emulate_cuda_kernel(trap, 1, n_threads=1)


def test_bounds_checks_from_the_view_header():
    kernel = CudaKernel(COLUMN, "scale_column")
    a = np.ones((4, 2))
    with pytest.raises(RuntimeError, match="crashed"):
        emulate_cuda_kernel(
            kernel,
            a,
            5,
            2.0,
            n_threads=4,
            options=("-DCUNUMPY_BOUNDS_CHECK",),
        )


@pytest.mark.parametrize("ndim", range(5, 17))
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("backend", ["emulation", "cuda"])
def test_high_dimensional_view_indexing_and_conversion(ndim, contiguous, backend):
    # Exercise every axis with unequal extents, including singleton dimensions.
    # Reading via a strided device helper also tests CArray -> Array conversion.
    ctype = f"{'C' if contiguous else ''}Array{ndim}D<double>"
    indices = ", ".join(f"i[{axis}]" for axis in range(ndim))
    source = f"""
    #include <cunumpy/array_view.cuh>
    #include <cunumpy/index.cuh>
    __device__ double read(Array{ndim}D<double> a, const long long* i) {{
        return a({indices});
    }}
    extern "C" __global__ void update({ctype} a) {{
        CUNUMPY_THREAD_1D(flat, a.size());
        long long i[{ndim}], remaining = flat;
        for (int axis = {ndim} - 1; axis >= 0; --axis) {{
            i[axis] = remaining % a.shape[axis];
            remaining /= a.shape[axis];
        }}
        a({indices}) = read(a, i) + flat + 1;
    }}
    """
    shape = (2, 3) + (1,) * (ndim - 3) + (4,)
    base = np.arange(48.0).reshape(shape[:-1] + (8,))
    if contiguous:
        a = base[..., :4].copy()
    else:
        a = base[..., ::-2].swapaxes(0, 1)
    expected = a.copy() + np.arange(a.size).reshape(a.shape) + 1
    kernel = CudaKernel(source, "update", options=("-DCUNUMPY_BOUNDS_CHECK",))
    if backend == "emulation":
        emulate_cuda_kernel(kernel, a, n_threads=a.size)
        np.testing.assert_array_equal(a, expected)
        if not contiguous:
            np.testing.assert_array_equal(
                base[..., 0::2], np.arange(48.0).reshape(base.shape)[..., 0::2]
            )
    else:
        import cunumpy as xp

        if not xp.cupy_available():
            pytest.skip("CuPy not installed or not functional")
        import cupy as cp

        device_base = cp.asarray(base)
        device = cp.asarray(a) if contiguous else device_base[..., ::-2].swapaxes(0, 1)
        kernel(device, n_threads=device.size)
        np.testing.assert_array_equal(cp.asnumpy(device), expected)


@pytest.mark.parametrize("contiguous", [False, True])
def test_high_dimensional_bounds_check(contiguous):
    ctype = f"{'C' if contiguous else ''}Array16D<double>"
    indices = ", ".join(["0"] * 15 + ["1"])
    source = f"""
    #include <cunumpy/array_view.cuh>
    extern "C" __global__ void invalid({ctype} a) {{ a({indices}) = 1; }}
    """
    with pytest.raises(RuntimeError, match="crashed"):
        emulate_cuda_kernel(
            CudaKernel(source, "invalid"),
            np.zeros((1,) * 16),
            n_threads=1,
            options=("-DCUNUMPY_BOUNDS_CHECK",),
        )


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """An empty disk cache and no libraries loaded; yields the list of compiler calls."""
    import subprocess

    from cunumpy import _emulation

    monkeypatch.setenv("CUNUMPY_EMULATION_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(_emulation, "_LIBRARIES", {})
    monkeypatch.setattr(_emulation, "_PREPARED", weakref.WeakKeyDictionary())
    calls = []
    run = subprocess.run

    def counting_run(command, *args, **kwargs):
        calls.append(command)
        return run(command, *args, **kwargs)

    monkeypatch.setattr(_emulation.subprocess, "run", counting_run)
    return calls


def test_a_kernel_is_compiled_once_for_all_launches(fresh_cache):
    kernel = CudaKernel(AXPY, "axpy")
    for n, a in [(10, 2.0), (1000, -0.5), (3, 7.0)]:
        x, y = np.arange(float(n)), np.ones(n)
        emulate_cuda_kernel(kernel, a, x, y, n, options=("-ffp-contract=off",))
        np.testing.assert_array_equal(y, 1.0 + a * np.arange(float(n)))
    # an equal kernel object shares the library
    emulate_cuda_kernel(
        CudaKernel(AXPY, "axpy"), 1.0, x, y, 3, options=("-ffp-contract=off",)
    )
    assert len(fresh_cache) == 1


def test_options_and_template_arguments_rebuild(fresh_cache):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for k in (2, 3, 2):
        emulate_cuda_kernel(
            CudaKernel(TEMPLATE, "power", template_args=(np.float32, k)), x, 3
        )
    assert x.tolist() == [1.0, 2.0**12, 3.0**12]
    assert len(fresh_cache) == 2
    emulate_cuda_kernel(
        CudaKernel(TEMPLATE, "power", template_args=(np.float32, 2)),
        x,
        3,
        options=("-DUNUSED=1",),
    )
    assert len(fresh_cache) == 3


def test_libraries_are_reused_from_the_disk_cache(fresh_cache, monkeypatch):
    from cunumpy import _emulation

    kernel = CudaKernel(AXPY, "axpy")
    y = np.zeros(4)
    emulate_cuda_kernel(kernel, 1.0, np.ones(4), y, 4)
    # a new process
    monkeypatch.setattr(_emulation, "_LIBRARIES", {})
    monkeypatch.setattr(_emulation, "_PREPARED", weakref.WeakKeyDictionary())
    emulate_cuda_kernel(kernel, 1.0, np.ones(4), y, 4)
    assert len(fresh_cache) == 1
    np.testing.assert_array_equal(y, 2.0)
    assert len(list(_emulation.emulation_cache_dir().glob("*.so"))) == 1


def test_without_disk_cache(fresh_cache, monkeypatch):
    from cunumpy import _emulation

    monkeypatch.setenv("CUNUMPY_EMULATION_CACHE", "0")
    assert _emulation.emulation_cache_dir() is None
    y = np.zeros(4)
    emulate_cuda_kernel(CudaKernel(AXPY, "axpy"), 1.0, np.ones(4), y, 4)
    np.testing.assert_array_equal(y, 1.0)


def test_compile_for_emulation_builds_without_a_launch(fresh_cache):
    from cunumpy.kernel_testing import compile_for_emulation

    compile_for_emulation(CudaKernel(AXPY, "axpy"))
    y = np.zeros(2)
    emulate_cuda_kernel(CudaKernel(AXPY, "axpy"), 1.0, np.ones(2), y, 2)
    assert len(fresh_cache) == 1
    with pytest.raises(RuntimeError, match="does not compile"):
        compile_for_emulation(
            CudaKernel('extern "C" __global__ void k(int n) { n = ; }', "k")
        )
    unparsed = CudaKernel(AXPY, "axpy", check_signature=False)
    compile_for_emulation(unparsed)  # a syntax check only


def test_arrays_are_used_in_place_and_aliases_see_each_other():
    source = r"""
    extern "C" __global__ void shift(const double* a, double* b, int n) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            for (int i = 1; i < n; ++i) b[i] = a[i - 1];
    }"""
    x = np.arange(5.0)
    emulate_cuda_kernel(CudaKernel(source, "shift"), x, x, 5, n_threads=1)
    np.testing.assert_array_equal(x, 0.0)  # serial: each write is read next
    readonly = np.arange(5.0)
    readonly.flags.writeable = False
    out = np.zeros(5)
    emulate_cuda_kernel(CudaKernel(source, "shift"), readonly, out, 5, n_threads=1)
    np.testing.assert_array_equal(out, [0, 0, 1, 2, 3])


def test_trap_in_a_kernel_with_barriers_is_reported():
    source = r"""
    extern "C" __global__ void k(double* x, int bad) {
        __shared__ double s[4];
        s[threadIdx.x] = x[threadIdx.x];
        __syncthreads();
        if (bad && threadIdx.x == 2) __trap();
        x[threadIdx.x] = s[3 - threadIdx.x];
    }"""
    kernel = CudaKernel(source, "k")
    with pytest.raises(RuntimeError, match="crashed"):
        emulate_cuda_kernel(kernel, np.arange(4.0), 1, grid=1, block=4)
    x = np.arange(4.0)
    emulate_cuda_kernel(kernel, x, 0, grid=1, block=4)  # still usable
    np.testing.assert_array_equal(x, [3, 2, 1, 0])


INLINE_ASM = r"""
extern "C" __global__ void map(double* x, int kind) {
    if (kind == 0) {
        x[0] = 1.0;
    } else if (kind == 1) {
        asm volatile("trap;");
    } else {
        asm("trap;");  // unknown kind
    }
}
"""


def test_inline_asm_is_trapped():
    kernel = CudaKernel(INLINE_ASM, "map")
    x = np.zeros(1)
    emulate_cuda_kernel(kernel, x, 0, n_threads=1)  # no extra options needed
    assert x[0] == 1.0
    for kind in (1, 2):
        with pytest.raises(RuntimeError, match="__trap"):
            emulate_cuda_kernel(kernel, x, kind, n_threads=1)
    # the option struphy used before still works
    emulate_cuda_kernel(kernel, x, 0, n_threads=1, options=("-Dasm(x)=__trap()",))


def test_compile_for_emulation_sees_changed_headers(fresh_cache, tmp_path):
    from cunumpy.kernel_testing import compile_for_emulation

    header = tmp_path / "factor.cuh"
    header.write_text("#define FACTOR 2.0\n")
    source = r"""
    #include "factor.cuh"
    extern "C" __global__ void k(double* x) {
        if (blockIdx.x == 0 && threadIdx.x == 0) x[0] *= FACTOR;
    }"""
    kernel = CudaKernel(source, "k", include_dirs=[tmp_path])
    x = np.ones(1)
    emulate_cuda_kernel(kernel, x, n_threads=1)
    header.write_text("#define FACTOR 3.0\n")
    emulate_cuda_kernel(kernel, x, n_threads=1)  # launches reuse the library
    assert x[0] == 4.0
    compile_for_emulation(kernel)  # as kernel.recompile() in emulated_launches
    emulate_cuda_kernel(kernel, x, n_threads=1)
    assert x[0] == 12.0
    assert len(fresh_cache) == 2
