"""Tests for `cunumpy.CudaKernel` and `cunumpy.parse_cuda_signature`.

Signature parsing and argument checking run everywhere: a small stand-in for a
device array (`FakeDeviceArray`) takes the place of CuPy arrays. Launching
kernels needs a GPU and is skipped without one.
"""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import CudaArguments, CudaKernel, parse_cuda_signature

AXPY = r"""
// y = a * x + y
extern "C" __global__
void axpy(double a, const double* __restrict__ x, double *y, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""

ALL_TYPES = r"""
#include <cupy/complex.cuh>
/* several kernels; the one we look for is not the first */
extern "C" __global__ void other(int n) {}
extern "C" __global__
void all_types(bool b, char c, unsigned char uc, short s, int i, unsigned u,
               long l, long long ll, unsigned long long ull, size_t sz,
               int64_t i64, uint32_t u32, float f, double d,
               complex<float> cf, thrust::complex<double> cd,
               const float* pf, int* __restrict__ pi, void* pv, double arr[])
{
}
"""


class FakeDeviceArray:
    """Enough of a CuPy array for the argument checks: a dtype and the interface."""

    __cuda_array_interface__ = {}

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)


def _skip_without_cupy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")


# ---------------------------------------------------------------------------
# signature parsing
# ---------------------------------------------------------------------------


def test_parse_axpy():
    params = parse_cuda_signature(AXPY, "axpy")
    assert [(p.name, p.ctype, p.pointer) for p in params] == [
        ("a", "double", False),
        ("x", "double", True),
        ("y", "double", True),
        ("n", "int", False),
    ]
    assert [p.dtype for p in params] == [np.dtype(np.float64)] * 3 + [
        np.dtype(np.int32)
    ]


def test_parse_all_types():
    params = {p.name: p for p in parse_cuda_signature(ALL_TYPES, "all_types")}
    expected = {
        "b": np.bool_,
        "c": np.int8,
        "uc": np.uint8,
        "s": np.int16,
        "i": np.int32,
        "u": np.uint32,
        "l": np.int64,
        "ll": np.int64,
        "ull": np.uint64,
        "sz": np.uint64,
        "i64": np.int64,
        "u32": np.uint32,
        "f": np.float32,
        "d": np.float64,
        "cf": np.complex64,
        "cd": np.complex128,
        "pf": np.float32,
        "pi": np.int32,
        "arr": np.float64,
    }
    for name, dtype in expected.items():
        assert params[name].dtype == np.dtype(dtype), name
    assert params["pv"].dtype is None and params["pv"].pointer
    assert params["arr"].pointer and not params["d"].pointer


def test_parse_no_parameters():
    assert parse_cuda_signature('extern "C" __global__ void f() {}', "f") == ()
    assert parse_cuda_signature('extern "C" __global__ void f(void) {}', "f") == ()


def test_parse_errors():
    with pytest.raises(ValueError, match="no __global__ function 'missing'"):
        parse_cuda_signature(AXPY, "missing")
    with pytest.raises(ValueError, match="unsupported type"):
        parse_cuda_signature("__global__ void f(MyStruct s) {}", "f")
    with pytest.raises(ValueError, match="unsupported type"):
        parse_cuda_signature("__global__ void f(double** p) {}", "f")
    # a commented-out kernel is not found
    with pytest.raises(ValueError, match="no __global__ function"):
        parse_cuda_signature("// __global__ void f(int n) {}", "f")


def test_unparsable_signature_can_be_skipped():
    source = "#define ARGS double* x, int n\n__global__ void f(ARGS) {}"
    with pytest.raises(ValueError):
        CudaKernel(source, "f")
    kernel = CudaKernel(source, "f", check_signature=False)
    assert kernel.signature is None
    values = kernel.prepare_args(1, 2.5, "anything")
    assert values == (1, 2.5, "anything")  # passed on as they are


# ---------------------------------------------------------------------------
# argument checks (no GPU needed)
# ---------------------------------------------------------------------------


def test_python_scalars_are_cast_to_the_declared_types():
    kernel = CudaKernel(AXPY, "axpy")
    x, y = FakeDeviceArray(np.float64), FakeDeviceArray(np.float64)

    a, x_out, y_out, n = kernel.prepare_args(2, x, y, 10)
    assert type(a) is np.float64 and a == 2.0  # int into double: cast, not garbage
    assert type(n) is np.int32 and n == 10
    assert x_out is x and y_out is y  # arrays are passed as they are

    a, _, _, n = kernel.prepare_args(0.5, x, y, True)
    assert type(a) is np.float64 and type(n) is np.int32 and n == 1


def test_wrong_scalars_raise():
    kernel = CudaKernel(AXPY, "axpy")
    x, y = FakeDeviceArray(np.float64), FakeDeviceArray(np.float64)

    with pytest.raises(TypeError, match=r"argument 3 \(int n\)"):
        kernel.prepare_args(1.0, x, y, 2.5)  # float into int
    with pytest.raises(OverflowError, match="out of range"):
        kernel.prepare_args(1.0, x, y, 2**31)  # overflows int
    with pytest.raises(TypeError, match="losing information"):
        kernel.prepare_args(1.0, x, y, np.int64(5))  # int64 into int
    with pytest.raises(TypeError):
        kernel.prepare_args("1.0", x, y, 5)


def test_numpy_scalars():
    kernel = CudaKernel(AXPY, "axpy")
    x, y = FakeDeviceArray(np.float64), FakeDeviceArray(np.float64)

    a, _, _, n = kernel.prepare_args(np.float32(1.5), x, y, np.int16(3))
    assert type(a) is np.float64 and type(n) is np.int32  # safe casts
    a, _, _, n = kernel.prepare_args(np.float64(1.5), x, y, np.int32(3))
    assert type(a) is np.float64 and type(n) is np.int32

    kernel_f = CudaKernel("__global__ void f(float a) {}", "f")
    with pytest.raises(TypeError, match="losing information"):
        kernel_f.prepare_args(np.float64(1.5))
    assert type(kernel_f.prepare_args(1.5)[0]) is np.float32


def test_array_checks():
    kernel = CudaKernel(AXPY, "axpy")
    x = FakeDeviceArray(np.float64)

    with pytest.raises(
        TypeError, match=r"argument 2 \(double\* y\) must be a CuPy array"
    ):
        kernel.prepare_args(1.0, x, np.zeros(3), 3)  # host array: never copied
    with pytest.raises(TypeError, match="must have dtype float64, got float32"):
        kernel.prepare_args(1.0, x, FakeDeviceArray(np.float32), 3)
    with pytest.raises(TypeError, match="takes 4 arguments"):
        kernel.prepare_args(1.0, x, x)

    void_kernel = CudaKernel("__global__ void f(void* p) {}", "f")
    void_kernel.prepare_args(FakeDeviceArray(np.int8))  # any dtype


def test_argument_objects_are_flattened():
    class Vectors(CudaArguments):
        def __init__(self, x, y, n):
            super().__init__(x, y, n)

    class Duck:
        def __init__(self, x, y, n):
            self.values = (x, y, n)

        def __cuda_args__(self):
            return self.values

    kernel = CudaKernel(AXPY, "axpy")
    x, y = FakeDeviceArray(np.float64), FakeDeviceArray(np.float64)
    for args in (Vectors(x, y, 7), Duck(x, y, 7)):
        a, x_out, y_out, n = kernel.prepare_args(2.0, args)
        assert x_out is x and y_out is y and n == 7 and type(n) is np.int32

    # the flattened arguments are checked too
    with pytest.raises(TypeError, match="takes 4 arguments"):
        kernel.prepare_args(2.0, CudaArguments(x, y))


def test_from_file(tmp_path):
    path = tmp_path / "axpy_cuda.cu"
    path.write_text(AXPY)
    kernel = CudaKernel.from_file(path, block_size=64)
    assert kernel.name == "axpy" and kernel.block_size == 64
    assert f"-I{tmp_path}" in kernel.options

    other = tmp_path / "saxpy.cu"
    other.write_text(AXPY)
    with pytest.raises(ValueError, match="does not end with"):
        CudaKernel.from_file(other)
    assert CudaKernel.from_file(other, "axpy").name == "axpy"


def test_launch_argument_validation():
    kernel = CudaKernel(AXPY, "axpy")
    with pytest.raises(ValueError, match="non-negative"):
        kernel(1.0, n_threads=-1)
    with pytest.raises(ValueError, match="block_size"):
        CudaKernel(AXPY, "axpy", block_size=0)
    # n_threads=0 checks the arguments but launches nothing (works without a GPU)
    x = FakeDeviceArray(np.float64)
    kernel(1.0, x, x, 0, n_threads=0)


def test_compile_without_gpu_raises():
    if xp.cupy_available():
        pytest.skip("a GPU is available")
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        CudaKernel(AXPY, "axpy").compile()


# ---------------------------------------------------------------------------
# launching (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 127, 128, 129, 1000])
def test_axpy_on_gpu(n):
    _skip_without_cupy()
    import cupy as cp

    kernel = CudaKernel(AXPY, "axpy")
    x = cp.arange(n, dtype=cp.float64)
    y = cp.ones(n)
    kernel(2, x, y, n, n_threads=n)
    assert cp.allclose(y, 2 * x + 1)
    assert kernel.compile() is kernel.compile()  # compiled once


def test_argument_object_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    n = 50
    x, y = cp.arange(n, dtype=cp.float64), cp.zeros(n)
    CudaKernel(AXPY, "axpy")(0.5, CudaArguments(x, y, n), n_threads=n)
    assert cp.allclose(y, 0.5 * x)


def test_stream_and_include_dirs_on_gpu(tmp_path):
    _skip_without_cupy()
    import cupy as cp

    (tmp_path / "helpers.cuh").write_text(
        "__device__ double twice(double v) { return 2 * v; }\n"
    )
    source = r"""
    #include "helpers.cuh"
    extern "C" __global__ void double_it(double* y, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) y[i] = twice(y[i]);
    }
    """
    kernel = CudaKernel(source, "double_it", include_dirs=[tmp_path], block_size=32)
    y = cp.ones(100)
    stream = cp.cuda.Stream()
    kernel(y, 100, n_threads=100, stream=stream)
    stream.synchronize()
    assert cp.all(y == 2)


def test_scalars_arrive_correctly_on_gpu():
    """Python scalars reach int/long long/float/double/bool parameters correctly."""
    _skip_without_cupy()
    import cupy as cp

    source = r"""
    extern "C" __global__
    void write(double* out, int a, long long b, float c, double d, bool e) {
        out[0] = a; out[1] = (double)b; out[2] = c; out[3] = d; out[4] = e;
    }
    """
    out = cp.zeros(5)
    CudaKernel(source, "write")(out, 3, 2**40, 1.5, 2, True, n_threads=1)
    assert out.get().tolist() == [3.0, float(2**40), 1.5, 2.0, 1.0]
