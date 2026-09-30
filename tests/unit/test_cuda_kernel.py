"""Tests for `cunumpy.CudaKernel` and `cunumpy.parse_cuda_signature`.

Signature parsing and argument checking run everywhere: a small stand-in for a
device array (`FakeDeviceArray`) takes the place of CuPy arrays. Launching
kernels needs a GPU and is skipped without one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import (
    CudaArguments,
    CudaKernel,
    CudaKernelVariants,
    CudaStruct,
    CudaStructValue,
    ctype_of,
    parse_cuda_signature,
)

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
    """Enough of a CuPy array for the argument checks: dtype, interface, address."""

    __cuda_array_interface__ = {}

    def __init__(self, dtype, ptr=0x1000):
        self.dtype = np.dtype(dtype)
        self.data = SimpleNamespace(ptr=ptr)


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
    with pytest.raises(ValueError, match="block sizes must be positive"):
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


# ---------------------------------------------------------------------------
# ctype_of
# ---------------------------------------------------------------------------


def test_ctype_of():
    assert ctype_of(np.float64) == "double"
    assert ctype_of("float32") == "float"
    assert ctype_of(np.dtype(np.int64)) == "long long"
    assert ctype_of(np.complex128) == "complex<double>"
    with pytest.raises(ValueError, match="no C type"):
        ctype_of(np.dtype("U3"))


# ---------------------------------------------------------------------------
# structs
# ---------------------------------------------------------------------------

PARTICLES = CudaStruct(
    "Particles",
    [
        ("x", "double*"),
        ("n", "int"),
        ("charge", "double"),
        ("alive", "bool*"),
        ("ids", "long long*"),
        ("weight", "float"),
    ],
)

PUSH_SOURCE = (
    PARTICLES.declaration
    + r"""
extern "C" __global__
void push(Particles p, double dt, double* out, unsigned long long* size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        size[0] = sizeof(Particles);
        out[0] = p.n; out[1] = p.charge; out[2] = (double)p.ids[1]; out[3] = p.weight;
    }
    if (i < p.n && p.alive[i]) p.x[i] += dt * p.charge;
}
"""
)


def test_struct_layout_and_declaration():
    assert PARTICLES.name == "Particles"
    assert [f.name for f in PARTICLES.fields] == [
        "x",
        "n",
        "charge",
        "alive",
        "ids",
        "weight",
    ]
    # C layout: 8 (x) + 4 (n) + 4 padding + 8 (charge) + 8 + 8 + 4 (weight) + 4 padding
    assert PARTICLES.dtype.itemsize == 48
    assert PARTICLES.dtype.fields["charge"][1] == 16
    assert "    double* x;" in PARTICLES.declaration
    assert PARTICLES.declaration.startswith("struct Particles {")


def test_struct_errors():
    with pytest.raises(ValueError, match="duplicate field"):
        CudaStruct("S", [("a", "int"), ("a", "double")])
    with pytest.raises(ValueError, match="invalid struct name"):
        CudaStruct("not a name", [("a", "int")])
    with pytest.raises(ValueError, match="unsupported type"):
        CudaStruct("S", [("a", "Other")])


def test_struct_values():
    x = FakeDeviceArray(np.float64, ptr=0xABC0)
    value = PARTICLES(
        x=x,
        n=3,
        charge=2,
        alive=FakeDeviceArray(np.bool_),
        ids=FakeDeviceArray(np.int64),
        weight=0.5,
    )
    assert isinstance(value, CudaStructValue) and value.struct is PARTICLES
    assert value["x"] is x
    assert value.packed["x"] == 0xABC0  # the device address
    assert value.packed["charge"] == 2.0 and value.packed["n"] == 3
    assert value.__cuda_args__() == (value.packed,)

    with pytest.raises(TypeError, match=r"missing fields \['ids', 'weight'\]"):
        PARTICLES(x=x, n=3, charge=2.0, alive=FakeDeviceArray(np.bool_))
    with pytest.raises(TypeError, match="must have dtype float64"):
        PARTICLES(
            x=FakeDeviceArray(np.float32),
            n=3,
            charge=2.0,
            alive=FakeDeviceArray(np.bool_),
            ids=FakeDeviceArray(np.int64),
            weight=0.5,
        )
    with pytest.raises(TypeError, match="must be a CuPy array"):
        PARTICLES(
            x=np.zeros(3),
            n=3,
            charge=2.0,
            alive=FakeDeviceArray(np.bool_),
            ids=FakeDeviceArray(np.int64),
            weight=0.5,
        )


def test_struct_parameters():
    kernel = CudaKernel(PUSH_SOURCE, "push", structs=[PARTICLES])
    param = kernel.signature[0]
    assert param.struct is PARTICLES and param.dtype == PARTICLES.dtype

    value = PARTICLES(
        x=FakeDeviceArray(np.float64),
        n=3,
        charge=1.0,
        alive=FakeDeviceArray(np.bool_),
        ids=FakeDeviceArray(np.int64),
        weight=1.0,
    )
    out, size = FakeDeviceArray(np.float64), FakeDeviceArray(np.uint64)
    packed, dt, _, _ = kernel.prepare_args(value, 1, out, size)
    assert packed is value.packed and type(dt) is np.float64

    other = CudaStruct("Particles", [("x", "double*")])
    with pytest.raises(TypeError, match="must be a value of struct Particles"):
        kernel.prepare_args(other(x=FakeDeviceArray(np.float64)), 1.0, out, size)
    with pytest.raises(TypeError, match="must be a value of struct"):
        kernel.prepare_args(1.0, 1.0, out, size)

    # without the struct, the parameter type is unknown
    with pytest.raises(ValueError, match="unsupported type 'Particles'"):
        CudaKernel(PUSH_SOURCE, "push")


def test_struct_definition_must_match():
    changed = CudaStruct("Particles", [("x", "double*"), ("n", "long long")])
    with pytest.raises(ValueError, match="does not match its CudaStruct"):
        CudaKernel(PUSH_SOURCE, "push", structs=[changed])
    # no definition in the source (e.g. in a header): nothing to compare
    source = r'extern "C" __global__ void f(Particles p) {}'
    CudaKernel(source, "f", structs=[PARTICLES])
    with pytest.raises(ValueError, match="only be passed by value"):
        CudaKernel(r"__global__ void f(Particles* p) {}", "f", structs=[PARTICLES])


def test_struct_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    n = 300
    x = cp.zeros(n)
    alive = cp.ones(n, dtype=bool)
    alive[::2] = False
    value = PARTICLES(
        x=x,
        n=n,
        charge=2.0,
        alive=alive,
        ids=cp.array([7, 42], dtype=cp.int64),
        weight=1.5,
    )
    out, size = cp.zeros(4), cp.zeros(1, dtype=cp.uint64)
    CudaKernel(PUSH_SOURCE, "push", structs=[PARTICLES])(
        value, 0.5, out, size, n_threads=n
    )
    assert int(size.get()[0]) == PARTICLES.dtype.itemsize  # same layout as in C
    assert out.get().tolist() == [n, 2.0, 42.0, 1.5]
    assert cp.all(x[1::2] == 1.0) and cp.all(x[::2] == 0.0)


# ---------------------------------------------------------------------------
# templates and generated variants
# ---------------------------------------------------------------------------

SCALE_TEMPLATE = r"""
#include <cupy/complex.cuh>
template <typename T, int N>
__global__ void scale(T* x, T factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = factor * x[i] * (T)N;
}
"""


def test_template_signature():
    kernel = CudaKernel(SCALE_TEMPLATE, "scale", template_args=(np.float32, 2))
    assert kernel.expression == "scale<float, 2>"
    assert [(p.ctype, p.pointer) for p in kernel.signature] == [
        ("float", True),
        ("float", False),
        ("int", False),
    ]
    assert CudaKernel(SCALE_TEMPLATE, "scale", template_args=("double", 1)).signature[
        0
    ].dtype == np.dtype(np.float64)

    with pytest.raises(ValueError, match="template with 2 parameters"):
        CudaKernel(SCALE_TEMPLATE, "scale")
    with pytest.raises(ValueError, match="template with 2 parameters"):
        CudaKernel(SCALE_TEMPLATE, "scale", template_args=("double",))
    with pytest.raises(ValueError, match="not a template"):
        CudaKernel(AXPY, "axpy", template_args=("double",))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_template_on_gpu(dtype):
    _skip_without_cupy()
    import cupy as cp

    kernel = CudaKernel(SCALE_TEMPLATE, "scale", template_args=(dtype, 3))
    x = cp.arange(10).astype(dtype)
    kernel(x, 2, 10, n_threads=10)
    assert cp.allclose(x, 6 * cp.arange(10).astype(dtype))


def _generated_source(ndim, ctype):
    index = " + ".join(f"i{k}" for k in range(ndim))
    params = ", ".join(f"int i{k}" for k in range(ndim))
    return f"""
    extern "C" __global__ void fill({ctype}* out, {params}) {{
        out[threadIdx.x] = ({ctype})({index});
    }}
    """


def test_variants():
    created = []

    def factory(ndim, dtype):
        created.append((ndim, dtype))
        return CudaKernel(_generated_source(ndim, ctype_of(dtype)), "fill")

    variants = CudaKernelVariants(factory)
    k3 = variants.get(3, np.float64)
    assert variants.get(3, np.float64) is k3  # created once
    assert variants.get(2, np.float32) is not k3
    assert created == [(3, np.float64), (2, np.float32)] and len(variants) == 2
    assert variants.keys() == [(3, np.float64), (2, np.float32)]
    assert [p.ctype for p in k3.signature] == ["double", "int", "int", "int"]

    with pytest.raises(TypeError, match="must return a CudaKernel"):
        CudaKernelVariants(lambda n: n).get(1)


def test_variants_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    variants = CudaKernelVariants(
        lambda ndim, dtype: CudaKernel(
            _generated_source(ndim, ctype_of(dtype)), "fill", block_size=1
        )
    )
    variants.compile_all([(2, np.float64), (1, np.int32)])
    assert all(variants.get(*key).is_compiled for key in variants.keys())

    out = cp.zeros(1)
    variants.get(2, np.float64)(out, 3, 4, n_threads=1)
    assert out.get()[0] == 7.0


# ---------------------------------------------------------------------------
# launch shapes and shared memory
# ---------------------------------------------------------------------------


def test_launch_shape():
    kernel = CudaKernel(AXPY, "axpy")  # block_size 128
    assert kernel.launch_shape(1000) == ((8,), (128,))
    assert kernel.launch_shape((300, 5)) == ((3, 5), (128, 1))
    assert kernel.launch_shape((300, 5), block=(16, 8)) == ((19, 1), (16, 8))
    assert kernel.launch_shape(grid=(4, 2), block=(8, 8)) == ((4, 2), (8, 8))
    assert kernel.launch_shape(0) == ((0,), (128,))

    k2 = CudaKernel(AXPY, "axpy", block_size=(16, 16))
    assert k2.block_size == (16, 16)
    assert k2.launch_shape((100, 33)) == ((7, 3), (16, 16))

    with pytest.raises(ValueError, match="different numbers of dimensions"):
        k2.launch_shape(100)
    with pytest.raises(TypeError, match="exactly one of n_threads and grid"):
        kernel.launch_shape()
    with pytest.raises(TypeError, match="exactly one of n_threads and grid"):
        kernel.launch_shape(10, grid=1)
    with pytest.raises(ValueError, match="at most 1024 threads"):
        CudaKernel(AXPY, "axpy", block_size=(64, 32))
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        kernel.launch_shape((1, 2, 3, 4))
    with pytest.raises(ValueError, match="shared_mem"):
        kernel(1.0, n_threads=1, shared_mem=-1)


MATRIX_SOURCE = r"""
extern "C" __global__ void add_indices(double* a, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < nx && j < ny) a[i * ny + j] += 10 * i + j;
}
"""

BLOCK_SUM_SOURCE = r"""
extern "C" __global__ void block_sum(const double* x, double* out, int n) {
    extern __shared__ double buffer[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    buffer[threadIdx.x] = i < n ? x[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) buffer[threadIdx.x] += buffer[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = buffer[0];
}
"""


def test_2d_launch_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    nx, ny = 37, 21
    expected = 10 * cp.arange(nx)[:, None] + cp.arange(ny)[None, :]
    kernel = CudaKernel(MATRIX_SOURCE, "add_indices", block_size=(8, 4))
    a = cp.zeros((nx, ny))
    kernel(a, nx, ny, n_threads=(nx, ny))
    assert cp.array_equal(a, expected)

    b = cp.zeros((nx, ny))  # explicit grid and block
    kernel(b, nx, ny, grid=(3, 2), block=(16, 16))
    assert cp.array_equal(b, expected)


def test_shared_memory_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    n, block = 1000, 128
    x = cp.arange(n, dtype=cp.float64)
    kernel = CudaKernel(BLOCK_SUM_SOURCE, "block_sum", block_size=block)
    grid = kernel.launch_shape(n)[0][0]
    out = cp.zeros(grid)
    kernel(x, out, n, n_threads=n, shared_mem=block * 8)
    assert float(out.sum()) == n * (n - 1) / 2
