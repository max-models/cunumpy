"""Tests for `cunumpy.CudaKernel` and `cunumpy.parse_cuda_signature`.

Signature parsing and argument checking run everywhere: a small stand-in for a
device array (`FakeDeviceArray`) takes the place of CuPy arrays. Launching
kernels needs a GPU and is skipped without one.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Final

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
    cuda_include_dir,
    parse_cuda_signature,
    write_cuda_header,
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
    """Enough of a CuPy array for the argument checks: dtype, interface, address.

    `shape` and `strides` (in bytes, C-contiguous by default) are needed for
    array view parameters.
    """

    def __init__(self, dtype, ptr=0x1000, shape=(1,), strides=None):
        self.dtype = np.dtype(dtype)
        self.data = SimpleNamespace(ptr=ptr)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        if strides is None:
            strides, stride = [], self.dtype.itemsize
            for n in reversed(self.shape):
                strides.insert(0, stride)
                stride *= n
        self.strides = tuple(strides)
        c_contiguous = self.strides == tuple(
            np.zeros(self.shape, dtype=self.dtype).strides
        )
        self.flags = SimpleNamespace(c_contiguous=c_contiguous)

    @property
    def __cuda_array_interface__(self):
        return {}


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
        _a, x_out, y_out, n = kernel.prepare_args(2.0, args)
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
    assert kernel.options[-1] == f"-I{cuda_include_dir()}"

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

PUSH_SOURCE = PARTICLES.declaration + r"""
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
# array views (cunumpy/array_view.cuh) and shipped headers
# ---------------------------------------------------------------------------

SCALE_COLUMN = r"""
#include "cunumpy/array_view.cuh"
#include <cunumpy/index.cuh>
extern "C" __global__
void scale_column(Array2D<double> a, long long column, double factor) {
    CUNUMPY_THREAD_1D(i, a.shape[0]);
    a(i, column) *= factor;
}
"""

VIEW_TYPES = r"""
#include <cupy/complex.cuh>
#include "cunumpy/array_view.cuh"
extern "C" __global__
void views(Array1D<double> a, const Array2D< long long > b, Array3D<float> c,
           Array2D<complex<double>> d, Array1D<const bool> e, int n) {}
"""


def test_cuda_include_dir_contains_the_headers():
    include = Path(cuda_include_dir())
    assert include.is_dir()
    for name in ("array_view.cuh", "index.cuh"):
        header = include / "cunumpy" / name
        assert header.is_file(), header
        text = header.read_text()
        assert "#ifndef CUNUMPY_" in text and "#endif" in text
    view = (include / "cunumpy" / "array_view.cuh").read_text()
    assert all(f"struct Array{n}D" in view for n in (1, 2, 3))
    assert "CUNUMPY_BOUNDS_CHECK" in view
    index = (include / "cunumpy" / "index.cuh").read_text()
    assert "CUNUMPY_THREAD_1D(i, n)" in index
    assert "CUNUMPY_GRID_STRIDE_1D(i, n)" in index


def test_kernels_get_the_cunumpy_include_option():
    kernel = CudaKernel(AXPY, "axpy", options=("-std=c++17",), include_dirs=["/x"])
    assert kernel.options == ("-std=c++17", "-I/x", f"-I{cuda_include_dir()}")


def test_parse_view_parameters():
    params = parse_cuda_signature(VIEW_TYPES, "views")
    assert [(p.name, p.ctype, p.view_ndim, p.dtype) for p in params] == [
        ("a", "Array1D<double>", 1, np.dtype(np.float64)),
        ("b", "Array2D<long long>", 2, np.dtype(np.int64)),
        ("c", "Array3D<float>", 3, np.dtype(np.float32)),
        ("d", "Array2D<complex<double>>", 2, np.dtype(np.complex128)),
        ("e", "Array1D<bool>", 1, np.dtype(np.bool_)),
        ("n", "int", None, np.dtype(np.int32)),
    ]
    assert not any(p.pointer or p.struct for p in params)

    with pytest.raises(ValueError, match="array views"):
        parse_cuda_signature("__global__ void f(Array2D<double>* a) {}", "f")
    with pytest.raises(ValueError, match="array views"):
        parse_cuda_signature("__global__ void f(Array2D<Other> a) {}", "f")
    with pytest.raises(ValueError, match="unsupported type"):
        parse_cuda_signature("__global__ void f(Array4D<double> a) {}", "f")


def test_view_parameters_pack_pointer_shape_and_strides():
    kernel = CudaKernel(SCALE_COLUMN, "scale_column")
    # a non-contiguous view: every second row of a (8, 3) float64 array
    a = FakeDeviceArray(np.float64, ptr=0xF00, shape=(4, 3), strides=(48, 8))
    packed, column, factor = kernel.prepare_args(a, 1, 2)
    assert isinstance(packed, np.void)
    assert packed.dtype.itemsize == 8 + 2 * 8 + 2 * 8  # data, shape[2], strides[2]
    assert packed.dtype.isalignedstruct
    assert packed["data"] == 0xF00
    assert packed["shape"].tolist() == [4, 3]
    assert packed["strides"].tolist() == [6, 1]  # in elements, not bytes
    assert type(column) is np.int64 and type(factor) is np.float64

    # 1D and 3D, contiguous
    kernel3 = CudaKernel(VIEW_TYPES, "views")
    values = kernel3.prepare_args(
        FakeDeviceArray(np.float64, shape=(5,)),
        FakeDeviceArray(np.int64, shape=(2, 3)),
        FakeDeviceArray(np.float32, shape=(2, 3, 4)),
        FakeDeviceArray(np.complex128, shape=(1, 1)),
        FakeDeviceArray(np.bool_, shape=(7,), strides=(2,)),
        1,
    )
    assert values[0]["shape"].tolist() == [5] and values[0]["strides"].tolist() == [1]
    assert values[2]["shape"].tolist() == [2, 3, 4]
    assert values[2]["strides"].tolist() == [12, 4, 1]
    assert values[4]["strides"].tolist() == [2]
    assert [v.dtype.itemsize for v in values[:5]] == [24, 40, 56, 40, 24]


def test_view_parameter_errors():
    kernel = CudaKernel(SCALE_COLUMN, "scale_column")
    with pytest.raises(
        TypeError, match=r"argument 0 \(Array2D<double> a\) must be a CuPy"
    ):
        kernel.prepare_args(np.zeros((2, 2)), 1, 2.0)
    with pytest.raises(TypeError, match="must have dtype float64, got float32"):
        kernel.prepare_args(FakeDeviceArray(np.float32, shape=(2, 2)), 1, 2.0)
    with pytest.raises(TypeError, match="must be a 2D array, got 1D"):
        kernel.prepare_args(FakeDeviceArray(np.float64, shape=(2,)), 1, 2.0)
    with pytest.raises(TypeError, match="not multiples of the element size"):
        kernel.prepare_args(
            FakeDeviceArray(np.float64, shape=(2, 2), strides=(20, 8)), 1, 2.0
        )


MARKERS = CudaStruct(
    "Markers",
    [("markers", "Array2D<double>"), ("valid", "Array1D<bool>"), ("n", "int")],
)

PUSH_MARKERS_SOURCE = r"""
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"
struct Markers {
    Array2D<double> markers;
    Array1D<bool> valid;
    int n;
};
extern "C" __global__
void push_markers(Markers m, double dt, unsigned long long* size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) size[0] = sizeof(Markers);
    CUNUMPY_THREAD_1D(ip, m.n);
    if (m.valid(ip)) m.markers(ip, 0) += dt * m.markers(ip, 1);
}
"""


def test_struct_with_view_fields():
    assert [f.view_ndim for f in MARKERS.fields] == [2, 1, None]
    assert MARKERS.has_views and not PARTICLES.has_views
    assert "    Array2D<double> markers;" in MARKERS.declaration
    # C layout: 40 (Array2D) + 24 (Array1D) + 4 (int) + 4 padding
    assert MARKERS.dtype.itemsize == 72
    assert MARKERS.dtype.fields["valid"][1] == 40
    assert MARKERS.dtype.fields["n"][1] == 64

    markers = FakeDeviceArray(np.float64, ptr=0xABC0, shape=(10, 6), strides=(96, 16))
    value = MARKERS(markers=markers, valid=FakeDeviceArray(np.bool_, shape=(10,)), n=10)
    assert value["markers"] is markers
    assert value.packed["markers"]["data"] == 0xABC0
    assert value.packed["markers"]["shape"].tolist() == [10, 6]
    assert value.packed["markers"]["strides"].tolist() == [12, 2]
    assert value.packed["valid"]["strides"].tolist() == [1]
    assert value.packed["n"] == 10

    with pytest.raises(TypeError, match="must be a 2D array, got 3D"):
        MARKERS(
            markers=FakeDeviceArray(np.float64, shape=(1, 1, 1)),
            valid=FakeDeviceArray(np.bool_, shape=(1,)),
            n=1,
        )
    with pytest.raises(TypeError, match="must have dtype bool"):
        MARKERS(markers=markers, valid=FakeDeviceArray(np.int8, shape=(10,)), n=10)

    # the definition in the source is compared, views included
    kernel = CudaKernel(PUSH_MARKERS_SOURCE, "push_markers", structs=[MARKERS])
    assert kernel.signature[0].struct is MARKERS
    changed = CudaStruct("Markers", [("markers", "Array3D<double>")])
    with pytest.raises(ValueError, match="does not match its CudaStruct"):
        CudaKernel(PUSH_MARKERS_SOURCE, "push_markers", structs=[changed])


def test_scale_column_of_non_contiguous_view_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    full = cp.arange(48, dtype=cp.float64).reshape(8, 6)
    expected = full.get()
    view = full[::2, 1:5]  # non-contiguous in both dimensions
    assert not view.flags.c_contiguous
    kernel = CudaKernel(SCALE_COLUMN, "scale_column", block_size=2)
    kernel(view, 1, 10.0, n_threads=view.shape[0])
    expected[::2, 2] *= 10.0  # column 1 of the view is column 2 of `full`
    assert np.array_equal(full.get(), expected)


def test_struct_with_view_fields_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    base = cp.zeros((20, 8))
    base[:, 3] = 1.0  # velocity column of the full array
    markers = base[::2, 2:5]  # rows 0, 2, ...; columns 2 (x), 3 (v), 4
    valid = cp.ones(10, dtype=bool)
    valid[0] = False
    size = cp.zeros(1, dtype=cp.uint64)
    kernel = CudaKernel(PUSH_MARKERS_SOURCE, "push_markers", structs=[MARKERS])
    kernel(MARKERS(markers=markers, valid=valid, n=10), 0.5, size, n_threads=10)
    assert int(size.get()[0]) == MARKERS.dtype.itemsize
    expected = np.zeros((20, 8))
    expected[:, 3] = 1.0
    expected[2::2, 2] = 0.5
    assert np.array_equal(base.get(), expected)


def test_bounds_check_traps_on_gpu():
    _skip_without_cupy()
    import cupy as cp

    kernel = CudaKernel(
        SCALE_COLUMN, "scale_column", options=("-DCUNUMPY_BOUNDS_CHECK",)
    )
    a = cp.ones((4, 3))
    kernel(a, 1, 2.0, n_threads=4)
    cp.cuda.Device().synchronize()
    assert a.get()[:, 1].tolist() == [2.0] * 4
    with pytest.raises(Exception):
        kernel(a, 5, 2.0, n_threads=4)  # column out of bounds
        cp.cuda.Device().synchronize()


# ---------------------------------------------------------------------------
# structs from pyccel annotations and header generation
# ---------------------------------------------------------------------------


class MarkerArguments:
    """Like struphy's MarkerArguments: pyccel annotations on __init__."""

    def __init__(
        self,
        markers: "float[:, :]",
        n_markers: "int",
        valid_mks: "bool[:]",
        weights: "Final[float[:]]",
        vdim: int,
        dt: float,
        use_perp: bool,
        cx: "const float[:,:,:]",
        ids: Final["int[:]"],
        f32: np.float32,
    ):
        self.markers = markers


def test_from_signature():
    struct = CudaStruct.from_signature(MarkerArguments.__init__, "MarkerArgs")
    assert struct.name == "MarkerArgs"
    assert [(f.name, f.ctype) for f in struct.fields] == [
        ("markers", "Array2D<double>"),
        ("n_markers", "long long"),
        ("valid_mks", "Array1D<bool>"),
        ("weights", "Array1D<double>"),
        ("vdim", "long long"),
        ("dt", "double"),
        ("use_perp", "bool"),
        ("cx", "Array3D<double>"),
        ("ids", "Array1D<long long>"),
        ("f32", "float"),
    ]
    assert struct.declaration == (
        "struct MarkerArgs {\n"
        "    Array2D<double> markers;\n"
        "    long long n_markers;\n"
        "    Array1D<bool> valid_mks;\n"
        "    Array1D<double> weights;\n"
        "    long long vdim;\n"
        "    double dt;\n"
        "    bool use_perp;\n"
        "    Array3D<double> cx;\n"
        "    Array1D<long long> ids;\n"
        "    float f32;\n"
        "};\n"
    )

    # 32-bit ints, single precision
    struct32 = CudaStruct.from_signature(
        MarkerArguments.__init__,
        "Args32",
        int_type="int",
        scalar_names={"float": "float"},
    )
    assert [f.ctype for f in struct32.fields][:2] == ["Array2D<float>", "int"]
    assert struct32.fields[8].ctype == "Array1D<int>"

    # a plain function, self is not special-cased there
    def kernel_args(x: "float[:]", n: int):
        pass

    assert [f.ctype for f in CudaStruct.from_signature(kernel_args, "A").fields] == [
        "Array1D<double>",
        "long long",
    ]


def test_from_signature_errors():
    def missing(x, n: int):
        pass

    def unknown(x: "str[:]"):
        pass

    def too_many(x: "float[:, :, :, :]"):
        pass

    def unparsable(x: "float[:](order=F)"):
        pass

    with pytest.raises(
        ValueError, match="parameter 'x' of .*missing.* no type annotation"
    ):
        CudaStruct.from_signature(missing, "A")
    with pytest.raises(ValueError, match="unsupported scalar type 'str'"):
        CudaStruct.from_signature(unknown, "A")
    with pytest.raises(ValueError, match="at most 3 dimensions"):
        CudaStruct.from_signature(too_many, "A")
    with pytest.raises(ValueError, match="cannot parse the annotation"):
        CudaStruct.from_signature(unparsable, "A")


def test_to_header(tmp_path):
    struct = CudaStruct.from_signature(MarkerArguments.__init__, "MarkerArgs")
    header = struct.to_header()
    assert header == (
        "// Generated by cunumpy.CudaStruct from the Python definition; do not edit.\n"
        "#ifndef MARKERARGS_CUH\n"
        "#define MARKERARGS_CUH\n"
        "\n"
        '#include "cunumpy/array_view.cuh"\n'
        "\n" + struct.declaration + "\n"
        "#endif  // MARKERARGS_CUH\n"
    )
    # writing, custom guard and includes
    path = tmp_path / "marker_args.cuh"
    written = struct.to_header(path, guard="STRUPHY_MARKER_ARGS", includes=["defs.cuh"])
    assert path.read_text() == written
    assert "#ifndef STRUPHY_MARKER_ARGS\n#define STRUPHY_MARKER_ARGS\n" in written
    assert '#include "cunumpy/array_view.cuh"\n#include "defs.cuh"\n' in written
    # the pattern: a committed header equals the generated one
    assert path.read_text() == struct.to_header(
        guard="STRUPHY_MARKER_ARGS", includes=["defs.cuh"]
    )
    # a struct without views does not include the array view header
    plain = PARTICLES.to_header()
    assert "array_view" not in plain and PARTICLES.declaration in plain
    # the generated header defines the struct as the kernel expects
    struct.check_source(header)
    CudaKernel(
        header + 'extern "C" __global__ void f(MarkerArgs a) {}', "f", structs=[struct]
    )


def test_write_cuda_header(tmp_path):
    path = tmp_path / "pusher_args.cuh"
    header = write_cuda_header(
        path, [MARKERS, PARTICLES], includes=["#include <cupy/complex.cuh>"]
    )
    assert path.read_text() == header
    assert header.startswith(
        "// Generated by cunumpy.CudaStruct from the Python definition; do not edit.\n"
        "#ifndef PUSHER_ARGS_CUH\n#define PUSHER_ARGS_CUH\n\n"
        '#include "cunumpy/array_view.cuh"\n#include <cupy/complex.cuh>\n\n'
    )
    assert header.index("struct Markers") < header.index("struct Particles")
    assert header.endswith("#endif  // PUSHER_ARGS_CUH\n")
    assert write_cuda_header(path, [PARTICLES], guard="G") == PARTICLES.to_header(
        guard="G"
    )
    for struct in (MARKERS, PARTICLES):
        struct.check_source(header)


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
    assert variants.keys() == list(variants) == [(3, np.float64), (2, np.float32)]
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
    assert all(variants.get(*key).is_compiled for key in variants)

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
