"""Tests for `cunumpy.testing`.

The pytest markers, the array collection and comparison of `assert_kernels_agree`
and the source generation of `device_function_kernel` run everywhere; running
the kernels needs a GPU and is skipped without one.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
import cunumpy.testing
from cunumpy import CudaArguments, CudaKernel, Kernel, parse_cuda_signature
from cunumpy.testing import (
    BACKENDS,
    _collect_arrays,
    _compare_results,
    assert_kernels_agree,
    backend,  # noqa: F401 - the fixture is used by name
    device_function_kernel,
    requires_cupy,
)

SCALE_CUDA = r"""
extern "C" __global__ void scale(double* x, double factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""


def scale(x, factor, n):
    for i in range(n):
        x[i] *= factor


def scale_wrong(x, factor, n):
    for i in range(n):
        x[i] *= factor + 1


def make_scale_args(backend_name, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(300))
    return (x, 2.0, x.size)


# ---------------------------------------------------------------------------
# pytest markers and fixture
# ---------------------------------------------------------------------------


def test_import_does_not_need_pytest():
    """cunumpy.testing (and cunumpy) import without importing pytest."""
    code = (
        "import sys, cunumpy, cunumpy.testing\n"
        "assert 'pytest' not in sys.modules\n"
        "assert 'device_function_kernel' in dir(cunumpy.testing)\n"
    )
    # the same source tree as this test, whether or not cunumpy is installed
    source_root = Path(cunumpy.__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(source_root)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_backends_and_marker():
    assert BACKENDS[0] == "numpy"
    assert BACKENDS[1].values == ("cupy",) and requires_cupy in BACKENDS[1].marks
    assert requires_cupy.name == "skipif"
    assert requires_cupy.args == (not xp.cupy_available(),)
    assert requires_cupy.kwargs["reason"] == "CuPy/GPU not available"
    with pytest.raises(AttributeError):
        cunumpy.testing.no_such_thing


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_parametrize_over_backends(backend_name):
    """The cupy case is skipped without a GPU; the numpy case always runs."""
    if backend_name == "cupy":
        assert xp.cupy_available()
    with xp.use_backend(backend_name):
        assert xp.get_backend() == backend_name


def test_backend_fixture(backend):  # noqa: F811 - the fixture imported above
    assert xp.get_backend() == backend
    if backend == "cupy":
        assert xp.cupy_available()


# ---------------------------------------------------------------------------
# assert_kernels_agree (no GPU)
# ---------------------------------------------------------------------------


def test_collect_arrays():
    class Holder:
        def __init__(self, x, values):
            self.x = x
            self.values = values
            self.n = 3

    x, y, z, w = np.zeros(3), np.ones(3), np.full(3, 2.0), np.full(3, 3.0)
    args = (x, 2.0, (y, 1), CudaArguments(z, 5), Holder(w, [x, y]), 7)
    found = _collect_arrays(args)
    assert found == {
        "argument 0": x,
        "argument 2[0]": y,
        "argument 3._cuda_args[0]": z,
        "argument 4.x": w,
        "argument 4.values[0]": x,
        "argument 4.values[1]": y,
    }
    assert _collect_arrays(args, outputs=(0, -2)) == {
        "argument 0": x,
        "argument 4.x": w,
        "argument 4.values[0]": x,
        "argument 4.values[1]": y,
    }
    assert _collect_arrays(args, outputs=()) == {}

    with pytest.raises(IndexError, match="output argument 6 does not exist"):
        _collect_arrays(args, outputs=(6,))
    with pytest.raises(TypeError, match="positional argument indices"):
        _collect_arrays(args, outputs=("out",))


def test_compare_results():
    host = {"argument 0": np.arange(4.0), "argument 2.x": np.ones(2)}
    device = {"argument 0": np.arange(4.0) * (1 + 1e-14), "argument 2.x": np.ones(2)}
    _compare_results(host, device, rtol=1e-12, atol=0.0)

    device["argument 2.x"] = np.array([1.0, 1.5])
    with pytest.raises(AssertionError, match="scale: argument 2.x differs"):
        _compare_results(host, device, 1e-12, 0.0, kernel_name="scale")
    _compare_results(host, device, 1e-12, 0.5)  # within tolerance

    with pytest.raises(AssertionError, match="not have the same array arguments"):
        _compare_results(host, {"argument 0": np.arange(4.0)}, 1e-12, 0.0)


def test_assert_kernels_agree_arguments():
    with pytest.raises(TypeError, match="expected a Kernel"):
        assert_kernels_agree(scale, make_scale_args, n_threads=300)
    with pytest.raises(ValueError, match="has no CUDA version"):
        assert_kernels_agree(Kernel(scale), make_scale_args, n_threads=300)
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    with pytest.raises(TypeError, match="n_threads"):
        assert_kernels_agree(kernel, make_scale_args)
    with pytest.raises(ValueError, match="n_calls"):
        assert_kernels_agree(kernel, make_scale_args, n_threads=300, n_calls=0)


def test_assert_kernels_agree_skips_without_cupy():
    if xp.cupy_available():
        pytest.skip("a GPU is available")
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    with pytest.raises(pytest.skip.Exception, match="CuPy/GPU not available"):
        assert_kernels_agree(kernel, make_scale_args, n_threads=300)


# ---------------------------------------------------------------------------
# assert_kernels_agree (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")
def test_assert_kernels_agree_on_gpu():
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    host = assert_kernels_agree(kernel, make_scale_args, n_threads=300, n_calls=2)
    expected = 4 * np.random.default_rng(0).random(300)
    np.testing.assert_allclose(host["argument 0"], expected)

    # the declared outputs of the host kernel are used by default
    declared = Kernel(
        scale, CudaKernel(SCALE_CUDA, "scale"), host_options={"outputs": (0,)}
    )
    assert list(assert_kernels_agree(declared, make_scale_args, n_threads=300)) == [
        "argument 0"
    ]

    wrong = Kernel(scale_wrong, CudaKernel(SCALE_CUDA, "scale"), name="scale")
    with pytest.raises(AssertionError, match="scale: argument 0 differs"):
        assert_kernels_agree(wrong, make_scale_args, n_threads=300)


@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")
def test_assert_kernels_agree_restores_backend():
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    with xp.use_backend("numpy"):
        assert_kernels_agree(kernel, make_scale_args, grid=3, block=128)
        assert xp.get_backend() == "numpy"


# ---------------------------------------------------------------------------
# device_function_kernel
# ---------------------------------------------------------------------------

FIND_SPAN = r"""
__device__ int find_span(const double* t, int p, double eta) {
    int span = p;
    while (t[span + 1] <= eta) span++;
    return span;
}
"""


def test_device_function_kernel_source():
    kernel = device_function_kernel(
        FIND_SPAN, "int find_span(const double* t, int p, double eta)"
    )
    assert kernel.name == "find_span_kernel"
    assert FIND_SPAN in kernel.source
    expected = (
        'extern "C" __global__ void find_span_kernel('
        "const double* t, const int* p, const double* eta, int* out, int n)"
    )
    assert expected in kernel.source
    assert "if (i >= n) return;" in kernel.source
    assert "out[i] = find_span(t, p[i], eta[i]);" in kernel.source

    params = parse_cuda_signature(kernel.source, "find_span_kernel")
    assert [(p.name, p.ctype, p.pointer) for p in params] == [
        ("t", "double", True),
        ("p", "int", True),
        ("eta", "double", True),
        ("out", "int", True),
        ("n", "int", False),
    ]
    assert kernel.signature == params


def test_device_function_kernel_options():
    kernel = device_function_kernel(
        "",
        "__device__ inline void fill(double* x, int n_cols, float value);",
        name="fill_all",
        includes=("helpers.cuh", "<cupy/complex.cuh>"),
        n_threads_param="count",
        block_size=64,
    )
    assert kernel.name == "fill_all" and kernel.block_size == 64
    assert kernel.source.startswith(
        '#include "helpers.cuh"\n#include <cupy/complex.cuh>\n'
    )
    assert (
        'extern "C" __global__ void fill_all('
        "double* x, const int* n_cols, const float* value, int count)"
    ) in kernel.source
    assert "fill(x, n_cols[i], value[i]);" in kernel.source  # void: no out
    assert "out" not in kernel.source
    assert [p.name for p in kernel.signature] == ["x", "n_cols", "value", "count"]

    no_params = device_function_kernel("", "double pi()")
    assert "pi_kernel(double* out, int n)" in no_params.source
    assert "out[i] = pi();" in no_params.source


def test_device_function_kernel_errors():
    with pytest.raises(ValueError, match="cannot parse the function prototype"):
        device_function_kernel("", "not a prototype")
    with pytest.raises(ValueError, match="unsupported return type 'MyStruct'"):
        device_function_kernel("", "MyStruct make(double x)")
    with pytest.raises(ValueError, match="returns a pointer"):
        device_function_kernel("", "double* first(double* x)")
    with pytest.raises(ValueError, match="unsupported type"):
        device_function_kernel("", "double f(double** x)")
    with pytest.raises(ValueError, match="'n' of 'f' clashes"):
        device_function_kernel("", "double f(const double* x, int n)")
    with pytest.raises(ValueError, match="'out' of 'f' clashes"):
        device_function_kernel("", "double f(double out)")
    # the clash is resolved with another generated name
    kernel = device_function_kernel(
        "", "double f(const double* x, int n)", n_threads_param="size"
    )
    assert (
        "f_kernel(const double* x, const int* n, double* out, int size)"
        in kernel.source
    )


@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")
def test_device_function_kernel_on_gpu():
    import cupy as cp

    sq = device_function_kernel(
        "__device__ double sq(double x) { return x * x; }", "double sq(double x)"
    )
    x = cp.arange(1000, dtype=cp.float64)
    out = cp.empty(1000)
    sq(x, out, 1000, n_threads=1000)
    assert cp.allclose(out, x * x)

    find_span = device_function_kernel(
        FIND_SPAN, "int find_span(const double* t, int p, double eta)"
    )
    t = cp.asarray([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    eta = cp.asarray([0.1, 0.6, 0.9])
    p = cp.full(3, 2, dtype=cp.int32)
    spans = cp.empty(3, dtype=cp.int32)
    find_span(t, p, eta, spans, 3, n_threads=3)
    assert spans.get().tolist() == [2, 3, 3]


def test_struct_arguments_are_compared_by_field_name():
    """A CudaStructArguments object (fields may be properties) gets the host names."""
    from cunumpy.testing import _collect_arrays

    class Owner:
        def __init__(self):
            self.markers = np.zeros((3, 4))
            self.weights = np.ones(3)

    class HostArguments:  # e.g. a Pyccel class
        def __init__(self, owner):
            self.markers = owner.markers
            self.weights = owner.weights
            self.n = 3

    class DeviceArguments(xp.CudaStructArguments):
        struct_name = "OwnerArgs"
        fields = (("markers", "Array2D<double>"), ("weights", "double*"), ("n", "int"))

        def __init__(self, owner):
            self._owner = owner  # not packed: no device arrays in this test

        @property
        def markers(self):
            return self._owner.markers

        @property
        def weights(self):
            return self._owner.weights

        n = 3

    owner = Owner()
    host = _collect_arrays((1.0, HostArguments(owner)))
    device = _collect_arrays((1.0, DeviceArguments(owner)))
    assert (
        sorted(host) == sorted(device) == ["argument 1.markers", "argument 1.weights"]
    )
    assert device["argument 1.markers"] is owner.markers

    struct = DeviceArguments.struct
    value = xp.CudaStructValue(
        struct, np.zeros((), struct.dtype)[()], vars(owner) | {"n": 3}
    )
    assert sorted(_collect_arrays((value,))) == [
        "argument 0.markers",
        "argument 0.weights",
    ]
