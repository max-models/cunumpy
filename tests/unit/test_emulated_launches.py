"""Struct parameters in the emulation, and `emulated_launches` on the fake CuPy."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cunumpy.arguments import CudaStruct
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from cunumpy.kernels import CudaKernel

pytestmark = pytest.mark.skipif(
    emulation_compiler() is None,
    reason="no C++ compiler for the emulation",
)

PARTICLES = CudaStruct(
    "Particles",
    [("markers", "Array2D<double>"), ("alive", "bool*"), ("n", "int")],
)
SOURCE = (
    '#include "cunumpy/array_view.cuh"\n'
    + PARTICLES.declaration
    + r"""
extern "C" __global__
void push(Particles p, double dt, double* total) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < p.n && p.alive[i]) {
        p.markers(i, 0) += dt * p.markers(i, 1);
        total[0] += p.markers(i, 0);
    }
}
"""
)


def particles():
    markers = np.arange(12.0).reshape(4, 3)
    return markers, np.array([True, True, False, True])


def expected(markers, alive, dt):
    out = markers.copy()
    out[alive, 0] += dt * out[alive, 1]
    return out


@pytest.mark.parametrize("kind", ["mapping", "object"])
def test_struct_given_as_mapping_or_object(kind):
    markers, alive = particles()
    want = expected(markers, alive, 0.5)
    total = np.zeros(1)
    fields = {"markers": markers, "alive": alive, "n": 4}
    value = fields if kind == "mapping" else type("Args", (), fields)()
    push = CudaKernel(SOURCE, "push", structs=[PARTICLES])
    emulate_cuda_kernel(push, value, 0.5, total, n_threads=4)
    np.testing.assert_allclose(markers, want)
    assert total[0] == pytest.approx(want[alive, 0].sum())


def test_struct_launch_shape_is_inferred_from_the_first_array():
    markers, alive = particles()
    push = CudaKernel(SOURCE, "push", structs=[PARTICLES])
    emulate_cuda_kernel(
        push, {"markers": markers, "alive": alive, "n": 4}, 1.0, np.zeros(1)
    )
    np.testing.assert_allclose(markers, expected(*particles(), 1.0))


def test_struct_with_a_missing_field_is_a_type_error():
    markers, alive = particles()
    push = CudaKernel(SOURCE, "push", structs=[PARTICLES])
    with pytest.raises(TypeError, match="no value for the field 'n'"):
        emulate_cuda_kernel(
            push, {"markers": markers, "alive": alive}, 1.0, np.zeros(1), n_threads=4
        )


SCRIPT = r"""
import numpy as np
import cunumpy as xp
from cunumpy.arguments import CudaStruct, CudaStructArguments
from cunumpy.kernel_testing import emulated_launches, host_buffer
from cunumpy.kernels import CudaKernel, Kernel

Particles = CudaStruct("Particles", [("x", "double*"), ("n", "int")])
source = Particles.declaration + '''
extern "C" __global__ void scale(Particles p, double f) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < p.n) p.x[i] *= f;
}'''
scale = CudaKernel(source, "scale", structs=[Particles])


class Args(CudaStructArguments):
    struct_name = "Particles"
    fields = (("x", "double*"), ("n", "int"))

    def __init__(self, x):
        self.x = x
        self.n = x.shape[0]
        self.pack()


x = xp.asarray(np.arange(5.0))
assert xp.get_backend() == "cupy"
try:
    scale(Particles(x=x, n=5), 2.0, n_threads=5)
except NotImplementedError:
    pass
else:
    raise AssertionError("the fake CuPy must refuse a launch outside the block")

with emulated_launches():
    scale(Particles(x=x, n=5), 2.0, n_threads=5)
    scale(Args(x), 3.0, n_threads=5)
    host_view = host_buffer(x)
np.testing.assert_allclose(host_buffer(x), np.arange(5.0) * 6.0)
assert host_view is host_buffer(x)  # not a copy

# a Kernel dispatches to the CUDA kernel on the CuPy backend
def host_scale(args, f):
    raise AssertionError("the CUDA kernel must run")

kernel = Kernel(host_scale, scale)
with emulated_launches():
    kernel(Args(x), 0.5, n_threads=5)
np.testing.assert_allclose(host_buffer(x), np.arange(5.0) * 3.0)

try:
    host_buffer(np.zeros(2))
except TypeError:
    pass
else:
    raise AssertionError("host_buffer must refuse a NumPy array")
print("emulated launches OK")
"""


def test_emulated_launches_on_the_fake_cupy():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUNUMPY_FAKE_CUPY="1", CUNUMPY_BACKEND="cupy")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), env.get("PYTHONPATH", "")) if p
    )
    env.pop("CUNUMPY_CUDA_DEBUG", None)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "emulated launches OK" in result.stdout


COMPILE_SCRIPT = r"""
import numpy as np
import cunumpy as xp
from cunumpy.kernel_testing import emulated_launches, host_buffer
from cunumpy.kernels import CudaKernel, CudaKernelVariants, Kernel, KernelCatalog

original = CudaKernel.compile
source = '''
template <int P>
__global__ void power(double* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { double v = x[i]; for (int k = 1; k < P; ++k) x[i] *= v; }
}'''
variants = CudaKernelVariants(lambda p: CudaKernel(source, "power", template_args=[p]))
scale = CudaKernel('''
extern "C" __global__ void scale(double* x, double f, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= f;
}''', "scale")
catalog = KernelCatalog({"scale": Kernel(lambda x, f, n: None, scale)})

with emulated_launches():
    assert scale.compile() is None
    assert scale.recompile() is None
    variants.compile_all([(2,), (3,)], jobs=2)
    assert catalog.compile_all() == ["scale"]
    x = xp.asarray(np.arange(4.0))
    scale(x, 2.0, 4, n_threads=4)
    variants.get(2)(x, 4, n_threads=4)
np.testing.assert_allclose(host_buffer(x), (2.0 * np.arange(4.0)) ** 2)
assert CudaKernel.compile is original

# a compile error shows up in compile(), without a launch
broken = CudaKernel('extern "C" __global__ void k(int n) { undefined_call(n); }', "k")
try:
    with emulated_launches():
        broken.compile()
except RuntimeError as error:
    assert "does not compile" in str(error), error
else:
    raise AssertionError("compile() must report the compile error")
assert CudaKernel.compile is original

# kernels the emulation cannot run are skipped by compile()
warp = CudaKernel('''extern "C" __global__ void w(double* x) {
    x[0] = __shfl_down_sync(0xffffffff, x[0], 1); }''', "w")
with emulated_launches():
    warp.compile()

# outside the block, the fake CuPy cannot compile
try:
    scale.compile()
except (RuntimeError, NotImplementedError):
    pass
else:
    raise AssertionError("compile() outside the block must not emulate")
print("emulated compile OK")
"""


def _run_on_fake_cupy(script):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUNUMPY_FAKE_CUPY="1", CUNUMPY_BACKEND="cupy")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), env.get("PYTHONPATH", "")) if p
    )
    env.pop("CUNUMPY_CUDA_DEBUG", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )


def test_compile_inside_emulated_launches_on_the_fake_cupy():
    result = _run_on_fake_cupy(COMPILE_SCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "emulated compile OK" in result.stdout


def test_compile_and_call_are_restored_after_an_exception():
    from cunumpy.kernel_testing import emulated_launches

    call, compile_ = CudaKernel.__call__, CudaKernel.compile
    with pytest.raises(KeyError), emulated_launches():
        assert CudaKernel.compile is not compile_
        raise KeyError("boom")
    assert CudaKernel.compile is compile_
    assert CudaKernel.__call__ is call
