"""Kernels chosen by where the arguments live, compiled host kernels, signature checks."""

import sys
import textwrap
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import (
    CompiledHostKernel,
    CudaArguments,
    CudaKernel,
    Kernel,
    KernelArguments,
    KernelCatalog,
)
from cunumpy import dispatch as dispatch_module

SCALE_CUDA = r"""
extern "C" __global__ void scale(double* x, double factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""


def scale(x, factor, n):
    for i in range(n):
        x[i] *= factor


class FakeDeviceArray:
    """Stands for a CuPy array (see the `fake_gpu` fixture)."""

    def __init__(self, n):
        self.shape = (n,)


@pytest.fixture
def fake_gpu(monkeypatch):
    """CuPy is the active backend, FakeDeviceArray counts as a device array, and
    CUDA launches are recorded instead of run."""
    monkeypatch.setattr(dispatch_module, "get_backend", lambda: "cupy")
    monkeypatch.setattr(
        dispatch_module, "_is_device_array", lambda a: isinstance(a, FakeDeviceArray)
    )
    launches = []

    def launch(self, *args, **kwargs):
        launches.append((self.name, args, kwargs))

    monkeypatch.setattr(CudaKernel, "__call__", launch)
    return launches


def test_dispatch_must_be_known():
    with pytest.raises(ValueError, match="dispatch must be one of"):
        Kernel(scale, dispatch="device")
    assert Kernel(scale).dispatch == "backend"


def test_arrays_dispatch_runs_host_arrays_on_the_host_while_cupy_is_active(fake_gpu):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    x = np.ones(4)
    kernel(x, 3.0, 4)  # NumPy arrays: the host kernel, although CuPy is active
    assert x.tolist() == [3.0] * 4 and fake_gpu == []


def test_backend_dispatch_sends_everything_to_cuda_on_cupy(fake_gpu):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    x = np.ones(4)
    kernel(x, 3.0, 4, n_threads=4)  # the old rule: the backend decides
    assert [name for name, *_ in fake_gpu] == ["scale"]
    assert x.tolist() == [1.0] * 4


def test_arrays_dispatch_runs_device_arrays_on_the_gpu(fake_gpu):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    x = FakeDeviceArray(4)
    kernel(x, 3.0, 4, n_threads=4)
    ((name, args, options),) = fake_gpu
    assert name == "scale" and args[0] is x and options["n_threads"] == 4
    with pytest.raises(ValueError, match="n_threads is required"):
        kernel(x, 3.0, 4)


def test_device_only_argument_objects_count_as_device(fake_gpu, monkeypatch):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")

    class Device(CudaArguments):
        pass

    kernel(Device(np.ones(1)), 2.0, 1, n_threads=1)
    assert len(fake_gpu) == 1

    class Both(KernelArguments):  # host and device form: does not decide
        def __host_args__(self):
            return np.ones(2)

    host = Kernel(lambda x, f, n: x.__setitem__(slice(None), f), dispatch="arrays")
    host(Both(), 5.0, 2)  # no device argument: the host kernel


def test_arrays_dispatch_without_cuda_kernel(fake_gpu):
    raising = Kernel(scale, dispatch="arrays")
    with pytest.raises(NotImplementedError, match="No CUDA version"):
        raising(FakeDeviceArray(2), 2.0, 2, n_threads=2)
    x = np.ones(2)
    raising(x, 2.0, 2)  # host arrays never need the CUDA kernel
    assert x.tolist() == [2.0, 2.0]


class CompiledFunction:
    """Like a compiled extension function: no Python signature to read."""

    __signature__ = "unreadable"  # inspect.signature raises TypeError

    def __call__(self, *args):
        return None


def test_check_signature():
    Kernel(scale, CudaKernel(SCALE_CUDA, "scale")).check_signature()  # same names

    def renamed(x, a, n): ...

    with pytest.raises(
        ValueError,
        match=r"host kernel takes \(x, a, n\), the CUDA kernel \(x, factor, n\)",
    ):
        Kernel(renamed, CudaKernel(SCALE_CUDA, "scale")).check_signature()

    def reordered(factor, x, n): ...

    with pytest.raises(ValueError, match="host kernel takes"):
        Kernel(
            reordered, CudaKernel(SCALE_CUDA, "scale"), name="scale"
        ).check_signature()

    # nothing to compare: no CUDA kernel, an unparsed signature, a compiled builtin
    Kernel(renamed).check_signature()
    Kernel(
        renamed, CudaKernel(SCALE_CUDA, "scale", check_signature=False)
    ).check_signature()
    compiled = CompiledFunction()
    Kernel(compiled, CudaKernel(SCALE_CUDA, "scale"), name="scale").check_signature()
    assert Kernel(compiled, name="m").host_parameters() is None


def test_catalog_check_signatures_lists_every_mismatch():
    def renamed(x, a, n): ...

    catalog = KernelCatalog()
    catalog.register(Kernel(scale, CudaKernel(SCALE_CUDA, "scale")))
    catalog.register(Kernel(renamed, CudaKernel(SCALE_CUDA, "scale"), name="one"))
    catalog.register(Kernel(renamed, CudaKernel(SCALE_CUDA, "scale"), name="two"))
    with pytest.raises(ValueError, match=r"(?s)'one'.*'two'"):
        catalog.check_signatures()


# ---------------------------------------------------------------------------
# compiled host kernels
# ---------------------------------------------------------------------------


def _module(name, source):
    module = SimpleNamespace(__name__=name)
    exec(textwrap.dedent(source), module.__dict__)  # noqa: S102
    return module


def test_compiled_host_kernel_compiles_once():
    module = _module("m", "def double(x):\n    x *= 2\n")
    compiled = []

    def compiler(mod):
        compiled.append(mod)
        return SimpleNamespace(double=lambda x: x.__setitem__(..., x * 10))

    kernel = CompiledHostKernel(module, "double", compiler)
    assert repr(kernel) == "CompiledHostKernel('double', not built)"
    x = np.ones(2)
    kernel(x)
    kernel(x)
    assert x.tolist() == [100.0, 100.0] and compiled == [module]
    assert kernel.compiled and kernel.error is None
    assert kernel.python is module.double


def test_compiled_host_kernel_falls_back():
    module = _module("m", "def double(x):\n    x *= 2\n")

    def failing(mod):
        raise ImportError("no pyccel")

    fallback_calls = []
    with_fallback = CompiledHostKernel(
        module, "double", failing, fallback=fallback_calls.append
    )
    with_fallback("x")
    assert fallback_calls == ["x"] and not with_fallback.compiled
    assert isinstance(with_fallback.error, ImportError)

    python = CompiledHostKernel(module, "double", failing)
    x = np.ones(2)
    with pytest.warns(RuntimeWarning, match="uncompiled Python version"):
        python(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python(x)  # warned once
    assert x.tolist() == [4.0, 4.0]


@pytest.fixture
def pyccel_style_package(tmp_path, monkeypatch):
    root = tmp_path / "demo_pyccel_pkg"
    (root / "scale").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "scale" / "__init__.py").write_text("")
    (root / "scale" / "scale_pyccel.py").write_text(
        "def scale(x: 'float[:]', factor: float, n: int):\n"
        "    for i in range(n):\n"
        "        x[i] *= factor\n"
    )
    (root / "scale" / "scale_cuda.cu").write_text(SCALE_CUDA)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "demo_pyccel_pkg"
    for module in [m for m in sys.modules if m.startswith("demo_pyccel_pkg")]:
        del sys.modules[module]


def test_from_package_with_compiled_hosts_and_array_dispatch(pyccel_style_package):
    calls = []

    def compiler(module):
        calls.append(module.__name__)
        return module  # "compiled": the Python module itself

    catalog = KernelCatalog.from_package(
        pyccel_style_package,
        host_suffix="_pyccel",
        dispatch="arrays",
        compile_host=compiler,
    )
    kernel = catalog["scale"]
    assert kernel.dispatch == "arrays"
    assert isinstance(kernel.host_kernel.kernel, CompiledHostKernel)
    catalog.check_signatures()  # x, factor, n on both sides
    x = np.ones(3)
    kernel(x, 2.0, 3)
    assert x.tolist() == [2.0] * 3 and calls == ["demo_pyccel_pkg.scale.scale_pyccel"]


def test_from_package_host_fallback(pyccel_style_package):
    def failing(module):
        raise RuntimeError("compiler missing")

    used = []
    catalog = KernelCatalog.from_package(
        pyccel_style_package,
        host_suffix="_pyccel",
        compile_host=failing,
        host_fallback=lambda name: lambda *args: used.append((name, args)),
    )
    catalog["scale"](np.ones(1), 2.0, 1)
    assert used and used[0][0] == "scale"
    mapping = KernelCatalog.from_package(
        pyccel_style_package,
        host_suffix="_pyccel",
        compile_host=failing,
        host_fallback={"scale": lambda *args: used.append("mapped")},
    )
    mapping["scale"](np.ones(1), 2.0, 1)
    assert used[-1] == "mapped"


def test_arrays_dispatch_on_gpu():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    with xp.use_backend("cupy"):
        host = np.ones(5)
        kernel(host, 2.0, 5)  # host array: the host kernel, no transfer
        device = cp.ones(5)
        kernel(device, 3.0, 5, n_threads=5)
    assert host.tolist() == [2.0] * 5
    assert device.get().tolist() == [3.0] * 5
