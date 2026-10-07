"""Kernels chosen by where the arguments live, compiled host kernels, signature checks."""

import importlib
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _dispatch as dispatch_module
from cunumpy.arguments import CudaArguments
from cunumpy.kernels import (
    CompiledHostKernel,
    CudaKernel,
    HostImplementations,
    Kernel,
    KernelCatalog,
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
        dispatch_module,
        "_is_device_array",
        lambda a: isinstance(a, FakeDeviceArray),
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


@pytest.mark.parametrize("dispatch", ["backend", "arrays"])
def test_explicit_device_implementation_dispatch(fake_gpu, dispatch):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch=dispatch)
    missing = Kernel(scale, dispatch=dispatch, missing_cuda="fallback")
    x = FakeDeviceArray(4)
    with xp.kernels.use_device_kernel_implementation("cuda"):
        assert kernel.selected(device=True) == "cuda"
        kernel(x, 3.0, 4)
        assert [name for name, *_ in fake_gpu] == ["scale"]
        with pytest.raises(LookupError, match="has no 'cuda' implementation"):
            missing(x, 3.0, 4)
        with pytest.raises(LookupError, match="has no 'cuda' implementation"):
            missing.selected(device=True)
    with (
        xp.kernels.use_device_kernel_implementation(None),
        pytest.warns(RuntimeWarning, match="No CUDA version"),
    ):
        assert missing.selected(device=True) == "host"


def test_device_implementation_does_not_change_host_selection(fake_gpu):
    with (
        xp.use_backend("numpy"),
        xp.kernels.use_host_kernel_implementation("python"),
        xp.kernels.use_device_kernel_implementation("cuda"),
    ):
        kernel = Kernel(scale, dispatch="arrays")
        x = np.ones(4)
        kernel(x, 3.0, 4)
        assert x.tolist() == [3.0] * 4 and fake_gpu == []
        assert xp.get_backend() == "numpy"
        assert xp.kernels.get_host_kernel_implementation() == "python"


def test_arrays_dispatch_runs_device_arrays_on_the_gpu(fake_gpu):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    x = FakeDeviceArray(4)
    kernel(x, 3.0, 4, n_threads=4)
    ((name, args, options),) = fake_gpu
    assert name == "scale" and args[0] is x and options["n_threads"] == 4
    kernel(x, 3.0, 4)
    assert fake_gpu[-1][2]["n_threads"] is None
    kernel.cuda_kernel.n_threads_from = None
    with pytest.raises(ValueError, match="n_threads is required"):
        kernel(x, 3.0, 4)


def test_device_only_argument_objects_count_as_device(fake_gpu, monkeypatch):
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")

    class Device(CudaArguments):
        pass

    kernel(Device(np.ones(1)), 2.0, 1, n_threads=1)
    assert len(fake_gpu) == 1

    class HostArgs:  # e.g. a pyccel argument class: no __cuda_args__
        def __init__(self):
            self.x = np.ones(2)

    def fill(args, value, n):
        args.x[:n] = value

    host_args = HostArgs()
    host = Kernel(fill, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    host(host_args, 5.0, 2)  # no device argument: the host kernel
    assert len(fake_gpu) == 1 and host_args.x.tolist() == [5.0, 5.0]


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
            reordered,
            CudaKernel(SCALE_CUDA, "scale"),
            name="scale",
        ).check_signature()

    # nothing to compare: no CUDA kernel, an unparsed signature, a compiled builtin
    Kernel(renamed).check_signature()
    Kernel(
        renamed,
        CudaKernel(SCALE_CUDA, "scale", check_signature=False),
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
        module,
        "double",
        failing,
        fallback=fallback_calls.append,
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
        "        x[i] *= factor\n",
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
    assert isinstance(kernel.host_kernel.kernel, HostImplementations)
    assert kernel.implementations == ("pyccel", "python", "cuda")
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


# ---------------------------------------------------------------------------
# one kernel folder, declared in its own __init__.py
# ---------------------------------------------------------------------------


@pytest.fixture
def self_declaring_package(tmp_path, monkeypatch):
    """`scale/__init__.py` declares its kernel with Kernel.from_folder.

    The folder has a pyccel, a NumPy, a numba (whose import fails) and a CUDA
    version of `scale`.
    """
    root = tmp_path / "demo_folder_pkg"
    (root / "scale").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "scale" / "scale_pyccel.py").write_text(
        "def scale(x: 'float[:]', factor: float, n: int):\n"
        "    for i in range(n):\n"
        "        x[i] *= factor\n",
    )
    (root / "scale" / "scale_numpy.py").write_text(
        "CALLS = []\n\n"
        "def scale(x, factor, n):\n"
        "    CALLS.append(n)\n"
        "    x[:n] *= factor\n",
    )
    (root / "scale" / "scale_numba.py").write_text(
        "import a_jit_library_that_is_not_installed\n",
    )
    (root / "scale" / "scale_cuda.cu").write_text(SCALE_CUDA)
    (root / "scale" / "__init__.py").write_text(
        "import cunumpy as xp\n\n"
        "COMPILED = []\n\n"
        "def _compile(module):\n"
        "    COMPILED.append(module.__name__)\n"
        "    return module\n\n"
        "kernel = xp.kernels.Kernel.from_folder(\n"
        "    __name__, host_suffix='_pyccel', dispatch='arrays',\n"
        "    compile_host=_compile, n_threads_from='first_array',\n"
        ")\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("demo_folder_pkg.scale")
    for module in [m for m in sys.modules if m.startswith("demo_folder_pkg")]:
        del sys.modules[module]


def test_from_folder_finds_every_implementation(self_declaring_package):
    folder = self_declaring_package
    kernel = folder.kernel
    assert kernel.name == "scale" and kernel.dispatch == "arrays"
    assert kernel.implementations == ("pyccel", "numba", "numpy", "python", "cuda")
    assert kernel.cuda_kernel.n_threads_from is not None
    kernel.check_signature()  # pyccel, NumPy and CUDA: x, factor, n; numba skipped
    assert kernel.selected() == "pyccel"  # the compiled version by default
    assert kernel.selected(device=True) == "cuda"
    x = np.ones(3)
    kernel(x, 2.0, 3)
    assert x.tolist() == [2.0] * 3
    assert folder.COMPILED == ["demo_folder_pkg.scale.scale_pyccel"]
    # the catalog of the parent package builds the same kernel
    catalog = KernelCatalog.from_package("demo_folder_pkg", host_suffix="_pyccel")
    assert catalog["scale"].implementations == kernel.implementations


def test_from_folder_needs_a_kernel_folder(self_declaring_package):
    with pytest.raises(FileNotFoundError, match="no host kernel scale_kernels.py"):
        Kernel.from_folder("demo_folder_pkg.scale")  # default suffix
    with pytest.raises(ModuleNotFoundError, match="not a package"):
        Kernel.from_folder("demo_folder_pkg.scale.scale_numpy")


def test_kernel_implementation_setting(self_declaring_package):
    kernel = self_declaring_package.kernel
    calls = importlib.import_module("demo_folder_pkg.scale.scale_numpy").CALLS
    assert xp.kernels.get_host_kernel_implementation() is None
    with xp.kernels.use_host_kernel_implementation("numpy"):
        assert xp.kernels.get_host_kernel_implementation() == "numpy"
        assert kernel.selected() == "numpy"
        x = np.ones(2)
        kernel(x, 3.0, 2)
        assert x.tolist() == [3.0, 3.0] and calls == [2]
        with xp.kernels.use_host_kernel_implementation("python"):
            kernel(x, 2.0, 2)  # the uncompiled pyccel source
        assert calls == [2] and x.tolist() == [6.0, 6.0]
    assert xp.kernels.get_host_kernel_implementation() is None
    assert self_declaring_package.COMPILED == []  # pyccel never needed
    # a chosen implementation that cannot run raises instead of running another
    xp.kernels.set_host_kernel_implementation("numba")
    try:
        with pytest.raises(LookupError, match="'numba' implementation .* unavailable"):
            kernel(np.ones(1), 2.0, 1)
    finally:
        xp.kernels.set_host_kernel_implementation(None)
    with pytest.raises(ValueError, match="kernel implementation must be one of"):
        xp.kernels.set_host_kernel_implementation("fortran")


def test_default_skips_unavailable_implementations():
    def no_pyccel():
        raise ImportError("pyccel missing")

    used = []
    with_numpy = HostImplementations(
        "scale",
        {
            "pyccel": no_pyccel,
            "numpy": lambda: used.append,
            "python": lambda: scale,
        },
    )
    assert with_numpy.selected() == "numpy" and not with_numpy.available("pyccel")
    assert isinstance(with_numpy.errors["pyccel"], ImportError)
    with_numpy("x")
    assert used == ["x"]
    only_python = HostImplementations(
        "scale",
        {"pyccel": no_pyccel, "python": lambda: scale},
    )
    x = np.ones(2)
    with pytest.warns(RuntimeWarning, match="uncompiled Python version"):
        only_python(x, 2.0, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        only_python(x, 2.0, 2)  # warned once
    assert x.tolist() == [4.0, 4.0]
    with pytest.raises(LookupError, match="has no 'numba' implementation"):
        only_python.get("numba")
    with pytest.raises(ValueError, match="unknown implementations"):
        HostImplementations("scale", {"python": lambda: scale, "julia": lambda: scale})


@pytest.mark.parametrize(
    "implementation,legacy,expected",
    [
        (None, None, None),
        ("", None, None),
        ("pyccel", None, "pyccel"),
        ("numba", None, "numba"),
        (" NuMpY ", None, "numpy"),
        ("python", None, "python"),
        (None, "numpy", None),
        ("numpy", "fortran", "numpy"),
    ],
)
def test_host_kernel_implementation_environment_variable(
    implementation, legacy, expected
):
    import os
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(xp.__file__).parents[1])
    env.pop("CUNUMPY_HOST_KERNEL_IMPLEMENTATION", None)
    env.pop("CUNUMPY_KERNEL_IMPLEMENTATION", None)
    if implementation is not None:
        env["CUNUMPY_HOST_KERNEL_IMPLEMENTATION"] = implementation
    if legacy is not None:
        env["CUNUMPY_KERNEL_IMPLEMENTATION"] = legacy
    code = (
        "import os, cunumpy as xp\n"
        f"assert xp.kernels.get_host_kernel_implementation() == {expected!r}\n"
        "os.environ['CUNUMPY_HOST_KERNEL_IMPLEMENTATION'] = 'python'\n"
        f"assert xp.kernels.get_host_kernel_implementation() == {expected!r}\n"
        "with xp.kernels.use_host_kernel_implementation('numpy'):\n"
        "    assert xp.kernels.get_host_kernel_implementation() == 'numpy'\n"
        f"assert xp.kernels.get_host_kernel_implementation() == {expected!r}\n"
        "xp.kernels.set_host_kernel_implementation(None)\n"
        "assert xp.kernels.get_host_kernel_implementation() is None\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_invalid_host_kernel_implementation_environment_variable():
    import os
    import subprocess

    code = "import cunumpy as xp; print(xp.kernels.get_host_kernel_implementation())"
    env = {**os.environ, "CUNUMPY_HOST_KERNEL_IMPLEMENTATION": "fortran"}
    env["PYTHONPATH"] = str(Path(xp.__file__).parents[1])
    failed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0 and "kernel implementation must be" in failed.stderr


def test_arrays_dispatch_calls_the_host_kernel_without_conversion(
    fake_gpu,
    monkeypatch,
):
    # CuPy is active, but host arguments never go through PyccelKernel's
    # device-to-host conversion: the choice already says they are host arrays
    def no_conversion(self, args, kwargs):
        raise AssertionError("conversion checked")

    monkeypatch.setattr(xp.kernels.PyccelKernel, "_needs_conversion", no_conversion)
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"), dispatch="arrays")
    x = np.ones(2)
    kernel(x, 2.0, 2)
    assert x.tolist() == [2.0, 2.0]
    # a host kernel that is told to convert keeps doing so
    forced = Kernel(
        xp.kernels.PyccelKernel(scale, use_cupy=True),
        CudaKernel(SCALE_CUDA, "scale"),
        dispatch="arrays",
    )
    with pytest.raises(AssertionError, match="conversion checked"):
        forced(x, 2.0, 2)


def test_check_signature_covers_every_host_implementation():
    def swapped(x, n, factor):
        pass

    kernel = Kernel(
        HostImplementations(
            "scale",
            {"python": lambda: scale, "numpy": lambda: swapped},
        ),
        CudaKernel(SCALE_CUDA, "scale"),
    )
    with pytest.raises(ValueError, match=r"the numpy version \(x, n, factor\)"):
        kernel.check_signature()
    module = _module("m", "def scale(x, factor, n):\n    x *= factor\n")
    compiled = Kernel(CompiledHostKernel(module, "scale", lambda m: m, swapped))
    with pytest.raises(ValueError, match=r"its fallback \(x, n, factor\)"):
        compiled.check_signature()


# ---------------------------------------------------------------------------
# kernel arguments on the side of the arrays
# ---------------------------------------------------------------------------


def test_as_kernel_array_on_the_host():
    grid = np.zeros((4, 3))
    fits = np.ones(5)
    assert xp.kernels.as_kernel_array(fits, like=grid, dtype=float) is fits  # no copy
    column = np.ones((5, 2))[:, 1]
    converted = xp.kernels.as_kernel_array(column, like=grid, dtype=float)
    assert converted.flags.c_contiguous and converted is not column
    ints = xp.kernels.as_kernel_array([1, 2], like=grid, dtype=float)
    assert ints.dtype == np.float64 and isinstance(ints, np.ndarray)


def test_kernel_output_writes_into_its_target():
    grid = np.zeros(3)
    out = np.zeros(4)
    with xp.kernels.kernel_output(out, like=grid, dtype=float) as buffer:
        assert buffer is out  # written directly
        buffer += 1.0
    strided = np.zeros((4, 2))[:, 0]
    with xp.kernels.kernel_output(strided, like=grid, dtype=float) as buffer:
        assert buffer is not strided
        buffer[...] = 7.0
    assert strided.tolist() == [7.0] * 4
    with (
        pytest.raises(RuntimeError),
        xp.kernels.kernel_output(strided, like=grid) as buffer,
    ):
        buffer[...] = 1.0
        raise RuntimeError("kernel failed")
    assert strided.tolist() == [7.0] * 4  # not copied back after an error


def test_kernel_arrays_follow_a_device_grid():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    grid = cp.zeros(3)
    on_device = xp.kernels.as_kernel_array(np.ones(4), like=grid, dtype=float)
    assert xp.is_gpu(on_device)
    host_out = np.zeros(4)
    with xp.kernels.kernel_output(host_out, like=grid, dtype=float) as buffer:
        assert xp.is_gpu(buffer)
        buffer[...] = 2.0
    assert host_out.tolist() == [2.0] * 4
