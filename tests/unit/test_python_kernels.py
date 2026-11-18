"""CUDA kernels written in Python (cupyx.jit, numba-cuda) and their dispatch."""

import importlib
import importlib.util
import os
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _dispatch as dispatch_module
from cunumpy.arguments import CudaArguments, CudaStruct
from cunumpy.kernel_testing import (
    assert_kernels_agree,
    fake_cupy_active,
    numba_simulator_active,
    requires_cupy,
    run_in_fake_cupy_subprocess,
)
from cunumpy.kernels import (
    CudaKernel,
    CupyJitKernel,
    Kernel,
    KernelCatalog,
    NumbaCudaKernel,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")

SCALE_CUDA = r"""
extern "C" __global__ void scale(double* x, double factor, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = factor * x[i];
}
"""


def scale(x, factor, out):
    out[:] = factor * x


class DeviceArray:
    """Stands for a CuPy array: has ``__cuda_array_interface__`` and a shape."""

    def __init__(self, n):
        self.shape = (n,)
        self.dtype = np.dtype(float)
        self.__cuda_array_interface__ = {"shape": self.shape}


class Recorder:
    """Records launches of Python device kernels instead of running them."""

    def __init__(self):
        self.launches = []

    def __call__(self, kernel, jitted, grid, block, values, shared_mem, stream):
        self.launches.append((kernel.implementation, grid, block, values, shared_mem))


@pytest.fixture
def launches(monkeypatch):
    """The CuPy backend is active, every launch is recorded, the JITs are no-ops."""
    recorder = Recorder()
    monkeypatch.setattr(dispatch_module, "get_backend", lambda: "cupy")
    monkeypatch.setattr(
        dispatch_module, "_is_device_array", lambda a: isinstance(a, DeviceArray)
    )
    for cls in (CupyJitKernel, NumbaCudaKernel):
        monkeypatch.setattr(cls, "_jit", lambda self, function: function)
        monkeypatch.setattr(
            cls,
            "_launch",
            lambda self, *rest: recorder(self, *rest),
        )

    def cuda_launch(self, *args, **kwargs):
        recorder.launches.append(("cuda_c", args, kwargs))

    monkeypatch.setattr(CudaKernel, "__call__", cuda_launch)
    return recorder.launches


def failing_loader():
    raise ImportError("No module named 'cupyx'")


# -- the kernel classes -------------------------------------------------------


def test_construction_checks():
    with pytest.raises(TypeError, match="exactly one of function and loader"):
        CupyJitKernel()
    with pytest.raises(TypeError, match="exactly one of function and loader"):
        CupyJitKernel(scale, loader=lambda: scale)
    with pytest.raises(TypeError, match="name is required"):
        NumbaCudaKernel(loader=lambda: scale)
    with pytest.raises(TypeError, match="must be callable"):
        CupyJitKernel(42)
    with pytest.raises(ValueError, match="at most 1024"):
        CupyJitKernel(scale, block_size=2048)


def test_name_signature_and_repr():
    kernel = CupyJitKernel(scale, block_size=(16, 16))
    assert kernel.name == kernel.expression == "scale"
    assert kernel.block_size == (16, 16)
    assert kernel.parameters() == ["x", "factor", "out"]
    assert repr(kernel) == "CupyJitKernel(name='scale', block_size=(16, 16))"
    assert kernel.launch_shape((100, 50)) == ((7, 4), (16, 16))
    assert CupyJitKernel(lambda *args: None, name="v").parameters() is None
    assert NumbaCudaKernel(max).parameters() is None  # no Python signature


def test_the_python_function_of_jit_kernels():
    call = {"__call__": lambda self, *args: None}
    jitted = type("_JitRawKernel", (), {"__module__": "cupyx.jit._interface", **call})()
    jitted._func = scale
    kernel = CupyJitKernel(jitted)
    assert kernel.name == "scale" and kernel.python_function is scale
    assert kernel.function is jitted and kernel.compile() is jitted  # already jitted

    dispatcher = type(
        "CUDADispatcher", (), {"__module__": "numba.cuda.dispatcher", **call}
    )()
    dispatcher.py_func = scale
    kernel = NumbaCudaKernel(dispatcher)
    assert kernel.name == "scale" and kernel.parameters() == ["x", "factor", "out"]
    assert kernel.compile() is dispatcher


def test_loading_is_lazy_and_failures_are_kept():
    calls = []

    def loader():
        calls.append(1)
        return failing_loader()

    kernel = CupyJitKernel(name="scale", loader=loader)
    assert calls == [] and not kernel.is_compiled
    assert not kernel.available()
    assert isinstance(kernel.error, ImportError)
    with pytest.raises(RuntimeError, match=r"cannot be loaded \(ImportError.*CuPy"):
        kernel.compile()
    with pytest.raises(ImportError):
        kernel.parameters()
    assert calls == [1]  # tried once

    bad = NumbaCudaKernel(name="scale", loader=lambda: 3)
    assert not bad.available()
    assert "returned int, not a function" in str(bad.error)
    with pytest.raises(RuntimeError, match="numba-cuda"):
        bad(DeviceArray(4), 2.0, DeviceArray(4))

    good = NumbaCudaKernel(name="scale", loader=lambda: scale)
    assert good.error is None and good.function is scale


def test_arguments_are_flattened_and_checked():
    kernel = CupyJitKernel(scale)
    x, out = DeviceArray(4), DeviceArray(4)
    assert kernel.prepare_args(x, 2.0, out) == (x, 2.0, out)
    assert kernel.prepare_args(CudaArguments(x, CudaArguments(2.0)), out) == (
        x,
        2.0,
        out,
    )
    with pytest.raises(TypeError, match="argument 0 is a NumPy array"):
        kernel.prepare_args(np.ones(4), 2.0, out)
    with pytest.raises(TypeError, match=r"argument 0\[1\] is a NumPy array"):
        kernel.prepare_args(CudaArguments(x, np.ones(1)), out)
    struct = CudaStruct("Pair", [("a", "double"), ("b", "int")])
    with pytest.raises(TypeError, match="argument 0 is a C struct"):
        kernel.prepare_args(struct(a=1.0, b=2), 2.0, out)
    with pytest.raises(
        TypeError, match=r"takes 3 arguments \(x, factor, out\), got 2$"
    ):
        kernel.prepare_args(x, out)
    with pytest.raises(TypeError, match="got 4 after flattening"):
        kernel.prepare_args(CudaArguments(x, 1.0, 2.0), out)


def test_launch_configuration(launches, monkeypatch):
    kernel = NumbaCudaKernel(scale, block_size=64)
    x, out = DeviceArray(100), DeviceArray(100)
    kernel(x, 2.0, out)  # n_threads from the first array
    kernel(x, 2.0, out, grid=3, block=32, shared_mem=16)
    kernel(x, 2.0, out, n_threads=0)  # empty grid: nothing to launch
    assert [launch[1:3] + launch[4:] for launch in launches] == [
        ((2,), (64,), 0),
        ((3,), (32,), 16),
    ]
    with pytest.raises(ValueError, match="shared_mem must be non-negative"):
        kernel(x, 2.0, out, shared_mem=-1)
    kernel.n_threads_from = None
    with pytest.raises(TypeError, match="exactly one of n_threads and grid"):
        kernel(x, 2.0, out)

    synced = []
    monkeypatch.setattr(
        NumbaCudaKernel, "_synchronize", lambda self, *a: synced.append(a)
    )
    NumbaCudaKernel(scale, debug=True)(x, 2.0, out, n_threads=4)
    with xp.cuda.cuda_debug(True):
        NumbaCudaKernel(scale)(x, 2.0, out, n_threads=4)
    NumbaCudaKernel(scale, debug=False)(x, 2.0, out, n_threads=4)
    assert synced == [(None, (1,), (128,))] * 2


def test_debug_synchronization_names_the_kernel():
    class Stream:
        def synchronize(self):
            raise RuntimeError("an illegal memory access was encountered")

    kernel = CupyJitKernel(scale)
    with pytest.raises(RuntimeError, match="cupy_jit kernel 'scale' with grid"):
        kernel._synchronize(Stream(), (1,), (128,))


def test_cupy_jit_uses_rawkernel_and_the_stream(monkeypatch):
    calls = []

    def rawkernel():
        def wrap(function):
            def jitted(grid, block, args, shared_mem=0):
                calls.append((function, grid, block, args, shared_mem))

            return jitted

        return wrap

    jit = types.SimpleNamespace(rawkernel=rawkernel)
    monkeypatch.setitem(sys.modules, "cupyx", types.SimpleNamespace(jit=jit))
    monkeypatch.setitem(sys.modules, "cupyx.jit", jit)

    class Stream:
        entered = 0

        def __enter__(self):
            Stream.entered += 1

        def __exit__(self, *exc):
            return False

    kernel = CupyJitKernel(scale)
    x, out = DeviceArray(4), DeviceArray(4)
    kernel(x, 2.0, out, stream=Stream(), shared_mem=8)
    kernel(x, 2.0, out)
    assert calls == [(scale, (1,), (128,), (x, 2.0, out), 8)] + [
        (scale, (1,), (128,), (x, 2.0, out), 0)
    ]
    assert Stream.entered == 1 and kernel.is_compiled


def test_numba_cuda_launches_on_the_cupy_stream(monkeypatch):
    launched = []

    class Dispatcher:
        def __init__(self, function):
            self.py_func = function

        def __getitem__(self, config):
            return lambda *args: launched.append((config, args))

    cuda = types.SimpleNamespace(
        jit=Dispatcher, external_stream=lambda ptr: f"external {ptr}"
    )
    monkeypatch.setitem(sys.modules, "numba", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "numba.cuda", cuda)
    current = types.SimpleNamespace(ptr=0)
    cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_current_stream=lambda: current)
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    kernel = NumbaCudaKernel(scale)
    x, out = DeviceArray(4), DeviceArray(4)
    kernel(x, 2.0, out)  # CuPy's null stream: numba's default stream
    current.ptr = 1234
    kernel(x, 2.0, out)  # CuPy's current stream
    kernel(x, 2.0, out, stream=types.SimpleNamespace(ptr=99), shared_mem=4)
    kernel(x, 2.0, out, stream="numba stream")
    assert [config for config, _ in launched] == [
        ((1,), (128,), 0, 0),
        ((1,), (128,), "external 1234", 0),
        ((1,), (128,), "external 99", 4),
        ((1,), (128,), "numba stream", 0),
    ]
    assert launched[0][1] == (x, 2.0, out)


# -- dispatch -------------------------------------------------------------------


def test_kernel_takes_python_device_kernels():
    kernel = Kernel(scale, cupy_jit_kernel=scale, numba_cuda_kernel=scale)
    assert isinstance(kernel.cupy_jit_kernel, CupyJitKernel)
    assert isinstance(kernel.numba_cuda_kernel, NumbaCudaKernel)
    assert kernel.cuda_kernel is None and kernel.has_cuda
    assert kernel.implementations == ("host", "cupy_jit", "numba_cuda")
    assert list(kernel.device_kernels) == ["cupy_jit", "numba_cuda"]
    assert (
        repr(kernel)
        == "Kernel(name='scale', device=(cupy_jit, numba_cuda), missing_cuda='raise')"
    )
    assert repr(Kernel(scale)) == (
        "Kernel(name='scale', device=(none), missing_cuda='raise')"
    )
    own = NumbaCudaKernel(scale, block_size=32)
    assert Kernel(scale, numba_cuda_kernel=own).numba_cuda_kernel is own
    with pytest.raises(TypeError, match="cupy_jit_kernel must be a CupyJitKernel"):
        Kernel(scale, cupy_jit_kernel=own)
    with pytest.raises(TypeError, match="numba_cuda_kernel must be a NumbaCudaKernel"):
        Kernel(scale, numba_cuda_kernel=CudaKernel(SCALE_CUDA, "scale"))
    with pytest.raises(TypeError, match="got int"):
        Kernel(scale, numba_cuda_kernel=1)


def test_automatic_selection_prefers_cuda_c(launches):
    x, out = DeviceArray(4), DeviceArray(4)
    full = Kernel(
        scale,
        CudaKernel(SCALE_CUDA, "scale"),
        cupy_jit_kernel=scale,
        numba_cuda_kernel=scale,
    )
    assert full.selected(device=True) == "cuda_c"
    python = Kernel(scale, cupy_jit_kernel=scale, numba_cuda_kernel=scale)
    assert python.selected(device=True) == "cupy_jit"
    numba = Kernel(
        scale,
        cupy_jit_kernel=CupyJitKernel(name="scale", loader=failing_loader),
        numba_cuda_kernel=scale,
    )
    assert numba.selected(device=True) == "numba_cuda"  # cupy_jit does not load
    for kernel in (full, python, numba):
        kernel(x, 2.0, out)
    assert [launch[0] for launch in launches] == ["cuda_c", "cupy_jit", "numba_cuda"]
    assert full.resolve(x, 2.0, out) is full.cuda_kernel
    with xp.kernels.use_device_kernel_implementation("cuda"):
        assert python.selected(device=True) == "cupy_jit"


def test_explicit_selection(launches):
    x, out = DeviceArray(4), DeviceArray(4)
    kernel = Kernel(
        scale,
        CudaKernel(SCALE_CUDA, "scale"),
        cupy_jit_kernel=scale,
        numba_cuda_kernel=NumbaCudaKernel(name="scale", loader=failing_loader),
    )
    with xp.kernels.use_device_kernel_implementation("cupy_jit"):
        assert kernel.selected(device=True) == "cupy_jit"
        kernel(x, 2.0, out)
        assert launches[-1][0] == "cupy_jit"
    with xp.kernels.use_device_kernel_implementation("cuda_c"):
        kernel(x, 2.0, out)
        assert launches[-1][0] == "cuda_c"
    with (
        xp.kernels.use_device_kernel_implementation("numba_cuda"),
        pytest.raises(LookupError, match="'numba_cuda' implementation of kernel "),
    ):
        kernel(x, 2.0, out)
    host_only = Kernel(scale, cupy_jit_kernel=scale, missing_cuda="fallback")
    with (
        xp.kernels.use_device_kernel_implementation("numba_cuda"),
        pytest.raises(LookupError, match="has no 'numba_cuda' implementation"),
    ):
        host_only.selected(device=True)


def test_python_kernels_that_do_not_load(launches):
    broken = CupyJitKernel(name="scale", loader=failing_loader)
    raising = Kernel(scale, cupy_jit_kernel=broken)
    with pytest.raises(
        NotImplementedError, match="cupy_jit cannot be loaded: ImportError"
    ):
        raising.selected(device=True)
    with (
        xp.kernels.use_device_kernel_implementation("cuda"),
        pytest.raises(LookupError, match="has no 'cuda' implementation.*cupy_jit"),
    ):
        raising.selected(device=True)
    fallback = Kernel(scale, cupy_jit_kernel=broken, missing_cuda="fallback")
    with pytest.warns(RuntimeWarning, match="cupy_jit cannot be loaded"):
        assert fallback.selected(device=True) == "host"
    assert launches == []


def test_compile_and_check_signature(launches):
    def other(x, a, out):
        pass

    kernel = Kernel(scale, cupy_jit_kernel=scale, numba_cuda_kernel=other)
    with pytest.raises(ValueError, match=r"the numba_cuda kernel \(x, a, out\)"):
        kernel.check_signature()
    Kernel(scale, cupy_jit_kernel=scale).check_signature()
    # a kernel that does not load is skipped by the check
    Kernel(
        scale, cupy_jit_kernel=CupyJitKernel(name="s", loader=failing_loader)
    ).check_signature()
    assert kernel.compile() and kernel.cupy_jit_kernel.is_compiled
    assert not Kernel(scale).compile()


def test_kernel_needs_a_launch_size(launches):
    kernel = Kernel(scale, cupy_jit_kernel=CupyJitKernel(scale, n_threads_from=None))
    with pytest.raises(ValueError, match="n_threads is required"):
        kernel(DeviceArray(4), 2.0, DeviceArray(4))
    kernel(DeviceArray(4), 2.0, DeviceArray(4), n_threads=4)
    assert launches[-1][1:3] == ((1,), (128,))


# -- kernel folders --------------------------------------------------------------


@pytest.fixture
def folder_package(tmp_path, monkeypatch):
    """A package with kernel folders holding Python device kernels."""
    root = tmp_path / "demo_python_device_pkg"
    files = {
        "__init__.py": "",
        "scale/__init__.py": "",
        "scale/scale_kernels.py": "def scale(x, factor, out):\n    out[:] = factor * x\n",
        "scale/scale_cupy_jit.py": textwrap.dedent(
            """\
            from cupyx import jit  # not installed here: loads only on the GPU

            @jit.rawkernel()
            def scale(x, factor, out):
                pass
            """
        ),
        "scale/scale_numba_cuda.py": "def scale(x, factor, out):\n    pass\n",
        "scale/scale_cuda_c.cu": SCALE_CUDA,
        "both/__init__.py": "",
        "both/both_kernels.py": "def both(x):\n    pass\n",
        "both/both_cuda.cu": SCALE_CUDA.replace("scale", "both"),
        "both/both_cuda_c.cu": SCALE_CUDA.replace("scale", "both"),
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "demo_python_device_pkg"
    for module in [m for m in sys.modules if m.startswith("demo_python_device_pkg")]:
        del sys.modules[module]


def test_from_folder_finds_the_device_kernels(folder_package, launches):
    kernel = Kernel.from_folder(f"{folder_package}.scale", block_size=64)
    assert kernel.implementations == (
        "pyccel",
        "python",
        "cuda_c",
        "cupy_jit",
        "numba_cuda",
    )
    assert kernel.cuda_path.name == "scale_cuda_c.cu"
    assert kernel.numba_cuda_kernel.block_size == 64
    assert kernel.cupy_jit_kernel.block_size == 64
    if importlib.util.find_spec("cupyx") is None:
        assert not kernel.cupy_jit_kernel.available()
    assert kernel.numba_cuda_kernel.parameters() == ["x", "factor", "out"]
    kernel.check_signature()
    with xp.kernels.use_device_kernel_implementation("numba_cuda"):
        kernel(DeviceArray(4), 2.0, DeviceArray(4))
    assert launches[-1][0] == "numba_cuda"
    renamed = Kernel.from_folder(
        f"{folder_package}.scale", cupy_jit_suffix="_none", numba_cuda_suffix="_none"
    )
    assert renamed.implementations == ("pyccel", "python", "cuda_c")
    with pytest.raises(ValueError, match="has both both_cuda.cu and both_cuda_c.cu"):
        Kernel.from_folder(f"{folder_package}.both")


def test_catalog_passes_the_suffixes(folder_package):
    root = Path(importlib.import_module(folder_package).__file__).parent
    (root / "both" / "both_cuda.cu").unlink()
    catalog = KernelCatalog.from_package(folder_package, numba_cuda_suffix="_none")
    assert catalog["scale"].implementations == (
        "pyccel",
        "python",
        "cuda_c",
        "cupy_jit",
    )
    assert catalog.with_cuda == ["both", "scale"]
    assert catalog.summary() == "CUDA kernels: 2 of 2"


# -- parity checks ----------------------------------------------------------------


def test_assert_kernels_agree_checks_the_implementations():
    kernel = Kernel(scale, cupy_jit_kernel=scale)
    with pytest.raises(
        ValueError, match=r"no 'numba_cuda' implementation \(it has cupy_jit\)"
    ):
        assert_kernels_agree(kernel, lambda b, s: (), implementations=["numba_cuda"])
    lazy = Kernel(scale, numba_cuda_kernel=NumbaCudaKernel(scale, n_threads_from=None))
    with pytest.raises(TypeError, match="required to launch the numba_cuda kernel"):
        assert_kernels_agree(lazy, lambda b, s: ())


def test_numba_simulator_active_without_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    assert not numba_simulator_active()


# -- on the fake CuPy -------------------------------------------------------------

FAKE_CUPY_JIT = textwrap.dedent(
    """\
    import numpy as np
    import pytest
    import cunumpy as xp  # installs the fake CuPy
    import cupy
    from cupyx import jit
    from cunumpy.kernel_testing import assert_kernels_agree
    from cunumpy.kernels import CupyJitKernel, Kernel

    def scale(x, factor, out):
        out[:] = factor * x

    @jit.rawkernel()
    def scale_jit(x, factor, out):
        i = jit.grid(1)
        if i < x.size:
            out[i] = factor * x[i]

    def wrong(x, factor, out):
        i = jit.grid(1)
        if i < x.size:
            out[i] = factor + x[i]

    def make_args(backend, seed):
        x = xp.to_cunumpy(np.random.default_rng(seed).random(37))
        return x, 3.0, xp.zeros(37)

    kernel = Kernel(scale, cupy_jit_kernel=scale_jit)
    assert kernel.cupy_jit_kernel.python_function.__name__ == "scale_jit"
    host = assert_kernels_agree(kernel, make_args, outputs=[2])
    assert host["argument 2"].shape == (37,)
    with pytest.raises(AssertionError, match=r"scale \\(cupy_jit\\)"):
        assert_kernels_agree(Kernel(scale, cupy_jit_kernel=wrong), make_args, outputs=[2])

    # 2D grids, atomics and the launch through __getitem__
    @jit.rawkernel()
    def histogram(cells, counts):
        i, j = jit.grid(2)
        nx, ny = jit.gridsize(2)
        if i < cells.shape[0] and j < cells.shape[1]:
            jit.atomic_add(counts, cells[i, j], 1)
            assert jit.blockDim.x == 4 and jit.gridDim.y == 2 and nx == 8 and ny == 6

    cells = cupy.asarray(np.arange(20).reshape(5, 4) % 3)
    counts = cupy.zeros(3, dtype=np.int32)
    CupyJitKernel(histogram, block_size=(4, 3))(cells, counts)
    assert counts.get().tolist() == [7, 7, 6]
    histogram[(2, 2), (4, 3)](cells, counts)
    assert counts.get().sum() == 40

    old = cupy.zeros(1)
    assert jit.atomic_exch(old.get().astype(np.float32), 0, 5.0) == 0.0
    assert jit.atomic_max([1], 0, 3) == 1
    with pytest.raises(TypeError, match="does not support int64"):
        jit.atomic_add(np.zeros(1, dtype=np.int64), 0, 1)  # as in CuPy
    assert jit.atomic_cas([1], 0, 1, 7) == 1 and jit.atomic_min([4], 0, 2) == 4
    assert jit.atomic_sub([4], 0, 1) == 4

    @jit.rawkernel()
    def barrier(x):
        jit.syncthreads()

    with pytest.raises(NotImplementedError, match="jit.syncthreads is not emulated"):
        barrier((1,), (1,), (cupy.zeros(1),))
    with pytest.raises(TypeError, match="NumPy arrays cannot be passed"):
        barrier((1,), (1,), (np.zeros(1),))
    with pytest.raises(TypeError, match="device function cannot be launched"):
        jit.rawkernel(device=True)(scale)((1,), (1,), ())
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        barrier((1, 1, 1, 1), (1,), ())
    with pytest.raises(RuntimeError, match="only defined inside"):
        jit.threadIdx.x
    print("ok")
    """
)


def test_cupy_jit_kernels_run_on_the_fake_cupy():
    result = run_in_fake_cupy_subprocess(FAKE_CUPY_JIT, env={"PYTHONPATH": SRC})
    assert result.stdout.strip() == "ok"


FAKE_NUMBA_CUDA = textwrap.dedent(
    """\
    import numpy as np
    import cunumpy as xp  # installs the fake CuPy
    from numba import cuda
    from cunumpy.kernel_testing import assert_kernels_agree, numba_simulator_active
    from cunumpy.kernels import Kernel

    def scale(x, factor, out):
        out[:] = factor * x

    @cuda.jit
    def scale_numba(x, factor, out):
        i = cuda.grid(1)
        if i < x.size:
            out[i] = factor * x[i]

    def make_args(backend, seed):
        x = xp.to_cunumpy(np.random.default_rng(seed).random(37))
        return x, 3.0, xp.zeros(37)

    assert numba_simulator_active()
    kernel = Kernel(scale, numba_cuda_kernel=scale_numba)
    assert_kernels_agree(kernel, make_args, outputs=[2])  # in the simulator
    print("ok")
    """
)


@pytest.mark.skipif(
    importlib.util.find_spec("numba_cuda") is None, reason="numba-cuda not installed"
)
def test_numba_cuda_kernels_run_in_the_simulator_on_the_fake_cupy():
    result = run_in_fake_cupy_subprocess(
        FAKE_NUMBA_CUDA, env={"PYTHONPATH": SRC, "NUMBA_ENABLE_CUDASIM": "1"}
    )
    assert result.stdout.strip() == "ok"


def test_the_emulated_jit_in_process(monkeypatch):
    from cunumpy import _fake_cupy, _fake_cupyx_jit

    class Array:  # a fake device array, its host buffer in .data
        def __init__(self, data):
            self.data = data

    monkeypatch.setattr(_fake_cupy, "host_buffer", lambda array: array.data)
    jit = _fake_cupyx_jit.make_module(types.SimpleNamespace(ndarray=Array))

    def fill(out, value):
        i, j, k = jit.threadIdx.x, jit.threadIdx.y, jit.blockIdx.z
        if jit.grid(1) < out.shape[0]:
            out[jit.grid(1)] = value + i + 10 * j + 100 * k

    kernel = jit.rawkernel()(fill)
    assert kernel._func is fill and kernel.__name__ == "fill"
    assert type(kernel).__module__.startswith("cupyx.jit")
    out = np.zeros(4)
    kernel(2, 2, (Array(out), 1.0))
    assert out.tolist() == [1.0, 2.0, 1.0, 2.0]
    kernel((1, 1, 2), (2, 2), (Array(out), 0.0))
    assert out.tolist() == [110.0, 111.0, 1.0, 2.0]  # last writers: k = 1, j = 1
    with pytest.raises(TypeError, match="NumPy arrays cannot be passed"):
        kernel(1, 1, (np.zeros(1), 0.0))
    with pytest.raises(RuntimeError, match="only defined inside"):
        _ = jit.blockDim.z
    assert jit.atomic_add([1], 0, 2) == 1 and jit.range is range
    with pytest.raises(NotImplementedError, match="jit.shared_memory"):
        jit.shared_memory(np.float64, 4)


# -- on a GPU -----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
N_GPU = 1024  # a multiple of the block size: SCALE_CUDA has no bounds check


def load_gpu_kernels(stem):
    """Import a module of GPU kernels next to this file (they import CuPy or numba)."""
    spec = importlib.util.spec_from_file_location(stem, HERE / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numba_cuda_on_gpu():
    """Skip without CuPy, a GPU or numba-cuda; fail instead with CUNUMPY_REQUIRE_NUMBA_CUDA=1."""
    reason = None
    if fake_cupy_active() or not xp.cupy_available():
        reason = "CuPy/GPU not available"
    else:
        try:
            from numba import cuda

            if not cuda.is_available():
                reason = "numba-cuda finds no GPU"
        except ImportError:
            reason = "numba-cuda not installed"
    if reason is not None:
        if os.environ.get("CUNUMPY_REQUIRE_NUMBA_CUDA", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            pytest.fail(f"CUNUMPY_REQUIRE_NUMBA_CUDA is set, but {reason}")
        pytest.skip(reason)
    return load_gpu_kernels("gpu_kernels_numba_cuda")


def make_scale_args(backend, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(N_GPU))
    return x, 3.0, xp.zeros(N_GPU)


def check_histogram(kernel):
    import cupy as cp

    cells_host = np.random.default_rng(1).integers(0, 5, size=(40, 30))
    counts = cp.zeros(5, dtype=np.int32)  # cupyx.jit has no int64 atomic_add
    kernel(cp.asarray(cells_host), counts)  # a 2D launch from the shape of cells
    np.testing.assert_array_equal(
        counts.get(), np.bincount(cells_host.ravel(), minlength=5)
    )


def check_streams_and_rules(kernel):
    import cupy as cp

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:  # CuPy work before and after the launch, on the current stream
        x = cp.arange(float(N_GPU))
        out = cp.empty_like(x)
        kernel(x, 2.0, out)
        total = out.sum()
    stream.synchronize()
    assert float(total) == 2.0 * N_GPU * (N_GPU - 1) / 2
    again = cp.zeros_like(x)
    kernel(x, 3.0, again, stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(again.get(), 3.0 * x.get())
    kernel(CudaArguments(x, 4.0), again)  # flattened
    np.testing.assert_array_equal(again.get(), 4.0 * x.get())
    with pytest.raises(TypeError, match="argument 0 is a NumPy array"):
        kernel(np.ones(4), 1.0, cp.zeros(4))
    with xp.cuda.cuda_debug(True):  # synchronizes after the launch
        kernel(x, 5.0, again)
    np.testing.assert_array_equal(again.get(), 5.0 * x.get())


@requires_cupy
def test_cupy_jit_kernels_on_the_gpu():
    jit_kernels = load_gpu_kernels("gpu_kernels_cupy_jit")
    kernel = Kernel(
        scale, CudaKernel(SCALE_CUDA, "scale"), cupy_jit_kernel=jit_kernels.scale
    )
    assert kernel.selected(device=True) == "cuda_c"
    assert_kernels_agree(kernel, make_scale_args, outputs=[2])  # cuda_c and cupy_jit
    check_streams_and_rules(kernel.cupy_jit_kernel)
    check_histogram(CupyJitKernel(jit_kernels.histogram, block_size=(8, 8)))
    # a plain function is compiled by cupyx.jit at the first launch
    plain = CupyJitKernel(jit_kernels.scale._func)
    with xp.use_backend("cupy"):  # make_scale_args builds arrays on the active backend
        plain(*make_scale_args("cupy", 0))
    assert plain.is_compiled


def test_numba_cuda_kernels_on_the_gpu():
    numba_kernels = numba_cuda_on_gpu()
    jit_kernels = load_gpu_kernels("gpu_kernels_cupy_jit")
    kernel = Kernel(
        scale,
        CudaKernel(SCALE_CUDA, "scale"),
        cupy_jit_kernel=jit_kernels.scale,
        numba_cuda_kernel=numba_kernels.scale,
    )
    assert kernel.implementations[-3:] == ("cuda_c", "cupy_jit", "numba_cuda")
    assert_kernels_agree(kernel, make_scale_args, outputs=[2])  # all three
    with xp.kernels.use_device_kernel_implementation("numba_cuda"):
        assert kernel.selected(device=True) == "numba_cuda"
    check_streams_and_rules(kernel.numba_cuda_kernel)
    check_histogram(NumbaCudaKernel(numba_kernels.histogram, block_size=(8, 8)))
    plain = NumbaCudaKernel(numba_kernels.scale.py_func)
    with xp.use_backend("cupy"):  # make_scale_args builds arrays on the active backend
        plain(*make_scale_args("cupy", 0))
    assert plain.is_compiled
