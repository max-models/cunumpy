"""Tests for the kernel-porting helpers added for struphy.

Device-ownership checks, structs from pyccel source files, argument objects
with a pyccel host class, launch sizes from the arguments, NaN checks, the
Fortran name limit, host parameters from pyccel stubs, test-arguments modules
and parity cases, struct parameters of device-function wrappers, the fake
CuPy, `mpi_buffer`, `segment_sum` and `require_version`. Everything runs
without a GPU; the fake CuPy runs in a subprocess so that it never replaces
CuPy in the test process.
"""

import os
import pickle
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
import cunumpy._cuda_kernel as cuda_kernel_module
import cunumpy.kernel_testing
from cunumpy._dispatch import FORTRAN_NAME_LIMIT, _pyccel_stub_parameters
from cunumpy.cuda import CudaKernel, CudaStruct
from cunumpy.kernel_testing import check_parity, device_function_kernel, parity_cases
from cunumpy.kernels import Kernel, KernelCatalog, PyccelStructArguments


class FakeDeviceArray:
    """Enough of a CuPy array for the argument checks, optionally on a device."""

    def __init__(self, dtype, ptr=0x1000, shape=(4,), device=None):
        self.dtype = np.dtype(dtype)
        self.data = SimpleNamespace(ptr=ptr)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.strides = tuple(np.zeros(self.shape, dtype=self.dtype).strides)
        self.flags = SimpleNamespace(c_contiguous=True)
        if device is not None:
            self.device = SimpleNamespace(id=device)

    @property
    def __cuda_array_interface__(self):
        return {}


SCALE = r"""
extern "C" __global__ void scale(double* x, double factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""

PYCCEL_SOURCE = textwrap.dedent("""
    "Argument classes, pyccel style."

    from numpy import shape


    class MarkerArguments:
        def __init__(
            self,
            markers: "float[:,:]",
            valid_mks: "bool[:]",
            Np: "int",
            first_pusher_idx: "int",
            bc_type: "int[:]",
        ):
            self.markers = markers
            self.valid_mks = valid_mks
            self.Np = Np
            self.n_markers = shape(markers)[0]
            self.first_init_idx = first_pusher_idx
            self.bc_type = bc_type
            self.scratch = markers[0]


    class DerhamArguments:
        def __init__(self, pn: "int[:]", tn1: "Final[float[:]]"):
            self.pn = pn
            self.tn1 = tn1
            self.bn1 = pn
    """)


# ---------------------------------------------------------------------------
# device-ownership check
# ---------------------------------------------------------------------------


def test_arrays_on_another_device_are_rejected(monkeypatch):
    kernel = CudaKernel(SCALE, "scale")
    monkeypatch.setattr(cuda_kernel_module, "_current_device_id", lambda: 0)
    kernel.prepare_args(FakeDeviceArray(np.float64), 2.0, 4)  # no device attribute
    kernel.prepare_args(FakeDeviceArray(np.float64, device=0), 2.0, 4)
    with pytest.raises(
        ValueError,
        match="on CUDA device 1, but the current device is 0",
    ):
        kernel.prepare_args(FakeDeviceArray(np.float64, device=1), 2.0, 4)
    # struct pointer fields go through the same check
    struct = CudaStruct("Vec", [("x", "double*"), ("n", "int")])
    with pytest.raises(ValueError, match="on CUDA device 1"):
        struct(x=FakeDeviceArray(np.float64, device=1), n=4)


def test_device_check_is_skipped_without_cupy(monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    assert cuda_kernel_module._current_device_id() is None
    CudaKernel(SCALE, "scale").prepare_args(
        FakeDeviceArray(np.float64, device=7),
        2.0,
        4,
    )


# ---------------------------------------------------------------------------
# CudaStruct.from_pyccel_class
# ---------------------------------------------------------------------------


def test_from_pyccel_class_from_source_and_file(tmp_path):
    struct = CudaStruct.from_pyccel_class(
        PYCCEL_SOURCE,
        "MarkerArguments",
        "MarkerArgs",
    )
    assert struct.name == "MarkerArgs"
    # fields are named after the attributes the parameters are stored in
    assert [(f.name, f.ctype) for f in struct.fields] == [
        ("markers", "Array2D<double>"),
        ("valid_mks", "Array1D<bool>"),
        ("Np", "long long"),
        ("first_init_idx", "long long"),
        ("bc_type", "Array1D<long long>"),
    ]
    path = tmp_path / "pusher_args_kernels.py"
    path.write_text(PYCCEL_SOURCE)
    from_file = CudaStruct.from_pyccel_class(path, "MarkerArguments")
    assert from_file.name == "MarkerArguments"
    assert from_file.dtype == struct.dtype
    same = CudaStruct.from_pyccel_class(
        str(path),
        "MarkerArguments",
        "A",
        int_type="int",
    )
    assert [f.ctype for f in same.fields][2] == "int"


def test_from_pyccel_class_options():
    keep = CudaStruct.from_pyccel_class(
        PYCCEL_SOURCE,
        "MarkerArguments",
        attribute_names=False,
    )
    assert [f.name for f in keep.fields][3] == "first_pusher_idx"
    fewer = CudaStruct.from_pyccel_class(
        PYCCEL_SOURCE,
        "MarkerArguments",
        exclude=("bc_type", "first_pusher_idx"),
    )
    assert [f.name for f in fewer.fields] == ["markers", "valid_mks", "Np"]
    final = CudaStruct.from_pyccel_class(PYCCEL_SOURCE, "DerhamArguments", "DerhamArgs")
    assert [(f.name, f.ctype) for f in final.fields] == [
        ("pn", "Array1D<long long>"),
        ("tn1", "Array1D<double>"),
    ]


def test_from_pyccel_class_errors():
    with pytest.raises(ValueError, match="no class 'Nope'"):
        CudaStruct.from_pyccel_class(PYCCEL_SOURCE, "Nope")
    with pytest.raises(ValueError, match="has no __init__"):
        CudaStruct.from_pyccel_class("class Empty:\n    pass\n", "Empty")
    with pytest.raises(ValueError, match="no type annotation"):
        CudaStruct.from_pyccel_class(
            "class A:\n    def __init__(self, x):\n        pass\n",
            "A",
        )


# ---------------------------------------------------------------------------
# PyccelStructArguments
# ---------------------------------------------------------------------------


class HostMarkers:
    """Stands in for a pyccel-compiled argument class."""

    instances = 0

    def __init__(self, markers, Np):
        type(self).instances += 1
        self.markers = markers
        self.Np = Np


class MarkerArguments(PyccelStructArguments):
    struct_name = "MarkerArgs"
    fields = (("markers", "Array2D<double>"), ("Np", "long long"), ("n_markers", "int"))
    host_class = HostMarkers
    host_fields = ("markers", "Np")

    def __init__(self, markers, Np):
        self.markers = markers
        self.Np = Np
        self.n_markers = markers.shape[0]


def test_pyccel_struct_arguments_host_form_is_built_once_and_follows_changes():
    HostMarkers.instances = 0
    markers = np.zeros((5, 3))
    args = MarkerArguments(markers, 5)
    host = args.__host_args__()
    assert isinstance(host, HostMarkers) and host.markers is markers and host.Np == 5
    assert args.__host_args__() is host and HostMarkers.instances == 1
    args.Np = 6  # a changed scalar: rebuilt
    assert args.__host_args__().Np == 6 and HostMarkers.instances == 2
    args.markers = np.zeros((7, 3))  # a replaced array: rebuilt
    assert args.__host_args__().markers is args.markers and HostMarkers.instances == 3
    # the host form is the Kernel's host argument
    seen = {}

    def push(m, dt):
        seen["host"] = m

    Kernel(push)(args, 0.1)
    assert seen["host"] is args.__host_args__()


def test_pyccel_struct_arguments_device_form_and_pickling():
    args = MarkerArguments(FakeDeviceArray(np.float64, shape=(5, 3)), 5)
    (packed,) = args.__cuda_args__()
    assert packed.dtype == MarkerArguments.struct.dtype
    assert int(packed["n_markers"]) == 5
    with pytest.raises(RuntimeError, match="no host form on the CuPy backend"):
        args.__host_args__()
    host_copy = MarkerArguments(np.ones((2, 3)), 2)
    host_copy.__host_args__()
    restored = pickle.loads(pickle.dumps(host_copy))
    assert "_host_value" not in restored.__dict__
    assert restored.__host_args__().markers.shape == (2, 3)


def test_pyccel_struct_arguments_host_copies(monkeypatch):
    class Copying(MarkerArguments):
        host_copies = True

    device = FakeDeviceArray(np.float64, shape=(2, 3))
    monkeypatch.setattr("cunumpy.xp.to_numpy", lambda a: np.full((2, 3), 7.0))
    host = Copying(device, 2).__host_args__()
    assert isinstance(host.markers, np.ndarray) and host.markers[0, 0] == 7.0


def test_pyccel_struct_arguments_requires_host_class():
    class NoHost(PyccelStructArguments):
        struct_name = "NoHost"
        fields = (("n", "int"),)

        def __init__(self):
            self.n = 1

    with pytest.raises(TypeError, match="host_class is not set"):
        NoHost().__host_args__()


# ---------------------------------------------------------------------------
# n_threads_from and check_finite
# ---------------------------------------------------------------------------


def _launching(kernel):
    """Replace compilation by a recorder of (grid, block) launches."""
    launches = []
    kernel.compile = lambda: (
        lambda grid, block, values, shared_mem: launches.append(grid)
    )
    return launches


def test_n_threads_from():
    kernel = CudaKernel(SCALE, "scale", n_threads_from=lambda args: args[2])
    launches = _launching(kernel)
    kernel(FakeDeviceArray(np.float64), 2.0, 300)
    assert launches == [(3,)]
    kernel(FakeDeviceArray(np.float64), 2.0, 300, n_threads=10)  # explicit wins
    assert launches[-1] == (1,)
    kernel.n_threads_from = None
    with pytest.raises(TypeError, match="exactly one of n_threads and grid"):
        kernel(FakeDeviceArray(np.float64), 2.0, 300)
    with pytest.raises(TypeError, match="callable"):
        kernel.n_threads_from = 5


def test_check_finite(monkeypatch):
    kernel = CudaKernel(SCALE, "scale", check_finite=True)
    assert kernel.check_finite
    _launching(kernel)
    kernel._synchronize_after_launch = lambda *a: None
    fake_cupy = ModuleType("cupy")
    fake_cupy.isfinite = np.isfinite
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    class Arr(FakeDeviceArray):
        def __init__(self, values):
            super().__init__(np.float64, shape=np.shape(values))
            self.values = np.asarray(values, dtype=np.float64)

        def __array__(self, dtype=None, copy=None):
            return self.values

    kernel(Arr([1.0, 2.0]), 2.0, 2, n_threads=2)
    with pytest.raises(RuntimeError, match="left a NaN or inf in argument 0"):
        kernel(Arr([1.0, np.nan]), 2.0, 2, n_threads=2)
    kernel.check_finite = False
    kernel(Arr([1.0, np.nan]), 2.0, 2, n_threads=2)


def test_device_arrays_in_struct_arguments():
    args = MarkerArguments(FakeDeviceArray(np.float64, shape=(5, 3)), 5)
    found = dict(
        cuda_kernel_module._device_arrays_in((1.0, args, FakeDeviceArray("f8"))),
    )
    assert set(found) == {"argument 1.markers", "argument 2"}


# ---------------------------------------------------------------------------
# dispatch: name length, pyccel stubs, test-arguments modules
# ---------------------------------------------------------------------------


def _forget_package(name):
    """Drop a temporary package from sys.modules (each test gets a new tmp_path)."""
    import importlib

    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[module]
    importlib.invalidate_caches()


@pytest.fixture
def helper_package(tmp_path, monkeypatch):
    _forget_package("helper_kernel_pkg")
    root = tmp_path / "helper_kernel_pkg"
    for name in ("scale", "no_cuda"):
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text("")
        (root / name / f"{name}_kernels.py").write_text(
            f"def {name}(x, factor, n):\n    for i in range(n):\n        x[i] *= factor\n",
        )
    (root / "scale" / "scale_cuda.cu").write_text(SCALE)
    (root / "scale" / "scale_test_args.py").write_text(
        "import numpy as np\nimport cunumpy as xp\n\nN_THREADS = 300\nRTOL = 1e-10\n\n\n"
        "def make_args(backend, seed):\n"
        "    x = xp.to_cunumpy(np.random.default_rng(seed).random(300))\n"
        "    return (x, 2.0, x.size)\n",
    )
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    _forget_package("helper_kernel_pkg")


def test_catalog_records_test_args_modules(helper_package):
    catalog = KernelCatalog.from_package("helper_kernel_pkg")
    assert (
        catalog["scale"].test_args_module == "helper_kernel_pkg.scale.scale_test_args"
    )
    assert catalog["no_cuda"].test_args_module is None
    module = catalog["scale"].test_args
    assert module.N_THREADS == 300 and catalog["scale"].test_args is module
    off = KernelCatalog.from_package("helper_kernel_pkg", test_args_suffix=None)
    assert off["scale"].test_args_module is None


def test_check_parity_reads_the_module(helper_package, monkeypatch):
    catalog = KernelCatalog.from_package("helper_kernel_pkg")
    calls = {}

    def fake_agree(kernel, make_args, **settings):
        calls["kernel"], calls["settings"] = kernel, settings
        return {"argument 0": np.zeros(1)}

    monkeypatch.setattr(cunumpy.kernel_testing, "assert_kernels_agree", fake_agree)
    check_parity(catalog["scale"], atol=1e-14)
    assert calls["kernel"] is catalog["scale"]
    assert calls["settings"] == {"n_threads": 300, "rtol": 1e-10, "atol": 1e-14}
    with pytest.raises(ValueError, match="no test-arguments module"):
        check_parity(catalog["no_cuda"])


def test_parity_cases_marks_kernels_without_test_args(helper_package):
    root = helper_package / "with_cuda_no_args"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "with_cuda_no_args_kernels.py").write_text(
        "def with_cuda_no_args(x, factor, n):\n    pass\n",
    )
    (root / "with_cuda_no_args_cuda.cu").write_text(
        SCALE.replace("scale", "with_cuda_no_args"),
    )
    catalog = KernelCatalog.from_package("helper_kernel_pkg")
    cases = parity_cases(catalog)
    assert [case.id for case in cases] == ["scale", "with_cuda_no_args"]
    assert cases[0].marks == ()
    (mark,) = cases[1].marks
    assert (
        mark.name == "skip"
        and "with_cuda_no_args_test_args.py" in mark.kwargs["reason"]
    )


def test_long_kernel_names_warn(tmp_path, monkeypatch):
    name = "k" * (FORTRAN_NAME_LIMIT - len("bind_c__kernels") + 1)
    root = tmp_path / "long_name_pkg"
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text("")
    (root / name / f"{name}_kernels.py").write_text(f"def {name}(x):\n    pass\n")
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_package("long_name_pkg")
    with pytest.warns(UserWarning, match="longer than Fortran's limit of 63"):
        KernelCatalog.from_package("long_name_pkg")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KernelCatalog.from_package("long_name_pkg", check_name_length=False)


def test_host_parameters_from_pyccel_stub(tmp_path):
    so = tmp_path / "push_kernels.cpython-313-darwin.so"
    (tmp_path / "__pyccel__").mkdir()
    (tmp_path / "__pyccel__" / "push_kernels.pyi").write_text(
        '#$ header metavar printer_imports="pyc_math_f90"\n'
        "from pyccel.decorators import low_level\n\n"
        "@low_level('push')\n"
        "def push(dt : 'float', stage : 'int', markers : 'float64[:,:](order=C)') -> None:\n"
        "    ...\n",
    )
    module = ModuleType("push_kernels")
    module.__file__ = str(so)
    compiled = SimpleNamespace(__self__=module, __name__="push")
    assert _pyccel_stub_parameters(compiled) == ["dt", "stage", "markers"]
    assert (
        _pyccel_stub_parameters(SimpleNamespace(__self__=module, __name__="x")) is None
    )
    assert _pyccel_stub_parameters(len) is None  # a builtin without a stub

    kernel = Kernel(compiled, CudaKernel(SCALE.replace("scale", "push"), "push"))
    # inspect.signature fails on the fake compiled function; the stub is used
    assert kernel.host_parameters() == ["dt", "stage", "markers"]
    with pytest.raises(ValueError, match=r"host kernel takes \(dt, stage, markers\)"):
        kernel.check_signature()


# ---------------------------------------------------------------------------
# device_function_kernel with struct parameters
# ---------------------------------------------------------------------------


def test_device_function_kernel_struct_parameters():
    domain = CudaStruct("DomainArgs", [("params", "double*"), ("kind", "int")])
    source = domain.declaration + (
        "__device__ double scale_x(const DomainArgs& d, double x) "
        "{ return d.params[0] * x; }\n"
    )
    kernel = device_function_kernel(
        source,
        "double scale_x(const DomainArgs& d, double x)",
        structs=[domain],
    )
    params = [(p.name, p.ctype, p.struct is not None) for p in kernel.signature]
    assert params == [
        ("d", "DomainArgs", True),
        ("x", "double", False),
        ("out", "double", False),
        ("n", "int", False),
    ]
    assert "void scale_x_kernel(DomainArgs d, const double* x, double* out, int n)" in (
        kernel.source
    )
    assert "out[i] = scale_x(d, x[i]);" in kernel.source
    with pytest.raises(ValueError, match="only structs can be passed by reference"):
        device_function_kernel(source, "double f(const double& x)")


# ---------------------------------------------------------------------------
# segment_sum, mpi_buffer, require_version
# ---------------------------------------------------------------------------


def test_segment_sum():
    keys = np.array([0, 2, 0, -1, 2])
    values = np.array([1.0, 2.0, 3.0, 100.0, 4.0])
    np.testing.assert_array_equal(
        xp.algorithms.segment_sum(values, keys, 4),
        [4.0, 0.0, 6.0, 0.0],
    )
    columns = np.stack([values, -values], axis=1)
    out = xp.algorithms.segment_sum(columns, keys, 3)
    np.testing.assert_array_equal(out, [[4.0, -4.0], [0.0, 0.0], [6.0, -6.0]])
    assert (
        xp.algorithms.segment_sum(np.array([1, 2]), np.array([1, 1]), 2).dtype
        == np.float64
    )
    assert (
        xp.algorithms.segment_sum(values.astype(np.float32), keys, 3).dtype
        == np.float32
    )
    complex_sum = xp.algorithms.segment_sum(values * (1 + 1j), keys, 3)
    np.testing.assert_allclose(complex_sum, [4 + 4j, 0, 6 + 6j])
    with pytest.raises(ValueError, match="smaller than n_segments"):
        xp.algorithms.segment_sum(values, keys, 2)
    with pytest.raises(ValueError, match="one entry per value"):
        xp.algorithms.segment_sum(values, keys[:2], 3)


def test_mpi_buffer_on_host_arrays():
    x = np.arange(3.0)
    with xp.mpi.mpi_buffer(x) as buf:
        assert buf is x
    with xp.mpi.mpi_buffer(x, send=False, recv=True) as buf:
        assert buf is x


def test_mpi_cuda_aware_setting():
    xp.mpi.set_mpi_cuda_aware(None)
    assert xp.mpi.get_mpi_cuda_aware() is None
    xp.mpi.set_mpi_cuda_aware(True)
    assert xp.mpi.get_mpi_cuda_aware() is True
    xp.mpi.set_mpi_cuda_aware(None)


def test_require_version(monkeypatch):
    xp.require_version("0.1")
    xp.require_version(xp.__version__)
    monkeypatch.setattr(xp, "__version__", "0.3.0")
    with pytest.raises(ImportError, match="0.4.0 or newer is required, but 0.3.0"):
        xp.require_version("0.4.0")
    monkeypatch.setattr(xp, "__version__", "0.0.0+unknown")
    xp.require_version("99.0")  # unknown version: not checked


# ---------------------------------------------------------------------------
# the fake CuPy (in a subprocess: it replaces cupy for the whole process)
# ---------------------------------------------------------------------------

FAKE_SCRIPT = r"""
import pickle
import numpy as np
import pytest
import cunumpy as xp
import cunumpy.kernel_testing as testing
from cunumpy.cuda import CudaKernel, CudaStruct
from cunumpy.kernels import KernelArguments

assert testing.fake_cupy_active()
assert xp.cupy_available() and xp.get_backend() == "cupy", xp.get_backend()
a = xp.zeros((4, 3))
assert type(a).__module__ == "cupy" and not isinstance(a, np.ndarray)
assert xp.is_gpu(a)
with pytest.raises(TypeError):
    np.asarray(a)
with pytest.raises(TypeError):
    a + np.ones(3)
assert a.sum().shape == ()
assert xp.to_numpy(a).shape == (4, 3)
b = pickle.loads(pickle.dumps(a))
assert xp.is_gpu(b)

# struct packing and the kernel argument checks work on fake device arrays
Vec = CudaStruct("Vec", [("x", "double*"), ("n", "int")])
assert int(Vec(x=xp.zeros(5), n=5).packed["x"]) != 0
scale = CudaKernel("extern \"C\" __global__ void scale(double* x, double f, int n) {}", "scale")
scale.prepare_args(xp.zeros(5), 2.0, 5)
with pytest.raises(TypeError):
    scale.prepare_args(np.zeros(5), 2.0, 5)
with pytest.raises(NotImplementedError):
    scale(xp.zeros(5), 2.0, 5, n_threads=5)

# kernels cannot run: the GPU tests are skipped
assert testing.requires_cupy.args[0] is True
assert testing.FAKE_SKIP_REASON in testing.requires_cupy.kwargs["reason"]

# as_device_array and the staging path of mpi_buffer
d = xp.as_device_array([1.0, 2.0], dtype=np.float64)
assert xp.is_gpu(d)
with pytest.raises(RuntimeError, match="not known whether MPI"):
    with xp.mpi.mpi_buffer(d):
        pass
with xp.profiling.count_transfers() as counter:
    with xp.mpi.mpi_buffer(d, recv=True, cuda_aware=False) as buf:
        assert isinstance(buf, np.ndarray) and buf.tolist() == [1.0, 2.0]
        buf[:] = [5.0, 6.0]
assert xp.to_numpy(d).tolist() == [5.0, 6.0]
assert sorted(e.kind for e in counter.events) == ["to_device", "to_host"]
with xp.mpi.mpi_buffer(d, cuda_aware=True) as buf:
    assert buf is d
producer = xp.cuda.create_stream()
event = xp.cuda.create_event()
with xp.cuda.stream(producer):
    assert xp.cuda.record_event(event, stream=producer) is event
xp.cuda.wait_event(event, stream=producer)
assert producer.done and event.done
assert producer.record().done
assert xp.backend_info()["versions"]["cupy"] == "0.0.0+cunumpy-fake"
print("fake cupy OK")
"""


def test_fake_cupy_in_subprocess():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUNUMPY_FAKE_CUPY="1", ARRAY_BACKEND="cupy")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), env.get("PYTHONPATH", "")) if p
    )
    env.pop("CUNUMPY_CUDA_DEBUG", None)
    result = subprocess.run(
        [sys.executable, "-c", FAKE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fake cupy OK" in result.stdout


def test_install_fake_cupy_refuses_a_real_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", ModuleType("cupy"))
    with pytest.raises(RuntimeError, match="real CuPy is already imported"):
        cunumpy.kernel_testing.install_fake_cupy()
    assert not cunumpy.kernel_testing.fake_cupy_active()
