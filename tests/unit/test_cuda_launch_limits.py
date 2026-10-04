"""Compilation/launch contracts with fake runtime limits and real GPU coverage."""

import io
import sys
import threading
import types

import pytest

import cunumpy as xp
from cunumpy import _cuda_kernel as implementation
from cunumpy.cuda import CudaKernel
from cunumpy.kernels import Kernel, KernelCatalog

EMPTY = 'extern "C" __global__ void empty() {}'
requires_gpu = pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")


@pytest.fixture
def runtime(monkeypatch):
    current = threading.local()
    current.device = 0
    raw_kernels, queries, compilations = [], [], []
    properties = {
        i: {
            "name": f"test GPU {i}".encode(),
            "maxThreadsDim": (1024, 1024, 64),
            "maxGridSize": (2**31 - 1, 65535, 65535),
            "maxThreadsPerBlock": 1024,
            "sharedMemPerBlock": 49152,
            "sharedMemPerBlockOptin": 98304,
        }
        for i in range(2)
    }

    def device_id():
        return getattr(current, "device", 0)

    class Device:
        def __init__(self, device):
            self.id = device

        def __enter__(self):
            self.previous = device_id()
            current.device = self.id
            return self

        def __exit__(self, *exc):
            current.device = self.previous

    class Raw:
        def __init__(self, source, name, options=()):
            self.source, self.name, self.options = source, name, options
            self.device = device_id()
            self.launches = []
            self.attributes = {
                "max_threads_per_block": 256,
                "shared_size_bytes": 4096,
                "max_dynamic_shared_size_bytes": 49152,
            }
            self.max_dynamic_shared_size_bytes = 49152
            raw_kernels.append(self)

        def compile(self, log_stream=None):
            compilations.append((self.device, self.name, self.options))
            if "bad_syntax" in self.source:
                raise RuntimeError("NVRTC rejected the source")
            if log_stream is not None:
                log_stream.write("compiled\n")

        def __call__(self, grid, block, args, shared_mem=0):
            assert device_id() == self.device
            self.launches.append((grid, block, shared_mem))

    class Module:
        def __init__(self, code, options, name_expressions):
            self.raw = Raw(code, name_expressions[0], options)
            self.compiled = False

        def compile(self, log_stream=None):
            self.raw.compile(log_stream)
            self.compiled = True

        def get_function(self, name):
            assert self.compiled
            assert name == self.raw.name
            return self.raw

    def get_properties(device):
        queries.append(device)
        return properties[device]

    cp = types.SimpleNamespace(
        RawKernel=Raw,
        RawModule=Module,
        cuda=types.SimpleNamespace(
            Device=Device,
            runtime=types.SimpleNamespace(
                getDevice=device_id, getDeviceProperties=get_properties
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(xp.xp, "cupy_available", lambda: True)
    monkeypatch.setattr(implementation, "_DEVICE_LIMITS", {})
    return types.SimpleNamespace(
        cp=cp,
        current=current,
        properties=properties,
        raw=raw_kernels,
        queries=queries,
        compilations=compilations,
    )


def test_compile_is_eager_cached_and_device_specific(runtime):
    kernel = CudaKernel(EMPTY, "empty")
    assert not kernel.is_compiled
    log = io.StringIO()
    first = kernel.compile(log_stream=log)
    assert log.getvalue() == "compiled\n"
    assert kernel.is_compiled
    assert kernel.compile() is first
    with runtime.cp.cuda.Device(1):
        assert not kernel.is_compiled
        second = kernel.compile()
        assert second is not first
        assert kernel.is_compiled
    assert kernel.compile() is first
    assert [c[0] for c in runtime.compilations] == [0, 1]
    assert runtime.queries == [0, 1]


def test_failed_compile_is_not_cached_and_can_be_retried(runtime, monkeypatch):
    kernel = CudaKernel(EMPTY, "empty")
    original = runtime.cp.RawKernel.compile

    def fail(self, log_stream=None):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(runtime.cp.RawKernel, "compile", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        kernel.compile()
    assert not kernel.is_compiled
    monkeypatch.setattr(runtime.cp.RawKernel, "compile", original)
    assert kernel.compile() is runtime.raw[-1]
    assert kernel.is_compiled


def test_recompile_refreshes_headers_and_debug_on_only_current_device(
    runtime, tmp_path
):
    header = tmp_path / "value.cuh"
    header.write_text("#define VALUE 1\n")
    kernel = CudaKernel(
        '#include "value.cuh"\n' + EMPTY, "empty", include_dirs=[tmp_path]
    )
    first = kernel.compile()
    with runtime.cp.cuda.Device(1):
        second = kernel.compile()
    header.write_text("#define VALUE 2\n")
    with xp.cuda.cuda_debug():
        rebuilt = kernel.recompile()
    assert rebuilt is not first
    assert rebuilt.options != first.options
    assert "-lineinfo" in rebuilt.options
    with runtime.cp.cuda.Device(1):
        assert kernel.compile() is second


def test_template_is_compiled_before_function_is_returned(runtime):
    kernel = CudaKernel(
        "template<typename T> __global__ void empty() {}",
        "empty",
        template_args=["double"],
    )
    assert kernel.compile().name == "empty<double>"
    assert len(runtime.compilations) == 1


def test_catalog_fails_during_setup_and_workers_keep_selected_device(runtime):
    good = CudaKernel(EMPTY, "empty")
    bad = CudaKernel(EMPTY + "\nbad_syntax", "empty")
    catalog = KernelCatalog(
        {"good": Kernel(lambda: None, good), "bad": Kernel(lambda: None, bad)}
    )
    with runtime.cp.cuda.Device(1):
        with pytest.raises(RuntimeError, match="NVRTC rejected"):
            catalog.compile_all(jobs=2)
        assert good.is_compiled
        assert not bad.is_compiled
    assert all(c[0] == 1 for c in runtime.compilations)


@pytest.mark.parametrize(
    "launch, expected",
    [
        ({"grid": (1, 65536)}, "grid\\[1\\]=65536"),
        ({"grid": 1, "block": (1, 1, 65)}, "block\\[2\\]=65"),
        ({"grid": 1, "block": 512}, "256 device/kernel limit"),
        ({"grid": 1, "shared_mem": 98304 - 4096 + 1}, "static shared memory 4096"),
    ],
)
def test_actual_limits_reject_invalid_launch_before_execution(
    runtime, launch, expected
):
    kernel = CudaKernel(EMPTY, "empty")
    with pytest.raises(ValueError, match=expected) as error:
        kernel(**launch)
    assert "test GPU 0" in str(error.value)
    assert runtime.raw[0].launches == []


def test_device_limits_can_be_stricter_than_kernel_limits(runtime):
    runtime.properties[0]["maxThreadsPerBlock"] = 64
    with pytest.raises(ValueError, match="64 device/kernel limit"):
        CudaKernel(EMPTY, "empty")(grid=1)


def test_static_memory_requires_opt_in_even_with_large_initial_dynamic_attribute(
    runtime,
):
    kernel = CudaKernel(EMPTY, "empty")
    raw = kernel.compile()
    state = kernel._compiled[0]
    assert state.dynamic_shared == 45056
    kernel(grid=1, shared_mem=49152 - 4096 + 1)
    assert raw.max_dynamic_shared_size_bytes == 45057


def test_opt_in_accounts_for_static_memory_and_is_per_device(runtime):
    kernel = CudaKernel(EMPTY, "empty")
    kernel(grid=1, shared_mem=45057)
    first = runtime.raw[-1]
    assert first.max_dynamic_shared_size_bytes == 45057
    kernel(grid=1, shared_mem=60000)
    kernel(grid=1, shared_mem=50000)
    assert first.max_dynamic_shared_size_bytes == 60000
    with runtime.cp.cuda.Device(1):
        kernel(grid=1, shared_mem=45057)
        assert runtime.raw[-1].max_dynamic_shared_size_bytes == 45057
    assert len(runtime.queries) == 2
    assert len(runtime.compilations) == 2


def test_stream_on_another_device_is_rejected(runtime):
    with pytest.raises(ValueError, match="stream belongs to another device"):
        CudaKernel(EMPTY, "empty")(grid=1, stream=types.SimpleNamespace(device_id=1))


@pytest.mark.parametrize("wrapped", [False, True])
def test_array_on_another_device_is_rejected_before_launch(runtime, wrapped):
    array = types.SimpleNamespace(
        __cuda_array_interface__={}, device=types.SimpleNamespace(id=1)
    )
    argument = (
        types.SimpleNamespace(__cuda_args__=lambda: (array,)) if wrapped else array
    )
    kernel = CudaKernel(EMPTY, "empty", check_signature=False)
    with pytest.raises(ValueError, match="belongs to CUDA device 1"):
        kernel(argument, grid=1)
    assert runtime.raw[0].launches == []


def test_empty_launch_does_not_compile(runtime):
    CudaKernel(EMPTY, "empty")(n_threads=0)
    assert runtime.compilations == []


@requires_gpu
def test_invalid_cuda_fails_at_compile_not_first_launch():
    kernel = CudaKernel(
        'extern "C" __global__ void broken() { missing_symbol(); }', "broken"
    )
    with pytest.raises(Exception, match="missing_symbol"):
        kernel.compile()
    assert not kernel.is_compiled


@requires_gpu
def test_kernel_launch_bounds_are_enforced():
    kernel = CudaKernel(
        'extern "C" __global__ __launch_bounds__(64) void limited() {}',
        "limited",
        check_signature=False,
        block_size=64,
    )
    raw = kernel.compile()
    assert raw.attributes["max_threads_per_block"] == 64
    with pytest.raises(ValueError, match="64 device/kernel limit"):
        kernel(grid=1, block=128)
    kernel(grid=1)
    xp.synchronize()


@requires_gpu
def test_same_kernel_compiles_on_multiple_devices():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two devices")
    kernel = CudaKernel(EMPTY, "empty")
    for device in (0, 1, 0):
        with cp.cuda.Device(device):
            kernel(grid=1)
            assert kernel.is_compiled
            cp.cuda.get_current_stream().synchronize()


@requires_gpu
def test_static_shared_memory_is_included_in_actual_gpu_limit():
    import cupy as cp

    source = r"""extern "C" __global__ void scratch(const double* x, double* out) {
        __shared__ double values[128];
        values[threadIdx.x] = x[threadIdx.x];
        __syncthreads();
        if (threadIdx.x == 0) out[0] = values[127];
    }"""
    kernel = CudaKernel(source, "scratch")
    raw = kernel.compile()
    static = raw.attributes["shared_size_bytes"]
    assert static >= 128 * 8
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    limit = max(
        properties["sharedMemPerBlock"], properties.get("sharedMemPerBlockOptin", 0)
    )
    with pytest.raises(ValueError, match="static shared memory"):
        kernel(cp.ones(128), cp.empty(1), n_threads=128, shared_mem=limit - static + 1)
