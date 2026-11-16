"""Fast paths: host segment sums, sort outputs, resolved kernels, bound launches, graphs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _cuda_kernel, _graphs
from cunumpy import _dispatch as dispatch_module
from cunumpy.algorithms import SegmentPlan, segment_sum, sort_by_key
from cunumpy.kernel_testing import (
    BACKENDS,
    emulate_cuda_kernel,
    emulated_launches,
    emulation_compiler,
    run_in_fake_cupy_subprocess,
)
from cunumpy.kernels import (
    BoundLaunch,
    CompiledHostKernel,
    CudaKernel,
    HostImplementations,
    Kernel,
    PyccelKernel,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")

AXPY = (
    'extern "C" __global__ void axpy(double a, const double* x, double* y, int n) {'
    " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i]; }"
)


def reference_sum(keys, values, n_segments, dtype):
    expected = np.zeros((n_segments, *values.shape[1:]), dtype=dtype)
    valid = keys >= 0
    np.add.at(expected, keys[valid], values[valid].astype(dtype))
    return expected


@pytest.mark.parametrize(
    "keys",
    [
        [3, 0, 3, 1, 0, 3],  # unsorted, all valid
        [3, -1, 0, 3, -2, 1],  # unsorted, dropped rows
        [0, 0, 1, 3, 3, 3],  # sorted, all valid
        [-2, -1, 0, 0, 3, 3],  # sorted, dropped rows first
        [0, 1, 2, 3, 3, 3],  # sorted, every segment present
        [-1, -1, -1, -1, -1, -1],  # nothing kept
    ],
)
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.complex128, np.int64, np.bool_]
)
@pytest.mark.parametrize("trailing", [(), (2,), (2, 3)])
def test_host_segment_sums_match_add_at(keys, dtype, trailing):
    keys = np.array(keys)
    values = (np.arange(6 * int(np.prod(trailing))) % 5).reshape(6, *trailing)
    values = values.astype(dtype)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * values[::-1]
    out_dtype = dtype if np.dtype(dtype).kind in "fc" else np.float64
    expected = reference_sum(keys, values, 4, out_dtype)
    plan = SegmentPlan(keys, 4)
    assert plan.deterministic
    out = np.full(expected.shape, 7, dtype=out_dtype)
    for _ in range(2):  # overwrites
        assert plan.sum(values, out=out) is out
        np.testing.assert_allclose(out, expected, rtol=1e-3)
    np.testing.assert_allclose(segment_sum(values, keys, 4), expected, rtol=1e-3)


def test_sorted_segment_sum_kernel_in_cpu_emulation():
    from cunumpy._algorithms import _SORTED_SUM_BLOCK, _sorted_sum_kernel

    if emulation_compiler() is None:
        pytest.skip("requires a C++ compiler")
    keys = np.repeat([0, 2, 3], [70, 3, 130])  # longer and shorter than a block
    values = np.random.default_rng(1).random((keys.size, 2))
    offsets = np.searchsorted(keys, np.arange(5)).astype(np.int64)
    out = np.full((4, 2), np.nan)
    emulate_cuda_kernel(
        _sorted_sum_kernel(np.float64),
        offsets,
        values,
        out,
        2,
        grid=4,
        block=_SORTED_SUM_BLOCK,
    )
    np.testing.assert_allclose(out, reference_sum(keys, values, 4, np.float64))


def test_sorted_segment_sum_kernel_is_cached():
    from cunumpy._algorithms import _sorted_sum_kernel

    assert _sorted_sum_kernel(np.float32) is _sorted_sum_kernel(np.float32)
    assert _sorted_sum_kernel(np.float32) is not _sorted_sum_kernel(np.float64)


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.complex128])
def test_gpu_sorted_plan_is_deterministic(dtype):
    import cupy as cp

    keys = np.sort(np.random.default_rng(2).integers(-1, 50, 5000))
    values = np.random.default_rng(3).random((keys.size, 3)).astype(dtype)
    plan = SegmentPlan(cp.asarray(keys), 50)
    assert plan.deterministic
    first = plan.sum(cp.asarray(values))
    for _ in range(3):
        cp.testing.assert_array_equal(plan.sum(cp.asarray(values)), first)
    np.testing.assert_allclose(
        cp.asnumpy(first), reference_sum(keys, values, 50, dtype), rtol=1e-2
    )
    assert not SegmentPlan(cp.asarray(keys[::-1].copy()), 50).deterministic


def test_sorted_segment_sum_on_the_fake_cupy():
    code = """
import numpy as np
import cunumpy as xp
xp.set_backend("cupy")
keys = xp.asarray([-1, 0, 0, 2])
plan = xp.algorithms.SegmentPlan(keys, 3)
assert plan.deterministic
total = plan.sum(xp.asarray([5.0, 1.0, 2.0, 4.0]))
assert xp.is_gpu(total)
np.testing.assert_array_equal(xp.to_numpy(total), [3.0, 0.0, 4.0])
print("ok")
"""
    assert "ok" in run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout


def test_cuda_segment_sum_paths_in_cpu_emulation():
    """The CUDA code of SegmentPlan.sum, with its kernels emulated on the CPU."""
    if emulation_compiler() is None:
        pytest.skip("requires a C++ compiler")
    code = """
import types
import numpy as np
import cunumpy as xp
from cunumpy import _algorithms, _fake_cupy
from cunumpy.kernel_testing import emulate_cuda_kernel
xp.set_backend("cupy")

def on_host_buffers(factory):
    def make(dtype):
        kernel = factory(dtype)
        def launch(*args, **config):
            args = [_fake_cupy.host_buffer(a) if xp.is_gpu(a) else a for a in args]
            emulate_cuda_kernel(kernel, *args, **config)
        return launch
    return make

_algorithms._sorted_sum_kernel = on_host_buffers(_algorithms._sorted_sum_kernel)
_algorithms._sum_kernel = on_host_buffers(_algorithms._sum_kernel)
_algorithms._fake_cupy = types.SimpleNamespace(is_active=lambda: False)
for keys, deterministic in (([-1, 0, 0, 2, 2], True), ([2, 0, -1, 0, 2], False)):
    for dtype in (np.float16, np.float32, np.float64, np.complex128):
        plan = xp.algorithms.SegmentPlan(xp.asarray(keys), 3)
        assert plan.deterministic == deterministic
        values = np.arange(10.0).reshape(5, 2).astype(dtype)
        got = xp.to_numpy(plan.sum(xp.asarray(values)))
        k = np.array(keys)
        expected = np.zeros((3, 2), dtype=dtype)
        np.add.at(expected, k[k >= 0], values[k >= 0])
        np.testing.assert_allclose(got, expected)
print("ok")
"""
    assert "ok" in run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout


@pytest.mark.parametrize("backend", BACKENDS)
def test_sort_by_key_writes_into_out(backend):
    rng = np.random.default_rng(4)
    n = 5000  # above the radix threshold
    with xp.use_backend(backend, strict=True):
        keys = xp.asarray(rng.integers(0, 300, n))
        rows = xp.asarray(rng.random((n, 3)))
        markers = xp.asarray(rng.random((4, n)))
        expected = sort_by_key(keys, rows)
        out = (xp.empty_like(keys), xp.empty(n, dtype=xp.int64), xp.empty_like(rows))
        got = sort_by_key(keys, rows, out=out)
        assert all(a is b for a, b in zip(got, out))
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(xp.to_numpy(a), xp.to_numpy(b))
        expected = sort_by_key(keys, markers, axis=-1)
        buffer = xp.empty_like(markers)
        got = sort_by_key(keys, markers, axis=-1, out=[None, None, buffer])
        assert got[2] is buffer
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(xp.to_numpy(a), xp.to_numpy(b))


def test_sort_by_key_rejects_unsuitable_out():
    keys = np.array([2, 0, 1])
    values = np.arange(3.0)
    readonly = np.empty(3)
    readonly.flags.writeable = False
    for out, match in [
        ((None, None), "3 entries"),
        ((None, None, np.empty(4)), "shape"),
        ((None, None, np.empty(3, dtype=np.float32)), "dtype"),
        ((None, np.empty(3, dtype=np.int32), None), "dtype"),
        ((None, None, np.empty(6)[::2]), "C-contiguous"),
        ((None, None, readonly), "writable"),
        ((None, None, values), "share memory"),
        ((keys, None, None), "share memory"),
    ]:
        with pytest.raises(ValueError, match=match):
            sort_by_key(keys, values, out=out)


def scale(x, out):
    out[:] = 2 * x


def test_resolved_kernels_are_the_host_functions():
    x, out = np.arange(3.0), np.zeros(3)
    kernel = Kernel(scale)
    run = kernel.resolve(x, out)
    assert run is scale
    run(x, out)
    np.testing.assert_array_equal(out, [0, 2, 4])
    assert Kernel(scale, dispatch="arrays").resolve(x, out) is scale
    assert PyccelKernel(scale).resolve(x, out) is scale
    forced = PyccelKernel(scale, use_cupy=True)
    assert forced.resolve(x, out) is forced
    assert Kernel(forced, dispatch="arrays").resolve(x, out) is forced

    implementations = HostImplementations(
        "scale", {"python": lambda: print, "numpy": lambda: scale}
    )
    assert implementations.resolve() is scale
    assert Kernel(implementations).resolve(x, out) is scale


def test_resolved_compiled_host_kernels():
    def compiler(module):
        return SimpleNamespace(sqrt=np.sqrt)

    def failing(module):
        raise RuntimeError("no Pyccel")

    assert CompiledHostKernel("math", "sqrt", compiler).resolve() is np.sqrt
    fallback = CompiledHostKernel("math", "sqrt", failing, fallback=np.sqrt)
    assert fallback.resolve() is np.sqrt
    python = CompiledHostKernel("math", "sqrt", failing)
    assert python.resolve() is python  # warns at the call
    with pytest.warns(RuntimeWarning, match="uncompiled"):
        assert python.resolve()(4.0) == 2.0


class FakeDeviceArray:
    shape = (3,)


def test_resolved_kernels_on_the_device(monkeypatch):
    monkeypatch.setattr(
        dispatch_module, "_is_device_array", lambda a: isinstance(a, FakeDeviceArray)
    )
    cuda = CudaKernel(AXPY, "axpy")
    kernel = Kernel(scale, cuda, dispatch="arrays")
    assert kernel.resolve(FakeDeviceArray(), FakeDeviceArray()) is cuda
    assert kernel.resolve(np.zeros(3), np.zeros(3)) is scale
    monkeypatch.setattr(dispatch_module, "get_backend", lambda: "cupy")
    assert Kernel(scale, cuda).resolve(np.zeros(3), np.zeros(3)) is cuda
    with pytest.raises(NotImplementedError):
        Kernel(scale).resolve(np.zeros(3), np.zeros(3))


def test_bound_launch_under_emulation():
    if emulation_compiler() is None:
        pytest.skip("requires a C++ compiler")
    axpy = CudaKernel(AXPY, "axpy")
    x, y = np.arange(3.0), np.zeros(3)
    launch = axpy.bind(2.0, x, y, 3, n_threads=3)
    assert isinstance(launch, BoundLaunch)
    assert launch.kernel is axpy
    assert "4 arguments" in repr(launch)
    with emulated_launches():
        launch()
        launch.launch()
    np.testing.assert_array_equal(y, [0, 4, 8])


class RecordingRaw:
    def __init__(self):
        self.launches = []

    def __call__(self, grid, block, values, shared_mem=0):
        self.launches.append((grid, block, values, shared_mem))


class RecordingStream:
    device_id = 0

    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def packed_launches(monkeypatch):
    """Launches without a GPU: the packed launch is recorded by a fake raw kernel."""
    raw = RecordingRaw()
    prepared = []

    def prepare(self, args, n_threads, grid, block, shared_mem):
        prepared.append(args)
        if n_threads == 0:
            return None
        return raw, 0, (1, 1, 1), (128, 1, 1), ("packed", *args), shared_mem

    monkeypatch.setattr(CudaKernel, "_prepare_launch", prepare)
    monkeypatch.setattr(CudaKernel, "debug_active", lambda self: False)
    monkeypatch.setattr(_cuda_kernel, "_current_device", lambda: 0)
    return raw, prepared


def test_bound_launch_packs_once(packed_launches, monkeypatch):
    raw, prepared = packed_launches
    launch = CudaKernel(AXPY, "axpy").bind(2.0, "x", "y", 3, n_threads=3)
    assert prepared == []  # packed at the first launch
    launch()
    launch()
    stream = RecordingStream()
    launch(stream=stream)
    assert len(prepared) == 1
    assert len(raw.launches) == 3
    assert raw.launches[0] == ((1, 1, 1), (128, 1, 1), ("packed", 2.0, "x", "y", 3), 0)
    assert stream.entered == 1
    stream.device_id = 1
    with pytest.raises(ValueError, match="another device"):
        launch(stream=stream)
    monkeypatch.setattr(_cuda_kernel, "_current_device", lambda: 1)
    with pytest.raises(ValueError, match="bound on CUDA device 0"):
        launch()


def test_bound_launch_of_an_empty_grid(packed_launches):
    raw, prepared = packed_launches
    launch = CudaKernel(AXPY, "axpy").bind(2.0, "x", "y", 0, n_threads=0)
    launch()
    launch()
    assert len(prepared) == 1
    assert raw.launches == []


def test_bound_launch_calls_the_kernel_in_debug_mode(packed_launches, monkeypatch):
    raw, _ = packed_launches
    calls = []
    monkeypatch.setattr(CudaKernel, "debug_active", lambda self: True)
    monkeypatch.setattr(
        CudaKernel, "__call__", lambda self, *args, **kwargs: calls.append(kwargs)
    )
    CudaKernel(AXPY, "axpy").bind(2.0, "x", "y", 3, n_threads=3)()
    assert calls == [
        {"stream": None, "n_threads": 3, "grid": None, "block": None, "shared_mem": 0}
    ]
    assert raw.launches == []


def test_graph_on_the_host_calls_the_function():
    calls = []
    graph = xp.cuda.capture_graph(lambda a, b=0: calls.append((a, b)), 1, b=2)
    assert calls == [(1, 2)]  # warm-up
    assert not graph.captured
    graph.launch()
    graph()
    assert calls == [(1, 2)] * 3
    assert "host" in repr(graph)
    xp.cuda.capture_graph(calls.append, 5, warmup=False)
    assert calls == [(1, 2)] * 3


def test_graph_capture_on_cuda(monkeypatch):
    events = []

    class Graph:
        def launch(self, stream=None):
            events.append(("launch", stream))

    class Stream:
        def __init__(self, non_blocking):
            assert non_blocking

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def begin_capture(self):
            events.append("begin")

        def end_capture(self):
            events.append("end")
            return Graph()

    device = SimpleNamespace(synchronize=lambda: events.append("sync"))
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(Stream=Stream, Device=lambda: device),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(_graphs, "_real_cupy", lambda: True)

    def step():
        events.append("step")

    graph = xp.cuda.capture_graph(step)
    assert events == ["step", "sync", "begin", "step", "end"]
    assert graph.captured
    assert "CUDA graph" in repr(graph)
    graph.launch("s")
    assert events[-1] == ("launch", "s")

    def failing():
        raise RuntimeError("boom")

    events.clear()
    with pytest.raises(RuntimeError, match="boom"):
        xp.cuda.capture_graph(failing, warmup=False)
    assert events == ["sync", "begin", "end"]  # the capture is ended


def test_real_cupy_detection():
    assert not _graphs._real_cupy()  # NumPy backend


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
def test_graph_and_bound_launch_on_a_gpu():
    import cupy as cp

    axpy = CudaKernel(AXPY, "axpy")
    x, y = cp.arange(3.0), cp.zeros(3)
    launch = axpy.bind(2.0, x, y, 3, n_threads=3)
    graph = xp.cuda.capture_graph(launch)  # warm-up runs it once
    graph.launch()
    launch()
    cp.cuda.Device().synchronize()
    np.testing.assert_array_equal(cp.asnumpy(y), [0, 6, 12])
