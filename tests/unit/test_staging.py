"""Tests for `xp.memory.HostStaging`: background copies of device arrays to the host."""

import types
from contextlib import nullcontext

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _staging as staging_module
from cunumpy.memory import HostStaging


def test_host_arrays_are_copied_at_once():
    staging = HostStaging((3,), np.float64)
    a = np.arange(3.0)
    copy = staging.copy(a)
    a[:] = -1.0  # the caller may overwrite its array right away
    assert copy.ready()
    assert copy.result().tolist() == [0.0, 1.0, 2.0]
    assert repr(staging) == "HostStaging(shape=(3,), dtype=float64, buffers=2)"


def test_shape_dtype_and_buffers_are_checked():
    staging = HostStaging(4, np.float32)
    with pytest.raises(ValueError, match="got an array of shape"):
        staging.copy(np.zeros(3, np.float32))
    with pytest.raises(ValueError, match="dtype float64"):
        staging.copy(np.zeros(4))
    with pytest.raises(ValueError, match="at least 1"):
        HostStaging(4, np.float32, buffers=0)


def test_stale_results_raise():
    staging = HostStaging((2,), np.float64, buffers=2)
    first = staging.copy(np.array([1.0, 1.0]))
    second = staging.copy(np.array([2.0, 2.0]))
    assert first.result().tolist() == [1.0, 1.0]
    third = staging.copy(np.array([3.0, 3.0]))  # reuses the first buffer
    with pytest.raises(RuntimeError, match="was reused"):
        first.result()
    with pytest.raises(RuntimeError, match="was reused"):
        first.ready()
    assert second.result().tolist() == [2.0, 2.0]
    assert third.result().tolist() == [3.0, 3.0]


class DeviceArray(np.ndarray):
    """Stands for a CuPy array; `get(stream, out)` is the asynchronous copy."""

    log = None

    @property
    def device(self):
        return types.SimpleNamespace(id=getattr(self, "_device_id", 0))

    def get(self, stream=None, out=None, blocking=True):
        assert blocking is False  # an asynchronous copy
        DeviceArray.log.append(("get", stream.name))
        out[...] = np.asarray(self)
        return out


class FakeEvent:
    def __init__(self, name):
        self.name = name
        self.done = False

    def synchronize(self):
        DeviceArray.log.append(("synchronize", self.name))
        self.done = True


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.device_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def record(self):
        self.count += 1
        event = FakeEvent(f"{self.name}#{self.count}")
        DeviceArray.log.append(("record", event.name))
        return event

    def wait_event(self, event):
        DeviceArray.log.append(("wait", self.name, event.name))


@pytest.fixture
def fake_device(monkeypatch):
    DeviceArray.log = []
    current = FakeStream("compute")
    cuda = types.SimpleNamespace(
        Stream=lambda non_blocking=False: FakeStream("staging"),
        get_current_stream=lambda: current,
        Device=lambda device: nullcontext(),
        runtime=types.SimpleNamespace(getDevice=lambda: 0),
    )

    def empty(shape, dtype):
        DeviceArray.log.append(("snapshot alloc",))
        return np.empty(shape, dtype).view(DeviceArray)

    fake = types.SimpleNamespace(cuda=cuda, empty=empty)
    monkeypatch.setattr(staging_module, "_cupy", lambda: fake)
    monkeypatch.setattr(
        staging_module,
        "_empty_pinned",
        lambda s, dtype: np.empty(s, dtype),
    )
    monkeypatch.setattr(
        staging_module,
        "_is_device_array",
        lambda a: isinstance(a, DeviceArray),
    )
    return DeviceArray.log


def test_device_copies_snapshot_then_copy_on_their_own_stream(fake_device):
    log = fake_device
    staging = HostStaging((3,), np.float64, buffers=2)
    rho = np.array([1.0, 2.0, 3.0]).view(DeviceArray)
    copy = staging.copy(rho)
    rho[...] = 0.0  # the next step overwrites the array: the snapshot keeps the data
    assert log == [
        ("snapshot alloc",),
        ("record", "compute#1"),  # after the producer's kernels and the snapshot
        ("wait", "staging", "compute#1"),
        ("get", "staging"),  # the copy to the host, on the staging stream
        ("record", "staging#1"),
    ]
    assert not copy.ready()
    assert copy.result().tolist() == [1.0, 2.0, 3.0]
    assert ("synchronize", "staging#1") in log


def test_a_buffer_is_reused_only_after_its_copy_finished(fake_device):
    log = fake_device
    staging = HostStaging((1,), np.float64, buffers=2)
    a = np.array([5.0]).view(DeviceArray)
    staging.copy(a)
    staging.copy(a)
    assert not any(entry[0] == "synchronize" for entry in log)  # two in flight
    staging.copy(a)  # the first buffer again: waits for its first copy
    assert ("synchronize", "staging#1") in log
    staging.synchronize()
    assert ("synchronize", "staging#2") in log


def test_copies_are_counted_as_transfers(fake_device):
    staging = HostStaging((2,), np.float64)
    with xp.profiling.count_transfers() as counter:
        staging.copy(np.zeros(2).view(DeviceArray))
    assert counter.total == 1
    assert counter.bytes_to_host == 16


def test_explicit_producer_stream_and_event_order_snapshot(fake_device):
    log = fake_device
    source = np.ones(3).view(DeviceArray)
    producer = FakeStream("producer")
    staging = HostStaging(3, source.dtype)
    staging.copy(source, stream=producer)
    assert ("record", "producer#1") in log
    log.clear()
    event = FakeEvent("produced")
    staging.copy(source, event=event)
    assert log.index(("wait", "compute", "produced")) < log.index(
        ("record", "compute#1")
    )
    assert ("synchronize", "produced") not in log
    with pytest.raises(ValueError, match="only one"):
        staging.copy(source, stream=producer, event=event)


def test_cpu_initialized_staging_upgrades_to_pinned_and_preserves_old_copy(
    fake_device, monkeypatch
):
    allocations = []

    def pinned(shape, dtype):
        allocations.append(shape)
        return np.empty(shape, dtype)

    monkeypatch.setattr(staging_module, "_empty_pinned", pinned)
    staging = HostStaging(2, np.float64)
    old = staging.copy(np.array([1.0, 2.0]))
    device_copy = staging.copy(np.array([3.0, 4.0]).view(DeviceArray))
    assert len(allocations) == staging.buffers
    np.testing.assert_array_equal(old.result(), [1.0, 2.0])
    np.testing.assert_array_equal(device_copy.result(), [3.0, 4.0])
    assert old.result() is not device_copy.result()


def test_device_binding_is_checked_before_reusing_any_buffer(fake_device, monkeypatch):
    staging = HostStaging(2, np.float64)
    source = np.ones(2).view(DeviceArray)
    copy = staging.copy(source)
    source._device_id = 1
    with pytest.raises(ValueError, match="device current"):
        staging.copy(source)
    monkeypatch.setattr(staging_module._cupy().cuda.runtime, "getDevice", lambda: 1)
    with pytest.raises(ValueError, match="bound to another"):
        staging.copy(source)
    assert copy.result().tolist() == [1.0, 1.0]


def test_wrong_stream_and_host_dependencies_are_rejected(fake_device):
    staging = HostStaging(2, np.float64)
    source = np.ones(2).view(DeviceArray)
    stream = FakeStream("wrong")
    stream.device_id = 1
    with pytest.raises(ValueError, match="stream belongs"):
        staging.copy(source, stream=stream)
    with pytest.raises(TypeError, match="CUDA producer"):
        staging.copy(source, event=xp.cuda.HostEvent())


def test_failed_copy_keeps_completion_before_slot_reuse(fake_device, monkeypatch):
    staging = HostStaging(1, np.float64, buffers=1)
    source = np.ones(1).view(DeviceArray)
    original = DeviceArray.get

    def fail(self, **kwargs):
        raise RuntimeError("copy failed after enqueue")

    monkeypatch.setattr(DeviceArray, "get", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        staging.copy(source)
    monkeypatch.setattr(DeviceArray, "get", original)
    result = staging.copy(source)
    assert ("synchronize", "staging#1") in fake_device
    assert result.result().tolist() == [1.0]


def test_host_copy_waits_for_previous_device_slot(fake_device):
    staging = HostStaging(1, np.float64, buffers=1)
    old = staging.copy(np.ones(1).view(DeviceArray))
    result = staging.copy(np.array([2.0]))
    assert ("synchronize", "staging#1") in fake_device
    with pytest.raises(RuntimeError, match="reused"):
        old.result()
    assert result.result().tolist() == [2.0]


@pytest.mark.parametrize("shape,dtype", [((-1,), float), ((2,), object)])
def test_invalid_staging_storage(shape, dtype):
    with pytest.raises((ValueError, TypeError)):
        HostStaging(shape, dtype)


def test_on_gpu():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    staging = HostStaging((1000,), np.float64)
    data = cp.arange(1000.0)
    copy = staging.copy(data)
    data *= 0.0  # overwritten right away on the compute stream
    np.testing.assert_array_equal(copy.result(), np.arange(1000.0))


@pytest.mark.parametrize("dependency", ["stream", "event"])
def test_nondefault_producer_on_gpu(dependency):
    if not xp.cupy_available():
        pytest.skip("requires CUDA")
    import cupy as cp

    source = cp.zeros(1000)
    cp.cuda.get_current_stream().synchronize()
    producer = cp.cuda.Stream(non_blocking=True)
    produced = cp.cuda.Event(disable_timing=True)
    with producer:
        source.fill(7.0)
        produced.record()
    staging = HostStaging(source.shape, source.dtype, buffers=1)
    host_copy = staging.copy(np.full(1000, 2.0))
    copy = staging.copy(
        source, **{dependency: producer if dependency == "stream" else produced}
    )
    np.testing.assert_array_equal(host_copy.result(), np.full(1000, 2.0))
    np.testing.assert_array_equal(copy.result(), np.full(1000, 7.0))
    # Copy completion makes the source safe to overwrite on another stream.
    source.fill(9.0)
    cp.cuda.get_current_stream().synchronize()
    again = staging.copy(source)
    with pytest.raises(RuntimeError, match="reused"):
        copy.result()
    np.testing.assert_array_equal(again.result(), np.full(1000, 9.0))


def test_staging_storage_description_cannot_change():
    staging = HostStaging(2, np.float64)
    with pytest.raises(AttributeError):
        staging.shape = (4,)
    with pytest.raises(AttributeError):
        staging.dtype = np.dtype(np.float32)


def test_staged_result_restores_callers_device_on_gpu():
    if not xp.cupy_available():
        pytest.skip("requires CUDA")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two devices")
    with cp.cuda.Device(0):
        staging = HostStaging(3, np.float64)
        copy = staging.copy(cp.arange(3.0))
    with cp.cuda.Device(1):
        np.testing.assert_array_equal(copy.result(), np.arange(3.0))
        assert cp.cuda.runtime.getDevice() == 1
        with pytest.raises(ValueError, match="bound to another"):
            staging.copy(cp.zeros(3))
