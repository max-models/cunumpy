"""Tests for `xp.memory.HostStaging`: background copies of device arrays to the host."""

import types

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import staging as staging_module
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
    )

    def empty(shape, dtype):
        DeviceArray.log.append(("snapshot alloc",))
        return np.empty(shape, dtype).view(DeviceArray)

    fake = types.SimpleNamespace(cuda=cuda, empty=empty)
    monkeypatch.setattr(staging_module, "_cupy", lambda: fake)
    monkeypatch.setattr(
        staging_module, "_empty_pinned", lambda s, dtype: np.empty(s, dtype)
    )
    monkeypatch.setattr(
        staging_module, "_is_device_array", lambda a: isinstance(a, DeviceArray)
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


def test_on_gpu():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    staging = HostStaging((1000,), np.float64)
    data = cp.arange(1000.0)
    copy = staging.copy(data)
    data *= 0.0  # overwritten right away on the compute stream
    np.testing.assert_array_equal(copy.result(), np.arange(1000.0))
