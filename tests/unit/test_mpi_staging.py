"""Host staging reuse and lifetime rules, without requiring an MPI installation."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _mpi
from cunumpy.mpi import MPIStaging, mpi_buffer


@pytest.fixture
def device(monkeypatch):
    log = []

    class Device:
        def __init__(self, id=0):
            self.id = id

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def synchronize(self):
            log.append(("device_sync", self.id))

    class Array:
        def __init__(self, data, id=0):
            self.values = np.asarray(data)
            self.shape, self.dtype = self.values.shape, self.values.dtype
            self.flags = self.values.flags
            self.device = Device(id)

        def get(self, out=None):
            log.append(("get", id(out)))
            np.copyto(out, self.values)
            return out

        def set(self, host):
            log.append(("set", id(host)))
            self.values[:] = host

        def __setitem__(self, index, value):
            self.values[index] = value.values

    def allocate_device(shape, dtype):
        log.append(("allocate_device",))
        return Array(np.empty(shape, dtype=dtype))

    stream = SimpleNamespace(synchronize=lambda: log.append(("stream_sync",)))
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=Device, get_current_stream=lambda: stream),
            empty=allocate_device,
        ),
    )
    monkeypatch.setattr(
        _mpi.array_api_compat,
        "is_cupy_array",
        lambda a: isinstance(a, Array),
    )

    def allocate(shape, dtype):
        log.append(("allocate",))
        return np.empty(shape, dtype=dtype)

    monkeypatch.setattr(_mpi, "_pinned_or_host_empty", allocate)
    return Array, log


def test_staging_reuses_storage_counts_copies_and_waits_for_copyback(device):
    Array, log = device
    data = Array([1.0, 2.0])
    staging = MPIStaging(2, np.float64)
    buffers = []
    with xp.profiling.count_transfers() as counter:
        for _ in range(3):
            with staging.buffer(data, recv=True, cuda_aware=False) as host:
                buffers.append(host)
                host += 1
    assert all(host is buffers[0] for host in buffers)
    assert log.count(("allocate",)) == 1
    assert log[-1] == ("stream_sync",)
    assert counter.to_host == counter.to_device == 3
    np.testing.assert_array_equal(data.values, [4, 5])


def test_strided_arrays_reuse_device_packing_and_preserve_unselected_rows(device):
    Array, log = device
    base = np.arange(24.0).reshape(8, 3)
    expected = base.copy()
    expected[::2] += 2
    data = Array(base[::2])
    staging = MPIStaging(data.shape, data.dtype)
    for _ in range(2):
        with staging.buffer(data, recv=True, cuda_aware=False) as host:
            host += 1
    np.testing.assert_array_equal(base, expected)
    assert log.count(("allocate",)) == log.count(("allocate_device",)) == 1


def test_staging_rejects_overlapping_uses_and_recovers_after_exception(device):
    Array, _ = device
    data = Array([1.0, 2.0])
    staging = MPIStaging(2, np.float64)
    with (
        pytest.raises(RuntimeError, match="already in use"),
        mpi_buffer(data, staging=staging, cuda_aware=False),
        staging.buffer(data, cuda_aware=False),
    ):
        pass
    with staging.buffer(data, cuda_aware=False) as host:
        np.testing.assert_array_equal(host, [1, 2])
    with (
        pytest.raises(ValueError, match="shape and dtype"),
        staging.buffer(Array([1.0]), cuda_aware=False),
    ):
        pass
    with (
        pytest.raises(ValueError, match="another CUDA device"),
        staging.buffer(Array([1.0, 2.0], id=1), cuda_aware=False),
    ):
        pass


def test_explicit_producer_dependencies_and_receive_only(device):
    Array, log = device
    data = Array([1.0, 2.0])
    producer = SimpleNamespace(synchronize=lambda: log.append(("producer",)))
    with mpi_buffer(
        data,
        send=False,
        recv=True,
        cuda_aware=False,
        event=producer,
    ) as host:
        host[:] = [9, 10]
    assert ("producer",) in log
    assert not any(entry[0] == "device_sync" for entry in log)
    np.testing.assert_array_equal(data.values, [9, 10])
    log.clear()
    with mpi_buffer(data, cuda_aware=True, stream=producer) as buf:
        assert buf is data
    assert log == [("producer",)]
    with pytest.raises(ValueError, match="only one"):
        _mpi.synchronize_for_mpi(data, stream=producer, event=producer)
    with pytest.raises(TypeError, match="CUDA producer"):
        _mpi.synchronize_for_mpi(data, event=xp.cuda.HostEvent())


def test_default_sync_waits_for_each_device(device):
    Array, log = device
    _mpi.synchronize_for_mpi(Array([1], id=0), Array([2], id=1), None)
    assert sorted(log) == [("device_sync", 0), ("device_sync", 1)]


def test_host_arrays_are_passed_through_and_invalid_staging_is_rejected():
    data = np.zeros(2)
    with MPIStaging(2, data.dtype).buffer(data) as buf:
        assert buf is data
    with pytest.raises(ValueError, match="non-negative"):
        MPIStaging((-1,), float)
    with pytest.raises(TypeError, match="object"):
        MPIStaging(2, object)


@pytest.mark.parametrize("cuda_aware", [False, True])
def test_exchange_device_buffers_stay_open_until_all_requests_finish(
    device, cuda_aware
):
    Array, log = device
    sends = [Array([1.0, 2.0]), Array([3.0, 4.0])]
    receives = [Array([0.0, 0.0]), Array([0.0, 0.0])]
    posted = []

    class Comm:
        def Irecv(self, buf, source, tag):
            posted.append(("recv", source, tag, buf))

            def wait():
                assert len(posted) == 4
                assert not any(entry[0] == "set" for entry in log)
                values = buf.values if cuda_aware else buf
                values[:] = [source, tag]

            return SimpleNamespace(Wait=wait)

        def Isend(self, buf, dest, tag):
            posted.append(("send", dest, tag, buf))

            def wait():
                assert not any(entry[0] == "set" for entry in log)
                values = buf.values if cuda_aware else buf
                np.testing.assert_array_equal(values, sends[dest - 1].values)

            return SimpleNamespace(Wait=wait)

        def Abort(self, code):
            pytest.fail("successful exchange must not abort")

    with xp.profiling.count_transfers() as counter:
        xp.mpi.exchange(
            Comm(),
            sends=((array, i + 1, 10 + i) for i, array in enumerate(sends)),
            receives=((array, i + 1, 20 + i) for i, array in enumerate(receives)),
            cuda_aware=cuda_aware,
        )
    assert [entry[:3] for entry in posted] == [
        ("recv", 1, 20),
        ("recv", 2, 21),
        ("send", 1, 10),
        ("send", 2, 11),
    ]
    for i, array in enumerate(receives):
        np.testing.assert_array_equal(array.values, [i + 1, 20 + i])
    assert counter.to_host == counter.to_device == (0 if cuda_aware else 2)
    if cuda_aware:
        assert posted[0][3] is receives[0]
        assert posted[2][3] is sends[0]


def test_exchange_prepares_every_buffer_before_posting(device):
    Array, log = device
    recv = Array([7.0])
    # A later invalid descriptor must fail before touching the communicator,
    # and unwinding preparation must not copy an uninitialized receive buffer.
    with pytest.raises(ValueError):
        xp.mpi.exchange(
            object(),
            receives=[(recv, 1, 0)],
            sends=[(Array([1.0]), 1)],
            cuda_aware=False,
        )
    np.testing.assert_array_equal(recv.values, [7])
    assert not any(entry[0] == "set" for entry in log)


@pytest.mark.parametrize("failure", ["receive", "send", "wait"])
def test_exchange_aborts_on_communication_failure(device, failure):
    Array, log = device
    events = []

    class Comm:
        def Irecv(self, buf, source, tag):
            if failure == "receive":
                raise RuntimeError("MPI failure")
            return SimpleNamespace(Wait=self.wait)

        def Isend(self, buf, dest, tag):
            if failure == "send":
                raise RuntimeError("MPI failure")
            return SimpleNamespace(Wait=self.wait)

        def wait(self):
            raise RuntimeError("MPI failure")

        def Abort(self, code):
            assert not any(entry[0] == "set" for entry in log)
            events.append(("abort", code))

    with pytest.raises(RuntimeError, match="MPI failure"):
        xp.mpi.exchange(
            Comm(),
            sends=[(Array([1.0]), 1, 0)],
            receives=[(Array([0.0]), 1, 0)],
            cuda_aware=False,
        )
    assert events == [("abort", 1)]
    assert not any(entry[0] == "set" for entry in log)


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_real_device_staging_reuses_host_storage_after_nondefault_producer(layout):
    import cupy as cp

    data = cp.zeros((4, 5))
    if layout == "F":
        data = cp.asfortranarray(data)
    elif layout == "strided":
        data = cp.zeros((8, 5))[::2]
    staging = MPIStaging(data.shape, data.dtype)
    producer = cp.cuda.Stream(non_blocking=True)
    cp.cuda.Device().synchronize()
    first, packed = None, None
    for value in (3, 8):
        with producer:
            data.fill(value)
        with staging.buffer(data, recv=True, cuda_aware=False, stream=producer) as host:
            if first is None:
                first = host
            assert host is first
            np.testing.assert_array_equal(host, np.full((4, 5), value))
            host[:] += 2
        np.testing.assert_array_equal(cp.asnumpy(data), np.full((4, 5), value + 2))
        if layout != "C":
            if packed is None:
                packed = staging._packed
            assert staging._packed is packed
