"""Physical copies, fallback markers, and retained-buffer ownership without CUDA."""

import sys
import types

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _mirror as mirrors
from cunumpy.memory import DeviceMirror


@pytest.fixture
def device(monkeypatch):
    cp = types.SimpleNamespace(current=0, copies=[], waits=[])

    class Array:
        def __init__(self, data):
            self.array = np.asarray(data)
            self.data = types.SimpleNamespace(ptr=self.array.ctypes.data)
            self.device = types.SimpleNamespace(id=cp.current)
            self.shape, self.dtype, self.flags = (
                self.array.shape,
                self.array.dtype,
                self.array.flags,
            )
            self.ndim = self.array.ndim

        def set(self, host):
            cp.copies.append("set")
            np.copyto(self.array, host)

        def get(self, out=None, stream=None):
            cp.copies.append(("get", stream))
            if out is None:
                return self.array.copy()
            np.copyto(out, self.array)
            return out

        def fill(self, value):
            self.array.fill(value)

        def __setitem__(self, index, value):
            self.array[index] = value.array if isinstance(value, Array) else value

    def asarray(value, dtype=None):
        if isinstance(value, Array):
            if dtype is None or value.dtype == np.dtype(dtype):
                return value
            value = value.array
        return Array(np.array(value, dtype=dtype, copy=True))

    def contiguous(value, dtype=None):
        value = asarray(value, dtype=dtype)
        return (
            value
            if value.flags.c_contiguous
            else Array(np.ascontiguousarray(value.array))
        )

    cp.ndarray = Array
    cp.asarray = asarray
    cp.ascontiguousarray = contiguous
    cp.zeros = lambda shape, dtype: Array(np.zeros(shape, dtype=dtype))
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDevice=lambda: cp.current),
        get_current_stream=lambda: types.SimpleNamespace(wait_event=cp.waits.append),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(xp.xp.array_backend, "_backend", "cupy")
    monkeypatch.setattr(xp.xp, "cupy_available", lambda: True)
    monkeypatch.setattr(mirrors, "_cupy_backend", lambda: xp.get_backend() == "cupy")
    monkeypatch.setattr(xp.xp, "_to_cupy", asarray)
    monkeypatch.setattr(
        xp.xp,
        "get_array_backend",
        lambda a: "cupy" if isinstance(a, Array) else "numpy",
    )
    return cp


def test_mirror_initialization_and_refreshes_count_each_copy_once(device):
    host = np.arange(4.0)
    mirror = DeviceMirror(host)
    with xp.profiling.count_transfers() as counter:
        mirror.to_device()
        mirror.to_device()
        mirror.to_host()
    assert counter.to_device == 2 and counter.to_host == 1
    assert counter.bytes_to_device == 2 * host.nbytes
    assert counter.bytes_to_host == host.nbytes
    assert [e.nbytes for e in counter.events] == [host.nbytes] * 3
    assert mirror.host is host


def test_mirror_refresh_breaks_no_transfer_assertion(device):
    mirror = DeviceMirror(np.ones(2))
    mirror.to_device()
    with (
        pytest.raises(AssertionError, match="DeviceMirror.to_device"),
        xp.profiling.assert_no_transfers(),
    ):
        mirror.to_device()
    with (
        pytest.raises(AssertionError, match="DeviceMirror.to_host"),
        xp.profiling.assert_no_transfers(),
    ):
        mirror.to_host()


def test_zero_allocates_without_transferring_initial_host_contents(device):
    mirror = DeviceMirror(np.ones(3))
    with xp.profiling.assert_no_transfers() as counter:
        mirror.zero()
        assert mirror.device is not mirror.host
    assert counter.total == 0
    np.testing.assert_array_equal(mirror.device.array, np.zeros(3))


@pytest.mark.parametrize("operation", ["device", "to_device", "to_host", "zero"])
def test_mirror_rejects_another_current_device_before_touching_storage(
    device, operation
):
    mirror = DeviceMirror(np.ones(3))
    mirror.to_device()
    device.current = 1
    with pytest.raises(ValueError, match="bound to CUDA device 0"):
        getattr(mirror, operation)() if operation != "device" else mirror.device
    assert device.copies == []


def test_mirror_explicit_dependencies_and_wrong_stream(device):
    mirror = DeviceMirror(np.ones(2))
    mirror.to_device()
    producer = types.SimpleNamespace(device_id=0)
    mirror.to_host(stream=producer)
    assert device.copies[-1] == ("get", producer)
    event = object()
    mirror.to_host(event=event)
    assert device.waits == [event]
    with pytest.raises(ValueError, match="only one"):
        mirror.to_host(stream=producer, event=event)
    with pytest.raises(ValueError, match="another device"):
        mirror.to_host(stream=types.SimpleNamespace(device_id=1))
    with pytest.raises(TypeError, match="CUDA producer"):
        mirror.to_host(event=xp.cuda.HostEvent())


def test_cpu_backend_refresh_does_not_touch_retained_device_buffer(device):
    mirror = DeviceMirror(np.ones(2))
    mirror.to_device()
    with xp.use_backend("numpy"), xp.profiling.assert_no_transfers():
        assert mirror.device is mirror.host
        mirror.zero()
        mirror.to_device().to_host()
    assert device.copies == []
    mirror.to_device()
    np.testing.assert_array_equal(mirror.device.array, [0.0, 0.0])


def test_host_argument_conversion_counts_bytes_and_identity_does_not(device):
    with xp.profiling.count_transfers() as counter:
        array = xp.as_device_array([1, 2, 3], dtype=np.float32)
        assert xp.as_device_array(array, dtype=np.float32) is array
    assert counter.to_device == 1 and counter.bytes_to_device == 12
    with (
        pytest.raises(AssertionError, match="as_device_array"),
        xp.profiling.assert_no_transfers(),
    ):
        xp.as_device_array(np.ones(2))


def test_device_only_dtype_and_layout_copies_are_distinguished(device):
    source = device.ndarray(np.zeros((4, 3))[:, ::2])
    with xp.profiling.assert_no_transfers() as counter:
        packed = xp.as_device_array(source)
        cast = xp.as_device_array(packed, dtype=np.float32)
    assert packed.flags.c_contiguous and cast.dtype == np.float32
    assert counter.to_host == counter.to_device == 0
    assert counter.device_copies == 2
    assert counter.bytes("device_copy") == packed.array.nbytes + cast.array.nbytes


def test_failed_transfer_is_not_recorded(device, monkeypatch):
    def fail(value):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(xp.xp, "_to_cupy", fail)
    with (
        xp.profiling.count_transfers() as counter,
        pytest.raises(RuntimeError, match="copy failed"),
    ):
        xp.to_cupy(np.zeros(2))
    assert counter.total == 0


def test_kernel_output_counts_host_back_to_device_copy(device):
    out = device.ndarray(np.zeros(4))
    with (
        xp.profiling.count_transfers() as counter,
        xp.kernels.kernel_output(out, like=np.zeros(4)) as buffer,
    ):
        buffer[:] = 3.0
    assert counter.to_host == counter.to_device == 1
    assert counter.bytes_to_host == counter.bytes_to_device == out.array.nbytes
    np.testing.assert_array_equal(out.array, np.full(4, 3.0))


def test_interoperable_device_reference_does_not_report_host_transfer(device):
    array = device.ndarray(np.ones(3))

    class Exporter:
        def __init__(self):
            self.__cuda_array_interface__ = {"data": (array.data.ptr, False)}

    exported = Exporter()
    device.asarray = lambda value, dtype=None: array
    with xp.profiling.assert_no_transfers() as counter:
        assert xp.as_device_array(exported) is array
    assert counter.total == 0


@pytest.mark.parametrize("helper", ["as_device_array", "as_kernel_array"])
def test_reference_only_layout_view_is_not_counted_as_copy(device, helper):
    scalar = device.ndarray(np.array(3.0))
    view = device.ndarray(scalar.array.reshape(1))
    device.ascontiguousarray = lambda value, dtype=None: view
    if helper == "as_device_array":
        # An interoperable exporter takes the conversion path. The resulting
        # one-dimensional view still references the original scalar's storage.
        exported = types.SimpleNamespace(
            __cuda_array_interface__={"data": (scalar.data.ptr, False)}
        )
        device.asarray = lambda value, dtype=None: scalar
        convert, value, kwargs = xp.as_device_array, exported, {}
    else:
        convert, value, kwargs = xp.kernels.as_kernel_array, scalar, {"like": scalar}
    with xp.profiling.count_transfers() as counter:
        assert convert(value, **kwargs) is view
    assert counter.total == 0
