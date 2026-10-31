"""Tests for optional framework interoperability."""

import sys
import types

import numpy as np
import pytest

from cunumpy.interop import _common, dlpack
from cunumpy.interop._common import require, target_backend
from cunumpy.profiling import count_transfers


def test_dlpack_helpers_use_array_protocol(monkeypatch):
    class Source:
        def __dlpack__(self):
            return "capsule"

    assert dlpack.to_dlpack(Source()) == "capsule"

    target = types.SimpleNamespace(
        from_dlpack=lambda source: np.array([source.__dlpack__()]),
    )
    monkeypatch.setattr("cunumpy.interop.dlpack.target_backend", lambda backend: target)
    result = dlpack.from_dlpack(Source())
    np.testing.assert_array_equal(result, ["capsule"])


def test_dlpack_rejects_objects_without_protocol():
    with pytest.raises(TypeError, match="DLPack protocol"):
        dlpack.to_dlpack(object())


def test_optional_dependency_error_is_actionable():
    with pytest.raises(ImportError, match="cunumpy\\[not-installed-framework\\]"):
        require("not-installed-framework")


def test_target_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be"):
        target_backend("jax")


class _Device:
    """An array on a fake CUDA device, with a DLPack device and a pointer."""

    shape, dtype, size = (3,), np.dtype("float64"), 3

    def __init__(self, ptr=1000, device=(2, 0)):
        self.ptr, self.device = ptr, device

    def __dlpack_device__(self):
        return self.device

    @property
    def __cuda_array_interface__(self):
        return {"data": (self.ptr, False)}


def test_memory_reads_the_dlpack_device():
    assert _common.memory(np.ones(2)) == _common.HOST
    assert _common.memory([1.0]) == _common.HOST
    assert _common.memory(_Device()) == (2, 0)
    assert _common.memory(_Device(device=(8, 0))) == _common.HOST  # Apple GPU

    class Broken:
        def __dlpack_device__(self):
            raise RuntimeError("no device")

    assert _common.memory(Broken()) == _common.HOST


def test_describe_memory():
    assert _common.describe_memory(_common.HOST) == "the host"
    assert _common.describe_memory((2, 1)) == "CUDA device 1"
    assert _common.describe_memory((99, 0)) == "DLPack device type 99 device 0"


def test_pointer_and_shared_memory():
    a = np.arange(4.0)
    assert _common.pointer(a) == a.ctypes.data
    assert _common.pointer(_Device(ptr=7)) == 7
    assert _common.pointer(memoryview(bytearray(4))) is not None
    assert _common.pointer(object()) is None
    assert _common.shares_memory(a, a)
    assert _common.shares_memory(a, a[:])
    assert not _common.shares_memory(a, a.copy())
    assert _common.shares_memory(a, np.ones(0))
    assert not _common.shares_memory(object(), a)


def test_check_copy_rejects_other_values():
    for copy in (None, True, False):
        _common.check_copy(copy)
    with pytest.raises(ValueError, match="copy must be"):
        _common.check_copy("never")


def test_dtype_name_is_framework_neutral():
    assert _common.dtype_name(np.float32) == "float32"
    assert _common.dtype_name("float64") == "float64"

    class TorchLike:
        def __str__(self):
            return "torch.float64"

    assert _common.dtype_name(TorchLike()) == "float64"


def test_check_dtype_raises_on_an_unrequested_cast():
    a = np.ones(2)
    _common.check_dtype(a, a, None, "X", "advice")
    _common.check_dtype(a, a.astype(np.float32), np.float32, "X", "advice")
    _common.check_dtype([1.0], a, None, "X", "advice")  # no source dtype
    with pytest.raises(TypeError, match="X would turn float64 into float32; ask"):
        _common.check_dtype(a, a.astype(np.float32), None, "X", "ask")


def test_finish_applies_the_copy_rule_and_records_copies():
    a = np.arange(3.0)
    with count_transfers() as counter:
        assert _common.finish(a, a, copy=None, target="X", duplicate=np.copy) is a
        assert counter.total == 0
        copied = _common.finish(a, a, copy=True, target="X", duplicate=np.copy)
        assert copied is not a and counter.count("device_copy") == 1
        _common.finish(a, a.copy(), copy=None, target="X", duplicate=np.copy)
        assert counter.count("device_copy") == 2
    with pytest.raises(ValueError, match="to X needs a copy"):
        _common.finish(a, a.copy(), copy=False, target="X", duplicate=np.copy)


def test_record_tells_host_and_device_copies_apart():
    host, device = np.ones(3), _Device()
    with count_transfers() as counter:
        _common.record(host, device)
        _common.record(device, host)
        _common.record(device, _Device(ptr=2))
    assert [event.kind for event in counter.events] == [
        "to_device",
        "to_host",
        "device_copy",
    ]
    assert counter.bytes("to_device") == 24
    _common.record(host, device)  # no counter active: nothing to do


def test_to_backend_numpy_shares_host_memory():
    a = np.arange(3.0)
    with count_transfers() as counter:
        result = _common.to_backend(
            a, "numpy", dtype=None, copy=False, host_array=np.asarray
        )
    assert np.shares_memory(result, a) and counter.total == 0
    cast = _common.to_backend(
        a, "numpy", dtype=np.float32, copy=None, host_array=np.asarray
    )
    assert cast.dtype == np.float32
    with pytest.raises(ValueError, match="needs a copy"):
        _common.to_backend(
            a, "numpy", dtype=np.float32, copy=False, host_array=np.asarray
        )


def test_to_backend_numpy_falls_back_to_the_host_array():
    a = np.arange(3.0)
    result = _common.to_backend(
        [0.0, 1.0, 2.0], "numpy", dtype=None, copy=None, host_array=lambda _: a
    )
    assert result is a


def test_to_backend_numpy_copies_a_device_array():
    device = _Device()
    with count_transfers() as counter:
        result = _common.to_backend(
            device, "numpy", dtype=None, copy=None, host_array=lambda _: np.ones(3)
        )
    np.testing.assert_array_equal(result, np.ones(3))
    assert counter.count("to_host") == 1
    with pytest.raises(ValueError, match="device to host"):
        _common.to_backend(
            device, "numpy", dtype=None, copy=False, host_array=lambda _: np.ones(3)
        )


def test_to_backend_cupy_paths(monkeypatch):
    received = []

    class Target:
        @staticmethod
        def asarray(array):
            received.append(("asarray", array))
            return _Device(ptr=5)

        @staticmethod
        def from_dlpack(source):
            if source.device != (2, 0):
                raise BufferError("wrong device")
            received.append(("from_dlpack", source))
            return source

    monkeypatch.setattr(_common, "target_backend", lambda backend: Target)
    a = np.ones(3)
    with count_transfers() as counter:
        _common.to_backend(a, "cupy", dtype=None, copy=None, host_array=np.asarray)
    assert received[-1][0] == "asarray" and counter.count("to_device") == 1
    with pytest.raises(ValueError, match="host to device"):
        _common.to_backend(a, "cupy", dtype=None, copy=False, host_array=np.asarray)

    device = _Device()
    result = _common.to_backend(device, "cupy", dtype=None, copy=False, host_array=None)
    assert result is device
    with pytest.raises(ValueError, match="CuPy cannot take a _Device on ROCm device 0"):
        _common.to_backend(
            _Device(device=(10, 0)), "cupy", dtype=None, copy=None, host_array=None
        )


def test_from_dlpack_passes_copy_on():
    calls = []
    target = types.SimpleNamespace(
        from_dlpack=lambda source, **kwargs: calls.append(kwargs) or source
    )
    a = np.ones(2)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("cunumpy.interop.dlpack.target_backend", lambda backend: target)
        assert dlpack.from_dlpack(a) is a
        dlpack.from_dlpack(a, copy=False)
    assert calls == [{}, {"copy": False}]
    with pytest.raises(ValueError, match="copy must be"):
        dlpack.from_dlpack(a, copy=1)
    assert dlpack.from_dlpack(a, copy=False).ctypes.data == a.ctypes.data


def test_from_dlpack_rejects_a_target_without_dlpack(monkeypatch):
    monkeypatch.setattr(
        "cunumpy.interop.dlpack.target_backend", lambda backend: types.SimpleNamespace()
    )
    with pytest.raises(TypeError, match="does not provide from_dlpack"):
        dlpack.from_dlpack(np.ones(2))


def test_target_backend_cupy_without_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="CuPy interoperability needs CuPy"):
        target_backend("cupy")


def test_device_error_names_the_device():
    error = _common.device_error(_Device(), "JAX", RuntimeError("boom"))
    assert "JAX cannot take a _Device on CUDA device 0 (boom)" in str(error)
