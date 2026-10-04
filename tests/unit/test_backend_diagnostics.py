"""Strict selection is transactional; diagnostic reports survive CUDA failures."""

import json
import sys
from types import SimpleNamespace

import pytest

import cunumpy as xp
from cunumpy import xp as backend


def test_strict_failure_preserves_backend_and_namespace(monkeypatch):
    with xp.use_backend("numpy"):
        zeros = xp.zeros
        monkeypatch.setattr(backend, "_CUPY_AVAILABLE_CACHE", False)
        monkeypatch.setattr(backend, "_CUPY_UNAVAILABLE_REASON", "driver mismatch")
        with pytest.raises(RuntimeError, match="driver mismatch"):
            xp.set_backend("cupy", strict=True)
        with (
            pytest.raises(RuntimeError, match="driver mismatch"),
            xp.use_backend("cupy", strict=True),
        ):
            pytest.fail("unavailable backend was entered")
        assert xp.get_backend() == "numpy" and xp.zeros is zeros
        xp.set_backend("cupy")  # preserve the existing fallback
        assert xp.get_backend() == "numpy"
        info = xp.backend_info()
        assert info["cuda_unavailable_reason"] == "driver mismatch"
        assert info["device"] is None
        json.dumps(info)


def test_availability_keeps_exception_details(monkeypatch):
    def unavailable():
        raise RuntimeError("bad driver")

    monkeypatch.setattr(backend, "_CUPY_AVAILABLE_CACHE", None)
    monkeypatch.setattr(backend, "_CUPY_UNAVAILABLE_REASON", None)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(is_available=unavailable))
    assert not backend.cupy_available()
    assert xp.backend_info()["cuda_unavailable_reason"] == "RuntimeError: bad driver"


def test_diagnostics_with_cuda_and_failed_inspection(monkeypatch):
    runtime = SimpleNamespace(
        getDeviceProperties=lambda device: {"name": b"test GPU"},
        getDeviceCount=lambda: 2,
        driverGetVersion=lambda: 13000,
        runtimeGetVersion=lambda: 12000,
    )
    cp = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=1, compute_capability="80"),
            runtime=runtime,
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(backend, "_CUPY_AVAILABLE_CACHE", True)
    info = xp.backend_info()
    assert info["device"]["id"] == 1 and info["device"]["name"] == "test GPU"
    assert info["versions"]["cupy"] == "test"
    assert info["cuda_unavailable_reason"] is None
    json.dumps(info)
    del runtime.getDeviceProperties
    info = xp.backend_info()
    assert info["cupy_available"] and "cuda_inspection_error" in info
    monkeypatch.setitem(sys.modules, "cupy", None)
    info = xp.backend_info()
    assert "ModuleNotFoundError" in info["cuda_inspection_error"]


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
def test_strict_cuda_selection_restores_namespace():
    with xp.use_backend("numpy"):
        zeros = xp.zeros
        with xp.use_backend("cupy", strict=True):
            assert xp.get_backend() == "cupy"
            assert xp.zeros is not zeros
        assert xp.zeros is zeros
