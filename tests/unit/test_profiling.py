"""Tests for the profiling helpers `nvtx_range` and `timed_region`. The NVTX
calls are checked against a fake ``cupy.cuda.nvtx`` module; the GPU parts
need a GPU."""

import sys
import time
from types import ModuleType

import pytest

import cunumpy as xp
from cunumpy import xp as xp_module


@pytest.fixture
def fake_nvtx(monkeypatch):
    """A fake ``cupy.cuda.nvtx`` recording its calls, with the CuPy backend
    selected (without CuPy installed, `use_backend("cupy")` would fall back
    to NumPy, so the backend name is patched directly)."""
    calls = []

    nvtx = ModuleType("cupy.cuda.nvtx")
    nvtx.RangePush = lambda message, id_color=-1: calls.append(
        ("push", message, id_color)
    )
    nvtx.RangePop = lambda: calls.append(("pop",))
    cuda = ModuleType("cupy.cuda")
    cuda.nvtx = nvtx
    cupy = ModuleType("cupy")
    cupy.cuda = cuda

    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvtx", nvtx)
    monkeypatch.setattr(xp_module.array_backend, "_backend", "cupy")
    # `synchronize()` imports cupy on this backend; make it a no-op instead.
    monkeypatch.setattr(xp_module, "synchronize", lambda: None)
    return calls


# --- NumPy backend: no-ops and timing ---------------------------------------


def test_nvtx_range_is_noop_on_numpy():
    with xp.use_backend("numpy"):
        with xp.nvtx_range("region") as r:
            assert r.name == "region"
        with xp.nvtx_range("colored", color=3):
            pass


def test_nvtx_range_as_decorator():
    @xp.nvtx_range("decorated")
    def add(a, b):
        return a + b

    with xp.use_backend("numpy"):
        assert add(1, 2) == 3
        assert add.__name__ == "add"


def test_nvtx_range_repr():
    assert repr(xp.nvtx_range("r", color=1)) == "nvtx_range(name='r', color=1)"


def test_timed_region_on_numpy():
    with xp.use_backend("numpy"), xp.timed_region("sleep") as timing:
        assert timing.name == "sleep"
        assert timing.elapsed is None
        time.sleep(0.02)

    assert isinstance(timing, xp.Timing)
    assert timing.elapsed >= 0.02
    assert timing.elapsed < 5.0
    assert timing.synced is False


def test_timed_region_without_sync_on_numpy():
    with xp.use_backend("numpy"), xp.timed_region("no sync", sync=False) as timing:
        pass
    assert timing.elapsed >= 0.0
    assert timing.synced is False


def test_timed_region_records_time_on_exception():
    with xp.use_backend("numpy"), pytest.raises(RuntimeError, match="boom"):
        with xp.timed_region("failing") as timing:
            raise RuntimeError("boom")
    assert timing.elapsed is not None
    assert timing.elapsed >= 0.0


# --- fake NVTX: push/pop order --------------------------------------------


def test_nvtx_range_pushes_and_pops(fake_nvtx):
    with xp.nvtx_range("outer"):
        fake_nvtx.append(("body",))
    assert fake_nvtx == [("push", "outer", -1), ("body",), ("pop",)]


def test_nvtx_range_color(fake_nvtx):
    with xp.nvtx_range("colored", color=5):
        pass
    assert fake_nvtx == [("push", "colored", 5), ("pop",)]


def test_nvtx_range_nested_and_reentrant(fake_nvtx):
    outer = xp.nvtx_range("outer")
    with outer:
        with xp.nvtx_range("inner"):
            pass
        with outer:  # same instance re-entered
            pass
    assert fake_nvtx == [
        ("push", "outer", -1),
        ("push", "inner", -1),
        ("pop",),
        ("push", "outer", -1),
        ("pop",),
        ("pop",),
    ]


def test_nvtx_range_pops_on_exception(fake_nvtx):
    with pytest.raises(ValueError), xp.nvtx_range("failing"):
        raise ValueError
    assert fake_nvtx == [("push", "failing", -1), ("pop",)]


def test_nvtx_range_decorator_pushes_and_pops(fake_nvtx):
    @xp.nvtx_range("decorated")
    def work():
        fake_nvtx.append(("body",))

    work()
    work()
    assert fake_nvtx == [("push", "decorated", -1), ("body",), ("pop",)] * 2


def test_timed_region_pushes_nvtx_range_and_syncs(fake_nvtx):
    with xp.timed_region("timed") as timing:
        fake_nvtx.append(("body",))
    assert fake_nvtx == [("push", "timed", -1), ("body",), ("pop",)]
    assert timing.synced is True
    assert timing.elapsed >= 0.0

    del fake_nvtx[:]
    with xp.timed_region("host only", sync=False) as timing:
        pass
    assert fake_nvtx == [("push", "host only", -1), ("pop",)]
    assert timing.synced is False


def test_nvtx_range_without_nvtx_module(monkeypatch):
    # CuPy backend but no NVTX (e.g. a build without it): still a no-op
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvtx", None)  # import fails
    monkeypatch.setattr(xp_module.array_backend, "_backend", "cupy")
    with xp.nvtx_range("no nvtx"):
        pass


# --- GPU --------------------------------------------------------------------


@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy not installed")
def test_nvtx_range_on_gpu():
    import cupy as cp

    with xp.use_backend("cupy"):
        with xp.nvtx_range("gpu region", color=2):
            x = cp.ones(1000)
            x += 1
        xp.synchronize()
    assert float(x[0]) == 2.0


@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy not installed")
def test_timed_region_on_gpu_includes_device_work():
    import cupy as cp

    def work():
        x = cp.ones(4_000_000)
        for _ in range(20):
            x = cp.sin(x) + 1.0
        return x

    with xp.use_backend("cupy"):
        work()  # warm up: allocations, kernel compilation
        xp.synchronize()

        # reference: explicit synchronize-and-time
        start = time.perf_counter()
        work()
        xp.synchronize()
        reference = time.perf_counter() - start

        with xp.timed_region("work") as timing:
            work()
        assert timing.synced is True
        assert cp.cuda.get_current_stream().done

    # a loose check: the synced time is not much shorter than the reference
    assert timing.elapsed >= 0.5 * reference
