"""Tests for `xp.kernels.fuse`: cupy.fuse for CuPy arrays, a plain call otherwise."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _fusion as fusion


def pressure(rho, T, gamma):
    return (gamma - 1.0) * rho * xp.exp(T)


def test_host_arrays_call_the_function():
    fused = xp.kernels.fuse(pressure)
    rho, T = np.full(4, 2.0), np.zeros(4)
    np.testing.assert_array_equal(fused(rho, T, 3.0), pressure(rho, T, 3.0))
    assert fused.__name__ == "pressure" and fused.__wrapped__ is pressure
    assert "fuse" in xp.kernels.__all__


def test_decorator_forms():
    @xp.kernels.fuse
    def double(x):
        return 2.0 * x

    @xp.kernels.fuse(kernel_name="triple_kernel")
    def triple(x):
        return 3.0 * x

    assert double(np.ones(2)).tolist() == [2.0, 2.0]
    assert triple(np.ones(2)).tolist() == [3.0, 3.0]


class FakeDeviceArray:
    def __init__(self, values):
        self.values = np.asarray(values)


def test_device_arrays_use_cupy_fuse_once(monkeypatch):
    created = []

    def fake_cupy_fuse(function, kernel_name):
        created.append(kernel_name)

        def fused(*args, **kwargs):
            assert xp.get_backend() in ("cupy", "numpy")  # numpy: no CuPy installed
            return ("fused", function.__name__, len(args), sorted(kwargs))

        return fused

    monkeypatch.setattr(fusion, "_cupy_fuse", fake_cupy_fuse)
    monkeypatch.setattr(
        fusion,
        "_is_device_array",
        lambda a: isinstance(a, FakeDeviceArray),
    )

    @xp.kernels.fuse(kernel_name="p")
    def p(rho, T, gamma=1.0):
        raise AssertionError("not called with device arrays")

    rho = FakeDeviceArray([1.0])
    assert p(rho, 0.0) == ("fused", "p", 2, [])
    assert p(1.0, 0.0, gamma=rho) == ("fused", "p", 2, ["gamma"])  # keyword array
    assert created == ["p"]  # fused once, reused
    with pytest.raises(AssertionError, match="not called"):
        p(np.ones(1), 0.0)  # host arrays: the function itself


def test_python_scalars_take_the_dtype_of_the_arrays(monkeypatch):
    seen = []
    monkeypatch.setattr(
        fusion,
        "_cupy_fuse",
        lambda function, kernel_name: lambda *a, **k: seen.append((a, k)),
    )
    monkeypatch.setattr(fusion, "_is_device_array", lambda a: hasattr(a, "dtype"))

    p = xp.kernels.fuse(pressure)
    p(np.ones(2), 0.5, 5.0 / 3.0)
    p(np.ones(2, dtype=np.float32), 0.5, gamma=2)
    p(np.ones(2, dtype=np.int64), True, 3)
    (a64, _), (a32, k32), (aint, _) = seen
    assert a64[1].dtype == a64[2].dtype == np.float64 and a64[2] == 5.0 / 3.0
    assert a32[1].dtype == k32["gamma"].dtype == np.float32
    assert aint[1] is True and aint[2].dtype == np.int64  # bools stay Python


def test_fuse_on_gpu():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    fused = xp.kernels.fuse(pressure)
    rho, T = cp.full(1000, 2.0), cp.linspace(0.0, 1.0, 1000)
    with xp.use_backend("cupy"):
        expected = pressure(rho, T, 5.0 / 3.0)
    cp.testing.assert_allclose(fused(rho, T, 5.0 / 3.0), expected, rtol=1e-14)
