"""host_call, evaluate_on_host and setup_on_host, with a stand-in for device arrays."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _host


class FakeDevice:
    """Stands in for a CuPy array: the host helpers only convert it."""

    def __init__(self, array):
        self.array = np.asarray(array)


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setattr(_host.xp, "is_gpu", lambda a: isinstance(a, FakeDevice))
    monkeypatch.setattr(
        _host.xp, "to_numpy", lambda a: a.array if isinstance(a, FakeDevice) else a
    )
    monkeypatch.setattr(_host.xp, "to_cupy", FakeDevice)


def test_host_arrays_are_a_plain_call():
    x = np.arange(3.0)
    out = xp.host_call(lambda a, scale: a * scale, x, scale=2.0)
    np.testing.assert_array_equal(out, 2 * x)


def test_device_arguments_run_on_numpy_and_return_on_device(device):
    seen = {}

    def fun(a, b, scale=1.0):
        seen["backend"] = xp.get_backend()
        seen["types"] = (type(a), type(b[0]))
        return a + b[0], 3.0

    a, b = FakeDevice([1.0, 2.0]), [FakeDevice([10.0, 20.0])]
    arr, scalar = xp.host_call(fun, a, b, scale=2.0)
    assert seen["backend"] == "numpy" and seen["types"] == (np.ndarray, np.ndarray)
    assert isinstance(arr, FakeDevice) and scalar == 3.0
    np.testing.assert_array_equal(arr.array, [11.0, 22.0])


def test_device_keyword_argument_is_detected(device):
    out = xp.host_call(lambda x=None: x * 2, x=FakeDevice([1.0]))
    assert isinstance(out, FakeDevice)


def test_evaluate_on_host_binds_self(device):
    class Model:
        offset = 1.0

        @xp.evaluate_on_host
        def f(self, x):
            """Docstring."""
            assert isinstance(x, np.ndarray)
            return x + self.offset

    assert Model.f.__doc__ == "Docstring."
    out = Model().f(FakeDevice([1.0]))
    np.testing.assert_array_equal(out.array, [2.0])


def test_setup_on_host_runs_init_on_numpy_backend():
    class Model:
        @xp.setup_on_host
        def __init__(self, n):
            self.backend = xp.get_backend()
            self.data = xp.arange(n)

    model = Model(3)
    assert model.backend == "numpy" and isinstance(model.data, np.ndarray)


def test_copies_are_counted(monkeypatch):
    """With the real conversions, only to_numpy/to_cupy on device arrays count."""
    with xp.profiling.count_transfers() as counter:
        xp.host_call(lambda a: a + 1, np.zeros(2))
    assert counter.total == 0
