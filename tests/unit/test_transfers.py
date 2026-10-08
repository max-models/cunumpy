"""Tests for `cunumpy.profiling.count_transfers` and `cunumpy.profiling.assert_no_transfers`.

Without a GPU there are no real device arrays, so most tests simulate one:
either by monkeypatching `cunumpy.xp.get_array_backend` (for `to_numpy` and
`to_cupy`), the conversion hooks of `cunumpy._kernel` (for `PyccelKernel`), or
`cunumpy._dispatch.get_backend` (for the `Kernel` fallback). The tests marked
with a GPU requirement exercise the real transfers.
"""

from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _dispatch as dispatch_module
from cunumpy import _kernel as kernel_module
from cunumpy import _transfers as transfers_module
from cunumpy import xp as xp_module
from cunumpy.kernels import Kernel, PyccelKernel
from cunumpy.profiling import TransferCounter, TransferEvent

THIS_FILE = str(Path(__file__))

requires_cupy = pytest.mark.skipif(
    not xp.cupy_available(),
    reason="CuPy not installed or not functional",
)


class _FakeDeviceArray:
    """Stand-in for a CuPy array: holds a NumPy array, copies it on `get()`."""

    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def get(self, order="C"):
        return self.data.copy(order=order)

    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, _FakeDeviceArray) else value


@pytest.fixture
def fake_device(monkeypatch):
    """Make cunumpy treat `_FakeDeviceArray` as a device array, without a GPU."""

    def get_array_backend(array):
        return "cupy" if isinstance(array, _FakeDeviceArray) else "numpy"

    monkeypatch.setattr(xp_module, "get_array_backend", get_array_backend)
    monkeypatch.setattr(xp_module, "_to_cupy", _FakeDeviceArray)
    monkeypatch.setattr(
        kernel_module,
        "_is_device_array",
        lambda v: isinstance(v, _FakeDeviceArray),
    )
    monkeypatch.setattr(kernel_module, "_device_to_host", lambda v: v.get())
    monkeypatch.setattr(kernel_module, "_host_to_device", _FakeDeviceArray)
    return _FakeDeviceArray


# ---------------------------------------------------------------------------
# What is counted, and what is not
# ---------------------------------------------------------------------------


def test_empty_counter():
    with xp.profiling.count_transfers() as counter:
        pass

    assert isinstance(counter, TransferCounter)
    assert counter.total == 0
    assert (counter.to_host, counter.to_device) == (0, 0)
    assert (counter.kernel_conversions, counter.fallbacks) == (0, 0)
    assert counter.events == [] and counter.kernel_conversion_calls == []
    assert counter.report().startswith("0 transfer(s) through cunumpy")
    assert repr(counter) == (
        "TransferCounter(to_host=0, to_device=0, kernel_conversion=0, fallback=0, device_copy=0, sync=0)"
    )


def test_to_numpy_of_numpy_array_is_not_a_transfer():
    with xp.profiling.count_transfers() as counter:
        xp.to_numpy(np.zeros(3))
        xp.to_numpy([1, 2, 3])
        with xp.use_backend("numpy"):
            xp.to_cunumpy(np.zeros(3))

    assert counter.total == 0


def test_no_counter_active_records_nothing(fake_device):
    """Transfers outside a block are not recorded anywhere."""
    xp.to_numpy(fake_device(np.zeros(3)))
    assert transfers_module._ACTIVE == []

    with xp.profiling.count_transfers() as counter:
        pass
    assert counter.total == 0


def test_to_numpy_of_device_array_is_counted(fake_device):
    with xp.profiling.count_transfers() as counter:
        host = xp.to_numpy(fake_device(np.arange(3.0)))

    assert np.array_equal(host, np.arange(3.0))
    assert counter.to_host == 1 and counter.total == 1
    (event,) = counter.events
    assert isinstance(event, TransferEvent)
    assert event.kind == "to_host"
    assert event.description == "to_numpy(shape=(3,), dtype=float64)"


def test_to_cupy_of_host_array_is_counted(fake_device):
    with xp.profiling.count_transfers() as counter:
        xp.to_cupy(np.zeros((2, 2)))
        xp.to_cupy([1, 2])

    assert counter.to_device == 2 and counter.total == 2
    assert counter.events[0].description == "to_cupy(shape=(2, 2), dtype=float64)"
    assert counter.events[1].description == "to_cupy(list)"


def test_to_cupy_of_device_array_is_not_counted(fake_device):
    with xp.profiling.count_transfers() as counter:
        xp.to_cupy(fake_device(np.zeros(2)))

    assert counter.total == 0


def test_to_cunumpy_counts_the_direction_it_delegates_to(fake_device, monkeypatch):
    device = fake_device(np.zeros(2))
    with xp.profiling.count_transfers() as counter, xp.use_backend("numpy"):
        xp.to_cunumpy(device)  # device -> host
        xp.to_cunumpy(np.zeros(2))  # already on the host

    assert counter.to_host == 1 and counter.to_device == 0

    monkeypatch.setattr(xp_module.array_backend, "_backend", "cupy")
    monkeypatch.setattr(xp_module, "cupy_available", lambda: True)
    with xp.profiling.count_transfers() as counter:
        xp.to_cunumpy(np.zeros(2))  # host -> device
        xp.to_cunumpy(device)  # already on the device

    assert counter.to_device == 1 and counter.to_host == 0


# ---------------------------------------------------------------------------
# Call sites and report
# ---------------------------------------------------------------------------


def test_where_points_at_the_caller_outside_cunumpy(fake_device):
    with xp.profiling.count_transfers() as counter:
        xp.to_numpy(fake_device(np.zeros(1)))
        line = _current_line() - 1

    (event,) = counter.events
    assert event.where == f"{THIS_FILE}:{line}"
    assert str(event) == f"{event.where}: to_host: {event.description}"


def test_where_skips_frames_inside_cunumpy(fake_device):
    """`to_cunumpy` calls `to_numpy`; the call site is still the test."""
    with xp.profiling.count_transfers() as counter, xp.use_backend("numpy"):
        xp.to_cunumpy(fake_device(np.zeros(1)))

    (event,) = counter.events
    assert event.where.startswith(THIS_FILE + ":")


def test_report_groups_events_by_kind_and_call_site(fake_device):
    device = fake_device(np.zeros(4))
    with xp.profiling.count_transfers() as counter:
        for _ in range(3):
            xp.to_numpy(device)
        xp.to_cupy(np.zeros(4))

    report = counter.report()
    lines = report.splitlines()
    assert lines[0] == (
        "4 transfer(s) through cunumpy "
        "(3 to_host, 1 to_device, 0 kernel_conversion, 0 fallback, 0 device_copy, 0 sync)"
    )
    assert "  to_host (3):" in lines
    assert "  to_device (1):" in lines
    assert "kernel_conversion (" not in report and "fallback (" not in report
    assert any(
        line.startswith(f"    {THIS_FILE}:")
        and line.endswith("to_numpy(shape=(4,), dtype=float64) (x3)")
        for line in lines
    )
    assert any(
        line.startswith(f"    {THIS_FILE}:")
        and line.endswith("to_cupy(shape=(4,), dtype=float64)")
        for line in lines
    )


# ---------------------------------------------------------------------------
# Nesting and assert_no_transfers
# ---------------------------------------------------------------------------


def test_nested_counters_each_see_their_own_block(fake_device):
    device = fake_device(np.zeros(1))
    with xp.profiling.count_transfers() as outer:
        xp.to_numpy(device)
        with xp.profiling.count_transfers() as inner:
            xp.to_numpy(device)
        xp.to_numpy(device)

    assert inner.total == 1
    assert outer.total == 3
    assert inner.events[0] == outer.events[1]
    assert transfers_module._ACTIVE == []


def test_counter_is_removed_when_the_block_raises(fake_device):
    with pytest.raises(RuntimeError), xp.profiling.count_transfers():
        raise RuntimeError

    assert transfers_module._ACTIVE == []


def test_assert_no_transfers_passes_without_transfers():
    with xp.profiling.assert_no_transfers() as counter:
        xp.to_numpy(np.zeros(3))

    assert counter.total == 0


def test_assert_no_transfers_raises_with_report(fake_device):
    with pytest.raises(AssertionError) as info, xp.profiling.assert_no_transfers():
        xp.to_numpy(fake_device(np.zeros(3)))

    message = str(info.value)
    assert "1 transfer(s) through cunumpy" in message
    assert f"{THIS_FILE}:" in message
    assert "to_numpy(shape=(3,), dtype=float64)" in message
    assert transfers_module._ACTIVE == []


def test_assert_no_transfers_lets_exceptions_through(fake_device):
    with pytest.raises(ValueError, match="inside"), xp.profiling.assert_no_transfers():
        xp.to_numpy(fake_device(np.zeros(3)))
        raise ValueError("inside")


# ---------------------------------------------------------------------------
# PyccelKernel conversions and Kernel fallback
# ---------------------------------------------------------------------------


def test_pyccel_kernel_conversion_is_counted_once_per_call(fake_device):
    def scale(x, y, factor):
        x[:] *= factor
        y[:] *= factor

    x = fake_device(np.ones(3))
    y = fake_device(np.ones(3))
    wrapped = PyccelKernel(scale, use_cupy=True)

    with xp.profiling.count_transfers() as counter:
        wrapped(x, y, 2.0)
        line = _current_line() - 1
        wrapped(x, x, 3.0)  # one array passed twice: converted once

    assert np.array_equal(x.data, np.full(3, 18.0))  # 1 * 2 * 3 * 3 (aliased)
    assert np.array_equal(y.data, np.full(3, 2.0))
    assert counter.kernel_conversions == 2
    assert counter.total == 8  # six copies and two conversion markers
    assert counter.to_host == counter.to_device == 3
    assert counter.bytes_to_host == counter.bytes_to_device == 3 * 3 * 8
    first, second = counter.kernel_conversion_calls
    assert first.kind == "kernel_conversion"
    assert first.description == (
        "PyccelKernel 'scale': 2 device array(s) copied to the host"
    )
    assert first.where == f"{THIS_FILE}:{line}"
    assert second.description == (
        "PyccelKernel 'scale': 1 device array(s) copied to the host"
    )


def test_pyccel_kernel_without_device_arrays_is_not_counted(fake_device):
    wrapped = PyccelKernel(lambda x: None, use_cupy=True)

    with xp.profiling.count_transfers() as counter:
        wrapped(np.zeros(3))  # conversion path, but nothing to convert
        with xp.use_backend("numpy"):
            PyccelKernel(lambda x: None)(np.zeros(3))

    assert counter.total == 0


def test_kernel_fallback_is_counted(monkeypatch):
    """Simulate the CuPy backend for `Kernel` only: the host kernel then runs
    with NumPy arrays, as `PyccelKernel` still sees the NumPy backend."""

    def shift(x, n):
        for i in range(n):
            x[i] += 1.0

    monkeypatch.setattr(dispatch_module, "get_backend", lambda: "cupy")
    kernel = Kernel(shift, missing_cuda="fallback")
    x = np.zeros(2)

    with xp.profiling.count_transfers() as counter:
        line = _current_line() + 2
        with pytest.warns(RuntimeWarning, match="copies its arrays"):
            kernel(x, 2)
        kernel(x, 2)

    assert np.all(x == 2.0)
    assert counter.fallbacks == 2 and counter.total == 2
    event = counter.events[0]
    assert event.kind == "fallback"
    assert event.description == (
        "Kernel 'shift' has no CUDA kernel: host kernel called on the CuPy backend"
    )
    assert event.where == f"{THIS_FILE}:{line}"
    assert "  fallback (2):" in counter.report()


def test_kernel_on_numpy_backend_is_not_a_fallback():
    def shift(x, n):
        x[:n] += 1.0

    kernel = Kernel(shift, missing_cuda="fallback")
    with xp.profiling.count_transfers() as counter, xp.use_backend("numpy"):
        kernel(np.zeros(2), 2)

    assert counter.total == 0


# ---------------------------------------------------------------------------
# Real transfers (GPU only)
# ---------------------------------------------------------------------------


@requires_cupy
def test_real_transfers_are_counted():
    import cupy as cp

    with xp.profiling.count_transfers() as counter:
        device = xp.to_cupy(np.arange(3.0))
        xp.to_cupy(device)  # already on the device
        host = xp.to_numpy(device)
        xp.to_numpy(host)  # already on the host
        with xp.use_backend("cupy"):
            xp.to_cunumpy(host)
            xp.to_cunumpy(device)

    assert isinstance(device, cp.ndarray)
    assert counter.to_device == 2 and counter.to_host == 1
    assert counter.total == 3
    assert all(event.where.startswith(THIS_FILE + ":") for event in counter.events)


@requires_cupy
def test_real_pyccel_kernel_conversion_is_counted():
    import cupy as cp

    def scale(x, factor):
        x[:] *= factor

    x = cp.ones(4)
    with xp.profiling.count_transfers() as counter:
        PyccelKernel(scale)(x, 2.0)

    assert cp.all(x == 2.0)
    assert counter.kernel_conversions == 1 and counter.total == 3
    assert counter.bytes_to_host == counter.bytes_to_device == x.nbytes
    assert counter.kernel_conversion_calls[0].description == (
        "PyccelKernel 'scale': 1 device array(s) copied to the host"
    )


@requires_cupy
def test_real_kernel_fallback_is_counted():
    import cupy as cp

    def scale(x, factor, n):
        for i in range(n):
            x[i] *= factor

    kernel = Kernel(scale, missing_cuda="fallback")
    x = cp.ones(3)
    with (
        xp.profiling.count_transfers() as counter,
        xp.use_backend("cupy"),
        pytest.warns(RuntimeWarning),
    ):
        kernel(x, 2.0, 3)

    assert cp.all(x == 2.0)
    assert counter.fallbacks == 1
    assert counter.kernel_conversions == 1, "the fallback converts on the host"
    assert counter.total == 4  # two copies, conversion and fallback markers


@requires_cupy
def test_assert_no_transfers_on_device_only_work():
    with xp.use_backend("cupy"), xp.profiling.assert_no_transfers():
        values = xp.arange(10, dtype=xp.float64)
        values = values * 2 + 1
        xp.synchronize()


def _current_line() -> int:
    """Line number of the caller."""
    import sys

    return sys._getframe(1).f_lineno
