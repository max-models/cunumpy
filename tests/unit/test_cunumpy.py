import sys

import numpy as np
import pytest

import cunumpy as xp
import cunumpy.xp as cxp  # the internal submodule, to inspect array_backend directly


def test_to_numpy():
    arr = xp.array([1, 2, 3])
    # Even if it's already numpy, to_numpy should work
    arr_np = xp.to_numpy(arr)
    assert isinstance(arr_np, np.ndarray)
    assert np.array_equal(arr_np, [1, 2, 3])


def test_to_cunumpy():
    arr = np.array([1, 2, 3])
    arr_xp = xp.to_cunumpy(arr)
    # Backend is numpy in tests usually
    assert isinstance(arr_xp, (np.ndarray, xp.ndarray))


def test_get_backend_and_is_gpu_cpu():
    arr = np.array([1, 2, 3])
    assert xp.get_backend(arr) == "numpy"
    assert xp.is_gpu(arr) is False
    assert xp.is_cpu(arr) is True


def test_get_array_module_numpy():
    arr = np.array([1, 2, 3])
    mod = xp.get_array_module(arr)
    assert "numpy" in mod.__name__
    assert mod.asarray(arr) is not None


def test_get_array_module_matches_array_not_global_backend():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    a_cpu = np.array([1, 2, 3])
    a_gpu = xp.to_cupy(a_cpu)

    with xp.use_backend("cupy"):
        # Global backend is cupy, but the array itself is on the host.
        assert "numpy" in xp.get_array_module(a_cpu).__name__

    with xp.use_backend("numpy"):
        # Global backend is numpy, but the array itself is on the device.
        assert "cupy" in xp.get_array_module(a_gpu).__name__


def test_same_backend_trivially_true_for_zero_or_one_array():
    assert xp.same_backend() is True
    assert xp.same_backend(np.array([1, 2, 3])) is True


def test_same_backend_true_for_multiple_numpy_arrays():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert xp.same_backend(a, b) is True


def test_assert_same_backend_does_not_raise_when_consistent():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    xp.assert_same_backend(a, b)  # must not raise


def test_same_backend_false_when_mismatched():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    a_cpu = np.array([1, 2, 3])
    a_gpu = xp.to_cupy(a_cpu)

    assert xp.same_backend(a_cpu, a_gpu) is False


def test_assert_same_backend_raises_with_informative_message_when_mismatched():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    a_cpu = np.array([1, 2, 3])
    a_gpu = xp.to_cupy(a_cpu)

    with pytest.raises(TypeError, match="mismatched backends"):
        xp.assert_same_backend(a_cpu, a_gpu)


def test_use_backend():
    # Initial backend should be numpy (default) in this test environment
    assert "numpy" in xp.xp.__name__

    with xp.use_backend("numpy"):
        assert "numpy" in xp.xp.__name__
        arr = xp.zeros(10)
        assert isinstance(arr, np.ndarray)

    assert "numpy" in xp.xp.__name__


def test_set_backend():
    # Set to numpy
    xp.set_backend("numpy")
    assert "numpy" in xp.xp.__name__
    arr = xp.array([1])
    assert isinstance(arr, np.ndarray)

    # Set to cupy (falls back to numpy if not available)
    xp.set_backend("cupy")
    # If cupy is not installed, xp.xp will be numpy module
    arr2 = xp.array([2])
    assert arr2 is not None


def test_backend_bools():
    with xp.use_backend("numpy"):
        assert xp.numpy_backend is True
        assert xp.cupy_backend is False


def test_version_is_own_package_version():
    # __version__ must resolve to cunumpy's own version, not be silently
    # proxied to the active backend module's __version__ via __getattr__.
    assert isinstance(xp.__version__, str)
    assert xp.__version__ != ""
    assert xp.__version__ != np.__version__


def test_set_backend_cupy_fallback_reports_effective_backend():
    """array_backend.backend must reflect what actually loaded, not what was requested."""
    try:
        import cupy  # noqa: F401

        cupy_installed = True
    except ImportError:
        cupy_installed = False

    xp.set_backend("cupy")

    if cupy_installed:
        # Only true in CI on MPCDF, where CuPy is actually available.
        assert cxp.array_backend.backend == "cupy"
        assert xp.cupy_backend is True
    else:
        # No silent lie: requesting cupy without it installed must fall
        # back to numpy *and* report "numpy", not "cupy".
        assert cxp.array_backend.backend == "numpy"
        assert xp.numpy_backend is True
        assert xp.cupy_backend is False

    xp.set_backend("numpy")


def test_use_backend_cupy_fallback_restores_correctly():
    try:
        import cupy  # noqa: F401

        pytest.skip("CuPy is installed; fallback behaviour is not exercised here")
    except ImportError:
        pass

    assert cxp.array_backend.backend == "numpy"

    with xp.use_backend("cupy"):
        # Falls back to numpy since CuPy isn't available here.
        assert cxp.array_backend.backend == "numpy"

    # And the previous state is restored afterwards.
    assert cxp.array_backend.backend == "numpy"


def test_to_numpy_does_not_misdetect_get_method_as_gpu_array():
    """Objects exposing a `.get` method (e.g. dict-like objects) must not be
    mistaken for CuPy arrays just because they happen to have a `.get`."""

    class MappingLike:
        def get(self, key, default=None):
            raise AssertionError(".get() should not be called for non-cupy objects")

        def __array__(self):
            return np.array([1, 2, 3])

    arr = xp.to_numpy(MappingLike())
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, [1, 2, 3])


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError):
        cxp.ArrayBackend(backend="tensorflow")


def test_set_device_is_noop_on_numpy():
    with xp.use_backend("numpy"):
        # Must not raise even though there's no GPU to select on the CPU backend.
        xp.set_device(0)


def test_set_device_selects_cuda_device():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")

    with xp.use_backend("cupy"):
        xp.set_device(0)
        assert cp.cuda.Device().id == 0


def test_array_backend_repr_reports_active_backend():
    with xp.use_backend("numpy"):
        assert repr(cxp.array_backend) == (
            "ArrayBackend(backend='numpy', module='array_api_compat.numpy')"
        )


def test_synchronize_warns_on_attribute_error(monkeypatch):
    """An AttributeError from CuPy's synchronize call (e.g. API mismatch)
    must surface as a warning, not be swallowed silently."""

    class BrokenDevice:
        def synchronize(self):
            raise AttributeError("simulated CuPy API mismatch")

    class FakeCupy:
        cuda = type("cuda", (), {"Device": staticmethod(lambda: BrokenDevice())})

    monkeypatch.setitem(sys.modules, "cupy", FakeCupy)
    monkeypatch.setattr(cxp.array_backend, "_backend", "cupy")

    try:
        with pytest.warns(RuntimeWarning, match="CuPy API mismatch"):
            xp.synchronize()
    finally:
        monkeypatch.setattr(cxp.array_backend, "_backend", "numpy")
