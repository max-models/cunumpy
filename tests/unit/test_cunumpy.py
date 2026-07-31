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
