import numpy as np
import pytest

import cunumpy as xp


def test_xp_array():

    arr = xp.array([1, 2])
    arr *= 2

    print(f"{arr = } {type(arr) = }")


def test_numpy_symbols_accessible():
    """All public numpy symbols must be reachable via cunumpy.

    This validates the runtime behaviour that the stub file (__init__.pyi)
    declares to Pylance so that `xp.<Tab>` shows numpy completions in VS Code.
    """
    # Exclude our custom methods from the numpy check
    custom_methods = [
        "to_numpy",
        "to_cupy",
        "to_cunumpy",
        "get_backend",
        "is_gpu",
        "is_cpu",
        "use_backend",
        "set_backend",
        "xp",
    ]
    missing = [
        name
        for name in np.__all__
        if not hasattr(xp, name) and name not in custom_methods
    ]
    assert missing == [], f"Symbols not accessible via cunumpy: {missing}"


def test_to_numpy():
    arr = xp.array([1, 2, 3])
    # Even if it's already numpy, to_numpy should work
    arr_np = xp.to_numpy(arr)
    assert isinstance(arr_np, np.ndarray)
    assert np.array_equal(arr_np, [1, 2, 3])


def test_to_cupy_not_available():
    try:
        import cupy

        pytest.skip("CuPy is installed, cannot test missing cupy error")
    except ImportError:
        pass

    arr = np.array([1, 2, 3])

    with pytest.raises(ImportError):
        xp.to_cupy(arr)


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
    # Accessing xp.xp triggers the dynamic __getattr__ in xp.py
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
    # We just verify it doesn't crash and we can still call things
    arr2 = xp.array([2])
    assert arr2 is not None


if __name__ == "__main__":
    test_xp_array()
    test_numpy_symbols_accessible()
