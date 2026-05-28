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


def test_get_backend_and_is_gpu():
    arr = np.array([1, 2, 3])
    assert xp.get_backend(arr) == "numpy"
    assert xp.is_gpu(arr) is False


if __name__ == "__main__":
    test_xp_array()
    test_numpy_symbols_accessible()
