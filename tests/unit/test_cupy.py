import numpy as np
import pytest

import cunumpy as xp


def test_to_cupy_available():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")

    with xp.use_backend("cupy"):
        arr = np.array([1, 2, 3])
        arr_cp = xp.to_cupy(arr)
        assert isinstance(arr_cp, cp.ndarray)


def test_to_cupy_not_available():
    try:
        import cupy

        pytest.skip("CuPy is installed, cannot test missing cupy error")
    except ImportError:
        pass

    with xp.use_backend("cupy"):
        arr = np.array([1, 2, 3])
        with pytest.raises(ImportError):
            xp.to_cupy(arr)


def test_synchronize():
    # Should not crash on any backend
    xp.synchronize()

    with xp.use_backend("numpy"):
        xp.synchronize()

    with xp.use_backend("cupy"):
        xp.synchronize()


def test_xp_array_cupy():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")

    with xp.use_backend("cupy"):
        arr = xp.array([1, 2])
        arr *= 2
        assert isinstance(arr, cp.ndarray)
        assert cp.asnumpy(arr).tolist() == [2, 4]
