import numpy as np
import pytest
import cunumpy as xp

def test_xp_array():
    arr = xp.array([1, 2])
    arr *= 2
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, [2, 4])

def test_numpy_symbols_accessible():
    """All public numpy symbols must be reachable via cunumpy.

    This validates the runtime behaviour that the stub file (__init__.pyi)
    declares to Pylance so that `xp.<Tab>` shows numpy completions in VS Code.
    """
    if xp.cupy_backend:
        pytest.skip("CuPy does not have 100% symbol parity with NumPy.")

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
        "synchronize",
        "numpy_backend",
        "cupy_backend",
        "xp",
    ]
    missing = [
        name
        for name in np.__all__
        if not hasattr(xp, name) and name not in custom_methods
    ]
    assert missing == [], f"Symbols not accessible via cunumpy: {missing}"
