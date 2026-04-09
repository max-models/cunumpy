import numpy as np

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
    missing = [
        name
        for name in np.__all__
        if not hasattr(xp, name)
    ]
    assert missing == [], f"Symbols not accessible via cunumpy: {missing}"


if __name__ == "__main__":
    test_xp_array()
    test_numpy_symbols_accessible()
