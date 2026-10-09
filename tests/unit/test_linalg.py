"""Tests for `solve_circulant`, the cunumpy fill-in for `cupyx.scipy.linalg`."""

import numpy as np
import pytest

from cunumpy._linalg import solve_circulant

scipy_linalg = pytest.importorskip("scipy.linalg")

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    ("c", "b", "kwargs"),
    [
        (RNG.random(5), RNG.random(5), {}),
        (RNG.random(5), RNG.random((5, 3)), {}),
        (RNG.random((2, 5)), RNG.random((5, 2)), {}),
        (RNG.random(6) + 1j * RNG.random(6), RNG.random(6), {}),
        (
            RNG.random((5, 3)),
            RNG.random((3, 5)),
            {"caxis": 0, "baxis": 1, "outaxis": -1},
        ),
        (np.array([1.0, 1.0, 0.0, 0.0]), np.ones(4), {"singular": "lstsq"}),
        (np.array([4, 1, 0, 1]), np.array([6, 6, 6, 6]), {}),
        (np.float32([4, 1, 0, 1]), np.float32([1, 2, 3, 4]), {}),
    ],
)
def test_matches_scipy(c, b, kwargs):
    x = solve_circulant(c, b, **kwargs)
    ref = scipy_linalg.solve_circulant(c, b, **kwargs)
    assert x.dtype == ref.dtype
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-12)


def test_singular_raises():
    with pytest.raises(np.linalg.LinAlgError, match="near singular"):
        solve_circulant(np.array([1.0, 1.0, 0.0, 0.0]), np.ones(4))


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="incompatible"):
        solve_circulant(np.ones(4), np.ones(5))
