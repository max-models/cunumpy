"""Backend numerical parity; performance comparisons belong in scope-profiler."""

import numpy as np
import pytest

import cunumpy as xp

pytestmark = pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")


def test_matmul_matches_numpy():
    rng = np.random.default_rng(123)
    a = rng.normal(size=(64, 96)).astype(np.float32)
    b = rng.normal(size=(96, 32)).astype(np.float32)
    with xp.use_backend("cupy", strict=True):
        actual = xp.to_numpy(xp.asarray(a) @ xp.asarray(b))
    np.testing.assert_allclose(actual, a @ b, rtol=2e-5, atol=2e-5)


def test_fft_matches_numpy():
    rng = np.random.default_rng(123)
    values = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(np.complex64)
    with xp.use_backend("cupy", strict=True):
        actual = xp.to_numpy(xp.fft.fft(xp.asarray(values)))
    np.testing.assert_allclose(actual, np.fft.fft(values), rtol=2e-5, atol=2e-5)
