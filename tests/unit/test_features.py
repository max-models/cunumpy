import numpy as np
import pytest

import cunumpy as xp


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_matrix_multiplication(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        # Test basic @ operator and matmul
        a = xp.array([[1, 2], [3, 4]], dtype=float)
        b = xp.array([[5, 6], [7, 8]], dtype=float)
        c = a @ b

        expected = np.array([[19, 22], [43, 50]])
        assert xp.array_equal(xp.to_numpy(c), expected)

        # Test linalg.norm
        norm = xp.linalg.norm(a)
        assert np.isclose(float(norm), np.linalg.norm([[1, 2], [3, 4]]))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_reductions_and_axes(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        a = xp.array([[1, 10, 100], [2, 20, 200]], dtype=float)

        assert xp.sum(a) == 333
        assert np.array_equal(xp.to_numpy(xp.max(a, axis=0)), [2, 20, 200])
        assert np.array_equal(xp.to_numpy(xp.min(a, axis=1)), [1, 2])
        assert xp.mean(a) == 333 / 6


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_complex_elementwise(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        a = xp.array([-1, 0, 1], dtype=float)

        # Exp and Log
        exp_a = xp.exp(a)
        assert np.allclose(xp.to_numpy(exp_a), np.exp([-1, 0, 1]))

        # Trig
        b = xp.array([0, xp.pi / 2], dtype=float)
        assert np.allclose(xp.to_numpy(xp.cos(b)), [1, 0], atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_broadcasting_logic(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        # 3D + 1D broadcasting
        a = xp.ones((2, 3, 4))
        b = xp.arange(4)
        c = a * b

        assert c.shape == (2, 3, 4)
        assert np.array_equal(xp.to_numpy(c[0, 0]), [0, 1, 2, 3])
        assert np.array_equal(xp.to_numpy(c[1, 2]), [0, 1, 2, 3])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_parity(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        # Create a signal with two frequencies
        t = xp.linspace(0, 1, 128)
        sig = xp.sin(2 * xp.pi * 5 * t) + 0.5 * xp.sin(2 * xp.pi * 20 * t)

        freqs = xp.fft.fft(sig)
        inv = xp.fft.ifft(freqs)

        # ifft(fft(x)) == x
        assert np.allclose(xp.to_numpy(inv.real), xp.to_numpy(sig))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_realistic_normalization_workflow(backend):
    """Workflow: Load data -> Compute Stats -> Normalize -> Mask Outliers."""
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        # 1. Create dummy data with clear outliers
        data = xp.array([1.0, 2.0, 3.0, 4.0, 100.0, -100.0])

        # 2. Normalize
        mean = xp.mean(data)
        std = xp.std(data)
        norm_data = (data - mean) / std

        # 3. Mask outliers (abs > 1.0 in this specific small set)
        mask = xp.abs(norm_data) < 1.0
        clean_data = data[mask]

        # Verify: -100 and 100 should be gone
        res = xp.to_numpy(xp.sort(clean_data))
        assert np.array_equal(res, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_stacking_and_concatenation(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        a = xp.array([1, 2, 3])
        b = xp.array([4, 5, 6])

        res_cat = xp.concatenate([a, b])
        assert np.array_equal(xp.to_numpy(res_cat), [1, 2, 3, 4, 5, 6])

        res_stack = xp.stack([a, b])
        assert res_stack.shape == (2, 3)
        assert np.array_equal(xp.to_numpy(res_stack[1]), [4, 5, 6])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_advanced_indexing(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        a = xp.arange(10).reshape(2, 5)

        # Pick specific elements: (0,1) and (1,3)
        rows = xp.array([0, 1])
        cols = xp.array([1, 3])

        indexed = a[rows, cols]
        assert np.array_equal(xp.to_numpy(indexed), [1, 8])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_random_generation(backend):
    if backend == "cupy" and not xp.has_cupy():
        pytest.skip("CuPy not installed or not functional")

    with xp.use_backend(backend):
        # Test reproducibility if we were to add seed (checking existing proxy)
        a = xp.random.normal(0, 1, size=(100, 100))
        assert a.shape == (100, 100)
        assert xp.abs(xp.mean(a)) < 0.5  # Basic statistical sanity
