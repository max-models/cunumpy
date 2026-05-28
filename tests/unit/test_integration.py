import numpy as np
import pytest

import cunumpy as xp


def has_cupy():
    try:
        import cupy

        return True
    except ImportError:
        return False


def test_data_movement_chain():
    """Test CPU -> GPU -> CPU multi-hop movement."""
    if not has_cupy():
        pytest.skip("CuPy not installed")

    # 1. Start on CPU
    data_orig = np.random.rand(100, 100).astype(np.float32)

    # 2. Move to GPU
    data_gpu = xp.to_cupy(data_orig)
    assert xp.is_gpu(data_gpu)

    # 3. Do operation on GPU
    with xp.use_backend("cupy"):
        res_gpu = xp.sin(data_gpu) ** 2 + xp.cos(data_gpu) ** 2

    # 4. Move back to CPU
    res_cpu = xp.to_numpy(res_gpu)
    assert isinstance(res_cpu, np.ndarray)
    assert np.allclose(res_cpu, 1.0)


def test_synchronize_logic():
    """Verify synchronize can be called and handles errors gracefully."""
    # This is more of a smoke test to ensure the path doesn't crash
    xp.synchronize()

    if has_cupy():
        import cupy as cp

        with xp.use_backend("cupy"):
            a = xp.random.rand(100)
            xp.synchronize()
            assert xp.is_gpu(a)


def test_fft_interop():
    """Test FFT between backends."""
    if not has_cupy():
        pytest.skip("CuPy not installed")

    # Create signal on CPU
    sig_cpu = np.random.rand(1024).astype(np.complex128)

    # Move to GPU and transform
    sig_gpu = xp.to_cupy(sig_cpu)
    freq_gpu = xp.fft.fft(sig_gpu)

    # Move frequencies to CPU and transform back
    freq_cpu = xp.to_numpy(freq_gpu)
    sig_reconstructed = np.fft.ifft(freq_cpu)

    assert np.allclose(sig_cpu, sig_reconstructed)


def test_mixed_backend_errors():
    """Verify that mixing backends in operations raises errors (standard NumPy/CuPy behavior)."""
    if not has_cupy():
        pytest.skip("CuPy not installed")

    a_cpu = np.array([1, 2, 3])
    a_gpu = xp.to_cupy(a_cpu)

    # This should fail because you can't add CPU and GPU arrays directly
    with pytest.raises(Exception):
        _ = a_cpu + a_gpu

    # But to_cunumpy should fix it
    a_gpu_fixed = xp.to_cunumpy(a_cpu)
    with xp.use_backend("cupy"):
        res = a_gpu + a_gpu_fixed
        assert xp.is_gpu(res)
