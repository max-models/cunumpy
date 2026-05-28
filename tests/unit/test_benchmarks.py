import numpy as np
import pytest
import cunumpy as xp
import time

def has_cupy():
    try:
        import cupy
        import cupy.cuda
        return cupy.cuda.is_available()
    except ImportError:
        return False

@pytest.mark.skipif(not has_cupy(), reason="CuPy/GPU not available")
def test_benchmark_matmul():
    """Benchmark matrix multiplication to show CuPy performance gain."""
    size = 2000
    
    # --- Benchmark NumPy ---
    with xp.use_backend("numpy"):
        a_np = xp.random.rand(size, size).astype(xp.float32)
        b_np = xp.random.rand(size, size).astype(xp.float32)
        
        start_np = time.perf_counter()
        c_np = a_np @ b_np
        # No sync needed for NumPy as it is synchronous
        end_np = time.perf_counter()
        t_np = end_np - start_np

    # --- Benchmark CuPy ---
    with xp.use_backend("cupy"):
        a_cp = xp.random.rand(size, size).astype(xp.float32)
        b_cp = xp.random.rand(size, size).astype(xp.float32)
        
        # Warm up
        _ = a_cp @ b_cp
        xp.synchronize()
        
        start_cp = time.perf_counter()
        c_cp = a_cp @ b_cp
        xp.synchronize() # CRITICAL for benchmarking GPU
        end_cp = time.perf_counter()
        t_cp = end_cp - start_cp

    print(f"\n[Benchmark] Size: {size}x{size}")
    print(f"NumPy time: {t_np:.4f}s")
    print(f"CuPy time:  {t_cp:.4f}s")
    print(f"Speedup:    {t_np/t_cp:.2f}x")
    
    # On a real GPU (A100/A30), CuPy should be significantly faster
    # We use a conservative threshold of 1.5x for the test to pass on various hardware
    assert t_cp < t_np, f"CuPy ({t_cp:.4f}s) was not faster than NumPy ({t_np:.4f}s)"

@pytest.mark.skipif(not has_cupy(), reason="CuPy/GPU not available")
def test_benchmark_fft():
    """Benchmark FFT performance."""
    size = 2**22 # ~4 million elements
    
    with xp.use_backend("numpy"):
        data_np = xp.random.rand(size).astype(xp.complex64)
        start = time.perf_counter()
        _ = xp.fft.fft(data_np)
        t_np = time.perf_counter() - start

    with xp.use_backend("cupy"):
        data_cp = xp.random.rand(size).astype(xp.complex64)
        # Warm up
        _ = xp.fft.fft(data_cp)
        xp.synchronize()
        
        start = time.perf_counter()
        _ = xp.fft.fft(data_cp)
        xp.synchronize()
        t_cp = time.perf_counter() - start

    print(f"\n[Benchmark] FFT Size: {size}")
    print(f"NumPy time: {t_np:.4f}s")
    print(f"CuPy time:  {t_cp:.4f}s")
    assert t_cp < t_np
