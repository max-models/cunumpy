"""Tests for the Philox4x32-10 generator: host functions and cunumpy/random.cuh."""

from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.cuda import cuda_include_dir
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from cunumpy.kernels import CudaKernel

# Known-answer vectors of Philox4x32-10 (Random123, kat_vectors)
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (
        (0xFFFFFFFF,) * 4,
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize(("counter", "key", "expected"), KAT)
def test_known_answers(counter, key, expected):
    out = xp.rng.philox4x32_10(np.array(counter, dtype=np.uint32), *key)
    assert out.dtype == np.uint32
    assert out.tolist() == list(expected)


def test_vectorized_over_counters():
    counters = np.array([k[0] for k in KAT], dtype=np.uint32)
    keys = np.array([k[1] for k in KAT], dtype=np.uint32)
    out = xp.rng.philox4x32_10(counters, keys[:, 0], keys[:, 1])
    assert out.tolist() == [list(k[2]) for k in KAT]


def test_uniforms_and_normals():
    ids = np.arange(200_000, dtype=np.uint64)
    u0, u1 = xp.rng.philox_uniform2(42, ids, 7)
    assert u0.shape == u1.shape == (200_000,)
    assert u0.min() >= 0.0 and u0.max() < 1.0
    assert abs(u0.mean() - 0.5) < 3e-3 and abs(u1.var() - 1 / 12) < 1e-3
    assert abs(np.corrcoef(u0, u1)[0, 1]) < 1e-2
    np.testing.assert_array_equal(xp.rng.philox_uniform(42, ids, 7), u0)
    z0, z1 = xp.rng.philox_normal2(42, ids, 7)
    assert abs(z0.mean()) < 1e-2 and abs(z1.std() - 1.0) < 1e-2
    np.testing.assert_array_equal(xp.rng.philox_normal(42, ids, 7), z0)


def test_streams_counters_and_seeds_differ():
    base = xp.rng.philox_uniform(1, 5, 9)
    assert xp.rng.philox_uniform(1, 5, 9) == base  # no state
    assert xp.rng.philox_uniform(1, 6, 9) != base
    assert xp.rng.philox_uniform(1, 5, 10) != base
    assert xp.rng.philox_uniform(2, 5, 9) != base
    # 64-bit stream and counter: the high words matter
    assert xp.rng.philox_uniform(1, 5 + 2**32, 9) != base
    assert xp.rng.philox_uniform(1, 5, 9 + 2**32) != base
    assert xp.rng.philox_uniform(1 + 2**32, 5, 9) != base


def test_broadcasting():
    u = xp.rng.philox_uniform(
        np.uint64(3),
        np.arange(4, dtype=np.uint64)[:, None],
        np.arange(5),
    )
    assert u.shape == (4, 5)
    assert u[2, 3] == xp.rng.philox_uniform(3, 2, 3)


def test_header_is_shipped():
    header = (Path(cuda_include_dir()) / "cunumpy" / "random.cuh").read_text()
    for name in ("cunumpy_philox4x32_10", "cunumpy_uniform2", "cunumpy_normal2"):
        assert name in header


SAMPLE = r"""
#include "cunumpy/random.cuh"
extern "C" __global__
void sample(double* u0, double* u1, double* z0, double* z1, long long n,
            unsigned long long seed, unsigned long long counter) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    cunumpy_uniform2(seed, (unsigned long long)i, counter, &u0[i], &u1[i]);
    cunumpy_normal2(seed, (unsigned long long)i, counter, &z0[i], &z1[i]);
}
"""


def _device_samples(run, n=1000, seed=2**40 + 17, counter=2**33 + 5):
    u0, u1, z0, z1 = (np.zeros(n) for _ in range(4))
    run(CudaKernel(SAMPLE, "sample"), u0, u1, z0, z1, n, seed, counter, n_threads=n)
    return (
        (u0, u1, z0, z1),
        xp.rng.philox_uniform2(seed, np.arange(n, dtype=np.uint64), counter),
        xp.rng.philox_normal2(seed, np.arange(n, dtype=np.uint64), counter),
    )


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
def test_header_matches_the_host_functions_in_emulation():
    (u0, u1, z0, z1), (h0, h1), (n0, n1) = _device_samples(emulate_cuda_kernel)
    np.testing.assert_array_equal(u0, h0)  # bit-identical
    np.testing.assert_array_equal(u1, h1)
    np.testing.assert_allclose(z0, n0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(z1, n1, rtol=1e-14, atol=1e-14)


def test_header_matches_the_host_functions_on_gpu():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    def run(kernel, *args, n_threads):
        device = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
        kernel(*device, n_threads=n_threads)
        for host, dev in zip(args, device):
            if isinstance(host, np.ndarray):
                host[...] = dev.get()

    (u0, u1, z0, z1), (h0, h1), (n0, n1) = _device_samples(run)
    np.testing.assert_array_equal(u0, h0)
    np.testing.assert_array_equal(u1, h1)
    np.testing.assert_allclose(z0, n0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z1, n1, rtol=1e-13, atol=1e-13)
    # the device-side generator on device arrays, too
    du0, _ = xp.rng.philox_uniform2(7, cp.arange(10, dtype=cp.uint64), 1)
    assert isinstance(du0, cp.ndarray)
    np.testing.assert_array_equal(
        du0.get(),
        xp.rng.philox_uniform2(7, np.arange(10, dtype=np.uint64), 1)[0],
    )
