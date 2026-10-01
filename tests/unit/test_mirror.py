"""Tests for `cunumpy.DeviceMirror` and the shipped `cunumpy/atomic.cuh` header.

On the NumPy backend a mirror is transparent: `device` is the host array and
transfers are no-ops. Device copies and the atomic kernel need a GPU and are
skipped without one.
"""

import os

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import CudaKernel, DeviceMirror

requires_gpu = pytest.mark.skipif(
    not xp.cupy_available(), reason="CuPy/GPU not available or not functional"
)

BIN_ADD = r"""
#include <cunumpy/atomic.cuh>

// out[bins[i]] += weights[i], many threads per bin
extern "C" __global__
void bin_add(const long long* bins, const double* weights, double* out, long long n)
{
    long long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) cunumpy_atomic_add(out + bins[i], weights[i]);
}
"""

BIN_ADD_2D = r"""
#include <cunumpy/atomic.cuh>

// out[rows[i], cols[i]] += weights[i] for a C-contiguous out of shape (n0, n1)
extern "C" __global__
void bin_add_2d(const long long* rows, const long long* cols,
                const double* weights, double* out, long long n1, long long n)
{
    long long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) cunumpy_atomic_add_2d(out, n1, rows[i], cols[i], weights[i]);
}
"""


# --- NumPy backend -----------------------------------------------------------


def test_device_is_host_on_numpy():
    host = np.arange(6, dtype=np.float64).reshape(2, 3)
    with xp.use_backend("numpy"):
        mirror = DeviceMirror(host)
        assert mirror.device is host
        assert mirror.host is host
    assert mirror.shape == (2, 3)
    assert mirror.dtype == np.float64
    assert "DeviceMirror" in repr(mirror)
    assert "(2, 3)" in repr(mirror)


def test_transfers_are_noops_and_chain_on_numpy():
    host = np.arange(4, dtype=np.float64)
    with xp.use_backend("numpy"):
        mirror = DeviceMirror(host)
        assert mirror.to_device() is mirror
        assert mirror.to_host() is mirror
        assert mirror.to_device().to_host() is mirror
    assert mirror.host is host
    np.testing.assert_array_equal(host, [0, 1, 2, 3])


def test_zero_clears_host_on_numpy():
    host = np.arange(4, dtype=np.float64)
    with xp.use_backend("numpy"):
        mirror = DeviceMirror(host)
        assert mirror.zero() is mirror
    assert mirror.host is host
    np.testing.assert_array_equal(host, 0)


def test_kernel_writes_reach_host_on_numpy():
    """The pattern accumulation code uses: zero, write into .device, to_host."""
    host = np.ones(3)
    with xp.use_backend("numpy"):
        mirror = DeviceMirror(host)
        mirror.zero()
        mirror.device[1] += 5.0  # what a host kernel would do
        mirror.to_host()
    np.testing.assert_array_equal(host, [0, 5, 0])


@pytest.mark.parametrize("bad", [[1.0, 2.0], (1, 2), 3.0, None])
def test_rejects_non_numpy_host(bad):
    with pytest.raises(TypeError, match="NumPy array"):
        DeviceMirror(bad)


def test_rejects_non_numpy_host_rebind():
    mirror = DeviceMirror(np.zeros(2))
    with pytest.raises(TypeError, match="NumPy array"):
        mirror.rebind([0.0, 0.0])


def test_reallocated_host_raises():
    host = np.zeros(4)
    mirror = DeviceMirror(host)
    host.resize(8, refcheck=False)  # the owner changed the buffer in place
    with pytest.raises(ValueError, match=r"\(4,\).*\(8,\)"):
        mirror.to_device()
    with pytest.raises(ValueError, match="rebind"):
        mirror.to_host()


def test_rebind():
    old = np.zeros(4)
    mirror = DeviceMirror(old)
    new = np.ones((2, 2), dtype=np.float32)
    assert mirror.rebind(new) is mirror
    assert mirror.host is new
    assert mirror.shape == (2, 2)
    assert mirror.dtype == np.float32
    with xp.use_backend("numpy"):
        assert mirror.device is new
        mirror.to_device().to_host()  # no longer mismatched


def test_cuda_include_dir_contains_atomic_header():
    include_dir = xp.cuda_include_dir()
    assert isinstance(include_dir, str)
    assert include_dir.endswith(os.path.join("cuda", "include"))
    header = os.path.join(include_dir, "cunumpy", "atomic.cuh")
    assert os.path.isfile(header)
    source = open(header).read()
    assert "cunumpy_atomic_add(double* p, double v)" in source
    assert "cunumpy_atomic_add(float* p, float v)" in source
    assert "cunumpy_atomic_add_2d(" in source
    assert "cunumpy_atomic_add_3d(" in source


def test_cuda_kernel_options_include_cunumpy_headers():
    flag = f"-I{xp.cuda_include_dir()}"
    kernel = CudaKernel(BIN_ADD, "bin_add")
    assert flag not in kernel.options  # options are as given
    assert flag in kernel.compile_options()
    # not duplicated when the caller adds it, and user dirs stay first
    kernel = CudaKernel(BIN_ADD, "bin_add", include_dirs=["/some/dir", flag[2:]])
    assert kernel.compile_options().count(flag) == 1
    assert kernel.compile_options()[0] == "-I/some/dir"
    kernel = CudaKernel(BIN_ADD, "bin_add", options=["-std=c++17", flag])
    options = kernel.compile_options()
    assert options[:2] == ("-std=c++17", flag)
    # the shipped atomic.cuh is part of the header hash
    assert len(options) == 3 and options[2].startswith("-DCUNUMPY_INCLUDE_HASH=0x")


# --- CuPy backend ------------------------------------------------------------


@requires_gpu
def test_device_is_cupy_array_on_gpu():
    import cupy as cp

    host = np.arange(6, dtype=np.float64).reshape(2, 3)
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(host)
        device = mirror.device
        assert isinstance(device, cp.ndarray)
        assert device.shape == host.shape
        assert device.dtype == host.dtype
        assert mirror.device is device  # allocated once
        np.testing.assert_array_equal(device.get(), host)  # filled from the host
    assert mirror.host is host


@requires_gpu
def test_to_device_copies_values():
    host = np.zeros(4)
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(host)
        device = mirror.device
        host[:] = [1, 2, 3, 4]
        assert mirror.to_device() is mirror
        assert mirror.device is device  # same buffer, no reallocation
        np.testing.assert_array_equal(device.get(), [1, 2, 3, 4])


@requires_gpu
def test_kernel_writes_land_in_same_host_object():
    host = np.ones(5)
    owner = {"data": host}  # stands in for the library that owns the buffer
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(host)
        mirror.zero()
        np.testing.assert_array_equal(host, 1)  # zero() does not touch the host
        mirror.device[2] += 5.0
        assert mirror.to_host() is mirror
    assert mirror.host is host
    assert owner["data"] is host
    np.testing.assert_array_equal(host, [0, 0, 5, 0, 0])


@requires_gpu
def test_zero_and_rebind_on_gpu():
    host = np.ones(3)
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(host)
        assert mirror.zero() is mirror
        np.testing.assert_array_equal(mirror.device.get(), 0)
        device = mirror.device
        mirror.rebind(np.full(3, 7.0))  # same shape: buffer kept
        assert mirror.device is device
        mirror.rebind(np.zeros((2, 2)))  # new shape: buffer dropped
        assert mirror.device is not device
        assert mirror.device.shape == (2, 2)


@requires_gpu
def test_atomic_add_matches_bincount():
    import cupy as cp

    rng = np.random.default_rng(0)
    n, n_bins = 100_000, 7
    bins = rng.integers(0, n_bins, n).astype(np.int64)
    weights = rng.random(n)
    expected = np.bincount(bins, weights=weights, minlength=n_bins)

    out = np.zeros(n_bins)
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(out)
        kernel = CudaKernel(BIN_ADD, "bin_add")
        mirror.zero()
        kernel(cp.asarray(bins), cp.asarray(weights), mirror.device, n, n_threads=n)
        mirror.to_host()
    np.testing.assert_allclose(out, expected, rtol=1e-12)


@requires_gpu
def test_atomic_add_2d_matches_bincount():
    import cupy as cp

    rng = np.random.default_rng(1)
    n, shape = 50_000, (3, 5)
    rows = rng.integers(0, shape[0], n).astype(np.int64)
    cols = rng.integers(0, shape[1], n).astype(np.int64)
    weights = rng.random(n)
    flat = rows * shape[1] + cols
    expected = np.bincount(flat, weights=weights, minlength=np.prod(shape)).reshape(
        shape
    )

    out = np.zeros(shape)
    with xp.use_backend("cupy"):
        mirror = DeviceMirror(out)
        kernel = CudaKernel(BIN_ADD_2D, "bin_add_2d")
        mirror.zero()
        kernel(
            cp.asarray(rows),
            cp.asarray(cols),
            cp.asarray(weights),
            mirror.device,
            shape[1],
            n,
            n_threads=n,
        )
        mirror.to_host()
    np.testing.assert_allclose(out, expected, rtol=1e-12)
