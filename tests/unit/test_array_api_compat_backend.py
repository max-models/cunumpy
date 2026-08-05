"""Verify that swapping `xp.xp` to `array_api_compat.numpy`/`array_api_compat.cupy`
(instead of raw `numpy`/`cupy`) actually behaves as intended on both backends.

The CuPy-side assertions here can only be meaningfully exercised on the MPCDF
GPU CI runner (no GPU is available in most dev/local environments), but the
NumPy-side assertions run everywhere so local runs still get partial coverage.
"""

import array_api_compat
import numpy as np
import pytest

import cunumpy as xp
import cunumpy.xp as cxp


def _skip_without_cupy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_backend_module_is_array_api_compat_wrapper(backend):
    if backend == "cupy":
        _skip_without_cupy()

    with xp.use_backend(backend):
        # NB: `xp.xp` (via the top-level cunumpy package) is the internal
        # `cunumpy.xp` submodule, not the active backend module -- go
        # through the ArrayBackend instance directly for that.
        assert cxp.array_backend.xp.__name__ == f"array_api_compat.{backend}"


def test_array_backend_repr_on_cupy():
    _skip_without_cupy()

    with xp.use_backend("cupy"):
        assert repr(cxp.array_backend) == (
            "ArrayBackend(backend='cupy', module='array_api_compat.cupy')"
        )


def test_to_cupy_returns_real_cupy_ndarray():
    _skip_without_cupy()

    import cupy as cp

    arr = xp.to_cupy(np.array([1, 2, 3]))
    assert type(arr) is cp.ndarray
    assert type(arr).__module__.startswith("cupy")


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.complex64]
)
def test_round_trip_preserves_values_and_dtype(dtype):
    _skip_without_cupy()

    original = np.arange(12, dtype=dtype).reshape(3, 4)
    gpu = xp.to_cupy(original)
    back = xp.to_numpy(gpu)

    assert back.dtype == original.dtype
    assert back.shape == original.shape
    assert np.array_equal(back, original)


@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 4), (2, 3, 4)])
def test_round_trip_preserves_shape(shape):
    _skip_without_cupy()

    original = np.random.rand(*shape).astype(np.float64)
    gpu = xp.to_cupy(original)
    back = xp.to_numpy(gpu)

    assert back.shape == shape
    assert np.allclose(back, original)


def test_get_backend_is_gpu_is_cpu_consistency():
    _skip_without_cupy()

    arr_cpu = np.array([1, 2, 3])
    arr_gpu = xp.to_cupy(arr_cpu)

    assert xp.get_backend(arr_cpu) == "numpy"
    assert xp.is_cpu(arr_cpu) is True
    assert xp.is_gpu(arr_cpu) is False

    assert xp.get_backend(arr_gpu) == "cupy"
    assert xp.is_gpu(arr_gpu) is True
    assert xp.is_cpu(arr_gpu) is False


def test_array_api_compat_agrees_with_cunumpy_on_real_gpu_array():
    """Sanity-check our get_backend() against array_api_compat's own
    detector directly, on a real (non-mocked) GPU array."""
    _skip_without_cupy()

    arr_gpu = xp.to_cupy(np.array([1, 2, 3]))
    assert array_api_compat.is_cupy_array(arr_gpu) is True
    assert array_api_compat.is_numpy_array(arr_gpu) is False


def test_asarray_device_kwarg_accepted_on_both_backends():
    """The wrapped asarray must accept `device=` on both backends, even
    though raw CuPy's asarray historically has a narrower signature."""
    with xp.use_backend("numpy"):
        arr = xp.asarray([1, 2, 3], device=None)
        assert isinstance(arr, np.ndarray)

    _skip_without_cupy()

    with xp.use_backend("cupy"):
        arr = xp.asarray([1, 2, 3], device=None)
        assert xp.is_gpu(arr)


def test_asarray_copy_semantics_parity_between_backends():
    """`copy=False` must avoid copying when no copy is needed, and
    `copy=True` must always copy -- consistently on both backends."""
    with xp.use_backend("numpy"):
        base = np.array([1.0, 2.0, 3.0])
        no_copy = xp.asarray(base, copy=False)
        assert np.shares_memory(no_copy, base)

        forced_copy = xp.asarray(base, copy=True)
        assert not np.shares_memory(forced_copy, base)

    _skip_without_cupy()

    import cupy as cp

    with xp.use_backend("cupy"):
        base_gpu = cp.array([1.0, 2.0, 3.0])
        no_copy_gpu = xp.asarray(base_gpu, copy=False)
        assert no_copy_gpu.data.ptr == base_gpu.data.ptr

        forced_copy_gpu = xp.asarray(base_gpu, copy=True)
        assert forced_copy_gpu.data.ptr != base_gpu.data.ptr


def test_asarray_dtype_kwarg_consistent_across_backends():
    for dtype in (np.float32, np.float64, np.int32, np.complex64):
        with xp.use_backend("numpy"):
            arr = xp.asarray([1, 2, 3], dtype=dtype)
            assert arr.dtype == dtype

        if not xp.cupy_available():
            continue

        with xp.use_backend("cupy"):
            arr = xp.asarray([1, 2, 3], dtype=dtype)
            assert arr.dtype == dtype


def test_set_device_and_synchronize_work_with_wrapped_backend():
    """set_device()/synchronize() still use raw CuPy internally for CUDA
    control (device management isn't part of the Array API standard) --
    verify they keep working correctly alongside the wrapped `xp.xp`."""
    _skip_without_cupy()

    import cupy as cp

    with xp.use_backend("cupy"):
        xp.set_device(0)
        assert cp.cuda.Device().id == 0

        arr = xp.asarray([1, 2, 3])
        xp.synchronize()
        assert xp.is_gpu(arr)


def test_cross_backend_arithmetic_matches_after_round_trip():
    """Values computed on the GPU via the wrapped backend must match the
    equivalent NumPy computation once brought back to the CPU."""
    _skip_without_cupy()

    data = np.random.rand(50).astype(np.float64)

    with xp.use_backend("numpy"):
        expected = xp.sin(xp.array(data)) ** 2 + xp.cos(xp.array(data)) ** 2

    with xp.use_backend("cupy"):
        gpu_data = xp.to_cupy(data)
        result_gpu = xp.sin(gpu_data) ** 2 + xp.cos(gpu_data) ** 2
        result = xp.to_numpy(result_gpu)

    assert np.allclose(result, expected)
    assert np.allclose(result, 1.0)
