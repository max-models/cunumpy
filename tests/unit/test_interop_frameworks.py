"""Tests for the JAX, PyTorch and MLX adapters with the real frameworks (skipped without them)."""

import numpy as np
import pytest

from cunumpy.profiling import assert_no_transfers, count_transfers


def _aligned(n, alignment=64):
    """A float64 array whose buffer is aligned for XLA, which shares only aligned memory."""
    buffer = np.empty(n * 8 + alignment, dtype=np.uint8)
    start = -buffer.ctypes.data % alignment
    return buffer[start : start + n * 8].view(np.float64)


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    from cunumpy.interop import jax as adapter

    enabled = jax.config.read("jax_enable_x64")
    yield jax, adapter
    jax.config.update("jax_enable_x64", enabled)


def test_jax_refuses_float64_without_x64(jax_x64):
    _, adapter = jax_x64
    adapter.enable_x64(False)
    assert not adapter.supports_float64()
    with pytest.raises(TypeError, match="JAX would turn float64 into float32; enable"):
        adapter.to_jax(np.arange(3.0))
    assert adapter.to_jax(np.arange(3.0), dtype="float32").dtype == np.float32


def test_jax_round_trip_shares_memory(jax_x64):
    jax, adapter = jax_x64
    adapter.enable_x64()
    assert adapter.supports_float64()
    assert adapter.supports_float64(jax.devices()[0])
    a = _aligned(4)
    a[:] = np.arange(4.0)
    with assert_no_transfers():
        value = adapter.to_jax(a, copy=False)
        back = adapter.from_jax(value, copy=False)
    assert value.dtype == np.float64
    assert back.ctypes.data == a.ctypes.data
    assert adapter.to_jax(value) is value


def test_jax_copies_and_reports_them(jax_x64):
    jax, adapter = jax_x64
    adapter.enable_x64()
    strided = np.arange(6.0).reshape(2, 3)[:, ::2]
    with count_transfers() as counter:
        value = adapter.to_jax(strided)  # JAX's DLPack takes no strides: a copy
        adapter.to_jax(np.arange(3.0), copy=True)
        adapter.from_jax(value, copy=True)
        adapter.to_jax([1.0, 2.0])
    assert counter.count("device_copy") == 4
    np.testing.assert_array_equal(np.asarray(value), strided)
    with pytest.raises(ValueError, match="to JAX needs a copy"):
        adapter.to_jax(strided, copy=False)
    with pytest.raises(ValueError, match="to JAX needs a copy"):
        adapter.to_jax(np.arange(3.0), dtype="float32", copy=False)
    placed = adapter.to_jax(np.arange(3.0), device=jax.devices()[0])
    assert placed.dtype == np.float64


def test_jax_rejects_tracers(jax_x64):
    jax, adapter = jax_x64

    @jax.jit
    def convert(value):
        adapter.from_jax(value)
        return value

    with pytest.raises(TypeError, match="traced JAX value"):
        convert(jax.numpy.ones(2))


def test_jax_rejects_a_device_it_does_not_have(jax_x64):
    _, adapter = jax_x64

    class OnCuda:
        dtype = np.dtype("float64")

        def __dlpack__(self, **kwargs):
            raise RuntimeError("not a capsule")

        def __dlpack_device__(self):
            return (2, 0)

    with pytest.raises(ValueError, match="JAX cannot take a OnCuda on CUDA device 0"):
        adapter.to_jax(OnCuda())


def test_torch_round_trip_shares_memory():
    torch = pytest.importorskip("torch")
    from cunumpy.interop import torch as adapter

    a = np.arange(4.0)
    with assert_no_transfers():
        tensor = adapter.to_torch(a, copy=False)
        back = adapter.from_torch(tensor, copy=False)
    assert tensor.dtype == torch.float64 and back.ctypes.data == a.ctypes.data
    assert adapter.to_torch(tensor) is tensor
    grad = torch.ones(2, requires_grad=True)
    np.testing.assert_array_equal(adapter.from_torch(grad), [1.0, 1.0])


def test_torch_copies_and_dtypes():
    torch = pytest.importorskip("torch")
    from cunumpy.interop import torch as adapter

    with count_transfers() as counter:
        adapter.to_torch(np.arange(3.0), copy=True)
        adapter.to_torch([1.0, 2.0])
        adapter.to_torch(np.arange(3.0), dtype=torch.float32)
    assert counter.count("device_copy") == 3
    with pytest.raises(ValueError, match="to PyTorch needs a copy"):
        adapter.to_torch(np.arange(3.0), dtype=torch.float32, copy=False)
    assert adapter.supports_float64() and adapter.supports_float64("cpu")
    assert not adapter.supports_float64("mps")
    with pytest.raises(TypeError, match="PyTorch would turn float64 into float32"):
        adapter.to_torch(np.arange(3.0), device="mps")
    assert adapter.to_torch(np.arange(3.0), device="cpu").dtype == torch.float64


def test_torch_on_an_apple_gpu():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    from cunumpy.interop import torch as adapter

    tensor = adapter.to_torch(np.ones(3, np.float32), device="mps")
    with count_transfers() as counter:
        back = adapter.from_torch(tensor)
    np.testing.assert_array_equal(back, np.ones(3))
    assert counter.count("device_copy") == 1  # unified memory: no host transfer


def test_mlx_float64_and_copies():
    mx = pytest.importorskip("mlx.core")
    from cunumpy.interop import mlx as adapter

    a = np.arange(3.0)
    with pytest.raises(TypeError, match="MLX would turn float64 into float32"):
        adapter.to_mlx(a, device=mx.gpu)
    assert adapter.to_mlx(a, device=mx.cpu).dtype == mx.float64
    assert adapter.to_mlx(a, dtype=mx.float32).dtype == mx.float32
    assert adapter.supports_float64(mx.cpu) and not adapter.supports_float64(mx.gpu)
    assert adapter.supports_float64() == (mx.default_device() == mx.cpu)
    with pytest.raises(ValueError, match="MLX copies into its own memory"):
        adapter.to_mlx(np.ones(3, np.float32), copy=False)
    with count_transfers() as counter:
        value = adapter.to_mlx(np.ones(3, np.float32))
        same = adapter.to_mlx(value)
        copied = adapter.to_mlx(value, copy=True)
    assert same is value and copied is not value
    assert counter.count("device_copy") == 2
    assert adapter.to_mlx(value, dtype=mx.float16).dtype == mx.float16


def test_mlx_to_numpy_shares_memory():
    mx = pytest.importorskip("mlx.core")
    from cunumpy.interop import mlx as adapter

    value = mx.array(np.ones(3, np.float32))
    with assert_no_transfers():
        view = adapter.from_mlx(value, copy=False)
    view[0] = 5.0
    assert value[0].item() == 5.0


def test_mlx_rejects_device_arrays():
    pytest.importorskip("mlx.core")
    from cunumpy.interop import mlx as adapter

    class OnCuda:
        def __dlpack_device__(self):
            return (2, 0)

    with pytest.raises(ValueError, match="MLX cannot take a OnCuda"):
        adapter.to_mlx(OnCuda())
