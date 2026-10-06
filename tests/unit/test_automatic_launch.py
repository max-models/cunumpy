"""Shape-based CUDA launch defaults, without allocating device memory."""

from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.arguments import CudaArguments, CudaStructArguments, CudaStructValue
from cunumpy.kernels import CudaKernel

SOURCE = 'extern "C" __global__ void work() {}'


@pytest.mark.parametrize(
    "shape,block,expected",
    [
        ((1000,), 128, ((8,), (128,))),
        ((1000, 6), 128, ((8,), (128,))),
        ((0, 6), 128, ((0,), (128,))),
        ((35, 19), (8, 4), ((5, 5), (8, 4))),
        ((35, 19, 7, 3), (8, 4, 2), ((5, 5, 4), (8, 4, 2))),
    ],
)
def test_default_inference_matches_leading_axes_to_block_dimensions(
    shape, block, expected
):
    kernel = CudaKernel(SOURCE, "work", block_size=block)
    args = (2.0, np.empty(shape), np.empty(1))
    assert kernel.launch_shape(args=args) == expected
    inferred = kernel.n_threads_from(args)
    assert inferred == (shape[0] if isinstance(block, int) else shape[: len(block)])


def test_block_override_changes_inferred_dimensions():
    kernel = CudaKernel(SOURCE, "work")
    args = (np.empty((35, 19)),)
    assert kernel.launch_shape(args=args, block=(8, 4)) == ((5, 5), (8, 4))
    kernel = CudaKernel(SOURCE, "work", block_size=(8, 4))
    assert kernel.launch_shape(args=args, block=16) == ((3,), (16,))


def test_explicit_sizes_bypass_inference():
    kernel = CudaKernel(SOURCE, "work")
    assert kernel.launch_shape(300, args=()) == ((3,), (128,))
    assert kernel.launch_shape(grid=4, args=()) == ((4,), (128,))
    with pytest.raises(TypeError, match="exactly one"):
        kernel.launch_shape(300, grid=4, args=())


def test_callbacks_and_explicit_opt_out():
    kernel = CudaKernel(SOURCE, "work", n_threads_from=lambda args: args[0])
    assert kernel.launch_shape(args=(300,)) == ((3,), (128,))
    kernel.n_threads_from = None
    with pytest.raises(TypeError, match="exactly one"):
        kernel.launch_shape(args=(np.empty(300),))
    kernel.n_threads_from = "auto"
    assert kernel.launch_shape(args=(np.empty(300),)) == ((3,), (128,))
    kernel.n_threads_from = "first_array"
    assert kernel.launch_shape(args=(np.empty((300, 7)),)) == ((3,), (128,))


def test_invalid_setting_preserves_auto_inference():
    kernel = CudaKernel(SOURCE, "work")
    with pytest.raises(TypeError, match="n_threads_from"):
        kernel.n_threads_from = "invalid"
    assert kernel.launch_shape(args=(np.empty((35, 19)),), block=(8, 4)) == (
        (5, 5),
        (8, 4),
    )


def test_missing_array_and_insufficient_dimensions_raise_clear_errors():
    kernel = CudaKernel(SOURCE, "work")
    with pytest.raises(TypeError, match="needs an array argument"):
        kernel.launch_shape(args=(2.0, np.array(3.0)))
    with pytest.raises(ValueError, match="cannot infer 2D n_threads"):
        kernel.launch_shape(args=(np.empty(3), np.empty((5, 7))), block=(4, 4))
    assert kernel.launch_shape(args=(np.array(3.0), np.empty(300))) == ((3,), (128,))


@pytest.mark.parametrize("kind", ["flattened", "struct_arguments", "struct_value"])
def test_inference_finds_arrays_in_supported_argument_objects(kind):
    array = np.empty((300, 6))
    if kind == "flattened":
        args = CudaArguments(3.0, array)
    elif kind == "struct_arguments":

        class Arguments(CudaStructArguments):
            struct_name = "Arguments"
            fields = (("count", "int"), ("data", "Array2D<double>"))

        # Shape inspection does not pack host arrays into CUDA structs.
        args = Arguments()
        args.count, args.data = 300, array
    else:
        struct = SimpleNamespace(
            fields=[SimpleNamespace(name="count"), SimpleNamespace(name="data")]
        )
        args = CudaStructValue(struct, None, {"count": 300, "data": array})
    kernel = CudaKernel(SOURCE, "work")
    assert kernel.launch_shape(args=(args, np.empty(1))) == ((3,), (128,))


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
def test_automatic_multidimensional_launch_on_gpu():
    import cupy as cp

    source = r"""
    #include <cunumpy/array_view.cuh>
    extern "C" __global__ void fill(Array2D<long long> a) {
        long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
        long long j = blockIdx.y * (long long)blockDim.y + threadIdx.y;
        if (i < a.shape[0] && j < a.shape[1]) a(i, j) = 10 * i + j;
    }
    """
    array = cp.zeros((5, 7), dtype=cp.int64)
    kernel = CudaKernel(source, "fill", block_size=(2, 4), options=("-lineinfo",))
    with xp.profiling.assert_no_transfers():
        kernel(array)
    np.testing.assert_array_equal(
        cp.asnumpy(array), 10 * np.arange(5)[:, None] + np.arange(7)
    )
