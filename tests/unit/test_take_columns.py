"""take_columns gathers columns of (ncomp, n) arrays like array[:, index], C-contiguous."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _algorithms
from cunumpy.kernels import use_host_kernel_implementation

DTYPES = [np.float64, np.float32, np.int64, np.int32, np.bool_, np.complex128]
IMPLEMENTATIONS = ["default", "numpy"]


def make(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 100).astype(dtype)


def run(implementation, *args, **kwargs):
    if implementation == "default":
        return xp.algorithms.take_columns(*args, **kwargs)
    with use_host_kernel_implementation(implementation):
        return xp.algorithms.take_columns(*args, **kwargs)


@pytest.fixture(params=IMPLEMENTATIONS)
def implementation(request, monkeypatch):
    # run the compiled kernel on the small test arrays too
    monkeypatch.setattr(_algorithms, "_TAKE_COLUMNS_MIN_COLUMNS", 0)
    return request.param


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (8, 1000), (3, 5000)])
def test_integer_index_matches_fancy_indexing(implementation, dtype, shape):
    array = make(shape, dtype)
    index = np.random.default_rng(1).integers(0, shape[1], 2 * shape[1] // 3 + 1)
    result = run(implementation, array, index)
    assert result.dtype == array.dtype
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, array[:, index])


def test_permutation_of_a_large_array(implementation):
    array = make((3, 100_000), np.float64)
    order = np.random.default_rng(2).permutation(100_000)
    np.testing.assert_array_equal(run(implementation, array, order), array[:, order])


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_index_dtypes(implementation, index_dtype):
    array = make((2, 50), np.float64)
    index = np.array([4, 0, 49, 4], dtype=index_dtype)
    np.testing.assert_array_equal(run(implementation, array, index), array[:, index])


def test_negative_indices_count_from_the_end(implementation):
    array = make((2, 10), np.float64)
    index = np.array([-1, 0, -10, 3])
    np.testing.assert_array_equal(run(implementation, array, index), array[:, index])


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_strided_column_prefix_view(implementation, dtype):
    storage = make((3, 40), dtype)
    view = storage[:, :25]  # the rows are further apart than the view is wide
    index = np.random.default_rng(3).permutation(25)
    np.testing.assert_array_equal(run(implementation, view, index), view[:, index])


def test_non_row_contiguous_input(implementation):
    base = make((9, 6), np.float64)
    for array in (base.T, base[:, ::2], np.asfortranarray(base)):
        index = np.arange(array.shape[1])[::-1]
        np.testing.assert_array_equal(
            run(implementation, array, index), array[:, index]
        )


def test_read_only_input(implementation):
    array = make((2, 20), np.float64)
    array.flags.writeable = False
    index = np.arange(20)[::-1]
    np.testing.assert_array_equal(run(implementation, array, index), array[:, index])


def test_empty_index_and_empty_array(implementation):
    array = make((3, 5), np.float64)
    result = run(implementation, array, np.array([], dtype=np.int64))
    assert result.shape == (3, 0)
    assert result.dtype == array.dtype
    assert run(implementation, np.zeros((0, 4)), np.array([1, 2])).shape == (0, 2)
    assert run(implementation, np.zeros((3, 0)), np.array([], dtype=int)).shape == (
        3,
        0,
    )
    assert run(implementation, array, np.zeros(5, bool)).shape == (3, 0)


def test_out_is_filled_and_returned(implementation):
    array = make((3, 30), np.float64)
    index = np.random.default_rng(4).permutation(30)[:20]
    out = np.full((3, 20), -1.0)
    result = run(implementation, array, index, out=out)
    assert result is out
    np.testing.assert_array_equal(out, array[:, index])


def test_out_may_be_a_row_block_of_a_larger_array(implementation):
    array = make((2, 10), np.float64)
    storage = np.zeros((4, 10))
    out = storage[:2]  # C-contiguous
    run(implementation, array, np.arange(10)[::-1], out=out)
    np.testing.assert_array_equal(storage[:2], array[:, ::-1])
    assert not storage[2:].any()


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("shape", [(3, 1), (3, 9), (8, 5000)])
def test_boolean_mask_matches_compress(implementation, dtype, shape):
    array = make(shape, dtype)
    mask = np.random.default_rng(5).random(shape[1]) < 0.4
    result = run(implementation, array, mask)
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.compress(mask, array, axis=1))


def test_mask_of_a_strided_view_with_out(implementation):
    storage = make((3, 40), np.float64)
    view = storage[:, :30]
    mask = np.arange(30) % 3 == 0
    out = np.empty((3, 10))
    run(implementation, view, mask, out=out)
    np.testing.assert_array_equal(out, view[:, mask])


def test_the_kernel_is_used_from_the_minimum_number_of_columns(monkeypatch):
    import cunumpy._kernels.take_columns as folder

    calls = []
    monkeypatch.setattr(folder, "kernel", lambda *args: calls.append(args))
    size = _algorithms._TAKE_COLUMNS_MIN_COLUMNS
    array = make((2, size), np.float64)
    xp.algorithms.take_columns(array, np.arange(size - 1))
    assert not calls
    xp.algorithms.take_columns(array, np.arange(size))
    assert len(calls) == 1


def test_error_arguments():
    array = np.zeros((2, 4))
    with pytest.raises(ValueError, match="2D"):
        xp.algorithms.take_columns(np.zeros(4), np.array([0]))
    with pytest.raises(ValueError, match="1D"):
        xp.algorithms.take_columns(array, np.zeros((2, 2), dtype=int))
    with pytest.raises(TypeError, match="integer or boolean"):
        xp.algorithms.take_columns(array, np.array([0.0]))
    with pytest.raises(ValueError, match="one entry per column"):
        xp.algorithms.take_columns(array, np.ones(3, bool))
    with pytest.raises(IndexError, match="out of bounds"):
        xp.algorithms.take_columns(array, np.array([0, 4]))
    with pytest.raises(IndexError, match="out of bounds"):
        xp.algorithms.take_columns(array, np.array([-5]))


def test_out_errors():
    array = np.zeros((2, 4))
    index = np.array([0, 1])
    with pytest.raises(ValueError, match="shape"):
        xp.algorithms.take_columns(array, index, out=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="dtype"):
        xp.algorithms.take_columns(array, index, out=np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="C-contiguous"):
        xp.algorithms.take_columns(array, index, out=np.zeros((2, 4))[:, ::2])
    read_only = np.zeros((2, 2))
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        xp.algorithms.take_columns(array, index, out=read_only)
    with pytest.raises(ValueError, match="overlap"):
        xp.algorithms.take_columns(array, np.arange(4), out=array)


def test_compiled_kernel_agrees_with_numpy_implementation(monkeypatch):
    monkeypatch.setattr(_algorithms, "_TAKE_COLUMNS_MIN_COLUMNS", 0)
    from cunumpy._kernels.take_columns import kernel

    pytest.importorskip("pyccel")
    assert kernel.selected() == "pyccel"
    array = make((3, 3000), np.float64)
    index = np.random.default_rng(6).permutation(3000)
    with use_host_kernel_implementation("pyccel"):
        compiled = xp.algorithms.take_columns(array, index)
    with use_host_kernel_implementation("numpy"):
        reference = xp.algorithms.take_columns(array, index)
    np.testing.assert_array_equal(compiled, reference)


def test_cupy_backend_matches_numpy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    array = make((3, 200), np.float64)
    index = np.random.default_rng(7).permutation(200)[:150]
    mask = index.size and np.random.default_rng(8).random(200) < 0.5
    with xp.use_backend("cupy"):
        device = xp.to_cupy(array)
        taken = xp.algorithms.take_columns(device, xp.to_cupy(index))
        masked = xp.algorithms.take_columns(device, xp.to_cupy(mask))
        out = xp.zeros((3, 150))
        xp.algorithms.take_columns(device, xp.to_cupy(index), out=out)
        negative = xp.algorithms.take_columns(device, xp.to_cupy(np.array([-1, 0])))
    np.testing.assert_array_equal(xp.to_numpy(taken), array[:, index])
    np.testing.assert_array_equal(xp.to_numpy(masked), array[:, mask])
    np.testing.assert_array_equal(xp.to_numpy(out), array[:, index])
    np.testing.assert_array_equal(xp.to_numpy(negative), array[:, [-1, 0]])
