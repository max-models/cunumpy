"""Tests for Morton keys (host functions and cunumpy/morton.cuh) and sort_by_key."""

from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.cuda import cuda_include_dir
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from cunumpy.kernels import CudaKernel


def interleave(cells, levels):
    """Bit-by-bit reference for morton_encode."""
    ndim = len(cells)
    key = 0
    for bit in range(levels):
        for axis, cell in enumerate(cells):
            key |= ((int(cell) >> bit) & 1) << (ndim * bit + axis)
    return key


@pytest.mark.parametrize("ndim", [2, 3])
def test_encode_matches_bitwise_reference(ndim):
    levels = xp.algorithms.MAX_MORTON_LEVELS[ndim]
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 2**levels, size=(ndim, 500), dtype=np.uint64)
    cells[:, 0] = 2**levels - 1  # all bits set
    cells[:, 1] = 0
    keys = xp.algorithms.morton_encode(*cells)
    assert keys.dtype == np.uint64
    expected = [interleave(cells[:, i], levels) for i in range(cells.shape[1])]
    assert keys.tolist() == expected


@pytest.mark.parametrize("ndim", [2, 3])
def test_decode_inverts_encode(ndim):
    levels = xp.algorithms.MAX_MORTON_LEVELS[ndim]
    rng = np.random.default_rng(1)
    cells = rng.integers(0, 2**levels, size=(ndim, 1000), dtype=np.uint64)
    decoded = xp.algorithms.morton_decode(xp.algorithms.morton_encode(*cells), ndim)
    for axis in range(ndim):
        np.testing.assert_array_equal(decoded[axis], cells[axis])


def test_encode_broadcasts_and_rejects_bad_dimensions():
    keys = xp.algorithms.morton_encode(np.arange(4)[:, None], np.arange(3))
    assert keys.shape == (4, 3)
    assert keys[2, 1] == interleave((2, 1), 2)
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        xp.algorithms.morton_encode(np.arange(3))


def test_keys_cells_and_clipping():
    levels = 3  # 8 cells per axis on [0, 1]
    positions = np.array(
        [
            [0.0, 0.0],
            [0.125, 0.0],  # on a cell boundary: the upper cell
            [0.999, 0.5],
            [1.0, 1.0],  # upper face: the last cell
            [-5.0, 7.0],  # outside: the nearest face
        ],
    )
    keys = xp.algorithms.morton_keys(positions, [0.0, 0.0], [1.0, 1.0], levels)
    cells = [(0, 0), (1, 0), (7, 4), (7, 7), (0, 7)]
    assert keys.tolist() == [interleave(c, levels) for c in cells]


def test_reversed_axis():
    # lower > upper on y: cell 0 at the top, like a quadtree with y < mid as
    # its second quadrant bit
    keys = xp.algorithms.morton_keys(
        np.array([[0.2, 0.9], [0.2, 0.1]]),
        [0, 1],
        [1, 0],
        1,
    )
    assert keys.tolist() == [0, 2]


def test_keys_validate_arguments():
    with pytest.raises(ValueError, match="levels"):
        xp.algorithms.morton_keys(np.zeros((3, 2)), [0, 0], [1, 1], 33)
    with pytest.raises(ValueError, match="levels"):
        xp.algorithms.morton_keys(np.zeros((3, 3)), [0, 0, 0], [1, 1, 1], 22)
    with pytest.raises(ValueError, match="differ"):
        xp.algorithms.morton_keys(np.zeros((3, 2)), [0, 0], [1, 0], 4)
    with pytest.raises(ValueError, match="need 2 values"):
        xp.algorithms.morton_keys(np.zeros((3, 2)), [0, 0, 0], [1, 1, 1], 4)
    with pytest.raises(ValueError, match=r"\(n, ndim\)"):
        xp.algorithms.morton_keys(np.zeros(3), [0, 0], [1, 1], 4)


def test_sorted_keys_make_tree_nodes_contiguous():
    rng = np.random.default_rng(2)
    positions = rng.random((2000, 2))
    levels = 10
    keys, _, sorted_positions = xp.algorithms.sort_by_key(
        xp.algorithms.morton_keys(positions, [0, 0], [1, 1], levels),
        positions,
    )
    for level in (1, 2, 3):
        node = keys >> np.uint64(2 * (levels - level))
        assert np.all(node[1:] >= node[:-1])
        # every point of a node lies in that node's square
        size = 0.5**level
        cx, cy = xp.algorithms.morton_decode(node, 2)
        assert np.all(np.floor(sorted_positions[:, 0] / size) == cx)
        assert np.all(np.floor(sorted_positions[:, 1] / size) == cy)


def test_sort_by_key_is_stable_and_reorders_all_arrays():
    keys = np.array([2, 0, 1, 0, 2], dtype=np.uint64)
    ids = np.arange(5)
    rows = np.arange(10.0).reshape(5, 2)
    sorted_keys, order, sorted_ids, sorted_rows = xp.algorithms.sort_by_key(
        keys,
        ids,
        rows,
    )
    assert order.dtype == np.int64
    assert order.tolist() == [1, 3, 2, 0, 4]
    assert sorted_keys.tolist() == [0, 0, 1, 2, 2]
    np.testing.assert_array_equal(sorted_ids, ids[order])
    np.testing.assert_array_equal(sorted_rows, rows[order])
    assert len(xp.algorithms.sort_by_key(keys)) == 2


def test_sort_by_key_validates_shapes():
    with pytest.raises(ValueError, match="1D"):
        xp.algorithms.sort_by_key(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="expected 3 entries"):
        xp.algorithms.sort_by_key(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="no axis 1"):
        xp.algorithms.sort_by_key(np.zeros(3), np.zeros(3), axis=1)


@pytest.mark.parametrize(
    ("dtype", "low", "high"),
    [
        (np.int64, 0, 300),  # one 16-bit pass, many equal keys
        (np.int64, 0, 400_000),  # two passes: cells of a 3D grid
        (np.int64, -(2**40), 2**40),  # negative keys, three passes
        (np.int32, -5, 70_000),
        (np.uint64, 0, 2**63),  # Morton keys, four passes
    ],
)
def test_radix_sort_of_integer_keys_matches_the_stable_argsort(dtype, low, high):
    keys = np.random.default_rng(5).integers(low, high, 50_000, dtype=dtype)
    _, order, sorted_ids = xp.algorithms.sort_by_key(keys, np.arange(keys.size))
    expected = np.argsort(keys, kind="stable")
    assert order.dtype == np.int64
    np.testing.assert_array_equal(order, expected)
    np.testing.assert_array_equal(sorted_ids, expected)


def test_sort_by_key_along_the_last_axis_of_component_major_arrays():
    keys = np.array([2, 0, 1, 0], dtype=np.int64)
    positions = np.arange(12.0).reshape(3, 4)
    weights = np.array([10.0, 11.0, 12.0, 13.0])
    _, order, sorted_positions, sorted_weights = xp.algorithms.sort_by_key(
        keys,
        positions,
        weights,
        axis=-1,
    )
    assert order.tolist() == [1, 3, 2, 0]
    np.testing.assert_array_equal(sorted_positions, positions[:, order])
    np.testing.assert_array_equal(sorted_weights, weights[order])


def test_header_is_shipped():
    header = (Path(cuda_include_dir()) / "cunumpy" / "morton.cuh").read_text()
    for name in (
        "cunumpy_morton_key2",
        "cunumpy_morton_key3",
        "cunumpy_morton_encode3",
    ):
        assert name in header


KEYS = r"""
#include "cunumpy/morton.cuh"
extern "C" __global__
void keys2(const double* pos, unsigned long long* key, long long n,
           double x0, double y0, double sx, double sy, int levels) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = cunumpy_morton_key2(pos[2 * i], pos[2 * i + 1], x0, y0, sx, sy, levels);
}
extern "C" __global__
void keys3(const double* pos, unsigned long long* key, long long n,
           double x0, double y0, double z0, double sx, double sy, double sz,
           int levels) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = cunumpy_morton_key3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2],
                                 x0, y0, z0, sx, sy, sz, levels);
}
"""


def _cases(ndim):
    rng = np.random.default_rng(3)
    lower = [-1.5, 0.25, 2.0][:ndim]
    upper = [2.5, -0.75, 3.0][:ndim]  # the second axis is reversed
    positions = rng.uniform(-2.0, 3.5, size=(997, ndim))
    levels = xp.algorithms.MAX_MORTON_LEVELS[ndim]
    # points on cell boundaries, where rounding would show
    edges = np.array(lower) + np.arange(5)[:, None] / xp.algorithms.morton_scales(
        lower,
        upper,
        levels,
    )
    positions = np.ascontiguousarray(np.vstack([positions, edges]))
    return positions, lower, upper, levels


def _device_keys(run, ndim):
    positions, lower, upper, levels = _cases(ndim)
    n = positions.shape[0]
    keys = np.zeros(n, dtype=np.uint64)
    scales = xp.algorithms.morton_scales(lower, upper, levels)
    args = (*lower, *scales.tolist(), levels)
    run(CudaKernel(KEYS, f"keys{ndim}"), positions, keys, n, *args, n_threads=n)
    return keys, xp.algorithms.morton_keys(positions, lower, upper, levels)


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
@pytest.mark.parametrize("ndim", [2, 3])
def test_header_matches_the_host_keys_in_emulation(ndim):
    device, host = _device_keys(emulate_cuda_kernel, ndim)
    np.testing.assert_array_equal(device, host)


def _run_on_gpu(kernel, *args, n_threads):
    import cupy as cp

    device = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    kernel(*device, n_threads=n_threads)
    for host, dev in zip(args, device):
        if isinstance(host, np.ndarray):
            host[...] = cp.asnumpy(dev)


@pytest.mark.parametrize("ndim", [2, 3])
def test_header_matches_the_host_keys_on_gpu(ndim):
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    device, host = _device_keys(_run_on_gpu, ndim)
    np.testing.assert_array_equal(device, host)


def test_cupy_arrays_stay_on_the_device():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    positions, lower, upper, levels = _cases(2)
    keys = xp.algorithms.morton_keys(cp.asarray(positions), lower, upper, levels)
    assert isinstance(keys, cp.ndarray)
    np.testing.assert_array_equal(
        cp.asnumpy(keys),
        xp.algorithms.morton_keys(positions, lower, upper, levels),
    )
    sorted_keys, order, _ = xp.algorithms.sort_by_key(keys, cp.asarray(positions))
    assert isinstance(order, cp.ndarray)
    assert bool((sorted_keys[1:] >= sorted_keys[:-1]).all())
    cx, _ = xp.algorithms.morton_decode(sorted_keys, 2)
    assert isinstance(cx, cp.ndarray)
