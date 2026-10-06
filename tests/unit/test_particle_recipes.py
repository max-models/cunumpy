"""The recipes of the "Particle codes" page (docs/source/examples/particle_recipes.py)."""

import importlib.util
from pathlib import Path

import numpy as np

import cunumpy as xp

_PATH = Path(__file__).resolve().parents[2] / "docs/source/examples/particle_recipes.py"
_spec = importlib.util.spec_from_file_location("particle_recipes", _PATH)
recipes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipes)


def test_remove_dead_and_compact_in_place():
    markers = np.arange(12.0).reshape(6, 2)
    alive = np.array([True, False, True, True, False, True])
    np.testing.assert_array_equal(recipes.remove_dead(markers, alive), markers[alive])
    buffer = markers.copy()
    n = recipes.compact_in_place(buffer, alive)
    assert n == 4
    np.testing.assert_array_equal(buffer[:n], markers[alive])


def test_sort_by_cell_and_deposit():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.1, 1.1, 1000)  # some outside: clipped to the edge cells
    weights = rng.random(1000)
    order, cell, offsets = recipes.sort_by_cell(x, 0.0, 0.1, 10)
    assert offsets[0] == 0 and offsets[-1] == 1000
    sorted_x = x[order]
    for c in range(10):
        members = sorted_x[offsets[c] : offsets[c + 1]]
        expected = np.clip(np.floor(members / 0.1), 0, 9)
        assert np.all(expected == c) and np.all(cell[offsets[c] : offsets[c + 1]] == c)
    # stable: the original order within a cell
    first_cell = order[offsets[3] : offsets[4]]
    assert np.all(np.diff(first_cell) > 0)
    deposit = recipes.deposit_nearest_cell(cell, weights[order], 10)
    expected = np.bincount(np.clip(np.floor(x / 0.1), 0, 9).astype(int), weights, 10)
    np.testing.assert_allclose(deposit, expected, rtol=1e-12)


def test_pack_for_ranks():
    markers = np.arange(10.0).reshape(5, 2)
    destination = np.array([2, 0, 2, 1, 0])
    sendbuf, counts = recipes.pack_for_ranks(markers, destination, 3)
    assert counts.tolist() == [2, 1, 2]
    np.testing.assert_array_equal(sendbuf, markers[[1, 4, 3, 0, 2]])


def test_exchange_on_one_rank():
    MPI = xp.mpi.get_mpi()
    markers = np.arange(8.0).reshape(4, 2)
    received = recipes.exchange(MPI.COMM_SELF, markers, np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(received, markers)


def test_thermal_velocities_are_reproducible():
    ids = np.arange(100_000, dtype=np.uint64)
    v0, v1 = recipes.thermal_velocities(7, ids, 3, 2.0)
    assert abs(v0.std() - 2.0) < 0.03 and abs(v1.mean()) < 0.03
    again, _ = recipes.thermal_velocities(7, ids[::-1], 3, 2.0)  # other order
    np.testing.assert_array_equal(again[::-1], v0)
