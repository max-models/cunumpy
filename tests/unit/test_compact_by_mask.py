"""compact_by_mask keeps the masked rows, in order, at the front of every array."""

import numpy as np
import pytest

import cunumpy as xp


def test_rows_are_moved_to_the_front_in_order_for_every_array():
    markers = np.arange(12.0).reshape(4, 3)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    ids = np.array([10, 11, 12, 13])
    alive = np.array([True, False, True, True])
    n = xp.algorithms.compact_by_mask(alive, markers, weights, ids)
    assert n == 3
    np.testing.assert_array_equal(markers[:n], [[0, 1, 2], [6, 7, 8], [9, 10, 11]])
    np.testing.assert_array_equal(weights[:n], [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(ids[:n], [10, 12, 13])


def test_all_true_none_true_and_empty():
    a = np.arange(4)
    assert xp.algorithms.compact_by_mask(np.ones(4, bool), a) == 4
    np.testing.assert_array_equal(a, np.arange(4))
    assert xp.algorithms.compact_by_mask(np.zeros(4, bool), a) == 0
    assert xp.algorithms.compact_by_mask(np.zeros(0, bool), np.zeros(0)) == 0


def test_matches_boolean_indexing_on_random_masks():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.random((50, 2))
        mask = rng.random(50) < 0.4
        expected = a[mask]
        n = xp.algorithms.compact_by_mask(mask, a)
        np.testing.assert_array_equal(a[:n], expected)


def test_argument_errors():
    with pytest.raises(TypeError, match="boolean"):
        xp.algorithms.compact_by_mask(np.array([1, 0]), np.zeros(2))
    with pytest.raises(ValueError, match="2 entries along axis 0"):
        xp.algorithms.compact_by_mask(np.array([True, False]), np.zeros(3))


def test_last_axis_compacts_component_major_and_scalar_arrays_together():
    positions = np.arange(12.0).reshape(3, 4)  # (ncomp, N)
    weights = np.array([1.0, 2.0, 3.0, 4.0])  # (N,)
    alive = np.array([False, True, True, False])
    n = xp.algorithms.compact_by_mask(alive, positions, weights, axis=-1)
    assert n == 2
    np.testing.assert_array_equal(positions[:, :n], [[1, 2], [5, 6], [9, 10]])
    np.testing.assert_array_equal(weights[:n], [2.0, 3.0])


def test_axis_matches_boolean_indexing_on_random_masks():
    rng = np.random.default_rng(2)
    for axis in (1, -1):
        a = rng.random((3, 40))
        mask = rng.random(40) < 0.5
        expected = a[:, mask]
        n = xp.algorithms.compact_by_mask(mask, a, axis=axis)
        np.testing.assert_array_equal(a[:, :n], expected)


def test_compacting_a_column_prefix_view_keeps_the_storage():
    storage = np.arange(20.0).reshape(2, 10)
    view = storage[:, :6]  # rows further apart than the view is wide
    n = xp.algorithms.compact_by_mask(np.arange(6) % 2 == 0, view, axis=-1)
    np.testing.assert_array_equal(storage[:, :n], [[0, 2, 4], [10, 12, 14]])
    np.testing.assert_array_equal(storage[:, 6:], [[6, 7, 8, 9], [16, 17, 18, 19]])


def test_axis_errors():
    with pytest.raises(ValueError, match="3 entries along axis -1"):
        xp.algorithms.compact_by_mask(np.ones(3, bool), np.zeros((3, 2)), axis=-1)
    with pytest.raises(ValueError, match="no axis 1"):
        xp.algorithms.compact_by_mask(np.ones(3, bool), np.zeros(3), axis=1)
