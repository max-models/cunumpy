"""Hilbert keys: a permutation of the cells along a curve without jumps."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import cunumpy as xp


@pytest.mark.parametrize(("ndim", "levels"), [(2, 1), (2, 3), (2, 5), (3, 1), (3, 3)])
def test_consecutive_keys_are_neighbouring_cells(ndim, levels) -> None:
    n = 2**levels
    cells = np.array(list(itertools.product(range(n), repeat=ndim)), dtype=np.int64).T
    keys = np.asarray(xp.algorithms.hilbert_encode(*cells, levels=levels))
    assert keys.dtype == np.uint64
    order = np.argsort(keys)
    assert np.array_equal(np.sort(keys), np.arange(n**ndim, dtype=np.uint64))
    walk = cells[:, order]
    steps = np.abs(np.diff(walk, axis=1)).sum(axis=0)
    assert np.all(steps == 1)  # one face at a time, no jumps
    assert np.all(walk[:, 0] == 0)  # it starts at the origin


def test_the_first_level_is_a_u_and_finer_levels_nest_in_it() -> None:
    keys = xp.algorithms.hilbert_encode(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), levels=1
    )
    assert np.asarray(keys).tolist() == [0, 1, 2, 3]
    # the cells of a node of the quadtree are a contiguous range of keys
    i, j = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    keys = np.asarray(xp.algorithms.hilbert_encode(i, j, levels=3))
    for a, b in itertools.product(range(4), repeat=2):
        block = np.sort(keys[2 * a : 2 * a + 2, 2 * b : 2 * b + 2].ravel())
        assert block[-1] - block[0] == 3 and block[0] % 4 == 0


def test_hilbert_keys_check_their_arguments() -> None:
    with pytest.raises(ValueError, match="2 or 3"):
        xp.algorithms.hilbert_encode(np.zeros(2, dtype=int), levels=2)
    with pytest.raises(ValueError, match="levels"):
        xp.algorithms.hilbert_encode(
            np.zeros(2, dtype=int), np.zeros(2, dtype=int), levels=0
        )
