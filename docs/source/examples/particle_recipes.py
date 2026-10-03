"""Recipes for particle codes with cunumpy: the code of the "Particle codes" page.

Every function runs on NumPy and on CuPy arrays (``xp`` follows the active
backend); ``tests/unit/test_particle_recipes.py`` runs them.
"""

from __future__ import annotations

import numpy as np

import cunumpy as xp


def remove_dead(markers, alive):
    """Drop the markers whose ``alive`` flag is False (a new, compact array)."""
    return markers[alive]


def compact_in_place(markers, alive):
    """Move the live markers to the front of a preallocated buffer; return their count.

    ``markers[:n]`` are the live markers afterwards, in their original order,
    and the rows behind them are free for injection. The right-hand side is a
    copy (boolean indexing), so source and destination may overlap.
    """
    n_alive = int(alive.sum())
    markers[:n_alive] = markers[alive]
    return n_alive


def sort_by_cell(positions, lower, cell_size, n_cells):
    """Order the markers by cell; return the order, the cells and the cell offsets.

    After ``markers = markers[order]`` the markers of cell ``c`` are
    ``markers[offsets[c]:offsets[c + 1]]``. The sort is stable, so markers
    keep their relative order within a cell (reproducible results).
    """
    cell = xp.floor((positions - lower) / cell_size).astype(xp.int64)
    cell = xp.clip(cell, 0, n_cells - 1)
    order = xp.argsort(cell, kind="stable")
    counts = xp.bincount(cell, minlength=n_cells)
    offsets = xp.zeros(n_cells + 1, dtype=xp.int64)
    offsets[1:] = xp.cumsum(counts)
    return order, cell[order], offsets


def deposit_nearest_cell(cell, weights, n_cells):
    """Sum the weights per cell without atomics: the sort-then-reduce deposit."""
    return xp.algorithms.segment_sum(weights, cell, n_cells)


def pack_for_ranks(markers, destination, n_ranks):
    """Group the markers by destination rank; return the send buffer and the counts.

    ``destination[i]`` is the rank marker ``i`` moves to (its own rank to stay).
    The markers for rank ``r`` are the rows ``displacements[r]`` to
    ``displacements[r] + counts[r]`` of the buffer.
    """
    order = xp.argsort(destination, kind="stable")
    counts = xp.bincount(destination, minlength=n_ranks)
    return xp.ascontiguousarray(markers[order]), counts


def exchange(comm, markers, destination):
    """Send every marker to its destination rank (``MPI_Alltoallv``); return the received ones.

    Works for host and device arrays, with or without CUDA-aware MPI
    (``xp.mpi.mpi_buffer`` stages device buffers through the host when needed).
    """
    n_ranks = comm.Get_size()
    width = markers.shape[1]
    sendbuf, counts = pack_for_ranks(markers, destination, n_ranks)
    send_counts = np.asarray(xp.to_numpy(counts), dtype=np.int64)
    recv_counts = np.empty(n_ranks, dtype=np.int64)
    comm.Alltoall(send_counts, recv_counts)  # how many rows come from each rank
    received = xp.empty((int(recv_counts.sum()), width), dtype=markers.dtype)

    def displacements(counts):
        return np.concatenate([[0], np.cumsum(counts)[:-1]])

    with (
        xp.mpi.mpi_buffer(sendbuf) as send,
        xp.mpi.mpi_buffer(received, send=False, recv=True) as recv,
    ):
        comm.Alltoallv(
            [send, send_counts * width, displacements(send_counts) * width, None],
            [recv, recv_counts * width, displacements(recv_counts) * width, None],
        )
    return received


def thermal_velocities(seed, particle_ids, step, v_th):
    """Maxwellian velocities from (seed, particle id, step): no generator state.

    The same numbers as ``cunumpy_normal2(seed, id, step, ...)`` in a kernel
    (up to the last bits of the math functions), whatever the order of the
    particles or the number of ranks.
    """
    z0, z1 = xp.rng.philox_normal2(seed, particle_ids, step)
    return v_th * z0, v_th * z1
