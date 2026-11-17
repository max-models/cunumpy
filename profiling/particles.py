"""Particle step, weak scaling.

The steps of a particle code on cunumpy's algorithms, with the same number of markers on every
rank: push the markers, send the ones that left the rank's slab of the unit cube to its
neighbours, sort them by their Morton key and deposit their weights per cell. Every rank owns
the slab ``rank / size <= x < (rank + 1) / size``; the domain is periodic.

Run it on its own (``python profiling/particles.py --profile-out prof.h5``, or under ``mpirun``)
or through ``scope-profiler jobs run profiling/particles_weak.toml``, which runs it at several
rank counts and uploads the results to the profiling catalogue.
"""

from __future__ import annotations

import argparse

import numpy as np
import scope_profiler as sp

import cunumpy as xp

LEVELS = 6  # Morton levels: 2**6 cells per direction


def leavers(positions, rank, size):
    """Return the destination of every marker: -1 to stay, else the neighbour (0 left, 1 right)."""
    owner = xp.floor(positions[:, 0] * size).astype(xp.int64) % size
    destination = xp.full(owner.shape, -1, dtype=xp.int64)
    if size > 1:
        destination[owner == (rank - 1) % size] = 0
        destination[owner == (rank + 1) % size] = 1
    return destination


def send_to_neighbours(comm, markers, destination):
    """Send the leaving markers to the left and right neighbour; return the markers kept."""
    rank, size = comm.Get_rank(), comm.Get_size()
    if size == 1:
        return markers
    left, right = (rank - 1) % size, (rank + 1) % size
    outgoing = [xp.ascontiguousarray(markers[destination == side]) for side in (0, 1)]
    counts_out = [np.array([len(block)], dtype=np.int64) for block in outgoing]
    counts_in = [np.zeros(1, dtype=np.int64) for _ in range(2)]
    xp.mpi.exchange(
        comm,
        sends=[(counts_out[0], left, 0), (counts_out[1], right, 1)],
        receives=[(counts_in[0], right, 0), (counts_in[1], left, 1)],
    )
    width = markers.shape[1]
    incoming = [xp.empty((int(c[0]), width), dtype=markers.dtype) for c in counts_in]
    xp.mpi.exchange(
        comm,
        sends=[(outgoing[0], left, 2), (outgoing[1], right, 3)],
        receives=[(incoming[0], right, 2), (incoming[1], left, 3)],
    )
    kept = markers[destination == -1]
    return xp.concatenate([kept, *incoming])


@sp.profile("step")
def step(comm, markers, dt):
    """One time step: push, exchange, sort and deposit; returns the markers and the density."""
    rank, size = comm.Get_rank(), comm.Get_size()
    with sp.region("push"):
        markers[:, :3] = (markers[:, :3] + dt * markers[:, 3:6]) % 1.0
    with sp.region("exchange"):
        markers = send_to_neighbours(comm, markers, leavers(markers[:, :3], rank, size))
    with sp.region("sort"):
        keys = xp.algorithms.morton_keys(
            markers[:, :3], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), LEVELS
        )
        keys, _, markers = xp.algorithms.sort_by_key(keys, markers)
    with sp.region("deposit"):
        plan = xp.algorithms.SegmentPlan(keys, 2 ** (3 * LEVELS))
        density = plan.sum(markers[:, 6])
    return markers, density


def run(markers_per_rank, steps, seed=0):
    """Run ``steps`` steps with ``markers_per_rank`` markers on every rank."""
    comm = xp.mpi.get_mpi().COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(seed + rank)
    markers = np.empty((markers_per_rank, 7))
    markers[:, 0] = (rank + rng.random(markers_per_rank)) / size  # x in the rank's slab
    markers[:, 1:3] = rng.random((markers_per_rank, 2))
    markers[:, 3:6] = rng.normal(0.0, 0.05, (markers_per_rank, 3))
    markers[:, 6] = 1.0 / markers_per_rank
    markers = xp.asarray(markers)
    dt = 0.5 / (size * 2**LEVELS)  # most markers stay in their slab
    for _ in range(steps):
        markers, density = step(comm, markers, dt)
    return markers, density


def main():
    """Run the case from the command line and write the profile."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markers", type=int, default=200_000, help="markers per rank")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profile-out", default="profiling_data.h5")
    args = parser.parse_args()
    with sp.session(file_path=args.profile_out, enable_tags=["cunumpy"], verbose=False):
        run(args.markers, args.steps)


if __name__ == "__main__":
    main()
