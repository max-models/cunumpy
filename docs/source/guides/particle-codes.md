# Particle codes

Recipes for the parts of a particle-in-cell (or any particle) code around the
kernels: removing and sorting markers, depositing without atomics, moving
markers between MPI ranks, random numbers per particle, output, and running a
time step as a CUDA graph. The NumPy recipes are the functions of
`docs/source/examples/particle_recipes.py`, which the test suite runs; each
works on NumPy and CuPy arrays alike.

## Remove dead markers

Markers leave the domain, hit a wall or are absorbed. Keep a boolean `alive`
array and compact:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: remove_dead
```

To avoid reallocating, keep the markers in a preallocated buffer with a count
of live rows, and move the live ones to the front; the free rows behind them
take injected markers:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: compact_in_place
```

`alive.sum()` on the GPU is a reduction plus one small transfer (the count);
do it once per step, not per species and kernel.

## Sort markers by cell

Sorting by cell makes deposits and gathers read and write memory in order,
which is often worth more than the sort costs, and gives each cell a
contiguous range of markers, which binary collisions and per-cell diagnostics
need:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: sort_by_cell
```

Apply `order` to every per-marker array (positions, velocities, weights, ids),
e.g. by keeping them as columns of one `(n, k)` array. A stable sort keeps the
result independent of how the markers were ordered before.
`xp.algorithms.sort_by_key(keys, positions, velocities, weights)` does the argsort and
the reordering of several arrays in one call.

For a tree code, or for better locality in 2D and 3D, sort by Morton key
(`xp.algorithms.morton_keys`, see the API page) instead of by cell: the markers of every
quadtree or octree node are then a contiguous range of the sorted arrays.

## Deposit without atomics: sort, then reduce

With markers sorted by cell, a nearest-cell deposit is a segmented sum,
deterministic on both backends:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: deposit_nearest_cell
```

For linear (cloud-in-cell) weights, deposit each of the two (2D: four, 3D:
eight) neighbours with its weight, one `segment_sum` each.

The two other GPU strategies, as kernels:

* **Global atomics** (`cunumpy_atomic_add` from `<cunumpy/atomic.cuh>`): the
  simplest; slow when many threads hit the same cells, and the summation order
  varies between runs (results differ in the last bits).
* **Per-block shared memory**: each block deposits into a copy of the grid in
  shared memory, then adds it to the global grid once per cell. Fast for small
  grids. Check the size with `xp.cuda.max_shared_memory_per_block()` and pass
  `shared_mem=` at the launch; above 48 KiB the kernel is set up for the larger
  limit automatically.

```c
extern "C" __global__
void deposit(Array1D<double> x, Array1D<double> w, Array1D<double> rho,
             double lower, double dx, int nx) {
    extern __shared__ double block_rho[];
    for (int k = threadIdx.x; k < nx; k += blockDim.x) block_rho[k] = 0.0;
    __syncthreads();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < x.shape[0]) { /* cunumpy_atomic_add(&block_rho[cell], ...) */ }
    __syncthreads();
    for (int k = threadIdx.x; k < nx; k += blockDim.x)
        cunumpy_atomic_add(&rho(k), block_rho[k]);
}
```

`cunumpy.kernel_testing.emulate_cuda_kernel(..., shared_mem=8 * nx)` runs such a
kernel on the CPU, barriers included, so it can be checked against the host
version without a GPU.

## Move markers between MPI ranks

With a spatial decomposition, markers that leave a rank's subdomain move to the
rank that owns their new position: compute each marker's destination rank,
group the markers by destination, exchange the counts, then the markers:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: pack_for_ranks
```

```{literalinclude} ../examples/particle_recipes.py
:pyobject: exchange
```

`xp.mpi.mpi_buffer` hands device arrays to MPI directly when it is CUDA-aware and
stages them through host memory otherwise (see [MPI](mpi.md)). Only the counts
are host arrays. Remove the markers that left with the compaction above, and
append the received ones.

## Random numbers per particle

A time loop is reproducible only if every random number comes from a seed.
Counter-based random numbers make that independent of the order of the
markers, the launch shape and the number of ranks: particle `id` at step
`step` always draws the same numbers. On the host:

```{literalinclude} ../examples/particle_recipes.py
:pyobject: thermal_velocities
```

and in a kernel, with `<cunumpy/random.cuh>`, the same numbers (uniforms
bit for bit):

```c
#include <cunumpy/random.cuh>
double z0, z1;
cunumpy_normal2(seed, particle_id, step, &z0, &z1);
v[i] = v_th * z0;
```

Use a different counter for every random decision of a step (e.g.
`4 * step + 0` for injection, `4 * step + 1` for collisions) so that they are
independent. For draws that need not be per particle (e.g. a collision
operator's own sampling), `xp.rng.random_streams` gives one seeded generator per
rank.

## Write output without stalling the GPU

```python
staging = xp.memory.HostStaging(rho.shape, rho.dtype)  # once
pending = []
for step in range(n_steps):
    advance()
    if step % output_every == 0:
        pending.append((step, staging.copy(rho)))  # returns at once
    while pending and pending[0][1].ready():
        s, copy = pending.pop(0)
        h5file[f"rho/{s}"] = copy.result()
```

`copy()` snapshots the array on the device and copies the snapshot to pinned
host memory on its own stream, so the next steps overwrite `rho` while the copy
runs. See `HostStaging` in the [API reference](../api.md).

## Run a time step as a CUDA graph

A step of a small simulation is dozens of short kernels, and the launch
overhead (a few microseconds each) can dominate. CuPy can capture the launches
of a step on a stream once and replay them:

```python
stream = cp.cuda.Stream(non_blocking=True)
catalog.compile_all()  # compile before capturing: no compilation in a graph
with stream:
    stream.begin_capture()
    step_kernels()  # the kernel launches of one step
    graph = stream.end_capture()
for _ in range(n_steps):
    graph.launch(stream)
stream.synchronize()
```

A graph replays exactly the captured launches: the same arrays (by address),
the same scalar arguments and launch shapes. So allocate every array before
capturing, keep scalars that change per step in device arrays, and keep host
synchronization (`.get()`, `float(x)`, `alive.sum()` read on the host, MPI)
outside the captured part. Debug mode skips its synchronization while a stream
is capturing.

## Choose the block size from the compiled kernel

```python
raw = kernel.compile()  # the cupy.RawKernel
raw.num_regs, raw.max_threads_per_block, raw.shared_size_bytes
```

A kernel that uses many registers per thread cannot run 1024 threads per block;
`max_threads_per_block` is the limit for this kernel on this device. Start
with 128 or 256 threads per block, then time a few sizes on the target GPU
(`xp.profiling.timed_region`) for the kernels that dominate a step.
