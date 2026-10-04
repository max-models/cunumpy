# Accumulation kernels

Particle-in-cell and finite-element codes often *scatter* contributions: every
particle adds its charge to the grid cells it overlaps. Two problems appear on
the GPU:

1. **Many threads write the same cell.** A plain `+=` loses updates when
   threads race. The writes must be atomic.
2. **The target buffer belongs to someone else.** It is often a NumPy array
   owned by a host library (a distributed stencil vector, a solver's
   right-hand side) that keeps using it on the host, for example for MPI
   reductions. It cannot simply be replaced by a CuPy array.

CuNumpy addresses the first with the header `cunumpy/atomic.cuh` and the second
with `DeviceMirror`.

## Atomic adds: `cunumpy/atomic.cuh`

```c
#include <cunumpy/atomic.cuh>

double cunumpy_atomic_add(double* p, double v);   // *p += v atomically, returns old value
float  cunumpy_atomic_add(float* p, float v);

// indexed variants for C-contiguous arrays of shape (n0, n1) and (n0, n1, n2)
double cunumpy_atomic_add_2d(double* data, long long n1, long long i, long long j, double v);
double cunumpy_atomic_add_3d(double* data, long long n1, long long n2,
                             long long i, long long j, long long k, double v);
```

The header is found by every `CudaKernel` automatically. On compute capability
6.0 and newer, `double` atomics are a hardware instruction; older devices use a
compare-and-swap loop.

## `DeviceMirror`

```python
mirror = xp.memory.DeviceMirror(host_array)
```

| Member | CuPy backend | NumPy backend |
| --- | --- | --- |
| `mirror.device` | CuPy array, allocated on first access as a copy of the host | the host array itself |
| `mirror.to_device()` | copies host into device array | no-op |
| `mirror.to_host()` | copies device into host array **in place** | no-op |
| `mirror.zero()` | zeroes the device array | zeroes the host array |
| `mirror.rebind(new_host)` | follows a reallocation by the owner | same |

`to_host()` writes into the existing host array, so the owning library keeps
its reference and sees the new values. Every method returns the mirror, so
calls chain. If the owner reallocated the host array with a new shape or dtype
and `rebind()` was not called, the mirror raises `ValueError` instead of copying
into the wrong buffer.

## A complete charge deposition

```python
import numpy as np

import cunumpy as xp

DEPOSIT = r"""
#include <cunumpy/atomic.cuh>
#include <cunumpy/index.cuh>

extern "C" __global__
void deposit(const double* x, const double* w, double* rho,
             long long n_particles, long long n_cells, double dx) {
    CUNUMPY_THREAD_1D(ip, n_particles);
    long long cell = (long long)(x[ip] / dx);
    if (cell >= 0 && cell < n_cells) cunumpy_atomic_add(&rho[cell], w[ip] / dx);
}
"""


def deposit_host(x, w, rho, n_particles, n_cells, dx):
    cells = (x[:n_particles] / dx).astype(np.int64)
    inside = (cells >= 0) & (cells < n_cells)
    np.add.at(rho, cells[inside], w[:n_particles][inside] / dx)


deposit = xp.kernels.Kernel(deposit_host, xp.cuda.CudaKernel(DEPOSIT, "deposit"))

# owned by a host library, e.g. a distributed vector
rho_host = np.zeros(64)
rho = xp.memory.DeviceMirror(rho_host)

rng = np.random.default_rng(0)
x = xp.to_cunumpy(rng.uniform(0.0, 1.0, 10_000))
w = xp.to_cunumpy(np.full(10_000, 1e-4))

rho.zero()
deposit(x, w, rho.device, x.size, rho_host.size, 1.0 / 64, n_threads=x.size)
rho.to_host()  # rho_host now holds the charge density, on both backends

# the host library continues with rho_host, e.g. an MPI Allreduce
```

The same lines run on the NumPy backend, where `rho.device is rho_host`, the
host kernel writes into it directly, and `to_host()` does nothing.

## Sort-then-reduce: `segment_sum`

The alternative to atomics: bin the particles (sort or compute a cell key per
particle), compute each particle's contribution into an array, and sum per
cell. The last step is `xp.algorithms.segment_sum(values, keys, n_segments)`, on either
backend:

```python
cell = ix + nx * (iy + ny * iz)  # (n_particles,), -1 for outside
weights = compute_weights(markers)  # (n_particles, 8), one per corner
rho_cells = xp.algorithms.segment_sum(weights, cell, nx * ny * nz)  # (n_cells, 8)
```

Negative keys drop the value; a 2D `values` is summed column by column. Measure
both strategies on a real case before choosing: atomics are simpler and often
fast enough, sort-then-reduce is deterministic (the summation order does not
depend on thread scheduling).

## Guidelines

* Create the mirror once and keep it with the owner of the buffer. Allocation
  happens on the first `device` access.
* Make exactly one `to_host()` per accumulation, after all kernels have written.
  The mirror's copies are explicit method calls, so they are easy to find in the
  code; note that only the first allocation of `device` (through `to_cupy()`) is
  recorded by `count_transfers()`, while `to_host()` and `to_device()` are not.
* `zero()` before accumulating unless the kernel should add to the previous
  contents; then call `to_device()` first if the host side changed.
* Atomics on one hot cell serialize. If most particles hit few cells, consider
  sorting particles by cell or accumulating per block in shared memory first.
  For a single value (a total charge, an energy), `cunumpy_block_sum_to` from
  `cunumpy/reduce.cuh` makes one atomic add per block instead of one per thread.
* Floating-point atomics make the summation order non-deterministic. Results
  differ between runs in the last bits; compare with a tolerance in tests.
