# Porting a particle-in-cell code

This example follows a small 1D electrostatic particle-in-cell (PIC) code from
a CPU-only program to a GPU port, kernel by kernel, the way a larger code base
would be ported. Every step leaves a working program.

A PIC time step has four phases:

1. **deposit**: each particle adds its charge to the grid cell it is in
   (scatter, needs atomics on the GPU);
2. **field solve**: compute the electric field from the charge density (here
   with an FFT, array-level code);
3. **accelerate**: each particle reads the field in its cell and updates its
   velocity (gather);
4. **push**: each particle moves.

Phases 1, 3 and 4 are loops over particles: kernels. Phase 2 is array code
that runs on either backend as it is.

## The layout

```text
pic/
├── __init__.py
├── simulation.py
└── kernels/
    ├── __init__.py                  # the KernelCatalog
    ├── deposit/deposit_kernels.py
    ├── accelerate/accelerate_kernels.py
    └── push/push_kernels.py
tests/
└── test_kernels.py
```

Each kernel has its own folder, so its CUDA version can be added next to it
later (`push/push_cuda.cu`). The `kernels` folder and every kernel folder are
packages (an `__init__.py` in each, possibly empty).

## Step 1: host kernels

The host kernels are written in the Pyccel style: annotated loops that Pyccel
can compile to fast native code. They also run unchanged as plain Python,
which is what this example does.

```python
# pic/kernels/push/push_kernels.py
from math import fmod


def push(x: "float[:]", v: "float[:]", n: int, dt: float, length: float):
    for i in range(n):
        x[i] = fmod(x[i] + dt * v[i], length)
        if x[i] < 0.0:
            x[i] += length
```

```python
# pic/kernels/deposit/deposit_kernels.py
def deposit(
    x: "float[:]", rho: "float[:]", n: int, dx: float, n_cells: int, weight: float
):
    for i in range(n):
        cell = min(int(x[i] / dx), n_cells - 1)
        rho[cell] += weight / dx
```

```python
# pic/kernels/accelerate/accelerate_kernels.py
def accelerate(
    x: "float[:]",
    v: "float[:]",
    e: "float[:]",
    n: int,
    dx: float,
    n_cells: int,
    qm_dt: float,
):
    for i in range(n):
        cell = min(int(x[i] / dx), n_cells - 1)
        v[i] += qm_dt * e[cell]
```

The catalog collects them. `outputs` tells the host wrapper which argument each
kernel writes, so the GPU fallback copies back only that one:

```python
# pic/kernels/__init__.py
import cunumpy as xp

OUTPUTS = {"push": (0,), "deposit": (1,), "accelerate": (1,)}

catalog = xp.kernels.KernelCatalog.from_package(
    __name__,
    missing_cuda="fallback",
    host_options=lambda name: {"outputs": OUTPUTS[name]},
)
```

The simulation calls the kernels through the catalog and does the field solve
with array operations:

```python
# pic/simulation.py
import numpy as np

import cunumpy as xp

from .kernels import catalog


class Simulation:
    def __init__(self, n_particles=100_000, n_cells=64, length=2 * np.pi, seed=0):
        rng = np.random.default_rng(seed)  # host data: identical on both backends
        x = rng.uniform(0.0, length, n_particles)
        v = rng.normal(0.0, 1.0, n_particles) + 0.1 * np.sin(x)

        self.n, self.n_cells, self.length = n_particles, n_cells, length
        self.dx = length / n_cells
        self.weight = length / n_particles  # mean density 1
        self.x = xp.to_cunumpy(x)
        self.v = xp.to_cunumpy(v)
        self.rho = xp.zeros(n_cells)
        self.e = xp.zeros(n_cells)
        self.k = xp.to_cunumpy(2 * np.pi * np.fft.rfftfreq(n_cells, d=self.dx))

    def solve_field(self):
        rho_k = xp.fft.rfft(self.rho - xp.mean(self.rho))
        e_k = xp.zeros_like(rho_k)
        e_k[1:] = -1j * rho_k[1:] / self.k[1:]  # E = -d(phi)/dx, phi'' = -rho
        self.e[:] = xp.fft.irfft(e_k, n=self.n_cells)

    def step(self, dt):
        self.rho[:] = 0.0
        catalog["deposit"](
            self.x,
            self.rho,
            self.n,
            self.dx,
            self.n_cells,
            self.weight,
            n_threads=self.n,
        )
        self.solve_field()
        catalog["accelerate"](
            self.x, self.v, self.e, self.n, self.dx, self.n_cells, -dt, n_threads=self.n
        )
        catalog["push"](self.x, self.v, self.n, dt, self.length, n_threads=self.n)

    def field_energy(self):
        return 0.5 * float(xp.sum(self.e**2)) * self.dx
```

This runs on the CPU, and, thanks to `missing_cuda="fallback"`, already on the
GPU backend too:

```python
import cunumpy as xp
from pic.kernels import catalog
from pic.simulation import Simulation

xp.set_backend("cupy")
print(catalog.summary())  # CUDA kernels: 0 of 3 (missing: accelerate, deposit, push)

sim = Simulation()
with xp.profiling.count_transfers() as counter:
    sim.step(0.1)
print(counter.report())
```

```text
6 transfer(s) through cunumpy (0 to_host, 0 to_device, 3 kernel_conversion, 3 fallback)
  ...
```

Every kernel call copies its arrays to the host and back. The results are
correct, which is the baseline for the port, but slow.

## Step 2: port the first kernel

Add `push/push_cuda.cu` next to the host kernel. It mirrors the host kernel's
argument list one to one:

```c
// pic/kernels/push/push_cuda.cu
#include <cunumpy/index.cuh>

extern "C" __global__
void push(double* x, const double* v, long long n, double dt, double length) {
    CUNUMPY_THREAD_1D(i, n);
    double xi = fmod(x[i] + dt * v[i], length);
    x[i] = xi < 0.0 ? xi + length : xi;
}
```

Nothing else changes. The catalog finds the file on the next start, and
`catalog["push"]` launches the CUDA kernel on the GPU backend:

```text
CUDA kernels: 1 of 3 (missing: accelerate, deposit)
```

The Python `int` and `float` arguments are cast to `long long` and `double` as
declared; passing, say, a NumPy array for `x` would raise a `TypeError` instead
of being copied.

## Step 3: test the port

Before porting more, make sure the CUDA kernel computes what the host kernel
computes. One parametrised test covers every ported kernel:

```python
# tests/test_kernels.py
import numpy as np
import pytest

import cunumpy as xp
from cunumpy.kernel_testing import assert_kernels_agree
from pic.kernels import catalog

N, N_CELLS, LENGTH = 10_000, 64, 2 * np.pi
DX = LENGTH / N_CELLS


def particles(seed):
    rng = np.random.default_rng(seed)
    x = xp.to_cunumpy(rng.uniform(0.0, LENGTH, N))
    v = xp.to_cunumpy(rng.normal(0.0, 1.0, N))
    return x, v


def push_args(backend, seed):
    x, v = particles(seed)
    return (x, v, N, 0.1, LENGTH)


def deposit_args(backend, seed):
    x, _ = particles(seed)
    return (x, xp.zeros(N_CELLS), N, DX, N_CELLS, LENGTH / N)


def accelerate_args(backend, seed):
    x, v = particles(seed)
    e = xp.to_cunumpy(np.sin(np.linspace(0.0, LENGTH, N_CELLS)))
    return (x, v, e, N, DX, N_CELLS, -0.1)


MAKE_ARGS = {"push": push_args, "deposit": deposit_args, "accelerate": accelerate_args}


@pytest.mark.parametrize("name, kernel", catalog.parity_cases())
def test_parity(name, kernel):
    # deposit sums with atomics in arbitrary order: allow round-off
    assert_kernels_agree(kernel, MAKE_ARGS[name], n_threads=N, rtol=1e-10)
```

Without a GPU, the test is skipped; on a GPU runner it compares every ported
kernel, and each newly added `.cu` file is tested automatically.

## Step 4: port the remaining kernels

The deposit writes many particles into few cells, so it needs atomic adds:

```c
// pic/kernels/deposit/deposit_cuda.cu
#include <cunumpy/atomic.cuh>
#include <cunumpy/index.cuh>

extern "C" __global__
void deposit(const double* x, double* rho, long long n, double dx,
             long long n_cells, double weight) {
    CUNUMPY_THREAD_1D(i, n);
    long long cell = min((long long)(x[i] / dx), n_cells - 1);
    cunumpy_atomic_add(&rho[cell], weight / dx);
}
```

```c
// pic/kernels/accelerate/accelerate_cuda.cu
#include <cunumpy/index.cuh>

extern "C" __global__
void accelerate(const double* x, double* v, const double* e, long long n,
                double dx, long long n_cells, double qm_dt) {
    CUNUMPY_THREAD_1D(i, n);
    long long cell = min((long long)(x[i] / dx), n_cells - 1);
    v[i] += qm_dt * e[cell];
}
```

Now the parity test covers all three, and the step should not transfer
anything. Turn that into a test:

```python
from cunumpy.kernel_testing import requires_cupy
from pic.simulation import Simulation


@requires_cupy
def test_step_stays_on_device():
    with xp.use_backend("cupy"):
        sim = Simulation(n_particles=N)
        sim.step(0.1)  # warm-up: compiles the kernels
        with xp.profiling.assert_no_transfers():
            sim.step(0.1)
```

## Step 5: lock it in and measure

With every kernel ported, switch the catalog to `missing_cuda="raise"`, so a
kernel added later without a CUDA version fails immediately on the GPU instead
of silently copying. Compile everything at start-up and time the loop:

```python
import cunumpy as xp
from pic.kernels import catalog
from pic.simulation import Simulation

xp.set_backend("cupy")
if xp.cupy_backend:
    catalog.compile_all(jobs=4)

sim = Simulation(n_particles=1_000_000)
with xp.profiling.timed_region("100 steps") as timing:
    for _ in range(100):
        with xp.profiling.nvtx_range("step"):
            sim.step(0.05)
print(
    f"{timing.elapsed / 100 * 1e3:.2f} ms/step, field energy {sim.field_energy():.4e}"
)
```

The same script, without `set_backend("cupy")`, runs the Pyccel-compiled (or,
as here, pure Python) host kernels on the CPU.

## Where to go from here

* The kernels take five to seven loose arguments. In a real code, group the
  particle data with [`KernelArguments` or a `CudaStruct`](../kernels/arguments.md).
* If the charge density belongs to a host library (a distributed vector
  exchanged over MPI), deposit into a [`DeviceMirror`](../kernels/accumulation.md).
* For several GPUs, split the particles over MPI ranks, bind one GPU per rank
  and reduce `rho` across ranks ([Multi-GPU programs with
  MPI](../guides/mpi.md)).
