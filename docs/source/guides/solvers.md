# Solvers and fluid updates

Particle pushes and deposits are only part of a plasma code. Field solves need
sparse matrices, iterative solvers and FFTs; fluid (MHD) updates are long
chains of pointwise operations; diagnostics reduce over all cells or
particles. This page covers the tools for that: `xp.scipy` (including splines),
`xp.optimize.newton`, `xp.kernels.fuse` and `xp.petsc.petsc_vec`, plus
in-kernel reductions with `cunumpy/reduce.cuh`.

## SciPy on both backends: `xp.scipy`

`xp.scipy` is SciPy on the NumPy backend and `cupyx.scipy` on the CuPy
backend, so solver code is written once:

```python
import numpy as np

import cunumpy as xp


def poisson_1d(rho, dx):
    """Solve -phi'' = rho with phi = 0 at both ends (second-order FD)."""
    n = rho.size
    main = xp.full(n, 2.0 / dx**2)
    off = xp.full(n - 1, -1.0 / dx**2)
    A = xp.scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    phi, info = xp.scipy.sparse.linalg.cg(A, rho)
    assert info == 0
    return phi


def periodic_poisson(rho, length):
    """Solve -phi'' = rho on a periodic grid with FFTs."""
    n = rho.size
    k = 2 * np.pi * xp.fft.rfftfreq(n, d=length / n)
    rho_k = xp.scipy.fft.rfft(rho)
    phi_k = xp.where(k == 0, 0.0, rho_k / xp.where(k == 0, 1.0, k**2))
    return xp.scipy.fft.irfft(phi_k, n)
```

* The subpackages forwarded are the ones `cupyx.scipy` has: `fft`, `linalg`,
  `ndimage`, `signal`, `sparse`, `sparse.linalg`, `sparse.csgraph`,
  `spatial`, `special`, `stats`, `interpolate`, `fftpack`.
* `cupyx.scipy` has only part of SciPy. A missing name raises
  `AttributeError` saying which backend lacks it; check with
  `xp.scipy.special.available("erfcx")` where a fallback is possible.
  Commonly missing: `special.jv` (only `j0`, `j1`, `yn`), `special.erfi`,
  `linalg.solve`/`inv` (use `xp.linalg`), `RectBivariateSpline`, and all of
  `integrate` and `optimize`.
* cunumpy fills in `xp.scipy.linalg.solve_circulant` on the CuPy backend: an
  FFT solve on the device, with SciPy's arguments (a circulant preconditioner
  per iteration stays on the GPU).
* Keyword arguments can differ between the two: SciPy's `cg` takes `rtol`
  (since SciPy 1.12), CuPy's takes `tol`. Pass tolerances only after checking
  both, or through a small wrapper.
* Assemble a sparse matrix once and move it to the device once:
  `A = xp.scipy.sparse.csr_matrix(host_matrix)` on the CuPy backend copies a
  SciPy matrix to the device. Do not rebuild matrices in the time loop.
* Special functions for distribution functions and cross sections
  (`erf`, `erfc`, Bessel functions `i0`, `i1`, `k0`, ...) are in
  `xp.scipy.special` on both backends.
* On the fake CuPy (tests without a GPU), `xp.scipy` runs SciPy with the CuPy
  rules: host arrays are rejected, results are fake device arrays, and each
  subpackage has only the names CuPy has.

## Splines and interpolation

`xp.scipy.interpolate` fits and evaluates splines on the active backend.
`cupyx.scipy.interpolate` has `UnivariateSpline` (and its interpolating and
least-squares variants), `BSpline`, `make_interp_spline`, `make_lsq_spline`,
`NdBSpline`, `CubicSpline`, `PchipInterpolator`, `RegularGridInterpolator`,
... but not the FITPACK 2D classes (`RectBivariateSpline`, ...) or
`splrep`/`splev`. An interpolating 2D spline on a grid is
`make_interp_spline` along each axis plus `NdBSpline`:

```python
interp = xp.scipy.interpolate
bx = interp.make_interp_spline(x, values, k=3)       # fit along x (axis 0)
by = interp.make_interp_spline(y, bx.c.T, k=3)       # then along y
spline = interp.NdBSpline((bx.t, by.t), by.c.T, (3, 3))
# RectBivariateSpline clamps points to the grid; NdBSpline extrapolates
points = xp.stack([xp.clip(R, x[0], x[-1]), xp.clip(Z, y[0], y[-1])], axis=-1)
dpsi_dR = spline(points, nu=(1, 0))
```

With `s=0` this is the spline `RectBivariateSpline(x, y, values, s=0)` fits
(same knots, values equal to round-off).

Fit once at setup and keep the spline object; evaluation then needs no
transfers.

## Many root-finding problems at once: `xp.optimize.newton`

SciPy's optimizers run on the host, and `cupyx.scipy` has no `optimize`.
Sequential scalar problems (one `fsolve`, a `quad` integral, an ODE) belong on
the host at setup (see `xp.setup_on_host`). Many independent scalar equations
(a flux surface on every ray, an inverse mapping at every marker) are one
array problem: `xp.optimize.newton` runs Newton's method (with `fprime`) or
the secant method on all of them at once, with the steps of
`scipy.optimize.newton` for an array `x0`:

```python
# radius where the flux reaches each level along each ray
def residual(r):
    return psi(R0 + r * cos_theta, Z0 + r * sin_theta) - levels

r = xp.optimize.newton(residual, xp.full(levels.shape, 0.3))
```

Each iteration synchronizes once to test convergence.
`full_output=True` returns `root`, `converged` and `zero_der` arrays.

## Fused elementwise updates: `xp.kernels.fuse`

On the GPU, `(gamma - 1) * (E - 0.5 * rho * u**2)` runs as five kernels, each
reading and writing a full temporary array. `xp.kernels.fuse` turns the whole
function into one kernel with `cupy.fuse` when it is called with CuPy arrays,
and calls it unchanged with NumPy arrays:

```python
@xp.kernels.fuse
def pressure(rho, mom, energy, gamma):
    u = mom / rho
    return (gamma - 1.0) * (energy - 0.5 * rho * u * u)


@xp.kernels.fuse
def maxwellian(v, n, u, v_th):
    return n / (xp.sqrt(2.0 * np.pi) * v_th) * xp.exp(-0.5 * ((v - u) / v_th) ** 2)


p = pressure(rho, mom, energy, 5.0 / 3.0)
```

Memory-bound chains like these typically get several times faster. The
function must be elementwise: arithmetic, comparisons, ufuncs (`xp.exp`,
`xp.sqrt`, ...), `xp.where`; no `if` on array values, no indexing. Test the
CuPy path (`cunumpy.kernel_testing.BACKENDS`): `cupy.fuse` reports a function it
cannot trace at the first call with CuPy arrays.

## PETSc without copies: `xp.petsc.petsc_vec`

When the field solve goes through PETSc, `xp.petsc.petsc_vec(array)` wraps an
array as a PETSc vector that uses the array's memory, a CUDA vector for a
CuPy array. The deposit writes into `rho`, PETSc reads it and writes `phi`,
the gather reads `phi`, and no data leaves the GPU:

```python
from petsc4py import PETSc

rho = xp.zeros(n_local)
phi = xp.zeros(n_local)
rho_vec, phi_vec = xp.petsc.petsc_vec(rho), xp.petsc.petsc_vec(phi)

A = assemble_laplacian()  # a PETSc Mat
A.setType("aijcusparse")  # keep the matrix on the GPU too
ksp = PETSc.KSP().create()
ksp.setOperators(A)

for step in range(n_steps):
    deposit(rho)
    xp.synchronize()  # CuPy finished writing rho
    ksp.solve(rho_vec, phi_vec)
    xp.synchronize()  # PETSc finished writing phi
    gather(phi)
```

This needs a petsc4py built with CUDA (or HIP) support; otherwise
`petsc_vec` raises for CuPy arrays rather than let PETSc work on a host copy.
The array must be C-contiguous and of `PETSc.ScalarType`, since anything else
would need a copy. With MPI, each process passes its local part and the
vector's communicator.

## Reductions inside kernels: `cunumpy/reduce.cuh`

Outside kernels, `xp.sum` and friends are the right tool. Inside a
hand-written kernel, for a diagnostic computed in the same pass as the push,
or to combine values in a block before one atomic write, include the header:

```c
#include <cunumpy/reduce.cuh>

extern "C" __global__ void push_and_energy(double* x, double* v, const double* E,
                                           long long n, double qm_dt, double dt,
                                           double half_m, double* energy) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    double e = 0.0;
    if (i < n) {
        v[i] += qm_dt * E[i];
        x[i] += dt * v[i];
        e = half_m * v[i] * v[i];
    }
    cunumpy_block_sum_to(energy, e);  // every thread calls it: no early return
}
```

Every thread must reach the reduction; block functions also support partial
warps. Warp functions accept explicit masks for subsets of lanes. See the API
reference for the warp and block functions and prefix scans.
