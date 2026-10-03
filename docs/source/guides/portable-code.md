# Writing backend-agnostic code

The goal is one code base that runs unchanged on NumPy and CuPy. Most array
code already does; this page collects the rules that keep it that way.

## Always go through the `xp` namespace

```python
import cunumpy as xp

grid = xp.linspace(0.0, 1.0, 1025)
field = xp.exp(-((grid - 0.5) ** 2) / 0.01)
energy = xp.sum(field**2) * (grid[1] - grid[0])
```

Two habits break portability:

* **Importing raw NumPy for array work.** `np.zeros(n)` always creates a host
  array, even when the rest of the program runs on the GPU. Use `numpy`
  directly only for things that must stay on the host: dtypes such as
  `np.float64`, file I/O, and data you will explicitly convert.
* **Binding names from `cunumpy`.** `xp.zeros` is looked up each time it is
  accessed, so it follows the active backend. `from cunumpy import zeros`
  captures the function of the backend that was active at import time and
  keeps calling it after `set_backend()`. Always write `xp.zeros(...)`. The
  same applies to `from cunumpy import numpy_backend`; read `xp.numpy_backend`
  where you need it.

## Follow the input, not the global setting

A function that *creates* arrays uses the active backend. A function that
*receives* arrays should compute on the library of its inputs, which may differ
from the active backend (a caller may have kept a NumPy reference while CuPy is
active). `get_array_module()` returns the matching compatibility module:

```python
def gradient(f, dx):
    array_xp = xp.get_array_module(f)
    out = array_xp.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    out[0] = (f[1] - f[0]) / dx
    out[-1] = (f[-1] - f[-2]) / dx
    return out
```

When a function combines several inputs, check that they agree. A mixed
NumPy/CuPy operation otherwise fails somewhere deep inside the library, or,
worse, silently converts:

```python
def axpy(a, x, y):
    xp.assert_same_backend(x, y)  # TypeError naming both backends
    return a * x + y
```

At the boundary of a library (the public constructor of a solver, for example),
normalize incoming arrays to the active backend once with `to_cunumpy()` and
keep them there:

```python
class Solver:
    def __init__(self, matrix, rhs):
        self.matrix = xp.to_cunumpy(matrix)  # copied once if needed
        self.rhs = xp.to_cunumpy(rhs)
```

## Be explicit about dtypes

NumPy and CuPy agree on dtype rules for array operations, but Python scalars and
lists are inferred, and code that mixes `float32` arrays with `float64` arrays
promotes silently. Pass a dtype when it matters, especially for arrays that will
be handed to CUDA kernels, which check dtypes strictly:

```python
x = xp.zeros(n, dtype=xp.float64)
idx = xp.arange(n, dtype=xp.int32)
w = xp.asarray([0.25, 0.75], dtype=xp.default_float_dtype())
```

`np.float64` and `xp.float64` are the same dtype object and both are accepted by
CuPy, so using NumPy dtypes for declarations is fine.

## Random numbers

`xp.rng.get_rng(seed)` returns a `Generator` of the active backend, so random data is
generated where it is used:

```python
rng = xp.rng.get_rng(seed=42)
velocities = rng.normal(0.0, 1.0, size=(n_particles, 3))
```

NumPy and CuPy generators do **not** produce the same sequence from the same
seed. When CPU and GPU runs must start from identical data (regression tests,
comparing results across machines), generate on the host and convert:

```python
import numpy as np

host = np.random.default_rng(42).normal(size=(n_particles, 3))
velocities = xp.to_cunumpy(host)
```

## Watch for hidden synchronization

On the GPU, operations are queued and run asynchronously. Anything that needs a
value on the host waits for the GPU and copies data back:

* `float(x)`, `int(x)`, `bool(x)`, `x.item()` on a device scalar;
* `if xp.any(mask): ...`, `while residual > tol: ...` (a device comparison in a
  Python condition);
* `print(array)`, `array.tolist()`, f-string formatting of a device value;
* iterating over a device array in a Python `for` loop.

None of these are errors, and they are free on NumPy. In a hot loop they stall
the GPU. Check convergence every few iterations instead of every iteration, and
keep reductions on the device until a value is really needed. These copies
bypass CuNumpy and are not seen by `count_transfers()`; use a profiler (see
[Timing and profiling](profiling.md)).

## Know where the libraries differ

CuNumpy does not wrap operations; it forwards them. Coverage is therefore that
of the installed CuPy version. The common differences:

* Some NumPy functions do not exist in CuPy, or exist with fewer options (for
  example parts of `numpy.polynomial`, object and string dtypes, structured
  arrays). Check the CuPy documentation for functions outside the core.
* Reductions such as `xp.sum(a)` return a NumPy scalar on NumPy but a 0-d
  device array on CuPy. Keep it as an array, or call `float()` and accept the
  synchronization.
* SciPy functions accept only NumPy arrays; CuPy has its own `cupyx.scipy`.
  Use `xp.scipy`, which is the one for the active backend (see
  [Solvers and fluid updates](solvers.md)); for functions `cupyx.scipy` lacks,
  convert with `to_numpy()` at that boundary.
* Plotting, HDF5 and most I/O libraries need host arrays: `to_numpy()` first.

When a function genuinely needs a different implementation per backend, branch
on the input's backend in one place rather than throughout the code:

```python
def solve_banded(ab, b):
    if xp.is_gpu(b):
        import cupyx.scipy.linalg as la
    else:
        import scipy.linalg as la
    return la.solve_banded((1, 1), ab, b)
```

## Test on both backends

`cunumpy.kernel_testing.BACKENDS` parametrizes a test over NumPy and CuPy, skipping the
CuPy case where there is no GPU, so the same test file runs on a laptop and in
GPU CI:

```python
import pytest
from cunumpy.kernel_testing import BACKENDS

import cunumpy as xp


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_of_linear_function(backend):
    with xp.use_backend(backend):
        x = xp.linspace(0.0, 1.0, 11)
        g = gradient(3.0 * x, x[1] - x[0])
        assert xp.get_array_backend(g) == backend
        assert float(xp.max(xp.abs(g - 3.0))) < 1e-12
```

See [Testing kernels](../kernels/testing.md) for the other test helpers.
