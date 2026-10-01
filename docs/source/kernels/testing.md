# Testing kernels

A GPU port is only as trustworthy as its comparison with the CPU version.
`cunumpy.testing` provides pytest helpers for exactly that, designed so that the
same test suite runs on a laptop without a GPU (GPU cases are skipped) and on a
GPU runner (everything runs).

```python
from cunumpy.testing import (
    BACKENDS,
    assert_kernels_agree,
    backend,
    device_function_kernel,
    requires_cupy,
)
```

`cunumpy.testing` is not imported by `import cunumpy`, and it imports pytest only
when one of its pytest objects is used.

## Run a test on both backends

`BACKENDS` is `["numpy", pytest.param("cupy", marks=requires_cupy)]`:

```python
import pytest

import cunumpy as xp
from cunumpy.testing import BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_norm(backend):
    with xp.use_backend(backend):
        assert float(xp.linalg.norm(xp.ones(4))) == 2.0
```

The `backend` fixture does the same and also activates the backend for the
whole test. Import it into `conftest.py` to make it available everywhere:

```python
# conftest.py
from cunumpy.testing import backend  # noqa: F401
```

```python
def test_energy_is_conserved(backend):
    state = make_state()              # arrays land on the active backend
    e0 = energy(state)
    for _ in range(100):
        step(state, 1e-3)
    assert abs(float(energy(state)) - float(e0)) < 1e-10
```

`requires_cupy` is a plain skip marker for GPU-only tests:

```python
from cunumpy.testing import requires_cupy


@requires_cupy
def test_kernel_compiles():
    with xp.use_backend("cupy"):
        assert catalog["push"].compile()
```

## Compare host and CUDA kernels: `assert_kernels_agree`

```python
import numpy as np

import cunumpy as xp
from cunumpy.testing import assert_kernels_agree


def make_axpy_args(backend, seed):
    rng = np.random.default_rng(seed)
    x = xp.to_cunumpy(rng.random(1000))
    y = xp.to_cunumpy(rng.random(1000))
    return (2.0, x, y, 1000)


def test_axpy_parity():
    assert_kernels_agree(axpy, make_axpy_args, n_threads=1000)
```

For each backend, NumPy then CuPy, it activates the backend, builds the
arguments with `make_args(backend, seed)`, calls the kernel (`n_calls` times),
then copies the CUDA results to the host and compares them with
`numpy.testing.assert_allclose(rtol=1e-12, atol=0)`. A failure names the
argument that differs. Without a GPU the test is skipped.

Things to know:

* **Build random data on the host.** NumPy and CuPy generators produce
  different sequences from the same seed, so use `numpy.random.default_rng` and
  convert with `to_cunumpy()`, as above.
* **Which arguments are compared**: `outputs=(2,)` selects them by index;
  otherwise the host kernel's declared `outputs` are used, and if there are
  none, every array argument. Arrays held by argument objects (one level deep,
  e.g. a `CudaArguments` object or a list) are compared too.
* **Tolerances**: the default `rtol=1e-12` suits deterministic kernels. Kernels
  with atomics or a different summation order need looser tolerances, for
  example `rtol=1e-10, atol=1e-14`.
* **`n_calls=10`** runs the kernel repeatedly on the same arguments, which
  catches state that is not reset between calls.
* It returns the host results by argument name for additional assertions.

### One test for the whole catalog

```python
import pytest

from my_sim.kernels import catalog
from my_sim.kernels.test_args import MAKE_ARGS, N_THREADS


@pytest.mark.parametrize("name, kernel", catalog.parity_cases())
def test_parity(name, kernel):
    assert_kernels_agree(kernel, MAKE_ARGS[name], n_threads=N_THREADS[name])
```

`parity_cases()` yields the kernels that have a CUDA version, so every newly
ported kernel is tested as soon as its `.cu` file exists (it needs an entry in
`MAKE_ARGS`, which fails loudly with a `KeyError` if forgotten).

## Test `__device__` helpers: `device_function_kernel`

Helpers such as B-spline evaluation or coordinate maps are `__device__`
functions in headers. Testing them through a whole kernel is indirect.
`device_function_kernel(header_source, signature)` generates an elementwise
kernel that calls the function once per thread:

```python
from pathlib import Path

import numpy as np

import cunumpy as xp
from cunumpy.testing import device_function_kernel, requires_cupy

BSPLINES = Path("my_sim/kernels/common/bsplines.cuh").read_text()


@requires_cupy
def test_find_span_matches_host():
    find_span = device_function_kernel(
        BSPLINES, "int find_span(const double* t, int p, double eta)"
    )
    t = np.linspace(0.0, 1.0, 17)
    eta = np.random.default_rng(0).random(1000)
    expected = np.array([find_span_host(t, 3, e) for e in eta])

    with xp.use_backend("cupy"):
        out = xp.empty(eta.size, dtype=xp.int32)
        find_span(
            xp.to_cupy(t),
            xp.full(eta.size, 3, dtype=xp.int32),  # scalar parameter -> one per thread
            xp.to_cupy(eta),
            out,
            eta.size,
            n_threads=eta.size,
        )
    np.testing.assert_array_equal(xp.to_numpy(out), expected)
```

In the generated kernel, pointer parameters are passed unchanged to every
thread (shared data), scalar parameters become per-thread arrays, the return
value of thread `i` goes to `out[i]`, and `n` is the number of elements. Extra
keyword arguments (`include_dirs`, `includes`, `block_size`) go to the
`CudaKernel`.

## Test that a step stays on the device

```python
@requires_cupy
def test_time_step_has_no_transfers():
    with xp.use_backend("cupy"):
        state = make_state()
        step(state, 1e-3)              # warm-up: compilation, allocations
        with xp.assert_no_transfers():
            step(state, 1e-3)
```

This catches `to_numpy()` calls, `PyccelKernel` conversions and `Kernel`
fallbacks that crept into the step. It does not see copies made outside
CuNumpy (see [Data movement](../guides/data-movement.md)).

## Test generated headers

When struct headers are generated with `write_cuda_header()` and committed,
add a test that regenerates them and compares (see [Kernel arguments and
structs](arguments.md), "Write the struct to a header"). A forgotten regeneration
then fails CI instead of producing a kernel that reads fields at wrong
offsets.

## CI setup

* Run the suite on a normal CPU runner: everything on NumPy runs, GPU cases
  are reported as skipped.
* Run the same suite on a GPU runner, optionally with `CUNUMPY_CUDA_DEBUG=1` so
  kernels are bounds-checked and errors are attributed to the right launch.
* Every so often, run the GPU suite under `compute-sanitizer` (see [Debugging
  CUDA kernels](debugging.md)).
