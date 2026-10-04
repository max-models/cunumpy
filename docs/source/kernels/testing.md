# Testing kernels

A GPU port is only as trustworthy as its comparison with the CPU version.
`cunumpy.kernel_testing` provides pytest helpers for exactly that, designed so that the
same test suite runs on a laptop without a GPU (GPU cases are skipped) and on a
GPU runner (everything runs).

```python
from cunumpy.kernel_testing import (
    BACKENDS,
    assert_kernels_agree,
    backend,
    device_function_kernel,
    requires_cupy,
)
```

`cunumpy.kernel_testing` is not imported by `import cunumpy`, and it imports pytest only
when one of its pytest objects is used.

## Run a test on both backends

`BACKENDS` is `["numpy", pytest.param("cupy", marks=requires_cupy)]`:

```python
import pytest

import cunumpy as xp
from cunumpy.kernel_testing import BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_norm(backend):
    with xp.use_backend(backend):
        assert float(xp.linalg.norm(xp.ones(4))) == 2.0
```

The `backend` fixture does the same and also activates the backend for the
whole test. Import it into `conftest.py` to make it available everywhere:

```python
# conftest.py
from cunumpy.kernel_testing import backend  # noqa: F401
```

```python
def test_energy_is_conserved(backend):
    state = make_state()  # arrays land on the active backend
    e0 = energy(state)
    for _ in range(100):
        step(state, 1e-3)
    assert abs(float(energy(state)) - float(e0)) < 1e-10
```

`requires_cupy` is a plain skip marker for GPU-only tests:

```python
from cunumpy.kernel_testing import requires_cupy


@requires_cupy
def test_kernel_compiles():
    with xp.use_backend("cupy"):
        assert catalog["push"].compile()
```

## Compare host and CUDA kernels: `assert_kernels_agree`

```python
import numpy as np

import cunumpy as xp
from cunumpy.kernel_testing import assert_kernels_agree


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
  e.g. a `CudaArguments` object or a list) are compared too. A
  `CudaStructArguments` object is read through its struct fields, so its
  arrays get the names of the host argument object's attributes
  (`argument 0.markers`), also when the fields are properties.
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

### Test arguments next to the kernel

Instead of one `make_args` per kernel in the test file, each kernel folder can
hold `<name>_test_args.py`:

```python
# my_sim/kernels/push/push_test_args.py
import numpy as np

import cunumpy as xp

N_THREADS = 1000  # or GRID; also BLOCK, RTOL, ATOL, N_CALLS, OUTPUTS, SEED


def make_args(backend, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(1000))
    return (x, 2.0, x.size)
```

`KernelCatalog.from_package()` records these modules (imported only when a
test asks for them), and the parity test of the whole package becomes

```python
from cunumpy.kernel_testing import check_parity, parity_cases


@pytest.mark.parametrize("kernel", parity_cases(catalog))
def test_parity(kernel):
    check_parity(kernel)
```

A kernel with a CUDA version but no test-arguments module shows up as skipped,
with the name of the missing file in the reason, so the report lists what is
left to do. `N_THREADS` may be a function of the argument tuple
(`lambda args: args[0].shape[0]`), or be omitted when the CUDA kernel has
`n_threads_from`. Keyword arguments of `check_parity` override the module.

## Test CUDA kernels without a GPU: `emulate_cuda_kernel`

On a CPU-only CI runner the parity tests are skipped, so nothing checks the
CUDA kernels' index and weight arithmetic. `emulate_cuda_kernel` runs a kernel
on the CPU instead: the source is compiled as C++ with the CUDA built-ins
replaced, and the kernel is called once per thread, one thread after another.
Compare it with the host kernel:

```python
import numpy as np
import pytest

from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

from my_sim.kernels import catalog

pytestmark = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


def test_gather_cuda_arithmetic():
    rng = np.random.default_rng(0)
    positions, field = rng.random((500, 2)), rng.normal(size=(17, 9, 2))
    expected, result = np.zeros((500, 2)), np.zeros((500, 2))
    catalog["gather"].host_kernel(
        positions, field, expected, 0.0, 0.0, 0.06, 0.11, 17, 9
    )
    emulate_cuda_kernel(
        catalog["gather"].cuda_kernel,
        positions,
        field,
        result,
        0.0,
        0.0,
        0.06,
        0.11,
        17,
        9,
        n_threads=500,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
```

Arrays are passed as NumPy arrays (any strides) and written back; scalars are
checked like in a launch. It catches wrong indices, clamping, periodic wrapping
and weights, i.e. most porting bugs of gather, scatter and push kernels. Block
shared memory and `__syncthreads` are emulated (pass `shared_mem=` for
`extern __shared__` arrays), so per-block deposits and shared-memory reductions
are covered too. It does not emulate concurrency between barriers or warp
intrinsics; kernels using the latter are refused with `NotImplementedError`, so
those still need a GPU run. The compiler may fuse multiply-adds as NVRTC does,
so compare with a tolerance of a few ulp.

## Test `__device__` helpers: `device_function_kernel`

Helpers such as B-spline evaluation or coordinate maps are `__device__`
functions in headers. Testing them through a whole kernel is indirect.
`device_function_kernel(header_source, signature)` generates an elementwise
kernel that calls the function once per thread:

```python
from pathlib import Path

import numpy as np

import cunumpy as xp
from cunumpy.kernel_testing import device_function_kernel, requires_cupy

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
value of thread `i` goes to `out[i]`, and `n` is the number of elements. A
struct parameter, by value or by `const` reference (`const DomainArgs& d`),
is passed through unchanged as well; give the struct types in `structs=`
and pass a `CudaStructArguments` object or a packed value, so helpers that
take the argument structs of the kernels are tested the same way. Extra
keyword arguments (`include_dirs`, `includes`, `block_size`) go to the
`CudaKernel`.

## Run the CuPy code paths without a GPU: the fake CuPy

`emulate_cuda_kernel` covers the kernels. Everything around them (argument
objects built from device arrays, struct packing, `as_device_array`,
transfer counting, the backend branches of a simulation) runs only with CuPy
present. For CI machines without a GPU, cunumpy ships a strict stand-in:

```bash
CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy pytest tests/
```

Its arrays live in host memory but are not NumPy arrays: `numpy.asarray(a)`
raises (as it does for real CuPy arrays, so a compiled host kernel rejects
them), CuPy functions reject NumPy arrays and lists, mixing the two raises,
reductions return 0-d arrays, and arrays have `data.ptr`, `device` and
`__cuda_array_interface__`. Kernels cannot run: `RawKernel` raises
`NotImplementedError`, `requires_cupy` skips and `assert_kernels_agree`
skips while the fake is active (`cunumpy.kernel_testing.fake_cupy_active()`). It
can also be installed from code, before the first backend use:

```python
# conftest.py
from cunumpy.kernel_testing import install_fake_cupy

install_fake_cupy()
```

Most host/device bugs (a NumPy array reaching a device argument object, a
device array reaching SciPy or MPI, a missing `xp.asarray`) show up this way
long before the code reaches a GPU. The fake is never installed when the
real CuPy is importable.

## Test that a step stays on the device

```python
@requires_cupy
def test_time_step_has_no_transfers():
    with xp.use_backend("cupy"):
        state = make_state()
        step(state, 1e-3)  # warm-up: compilation, allocations
        with xp.profiling.assert_no_transfers():
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
  are reported as skipped, and `emulate_cuda_kernel` tests check the CUDA
  kernels' arithmetic (the runner needs a C++ compiler, which Linux images
  have).
* Run it a second time with `CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy`, so the
  CuPy code paths are exercised on the CPU runner too (kernel launches are
  skipped).
* Run the same suite on a GPU runner, optionally with `CUNUMPY_CUDA_DEBUG=1` so
  kernels are bounds-checked and errors are attributed to the right launch.
* Every so often, run the GPU suite under `compute-sanitizer` (see [Debugging
  CUDA kernels](debugging.md)).
