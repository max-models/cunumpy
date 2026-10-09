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
when one of its pytest objects is used, so `device_function_kernel` and the
emulation work without pytest. It is not called `cunumpy.testing`, because a
submodule of that name would replace NumPy's `xp.testing` once imported.

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
* **Which arguments are compared**: `outputs=(2,)` selects them by index and
  `outputs=("markers",)` by parameter name (the names of the host function).
  `"markers.positions"` compares only that field of a struct or argument
  object, leaving out fields the two kernels fill differently (scratch buffers,
  for instance); otherwise the host kernel's declared `outputs` are used, and if there are
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

Arrays are passed as NumPy arrays (any strides) and the kernel writes into
them; scalars are checked like in a launch. Each kernel is compiled once into a
shared library and cached (in the process and in `~/.cache/cunumpy/emulation`,
see `emulation_cache_dir()`), so further launches with other values and sizes
cost no compilation. It catches wrong indices, clamping, periodic wrapping
and weights, i.e. most porting bugs of gather, scatter and push kernels. Block
shared memory and `__syncthreads` are emulated (pass `shared_mem=` for
`extern __shared__` arrays), so per-block deposits and shared-memory reductions
are covered too. It does not emulate concurrency between barriers or warp
intrinsics; kernels using the latter are refused with `NotImplementedError`, so
those still need a GPU run. Inline PTX is not emulated either: `asm(...)` and
`asm volatile(...)` compile but trap when reached, so a PTX branch a test never
takes needs no extra options, and reaching it raises `RuntimeError`. The compiler may fuse multiply-adds as NVRTC does,
so compare with a tolerance of a few ulp, or pass `options=("-ffp-contract=off",)`
for NumPy's rounding.

How the emulation works, for when a result surprises you:

* The source is compiled as C++17 with `CXX` (else `c++`, see
  `emulation_compiler()`), with `threadIdx`, `blockIdx`, `blockDim`, `gridDim`,
  the atomics (`atomicAdd`, `atomicMin`, ..., as plain operations), `__ldg`,
  `rsqrt` and `__trap` replaced. The shipped headers, the kernel's include
  directories and its `-D` options apply. `CUNUMPY_EMULATION_CACHE` sets the
  disk cache (`0`: none).
* The library runs in the Python process, on the arrays themselves. An array
  that is not writeable, or whose layout a parameter cannot take, is passed as
  a copy and written back, so aliased arguments see each other's writes.
* `__shared__` variables exist once per block (blocks run one after another),
  `extern __shared__` arrays point into a buffer of `shared_mem` bytes, and in a
  kernel that calls `__syncthreads` the threads of a block run as coroutines
  (POSIX `ucontext`, each with its own stack), so every thread reaches a barrier
  before any continues past it.
* Not emulated: concurrency between barriers (races and atomic ordering never
  show), warp intrinsics (`__syncwarp`, shuffles, votes: `NotImplementedError`,
  also from an included header), complex scalars and inline PTX.
* A kernel that does not compile, or traps (an out-of-bounds index with
  `-DCUNUMPY_BOUNDS_CHECK`, `__trap()`, inline `asm`), raises `RuntimeError`.
  Other crashes, such as a segmentation fault from an unchecked out-of-bounds
  index, end the process: run such tests through
  `run_in_fake_cupy_subprocess()`, which prints the traceback with
  `faulthandler`.

### Struct parameters and `emulated_launches()`

A kernel with struct parameters takes, for each struct, a dictionary of field
names to values, a `CudaStructValue`, or any object with an attribute per field
(a host argument class, a `CudaStructArguments`); the arrays in it are updated in
place:

```python
emulate_cuda_kernel(
    push,
    {"markers": markers, "alive": alive, "n": 4},  # the struct Particles
    0.5,
    total,
    n_threads=4,
)
```

To test code that *launches* kernels (a `Kernel` on the CuPy backend, a
propagator), wrap it in `emulated_launches()`. With the fake CuPy (below) every
`CudaKernel` launch in the block then runs through the emulation on the host
buffers of the fake arrays, so the device arrays hold the results:

```python
from cunumpy.kernel_testing import emulated_launches, host_buffer

with emulated_launches():
    propagator(dt)  # CuPy backend = the fake CuPy
np.testing.assert_allclose(host_buffer(markers), expected)
```

`host_buffer(array)` is the NumPy array behind a fake CuPy array (not a copy).
Launches are serial, so use small problems.

No CUDA is compiled inside the block either: `kernel.compile()`,
`recompile()` and the `compile_all()` methods build the emulation library
instead (and return None), so a program that compiles its kernels before the
time loop runs unchanged and still reports compile errors there.
`fake_cupy_session()` does all of it for a whole program, activating the CuPy
backend too:

```python
from cunumpy.kernel_testing import fake_cupy_session, requires_device_backend


@requires_device_backend  # a GPU, or the fake CuPy (device_backend_available())
def test_simulation_on_the_cupy_backend():
    with fake_cupy_session():  # only with the fake CuPy
        sim.run()
```

The fake CuPy must be installed before cunumpy is used, so a test process
that runs on NumPy cannot switch to it. `run_in_fake_cupy_subprocess(code)`
runs the code in a serial child process on the fake CuPy (outside the MPI
job, only on rank 0 under MPI) and fails the test with the signal or exit code
and the end of the child's output if it fails.

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

The prototype above generates this kernel, named `<function>_kernel` unless
`name=` is given:

```c
extern "C" __global__ void find_span_kernel(
    const double* t, const int* p, const double* eta, int* out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = find_span(t, p[i], eta[i]);
}
```

In the generated kernel, pointer parameters are passed unchanged to every
thread (shared data), scalar parameters become per-thread arrays, the return
value of thread `i` goes to `out[i]` (a `void` function has no `out`), and `n`
is the number of elements. A parameter of the function named `out` or `n`
raises `ValueError`; rename the generated ones with `out_param=` and
`n_threads_param=`. A
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

Syncs (the host waiting for the device) are recorded as well, in
`counter.syncs`, and are accepted unless you ask for none:
`assert_no_transfers(syncs=True)`. They include `xp.synchronize()` and the waits
of the MPI helpers; on the fake CuPy also `float(a)`, `int(a)`, `bool(a)`,
`a.item()` and `a.tolist()`, which stall the real CuPy too but cannot be
observed there from Python.

## Test generated headers

When struct headers are generated with `write_cuda_header()` and committed,
add a test that regenerates them and compares (see [Kernel arguments and
structs](arguments.md), "Write the struct to a header"). A forgotten regeneration
then fails CI instead of producing a kernel that reads fields at wrong
offsets.

## CI setup

* With `CUNUMPY_REQUIRE_CUDA=1` the GPU markers fail instead of skipping:
  `requires_cupy` (as an error when the test is set up), the `cupy` run of the
  `backend` fixture (which also activates CuPy strictly, never falling back to
  NumPy) and `assert_kernels_agree`. Set it on the GPU CI job, so a broken CuPy
  or driver cannot pass as a set of skipped tests.

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
