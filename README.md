# CuNumpy

CuNumpy lets a Python program use a NumPy-like API while choosing NumPy arrays
on the CPU or CuPy arrays on an NVIDIA GPU. In the simplest case, replace
`import numpy as np` with `import cunumpy as xp`; the array operations you
already know then run on the selected backend.

```python
import cunumpy as xp

values = xp.arange(5, dtype=xp.float64)
print(values * 2)
print(xp.get_backend())  # 'numpy' by default
```

CuNumpy selects an array library for newly requested operations. It does not
move existing arrays just because the selected backend changes. This guide
covers backend selection, array movement, mixed CPU/GPU workflows, and the
helper APIs CuNumpy provides around NumPy and CuPy.

The top level of `cunumpy` is the NumPy (or CuPy) namespace plus backend
selection and array conversion. The helpers are in submodules, so that they
never hide a NumPy name:

| Submodule | Contents |
|---|---|
| `xp.cuda` | CUDA only: `CudaKernel`, `CudaStruct`, CUDA headers, devices, streams |
| `xp.kernels` | `Kernel`, `KernelCatalog`, `PyccelKernel`, host implementations, `fuse` |
| `xp.rng` | `random_streams`, `get_rng`, `philox_*` |
| `xp.algorithms` | `morton_*`, `sort_by_key`, `cell_offsets`, `segment_boundaries`, `segment_sum`, `SegmentPlan` |
| `xp.mpi` | `mpi_buffer`, reusable `MPIStaging`, CUDA-aware MPI |
| `xp.profiling` | `timed_region`, `nvtx_range`, `count_transfers` |
| `xp.memory` | `HostStaging`, `DeviceMirror` |
| `xp.petsc` | `petsc_vec` |
| `cunumpy.kernel_testing` | pytest helpers for host/CUDA kernel pairs |

Everything except `xp.cuda` works on both backends.

## Install

```bash
python -m pip install cunumpy
```

NumPy and `array-api-compat` are installed as dependencies. To use a GPU,
install a CuPy package compatible with your CUDA environment as well. CuPy
installation depends on the CUDA version and platform; follow the CuPy
installation instructions for your system. CuNumpy does not install CUDA.

`array-api-compat` supplies NumPy and CuPy compatibility modules with more
consistent behavior for shared array operations. CuNumpy uses them internally;
your arrays remain ordinary NumPy or CuPy arrays. See [why CuNumpy uses
`array-api-compat`](docs/source/array-api-compat.md) for a plain-language
explanation and examples.

## Choose a backend

CuNumpy starts with NumPy unless `CUNUMPY_BACKEND=cupy` is set before import.
You can also choose at runtime:

```python
import cunumpy as xp

xp.set_backend("cupy")
print(xp.get_backend())  # 'cupy' if CuPy and CUDA are functional

values = xp.arange(5)  # created by the active backend
```

The accepted backend names are `"numpy"` and `"cupy"`. If CuPy is requested
but unavailable or not functional, CuNumpy falls back to NumPy. Always check
`get_backend()` when the effective backend matters, such as when reporting
configuration or deciding whether GPU-specific work will happen.

Use `xp.set_backend("cupy", strict=True)` to raise when CUDA is unavailable,
preserving the previous backend. `xp.backend_info()` returns structured backend,
dependency, and CUDA diagnostics. Reusable streams/events, MPI staging, cell
ranges, and prepared reductions are described in the
[execution helpers guide](docs/source/guides/execution-helpers.md).

Use `use_backend()` for a temporary selection. It restores the previous
selection when the block exits, including when an exception is raised:

```python
with xp.use_backend("numpy"):
    cpu_values = xp.linspace(0, 1, 100)
    assert xp.get_backend() == "numpy"

# The previous global backend is active again here.
```

The backend selection is process-wide shared state. Do not switch it
independently from multiple threads or async tasks; those changes can
interfere. A context manager is useful for sequential code, tests, and
notebooks.

## Understand the two backend questions

The active backend controls which library CuNumpy exposes through its NumPy
like operations. The array backend reports where one particular array lives.
These can differ: changing the active backend does not convert arrays that
already exist.

```python
xp.set_backend("numpy")
cpu_values = xp.arange(3)

gpu_values = xp.to_cupy(cpu_values)  # explicit transfer
print(xp.get_backend())  # 'numpy'
print(xp.get_array_backend(gpu_values))  # 'cupy'
```

Use `is_cpu(array)`, `is_gpu(array)`, or `get_array_backend(array)` when
dispatch should follow the array passed to a function. `get_array_module()`
returns the matching `array_api_compat` module, which is useful when writing
backend-generic functions:

```python
def vector_norm(values):
    array_xp = xp.get_array_module(values)
    return array_xp.sqrt(array_xp.sum(values * values))
```

## Move data between CPU and GPU

Transfers are explicit so it is clear when data crosses the CPU/GPU boundary:

```python
host = xp.to_numpy(gpu_values)  # CuPy -> NumPy (host)
device = xp.to_cupy(host)  # NumPy/array-like -> CuPy (device)
active = xp.to_cunumpy(host)  # convert to the currently selected backend
```

`to_numpy()` also accepts ordinary array-like values. `to_cupy()` raises
`ImportError` when CuPy or a functional CUDA runtime is unavailable.
`to_cunumpy()` is useful at API boundaries where the consumer expects the
currently selected backend. It does not change the original array.

Avoid transferring data inside a tight loop. Keep intermediate arrays on one
backend and move only at boundaries such as file I/O, plotting, or a
CPU-only library call. For example:

```python
with xp.use_backend("cupy"):
    signal = xp.asarray(host_signal)
    filtered = xp.fft.rfft(signal)
    result = xp.to_numpy(filtered)  # one transfer for a CPU-only consumer
```

To verify that a block, such as a time step, makes no transfer at all, count
them: `count_transfers()` records every `to_numpy()`, `to_cupy()` and
`to_cunumpy()` call that actually copies, mirror/staging refreshes, argument
conversion and kernel output copy-back, with call sites and payload byte counts.
Host kernel conversions and fallbacks have separate explanatory markers.
`assert_no_transfers()` rejects host/device movement and permits device-only
conversions. Only CuNumpy execution/conversion helpers are counted; forwarded
backend calls such as `xp.asarray()` and raw
`cupy.ndarray.get()` or `cupy.asarray()` calls need a profiler such as `nsys`.

```python
with xp.profiling.count_transfers() as counter:
    propagator(dt)

assert counter.to_host == counter.to_device == 0, counter.report()
print(counter.bytes_to_host, counter.bytes_to_device)
```

## Random numbers and dtypes

`get_rng(seed)` returns a random generator for the active backend. NumPy and
CuPy have similar generator APIs, though exact bit-for-bit sequences are not
guaranteed to match between libraries:

```python
rng = xp.rng.get_rng(seed=42)
samples = rng.normal(size=1000)
```

Use `default_float_dtype()` when code needs to explicitly request the active
backend's `float64` dtype rather than rely on Python scalar inference:

```python
x = xp.asarray([1.0, 2.0], dtype=xp.default_float_dtype())
```

## GPU selection and memory helpers

These helpers are useful for multi-GPU programs and for understanding CuPy's
memory behavior:

```python
print("visible GPUs:", xp.cuda.device_count())
xp.cuda.set_device(0)  # selects CUDA device 0 when CuPy is active
print("memory (free, total):", xp.cuda.memory_info())
```

`set_device()` is a no-op on NumPy. `device_count()` checks visible CUDA
hardware even if the active backend is NumPy; it returns zero when CuPy/CUDA
cannot be used. `memory_info()` returns `(free_bytes, total_bytes)` on the
active CuPy device and `None` on NumPy. `set_device_for_rank(rank)` is a
round-robin convenience for MPI layouts where local ranks map contiguously to
GPUs. If your scheduler uses a different mapping, select the device directly.

For MPI programs with one rank per GPU, the startup sequence is:

1. `bind_local_device()` selects the GPU from the node-local rank that the MPI
   launcher exports (`local_rank()`) and creates its CUDA context. It runs
   before MPI is initialized because a CUDA-aware MPI binds to the device that
   is current at `MPI_Init`; without it, every rank of a node would use
   device 0.
2. `from mpi4py import MPI` initializes MPI.
3. `require_cuda_aware_mpi()` (or `mpi_is_cuda_aware(comm)`) checks, with one
   tiny device `Sendrecv` on every rank, that the MPI library can pass device
   buffers at all. Passing CuPy arrays to a plain MPI build segfaults or
   silently sends garbage; the check turns that into a clear error at
   startup. It is a no-op on the NumPy backend.
4. `synchronize_for_mpi(*buffers)` before every MPI call with device buffers:
   kernels run asynchronously, and MPI would otherwise send a buffer a kernel
   is still writing, without an error.

```python
xp.set_backend("cupy")
xp.cuda.bind_local_device()  # before MPI_Init
from mpi4py import MPI  # MPI_Init

xp.mpi.require_cuda_aware_mpi()  # once, on all ranks

xp.mpi.synchronize_for_mpi(send, recv)
MPI.COMM_WORLD.Sendrecv(send, dest, recvbuf=recv, source=source)
```

CuPy caches released allocations in memory pools. This can make process-level
GPU memory appear occupied after arrays go out of scope. `free_memory()` asks
CuPy to release currently free cached blocks; it does not free memory still
referenced by live arrays.

`pin_memory(host_array)` makes a pinned host copy, which can improve transfer
throughput for workloads that explicitly manage asynchronous transfers.
`stream()` creates a non-blocking CuPy stream and yields it; it yields `None`
on NumPy. GPU work is asynchronous, so synchronize before reading results on
the host:

```python
with xp.cuda.stream():
    device = xp.to_cupy(host)
    transformed = xp.fft.fft(device)

xp.synchronize()
result = xp.to_numpy(transformed)
```

Because GPU work is asynchronous, a wall-clock timer around a kernel launch
measures the launch, not the kernel. `timed_region(name)` synchronizes the
device before reading the clock (on NumPy it is a plain timer), and
`nvtx_range(name)` marks a region so it shows up in `nsys`/Nsight; both are
no-ops or plain timers on NumPy, and `nvtx_range` also works as a decorator:

```python
with xp.profiling.timed_region("fft") as timing:
    transformed = xp.fft.fft(device)
print(timing.elapsed, timing.synced)


@xp.profiling.nvtx_range("step")
def step(dt): ...
```

## Use NumPy-only kernels with CuPy arrays

`PyccelKernel` adapts a callable that expects NumPy arrays. When conversion is
needed, CuNumpy copies CuPy inputs to the host, calls the wrapped function,
copies in-place output changes back to the device, and moves returned NumPy
arrays to CuPy. With NumPy inputs, the wrapper calls the function directly.
CuNumpy does not compile functions or import Pyccel for you.

```python
import cunumpy as xp


def scale_in_place(values, factor):
    values[:] *= factor
    return values


scale = xp.kernels.PyccelKernel(scale_in_place, outputs=(0,))

with xp.use_backend("cupy"):
    values = xp.arange(5, dtype=xp.float64)
    returned = scale(values, 3.0)
    xp.synchronize()
```

By default every converted argument is copied back, because the wrapper cannot
know which arguments the kernel changed. `outputs=(0,)` declares that
positional argument 0 is written, avoiding unnecessary copy-back for
read-only inputs. For a keyword call, declare the keyword name, such as
`outputs=("out",)`. A wrong declaration can leave GPU output values stale.
The wrapper can also traverse arrays nested in lists, tuples, dictionaries,
and selected application objects; see the full [API reference](docs/source/api.md)
for `object_modules`, `is_array`, aliasing, and output declarations.

## Write CUDA kernels next to host kernels

`CudaKernel` wraps a CUDA C kernel (compiled with NVRTC through
`cupy.RawKernel`) so that it is called with the same arguments as the host
kernel it mirrors. Thread counts default to the first array's leading shape
axes: one thread per row for 1D blocks, matching axes for 2D/3D blocks. Explicit
`n_threads`, `grid`, or a custom `n_threads_from` controls the launch when needed.
Arrays are never copied: they
must be C-contiguous CuPy arrays. The `extern "C" __global__` signature is
parsed once and every call is checked against it: Python scalars are cast to
the declared C types, and a wrong argument count, an array of the wrong dtype
or a non-contiguous view, or a scalar that does not fit its type raises instead
of silently producing wrong values.

`Kernel` pairs a host kernel with its CUDA kernel and calls the one matching
the active backend, so kernels can be ported to CUDA one at a time:

```python
import cunumpy as xp

AXPY = r"""
extern "C" __global__
void axpy(double a, const double* x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""


def axpy(a, x, y, n):  # host version, e.g. compiled with Pyccel
    for i in range(n):
        y[i] += a * x[i]


kernel = xp.kernels.Kernel(axpy, xp.cuda.CudaKernel(AXPY, "axpy"))

with xp.use_backend("cupy"):
    x = xp.arange(1000, dtype=xp.float64)
    y = xp.zeros(1000)
    kernel(2.0, x, y, 1000)  # infer n_threads = x.shape[0], run the CUDA kernel
```

On the CuPy backend, a `Kernel` without CUDA kernel raises
`NotImplementedError` (or, with `missing_cuda="fallback"`, runs the host kernel
through `PyccelKernel`, with host copies; `host_options` configure that
`PyccelKernel`). `KernelCatalog.from_package()` collects kernel pairs from a
package with one folder per kernel (`name/name_kernels.py` and
`name/name_cuda.cu`), and `catalog.compile_all()` compiles all CUDA kernels at
setup.

Groups of arguments can be passed as one: objects implementing
`__cuda_args__()` (see `CudaArguments`) are flattened into several kernel
arguments, and `CudaStruct` defines a C struct once (its C `declaration` and
the matching memory layout) and packs values into it, which the kernel takes
as one parameter:

```python
Vec = xp.cuda.CudaStruct("Vec", [("data", "double*"), ("n", "int")])
scale = xp.cuda.CudaKernel(
    Vec.declaration
    + r"""
    extern "C" __global__ void scale(Vec v, double a) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < v.n) v.data[i] *= a;
    }""",
    "scale",
    structs=[Vec],
)
scale(Vec(data=y, n=y.size), 0.5, n_threads=y.size)
```

When building such argument objects, `xp.as_device_array(value, dtype,
ndim=None)` applies the "reference or copy once" rule: a CuPy array that
already has the dtype and is C-contiguous is returned as it is, anything else
(a tuple such as `degree = (3, 3, 3)`, a host array, another dtype, a
non-contiguous view) is converted; dtype and layout changes can require separate
device copies. Call it once when the object is
built, not per kernel call; on the NumPy backend it raises, so host data is
never copied to the device implicitly.
When the host kernel takes such a group as one object too (e.g. a Pyccel class
holding NumPy arrays), give the group both forms with `KernelArguments`:
`__host_args__()` returns the object for the host kernel, `__cuda_args__()`
the flattened device arguments. `Kernel` and `PyccelKernel` resolve
`__host_args__()` on the host path and `CudaKernel` flattens `__cuda_args__()`
on the CUDA path, so the call site is the same on both backends and each form
can be built lazily on first access (a CPU run never builds device arguments):

```python
class ParticleArguments(xp.kernels.KernelArguments):
    def __init__(self, markers):
        self.markers = markers
        self._host = None

    def __host_args__(self):
        if self._host is None:
            self._host = MarkerArguments(self.markers)  # Pyccel class
        return self._host

    def __cuda_args__(self):
        return (self.markers, self.markers.shape[0])


kernel(particles.kernel_args, dt, n_threads=n)  # host or CUDA kernel
```

Kernels ported from pyccel index arrays like `markers[ip, j]`, which needs
shapes and strides rather than bare pointers. The shipped header
`cunumpy/array_view.cuh` (found by every `CudaKernel`) provides the strided
views `Array1D<T>` to `Array4D<T>`; a parameter or struct field of that type
takes a CuPy array, contiguous or not, and indexes `a(i, j)`. The struct can be
generated from the annotations of the pyccel argument class, so the Python
class is the one definition, and written to a header that a test keeps in sync:

```python
class MarkerArguments:
    def __init__(self, markers: "float[:, :]", n_markers: int, valid: "bool[:]"): ...


MarkerArgs = xp.cuda.CudaStruct.from_signature(MarkerArguments.__init__, "MarkerArgs")
MarkerArgs.to_header(
    "marker_args.cuh"
)  # Array2D<double> markers; long long n_markers; ...
push = xp.cuda.CudaKernel(
    r"""
    #include "marker_args.cuh"
    #include <cunumpy/index.cuh>
    extern "C" __global__ void push(MarkerArgs m, double dt) {
        CUNUMPY_THREAD_1D(ip, m.n_markers);
        if (m.valid(ip)) m.markers(ip, 0) += dt * m.markers(ip, 3);
    }""",
    "push",
    structs=[MarkerArgs],
    include_dirs=["."],
)
push(
    MarkerArgs(markers=markers, n_markers=markers.shape[0], valid=valid),
    0.1,
    n_threads=markers.shape[0],
)
```

Launches can be 1D to 3D (`n_threads=(nx, ny)`, `block_size=(16, 16)`) or use
an explicit `grid`, with dynamic shared memory (`shared_mem`) and a `stream`.
C++ function templates are instantiated with `template_args`, and
`CudaKernelVariants` caches kernels whose source is generated per variant
(e.g. per dimension and dtype). See the [API reference](docs/source/api.md) for
details.

Kernels run asynchronously, so a CUDA error (an illegal memory access, say)
normally surfaces at a later `.get()` or MPI call, far from the kernel that
caused it. In debug mode, enabled with `xp.cuda.set_cuda_debug(True)`, the
context manager `xp.cuda.cuda_debug()`, `CudaKernel(..., debug=True)` or the
environment variable `CUNUMPY_CUDA_DEBUG=1`, kernels are compiled with
`-lineinfo` and `-DCUNUMPY_BOUNDS_CHECK` and every launch is synchronized, so
the error is raised as a `RuntimeError` naming the kernel and its launch shape.
To find the faulting line and out-of-bounds accesses that do not crash, the
next step is NVIDIA's memory checker:
`CUNUMPY_CUDA_DEBUG=1 compute-sanitizer python -m pytest ...`.

## Test kernel pairs

`cunumpy.kernel_testing` helps to test the ports with pytest. `assert_kernels_agree`
builds the arguments on both backends, runs the host and the CUDA kernel and
compares the arrays they wrote; with `catalog.parity_cases()`, one
parametrised test covers every ported kernel of a catalog. `BACKENDS` and
`requires_cupy` parametrize tests over the backends, skipping CuPy without a
GPU, and `device_function_kernel` wraps a `__device__` helper in an elementwise
kernel so it can be checked against its host version without writing a test
kernel:

```python
import pytest
from cunumpy.kernel_testing import assert_kernels_agree


def make_args(backend, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(1000))
    return (x, 2.0, x.size)


@pytest.mark.parametrize("name, kernel", catalog.parity_cases())
def test_parity(name, kernel):
    assert_kernels_agree(kernel, make_args, n_threads=1000)
```

Accumulation kernels often write into a buffer that another library owns on
the host (a stencil vector's `_data`, exchanged over MPI). `DeviceMirror`
pairs that NumPy array with a device copy: `mirror.device` is the CuPy array
on the GPU and the host array itself on the CPU, `to_host()` copies back in
place (the host array keeps its identity) and `zero()` clears the buffer, so
the one transfer per accumulation is explicit. The shipped header
`cunumpy/atomic.cuh` (found automatically, see `cuda_include_dir()`) provides
`cunumpy_atomic_add()` and 2D/3D indexed variants for the many-threads-to-one-cell
writes:

```python
mirror = xp.memory.DeviceMirror(vector._data)
mirror.zero()
accumulate(markers, mirror.device, n_threads=n_markers)
mirror.to_host()  # vector._data holds the result on both backends
```

## Pyodide

CuNumpy supports the NumPy backend in Pyodide. It does not provide CuPy/CUDA
there. Ordinary Python callables can be wrapped with `PyccelKernel` without
compilation. See the [Pyodide guide](docs/source/pyodide.md) for a complete
installation example and compatibility notes.

## Documentation

The full documentation lives in [`docs/source`](docs/source/index.md) and is
published at <https://max-models.github.io/cunumpy/>:

* Getting started: [installation](docs/source/installation.md) and a
  [quickstart](docs/source/quickstart.md) with a map of which guide covers what.
* User guide: [choosing a backend](docs/source/guides/backends.md),
  [backend-agnostic code](docs/source/guides/portable-code.md),
  [data movement](docs/source/guides/data-movement.md),
  [devices, memory and streams](docs/source/guides/gpu-devices.md),
  [MPI with one rank per GPU](docs/source/guides/mpi.md),
  [timing and profiling](docs/source/guides/profiling.md).
* Porting kernels: [overview](docs/source/kernels/overview.md),
  [`PyccelKernel`](docs/source/kernels/pyccel-kernel.md),
  [`CudaKernel`](docs/source/kernels/cuda-kernel.md),
  [`Kernel` and `KernelCatalog`](docs/source/kernels/dispatch.md),
  [argument objects and structs](docs/source/kernels/arguments.md),
  [accumulation kernels](docs/source/kernels/accumulation.md),
  [debugging](docs/source/kernels/debugging.md),
  [testing](docs/source/kernels/testing.md).
* [Worked examples](docs/source/examples/index.md),
  [best practices](docs/source/best-practices.md),
  [troubleshooting](docs/source/troubleshooting.md),
  [Pyodide](docs/source/pyodide.md) and the
  [API reference](docs/source/api.md).

### For AI coding assistants

[`src/cunumpy/LLM_GUIDE.md`](src/cunumpy/LLM_GUIDE.md) is a compact,
self-contained guide to the API and its rules for LLM-based coding assistants.
It ships inside the installed package, so an assistant working in a project that
depends on CuNumpy can read it from `site-packages/cunumpy/LLM_GUIDE.md`, or
locate it with:

```bash
python -c "import cunumpy, pathlib; print(pathlib.Path(cunumpy.__file__).parent / 'LLM_GUIDE.md')"
```
