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

CuNumpy starts with NumPy unless `ARRAY_BACKEND=cupy` is set before import.
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
`to_cunumpy()` call that actually copies, every `PyccelKernel` call that
converts device arrays, and every `Kernel` fallback to the host kernel, with
the call site of each. `assert_no_transfers()` raises with that report if
anything was counted. Only transfers made through CuNumpy are seen; raw
`cupy.ndarray.get()` or `cupy.asarray()` calls need a profiler such as `nsys`.

```python
with xp.count_transfers() as counter:
    propagator(dt)

assert counter.total == 0, counter.report()
```

## Random numbers and dtypes

`get_rng(seed)` returns a random generator for the active backend. NumPy and
CuPy have similar generator APIs, though exact bit-for-bit sequences are not
guaranteed to match between libraries:

```python
rng = xp.get_rng(seed=42)
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
print("visible GPUs:", xp.device_count())
xp.set_device(0)  # selects CUDA device 0 when CuPy is active
print("memory (free, total):", xp.memory_info())
```

`set_device()` is a no-op on NumPy. `device_count()` checks visible CUDA
hardware even if the active backend is NumPy; it returns zero when CuPy/CUDA
cannot be used. `memory_info()` returns `(free_bytes, total_bytes)` on the
active CuPy device and `None` on NumPy. `set_device_for_rank(rank)` is a
round-robin convenience for MPI layouts where local ranks map contiguously to
GPUs. If your scheduler uses a different mapping, select the device directly.

For MPI programs with one rank per GPU, `bind_local_device()` selects the GPU
from the node-local rank that the MPI launcher exports (`local_rank()`), so it
can run before MPI is initialized, as CUDA-aware MPI requires. Before passing
device buffers to MPI, call `synchronize_for_mpi(*buffers)`: kernels run
asynchronously, and MPI would otherwise send a buffer a kernel is still
writing, without an error.

```python
xp.set_backend("cupy")
xp.bind_local_device()  # before MPI_Init
from mpi4py import MPI

xp.synchronize_for_mpi(send, recv)
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
with xp.stream():
    device = xp.to_cupy(host)
    transformed = xp.fft.fft(device)

xp.synchronize()
result = xp.to_numpy(transformed)
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


scale = xp.PyccelKernel(scale_in_place, outputs=(0,))

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
kernel it mirrors, plus the number of threads. Arrays are never copied: they
must be CuPy arrays. The `extern "C" __global__` signature is parsed once and
every call is checked against it: Python scalars are cast to the declared C
types, and a wrong argument count, an array of the wrong dtype, or a scalar
that does not fit its type raises instead of silently producing wrong values.

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


kernel = xp.Kernel(axpy, xp.CudaKernel(AXPY, "axpy"))

with xp.use_backend("cupy"):
    x = xp.arange(1000, dtype=xp.float64)
    y = xp.zeros(1000)
    kernel(2.0, x, y, 1000, n_threads=1000)  # runs the CUDA kernel
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
Vec = xp.CudaStruct("Vec", [("data", "double*"), ("n", "int")])
scale = xp.CudaKernel(
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

Launches can be 1D to 3D (`n_threads=(nx, ny)`, `block_size=(16, 16)`) or use
an explicit `grid`, with dynamic shared memory (`shared_mem`) and a `stream`.
C++ function templates are instantiated with `template_args`, and
`CudaKernelVariants` caches kernels whose source is generated per variant
(e.g. per dimension and dtype). See the [API reference](docs/source/api.md) for
details.

## Pyodide

CuNumpy supports the NumPy backend in Pyodide. It does not provide CuPy/CUDA
there. Ordinary Python callables can be wrapped with `PyccelKernel` without
compilation. See the [Pyodide guide](docs/source/pyodide.md) for a complete
installation example and compatibility notes.

## Documentation

The [user guide](docs/source/quickstart.md) explains common workflows. The
[API reference](docs/source/api.md) documents each helper and its behavior.
The [Pyodide guide](docs/source/pyodide.md) covers WebAssembly usage.
