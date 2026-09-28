# User guide

CuNumpy gives CPU and GPU programs a common NumPy-like entry point. This guide
starts with the default CPU workflow and then shows backend selection,
array-aware dispatch, data transfers, and GPU utilities.

## Install and import

Install the package with pip:

```bash
python -m pip install cunumpy
```

CuNumpy depends on NumPy and `array-api-compat`. To use NVIDIA GPUs, install a
CuPy distribution that matches your CUDA environment separately. CUDA itself
is not installed by CuNumpy.

`array-api-compat` is a small adapter that gives NumPy and CuPy a more
consistent interface for shared array operations. You do not need to import
it directly when using CuNumpy. See [Why CuNumpy uses
`array-api-compat`](array-api-compat.md) for a beginner-friendly explanation
and examples.

Import CuNumpy using the familiar alias `xp`:

```python
import cunumpy as xp

a = xp.array([1.0, 2.0, 3.0])
b = xp.asarray([4.0, 5.0, 6.0])
print(xp.dot(a, b))
```

Most common array operations are available through the active NumPy-like
namespace, including array creation, indexing, arithmetic, reductions,
linear algebra, random operations via `xp.random`, and FFTs via `xp.fft`.
CuNumpy aims to keep the interface familiar; NumPy and CuPy are separate
libraries, so niche functions and edge-case behavior can differ. Consult the
upstream library documentation for operation-specific details.

## Configure the active backend

The default is NumPy. Select CuPy using an environment variable before
starting Python:

```bash
ARRAY_BACKEND=cupy python my_program.py
```

Or change the active backend within a running program:

```python
xp.set_backend("cupy")
print(xp.get_backend())  # active backend name

data = xp.arange(1000)   # created on the GPU while CuPy is active
```

Only `"numpy"` and `"cupy"` are valid names. If CuPy is requested but is
missing or fails its availability check, initialization falls back to NumPy.
Inspect `xp.get_backend()` after selection if the effective backend matters.
The boolean properties `xp.numpy_backend` and `xp.cupy_backend` are also
available for conditional code.

Use a context manager when only a section should use a particular backend:

```python
xp.set_backend("cupy")

with xp.use_backend("numpy"):
    reference = xp.linspace(0, 1, 100)
    print(xp.get_backend())  # 'numpy'

print(xp.get_backend())      # 'cupy' again
```

The old backend is restored even if the block raises an exception. Backend
selection is a process-wide setting and is not thread-safe; concurrent code
must not switch it independently from different threads or async tasks.

## The active backend and an array's backend

These are deliberately separate concepts:

* `xp.get_backend()` returns the globally selected backend for new CuNumpy
  operations.
* `xp.get_array_backend(array)` returns `"numpy"` or `"cupy"` according to
  where a particular array is stored.
* `xp.get_array_module(array)` returns the matching `array_api_compat` module.

Changing the global selection does not migrate arrays already created. For
example, the active backend can be NumPy while a CuPy array is still alive:

```python
xp.set_backend("cupy")
gpu_values = xp.arange(4)

xp.set_backend("numpy")
print(xp.get_backend())                  # 'numpy'
print(xp.get_array_backend(gpu_values))   # 'cupy'
```

For a function that should follow its input array rather than global state,
use `get_array_module()`:

```python
def normalize(values):
    array_xp = xp.get_array_module(values)
    length = array_xp.sqrt(array_xp.sum(values * values))
    return values / length
```

This is often the right approach for reusable functions called with arrays
created by other libraries. `is_cpu(values)` and `is_gpu(values)` are concise
boolean checks. `same_backend(a, b)` checks whether all given arrays share a
backend, and `assert_same_backend(a, b)` raises a `TypeError` with the detected
backends if they do not.

## Convert arrays explicitly

Use explicit conversion at CPU/GPU boundaries:

```python
host_array = xp.to_numpy(gpu_array)
device_array = xp.to_cupy(host_array)
active_array = xp.to_cunumpy(host_array)
```

`to_numpy()` always returns a host-side NumPy array. For a NumPy input it
uses `numpy.asarray`, so an existing array or view can be returned without a
copy. For a CuPy input it transfers data from device to host. `to_cupy()`
converts an array-like object to a CuPy array, and requires working CuPy/CUDA.
`to_cunumpy()` converts to whichever backend is currently active.

Conversions do not mutate the source array. To avoid repeated transfer costs,
keep data on one device for a whole computational phase and transfer results
once at a boundary:

```python
with xp.use_backend("cupy"):
    signal_gpu = xp.asarray(signal_host)
    spectrum_gpu = xp.fft.rfft(signal_gpu)
    spectrum_host = xp.to_numpy(spectrum_gpu)
```

Mixing NumPy and CuPy arrays in the same operation is usually an error. Use
`assert_same_backend` to fail early in APIs that require matched arrays, or
convert the inputs to a common backend first.

## Random generators and floating-point dtypes

`xp.get_rng(seed)` chooses a random generator for the active backend:

```python
rng = xp.get_rng(seed=1234)
noise = rng.normal(loc=0.0, scale=1.0, size=10_000)
```

The interface is similar across NumPy and CuPy, but generated values are not
expected to be bitwise identical across backends. When reproducibility
matters, keep the backend and library versions fixed as well as the seed.

`xp.default_float_dtype()` returns the active module's `float64` dtype object.
It can be used to request an explicit portable precision:

```python
weights = xp.asarray([0.25, 0.75], dtype=xp.default_float_dtype())
```

Explicit dtypes are especially useful when code must not rely on backend- or
platform-specific inference for Python integers and floats.

## Work with CUDA devices

`xp.device_count()` reports the number of visible CUDA devices, or zero if
CuPy/CUDA is unavailable. It checks hardware visibility regardless of the
active array backend.

```python
count = xp.device_count()
if count:
    xp.set_backend("cupy")
    xp.set_device(0)
    values = xp.arange(100)
```

`set_device(device_id)` selects a CUDA device when the active backend is
CuPy and does nothing on NumPy. For a common one-rank-per-GPU MPI layout,
`set_device_for_rank(rank)` selects `rank % device_count()` and returns the
selected ID. Its default assumes ranks are arranged in contiguous device
blocks on each node; supply `devices_per_node` or call `set_device()` yourself
if your process mapping differs.

`memory_info()` returns `(free_bytes, total_bytes)` for the active CUDA device,
or `None` on NumPy. This queries CUDA's runtime memory accounting, not only
CuPy's allocator. CuPy keeps freed blocks in pools for reuse, so cached memory
may remain visible as allocated. `free_memory()` releases free cached blocks
from CuPy's device and pinned-host pools; live arrays remain allocated.

## Streams and synchronization

CUDA operations are generally asynchronous with respect to the host. A stream
orders queued work; synchronization waits until that work has completed.
`xp.stream()` creates a non-blocking CuPy stream and yields it. On NumPy it is
a no-op context manager that yields `None`:

```python
with xp.stream() as stream:
    device_values = xp.to_cupy(host_values)
    result = xp.exp(device_values)

xp.synchronize()  # wait before host code consumes the result
host_result = xp.to_numpy(result)
```

For finer control, synchronize the yielded stream with `stream.synchronize()`.
Use `xp.synchronize()` when you need to wait for all work on the current
device. Synchronization is a no-op on NumPy.

`pin_memory(host_array)` creates a pinned (page-locked) host copy. Pinned
memory can improve host/device transfer throughput, especially for workloads
that overlap transfers and computation, but should be used when transfer
profiling indicates it is useful:

```python
pinned = xp.pin_memory(host_array)
```

## Adapt NumPy-only or Pyccel kernels

`xp.PyccelKernel` wraps a callable that expects NumPy arrays. It is useful for
compiled Pyccel functions and for other host-only kernels. It is an adapter,
not a compiler: compilation and imports remain the application's
responsibility.

With the NumPy backend and NumPy arguments, the wrapped callable runs
directly. When conversion is needed (active CuPy backend or CuPy arrays among
the arguments), CuNumpy makes host copies of device arrays, invokes the
callable, copies declared in-place outputs back to the device, and converts
returned NumPy arrays to CuPy.

```python
def axpy(alpha, x, y, out):
    out[:] = alpha * x + y
    return out

axpy_cpu = xp.PyccelKernel(axpy, outputs=(3,))

with xp.use_backend("cupy"):
    x = xp.arange(8, dtype=xp.float64)
    y = xp.ones_like(x)
    out = xp.empty_like(x)
    result = axpy_cpu(2.0, x, y, out)
```

By default (`outputs=None`), all converted arrays are copied back after a
converted call. Specify only arguments the kernel may mutate to avoid copying
read-only inputs back. Positional arguments are declared by index; keyword
arguments by name. An argument passed by keyword must be declared by name,
because compiled builtins do not expose a signature for mapping it to a
position. A wrong output declaration can silently discard an in-place update
on the host copy, so include every argument the kernel writes.

Lists, tuples, and dictionaries are traversed recursively. For instances from
your own modules, pass their module prefixes in `object_modules`, for example
`object_modules=("my_simulation.",)`. The wrapper shallow-copies those objects
and traverses their attributes, preserving the caller's original references.
`is_array` customizes which host result types are converted back to CuPy; its
default recognizes `numpy.ndarray`.

Aliased device arrays are converted only once per call, so if the same array
appears in multiple arguments the kernel sees the same host array. Reference
cycles in supported containers are handled. `use_cupy=True` forces conversion
on and `use_cupy=False` forces it off; leave it as `None` for per-call
automatic selection.

## Pyodide and WebAssembly

CuNumpy's NumPy backend works in Pyodide. CuPy/CUDA and native Pyccel
compilation are not supported in that environment. Python-source functions
can still be wrapped with `PyccelKernel`; keep `use_cupy` unset or false.
Follow the [Pyodide guide](pyodide.md) for an install-and-run example and
runtime constraints.
