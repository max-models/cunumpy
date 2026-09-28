# API reference

CuNumpy exports the active NumPy-like namespace and a set of helpers for
backend selection, array inspection and conversion, hardware control, and
host-only kernels. Most examples use `import cunumpy as xp`.

## NumPy-like namespace

```python
import cunumpy as xp

values = xp.arange(5)
total = xp.sum(values)
```

At runtime, NumPy-like attributes such as `array`, `sum`, `fft`, and `linalg`
are forwarded to the currently selected `array-api-compat` NumPy or CuPy
module. CuNumpy does not wrap every operation individually. The available
operations and some details can therefore vary with the installed NumPy and
CuPy versions. In normal use, access those operations through the top-level
`cunumpy` namespace, commonly imported as `xp`.

NumPy and CuPy are not interchangeable for every function or object. A
function that needs to follow an input array's location should use
`get_array_module(array)` instead of assuming the global backend matches it.

## Backend selection

### `set_backend(backend)`

Selects the process-wide backend used for new NumPy-like operations. Supported
values are `"numpy"` and `"cupy"`:

```python
xp.set_backend("cupy")
values = xp.arange(10)
```

Requesting CuPy selects it only when CuPy and its CUDA runtime are functional;
otherwise CuNumpy falls back to NumPy. Check `get_backend()` to inspect the
effective selection. Changing the selection does not move arrays that have
already been created.

The initial backend is NumPy unless `ARRAY_BACKEND=cupy` is set before CuNumpy
is imported. Other values of this environment variable result in the NumPy
default.

Backend state is shared process-wide and `set_backend()` is not thread-safe.
Concurrent tasks that change the backend may race.

### `get_backend()`

Returns the active global backend name, either `"numpy"` or `"cupy"`. This is
the getter paired with `set_backend()`:

```python
xp.set_backend("numpy")
assert xp.get_backend() == "numpy"
```

### `use_backend(backend)`

Context manager that temporarily selects a backend and restores the previous
backend on exit, even if the block raises an exception:

```python
with xp.use_backend("numpy"):
    reference = xp.zeros(10)
```

As backend selection is shared process state, this context manager is suited
to sequential use rather than concurrent backend switching.

### `numpy_backend`, `cupy_backend`

Boolean properties indicating whether the currently selected global backend
is NumPy or CuPy:

```python
if xp.cupy_backend:
    print("new arrays are being created on the GPU")
```

For a string value, prefer `get_backend()`.

## Inspect arrays and select operations

### `get_array_backend(array)`

Returns `"numpy"` or `"cupy"` according to the given array's type. It reports
the array's location, not the active global selection:

```python
xp.set_backend("numpy")
values_gpu = xp.to_cupy([1, 2, 3])
assert xp.get_backend() == "numpy"
assert xp.get_array_backend(values_gpu) == "cupy"
```

### `get_array_module(array)`

Returns the `array-api-compat` module matching the array: the NumPy module for
a NumPy array or the CuPy module for a CuPy array. This supports functions
that dispatch based on their input rather than global state:

```python
def standardize(values):
    array_xp = xp.get_array_module(values)
    mean = array_xp.mean(values)
    scale = array_xp.std(values)
    return (values - mean) / scale
```

The returned module is an `array-api-compat` module, consistent with the
active module used by CuNumpy. It is not necessarily identical to importing
raw `numpy` or raw `cupy`.

### `is_cpu(array)`, `is_gpu(array)`

Return booleans indicating whether an array is a NumPy (CPU) or CuPy (GPU)
array. They are convenience checks equivalent to comparing
`get_array_backend(array)` with `"numpy"` or `"cupy"`.

### `same_backend(*arrays)`

Returns `True` if all provided arrays have the same backend. Zero or one
argument is considered to match:

```python
if xp.same_backend(position, velocity):
    update(position, velocity)
```

### `assert_same_backend(*arrays)`

Raises `TypeError` with the detected backend names if arrays do not all share
a backend. Use it at API boundaries to provide a clear error before a mixed
NumPy/CuPy operation fails deeper in a library:

```python
def combine(left, right):
    xp.assert_same_backend(left, right)
    return left + right
```

## Convert arrays

### `to_numpy(array)`

Converts to a host-side NumPy array. CuPy arrays are copied from device to
host. Other array-like inputs are passed through `numpy.asarray`; NumPy arrays
may therefore be returned as-is rather than copied.

### `to_cupy(array)`

Converts an array-like input to a CuPy array. Raises `ImportError` if CuPy or
CUDA is unavailable or not functional. The source is not modified.

### `to_cunumpy(array)`

Converts to the currently active backend. This is convenient at an API
boundary when an input should be normalized to the configured backend:

```python
normalized = xp.to_cunumpy(input_array)
assert xp.get_array_backend(normalized) == xp.get_backend()
```

Each conversion returns a suitable array; it does not change the active
backend or mutate the source.

## Random numbers and dtype

### `get_rng(seed=None)`

Returns a NumPy or CuPy `Generator` matching the active backend:

```python
rng = xp.get_rng(seed=7)
samples = rng.uniform(size=100)
```

The generator APIs are similar, but seeds do not guarantee identical random
sequences across NumPy and CuPy.

### `default_float_dtype()`

Returns the active backend module's `float64` dtype object. Pass it to array
creation when code requires an explicit precision:

```python
values = xp.asarray([0.1, 0.2], dtype=xp.default_float_dtype())
```

## CUDA devices and memory

### `cupy_available()`

Returns whether CuPy can be imported and reports itself functional. The result
is cached for the process. This checks availability, not whether every
GPU-specific operation will succeed later.

### `device_count()`

Returns the number of visible CUDA devices. Returns zero when CuPy/CUDA is
unavailable or querying the runtime fails. This is independent of the active
backend, so it may return a positive number while NumPy is selected.

### `set_device(device_id)`

Selects a CUDA device when CuPy is active; it is a no-op on NumPy. The device
must be valid for the current CUDA process.

### `set_device_for_rank(rank, devices_per_node=None)`

Selects a device using `rank % devices_per_node` and returns its ID. If
`devices_per_node` is omitted, it uses `device_count()`. When no devices are
visible, it returns `0` without selecting a device. This helper assumes
contiguous rank-to-device mapping on each node, suitable for a common
one-rank-per-GPU MPI layout. Use `set_device()` directly when the scheduler's
mapping differs:

```python
device_id = xp.set_device_for_rank(mpi_rank)
```

### `memory_info()`

Returns `(free_bytes, total_bytes)` reported by the CUDA runtime for the
active device, or `None` on NumPy. The values cover the device, not only
allocations owned by CuPy.

### `free_memory()`

Releases currently free blocks in CuPy's device and pinned-host memory pools.
It is a no-op on NumPy. It does not release blocks still referenced by live
arrays. CuPy normally caches freed allocations for reuse, so cached memory
does not necessarily indicate a leak.

### `pin_memory(array)`

Copies a host array to page-locked (pinned) host memory. Pinned memory can
improve host/device transfer throughput in suitable asynchronous workloads.
Raises `ImportError` if CuPy is unavailable. If the input may be a CuPy array,
first transfer it with `to_numpy()`.

### `synchronize()`

Waits for queued work on the current CUDA device to finish. This is useful
before reading asynchronously computed results from host code. It is a no-op
on NumPy.

### `stream()`

Context manager that creates a non-blocking CuPy stream and yields it. Work
issued in the block is enqueued on that stream. On NumPy, yields `None` and
does nothing:

```python
with xp.stream() as work_stream:
    result = xp.to_cupy(host_values) * 2

work_stream.synchronize()  # on CuPy; the yielded value is None on NumPy
```

Do not call methods on the yielded value without checking the backend. Use
`xp.synchronize()` for code that should work on both backends.

## `PyccelKernel`

### Constructor

```python
xp.PyccelKernel(
    kernel,
    use_cupy=None,
    object_modules=(),
    is_array=None,
    outputs=None,
)
```

Wraps a callable expecting host NumPy arrays so it can be used with CuPy
arrays. This can adapt a Pyccel-compiled kernel or an ordinary Python
callable. CuNumpy does not compile the callable or import Pyccel.

When no conversion is needed, the original callable is invoked directly. If
conversion is needed, CuNumpy recursively replaces CuPy arrays in supported
arguments with NumPy host copies, calls the kernel, copies in-place updates
back to the corresponding CuPy arrays, and converts returned NumPy arrays to
CuPy arrays.

### Parameters

* `kernel`: callable that accepts the host-side arguments.
* `use_cupy`: `None` (default) chooses conversion for each call when the
  global backend is CuPy or a CuPy array is present. `True` forces conversion;
  `False` disables it.
* `object_modules`: module prefixes whose instances should be shallow-copied
  and traversed by attributes. For example,
  `object_modules=("my_project.",)`.
* `is_array`: predicate for host array values to convert back to CuPy. The
  default is `isinstance(value, numpy.ndarray)`.
* `outputs`: sequence of arguments the kernel may write to. Entries are
  positional indices or keyword names. If omitted, every converted array is
  copied back.

### Example and output declarations

```python
def scale_and_shift(scale, values, out):
    out[:] = scale * values + 1
    return out

kernel = xp.PyccelKernel(scale_and_shift, outputs=(2,))

with xp.use_backend("cupy"):
    values = xp.arange(8, dtype=xp.float64)
    out = xp.empty_like(values)
    result = kernel(2.0, values, out)
```

`outputs=(2,)` marks only `out` for copy-back. Read-only `values` is copied
to the host for the call but not transferred back. Declare a positional
argument by index (negative indices count from the end), and a keyword
argument by its name, such as `outputs=("out",)` for `kernel(..., out=out)`.
The forms are not interchangeable because compiled builtins may not expose a
Python signature. Invalid declarations raise `IndexError` or `KeyError`.

An incorrect declaration is a correctness bug: if the kernel writes an
argument that was not declared, the device array will not receive that
update. Use `outputs=()` only when no converted input is mutated. Containers
and selected objects can be declared as outputs; every nested supported
array is then copied back. If an array is reachable through multiple paths,
declaring any path that includes it is sufficient.

### Supported containers, aliasing, and returns

Tuples, lists, and dictionaries are traversed recursively. Objects are
traversed only when their class module starts with one of the configured
`object_modules` prefixes; those objects are shallow-copied, and their
attributes are converted on the copy. Other objects are passed to the kernel
unchanged.

The wrapper memoizes conversions within a call. If the same device array is
passed more than once or appears inside a supported container, the host kernel
sees the same NumPy array object, preserving aliasing. Supported reference
cycles terminate safely. Returned NumPy arrays (and arrays inside tuples or
lists) are converted back using `is_array`; dictionaries in return values are
not recursively converted. On the NumPy path, the original return value and
normal Python mutation and exception behavior are preserved.

## Version

`xp.__version__` is the installed package version. When package metadata is
not available (for example, some source-tree imports), it is
`"0.0.0+unknown"`.
