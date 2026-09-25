# API

The `cunumpy` package exposes the following helper functions in addition to the standard NumPy/CuPy API.

## Backend Management

### `to_numpy(array)`
Converts an array to a NumPy array on the CPU.

### `to_cupy(array)`
Converts an array to a CuPy array on the GPU. Raises `ImportError` if CuPy is not available.

### `to_cunumpy(array)`
Converts an array to the currently active backend.

### `get_backend(array)`
Returns the name of the backend (`"numpy"` or `"cupy"`) for the given array.

### `get_array_module(array)`
Returns the array-api-compat module (`numpy` or `cupy`) matching the given array's own backend — not necessarily the currently active global backend. Useful for writing functions that dispatch correctly on whatever array they receive, independent of `set_backend`/`use_backend`:

```python
def norm(array):
    xp_ = xp.get_array_module(array)
    return xp_.sqrt(xp_.sum(array**2))
```

### `is_gpu(array)`
Returns `True` if the array is stored on a GPU (CuPy).

### `is_cpu(array)`
Returns `True` if the array is stored on a CPU (NumPy).

## Global Configuration

### `numpy_backend`
Boolean property that returns `True` if the currently active global backend is NumPy.

### `cupy_backend`
Boolean property that returns `True` if the currently active global backend is CuPy.

### `set_backend(backend_name)`
Globally sets the active backend for all `cunumpy` operations. `backend_name` should be `"numpy"` or `"cupy"`.

### `use_backend(backend_name)`
A context manager that temporarily sets the active backend.

```python
with xp.use_backend("numpy"):
    # Operations here use NumPy
    pass
```

## Hardware Control

### `synchronize()`
Blocks until all preceding GPU operations are complete. This is a no-op when using the NumPy backend.

### `device_count()`
Returns the number of visible CUDA devices. Returns `0` on the NumPy backend or if CuPy/CUDA is unavailable. Independent of the currently active backend.

### `set_device_for_rank(rank, devices_per_node=None)`
Convenience for one-MPI-rank-per-GPU codes: selects device `rank % devices_per_node` via `set_device()` and returns the device id chosen. `devices_per_node` defaults to `device_count()`. No-op (returns `0`) with no visible devices.

```python
xp.set_device_for_rank(mpi_rank)  # each rank picks its own GPU
```

### `memory_info()`
Returns `(free, total)` bytes of memory on the active CUDA device, or `None` on the NumPy backend.

### `free_memory()`
Releases all free blocks held by CuPy's device and pinned-host memory pools. No-op on the NumPy backend. CuPy caches freed memory rather than returning it to the driver immediately, which can look like a leak in long-running processes.

### `pin_memory(array)`
Copies a host array into pinned (page-locked) CUDA host memory, which transfers to/from the GPU faster than regular pageable memory. Raises `ImportError` if CuPy is unavailable.

### `stream()`
Context manager for a CUDA stream, to overlap transfers and compute. No-op (yields `None`) on the NumPy backend.

```python
with xp.stream() as s:
    arr = xp.to_cupy(host_array)  # enqueued on the new stream
xp.synchronize()  # wait for it before reading results
```

### `get_rng(seed=None)`
Returns a `numpy.random.Generator` or `cupy.random.Generator` matching the active backend, so callers don't have to branch on the backend themselves.

### `default_float_dtype()`
Returns the active backend's `float64` dtype object. NumPy and CuPy resolve Python literals and the bare `dtype=float` spelling to a platform- or backend-dependent default; pass this explicitly when a specific, portable precision matters.

## Compiled Kernels

### `PyccelKernel(kernel, use_cupy=None, object_modules=(), outputs=None)`
Wraps a kernel compiled with [pyccel](https://github.com/pyccel/pyccel) — which only accepts NumPy arrays — so that it can be called with CuPy arrays as well.

On the CuPy backend the arguments are copied to the host before the call, any in-place updates the kernel makes are copied back to the device afterwards, and arrays returned by the kernel are moved back to the device. On the NumPy backend the kernel is called directly, without any conversion.

```python
import cunumpy as xp
from my_package.kernels import axpy  # pyccelized kernel

axpy = xp.PyccelKernel(axpy)

with xp.use_backend("cupy"):
    x = xp.arange(10, dtype=xp.float64)
    y = xp.ones(10, dtype=xp.float64)
    out = xp.zeros(10, dtype=xp.float64)
    axpy(2.0, x, y, out)  # `out` is updated in place, on the GPU
```

Tuples, lists and dicts are traversed recursively. Pass `object_modules` to also traverse the attributes of your own objects, e.g. `object_modules=("struphy.", "feectools.")`; instances from other modules are handed to the kernel untouched.

By default, only `numpy.ndarray` values are recognized as arrays to convert back to the device. Pass `is_array` to recognize a different (or additional) host array type instead, e.g. `is_array=lambda v: isinstance(v, (np.ndarray, np.ma.MaskedArray))`.

#### Declaring outputs

By default every array that was copied to the host is copied back afterwards, since the wrapper cannot know which ones the kernel wrote to. Most pyccel kernels write to one `out` argument and only read the rest, so `outputs` lets you skip the needless transfers:

```python
interpolate = xp.PyccelKernel(some_interpolation_kernel, outputs=(5,))

interpolate(x, y, z, basis, coeffs, out)  # `out` is argument 5
```

Only the declared arguments are copied back; `basis` and `coeffs` make the trip to the host and no further. Containers and traversed objects may be declared too — every array nested inside them is copied back. An array that is *also* reachable from a declared output (e.g. passed as both an input and the output) is still copied back.

Declare positional arguments by index (negatives count from the end) and keyword arguments by name, e.g. `outputs=("out",)` for `interpolate(..., out=out)`. The two are not interchangeable: pyccel-compiled kernels are builtins with no introspectable signature, so the wrapper cannot map a name onto a position. A declaration that matches no argument of the call raises `IndexError`/`KeyError` rather than silently copying nothing back.

`outputs=()` declares that the kernel writes to none of its arguments. Leaving `outputs` unset keeps the always-correct default. Note that a *wrong* declaration is a silent-wrong-answer bug: an argument the kernel writes to but that you did not declare keeps its stale values on the GPU.

#### Aliasing and cycles

Conversion is identity-aware: an array reachable by several paths — passed as two arguments, or both directly and as an attribute of a traversed object — becomes a single array on the host, so the kernel sees the aliasing the caller intended and no in-place update is lost on the way back. Reference cycles are handled rather than recursed into.

Set `use_cupy` explicitly to force conversion on or off. By default it is decided per call: conversion happens when the active backend is CuPy, or when a CuPy array is passed in.
