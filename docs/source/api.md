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

## Compiled Kernels

### `PyccelKernel(kernel, use_cupy=None, object_modules=())`
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

Conversion is identity-aware: an array reachable by several paths — passed as two arguments, or both directly and as an attribute of a traversed object — becomes a single array on the host, so the kernel sees the aliasing the caller intended and no in-place update is lost on the way back. Reference cycles are handled rather than recursed into.

Set `use_cupy` explicitly to force conversion on or off. By default it is decided per call: conversion happens when the active backend is CuPy, or when a CuPy array is passed in.
