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
