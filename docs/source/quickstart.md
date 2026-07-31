# Quickstart

## Installation

Install `cunumpy` directly from PyPI:

```bash
pip install cunumpy
```

## Basic Usage

Replace `import numpy as np` with `import cunumpy as xp`. By default, it will use NumPy if CuPy is not installed or configured.

```python
import cunumpy as xp

# Create an array (automatically uses the active backend)
arr = xp.array([1, 2, 3])

print(f"Array type: {type(arr)}")
print(f"Active backend: {xp.get_backend(arr)}")
```

## Backend Control

You can control the backend using environment variables:

```bash
export ARRAY_BACKEND=cupy
```

Or programmatically:

```python
import cunumpy as xp

# Globally set backend
xp.set_backend("cupy")

# Scoped backend switching
with xp.use_backend("numpy"):
    arr_cpu = xp.zeros(10)
    print(xp.is_cpu(arr_cpu))  # True

# Explicit conversion
arr_np = xp.to_numpy(arr)
arr_cp = xp.to_cupy(arr)
```

## Synchronization

When using the GPU, it's important to synchronize for accurate timing:

```python
import time
import cunumpy as xp

xp.set_backend("cupy")
a = xp.random.rand(1000, 1000)

start = time.time()
b = xp.dot(a, a)
xp.synchronize()  # Wait for GPU
end = time.time()

print(f"Elapsed: {end - start:.4f}s")
```
