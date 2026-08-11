# CuNumpy

Simple wrapper for numpy and cupy. Replace `import numpy as np` with `import cunumpy as xp`.

# Install

```bash
pip install cunumpy
```

Example usage:

```
export ARRAY_BACKEND=cupy
```

```python
import cunumpy as xp

xp.set_backend("cupy")

arr = xp.array([1, 2])

print(f"{type(arr) = }")
print(f"{xp.__version__ = }")

# Convert to NumPy
arr_np = xp.to_numpy(arr)

# Convert to active backend
arr_xp = xp.to_cunumpy(arr)

# Inspect backend
print(f"{xp.get_backend(arr) = }")
print(f"{xp.is_gpu(arr) = }")
print(f"{xp.is_cpu(arr) = }")

# Temporarily switch backend
with xp.use_backend("numpy"):
    # This code runs on CPU even if ARRAY_BACKEND=cupy
    arr_cpu = xp.zeros(100)

# Set backend globally
xp.set_backend("cupy")

# Synchronize GPU operations (no-op on CPU)
xp.synchronize()
```

Output:

```
type(arr) = <class 'cupy.ndarray'>
xp.__version__ = '0.1.4'
xp.get_backend(arr) = 'cupy'
xp.is_gpu(arr) = True
xp.is_cpu(arr) = False
```

# Build docs


```
make html
cd ../
open docs/_build/html/index.html
```
