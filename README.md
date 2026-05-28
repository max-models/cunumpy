# CuNumpy

Simple wrapper for numpy and cupy. Replace `import numpy as np` with `import cunumpy as xp`.

# Install

Create and activate python environment

```
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the code and requirements with pip

```
pip install -e .
```

Example usage:

```
export ARRAY_BACKEND=cupy
```

```python
import cunumpy as xp
arr = xp.array([1,2])

print(type(arr))
print(xp.__version__)

# Convert to NumPy
arr_np = xp.to_numpy(arr)

# Convert to active backend
arr_xp = xp.to_cunumpy(arr)

# Inspect backend
print(xp.get_backend(arr))
print(xp.is_gpu(arr))
print(xp.is_cpu(arr))
```

# Build docs


```
make html
cd ../
open docs/_build/html/index.html
```
