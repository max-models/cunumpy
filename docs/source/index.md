# CuNumpy

CuNumpy provides a NumPy-like interface that can create and operate on NumPy
arrays on the CPU or CuPy arrays on an NVIDIA GPU. Select the backend once
for a program or temporarily for a section of code; use explicit conversion
helpers when data needs to cross between host and device.

```python
import cunumpy as xp

xp.set_backend("cupy")
values = xp.arange(1_000)
print(xp.get_backend())  # 'cupy' when CuPy/CUDA is functional
```

Install with `python -m pip install cunumpy`. GPU use also requires a CuPy
installation that matches the system's CUDA setup. CuNumpy falls back to
NumPy if CuPy cannot be used.

```{toctree}
:maxdepth: 2
:caption: Guides and reference:

quickstart
array-api-compat
api
pyodide
```
