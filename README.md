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

# Pyodide

cuNumPy supports Pyodide with the NumPy backend. In an initialized Pyodide
JavaScript runtime, install the package and run a Python-source kernel:

```javascript
await pyodide.loadPackage("micropip");
await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("cunumpy")

    import cunumpy as xp
    xp.set_backend("numpy")

    def scale(values, factor):
        values[:] *= factor
        return values

    values = xp.array([1.0, 2.0, 3.0])
    result = xp.PyccelKernel(scale)(values, 2.0)
    assert result is values
    print(xp.to_numpy(result))  # [2. 4. 6.]
`);
```

NumPy is the default backend when `ARRAY_BACKEND` is unset. CuPy/CUDA and
native Pyccel compilation are not supported in Pyodide. Despite its name,
`PyccelKernel` accepts ordinary Python callables and neither imports Pyccel nor
compiles code. Applications must supply Python-source kernels with
Pyodide-compatible imports.

CI tests the built wheel in Pyodide's WebAssembly runtime under Node.js, including
array operations, conversions, mutation/aliasing, backend context restoration,
and Python kernels, while rejecting CuPy or Pyccel imports. This does not test
browser-specific integration such as page loading or workers. See the
[Pyodide guide](docs/source/pyodide.md) for details and local test commands.

# Development tests

```bash
pip install -e '.[test]'
pytest tests/portable
```

The `test` extra is compiler-free. For compiled-kernel tests, install
`'.[test-compiled]'` and a working native compiler, then run `pytest`.
The `dev` extra includes these compiled-test dependencies as before.

# Build docs


```
make html
cd ../
open docs/_build/html/index.html
```
