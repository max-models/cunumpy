# Pyodide

cuNumPy uses Pyodide's NumPy build with the existing `numpy` backend. No separate
backend or special wheel is needed. The supported path is CPU array operations
and ordinary Python callables wrapped in `PyccelKernel`. CuPy/CUDA execution and
native Pyccel compilation are outside this support.

## Installation and use

After [initializing Pyodide](https://pyodide.org/en/stable/usage/index.html), load
`micropip` and install cuNumPy. Its dependency resolver supplies NumPy and
`array-api-compat`; see [Pyodide package loading](https://pyodide.org/en/stable/usage/loading-packages.html).

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

NumPy is selected by default when `ARRAY_BACKEND` is unset. Leave it unset or set
it to `numpy` before importing cuNumPy. `to_numpy` and `to_cunumpy` preserve
existing NumPy arrays, including views. `synchronize` and `set_device` are no-ops
on this backend.

`PyccelKernel` does not import Pyccel or compile functions. With NumPy arguments
and its default settings, it calls your function directly: argument identity,
in-place mutation, return values, and exceptions retain normal Python semantics.
The `outputs` option controls GPU copy-back only; it does not prevent a Python
kernel from mutating CPU arguments. Keep `use_cupy` unset or `False` in Pyodide.

Applications such as Struphy must choose Python-source kernels and ensure their
imports (including any Pyccel decorators) work without a native compiler. Wrapping
a function does not make its own dependencies Pyodide-compatible.

## Testing the built wheel

From the repository root, with Python and Node.js 22 installed:

```bash
python -m pip install build '.[test]'
python -m build --wheel
python -m pytest -q tests/portable
npm ci --prefix tests/pyodide
node tests/pyodide/run.mjs dist/cunumpy-*-py3-none-any.whl
```

Pass exactly one wheel (an explicit path if `dist` contains multiple versions).
Network access is required to download the Pyodide packages and wheel dependencies.

CI pins Pyodide 314.0.6 and executes the portable suite in its actual WebAssembly
Python runtime under Node.js. The runner installs the built wheel with `micropip`,
including dependency resolution, without exposing the source tree. It checks the
default NumPy backend, installed version metadata, and rejects attempts to import
CuPy or Pyccel during import and CPU execution. The suite covers array operations,
conversions, backend context restoration after normal and exceptional exits, and
Python-kernel return values, mutation, aliasing, and exceptions.

This is a runtime compatibility check; browser-specific page integration,
workers, and hosting configuration are not exercised by this CI job.

The `test` extra installs compiler-free test dependencies. Native compiled-kernel
tests use the separate `test-compiled` extra and require a working compiler.
