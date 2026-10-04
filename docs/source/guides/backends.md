# Choosing a backend

CuNumpy has two backends: `"numpy"` (CPU) and `"cupy"` (NVIDIA GPU). The
*active backend* decides which library `xp.<function>` calls, and therefore
where newly created arrays live.

## Select the backend at start-up

The backend is NumPy unless the environment variable `ARRAY_BACKEND=cupy` is
set when CuNumpy is first imported:

```bash
ARRAY_BACKEND=cupy python simulate.py
```

This is the least intrusive option for scripts and batch jobs: the code does
not change, and a job script decides whether it runs on a GPU. The variable is
read once, at import; setting it later in `os.environ` has no effect.

To choose from inside the program, for example from a command-line flag, call
`set_backend()` once, early, before arrays are created:

```python
import argparse

import cunumpy as xp

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

xp.set_backend("cupy" if args.gpu else "numpy")
print(f"running on {xp.get_backend()}")
```

## Check what you actually got

Requesting CuPy is a request, not a guarantee. If CuPy is not installed, or
installed but not functional (no driver, no visible GPU, a CUDA version
mismatch), CuNumpy falls back to NumPy without raising. Always read the
effective backend back when it matters:

```python
xp.set_backend("cupy")
if xp.get_backend() != "cupy":
    raise SystemExit("this run needs a GPU, but CuPy is not usable")
```

`xp.cupy_available()` answers the question without changing the backend.
`xp.numpy_backend` and `xp.cupy_backend` are booleans for the active backend,
handy in conditionals:

```python
if xp.cupy_backend:
    xp.cuda.bind_local_device()
```

## Switch temporarily

`use_backend()` selects a backend for a block and restores the previous one on
exit, also if the block raises. It is the right tool for tests, notebooks, and
CPU reference computations inside a GPU program:

```python
xp.set_backend("cupy")

with xp.use_backend("numpy"):
    reference = xp.linspace(0.0, 1.0, 100)  # NumPy array

print(xp.get_backend())  # 'cupy' again
```

The selection is a single process-wide setting. Do not switch it from several
threads or async tasks at once: one task's `use_backend()` changes the backend
seen by all others.

## The active backend is not the array's backend

Changing the backend never moves existing arrays. A program can have NumPy
active while CuPy arrays are alive, and vice versa:

```python
xp.set_backend("cupy")
on_gpu = xp.arange(4)

xp.set_backend("numpy")
print(xp.get_backend())  # 'numpy': what xp.* creates now
print(xp.get_array_backend(on_gpu))  # 'cupy': where this array lives
```

Two families of functions answer the two questions:

| Question | Functions |
| --- | --- |
| Which library does `xp.*` use now? | `get_backend()`, `numpy_backend`, `cupy_backend` |
| Where does this array live? | `get_array_backend(a)`, `is_cpu(a)`, `is_gpu(a)`, `get_array_module(a)` |
| Do these arrays live in the same place? | `same_backend(*arrays)`, `assert_same_backend(*arrays)` |

Code that creates arrays uses the first family; code that receives arrays
uses the second (see [Writing backend-agnostic code](portable-code.md)).

## Recommended patterns

* **Choose once, at the top.** Select the backend in the entry point (the
  `main()` of a script, a configuration loader), not inside library functions.
  Library code should not call `set_backend()`.
* **Library code follows its inputs.** Use `get_array_module(array)` or check
  with `assert_same_backend()` instead of reading the global setting.
* **Use `use_backend()` for scoped work.** It is exception-safe and makes the
  scope obvious; a bare `set_backend()` in the middle of a function changes
  the state for everything that runs afterwards.
* **Log the effective backend.** Print `xp.get_backend()`, `xp.cuda.device_count()`
  and `xp.__version__` in the run's output so results can be traced to the
  hardware they ran on.

## What is behind `xp`

`xp.<name>` resolves `name` on `array_api_compat.numpy` or
`array_api_compat.cupy`, depending on the active backend. These are thin
compatibility layers over the real libraries, so the arrays are plain
`numpy.ndarray` and `cupy.ndarray` objects. [Why CuNumpy uses
`array-api-compat`](../array-api-compat.md) explains the layer in more detail.
