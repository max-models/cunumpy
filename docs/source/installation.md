# Installation

## Install the package

```bash
python -m pip install cunumpy
```

This installs CuNumpy with its two dependencies, NumPy and `array-api-compat`.
That is all that is needed for the CPU (NumPy) backend, and CuNumpy runs on any
platform NumPy runs on, including [Pyodide](pyodide.md). CuNumpy requires
Python 3.10 or newer.

## Add GPU support

The GPU backend uses [CuPy](https://cupy.dev). CuNumpy does not install CuPy or
CUDA itself, because the right CuPy package depends on the CUDA version of the
machine. Install the CuPy wheel that matches your CUDA toolkit or driver, for
example:

```bash
python -m pip install cupy-cuda12x   # CUDA 12.x
```

On HPC clusters, load the site's CUDA module first and check the CuPy
installation guide for the matching package name. Then verify that CuNumpy can
use it:

```python
import cunumpy as xp

print("CuPy usable:", xp.cupy_available())
print("visible GPUs:", xp.device_count())

xp.set_backend("cupy")
print("active backend:", xp.get_backend())  # 'cupy' if the GPU works
```

If `cupy_available()` is `False`, CuNumpy quietly falls back to NumPy when CuPy
is requested. See [Troubleshooting](troubleshooting.md) for the usual causes.

## Optional extras

| Extra | Installs | Use it for |
| --- | --- | --- |
| `cunumpy[test]` | `pytest`, `coverage` | running the test suite, using `cunumpy.testing` |
| `cunumpy[test-compiled]` | the above plus `pyccel` | tests that compile host kernels with Pyccel |
| `cunumpy[docs]` | Sphinx, MyST, the book theme | building this documentation |
| `cunumpy[dev]` | all of the above plus formatters | developing CuNumpy itself |

For MPI programs, install `mpi4py` against your MPI library as usual. For GPU
MPI the library must be CUDA-aware; see [Multi-GPU programs with
MPI](guides/mpi.md).

## Install from source

```bash
git clone https://github.com/max-models/cunumpy.git
cd cunumpy
python -m pip install -e ".[dev]"
python -m pytest tests/unit
```

Tests that need a GPU are skipped automatically where CuPy is not functional.

## Environment variables

| Variable | Effect |
| --- | --- |
| `ARRAY_BACKEND=cupy` | start with the CuPy backend instead of NumPy (read once, at import) |
| `CUNUMPY_CUDA_DEBUG=1` | enable [CUDA debug mode](kernels/debugging.md) for all kernels |

MPI launchers also export node-local rank variables (`OMPI_COMM_WORLD_LOCAL_RANK`,
`SLURM_LOCALID`, ...), which `xp.local_rank()` reads to pick a GPU per process.

## Build the documentation

```bash
python -m pip install ".[docs]"
cd docs
make html   # output in docs/build/html
```
