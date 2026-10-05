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
print("visible GPUs:", xp.cuda.device_count())

xp.set_backend("cupy")
print("active backend:", xp.get_backend())  # 'cupy' if the GPU works
```

If `cupy_available()` is `False`, CuNumpy quietly falls back to NumPy when CuPy
is requested. See [Troubleshooting](troubleshooting.md) for the usual causes.

## Optional extras

| Extra | Installs | Use it for |
| --- | --- | --- |
| `cunumpy[test]` | `pytest`, `coverage` | running the test suite, using `cunumpy.kernel_testing` |
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
| `CUNUMPY_BACKEND=cupy` | start with the CuPy backend instead of NumPy (read once, at import) |
| `CUNUMPY_CUDA_DEBUG=1` | enable [CUDA debug mode](kernels/debugging.md) for all kernels |
| `CUNUMPY_HOST_KERNEL_IMPLEMENTATION=numpy` | choose the host kernel implementation (read at import) |
| `CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION=cuda` | require CUDA for device kernel dispatch (read at import); unset allows the kernel's configured fallback |
| `CUNUMPY_MPI=1` / `0` | require MPI / use serial MPI regardless of launcher detection |
| `CUNUMPY_FAKE_CUPY=1` | install the strict CPU stand-in for CuPy for tests |
| `CUNUMPY_REQUIRE_CUDA=1` | require a real usable GPU when starting the test suite (CI guard) |

Use `CUNUMPY_BACKEND` in job scripts; the former `ARRAY_BACKEND` setting is no
longer read. Standard toolchain/device variables such as `CXX` and
`CUDA_VISIBLE_DEVICES` keep their standard meanings.

Use `CUNUMPY_HOST_KERNEL_IMPLEMENTATION` instead of the former
`CUNUMPY_KERNEL_IMPLEMENTATION`, which is no longer read. This selects host
implementations only; CUDA dispatch is unaffected. The Python functions
`set_host_kernel_implementation`, `get_host_kernel_implementation`, and
`use_host_kernel_implementation` provide runtime selection under `xp.kernels`.

MPI launchers also export node-local rank variables (`OMPI_COMM_WORLD_LOCAL_RANK`,
`SLURM_LOCALID`, ...), which `xp.mpi.local_rank()` reads to pick a GPU per process.

## Build the documentation

```bash
python -m pip install ".[docs]"
cd docs
make html   # output in docs/build/html
```
