# CuNumpy

CuNumpy lets one Python code base run on NumPy (CPU) or CuPy (NVIDIA GPU).
Replace `import numpy as np` with `import cunumpy as xp`, choose a backend, and
the array code you already have runs on the selected device. For codes whose
time is spent in compiled loops, CuNumpy adds a kernel layer to port those
kernels to CUDA one at a time, with the CPU version kept as the tested
reference.

```python
import cunumpy as xp

xp.set_backend("cupy")       # falls back to NumPy without a usable GPU
values = xp.arange(1_000)
print(xp.sum(values**2), xp.get_backend())
```

Install with `python -m pip install cunumpy`, plus a CuPy wheel matching your
CUDA version for GPU use. Start with the [Quickstart](quickstart.md).

## Where to go

* **New to CuNumpy**: [Installation](installation.md), then the
  [Quickstart](quickstart.md).
* **Running array code on CPU and GPU**: the *User guide* pages on backends,
  backend-agnostic code, data movement, devices, MPI, and profiling.
* **Porting compiled kernels to CUDA**: start at [Porting kernels to the
  GPU](kernels/overview.md).
* **Complete programs**: [Worked examples](examples/index.md).
* **Looking up a function**: the [API reference](api.md).
* **Using an AI assistant**: point it at [the guide for AI
  assistants](ai-assistants.md), which ships with the package as
  `cunumpy/LLM_GUIDE.md`.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

guides/backends
guides/portable-code
guides/data-movement
guides/gpu-devices
guides/mpi
guides/particle-codes
guides/solvers
guides/profiling
array-api-compat
```

```{toctree}
:maxdepth: 2
:caption: Porting kernels

kernels/overview
kernels/pyccel-kernel
kernels/cuda-kernel
kernels/dispatch
kernels/arguments
kernels/accumulation
kernels/debugging
kernels/testing
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 1
:caption: More

best-practices
troubleshooting
pyodide
ai-assistants
api
```
