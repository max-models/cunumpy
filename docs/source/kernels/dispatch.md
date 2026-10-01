# Pairing host and CUDA kernels

`Kernel` holds a host kernel and, once it is written, its CUDA counterpart, and
calls the one matching the active backend. `KernelCatalog` collects all kernels
of a package. Together they let a code base port its kernels incrementally while
every call site stays the same.

## `Kernel`

```python
import cunumpy as xp

AXPY = r"""
extern "C" __global__
void axpy(double a, const double* x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""


def axpy_host(a, x, y, n):
    for i in range(n):
        y[i] += a * x[i]


axpy = xp.Kernel(axpy_host, xp.CudaKernel(AXPY, "axpy"), name="axpy")

axpy(2.0, x, y, x.size, n_threads=x.size)
```

* On the **NumPy backend** the host kernel is called with the positional
  arguments; `n_threads`, `grid`, `block`, `shared_mem` and `stream` are
  ignored.
* On the **CuPy backend** the CUDA kernel is launched; `n_threads` (or `grid`)
  is required.
* A plain host function is wrapped in a [`PyccelKernel`](pyccel-kernel.md);
  pass `host_options={"outputs": (2,)}` to configure that wrapper, or pass a
  `PyccelKernel` you built yourself.
* Kernels take positional arguments only. [`KernelArguments`](arguments.md)
  objects are resolved per backend.
* `kernel.compile()` compiles the CUDA kernel now; `kernel.has_cuda` tells
  whether there is one.

### Kernels without CUDA version

`Kernel(host, None)` is a kernel not yet ported. On the CuPy backend:

* `missing_cuda="raise"` (default) raises `NotImplementedError`, naming
  `cuda_path` if given, the file where the CUDA kernel is expected.
* `missing_cuda="fallback"` calls the host kernel through `PyccelKernel`,
  copying the arrays to the host and back at every call. A `RuntimeWarning` is
  emitted once per kernel, and `count_transfers()` records a `fallback` event
  per call.

For the fallback to find device arrays inside your own objects, the wrapper
needs to know them: `host_options={"object_modules": ("my_package.",)}`.

`kernel.get_kernel()` returns the kernel for the active backend. Calling it
once at setup surfaces a missing CUDA kernel immediately instead of in the
middle of a run.

## `KernelCatalog`: one folder per kernel

`KernelCatalog.from_package()` finds kernels by convention:

```text
my_sim/kernels/
├── __init__.py               # catalog = xp.KernelCatalog.from_package(__name__)
├── push/
│   ├── push_kernels.py       # def push(...): ...        host kernel
│   └── push_cuda.cu          # __global__ void push(...)  CUDA kernel
├── deposit/
│   ├── deposit_kernels.py
│   └── deposit_cuda.cu
└── sort/
    └── sort_kernels.py       # not ported yet
```

```python
# my_sim/kernels/__init__.py
import cunumpy as xp

catalog = xp.KernelCatalog.from_package(__name__, missing_cuda="fallback")
```

```python
# anywhere in the code
from my_sim.kernels import catalog

catalog["push"](markers, dt, n_markers, n_threads=n_markers)
```

For every subfolder `<name>` containing `<name>_kernels.py`, the function
`<name>` in that module is the host kernel, and `<name>_cuda.cu` in the same
folder, if present, provides the CUDA kernel `__global__ void <name>(...)`.
Other `__global__` functions in that file are ignored by the catalog (load them
with `CudaKernel.all_from_file`). The suffixes are configurable with
`host_suffix` and `cuda_suffix`.

Options:

* `missing_cuda`: passed to every `Kernel`.
* `host_options`: `PyccelKernel` options for all host kernels, or a function of
  the kernel name for per-kernel output declarations:

  ```python
  OUTPUTS = {"push": (0,), "deposit": (2,)}
  catalog = xp.KernelCatalog.from_package(
      __name__,
      host_options=lambda name: {"outputs": OUTPUTS.get(name)},
  )
  ```

* `include_dirs`: extra include directories for the CUDA kernels. By default
  the source root of the top-level package is one, so a kernel can write
  `#include "my_sim/kernels/common.cuh"`. Each kernel's own folder is always
  searched.
* Other keyword arguments (`block_size`, `structs`, `options`, ...) go to every
  `CudaKernel.from_file()`.

A catalog is a read-only mapping: `catalog["push"]`, `"push" in catalog`,
`len(catalog)`, iteration over names. `KernelCatalog()` plus
`catalog.register(kernel)` builds one by hand.

## Porting status and setup

```python
print(catalog.summary())
# CUDA kernels: 2 of 3 (missing: sort)

catalog.without_cuda    # ['sort']
catalog.with_cuda       # ['deposit', 'push']
```

`summary()` is handy for a `--status` command line flag or the start-up log.

At setup, on the GPU backend, compile everything at once:

```python
if xp.cupy_backend:
    catalog.compile_all(jobs=8)   # threads; jobs=None uses all CPUs
```

All kernels are compiled even if one fails; the first error is raised
afterwards. On later runs CuPy loads the binaries from its disk cache, so this
is fast.

## Recommended project layout

* One folder per kernel, the host kernel and its CUDA port side by side, so a
  reviewer sees both at once.
* Shared device helpers (`__device__` functions) in `.cuh` headers next to the
  kernels or in a `common/` folder, included with quotes. Test them with
  `device_function_kernel` ([Testing kernels](testing.md)).
* Generated struct headers (`write_cuda_header`) committed next to the kernels,
  with a test that they are up to date.
* One parametrised parity test over `catalog.parity_cases()`.
* `missing_cuda="fallback"` while porting, `"raise"` once the time loop is fully
  ported.
