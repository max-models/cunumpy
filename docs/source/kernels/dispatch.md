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

## Compiled Pyccel host kernels

By default the host kernel is the Python function itself, which is fine for
NumPy-vectorized code but slow for the loops of a Pyccel kernel. With
`compile_host`, each host kernel is compiled on its first call and cached on
disk (`cunumpy.pyccel.compile_cached`), and `host_fallback` gives the version
to use when compilation is not possible (no Pyccel, no compiler):

```python
import cunumpy as xp
from cunumpy.pyccel import compile_cached

from my_sim.kernels.numpy_versions import NUMPY_VERSIONS  # {"push": push_numpy, ...}

catalog = xp.KernelCatalog.from_package(
    __name__,
    host_suffix="_pyccel",          # push/push_pyccel.py next to push/push_cuda.cu
    compile_host=compile_cached,
    host_fallback=NUMPY_VERSIONS,
)
```

The build is keyed on the source and the Pyccel version, so the first run
after an edit or an upgrade compiles again; later runs and other MPI ranks load
the cached build. A host kernel is a `cunumpy.pyccel.CompiledHostKernel`:
`catalog["push"].host_kernel.kernel.compiled` reports whether the compiled
version is available.

## Choosing the kernel by where the arrays are

`Kernel` picks the CUDA kernel on the CuPy backend. Real codes also hand host
arrays to kernels while CuPy is active: a diagnostic on a copy, a buffer staged
for MPI, a path that has no GPU version yet. With `dispatch="arrays"` such calls
run the host kernel:

```python
catalog = xp.KernelCatalog.from_package(__name__, dispatch="arrays")

catalog["gather"](positions, field, result, n_threads=n)  # CUDA if positions are CuPy
catalog["gather"](host_positions, host_field, host_result)  # host kernel, also on CuPy
```

The CUDA kernel runs if any top-level argument is on the GPU: a CuPy array, or
a device-only argument object (`CudaArguments`, a struct value). A
`KernelArguments` object has both forms and does not decide on its own.

## Same parameters on both sides

The call site is the same for both kernels only if they take the same
parameters in the same order. Check it once, in a test:

```python
def test_kernel_signatures():
    catalog.check_signatures()
```

It compares the parameter names of each host function (for a compiled host
kernel, its Python source, or the `__pyccel__/<module>.pyi` stub that pyccel
writes next to a compiled extension module) with the parsed `__global__`
signature, and lists every kernel that differs, e.g.
`kernel 'push': the host kernel takes (x, v, dt, n), the CUDA kernel (x, v, n, dt)`.

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

`from_package()` also warns about kernel names that pyccel's Fortran backend
cannot compile: the wrapper module `bind_c_<name>_kernels` must fit Fortran's
63-character limit, so a kernel name has at most 48 characters
(`check_name_length=False` silences it, e.g. for the C backend).

## Launch size from the arguments

When the number of threads is a property of an argument (the marker count of
a particle argument object), set it once instead of at every call site:

```python
catalog["push"].cuda_kernel.n_threads_from = lambda args: args[0].n_markers
push(args_markers, dt)  # no n_threads needed, on either backend
```

An explicit `n_threads` or `grid` still wins.

## Recommended project layout

* One folder per kernel, the host kernel and its CUDA port side by side, so a
  reviewer sees both at once.
* Shared device helpers (`__device__` functions) in `.cuh` headers next to the
  kernels or in a `common/` folder, included with quotes. Test them with
  `device_function_kernel` ([Testing kernels](testing.md)).
* Generated struct headers (`write_cuda_header`) committed next to the kernels,
  with a test that they are up to date.
* One parametrised parity test over `catalog.parity_cases()`, and
  `catalog.check_signatures()` in a test.
* Declare the `.cu` and `.cuh` files as package data (e.g.
  `[tool.setuptools.package-data] my_sim = ["kernels/*/*_cuda.cu",
  "kernels/*.cuh"]`); otherwise wheels contain the host kernels but no CUDA
  sources, and every kernel looks unported after `pip install`.
* `missing_cuda="fallback"` while porting, `"raise"` once the time loop is fully
  ported.
