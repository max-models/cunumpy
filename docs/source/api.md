# API overview

CuNumpy exports the active NumPy-like namespace and a set of helpers for
backend selection, array inspection and conversion, hardware control, and
host-only kernels. Most examples use `import cunumpy as xp`. This page shows how
the package is laid out; every function and class is documented in the [API
reference](reference/index.md), generated from the docstrings, and the guides
explain how to use them together.

## NumPy-like namespace

```python
import cunumpy as xp

values = xp.arange(5)
total = xp.sum(values)
```

At runtime, NumPy-like attributes such as `array`, `sum`, `fft`, and `linalg`
are those of the currently selected `array-api-compat` NumPy or CuPy module.
They are copied into the `cunumpy` namespace and replaced when the backend
changes, so `xp.sum` costs the same as `numpy.sum` (a switch between NumPy and
CuPy takes some tens of microseconds; avoid switching inside a hot loop).
CuNumpy does not wrap every operation individually. The available operations
and some details can therefore vary with the installed NumPy and CuPy
versions. In normal use, access those operations through the top-level
`cunumpy` namespace, commonly imported as `xp`.

For an explanation of what the compatibility module does and why CuNumpy
uses it, read [Why CuNumpy uses `array-api-compat`](array-api-compat.md).

NumPy and CuPy are not interchangeable for every function or object. A
function that needs to follow an input array's location should use
`get_array_module(array)` instead of assuming the global backend matches it.

## Module layout

The top level of `cunumpy` is the NumPy (or CuPy) namespace plus the functions
that select the backend and convert arrays. Everything else is in a submodule,
imported with `cunumpy` (`xp.kernels.CudaKernel`, `xp.rng.random_streams`, ...).
The submodules are named so that they do not hide a NumPy name (`rng`, not
`random`):

| Submodule | Backends | Contents |
|---|---|---|
| [`cunumpy`](reference/top-level.md) | both | NumPy/CuPy namespace, backend selection, array inspection and conversion, `as_device_array`, `to_host_async`, `synchronize`, `host_call`, `evaluate_on_host`, `setup_on_host`, `scipy`, `require_version` |
| [`cunumpy.kernels`](reference/kernels.md) | both | `Kernel`, `KernelCatalog`, `PyccelKernel`, `CudaKernel`, `CudaKernelVariants`, `MetalKernel`, `metal_available`, host implementations, `as_kernel_array`, `kernel_output`, `fuse` |
| [`cunumpy.arguments`](reference/arguments.md) | CUDA only | `CudaArguments`, `CudaStruct`, `CudaStructArguments`, `CudaStructValue`, `write_cuda_header` |
| [`cunumpy.cuda`](reference/cuda.md) | CUDA only | device selection and memory, `stream`, streams/events, `pin_memory`, debug mode, CUDA headers and source tools (`cuda_include_dir`, `parse_cuda_signature`) |
| [`cunumpy.rng`](reference/rng.md) | both | `random_streams`, `get_rng`, `philox_*` |
| [`cunumpy.algorithms`](reference/algorithms.md) | both | `morton_*`, `sort_by_key`, `segment_sum`, `compact_by_mask` |
| [`cunumpy.mpi`](reference/mpi.md) | both | `mpi_buffer`, `MPIStaging`, CUDA-aware MPI detection, `synchronize_for_mpi`, and the MPI stand-ins of maybempi (`get_mpi`, `local_rank`, ...) |
| [`cunumpy.profiling`](reference/profiling.md) | both | `timed_region`, `nvtx_range`, `count_transfers`, `assert_no_transfers`, `TransferBudget` |
| [`cunumpy.memory`](reference/memory.md) | both | `HostStaging`, `HostCopy`, `DeviceMirror` |
| [`cunumpy.petsc`](reference/petsc.md) | both | `petsc_vec` |
| [`cunumpy.kernel_testing`](reference/kernel-testing.md) | both | pytest helpers for host/CUDA kernel pairs (not imported by `import cunumpy`) |

"Both" means the functions work on NumPy and CuPy arrays; the functions of
`cunumpy.cuda` do nothing (or return `None`/`0`) on the NumPy backend.

The CUDA headers shipped for hand-written kernels (`cunumpy/atomic.cuh`,
`cunumpy/reduce.cuh`, `cunumpy/random.cuh`, ...) have no Python API; they are
documented in [CUDA headers](kernels/cuda-headers.md).

Each name has one import path, through its submodule: `xp.kernels.CudaKernel`,
`xp.arguments.CudaStruct`, `xp.mpi.mpi_buffer`, `xp.rng.random_streams`. The top
level of `cunumpy` is the NumPy/CuPy namespace plus backend selection and array
conversion, so it never hides a NumPy or CuPy name (`xp.fuse` is CuPy's `fuse`;
cunumpy's is `xp.kernels.fuse`). Modules starting with `_` are private.
