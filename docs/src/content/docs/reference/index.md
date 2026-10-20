---
title: "API reference"
description: "One page per module, generated from the docstrings."
---

Generated from the docstrings. Each name has one import path, through the
module it is listed under (`xp.kernels.CudaKernel`, `xp.rng.random_streams`).
For an overview of the module layout and the NumPy-like namespace, see the
[API overview](/cunumpy/reference/overview/).

- [Top level: `cunumpy`](/cunumpy/api/cunumpy/): The NumPy/CuPy namespace (see the [API overview](/cunumpy/reference/overview/)) plus backend selection, array inspection and conversion, and host execution helpers.
- [Kernels: `cunumpy.kernels`](/cunumpy/api/cunumpy/kernels/): Kernels that run on either backend: dispatch, host implementations, CUDA and Metal kernels, fusion.
- [Kernel arguments: `cunumpy.arguments`](/cunumpy/api/cunumpy/arguments/): Structs and argument objects passed to CUDA kernels.
- [CUDA runtime: `cunumpy.cuda`](/cunumpy/api/cunumpy/cuda/): Devices, memory, streams, events, debug mode and CUDA source tools. These do nothing on the NumPy backend.
- [Random numbers: `cunumpy.rng`](/cunumpy/api/cunumpy/rng/): Reproducible random streams and counter-based Philox generators.
- [Algorithms: `cunumpy.algorithms`](/cunumpy/api/cunumpy/algorithms/): Morton keys, sorting by key, segment sums and compaction.
- [Optimization: `cunumpy.optimize`](/cunumpy/api/cunumpy/optimize/): Batched Newton and secant iterations on the device, and SciPy's `fsolve`, `root` and `minimize` with callbacks on the device.
- [Integration: `cunumpy.integrate`](/cunumpy/api/cunumpy/integrate/): SciPy's `quad` and `odeint` with callbacks on the device.
- [MPI: `cunumpy.mpi`](/cunumpy/api/cunumpy/mpi/): MPI buffers, CUDA-aware MPI detection and synchronization. `cunumpy.mpi` also re-exports the MPI stand-ins of [maybempi](https://max-models.github.io/maybempi/), documented there: `get_mpi`, `is_serial`, `launched_under_mpi`, `local_rank`, `OVERRIDE_VARIABLE`, `SerialMPI`, `SerialComm`, `SerialRequest` and `SerialStatus`.
- [Profiling: `cunumpy.profiling`](/cunumpy/api/cunumpy/profiling/): Timed regions, NVTX ranges and host/device transfer counting.
- [Memory: `cunumpy.memory`](/cunumpy/api/cunumpy/memory/): Pinned host staging, asynchronous host copies and device mirrors.
- [PETSc: `cunumpy.petsc`](/cunumpy/api/cunumpy/petsc/): PETSc vectors that share memory with arrays.
- [Kernel testing: `cunumpy.kernel_testing`](/cunumpy/api/cunumpy/kernel_testing/): Pytest helpers for host/CUDA kernel pairs. Not imported by `import cunumpy`.
