# Best practices

A condensed checklist. Each item links to the guide with the reasoning.

## Structure

* Import as `import cunumpy as xp` and always call `xp.<name>(...)`; never
  `from cunumpy import <array function>`. ([Backend-agnostic
  code](guides/portable-code.md))
* Choose the backend once, in the entry point, with `ARRAY_BACKEND` or
  `set_backend()`. Library code never calls `set_backend()`. ([Choosing a
  backend](guides/backends.md))
* Read back `xp.get_backend()` after requesting CuPy; log it with
  `xp.device_count()` and `xp.__version__`.
* Use `use_backend()` for scoped switches (tests, CPU reference computations),
  never from several threads at once.
* Functions that receive arrays follow them with `get_array_module()`; functions
  that combine arrays check them with `assert_same_backend()`.

## Data

* Create arrays with `xp.*` so they are born on the right device. Use `numpy`
  directly only for host-only data and dtypes.
* Convert explicitly, at boundaries: `to_cunumpy()` on input, `to_numpy()` for
  output, plotting and host-only libraries. ([Data
  movement](guides/data-movement.md))
* Keep the time loop free of transfers and of host synchronization (`float()`,
  `.item()`, `print`, `if` on device values). Verify with
  `assert_no_transfers()` in a test.
* Give dtypes explicitly for arrays that go to kernels.
* Generate random test data on the host with a seed, then convert; NumPy and
  CuPy generators differ.

## Kernels

* Start with `KernelCatalog.from_package(..., missing_cuda="fallback")`, port
  kernels by profile order, switch to `"raise"` when done. ([Porting
  kernels](kernels/overview.md))
* Keep the CUDA kernel's argument list identical to the host kernel's; put both
  in one folder, and test it with `catalog.check_signatures()`.
* Compile Pyccel host kernels ahead of time with the `pyccel` command, or at run
  time with `from_package(..., compile_host=...)`, and keep a NumPy version as
  `host_fallback`.
* Use `dispatch="arrays"` if host arrays reach kernels while CuPy is active.
* Ship `.cu`/`.cuh` files as package data.
* Declare `outputs` on host kernels so the fallback copies back only what was
  written, and never forget an argument that is written.
* Pass device arrays to `CudaKernel`; build argument objects once with
  `as_device_array()`. ([Kernel arguments](kernels/arguments.md))
* Use `long long` indices, `CUNUMPY_THREAD_1D` guards, `const` on inputs, and
  `cunumpy_atomic_add` for scatter writes.
* Use array views (`Array2D<double>`) instead of hand-computed offsets for
  multi-dimensional data.
* Generate structs from the host argument class (`CudaStruct.from_signature`)
  and test that committed headers are up to date.
* Compile at setup (`catalog.compile_all(jobs=...)`).
* Keep signature checks on; disable them (`check_signature=False`) only for
  tiny kernels in hot loops after they are tested.

## Verification

* Parametrize tests with `BACKENDS` or the `backend` fixture; they run
  everywhere and use the GPU where there is one. ([Testing
  kernels](kernels/testing.md))
* One `assert_kernels_agree` test over `catalog.parity_cases()`.
* `emulate_cuda_kernel` tests, so CPU-only CI checks the CUDA arithmetic.
* Seed `xp.random_streams` with `(seed, rank)` and draw only from it.
* Debug crashes with `CUNUMPY_CUDA_DEBUG=1`, then `compute-sanitizer`.
  ([Debugging](kernels/debugging.md))
* Time with `timed_region()`, profile with `nvtx_range()` and `nsys`.
  ([Profiling](guides/profiling.md))

## MPI

* `set_backend("cupy")`, `bind_local_device()`, then `from mpi4py import MPI`,
  then `require_cuda_aware_mpi()`. ([MPI](guides/mpi.md))
* `synchronize_for_mpi(*buffers)` before every MPI call with device buffers.
