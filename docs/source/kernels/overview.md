# Porting kernels to the GPU

Array-level code (`xp.sum`, `xp.fft`, slicing, broadcasting) runs on the GPU as
soon as the backend is CuPy. Many scientific codes, however, spend their time in
*kernels*: loops over particles or grid cells written in Python and compiled
for the CPU, for example with [Pyccel](https://github.com/pyccel/pyccel) or
Numba. Those kernels take NumPy arrays and cannot run on device memory.

CuNumpy's kernel layer lets such a code base move to the GPU **one kernel at a
time**, with the CPU version kept as the reference and the call sites
unchanged.

## The building blocks

| Class | What it does | Guide |
| --- | --- | --- |
| `PyccelKernel` | calls a host kernel with CuPy arrays by copying them to the host and back | [Host kernels with GPU data](pyccel-kernel.md) |
| `CudaKernel` | wraps a CUDA C kernel, checks every call against its signature | [Writing CUDA kernels](cuda-kernel.md) |
| `Kernel` | a host kernel plus its CUDA kernel; calls the one matching the backend | [Pairing host and CUDA kernels](dispatch.md) |
| `KernelCatalog` | all `Kernel`s of a package, found by folder convention | [Pairing host and CUDA kernels](dispatch.md) |
| `CudaArguments`, `CudaStruct`, `CudaStructArguments` | pass a group of arrays and scalars as one argument | [Kernel arguments and structs](arguments.md) |
| `DeviceMirror`, `cunumpy/atomic.cuh` | scatter-add into a buffer owned by a host library | [Accumulation kernels](accumulation.md) |
| `cunumpy.kernel_testing` | check that host and CUDA kernels compute the same | [Testing kernels](testing.md) |

## Which one do I need?

* **"I just want my existing code to run with `CUNUMPY_BACKEND=cupy`."** Wrap the
  host kernels in `PyccelKernel`. Everything works, but every kernel call
  copies its arrays to the host and back. This is a correct starting point, not
  a fast one.
* **"I have one hot kernel and want it fast on the GPU."** Write a CUDA version
  and launch it with `CudaKernel`.
* **"I have dozens of kernels and will port them over months."** Put each
  host/CUDA pair in a `Kernel`, collect them in a `KernelCatalog`, and test each
  pair with `assert_kernels_agree`. Unported kernels either raise or fall back
  to the host version.
* **"My kernels take ten arrays each."** Group them in a `CudaStruct`, or a
  `CudaStructArguments` class next to the host argument class.

## A typical porting workflow

1. **Run everything on the host.** The code base uses `import cunumpy as xp`
   and the NumPy backend; kernels are plain host functions (Pyccel, Numba, or
   pure Python).
2. **Make the GPU backend work, slowly.** Collect the kernels in a
   `KernelCatalog` with `missing_cuda="fallback"`. On CuPy every kernel now runs
   through `PyccelKernel`, with host copies. Results must match the CPU run.
   `count_transfers()` shows how many copies each step makes; `catalog.summary()`
   shows how many kernels are left (`CUDA kernels: 0 of 42`).
3. **Port the most expensive kernel.** Write `name/name_cuda.cu` next to the
   host kernel, mirroring its argument list. The catalog picks it up
   automatically. A parity test compares it with the host version.
4. **Repeat**, ordered by a profile (see [Timing and
   profiling](../guides/profiling.md)), until the kernels in the time loop are
   ported. `assert_no_transfers()` around a time step verifies that no fallback
   or copy is left.
5. **Switch to `missing_cuda="raise"`** so a kernel added later without CUDA
   version fails loudly instead of silently copying.

The [particle pusher example](../examples/particle-pusher.md) walks through
these steps on a small code.

## Design rules the kernel layer follows

Knowing these makes the error messages predictable:

* **CUDA kernels never copy.** `CudaKernel` accepts only device arrays for
  pointer parameters. A NumPy array raises `TypeError` instead of being copied
  to the device behind your back.
* **Signatures are checked.** CuPy's `RawKernel` reads arguments with the size
  declared in C and never checks them, so a Python `int` passed as `double`
  arrives as garbage. `CudaKernel` parses the signature and casts or rejects
  every argument.
* **Copies of the fallback path are visible.** `PyccelKernel` conversions and
  `Kernel` fallbacks are recorded by `count_transfers()` and warned about once.
* **The CUDA kernel mirrors the host kernel.** Same name, same argument order,
  plus a launch shape (`n_threads`). The call site does not branch on the
  backend.
