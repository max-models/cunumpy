# Troubleshooting

## Backend and installation

**`set_backend("cupy")` but `get_backend()` returns `"numpy"`.**
CuPy is missing or not functional, and CuNumpy fell back to NumPy. Check
`xp.cupy_available()`, then try `python -c "import cupy; cupy.arange(3)"` to see
the real error. Typical causes: no CuPy installed, a CuPy wheel for a different
CUDA major version, no GPU visible (`CUDA_VISIBLE_DEVICES` empty, a login node
without GPUs), or a driver too old for the CUDA runtime.

**`ARRAY_BACKEND=cupy` has no effect.** The variable is read once, when
CuNumpy is first imported. Setting `os.environ["ARRAY_BACKEND"]` after the
import does nothing; use `xp.set_backend()`.

**Code still runs on the CPU after `set_backend("cupy")`.** Look for
`import numpy as np` used for array creation, and for `from cunumpy import
zeros`-style imports, which bind the function of the backend active at import
time. See [Writing backend-agnostic code](guides/portable-code.md).

## Arrays and transfers

**`TypeError` mixing NumPy and CuPy arrays.** An operation got one array of
each kind. Find where the host array came from (often `np.` instead of `xp.`,
or data loaded from disk without `to_cunumpy()`), or check inputs with
`xp.assert_same_backend()` at function entry.

**The GPU version is slower than the CPU version.** Usually transfers in the
loop or host synchronization. Run a step inside `xp.profiling.count_transfers()` and read
the report; look for `float()`, `.item()`, `print()` or `if` on device values;
profile with `nsys` ([Timing and profiling](guides/profiling.md)). Also check
that the problem is large enough: GPUs need many thousands of elements per
operation to pay off.

**GPU memory looks full although arrays were deleted.** CuPy's memory pool
keeps freed blocks for reuse. `xp.cuda.free_memory()` returns them to the driver.
Memory that remains in use is referenced by live arrays.

## Kernels

**`TypeError: argument ... must be a CuPy array ... never copied to the
device`.** A NumPy array was passed to a `CudaKernel` pointer parameter.
CUDA kernels never copy implicitly; create the array on the device or convert it
once with `xp.to_cupy()` / `xp.as_device_array()`.

**`TypeError` about a dtype, or a non-contiguous array.** The array's dtype
must equal the C pointer type (`double*` needs `float64`, `int*` needs `int32`).
Views such as `a[:, 0]` are rejected for pointer parameters; use
`xp.ascontiguousarray()` or an `Array2D<T>` parameter.

**`OverflowError` for a scalar argument.** The Python integer does not fit the
declared C type, for example a size above 2^31 passed as `int`. Use `long long`
in the kernel.

**`NotImplementedError: No CUDA version of kernel ...`.** A `Kernel` without
CUDA version was called on the CuPy backend with `missing_cuda="raise"`. Port
the kernel, or use `missing_cuda="fallback"` while porting.

**`RuntimeWarning: No CUDA version of kernel ...: calling the host kernel`.**
The fallback is running, with host copies at every call. `catalog.summary()`
lists the kernels still to port.

**After a `PyccelKernel` call, device arrays have old values.** The kernel
wrote an argument that is not in `outputs`. Add it, or remove `outputs` to copy
everything back.

**`illegal memory access` somewhere unrelated.** An earlier kernel failed
asynchronously. Re-run with `CUNUMPY_CUDA_DEBUG=1` to raise at the guilty
launch, then use `compute-sanitizer` ([Debugging CUDA
kernels](kernels/debugging.md)). Restart the process afterwards; the CUDA
context is unusable.

**A change in a `.cuh` header is ignored.** Headers included with quotes, and
cunumpy's own headers in either form, are hashed into the compile options and
trigger recompilation; other headers included with angle brackets are not.
Also, a kernel object compiles once per process: restart the process after
editing.

**After `pip install`, every kernel has no CUDA version.** The `.cu` files are
not package data, so the wheel contains only the Python files. Declare them,
e.g. `[tool.setuptools.package-data] my_sim = ["kernels/*/*_cuda.cu",
"kernels/*.cuh"]`, and check the wheel's contents.

**`TypeError` for a CUDA kernel called with host arrays on the CuPy backend.**
The code hands host arrays to a kernel while CuPy is active. Create the kernels
with `dispatch="arrays"` so such calls run the host kernel.

**`ValueError: ... the host kernel takes (...), the CUDA kernel (...)`.** From
`check_signature()`: the two versions of a kernel take different parameters,
so one of them reads its arguments in the wrong order. Make the lists equal.

**The first time step is much slower.** Kernels are compiled on first call.
Compile at setup with `catalog.compile_all()` or `kernel.compile()`; later runs
load from CuPy's disk cache.

## MPI

**All ranks use GPU 0.** Call `xp.cuda.bind_local_device()` before `from mpi4py
import MPI`.

**Segfault in the first MPI call with a CuPy array.** The MPI library is not
CUDA-aware. `xp.mpi.require_cuda_aware_mpi()` at start-up gives a clear message;
load a CUDA-aware MPI module or rebuild `mpi4py` against one.

**Occasionally wrong data after an exchange.** A kernel was still writing the
send buffer. Call `xp.mpi.synchronize_for_mpi(send, recv)` before the MPI call.

See [Multi-GPU programs with MPI](guides/mpi.md).
