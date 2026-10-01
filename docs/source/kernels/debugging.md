# Debugging CUDA kernels

Kernel launches are asynchronous. When a kernel reads out of bounds, the error
is reported by the *next* operation that waits for the GPU (a `.get()`, an MPI
call, the next allocation), often far from the kernel that caused it, with a
message like `cudaErrorIllegalAddress: an illegal memory access was
encountered`. Out-of-bounds accesses that stay inside allocated memory produce
no error at all, only wrong numbers.

## Step 1: debug mode

Enable debug mode in any of these ways:

```bash
CUNUMPY_CUDA_DEBUG=1 python simulate.py            # whole process
```

```python
xp.set_cuda_debug(True)          # globally, from now on
with xp.cuda_debug():            # for a block
    ...
xp.CudaKernel(src, "push", debug=True)   # one kernel, regardless of the global setting
```

In debug mode a `CudaKernel`:

* is compiled with `-lineinfo` (source lines for `compute-sanitizer` and
  profilers) and `-DCUNUMPY_BOUNDS_CHECK`, which turns on bounds checks in
  `Array1D`/`Array2D`/`Array3D` views (an out-of-bounds index prints the index
  and shape, then traps);
* synchronizes after every launch, so a failure raises at the launch that
  caused it, as a `RuntimeError` naming the kernel and its grid and block, with
  the CUDA error chained.

```python
with xp.cuda_debug():
    push = xp.CudaKernel(SOURCE, "push")
    push(markers, dt, n, n_threads=n)
# RuntimeError: CUDA error after launching kernel 'push' with grid (79,) and block (128,): ...
```

Compile options are fixed when a kernel is compiled. A kernel that was already
compiled before debug mode was enabled keeps its options (the synchronization
still applies). Enable debug mode before the kernels are first called, or set
`CUNUMPY_CUDA_DEBUG=1` in the environment. `kernel.debug_active()` and
`kernel.compile_options()` show what applies to a kernel.

Use `-DCUNUMPY_BOUNDS_CHECK` in your own code too:

```c
#ifdef CUNUMPY_BOUNDS_CHECK
    if (cell < 0 || cell >= n_cells) { printf("bad cell %lld\n", cell); __trap(); }
#endif
```

`-G` (full device debug symbols) is not available with NVRTC.

## Step 2: compute-sanitizer

Debug mode tells you *which* kernel failed. NVIDIA's memory checker tells you
*where*, thanks to `-lineinfo`, and also finds out-of-bounds accesses that do
not crash:

```bash
CUNUMPY_CUDA_DEBUG=1 compute-sanitizer python -m pytest tests/test_push.py
CUNUMPY_CUDA_DEBUG=1 compute-sanitizer --tool racecheck python check_deposit.py
```

The `memcheck` tool (default) finds invalid reads and writes, `racecheck`
shared-memory races, `initcheck` reads of uninitialized device memory. Run them
on a small problem; they slow execution down by a large factor.

## Step 3: compare with the host kernel

Most porting bugs are not crashes but differences: an index off by one, a
wrong stride, a missing `if`. The host kernel is the reference.
`assert_kernels_agree()` runs both on the same inputs and names the argument
that differs (see [Testing kernels](testing.md)). For `__device__` helpers,
`device_function_kernel()` exposes a single function to Python so it can be
compared value by value with its host version.

## Common causes

| Symptom | Likely cause |
| --- | --- |
| `TypeError` at the call, before launch | argument does not match the C signature: host array, wrong dtype, non-contiguous view, wrong count. Read the message; it names the parameter. |
| illegal memory access | index beyond the array: missing `if (i >= n) return;`, wrong `n`, wrong stride arithmetic. Use array views with bounds checks. |
| correct on small inputs, wrong on large ones | `int` overflow in index computations; use `long long`. |
| results differ slightly between runs | floating-point atomics in a different order; compare with a tolerance. |
| results occasionally garbage after MPI | missing `synchronize_for_mpi()` before the MPI call. |
| a header change has no effect | the header is included with angle brackets (not hashed) or the kernel object was compiled before the change; see "Headers and the compile cache" in [Writing CUDA kernels](cuda-kernel.md). |

After an illegal memory access the CUDA context of the process is unusable: all
later CUDA calls fail. Restart the process (or the pytest run) after fixing the
kernel.
