# Writing CUDA kernels

`CudaKernel` wraps a CUDA C kernel so it can be called from Python like the
host kernel it mirrors, plus a launch shape. It builds on CuPy's `RawKernel`
(compiled at run time with NVRTC, no `nvcc` or build step needed) and adds what
`RawKernel` lacks: signature checks, launch-shape arithmetic, header handling,
and debug support.

## A first kernel

```python
import cunumpy as xp

AXPY = r"""
extern "C" __global__
void axpy(double a, const double* x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""

axpy = xp.kernels.CudaKernel(AXPY, "axpy")

xp.set_backend("cupy")
x = xp.arange(10_000, dtype=xp.float64)
y = xp.zeros_like(x)
axpy(2.0, x, y, x.size, n_threads=x.size)
```

The essentials:

* The kernel is declared `extern "C" __global__` so its name is not mangled
  (templates are the exception, see below).
* Arguments are passed in the order of the C signature. `n_threads` is the
  number of threads to launch; CuNumpy computes the grid as
  `ceil(n_threads / block_size)`. Threads beyond `n` must return early, as
  above, because the last block is usually only partially used.
* Compilation happens on the first call, or when `axpy.compile()` is called.
  CuPy caches the binary on disk (`~/.cupy/kernel_cache`), so later runs load it.
* Creating a `CudaKernel` does not import CuPy. Kernels can be defined at module
  level in code that also runs on machines without a GPU.

## What the signature checks catch

`RawKernel` reads each argument with the size declared in C and never checks
it: a Python `int` passed to a `double` parameter, a `float64` array passed as
`float*`, or a NumPy array instead of a CuPy array produce wrong numbers or
crashes without an error. `CudaKernel` parses the signature once and checks
every call:

| Parameter type | Accepted | Rejected |
| --- | --- | --- |
| `double*`, `const int*`, ... | C-contiguous CuPy array of exactly that dtype | NumPy arrays (`TypeError`, never copied), other dtypes, non-contiguous views |
| `void*` | C-contiguous CuPy array of any dtype | host arrays, views |
| `double`, `float`, `complex<double>` | Python `int`/`float`, NumPy scalars that cast safely | strings, arrays, unsafe casts (`np.float64` into `float`) |
| `int`, `long long`, `size_t`, `int64_t`, ... | Python `int` and NumPy integers whose value is in range, `bool` | out-of-range values (`OverflowError`), floats |
| `Array1D<T>` ... `Array16D<T>` | CuPy array of dtype `T` and that ndim, contiguous or not | wrong dtype or ndim |
| a `CudaStruct` type | a value of that struct | anything else |

C types map to NumPy dtypes as on 64-bit Linux: `int` is `int32`, `long` and
`long long` are `int64`, `float` is `float32`, `double` is `float64`.
`xp.cuda.ctype_of(np.float64)` returns `"double"`, useful when generating source.

A wrong argument count, dtype or layout raises before anything is launched,
with the parameter name in the message:

```python
axpy(2, x, y, x.size, n_threads=x.size)  # fine: 2 is cast to double
axpy(2.0, xp.to_numpy(x), y, x.size, n_threads=x.size)
# TypeError: argument 1 (double* x) must be a CuPy array, got ndarray; arrays are never copied to the device
```

Non-contiguous views such as `a[:, 0]` are rejected for pointer parameters
because the kernel would read them as a flat buffer. Make them contiguous with
`xp.ascontiguousarray()` (a copy), or use an array view parameter (below),
which carries the strides.

The checks cost about 0.3 µs per argument. For tiny kernels called millions of
times, `check_signature=False` disables them once the calls are known to be
correct; the kernel then behaves like a raw `RawKernel`.

## Launch configuration

```python
kernel(*args, n_threads=None, grid=None, block=None, shared_mem=0, stream=None)
```

Omit both `n_threads` and `grid` to infer the launch from the first array's shape.
The default `n_threads_from="auto"` uses `shape[0]` for a 1D block (one thread per
row/particle), `shape[:2]` for a 2D block, and `shape[:3]` for a 3D block. Axes map
to CUDA x, y, z. Scalar arguments and zero-dimensional arrays are skipped;
arrays inside supported argument objects are searched in field/argument order.
Per-call block overrides determine the inference dimensions too.

```python
axpy(2.0, x, y, x.size)  # infer n_threads = x.shape[0]
```

Explicit sizes override inference. Set `n_threads_from=lambda args: args[0].size`
for a flattened element kernel, or another callback for a different axis order
or logical work count. `n_threads_from=None` requires explicit sizes;
`"first_array"` always selects only the first axis. `launch_shape(args=(...))`
inspects the inferred grid/block without compilation or a launch. CPU emulation
and host/CUDA parity tests use the same defaults.

* **1D**: `n_threads=n` with the default `block_size=128` (set per kernel with
  `CudaKernel(..., block_size=256)`).
* **2D/3D**: `n_threads=(nx, ny)` and a tuple block, e.g.
  `CudaKernel(src, "stencil", block_size=(16, 16))`. In the kernel, use
  `blockIdx.y`/`threadIdx.y` for the second dimension.
* **Explicit grid**: `grid=(n_blocks,)` instead of `n_threads`, for kernels that
  loop internally (grid-stride loops) or need a specific number of blocks.
* **Per-call block**: `block=(32, 8)` overrides `block_size` for one call.
* **Dynamic shared memory**: `shared_mem` bytes per block for
  `extern __shared__` arrays.
* **Stream**: `stream=s` queues the launch on a stream, see [Devices, memory and
  streams](../guides/gpu-devices.md).

Launches validate block/grid dimensions, threads per block against both device
and compiled-kernel limits, and static plus dynamic shared memory. Dynamic
storage above the default allowance opts in where supported. Queries and opt-in
state are cached per device. Invalid configurations raise before launching.

A zero-sized launch (`n_threads=0`) launches nothing. `launch_shape()` returns
the `(grid, block)` a call would use, which is how to size per-block outputs:

```python
BLOCK_SUM = r"""
extern "C" __global__ void block_sum(const double* x, double* out, int n) {
    extern __shared__ double buffer[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    buffer[threadIdx.x] = i < n ? x[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) buffer[threadIdx.x] += buffer[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = buffer[0];
}
"""
block_sum = xp.kernels.CudaKernel(BLOCK_SUM, "block_sum", block_size=128)
(n_blocks,), (threads,) = block_sum.launch_shape(x.size)
partial = xp.zeros(n_blocks)
block_sum(x, partial, x.size, n_threads=x.size, shared_mem=threads * 8)
total = float(partial.sum())
```

## Kernels in files

Keeping CUDA source in `.cu` files gives editor support and lets kernels share
headers:

```python
push = xp.kernels.CudaKernel.from_file(
    "kernels/push/push_cuda.cu"
)  # kernel name "push"
```

`from_file` derives the kernel name from the file name minus the `_cuda.cu`
suffix (override with `name=` or `suffix=`), and adds the file's directory to the
include path, so `#include "helpers.cuh"` next to it works.

A file with several small kernels is loaded at once with `all_from_file`, which
returns a dict by name; the kernels share one compilation:

```python
ops = xp.kernels.CudaKernel.all_from_file("kernels/vector_ops.cu", block_size=256)
ops["scale"](x, 2.0, x.size, n_threads=x.size)
ops["shift"](x, 1.0, x.size, n_threads=x.size)
```

`xp.cuda.cuda_kernel_names(source)` lists the `__global__` functions of a source.

## Headers and the compile cache

Pass extra include directories with `include_dirs=[...]` and NVRTC flags with
`options=("-std=c++17",)`. The headers shipped with CuNumpy are always found:

| Header | Provides |
| --- | --- |
| `<cunumpy/index.cuh>` | `CUNUMPY_THREAD_1D(i, n)`, `_2D`, `_3D`, `CUNUMPY_GRID_STRIDE_1D(i, n)` |
| `<cunumpy/array_view.cuh>` | strided views `Array1D<T>` to `Array16D<T>` |
| `<cunumpy/atomic.cuh>` | `cunumpy_atomic_add` and indexed 2D/3D variants, see [Accumulation kernels](accumulation.md) |
| `<cunumpy/morton.cuh>` | Morton (Z-order) keys `cunumpy_morton_key2(x, y, ...)`, `_key3`, equal to `xp.algorithms.morton_keys` on the host |
| `<cunumpy/random.cuh>` | counter-based random numbers `cunumpy_uniform(seed, stream, counter)`, `cunumpy_normal2(...)`, equal to `xp.rng.philox_uniform` on the host |

`xp.cuda.cuda_include_dir()` returns their directory for use with other compilers.

CuPy's disk cache is keyed on the source string and the options only, so
editing an included header would normally *not* trigger a recompile.
`CudaKernel` resolves the `#include "..."` files of a source (recursively),
hashes their contents, and adds `-DCUNUMPY_INCLUDE_HASH=0x...` to the options,
so a changed header produces a new cache entry. `kernel.included_headers` lists
the resolved files and `kernel.compile_options()` the options actually used.
System includes in angle brackets are not hashed.

## Index macros and array views

Kernels ported from Python index multi-dimensional arrays like `markers[ip, 3]`.
With bare pointers, the CUDA version has to receive the shape and compute
`markers[ip * n_cols + 3]` by hand, and breaks for non-contiguous arrays. Array
views carry shape and strides instead:

```python
SCALE_COLUMN = r"""
#include <cunumpy/array_view.cuh>
#include <cunumpy/index.cuh>

extern "C" __global__
void scale_column(Array2D<double> a, long long column, double factor) {
    CUNUMPY_THREAD_1D(i, a.shape[0]);   // long long i; returns if i >= shape[0]
    a(i, column) *= factor;
}
"""
scale_column = xp.kernels.CudaKernel(SCALE_COLUMN, "scale_column")

markers = xp.zeros((1000, 7))
view = markers[::2, 1:5]  # non-contiguous view is fine
scale_column(view, 1, 10.0, n_threads=view.shape[0])
```

An `ArrayND<T>` parameter takes a CuPy array of dtype `T` and `N` dimensions; it
is passed by value as pointer, shape and strides (in elements). `a(i, j)`
returns a reference, `a.shape[k]` and `a.size()` give the extents. Compiling
with `-DCUNUMPY_BOUNDS_CHECK` (automatic in [debug mode](debugging.md)) checks
every index against the shape and traps with a message on violation.

The index macros remove the boilerplate of thread index computation:

```c
CUNUMPY_THREAD_1D(i, n);               // long long i; return if i >= n
CUNUMPY_THREAD_2D(i, j, ni, nj);       // 2D launch, n_threads=(ni, nj)
CUNUMPY_THREAD_3D(i, j, k, ni, nj, nk);
CUNUMPY_GRID_STRIDE_1D(i, n) { ... }   // loop; launch with any grid
```

## Templates and generated kernels

C++ function templates are instantiated with `template_args`; dtypes are
converted with `ctype_of`, and the instantiated signature is checked as usual:

```python
import numpy as np

SCALE = r"""
template <typename T>
__global__ void scale(T* x, T factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""
scale_f64 = xp.kernels.CudaKernel(SCALE, "scale", template_args=(np.float64,))
scale_f32 = xp.kernels.CudaKernel(SCALE, "scale", template_args=(np.float32,))
```

When the source itself is generated per variant (unrolled loops per dimension,
per spline degree, per dtype), `CudaKernelVariants` creates one kernel per key
on first use and caches it:

```python
def make_matvec(ndim, dtype):
    return xp.kernels.CudaKernel(
        generate_source(ndim, xp.cuda.ctype_of(dtype)), "matvec"
    )


matvec = xp.kernels.CudaKernelVariants(make_matvec)
matvec.get(3, np.float64)(mat, x, out, n_threads=out.size)
matvec.compile_all([(3, np.float64), (3, np.complex128)], jobs=4)  # at setup
```

## Compile at setup

The first call of each kernel compiles it, which takes from a fraction of a
second to several seconds. Call `kernel.compile()` (or
`catalog.compile_all(jobs=8)` for a whole catalog, in parallel threads) during
setup so the first time step is not slower than the rest and compilation errors
appear before the simulation starts. `kernel.is_compiled` reports successful
compilation on the current device. Failed compilation can be retried. Compiling
requires a GPU (`RuntimeError` otherwise). Parallel catalog compilation preserves
the device selected by its caller.

`kernel.compile(log_stream=log)` accepts a writable compiler log stream.
`kernel.recompile(log_stream=log)` refreshes headers and debug options on the
current device; other devices keep their compiled versions. Finish in-flight
launches before rebuilding. Scope-profiler can inspect the returned CuPy kernel's
existing `attributes`; CuNumpy does not add a separate resource-reporting API.

## Tips for writing kernels

* **Mirror the host kernel's argument list.** Same order, same names. Then the
  call site is identical on both backends (see [Pairing host and CUDA
  kernels](dispatch.md)) and parity tests can share argument builders.
* **Use `long long` for indices into large arrays.** `int` overflows above
  2^31 elements; the index macros already declare `long long`.
* **Guard the tail.** Every 1D kernel needs `if (i >= n) return;` (or
  `CUNUMPY_THREAD_1D`).
* **Keep `const` on read-only pointers.** It documents intent and lets the
  compiler use the read-only cache.
* **Many threads writing one location need atomics.** Use `cunumpy_atomic_add`
  ([Accumulation kernels](accumulation.md)).
* **When something crashes, enable debug mode** before anything else ([Debugging
  CUDA kernels](debugging.md)).
