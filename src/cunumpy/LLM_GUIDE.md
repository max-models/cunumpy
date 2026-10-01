# cunumpy: guide for AI coding assistants

This file tells AI coding assistants (and people in a hurry) how to write
correct code with `cunumpy`. It ships inside the installed package
(`<site-packages>/cunumpy/LLM_GUIDE.md`; locate it with
`python -c "import cunumpy, pathlib; print(pathlib.Path(cunumpy.__file__).parent / 'LLM_GUIDE.md')"`).
It is self-contained. The full documentation is at
https://max-models.github.io/cunumpy/ and in `docs/source/` of the repository.

## What cunumpy is

* A drop-in NumPy-like namespace: `import cunumpy as xp`, then `xp.zeros`,
  `xp.fft.rfft`, `xp.linalg.norm`, ... Every attribute is forwarded at access
  time to `array_api_compat.numpy` or `array_api_compat.cupy`, depending on the
  **active backend** (`"numpy"` = CPU, `"cupy"` = NVIDIA GPU). Results are plain
  `numpy.ndarray` / `cupy.ndarray`; there is no wrapper array type.
* Helpers for explicit host/device transfers, device/stream/memory control, MPI
  with one rank per GPU, profiling.
* A kernel layer for porting compiled CPU kernels (Pyccel, Numba, Python loops)
  to CUDA one at a time: `PyccelKernel`, `CudaKernel`, `Kernel`,
  `KernelCatalog`, `KernelArguments`, `CudaStruct`, `DeviceMirror`, and test
  helpers in `cunumpy.testing`.

## Hard rules

1. **Always `import cunumpy as xp` and call `xp.<name>(...)` at the point of use.**
   Never `from cunumpy import zeros, arange, ...`: that binds the function of the
   backend active at import time. Same for `numpy_backend`/`cupy_backend`: read
   `xp.cupy_backend` each time.
2. **Do not use `np.<array function>` for data that should follow the backend.**
   `np.zeros` always allocates on the host. Using `numpy` for dtypes
   (`np.float64`), host-only I/O, and host-side random test data is fine.
3. **Select the backend once, at the program entry point** (`ARRAY_BACKEND=cupy`
   env var before import, or `xp.set_backend("cupy")` early). Library code must
   not call `set_backend()`. Use `with xp.use_backend(...)` for scoped switches
   (tests, CPU references). Backend state is process-global and not thread-safe.
4. **Requesting CuPy can silently fall back to NumPy** (no CuPy, no GPU, CUDA
   mismatch). Check `xp.get_backend()` after `set_backend("cupy")` if it matters.
5. **Changing the backend never moves existing arrays.** Transfers are always
   explicit: `xp.to_numpy(a)`, `xp.to_cupy(a)`, `xp.to_cunumpy(a)`.
6. **No transfers or host syncs inside time loops.** Avoid `to_numpy`, `float(x)`,
   `x.item()`, `print(x)`, `if device_value:` in hot loops. Verify with
   `xp.assert_no_transfers()` in tests.
7. **`CudaKernel` never copies.** Pointer parameters require C-contiguous CuPy
   arrays of the exact dtype; NumPy arrays raise `TypeError`. Do not "fix" that
   error by disabling checks; convert once with `xp.to_cupy` /
   `xp.as_device_array` where the data is created.
8. **Keep `check_signature=True`** (the default). Only disable it for tiny,
   already-tested kernels in hot loops.
9. **A CUDA kernel mirrors its host kernel**: same name, same positional
   argument order. `Kernel` calls take positional arguments only, plus the
   launch keywords `n_threads=`/`grid=`/`block=`/`shared_mem=`/`stream=`.
10. **Declare `outputs` correctly on `PyccelKernel`** (and via `host_options`).
    An argument the kernel writes but that is not declared leaves stale device
    data, silently. If unsure, leave `outputs=None` (copies everything back).

## Decision guide

| Task | Use |
| --- | --- |
| array math that should run on CPU or GPU | `xp.*` functions |
| function receiving arrays from elsewhere | `array_xp = xp.get_array_module(a)`; `xp.assert_same_backend(a, b)` |
| normalize inputs at an API boundary | `xp.to_cunumpy(a)` once |
| hand data to SciPy/matplotlib/h5py | `xp.to_numpy(a)` |
| call an existing NumPy-only kernel with GPU arrays (slow, correct) | `xp.PyccelKernel(fn, outputs=(...))` |
| launch a hand-written CUDA C kernel | `xp.CudaKernel(source, "name")` / `CudaKernel.from_file(path)` |
| host kernel + CUDA port, chosen by backend | `xp.Kernel(host_fn, cuda_kernel_or_None)` |
| many kernels in a package, ported incrementally | `xp.KernelCatalog.from_package(__name__, missing_cuda="fallback")` |
| group arrays/scalars into one kernel argument | `xp.CudaArguments` (device only), `xp.KernelArguments` (host object + device tuple), `xp.CudaStruct` (C struct), `xp.CudaStructArguments` (C struct as a class) |
| CUDA struct from a Pyccel argument class | `xp.CudaStruct.from_signature(Cls.__init__, "Name")`, `xp.write_cuda_header(...)` |
| kernel writes into a host buffer owned by another library | `xp.DeviceMirror(host_array)` + `<cunumpy/atomic.cuh>` |
| N-D indexing in CUDA, non-contiguous arrays | `Array1D<T>`..`Array3D<T>` params from `<cunumpy/array_view.cuh>` |
| one MPI rank per GPU | `bind_local_device()` → `from mpi4py import MPI` → `require_cuda_aware_mpi()` → `synchronize_for_mpi(...)` before each call |
| timing GPU code | `with xp.timed_region("name") as t:` → `t.elapsed` |
| profiler markers | `xp.nvtx_range("name")` (context manager or decorator) |
| find transfers | `with xp.count_transfers() as c: ...; print(c.report())` |
| debug an illegal memory access | `CUNUMPY_CUDA_DEBUG=1`, then `compute-sanitizer` |
| test on both backends | `cunumpy.testing.BACKENDS`, `backend` fixture, `requires_cupy` |
| test CUDA vs host kernel | `cunumpy.testing.assert_kernels_agree(kernel, make_args, n_threads=...)` |

## API cheat sheet

Backend and inspection:

```python
xp.set_backend("numpy" | "cupy")     # global; falls back to numpy if cupy unusable
xp.get_backend() -> "numpy" | "cupy" # active backend
with xp.use_backend("numpy"): ...    # temporary, exception-safe
xp.numpy_backend, xp.cupy_backend    # bools for the active backend
xp.cupy_available() -> bool          # CuPy importable and functional (cached)
xp.get_array_backend(a) -> "numpy" | "cupy"   # where this array lives
xp.get_array_module(a)               # array_api_compat module for a's backend
xp.is_cpu(a), xp.is_gpu(a)
xp.same_backend(*arrays) -> bool
xp.assert_same_backend(*arrays)      # TypeError if mixed
xp.__version__
```

Transfers:

```python
xp.to_numpy(a)        # -> numpy.ndarray; copies only if a is CuPy
xp.to_cupy(a)         # -> cupy.ndarray; copies only if a is not CuPy; ImportError w/o CuPy
xp.to_cunumpy(a)      # -> array of the active backend
xp.as_device_array(value, dtype=None, ndim=None, *, name=None)
                      # contiguous CuPy array of dtype: returned as is; else one copy;
                      # RuntimeError on the NumPy backend
with xp.count_transfers() as c: ...   # c.total, c.to_host, c.to_device,
                                      # c.kernel_conversions, c.fallbacks, c.events, c.report()
with xp.assert_no_transfers(): ...    # AssertionError with report if anything copied
```

Only transfers through cunumpy are counted (not raw `cupy.asarray`, `.get()`,
`float(device_scalar)`, or `DeviceMirror.to_host()/to_device()`).

Random numbers and dtypes:

```python
rng = xp.get_rng(seed=None)   # numpy or cupy Generator for the active backend
xp.default_float_dtype()      # float64 of the active backend
```

NumPy and CuPy generators give different sequences for the same seed. For
identical data on both backends: `xp.to_cunumpy(np.random.default_rng(s).random(n))`.

Devices, memory, streams (all safe on NumPy: no-ops / neutral values):

```python
xp.device_count() -> int              # visible GPUs, 0 without CuPy
xp.set_device(i)
xp.memory_info() -> (free, total) | None
xp.free_memory()                      # release CuPy pool cached blocks
xp.synchronize()
with xp.stream() as s: ...            # s is None on NumPy; prefer xp.synchronize()
xp.pin_memory(host_array)             # pinned copy; needs CuPy
```

MPI:

```python
xp.set_backend("cupy")
xp.bind_local_device()        # BEFORE `from mpi4py import MPI`; uses local_rank()
from mpi4py import MPI
xp.require_cuda_aware_mpi()   # collective; RuntimeError if MPI can't take device buffers
xp.mpi_is_cuda_aware(comm=None) -> bool
xp.local_rank() -> int        # node-local rank from launcher env vars
xp.synchronize_for_mpi(*buffers)  # before every MPI call that touches device buffers
```

Profiling:

```python
with xp.timed_region("name", sync=True) as t: ...   # t.name, t.elapsed (s), t.synced
with xp.nvtx_range("name", color=None): ...         # also usable as @decorator
```

`PyccelKernel`:

```python
k = xp.PyccelKernel(fn, use_cupy=None, object_modules=(), is_array=None, outputs=None)
k(*args, **kwargs)
```

NumPy path: calls `fn` directly. CuPy path: copies CuPy arrays (also inside
lists/tuples/dicts and objects whose class module starts with an
`object_modules` prefix) to the host, calls `fn`, copies `outputs` (positional
indices or keyword names; default: all) back, converts returned NumPy arrays to
CuPy. Does not compile anything.

`CudaKernel`:

```python
k = xp.CudaKernel(source, name, *, block_size=128, options=(), include_dirs=(),
                  source_dir=None, structs=(), template_args=None,
                  check_signature=True, debug=None)
k = xp.CudaKernel.from_file("push/push_cuda.cu")           # name "push"
ks = xp.CudaKernel.all_from_file("ops.cu")                 # dict name -> kernel
k(*args, n_threads=None, grid=None, block=None, shared_mem=0, stream=None)
k.compile(); k.is_compiled; k.launch_shape(n_threads) -> (grid, block)
k.included_headers; k.compile_options(); k.debug_active()
xp.CudaKernelVariants(factory).get(*key); .compile_all(keys, jobs=1)
xp.ctype_of(np.float64) == "double"
xp.cuda_kernel_names(source); xp.parse_cuda_signature(source, name)
xp.cuda_include_dir()
```

* Source must declare `extern "C" __global__` (templates: plain `__global__`
  plus `template_args=(np.float64, 3)`).
* Type mapping (LP64): `int`=int32, `long`/`long long`=int64, `float`=float32,
  `double`=float64, `complex<double>`=complex128, `bool`=bool; `int64_t`,
  `size_t` etc. supported.
* Scalars: Python `int`/`float`/`bool` are cast with range checks
  (`OverflowError`); NumPy scalars must match or cast safely.
* Pointer params: C-contiguous CuPy arrays, exact dtype (`void*` any).
  `ArrayND<T>` params: CuPy arrays of dtype T and ndim N, any strides.
* Compiled lazily with NVRTC on first call; cached on disk by CuPy; quoted
  `#include "..."` headers are hashed into the options so edits recompile.
* Creating a `CudaKernel` does not import CuPy; compiling needs a GPU.

Shipped CUDA headers (always on the include path):

```c
#include <cunumpy/index.cuh>       // CUNUMPY_THREAD_1D(i, n) /_2D/_3D, CUNUMPY_GRID_STRIDE_1D(i, n) {...}
#include <cunumpy/array_view.cuh>  // Array1D<T>..Array3D<T>: data, shape[], strides[] (elements), a(i, j), size()
#include <cunumpy/atomic.cuh>      // cunumpy_atomic_add(double*|float*, v), _2d(data, n1, i, j, v), _3d(...)
```

`-DCUNUMPY_BOUNDS_CHECK` (on in debug mode) bounds-checks array view indexing.

`Kernel` and `KernelCatalog`:

```python
k = xp.Kernel(host_kernel, cuda_kernel=None, *, name=None,
              missing_cuda="raise" | "fallback", cuda_path=None, host_options=None)
k(*args, n_threads=..., grid=None, block=None, shared_mem=0, stream=None)
k.get_kernel(); k.compile(); k.has_cuda

catalog = xp.KernelCatalog.from_package(__name__, *, host_suffix="_kernels",
    cuda_suffix="_cuda.cu", missing_cuda="raise", host_options=None,
    include_dirs=None, **cuda_options)
catalog["push"]; catalog.summary(); catalog.with_cuda; catalog.without_cuda
catalog.compile_all(jobs=1); catalog.parity_cases(); catalog.register(kernel)
```

Package layout for `from_package`: `pkg/<name>/<name>_kernels.py` defines
function `<name>` (host); optional `pkg/<name>/<name>_cuda.cu` defines
`__global__ void <name>(...)`. Kernel folders must be importable packages.
`host_options` may be a function of the kernel name, e.g.
`lambda name: {"outputs": OUTPUTS[name]}`.

Argument objects:

```python
class Dev(xp.CudaArguments):            # flattened into several CUDA params
    def __init__(self, x, n): super().__init__(x, n)

class Args(xp.KernelArguments):          # one object, host form + device form
    def __host_args__(self): return host_object        # host kernel gets this
    def __cuda_args__(self): return (arr, n, ...)      # CUDA kernel gets these, flattened

S = xp.CudaStruct("S", [("x", "double*"), ("n", "long long"), ("a", "Array2D<double>")])
S.declaration; S.dtype; S.to_header(path); value = S(x=..., n=..., a=...)
S.verify_layout()                        # GPU test: compiler layout == S.dtype (also verify_layout("hdr.cuh"))
S = xp.CudaStruct.from_signature(Cls.__init__, "S", int_type="long long")
xp.write_cuda_header("args.cuh", [S1, S2])

class A(xp.CudaStructArguments):         # the struct as a class; A.struct is the CudaStruct
    struct_name = "A"
    fields = (("x", "double*"), ("n", "int"))
    def __init__(self, x):
        self.x, self.n = x, x.shape[0]
        self.pack()                      # repacks itself when a field changes; copies repack
xp.CudaKernel(S.declaration + src, "k", structs=[S])
xp.resolve_host_args(args, kwargs)
```

Only top-level arguments are resolved. Cache both forms lazily and invalidate
them when the underlying arrays are replaced. A packed struct holds device
addresses: re-pack after replacing an array (a `CudaStructArguments` does this
itself at the next launch; make its fields properties to follow an owner's arrays).

`DeviceMirror`:

```python
m = xp.DeviceMirror(host_numpy_array)  # TypeError if not numpy.ndarray
m.device        # CuPy copy (lazy) on CuPy; the host array itself on NumPy
m.zero(); m.to_device(); m.to_host()   # to_host copies in place; no-ops on NumPy
m.rebind(new_host_array)               # after the owner reallocates
```

Debugging:

```python
xp.set_cuda_debug(True); xp.get_cuda_debug(); with xp.cuda_debug(): ...
xp.CudaKernel(..., debug=True)    # env: CUNUMPY_CUDA_DEBUG=1
```

Debug mode adds `-lineinfo -DCUNUMPY_BOUNDS_CHECK` at compile time and
synchronizes after each launch (errors become `RuntimeError` naming the kernel).
Enable it before kernels compile. After an illegal memory access, the process
must be restarted.

Testing (`import cunumpy.testing`; not imported by `import cunumpy`):

```python
from cunumpy.testing import BACKENDS, backend, requires_cupy, \
    assert_kernels_agree, device_function_kernel

@pytest.mark.parametrize("backend", BACKENDS)   # "numpy" always, "cupy" if GPU
def test_x(backend):
    with xp.use_backend(backend): ...

assert_kernels_agree(kernel, make_args, *, n_threads=None, grid=None, block=None,
                     rtol=1e-12, atol=0.0, n_calls=1, outputs=None, seed=0)
# make_args(backend, seed) -> tuple of positional args, built with the backend active

k = device_function_kernel(header_source, "int f(const double* t, int p, double x)")
k(t, p_array, x_array, out, n, n_threads=n)    # scalars become per-thread arrays
```

## Canonical patterns

Backend-agnostic function:

```python
def normalize(v):
    array_xp = xp.get_array_module(v)
    return v / array_xp.linalg.norm(v)
```

Script entry point:

```python
import cunumpy as xp

def main(use_gpu: bool):
    xp.set_backend("cupy" if use_gpu else "numpy")
    print("backend:", xp.get_backend())
    data = xp.to_cunumpy(load_host_data())       # one transfer in
    for _ in range(n_steps):
        data = update(data)                      # no transfers here
    save(xp.to_numpy(data))                      # one transfer out
```

Kernel pair:

```python
SRC = r"""
#include <cunumpy/index.cuh>
extern "C" __global__
void scale(double* x, double a, long long n) {
    CUNUMPY_THREAD_1D(i, n);
    x[i] *= a;
}
"""

def scale_host(x: "float[:]", a: float, n: int):
    for i in range(n):
        x[i] *= a

scale = xp.Kernel(scale_host, xp.CudaKernel(SRC, "scale"),
                  host_options={"outputs": (0,)})
scale(x, 2.0, x.size, n_threads=x.size)
```

Parity test:

```python
def make_args(backend, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(1000))
    return (x, 2.0, x.size)

def test_scale():
    assert_kernels_agree(scale, make_args, n_threads=1000)
```

## Common mistakes to avoid

* Writing `if xp.get_backend() == "cupy": cupy.foo(a) else: numpy.foo(a)` for
  functions that `xp.foo` already dispatches. Branch only when the libraries
  genuinely differ (SciPy vs `cupyx.scipy`), and branch on the *array*
  (`xp.is_gpu(a)`), not the global setting.
* Calling `xp.to_cupy()` inside a loop or per kernel call.
* Passing NumPy arrays or non-contiguous views (`a[:, 0]`) to `CudaKernel`
  pointer parameters. Use `xp.ascontiguousarray()` or an `Array2D<T>` param.
* `int` loop indices in CUDA for arrays that may exceed 2^31 elements; use
  `long long` (the index macros do).
* Forgetting `if (i >= n) return;` (or `CUNUMPY_THREAD_1D`) in a kernel.
* Plain `+=` from many threads into one cell; use `cunumpy_atomic_add`.
* Timing GPU code with `time.perf_counter()` without synchronizing; use
  `xp.timed_region()`.
* Importing `mpi4py.MPI` before `xp.bind_local_device()`; skipping
  `xp.synchronize_for_mpi()` before MPI calls on device buffers.
* Comparing CPU and GPU results with exact equality where the GPU uses atomics
  or a different reduction order; use a tolerance.
* Assuming `xp.random.seed(s)` gives the same numbers on both backends.
* Expecting `DeviceMirror` copies or raw CuPy conversions to show up in
  `count_transfers()`.
* Writing code that requires CuPy at import time. `import cunumpy` and creating
  `CudaKernel` objects work without CuPy; import `cupy` lazily, only on the GPU
  path, if you need it at all.
