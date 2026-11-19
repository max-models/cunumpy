---
title: "CUDA kernels in Python"
description: "CupyJitKernel (cupyx.jit) and NumbaCudaKernel (numba-cuda) run CUDA kernels written in Python, launched and dispatched like CudaKernel."
---

A CUDA kernel does not have to be written in CUDA C. Two JIT compilers turn a
Python function into a CUDA kernel:

* [`cupyx.jit`](https://docs.cupy.dev/en/stable/user_guide/kernel.html#raw-kernels-in-python)
  is part of CuPy: no further dependency, and the kernel runs on CuPy's streams
  and memory pool;
* [numba-cuda](https://nvidia.github.io/numba-cuda/) (`pip install
  'numba-cuda[cu12]'`) covers more of Python and of CUDA (device functions,
  cooperative groups) and has a CPU simulator.

cunumpy wraps them in `CupyJitKernel` and `NumbaCudaKernel`. Both are launched
like a [`CudaKernel`](/cunumpy/kernels/cuda-kernel/) (`n_threads`, `grid`,
`block`, `shared_mem`, `stream`, launch size inferred from the first array) and
can sit next to it in a [`Kernel`](/cunumpy/kernels/dispatch/). A typical port
goes from the host kernel to a Python version written in an afternoon, and to
CUDA C only for the kernels a profile shows to be worth it.

## A first kernel

```python
import cunumpy as xp
from cupyx import jit


@jit.rawkernel()
def scale(x, factor, out):
    i = jit.grid(1)
    if i < x.size:
        out[i] = factor * x[i]


kernel = xp.kernels.CupyJitKernel(scale, block_size=256)
x = xp.linspace(0.0, 1.0, 10_000)
out = xp.empty_like(x)
kernel(x, 2.0, out)  # 40 blocks of 256 threads, from x.shape[0]
```

The same with numba-cuda:

```python
from numba import cuda


@cuda.jit
def scale(x, factor, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = factor * x[i]


kernel = xp.kernels.NumbaCudaKernel(scale, block_size=256)
kernel(x, 2.0, out)  # CuPy arrays, no copies
```

A plain (undecorated) function works too: the wrapper applies
`jit.rawkernel()` or `cuda.jit` itself, at the first launch.

## The rules

They follow those of `CudaKernel` where a JIT allows:

* **Device arrays only.** A NumPy array raises `TypeError` instead of being
  copied to the GPU (numba would copy it silently, and back after the launch).
* **Argument objects are flattened.** A
  [`CudaArguments`](/cunumpy/kernels/arguments/) (anything with
  `__cuda_args__()`) becomes its values, in order. C structs
  (`CudaStruct`, `CudaStructArguments`) raise: only CUDA C kernels take
  structs by value.
* **The number of arguments is checked** against the Python function. The
  types are not: the JIT compiles a version for each new set of argument types
  (a Python `int` becomes `int64`, a `float` `float64`), so pass NumPy scalars
  (`np.int32(n)`) where the type matters.
* **Streams.** `CupyJitKernel` launches on CuPy's current stream (or `stream`).
  `NumbaCudaKernel` does too: it hands CuPy's stream to numba with
  `cuda.external_stream`, so the launch is ordered with the surrounding CuPy
  work. A numba stream (no `ptr` attribute) is used as it is.
* **Debug mode** ([`set_cuda_debug`](/cunumpy/kernels/debugging/), or
  `debug=True`) synchronizes after every launch and raises a `RuntimeError`
  naming the kernel.
* **Lazy loading.** The function can come from a `loader` (for example an
  import), called at the first use; a kernel can thus be built without CuPy or
  numba installed. `kernel.available()` loads it, `kernel.error` holds why it
  failed, `kernel.compile()` raises a `RuntimeError` naming what is missing.

What the JITs cannot do, write in CUDA C: cunumpy's headers (`random.cuh`,
`atomic.cuh`, `reduce.cuh`, ...) are not available in Python kernels.

## In a `Kernel`

A `Kernel` takes up to three device implementations:

```python
push = xp.kernels.Kernel(
    push_host,
    cuda_kernel=None,  # CudaKernel, "cuda_c"
    cupy_jit_kernel=push_jit,  # CupyJitKernel or a function, "cupy_jit"
    numba_cuda_kernel=None,  # NumbaCudaKernel or a function, "numba_cuda"
)
push.implementations  # ('host', 'cupy_jit')
push.selected(device=True)  # 'cupy_jit'
```

In a kernel folder they are found by name, next to the host kernel:

```text
my_sim/kernels/push/
├── push_kernels.py       # def push(...)                  host kernel
├── push_cuda.cu          # __global__ void push(...)      "cuda_c" (or push_cuda_c.cu)
├── push_cupy_jit.py      # @jit.rawkernel() def push(...)  "cupy_jit"
└── push_numba_cuda.py    # @cuda.jit def push(...)        "numba_cuda"
```

Each module is imported at the first device call, so a folder with a
`cupyx.jit` kernel still loads on a machine without CuPy. `block_size`,
`n_threads_from` and `debug` of `from_folder`/`from_package` apply to all
device kernels; the suffixes are set with `cupy_jit_suffix` and
`numba_cuda_suffix`.

On the device a call runs the first implementation that loads, in the order
`"cuda_c"`, `"cupy_jit"`, `"numba_cuda"`: a hand-written CUDA C kernel replaces
the Python one as soon as it exists. Choose one explicitly to compare them or to
time them:

```python
with xp.kernels.use_device_kernel_implementation("cupy_jit"):
    push(markers, dt)  # LookupError if push has no cupy_jit version
```

| Setting | Runs on the device |
| --- | --- |
| `None` (default) | the first available of `cuda_c`, `cupy_jit`, `numba_cuda`; else the host kernel with `missing_cuda="fallback"` |
| `"cuda"` | the same, but never the host fallback (`LookupError`) |
| `"cuda_c"`, `"cupy_jit"`, `"numba_cuda"` | only that implementation (`LookupError` if missing or not loadable) |

`CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION=cupy_jit` sets it at import.
`kernel.check_signature()` (and `catalog.check_signatures()`) compares the
parameter names of the Python kernels with the host kernel's.

## Testing without a GPU

[`assert_kernels_agree`](/cunumpy/kernels/testing/) compares every device
implementation of a kernel with the host one, each in turn. Two of them also run
without a GPU, on the [fake CuPy](/cunumpy/kernels/testing/#run-the-cupy-code-paths-without-a-gpu-the-fake-cupy):

* the fake CuPy emulates `cupyx.jit`: a kernel runs on the CPU, one thread
  after another, on the host buffers of the fake arrays. `jit.grid`,
  `jit.gridsize`, the index variables and the atomics work; `jit.syncthreads`,
  `jit.shared_memory` and warp functions raise `NotImplementedError`;
* numba's simulator (`NUMBA_ENABLE_CUDASIM=1`, set before numba is imported)
  runs `NumbaCudaKernel` launches on the CPU, also on fake CuPy arrays.

```bash
CUNUMPY_FAKE_CUPY=1 NUMBA_ENABLE_CUDASIM=1 pytest tests/test_parity.py
```

On the fake CuPy, `assert_kernels_agree` checks the implementations that can
run there (`"cupy_jit"`, and `"numba_cuda"` under the simulator) and skips the
test only if none can. The emulations are serial and follow Python arithmetic,
so they find indexing and logic errors, not races or differences in rounding:
run the parity tests on a GPU too.
