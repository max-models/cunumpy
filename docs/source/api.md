# API reference

CuNumpy exports the active NumPy-like namespace and a set of helpers for
backend selection, array inspection and conversion, hardware control, and
host-only kernels. Most examples use `import cunumpy as xp`.

## NumPy-like namespace

```python
import cunumpy as xp

values = xp.arange(5)
total = xp.sum(values)
```

At runtime, NumPy-like attributes such as `array`, `sum`, `fft`, and `linalg`
are forwarded to the currently selected `array-api-compat` NumPy or CuPy
module. CuNumpy does not wrap every operation individually. The available
operations and some details can therefore vary with the installed NumPy and
CuPy versions. In normal use, access those operations through the top-level
`cunumpy` namespace, commonly imported as `xp`.

For an explanation of what the compatibility module does and why CuNumpy
uses it, read [Why CuNumpy uses `array-api-compat`](array-api-compat.md).

NumPy and CuPy are not interchangeable for every function or object. A
function that needs to follow an input array's location should use
`get_array_module(array)` instead of assuming the global backend matches it.

## Backend selection

### `set_backend(backend)`

Selects the process-wide backend used for new NumPy-like operations. Supported
values are `"numpy"` and `"cupy"`:

```python
xp.set_backend("cupy")
values = xp.arange(10)
```

Requesting CuPy selects it only when CuPy and its CUDA runtime are functional;
otherwise CuNumpy falls back to NumPy. Check `get_backend()` to inspect the
effective selection. Changing the selection does not move arrays that have
already been created.

The initial backend is NumPy unless `ARRAY_BACKEND=cupy` is set before CuNumpy
is imported. Other values of this environment variable result in the NumPy
default.

Backend state is shared process-wide and `set_backend()` is not thread-safe.
Concurrent tasks that change the backend may race.

### `get_backend()`

Returns the active global backend name, either `"numpy"` or `"cupy"`. This is
the getter paired with `set_backend()`:

```python
xp.set_backend("numpy")
assert xp.get_backend() == "numpy"
```

### `use_backend(backend)`

Context manager that temporarily selects a backend and restores the previous
backend on exit, even if the block raises an exception:

```python
with xp.use_backend("numpy"):
    reference = xp.zeros(10)
```

As backend selection is shared process state, this context manager is suited
to sequential use rather than concurrent backend switching.

### `numpy_backend`, `cupy_backend`

Boolean properties indicating whether the currently selected global backend
is NumPy or CuPy:

```python
if xp.cupy_backend:
    print("new arrays are being created on the GPU")
```

For a string value, prefer `get_backend()`.

## Inspect arrays and select operations

### `get_array_backend(array)`

Returns `"numpy"` or `"cupy"` according to the given array's type. It reports
the array's location, not the active global selection:

```python
xp.set_backend("numpy")
values_gpu = xp.to_cupy([1, 2, 3])
assert xp.get_backend() == "numpy"
assert xp.get_array_backend(values_gpu) == "cupy"
```

### `get_array_module(array)`

Returns the `array-api-compat` module matching the array: the NumPy module for
a NumPy array or the CuPy module for a CuPy array. This supports functions
that dispatch based on their input rather than global state:

```python
def standardize(values):
    array_xp = xp.get_array_module(values)
    mean = array_xp.mean(values)
    scale = array_xp.std(values)
    return (values - mean) / scale
```

The returned module is an `array-api-compat` module, consistent with the
active module used by CuNumpy. It is not necessarily identical to importing
raw `numpy` or raw `cupy`.

### `is_cpu(array)`, `is_gpu(array)`

Return booleans indicating whether an array is a NumPy (CPU) or CuPy (GPU)
array. They are convenience checks equivalent to comparing
`get_array_backend(array)` with `"numpy"` or `"cupy"`.

### `same_backend(*arrays)`

Returns `True` if all provided arrays have the same backend. Zero or one
argument is considered to match:

```python
if xp.same_backend(position, velocity):
    update(position, velocity)
```

### `assert_same_backend(*arrays)`

Raises `TypeError` with the detected backend names if arrays do not all share
a backend. Use it at API boundaries to provide a clear error before a mixed
NumPy/CuPy operation fails deeper in a library:

```python
def combine(left, right):
    xp.assert_same_backend(left, right)
    return left + right
```

## Convert arrays

### `to_numpy(array)`

Converts to a host-side NumPy array. CuPy arrays are copied from device to
host. Other array-like inputs are passed through `numpy.asarray`; NumPy arrays
may therefore be returned as-is rather than copied.

### `to_cupy(array)`

Converts an array-like input to a CuPy array. Raises `ImportError` if CuPy or
CUDA is unavailable or not functional. The source is not modified.

### `to_cunumpy(array)`

Converts to the currently active backend. This is convenient at an API
boundary when an input should be normalized to the configured backend:

```python
normalized = xp.to_cunumpy(input_array)
assert xp.get_array_backend(normalized) == xp.get_backend()
```

Each conversion returns a suitable array; it does not change the active
backend or mutate the source.

## Count transfers

A transfer inside a time loop is the classic performance bug of a GPU port:
every step then waits for the device and copies an array. These helpers let a
test verify that a block of code does not transfer at all.

### `count_transfers()`

Context manager yielding a `TransferCounter` that records every host/device
transfer made through CuNumpy while the block runs, with the call site of
each:

```python
with xp.count_transfers() as counter:
    propagator(dt)

assert counter.total == 0, counter.report()
```

Four kinds of events are recorded:

* `to_host`: `to_numpy()` (or `to_cunumpy()`) called with a CuPy array;
* `to_device`: `to_cupy()` (or `to_cunumpy()`) called with anything that is not
  a CuPy array already;
* `kernel_conversion`: a `PyccelKernel` call that copied device arrays to the
  host (and back), one event per call, naming the kernel and the number of
  arrays converted;
* `fallback`: a `Kernel` without CUDA kernel calling its host kernel on the
  CuPy backend (`missing_cuda="fallback"`), one event per call, naming the
  kernel. The host copies it makes are counted as one `kernel_conversion`
  event in addition.

Only real transfers count: `to_numpy()` of a NumPy array or `to_cupy()` of a
CuPy array records nothing. The counter has the attributes `to_host`,
`to_device`, `kernel_conversions`, `fallbacks` (counts per kind), `total`,
`events` (a list of `TransferEvent(kind, description, where)`, where `where`
is the `file:line` of the caller outside CuNumpy) and
`kernel_conversion_calls` (the `kernel_conversion` events). `report()` returns
a multi-line string with the events grouped by kind and call site, with
counts:

```text
4 transfer(s) through cunumpy (3 to_host, 1 to_device, 0 kernel_conversion, 0 fallback)
  to_host (3):
    /home/me/sim/diagnostics.py:42: to_numpy(shape=(100000,), dtype=float64) (x3)
  to_device (1):
    /home/me/sim/setup.py:17: to_cupy(shape=(100000,), dtype=float64)
```

Blocks can be nested; every active counter sees the transfers made inside it.
When no counter is active, the instrumentation costs a single check per call.
Like the backend selection, the active counters are process-wide state and
not thread-safe.

**Limitation:** only transfers made through CuNumpy are seen. Raw
`cupy.ndarray.get()`, `cupy.asarray(numpy_array)`, `numpy.asarray(cupy_array)`,
`float(device_array)`, and implicit conversions inside other libraries are
not counted. Use `nsys` (or CuPy's profiling hooks) to find those.

### `assert_no_transfers()`

Context manager that raises `AssertionError` with the counter's `report()` if
the block makes a transfer through CuNumpy. It yields the `TransferCounter`
too. An exception raised inside the block propagates as it is:

```python
def test_time_step_stays_on_the_device():
    with xp.assert_no_transfers():
        propagator(dt)
### `as_device_array(value, dtype=None, ndim=None, *, name=None)`

The "reference or copy once" rule for building CUDA argument objects
(`CudaArguments` subclasses, `CudaStruct` values). Call it once when the
argument object is built, never per kernel call:

* a CuPy array that already has `dtype` (any dtype if `dtype` is `None`) and
  is C-contiguous is returned unchanged, the same object without a copy, so
  kernels write into the caller's array;
* anything else becomes one C-contiguous device copy,
  `cupy.ascontiguousarray(cupy.asarray(value, dtype))`: tuples and lists
  (`degree = (3, 3, 3)`), host NumPy arrays (one explicit transfer at build
  time), device arrays of another dtype, and non-contiguous views.

The result passes the pointer checks of `CudaKernel` and `CudaStruct`. On the
NumPy backend it raises `RuntimeError`: device arguments are only built when
running on CuPy, and host data is never copied to the device implicitly. If
`ndim` is given and the result has another number of dimensions, it raises
`ValueError`; `name` is the argument name used in error messages.

```python
class DeviceParticles(xp.CudaArguments):
    def __init__(self, markers, degree):
        self.markers = xp.as_device_array(markers, np.float64, ndim=2, name="markers")
        self.degree = xp.as_device_array(degree, np.int32, ndim=1, name="degree")
        super().__init__(self.markers, self.degree, self.markers.shape[0])
```

## Random numbers and dtype

### `get_rng(seed=None)`

Returns a NumPy or CuPy `Generator` matching the active backend:

```python
rng = xp.get_rng(seed=7)
samples = rng.uniform(size=100)
```

The generator APIs are similar, but seeds do not guarantee identical random
sequences across NumPy and CuPy.

### `default_float_dtype()`

Returns the active backend module's `float64` dtype object. Pass it to array
creation when code requires an explicit precision:

```python
values = xp.asarray([0.1, 0.2], dtype=xp.default_float_dtype())
```

## CUDA devices and memory

### `cupy_available()`

Returns whether CuPy can be imported and reports itself functional. The result
is cached for the process. This checks availability, not whether every
GPU-specific operation will succeed later.

### `device_count()`

Returns the number of visible CUDA devices. Returns zero when CuPy/CUDA is
unavailable or querying the runtime fails. This is independent of the active
backend, so it may return a positive number while NumPy is selected.

### `set_device(device_id)`

Selects a CUDA device when CuPy is active; it is a no-op on NumPy. The device
must be valid for the current CUDA process.

### `set_device_for_rank(rank, devices_per_node=None)`

Selects a device using `rank % devices_per_node` and returns its ID. If
`devices_per_node` is omitted, it uses `device_count()`. When no devices are
visible, it returns `0` without selecting a device. This helper assumes
contiguous rank-to-device mapping on each node, suitable for a common
one-rank-per-GPU MPI layout. Use `set_device()` directly when the scheduler's
mapping differs:

```python
device_id = xp.set_device_for_rank(mpi_rank)
```

### `local_rank()`

The rank of the process within its node, read from the environment variables
that MPI launchers export (Open MPI, MVAPICH2, Intel MPI/MPICH, PMI, Cray
PALS, Slurm, `LOCAL_RANK`), or `0` if none is set. The launcher sets them
before `MPI_Init`, so this works before MPI is initialized and without
importing `mpi4py`.

### `bind_local_device()`

Selects device `local_rank() % device_count()` for this process and creates its
CUDA context. Returns the device id, or `None` on the NumPy backend or without
devices. Call it before `MPI_Init` (before importing `mpi4py.MPI`), so that a
CUDA-aware MPI sees the right device; otherwise all ranks of a node would use
device 0. If the launcher gives each rank its own device through
`CUDA_VISIBLE_DEVICES`, each process sees one device and selects it:

```python
import cunumpy as xp

xp.set_backend("cupy")
xp.bind_local_device()
from mpi4py import MPI  # initializes MPI after the device is bound
```

Unlike `set_device_for_rank()`, it needs no MPI rank, and it uses the rank
within the node rather than assuming contiguous ranks per node.

### `mpi_is_cuda_aware(comm=None, *, method="probe")`

Checks whether the MPI library can send and receive device (CuPy) buffers,
which needs a CUDA-aware MPI build; with a plain build, passing a CuPy array
to MPI segfaults or silently sends garbage. Returns `False` on the NumPy
backend and without a functional CuPy, without importing `mpi4py`: the
question only makes sense with device buffers. `comm` defaults to
`mpi4py.MPI.COMM_WORLD`, and `mpi4py` is imported only then.

The check is collective: every rank of `comm` must call it, and all ranks get
the same result. Each rank sends a tiny device array to rank
`(rank + 1) % size` and receives from `(rank - 1) % size` with `Sendrecv`
(after `synchronize_for_mpi()`; with a single rank, it sends to itself), checks
the received values, and the ranks combine their outcomes with
`allreduce(op=LAND)`. Any exception in the exchange, on any rank, gives
`False`. Only `method="probe"` exists: `mpi4py` does not expose the library
query (`MPIX_Query_cuda_support`) and the library version string is not a
reliable indicator.

An MPI library that is not CUDA-aware may also read the device address as a
host address and crash the process. A segfault inside this call therefore
means the same thing as `False`. Call it once at startup, after
`bind_local_device()` and `MPI_Init`, before any communication of device
buffers.

### `require_cuda_aware_mpi(comm=None)`

Raises `RuntimeError`, explaining how to get a CUDA-aware build (Open MPI
`--with-cuda`, MPICH with a CUDA-enabled UCX, the site's CUDA-aware MPI
module), if `mpi_is_cuda_aware(comm)` returns `False` on the CuPy backend.
No-op on the NumPy backend. The complete startup sequence for one rank per
GPU:

```python
import cunumpy as xp

xp.set_backend("cupy")
xp.bind_local_device()  # 1. select the GPU, before MPI_Init
from mpi4py import MPI  # 2. MPI_Init, on the bound device

xp.require_cuda_aware_mpi()  # 3. clear error instead of a segfault later

xp.synchronize_for_mpi(send, recv)  # 4. before every MPI call with device buffers
MPI.COMM_WORLD.Sendrecv(send, dest, recvbuf=recv, source=source)
```

### `synchronize_for_mpi(*arrays)`

Waits for the work pending on the current stream if at least one of `arrays`
is a CuPy array; `None` entries and host arrays are ignored, so it costs
nothing for host buffers and on the NumPy backend. Call it before every MPI
call that sends or receives device buffers: CuPy launches kernels
asynchronously and MPI knows nothing about CUDA streams, so a buffer that a
kernel is still writing would be sent as it is at that moment, without an
error:

```python
xp.synchronize_for_mpi(send_buffer, recv_buffer)
comm.Sendrecv(send_buffer, dest, recvbuf=recv_buffer, source=source)
```

No synchronization is needed after MPI returns: kernels launched afterwards see
the received data.

### `memory_info()`

Returns `(free_bytes, total_bytes)` reported by the CUDA runtime for the
active device, or `None` on NumPy. The values cover the device, not only
allocations owned by CuPy.

### `free_memory()`

Releases currently free blocks in CuPy's device and pinned-host memory pools.
It is a no-op on NumPy. It does not release blocks still referenced by live
arrays. CuPy normally caches freed allocations for reuse, so cached memory
does not necessarily indicate a leak.

### `pin_memory(array)`

Copies a host array to page-locked (pinned) host memory. Pinned memory can
improve host/device transfer throughput in suitable asynchronous workloads.
Raises `ImportError` if CuPy is unavailable. If the input may be a CuPy array,
first transfer it with `to_numpy()`.

### `synchronize()`

Waits for queued work on the current CUDA device to finish. This is useful
before reading asynchronously computed results from host code. It is a no-op
on NumPy.

### `stream()`

Context manager that creates a non-blocking CuPy stream and yields it. Work
issued in the block is enqueued on that stream. On NumPy, yields `None` and
does nothing:

```python
with xp.stream() as work_stream:
    result = xp.to_cupy(host_values) * 2

work_stream.synchronize()  # on CuPy; the yielded value is None on NumPy
```

Do not call methods on the yielded value without checking the backend. Use
`xp.synchronize()` for code that should work on both backends.

## Profiling

CUDA kernels run asynchronously: a wall-clock timer around a launch measures
the launch, not the kernel, and regions of an application profiler are not
visible to `nsys`. These helpers address both; they are no-ops (or plain
timers) on NumPy, so instrumented code runs unchanged on both backends.

### `nvtx_range(name, color=None)`

Context manager and decorator marking a code region as an NVTX range. On CuPy
it calls `cupy.cuda.nvtx.RangePush(name)` on entry and `RangePop()` on exit
(also when the block raises), so the region appears on the `nsys`/Nsight
timeline next to the kernels launched inside it. `color` is an optional index
into NVTX's colour table (the `id_color` argument of `RangePush`). On NumPy,
or if NVTX is not available in the CuPy build, it does nothing. The same
instance may be nested or re-entered, e.g. as the decorator of a recursive
function.

```python
with xp.nvtx_range("push markers"):
    kernel(markers, dt, n_threads=n)


@xp.nvtx_range("accumulate")
def accumulate(particles, grid):
    ...
```

### `timed_region(name, *, sync=True)`

Context manager timing a code region, including the device work it queues.
It yields a `Timing` object whose `elapsed` (seconds, from
`time.perf_counter`) is set when the block exits, also when it raises. On
CuPy it synchronizes the device on entry, so earlier queued work is not
charged to the region, and, if `sync` is true, again on exit before reading
the clock; `synced` records whether that happened. It also pushes an
`nvtx_range()` of the same name. On NumPy it is a plain timer and `synced`
is `False`. With `sync=False` only the host time is measured.

```python
with xp.timed_region("push markers") as timing:
    kernel(markers, dt, n_threads=n)

print(f"{timing.name}: {timing.elapsed:.4f} s, synced={timing.synced}")
```

### `Timing`

Dataclass returned by `timed_region()`, with the fields `name` (`str`),
`elapsed` (`float`, `None` until the block exits) and `synced` (`bool`).

## `PyccelKernel`

### Constructor

```python
xp.PyccelKernel(
    kernel,
    use_cupy=None,
    object_modules=(),
    is_array=None,
    outputs=None,
)
```

Wraps a callable expecting host NumPy arrays so it can be used with CuPy
arrays. This can adapt a Pyccel-compiled kernel or an ordinary Python
callable. CuNumpy does not compile the callable or import Pyccel.

When no conversion is needed, the original callable is invoked directly. If
conversion is needed, CuNumpy recursively replaces CuPy arrays in supported
arguments with NumPy host copies, calls the kernel, copies in-place updates
back to the corresponding CuPy arrays, and converts returned NumPy arrays to
CuPy arrays.

### Parameters

* `kernel`: callable that accepts the host-side arguments.
* `use_cupy`: `None` (default) chooses conversion for each call when the
  global backend is CuPy or a CuPy array is present. `True` forces conversion;
  `False` disables it.
* `object_modules`: module prefixes whose instances should be shallow-copied
  and traversed by attributes. For example,
  `object_modules=("my_project.",)`.
* `is_array`: predicate for host array values to convert back to CuPy. The
  default is `isinstance(value, numpy.ndarray)`.
* `outputs`: sequence of arguments the kernel may write to. Entries are
  positional indices or keyword names. If omitted, every converted array is
  copied back.

### Example and output declarations

```python
def scale_and_shift(scale, values, out):
    out[:] = scale * values + 1
    return out

kernel = xp.PyccelKernel(scale_and_shift, outputs=(2,))

with xp.use_backend("cupy"):
    values = xp.arange(8, dtype=xp.float64)
    out = xp.empty_like(values)
    result = kernel(2.0, values, out)
```

`outputs=(2,)` marks only `out` for copy-back. Read-only `values` is copied
to the host for the call but not transferred back. Declare a positional
argument by index (negative indices count from the end), and a keyword
argument by its name, such as `outputs=("out",)` for `kernel(..., out=out)`.
The forms are not interchangeable because compiled builtins may not expose a
Python signature. Invalid declarations raise `IndexError` or `KeyError`.

An incorrect declaration is a correctness bug: if the kernel writes an
argument that was not declared, the device array will not receive that
update. Use `outputs=()` only when no converted input is mutated. Containers
and selected objects can be declared as outputs; every nested supported
array is then copied back. If an array is reachable through multiple paths,
declaring any path that includes it is sufficient.

### Supported containers, aliasing, and returns

Tuples, lists, and dictionaries are traversed recursively. Objects are
traversed only when their class module starts with one of the configured
`object_modules` prefixes; those objects are shallow-copied, and their
attributes are converted on the copy. Other objects are passed to the kernel
unchanged.

The wrapper memoizes conversions within a call. If the same device array is
passed more than once or appears inside a supported container, the host kernel
sees the same NumPy array object, preserving aliasing. Supported reference
cycles terminate safely. Returned NumPy arrays (and arrays inside tuples or
lists) are converted back using `is_array`; dictionaries in return values are
not recursively converted. On the NumPy path, the original return value and
normal Python mutation and exception behavior are preserved.

## `CudaKernel`

### Constructor

```python
xp.CudaKernel(
    source,
    name,
    *,
    block_size=128,
    options=(),
    include_dirs=(),
    source_dir=None,
    structs=(),
    template_args=None,
    check_signature=True,
    debug=None,
)
xp.CudaKernel.from_file(path, name=None, *, suffix="_cuda.cu", **kwargs)
```

Wraps the `__global__` function `name` in the CUDA C `source` (declared
`extern "C"`, unless it is a template). The kernel is compiled with NVRTC
through CuPy on the first call (or by `compile()`), and cached, also on disk by
CuPy. CuPy is imported only then, so kernels can be created and their
signatures parsed without CuPy; `compile()` raises `RuntimeError` without a
GPU.

`from_file` reads the source from a file; the kernel name defaults to the file
name without `suffix` (`axpy_cuda.cu` -> `axpy`), and the directory of the file
is added to the include directories and is the `source_dir`.

### Parameters

* `block_size`: threads per block, an integer for 1D launches or a tuple of 1
  to 3 integers, e.g. `(16, 16)`; at most 1024 threads in total.
* `options`: additional NVRTC options, e.g. `("-std=c++17",)`.
* `include_dirs`: directories for `#include`, passed as `-I<dir>`.
* `source_dir`: the directory the source was read from, where
  `#include "..."` files are looked up first (set by `from_file`).
* `structs`: `CudaStruct` types that the kernel takes as parameters (by
  value), see `CudaStruct` below.
* `template_args`: template arguments if `name` is a function template, see
  "Templates and generated kernels" below.
* `check_signature`: parse the signature and check every call against it
  (default). Raises `ValueError` if the signature cannot be parsed, e.g. with
  macros or pointers to pointers in the parameter list; pass `False` to launch
  with the arguments as they are, like `cupy.RawKernel`.
* `debug`: `None` (default) follows the global debug setting, `True`/`False`
  fix it for this kernel, see "Debugging" below.

Properties: `name`, `expression` (`name`, or the template instantiation such
as `"scale<double, 3>"`), `source`, `block_size`, `options`, `structs`,
`template_args`, `signature`, `is_compiled`, `debug`.
as `"scale<double, 3>"`), `source`, `block_size`, `options`, `include_dirs`,
`source_dir`, `included_headers`, `structs`, `template_args`, `signature`,
`is_compiled`.

### Included headers and the compile cache

CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), keyed on the
source string and the compiler options only: a file pulled in through
`#include "..."` is not part of the key, so editing a shared `.cuh` header
would not recompile the kernels that include it. `CudaKernel` therefore
resolves the quoted includes of its source when it compiles and adds a define
with a hash of their contents to the options:

```python
kernel = xp.CudaKernel.from_file("push/push_cuda.cu", include_dirs=[src_root])
kernel.included_headers   # (Path('push/helpers.cuh'), Path('.../common.cuh'))
kernel.options            # ('-Ipush', '-I<src_root>')
kernel.compile_options()  # options + ('-DCUNUMPY_INCLUDE_HASH=0x3f9a...',)
```

* `included_headers`: the header files the source includes with
  `#include "name"`, recursively, each once in order of first inclusion. A
  name is looked up relative to the including file (`source_dir` for the
  kernel source, the header's own directory for nested includes), then in
  `include_dirs` in order, like NVRTC does. System headers in angle brackets
  and includes that cannot be found are ignored (NVRTC reports the latter).
  Recomputed at every access, so it follows the files on disk.
* `compile_options()`: the options passed to CuPy at compile time: `options`
  plus `-DCUNUMPY_INCLUDE_HASH=0x<hash>` if the source includes any header,
  where the hash covers the contents of `included_headers` (not their paths).
  A changed header gives another define, hence another cache entry. Sources
  without quoted includes never touch the file system.

The two building blocks are available on their own:

* `xp.resolve_includes(source, include_dirs=(), *, base_dir=None)`: the
  resolved header paths of a source, as a list.
* `xp.include_hash(paths)`: the first 16 hex digits of the SHA-256 digest of
  the contents of the files, in order.

### Calling

```python
kernel(*args, n_threads=None, grid=None, block=None, shared_mem=0, stream=None)
```

Launches the kernel on `stream` (the current stream if `None`). The launch
shape is given either by `n_threads` or by `grid`:

* `n_threads`: number of threads, an integer or a tuple of 1 to 3 integers such
  as `(nx, ny)`. The grid is `ceil(n_threads / block)` per dimension. With a 1D
  `block_size` and multi-dimensional `n_threads`, the block is
  `(block_size, 1, ...)`.
* `grid`: number of blocks per dimension, instead of `n_threads`.
* `block`: block shape for this call, instead of `block_size`.
* `shared_mem`: dynamic shared memory per block in bytes, for
  `extern __shared__` arrays.

Nothing is launched if the grid has a zero dimension (e.g. `n_threads=0`).
`kernel.launch_shape(n_threads=None, *, grid=None, block=None)` returns the
`(grid, block)` a call would use, e.g. to size a per-block output:

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
block_sum = xp.CudaKernel(BLOCK_SUM, "block_sum", block_size=128)
(n_blocks,), _ = block_sum.launch_shape(x.size)
partial = xp.zeros(n_blocks)
block_sum(x, partial, x.size, n_threads=x.size, shared_mem=128 * 8)
```

In a 2D kernel, use `blockIdx.y`/`threadIdx.y` for the second dimension and
launch with `n_threads=(nx, ny)` and, e.g., `block_size=(16, 16)`.

### Argument checks

The arguments are prepared by `kernel.prepare_args(*args)`:

* arguments with a `__cuda_args__()` method are replaced by the values it
  returns (see `CudaArguments` and `CudaStruct` below);
* with a checked signature, the number of arguments must match, and
  * pointer parameters take C-contiguous CuPy arrays whose dtype matches the
    pointed-to type (any dtype for `void*`); host arrays raise `TypeError`,
    they are never copied to the device, and so do non-contiguous views such
    as `a[:, 0:3]`, which the kernel would read as a flat buffer (build the
    arrays with `as_device_array()` or `cupy.ascontiguousarray()`);
  * struct parameters take values of that `CudaStruct`;
  * Python scalars are cast to the declared type: `int` into integer (with a
    range check, `OverflowError`), floating-point and complex parameters,
    `float` into floating-point and complex parameters, `bool` into boolean
    and integer parameters; anything else raises `TypeError`;
  * NumPy scalars are passed as they are if their dtype matches, cast if the
    cast is safe (e.g. `np.float32` into `double`), and raise `TypeError`
    otherwise (e.g. `np.float64` into `float`).

This matters because `cupy.RawKernel` reads each argument with the size
declared in the signature and does not check types: an integer passed to a
`double` parameter, or a `double` passed to a `float` parameter, arrives as a
wrong value without an error. The checks cost about 0.3 µs per argument (about
10 µs for a kernel with 29 arguments, measured on an H100 node, where the launch
itself costs about as much), which is negligible for kernels that run for
100 µs or more. For very short kernels called in a hot loop, pass
`check_signature=False` once the calls are known to be correct.

C types are mapped to NumPy dtypes as on Linux (LP64): `int` is `int32`,
`long` and `long long` are `int64`, `float` is `float32`, `double` is
`float64`, `complex<double>` is `complex128`; fixed-width types such as
`int64_t` and `size_t` are supported too. `xp.ctype_of(dtype)` gives the C
type of a dtype (`xp.ctype_of(np.float64) == "double"`), e.g. to generate
source. `xp.parse_cuda_signature(source, name, *, structs=(),
template_args=None)` returns the parsed parameters (`CudaParameter` tuples of
`name`, `ctype`, `dtype`, `pointer`, `struct`).

### Templates and generated kernels

A function template is instantiated with `template_args`: C types (or NumPy
dtypes, converted with `ctype_of`) for type parameters, integers or booleans
for non-type parameters. The template parameters are substituted into the
signature, so calls are checked as for any other kernel:

```python
SCALE = r"""
template <typename T, int N>
__global__ void scale(T* x, T factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = factor * x[i] * (T)N;
}
"""
scale_f64 = xp.CudaKernel(SCALE, "scale", template_args=(np.float64, 3))
scale_f64(x, 2.0, x.size, n_threads=x.size)  # instantiation scale<double, 3>
```

For kernels whose source is generated per variant (e.g. per number of
dimensions and dtype), `CudaKernelVariants` creates and caches one kernel per
key:

```python
matvec = xp.CudaKernelVariants(
    lambda ndim, dtype: xp.CudaKernel(make_source(ndim, xp.ctype_of(dtype)), "matvec")
)
matvec.get(3, np.float64)(mat, x, out, n_threads=out.size)  # created once
matvec.compile_all([(3, np.float64), (3, np.complex128)])   # at setup
```

`get(*key)` calls the factory the first time a key is used; `keys()`,
iteration and `len()` give the variants created so far; `compile_all(keys=())` creates the
given variants and compiles all of them.

### Debugging

```python
xp.set_cuda_debug(enabled)
xp.get_cuda_debug()
xp.cuda_debug(enabled=True)   # context manager
xp.CudaKernel(..., debug=None)
kernel.debug_active()
kernel.compile_options()
xp.DEBUG_OPTIONS  # ("-lineinfo", "-DCUNUMPY_BOUNDS_CHECK")
```

Kernel launches are asynchronous: a CUDA error such as an illegal memory
access or a launch failure is reported by the next operation that
synchronizes (a `.get()`, an MPI call, ...), which may be far from the kernel
that caused it. In debug mode, a `CudaKernel`

* is compiled with `-lineinfo` (source line information for
  `compute-sanitizer` and profilers) and `-DCUNUMPY_BOUNDS_CHECK` (bounds
  checks in cunumpy's array views, and available to your own `#ifdef`s),
  unless the option is already among its `options`. `-G` (device debug
  symbols) is not added, because NVRTC does not support it;
* synchronizes the stream after every launch (the `stream` passed, else the
  current one), so an error is raised at the launch that caused it, as a
  `RuntimeError` that names the kernel and its grid and block, with the CuPy
  error chained.

Debug mode is enabled globally with `xp.set_cuda_debug(True)`, temporarily
with the context manager `xp.cuda_debug()`, or before starting Python with
the environment variable `CUNUMPY_CUDA_DEBUG=1` (`true`, `yes` and `on` work
too); `xp.get_cuda_debug()` returns the current setting. A kernel created with
`debug=None` (the default) reads the global setting at every launch, so
enabling it also affects kernels created earlier; `debug=True` or
`debug=False` fix the mode for one kernel. Only the compile options are fixed
at compile time: a kernel compiled before debug mode was enabled keeps its
options, so call `compile()` after enabling, or create the kernels after
enabling. `kernel.debug_active()` tells whether debug mode applies to a
kernel now, and `kernel.compile_options()` returns the options a compilation
now would use.

```python
with xp.cuda_debug():
    kernel = xp.CudaKernel(SOURCE, "kernel")
    kernel(x, y, n, n_threads=n)  # RuntimeError: CUDA error after launching kernel 'kernel' ...
```

The `RuntimeError` says which kernel failed, not where. The next step is
NVIDIA's memory checker, which reports the faulting source line (thanks to
`-lineinfo`) and also finds out-of-bounds accesses that do not crash:

```bash
CUNUMPY_CUDA_DEBUG=1 compute-sanitizer python -m pytest tests/unit/test_my_kernel.py
```

Note that after an illegal memory access the CUDA context is unusable; the
process (or the pytest run) has to be restarted.

## `CudaStruct`

```python
Particles = xp.CudaStruct(
    "Particles",
    [("x", "double*"), ("v", "double*"), ("n", "int"), ("charge", "double")],
)
source = Particles.declaration + r"""
extern "C" __global__ void push(Particles p, double dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < p.n) p.x[i] += dt * p.charge * p.v[i];
}
"""
push = xp.CudaKernel(source, "push", structs=[Particles])
push(Particles(x=x, v=v, n=x.size, charge=-1.0), 0.1, n_threads=x.size)
```

A C struct passed to kernels by value. It groups arguments, e.g. all arrays
describing a set of particles, into one kernel parameter, so adding a field
changes one definition instead of every kernel signature.

`CudaStruct(name, fields)` takes the fields as `(name, C type)` pairs; scalar
fields and pointers to the scalar types above (or `void*`) are supported.

* `declaration`: the C definition of the struct, to put in the CUDA source
  or a header.
* `dtype`: the NumPy structured dtype with the memory layout of the C struct
  (C alignment and padding; pointers stored as 64-bit device addresses).
* `fields`: the parsed fields (`CudaParameter` tuples).
* `check_source(source)`: raises `ValueError` if `source` defines the struct
  with other fields; a kernel created with `structs=[...]` does this check.
* Calling the struct with keyword arguments, one per field, packs the values:
  pointer fields take C-contiguous CuPy arrays of the declared dtype (never
  copied), scalar fields are checked and cast like scalar kernel arguments.

The result is a `CudaStructValue`: it keeps references to the arrays it points
to (the packed struct only holds their addresses, so keep the value alive while
the kernel may run), gives access to the field values with
`value["field"]`, holds the packed struct in `value.packed`, and is flattened
into it when passed to a kernel.

## `CudaArguments`

```python
class Particles(xp.CudaArguments):
    def __init__(self, positions, velocities):
        self.positions = positions
        super().__init__(positions, velocities, positions.shape[0])

kernel(dt, Particles(x, v), n_threads=x.shape[0])
```

Base class for objects passed to a `CudaKernel` as one argument that stands
for several kernel parameters. `CudaArguments(*values)` stores the values;
`__cuda_args__()` returns them. Subclassing is optional: any object with a
`__cuda_args__()` method returning a tuple is flattened. This lets an
application keep its host argument objects (e.g. Pyccel classes holding NumPy
arrays) and matching device argument objects that reference the same data on
the device, and pass either to the same call. A `CudaArguments` object may also
return struct values (`CudaStructValue.packed`) among its values.

## `KernelArguments`

```python
class ParticleArguments(xp.KernelArguments):
    def __init__(self, particles):
        self._particles = particles
        self._host = None
        self._cuda = None

    def __host_args__(self):
        if self._host is None:  # e.g. a Pyccel class holding NumPy arrays
            self._host = MarkerArguments(self._particles.markers)
        return self._host

    def __cuda_args__(self):
        if self._cuda is None:  # device arrays and scalars, flattened
            markers = self._particles.markers
            self._cuda = (markers, markers.shape[0], markers.shape[1])
        return self._cuda


class Particles:
    @property
    def kernel_args(self):
        if self._kernel_args is None:
            self._kernel_args = ParticleArguments(self)
        return self._kernel_args


push(particles.kernel_args, dt, n_threads=n)  # same call on both backends
```

Base class for argument objects that have a host form and a device form. A
group of arrays, e.g. the marker data of a particle species, is typically
passed to the host kernel as one object holding NumPy arrays (a Pyccel class)
and to the CUDA kernel as several device arrays and scalars. `KernelArguments`
lets one object stand for both, so a `Kernel` call never branches on the
backend:

* `__host_args__()` returns the single object the host kernel receives in that
  position. `Kernel` (on the NumPy backend) and `PyccelKernel` (always, so the
  `missing_cuda="fallback"` path works with the same objects) replace the
  argument by this value.
* `__cuda_args__()` returns the tuple of CUDA kernel arguments the object
  stands for, the `CudaArguments` protocol above; `CudaKernel` flattens it.

Only top-level positional and keyword arguments are resolved, not objects
nested in tuples, lists or dicts. The check is made on the type, like for
`__cuda_args__`: an instance attribute named `__host_args__` (e.g. a stored
object) is not treated as the protocol. Subclassing is optional; both methods
of the base class raise `NotImplementedError`, so a subclass overrides the ones
it supports (a `KernelArguments` without `__cuda_args__` raises when it reaches
a `CudaKernel`).

In the example above both forms are built lazily on first access and cached,
so a CPU run never builds device arguments and a GPU run never builds the host
object. The owner is responsible for invalidating the cache (setting the
stored forms to `None`, or replacing the `ParticleArguments` object) when its
arrays are replaced, e.g. after resizing, `deepcopy` or unpickling.

### `resolve_host_args(args, kwargs=None)`

Returns `(args, kwargs)` with every top-level argument whose type defines a
callable `__host_args__()` replaced by its result; everything else is passed
through untouched. `Kernel` and `PyccelKernel` call it before the host kernel;
it is exported for code that calls host kernels by other means:

```python
args, kwargs = xp.resolve_host_args((particles.kernel_args, dt), {"out": out})
host_push(*args, **kwargs)
```

## `Kernel`

```python
xp.Kernel(
    host_kernel,
    cuda_kernel=None,
    *,
    name=None,
    missing_cuda="raise",
    cuda_path=None,
    host_options=None,
)
```

A host kernel (a `PyccelKernel`; other callables are wrapped in one) and its
CUDA counterpart. `kernel.get_kernel()` returns the host kernel on the NumPy
backend and the CUDA kernel on the CuPy backend; call it once at setup to fail
early if a CUDA kernel is missing.

```python
kernel(*args, n_threads=None, grid=None, block=None, shared_mem=0, stream=None)
```

calls the kernel of the active backend. The launch arguments are passed to the
CUDA kernel (`n_threads` or `grid` is required there) and ignored by the host
kernel. Arguments implementing `KernelArguments` are replaced by their
`__host_args__()` on the host path and flattened via `__cuda_args__()` on the
CUDA path. `kernel.compile()` compiles the CUDA kernel now and returns whether
there is one.

Without a CUDA kernel on the CuPy backend, `missing_cuda="raise"` raises
`NotImplementedError` (naming `cuda_path`, if given), and
`missing_cuda="fallback"` calls the host kernel through `PyccelKernel`, which
copies the arrays to the host and back at every call (a `RuntimeWarning` is
emitted once).

`host_options` are keyword arguments for the `PyccelKernel` that wraps a plain
callable `host_kernel`, e.g. `{"object_modules": ("my_package.",), "outputs":
(2,)}`. They matter for the fallback: `object_modules` lets it find the device
arrays inside application objects, and `outputs` limits the copies back to the
device. Passing `host_options` together with a `PyccelKernel` raises
`ValueError`; configure that `PyccelKernel` directly.

Properties: `name`, `host_kernel`, `cuda_kernel`, `has_cuda`, `missing_cuda`,
`cuda_path`.

## `KernelCatalog`

```python
catalog = xp.KernelCatalog.from_package(
    package,
    *,
    host_suffix="_kernels",
    cuda_suffix="_cuda.cu",
    missing_cuda="raise",
    host_options=None,
    include_dirs=None,
    **cuda_options,
)
kernel = catalog["push"]
```

A read-only mapping from names to `Kernel` objects. `from_package` scans the
subfolders of `package`: for every folder `<name>` containing the module
`<name><host_suffix>.py`, the function `<name>` of that module is the host
kernel, and `<name><cuda_suffix>` in the same folder, if present, is the CUDA
kernel (`__global__` function `<name>`). Typically called in the package's
`__init__.py`:

```text
my_kernels/
├── __init__.py              # catalog = xp.KernelCatalog.from_package(__name__)
├── push/
│   ├── push_kernels.py      # def push(...): ...
│   └── push_cuda.cu         # __global__ void push(...)
└── deposit/
    └── deposit_kernels.py   # no CUDA kernel yet
```

* `host_options`: `PyccelKernel` options for the host kernels (see `Kernel`),
  the same for all kernels or a function of the kernel name, e.g.
  `lambda name: {"outputs": OUTPUTS[name]}`.
* `include_dirs`: include directories of the CUDA kernels, in addition to
  each kernel's own folder. By default the source root of the top-level
  package (the directory containing it), so that a kernel of
  `my_pkg.kernels` can `#include "my_pkg/common.cuh"`. Headers found this
  way take part in the compile cache key, see "Included headers and the
  compile cache" under `CudaKernel`.
* `cuda_options`: passed on to `CudaKernel.from_file`, e.g. `block_size` or
  `structs`.

`catalog.without_cuda` lists the kernels still to port.
`catalog.compile_all()` compiles every CUDA kernel and returns their names;
call it at setup so that the first time step does not pay for compilation
(after the first run, CuPy loads the kernels from its disk cache).
`catalog.parity_cases()` returns the `(name, kernel)` pairs of the kernels
that have a CUDA kernel, for a parametrised parity test (see "Testing
utilities"). `KernelCatalog(kernels)` and `catalog.register(kernel, name=None)`
build a catalog by hand.

## Testing utilities

```python
from cunumpy.testing import (
    BACKENDS,
    assert_kernels_agree,
    device_function_kernel,
    requires_cupy,
)
```

`cunumpy.testing` holds helpers for testing kernels with pytest. It is not
imported by `import cunumpy` (so `xp.testing` remains NumPy's or CuPy's
`testing` module until `cunumpy.testing` is imported), and it imports pytest
only when one of its pytest objects is used, so `device_function_kernel` works
without pytest.

### `requires_cupy`, `BACKENDS`, `backend`

`requires_cupy` is `pytest.mark.skipif(not cupy_available(), reason="CuPy/GPU
not available")`, for tests that need a GPU. `BACKENDS` is
`["numpy", pytest.param("cupy", marks=requires_cupy)]`, so a test parametrised
with it runs on NumPy everywhere and on CuPy where a GPU is available:

```python
@pytest.mark.parametrize("backend", BACKENDS)
def test_norm(backend):
    with xp.use_backend(backend):
        assert xp.linalg.norm(xp.ones(4)) == 2.0
```

The `backend` fixture does the same and activates the backend for the test;
import it into a `conftest.py` (`from cunumpy.testing import backend`) or the
test module, then take `backend` as a test argument.

### `assert_kernels_agree(kernel, make_args, ...)`

```python
assert_kernels_agree(
    kernel,
    make_args,
    *,
    n_threads=None,
    grid=None,
    block=None,
    rtol=1e-12,
    atol=0.0,
    n_calls=1,
    outputs=None,
    seed=0,
)
```

Checks that the host and the CUDA version of a `Kernel` compute the same. For
each backend, `"numpy"` then `"cupy"`, the backend is activated with
`use_backend`, the positional arguments are built with `make_args(backend,
seed)` (a tuple or list; kernels take positional arguments only), the kernel is
called `n_calls` times (with `n_threads`, `grid` and `block` on CuPy), and the
arrays among the arguments are collected. The CUDA results are copied to the
host and compared with the host results using `numpy.testing.assert_allclose`
with `rtol` and `atol`; the `AssertionError` names the argument that differs.
The test is skipped (`pytest.skip`) without a GPU, and `ValueError` is raised
for a kernel without CUDA version. The host arrays are returned by argument
name (`"argument 0"`, `"argument 3.x"`) for further checks.

`make_args` runs with the backend active, so arrays created through `cunumpy`
land on it. NumPy and CuPy generators do not produce the same random sequence
from one seed, so build random data on the host and convert it:

```python
def make_args(backend, seed):
    x = xp.to_cunumpy(np.random.default_rng(seed).random(1000))
    return (x, 2.0, x.size)


def test_scale():
    assert_kernels_agree(catalog["scale"], make_args, n_threads=1000)
```

`outputs` selects the arguments to compare by index (negative indices count
from the end), like `PyccelKernel(outputs=...)`; by default the `outputs`
declared by the host kernel are used, and if it declares none, every argument.
An argument that is an array is compared directly. For a tuple, list, dict or
object argument (e.g. a `CudaArguments` object), the arrays it holds one level
deep are compared, plus the arrays in a container attribute of an object.

Together with `KernelCatalog.parity_cases()`, one test covers a catalog:

```python
MAKE_ARGS = {"scale": make_scale_args, "push": make_push_args}


@pytest.mark.parametrize("name, kernel", catalog.parity_cases())
def test_parity(name, kernel):
    assert_kernels_agree(kernel, MAKE_ARGS[name], n_threads=1000)
```

### `device_function_kernel(header_source, signature, ...)`

```python
device_function_kernel(
    header_source,
    signature,
    *,
    name=None,
    includes=(),
    n_threads_param="n",
    out_param="out",
    **cuda_kernel_options,
)
```

Generates an elementwise `extern "C" __global__` kernel that calls a
`__device__` function once per thread and returns it as a `CudaKernel`, so
device helpers (B-spline evaluation, mapping evaluation, small linear algebra)
can be run from Python on many inputs at once and compared with their host
versions. `header_source` is the CUDA source defining the function (or the
content of its header; `includes` adds `#include` lines before it, with quotes,
or with angle brackets for `"<cupy/complex.cuh>"`), and `signature` is its C
prototype, e.g. `"int find_span(const double* t, int p, double eta)"`.
Additional keyword arguments such as `include_dirs` and `block_size` go to
`CudaKernel`.

The generated kernel takes the parameters of the function in their order,
followed by the output array and the number of elements:

* a pointer parameter is kept as it is and passed unchanged to every call (an
  array shared by all threads);
* a scalar parameter `T x` becomes a device array `const T* x` of length `n`,
  and thread `i` calls the function with `x[i]`;
* the return value of thread `i` is stored in `out[i]` (`R* out`, with `R` the
  return type); a `void` function has no `out`;
* `int n` is the number of elements; threads `i >= n` do nothing.

The prototype above gives:

```c
extern "C" __global__ void find_span_kernel(
    const double* t, const int* p, const double* eta, int* out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = find_span(t, p[i], eta[i]);
}
```

```python
find_span = device_function_kernel(BSPLINES_CUH, "int find_span(const double* t, int p, double eta)")
find_span(t, p, eta, spans, eta.size, n_threads=eta.size)
```

The kernel is named `<function>_kernel` unless `name` is given. Scalar
parameters and return types are those `CudaKernel` supports; a struct or
pointer return type, an unsupported parameter type, or a parameter named like
`out_param` or `n_threads_param` raises `ValueError` (rename the generated
parameter in that case).

## Version

`xp.__version__` is the installed package version. When package metadata is
not available (for example, some source-tree imports), it is
`"0.0.0+unknown"`.
