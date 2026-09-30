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
    structs=(),
    template_args=None,
    check_signature=True,
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
is added to the include directories.

### Parameters

* `block_size`: threads per block, an integer for 1D launches or a tuple of 1
  to 3 integers, e.g. `(16, 16)`; at most 1024 threads in total.
* `options`: additional NVRTC options, e.g. `("-std=c++17",)`.
* `include_dirs`: directories for `#include`, passed as `-I<dir>`.
* `structs`: `CudaStruct` types that the kernel takes as parameters (by
  value), see `CudaStruct` below.
* `template_args`: template arguments if `name` is a function template, see
  "Templates and generated kernels" below.
* `check_signature`: parse the signature and check every call against it
  (default). Raises `ValueError` if the signature cannot be parsed, e.g. with
  macros or pointers to pointers in the parameter list; pass `False` to launch
  with the arguments as they are, like `cupy.RawKernel`.

Properties: `name`, `expression` (`name`, or the template instantiation such
as `"scale<double, 3>"`), `source`, `block_size`, `options`, `structs`,
`template_args`, `signature`, `is_compiled`.

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
* `cuda_options`: passed on to `CudaKernel.from_file`, e.g. `block_size`,
  `include_dirs` or `structs`.

`catalog.without_cuda` lists the kernels still to port.
`catalog.compile_all()` compiles every CUDA kernel and returns their names;
call it at setup so that the first time step does not pay for compilation
(after the first run, CuPy loads the kernels from its disk cache).
`KernelCatalog(kernels)` and `catalog.register(kernel, name=None)` build a
catalog by hand.

## Version

`xp.__version__` is the installed package version. When package metadata is
not available (for example, some source-tree imports), it is
`"0.0.0+unknown"`.
