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
are those of the currently selected `array-api-compat` NumPy or CuPy module.
They are copied into the `cunumpy` namespace and replaced when the backend
changes, so `xp.sum` costs the same as `numpy.sum` (a switch between NumPy and
CuPy takes some tens of microseconds; avoid switching inside a hot loop). CuNumpy does not wrap every operation individually. The available
operations and some details can therefore vary with the installed NumPy and
CuPy versions. In normal use, access those operations through the top-level
`cunumpy` namespace, commonly imported as `xp`.

For an explanation of what the compatibility module does and why CuNumpy
uses it, read [Why CuNumpy uses `array-api-compat`](array-api-compat.md).

NumPy and CuPy are not interchangeable for every function or object. A
function that needs to follow an input array's location should use
`get_array_module(array)` instead of assuming the global backend matches it.

## Module layout

The top level of `cunumpy` is the NumPy (or CuPy) namespace plus the functions
that select the backend and convert arrays. Everything else is in a submodule,
imported with `cunumpy` (`xp.kernels.CudaKernel`, `xp.rng.random_streams`, ...).
The submodules are named so that they do not hide a NumPy name (`rng`, not
`random`):

| Submodule | Backends | Contents |
|---|---|---|
| `cunumpy` | both | NumPy/CuPy namespace, backend selection, array inspection and conversion, `synchronize`, `host_call`, `evaluate_on_host`, `setup_on_host`, `scipy`, `require_version` |
| `cunumpy.kernels` | both | `Kernel`, `KernelCatalog`, `PyccelKernel`, `CudaKernel`, `CudaKernelVariants`, `MetalKernel`, `metal_available`, host implementations, `as_kernel_array`, `kernel_output`, `fuse` |
| `cunumpy.arguments` | CUDA only | `CudaArguments`, `CudaStruct`, `CudaStructArguments`, `CudaStructValue`, `write_cuda_header` |
| `cunumpy.cuda` | CUDA only | device selection and memory, `stream`, streams/events, `pin_memory`, debug mode, CUDA headers and source tools (`cuda_include_dir`, `parse_cuda_signature`) |
| `cunumpy.rng` | both | `random_streams`, `get_rng`, `philox_*` |
| `cunumpy.algorithms` | both | `morton_*`, `sort_by_key`, `segment_sum`, `compact_by_mask` |
| `cunumpy.mpi` | both | `mpi_buffer`, CUDA-aware MPI detection, `local_rank`, `synchronize_for_mpi` |
| `cunumpy.profiling` | both | `timed_region`, `nvtx_range`, `count_transfers`, `assert_no_transfers`, `TransferBudget` |
| `cunumpy.memory` | both | `HostStaging`, `HostCopy`, `DeviceMirror` |
| `cunumpy.petsc` | both | `petsc_vec` |
| `cunumpy.kernel_testing` | both | pytest helpers for host/CUDA kernel pairs (not imported by `import cunumpy`) |

"Both" means the functions work on NumPy and CuPy arrays; the functions of
`cunumpy.cuda` do nothing (or return `None`/`0`) on the NumPy backend.

Each name has one import path, through its submodule: `xp.kernels.CudaKernel`,
`xp.arguments.CudaStruct`, `xp.mpi.mpi_buffer`, `xp.rng.random_streams`. The top
level of `cunumpy` is the NumPy/CuPy namespace plus backend selection and array
conversion, so it never hides a NumPy or CuPy name (`xp.fuse` is CuPy's `fuse`;
cunumpy's is `xp.kernels.fuse`). Modules starting with `_` are private.

## Version

### `require_version(minimum)`

Raises `ImportError` if the installed cunumpy is older than `minimum`
(`xp.require_version("0.4.0")`). Only the numeric parts are compared; nothing
is checked when the version is unknown (not installed as a package).

## Backend selection

### `set_backend(backend, *, strict=False)`

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

With `strict=True`, requesting unavailable CuPy raises `RuntimeError` with the
availability failure reason, preserving the previous backend and namespace.
`use_backend(backend, strict=True)` offers the same guarantee.

### `backend_info()`

Returns a JSON-compatible dictionary with the selected backend, cached CUDA
availability and failure reason, dependency versions, and current device/runtime
information when available. Inspection errors are included in the result. It
does not change the backend or initialize MPI.

The initial backend is NumPy unless `CUNUMPY_BACKEND=cupy` is set before CuNumpy
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

F-contiguous CuPy inputs keep F order on the host; other CuPy inputs become
C-contiguous. Arbitrary device strides are not preserved. See
[Array ordering and strides](guides/array-ordering.md).

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

### `to_host_async(array)`

```python
pending = xp.to_host_async(residual_norm)  # a device scalar; returns at once
...                                         # queue the next iteration's kernels
if pending.ready() and pending.result() < tol:
    break
```

Starts copying a device scalar or small array to the host without waiting for
it, so that the host can keep queueing kernels (e.g. a convergence test read one
iteration late, without a sync per iteration). The copy runs on a separate
stream into page-locked memory, after the work queued so far on the current
stream. The returned `memory.HostCopy` has `ready()` (never waits) and
`result()` (waits only if the copy has not finished; a NumPy scalar for a 0-d
array, else a NumPy array). `count_transfers()` records a `to_host` event with
`blocking=False`, and a wait in `result()` as a `sync`.

On the NumPy backend the value is copied at once and `ready()` is always True;
on the fake CuPy too, and the copy is counted. For large arrays copied
repeatedly, use `memory.HostStaging`, which reuses its buffers.

### `algorithms.segment_sum(values, keys, n_segments, *, out=None)`

`out[k] = sum(values[i] for keys[i] == k)` on the backend of `keys`, with
one CUDA accumulation launch for all trailing components. `values` has shape
`(n, ...)`; a negative key drops the row; integer keys must be smaller than
`n_segments`. The result has shape `(n_segments, ...)`, keeps a floating-point
or complex dtype, and is `float64` otherwise. Optional `out` is overwritten,
must match shape/dtype, be writable and C-contiguous, and must not alias values.
Mixed backends raise. CUDA float16 accumulates in float32; floating-point atomic
accumulation order is not deterministic.

### `algorithms.SegmentPlan(keys, n_segments)`

Copies and validates keys once, then `plan.sum(values, out=None)` reuses them.
Changing the original keys does not change the plan. GPU preparation may
synchronize; repeated sums avoid key validation and per-column scalar reads.
A GPU plan requires its device current and values/output on that device.

### `algorithms.cell_offsets(sorted_cells, n_cells)`

Returns `n_cells+1` int64 offsets on the input backend. Cell k occupies the
half-open slice `[offsets[k], offsets[k+1])`. Missing cells have equal endpoints.
Input must be sorted integer IDs in `[0, n_cells)`; filter negative IDs first.

### `algorithms.segment_boundaries(sorted_keys)`

Returns `(unique_keys, starts, stops)` for runs of sorted integer keys. Supports
sparse, negative, and uint64 Morton keys. Starts/stops are int64 half-open indices.
Empty input returns three empty arrays. Validation and variable-length GPU
output may synchronize; prepare boundaries outside repeated operations.

### `algorithms.sort_by_key(keys, *arrays, axis=0)`

Stable argsort of the 1D `keys` (CuPy's radix sort on the device), applied to
every array along `axis`, in one call:

```python
keys, order, positions, charges = xp.algorithms.sort_by_key(keys, positions, charges)
# component-major (ncomp, N) markers: sort along the last axis of each array
keys, order, positions, weights = xp.algorithms.sort_by_key(
    keys, positions, weights, axis=-1
)
```

Returns `(keys[order], order, *(take(a, order, axis) for a in arrays))`,
`order` as `int64`. Equal keys keep their order, so the result is reproducible.
On NumPy, integer keys of more than 4096 entries are sorted by an LSD radix
sort on 16-bit digits (one stable `argsort` of `uint16` per digit, as many as
the key range needs), about ten times faster than the stable sort of 64-bit
integers.

### `algorithms.compact_by_mask(mask, *arrays, axis=0)`

Moves the rows where the boolean `mask` is True to the front of every array, in
place and in their original order, and returns how many there are. Typical use:
keep the live particles at the front of the marker arrays.

```python
n = xp.algorithms.compact_by_mask(alive, markers, weights)
markers, weights = markers[:n], weights[:n]
```

The rows after the first `n` are unspecified. The count is needed on the host,
so on CuPy each call synchronizes once. The mask and the arrays must be on the
same backend. `axis` is the axis of every array that the mask indexes; `-1`
compacts component-major arrays, `(ncomp, N)` next to `(N,)`, along their
marker axis:

```python
n = xp.algorithms.compact_by_mask(alive, positions, weights, axis=-1)
positions, weights = positions[:, :n], weights[:n]
```

## Count transfers

A transfer inside a time loop is the classic performance bug of a GPU port:
every step then waits for the device and copies an array. These helpers let a
test verify that a block of code does not transfer at all.

### `profiling.count_transfers(into=None)`

Context manager yielding a `TransferCounter` that records every host/device
transfer made through CuNumpy while the block runs, with the call site of
each:

```python
with xp.profiling.count_transfers() as counter:
    propagator(dt)

assert counter.total == 0, counter.report()
```

Six kinds of events are recorded:

* `to_host`: an actual device-to-host copy through conversion, mirror, staging,
  serial MPI, or host kernel helpers;
* `to_device`: an actual host-to-device copy through those helpers, including
  `as_device_array()` and `kernel_output()` copy-back;
* `kernel_conversion`: a `PyccelKernel` call that copied device arrays to the
  host (and back), one event per call, naming the kernel and the number of
  arrays converted;
* `fallback`: a `Kernel` without CUDA kernel calling its host kernel on the
  CuPy backend (`missing_cuda="fallback"`), one event per call, naming the
  kernel. Physical copies and a `kernel_conversion` marker are recorded separately;
* `device_copy`: a device-only dtype/layout conversion through CuNumpy helpers;
* `sync`: the host waited for the device: `xp.synchronize()`, the waits of the MPI
  helpers (`synchronize_for_mpi()`, `mpi_buffer()` staging) and of the CUDA debug
  mode, and, on the fake CuPy, a scalar read of a device array (`float(a)`,
  `int(a)`, `bool(a)`, `a.item()`, `a.tolist()`). They are in `counter.syncs` and in
  the report, but not in `total`.

Only real transfers count: `to_numpy()` of a NumPy array or `to_cupy()` of a
CuPy array records nothing. The counter has the attributes `to_host`,
`to_device`, `kernel_conversions`, `fallbacks` (counts per kind), `total`,
`device_copies`, `bytes_to_host`, `bytes_to_device`, and `bytes(kind)`.
`total` includes physical copies and conversion/fallback markers. Byte totals
include only physical copies, so markers do not double-count bytes.
`events` is a list of `TransferEvent(kind, description, where, nbytes=None,
blocking=True, implicit=False)`; `where` is the caller's `file:line` and `nbytes`
is None for markers/unknown sizes. `blocking` is False for the copies that the host
does not wait for (`to_host_async()`, `HostStaging.copy()`), and `implicit` is True
for the scalar reads of the fake CuPy (`float(a)`), which are `sync` events; the
report marks them `[async]` and `[implicit]`.
`kernel_conversion_calls` selects the conversion markers. `report()` returns
a multi-line string with the events grouped by kind and call site, with
counts:

```text
4 transfer(s) through cunumpy (3 to_host, 1 to_device, 0 kernel_conversion, 0 fallback, 0 device_copy)
  to_host (3):
    /home/me/sim/diagnostics.py:42: to_numpy(shape=(100000,), dtype=float64) (x3)
  to_device (1):
    /home/me/sim/setup.py:17: to_cupy(shape=(100000,), dtype=float64)
```

Blocks can be nested; every active counter sees the transfers made inside it.
`count_transfers(counter)` adds the events to an existing counter instead, e.g. to
accumulate over several calls; a counter that is already active is not added
again, so each event is counted once. Like every `contextlib.contextmanager`, it
is also a decorator: `@count_transfers(counter)`.
When no counter is active, the instrumentation costs a single check per call.
Like the backend selection, the active counters are process-wide state and
not thread-safe.

**Limitation:** only transfers made through CuNumpy are seen. Raw
`cupy.ndarray.get()`, `cupy.asarray(numpy_array)`, `numpy.asarray(cupy_array)`,
forwarded backend calls such as `xp.asarray()`, and implicit conversions inside
other libraries are not counted. Neither is `float(device_array)` (an implicit
sync) with the real CuPy, which cannot be observed from Python; the fake CuPy
counts it. Use `nsys` (or CuPy's profiling hooks) to find those.

### `profiling.assert_no_transfers(*, syncs=False)`

Context manager that raises `AssertionError` with the counter's `report()` if
the block makes a host/device transfer or host fallback through CuNumpy.
Device-only dtype/layout conversions are allowed, and so are syncs unless
`syncs=True`. It yields the `TransferCounter` too. An exception raised inside the block propagates as it is:

```python
def test_time_step_stays_on_the_device():
    with xp.profiling.assert_no_transfers():
        propagator(dt)
```

### `profiling.TransferBudget(*, started=True)`

Counts the transfers per phase of a program (the time step, the diagnostics,
the output) and checks a rule for each phase:

```python
budget = xp.profiling.TransferBudget(started=False)
model.integrate = budget.count("integrate")(model.integrate)  # a decorator
...                                    # setup: not counted
budget.start()
for step in range(n_steps):
    model.integrate(dt)
    with budget.phase("output"):       # or a context
        save(model)

budget.require("integrate", allow={"to_host": {"max_nbytes": 8}}, calls=n_steps)
budget.require("output", allow={"to_host": {"max_count": n, "max_total_bytes": b}})
budget.check()  # AssertionError with budget.report() if a rule is broken
```

`phase(name)` counts its block in phase `name` and yields the phase's
`TransferCounter`; `count(name)` is a decorator that does the same for every
call. The events of a phase accumulate over its calls (`budget[name]`,
`budget.phases`), and `budget.calls[name]` counts the calls. Phases nest: an
event is counted in the innermost phase only, and a phase entered again inside
itself (recursion) counts each event and call once. Until `start()` (with
`started=False`), and after `stop()`, phases run uncounted.

`require(phase, allow=None, *, ignore=("device_copy", "sync"), calls=None)`
sets the rule of a phase: the event kinds in `allow` are allowed, each with
optional limits, every other kind is forbidden except those in `ignore` (unless
they are in `allow`). The limits are `max_nbytes` (each event; an event of
unknown size breaks it), `max_count` and `max_total_bytes` (the whole phase),
and `blocking` and `implicit` (the events must have this value), e.g.
`{"to_host": {"max_nbytes": 8, "blocking": False}}` to allow only non-blocking
scalar copies, or `{"sync": {"implicit": False}}` with `ignore=("device_copy",)`
to reject the scalar reads of the fake CuPy. `calls` is the number of calls the
phase must have had. `violations()` lists the broken rules, with the offending
events and where they happened; `report()` shows every phase's events and the
broken rules; `check()` raises `AssertionError` with the report.

### `as_device_array(value, dtype=None, ndim=None, *, name=None)`

The "reference or copy once" rule for building CUDA argument objects
(`CudaArguments` subclasses, `CudaStruct` values). Call it once when the
argument object is built, never per kernel call:

* a CuPy array that already has `dtype` (any dtype if `dtype` is `None`) and
  is C-contiguous is returned unchanged, the same object without a copy, so
  kernels write into the caller's array;
* anything else is converted to a C-contiguous device array using
  `cupy.ascontiguousarray(cupy.asarray(value, dtype))`: tuples and lists
  (`degree = (3, 3, 3)`), host NumPy arrays (one explicit transfer at build
  time), device arrays of another dtype, and non-contiguous views.

Dtype and layout conversion can require separate device copies; each copy
through this helper appears in transfer accounting.

The result passes the pointer checks of `CudaKernel` and `CudaStruct`. On the
NumPy backend it raises `RuntimeError`: device arguments are only built when
running on CuPy, and host data is never copied to the device implicitly. If
`ndim` is given and the result has another number of dimensions, it raises
`ValueError`; `name` is the argument name used in error messages.

```python
class DeviceParticles(xp.arguments.CudaArguments):
    def __init__(self, markers, degree):
        self.markers = xp.as_device_array(markers, np.float64, ndim=2, name="markers")
        self.degree = xp.as_device_array(degree, np.int32, ndim=1, name="degree")
        super().__init__(self.markers, self.degree, self.markers.shape[0])
```

### `kernels.as_kernel_array(value, like, dtype=None, *, strided=False)`, `kernels.kernel_output(out, like, dtype=None, *, strided=False)`

For the arguments of a `Kernel` with `dispatch="arrays"`, whose choice follows
the arrays. `as_kernel_array` returns `value` on the side of `like` (a CuPy
array if `like` is one, a NumPy array otherwise), C-contiguous and with `dtype`
(any if `None`): `value` itself if it already is such an array, else a conversion,
moved across if needed (counted by `count_transfers()`). `kernel_output` is a
context manager yielding the buffer for an array the kernel writes: `out`
itself if `as_kernel_array` takes it unchanged, else a converted copy whose
contents are written into `out` (on its own side) when the block ends without
an error.

With `strided=True` a NumPy array for a host kernel is also taken unchanged
when it is in C order with gaps (positive strides, each at least the extent of
the next axis), e.g. the first `n` columns `storage[:, :n]` of a marker buffer.
Pyccel's wrappers take such arrays without a copy; a host kernel that needs
contiguous memory must not use it. F-ordered, transposed and negative-stride
arrays are still copied, and CuPy arrays are always made C-contiguous.

```python
with xp.kernels.kernel_output(result, like=field, dtype=float) as buffer:
    gather(
        xp.kernels.as_kernel_array(positions, like=field, dtype=float), field, buffer
    )
```

## Random numbers and dtype

### `rng.get_rng(seed=None)`

Returns a NumPy or CuPy `Generator` matching the active backend:

```python
rng = xp.rng.get_rng(seed=7)
samples = rng.uniform(size=100)
```

The generator APIs are similar, but seeds do not guarantee identical random
sequences across NumPy and CuPy.

### `rng.random_streams`

```python
xp.rng.random_streams.seed(42, rank=comm.Get_rank(), bit_generator="PCG64")
v = xp.rng.random_streams.normal(0.0, v_th, (n, 3))
rng = xp.rng.random_streams.generator()  # numpy or cupy Generator
own = xp.rng.random_streams.make_generator(seed)  # a component's own generator
```

One seeded random generator per process and backend, for reproducible MPI
runs: after `seed(value, rank)` every draw of the process comes from the stream
`(value, rank)` (a NumPy `SeedSequence` with the rank as spawn key), so the same
seed and number of ranks give the same results and the ranks' streams are
independent. `seed(None)` (or no seed) seeds from the operating system.
`bit_generator` selects the NumPy bit generator (`MT19937`, `PCG64`,
`PCG64DXSM`, `Philox`, `SFC64`); the CuPy generator uses CuPy's default.
`seed` also seeds NumPy's global state (and CuPy's on the CuPy backend) for code
that calls `np.random.*` directly.

`generator(backend=None)` returns the process generator of a backend (the
active one by default), created on first use. `make_generator(seed=None,
backend=None)` returns a separate generator for a component with a seed of its
own, and the process generator otherwise. `random`, `standard_normal`,
`normal` and `uniform` draw from the process generator (or `rng=`); `normal`
and `uniform` fall back to `standard_normal` and `random` for CuPy generators
without those methods. `xp.rng.RandomStreams()` makes an independent instance.

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

### `cuda.device_count()`

Returns the number of visible CUDA devices. Returns zero when CuPy/CUDA is
unavailable or querying the runtime fails. This is independent of the active
backend, so it may return a positive number while NumPy is selected.

### `cuda.set_device(device_id)`

Selects a CUDA device when CuPy is active; it is a no-op on NumPy. The device
must be valid for the current CUDA process.

### `cuda.set_device_for_rank(rank, devices_per_node=None)`

Selects a device using `rank % devices_per_node` and returns its ID. If
`devices_per_node` is omitted, it uses `device_count()`. When no devices are
visible, it returns `0` without selecting a device. This helper assumes
contiguous rank-to-device mapping on each node, suitable for a common
one-rank-per-GPU MPI layout. Use `set_device()` directly when the scheduler's
mapping differs:

```python
device_id = xp.cuda.set_device_for_rank(mpi_rank)
```

### `mpi.get_mpi(use_mpi=None)`, `mpi.launched_under_mpi()`

Importing `mpi4py.MPI` starts MPI (`MPI_Init`), which takes time and makes
every collective cost something even on one process. `launched_under_mpi()`
tells, from the environment that `mpirun`/`mpiexec`/`srun` set up and without
importing mpi4py, whether the process belongs to an MPI job
(`MAYBEMPI=1`/`0` overrides it). These functions and the stand-in come from
[maybempi](https://max-models.github.io/maybempi/) and are re-exported in `xp.mpi`. `get_mpi()` returns `mpi4py.MPI` then, and
otherwise the serial stand-in, so that the same code runs with and without
MPI:

```python
MPI = xp.mpi.get_mpi()  # decided once per process
comm = MPI.COMM_WORLD
comm.Allreduce(MPI.IN_PLACE, rho, op=MPI.SUM)  # nothing to do on one process
n_total = comm.allreduce(n_local)  # n_local itself
if xp.mpi.is_serial(MPI):
    ...  # a serial run
```

`get_mpi(True)` imports mpi4py (`ImportError` if missing), `get_mpi(False)`
returns the stand-in. Launched under MPI without mpi4py installed, `get_mpi()`
warns (`RuntimeWarning`) and returns the stand-in.

### `mpi.SerialMPI`, `mpi.SerialComm`

The stand-in for `mpi4py.MPI` and its communicators of size 1.
`SerialComm` returns (object methods) or copies (buffer methods) what one rank
gets: `bcast`, `allreduce`, `reduce` and `scan` return their argument,
`gather`/`allgather` return `[x]`, `scatter([x])` returns `x`;
`Allreduce`/`Reduce`/`Allgather`/`Gather`/`Scatter`/`Alltoall`/`Scan` copy the
send buffer into the receive buffer (nothing with `IN_PLACE`), the vector
forms (`Allgatherv`, `Gatherv`, `Scatterv`) use the displacement of rank 0,
`Bcast` and `Barrier` do nothing, and `sendrecv`/`Sendrecv` work to and from
rank 0 or `PROC_NULL`. Buffers are NumPy or CuPy arrays, or mpi4py buffer
specifications (`[array, MPI.DOUBLE]`). Non-blocking versions return completed
requests. Any other method raises `AttributeError`, so a missing feature shows
instead of doing nothing. `SerialMPI` has `COMM_WORLD`, `COMM_SELF`,
`COMM_NULL` (false), `IN_PLACE`, the reduction operations, the common
datatypes (`isinstance(MPI.DOUBLE, MPI.Datatype)` holds), `PROC_NULL`, `ROOT`,
`ANY_SOURCE`, `UNDEFINED`, `Comm`/`Intracomm`, `Request`, `Prequest`, `Status`,
`Wtime()` and `Is_initialized()` (False).

### `mpi.local_rank()`

The rank of the process within its node, read from the environment variables
that MPI launchers export (Open MPI, MVAPICH2, Intel MPI/MPICH, PMI, Cray
PALS, Slurm, `LOCAL_RANK`), or `0` if none is set. The launcher sets them
before `MPI_Init`, so this works before MPI is initialized and without
importing `mpi4py`.

### `cuda.bind_local_device()`

Selects device `local_rank() % device_count()` for this process and creates its
CUDA context. Returns the device id, or `None` on the NumPy backend or without
devices. Call it before `MPI_Init` (before importing `mpi4py.MPI`), so that a
CUDA-aware MPI sees the right device; otherwise all ranks of a node would use
device 0. If the launcher gives each rank its own device through
`CUDA_VISIBLE_DEVICES`, each process sees one device and selects it:

```python
import cunumpy as xp

xp.set_backend("cupy")
xp.cuda.bind_local_device()
from mpi4py import MPI  # initializes MPI after the device is bound
```

Unlike `set_device_for_rank()`, it needs no MPI rank, and it uses the rank
within the node rather than assuming contiguous ranks per node.

### `mpi.mpi_is_cuda_aware(comm=None, *, method="probe")`

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

### `mpi.require_cuda_aware_mpi(comm=None)`

Raises `RuntimeError`, explaining how to get a CUDA-aware build (Open MPI
`--with-cuda`, MPICH with a CUDA-enabled UCX, the site's CUDA-aware MPI
module), if `mpi_is_cuda_aware(comm)` returns `False` on the CuPy backend.
No-op on the NumPy backend. The complete startup sequence for one rank per
GPU:

```python
import cunumpy as xp

xp.set_backend("cupy")
xp.cuda.bind_local_device()  # 1. select the GPU, before MPI_Init
from mpi4py import MPI  # 2. MPI_Init, on the bound device

xp.mpi.require_cuda_aware_mpi()  # 3. clear error instead of a segfault later

xp.mpi.synchronize_for_mpi(send, recv)  # 4. before every MPI call with device buffers
MPI.COMM_WORLD.Sendrecv(send, dest, recvbuf=recv, source=source)
```

### `mpi.synchronize_for_mpi(*arrays, stream=None, event=None)`

Waits for all work on devices owning the CuPy arrays, or only the explicit
producer stream/event when supplied (pass at most one). The dependency must
cover all supplied device buffers. `None` entries and host arrays are ignored, so it costs
nothing for host buffers and on the NumPy backend. Call it before every MPI
call that sends or receives device buffers: CuPy launches kernels
asynchronously and MPI knows nothing about CUDA streams, so a buffer that a
kernel is still writing would be sent as it is at that moment, without an
error:

```python
xp.mpi.synchronize_for_mpi(send_buffer, recv_buffer)
comm.Sendrecv(send_buffer, dest, recvbuf=recv_buffer, source=source)
```

No synchronization is needed after MPI returns: kernels launched afterwards see
the received data.

### `mpi.mpi_buffer(array, *, send=True, recv=False, cuda_aware=None, staging=None, stream=None, event=None)`

Context manager yielding the buffer to pass to MPI for `array`: a host array
unchanged; a device array unchanged (after `synchronize_for_mpi`) when MPI is
CUDA-aware; otherwise a pinned host staging buffer, filled from the device
before the block (`send`) and copied back after it (`recv`), both counted by
`count_transfers()`. `cuda_aware=None` uses the answer recorded by
`mpi_is_cuda_aware()` or `set_mpi_cuda_aware()`; without one, a device array
raises `RuntimeError`.

Pass `stream` or `event` for producer synchronization. Receive-only device
staging also waits for preceding device work. Receive copy-back completes before
the context releases host storage. With nonblocking MPI, call `request.Wait()`
inside the context before reading/releasing the buffer.

### `mpi.MPIStaging(shape, dtype)`

Caller-owned reusable host staging storage, pinned when available and allocated
lazily. Use `staging.buffer(array, **options)` or
`mpi_buffer(array, staging=staging, **options)`. Device staging is bound to a
shape, dtype, and device, and rejects overlapping uses. Host arrays still pass
through unchanged. Use separate staging instances for simultaneously active
send and receive buffers.

Non-C-contiguous device arrays use a reusable device packing buffer; received
values are written back into the original view, preserving its storage.

### `mpi.set_mpi_cuda_aware(value)`, `mpi.get_mpi_cuda_aware()`

Record (or read) whether MPI can take device buffers, for `mpi_buffer()`.
`mpi_is_cuda_aware()` records its own result; set it by hand when the answer
is known otherwise, or `None` to forget it.

### `cuda.memory_info()`

Returns `(free_bytes, total_bytes)` reported by the CUDA runtime for the
active device, or `None` on NumPy. The values cover the device, not only
allocations owned by CuPy.

### `cuda.free_memory()`

Releases currently free blocks in CuPy's device and pinned-host memory pools.
It is a no-op on NumPy. It does not release blocks still referenced by live
arrays. CuPy normally caches freed allocations for reuse, so cached memory
does not necessarily indicate a leak.

### `cuda.max_shared_memory_per_block(device=None, *, opt_in=False)`

The bytes of shared memory a block may use on the current (or given) device,
from the device attributes, e.g. to decide whether a per-block copy of a grid
fits. `opt_in=True` gives the larger limit of newer GPUs, which a kernel uses
only after setting `max_dynamic_shared_size_bytes` on its compiled
`cupy.RawKernel`. Without CuPy it returns `DEFAULT_SHARED_MEMORY_PER_BLOCK`
(48 KiB, which every CUDA device provides).

### `cuda.cuda_include_dir()`

Returns the directory (as `str`) of the CUDA headers shipped with CuNumpy,
`cunumpy/array_view.cuh`, `cunumpy/atomic.cuh`, `cunumpy/index.cuh` and `cunumpy/reduce.cuh`. `CudaKernel` adds it to its NVRTC options as
`-I<dir>` automatically (and only once), so kernel sources can write
`#include <cunumpy/atomic.cuh>` without configuration. Use it to pass the
same headers to other compilers.

### `cuda.pin_memory(array)`

Copies a host array to page-locked (pinned) host memory. Pinned memory can
improve host/device transfer throughput in suitable asynchronous workloads.
Raises `ImportError` if CuPy is unavailable. If the input may be a CuPy array,
first transfer it with `to_numpy()`.

### `synchronize()`

Waits for queued work on the current CUDA device to finish. This is useful
before reading asynchronously computed results from host code. It is a no-op
on NumPy.

### `cuda.stream(existing=None)`

Context manager that creates a non-blocking CuPy stream and yields it. Work
issued in the block is enqueued on that stream. On NumPy, yields `None` and
does nothing:

```python
with xp.cuda.stream() as work_stream:
    result = xp.to_cupy(host_values) * 2

work_stream.synchronize()  # on CuPy; the yielded value is None on NumPy
```

Do not call methods on the yielded value without checking the backend. Use
`xp.synchronize()` for code that should work on both backends.

Pass `existing` to select a previously allocated stream for the context.

### `cuda.create_stream(non_blocking=True)`, `cuda.create_event(timing=False)`

Create reusable CuPy streams and events. On CPU they return synchronous
`HostStream`/`HostEvent` objects supporting context selection, `.synchronize()`,
`.done`, recording and waits. Events disable timing by default. CUDA streams
belong to the device current at construction; keep that device current while
using them.

### `cuda.record_event(event=None, *, stream=None)`, `cuda.wait_event(event, *, stream=None)`

Record an existing or newly created completion event on the producer stream,
then enqueue a wait on the consumer stream without blocking the CPU. Omitting
the stream uses the current CUDA stream. All consumers of an event recording
must enqueue their waits before that event is re-recorded. See the
[execution helpers guide](guides/execution-helpers.md) for examples.

## Profiling

CUDA kernels run asynchronously: a wall-clock timer around a launch measures
the launch, not the kernel, and regions of an application profiler are not
visible to `nsys`. These helpers address both; they are no-ops (or plain
timers) on NumPy, so instrumented code runs unchanged on both backends.

### `profiling.nvtx_range(name, color=None)`

Context manager and decorator marking a code region as an NVTX range. On CuPy
it calls `cupy.cuda.nvtx.RangePush(name)` on entry and `RangePop()` on exit
(also when the block raises), so the region appears on the `nsys`/Nsight
timeline next to the kernels launched inside it. `color` is an optional index
into NVTX's colour table (the `id_color` argument of `RangePush`). On NumPy,
or if NVTX is not available in the CuPy build, it does nothing. The same
instance may be nested or re-entered, e.g. as the decorator of a recursive
function.

```python
with xp.profiling.nvtx_range("push markers"):
    kernel(markers, dt, n_threads=n)


@xp.profiling.nvtx_range("accumulate")
def accumulate(particles, grid): ...
```

### `profiling.timed_region(name, *, sync=True)`

Context manager timing a code region, including the device work it queues.
It yields a `Timing` object whose `elapsed` (seconds, from
`time.perf_counter`) is set when the block exits, also when it raises. On
CuPy it synchronizes the device on entry, so earlier queued work is not
charged to the region, and, if `sync` is true, again on exit before reading
the clock; `synced` records whether that happened. It also pushes an
`nvtx_range()` of the same name. On NumPy it is a plain timer and `synced`
is `False`. With `sync=False` only the host time is measured.

```python
with xp.profiling.timed_region("push markers") as timing:
    kernel(markers, dt, n_threads=n)

print(f"{timing.name}: {timing.elapsed:.4f} s, synced={timing.synced}")
```

### `profiling.Timing`

Dataclass returned by `timed_region()`, with the fields `name` (`str`),
`elapsed` (`float`, `None` until the block exits) and `synced` (`bool`).

## `kernels.PyccelKernel`

### Constructor

```python
xp.kernels.PyccelKernel(
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
  indices or parameter names; a name also finds a positional argument, and an
  index a keyword argument, when the parameter names are known (from the
  Python signature, or `parameters`). If omitted, every converted array is
  copied back.
* `parameters`: the names of the positional parameters, or a function that
  returns them, for compiled kernels without a Python signature. A `Kernel`
  supplies the names of its host function.

`kernels.outputs_from_annotations(function)` returns the parameters a kernel may
write to, read from its annotations: everything that is not `Final`, `const` or a
scalar. Use it as `outputs=outputs_from_annotations(push)`, or pass
`outputs="annotations"` to `Kernel.from_folder()` / `KernelCatalog.from_package()`.

### Example and output declarations

```python
def scale_and_shift(scale, values, out):
    out[:] = scale * values + 1
    return out


kernel = xp.kernels.PyccelKernel(scale_and_shift, outputs=(2,))

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

## `kernels.MetalKernel`

A Metal Shading Language kernel for the GPU of an Apple silicon Mac, run with
[MLX](https://github.com/ml-explore/mlx) (`pip install 'cunumpy[metal]'`). It
takes NumPy arrays and fills the output arrays you pass, so no backend switch
is needed.

```python
import numpy as np
import cunumpy as xp

scale = xp.kernels.MetalKernel(
    "uint i = thread_position_in_grid.x; y[i] = a[0] * x[i];",
    inputs=["x", "a"],
    outputs=["y"],
)
x = np.arange(8, dtype=np.float32)
y = np.empty_like(x)
scale(x, 2.0, out=y)
```

`MetalKernel(source, inputs, outputs, *, name="cunumpy_kernel", header="",
threadgroup=256, float64="error", atomic_outputs=False, init_value=None)`

- `source` is the body of the kernel function. MLX generates the signature: each
  name in `inputs` and `outputs` is a pointer to the flat, row-major data of that
  array, so `x[i]` is the flat index. `thread_position_in_grid` and the other
  Metal attributes used in the body are added automatically. `header` goes before
  the function (includes, defines, helper functions).
- Calling the kernel: `kernel(*inputs, out=array_or_arrays, n_threads=None,
  template=None)`. `n_threads` is the total thread count (default: first axis of
  the first output). `template` gives compile-time constants, e.g.
  `template={"NSTEPS": 200}`. It returns the output array, or a tuple of them.
- Outputs are uninitialized: write every element, pass the old array as an input
  too if the kernel reads it, or set `init_value`.
- **float32 only.** The Apple GPU has no float64: a float64 input or output
  raises `TypeError`. With `float64="cast"` float64 data is computed in float32
  (a push over 200 steps agreed with float64 to about 3e-5).
- Every call copies the inputs to MLX arrays and the results back, counted as
  `to_device` and `to_host` transfers by `count_transfers()`. On a 4M-particle
  push these copies were about 7 ms next to an 11.5 ms kernel.
- `xp.kernels.metal_available()` is True if MLX is installed and a Metal GPU is
  present. Without them, calling a `MetalKernel` raises `ImportError` or
  `RuntimeError`.

## `kernels.CudaKernel`

### Constructor

```python
xp.kernels.CudaKernel(
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
    n_threads_from="auto",
    check_finite=False,
)
xp.kernels.CudaKernel.from_file(path, name=None, *, suffix="_cuda.cu", **kwargs)
xp.kernels.CudaKernel.all_from_file(path, **kwargs)
```

Wraps the `__global__` function `name` in the CUDA C `source` (declared
`extern "C"`, unless it is a template). The kernel is compiled with NVRTC
through CuPy on the first call (or by `compile()`), and cached, also on disk by
CuPy. CuPy is imported only then, so kernels can be created and their
signatures parsed without CuPy; `compile()` raises `RuntimeError` without a
GPU. `compile(log_stream=None)` compiles eagerly and returns the CuPy raw kernel;
repeated calls reuse successful state on the current device. `is_compiled`
reports that device's state. `recompile(log_stream=None)` refreshes headers and
options on the current device; finish in-flight launches before rebuilding.
Compiler failures remain retryable and a writable `log_stream` receives compiler
output. Catalog compilation therefore reports errors during setup.

`from_file` reads the source from a file; the kernel name defaults to the file
name without `suffix` (`axpy_cuda.cu` -> `axpy`), and the directory of the file
is added to the include directories and is the `source_dir`.

`all_from_file` loads every `__global__` function of a file, for files that
group several small kernels, and returns a `dict` of kernels by name in the
order of the source. The kernels share the source and options, so CuPy
compiles the file once. `xp.cuda.cuda_kernel_names(source)` lists the `__global__`
functions of a source string (ignoring comments).

```python
kernels = xp.kernels.CudaKernel.all_from_file("small_kernels.cu", block_size=64)
kernels["scale"](x, 2.0, x.size, n_threads=x.size)
kernels["shift"](x, 1.0, x.size, n_threads=x.size)
```

### Parameters

* `block_size`: threads per block, an integer for 1D launches or a tuple of 1
  to 3 integers, e.g. `(16, 16)`; at most 1024 threads in total.
* `options`: additional NVRTC options, e.g. `("-std=c++17",)`.
* `include_dirs`: directories for `#include`, passed as `-I<dir>`. The headers
  shipped with cunumpy (see "CUDA headers and array views" below) are always
  found.
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
as `"scale<double, 3>"`), `source`, `block_size`, `options`, `include_dirs`,
`source_dir`, `included_headers`, `structs`, `template_args`, `signature`,
`is_compiled`, `debug`.

### Included headers and the compile cache

CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), keyed on the
source string and the compiler options only: a file pulled in through
`#include "..."` is not part of the key, so editing a shared `.cuh` header
would not recompile the kernels that include it. `CudaKernel` therefore
resolves the quoted includes of its source when it compiles and adds a define
with a hash of their contents to the options:

```python
kernel = xp.kernels.CudaKernel.from_file("push/push_cuda.cu", include_dirs=[src_root])
kernel.included_headers  # (Path('push/helpers.cuh'), Path('.../common.cuh'))
kernel.options  # ('-Ipush', '-I<src_root>')
kernel.compile_options()  # options + ('-DCUNUMPY_INCLUDE_HASH=0x3f9a...',)
```

* `included_headers`: the header files the source includes with
  `#include "name"`, recursively, each once in order of first inclusion. A
  name is looked up relative to the including file (`source_dir` for the
  kernel source, the header's own directory for nested includes), then in
  `include_dirs` in order, then in cunumpy's header directory, like NVRTC
  does. cunumpy's shipped headers are tracked also when included in angle
  brackets (`#include <cunumpy/reduce.cuh>`), so upgrading cunumpy with a
  changed header recompiles the kernels that use it. Other angle-bracket
  (system) headers and includes that cannot be found are ignored (NVRTC
  reports the latter). Recomputed at every access, so it follows the files on
  disk.
* `compile_options()`: the options passed to CuPy at compile time: `options`
  plus `-DCUNUMPY_INCLUDE_HASH=0x<hash>` if the source includes any header,
  where the hash covers the contents of `included_headers` (not their paths).
  A changed header gives another define, hence another cache entry. Sources
  without includes never touch the file system.

The two building blocks are available on their own:

* `xp.cuda.resolve_includes(source, include_dirs=(), *, base_dir=None,
  angle_dirs=())`: the resolved header paths of a source, as a list;
  `angle_dirs` are searched last and also for `#include <name>`.
* `xp.cuda.include_hash(paths)`: the first 16 hex digits of the SHA-256 digest of
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
  `extern __shared__` arrays. Static plus dynamic storage is checked against
  the device's opt-in limit. Dynamic storage above the default allowance after
  subtracting static storage opts in through `max_dynamic_shared_size_bytes`.
  Block/grid dimensions and device/kernel thread limits are also checked.

Without `n_threads` and `grid`, the default `n_threads_from="auto"` infers the
launch from the first array argument, skipping scalars and zero-dimensional
arrays. Arrays inside `CudaArguments`, `CudaStructArguments`, and struct values
are searched in argument/field order. The effective block shape determines
how many leading array axes are used: a 1D block uses `shape[0]` (one thread per
row/particle), a 2D block uses `shape[:2]`, and a 3D block uses `shape[:3]`. These
axes map to CUDA x, y, z. A `block=` override changes the inference dimensions.
Missing arrays or insufficient array dimensions raise with instructions to
give an explicit size.

```python
push = xp.kernels.CudaKernel.from_file("push_cuda.cu")
push(positions, velocities, e_field, dt)  # n_threads = positions.shape[0]
```

Explicit `n_threads` or `grid` always takes precedence. Set `n_threads_from`
(constructor argument or settable property) to a callable such as
`lambda args: args[0].size` for flattened element kernels, or
`lambda args: args[0].shape[::-1]` for kernels whose x index follows columns.
`"first_array"` always uses the first axis and `"last_axis"` the last one (one
thread per entry of a component-major `(ncomp, N)` or `(N,)` array); None disables inference and requires
explicit launch sizes. The same defaults apply through `kernels.Kernel` and
`kernel_testing.assert_kernels_agree`, and in CPU emulation.

Nothing is launched if the grid has a zero dimension (e.g. `n_threads=0`).
`kernel.launch_shape(n_threads=None, *, grid=None, block=None, args=None)` returns the
`(grid, block)` a call would use, e.g. to size a per-block output:

Supply `args=(...)` to inspect an automatically inferred launch without running
the kernel, e.g. `kernel.launch_shape(args=(positions, velocities, e_field, dt))`.

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
  * array view parameters (`Array2D<double>`, see "CUDA headers and array
    views" below) take CuPy arrays of the declared dtype and number of
    dimensions, contiguous or not, and are packed into (pointer, shape,
    strides in elements);
  * Python scalars are cast to the declared type: `int` into integer (with a
    range check, `OverflowError`), floating-point and complex parameters,
    `float` into floating-point and complex parameters, `bool` into boolean
    and integer parameters; anything else raises `TypeError`;
  * NumPy scalars are passed as they are if their dtype matches, cast if the
    cast is safe (e.g. `np.float32` into `double`), and raise `TypeError`
    otherwise (e.g. `np.float64` into `float`). NumPy integer scalars are
    checked by value, like Python ints: `np.int64(5)` fits an `int`
    parameter, `np.int64(2**31)` raises `OverflowError`.

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
`int64_t` and `size_t` are supported too. `xp.cuda.ctype_of(dtype)` gives the C
type of a dtype (`xp.cuda.ctype_of(np.float64) == "double"`), e.g. to generate
source. `xp.cuda.parse_cuda_signature(source, name, *, structs=(),
template_args=None)` returns the parsed parameters (`CudaParameter` tuples of
`name`, `ctype`, `dtype`, `pointer`, `struct`, `view_ndim`).

### CUDA headers and array views

```python
SCALE_COLUMN = r"""
#include <cunumpy/array_view.cuh>
#include <cunumpy/index.cuh>
extern "C" __global__
void scale_column(Array2D<double> a, long long column, double factor) {
    CUNUMPY_THREAD_1D(i, a.shape[0]);  // long long i; returns if i >= shape[0]
    a(i, column) *= factor;
}
"""
scale_column = xp.kernels.CudaKernel(SCALE_COLUMN, "scale_column")
view = markers[::2, 1:5]  # non-contiguous is fine
scale_column(view, 1, 10.0, n_threads=view.shape[0])
```

cunumpy ships CUDA headers that every `CudaKernel` finds automatically;
`xp.cuda.cuda_include_dir()` returns their directory (a `str`) for other compilers
(`-I<dir>`).

`cunumpy/array_view.cuh` defines the strided views `Array1D<T>` to
`Array16D<T>` (for example, a 4D view can describe a 3D grid of vector
components `(nx, ny, nz, ncomp)`): `T* data`, `long long shape[ndim]`, `long long
strides[ndim]` (in elements, not bytes), `operator()(i, j, ...)` returning a
reference to the element, and `size()`. A kernel indexes `a(i, j)` like the
pyccel kernel it is ported from indexes `a[i, j]`, without hand-passed sizes.
Compiling with `options=("-DCUNUMPY_BOUNDS_CHECK",)` checks every index against
the shape (an out-of-bounds index prints a message and traps the kernel).

A kernel parameter or a `CudaStruct` field of type `Array<n>D<T>`, for the
scalar C types above, takes a CuPy array of that dtype and number of dimensions
(dtype and ndim mismatches raise `TypeError`), contiguous or not: it is packed
by value into pointer, shape and strides with the memory layout of the C
struct (8-byte aligned, `sizeof == 8 * (1 + 2 * ndim)`; the header checks this
with `static_assert`). `CudaParameter.view_ndim` is the number of dimensions of
such a parameter.

`cunumpy/index.cuh` defines `CUNUMPY_THREAD_1D(i, n)` (declares `long long i`
as the global thread index and returns if `i >= n`), `CUNUMPY_THREAD_2D(i, j,
ni, nj)`, `CUNUMPY_THREAD_3D(i, j, k, ni, nj, nk)` and the grid-stride loop
`CUNUMPY_GRID_STRIDE_1D(i, n) { ... }`.

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
scale_f64 = xp.kernels.CudaKernel(SCALE, "scale", template_args=(np.float64, 3))
scale_f64(x, 2.0, x.size, n_threads=x.size)  # instantiation scale<double, 3>
```

For kernels whose source is generated per variant (e.g. per number of
dimensions and dtype), `CudaKernelVariants` creates and caches one kernel per
key:

```python
matvec = xp.kernels.CudaKernelVariants(
    lambda ndim, dtype: xp.kernels.CudaKernel(
        make_source(ndim, xp.cuda.ctype_of(dtype)), "matvec"
    )
)
matvec.get(3, np.float64)(mat, x, out, n_threads=out.size)  # created once
matvec.compile_all([(3, np.float64), (3, np.complex128)])  # at setup
```

`get(*key)` calls the factory the first time a key is used; `keys()`,
iteration and `len()` give the variants created so far; `compile_all(keys=(), jobs=1)`
creates the given variants and compiles all of them, `jobs` at a time in
threads (see `KernelCatalog.compile_all`).

### Debugging

```python
xp.cuda.set_cuda_debug(enabled)
xp.cuda.get_cuda_debug()
xp.cuda.cuda_debug(enabled=True)  # context manager
xp.kernels.CudaKernel(..., debug=None)
kernel.debug_active()
kernel.compile_options()
xp.cuda.DEBUG_OPTIONS  # ("-lineinfo", "-DCUNUMPY_BOUNDS_CHECK")
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

Debug mode is enabled globally with `xp.cuda.set_cuda_debug(True)`, temporarily
with the context manager `xp.cuda.cuda_debug()`, or before starting Python with
the environment variable `CUNUMPY_CUDA_DEBUG=1` (`true`, `yes` and `on` work
too); `xp.cuda.get_cuda_debug()` returns the current setting. A kernel created with
`debug=None` (the default) reads the global setting at every launch, so
enabling it also affects kernels created earlier; `debug=True` or
`debug=False` fix the mode for one kernel. Only the compile options are fixed
at compile time: a kernel compiled before debug mode was enabled keeps its
options, so call `recompile()` after enabling, or create the kernels after
enabling. `kernel.debug_active()` tells whether debug mode applies to a
kernel now, and `kernel.compile_options()` returns the options a compilation
now would use.

```python
with xp.cuda.cuda_debug():
    kernel = xp.kernels.CudaKernel(SOURCE, "kernel")
    kernel(
        x, y, n, n_threads=n
    )  # RuntimeError: CUDA error after launching kernel 'kernel' ...
```

The `RuntimeError` says which kernel failed, not where. The next step is
NVIDIA's memory checker, which reports the faulting source line (thanks to
`-lineinfo`) and also finds out-of-bounds accesses that do not crash:

```bash
CUNUMPY_CUDA_DEBUG=1 compute-sanitizer python -m pytest tests/unit/test_my_kernel.py
```

Note that after an illegal memory access the CUDA context is unusable; the
process (or the pytest run) has to be restarted.

## `arguments.CudaStruct`

```python
Particles = xp.arguments.CudaStruct(
    "Particles",
    [("x", "double*"), ("v", "double*"), ("n", "int"), ("charge", "double")],
)
source = (
    Particles.declaration
    + r"""
extern "C" __global__ void push(Particles p, double dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < p.n) p.x[i] += dt * p.charge * p.v[i];
}
"""
)
push = xp.kernels.CudaKernel(source, "push", structs=[Particles])
push(Particles(x=x, v=v, n=x.size, charge=-1.0), 0.1, n_threads=x.size)
```

A C struct passed to kernels by value. It groups arguments, e.g. all arrays
describing a set of particles, into one kernel parameter, so adding a field
changes one definition instead of every kernel signature.

`CudaStruct(name, fields)` takes the fields as `(name, C type)` pairs; scalar
fields, pointers to the scalar types above (or `void*`), and array views
`Array1D<T>` to `Array16D<T>` of those scalar types (see "CUDA headers and
array views") are supported.

* `declaration`: the C definition of the struct, to put in the CUDA source
  or a header. A struct with array view fields (`has_views`) needs
  `#include "cunumpy/array_view.cuh"` before it; `to_header()` adds it.
* `dtype`: the NumPy structured dtype with the memory layout of the C struct
  (C alignment and padding; pointers stored as 64-bit device addresses).
* `fields`: the parsed fields (`CudaParameter` tuples).
* `check_source(source)`: raises `ValueError` if `source` defines the struct
  with other fields; a kernel created with `structs=[...]` does this check.
* `verify_layout(include=None, *, include_dirs=(), options=())` (needs CuPy):
  compiles and runs a one-thread kernel that reports `sizeof`, `alignof` and
  every field offset as the CUDA compiler lays the struct out, and raises
  `ValueError` listing the differences from `dtype`. Returns the measured
  layout as a dict. With `include` (a header file name or `#include` line,
  found in `include_dirs`) the struct is defined by that header instead of
  `declaration`, which checks a hand-written or generated header. Call it once
  per struct in a GPU test, and on every new platform (e.g. ROCm).
  `layout_source(include=None)` returns the kernel source.
* Calling the struct with keyword arguments, one per field, packs the values:
  pointer fields take C-contiguous CuPy arrays of the declared dtype (never
  copied), array view fields take CuPy arrays of the declared dtype and number
  of dimensions (contiguous or not), scalar fields are checked and cast like
  scalar kernel arguments.

The result is a `CudaStructValue`: it keeps references to the arrays it points
to (the packed struct only holds their addresses, so keep the value alive while
the kernel may run), gives access to the field values with
`value["field"]`, holds the packed struct in `value.packed`, and is flattened
into it when passed to a kernel.

### Structs from Python annotations

```python
class MarkerArguments:  # the pyccel argument class, e.g. in struphy
    def __init__(self, markers: "float[:, :]", n_markers: int, valid: "bool[:]"): ...


MarkerArgs = xp.arguments.CudaStruct.from_signature(
    MarkerArguments.__init__, "MarkerArgs"
)
print(MarkerArgs.declaration)
# struct MarkerArgs {
#     Array2D<double> markers;
#     long long n_markers;
#     Array1D<bool> valid;
# };
```

`CudaStruct.from_signature(func, name, *, int_type="long long",
scalar_names=None)` builds the struct from the annotated parameters of `func`
(one field per parameter, in order; `self` is skipped), so that the Python
class is the one definition of the arguments on the host and on the device.
Annotations are written in the pyccel style, as strings or real types:
`"float"`/`float` -> `double`, `"int"`/`int` -> `int_type` (`"long long"` by
default, since pyccel integers are 64-bit), `"bool"`/`bool` -> `bool`, NumPy
scalar types such as `np.float32` -> `float`, and an array `"float[:, :]"` ->
`Array2D<double>` (1 to 3 dimensions; `Final[...]` and `const` are ignored).
`scalar_names` adds or changes mappings from annotation scalar names to C
types, e.g. `{"float": "float"}` for single precision. A parameter without
annotation, or with an annotation that cannot be mapped, raises `ValueError`.

### Structs from a pyccel source file

`CudaStruct.from_pyccel_class(source, class_name, name=None, *, int_type="long long",
scalar_names=None, exclude=(), attribute_names=True)` builds a struct from the
annotated `__init__` of a class in a `.py` file (or a source string), parsed
with `ast` and never imported, for classes whose module is compiled by pyccel.
Fields are named after the attributes the parameters are stored in
(`self.<attribute> = <parameter>`; `attribute_names=False` keeps the parameter
names), parameters in `exclude` are skipped, and the mappings are those of
`from_signature`. Raises `ValueError` if the class or its `__init__` is missing
or an annotation cannot be mapped.

### Generating headers

```python
xp.arguments.write_cuda_header("pusher_args.cuh", [MarkerArgs, DomainArgs])
```

`struct.to_header(path=None, *, guard=None, includes=())` returns the struct
definition as a header: an include guard (`<NAME>_CUH` by default),
`#include "cunumpy/array_view.cuh"` if the struct has array view fields, the
`includes` (file names or `#include` lines), and the definition. With `path`
the header is also written. `xp.arguments.write_cuda_header(path, structs, guard=None,
*, includes=())` writes several structs to one header (the guard defaults to
the file name, `pusher_args.cuh` -> `PUSHER_ARGS_CUH`) and returns the source.

The pattern: write the header once (at setup, or in a script), commit it next
to the kernels that `#include` it, and keep it in sync with a test:

```python
def test_pusher_args_header_is_up_to_date():
    generated = xp.arguments.write_cuda_header(
        tmp_path / "pusher_args.cuh", [MarkerArgs, DomainArgs]
    )
    assert Path("kernels/pusher_args.cuh").read_text() == generated
```

Kernels created with `structs=[MarkerArgs, ...]` also check a definition in
their own source against the Python definition (`check_source`).

## `arguments.CudaStructArguments`

```python
class MarkerArguments(xp.arguments.CudaStructArguments):
    struct_name = "MarkerArgs"
    fields = (("markers", "Array2D<double>"), ("valid", "bool*"), ("n_markers", "int"))

    def __init__(self, markers, valid):
        self.markers = markers
        self.valid = valid
        self.n_markers = markers.shape[0]
        self.pack()


push = xp.kernels.CudaKernel(source, "push", structs=[MarkerArguments.struct])
push(MarkerArguments(markers, valid), dt, n_threads=markers.shape[0])
```

Base class for argument objects that are one C struct: the class form of
`CudaStruct`. A subclass sets `struct_name` and `fields` (`(field name, C type)`
pairs, as for `CudaStruct`), stores every field as an attribute of the same
name, and calls `pack()` at the end of its constructor. The `CudaStruct` is
built once per subclass when the class is defined and is the class attribute
`struct` (for `structs=[...]`, `declaration`, `to_header()`,
`write_cuda_header()`); invalid field types raise when the class is defined.

* `pack()` packs the field attributes, with the checks of `CudaStruct`
  (C-contiguous CuPy arrays of the declared dtype, range-checked scalars). A
  field without an attribute raises `AttributeError`.
* The packed struct always matches the current attributes: at every use
  (`packed`, `__cuda_args__()`, so at every launch) the device address, shape
  and strides of each array field and the value of each scalar field are
  compared with what was packed, and the struct is packed again if anything
  changed. Fields may be properties that read an owner's current arrays, so a
  resized array is picked up at the next launch. Calling `pack()` again is
  never needed.
* `packed` is the packed struct (`numpy.void`); `__cuda_args__()` returns
  `(packed,)`, so a `CudaKernel` receives the struct.
* Copies (`copy.copy`, `copy.deepcopy`) and unpickled objects are packed again
  from their own arrays; the packed struct is not part of the pickled state.
* A subclass that sets neither `struct_name` nor `fields` is an intermediate
  base class (its instances cannot be packed); setting only one raises
  `TypeError`. Subclasses of a complete class inherit its struct.

## `arguments.CudaArguments`

```python
class Particles(xp.arguments.CudaArguments):
    def __init__(self, positions, velocities):
        self.positions = positions
        super().__init__(positions, velocities, positions.shape[0])


kernel(dt, Particles(x, v), n_threads=x.shape[0])
```

Base class for objects passed to a `CudaKernel` as one argument that stands
for several kernel parameters. `CudaArguments(*values)` stores the values;
`__cuda_args__()` returns them. Subclassing is optional: any object with a
`__cuda_args__()` method returning a tuple is flattened. Host kernels
receive their own argument objects (e.g. Pyccel classes holding NumPy arrays)
unchanged: the caller passes the host or the CUDA object, cunumpy never
converts one into the other. A `CudaArguments` object may also return struct
values (`CudaStructValue.packed`) among its values.

## `kernels.Kernel`

```python
xp.kernels.Kernel(
    host_kernel,
    cuda_kernel=None,
    *,
    name=None,
    missing_cuda="raise",
    cuda_path=None,
    host_options=None,
    dispatch="backend",
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
CUDA kernel and ignored by the host kernel. Omitted sizes use the CUDA kernel's
shape-based default or its configured callback. Arguments are passed to the
selected kernel as they are (a `CudaKernel` flattens `__cuda_args__()`
objects). `kernel.compile()` compiles the CUDA kernel now and returns whether
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

`dispatch` decides which kernel a call runs. `"backend"` (default): the CUDA
kernel on the CuPy backend, the host kernel on the NumPy backend.
`"arrays"`: the CUDA kernel if any top-level argument lives on the GPU (a CuPy
array, or a CUDA argument object: one with `__cuda_args__()`, such as a
`CudaArguments` or a struct value), else the host
kernel, whatever the backend. Use `"arrays"` in codes that hand host arrays to
kernels while CuPy is active (diagnostics, MPI staging, CPU fallbacks): those
calls then run the host kernel instead of failing in the CUDA argument checks,
calling the host function directly (no device-array conversion).
`missing_cuda` applies to device arguments without a CUDA kernel.

`kernel.check_signature()` checks that the host and CUDA kernels (and the
fallback of a `CompiledHostKernel`) take the same parameters in the same order (the names of the Python host function, or of the
uncompiled Python version of a `CompiledHostKernel`, against the parsed
`__global__` signature) and raises `ValueError` showing both lists otherwise.
It does nothing without a CUDA kernel, with `check_signature=False`, or when
the host kernel has no Python signature (a compiled function).
`kernel.host_parameters()` returns the host names, or None.

Properties: `name`, `host_kernel`, `cuda_kernel`, `has_cuda`, `missing_cuda`,
`cuda_path`, `dispatch`.

### Test arguments and compiled host kernels

* `Kernel(..., test_args="pkg.push.push_test_args")`,
  `Kernel.test_args_module` and `Kernel.test_args` (the module, imported on
  first access): the test-arguments module of the kernel, set by
  `KernelCatalog.from_package()`; see `check_parity` below.
* `Kernel.host_parameters()` falls back to the `__pyccel__/<module>.pyi` stub
  for a pyccel-compiled host function, so `check_signature()` works for
  compiled kernels.
* `Kernel.__call__` infers thread counts by default; custom `n_threads_from`
  callbacks are supported, and None requires explicit sizes.

## `kernels.KernelCatalog`

```python
catalog = xp.kernels.KernelCatalog.from_package(
    package,
    *,
    host_suffix="_kernels",
    cuda_suffix="_cuda.cu",
    missing_cuda="raise",
    host_options=None,
    include_dirs=None,
    dispatch="backend",
    compile_host=None,
    host_fallback=None,
    **cuda_options,
)
kernel = catalog["push"]
```

A read-only mapping from names to `Kernel` objects. `from_package` scans the
subfolders of `package`: for every folder `<name>` containing the module
`<name><host_suffix>.py`, the function `<name>` of that module is the host
kernel, and `<name><cuda_suffix>` in the same folder, if present, is the CUDA
kernel (`__global__` function `<name>`). Other `__global__` functions in that
file are ignored by the catalog; they can be loaded with
`CudaKernel.all_from_file`. Typically called in the package's `__init__.py`:

```text
my_kernels/
├── __init__.py              # catalog = xp.kernels.KernelCatalog.from_package(__name__)
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
* `dispatch`: passed on to every `Kernel` (`"backend"` or `"arrays"`).
* `compile_host`: your function that compiles a host kernel module (cunumpy
  does not compile anything itself), e.g. a wrapper around `pyccel.epyccel`
  with a cache. Each host kernel is then a `CompiledHostKernel` (below),
  compiled on its first call. Without it, the plain Python functions are
  called.
* `host_fallback`: for kernels whose compilation fails, a callable with the
  same arguments (e.g. a vectorized NumPy version), given as a mapping from
  names or a function of the name. Without one, a failed compilation runs the
  uncompiled Python function, with a warning.
* `cuda_options`: passed on to `CudaKernel.from_file`, e.g. `block_size` or
  `structs`.

A Pyccel package with the layout `<name>/<name>_pyccel.py` and
`<name>/<name>_cuda.cu`, compiled host kernels and dispatch by argument:

```python
catalog = xp.kernels.KernelCatalog.from_package(
    __name__,
    host_suffix="_pyccel",
    dispatch="arrays",
    compile_host=my_pkg.compile_kernels,  # e.g. pyccel.epyccel with a cache
    host_fallback=NUMPY_VERSIONS,  # {"gather": gather_numpy, ...}
)
```

`catalog.check_signatures()` runs `check_signature()` on every kernel and
raises one `ValueError` listing every kernel whose host and CUDA parameters
differ; call it in a unit test of a ported package.

`catalog.without_cuda` lists the kernels still to port, `catalog.with_cuda`
the ported ones. `catalog.summary()` (also `str(catalog)`) is one line on the
porting status, e.g. for a `--status` command:
`"CUDA kernels: 3 of 60 (missing: a, b, c)"`; at most `max_missing=10` names
are listed before `...`.

`catalog.compile_all(jobs=1)` compiles every CUDA kernel and returns their
names; call it at setup so that the first time step does not pay for
compilation (after the first run, CuPy loads the kernels from its disk cache).
With `jobs > 1` the kernels are compiled in that many threads (NVRTC releases
the GIL; all threads use the current device), `jobs=None` uses the number of
CPUs. All kernels are compiled even if one fails; the first error is raised
afterwards.

`catalog.parity_cases()` returns the `(name, kernel)` pairs of the kernels
that have a CUDA kernel, for a parametrised parity test (see "Testing
utilities"). `KernelCatalog(kernels)` and `catalog.register(kernel, name=None)`
build a catalog by hand.

## `kernels.Kernel.from_folder`

```python
# my_sim/kernels/push/__init__.py
kernel = xp.kernels.Kernel.from_folder(
    __name__, host_suffix="_pyccel", dispatch="arrays", compile_host=compile_kernels
)
kernel.implementations  # ("pyccel", "numpy", "python", "cuda")
kernel.selected()  # "pyccel": what a call with host arrays runs now
```

The kernel of one kernel folder `package` (its dotted name, `__name__` in its
`__init__.py`). Each version of `<name>` in the folder is an implementation:
`<name><host_suffix>.py` gives `"pyccel"` (compiled with `compile_host` on
first use, as it is without one) and `"python"` (uncompiled), `<name>_numba.py`
gives `"numba"`, `<name>_numpy.py` gives `"numpy"`, and `<name><cuda_suffix>` the
CUDA kernel. `extra_implementations` adds host implementations that are not
files, as loaders by name (`{"numpy": lambda: push_numpy}`). Takes the options of
`KernelCatalog.from_package` for one kernel: `host_suffix`, `cuda_suffix`,
`test_args_suffix`, `check_name_length`, `missing_cuda`, `host_options`,
`include_dirs`, `dispatch`, `compile_host` and CUDA options such as
`block_size` or `n_threads_from`. Raises `FileNotFoundError` if the folder has
no host kernel module and `ModuleNotFoundError` if `package` is not a package.
`kernel.selected(device=True)` names the implementation for device arguments.

## `kernels.HostImplementations`, `kernels.set_host_kernel_implementation`

```python
host = xp.kernels.HostImplementations(
    "push",
    {"pyccel": load_compiled, "numpy": lambda: push_numpy, "python": lambda: push},
)
host(*args)  # the default implementation
xp.kernels.set_host_kernel_implementation("numpy")  # every kernel: like xp.set_backend
with xp.kernels.use_host_kernel_implementation("python"):  # like xp.use_backend
    host(*args)
```

The host implementations of one kernel (names from `xp.kernels.HOST_IMPLEMENTATIONS`:
`"pyccel"`, `"numba"`, `"numpy"`, `"python"`; `"python"` is required), each
given as a loader that returns the function or raises if it is unavailable.
Loaded on first use; `available(name)` loads and reports, `get(name)` returns it
or raises `LookupError` (missing, or failed to load with the error as cause),
`errors` maps names to load errors, `names` lists them, `python` is the
uncompiled function, `build()` loads the default now. A call runs
`selected()`: the implementation set with `set_host_kernel_implementation(name)` (or
`use_host_kernel_implementation`, or the environment variable
`CUNUMPY_HOST_KERNEL_IMPLEMENTATION` read at import), which raises if the kernel
lacks it or cannot load it, else the default: the first available of pyccel,
numba and NumPy, and else `"python"` with a `RuntimeWarning` (once).
`get_host_kernel_implementation()` reads the setting; `None` is the default. The
setting is global, not per thread, and applies to host calls only.

## Device kernel implementation selection

```python
xp.kernels.set_device_kernel_implementation("cuda")
xp.kernels.get_device_kernel_implementation()  # "cuda"
with xp.kernels.use_device_kernel_implementation(None):
    push(positions, velocities, dt)  # automatic selection
xp.kernels.set_device_kernel_implementation(None)
```

`xp.kernels.DEVICE_IMPLEMENTATIONS` is currently `("cuda",)`. The setter accepts
`"cuda"` or `None` (automatic selection); unsupported values raise `ValueError`
without changing the setting. The getter reports the requested setting, so
it returns `None` in automatic mode even when a call would use CUDA.
`CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION` initializes the setting at import;
an unset or empty value means automatic selection. Environment values are
case-insensitive and stripped of whitespace; unsupported values fail at import.

Explicit `"cuda"` requires a CUDA implementation for device dispatch: missing
implementations raise `LookupError`, even with `missing_cuda="fallback"`.
Automatic mode preserves that per-kernel fallback policy. This affects
`Kernel` and `KernelCatalog` device dispatch; it does not switch the array
backend, alter host calls, or affect direct `CudaKernel`/CuPy RawKernel calls.
The context manager restores the previous setting even after an exception.
The setting is global, not per thread.

## `kernels.CompiledHostKernel`

```python
kernel = xp.kernels.CompiledHostKernel(
    my_kernels_module, "push", compiler, fallback=push_numpy
)
kernel(*args)
```

A host kernel compiled on its first call, for `KernelCatalog.from_package(...,
compile_host=...)` or by hand. cunumpy does not compile anything itself:
`compiler(module)` is your function returning the compiled form of the module
(with the same function names), e.g. a wrapper around `pyccel.epyccel` with an
on-disk cache, or one that imports modules compiled ahead of time with the
`pyccel` command. If compilation fails, the kernel calls `fallback`, or else
the uncompiled Python function with a `RuntimeWarning`. `kernel.compiled`
builds and reports whether that worked (so that callers can choose another
path), `kernel.error` is the exception of a failed build, `kernel.python` the
uncompiled function, `kernel.fallback` the fallback and `kernel.build()`
compiles now.

## Testing utilities

```python
from cunumpy.kernel_testing import (
    BACKENDS,
    assert_kernels_agree,
    device_function_kernel,
    emulate_cuda_kernel,
    emulated_launches,
    fake_cupy_session,
    requires_cupy,
    requires_device_backend,
    run_in_fake_cupy_subprocess,
)
```

`cunumpy.kernel_testing` holds helpers for testing kernels with pytest. It is
not imported by `import cunumpy`, and it imports pytest only when one of its
pytest objects is used, so `device_function_kernel` works without pytest.
It is not called `cunumpy.testing`, because a submodule of that name would
replace NumPy's `xp.testing` once imported.

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
import it into a `conftest.py` (`from cunumpy.kernel_testing import backend`) or the
test module, then take `backend` as a test argument.

### `requires_device_backend`, `device_backend_available()`

`device_backend_available()` tells whether a CuPy-backend program can run: a
GPU, or the fake CuPy. That is more than `requires_cupy` allows (CUDA kernels
can be launched): on the fake CuPy, launches only run inside
`emulated_launches()` (or `fake_cupy_session()`). `requires_device_backend`
skips tests without either; with `CUNUMPY_REQUIRE_CUDA` set it fails them
instead.

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
find_span = device_function_kernel(
    BSPLINES_CUH, "int find_span(const double* t, int p, double eta)"
)
find_span(t, p, eta, spans, eta.size, n_threads=eta.size)
```

The kernel is named `<function>_kernel` unless `name` is given. Scalar
parameters and return types are those `CudaKernel` supports; a struct or
pointer return type, an unsupported parameter type, or a parameter named like
`out_param` or `n_threads_param` raises `ValueError` (rename the generated
parameter in that case).

### `parity_cases(catalog)`, `check_parity(kernel, **overrides)`

`parity_cases(catalog)` returns one `pytest.param(kernel, id=name)` per kernel
of `catalog.parity_cases()`; a kernel without a test-arguments module is
marked `skip` with a reason naming the missing `<name>_test_args.py`.
`check_parity(kernel)` runs `assert_kernels_agree` with the module's
`make_args` and the settings of `TEST_ARGS_SETTINGS` (`N_THREADS`, `GRID`,
`BLOCK`, `RTOL`, `ATOL`, `N_CALLS`, `OUTPUTS`, `SEED`), overridden by keyword
arguments. Raises `ValueError` without a module and `TypeError` without a
callable `make_args`.

### `install_fake_cupy()`, `fake_cupy_active()`

Install the fake CuPy of `cunumpy._fake_cupy` (a strict host stand-in for CuPy
for machines without a GPU; also installed by `CUNUMPY_FAKE_CUPY=1` when
cunumpy is imported), and tell whether it is active. `requires_cupy` and
`assert_kernels_agree` skip while it is. `install_fake_cupy()` raises if the
real CuPy was imported already or cunumpy already checked for CuPy.

### `emulate_cuda_kernel(kernel, *args, n_threads=None, grid=None, block=None, compiler=None, options=(), shared_mem=0)`

```python
x, y = rng.random(1000), np.zeros(1000)
emulate_cuda_kernel(axpy, 2.0, x, y, 1000, n_threads=1000)
np.testing.assert_allclose(y, 2.0 * x, rtol=1e-15)
```

Runs a `CudaKernel` on the CPU, serially, as if it were launched with `args`,
so that CI without a GPU can compare a kernel with its host version. The
kernel source is compiled as C++ (C++17, `CXX` or `c++`; see
`emulation_compiler()`) with the CUDA built-ins replaced: `threadIdx`,
`blockIdx`, `blockDim`, `gridDim`, atomics (`atomicAdd`, `atomicMin`, ...,
plain operations), `__ldg`, `rsqrt`, `__trap`. The shipped headers and the
kernel's include directories and `-D` options apply. Then the kernel is called
once per thread, for every block and thread index of the launch shape.

Each kernel is compiled once (for each compiler, options and template
arguments) into a shared library that takes the arguments at run time, so all
later launches, with any values and array sizes, reuse it. Libraries are cached
in the process and on disk, like CuPy's kernel cache: in
`emulation_cache_dir()`, which is `CUNUMPY_EMULATION_CACHE` (`0`: no disk
cache), else `$XDG_CACHE_HOME/cunumpy/emulation` or `~/.cache/cunumpy/emulation`.
The library runs in the Python process, on the arrays themselves.

Arguments follow the signature, with NumPy arrays in place of CuPy arrays:
pointer and view parameters (`Array1D<T>` to `Array16D<T>`) take arrays of the
declared dtype (and ndim); the kernel writes into them directly (an array that
is not writeable, or whose layout a parameter cannot take, is passed as a
copy and written back), so any strides work and aliased arguments see each
other's writes. Struct parameters take a mapping of field names to values, a
`CudaStructValue`, or an object with an attribute per field. Scalars are
checked and cast like in a launch. Like NVRTC by default, the compiler may fuse
`a * b + c` into an FMA, so compare with NumPy using a tolerance of a few ulp,
or pass `options=("-ffp-contract=off",)` for NumPy's rounding.

Block shared memory and `__syncthreads` are emulated: `__shared__` variables
are one copy per block (blocks run one after another), `extern __shared__`
arrays point into a buffer of `shared_mem` bytes, and in a kernel that calls
`__syncthreads` the threads of a block run as coroutines (POSIX `ucontext`,
each with its own stack), so that every thread reaches a barrier before any
thread continues past it. Per-block deposits, shared-memory reductions and
tiled kernels work.

Not emulated: concurrency between barriers (races and atomic ordering never
show), warp intrinsics, complex scalars and inline PTX. Inline `asm(...)` and
`asm volatile(...)` statements compile, but trap when reached, so a kernel with
a PTX branch that a test never takes (e.g. `asm("trap;")` for an unknown case)
runs without extra options. A kernel, or a header it includes, using
`__syncwarp`, warp shuffles or votes raises `NotImplementedError` (serial
threads would give wrong results). A kernel that does not compile, or traps (an
out-of-bounds index with `-DCUNUMPY_BOUNDS_CHECK`, `__trap()`, inline `asm`),
raises `RuntimeError`. Other crashes, such as a segmentation fault from an
unchecked out-of-bounds index, end the process; run such code in a child
process (`run_in_fake_cupy_subprocess()`, which prints the traceback with
`faulthandler`).

### `compile_for_emulation(kernel, *, compiler=None, options=())`, `emulation_cache_dir()`

`compile_for_emulation()` builds (or fetches from the cache) the emulation
library of a kernel without launching it, so compile errors show up early;
it raises `RuntimeError` with the compiler output. A kernel without a parsed
signature is only checked for syntax and type errors (`-fsyntax-only`).
`emulation_cache_dir()` is the disk cache, or None.

### `emulated_launches(*, compiler=None, options=())`

```python
with emulated_launches():
    propagator(dt)  # CuPy backend = the fake CuPy: the kernels run on the CPU
np.testing.assert_allclose(host_buffer(markers), expected)
```

Inside the block every `CudaKernel` launch runs through `emulate_cuda_kernel`,
on the host buffers of the fake CuPy arrays (`host_buffer(array)` is the NumPy
array behind one), with the `compiler` and `options` of the block. Argument
objects are flattened like in a launch. No CUDA is compiled in the block
either: `CudaKernel.compile()` (and so `recompile()`,
`CudaKernelVariants.compile_all()` and `KernelCatalog.compile_all()`) calls
`compile_for_emulation()` and returns None instead of a `cupy.RawKernel`, so a
program that compiles its kernels up front still reports compile errors there.
Kernels the emulation cannot run (warp intrinsics) are skipped by `compile()`.
This holds on a GPU too: the block means "no CUDA here". The original methods
are restored when the block exits, also by an exception.

### `fake_cupy_session(*, compiler=None, options=())`

```python
with fake_cupy_session():
    sim.run()  # CuPy backend; kernels compiled up front and launched on the CPU
```

Everything a CuPy-backend program needs to run on the CPU: activates the CuPy
backend (the fake CuPy) and `emulated_launches(compiler=..., options=...)`.
Raises `RuntimeError` if the fake CuPy is not active.

### `run_in_fake_cupy_subprocess(code, *, env=None, timeout=None)`

```python
def test_domain_on_the_fake_cupy():
    run_in_fake_cupy_subprocess("from my_sim.tests import check_domain; check_domain()")
```

The fake CuPy must be installed before anything imports cunumpy, so a test
process that already uses cunumpy cannot switch to it. This runs
`python -X faulthandler -c code` in a child process with `CUNUMPY_FAKE_CUPY=1`
and `OMP_NUM_THREADS=1`, without the environment variables of an MPI launcher
and with `MAYBEMPI=0`, so the child runs serially and does not join the
parent's MPI job; `env` adds variables. Under MPI only rank 0 starts the child,
the other ranks skip the test. If the child fails (or runs longer than
`timeout` seconds), the test fails with the exit code or the signal (e.g.
`SIGSEGV`) and the last 50 lines of its stdout and stderr. Returns the
`subprocess.CompletedProcess` otherwise.

## `memory.HostStaging`

```python
staging = xp.memory.HostStaging(rho.shape, rho.dtype, buffers=2)
copy = staging.copy(rho)  # returns at once; rho may be overwritten
...
if copy.ready():
    h5file["rho"] = copy.result()  # a NumPy array
```

Copies device arrays to page-locked host buffers in the background, so output
overlaps the next time steps. `copy(array)` snapshots the array on the device
(on the current stream, after the kernels that wrote it) and copies the
snapshot to the next of `buffers` pinned host buffers on its own stream. It
waits only if that buffer's previous copy has not finished, so the program
runs at most `buffers` copies ahead. The returned `StagedCopy` has `ready()`
(never waits) and `result()` (waits, returns the host buffer, valid until the
buffer is reused `buffers` copies later; a stale result raises
`RuntimeError`). `staging.synchronize()` waits for all copies. Host arrays and
the NumPy backend copy at once. Device copies are counted by
`count_transfers()`. The arrays must have the staging shape and dtype.

`copy(array, *, stream=None, event=None)` accepts an explicit producer stream or
event (at most one). An event queues a wait before snapshotting on the current
stream. Later writes on another stream must wait for the snapshot/copy;
`copy.result()` is a conservative completion point. Keep the source's device
current when submitting copies. Storage binds to that device and rejects another
device. `ready()`/`result()` temporarily select the owning device and restore the
caller's device. The first GPU use of CPU-initialized storage allocates fresh
pinned slots; previous completed CPU handles keep their snapshots. `shape` and
`dtype` are read-only.

## `memory.DeviceMirror`

```python
mirror = xp.memory.DeviceMirror(host_array)
```

Pairs a host NumPy array that another library owns and keeps using on the host
(for example a stencil vector's `_data` that is exchanged over MPI) with a
device copy of the same shape and dtype, for accumulation kernels that must
write into that buffer. `host_array` must be a `numpy.ndarray`; anything else
raises `TypeError`.

* `device`: the array kernels write into. On the CuPy backend it is a CuPy
  array, allocated on first access as a copy of the host (the only implicit
  transfer). On the NumPy backend it is the host array itself, so the same
  code runs without any copy on the CPU.
* `to_device()`: copies the host array into the existing device array;
  `to_host(stream=None, event=None)`: copies the device array into the host array, in place, so the
  host array keeps its identity and the owning library sees the new values.
  Both are no-ops on the NumPy backend.
* `zero()`: zeroes the device array (allocating it empty if needed), or the
  host array on the NumPy backend.
* `rebind(host_array)`: follows a reallocation by the owner; the device array
  is kept if shape and dtype are unchanged. If the host array's shape or dtype
  changed without a `rebind()`, `device`, `to_device()` and `to_host()` raise
  `ValueError`.
* `host`, `shape`, `dtype` properties. `to_device()`, `to_host()`, `zero()`
  and `rebind()` return the mirror, for chaining.

Make the mirror's device current before using its device storage. A mismatched
device raises before copying or zeroing. Pass an explicit producer `stream=` or
`event=` to `to_host()` when production happened outside the current stream;
the host copy is complete on return. Initial uploads and refreshes in both
directions appear in transfer accounting, with payload byte counts. Switching
to NumPy uses the host array; explicitly refresh with `to_device()` after CPU
changes before resuming GPU work.

The transfers are explicit so that one per accumulation is visible and
bounded:

```python
mirror = xp.memory.DeviceMirror(vector._data)
mirror.zero()
accumulate(markers, mirror.device, n_threads=n_markers)  # a Kernel
mirror.to_host()  # vector._data now holds the result, same object
```

### `cunumpy/atomic.cuh`

A CUDA header shipped with the package (found through `cuda_include_dir()`,
which `CudaKernel` adds automatically) for the many-threads-to-one-cell writes
of accumulation kernels:

```c
#include <cunumpy/atomic.cuh>

double cunumpy_atomic_add(double* p, double v);   // *p += v, returns old *p
float  cunumpy_atomic_add(float* p, float v);
double cunumpy_atomic_add_2d(double* data, long long n1,
                             long long i, long long j, double v);
double cunumpy_atomic_add_3d(double* data, long long n1, long long n2,
                             long long i, long long j, long long k, double v);
```

The scalar and indexed helpers also support `int`, `unsigned int`, `long long`,
and `unsigned long long`. Signed 64-bit addition uses CUDA's unsigned 64-bit
atomic and modulo-2^64 arithmetic. The indexed helpers address C-contiguous arrays of shape
`(n0, n1)` and `(n0, n1, n2)`. They wrap `atomicAdd`, a hardware instruction
for `double` from compute capability 6.0 (sm_60) on; older devices use a
compare-and-swap loop.

### `cunumpy/random.cuh` and `rng.philox_uniform`

Counter-based random numbers (Philox4x32-10, as in Random123 and cuRAND): a
pure function of a key and a counter, with no generator state, so each thread
draws from `(seed, stream, counter)`, e.g. `(seed, particle id, step)`, and
the host computes the same numbers:

```c
#include <cunumpy/random.cuh>

cunumpy_u32x4 cunumpy_philox4x32_10(cunumpy_u32x4 ctr, unsigned int key0, unsigned int key1);
double cunumpy_uniform(seed, stream, counter);              // [0, 1), 53 bits
void   cunumpy_uniform2(seed, stream, counter, &u0, &u1);  // two from one call
double cunumpy_normal(seed, stream, counter);               // Box-Muller
void   cunumpy_normal2(seed, stream, counter, &z0, &z1);
```

(`seed`, `stream` and `counter` are `unsigned long long`.)

```python
ids = xp.arange(n, dtype=xp.uint64)
u0, u1 = xp.rng.philox_uniform2(seed, ids, step)  # == cunumpy_uniform2 in thread i
z0, z1 = xp.rng.philox_normal2(seed, ids, step)
words = xp.rng.philox4x32_10(counter_words, key0, key1)  # the raw generator
```

`xp.rng.philox_uniform`, `philox_uniform2`, `philox_normal`, `philox_normal2` and
`philox4x32_10` broadcast their arguments and return NumPy or CuPy arrays,
matching the inputs. The uniform numbers equal the kernel's bit for bit; the
normal numbers can differ in the last bits (`log`, `sqrt`, `sin`, `cos` on the
GPU are not the host's). The generator passes the Random123 known-answer
tests. Use a different `counter` for every random decision of a step.

### `cunumpy/morton.cuh` and `algorithms.morton_keys`

Morton (Z-order) keys: the bits of a point's integer cell coordinates,
interleaved into one `uint64`. Sorted by key, nearby points are nearby in
memory, and the points of every node of a quadtree (2D) or octree (3D) on the
same box form a contiguous range, the starting point of tree builds on the
GPU.

```python
keys = xp.algorithms.morton_keys(
    positions, lower, upper, levels
)  # (n, 2|3) -> (n,) uint64
keys, order, positions = xp.algorithms.sort_by_key(keys, positions)
node = keys >> np.uint64(ndim * (levels - level))  # node index at `level`
cells = xp.algorithms.morton_decode(node, ndim)  # its integer coordinates
key = xp.algorithms.morton_encode(ix, iy)  # from integer cells
scales = xp.algorithms.morton_scales(
    lower, upper, levels
)  # 2**levels / (upper - lower)
```

```c
#include <cunumpy/morton.cuh>

unsigned long long cunumpy_morton_key2(x, y, lower_x, lower_y, scale_x, scale_y, levels);
unsigned long long cunumpy_morton_key3(x, y, z, lower_x, ..., scale_x, ..., levels);
unsigned long long cunumpy_morton_encode2(ix, iy);   // and _encode3(ix, iy, iz)
unsigned long long cunumpy_morton_cell(x, lower, scale, levels);
unsigned long long cunumpy_morton_spread2(v);       // and _compact2, _spread3, _compact3
```

`levels` is the number of bits per axis, at most 32 in 2D and 21 in 3D
(`xp.algorithms.MAX_MORTON_LEVELS`). Axis 0 is the lowest bit of every group of `ndim`
bits; the top group is the child of the root. The cell along an axis is
`floor((x - lower) * scale)` clipped to `[0, 2**levels - 1]`: points on a cell
boundary go to the upper cell, points outside the box to the nearest face, and
`lower > upper` reverses the axis. Given the `morton_scales` of the host, the
kernel functions return the host keys bit for bit. All host functions run on
NumPy and CuPy arrays.

### `cunumpy/reduce.cuh`

Warp- and block-level reductions for hand-written kernels: in-kernel
diagnostics (energy, momentum, total charge, the maximum velocity for a CFL
check) and combining values in a block before one atomic write.

```c
#include <cunumpy/reduce.cuh>

T cunumpy_warp_sum(T v, unsigned mask = 0xffffffffu); // also min/max
T cunumpy_block_sum(T v);  T cunumpy_block_min(T v);  T cunumpy_block_max(T v);
void cunumpy_block_sum_to(T* out, T v);  // *out += block sum, one atomic per block
int cunumpy_block_thread();   // linear thread index in a 1D-3D block
int cunumpy_block_threads();  // threads per block
```

`T` is `int`, `unsigned`, `long long`, `unsigned long long`, `float` or
`double`; `block_sum_to` supports the same types. Every
thread gets the result. Rules: every thread of the block calls block
functions (no early `return`; threads without a value pass the identity, e.g.
`0.0` for a sum). Warp functions support arbitrary nonzero masks: every named
lane calls with the same mask, and only named lanes participate. The default
requires all 32 lanes. Block functions handle partial warps and any legal block
size. The block functions use 32 values of static shared memory per
type and may be called several times in a kernel.

```c
extern "C" __global__ void kinetic_energy(const double* v, long long n,
                                          double mass, double* energy) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    double e = i < n ? 0.5 * mass * v[i] * v[i] : 0.0;
    cunumpy_block_sum_to(energy, e);  // zero *energy before the launch
}
```

### `cunumpy/scan.cuh`

Warp/block prefix sums for compaction and binning:

```c
#include <cunumpy/scan.cuh>
T cunumpy_warp_inclusive_sum(T v, unsigned mask = 0xffffffffu);
T cunumpy_warp_exclusive_sum(T v, unsigned mask = 0xffffffffu);
T cunumpy_block_inclusive_sum(T v);
T cunumpy_block_exclusive_sum(T v);
```

Same shuffle arithmetic types and collective participation rules as reductions.
Warp order follows increasing participating lane IDs, including sparse masks.
Block order follows linear thread index (x fastest), with partial warps supported.
Exclusive scans start at zero. Block scans use 32 shared values per type and
specialization; repeated calls are supported. These are block-local operations;
a global compaction needs a separate pass to combine block totals.

## `scipy`

```python
A = xp.scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
x, info = xp.scipy.sparse.linalg.cg(A, b)
rho_k = xp.scipy.fft.rfftn(rho)
```

SciPy for the active backend: `scipy` on NumPy, `cupyx.scipy` on CuPy. The
forwarded subpackages are those `cupyx.scipy` has (`SUBMODULES`): `fft`,
`fftpack`, `interpolate`, `linalg`, `ndimage`, `signal`, `sparse`,
`sparse.csgraph`, `sparse.linalg`, `spatial`, `special`, `stats`. Names are
looked up at every access, so a backend switch takes effect immediately
(Python caches the imports). Nothing is imported until a name is used; SciPy
is not a dependency of cunumpy.

* A name missing on the active backend (`cupyx.scipy` covers part of SciPy)
  raises `AttributeError` naming the backend. Keyword arguments can differ
  too (SciPy's `cg(..., rtol=)` is `tol=` in CuPy).
* `xp.scipy.special.available("erfcx")` checks a name without raising.
* `xp.scipy.sparse.linalg.resolve()` returns the module itself.
* A missing SciPy (NumPy backend) or CuPy raises `ImportError` with the module
  it needs.

Sparse matrices assembled on the host move to the device once, with the
constructor of the device type: `xp.scipy.sparse.csr_matrix(host_matrix)` on
the CuPy backend copies a SciPy matrix; `matrix.get()` copies back.

## `kernels.fuse(function=None, *, kernel_name=None)`

```python
@xp.kernels.fuse
def pressure(rho, T, gamma):
    return (gamma - 1.0) * rho * T
```

Compiles an elementwise function into one kernel with `cupy.fuse` when it is
called with a CuPy array (positional or keyword), with the CuPy backend active
while it is traced, so `xp.exp` etc. resolve to CuPy ufuncs; other calls run
the function as it is. The fused kernel is created on first use and reused.
`kernel_name` names it in profilers (default: the function name). The function
must be elementwise in the sense of `cupy.fuse`: arithmetic, comparisons,
ufuncs, `xp.where`, and supported reductions as the last operation; no Python
control flow on array values or indexing. Test the CuPy path: a function
`cupy.fuse` cannot trace raises at its first call with CuPy arrays.

## `petsc.petsc_vec(array, comm=None)`

```python
b_vec = xp.petsc.petsc_vec(b)  # b: NumPy or CuPy array, shared, never copied
x_vec = xp.petsc.petsc_vec(x)
xp.synchronize()
ksp.solve(b_vec, x_vec)  # PETSc writes into x
xp.synchronize()
```

A `petsc4py.PETSc.Vec` that shares the memory of a C-contiguous array of
`PETSc.ScalarType`, through DLPack: a `seq`/`mpi` vector for a NumPy array, a
`seqcuda`/`mpicuda` (or HIP) vector for a CuPy array. The vector keeps a
reference to the array. With several processes, `array` is this process's
part of the vector (`comm`, default `COMM_WORLD`).

* Another dtype raises `TypeError` and a non-contiguous array `ValueError`
  (both would need a copy).
* A CuPy array with a petsc4py built without CUDA/HIP raises `RuntimeError`
  instead of PETSc working on a host copy.
* Synchronize between CuPy and PETSc work on the same memory (they may use
  different streams).
* For the solve to stay on the GPU, the matrix must be a GPU type too
  (`aijcusparse`, or `-mat_type aijcusparse -vec_type cuda`).

## Version

`xp.__version__` is the installed package version. When package metadata is
not available (for example, some source-tree imports), it is
`"0.0.0+unknown"`.
