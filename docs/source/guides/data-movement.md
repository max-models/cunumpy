# Moving data between host and device

On a GPU system, NumPy arrays live in host (CPU) memory and CuPy arrays live in
device (GPU) memory. Copying between the two goes over PCIe or NVLink and is
often slower than the computation itself. CuNumpy therefore never moves data
implicitly: every transfer is a visible function call.

## The three conversion functions

| Function | Returns | Copies when |
| --- | --- | --- |
| `xp.to_numpy(a)` | `numpy.ndarray` (host) | `a` is a CuPy array (device to host) |
| `xp.to_cupy(a)` | `cupy.ndarray` (device) | `a` is not a CuPy array (host to device) |
| `xp.to_cunumpy(a)` | an array of the *active* backend | `a` lives on the other backend |

* `to_numpy()` accepts anything `numpy.asarray` accepts (lists, tuples, NumPy
  arrays, views). A NumPy array is returned as it is, without a copy.
* `to_cupy()` raises `ImportError` when CuPy or CUDA is not usable. A CuPy
  array is returned as it is.
* `to_cunumpy()` is the portable choice: on the NumPy backend it behaves like
  `to_numpy()`, on CuPy like `to_cupy()`.
* None of them modify the source array or change the active backend.

CuPy-to-host conversions preserve F-contiguous layout; other device inputs
become C-contiguous host arrays. See [Array ordering and strides](array-ordering.md)
for conversion rules, copies, and compiled kernel requirements.

## Transfer at boundaries, not in loops

Load or generate data, move it to the device once, run the whole computation
there, and bring back only what the host needs:

```python
import numpy as np

import cunumpy as xp

xp.set_backend("cupy")

signal = xp.to_cunumpy(np.load("signal.npy"))  # one host-to-device copy
spectrum = xp.abs(xp.fft.rfft(signal)) ** 2
peak = int(xp.argmax(spectrum))  # one tiny device-to-host copy
np.save("spectrum.npy", xp.to_numpy(spectrum))  # one device-to-host copy
```

Typical boundaries are file I/O, plotting, calls into host-only libraries
(SciPy, h5py, matplotlib), and diagnostics that run every N steps rather than
every step.

An anti-pattern to avoid:

```python
for step in range(n_steps):
    state = xp.to_cupy(state)  # copied up every step
    state = advance(state, dt)
    state = xp.to_numpy(state)  # and down again
    if step % 100 == 0:
        write_output(state)
```

Move the conversions out of the loop and convert inside the `if` only.

## Count the transfers

The classic performance bug of a GPU port is a transfer that sneaks into the
time loop. `count_transfers()` records copies made through CuNumpy's conversion
and execution helpers, with the file and line that caused them:

```python
with xp.profiling.count_transfers() as counter:
    for _ in range(10):
        step(state, dt)

print(counter.total)
print(counter.report())
```

```text
20 transfer(s) through cunumpy (10 to_host, 10 to_device, 0 kernel_conversion, 0 fallback, 0 device_copy)
  to_host (10):
    /home/me/sim/diagnostics.py:42: to_numpy(shape=(100000,), dtype=float64) (x10)
  to_device (10):
    /home/me/sim/step.py:17: to_cupy(shape=(100000,), dtype=float64) (x10)
```

Physical copies are recorded as `to_host`, `to_device`, or `device_copy`
(device-only dtype/layout conversions). Mirror refreshes, staging, and kernel
output copies are included. Each copy has an `event.nbytes` payload size;
`counter.bytes_to_host`, `counter.bytes_to_device`, and `counter.bytes(kind)`
sum the known sizes.

`kernel_conversion` marks a [`PyccelKernel`](../kernels/pyccel-kernel.md) call
that copied device arrays to the host and back, and `fallback` marks a
[`Kernel`](../kernels/dispatch.md) without a CUDA version running its host kernel
on the GPU backend. Physical copies are recorded separately from these markers,
which add no bytes. `counter.total` includes markers; `counter.to_host +
counter.to_device` counts physical host/device copies. Reference-only calls,
such as `to_numpy()` of a NumPy array, record nothing.

In tests, `assert_no_transfers()` rejects host/device copies and host execution
on the GPU backend, with the report below. It allows device-only conversions:

```python
def test_step_stays_on_device():
    state = make_state()
    with xp.profiling.assert_no_transfers():
        step(state, dt)
```

The counter only sees transfers made through CuNumpy helpers. Forwarded backend
operations such as `xp.asarray(host)`, raw `cupy.asarray(host)`,
`device_array.get()`, `float(device_scalar)` and conversions inside other
libraries are invisible to it; use `nsys` to find those (see [Timing and
profiling](profiling.md)).

## Build device arguments once: `as_device_array`

Objects that hold arrays for CUDA kernels (see [Kernel arguments and
structs](../kernels/arguments.md)) should reference existing device arrays and
copy only what is not already in the right form. `as_device_array(value, dtype,
ndim=None, *, name=None)` implements that rule:

* a C-contiguous CuPy array with the requested dtype is returned unchanged (the
  same object, so kernels write into the caller's array);
* anything else (a tuple such as `(3, 3, 3)`, a host array, another dtype, a
  non-contiguous view) becomes one C-contiguous device copy.

```python
import numpy as np

degree = xp.as_device_array((3, 3, 3), np.int32, ndim=1, name="degree")
markers = xp.as_device_array(markers, np.float64, ndim=2, name="markers")
```

Call it when the argument object is built, never per kernel call. On the NumPy
backend it raises `RuntimeError`, so host data is never copied to a device by
accident.

## Buffers owned by another library: `DeviceMirror`

Sometimes the array a GPU kernel must write into belongs to a host library (a
distributed vector exchanged over MPI, a buffer a solver keeps using).
`DeviceMirror(host_array)` pairs such an array with a device copy and makes the
two transfers explicit: `mirror.device` is the array kernels write into, and
`mirror.to_host()` copies the result back into the original host array in
place. On the NumPy backend `mirror.device` *is* the host array and the copies
are no-ops. See [Accumulation kernels](../kernels/accumulation.md) for the full
pattern.

Both directions appear in transfer accounting. A retained device buffer is
bound to its CUDA device; make that device current before using it. Pass
`mirror.to_host(stream=producer)` or `mirror.to_host(event=completion)` when
the kernel ran on another stream. The refresh blocks until the host data is
ready. After CPU work changes the host buffer, call `mirror.to_device()` before
resuming GPU work.

## Pinned memory

Host-to-device copies from page-locked ("pinned") host memory are faster and
can overlap with computation on a stream. `xp.cuda.pin_memory(host_array)` returns a
pinned copy:

```python
pinned = xp.cuda.pin_memory(np.load("snapshot.npy"))
with xp.cuda.stream():
    device = xp.to_cupy(pinned)
```

Pinned memory is a limited system resource; use it for large, repeatedly
transferred buffers after a profile shows transfers matter.
`pin_memory()` requires CuPy.

## Host-only code: `host_call`

Some code can only run on the host: a SciPy spline, a file reader, an external
equilibrium code. `xp.host_call(fun, *args, **kwargs)` calls it with arguments of
either backend. Device arrays are copied to the host, `fun` runs on the NumPy
backend, and array results are copied back, once per call. `@xp.evaluate_on_host`
does the same for a method, and `@xp.setup_on_host` runs an `__init__` on the NumPy
backend, so the object holds only host data. The copies are counted by
`count_transfers()`. Use it for setup and diagnostics, not in a time loop.

```python
values = xp.host_call(spline, x)  # x on the device -> values on the device


class Equilibrium:
    @xp.setup_on_host
    def __init__(self, path):
        self.spline = read_spline(path)  # NumPy and SciPy

    @xp.evaluate_on_host
    def pressure(self, x):
        return self.spline(x)
```
