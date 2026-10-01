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

## Transfer at boundaries, not in loops

Load or generate data, move it to the device once, run the whole computation
there, and bring back only what the host needs:

```python
import numpy as np

import cunumpy as xp

xp.set_backend("cupy")

signal = xp.to_cunumpy(np.load("signal.npy"))   # one host-to-device copy
spectrum = xp.abs(xp.fft.rfft(signal)) ** 2
peak = int(xp.argmax(spectrum))                  # one tiny device-to-host copy
np.save("spectrum.npy", xp.to_numpy(spectrum))   # one device-to-host copy
```

Typical boundaries are file I/O, plotting, calls into host-only libraries
(SciPy, h5py, matplotlib), and diagnostics that run every N steps rather than
every step.

An anti-pattern to avoid:

```python
for step in range(n_steps):
    state = xp.to_cupy(state)       # copied up every step
    state = advance(state, dt)
    state = xp.to_numpy(state)      # and down again
    if step % 100 == 0:
        write_output(state)
```

Move the conversions out of the loop and convert inside the `if` only.

## Count the transfers

The classic performance bug of a GPU port is a transfer that sneaks into the
time loop. `count_transfers()` records every copy made through CuNumpy in a
block, with the file and line that caused it:

```python
with xp.count_transfers() as counter:
    for _ in range(10):
        step(state, dt)

print(counter.total)
print(counter.report())
```

```text
20 transfer(s) through cunumpy (10 to_host, 10 to_device, 0 kernel_conversion, 0 fallback)
  to_host (10):
    /home/me/sim/diagnostics.py:42: to_numpy(shape=(100000,), dtype=float64) (x10)
  to_device (10):
    /home/me/sim/step.py:17: to_cupy(shape=(100000,), dtype=float64) (x10)
```

Four kinds of events are recorded: `to_host`, `to_device`,
`kernel_conversion` (a [`PyccelKernel`](../kernels/pyccel-kernel.md) that copied
device arrays to the host and back) and `fallback` (a
[`Kernel`](../kernels/dispatch.md) without CUDA version running its host kernel
on the GPU backend). Calls that do not copy, such as `to_numpy()` of a NumPy
array, are not counted, so a `count_transfers()` block on the NumPy backend
reports zero.

In tests, `assert_no_transfers()` turns this into a check that fails with the
report:

```python
def test_step_stays_on_device():
    state = make_state()
    with xp.assert_no_transfers():
        step(state, dt)
```

The counter only sees transfers made through CuNumpy. Raw `cupy.asarray(host)`,
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

## Pinned memory

Host-to-device copies from page-locked ("pinned") host memory are faster and
can overlap with computation on a stream. `xp.pin_memory(host_array)` returns a
pinned copy:

```python
pinned = xp.pin_memory(np.load("snapshot.npy"))
with xp.stream():
    device = xp.to_cupy(pinned)
```

Pinned memory is a limited system resource; use it for large, repeatedly
transferred buffers after a profile shows transfers matter.
`pin_memory()` requires CuPy.
