# Devices, memory and streams

These helpers wrap the parts of CuPy's device API that a portable program
needs. All of them are safe to call on the NumPy backend, where they are no-ops
or return neutral values, so the calls can stay in the code unconditionally.

| Function | CuPy backend | NumPy backend |
| --- | --- | --- |
| `cupy_available()` | `True` if CuPy imports and works | same check (backend-independent) |
| `device_count()` | number of visible GPUs | number of visible GPUs, `0` without CuPy |
| `set_device(i)` | makes device `i` current | no-op |
| `memory_info()` | `(free_bytes, total_bytes)` of the current device | `None` |
| `free_memory()` | releases cached blocks of CuPy's pools | no-op |
| `synchronize()` | waits for the current device | no-op |
| `stream()` | yields a new non-blocking stream | yields `None` |
| `pin_memory(a)` | pinned host copy | raises `ImportError` without CuPy |

## Select a GPU

On a multi-GPU workstation, pick the device before creating arrays:

```python
xp.set_backend("cupy")
print("GPUs:", xp.cuda.device_count())
xp.cuda.set_device(1)          # arrays created from now on live on GPU 1
values = xp.zeros(10**6)
```

The usual alternative is to restrict visibility from outside the process,
which also works for libraries that do not know about CuNumpy:

```bash
CUDA_VISIBLE_DEVICES=1 ARRAY_BACKEND=cupy python simulate.py
```

For MPI programs with one rank per GPU, use `bind_local_device()` instead
(see [Multi-GPU programs with MPI](mpi.md)).

## Memory

```python
free, total = xp.cuda.memory_info()
print(f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
```

`memory_info()` reports the CUDA runtime's view of the whole device, including
memory held by other processes and by CuPy's caches.

CuPy does not return freed memory to the driver. It keeps released blocks in a
memory pool and reuses them for the next allocation, which makes allocation
cheap. As a consequence, `nvidia-smi` shows memory as used after arrays went
out of scope. That is not a leak. `free_memory()` returns the currently
unused cached blocks to the driver, for example before handing the GPU to
another library or between phases with very different memory needs:

```python
del large_temporary
xp.cuda.free_memory()
```

It cannot free memory still referenced by live arrays. If memory keeps
growing, look for arrays kept alive by lists, caches or closures.

## Asynchronous execution and synchronization

A CuPy operation returns as soon as the work is queued; the GPU runs it later.
This is what makes GPUs fast, and it has three practical consequences:

1. Reading a value on the host (`to_numpy()`, `float()`, `print`) waits for all
   queued work that produces it. This happens automatically.
2. Timing with `time.perf_counter()` around a launch measures the launch, not
   the work. Use `xp.profiling.timed_region()` (see [Timing and profiling](profiling.md)).
3. Libraries that read device memory without CuPy's knowledge, most
   importantly MPI, need an explicit `xp.synchronize()` (or
   `xp.mpi.synchronize_for_mpi()`) before they access a buffer.

## Streams

Work on one stream runs in order; work on different streams may overlap.
`xp.cuda.stream()` creates a non-blocking stream and makes it current for the block:

```python
with xp.cuda.stream() as s:
    device = xp.to_cupy(pinned_host)   # copy and compute queued on s
    result = xp.fft.fft(device)

xp.synchronize()                       # wait before using the result elsewhere
```

On NumPy, `stream()` yields `None`, so do not call methods on the yielded value
in portable code; use `xp.synchronize()` instead of `s.synchronize()`.
CUDA kernels take a `stream=` argument as well (see [Writing CUDA
kernels](../kernels/cuda-kernel.md)).

Streams help when independent pieces of work (a transfer and an unrelated
kernel, or several small kernels) can overlap. Most array-level code gains
nothing from them; reach for streams after a profile shows idle gaps.
