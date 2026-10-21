---
title: "Timing and profiling"
description: "GPU code needs two adjustments to the usual timing habits."
---

GPU code needs two adjustments to the usual timing habits: work is
asynchronous, so a timer must wait for the device, and the interesting events
(kernels, copies) happen outside Python, so a GPU profiler is needed to see
them. CuNumpy's helpers handle both and run unchanged on the NumPy backend.

## Time a region: `timed_region`

```python
with xp.profiling.timed_region("field solve") as timing:
    solve(field)

print(f"{timing.name}: {timing.elapsed * 1e3:.2f} ms (synced={timing.synced})")
```

On CuPy, `timed_region` synchronizes the device on entry (so earlier queued
work is not charged to this region) and on exit (so the region's own work is
included), and marks the region as an NVTX range. On NumPy it is a plain
`time.perf_counter()` timer and `synced` is `False`. With `sync=False` only the
host time, that is the launch overhead, is measured.

Compare with the naive version, which on a GPU typically reports a few
microseconds regardless of the problem size:

```python
start = time.perf_counter()
solve(field)  # only queues the kernels
elapsed = time.perf_counter() - start  # wrong on the GPU
```

Timing tips:

* **Warm up first.** The first call of a CUDA kernel compiles it (or loads it
  from CuPy's disk cache), and the first CuPy operations initialize the CUDA
  context. Run one step before measuring, or compile at setup with
  `catalog.compile_all()` (see [Pairing host and CUDA kernels](/cunumpy/kernels/dispatch/)).
* **Time many iterations.** Wrap a loop of steps rather than one short step;
  the synchronization itself costs a few microseconds.
* **Do not leave synchronizing timers in production hot loops.** Each one
  stalls the pipeline; use NVTX ranges there and a profiler to read them.

## Mark regions for Nsight: `nvtx_range`

```python
with xp.profiling.nvtx_range("push markers"):
    push(markers, dt, n_threads=n_markers)


@xp.profiling.nvtx_range("time step")
def step(state, dt): ...
```

On CuPy this pushes an NVTX range, so the region appears as a labelled bar on
the Nsight Systems timeline directly above the kernels it launched. It does not
synchronize and costs well under a microsecond. On NumPy, or without NVTX
support, it does nothing. `color` selects an entry of NVTX's colour table.

Record a profile with:

```bash
nsys profile -t cuda,nvtx -o step_profile python simulate.py --gpu --steps 20
nsys stats step_profile.nsys-rep     # summary tables in the terminal
```

and open the `.nsys-rep` file in Nsight Systems to see the timeline.

## Find unwanted transfers

A host-device copy in the time loop is the most common reason a GPU port is
slower than expected. `count_transfers()` lists every copy made through CuNumpy
with its call site:

```python
with xp.profiling.count_transfers() as counter:
    for _ in range(10):
        step(state, dt)
print(counter.report())
```

Copies made outside CuNumpy (`float(x)` on a device value, `cupy.asarray`,
another library converting internally) show up in Nsight as `memcpy` rows on the
timeline; NVTX ranges tell you which part of the step issued them. See [Moving
data between host and device](/cunumpy/guides/data-movement/).

## Cunumpy's own regions in scope-profiler

Cunumpy ships [scope-profiler](https://github.com/max-models/scope-profiler)
regions around its coarse operations: every `Kernel` call
(`cunumpy.kernel.<name>`), CUDA and Pyccel compilation, `sort_by_key`,
`segment_sum` (`cunumpy.SegmentPlan.sum`), `compact_by_mask`, `morton_keys`, the
solvers, `exchange` and the host/device copies of `mpi_buffer` and
`DeviceMirror`. They are off by default, so an application that profiles itself
does not see them, and cost at most about 0.2 µs per call while off. Turn them on with the `cunumpy` tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["cunumpy"])
```

or `SCOPE_PROFILER_ENABLE_TAGS=cunumpy`, or
`scope-profiler run --enable-tag cunumpy simulate.py` (`"*"` enables every
library's tag). The regions time the host side: on CuPy, a kernel region
measures the launch, not the kernel; wrap a synchronizing `timed_region` around
a loop of steps for device times.

## A profiling checklist

1. Run a few steps with `count_transfers()` and remove every transfer from the
   step.
2. Wrap the phases of a step in `nvtx_range()` and record an `nsys` profile.
3. Look for gaps between kernels (host overhead, synchronization) and for the
   longest kernels.
4. Use `timed_region()` around a loop of steps to measure the improvement.
