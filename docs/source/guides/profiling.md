# Timing and profiling

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
solve(field)                         # only queues the kernels
elapsed = time.perf_counter() - start  # wrong on the GPU
```

Timing tips:

* **Warm up first.** The first call of a CUDA kernel compiles it (or loads it
  from CuPy's disk cache), and the first CuPy operations initialize the CUDA
  context. Run one step before measuring, or compile at setup with
  `catalog.compile_all()` (see [Pairing host and CUDA kernels](../kernels/dispatch.md)).
* **Time many iterations.** Wrap a loop of steps rather than one short step;
  the synchronization itself costs a few microseconds.
* **Do not leave synchronizing timers in production hot loops.** Each one
  stalls the pipeline; use NVTX ranges there and a profiler to read them.

## Mark regions for Nsight: `nvtx_range`

```python
with xp.profiling.nvtx_range("push markers"):
    push(markers, dt, n_threads=n_markers)


@xp.profiling.nvtx_range("time step")
def step(state, dt):
    ...
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
data between host and device](data-movement.md).

## A profiling checklist

1. Run a few steps with `count_transfers()` and remove every transfer from the
   step.
2. Wrap the phases of a step in `nvtx_range()` and record an `nsys` profile.
3. Look for gaps between kernels (host overhead, synchronization) and for the
   longest kernels.
4. Use `timed_region()` around a loop of steps to measure the improvement.
