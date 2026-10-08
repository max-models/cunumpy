# Reusable execution and grouping helpers

Allocate streams, staging storage, and grouping plans during setup, then reuse
them in timesteps. CPU equivalents keep the same call sites usable on both backends.

## Require CUDA and inspect the environment

```python
import json
import cunumpy as xp

xp.set_backend("cupy", strict=True)
print(json.dumps(xp.backend_info(), indent=2))
```

Strict selection raises with the cached availability failure reason when CUDA
cannot be used, preserving the previous backend and array functions.
`use_backend("cupy", strict=True)` behaves the same way. Without strict selection,
the existing fallback to NumPy is preserved. Diagnostics include dependency
versions and current device/runtime information; they do not initialize MPI.

## Reuse streams and events

```python
producer = xp.cuda.create_stream()
consumer = xp.cuda.create_stream()
produced = xp.cuda.create_event()
consumed = xp.cuda.create_event()
values = xp.zeros(1000)
result = xp.zeros_like(values)
xp.synchronize()  # finish initialization before using nonblocking streams

for step in range(3):
    with xp.cuda.stream(producer):
        if step:
            xp.cuda.wait_event(consumed, stream=producer)
        values.fill(step + 1)
        xp.cuda.record_event(produced, stream=producer)
    with xp.cuda.stream(consumer):
        xp.cuda.wait_event(produced, stream=consumer)
        result += values
        xp.cuda.record_event(consumed, stream=consumer)

consumed.synchronize()
host_result = xp.to_numpy(result)
```

CUDA waits order consumer work without blocking the CPU. `.done` checks completion
and `.synchronize()` waits. Enqueue all consumers of an event recording before
re-recording it. Retain the arrays until their queued work finishes.

CPU factories return synchronous `HostStream`/`HostEvent` objects with the same
completion interface. Legacy `cuda.stream()` still yields `None` on CPU. Keep a
CUDA stream's device current while using it. Events disable timing by default;
performance measurement and reporting belong in scope-profiler.

## Reuse MPI staging buffers

```python
MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD
send = xp.arange(100, dtype=xp.float64)
recv = xp.zeros_like(send)
send_staging = xp.mpi.MPIStaging(send.shape, send.dtype)
recv_staging = xp.mpi.MPIStaging(recv.shape, recv.dtype)

with (
    send_staging.buffer(send, cuda_aware=False) as sendbuf,
    recv_staging.buffer(recv, send=False, recv=True, cuda_aware=False) as recvbuf,
):
    comm.Sendrecv(sendbuf, dest=0, recvbuf=recvbuf, source=0)
```

This example exchanges with rank 0; use application-specific peer ranks for a
distributed exchange. Each staging object lazily allocates one reusable host
buffer, pinned when available. Copies appear in transfer accounting, and receive
copy-back completes before storage is released. Host arrays pass through unchanged.
Use separate objects for simultaneous send/receive contexts: overlapping device
staging use raises.

Fortran-order and strided device arrays use an additional C-order device packing
buffer, also allocated once and reused. Receive data is scattered back into the
original array view before the context exits.

`mpi_buffer(array, staging=...)` is equivalent. Device staging has a fixed shape,
dtype, and device. For nonblocking MPI, call **`request.Wait()` inside the context**
before storage is released or received data is copied back. Requests are not
tracked automatically.

Pass `stream=producer` or `event=completion` to wait for that explicit producer.
Without either, CuNumpy synchronizes all work on the buffer's device.
`synchronize_for_mpi(*arrays, stream=..., event=...)` follows the same rule; pass
at most one dependency, covering all reads/writes of the supplied buffers.

## Order output snapshots and mirror refreshes

```python
staging = xp.memory.HostStaging(field.shape, field.dtype, buffers=2)
copy = staging.copy(field, stream=producer)
# Alternatively: copy = staging.copy(field, event=produced)
host_field = copy.result()  # complete host data for an output writer
```

The snapshot runs on the supplied producer stream, or on the current stream
after waiting for the supplied event. Without either, the current stream must
already cover production. Subsequent writes on that stream are ordered after
the snapshot; another stream must wait before overwriting the source. Waiting
for `copy.result()` is a conservative way to establish completion.

Make the source device current when submitting copies. The staging instance
binds to that device and rejects another. Completion checks temporarily select
the owning device and restore the caller's device. CPU-initialized storage gets
fresh pinned slots at first GPU use; earlier CPU snapshots stay valid. Reusing
a slot still invalidates its earlier handle, so copy a returned host buffer if
the writer must retain it.

`DeviceMirror.to_host(stream=producer)` and `.to_host(event=produced)` provide the
same explicit producer choices for synchronous refreshes into another library's
host array. Make the mirror's device current. After CPU work changes its host
array, explicitly call `.to_device()` before resuming GPU work.

## Check copies and prepare kernels

```python
with xp.profiling.count_transfers() as copies:
    mirror.to_device()
    mirror.to_host()
print(copies.bytes_to_device, copies.bytes_to_host)
```

Transfers through execution/conversion helpers carry payload byte counts,
including mirror refreshes, MPI/output staging, argument conversion, and kernel
output copy-back. Conversion/fallback markers describe calls and add no bytes.
`total` includes these markers; use `to_host + to_device` for physical boundary
copy counts. Device-only conversions are recorded separately as `device_copy`
and are permitted by `assert_no_transfers()`. Forwarded backend operations such
as `xp.asarray()` and external library copies are outside this accounting.

`kernel.compile(log_stream=log)` compiles eagerly and caches successful state
per device; catalog setup catches CUDA compiler failures before timesteps.
`kernel.recompile()` explicitly refreshes headers/debug options. Each launch
checks actual block/grid, kernel thread, and static-plus-dynamic shared-memory
limits. These execution checks are independent of scope-profiler.

GPU CI requires a real CUDA device (`CUNUMPY_REQUIRE_CUDA=1`; the GPU markers of
`cunumpy.kernel_testing` then fail instead of skipping) and runs focused
`memcheck`, `racecheck`, and `synccheck` jobs with nonzero sanitizer error exits.
Numerical parity tests are separate from performance comparisons.

## Keep the live particles: `compact_by_mask`

```python
n = xp.algorithms.compact_by_mask(alive, markers, weights)
markers, weights = markers[:n], weights[:n]
```

Moves the rows where the boolean mask is True to the front of every array, in
place and in order, and returns their number. The rows after the first `n` are
unspecified. The count is needed on the host, so on CuPy each call
synchronizes once. For component-major arrays, `(ncomp, N)` next to `(N,)`,
pass `axis=-1` and continue with `positions[:, :n]`.

## Prepare cell ranges and segment reductions

```python
cells = xp.asarray([2, 0, 2, 4, 0], dtype=xp.int64)
values = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
cells, order, values = xp.algorithms.sort_by_key(cells, values)
offsets = xp.algorithms.cell_offsets(cells, n_cells=6)
# [0, 2, 2, 4, 4, 5, 5]; cell k uses [offsets[k], offsets[k+1]).
unique, starts, stops = xp.algorithms.segment_boundaries(cells)
# unique=[0, 2, 4], starts=[0, 2, 4], stops=[2, 4, 5].
plan = xp.algorithms.SegmentPlan(cells, n_segments=6)
totals = xp.empty(6)
plan.sum(values, out=totals)  # [7, 0, 4, 0, 4, 0]
```

Dense offsets require sorted integer IDs in `[0, n_cells)`; filter negative IDs
first. Sparse run boundaries also accept negative IDs and uint64 Morton keys.
Results remain on the input backend. Setup validation and variable-length GPU
results may synchronize: prepare outside tight loops and pass GPU offsets to
device kernels instead of reading them individually in Python.

`SegmentPlan` snapshots/validates keys once and drops negative-key rows. Repeated
sums overwrite their output and accept arbitrary trailing component dimensions.
CUDA reduces all components in one accumulation launch without per-column scalar
reads. Reusing `out` avoids output allocation; noncontiguous values, dtype
conversion, and float16 accumulation can still require temporary storage.

Floating/complex output dtypes match the input; integer/bool values produce
float64. CUDA float16 accumulates in float32. Floating-point atomic order can
vary, so compare with tolerances. Output must match shape/dtype, be C-contiguous,
and not alias values. A GPU plan requires its device current and all arrays on
that device. Mixed backends raise. Occasional calls can use
`segment_sum(values, keys, n_segments, out=...)` with the same semantics.

## CUDA scans, masked reductions, and integer atomics

```c
#include <cunumpy/scan.cuh>
#include <cunumpy/atomic.cuh>

extern "C" __global__ void compact_flags(const int* keep, long long n,
                                         int* local_offsets, int* block_counts) {
    int t = cunumpy_block_thread();
    long long i = (long long)blockIdx.x * cunumpy_block_threads() + t;
    int flag = i < n ? (keep[i] != 0) : 0;
    int offset = cunumpy_block_exclusive_sum(flag);
    int count = cunumpy_block_sum(flag);
    if (i < n) local_offsets[i] = offset;
    if (t == 0) block_counts[blockIdx.x] = count;
}
```

Offsets are block-local; global compaction needs another prefix sum of block
counts and a scatter pass. Inclusive/exclusive warp/block sums are provided.
Block collectives accept partial warps and any legal shape, with x fastest.
Every block thread must participate, including padding threads; do not use an
early-return indexing macro before a collective.

Warp reductions and scans accept optional nonzero masks, including sparse masks.
Only named lanes participate and all must call with the same mask. Scan order
follows increasing named lane IDs. The default requires all 32 lanes. These rules
follow [NVIDIA's warp intrinsic constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

`cunumpy_atomic_add` and its indexed 2D/3D variants also accept `int`, `unsigned int`,
`long long`, and `unsigned long long`, returning the old value. Signed 64-bit
addition uses CUDA's unsigned instruction with modulo-2^64 arithmetic.

Small fixed-size matrix helpers remain deferred: this repository currently has
no repeated CUDA consumers establishing a shared routine or storage convention.
