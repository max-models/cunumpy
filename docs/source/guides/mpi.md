# Multi-GPU programs with MPI

The common layout for GPU simulations is one MPI rank per GPU. Getting it right
takes four steps, each of which fails silently or with a segfault if skipped.
CuNumpy provides one helper per step.

## The start-up sequence

```python
import cunumpy as xp

xp.set_backend("cupy")
xp.bind_local_device()        # 1. pick this rank's GPU, before MPI_Init

from mpi4py import MPI        # 2. MPI_Init happens here

xp.require_cuda_aware_mpi()   # 3. fail clearly if MPI cannot take GPU buffers

comm = MPI.COMM_WORLD
send = xp.full(1000, comm.rank, dtype=xp.float64)
recv = xp.empty_like(send)

xp.synchronize_for_mpi(send, recv)  # 4. before every MPI call on device buffers
comm.Sendrecv(send, dest=(comm.rank + 1) % comm.size,
              recvbuf=recv, source=(comm.rank - 1) % comm.size)
```

The same file runs on the NumPy backend: `bind_local_device()` returns `None`,
`require_cuda_aware_mpi()` does nothing, and `synchronize_for_mpi()` ignores
host buffers.

### 1. Bind the device before `MPI_Init`

Without device selection every rank on a node uses GPU 0. A CUDA-aware MPI also
binds to whatever device is current when `MPI_Init` runs, so the device must be
chosen *before* `from mpi4py import MPI`. At that point MPI cannot be asked for
the rank yet, but launchers export the node-local rank in environment
variables, which `xp.local_rank()` reads (Open MPI, MVAPICH2, Intel MPI/MPICH,
PMI, Cray PALS, Slurm and `LOCAL_RANK`).

`xp.bind_local_device()` selects device `local_rank() % device_count()`,
creates its CUDA context, and returns the device id. If the launcher already
restricts each rank to one GPU (`CUDA_VISIBLE_DEVICES` per rank, or Slurm's
`--gpus-per-task=1`), every process sees one device and selects it.

`set_device_for_rank(rank)` is an older alternative that needs the global MPI
rank and assumes ranks are numbered contiguously per node. Prefer
`bind_local_device()`.

### 2. Initialize MPI

Importing `mpi4py.MPI` initializes MPI by default. Do it after step 1.

### 3. Check that MPI is CUDA-aware

Only an MPI library built with CUDA support can send and receive CuPy arrays
directly. A plain build reads the device pointer as a host address: the result
is a segfault, or garbage values without any error. `require_cuda_aware_mpi()`
performs a tiny `Sendrecv` of a device array on every rank, verifies the
received values, and raises a `RuntimeError` with hints on getting a
CUDA-aware build if it fails.

The check is collective: call it once, on all ranks, at start-up. The boolean
variant `mpi_is_cuda_aware(comm)` lets a program choose a fallback instead,
such as staging buffers through the host:

```python
if xp.cupy_backend and not xp.mpi_is_cuda_aware(comm):
    stage_through_host = True
```

A segfault *inside* the check means the same as `False`: the MPI library is not
CUDA-aware.

### 4. Synchronize before every MPI call

CuPy kernels run asynchronously, and MPI knows nothing about CUDA streams. A
buffer that a kernel is still writing would be sent as it is at that moment.
`synchronize_for_mpi(*buffers)` waits for the current stream if any argument is
a CuPy array, and costs nothing otherwise:

```python
def exchange_halo(field, comm, left, right):
    # field has shape (nx + 2, ny): one ghost row on each side
    send_l, send_r = field[1], field[-2]
    recv_l, recv_r = xp.empty_like(send_l), xp.empty_like(send_r)
    xp.synchronize_for_mpi(send_l, send_r)
    comm.Sendrecv(send_l, dest=left, recvbuf=recv_r, source=right)
    comm.Sendrecv(send_r, dest=right, recvbuf=recv_l, source=left)
    field[0], field[-1] = recv_l, recv_r
```

Note that MPI buffers must be contiguous: rows of a C-ordered array are,
columns are not (copy them with `xp.ascontiguousarray()` first). No
synchronization is needed after MPI returns; kernels launched afterwards see
the received data.

## Launching

```bash
# Open MPI, 4 GPUs on one node
mpirun -n 4 python simulate.py --gpu

# Slurm, 2 nodes with 4 GPUs each
srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 python simulate.py --gpu
```

Print the binding once at start-up to catch mapping errors early:

```python
device = xp.bind_local_device()
from mpi4py import MPI
print(f"rank {MPI.COMM_WORLD.rank}: local rank {xp.local_rank()}, device {device}")
```

## Common failures

| Symptom | Cause |
| --- | --- |
| all ranks on a node run on GPU 0 | no device binding, or binding after `MPI_Init` |
| segfault in the first `Send`/`Recv` of a CuPy array | MPI is not CUDA-aware; `require_cuda_aware_mpi()` turns this into a clear error |
| received data is occasionally stale or zero | missing `synchronize_for_mpi()` before the call |
| `BufferError` or wrong values with array slices | non-contiguous buffer passed to MPI |
