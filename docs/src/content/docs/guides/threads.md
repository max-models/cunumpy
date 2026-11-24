---
title: "Threads: BLAS and OpenMP"
description: "Keep BLAS threads from slowing the OpenMP kernels, and fit the threads of the ranks to the cores."
---

A process that uses cunumpy usually runs two kinds of thread pools: the OpenMP
pool of the compiled kernels (Pyccel's `#$ omp parallel for`), and the BLAS that
NumPy and SciPy load (OpenBLAS in the wheels, MKL or BLIS elsewhere), each with
one thread per core. Under MPI every rank starts its own. That goes wrong in two
ways:

- **Spinning BLAS threads.** After a BLAS call (`numpy.dot`, `vdot`, a SciPy
  solve) OpenBLAS keeps its threads spinning for a while, and an OpenMP kernel
  that runs next competes with them for the cores. In picnic a kernel of about
  10 µs took about 5 ms after a `vdot`, on one rank with four threads; limiting
  BLAS to one thread made a 3D particle step 1.7 times faster (402 to 236 ms).
- **Oversubscription.** Four ranks on an eight-core node, each with eight BLAS
  or OpenMP threads, run 32 threads on 8 cores. In struphy the SuperLU solves of
  its projectors became hundreds of times slower that way.

## One call at the start: `configure_threads`

```python
from maybempi import MPI
import cunumpy as xp

report = xp.threads.configure_threads(MPI.COMM_WORLD)
print(report.summary())
# 8 cores, 1 rank(s) per node, 8 OpenMP and 1 BLAS thread(s) per rank
```

It limits every loaded BLAS to one thread and warns with
`OversubscriptionWarning` when the ranks of a node run more OpenMP threads than
the node has cores (ranks bound to their own cores are judged by those). It is
collective: every rank must call it. Call it after NumPy and SciPy are imported,
since a BLAS loaded later starts with all cores.

- A BLAS whose own variable is set keeps its count: `OPENBLAS_NUM_THREADS` (or
  `GOTO_NUM_THREADS`), `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`
  (`xp.threads.BLAS_THREAD_ENV_VARS`). An MKL setting does not exempt OpenBLAS.
- The OpenMP pool is never changed. Set it with `OMP_NUM_THREADS`: under MPI, the
  cores per rank, with the ranks bound to them
  (`mpirun --map-by slot:PE=<threads> --bind-to core`, or
  `srun --cpus-per-task=<threads>`).
- A BLAS built with OpenMP shares the OpenMP pool; it is left alone, with a
  warning, so that the kernels keep their threads.
- Without threadpoolctl, or without a BLAS it can control (Apple's Accelerate),
  nothing changes and a warning says so.

`blas=None` only reports and warns; `warn=False` stays silent.

## Only around a block: `blas_threads`

```python
with xp.threads.blas_threads(1):
    for _ in range(steps):
        step()  # OpenMP kernels and BLAS calls, without spinning BLAS threads
```

`limit_blas_threads(n)` sets the BLAS pools and returns their former counts, and
`restore_blas_threads(previous)` gives them back.

## What a process has: `thread_report`

`xp.threads.thread_report(comm)` returns a `ThreadReport`: the cores of the
process (its affinity mask), the ranks per node, the OpenMP and BLAS threads,
every pool (`thread_pools()`), the thread environment variables, and the flags
`oversubscribed` and `blas_competes`. `report.as_dict()` is plain data, for the
metadata of a benchmark. From the shell, also under `mpirun`:

```bash
python -m cunumpy.threads            # one line per rank, then the pools
python -m cunumpy.threads --limit    # after limiting BLAS
python -m cunumpy.threads --json
```

## When to keep BLAS threads

A program whose time goes into large dense linear algebra (matrix products,
dense factorizations) on one rank, with no OpenMP kernels between, gains from
threaded BLAS: give it its threads with `OPENBLAS_NUM_THREADS`, or run that part
in `blas_threads(n)` with the cores. Measure both: in picnic's profiling
benchmark, one BLAS thread made no case slower.
