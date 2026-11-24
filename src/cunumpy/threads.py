"""The thread pools of a process: BLAS, OpenMP, and how many share the cores.

NumPy and SciPy each load a BLAS (OpenBLAS in the wheels, MKL or BLIS elsewhere)
that starts one thread per core, next to the OpenMP pool of compiled kernels
(Pyccel's ``#$ omp parallel for``), in every MPI rank. Two problems follow:

* **Spinning BLAS threads.** After a BLAS call (``numpy.dot``, ``vdot``, a
  SciPy solve) OpenBLAS keeps its threads spinning for a while; an OpenMP kernel
  that runs next competes with them for the cores. In picnic a kernel of about
  10 us took about 5 ms after a ``vdot``, on one rank.
* **Oversubscription.** Under MPI every rank starts its own pools: four ranks on
  eight cores with eight BLAS threads each run 32 threads on 8 cores, and BLAS
  calls (SuperLU solves) became hundreds of times slower in struphy.

:func:`configure_threads` is the one call for the start of an application:
it limits every BLAS to one thread (unless the environment sets that library's
thread count) and warns when the OpenMP threads of the ranks on a node exceed
its cores. :func:`thread_report` tells what the process has, e.g. for the
metadata of a benchmark::

    report = xp.threads.configure_threads(comm)
    report.as_dict()     # cores, ranks per node, OpenMP and BLAS threads, env

:func:`limit_blas_threads` and the context manager :func:`blas_threads` set the
BLAS pools alone. The OpenMP pool is never changed: set it with
``OMP_NUM_THREADS`` (under MPI, the cores per rank). See :doc:`/guides/threads`.

The pools are found with threadpoolctl; without it (or without a BLAS it can
control, such as Apple's Accelerate) the functions change nothing and say so.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "THREAD_ENV_VARS",
    "OversubscriptionWarning",
    "ThreadPool",
    "ThreadReport",
    "available_cores",
    "blas_threads",
    "configure_threads",
    "limit_blas_threads",
    "ranks_per_node",
    "restore_blas_threads",
    "thread_pools",
    "thread_report",
]

#: The environment variables that set the threads of each BLAS (threadpoolctl's
#: ``internal_api``). A library whose variable is set keeps its thread count.
BLAS_THREAD_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openblas": ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"),
    "mkl": ("MKL_NUM_THREADS",),
    "blis": ("BLIS_NUM_THREADS",),
}

#: Every environment variable about threads that :func:`thread_report` records.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
)


class OversubscriptionWarning(RuntimeWarning):
    """More threads than cores on a node: the OpenMP threads of its ranks, or BLAS next to them."""


@dataclass(frozen=True)
class ThreadPool:
    """One thread pool of the process, as threadpoolctl reports it.

    ``user_api`` is ``"blas"`` or ``"openmp"``, ``internal_api`` the library
    (``"openblas"``, ``"mkl"``, ``"blis"``, ``"openmp"``), ``prefix`` the name
    of the shared library (NumPy and SciPy load OpenBLAS twice, as
    ``libscipy_openblas`` with different versions).
    """

    user_api: str
    internal_api: str
    prefix: str
    num_threads: int
    version: str | None = None
    filepath: str | None = None


@dataclass(frozen=True)
class ThreadReport:
    """The threads of this process and of its node (see :func:`thread_report`).

    ``cores`` are the cores this process may run on (its affinity mask), and
    ``ranks_per_node`` the MPI ranks sharing the node. ``openmp_threads`` and
    ``blas_threads`` are the largest pools of each kind (0 when none is loaded).
    ``oversubscribed`` is true when the ranks of the node run more OpenMP threads
    than the node has cores; ``blas_competes`` when a BLAS with several threads
    sits next to several OpenMP threads, whose kernels its spinning threads slow.
    ``blas_limited`` lists the BLAS libraries :func:`configure_threads` limited.
    """

    cores: int
    ranks_per_node: int
    openmp_threads: int
    blas_threads: int
    oversubscribed: bool
    blas_competes: bool
    pools: tuple[ThreadPool, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    blas_limited: tuple[str, ...] = ()
    threadpoolctl: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the report as plain data (for JSON, e.g. a benchmark's metadata)."""
        return asdict(self)

    def summary(self) -> str:
        """One line: cores, ranks, threads, and the problems found."""
        text = (
            f"{self.cores} cores, {self.ranks_per_node} rank(s) per node, "
            f"{self.openmp_threads} OpenMP and {self.blas_threads} BLAS thread(s) per rank"
        )
        if self.oversubscribed:
            text += "; oversubscribed"
        if self.blas_competes:
            text += "; BLAS threads compete with OpenMP"
        return text


def _threadpoolctl():
    """threadpoolctl, or None if it is not installed (e.g. some Pyodide builds)."""
    try:
        import threadpoolctl
    except ImportError:
        return None
    return threadpoolctl


def available_cores() -> int:
    """Return the cores this process may run on: its affinity mask, else the CPU count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def ranks_per_node(comm: Any = None) -> int:
    """Return the number of ranks of ``comm`` on this rank's node.

    Collective: every rank of ``comm`` must call it. Uses the shared-memory split
    of MPI (``Split_type(COMM_TYPE_SHARED)``).

    Parameters
    ----------
    comm : MPI communicator or None
        The ranks to count; ``None`` counts this process alone.

    Returns
    -------
    int
        The ranks on this node; 1 without a communicator or if the MPI library
        has no shared-memory split.
    """
    if comm is None or comm.Get_size() == 1:
        return 1
    try:
        from maybempi import MPI

        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    except (AttributeError, NotImplementedError, ImportError):
        return 1
    try:
        return int(local.Get_size())
    finally:
        free = getattr(local, "Free", None)
        if free is not None:
            free()


def thread_pools() -> list[ThreadPool]:
    """Return the BLAS and OpenMP pools loaded in this process (empty without threadpoolctl)."""
    threadpoolctl = _threadpoolctl()
    if threadpoolctl is None:
        return []
    return [
        ThreadPool(
            user_api=str(info.get("user_api")),
            internal_api=str(info.get("internal_api")),
            prefix=str(info.get("prefix")),
            num_threads=int(info.get("num_threads") or 0),
            version=info.get("version"),
            filepath=info.get("filepath"),
        )
        for info in threadpoolctl.threadpool_info()
    ]


def _env_overrides(internal_api: str) -> bool:
    """Whether the environment sets the threads of this BLAS itself."""
    return any(
        os.environ.get(name) for name in BLAS_THREAD_ENV_VARS.get(internal_api, ())
    )


def _openmp_threads(pools: Sequence[ThreadPool]) -> int:
    """Return the OpenMP threads a kernel gets: the largest loaded pool, else ``OMP_NUM_THREADS``."""
    counts = [pool.num_threads for pool in pools if pool.user_api == "openmp"]
    if counts:
        return max(counts)
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "").split(",")[0]))
    except ValueError:
        return available_cores()


def limit_blas_threads(
    num_threads: int = 1, *, respect_env: bool = True
) -> dict[str, int]:
    """Set every loaded BLAS to ``num_threads`` threads; return the previous counts.

    The result maps each changed library (its file path) to its former thread
    count, for :func:`restore_blas_threads`. With ``respect_env`` a library whose
    own variable is set (:data:`BLAS_THREAD_ENV_VARS`, e.g.
    ``OPENBLAS_NUM_THREADS``) is left alone: that is the user's choice. The OpenMP
    pools are kept as they were; a BLAS built with OpenMP that shares the OpenMP
    pool is left alone, with a warning. Load NumPy and SciPy's linear algebra
    before calling it: a BLAS loaded later starts with all cores.

    Parameters
    ----------
    num_threads : int
        The threads of every BLAS, at least 1.
    respect_env : bool
        Leave a library alone whose own thread variable is set.

    Returns
    -------
    dict of str to int
        The former thread count of every library changed, by file path.

    Raises
    ------
    ValueError
        If ``num_threads`` is less than 1.
    """
    if num_threads < 1:
        raise ValueError(f"num_threads must be at least 1, got {num_threads}")
    threadpoolctl = _threadpoolctl()
    if threadpoolctl is None:
        warnings.warn(
            "threadpoolctl is not installed: the BLAS threads are not limited",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    controller = threadpoolctl.ThreadpoolController()
    openmp_before = {
        lib.filepath: lib.num_threads
        for lib in controller.lib_controllers
        if lib.user_api == "openmp"
    }
    previous: dict[str, int] = {}
    for lib in controller.lib_controllers:
        if lib.user_api != "blas" or lib.num_threads == num_threads:
            continue
        if respect_env and _env_overrides(lib.internal_api):
            continue
        before = lib.num_threads
        lib.set_num_threads(num_threads)
        coupled = any(
            other.num_threads != openmp_before[other.filepath]
            for other in threadpoolctl.ThreadpoolController().lib_controllers
            if other.user_api == "openmp" and other.filepath in openmp_before
        )
        if coupled:
            # an OpenMP-built BLAS shares the OpenMP pool: limiting it would
            # limit the kernels too, so leave both as they were
            lib.set_num_threads(before)
            for other in threadpoolctl.ThreadpoolController().lib_controllers:
                if other.user_api == "openmp" and other.filepath in openmp_before:
                    other.set_num_threads(openmp_before[other.filepath])
            warnings.warn(
                f"{lib.prefix} shares the OpenMP thread pool; its threads are not "
                "limited (set OMP_NUM_THREADS for both)",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        previous[lib.filepath] = before
    return previous


def restore_blas_threads(previous: dict[str, int]) -> None:
    """Give the BLAS libraries of ``previous`` (from :func:`limit_blas_threads`) their counts back.

    Parameters
    ----------
    previous : dict of str to int
        The result of :func:`limit_blas_threads`.
    """
    threadpoolctl = _threadpoolctl()
    if threadpoolctl is None or not previous:
        return
    for lib in threadpoolctl.ThreadpoolController().lib_controllers:
        if lib.filepath in previous:
            lib.set_num_threads(previous[lib.filepath])


@contextmanager
def blas_threads(
    num_threads: int = 1, *, respect_env: bool = True
) -> Iterator[dict[str, int]]:
    """Run a block with every BLAS at ``num_threads`` threads, then restore them.

    ::

        with xp.threads.blas_threads(1):
            step()   # OpenMP kernels and BLAS calls, without spinning BLAS threads

    Parameters
    ----------
    num_threads : int
        The threads of every BLAS inside the block.
    respect_env : bool
        Leave a library alone whose own thread variable is set.

    Yields
    ------
    dict of str to int
        The former thread counts (see :func:`limit_blas_threads`).
    """
    previous = limit_blas_threads(num_threads, respect_env=respect_env)
    try:
        yield previous
    finally:
        restore_blas_threads(previous)


def _bound(cores: int, ranks: int) -> bool:
    """Whether the ranks look bound to their own cores: each sees a part of the node."""
    total = os.cpu_count() or cores
    return ranks > 1 and cores * ranks <= total


def thread_report(
    comm: Any = None, *, blas_limited: Sequence[str] = ()
) -> ThreadReport:
    """Report the threads this process runs, and whether they fit the node's cores.

    Collective when ``comm`` has several ranks (it counts the ranks per node).

    Parameters
    ----------
    comm : MPI communicator or None
        The ranks of the application; ``None`` for a single process.
    blas_limited : sequence of str
        The BLAS libraries that were limited, recorded in the report.

    Returns
    -------
    ThreadReport
        Cores, ranks per node, thread counts, pools, environment and flags.
    """
    pools = tuple(thread_pools())
    cores = available_cores()
    ranks = ranks_per_node(comm)
    openmp = _openmp_threads(pools)
    blas = max(
        (pool.num_threads for pool in pools if pool.user_api == "blas"), default=0
    )
    # a rank bound to its own cores must fit its threads on them; unbound ranks
    # all see the whole node and share it
    oversubscribed = openmp > cores if _bound(cores, ranks) else ranks * openmp > cores
    return ThreadReport(
        cores=cores,
        ranks_per_node=ranks,
        openmp_threads=openmp,
        blas_threads=blas,
        oversubscribed=bool(oversubscribed),
        blas_competes=bool(blas > 1 and openmp > 1),
        pools=pools,
        env={name: os.environ[name] for name in THREAD_ENV_VARS if name in os.environ},
        blas_limited=tuple(blas_limited),
        threadpoolctl=_threadpoolctl() is not None,
    )


def configure_threads(
    comm: Any = None, *, blas: int | None = 1, warn: bool = True
) -> ThreadReport:
    """Set up the threads of an application at its start; return the report.

    Limits every BLAS to ``blas`` threads (``None`` leaves them; a library whose
    own variable is set keeps its count, see :func:`limit_blas_threads`), then
    warns with :class:`OversubscriptionWarning` if the ranks on the node run more
    OpenMP threads than it has cores. Collective when ``comm`` has several ranks;
    every rank must call it.

    Parameters
    ----------
    comm : MPI communicator or None
        The ranks of the application; ``None`` for a single process.
    blas : int or None
        The threads of every BLAS; ``None`` leaves them as they are.
    warn : bool
        Warn about oversubscription and about BLAS threads next to OpenMP ones.

    Returns
    -------
    ThreadReport
        The threads after the change (see :func:`thread_report`).
    """
    limited: dict[str, int] = {}
    if blas is not None:
        with warnings.catch_warnings():
            if not warn:
                warnings.simplefilter("ignore")
            limited = limit_blas_threads(blas)
    report = thread_report(
        comm,
        blas_limited=tuple(
            pool.prefix for pool in thread_pools() if pool.filepath in limited
        ),
    )
    if warn and report.oversubscribed:
        warnings.warn(
            f"{report.summary()}: set OMP_NUM_THREADS to the cores per rank "
            "(and bind the ranks, e.g. mpirun --map-by slot:PE=<threads> --bind-to core)",
            OversubscriptionWarning,
            stacklevel=2,
        )
    elif warn and report.blas_competes:
        warnings.warn(
            f"{report.summary()}: BLAS threads spin after each call and slow the "
            "OpenMP kernels run next; set OPENBLAS_NUM_THREADS=1 (or the library's "
            "own variable)",
            OversubscriptionWarning,
            stacklevel=2,
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Print the thread report: ``python -m cunumpy.threads`` (``--limit`` limits BLAS first).

    Under ``mpirun`` every rank prints its line, so a job script can check its
    binding before the run.

    Parameters
    ----------
    argv : sequence of str or None
        The arguments; ``None`` reads the command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cunumpy.threads", description=main.__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--limit", action="store_true", help="limit every BLAS to one thread first"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    import json

    import numpy  # noqa: F401 -- load NumPy's BLAS, so that it is reported
    from maybempi import MPI

    comm = MPI.COMM_WORLD
    report = configure_threads(comm, blas=1 if args.limit else None, warn=False)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return
    prefix = f"rank {comm.Get_rank()}: " if comm.Get_size() > 1 else ""
    print(prefix + report.summary())
    for pool in report.pools:
        print(
            f"{prefix}  {pool.user_api:6} {pool.prefix:24} {pool.num_threads:3} thread(s)"
        )
    for name, value in report.env.items():
        print(f"{prefix}  {name}={value}")


if __name__ == "__main__":
    main()
