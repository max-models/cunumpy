from __future__ import annotations

import importlib
import logging
import os
import time
import warnings
from collections.abc import Generator
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import array_api_compat
import array_api_compat.numpy as np

from .transfers import _ACTIVE as _COUNTERS
from .transfers import _describe, _record

BackendType = Literal["numpy", "cupy"]

_logger = logging.getLogger(__name__)


_CUPY_AVAILABLE_CACHE = None


def cupy_available() -> bool:
    """Check if CuPy is available and functional."""
    global _CUPY_AVAILABLE_CACHE
    if _CUPY_AVAILABLE_CACHE is not None:
        return _CUPY_AVAILABLE_CACHE

    try:
        import cupy as cp

        # Check if a GPU is available
        _CUPY_AVAILABLE_CACHE = cp.is_available()
        return _CUPY_AVAILABLE_CACHE
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        _CUPY_AVAILABLE_CACHE = False
        return False


class ArrayBackend:
    """Holds the process-wide active backend (NumPy or CuPy).

    Not thread-safe: `set_backend`/`use_backend` mutate this single shared
    instance in place, so concurrent code (threads, async tasks) switching
    backends independently will race. Safe for the typical single-threaded
    script/notebook usage this library targets.
    """

    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        if backend.lower() not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")

        self._backend: BackendType = "cupy" if backend.lower() == "cupy" else "numpy"
        self._xp: ModuleType = np  # Placeholder

        # Import numpy/cupy
        self._xp = self._load_backend(self._backend, verbose)

    def _load_backend(self, backend: BackendType, verbose: bool = False) -> ModuleType:
        if backend == "cupy":
            if cupy_available():
                import array_api_compat.cupy as cp

                self._backend = "cupy"
                return cp
            else:
                if verbose:
                    print(
                        "CuPy not available or not functional. Falling back to NumPy."
                    )
                self._backend = "numpy"
                return np
        self._backend = "numpy"
        return np

    def __repr__(self) -> str:
        return f"ArrayBackend(backend={self._backend!r}, module={self._xp.__name__!r})"

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @contextmanager
    def use_backend(self, backend: BackendType) -> Generator[None, None, None]:
        """Temporarily change the backend."""
        if backend not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")
        old_backend = self._backend
        old_xp = self._xp

        self._backend = backend
        self._xp = self._load_backend(backend)

        try:
            yield
        finally:
            self._backend = old_backend
            self._xp = old_xp


array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)


def use_backend(backend: BackendType) -> Generator[None, None, None]:
    """Temporarily change the backend."""
    return array_backend.use_backend(backend)


def set_backend(backend: BackendType) -> None:
    """Set the backend globally."""
    if backend not in ("numpy", "cupy"):
        raise ValueError("Array backend must be either 'numpy' or 'cupy'.")
    array_backend._backend = backend
    array_backend._xp = array_backend._load_backend(backend)


def get_backend() -> BackendType:
    """Return the currently active global backend name."""
    return array_backend.backend


def _cupy_backend() -> bool:
    """Check if the active global backend is CuPy."""
    return array_backend.backend == "cupy"


def _numpy_backend() -> bool:
    """Check if the active global backend is NumPy."""
    return array_backend.backend == "numpy"


def set_device(device_id: int) -> None:
    """Select the active CUDA device for the current process (no-op on NumPy)."""
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.cuda.Device(device_id).use()


def device_count() -> int:
    """Number of visible CUDA devices.

    Returns 0 on the NumPy backend, or if CuPy/CUDA is not available.
    Independent of the currently active backend -- this reports what
    hardware is visible, not what `xp.xp` currently dispatches to.
    """
    if not cupy_available():
        return 0

    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        return 0


def set_device_for_rank(rank: int, devices_per_node: int | None = None) -> int:
    """Select a CUDA device for an MPI rank, round-robin across the node.

    Convenience for one-rank-per-GPU codes: computes
    ``device_id = rank % devices_per_node`` and calls `set_device()` with
    it. `devices_per_node` defaults to `device_count()`. Returns the
    selected device id, or 0 as a no-op if there are no visible devices.

    This assumes ranks map to devices in contiguous blocks per node (i.e.
    local rank == ``rank % devices_per_node``); codes with a different
    rank-to-device layout should call `set_device()` directly instead.
    """
    n = devices_per_node if devices_per_node is not None else device_count()
    if n == 0:
        return 0

    device_id = rank % n
    set_device(device_id)
    return device_id


# Node-local rank of the process, as exported by common MPI launchers. They are
# set before ``MPI_Init``, so the device can be chosen before MPI starts.
_LOCAL_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_LOCAL_RANK",  # Open MPI
    "MV2_COMM_WORLD_LOCAL_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Intel MPI, MPICH (Hydra)
    "PMI_LOCAL_RANK",  # MPICH / PMI
    "PALS_LOCAL_RANKID",  # Cray PALS
    "SLURM_LOCALID",  # Slurm (srun)
    "LOCAL_RANK",  # torchrun and others
)


def local_rank() -> int:
    """Rank of this process within its node, from the MPI launcher's environment.

    Reads the node-local rank that common launchers export (Open MPI, MVAPICH2,
    Intel MPI/MPICH, PMI, Cray PALS, Slurm, ``LOCAL_RANK``). These variables are
    set before ``MPI_Init``, so this works before MPI is initialized, and
    without importing ``mpi4py``. Returns 0 if none is set (e.g. a serial run).
    """
    for variable in _LOCAL_RANK_VARIABLES:
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def bind_local_device() -> int | None:
    """Bind this process to one GPU of its node, by node-local rank.

    Selects device ``local_rank() % device_count()`` and creates its CUDA
    context. Call it before ``MPI_Init`` (i.e. before importing
    ``mpi4py.MPI``), so that CUDA-aware MPI sees the right device. Without it,
    every rank on a node would use device 0. If the launcher already restricts
    each rank to its own device with ``CUDA_VISIBLE_DEVICES``, every process
    sees a single device and selects it.

    Returns
    -------
    int | None
        The selected device id, or None on the NumPy backend or if no device
        is available.
    """
    if array_backend.backend != "cupy":
        return None
    count = device_count()
    if count == 0:
        return None

    import cupy as cp

    device_id = local_rank() % count
    cp.cuda.Device(device_id).use()
    cp.cuda.Stream.null.synchronize()  # creates the CUDA context now
    return device_id


def synchronize_for_mpi(*arrays: Any) -> None:
    """Wait for pending device work before MPI reads or writes `arrays`.

    CuPy launches kernels asynchronously; MPI does not know about CUDA streams.
    Passing a device buffer to MPI while a kernel is still writing it sends
    whatever is in memory at that moment -- silently wrong data, no error. Call
    this before every MPI call that uses device buffers. It synchronizes the
    current stream only if at least one of `arrays` is a CuPy array, so host
    buffers and the NumPy backend cost nothing. (After MPI returns, no
    synchronization is needed: kernels launched later see the received data.)

    Parameters
    ----------
    *arrays
        The buffers about to be passed to MPI; ``None`` entries are ignored.
    """
    if not any(array_api_compat.is_cupy_array(a) for a in arrays if a is not None):
        return

    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


def _mpi_module() -> Any:
    """Import and return ``mpi4py.MPI``, with a clear error if it is missing."""
    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            "mpi4py is required for the CUDA-aware MPI check: install it, or "
            "pass a communicator explicitly."
        ) from e
    return MPI


def _device_buffers_in_use() -> bool:
    """Whether MPI calls of this process would carry device (CuPy) buffers."""
    return array_backend.backend == "cupy" and cupy_available()


def _mpi_probe_buffers(rank: int, n: int = 4) -> tuple[Any, Any]:
    """Device send and receive buffers for the CUDA-aware MPI probe.

    The send buffer of rank ``r`` holds ``r + arange(n)``, so the receiver can
    verify that the values came from the expected source.
    """
    import cupy as cp

    send = cp.arange(n, dtype=cp.float64) + rank
    recv = cp.empty(n, dtype=cp.float64)
    return send, recv


def mpi_is_cuda_aware(comm: Any = None, *, method: str = "probe") -> bool:
    """Check whether the MPI library can send and receive device buffers.

    Passing CuPy arrays to MPI requires a CUDA-aware MPI build (e.g. Open MPI
    with ``--with-cuda``); with a plain build the call segfaults or silently
    sends garbage. This is a collective call: every rank of `comm` must call
    it, and all ranks receive the same result.

    Parameters
    ----------
    comm
        The communicator to check; ``None`` means ``mpi4py.MPI.COMM_WORLD``
        (``mpi4py`` is imported only then, and only on the CuPy backend).
    method
        Only ``"probe"`` is available: each rank sends a tiny device buffer to
        rank ``(rank + 1) % size`` and receives from ``(rank - 1) % size`` with
        ``Sendrecv`` (with one rank, to itself), checks the received values,
        and the ranks agree with ``allreduce(op=LAND)``. Any exception in the
        exchange, on any rank, gives ``False``. ``mpi4py`` does not expose the
        library's own query (``MPIX_Query_cuda_support``), and the library
        version string is not a reliable indicator, so no ``"query"`` method
        is offered.

    Returns
    -------
    bool
        ``True`` if all ranks exchanged a device buffer successfully. ``False``
        on the NumPy backend and without a functional CuPy, without touching
        MPI: the question only makes sense with device buffers.

    Notes
    -----
    Not every failure is an exception: an MPI library that is not CUDA-aware
    may read the device address as a host address and crash the process. A
    segfault inside this call therefore also means that the MPI build is not
    CUDA-aware. Call it once at startup, after `bind_local_device()` and
    ``MPI_Init``, before any communication of device buffers.
    """
    if method != "probe":
        raise ValueError(
            f"Unknown method {method!r}; only 'probe' is available (mpi4py does "
            "not expose a reliable CUDA support query)."
        )
    if not _device_buffers_in_use():
        return False

    MPI = _mpi_module()
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size
    dest = (rank + 1) % size
    source = (rank - 1) % size

    ok = False
    try:
        send, recv = _mpi_probe_buffers(rank)
        synchronize_for_mpi(send, recv)
        comm.Sendrecv(send, dest, recvbuf=recv, source=source)
        expected = np.arange(send.size, dtype=np.float64) + source
        ok = bool(np.array_equal(to_numpy(recv), expected))
    except Exception as e:  # noqa: BLE001 - any failure means "not CUDA-aware"
        _logger.debug("CUDA-aware MPI probe failed on rank %d: %r", rank, e)
        ok = False

    return bool(comm.allreduce(ok, op=MPI.LAND))


def require_cuda_aware_mpi(comm: Any = None) -> None:
    """Raise if device buffers cannot be passed to MPI (no-op on NumPy).

    Runs `mpi_is_cuda_aware()` and raises `RuntimeError` with instructions
    when it returns ``False`` on the CuPy backend. Collective: every rank of
    `comm` must call it. On the NumPy backend, or without a functional CuPy,
    it returns without touching MPI.

    Parameters
    ----------
    comm
        The communicator to check; ``None`` means ``mpi4py.MPI.COMM_WORLD``.
    """
    if not _device_buffers_in_use():
        return
    if mpi_is_cuda_aware(comm):
        return
    raise RuntimeError(
        "The MPI library cannot send or receive device (CuPy) buffers: it is "
        "not CUDA-aware. Use a CUDA-aware MPI build, e.g. Open MPI configured "
        "with --with-cuda (and UCX built with CUDA), or MPICH with "
        "--with-device=ch4:ucx on a CUDA-enabled UCX; on clusters, load the "
        "CUDA-aware MPI module (its name differs per site) and rebuild mpi4py "
        "against it. Alternatively, copy the buffers to the host with "
        "xp.to_numpy() before every MPI call."
    )


def memory_info() -> tuple[int, int] | None:
    """Return `(free, total)` bytes of memory on the active CUDA device.

    Returns `None` on the NumPy backend. Queries the CUDA runtime directly,
    so it reflects the whole device rather than just CuPy's memory pool.
    """
    if array_backend.backend != "cupy":
        return None

    import cupy as cp

    return cp.cuda.runtime.memGetInfo()


#: Dynamic shared memory per block that every CUDA device provides without an
#: opt-in (48 KiB); also the answer of :func:`max_shared_memory_per_block`
#: without a GPU.
DEFAULT_SHARED_MEMORY_PER_BLOCK = 48 * 1024


def max_shared_memory_per_block(
    device: int | None = None, *, opt_in: bool = False
) -> int:
    """Bytes of shared memory a block of a CUDA kernel may use on `device`.

    Use it to decide whether a per-block buffer (e.g. a copy of a small grid
    for a deposit) fits, instead of a hard-coded limit.

    Parameters
    ----------
    device : int | None
        CUDA device id; the current device by default.
    opt_in : bool
        The larger limit a kernel can opt in to on newer GPUs (e.g. 99 or 227
        KiB). Using more than the default 48 KiB needs the kernel attribute
        ``max_dynamic_shared_size_bytes`` set on the compiled ``cupy.RawKernel``
        (``kernel.compile()``).

    Returns
    -------
    int
        The limit in bytes; :data:`DEFAULT_SHARED_MEMORY_PER_BLOCK` if CuPy is
        not available (so that code choosing a GPU strategy runs everywhere).
    """
    if not cupy_available():
        return DEFAULT_SHARED_MEMORY_PER_BLOCK
    import cupy as cp

    dev = cp.cuda.Device() if device is None else cp.cuda.Device(device)
    key = "MaxSharedMemoryPerBlockOptin" if opt_in else "MaxSharedMemoryPerBlock"
    return int(dev.attributes.get(key, DEFAULT_SHARED_MEMORY_PER_BLOCK))


def free_memory() -> None:
    """Release all free blocks held by CuPy's memory pools (no-op on NumPy).

    CuPy caches freed device (and pinned host) memory in pools rather than
    returning it to the driver/OS immediately, which can look like a leak
    in long-running processes. Call this to give it back.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def pin_memory(array: Any) -> Any:
    """Copy a host array into pinned (page-locked) CUDA host memory.

    Pinned memory transfers to/from the GPU faster than regular pageable
    memory, since the driver can DMA it directly. `array` must already be
    on the host (use `to_numpy()` first if it may be on the GPU). Raises
    `ImportError` if CuPy is not available.
    """
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import cupy as cp

    array = np.asarray(array)
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    pinned = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    pinned[...] = array
    return pinned


@contextmanager
def stream() -> Generator[Any, None, None]:
    """Context manager for a CUDA stream, to overlap transfers and compute.

    On the CuPy backend, operations issued inside the block are enqueued on
    a new, non-blocking stream rather than the default one. Call
    `xp.synchronize()` (or the yielded stream's own `.synchronize()`) before
    reading results computed inside the block. No-op on the NumPy backend,
    where it yields `None`.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        with cp.cuda.Stream(non_blocking=True) as s:
            yield s
    else:
        yield None


def get_rng(seed: int | None = None) -> Any:
    """Return a random Generator matching the active backend.

    NumPy and CuPy both provide `default_rng(seed)`, returning a
    `Generator` with a largely-compatible distribution API, but picking the
    right one requires branching on the backend -- this does that for you.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        return cp.random.default_rng(seed)

    import numpy as numpy_raw

    return numpy_raw.random.default_rng(seed)


def default_float_dtype() -> Any:
    """Return the active backend's `float64` dtype object.

    NumPy and CuPy resolve Python `int`/`float` literals and the bare
    `dtype=float` spelling to a platform- or backend-dependent default
    (e.g. NumPy's default integer width differs between Windows and
    Linux/macOS). Use ``dtype=xp.default_float_dtype()`` instead of
    ``dtype=float`` when a specific, portable precision matters.
    """
    return array_backend.xp.float64


def synchronize() -> None:
    """Wait for all kernels in all streams on current device to complete."""
    if array_backend.backend == "cupy":
        try:
            import cupy as cp

            cp.cuda.Device().synchronize()
        except ImportError:
            pass
        except AttributeError as e:
            warnings.warn(
                f"CuPy synchronize() failed unexpectedly, this may indicate a "
                f"CuPy API mismatch: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def _nvtx_module() -> ModuleType | None:
    """Return ``cupy.cuda.nvtx`` on the CuPy backend, else None.

    None is also returned when NVTX is not available in this CuPy build, so
    callers can degrade to a no-op instead of failing.
    """
    if array_backend.backend != "cupy":
        return None
    try:
        return importlib.import_module("cupy.cuda.nvtx")
    except Exception:  # noqa: BLE001 - tolerate any missing/broken NVTX
        return None


class nvtx_range(ContextDecorator):
    """Mark a code region as an NVTX range, visible in ``nsys`` and Nsight.

    On the CuPy backend the block is wrapped in ``cupy.cuda.nvtx.RangePush``
    / ``RangePop``, so the region appears on the profiler timeline next to
    the kernels it launches. On the NumPy backend, or if NVTX is not
    available, it is a no-op. The range is popped when the block raises.

    It can also be used as a decorator, and the same instance can be nested
    or re-entered (e.g. on a recursive function).

    Parameters
    ----------
    name : str
        Name shown in the profiler.
    color : int, optional
        Index into NVTX's colour table (``id_color`` of ``RangePush``).
        ``None`` uses the default colour.

    Examples
    --------
    >>> with xp.nvtx_range("push markers"):
    ...     kernel(markers, dt, n_threads=n)

    >>> @xp.nvtx_range("accumulate")
    ... def accumulate(...):
    ...     ...
    """

    def __init__(self, name: str, color: int | None = None) -> None:
        self.name = str(name)
        self.color = color
        self._stack: list[ModuleType | None] = []

    def __enter__(self):
        nvtx = _nvtx_module()
        if nvtx is not None:
            if self.color is None:
                nvtx.RangePush(self.name)
            else:
                nvtx.RangePush(self.name, id_color=int(self.color))
        self._stack.append(nvtx)
        return self

    def __exit__(self, *exc_info: object) -> None:
        nvtx = self._stack.pop()
        if nvtx is not None:
            nvtx.RangePop()

    def __repr__(self) -> str:
        return f"nvtx_range(name={self.name!r}, color={self.color!r})"


@dataclass
class Timing:
    """Result of a `timed_region()` block.

    Attributes
    ----------
    name : str
        Name of the region.
    elapsed : float | None
        Wall-clock seconds spent in the block; ``None`` until the block
        exits.
    synced : bool
        Whether the device was synchronized before the clock was read, i.e.
        whether `elapsed` includes the queued device work.
    """

    name: str
    elapsed: float | None = None
    synced: bool = False


@contextmanager
def timed_region(name: str, *, sync: bool = True) -> Generator[Timing, None, None]:
    """Time a code region, including the device work it queues.

    CUDA kernels run asynchronously: a wall-clock timer around a launch
    measures the launch, not the kernel. On the CuPy backend this context
    manager synchronizes the device on entry (so earlier queued work is not
    charged to the region) and, if `sync` is true, again on exit before the
    clock is read, so the measured time includes the kernels launched in
    the block. It also pushes an `nvtx_range()` of the same name, so the
    region shows in ``nsys``. On the NumPy backend it only times the block.
    The time is recorded when the block raises as well.

    Parameters
    ----------
    name : str
        Name of the region (also the NVTX range name).
    sync : bool, default True
        Synchronize the device before reading the clock on exit. With
        ``False`` the time is the host time only, as for a plain timer.

    Yields
    ------
    Timing
        `elapsed` is set (in seconds, from ``time.perf_counter``) when the
        block exits; `synced` tells whether the device was synchronized.

    Examples
    --------
    >>> with xp.timed_region("push markers") as timing:
    ...     kernel(markers, dt, n_threads=n)
    >>> print(f"{timing.name}: {timing.elapsed:.3f} s (synced={timing.synced})")
    """
    timing = Timing(name=str(name))
    on_gpu = array_backend.backend == "cupy"
    with nvtx_range(timing.name):
        if sync and on_gpu:
            synchronize()
        start = time.perf_counter()
        try:
            yield timing
        finally:
            if sync and on_gpu:
                synchronize()
                timing.synced = True
            timing.elapsed = time.perf_counter() - start


# CUDA debug mode: `CudaKernel` compiles with line information and bounds
# checks, and synchronizes after every launch so that asynchronous CUDA errors
# are raised at the kernel that caused them.
_CUDA_DEBUG_TRUE = ("1", "true", "yes", "on")


def _debug_from_env(value: str | None) -> bool:
    """Whether the value of ``CUNUMPY_CUDA_DEBUG`` enables the debug mode.

    ``"1"``, ``"true"``, ``"yes"`` and ``"on"`` (any case, surrounding
    whitespace ignored) enable it; anything else, including unset, does not.
    """
    if value is None:
        return False
    return value.strip().lower() in _CUDA_DEBUG_TRUE


_cuda_debug: bool = _debug_from_env(os.getenv("CUNUMPY_CUDA_DEBUG"))


def set_cuda_debug(enabled: bool) -> None:
    """Enable or disable the CUDA debug mode globally.

    In debug mode, `CudaKernel`s created with ``debug=None`` (the default)
    compile with ``-lineinfo`` and ``-DCUNUMPY_BOUNDS_CHECK``, and synchronize
    the stream after every launch, re-raising an asynchronous CUDA error as a
    ``RuntimeError`` naming the kernel that caused it. The setting is read at
    every launch, so it also applies to kernels created earlier; only their
    compile options are fixed once they are compiled (call ``compile()`` again
    or create the kernels after enabling debug mode). Initialised from the
    environment variable ``CUNUMPY_CUDA_DEBUG`` (``1``, ``true``, ``yes`` or
    ``on``) at import.
    """
    global _cuda_debug
    _cuda_debug = bool(enabled)


def get_cuda_debug() -> bool:
    """Whether the CUDA debug mode is enabled globally, see `set_cuda_debug`."""
    return _cuda_debug


@contextmanager
def cuda_debug(enabled: bool = True) -> Generator[None, None, None]:
    """Temporarily enable (or disable) the CUDA debug mode.

    Restores the previous setting on exit, see `set_cuda_debug`.
    """
    previous = _cuda_debug
    set_cuda_debug(enabled)
    try:
        yield
    finally:
        set_cuda_debug(previous)


def _to_numpy(array: Any) -> np.ndarray:
    """`to_numpy` without transfer counting, for internal use."""
    if get_array_backend(array) == "cupy":
        return array.get()

    return np.asarray(array)


def _to_cupy(array: Any) -> Any:
    """`to_cupy` without transfer counting, for internal use."""
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import array_api_compat.cupy as cp

    return cp.asarray(array)


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array to a NumPy array.

    A CuPy array is copied to the host, which `count_transfers()` counts as a
    ``to_host`` transfer; anything else is passed through `numpy.asarray`.
    """
    if _COUNTERS and get_array_backend(array) == "cupy":
        _record("to_host", f"to_numpy({_describe(array)})")
    return _to_numpy(array)


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array.

    Anything that is not a CuPy array already is copied to the device, which
    `count_transfers()` counts as a ``to_device`` transfer.
    """
    if _COUNTERS and get_array_backend(array) != "cupy":
        _record("to_device", f"to_cupy({_describe(array)})")
    return _to_cupy(array)


def as_device_array(
    value: Any,
    dtype: Any = None,
    ndim: int | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Reference `value` on the device, or make one device copy of it.

    The "reference or copy once" rule for building CUDA argument objects
    (`CudaArguments` subclasses, `CudaStruct` values): call it once when the
    argument object is built, never per kernel call. A CuPy array that already
    has the requested `dtype` (any dtype if `dtype` is None) and is C-contiguous
    is returned unchanged, the same object without a copy, so kernels write
    into the caller's array. Anything else is converted with one device copy,
    ``cupy.ascontiguousarray(cupy.asarray(value, dtype))``: a tuple or list
    (e.g. ``degree = (3, 3, 3)``), a host NumPy array (one explicit transfer
    at build time), a device array of another dtype, or a non-contiguous view.
    The result passes the pointer checks of `CudaKernel` and `CudaStruct`.

    Raises on the NumPy backend: device argument objects are only built when
    running on CuPy, and host data is never copied to the device implicitly.

    Parameters
    ----------
    value : array-like
        A CuPy array, a NumPy array, or a sequence of numbers.
    dtype : dtype-like, optional
        The dtype the kernel expects, e.g. the pointed-to type of the
        parameter. None keeps the dtype of `value`.
    ndim : int, optional
        The expected number of dimensions of the result.
    name : str, optional
        Name of the argument, used in error messages.

    Returns
    -------
    cupy.ndarray
        `value` itself, or a C-contiguous device copy with dtype `dtype`.

    Raises
    ------
    RuntimeError
        The active backend is not CuPy.
    ValueError
        `ndim` is given and the array has another number of dimensions.
    """
    what = f"device argument {name!r}" if name is not None else "device argument"
    if array_backend.backend != "cupy":
        raise RuntimeError(
            f"{what}: the active backend is {array_backend.backend!r}; device "
            "arguments are only built on the CuPy backend, and host data is never "
            "copied to the device implicitly (build host arguments instead)"
        )

    import cupy as cp

    if (
        isinstance(value, cp.ndarray)
        and value.flags.c_contiguous
        and (dtype is None or value.dtype == np.dtype(dtype))
    ):
        result = value
    else:
        result = cp.ascontiguousarray(cp.asarray(value, dtype=dtype))
    if ndim is not None and result.ndim != ndim:
        raise ValueError(
            f"{what} must have {ndim} dimension(s), got {result.ndim} "
            f"(shape {result.shape})"
        )
    return result


def to_cunumpy(array: Any) -> Any:
    """Convert an array to the currently active backend.

    Delegates to `to_cupy()` or `to_numpy()`, so an actual copy is counted by
    `count_transfers()` as a ``to_device`` or ``to_host`` transfer.
    """
    if array_backend.backend == "cupy" and cupy_available():
        return to_cupy(array)
    return to_numpy(array)


def get_array_backend(array: Any) -> BackendType:
    """Return 'cupy' or 'numpy' depending on the array's type."""
    return "cupy" if array_api_compat.is_cupy_array(array) else "numpy"


def get_array_module(array: Any) -> ModuleType:
    """Return the array-api-compat module matching `array`'s own backend.

    Unlike `xp.xp`, which reflects the process-wide active backend, this
    dispatches on the array itself -- useful for writing functions that
    operate correctly regardless of what `set_backend`/`use_backend` last
    selected. Mirrors `cupy.get_array_module`, but also works in Pyodide
    (where `cupy` cannot be imported) and returns array-api-compat modules
    for standard-conformant behavior, consistent with `xp.xp`.
    """
    if get_array_backend(array) == "cupy":
        import array_api_compat.cupy as cp

        return cp
    return np


def is_gpu(array: Any) -> bool:
    """Check if the array is stored on a GPU (CuPy)."""
    return get_array_backend(array) == "cupy"


def is_cpu(array: Any) -> bool:
    """Check if the array is stored on a CPU (NumPy)."""
    return get_array_backend(array) == "numpy"


def same_backend(*arrays: Any) -> bool:
    """Return True if all given arrays live on the same backend.

    Trivially True for zero or one array.
    """
    if len(arrays) <= 1:
        return True
    backends = {get_array_backend(array) for array in arrays}
    return len(backends) == 1


def assert_same_backend(*arrays: Any) -> None:
    """Raise TypeError if the given arrays don't all live on the same backend.

    Mixing NumPy and CuPy arrays in an operation typically fails with a
    confusing, backend-internal error (e.g. a CuPy kernel dispatch error
    complaining about an "unsupported type"). Call this upfront to fail
    with a clear message instead. Use `to_cunumpy()`/`to_numpy()`/`to_cupy()`
    to align mismatched arrays onto one backend first.
    """
    if not same_backend(*arrays):
        backends = [get_array_backend(array) for array in arrays]
        raise TypeError(
            f"Arrays are on mismatched backends: {backends}. Use "
            "xp.to_cunumpy()/xp.to_numpy()/xp.to_cupy() to align them first."
        )


# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    import numpy as xp  # noqa: F401 - type-checker-only alias for autocompletion
else:
    # Use module-level __getattr__ for dynamic xp (Python 3.7+)
    def __getattr__(name):
        if name == "xp":
            return array_backend.xp
        if name == "numpy_backend":
            return _numpy_backend()
        if name == "cupy_backend":
            return _cupy_backend()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
