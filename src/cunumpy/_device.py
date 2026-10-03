"""CUDA devices, memory, streams and the CUDA debug mode (see :mod:`cunumpy.cuda`)."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import array_api_compat.numpy as np

from ._mpi import local_rank
from .xp import array_backend, cupy_available


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
