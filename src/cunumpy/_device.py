"""CUDA devices, memory, streams and the CUDA debug mode (see :mod:`cunumpy.cuda`)."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import array_api_compat.numpy as np
from maybempi import local_rank

from cunumpy.xp import array_backend, cupy_available


def set_device(device_id: int) -> None:
    """Select the active CUDA device of this process.

    Does nothing on the NumPy backend.

    Parameters
    ----------
    device_id : int
        A device id valid for this process.

    See Also
    --------
    set_device_for_rank : Select the device from an MPI rank.
    bind_local_device : Select the device from the node-local rank.

    Examples
    --------
    >>> xp.cuda.set_device(0)  # no-op on NumPy
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.cuda.Device(device_id).use()


def device_count() -> int:
    """Return the number of visible CUDA devices.

    Reports the visible hardware, independent of the active backend: it can be
    positive while NumPy is selected.

    Returns
    -------
    int
        The device count, or 0 if CuPy/CUDA is unavailable or the query fails.

    Examples
    --------
    >>> n = xp.cuda.device_count()  # 0 on a machine without GPUs
    """
    if not cupy_available():
        return 0

    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        return 0


def is_hip() -> bool:
    """Return whether the active CuPy build targets AMD ROCm/HIP.

    CuPy's own API (``cp.cuda.Device``, ``memory_info``, streams, ...) is the
    same on a ROCm build; this only distinguishes the GPU vendor for the
    parts that differ, such as :meth:`~cunumpy.kernels.CudaKernel.compile_options`
    (NVRTC-only flags) and the warp-level primitives of
    ``cunumpy/reduce.cuh`` and ``cunumpy/scan.cuh`` (not yet ported to HIP,
    see :doc:`/kernels/cuda-headers`).

    Returns
    -------
    bool
        True if CuPy is available and reports a HIP runtime, False for a
        CUDA build or when CuPy/a GPU is unavailable.

    Examples
    --------
    >>> xp.cuda.is_hip()  # on a CUDA build, or without CuPy
    False
    """
    if not cupy_available():
        return False

    import cupy as cp

    try:
        return bool(cp.cuda.runtime.is_hip)
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        return False


def set_device_for_rank(rank: int, devices_per_node: int | None = None) -> int:
    """Select device ``rank % devices_per_node`` for an MPI rank.

    For one-rank-per-GPU codes whose ranks fill each node in contiguous blocks;
    otherwise use :func:`bind_local_device` or :func:`set_device`.

    Parameters
    ----------
    rank : int
        The MPI rank of this process.
    devices_per_node : int, optional
        Devices per node; :func:`device_count` by default.

    Returns
    -------
    int
        ``rank % devices_per_node``, selected on the CuPy backend only (on
        NumPy the id is returned without selecting anything). 0, with nothing
        selected, if the number of devices per node is 0, e.g. when
        `devices_per_node` is omitted and no device is visible.

    See Also
    --------
    bind_local_device : Uses the node-local rank instead.

    Examples
    --------
    >>> xp.cuda.set_device_for_rank(3, devices_per_node=4)
    3
    """
    n = devices_per_node if devices_per_node is not None else device_count()
    if n == 0:
        return 0

    device_id = rank % n
    set_device(device_id)
    return device_id


def bind_local_device() -> int | None:
    """Bind this process to device ``local_rank() % device_count()`` of its node.

    Also creates the CUDA context. Call it before ``MPI_Init`` (before importing
    ``mpi4py.MPI``) so that CUDA-aware MPI sees the right device; otherwise every
    rank of a node uses device 0. With ``CUDA_VISIBLE_DEVICES`` set per rank, each
    process sees and selects one device. Needs no MPI rank, unlike
    :func:`set_device_for_rank`.

    Returns
    -------
    int or None
        The selected device id, or None on the NumPy backend or without devices.

    Examples
    --------
    >>> print(xp.cuda.bind_local_device())  # on NumPy
    None
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
    """Return the free and total memory of the active CUDA device.

    Queries the CUDA runtime, so it covers the whole device, not only CuPy's
    memory pool.

    Returns
    -------
    tuple of int or None
        ``(free, total)`` in bytes, or None on the NumPy backend.

    Examples
    --------
    >>> print(xp.cuda.memory_info())  # on NumPy
    None
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
    device: int | None = None,
    *,
    opt_in: bool = False,
) -> int:
    """Return the bytes of shared memory a CUDA block may use on `device`.

    Use it to decide whether a per-block buffer (e.g. a copy of a small grid for
    a deposit) fits, instead of a hard-coded limit.

    Parameters
    ----------
    device : int, optional
        CUDA device id; the current device by default.
    opt_in : bool, optional
        Return the larger limit a kernel can opt in to on newer GPUs (e.g. 99 or
        227 KiB). Using more than 48 KiB needs ``max_dynamic_shared_size_bytes``
        set on the compiled ``cupy.RawKernel``.

    Returns
    -------
    int
        The limit in bytes; :data:`DEFAULT_SHARED_MEMORY_PER_BLOCK` without CuPy,
        so that code choosing a GPU strategy runs everywhere.

    Examples
    --------
    >>> xp.cuda.max_shared_memory_per_block()  # without CuPy
    49152
    """
    if not cupy_available():
        return DEFAULT_SHARED_MEMORY_PER_BLOCK
    import cupy as cp

    # `cp.cuda.Device().attributes` is unreliable here on a HIP/ROCm build (it
    # can report 0 for these keys); `getDeviceProperties()`'s `sharedMemPerBlock`
    # and `sharedMemPerBlockOptin` fields are the ones `CudaKernel` itself
    # already relies on for the real launch-time check, on both backends.
    dev_id = cp.cuda.Device().id if device is None else device
    properties = cp.cuda.runtime.getDeviceProperties(dev_id)
    base = int(properties.get("sharedMemPerBlock", 0)) or DEFAULT_SHARED_MEMORY_PER_BLOCK
    if not opt_in:
        return base
    # The "opt in" larger dynamic-shared-memory limit is an NVIDIA concept
    # (cudaFuncAttributeMaxDynamicSharedMemorySize); on a HIP build there is no
    # larger limit to opt into, so the base limit is also the opt-in one.
    return max(base, int(properties.get("sharedMemPerBlockOptin", 0)))


def free_memory() -> None:
    """Release the free blocks of CuPy's device and pinned memory pools.

    CuPy caches freed memory for reuse, which can look like a leak in
    long-running processes. Blocks still used by live arrays are kept. Does
    nothing on the NumPy backend.

    Examples
    --------
    >>> xp.cuda.free_memory()  # no-op on NumPy
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def pin_memory(array: Any) -> Any:
    """Copy a host array into pinned (page-locked) host memory.

    Pinned memory transfers to and from the GPU faster than pageable memory.

    Parameters
    ----------
    array : array_like
        A host array; use :func:`~cunumpy.to_numpy` first if it may be on the GPU.

    Returns
    -------
    numpy.ndarray
        A copy of `array` backed by pinned memory.

    Raises
    ------
    ImportError
        If CuPy is not available.

    Examples
    --------
    >>> pinned = xp.cuda.pin_memory(np.zeros(1000))  # doctest: +SKIP
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
def stream(existing: Any = None) -> Generator[Any, None, None]:
    """Run the work issued in a ``with`` block on a CUDA stream.

    On CuPy, yields a new non-blocking stream (or selects `existing`); call
    :func:`~cunumpy.synchronize` or the stream's ``synchronize()`` before reading
    results computed in the block. On NumPy, yields None and does nothing, so
    portable code calls :func:`~cunumpy.synchronize` rather than methods of the
    yielded stream.

    Parameters
    ----------
    existing : stream, optional
        A reusable stream from :func:`create_stream` to select instead of a new
        one; also works with its CPU equivalent :class:`HostStream`.

    Yields
    ------
    stream or None
        The selected stream, or None on the NumPy backend without `existing`.

    Examples
    --------
    >>> with xp.cuda.stream() as s:
    ...     y = xp.ones(3) * 2
    >>> print(s)  # on NumPy
    None
    """
    if existing is not None:
        with existing:
            yield existing
    elif array_backend.backend == "cupy":
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
    """Whether a ``CUNUMPY_CUDA_DEBUG`` value enables the debug mode."""
    if value is None:
        return False
    return value.strip().lower() in _CUDA_DEBUG_TRUE


_cuda_debug: bool = _debug_from_env(os.getenv("CUNUMPY_CUDA_DEBUG"))


def set_cuda_debug(enabled: bool) -> None:
    """Enable or disable the CUDA debug mode globally.

    In debug mode, a :class:`~cunumpy.kernels.CudaKernel` created with
    ``debug=None`` (the default) compiles with ``-lineinfo`` and
    ``-DCUNUMPY_BOUNDS_CHECK`` and synchronizes after every launch, re-raising an
    asynchronous CUDA error as a ``RuntimeError`` naming the kernel. The setting
    is read at every launch, but compile options are fixed once a kernel is
    compiled. Initialized at import from ``CUNUMPY_CUDA_DEBUG`` (``1``, ``true``,
    ``yes`` or ``on``). See :doc:`/kernels/debugging`.

    Parameters
    ----------
    enabled : bool
        Whether to enable the debug mode.

    See Also
    --------
    cuda_debug : Enable it temporarily.

    Examples
    --------
    >>> xp.cuda.set_cuda_debug(True)
    >>> xp.cuda.get_cuda_debug()
    True
    >>> xp.cuda.set_cuda_debug(False)
    """
    global _cuda_debug
    _cuda_debug = bool(enabled)


def get_cuda_debug() -> bool:
    """Return whether the CUDA debug mode is enabled globally.

    Returns
    -------
    bool
        The setting of :func:`set_cuda_debug`.

    Examples
    --------
    >>> xp.cuda.get_cuda_debug()
    False
    """
    return _cuda_debug


@contextmanager
def cuda_debug(enabled: bool = True) -> Generator[None, None, None]:
    """Enable (or disable) the CUDA debug mode inside a ``with`` block.

    Restores the previous setting on exit, see :func:`set_cuda_debug`.

    Parameters
    ----------
    enabled : bool, optional
        The setting inside the block, True by default.

    Yields
    ------
    None

    Examples
    --------
    >>> with xp.cuda.cuda_debug():
    ...     xp.cuda.get_cuda_debug()
    True
    """
    previous = _cuda_debug
    set_cuda_debug(enabled)
    try:
        yield
    finally:
        set_cuda_debug(previous)
