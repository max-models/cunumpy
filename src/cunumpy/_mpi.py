"""MPI with NumPy or CuPy buffers (see :mod:`cunumpy.mpi`)."""

from __future__ import annotations

import logging
import operator
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import array_api_compat
import array_api_compat.numpy as np
from maybempi import get_mpi

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _describe, _nbytes, _record, _record_sync
from cunumpy.xp import array_backend, cupy_available, to_numpy

_logger = logging.getLogger(__name__)


def synchronize_for_mpi(*arrays: Any, stream: Any = None, event: Any = None) -> None:
    """Wait for pending device work before MPI reads or writes `arrays`.

    CuPy launches kernels asynchronously and MPI knows nothing about CUDA
    streams: a device buffer that a kernel is still writing is sent as it is,
    silently wrong. Call this before every MPI call with device buffers. No
    synchronization is needed after MPI returns. Host arrays and the NumPy
    backend cost nothing.

    Parameters
    ----------
    *arrays : array or None
        The buffers about to be passed to MPI; None and host arrays are
        ignored.
    stream : cupy.cuda.Stream, optional
        Producer stream to wait for, instead of the whole devices owning
        `arrays`. It must cover all the device buffers.
    event : cupy.cuda.Event, optional
        Producer event to wait for instead; pass at most one of `stream` and
        `event`.

    Raises
    ------
    ValueError
        If both `stream` and `event` are given.
    TypeError
        If device buffers come with a host stream or event.

    See Also
    --------
    mpi_buffer : Synchronizes, and stages through the host if needed.

    Examples
    --------
    >>> send, recv = np.ones(4), np.empty(4)
    >>> xp.mpi.synchronize_for_mpi(send, recv)  # no-op for host arrays
    """
    if stream is not None and event is not None:
        raise ValueError("pass only one of stream and event")
    devices = {a.device.id for a in arrays if array_api_compat.is_cupy_array(a)}
    if not devices:
        return
    from cunumpy._streams import HostEvent, HostStream

    if isinstance(event, HostEvent) or isinstance(stream, HostStream):
        raise TypeError("device buffers require a CUDA producer stream or event")
    if _COUNTERS:
        _record_sync("synchronize_for_mpi()")
    if event is not None:
        event.synchronize()
        return
    if stream is not None:
        stream.synchronize()
        return

    import cupy as cp

    for device in devices:
        cp.cuda.Device(device).synchronize()


# the result of the last mpi_is_cuda_aware() probe (or of set_mpi_cuda_aware()),
# used by mpi_buffer(); None until one of them was called
_MPI_CUDA_AWARE: bool | None = None


def set_mpi_cuda_aware(value: bool | None) -> None:
    """Tell :func:`mpi_buffer` whether MPI can take device buffers.

    :func:`mpi_is_cuda_aware` records the result of its probe itself; call
    this when the answer is known otherwise (e.g. from the cluster
    documentation, or to force host staging in a test).

    Parameters
    ----------
    value : bool or None
        Whether MPI is CUDA-aware; None forgets the setting.

    See Also
    --------
    get_mpi_cuda_aware : Read the setting.

    Examples
    --------
    >>> xp.mpi.set_mpi_cuda_aware(False)  # always stage device buffers
    >>> xp.mpi.get_mpi_cuda_aware()
    False
    >>> xp.mpi.set_mpi_cuda_aware(None)
    """
    global _MPI_CUDA_AWARE
    _MPI_CUDA_AWARE = None if value is None else bool(value)


def get_mpi_cuda_aware() -> bool | None:
    """Return whether MPI was recorded as CUDA-aware.

    Returns
    -------
    bool or None
        The answer recorded by :func:`mpi_is_cuda_aware` or
        :func:`set_mpi_cuda_aware`; None if nothing was recorded.

    Examples
    --------
    >>> xp.mpi.get_mpi_cuda_aware() is None
    True
    """
    return _MPI_CUDA_AWARE


def _pinned_or_host_empty(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """Return a host buffer for staging, pinned when CuPy can allocate pinned memory."""
    try:
        import cupy as cp

        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        mem = cp.cuda.alloc_pinned_memory(max(nbytes, 1))
        return np.frombuffer(mem, dtype, int(np.prod(shape, dtype=np.int64))).reshape(
            shape,
        )
    except Exception:  # noqa: BLE001 - no CuPy, no pinned memory, the fake CuPy, ...
        return np.empty(shape, dtype=dtype)


class MPIStaging:
    """Reusable host storage for :func:`mpi_buffer` staging of device arrays.

    Make one per buffer that is in use at the same time (e.g. one for sending,
    one for receiving). The host storage is allocated on first use, pinned
    when available, and bound to one shape, dtype and CUDA device; only one
    :func:`mpi_buffer` context may use it at a time. Non-C-contiguous device
    arrays go through a reusable device packing buffer, and received values
    are written back into the original view. Host arrays pass through
    unchanged.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the staged arrays.
    dtype : dtype-like
        Their dtype (not object).

    Raises
    ------
    ValueError
        For a negative shape.
    TypeError
        For an object dtype.

    Examples
    --------
    >>> staging = xp.mpi.MPIStaging((3,), np.float64)
    >>> rho = np.ones(3)
    >>> with staging.buffer(rho, recv=True) as buf:
    ...     buf is rho  # host arrays are not staged
    True
    """

    def __init__(self, shape: int | tuple[int, ...], dtype: Any) -> None:
        self._shape = (
            (operator.index(shape),)
            if isinstance(shape, (int, np.integer))
            else tuple(operator.index(n) for n in shape)
        )
        if any(n < 0 for n in self.shape):
            raise ValueError("staging shape must be non-negative")
        self._dtype = np.dtype(dtype)
        if self.dtype.hasobject:
            raise TypeError("MPI staging does not support object dtype")
        self._host: np.ndarray | None = None
        self._packed: Any = None
        self._device: int | None = None
        self._active = False

    @property
    def shape(self) -> tuple[int, ...]:
        """The fixed staging shape."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The fixed staging dtype."""
        return self._dtype

    def buffer(self, array: Any, **kwargs: Any) -> Any:
        """Return ``mpi_buffer(array, staging=self, **kwargs)``.

        Parameters
        ----------
        array : array
            The buffer of the MPI call.
        **kwargs
            Further arguments of :func:`mpi_buffer`.

        Returns
        -------
        context manager
            Yields the buffer to pass to MPI.
        """
        return mpi_buffer(array, staging=self, **kwargs)

    def _transfer_array(self, array: Any) -> Any:
        """Return `array`, or reusable C-order device storage for another layout."""
        if array.flags.c_contiguous:
            return array
        if self._packed is None:
            import cupy as cp

            self._packed = cp.empty(self.shape, dtype=self.dtype)
        return self._packed

    @contextmanager
    def _lease(self, array: Any) -> Generator[np.ndarray, None, None]:
        if self._active:
            raise RuntimeError("MPI staging buffer is already in use")
        if tuple(array.shape) != self.shape or np.dtype(array.dtype) != self.dtype:
            raise ValueError("MPI staging shape and dtype must match the array")
        device = array.device.id
        if self._device is not None and device != self._device:
            raise ValueError("MPI staging buffer is bound to another CUDA device")
        self._device = device
        if self._host is None:
            self._host = _pinned_or_host_empty(self.shape, self.dtype)
        self._active = True
        try:
            yield self._host
        finally:
            self._active = False


@contextmanager
def mpi_buffer(
    array: Any,
    *,
    send: bool = True,
    recv: bool = False,
    cuda_aware: bool | None = None,
    staging: MPIStaging | None = None,
    stream: Any = None,
    event: Any = None,
) -> Generator[Any, None, None]:
    """Yield the buffer to hand to MPI for `array`: the array itself, or a host copy.

    One MPI call site for both backends and both kinds of MPI builds. A host
    array is yielded unchanged. A device array is yielded unchanged after
    :func:`synchronize_for_mpi` when MPI is CUDA-aware; otherwise it is staged
    through a host buffer (pinned when available), copied from the device
    before the block (`send`) and back after it (`recv`), both counted by
    :func:`~cunumpy.profiling.count_transfers`. The copy-back has finished
    when the block ends. With nonblocking MPI, call ``request.Wait()`` inside
    the block.

    Parameters
    ----------
    array : array
        The buffer of the MPI call: a NumPy or CuPy array.
    send : bool, optional
        Whether MPI reads the buffer (copy device to host before the block).
    recv : bool, optional
        Whether MPI writes the buffer (copy host to device after the block).
    cuda_aware : bool, optional
        Whether MPI can take device buffers. None uses the answer recorded by
        :func:`mpi_is_cuda_aware` or :func:`set_mpi_cuda_aware`.
    staging : MPIStaging, optional
        Reusable host storage; a new one is allocated for each call otherwise.
    stream : cupy.cuda.Stream, optional
        Producer stream to wait for; all work on the array's device by
        default.
    event : cupy.cuda.Event, optional
        Producer event to wait for instead; pass at most one of `stream` and
        `event`.

    Yields
    ------
    array
        The buffer to pass to MPI.

    Raises
    ------
    RuntimeError
        For a device array when `cuda_aware` is None and nothing was recorded:
        call :func:`mpi_is_cuda_aware` (collective) once at startup, or
        :func:`set_mpi_cuda_aware`.

    See Also
    --------
    MPIStaging : Reusable staging storage.

    Examples
    --------
    >>> MPI = xp.mpi.get_mpi()  # the serial stand-in without an MPI launcher
    >>> rho = np.ones(3)
    >>> with xp.mpi.mpi_buffer(rho, recv=True) as buf:
    ...     MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    >>> rho
    array([1., 1., 1.])
    """
    if not array_api_compat.is_cupy_array(array):
        yield array
        return
    if cuda_aware is None:
        cuda_aware = _MPI_CUDA_AWARE
    if cuda_aware is None:
        raise RuntimeError(
            "mpi_buffer(): it is not known whether MPI can take device buffers; "
            "call xp.mpi.mpi_is_cuda_aware(comm) once at startup (every rank), or "
            "xp.mpi.set_mpi_cuda_aware(True/False), or pass cuda_aware=",
        )
    if cuda_aware:
        synchronize_for_mpi(array, stream=stream, event=event)
        yield array
        return
    if staging is None:
        staging = MPIStaging(tuple(array.shape), array.dtype)
    import cupy as cp

    with cp.cuda.Device(array.device.id), staging._lease(array) as host:
        # Even receive-only buffers may still be read/written by a GPU kernel.
        synchronize_for_mpi(array, stream=stream, event=event)
        transfer_array = staging._transfer_array(array)
        if send:
            if transfer_array is not array:
                transfer_array[...] = array
            transfer_array.get(out=host)
            if _COUNTERS:
                _record(
                    "to_host",
                    f"mpi_buffer({_describe(array)}) staging for send",
                    nbytes=_nbytes(array),
                )
        yield host
        if recv:
            transfer_array.set(host)
            if _COUNTERS:
                _record(
                    "to_device",
                    f"mpi_buffer({_describe(array)}) staging for recv",
                    nbytes=_nbytes(array),
                )
            # Ensure MPI's host buffer can be reused immediately on context exit.
            if transfer_array is not array:
                array[...] = transfer_array
            if _COUNTERS:
                _record_sync("mpi_buffer() staging for recv")
            cp.cuda.get_current_stream().synchronize()


def _device_buffers_in_use() -> bool:
    """Whether MPI calls of this process would carry device (CuPy) buffers."""
    return array_backend.backend == "cupy" and cupy_available()


def _mpi_probe_buffers(rank: int, n: int = 4) -> tuple[Any, Any]:
    """Device send/receive buffers for the CUDA-aware probe; rank ``r`` sends ``r + arange(n)``."""
    import cupy as cp

    send = cp.arange(n, dtype=cp.float64) + rank
    recv = cp.empty(n, dtype=cp.float64)
    return send, recv


def mpi_is_cuda_aware(comm: Any = None, *, method: str = "probe") -> bool:
    """Check whether the MPI library can send and receive device buffers.

    Passing CuPy arrays to MPI needs a CUDA-aware MPI build (e.g. Open MPI
    with ``--with-cuda``); with a plain build the call segfaults or silently
    sends garbage. Collective: every rank of `comm` must call it, and all get
    the same result, which is recorded for :func:`mpi_buffer`. Call it once
    at startup, after :func:`~cunumpy.cuda.bind_local_device` and
    ``MPI_Init``.

    Parameters
    ----------
    comm : Comm, optional
        The communicator; ``COMM_WORLD`` of ``cunumpy.mpi.get_mpi()`` by
        default.
    method : {"probe"}, optional
        Each rank sends a tiny device buffer to rank ``(rank + 1) % size``
        with ``Sendrecv``, checks what it received, and the ranks agree with
        ``allreduce(op=LAND)``; any exception gives False. mpi4py does
        not expose ``MPIX_Query_cuda_support``, so there is no other method.

    Returns
    -------
    bool
        True if all ranks exchanged a device buffer correctly. False on the
        NumPy backend and without a functional CuPy, without touching MPI.

    Raises
    ------
    ValueError
        For a `method` other than ``"probe"``.

    See Also
    --------
    require_cuda_aware_mpi : Raise with instructions instead of returning False.

    Notes
    -----
    A library that is not CUDA-aware may read the device address as a host
    address and crash the process: a segfault inside this call means the same
    as False. See :doc:`/guides/mpi`.

    Examples
    --------
    >>> xp.mpi.mpi_is_cuda_aware()  # NumPy backend: no device buffers
    False
    """
    if method != "probe":
        raise ValueError(
            f"Unknown method {method!r}; only 'probe' is available (mpi4py does "
            "not expose a reliable CUDA support query).",
        )
    if not _device_buffers_in_use():
        return False

    MPI = get_mpi()
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

    result = bool(comm.allreduce(ok, op=MPI.LAND))
    set_mpi_cuda_aware(result)
    return result


def require_cuda_aware_mpi(comm: Any = None) -> None:
    """Raise if device buffers cannot be passed to MPI.

    Runs :func:`mpi_is_cuda_aware` on the CuPy backend. Collective: every rank
    of `comm` must call it. On the NumPy backend, or without a functional
    CuPy, it returns without touching MPI.

    Parameters
    ----------
    comm : Comm, optional
        The communicator; ``COMM_WORLD`` of ``cunumpy.mpi.get_mpi()`` by
        default.

    Raises
    ------
    RuntimeError
        If MPI is not CUDA-aware, explaining how to get a CUDA-aware build.

    Examples
    --------
    >>> xp.mpi.require_cuda_aware_mpi()  # no-op on the NumPy backend
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
        "xp.to_numpy() before every MPI call.",
    )
