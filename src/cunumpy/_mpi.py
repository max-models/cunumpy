"""MPI with NumPy or CuPy buffers (see :mod:`cunumpy.mpi`)."""

from __future__ import annotations

import logging
import operator
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import array_api_compat
import array_api_compat.numpy as np

from ._mpi_serial import _LOCAL_RANK_VARIABLES, local_rank  # noqa: F401 - re-exported
from ._transfers import _ACTIVE as _COUNTERS
from ._transfers import _describe, _record
from .xp import array_backend, cupy_available, to_numpy

_logger = logging.getLogger(__name__)


def synchronize_for_mpi(*arrays: Any, stream: Any = None, event: Any = None) -> None:
    """Wait for pending device work before MPI reads or writes `arrays`.

    CuPy launches kernels asynchronously; MPI does not know about CUDA streams.
    Passing a device buffer to MPI while a kernel is still writing it sends
    whatever is in memory at that moment -- silently wrong data, no error. Call
    this before every MPI call that uses device buffers. It synchronizes the
    devices owning the arrays if no producer `stream` or `event` is supplied.
    An explicit dependency waits only for that producer. Host
    buffers and the NumPy backend cost nothing. (After MPI returns, no
    synchronization is needed: kernels launched later see the received data.)

    Parameters
    ----------
    *arrays
        The buffers about to be passed to MPI; ``None`` entries are ignored.
    """
    if stream is not None and event is not None:
        raise ValueError("pass only one of stream and event")
    devices = {a.device.id for a in arrays if array_api_compat.is_cupy_array(a)}
    if not devices:
        return
    from ._streams import HostEvent, HostStream

    if isinstance(event, HostEvent) or isinstance(stream, HostStream):
        raise TypeError("device buffers require a CUDA producer stream or event")
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
    """Tell `mpi_buffer()` whether MPI can take device buffers.

    `mpi_is_cuda_aware()` records its result itself; call this instead when
    the answer is known otherwise (e.g. from the cluster documentation, or to
    force host staging for a test). ``None`` forgets the setting.
    """
    global _MPI_CUDA_AWARE
    _MPI_CUDA_AWARE = None if value is None else bool(value)


def get_mpi_cuda_aware() -> bool | None:
    """The recorded answer of `mpi_is_cuda_aware()`/`set_mpi_cuda_aware()`, or None."""
    return _MPI_CUDA_AWARE


def _pinned_or_host_empty(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """A host buffer for staging, pinned when CuPy can allocate pinned memory."""
    try:
        import cupy as cp

        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        mem = cp.cuda.alloc_pinned_memory(max(nbytes, 1))
        return np.frombuffer(mem, dtype, int(np.prod(shape, dtype=np.int64))).reshape(
            shape
        )
    except Exception:  # noqa: BLE001 - no CuPy, no pinned memory, the fake CuPy, ...
        return np.empty(shape, dtype=dtype)


class MPIStaging:
    """Reusable host storage for blocking MPI exchanges of device arrays.

    Allocate once per send/receive buffer. Storage is allocated lazily, pinned
    when available, and bound to one shape, dtype, and CUDA device. Only one
    context may use it at a time. For nonblocking MPI, call request.Wait() inside
    the context: the buffer must not be released or copied back while MPI uses it.
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
        """Equivalent to ``mpi_buffer(array, staging=self, **kwargs)``."""
        return mpi_buffer(array, staging=self, **kwargs)

    def _transfer_array(self, array: Any) -> Any:
        """Reuse C-order device storage when the MPI array has another layout."""
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
    """The buffer to hand to MPI for `array`: the array itself, or a host copy.

    One MPI call site for both backends and both kinds of MPI builds::

        with xp.mpi.mpi_buffer(markers_out) as sendbuf, xp.mpi.mpi_buffer(
            markers_in, send=False, recv=True
        ) as recvbuf:
            comm.Sendrecv(sendbuf, dest, recvbuf=recvbuf, source=source)

    * A host array (NumPy backend, or a NumPy array on the CuPy backend) is
      yielded unchanged.
    * A device array with CUDA-aware MPI is yielded unchanged after
      `synchronize_for_mpi()`, so MPI reads what the kernels wrote.
    * A device array without CUDA-aware MPI is staged through a host buffer
      (pinned memory when available): with `send`, the array is copied to the
      host first (counted as a ``to_host`` transfer by `count_transfers()`);
      with `recv`, the host buffer is copied back into the array when the
      block ends (a ``to_device`` transfer). The device array itself is never
      given to MPI.

    Parameters
    ----------
    array
        The buffer of the MPI call: a NumPy or CuPy array.
    send : bool
        Whether MPI reads the buffer (copy device to host before the block).
    recv : bool
        Whether MPI writes the buffer (copy host to device after the block).
    cuda_aware : bool | None
        Whether MPI can take device buffers. None uses the answer recorded by
        `mpi_is_cuda_aware()` or `set_mpi_cuda_aware()`.
    staging : MPIStaging | None
        Reusable storage for host staging. For nonblocking MPI, wait for the
        request inside the context before releasing this storage.
    stream, event
        Optional producer dependency (only one); without it all work on the
        array's device is synchronized before MPI accesses the buffer.

    Raises
    ------
    RuntimeError
        For a device array when `cuda_aware` is None and nothing was recorded:
        call `mpi_is_cuda_aware(comm)` (collective) once at startup, or
        `set_mpi_cuda_aware()`.
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
            "xp.mpi.set_mpi_cuda_aware(True/False), or pass cuda_aware="
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
            if _COUNTERS:
                _record("to_host", f"mpi_buffer({_describe(array)}) staging for send")
            if transfer_array is not array:
                transfer_array[...] = array
            transfer_array.get(out=host)
        yield host
        if recv:
            if _COUNTERS:
                _record("to_device", f"mpi_buffer({_describe(array)}) staging for recv")
            # Ensure MPI's host buffer can be reused immediately on context exit.
            transfer_array.set(host)
            if transfer_array is not array:
                array[...] = transfer_array
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

    result = bool(comm.allreduce(ok, op=MPI.LAND))
    set_mpi_cuda_aware(result)
    return result


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
