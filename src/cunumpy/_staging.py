"""Copy device arrays to the host in the background, for output that should not stall the GPU.

``array.get()`` waits for the GPU and then for the copy, while no kernel runs.
:class:`~cunumpy.memory.HostStaging` (for large arrays copied repeatedly) and
:func:`~cunumpy.to_host_async` (for scalars and small arrays) overlap the copy
with the next kernels. Host arrays (and the NumPy backend) are copied at once,
so the code is the same on both backends. See :doc:`/guides/particle-codes`.
"""

from __future__ import annotations

import operator
from contextlib import nullcontext
from typing import Any

import array_api_compat
import numpy as np

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _describe, _nbytes, _record, _record_sync

__all__ = ["HostCopy", "HostStaging", "StagedCopy", "to_host_async"]

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array


def _cupy() -> Any:
    import cupy

    return cupy


def _empty_pinned(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    import cupyx

    return cupyx.empty_pinned(shape, dtype=dtype)


class _Slot:
    def __init__(self, host: np.ndarray, device_id: int | None = None) -> None:
        self.host = host
        self.device: Any = None  # the snapshot, allocated on the first device copy
        self.event: Any = None  # recorded after the copy to the host
        self.generation = 0
        self.device_id = device_id

    def context(self) -> Any:
        return (
            nullcontext()
            if self.device_id is None
            else _cupy().cuda.Device(self.device_id)
        )


class StagedCopy:
    """A background copy started by :meth:`HostStaging.copy`.

    Not created directly. Its host buffer is valid until the staging reuses
    it, `buffers` copies later.

    Parameters
    ----------
    slot : object
        The staging buffer (internal).
    generation : int
        The buffer's use count when this copy started.
    """

    def __init__(self, slot: _Slot, generation: int) -> None:
        self._slot = slot
        self._generation = generation

    def _check(self) -> None:
        if self._slot.generation != self._generation:
            raise RuntimeError(
                "this staged copy's host buffer was reused by a later copy; call "
                "result() before starting more copies than there are buffers, or "
                "keep result().copy()",
            )

    def ready(self) -> bool:
        """Whether the copy has finished (never waits); raises if the buffer was reused."""
        self._check()
        if self._slot.event is None:
            return True
        with self._slot.context():
            return bool(self._slot.event.done)

    def result(self) -> np.ndarray:
        """Wait for the copy and return the host array.

        Returns
        -------
        numpy.ndarray
            The staging buffer itself, valid until it is reused; keep
            ``result().copy()`` to hold it longer.

        Raises
        ------
        RuntimeError
            If the buffer was already reused by a later copy.
        """
        self._check()
        if self._slot.event is not None:
            with self._slot.context():
                self._slot.event.synchronize()
        return self._slot.host


class HostStaging:
    """Page-locked host buffers that device arrays are copied to in the background.

    :meth:`copy` snapshots the array on the device, so later steps may
    overwrite it, then copies the snapshot to the next of `buffers` pinned host
    buffers on its own stream. It waits only if that buffer's previous copy is
    still running, so the program runs at most `buffers` copies ahead. Host
    arrays are copied at once into the same buffers. Device copies are counted
    by :func:`~cunumpy.profiling.count_transfers`.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the arrays to copy.
    dtype : dtype-like
        Their dtype (not object).
    buffers : int, optional
        Number of buffers (and device snapshots) used in turn: how many
        copies may be in flight. The default 2 lets one copy run while the
        previous result is written out.

    Raises
    ------
    ValueError
        If `buffers` < 1 or `shape` is negative.
    TypeError
        For an object dtype.

    See Also
    --------
    cunumpy.to_host_async : One-off copy of a scalar or small array.

    Notes
    -----
    GPU storage is bound to the first source device; make that device current
    when submitting copies. The first GPU copy of an instance used on the host
    allocates new pinned buffers; earlier host results keep theirs.

    Examples
    --------
    >>> staging = xp.memory.HostStaging((2, 3), np.float64)
    >>> copy = staging.copy(np.ones((2, 3)))  # returns at once on the GPU
    >>> copy.ready()
    True
    >>> copy.result()
    array([[1., 1., 1.],
           [1., 1., 1.]])
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        dtype: Any,
        buffers: int = 2,
    ) -> None:
        buffers = operator.index(buffers)
        if buffers < 1:
            raise ValueError(f"buffers must be at least 1, got {buffers}")
        self._shape = (
            (operator.index(shape),)
            if isinstance(shape, (int, np.integer))
            else tuple(operator.index(n) for n in shape)
        )
        if any(n < 0 for n in self.shape):
            raise ValueError("staging shape must be non-negative")
        self._dtype = np.dtype(dtype)
        if self.dtype.hasobject:
            raise TypeError("HostStaging does not support object dtype")
        self._slots: list[_Slot] = []
        self._n_buffers = buffers
        self._next = 0
        self._stream: Any = None
        self._device_id: int | None = None

    def __repr__(self) -> str:
        return (
            f"HostStaging(shape={self.shape}, dtype={self.dtype}, "
            f"buffers={self._n_buffers})"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The fixed shape of each snapshot."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The fixed dtype of each snapshot."""
        return self._dtype

    @property
    def buffers(self) -> int:
        """Number of buffers used in turn."""
        return self._n_buffers

    def _slot(self, device: bool) -> _Slot:
        if not self._slots or (device and self._slots[0].device_id is None):
            allocate = _empty_pinned if device else np.empty
            # On the first GPU copy, replace ordinary host storage with pinned
            # slots. Existing CPU handles keep their own completed snapshots.
            self._slots = [
                _Slot(allocate(self.shape, self.dtype), self._device_id)
                for _ in range(self._n_buffers)
            ]
            self._next = 0
        slot = self._slots[self._next]
        self._next = (self._next + 1) % self._n_buffers
        return slot

    def copy(self, array: Any, *, stream: Any = None, event: Any = None) -> StagedCopy:
        """Start copying `array` to the host and return at once.

        Parameters
        ----------
        array : array
            A NumPy or CuPy array of the staging shape and dtype. Later writes
            on the snapshot's stream are ordered after the snapshot; another
            stream must wait for it before overwriting the array.
        stream : cupy.cuda.Stream, optional
            Producer stream to snapshot on; the current stream by default.
        event : cupy.cuda.Event, optional
            Producer event the current stream waits for before the snapshot.
            Pass at most one of `stream` and `event`; host copies ignore both.

        Returns
        -------
        StagedCopy
            ``ready()`` tells whether the copy has finished, ``result()``
            waits for it and returns the host array.

        Raises
        ------
        ValueError
            For another shape or dtype, both `stream` and `event`, or a
            device other than the current one or the one the staging is
            bound to.
        """
        if stream is not None and event is not None:
            raise ValueError("pass only one of stream and event")
        if tuple(array.shape) != self.shape or np.dtype(array.dtype) != self.dtype:
            raise ValueError(
                f"HostStaging for {self.shape} {self.dtype} got an array of shape "
                f"{tuple(array.shape)} and dtype {array.dtype}",
            )
        device = _is_device_array(array)
        if device:
            from cunumpy._streams import HostEvent, HostStream

            if isinstance(event, HostEvent) or isinstance(stream, HostStream):
                raise TypeError("device copies require a CUDA producer stream or event")
            cp = _cupy()
            device_id = array.device.id
            if cp.cuda.runtime.getDevice() != device_id:
                raise ValueError("make the source array's CUDA device current")
            if self._device_id is not None and self._device_id != device_id:
                raise ValueError("HostStaging is bound to another CUDA device")
            if stream is not None and getattr(stream, "device_id", device_id) not in (
                device_id,
                -1,
            ):
                raise ValueError(
                    "HostStaging producer stream belongs to another device"
                )
            self._device_id = device_id
        slot = self._slot(device)
        if slot.event is not None:
            with slot.context():
                slot.event.synchronize()  # the buffer's previous copy is done
        slot.generation += 1
        if not device:
            np.copyto(slot.host, np.asarray(array))
            slot.event = None
            return StagedCopy(slot, slot.generation)

        if self._stream is None:
            self._stream = cp.cuda.Stream(non_blocking=True)
        # Snapshot on the producer stream, or wait for a supplied event on the
        # current stream. Later writes must be ordered after this snapshot.
        producer = cp.cuda.get_current_stream() if stream is None else stream
        with producer:
            if event is not None:
                producer.wait_event(event)
            if slot.device is None:
                slot.device = cp.empty(self.shape, dtype=self.dtype)
            slot.device[...] = array
            snapshot_done = producer.record()
        # copy the snapshot to the pinned buffer on the staging stream
        self._stream.wait_event(snapshot_done)
        try:
            try:
                # blocking=False (CuPy >= 13): return once the copy is enqueued
                slot.device.get(stream=self._stream, out=slot.host, blocking=False)
            except TypeError:  # older CuPy
                slot.device.get(stream=self._stream, out=slot.host)
            if _COUNTERS:
                _record(
                    "to_host",
                    f"HostStaging.copy({self.shape} {self.dtype})",
                    nbytes=_nbytes(array),
                    blocking=False,
                )
        finally:
            # Retain completion even if a copy failed after enqueuing work.
            slot.event = self._stream.record()
        return StagedCopy(slot, slot.generation)

    def synchronize(self) -> None:
        """Wait for all copies in flight."""
        for slot in self._slots:
            if slot.event is not None:
                with slot.context():
                    slot.event.synchronize()


class HostCopy:
    """A device-to-host copy started by :func:`~cunumpy.to_host_async`.

    Not created directly.

    Parameters
    ----------
    host : numpy.ndarray
        The host buffer the copy writes into.
    event : cupy.cuda.Event, optional
        Recorded after the copy; None for a finished copy.
    source : array, optional
        The device array, kept alive until the copy is done.
    """

    def __init__(self, host: np.ndarray, event: Any = None, source: Any = None) -> None:
        self._host = host
        self._event = event
        self._source = source  # the device array stays alive until the copy is done

    def ready(self) -> bool:
        """Whether the copy has finished (never waits)."""
        return self._event is None or bool(self._event.done)

    def result(self) -> Any:
        """Return the value on the host, waiting only if the copy has not finished.

        A wait is counted as a ``sync`` by
        :func:`~cunumpy.profiling.count_transfers`.

        Returns
        -------
        numpy scalar or numpy.ndarray
            A scalar for a 0-d array, else an array (page-locked on the GPU).
        """
        if self._event is not None:
            if not self._event.done:
                if _COUNTERS:
                    _record_sync("to_host_async(...).result()")
                self._event.synchronize()
            self._event = self._source = None
        return self._host[()] if self._host.ndim == 0 else self._host


# one copy stream per device
_COPY_STREAMS: dict[int, Any] = {}


def to_host_async(array: Any) -> HostCopy:
    """Start copying a device scalar or small array to the host without waiting.

    The copy runs on a separate stream into page-locked memory, after the work
    queued so far on the current stream, so the host can keep queueing kernels
    (e.g. a convergence test read one iteration late). The value is that of
    the array when the queued work is done, even if later kernels overwrite
    it. A host array is copied at once. Counted as a ``to_host`` transfer with
    ``blocking=False`` by :func:`~cunumpy.profiling.count_transfers`.

    Parameters
    ----------
    array : array
        A NumPy or CuPy array or scalar.

    Returns
    -------
    HostCopy
        ``ready()`` tells whether the copy has finished (never waits),
        ``result()`` returns the value.

    See Also
    --------
    cunumpy.memory.HostStaging : For large arrays copied repeatedly.

    Examples
    --------
    >>> pending = xp.to_host_async(np.float64(3.5))  # e.g. a residual norm
    >>> pending.ready()
    True
    >>> pending.result()
    np.float64(3.5)
    """
    if not _is_device_array(array):
        return HostCopy(np.array(array))
    from cunumpy import _fake_cupy

    nbytes = _nbytes(array)
    if _fake_cupy.is_active():
        host = array.get()
        event = None
        source = None
    else:
        cp = _cupy()
        with cp.cuda.Device(array.device.id):
            source = cp.ascontiguousarray(array)  # on the current stream
            host = _empty_pinned(source.shape, source.dtype)
            queued = cp.cuda.get_current_stream().record()
            stream = _COPY_STREAMS.get(array.device.id)
            if stream is None:
                stream = cp.cuda.Stream(non_blocking=True)
                _COPY_STREAMS[array.device.id] = stream
            stream.wait_event(queued)
            try:
                # blocking=False (CuPy >= 13): return once the copy is enqueued
                source.get(stream=stream, out=host, blocking=False)
            except TypeError:  # older CuPy
                source.get(stream=stream, out=host)
            event = stream.record()
    if _COUNTERS:
        _record(
            "to_host",
            f"to_host_async({_describe(array)})",
            nbytes=nbytes,
            blocking=False,
        )
    return HostCopy(host, event, source)
