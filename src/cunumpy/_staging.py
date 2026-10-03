"""Copy device arrays to the host in the background, for output that should not stall the GPU.

Writing a field or the markers to HDF5 every few steps needs host arrays, and
``array.get()`` waits for the GPU and then for the copy, while no kernel runs.
:class:`HostStaging` overlaps the copy with the next time steps::

    staging = xp.memory.HostStaging(rho.shape, rho.dtype)        # once
    for step in range(n_steps):
        advance(...)
        if step % output_every == 0:
            pending.append((step, staging.copy(rho)))      # returns at once
        while pending and pending[0][1].ready():
            s, copy = pending.pop(0)
            h5file[f"rho/{s}"] = copy.result()             # a NumPy array

Each :meth:`HostStaging.copy` first snapshots the array on the device (on the
current stream, after the kernels that wrote it, so the next steps may
overwrite the array), then copies the snapshot to a page-locked host buffer on
its own stream. The buffers are reused in turn: with ``buffers=2``, a third
copy waits until the first one has finished, so the program never runs more
than ``buffers`` copies ahead of the output. :meth:`StagedCopy.result` waits for
its copy and returns the host buffer, valid until that buffer is reused
(``buffers`` copies later); a stale result raises instead of returning another
step's data. Copy it (``result().copy()``) to keep it longer.

Host arrays (and the NumPy backend) are copied at once, into the same buffers,
so the code is the same on both backends.
"""

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np

from ._transfers import _ACTIVE as _COUNTERS
from ._transfers import _record

__all__ = ["HostStaging", "StagedCopy"]

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array


def _cupy() -> Any:
    import cupy

    return cupy


def _empty_pinned(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    import cupyx

    return cupyx.empty_pinned(shape, dtype=dtype)


class _Slot:
    def __init__(self, host: np.ndarray) -> None:
        self.host = host
        self.device: Any = None  # the snapshot, allocated on the first device copy
        self.event: Any = None  # recorded after the copy to the host
        self.generation = 0


class StagedCopy:
    """A copy started by :meth:`HostStaging.copy`."""

    def __init__(self, slot: _Slot, generation: int) -> None:
        self._slot = slot
        self._generation = generation

    def _check(self) -> None:
        if self._slot.generation != self._generation:
            raise RuntimeError(
                "this staged copy's host buffer was reused by a later copy; call "
                "result() before starting more copies than there are buffers, or "
                "keep result().copy()"
            )

    def ready(self) -> bool:
        """Whether the copy has finished (never waits)."""
        self._check()
        return self._slot.event is None or bool(self._slot.event.done)

    def result(self) -> np.ndarray:
        """Wait for the copy and return the host array.

        The array is the staging buffer itself: valid until the buffer is reused.

        Raises
        ------
        RuntimeError
            If the buffer was already reused by a later copy.
        """
        self._check()
        if self._slot.event is not None:
            self._slot.event.synchronize()
        return self._slot.host


class HostStaging:
    """Page-locked host buffers that device arrays are copied to in the background.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the arrays to copy.
    dtype : dtype-like
        Their dtype.
    buffers : int
        Number of buffers (and device snapshots) used in turn: how many copies
        may be in flight at once. 2 (double buffering) lets one copy run while
        the previous result is written out.
    """

    def __init__(
        self, shape: tuple[int, ...] | int, dtype: Any, buffers: int = 2
    ) -> None:
        if buffers < 1:
            raise ValueError(f"buffers must be at least 1, got {buffers}")
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = np.dtype(dtype)
        self._slots: list[_Slot] = []
        self._n_buffers = buffers
        self._next = 0
        self._stream: Any = None

    def __repr__(self) -> str:
        return (
            f"HostStaging(shape={self.shape}, dtype={self.dtype}, "
            f"buffers={self._n_buffers})"
        )

    @property
    def buffers(self) -> int:
        """Number of buffers used in turn."""
        return self._n_buffers

    def _slot(self, device: bool) -> _Slot:
        if not self._slots:
            allocate = _empty_pinned if device else np.empty
            self._slots = [
                _Slot(allocate(self.shape, self.dtype)) for _ in range(self._n_buffers)
            ]
        slot = self._slots[self._next]
        self._next = (self._next + 1) % self._n_buffers
        return slot

    def copy(self, array: Any) -> StagedCopy:
        """Start copying `array` to the host and return at once.

        Parameters
        ----------
        array : cupy.ndarray | numpy.ndarray
            An array of the staging shape and dtype. It may be overwritten
            right after this call: a device array is snapshotted first.

        Returns
        -------
        StagedCopy
            ``ready()`` tells whether the copy has finished, ``result()``
            waits for it and returns the host array.
        """
        if tuple(array.shape) != self.shape or np.dtype(array.dtype) != self.dtype:
            raise ValueError(
                f"HostStaging for {self.shape} {self.dtype} got an array of shape "
                f"{tuple(array.shape)} and dtype {array.dtype}"
            )
        device = _is_device_array(array)
        slot = self._slot(device)
        if slot.event is not None:
            slot.event.synchronize()  # the buffer's previous copy is done
        slot.generation += 1
        if not device:
            np.copyto(slot.host, np.asarray(array))
            slot.event = None
            return StagedCopy(slot, slot.generation)

        cp = _cupy()
        if self._stream is None:
            self._stream = cp.cuda.Stream(non_blocking=True)
        if slot.device is None:
            slot.device = cp.empty(self.shape, dtype=self.dtype)
        if _COUNTERS:
            _record("to_host", f"HostStaging.copy({self.shape} {self.dtype})")
        # snapshot on the producer's stream, after the kernels that wrote `array`
        slot.device[...] = array
        snapshot_done = cp.cuda.get_current_stream().record()
        # copy the snapshot to the pinned buffer on the staging stream
        self._stream.wait_event(snapshot_done)
        try:
            # blocking=False (CuPy >= 13): return once the copy is enqueued
            slot.device.get(stream=self._stream, out=slot.host, blocking=False)
        except (
            TypeError
        ):  # older CuPy: a copy on a stream into pinned memory is asynchronous
            slot.device.get(stream=self._stream, out=slot.host)
        slot.event = self._stream.record()
        return StagedCopy(slot, slot.generation)

    def synchronize(self) -> None:
        """Wait for all copies in flight."""
        for slot in self._slots:
            if slot.event is not None:
                slot.event.synchronize()
