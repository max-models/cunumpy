"""Host buffers owned by other libraries, mirrored on the device.

Accumulation kernels may have to write into a buffer that another library owns
and keeps using on the host (e.g. a stencil vector's ``_data`` exchanged over
MPI). :class:`~cunumpy.memory.DeviceMirror` makes the one transfer per
accumulation explicit and keeps the host array's identity. See
:doc:`/guides/data-movement`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _describe, _nbytes, _record
from cunumpy.xp import _cupy_backend, to_cupy

__all__ = ["DeviceMirror"]


class DeviceMirror:
    """A host NumPy array paired with a lazily created device copy.

    On the CuPy backend, :attr:`device` is a CuPy array of the same shape and
    dtype, allocated as a copy of the host on first use. On the NumPy backend
    it is the host array itself, so kernels write straight into it. Transfers
    are explicit and in place (:meth:`to_device`, :meth:`to_host`, no-ops on
    NumPy) and counted by :func:`~cunumpy.profiling.count_transfers`. Make the
    mirror's CUDA device current before using it.

    Parameters
    ----------
    host : numpy.ndarray
        The host buffer, owned by the caller or another library.

    Raises
    ------
    TypeError
        If `host` is not a NumPy array.

    Examples
    --------
    >>> data = np.ones((4, 3))  # owned by another library
    >>> mirror = xp.memory.DeviceMirror(data)
    >>> accumulate(mirror.zero().device, n_threads=n)  # doctest: +SKIP
    >>> mirror.to_host().host is data  # same object, new values
    True
    """

    def __init__(self, host: np.ndarray) -> None:
        self._host = self._check_host(host)
        self._shape = host.shape
        self._dtype = host.dtype
        self._device: Any = None

    @staticmethod
    def _check_host(host: Any) -> np.ndarray:
        if not isinstance(host, np.ndarray):
            raise TypeError(
                "DeviceMirror needs a NumPy array as host buffer, got "
                f"{type(host).__name__}",
            )
        return host

    def __repr__(self) -> str:
        where = "host" if self._device is None else "device"
        return (
            f"DeviceMirror(shape={self._shape}, dtype={self._dtype}, buffer={where!r})"
        )

    @property
    def host(self) -> np.ndarray:
        """The host array, the same object that was passed in."""
        return self._host

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the mirrored array."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Dtype of the mirrored array."""
        return self._dtype

    @property
    def device(self) -> Any:
        """The array kernels write into: CuPy (copied from the host on first use), or the host on NumPy."""
        if not _cupy_backend():
            return self._host
        self._check_bound()
        self._check_device()
        if self._device is None:
            # The only host -> device allocation; it goes through to_cupy so
            # that it is visible wherever transfers are accounted for.
            self._device = to_cupy(self._host)
        return self._device

    def _check_device(self) -> None:
        if self._device is not None:
            import cupy as cp

            if cp.cuda.runtime.getDevice() != self._device.device.id:
                raise ValueError(
                    f"DeviceMirror is bound to CUDA device {self._device.device.id}; "
                    "make that device current before using the mirror",
                )

    def _check_bound(self) -> None:
        """Raise if the host array no longer has the shape/dtype it was bound with."""
        if self._host.shape != self._shape or self._host.dtype != self._dtype:
            raise ValueError(
                "the host array was reallocated: DeviceMirror was bound to "
                f"shape {self._shape} and dtype {self._dtype}, but the host now "
                f"has shape {self._host.shape} and dtype {self._host.dtype}; "
                "call rebind(host) with the new array",
            )

    def rebind(self, host: np.ndarray) -> DeviceMirror:
        """Bind to a new host array, e.g. after its owner reallocated it.

        The device array is kept if shape and dtype are unchanged (not
        refreshed: call :meth:`to_device`), otherwise it is allocated again on
        the next use.

        Parameters
        ----------
        host : numpy.ndarray
            The new host buffer.

        Returns
        -------
        DeviceMirror
            `self`, for chaining.
        """
        self._host = self._check_host(host)
        if host.shape != self._shape or host.dtype != self._dtype:
            self._device = None
        self._shape = host.shape
        self._dtype = host.dtype
        return self

    def to_device(self) -> DeviceMirror:
        """Copy the host array into the device array (no-op on the NumPy backend).

        Returns
        -------
        DeviceMirror
            `self`, for chaining.

        Raises
        ------
        ValueError
            If the host array was reallocated with another shape or dtype
            (call :meth:`rebind`).
        """
        self._check_bound()
        if not _cupy_backend():
            return self
        self._check_device()
        if self._device is None:
            _ = self.device  # allocates as a copy of the host, through to_cupy
        else:
            # host -> device copy into the existing device buffer
            self._device.set(self._host)
            if _COUNTERS:
                _record(
                    "to_device",
                    f"DeviceMirror.to_device({_describe(self._host)})",
                    nbytes=_nbytes(self._host),
                )
        return self

    def to_host(self, *, stream: Any = None, event: Any = None) -> DeviceMirror:
        """Copy the device array into the host array, in place, and wait for it.

        The host array keeps its identity, so the library that holds it sees
        the new values. No-op on the NumPy backend and before the device array
        exists.

        Parameters
        ----------
        stream : cupy.cuda.Stream, optional
            Producer stream to copy on; by default only the current stream's
            work is ordered before the copy.
        event : cupy.cuda.Event, optional
            Producer event the current stream waits for before the copy. Pass
            at most one of `stream` and `event`.

        Returns
        -------
        DeviceMirror
            `self`, for chaining.

        Raises
        ------
        ValueError
            If the host array was reallocated with another shape or dtype
            (call :meth:`rebind`), or for both `stream` and `event`.
        """
        if stream is not None and event is not None:
            raise ValueError("pass only one of stream and event")
        self._check_bound()
        if not _cupy_backend() or self._device is None:
            return self
        self._check_device()
        from cunumpy._streams import HostEvent, HostStream

        if isinstance(event, HostEvent) or isinstance(stream, HostStream):
            raise TypeError("device copies require a CUDA producer stream or event")
        import cupy as cp

        if stream is not None and getattr(
            stream, "device_id", self._device.device.id
        ) not in (
            self._device.device.id,
            -1,
        ):
            raise ValueError("DeviceMirror producer stream belongs to another device")
        if event is not None:
            cp.cuda.get_current_stream().wait_event(event)
        # device -> host copy into the existing host buffer
        if stream is None:
            self._device.get(out=self._host)
        else:
            self._device.get(out=self._host, stream=stream)
        if _COUNTERS:
            _record(
                "to_host",
                f"DeviceMirror.to_host({_describe(self._host)})",
                nbytes=_nbytes(self._host),
            )
        return self

    def zero(self) -> DeviceMirror:
        """Zero the array kernels write into (the device array, or the host on NumPy).

        Returns
        -------
        DeviceMirror
            `self`, for chaining.
        """
        self._check_bound()
        if not _cupy_backend():
            self._host.fill(0)
            return self
        self._check_device()
        if self._device is None:
            self._check_bound()
            import cupy as cp

            # allocated empty: no host -> device copy is needed to zero it
            self._device = cp.zeros(self._shape, dtype=self._dtype)
        else:
            self._device.fill(0)
        return self
