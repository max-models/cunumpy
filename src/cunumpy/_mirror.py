"""Host buffers owned by other libraries, mirrored on the device.

Accumulation kernels write into buffers that another library owns and keeps
using on the host (e.g. a stencil vector's ``_data`` that is then exchanged
over MPI). On the GPU the kernel writes into a device array instead, and one
transfer per accumulation is unavoidable; :class:`DeviceMirror` makes that
transfer explicit and keeps the host array's identity, so the owning library
still sees its own buffer.

Every host/device copy of a mirror goes through ``xp.to_cupy`` or the
mirror's own ``to_device()``/``to_host()`` calls, so that transfers are easy
to find (and to account for) in application code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy.xp import _cupy_backend, to_cupy

__all__ = ["DeviceMirror"]


class DeviceMirror:
    """A host NumPy array paired with a lazily created device copy.

    On the CuPy backend, `device` is a CuPy array of the same shape and dtype
    as the host array, allocated (and filled from the host) on first use.
    On the NumPy backend, `device` is the host array itself, so a kernel that
    is given ``mirror.device`` writes straight into the host buffer, without
    any copy.

    Transfers are explicit: `to_device()` copies host to device and
    `to_host()` copies device to host, in place, so the host array keeps its
    identity. Both are no-ops on the NumPy backend. `zero()` clears the
    buffer the kernel writes into.

    Parameters
    ----------
    host : numpy.ndarray
        The host buffer, owned by the caller (or by another library).

    Raises
    ------
    TypeError
        If `host` is not a NumPy array.

    Examples
    --------
    >>> data = np.zeros((4, 3))  # owned by another library
    >>> mirror = DeviceMirror(data)
    >>> mirror.zero()  # doctest: +SKIP
    >>> accumulate(mirror.device, n_threads=n)  # doctest: +SKIP
    >>> mirror.to_host()  # doctest: +SKIP
    >>> mirror.host is data
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
        """The array kernels write into: a CuPy array, or the host on NumPy.

        On the CuPy backend the device array is allocated on first access, as
        a copy of the host array. On the NumPy backend this is the host array
        itself.
        """
        if not _cupy_backend():
            return self._host
        if self._device is None:
            self._check_bound()
            # The only host -> device allocation; it goes through to_cupy so
            # that it is visible wherever transfers are accounted for.
            self._device = to_cupy(self._host)
        return self._device

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

        The device buffer is kept if the shape and dtype are unchanged (it
        is not refreshed: call `to_device()` for that), otherwise it is
        dropped and allocated again on the next use.

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
        """Copy the host array to the device (no-op on the NumPy backend).

        Raises
        ------
        ValueError
            If the host array was reallocated with another shape or dtype.
        """
        self._check_bound()
        if not _cupy_backend():
            return self
        if self._device is None:
            _ = self.device  # allocates as a copy of the host, through to_cupy
        else:
            # host -> device copy into the existing device buffer
            self._device.set(self._host)
        return self

    def to_host(self) -> DeviceMirror:
        """Copy the device array into the host array, in place.

        The host array keeps its identity, so a library that holds the buffer
        sees the new values. No-op on the NumPy backend, and if no device
        array has been created yet.

        Raises
        ------
        ValueError
            If the host array was reallocated with another shape or dtype.
        """
        self._check_bound()
        if not _cupy_backend() or self._device is None:
            return self
        # device -> host copy into the existing host buffer
        self._device.get(out=self._host)
        return self

    def zero(self) -> DeviceMirror:
        """Zero the array kernels write into (the device array, or the host on NumPy)."""
        if not _cupy_backend():
            self._host.fill(0)
            return self
        if self._device is None:
            self._check_bound()
            import cupy as cp

            # allocated empty: no host -> device copy is needed to zero it
            self._device = cp.zeros(self._shape, dtype=self._dtype)
        else:
            self._device.fill(0)
        return self
