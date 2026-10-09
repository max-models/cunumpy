"""Reusable CUDA streams and events, with synchronous CPU equivalents."""

from __future__ import annotations

from typing import Any

from cunumpy.xp import get_backend


class HostEvent:
    """Already-completed CPU event with the completion interface of ``cupy.cuda.Event``.

    Returned by :func:`create_event` on the NumPy backend.

    Examples
    --------
    >>> event = xp.cuda.create_event()  # a HostEvent on NumPy
    >>> event.done
    True
    """

    @property
    def done(self) -> bool:
        """Whether the event has completed; always True."""
        return True

    def record(self, stream: Any = None) -> None:
        """Record the event on a host stream; does nothing.

        Parameters
        ----------
        stream : HostStream, optional
            The stream to record on.

        Raises
        ------
        TypeError
            If `stream` is not a :class:`HostStream`.
        """
        if stream is not None and not isinstance(stream, HostStream):
            raise TypeError("a host event requires a host stream")

    def synchronize(self) -> None:
        """Wait for the event; returns at once."""


class HostStream:
    """Synchronous CPU stream with the interface of ``cupy.cuda.Stream``.

    Returned by :func:`create_stream` on the NumPy backend; reusable and nestable
    as a context manager.

    Examples
    --------
    >>> s = xp.cuda.create_stream()  # a HostStream on NumPy
    >>> with s:
    ...     y = xp.ones(3)
    >>> s.record().done
    True
    """

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    @property
    def done(self) -> bool:
        """Whether all work on the stream has completed; always True."""
        return True

    def synchronize(self) -> None:
        """Wait for the work on the stream; returns at once."""

    def record(self, event: Any = None) -> HostEvent:
        """Record an event on this stream.

        Parameters
        ----------
        event : HostEvent, optional
            The event to record; a new one by default.

        Returns
        -------
        HostEvent
            The recorded event.
        """
        event = HostEvent() if event is None else event
        event.record(self)
        return event

    def wait_event(self, event: Any) -> None:
        """Order later work after `event`; does nothing.

        Parameters
        ----------
        event : HostEvent
            The event to wait for.

        Raises
        ------
        TypeError
            If `event` is not a :class:`HostEvent`.
        """
        if not isinstance(event, HostEvent):
            raise TypeError("a host stream requires a host event")


def create_stream(*, non_blocking: bool = True) -> Any:
    """Allocate a reusable CUDA stream, or a :class:`HostStream` on NumPy.

    A CUDA stream belongs to the device current at construction; keep that device
    current while selecting the stream and submitting work to it.

    Parameters
    ----------
    non_blocking : bool, optional
        Create a stream that does not synchronize with the default stream; True
        by default.

    Returns
    -------
    cupy.cuda.Stream or HostStream
        The stream, usable with :func:`stream`.

    Examples
    --------
    >>> s = xp.cuda.create_stream()
    >>> with xp.cuda.stream(s):
    ...     y = xp.ones(3) * 2
    >>> s.synchronize()
    """
    if get_backend() == "numpy":
        return HostStream()
    import cupy as cp

    return cp.cuda.Stream(non_blocking=non_blocking)


def create_event(*, timing: bool = False) -> Any:
    """Allocate a reusable completion event, or a :class:`HostEvent` on NumPy.

    Host events are always complete and do not measure time.

    Parameters
    ----------
    timing : bool, optional
        Enable CUDA timing, for CuPy's elapsed-time functions; False by default.

    Returns
    -------
    cupy.cuda.Event or HostEvent
        The event.

    Examples
    --------
    >>> xp.cuda.create_event().done
    True
    """
    if get_backend() == "numpy":
        return HostEvent()
    import cupy as cp

    return cp.cuda.Event(disable_timing=not timing)


def record_event(event: Any = None, *, stream: Any = None) -> Any:
    """Record a completion event on `stream`.

    Re-recording an event replaces its completion point: enqueue all waits on
    one recording before reusing the event. See :doc:`/guides/execution-helpers`.

    Parameters
    ----------
    event : event, optional
        The event to record; a new one from :func:`create_event` by default.
    stream : stream, optional
        The producer stream; the current stream by default.

    Returns
    -------
    event
        The recorded event.

    See Also
    --------
    wait_event : Make another stream wait for the event.

    Examples
    --------
    >>> producer, consumer = xp.cuda.create_stream(), xp.cuda.create_stream()
    >>> event = xp.cuda.record_event(stream=producer)
    >>> xp.cuda.wait_event(event, stream=consumer)
    """
    if event is None:
        event = HostEvent() if isinstance(stream, HostStream) else create_event()
    event.record(stream)
    return event


def wait_event(event: Any, *, stream: Any = None) -> None:
    """Order future work on `stream` after `event`, without blocking the CPU.

    Parameters
    ----------
    event : event
        An event recorded with :func:`record_event`.
    stream : stream, optional
        The consumer stream; the current stream by default.

    See Also
    --------
    record_event : Record the event to wait for.

    Examples
    --------
    >>> event = xp.cuda.record_event(stream=xp.cuda.create_stream())
    >>> xp.cuda.wait_event(event)
    """
    if stream is None:
        if isinstance(event, HostEvent):
            return
        import cupy as cp

        stream = cp.cuda.get_current_stream()
    stream.wait_event(event)
