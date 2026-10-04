"""Reusable CUDA streams/events with synchronous CPU equivalents."""

from __future__ import annotations

from typing import Any

from cunumpy.xp import get_backend


class HostEvent:
    """Already-completed CPU event, matching CuPy's completion interface."""

    @property
    def done(self) -> bool:
        return True

    def record(self, stream: Any = None) -> None:
        if stream is not None and not isinstance(stream, HostStream):
            raise TypeError("a host event requires a host stream")

    def synchronize(self) -> None:
        pass


class HostStream:
    """Synchronous CPU stream; reusable and nestable as a context manager."""

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    @property
    def done(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def record(self, event: Any = None) -> HostEvent:
        event = HostEvent() if event is None else event
        event.record(self)
        return event

    def wait_event(self, event: Any) -> None:
        if not isinstance(event, HostEvent):
            raise TypeError("a host stream requires a host event")


def create_stream(*, non_blocking: bool = True) -> Any:
    """Allocate a reusable CuPy stream, or a synchronous :class:`HostStream`.

    CUDA streams belong to the device current at construction. Keep the same
    device current while selecting the stream and submitting work to it.
    """
    if get_backend() == "numpy":
        return HostStream()
    import cupy as cp

    return cp.cuda.Stream(non_blocking=non_blocking)


def create_event(*, timing: bool = False) -> Any:
    """Allocate a reusable completion event; CPU events are always complete.

    CUDA timing is disabled by default. Pass `timing=True` for events used
    with CuPy's elapsed-time functions; host events do not measure time.
    """
    if get_backend() == "numpy":
        return HostEvent()
    import cupy as cp

    return cp.cuda.Event(disable_timing=not timing)


def record_event(event: Any = None, *, stream: Any = None) -> Any:
    """Record completion on `stream` (current stream by default).

    Re-recording an event replaces its completion point: enqueue all consumers
    of a recording before reusing the event for a subsequent producer.
    """
    if event is None:
        event = HostEvent() if isinstance(stream, HostStream) else create_event()
    event.record(stream)
    return event


def wait_event(event: Any, *, stream: Any = None) -> None:
    """Order future work on `stream` after `event`, without blocking the CPU."""
    if stream is None:
        if isinstance(event, HostEvent):
            return
        import cupy as cp

        stream = cp.cuda.get_current_stream()
    stream.wait_event(event)
