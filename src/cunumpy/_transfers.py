"""Counting host/device transfers made through cunumpy.

Moving data between the host and the device is the classic performance bug of
a GPU port: a single ``to_numpy()`` inside a time loop makes every step wait for
the device and copy an array. :func:`count_transfers` records every transfer
that goes through cunumpy while the block runs, together with the call site
that caused it, so a test can verify that a time step does not transfer at
all::

    with xp.profiling.count_transfers() as counter:
        propagator(dt)
    assert counter.total == 0, counter.report()

or, to reject host/device copies and host execution on the GPU backend::

    with xp.profiling.assert_no_transfers():
        propagator(dt)

Counted are

* ``to_host``: :func:`~cunumpy.to_numpy` (and :func:`~cunumpy.to_cunumpy`)
  called with a device array;
* ``to_device``: :func:`~cunumpy.to_cupy` (and :func:`~cunumpy.to_cunumpy`)
  called with anything that is not already a device array;
* ``kernel_conversion``: a :class:`~cunumpy.kernels.PyccelKernel` call that copied
  device arrays to the host (and back), one event per call;
* ``fallback``: a :class:`~cunumpy.kernels.Kernel` without CUDA kernel calling its host
  kernel on the CuPy backend (``missing_cuda="fallback"``), one event per call;
* ``device_copy``: device-only dtype/layout conversions in CuNumpy helpers.

Mirror refreshes, argument conversions, staging, serial MPI and kernel output
copy-back are also counted. Each physical host/device copy is recorded with its
payload size; conversion/fallback markers have no byte count. ``total`` counts
all observations, including markers. ``assert_no_transfers`` allows device-only
copies and rejects host/device copies and host fallback/conversion markers.

Limitations
-----------
Only transfers made *through cunumpy* are seen. Raw ``cupy.ndarray.get()``,
``cupy.asarray(numpy_array)``, ``numpy.asarray(cupy_array)``, ``float(device_array)``,
forwarded backend operations such as ``xp.asarray`` and implicit conversions
inside other libraries are not counted; use ``nsys``
(or CuPy's own profiling hooks) to find those.

Like the backend selection, the set of active counters is process-wide state
and not thread-safe: counters started in one thread see the transfers of every
thread.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TransferCounter",
    "TransferEvent",
    "assert_no_transfers",
    "count_transfers",
]

#: Event kinds, in the order they are reported.
KINDS = ("to_host", "to_device", "kernel_conversion", "fallback", "device_copy")

# The currently active counters, innermost last. Instrumented code checks
# ``if _ACTIVE:`` before doing any work, so the overhead of an inactive counter
# is a single truth test. The list object is shared with the instrumented
# modules and mutated in place; never rebind it.
_ACTIVE: list[TransferCounter] = []

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class TransferEvent:
    """One recorded transfer.

    Attributes
    ----------
    kind : str
        One of ``"to_host"``, ``"to_device"``, ``"kernel_conversion"`` or
        ``"fallback"``, or ``"device_copy"``.
    description : str
        What was transferred, e.g. ``"to_numpy(shape=(1000,), dtype=float64)"``
        or ``"PyccelKernel 'push': 3 array(s) copied to the host"``.
    where : str
        The call site outside cunumpy, as ``"file:line"``.
    nbytes : int | None
        Payload bytes of a physical copy. None for markers or unknown sizes.
    """

    kind: str
    description: str
    where: str
    nbytes: int | None = None

    def __str__(self) -> str:
        return f"{self.where}: {self.kind}: {self.description}"


class TransferCounter:
    """Transfers recorded while a :func:`count_transfers` block runs.

    Attributes
    ----------
    events : list[TransferEvent]
        Every recorded transfer, in order.
    """

    def __init__(self) -> None:
        self.events: list[TransferEvent] = []

    def __repr__(self) -> str:
        counts = ", ".join(f"{kind}={self.count(kind)}" for kind in KINDS)
        return f"TransferCounter({counts})"

    def _add(self, event: TransferEvent) -> None:
        self.events.append(event)

    def count(self, kind: str) -> int:
        """Number of events of `kind`."""
        return sum(1 for event in self.events if event.kind == kind)

    @property
    def to_host(self) -> int:
        """Number of device-to-host copies (`to_numpy`, `to_cunumpy`)."""
        return self.count("to_host")

    @property
    def to_device(self) -> int:
        """Number of host-to-device copies (`to_cupy`, `to_cunumpy`)."""
        return self.count("to_device")

    @property
    def kernel_conversions(self) -> int:
        """Number of `PyccelKernel` calls that copied device arrays to the host."""
        return self.count("kernel_conversion")

    @property
    def kernel_conversion_calls(self) -> list[TransferEvent]:
        """The `PyccelKernel` conversion events: kernel name, arrays, call site."""
        return [event for event in self.events if event.kind == "kernel_conversion"]

    @property
    def fallbacks(self) -> int:
        """Number of `Kernel` calls that fell back to the host kernel on CuPy."""
        return self.count("fallback")

    @property
    def total(self) -> int:
        """Number of observations, including conversion/fallback markers."""
        return len(self.events)

    def bytes(self, kind: str) -> int:
        """Known bytes copied for `kind`; markers and unknown sizes add zero."""
        return sum(e.nbytes or 0 for e in self.events if e.kind == kind)

    @property
    def bytes_to_host(self) -> int:
        """Known payload bytes copied from device to host."""
        return self.bytes("to_host")

    @property
    def bytes_to_device(self) -> int:
        """Known payload bytes copied from host to device."""
        return self.bytes("to_device")

    @property
    def device_copies(self) -> int:
        """Device-only conversions recorded by CuNumpy helpers."""
        return self.count("device_copy")

    def report(self) -> str:
        """A readable summary: events grouped by kind and call site, with counts."""
        counts = ", ".join(f"{self.count(kind)} {kind}" for kind in KINDS)
        lines = [f"{self.total} transfer(s) through cunumpy ({counts})"]
        for kind in KINDS:
            events = [event for event in self.events if event.kind == kind]
            if not events:
                continue
            lines.append(f"  {kind} ({len(events)}):")
            grouped: dict[tuple[str, str], int] = {}
            for event in events:
                key = (event.where, event.description)
                grouped[key] = grouped.get(key, 0) + 1
            for (where, description), n in grouped.items():
                times = f" (x{n})" if n > 1 else ""
                lines.append(f"    {where}: {description}{times}")
        return "\n".join(lines)


def _caller() -> str:
    """The innermost ``file:line`` on the stack that is outside cunumpy."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not os.path.abspath(filename).startswith(_PACKAGE_DIR):
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "<unknown>"


def _record(kind: str, description: str, *, nbytes: int | None = None) -> None:
    """Record a transfer in every active counter (call only ``if _ACTIVE:``)."""
    event = TransferEvent(kind, description, _caller(), nbytes)
    for counter in _ACTIVE:
        counter._add(event)


def _describe(array: Any) -> str:
    """``shape=..., dtype=...`` of an array, or the type name otherwise."""
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return type(array).__name__
    return f"shape={tuple(shape)}, dtype={dtype}"


def _nbytes(array: Any) -> int | None:
    """Array payload size, without copying data or reading a device scalar."""
    shape, dtype = getattr(array, "shape", None), getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return None
    return math.prod(shape) * dtype.itemsize


def _device_pointer(array: Any) -> int | None:
    pointer = getattr(getattr(array, "data", None), "ptr", None)
    if pointer is not None:
        return pointer
    interface = getattr(array, "__cuda_array_interface__", {})
    data = interface.get("data")
    return None if data is None else data[0]


def _is_device_copy(source: Any, result: Any) -> bool:
    """Distinguish a newly allocated conversion from a view of the same storage."""
    if result is source:
        return False
    pointer = _device_pointer(source)
    return pointer is None or pointer != _device_pointer(result)


@contextmanager
def count_transfers() -> Generator[TransferCounter, None, None]:
    """Count the host/device transfers made through cunumpy in the block.

    Yields a :class:`TransferCounter` that records every ``to_numpy``,
    ``to_cupy`` and ``to_cunumpy`` call that actually copies, every
    :class:`~cunumpy.kernels.PyccelKernel` call that converts device arrays and every
    :class:`~cunumpy.kernels.Kernel` fallback to the host kernel, with the call site of
    each. Nothing is counted for calls that do not copy, e.g. ``to_numpy`` of a
    NumPy array.

    Blocks can be nested; each active counter sees the transfers made inside
    it. Transfers that bypass cunumpy (raw ``cupy.ndarray.get()``,
    ``cupy.asarray(numpy_array)``, conversions inside other libraries) are not
    seen; see the module documentation.

    Examples
    --------
    >>> with xp.profiling.count_transfers() as counter:
    ...     propagator(dt)
    >>> assert counter.total == 0, counter.report()
    """
    counter = TransferCounter()
    _ACTIVE.append(counter)
    try:
        yield counter
    finally:
        _ACTIVE.remove(counter)


@contextmanager
def assert_no_transfers() -> Generator[TransferCounter, None, None]:
    """Raise ``AssertionError`` if the block makes a transfer through cunumpy.

    A :func:`count_transfers` block that, on exit, raises with the counter's
    :meth:`~TransferCounter.report` if a host/device copy or host conversion/
    fallback was counted. Device-only conversions are allowed. Only checked if
    the block exits normally; an exception raised inside propagates as it is.

    Examples
    --------
    >>> with xp.profiling.assert_no_transfers():
    ...     propagator(dt)
    """
    with count_transfers() as counter:
        yield counter
    if any(event.kind != "device_copy" for event in counter.events):
        raise AssertionError(
            "host/device transfers inside a block that must not transfer:\n"
            + counter.report(),
        )
