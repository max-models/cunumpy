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
* ``device_copy``: device-only dtype/layout conversions in CuNumpy helpers;
* ``sync``: the host waited for the device: :func:`~cunumpy.synchronize`, the
  waits of the MPI helpers and of the CUDA debug mode, and, on the fake CuPy,
  a scalar read of a device array (``float(a)``, ``int(a)``, ``bool(a)``,
  ``a.item()``, ``a.tolist()``). Syncs are listed in :attr:`TransferCounter.syncs`
  and the report but not in :attr:`~TransferCounter.total`, and
  :func:`assert_no_transfers` accepts them unless called with ``syncs=True``.

Mirror refreshes, argument conversions, staging, serial MPI and kernel output
copy-back are also counted. Each physical host/device copy is recorded with its
payload size; conversion/fallback markers have no byte count. ``total`` counts
all observations, including markers. ``assert_no_transfers`` allows device-only
copies and rejects host/device copies and host fallback/conversion markers.

Limitations
-----------
Only transfers made *through cunumpy* are seen. Raw ``cupy.ndarray.get()``,
``cupy.asarray(numpy_array)``, ``numpy.asarray(cupy_array)``, ``float(device_array)``
(an implicit sync of the real CuPy, which cannot be observed from Python; the fake
CuPy reports it),
forwarded backend operations such as ``xp.asarray`` and implicit conversions
inside other libraries are not counted; use ``nsys``
(or CuPy's own profiling hooks) to find those.

Like the backend selection, the set of active counters is process-wide state
and not thread-safe: counters started in one thread see the transfers of every
thread.
"""

from __future__ import annotations

import functools
import math
import os
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TransferBudget",
    "TransferCounter",
    "TransferEvent",
    "assert_no_transfers",
    "count_transfers",
]

#: Event kinds, in the order they are reported.
KINDS = (
    "to_host",
    "to_device",
    "kernel_conversion",
    "fallback",
    "device_copy",
    "sync",
)

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
    blocking : bool
        Whether the host waited for the copy. False for the copies started by
        :func:`~cunumpy.to_host_async`.
    implicit : bool
        Whether the event was not asked for explicitly: a scalar read of a
        device array (``float(a)``) on the fake CuPy, recorded as a ``sync``.
    """

    kind: str
    description: str
    where: str
    nbytes: int | None = None
    blocking: bool = True
    implicit: bool = False

    @property
    def label(self) -> str:
        """The description, marked ``[async]`` or ``[implicit]`` where it applies."""
        marks = ("" if self.blocking else " [async]") + (
            " [implicit]" if self.implicit else ""
        )
        return self.description + marks

    def __str__(self) -> str:
        return f"{self.where}: {self.kind}: {self.label}"


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
    def syncs(self) -> int:
        """Number of times the host waited for the device (see the module documentation)."""
        return self.count("sync")

    @property
    def total(self) -> int:
        """Number of observations, including conversion/fallback markers (not syncs)."""
        return sum(1 for event in self.events if event.kind != "sync")

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
                key = (event.where, event.label)
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


def _record(
    kind: str,
    description: str,
    *,
    nbytes: int | None = None,
    blocking: bool = True,
    implicit: bool = False,
) -> None:
    """Record a transfer in every active counter (call only ``if _ACTIVE:``)."""
    event = TransferEvent(kind, description, _caller(), nbytes, blocking, implicit)
    for counter in _ACTIVE:
        counter._add(event)


def _record_sync(description: str, *, implicit: bool = False) -> None:
    """Record that the host waits for the device (call only ``if _ACTIVE:``)."""
    _record("sync", description, implicit=implicit)


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
def count_transfers(
    into: TransferCounter | None = None,
) -> Generator[TransferCounter, None, None]:
    """Count the host/device transfers made through cunumpy in the block.

    Yields a :class:`TransferCounter` that records every ``to_numpy``,
    ``to_cupy`` and ``to_cunumpy`` call that actually copies, every
    :class:`~cunumpy.kernels.PyccelKernel` call that converts device arrays and every
    :class:`~cunumpy.kernels.Kernel` fallback to the host kernel, with the call site of
    each. Nothing is counted for calls that do not copy, e.g. ``to_numpy`` of a
    NumPy array.

    Blocks can be nested; each active counter sees the transfers made inside
    it. With `into`, the events are added to that counter (e.g. to accumulate
    over several calls); a counter that is already active is not added again,
    so nested blocks with the same counter count each event once. Like any
    context manager made with :func:`contextlib.contextmanager`, it is also a
    decorator: ``@count_transfers(counter)``. Transfers that bypass cunumpy (raw ``cupy.ndarray.get()``,
    ``cupy.asarray(numpy_array)``, conversions inside other libraries) are not
    seen; see the module documentation.

    Examples
    --------
    >>> with xp.profiling.count_transfers() as counter:
    ...     propagator(dt)
    >>> assert counter.total == 0, counter.report()
    """
    counter = TransferCounter() if into is None else into
    if any(active is counter for active in _ACTIVE):
        yield counter
        return
    _ACTIVE.append(counter)
    try:
        yield counter
    finally:
        _ACTIVE.remove(counter)


@contextmanager
def assert_no_transfers(
    *, syncs: bool = False
) -> Generator[TransferCounter, None, None]:
    """Raise ``AssertionError`` if the block makes a transfer through cunumpy.

    Syncs (the host waiting for the device) are accepted unless `syncs` is
    True, e.g. ``assert_no_transfers(syncs=True)`` for a time step that must
    never stall on the device.

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
    ignored = ("device_copy",) if syncs else ("device_copy", "sync")
    if any(event.kind not in ignored for event in counter.events):
        raise AssertionError(
            "host/device transfers inside a block that must not transfer:\n"
            + counter.report(),
        )


class _PhaseRouter(TransferCounter):
    """The counter a :class:`TransferBudget` keeps active: it adds every event
    to the innermost phase of the budget only."""

    def __init__(self, budget: TransferBudget) -> None:
        super().__init__()
        self._budget = budget

    def _add(self, event: TransferEvent) -> None:
        self._budget[self._budget._stack[-1]]._add(event)


_LIMITS = ("max_nbytes", "max_count", "max_total_bytes", "blocking", "implicit")


@dataclass
class _Rule:
    allow: dict[str, dict[str, Any]]
    ignore: tuple[str, ...]
    calls: int | None


class TransferBudget:
    """Transfers counted per phase of a program, checked against rules per phase.

    A time loop typically has phases with different budgets: the time step must
    not copy arrays between host and device at all, the diagnostics may copy a
    few scalars to the host, the output each saved array once. A budget counts
    the transfers of each phase (with :func:`count_transfers`) and checks them::

        budget = TransferBudget(started=False)
        model.integrate = budget.count("integrate")(model.integrate)
        ...                       # setup: not counted
        budget.start()
        for step in range(n_steps):
            model.integrate(dt)
            with budget.phase("output"):
                save(model)
        budget.require("integrate", allow={"to_host": dict(max_nbytes=8)}, calls=n_steps)
        budget.require("output", allow={"to_host": dict(max_count=n, max_total_bytes=b)})
        budget.check()            # AssertionError with report() if a rule is broken

    Phases nest: an event is counted in the innermost phase only, and a phase
    entered again inside itself (recursion) counts each event and call once.

    Parameters
    ----------
    started : bool
        Whether to count from the start; otherwise phases run uncounted until
        :meth:`start`.

    Attributes
    ----------
    phases : dict[str, TransferCounter]
        The events of each phase, accumulated over its calls.
    calls : dict[str, int]
        How many times each phase was entered while counting.
    """

    def __init__(self, *, started: bool = True) -> None:
        self.phases: dict[str, TransferCounter] = {}
        self.calls: dict[str, int] = {}
        self.started = started
        self._rules: dict[str, _Rule] = {}
        self._stack: list[str] = []
        self._router = _PhaseRouter(self)

    def __getitem__(self, phase: str) -> TransferCounter:
        """The counter of `phase` (empty if it has not run)."""
        return self.phases.setdefault(phase, TransferCounter())

    def start(self) -> None:
        """Count the phases from now on."""
        self.started = True

    def stop(self) -> None:
        """Stop counting; phases run uncounted until :meth:`start`."""
        self.started = False

    @contextmanager
    def phase(self, name: str) -> Generator[TransferCounter, None, None]:
        """Count the transfers of the block in phase `name` (accumulated)."""
        counter = self[name]
        if not self.started:
            yield counter
            return
        if name not in self._stack:
            self.calls[name] = self.calls.get(name, 0) + 1
        self._stack.append(name)
        try:
            with count_transfers(into=self._router):
                yield counter
        finally:
            self._stack.pop()

    def count(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """A decorator: every call of the function is counted in phase `name`."""

        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(function)
            def counted(*args: Any, **kwargs: Any) -> Any:
                with self.phase(name):
                    return function(*args, **kwargs)

            return counted

        return decorate

    def require(
        self,
        phase: str,
        allow: Mapping[str, Mapping[str, Any] | None] | None = None,
        *,
        ignore: Sequence[str] = ("device_copy", "sync"),
        calls: int | None = None,
    ) -> None:
        """Set the rule of `phase`, checked by :meth:`check`.

        Parameters
        ----------
        phase : str
            The phase.
        allow : Mapping[str, Mapping | None] | None
            The event kinds the phase may have, each with its limits (None or
            ``{}``: any number). Every other kind is forbidden, except those in
            `ignore`. Limits: ``max_nbytes`` (of each event; events of unknown
            size break it), ``max_count`` and ``max_total_bytes`` (over the
            whole phase), ``blocking`` and ``implicit`` (the events must have
            this value of :attr:`TransferEvent.blocking` / ``implicit``), e.g.
            ``{"to_host": dict(max_nbytes=8, blocking=False)}``.
        ignore : Sequence[str]
            Kinds that are not checked unless they are in `allow`: by default
            device-only copies and syncs (the host waiting for the device).
        calls : int | None
            The number of times the phase must have been entered.
        """
        allow = {kind: dict(limits or {}) for kind, limits in (allow or {}).items()}
        for kind, limits in allow.items():
            if kind not in KINDS:
                raise ValueError(f"unknown transfer kind {kind!r}; kinds: {KINDS}")
            unknown = set(limits) - set(_LIMITS)
            if unknown:
                raise ValueError(
                    f"unknown limits {sorted(unknown)} for {kind!r}; limits: {_LIMITS}"
                )
        self._rules[phase] = _Rule(allow, tuple(ignore), calls)

    def violations(self) -> list[str]:
        """The broken rules, one line each, with the offending events."""
        found = []
        for name, rule in self._rules.items():
            events = self[name].events
            n_calls = self.calls.get(name, 0)
            if rule.calls is not None and n_calls != rule.calls:
                found.append(f"{name}: {n_calls} call(s), expected {rule.calls}")
            for event in events:
                if event.kind not in rule.allow:
                    if event.kind not in rule.ignore:
                        found.append(f"{name}: not allowed: {event}")
                    continue
                limits = rule.allow[event.kind]
                limit = limits.get("max_nbytes")
                if limit is not None and (event.nbytes is None or event.nbytes > limit):
                    found.append(
                        f"{name}: {event.nbytes} bytes > max_nbytes={limit}: {event}"
                    )
                for flag in ("blocking", "implicit"):
                    if flag in limits and getattr(event, flag) != limits[flag]:
                        found.append(
                            f"{name}: {flag}={getattr(event, flag)} not allowed: {event}"
                        )
            for kind, limits in rule.allow.items():
                of_kind = [e for e in events if e.kind == kind]
                limit = limits.get("max_count")
                if limit is not None and len(of_kind) > limit:
                    found.append(f"{name}: {len(of_kind)} {kind} > max_count={limit}")
                limit = limits.get("max_total_bytes")
                total = sum(e.nbytes or 0 for e in of_kind)
                if limit is not None and total > limit:
                    found.append(
                        f"{name}: {total} bytes of {kind} > max_total_bytes={limit}"
                    )
        return found

    def report(self) -> str:
        """The events of every phase (see :meth:`TransferCounter.report`) and the broken rules."""
        lines = []
        for name, counter in self.phases.items():
            lines.append(f"{name} ({self.calls.get(name, 0)} call(s)):")
            lines += [f"  {line}" for line in counter.report().splitlines()]
        found = self.violations()
        if found:
            lines.append("broken rules:")
            lines += [f"  {line}" for line in found]
        return "\n".join(lines)

    def check(self) -> None:
        """Raise ``AssertionError`` with :meth:`report` if a rule is broken."""
        if self.violations():
            raise AssertionError("transfer budget exceeded:\n" + self.report())
