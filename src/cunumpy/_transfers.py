"""Counting host/device transfers made through cunumpy.

:func:`count_transfers` records every transfer that goes through cunumpy while
a block runs, with the call site that caused it, :func:`assert_no_transfers`
rejects them, and :class:`TransferBudget` checks them per phase of a program.
All are public in :mod:`cunumpy.profiling`; see :doc:`/guides/profiling`.

Only transfers made *through cunumpy* are seen: raw ``cupy.ndarray.get()``,
``cupy.asarray(numpy_array)``, ``numpy.asarray(cupy_array)``, forwarded
backend operations such as ``xp.asarray`` and conversions inside other
libraries are not, nor is ``float(device_array)`` on the real CuPy (the fake
CuPy reports it); use ``nsys`` for those. Like the backend selection, the set
of active counters is process-wide and not thread-safe.
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
    """One transfer recorded by :func:`count_transfers`.

    Attributes
    ----------
    kind : str
        ``"to_host"``, ``"to_device"``, ``"kernel_conversion"``,
        ``"fallback"``, ``"device_copy"`` or ``"sync"`` (see
        :func:`count_transfers`).
    description : str
        What was transferred, e.g. ``"to_numpy(shape=(1000,), dtype=float64)"``
        or ``"PyccelKernel 'push': 3 array(s) copied to the host"``.
    where : str
        The call site outside cunumpy, as ``"file:line"``.
    nbytes : int or None
        Payload bytes of a physical copy; None for markers or unknown sizes.
    blocking : bool
        Whether the host waited for the copy. False for the copies of
        :func:`~cunumpy.to_host_async` and
        :meth:`HostStaging.copy <cunumpy.memory.HostStaging.copy>`.
    implicit : bool
        Whether the event was not asked for explicitly: a scalar read of a
        device array (``float(a)``) on the fake CuPy, recorded as a ``sync``.

    Examples
    --------
    >>> event = xp.profiling.TransferEvent("to_host", "x", "step.py:12", nbytes=8)
    >>> print(event)
    step.py:12: to_host: x
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
    """The transfers recorded while a :func:`count_transfers` block runs.

    The properties count the events by kind; :meth:`report` lists them by
    call site. Usually created by :func:`count_transfers`.

    Attributes
    ----------
    events : list of TransferEvent
        Every recorded transfer, in order.

    Examples
    --------
    >>> counter = xp.profiling.TransferCounter()
    >>> event = xp.profiling.TransferEvent("to_host", "x", "step.py:12", nbytes=8)
    >>> counter.events.append(event)
    >>> counter.to_host, counter.bytes_to_host, counter.total
    (1, 8, 1)
    """

    def __init__(self) -> None:
        self.events: list[TransferEvent] = []

    def __repr__(self) -> str:
        counts = ", ".join(f"{kind}={self.count(kind)}" for kind in KINDS)
        return f"TransferCounter({counts})"

    def _add(self, event: TransferEvent) -> None:
        self.events.append(event)

    def count(self, kind: str) -> int:
        """Return the number of events of one kind.

        Parameters
        ----------
        kind : str
            An event kind, e.g. ``"to_host"``.

        Returns
        -------
        int
            The number of events of `kind`.
        """
        return sum(1 for event in self.events if event.kind == kind)

    @property
    def to_host(self) -> int:
        """Number of device-to-host copies (``to_host`` events)."""
        return self.count("to_host")

    @property
    def to_device(self) -> int:
        """Number of host-to-device copies (``to_device`` events)."""
        return self.count("to_device")

    @property
    def kernel_conversions(self) -> int:
        """Number of ``PyccelKernel`` calls that copied device arrays to the host."""
        return self.count("kernel_conversion")

    @property
    def kernel_conversion_calls(self) -> list[TransferEvent]:
        """The ``kernel_conversion`` events: kernel name, arrays, call site."""
        return [event for event in self.events if event.kind == "kernel_conversion"]

    @property
    def fallbacks(self) -> int:
        """Number of ``Kernel`` calls that fell back to the host kernel on CuPy."""
        return self.count("fallback")

    @property
    def syncs(self) -> int:
        """Number of times the host waited for the device (``sync`` events)."""
        return self.count("sync")

    @property
    def total(self) -> int:
        """Number of events except syncs, including conversion/fallback markers."""
        return sum(1 for event in self.events if event.kind != "sync")

    def bytes(self, kind: str) -> int:
        """Return the known bytes copied by the events of one kind.

        Parameters
        ----------
        kind : str
            An event kind, e.g. ``"to_host"``.

        Returns
        -------
        int
            The sum of `nbytes`; markers and unknown sizes add zero.
        """
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
        """Number of device-only conversions (``device_copy`` events)."""
        return self.count("device_copy")

    def report(self) -> str:
        """Return a readable summary, grouped by kind and call site.

        Returns
        -------
        str
            One header line with the counts per kind, then one line per call
            site and description, e.g. ``"step.py:12: to_numpy(...) (x3)"``.
        """
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
    """Return the innermost ``file:line`` on the stack outside cunumpy."""
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

    Records every transfer cunumpy makes while the block runs, with its call
    site. Calls that do not copy record nothing (e.g. ``to_numpy`` of a NumPy
    array); with no counter active, a call costs one truth test. Blocks can
    be nested; each active counter sees the transfers made inside it. Like any
    :func:`contextlib.contextmanager`, it is also a decorator. Transfers that
    bypass cunumpy (raw ``cupy.ndarray.get()``, conversions inside other
    libraries) are not seen. Process-wide state, not thread-safe.

    Parameters
    ----------
    into : TransferCounter, optional
        Add the events to this counter instead of a new one (e.g. to
        accumulate over calls). A counter that is already active is not added
        again, so each event is counted once.

    Yields
    ------
    TransferCounter
        The counter, filled while the block runs.

    See Also
    --------
    assert_no_transfers : Fail if the block transfers.
    TransferBudget : Count and check transfers per phase.

    Notes
    -----
    The event kinds are:

    * ``to_host`` / ``to_device``: a physical copy between host and device
      (conversions such as :func:`~cunumpy.to_numpy` and
      :func:`~cunumpy.to_cupy`, mirrors, staging, serial MPI, kernel output
      copy-back), with its size in bytes;
    * ``kernel_conversion``: a :class:`~cunumpy.kernels.PyccelKernel` call that
      copied device arrays to the host and back, one marker per call;
    * ``fallback``: a :class:`~cunumpy.kernels.Kernel` without CUDA version
      that ran its host kernel on the CuPy backend, one marker per call;
    * ``device_copy``: a device-only dtype/layout conversion;
    * ``sync``: the host waited for the device (:func:`~cunumpy.synchronize`,
      the MPI helpers, the CUDA debug mode and, on the fake CuPy, scalar
      reads such as ``float(a)``). Not included in
      :attr:`TransferCounter.total`.

    Examples
    --------
    >>> with xp.profiling.count_transfers() as counter:
    ...     y = xp.to_numpy(xp.ones(3))  # a NumPy array: no copy
    >>> counter.total
    0
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
    """Fail if the block makes a host/device transfer through cunumpy.

    A :func:`count_transfers` block that, on exit, rejects host/device copies
    and host conversion/fallback markers. Device-only conversions are allowed.
    Only checked if the block exits normally; an exception raised inside
    propagates as it is.

    Parameters
    ----------
    syncs : bool, optional
        Also reject syncs (the host waiting for the device), e.g. for a time
        step that must never stall. Default False.

    Yields
    ------
    TransferCounter
        The counter of the block.

    Raises
    ------
    AssertionError
        If a transfer was counted; the message is the counter's
        :meth:`~TransferCounter.report`.

    See Also
    --------
    count_transfers : Count without failing.

    Examples
    --------
    >>> with xp.profiling.assert_no_transfers():
    ...     y = 2.0 * xp.ones(3)
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
    """Counter that adds every event to the innermost phase of a budget."""

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
    """Count transfers per phase of a program and check a rule for each phase.

    E.g. the time step must not copy between host and device, the diagnostics
    may copy a few scalars, the output each saved array once. Mark phases with
    :meth:`phase` or :meth:`count`, set rules with :meth:`require` and call
    :meth:`check`. Events of a phase accumulate over its calls. Phases nest:
    an event is counted in the innermost phase only, and a phase re-entered
    inside itself (recursion) counts each event and call once.

    Parameters
    ----------
    started : bool, optional
        Count from the start (default); with False, phases run uncounted
        until :meth:`start`.

    Attributes
    ----------
    phases : dict of str to TransferCounter
        The events of each phase, accumulated over its calls.
    calls : dict of str to int
        How many times each phase was entered while counting.
    started : bool
        Whether phases are counted.

    See Also
    --------
    count_transfers : Count the transfers of one block.

    Examples
    --------
    >>> budget = xp.profiling.TransferBudget()
    >>> for step in range(3):
    ...     with budget.phase("step"):
    ...         x = 2.0 * xp.ones(4)
    >>> budget.require("step", allow={"to_host": {"max_nbytes": 8}}, calls=3)
    >>> budget.check()
    >>> budget.violations()
    []
    """

    def __init__(self, *, started: bool = True) -> None:
        self.phases: dict[str, TransferCounter] = {}
        self.calls: dict[str, int] = {}
        self.started = started
        self._rules: dict[str, _Rule] = {}
        self._stack: list[str] = []
        self._router = _PhaseRouter(self)

    def __getitem__(self, phase: str) -> TransferCounter:
        """Return the counter of `phase` (empty if it has not run)."""
        return self.phases.setdefault(phase, TransferCounter())

    def start(self) -> None:
        """Count the phases from now on."""
        self.started = True

    def stop(self) -> None:
        """Stop counting; phases run uncounted until :meth:`start`."""
        self.started = False

    @contextmanager
    def phase(self, name: str) -> Generator[TransferCounter, None, None]:
        """Count the transfers of the block in a phase.

        Parameters
        ----------
        name : str
            The phase; its events accumulate over calls.

        Yields
        ------
        TransferCounter
            The counter of the phase (``budget[name]``).
        """
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
        """Return a decorator that counts every call of a function in a phase.

        Parameters
        ----------
        name : str
            The phase.

        Returns
        -------
        callable
            The decorator, e.g. ``f = budget.count("integrate")(f)``.
        """

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
        """Set the rule of a phase, checked by :meth:`check`.

        Parameters
        ----------
        phase : str
            The phase.
        allow : mapping, optional
            The event kinds the phase may have, each with its limits (None or
            ``{}``: any number); every other kind is forbidden unless in
            `ignore`. Limits: ``max_nbytes`` (each event; an unknown size
            breaks it), ``max_count`` and ``max_total_bytes`` (whole phase),
            ``blocking`` and ``implicit`` (required value of the event field),
            e.g. ``{"to_host": {"max_nbytes": 8, "blocking": False}}``.
        ignore : sequence of str, optional
            Kinds not checked unless in `allow`; by default ``device_copy``
            and ``sync``.
        calls : int, optional
            The number of times the phase must have been entered.

        Raises
        ------
        ValueError
            If `allow` has an unknown kind or limit.
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
        """Return the broken rules.

        Returns
        -------
        list of str
            One line per broken rule, with the offending event and its call
            site.
        """
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
        """Return the events of every phase and the broken rules.

        Returns
        -------
        str
            :meth:`TransferCounter.report` of each phase, then the
            :meth:`violations`.
        """
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
        """Raise if a rule is broken.

        Raises
        ------
        AssertionError
            If a rule is broken; the message is :meth:`report`.
        """
        if self.violations():
            raise AssertionError("transfer budget exceeded:\n" + self.report())
