"""Tests for `xp.profiling.TransferBudget` and nesting of `count_transfers`."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cunumpy._transfers import _record, _record_sync
from cunumpy.profiling import TransferBudget, TransferCounter, count_transfers


def download(nbytes=8, blocking=True):
    _record("to_host", f"download({nbytes})", nbytes=nbytes, blocking=blocking)


def test_count_transfers_into_a_counter_and_as_a_decorator():
    counter = TransferCounter()

    @count_transfers(counter)
    def step():
        download()

    step()
    step()
    assert counter.to_host == 2
    with count_transfers(counter), count_transfers(counter):  # nested: counted once
        download()
    assert counter.to_host == 3
    with count_transfers() as outer, count_transfers() as inner:
        download()
    assert outer.to_host == inner.to_host == 1


def test_phases_accumulate_and_count_calls():
    budget = TransferBudget()

    @budget.count("integrate")
    def integrate():
        download()

    for _ in range(3):
        integrate()
    with budget.phase("output") as output:
        download(800)
    assert budget["integrate"].to_host == 3
    assert budget.calls == {"integrate": 3, "output": 1}
    assert output.bytes_to_host == 800
    assert "integrate (3 call(s))" in budget.report()


def test_counting_starts_with_start():
    budget = TransferBudget(started=False)
    with budget.phase("integrate"):
        download()  # setup
    budget.start()
    with budget.phase("integrate"):
        download()
    budget.stop()
    with budget.phase("integrate"):
        download()
    assert budget["integrate"].to_host == 1
    assert budget.calls == {"integrate": 1}


def test_nested_phases_count_each_event_once():
    budget = TransferBudget()

    @budget.count("integrate")
    def integrate(depth):
        download()
        if depth:
            integrate(depth - 1)  # recursion: the same phase
            with budget.phase("diagnostics"):
                download(16)

    integrate(1)
    assert budget["integrate"].to_host == 2
    assert budget["diagnostics"].to_host == 1  # not also in "integrate"
    assert budget.calls == {"integrate": 1, "diagnostics": 1}


def test_rules_that_pass():
    budget = TransferBudget()
    for _ in range(2):
        with budget.phase("integrate"):
            download(8)
            _record("device_copy", "astype", nbytes=80)
            _record_sync("synchronize()")
    with budget.phase("output"):
        download(400)
        download(400)
    budget.require("integrate", allow={"to_host": {"max_nbytes": 8}}, calls=2)
    budget.require(
        "output", allow={"to_host": {"max_count": 2, "max_total_bytes": 800}}
    )
    budget.require("never_run", calls=0)
    assert budget.violations() == []
    budget.check()


def test_rules_that_fail_report_the_events_and_where():
    budget = TransferBudget()
    with budget.phase("integrate"):
        download(80)
        _record("fallback", "Kernel 'push' ran its host kernel")
    with budget.phase("output"):
        for _ in range(3):
            download(400)
    budget.require("integrate", allow={"to_host": {"max_nbytes": 8}}, calls=2)
    budget.require(
        "output", allow={"to_host": {"max_count": 2, "max_total_bytes": 1000}}
    )
    with pytest.raises(AssertionError) as info:
        budget.check()
    report = str(info.value)
    assert "integrate: 1 call(s), expected 2" in report
    assert "80 bytes > max_nbytes=8" in report
    assert "not allowed" in report and "fallback" in report
    assert "3 to_host > max_count=2" in report
    assert "1200 bytes of to_host > max_total_bytes=1000" in report
    assert f"{Path(__file__).name}:" in report  # where it happened


def test_blocking_and_implicit_events_are_told_apart():
    budget = TransferBudget()
    with budget.phase("solve"):
        download(8, blocking=False)
        _record_sync("float(a)", implicit=True)
    budget.require(
        "solve",
        allow={"to_host": {"blocking": False}, "sync": {"implicit": False}},
    )
    found = budget.violations()
    assert len(found) == 1 and "implicit=True not allowed" in found[0]
    with budget.phase("solve"):
        download(8)  # blocking
    assert any("blocking=True not allowed" in v for v in budget.violations())


def test_unknown_kinds_and_limits_are_refused():
    budget = TransferBudget()
    with pytest.raises(ValueError, match="unknown transfer kind"):
        budget.require("x", allow={"to_hots": None})
    with pytest.raises(ValueError, match="unknown limits"):
        budget.require("x", allow={"to_host": {"max_bytes": 8}})


def test_fake_cupy_scalar_reads_are_implicit_syncs():
    code = """
import cunumpy as xp

with xp.use_backend("cupy"), xp.profiling.count_transfers() as counter:
    a = xp.ones(3)
    float(a.sum())
    xp.synchronize()
    copy = xp.to_host_async(a.sum())
    assert copy.ready() and copy.result() == 3.0
syncs = [e for e in counter.events if e.kind == "sync"]
assert [e.implicit for e in syncs] == [True, False], syncs
(event,) = [e for e in counter.events if e.kind == "to_host"]
assert not event.blocking and event.nbytes == 8
print("ok")
"""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUNUMPY_FAKE_CUPY="1", PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
