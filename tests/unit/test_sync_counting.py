"""count_transfers counts syncs: synchronize(), and scalar reads on the fake CuPy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cunumpy as xp
from cunumpy._transfers import TransferCounter, _record_sync


def test_sync_events_are_listed_but_not_in_the_total():
    with xp.profiling.count_transfers() as counter:
        _record_sync("something")
    assert counter.syncs == 1 and counter.total == 0
    assert "sync (1)" in counter.report()


def test_assert_no_transfers_accepts_syncs_unless_asked():
    with xp.profiling.assert_no_transfers():
        _record_sync("something")
    with (
        pytest.raises(AssertionError, match="sync"),
        xp.profiling.assert_no_transfers(syncs=True),
    ):
        _record_sync("something")


def test_numpy_backend_synchronize_is_not_a_sync():
    with xp.profiling.count_transfers() as counter:
        xp.synchronize()
    assert counter.syncs == 0
    assert isinstance(counter, TransferCounter)


SCRIPT = r"""
import cunumpy as xp

a = xp.asarray([1.0, 2.0, 3.0])
with xp.profiling.count_transfers() as counter:
    float(a[0]); int(a[1]); bool(a[2]); a.sum().item(); a.tolist()
    xp.synchronize()
    b = a * 2  # device work: no sync
assert counter.syncs == 6, counter.report()
assert counter.total == 0
with xp.profiling.assert_no_transfers():
    float(a[0])
try:
    with xp.profiling.assert_no_transfers(syncs=True):
        float(a[0])
except AssertionError:
    pass
else:
    raise AssertionError("syncs=True must reject a scalar read")
print("sync counting OK")
"""


def test_scalar_reads_and_synchronize_are_counted_on_the_fake_cupy():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUNUMPY_FAKE_CUPY="1", CUNUMPY_BACKEND="cupy")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), env.get("PYTHONPATH", "")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sync counting OK" in result.stdout
