"""Tests for `cunumpy.threads`: the BLAS limit, its restore, and the thread report.

The limit is checked on the BLAS that NumPy and SciPy load; the oversubscription
logic on synthetic pools, so it does not depend on the machine.
"""

import ctypes
import ctypes.util
import json
import warnings

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import threads

threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_counts():
    return {
        pool.filepath: pool.num_threads
        for pool in threads.thread_pools()
        if pool.user_api == "blas"
    }


@pytest.fixture
def blas():
    """The loaded BLAS pools at two threads or more (skip without a controllable BLAS)."""
    np.dot(np.ones(4), np.ones(4))  # make sure NumPy's BLAS is loaded
    counts = _blas_counts()
    if not counts:
        pytest.skip("no BLAS that threadpoolctl controls")
    controller = threadpoolctl.ThreadpoolController()
    before = {lib.filepath: lib.num_threads for lib in controller.lib_controllers}
    for lib in controller.lib_controllers:
        if lib.user_api == "blas":
            lib.set_num_threads(2)
    yield counts
    for lib in threadpoolctl.ThreadpoolController().lib_controllers:
        if lib.filepath in before:
            lib.set_num_threads(before[lib.filepath])


@pytest.fixture
def no_blas_env(monkeypatch):
    for names in threads.BLAS_THREAD_ENV_VARS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)


def test_module_is_part_of_the_namespace():
    assert xp.threads is threads
    assert "threads" in xp.__all__


def test_limit_and_restore(blas, no_blas_env):
    previous = threads.limit_blas_threads(1)
    assert set(previous) == set(blas)
    assert set(previous.values()) == {2}
    assert set(_blas_counts().values()) == {1}
    threads.restore_blas_threads(previous)
    assert set(_blas_counts().values()) == {2}


def test_context_manager_restores(blas, no_blas_env):
    with threads.blas_threads(1) as previous:
        assert previous
        assert set(_blas_counts().values()) == {1}
    assert set(_blas_counts().values()) == {2}


def test_a_library_set_in_the_environment_keeps_its_threads(blas, monkeypatch):
    for names in threads.BLAS_THREAD_ENV_VARS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    previous = threads.limit_blas_threads(1)
    pools = {pool.filepath: pool for pool in threads.thread_pools()}
    for path in blas:
        if pools[path].internal_api == "openblas":
            assert path not in previous
            assert pools[path].num_threads == 2
    # an MKL setting does not exempt OpenBLAS
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    previous = threads.limit_blas_threads(1)
    threads.restore_blas_threads(previous)
    assert all(
        path in previous
        for path, pool in pools.items()
        if pool.internal_api == "openblas"
    )


def test_openmp_pool_is_not_changed(blas, no_blas_env):
    path = ctypes.util.find_library("gomp")
    if path is None:
        pytest.skip("no libgomp")
    ctypes.CDLL(path)
    before = {
        pool.filepath: pool.num_threads
        for pool in threads.thread_pools()
        if pool.user_api == "openmp"
    }
    if not before:
        pytest.skip("threadpoolctl does not see the OpenMP runtime")
    with threads.blas_threads(1):
        after = {
            pool.filepath: pool.num_threads
            for pool in threads.thread_pools()
            if pool.user_api == "openmp"
        }
    assert after == before


def test_limit_rejects_zero_threads():
    with pytest.raises(ValueError, match="at least 1"):
        threads.limit_blas_threads(0)


def test_without_threadpoolctl_nothing_changes(monkeypatch):
    monkeypatch.setattr(threads, "_threadpoolctl", lambda: None)
    assert threads.thread_pools() == []
    with pytest.warns(RuntimeWarning, match="threadpoolctl"):
        assert threads.limit_blas_threads(1) == {}
    report = threads.thread_report()
    assert report.threadpoolctl is False
    assert report.blas_threads == 0


def _pools(openmp, blas):
    pools = []
    if openmp:
        pools.append(threads.ThreadPool("openmp", "openmp", "libgomp", openmp))
    if blas:
        pools.append(threads.ThreadPool("blas", "openblas", "libopenblas", blas))
    return pools


@pytest.mark.parametrize(
    ("cores", "ranks", "openmp", "blas", "oversubscribed", "competes"),
    [
        (8, 1, 8, 1, False, False),  # one rank on all cores, BLAS limited
        (8, 1, 8, 8, False, True),  # BLAS threads next to the OpenMP kernels
        (8, 4, 8, 1, True, False),  # four unbound ranks with eight threads each
        (8, 4, 2, 1, False, False),  # four ranks with two threads: fits
        (8, 2, 4, 1, False, False),
    ],
)
def test_report_flags(
    monkeypatch, cores, ranks, openmp, blas, oversubscribed, competes
):
    monkeypatch.setattr(threads, "thread_pools", lambda: _pools(openmp, blas))
    monkeypatch.setattr(threads, "available_cores", lambda: cores)
    monkeypatch.setattr(threads, "ranks_per_node", lambda comm: ranks)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: cores)
    report = threads.thread_report()
    assert (report.oversubscribed, report.blas_competes) == (oversubscribed, competes)
    assert report.openmp_threads == openmp
    assert report.blas_threads == blas
    json.dumps(report.as_dict())  # plain data
    assert f"{ranks} rank(s) per node" in report.summary()


def test_bound_ranks_are_judged_by_their_own_cores(monkeypatch):
    """Four ranks bound to two cores each of an eight-core node, with two threads each."""
    monkeypatch.setattr(threads, "thread_pools", lambda: _pools(2, 1))
    monkeypatch.setattr(threads, "available_cores", lambda: 2)
    monkeypatch.setattr(threads, "ranks_per_node", lambda comm: 4)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 8)
    assert not threads.thread_report().oversubscribed
    monkeypatch.setattr(threads, "thread_pools", lambda: _pools(4, 1))
    assert threads.thread_report().oversubscribed


def test_openmp_threads_from_the_environment(monkeypatch):
    monkeypatch.setattr(threads, "thread_pools", list)
    monkeypatch.setenv("OMP_NUM_THREADS", "3,2")
    assert threads.thread_report().openmp_threads == 3


def test_configure_threads_warns_when_oversubscribed(monkeypatch):
    monkeypatch.setattr(threads, "limit_blas_threads", lambda n: {})
    monkeypatch.setattr(threads, "thread_pools", lambda: _pools(8, 1))
    monkeypatch.setattr(threads, "available_cores", lambda: 8)
    monkeypatch.setattr(threads, "ranks_per_node", lambda comm: 4)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 8)
    with pytest.warns(threads.OversubscriptionWarning, match="OMP_NUM_THREADS"):
        report = threads.configure_threads()
    assert report.oversubscribed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        threads.configure_threads(warn=False)


def test_configure_threads_limits_blas(blas, no_blas_env, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    report = threads.configure_threads()
    try:
        assert report.blas_threads == 1
        assert not report.blas_competes
        assert report.blas_limited
        assert report.env["OMP_NUM_THREADS"] == "1"
    finally:
        threadpoolctl.threadpool_limits(limits=2, user_api="blas")


def test_ranks_per_node_without_mpi():
    from maybempi import MPI

    assert threads.ranks_per_node(None) == 1
    assert threads.ranks_per_node(MPI.COMM_WORLD) == MPI.COMM_WORLD.Get_size()


def test_command_line_prints_the_report(capsys, blas, no_blas_env):
    threads.main(["--limit"])
    out = capsys.readouterr().out
    assert "rank(s) per node" in out
    assert "blas" in out
    threads.main(["--json"])
    assert json.loads(capsys.readouterr().out)["cores"] >= 1
