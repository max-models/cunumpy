"""Tests for the CUDA-aware MPI check: `mpi_is_cuda_aware` and
`require_cuda_aware_mpi`. They need neither MPI nor a GPU: the communicator is
a fake and the probe buffers are host arrays. The last test needs both."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import xp as xp_module


class FakeComm:
    """A communicator recording the probe's calls.

    With one rank, `Sendrecv` hands the sent buffer back; with more, it fills
    the receive buffer with what the source rank would have sent.
    """

    def __init__(self, rank=0, size=1, fail=None, corrupt=False, allreduce_result=None):
        self.rank = rank
        self.size = size
        self.fail = fail
        self.corrupt = corrupt
        self.allreduce_result = allreduce_result
        self.sendrecv_calls = []
        self.allreduce_calls = []

    def Sendrecv(self, sendbuf, dest, recvbuf=None, source=None):
        self.sendrecv_calls.append((dest, source))
        if self.fail is not None:
            raise self.fail
        if self.corrupt:
            recvbuf[...] = -1.0
        else:
            recvbuf[...] = sendbuf - self.rank + source

    def allreduce(self, value, op=None):
        self.allreduce_calls.append((value, op))
        return value if self.allreduce_result is None else self.allreduce_result


@pytest.fixture
def no_mpi4py(monkeypatch):
    """Make any import of mpi4py fail."""
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)


@pytest.fixture
def fake_mpi(monkeypatch):
    """A fake `mpi4py.MPI` module with `COMM_WORLD` and `LAND`."""
    comm = FakeComm()
    MPI = SimpleNamespace(COMM_WORLD=comm, LAND="LAND")
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=MPI))
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", MPI)
    return comm


@pytest.fixture
def device_buffers(monkeypatch):
    """Pretend the CuPy backend is active, with host arrays as probe buffers."""
    monkeypatch.setattr(xp_module, "_device_buffers_in_use", lambda: True)

    def buffers(rank, n=4):
        return np.arange(n, dtype=np.float64) + rank, np.empty(n, dtype=np.float64)

    monkeypatch.setattr(xp_module, "_mpi_probe_buffers", buffers)


def test_numpy_backend_returns_false_without_mpi(no_mpi4py):
    with xp.use_backend("numpy"):
        assert xp.mpi.mpi_is_cuda_aware() is False
        assert xp.mpi.mpi_is_cuda_aware(FakeComm()) is False
        assert xp.mpi.require_cuda_aware_mpi() is None  # no-op, mpi4py not imported


def test_unknown_method():
    with pytest.raises(ValueError, match="probe"):
        xp.mpi.mpi_is_cuda_aware(FakeComm(), method="query")


def test_probe_succeeds(device_buffers, fake_mpi):
    comm = FakeComm()
    assert xp.mpi.mpi_is_cuda_aware(comm) is True
    assert comm.sendrecv_calls == [(0, 0)]  # size 1: to and from itself
    assert comm.allreduce_calls == [(True, "LAND")]


def test_probe_uses_comm_world_by_default(device_buffers, fake_mpi):
    assert xp.mpi.mpi_is_cuda_aware() is True
    assert fake_mpi.sendrecv_calls == [(0, 0)]


def test_probe_neighbours(device_buffers, fake_mpi):
    comm = FakeComm(rank=3, size=4)
    assert xp.mpi.mpi_is_cuda_aware(comm) is True
    assert comm.sendrecv_calls == [(0, 2)]  # to the next rank, from the previous


def test_probe_exception_gives_false(device_buffers, fake_mpi):
    comm = FakeComm(fail=RuntimeError("MPI_ERR_BUFFER"))
    assert xp.mpi.mpi_is_cuda_aware(comm) is False
    assert comm.allreduce_calls == [(False, "LAND")]


def test_probe_wrong_values_give_false(device_buffers, fake_mpi):
    comm = FakeComm(corrupt=True)
    assert xp.mpi.mpi_is_cuda_aware(comm) is False


def test_probe_other_rank_failed(device_buffers, fake_mpi):
    comm = FakeComm(allreduce_result=False)  # this rank ok, another one not
    assert xp.mpi.mpi_is_cuda_aware(comm) is False


def test_require_raises(device_buffers, fake_mpi):
    comm = FakeComm(fail=RuntimeError("MPI_ERR_BUFFER"))
    with pytest.raises(RuntimeError, match="CUDA-aware"):
        xp.mpi.require_cuda_aware_mpi(comm)
    xp.mpi.require_cuda_aware_mpi(FakeComm())  # succeeds silently


def test_probe_on_comm_world():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    pytest.importorskip("mpi4py")
    with xp.use_backend("cupy"):
        assert isinstance(xp.mpi.mpi_is_cuda_aware(), bool)
