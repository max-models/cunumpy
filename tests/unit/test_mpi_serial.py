"""Tests for `xp.mpi.get_mpi`, `launched_under_mpi` and the serial stand-in `SerialComm`."""

import sys
import types

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _mpi_serial
from cunumpy.mpi import SerialMPI, get_mpi, launched_under_mpi

MPI = get_mpi(False)
comm = MPI.COMM_WORLD


@pytest.fixture
def clean_env(monkeypatch):
    """No launcher variables, no override, no mpi4py imported, no cached decision."""
    for variable in (*_mpi_serial._LAUNCHER_VARIABLES, _mpi_serial.OVERRIDE_VARIABLE):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.setattr(_mpi_serial, "_AUTO_MPI", None)
    return monkeypatch


@pytest.fixture
def fake_mpi4py(clean_env):
    """A stand-in for an installed mpi4py (without starting MPI)."""
    module = types.ModuleType("mpi4py.MPI")
    module.Is_initialized = lambda: True
    package = types.ModuleType("mpi4py")
    package.MPI = module
    clean_env.setitem(sys.modules, "mpi4py", package)
    clean_env.setitem(sys.modules, "mpi4py.MPI", module)
    return module


@pytest.fixture
def no_mpi4py(clean_env):
    clean_env.setitem(sys.modules, "mpi4py", None)
    clean_env.setitem(sys.modules, "mpi4py.MPI", None)
    return clean_env


# ------------------------------------------------------------ the decision


def test_serial_by_default(clean_env):
    assert launched_under_mpi() is False
    assert isinstance(get_mpi(), SerialMPI)


@pytest.mark.parametrize("variable", _mpi_serial._LAUNCHER_VARIABLES)
def test_launcher_variables(clean_env, variable):
    clean_env.setenv(variable, "3")
    assert launched_under_mpi() is True


def test_slurm_batch_script_is_not_an_mpi_launch(clean_env):
    clean_env.setenv("SLURM_PROCID", "0")
    assert launched_under_mpi() is False


@pytest.mark.parametrize(
    ("value", "launcher", "expected"),
    [("1", False, True), ("on", False, True), ("0", True, False), ("no", True, False),
     ("maybe", True, True), ("maybe", False, False)],
)  # fmt: skip
def test_override(clean_env, value, launcher, expected):
    clean_env.setenv("CUNUMPY_MPI", value)
    if launcher:
        clean_env.setenv("PMI_RANK", "0")
    assert launched_under_mpi() is expected


def test_initialized_mpi4py_counts_as_mpi(fake_mpi4py):
    assert launched_under_mpi() is True
    fake_mpi4py.Is_initialized = lambda: False
    assert launched_under_mpi() is False


def test_get_mpi_under_a_launcher(fake_mpi4py, clean_env):
    fake_mpi4py.Is_initialized = lambda: False
    clean_env.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert get_mpi() is fake_mpi4py
    clean_env.delenv("OMPI_COMM_WORLD_RANK")
    assert get_mpi() is fake_mpi4py  # decided once per process


def test_get_mpi_explicit(fake_mpi4py):
    assert get_mpi(True) is fake_mpi4py
    assert get_mpi(False) is MPI


def test_launcher_without_mpi4py_warns(no_mpi4py):
    no_mpi4py.setenv("PMI_RANK", "0")
    with pytest.warns(RuntimeWarning, match="mpi4py is not installed"):
        assert isinstance(get_mpi(), SerialMPI)


def test_explicit_mpi_without_mpi4py_raises(no_mpi4py):
    with pytest.raises(ImportError):
        get_mpi(True)


def test_exported():
    assert xp.mpi.get_mpi is get_mpi
    assert xp.mpi.local_rank is _mpi_serial.local_rank
    assert "get_mpi" not in xp._MOVED  # never was at the top level


# -------------------------------------------------- the serial communicator


def test_queries():
    assert comm.rank == 0 and comm.size == 1
    assert comm.Get_rank() == 0 and comm.Get_size() == 1
    assert isinstance(comm, MPI.Comm) and isinstance(comm, MPI.Intracomm)
    assert comm.Is_intra() and not comm.Is_inter()
    assert MPI.COMM_SELF.Get_size() == 1
    assert repr(comm) == "SerialComm(COMM_WORLD)"


def test_object_collectives_return_the_value():
    value = {"a": 1}
    assert comm.bcast(value) is value
    assert comm.allreduce(5, op=MPI.SUM) == 5
    assert comm.allreduce(2.5, op=MPI.MAX) == 2.5
    assert comm.reduce(7) == 7
    assert comm.scan(3) == 3
    assert comm.exscan(3) is None  # as mpi4py on rank 0
    assert comm.gather(value) == [value]
    assert comm.allgather(value) == [value]
    assert comm.scatter([value]) is value
    assert comm.alltoall([value]) == [value]
    assert comm.sendrecv(value, dest=0, source=0) is value
    assert comm.ibcast(value).wait() is value
    assert comm.iallreduce(4).wait() == 4
    assert comm.Barrier() is None and comm.barrier() is None


def test_object_collectives_check_sizes_and_ranks():
    with pytest.raises(ValueError, match="1 item"):
        comm.scatter([1, 2])
    with pytest.raises(ValueError, match="only rank 0"):
        comm.bcast(1, root=1)
    with pytest.raises(ValueError, match="only rank 0"):
        comm.sendrecv(1, dest=1)
    assert comm.sendrecv(1, dest=MPI.PROC_NULL, source=MPI.PROC_NULL) is None


def test_unknown_methods_raise():
    # MockComm returned None for everything, which hid missing support
    with pytest.raises(AttributeError):
        _ = comm.Create_cart
    with pytest.raises(AttributeError):
        _ = MPI.Win


def test_buffer_collectives_copy():
    send = np.arange(6.0).reshape(2, 3)
    for call in (
        lambda r: comm.Allreduce(send, r, op=MPI.SUM),
        lambda r: comm.Reduce(send, r, op=MPI.SUM, root=0),
        lambda r: comm.Allgather(send, r),
        lambda r: comm.Gather(send, r),
        lambda r: comm.Scatter(send, r),
        lambda r: comm.Alltoall(send, r),
        lambda r: comm.Scan(send, r),
        lambda r: comm.Sendrecv(send, dest=0, recvbuf=r, source=0),
        lambda r: comm.Iallreduce(send, r).Wait(),
        lambda r: comm.Iallgather(send, r).Wait(),
    ):
        recv = np.zeros_like(send)
        call(recv)
        np.testing.assert_array_equal(recv, send)


def test_buffer_specs_and_in_place():
    data = np.arange(4.0)
    recv = np.zeros(4)
    comm.Allreduce([data, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
    np.testing.assert_array_equal(recv, data)
    before = data.copy()
    comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
    comm.Bcast(data, root=0)
    comm.Ibcast(data).Wait()
    np.testing.assert_array_equal(data, before)
    comm.Exscan(data, recv)  # undefined on rank 0: untouched
    np.testing.assert_array_equal(recv, before)


def test_vector_collectives_use_the_displacement():
    send = np.array([1.0, 2.0])
    recv = np.zeros(5)
    comm.Allgatherv(send, [recv, [2], [3], MPI.DOUBLE])
    np.testing.assert_array_equal(recv, [0, 0, 0, 1, 2])
    recv[:] = 0
    comm.Gatherv(send, [recv, [2], [1]])
    np.testing.assert_array_equal(recv, [0, 1, 2, 0, 0])
    part = np.zeros(2)
    comm.Scatterv([np.arange(5.0), [2], [2], MPI.DOUBLE], part)
    np.testing.assert_array_equal(part, [2, 3])


def test_buffer_errors():
    with pytest.raises(ValueError, match="too small"):
        comm.Allreduce(np.ones(3), np.zeros(2))
    with pytest.raises(ValueError, match="C-contiguous"):
        comm.Allreduce(np.ones(2), np.zeros((2, 2))[:, 0])
    with pytest.raises(ValueError, match="only rank 0"):
        comm.Sendrecv(np.ones(2), dest=1, recvbuf=np.zeros(2))
    comm.Sendrecv(np.ones(2), dest=MPI.PROC_NULL, recvbuf=np.zeros(2))


class _DeviceArray:
    """Duck-typed device array: `.get()` copies it to the host."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.flags = self._data.flags

    def get(self):
        return self._data.copy()

    def reshape(self, *shape):
        return _DeviceArray(self._data.reshape(*shape))

    @property
    def size(self):
        return self._data.size


def test_device_source_to_host_target():
    recv = np.zeros(3)
    comm.Allreduce(_DeviceArray([1.0, 2.0, 3.0]), recv)
    np.testing.assert_array_equal(recv, [1, 2, 3])


def test_split_dup_and_requests():
    assert comm.Split(0, 0).Get_size() == 1
    assert comm.Split(MPI.UNDEFINED) is MPI.COMM_NULL
    assert comm.Dup().Get_rank() == 0 and comm.Clone().Get_size() == 1
    requests = [comm.Ibarrier(), comm.Ibcast(np.zeros(1))]
    assert MPI.Request.Waitall(requests) is None
    assert MPI.Request.Testall(requests) is True
    assert MPI.Request.waitall([comm.ibcast(1)]) == [1]
    status = MPI.Status()
    assert status.Get_source() == 0 and status.Get_tag() == 0


def test_module_functions():
    assert MPI.Is_initialized() is False and MPI.Is_finalized() is False
    assert MPI.Wtime() > 0 and MPI.Wtick() > 0
    assert isinstance(MPI.Get_processor_name(), str)
    assert repr(MPI.SUM) == "SerialMPI.SUM"
    assert MPI.PROC_NULL == -2 and MPI.ANY_SOURCE == -1 and MPI.ROOT == -3
    assert isinstance(MPI, SerialMPI) and get_mpi(False) is MPI


def test_constants_used_by_struphy_and_feectools():
    assert isinstance(MPI.DOUBLE, MPI.Datatype) and not isinstance(
        MPI.SUM, MPI.Datatype
    )
    assert isinstance(MPI.LOR, MPI.Op)
    assert MPI._typedict[np.dtype(np.float64).char] is MPI.DOUBLE
    # null handles are false, communicators true, like in mpi4py
    assert not MPI.COMM_NULL and not MPI.DATATYPE_NULL
    assert comm and comm != MPI.COMM_NULL
    MPI.Prequest.Startall([])
    assert MPI.Prequest.Waitall([]) is None
    # usable in annotations evaluated at definition time
    assert (MPI.Intracomm | None) is not None


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.skipif(
    not xp.cupy_available(), reason="CuPy/GPU not available"))])  # fmt: skip
def test_buffers_on_either_backend(backend):
    with xp.use_backend(backend):
        send = xp.arange(4.0)
        recv = xp.zeros(4)
        comm.Allreduce(send, recv, op=MPI.SUM)
        np.testing.assert_array_equal(xp.to_numpy(recv), np.arange(4.0))
        host = np.zeros(4)
        comm.Allgather(send, host)  # a device array into a host buffer
        np.testing.assert_array_equal(host, np.arange(4.0))


def test_matches_mpi4py_on_one_process():
    """The same calls on mpi4py's COMM_SELF give the same results (if mpi4py is there)."""
    real = pytest.importorskip("mpi4py.MPI")
    self_comm = real.COMM_SELF
    assert self_comm.allreduce(5) == comm.allreduce(5)
    assert self_comm.gather(3) == comm.gather(3)
    assert self_comm.scatter([4]) == comm.scatter([4])
    assert self_comm.exscan(3) == comm.exscan(3)
    send, recv_real, recv_serial = np.arange(3.0), np.zeros(3), np.zeros(3)
    self_comm.Allreduce(send, recv_real, op=real.SUM)
    comm.Allreduce(send, recv_serial, op=MPI.SUM)
    np.testing.assert_array_equal(recv_real, recv_serial)
