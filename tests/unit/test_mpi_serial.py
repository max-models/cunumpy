"""`xp.mpi` re-exports maybempi's launcher detection and serial stand-in.

The stand-in itself is tested in maybempi; these tests check the cunumpy side:
the re-exports, transfer counting and NumPy/CuPy buffers.
"""

import maybempi
import numpy as np
import pytest

import cunumpy as xp

MPI = xp.mpi.get_mpi(False)
comm = MPI.COMM_WORLD


def test_reexported_from_maybempi():
    for name in (
        "OVERRIDE_VARIABLE",
        "SerialComm",
        "SerialMPI",
        "SerialRequest",
        "SerialStatus",
        "get_mpi",
        "is_serial",
        "launched_under_mpi",
        "local_rank",
    ):
        assert getattr(xp.mpi, name) is getattr(maybempi, name), name
    assert xp.mpi.OVERRIDE_VARIABLE == "MAYBEMPI"
    assert xp.mpi.is_serial(MPI) and xp.mpi.is_serial(comm)


class _DeviceArray:
    """Duck-typed device array: `.get()` copies it to the host."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.flags = self._data.flags
        self.nbytes = self._data.nbytes

    def get(self):
        return self._data.copy()

    def reshape(self, *shape):
        return _DeviceArray(self._data.reshape(*shape))

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def size(self):
        return self._data.size


def test_serial_copies_between_host_and_device_are_counted():
    with xp.profiling.count_transfers() as counter:
        comm.Allreduce(_DeviceArray([1.0, 2.0, 3.0]), np.zeros(3))
        comm.Allgather(np.ones(2), _DeviceArray(np.zeros(2)))
        comm.Allreduce(np.ones(2), np.zeros(2))  # host to host: not a transfer
    assert [(e.kind, e.nbytes) for e in counter.events] == [
        ("to_host", 24),
        ("to_device", 16),
    ]


@pytest.mark.parametrize(
    "backend", [
        "numpy", pytest.param(
            "cupy", marks=pytest.mark.skipif(
            not xp.cupy_available(), reason="CuPy/GPU not available",
            ),
        ),
    ],
)  # fmt: skip
def test_buffers_on_either_backend(backend):
    with xp.use_backend(backend):
        send = xp.arange(4.0)
        recv = xp.zeros(4)
        comm.Allreduce(send, recv, op=MPI.SUM)
        np.testing.assert_array_equal(xp.to_numpy(recv), np.arange(4.0))
        host = np.zeros(4)
        comm.Allgather(send, host)  # a device array into a host buffer
        np.testing.assert_array_equal(host, np.arange(4.0))
