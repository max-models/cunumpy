"""MPI with NumPy or CuPy arrays, and serial runs without MPI.

:func:`get_mpi` returns ``mpi4py.MPI`` when the process was started by an MPI
launcher (:func:`launched_under_mpi`, decided from the environment without
importing mpi4py), and :class:`SerialMPI` otherwise: a stand-in whose
``COMM_WORLD`` is a :class:`SerialComm` of size 1, so that the same code runs
serially without starting MPI::

    import cunumpy as xp

    MPI = xp.mpi.get_mpi()
    comm = MPI.COMM_WORLD
    with xp.mpi.mpi_buffer(rho, send=True, recv=True) as buf:
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

:func:`mpi_buffer` hands an array to mpi4py: the device array itself when the
MPI library is CUDA-aware, a host copy otherwise; :func:`mpi_is_cuda_aware`
finds out which. :func:`local_rank` is the rank of this process on its node
(to pick a GPU, see :func:`cunumpy.cuda.bind_local_device`).

mpi4py is imported only by the functions that need it.
"""

from ._mpi import (
    get_mpi_cuda_aware,
    mpi_buffer,
    mpi_is_cuda_aware,
    require_cuda_aware_mpi,
    set_mpi_cuda_aware,
    synchronize_for_mpi,
)
from ._mpi_serial import (
    OVERRIDE_VARIABLE,
    SerialComm,
    SerialMPI,
    SerialRequest,
    SerialStatus,
    get_mpi,
    launched_under_mpi,
    local_rank,
)

__all__ = [
    "OVERRIDE_VARIABLE",
    "SerialComm",
    "SerialMPI",
    "SerialRequest",
    "SerialStatus",
    "get_mpi",
    "get_mpi_cuda_aware",
    "launched_under_mpi",
    "local_rank",
    "mpi_buffer",
    "mpi_is_cuda_aware",
    "require_cuda_aware_mpi",
    "set_mpi_cuda_aware",
    "synchronize_for_mpi",
]
