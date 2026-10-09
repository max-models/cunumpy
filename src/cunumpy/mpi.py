"""MPI with NumPy or CuPy arrays, and serial runs without MPI.

``get_mpi`` returns ``mpi4py.MPI`` when the process was started by an MPI
launcher (``launched_under_mpi``, decided from the environment without
importing mpi4py), and ``SerialMPI`` otherwise: a stand-in whose
``COMM_WORLD`` is a ``SerialComm`` of size 1, so that the same code runs
serially without starting MPI. These come from the `maybempi
<https://max-models.github.io/maybempi/>`_ package and are re-exported here;
``MAYBEMPI=1``/``0`` (``OVERRIDE_VARIABLE``) overrides the launcher detection::

    MPI = xp.mpi.get_mpi()
    comm = MPI.COMM_WORLD
    with xp.mpi.mpi_buffer(rho, send=True, recv=True) as buf:
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

:func:`mpi_buffer` hands an array to mpi4py: the device array itself when the
MPI library is CUDA-aware, a host copy otherwise; :func:`mpi_is_cuda_aware`
finds out which. ``local_rank`` is the rank of this process on its node
(to pick a GPU, see :func:`cunumpy.cuda.bind_local_device`).

mpi4py is imported only by the functions that need it. See :doc:`/guides/mpi`.
"""

from maybempi import (
    OVERRIDE_VARIABLE,
    SerialComm,
    SerialMPI,
    SerialRequest,
    SerialStatus,
    get_mpi,
    is_serial,
    launched_under_mpi,
    local_rank,
    set_copy_hook,
)

from cunumpy._mpi import (
    MPIStaging,
    get_mpi_cuda_aware,
    mpi_buffer,
    mpi_is_cuda_aware,
    require_cuda_aware_mpi,
    set_mpi_cuda_aware,
    synchronize_for_mpi,
)
from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _nbytes, _record


def _count_serial_copy(kind: str, array: object) -> None:
    """Count the host/device copies of the serial stand-in's buffer collectives."""
    if _COUNTERS:
        _record(
            kind,
            f"SerialComm receive ({kind.replace('_', ' ')})",
            nbytes=_nbytes(array),
        )


set_copy_hook(_count_serial_copy)

__all__ = [
    "OVERRIDE_VARIABLE",
    "MPIStaging",
    "SerialComm",
    "SerialMPI",
    "SerialRequest",
    "SerialStatus",
    "get_mpi",
    "get_mpi_cuda_aware",
    "is_serial",
    "launched_under_mpi",
    "local_rank",
    "mpi_buffer",
    "mpi_is_cuda_aware",
    "require_cuda_aware_mpi",
    "set_mpi_cuda_aware",
    "synchronize_for_mpi",
]
