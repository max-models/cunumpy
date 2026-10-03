"""MPI with NumPy or CuPy arrays.

:func:`mpi_buffer` hands an array to mpi4py: the device array itself when the
MPI library is CUDA-aware, a host copy otherwise; :func:`mpi_is_cuda_aware`
finds out which. :func:`local_rank` is the rank of this process on its node
(to pick a GPU, see :func:`cunumpy.cuda.bind_local_device`)::

    import cunumpy as xp

    with xp.mpi.mpi_buffer(rho, send=True, recv=True) as buf:
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

mpi4py is imported only by the functions that need it.
"""

from ._mpi import (
    get_mpi_cuda_aware,
    local_rank,
    mpi_buffer,
    mpi_is_cuda_aware,
    require_cuda_aware_mpi,
    set_mpi_cuda_aware,
    synchronize_for_mpi,
)

__all__ = [
    "get_mpi_cuda_aware",
    "local_rank",
    "mpi_buffer",
    "mpi_is_cuda_aware",
    "require_cuda_aware_mpi",
    "set_mpi_cuda_aware",
    "synchronize_for_mpi",
]
