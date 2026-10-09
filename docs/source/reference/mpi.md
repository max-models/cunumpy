# MPI: `cunumpy.mpi`

MPI buffers, CUDA-aware MPI detection and synchronization.

`cunumpy.mpi` also re-exports the MPI stand-ins of
[maybempi](https://max-models.github.io/maybempi/), documented there:
`get_mpi`, `is_serial`, `launched_under_mpi`, `local_rank`,
`OVERRIDE_VARIABLE`, `SerialMPI`, `SerialComm`, `SerialRequest` and
`SerialStatus`.

```{eval-rst}
.. automodule:: cunumpy.mpi
   :members:
   :special-members: __call__
   :exclude-members: get_mpi, is_serial, launched_under_mpi, local_rank, OVERRIDE_VARIABLE, SerialMPI, SerialComm, SerialRequest, SerialStatus
```
