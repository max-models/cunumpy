"""PETSc vectors that share memory with NumPy or CuPy arrays (no copies).

:func:`petsc_vec` wraps an array as a PETSc vector that uses its memory, so a
GPU code can hand its CuPy right-hand side and solution to a PETSc solver
without copying them to the host at every solve::

    b_vec, phi_vec = xp.petsc.petsc_vec(b), xp.petsc.petsc_vec(phi)
    xp.synchronize()           # CuPy work on b done before PETSc reads it
    ksp.solve(b_vec, phi_vec)  # writes into phi
    xp.synchronize()           # PETSc done before CuPy reads phi

For the solve to stay on the GPU, the matrix must be a GPU type as well
(``mat.setType("aijcusparse")``, or ``-mat_type aijcusparse -vec_type cuda``);
otherwise PETSc copies the vectors to the host for the matrix products.
petsc4py is imported only when :func:`petsc_vec` is called. See
:doc:`/guides/solvers`.
"""

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np

__all__ = ["petsc_vec"]

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array

_DEVICE_VEC_TYPES = ("cuda", "hip")


def _petsc() -> Any:
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise ImportError(
            "xp.petsc.petsc_vec needs petsc4py (pip install petsc4py)",
        ) from error
    return PETSc


def petsc_vec(array: Any, comm: Any = None) -> Any:
    """Wrap `array` as a PETSc vector that shares its memory, through DLPack.

    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        C-contiguous array of PETSc's scalar type (``PETSc.ScalarType``,
        usually float64). A multi-dimensional array is seen by PETSc in
        row-major order, as ``array.ravel()``. The array is never copied.
    comm : mpi4py.MPI.Comm or PETSc.Comm, optional
        Communicator of the vector; PETSc's default (``COMM_WORLD``) if None.
        With several processes, `array` is this process's part.

    Returns
    -------
    petsc4py.PETSc.Vec
        A sequential or MPI vector (``seq``/``mpi`` for a NumPy array,
        ``seqcuda``/``mpicuda`` or ``seqhip``/``mpihip`` for a CuPy array).
        It keeps a reference to `array`, which therefore stays alive as long
        as the vector.

    Raises
    ------
    TypeError
        If `array` has another dtype than ``PETSc.ScalarType``, or is not an
        array.
    ValueError
        If `array` is not C-contiguous.
    RuntimeError
        If `array` is a CuPy array and petsc4py has no CUDA or HIP support
        (PETSc would otherwise silently work on a host copy).
    ImportError
        If petsc4py is not installed.

    Notes
    -----
    PETSc and CuPy may run on different streams: synchronize
    (:func:`cunumpy.synchronize`) before PETSc reads an array that CuPy wrote,
    and before CuPy reads a vector that PETSc wrote.

    Examples
    --------
    >>> b = xp.zeros(100)
    >>> b_vec = xp.petsc.petsc_vec(b)  # doctest: +SKIP
    >>> b_vec.getSize()  # doctest: +SKIP
    100
    """
    PETSc = _petsc()
    device = _is_device_array(array)
    if not (device or isinstance(array, np.ndarray)):
        raise TypeError(
            f"petsc_vec takes a NumPy or CuPy array, got {type(array).__name__}",
        )
    scalar = np.dtype(PETSc.ScalarType)
    if array.dtype != scalar:
        raise TypeError(
            f"petsc_vec needs an array of PETSc's scalar type {scalar}, got "
            f"{array.dtype} (convert it once, outside the time loop)",
        )
    if not array.flags.c_contiguous:
        raise ValueError("petsc_vec needs a C-contiguous array (no copy is made)")

    try:
        vec = PETSc.Vec().createWithDLPack(array, comm=comm)
    except PETSc.Error as error:
        if device:
            raise RuntimeError(
                "petsc4py could not wrap the CuPy array; it needs a PETSc built "
                "with CUDA or HIP support (--with-cuda / --with-hip)",
            ) from error
        raise
    if device and not any(t in vec.getType() for t in _DEVICE_VEC_TYPES):
        vec.destroy()
        raise RuntimeError(
            f"petsc4py created a {vec.getType()!r} vector for a CuPy array; it "
            "needs a PETSc built with CUDA or HIP support",
        )
    vec.setAttr("cunumpy_array", array)  # the vector does not own the memory
    return vec
