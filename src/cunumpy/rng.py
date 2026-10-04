"""Random numbers on either backend.

:data:`random_streams` is one seeded generator per process and backend, with a
separate stream on every MPI rank; :func:`get_rng` returns a fresh NumPy or
CuPy ``Generator`` for the active backend; the ``philox_*`` functions return
the counter-based random numbers of ``cunumpy/random.cuh`` on the host, the
same as a CUDA kernel draws::

    import cunumpy as xp

    xp.rng.random_streams.seed(42, rank=comm.Get_rank())
    v = xp.rng.random_streams.normal(0.0, v_th, (n, 3))
    u0, u1 = xp.rng.philox_uniform2(seed, ids, step)

Named ``rng`` and not ``random`` so that ``xp.random`` stays NumPy's (CuPy's)
``random`` module.
"""

from cunumpy._philox import (
    philox4x32_10,
    philox_normal,
    philox_normal2,
    philox_uniform,
    philox_uniform2,
)
from cunumpy._random_streams import (
    BIT_GENERATORS,
    RandomStreams,
    get_rng,
    random_streams,
)

__all__ = [
    "BIT_GENERATORS",
    "RandomStreams",
    "get_rng",
    "philox4x32_10",
    "philox_normal",
    "philox_normal2",
    "philox_uniform",
    "philox_uniform2",
    "random_streams",
]
