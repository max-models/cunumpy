"""Counter-based random numbers (Philox4x32-10), the same on host and device.

The host side of ``cunumpy/random.cuh``: :func:`philox_uniform` returns, for
every (seed, stream, counter), exactly the doubles that ``cunumpy_uniform`` /
``cunumpy_uniform2`` return in a kernel, so a kernel that samples random
numbers can be compared with its host version element by element::

    ids = xp.arange(n, dtype=xp.uint64)              # one stream per particle
    u0, u1 = xp.rng.philox_uniform2(seed, ids, step)      # what each GPU thread draws
    z0, z1 = xp.rng.philox_normal2(seed, ids, step)

The numbers are a pure function of the key (``seed``, 64 bit) and the 128-bit
counter (``counter`` and ``stream``, 64 bit each): no state, independent of the
launch shape. Arguments broadcast like NumPy arrays; the result is a NumPy or a
CuPy array, matching the inputs (NumPy for Python ints). Uniform numbers are
bit-identical to the device ones; normal numbers (Box-Muller with ``log``,
``sqrt``, ``sin``, ``cos``) can differ in the last bits, since the device math
functions are not the host's.
"""

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np

__all__ = [
    "philox4x32_10",
    "philox_normal",
    "philox_normal2",
    "philox_uniform",
    "philox_uniform2",
]

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK32 = 0xFFFFFFFF


def _module(*values: Any) -> Any:
    """CuPy if any value is a CuPy array, else NumPy."""
    for value in values:
        if array_api_compat.is_cupy_array(value):
            import cupy

            return cupy
    return np


def philox4x32_10(counter: Any, key0: Any, key1: Any) -> Any:
    """Philox4x32-10 of a ``(..., 4)`` array of uint32 counter words and a 2x32-bit key.

    Parameters
    ----------
    counter : array of uint32, shape (..., 4)
        The counter words ``(c0, c1, c2, c3)``.
    key0, key1 : int or array of uint32
        The key words, broadcast against ``counter[..., 0]``.

    Returns
    -------
    array of uint32, shape (..., 4)
        The random words, as ``cunumpy_philox4x32_10`` computes them.
    """
    xp = _module(counter, key0, key1)
    ctr = xp.asarray(counter, dtype=xp.uint64)
    c0, c1, c2, c3 = (ctr[..., i] for i in range(4))
    k0 = xp.asarray(key0, dtype=xp.uint64) & _MASK32
    k1 = xp.asarray(key1, dtype=xp.uint64) & _MASK32
    m0, m1 = xp.uint64(_M0), xp.uint64(_M1)
    w0, w1 = xp.uint64(_W0), xp.uint64(_W1)
    mask, shift = xp.uint64(_MASK32), xp.uint64(32)
    for _ in range(10):
        p0 = m0 * c0  # < 2^64: exact in uint64
        p1 = m1 * c2
        c0, c1, c2, c3 = (
            (p1 >> shift) ^ c1 ^ k0,
            p1 & mask,
            (p0 >> shift) ^ c3 ^ k1,
            p0 & mask,
        )
        k0 = (k0 + w0) & mask
        k1 = (k1 + w1) & mask
    return xp.stack(xp.broadcast_arrays(c0, c1, c2, c3), axis=-1).astype(xp.uint32)


def _random_bits(seed: Any, stream: Any, counter: Any) -> Any:
    xp = _module(seed, stream, counter)
    seed = xp.asarray(seed, dtype=xp.uint64)
    stream = xp.asarray(stream, dtype=xp.uint64)
    counter = xp.asarray(counter, dtype=xp.uint64)
    mask, shift = xp.uint64(_MASK32), xp.uint64(32)
    seed, stream, counter = xp.broadcast_arrays(seed, stream, counter)
    words = xp.stack(
        [counter & mask, counter >> shift, stream & mask, stream >> shift],
        axis=-1,
    )
    return philox4x32_10(words, seed & mask, seed >> shift)


def _to_uniform(xp: Any, hi: Any, lo: Any) -> Any:
    bits = (hi.astype(xp.uint64) << xp.uint64(32)) | lo.astype(xp.uint64)
    return (bits >> xp.uint64(11)).astype(xp.float64) * (1.0 / 9007199254740992.0)


def philox_uniform2(seed: Any, stream: Any, counter: Any) -> tuple[Any, Any]:
    """Two uniform doubles in [0, 1) per (seed, stream, counter), as ``cunumpy_uniform2``.

    Parameters
    ----------
    seed : int or array
        The key (64 bit).
    stream : int or array
        The stream, e.g. a particle id (64 bit).
    counter : int or array
        The draw index, e.g. the time step (64 bit).

    Returns
    -------
    tuple of arrays
        ``(u0, u1)``, broadcast to the shape of the arguments.
    """
    xp = _module(seed, stream, counter)
    r = _random_bits(seed, stream, counter)
    return _to_uniform(xp, r[..., 0], r[..., 1]), _to_uniform(xp, r[..., 2], r[..., 3])


def philox_uniform(seed: Any, stream: Any, counter: Any) -> Any:
    """One uniform double in [0, 1) per (seed, stream, counter), as ``cunumpy_uniform``."""
    return philox_uniform2(seed, stream, counter)[0]


def philox_normal2(seed: Any, stream: Any, counter: Any) -> tuple[Any, Any]:
    """Two standard normal doubles per (seed, stream, counter), as ``cunumpy_normal2``.

    Box-Muller of :func:`philox_uniform2`; equal to the device numbers up to the
    last bits of the math functions.
    """
    xp = _module(seed, stream, counter)
    u0, u1 = philox_uniform2(seed, stream, counter)
    radius = xp.sqrt(-2.0 * xp.log(1.0 - u0))
    angle = 6.283185307179586 * u1
    return radius * xp.cos(angle), radius * xp.sin(angle)


def philox_normal(seed: Any, stream: Any, counter: Any) -> Any:
    """One standard normal double per (seed, stream, counter), as ``cunumpy_normal``."""
    return philox_normal2(seed, stream, counter)[0]
