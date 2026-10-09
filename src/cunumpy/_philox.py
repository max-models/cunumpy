"""Counter-based random numbers (Philox4x32-10), the same on host and device.

The host side of ``cunumpy/random.cuh``: for every (seed, stream, counter) the
``philox_*`` functions return the numbers a kernel draws, so a kernel can be
compared with its host version element by element. The numbers are a pure
function of the 64-bit key (``seed``) and the 128-bit counter (``counter``,
``stream``): no state, independent of the launch shape. Arguments broadcast;
the result is a NumPy or CuPy array matching the inputs (NumPy for Python
ints). Uniform numbers are bit-identical to the device ones; normal numbers
can differ in the last bits, since the device math functions are not the
host's.
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
    """Return Philox4x32-10 of uint32 counter words and a 2x32-bit key.

    The raw generator, as ``cunumpy_philox4x32_10`` in ``cunumpy/random.cuh``
    computes it; it passes the Random123 known-answer tests.

    Parameters
    ----------
    counter : array of uint32, shape (..., 4)
        The counter words ``(c0, c1, c2, c3)``.
    key0, key1 : int or array of uint32
        The key words, broadcast against ``counter[..., 0]``.

    Returns
    -------
    array of uint32, shape (..., 4)
        The random words, on the backend of the inputs.

    See Also
    --------
    philox_uniform2 : Uniform doubles from (seed, stream, counter).

    Examples
    --------
    >>> xp.rng.philox4x32_10(np.zeros(4, dtype=np.uint32), 0, 0)
    array([1713891541, 3781805453, 3159862348, 2600524760], dtype=uint32)
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
    """Return two uniform doubles in [0, 1) per (seed, stream, counter).

    Bit-identical to ``cunumpy_uniform2`` in a kernel (53 random bits each).
    Use a different `counter` for every random decision of a step.

    Parameters
    ----------
    seed : int or array of uint64
        The key.
    stream : int or array of uint64
        The stream, e.g. a particle id.
    counter : int or array of uint64
        The draw index, e.g. the time step.

    Returns
    -------
    u0 : array of float64
        The first number, broadcast to the shape of the arguments, on their
        backend.
    u1 : array of float64
        The second number.

    See Also
    --------
    philox_uniform : The first of the two numbers.
    philox_normal2 : Normal numbers from the same draw.

    Examples
    --------
    >>> ids = xp.arange(3, dtype=xp.uint64)  # one stream per particle
    >>> u0, u1 = xp.rng.philox_uniform2(42, ids, 0)
    >>> u0
    array([0.61295988, 0.01005884, 0.3984137 ])
    """
    xp = _module(seed, stream, counter)
    r = _random_bits(seed, stream, counter)
    return _to_uniform(xp, r[..., 0], r[..., 1]), _to_uniform(xp, r[..., 2], r[..., 3])


def philox_uniform(seed: Any, stream: Any, counter: Any) -> Any:
    """Return one uniform double in [0, 1) per (seed, stream, counter).

    Bit-identical to ``cunumpy_uniform`` in a kernel: the first number of
    :func:`philox_uniform2`.

    Parameters
    ----------
    seed : int or array of uint64
        The key.
    stream : int or array of uint64
        The stream, e.g. a particle id.
    counter : int or array of uint64
        The draw index, e.g. the time step.

    Returns
    -------
    array of float64
        Broadcast to the shape of the arguments, on their backend.

    Examples
    --------
    >>> xp.rng.philox_uniform(42, xp.arange(3, dtype=xp.uint64), 0)
    array([0.61295988, 0.01005884, 0.3984137 ])
    """
    return philox_uniform2(seed, stream, counter)[0]


def philox_normal2(seed: Any, stream: Any, counter: Any) -> tuple[Any, Any]:
    """Return two standard normal doubles per (seed, stream, counter).

    Box-Muller of :func:`philox_uniform2`, as ``cunumpy_normal2`` in a kernel;
    equal to the device numbers up to the last bits of ``log``, ``sqrt``,
    ``sin`` and ``cos``.

    Parameters
    ----------
    seed : int or array of uint64
        The key.
    stream : int or array of uint64
        The stream, e.g. a particle id.
    counter : int or array of uint64
        The draw index, e.g. the time step.

    Returns
    -------
    z0 : array of float64
        The first number, broadcast to the shape of the arguments, on their
        backend.
    z1 : array of float64
        The second number.

    Examples
    --------
    >>> z0, z1 = xp.rng.philox_normal2(42, xp.arange(1000, dtype=xp.uint64), 0)
    >>> z0.shape, z0.dtype
    ((1000,), dtype('float64'))
    """
    xp = _module(seed, stream, counter)
    u0, u1 = philox_uniform2(seed, stream, counter)
    radius = xp.sqrt(-2.0 * xp.log(1.0 - u0))
    angle = 6.283185307179586 * u1
    return radius * xp.cos(angle), radius * xp.sin(angle)


def philox_normal(seed: Any, stream: Any, counter: Any) -> Any:
    """Return one standard normal double per (seed, stream, counter).

    As ``cunumpy_normal`` in a kernel: the first number of
    :func:`philox_normal2`.

    Parameters
    ----------
    seed : int or array of uint64
        The key.
    stream : int or array of uint64
        The stream, e.g. a particle id.
    counter : int or array of uint64
        The draw index, e.g. the time step.

    Returns
    -------
    array of float64
        Broadcast to the shape of the arguments, on their backend.

    Examples
    --------
    >>> xp.rng.philox_normal(42, xp.arange(4, dtype=xp.uint64), 0).shape
    (4,)
    """
    return philox_normal2(seed, stream, counter)[0]
