"""SciPy linear algebra that ``cupyx.scipy.linalg`` lacks, on the array's backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy.xp import get_array_module


def _moveaxis(xpm: Any, a: Any, source: int, destination: int) -> Any:
    if source % a.ndim == destination % a.ndim:
        return a
    return xpm.moveaxis(a, source, destination)


def _as_inexact(xpm: Any, a: Any) -> Any:
    a = xpm.asarray(a)
    if a.ndim == 0:
        a = xpm.reshape(a, (1,))
    if not xpm.isdtype(a.dtype, ("real floating", "complex floating")):
        a = xpm.astype(a, xpm.float64)
    return a


def solve_circulant(
    c: Any,
    b: Any,
    singular: str = "raise",
    tol: float | None = None,
    caxis: int = -1,
    baxis: int = 0,
    outaxis: int = 0,
) -> Any:
    """Solve ``C x = b`` for a circulant matrix ``C`` with first column `c`.

    The steps and arguments of :func:`scipy.linalg.solve_circulant`, with FFTs
    on the backend of `c`: on the CuPy backend nothing leaves the device, and
    the singularity check synchronizes once. Available as
    ``xp.scipy.linalg.solve_circulant`` on both backends.

    Parameters
    ----------
    c : array
        First column of ``C``; a batch of columns along the other axes.
    b : array
        Right-hand side(s), of length ``len(c)`` along `baxis`.
    singular : {"raise", "lstsq"}, optional
        On a (near) singular ``C``: raise, or return the least-squares
        solution.
    tol : float, optional
        Eigenvalues of ``C`` with magnitude at most `tol` count as zero.
        Default ``max(abs(fft(c))) * len(c) * eps``.
    caxis, baxis, outaxis : int, optional
        The axis of `c`, `b` and the result along which the vectors lie.

    Returns
    -------
    array
        The solution ``x``, real unless `c` or `b` is complex.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``C`` is near singular and `singular` is ``"raise"``.
    ValueError
        If the lengths of `c` and `b` differ.

    Examples
    --------
    >>> c = xp.asarray([4.0, 1.0, 0.0, 1.0])
    >>> xp.scipy.linalg.solve_circulant(c, xp.asarray([6.0, 6.0, 6.0, 6.0]))
    array([1., 1., 1., 1.])
    """
    xpm = get_array_module(c)
    c = _as_inexact(xpm, c)
    b = _as_inexact(xpm, b)
    n = c.shape[caxis]
    if b.shape[baxis] != n:
        raise ValueError(f"Shapes of c {c.shape} and b {b.shape} are incompatible")
    fc = xpm.fft.fft(_moveaxis(xpm, c, caxis, -1), axis=-1)
    abs_fc = xpm.abs(fc)
    if tol is None:
        # the tolerance of numpy.linalg.matrix_rank, as in SciPy
        tol = xpm.max(abs_fc, axis=-1, keepdims=True) * n * np.finfo(np.float64).eps
    near_zeros = abs_fc <= tol
    is_near_singular = bool(xpm.any(near_zeros))
    if is_near_singular:
        if singular == "raise":
            raise np.linalg.LinAlgError("near singular circulant matrix.")
        fc = xpm.where(near_zeros, xpm.ones_like(fc), fc)
    fb = xpm.fft.fft(_moveaxis(xpm, b, baxis, -1), axis=-1)
    q = fb / fc
    if is_near_singular:
        q = xpm.where(near_zeros, xpm.zeros_like(q), q)
    x = xpm.fft.ifft(q, axis=-1)
    if not (
        xpm.isdtype(c.dtype, "complex floating")
        or xpm.isdtype(b.dtype, "complex floating")
    ):
        x = xpm.real(x)
    return _moveaxis(xpm, x, -1, outaxis)
