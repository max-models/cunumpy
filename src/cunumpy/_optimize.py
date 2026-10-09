"""Batched root finding on either backend (see :mod:`cunumpy.optimize`)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

from cunumpy.xp import array_backend, get_array_backend, get_array_module


class NewtonResult(NamedTuple):
    """The result of :func:`newton` with ``full_output=True``.

    Attributes
    ----------
    root : array
        The estimated roots.
    converged : array of bool
        Where the last step was smaller than the tolerance (or ``func`` was
        exactly zero).
    zero_der : array of bool
        Where the iteration stopped on a zero derivative (Newton) or on equal
        function values (secant) without converging.
    """

    root: Any
    converged: Any
    zero_der: Any


def _module(x0: Any) -> Any:
    if get_array_backend(x0) == "cupy":
        return get_array_module(x0)
    return array_backend.xp


def newton(
    func: Callable[..., Any],
    x0: Any,
    fprime: Callable[..., Any] | None = None,
    args: tuple = (),
    tol: float = 1.48e-8,
    maxiter: int = 50,
    rtol: float = 0.0,
    full_output: bool = False,
) -> Any:
    """Find a root of each of many independent scalar equations at once.

    Newton's method if `fprime` is given, the secant method otherwise, on an
    array of starting points: ``func(x)`` returns ``f_i(x_i)`` for each entry.
    The steps of :func:`scipy.optimize.newton` with an array `x0`, but on the
    array's own backend (NumPy or CuPy) with fixed shapes (no boolean
    indexing), so every step runs on the device. Points that have converged
    keep taking (tiny) steps until all have, as in SciPy. Each iteration
    synchronizes once, to test for convergence.

    Parameters
    ----------
    func : callable
        ``func(x, *args)``: the residuals at `x`, an array of `x`'s shape.
    x0 : array_like
        The starting points. A NumPy or CuPy array stays on its backend; a
        scalar or list goes to the active backend.
    fprime : callable, optional
        ``fprime(x, *args)``: the derivatives ``f_i'(x_i)``. Without it the
        secant method is used.
    args : tuple, optional
        Extra arguments of `func` and `fprime`.
    tol : float, optional
        Absolute tolerance on the step.
    maxiter : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Relative tolerance on the step: a point has converged when
        ``abs(step) < tol + rtol * abs(x)``.
    full_output : bool, optional
        Return a :class:`NewtonResult` instead of only the roots.

    Returns
    -------
    array or NewtonResult
        The roots, of `x0`'s shape (a 0-d array for a scalar `x0`), or the
        :class:`NewtonResult` if `full_output` is true.

    Warns
    -----
    RuntimeWarning
        Without `full_output`, if some points did not converge.

    See Also
    --------
    scipy.optimize.newton : The SciPy version (host only).

    Examples
    --------
    >>> s = xp.asarray([2.0, 3.0, 4.0])
    >>> xp.optimize.newton(lambda x: x**2 - s, xp.ones(3), fprime=lambda x: 2 * x)
    array([1.41421356, 1.73205081, 2.        ])
    >>> xp.optimize.newton(lambda x: x**3 - s, xp.ones(3))
    array([1.25992105, 1.44224957, 1.58740105])
    """
    xpm = _module(x0)
    p = xpm.astype(xpm.asarray(x0), xpm.float64, copy=True)
    failures = xpm.ones(p.shape, dtype=bool)
    nz_der = xpm.ones(p.shape, dtype=bool)

    def tolerance(x: Any) -> Any:
        return tol + rtol * xpm.abs(x)

    if fprime is not None:
        for _ in range(maxiter):
            fval = func(p, *args)
            if not bool(xpm.any(fval != 0)):
                failures = fval != 0
                break
            fder = fprime(p, *args)
            nz_der = fder != 0
            if not bool(xpm.any(nz_der)):
                break
            dp = fval / xpm.where(nz_der, fder, 1.0)
            p = xpm.where(nz_der, p - dp, p)
            failures = xpm.where(nz_der, xpm.abs(dp) >= tolerance(p), failures)
            if not bool(xpm.any(failures & nz_der)):
                break
    else:
        dx = float(xpm.finfo(xpm.float64).eps) ** 0.33
        p1 = p * (1 + dx) + xpm.where(p >= 0, dx, -dx)
        q0 = func(p, *args)
        q1 = func(p1, *args)
        active = xpm.ones(p.shape, dtype=bool)
        for _ in range(maxiter):
            nz_der = q1 != q0
            if not bool(xpm.any(nz_der)):
                p = (p1 + p) / 2.0
                break
            dp = q1 * (p1 - p) / xpm.where(nz_der, q1 - q0, 1.0)
            new = xpm.where(nz_der, p1 - dp, p)
            new = xpm.where(~nz_der & active, (p1 + p) / 2.0, new)
            p = new
            active = active & nz_der
            failures = xpm.where(nz_der, xpm.abs(dp) >= tolerance(p), failures)
            if not bool(xpm.any(failures & nz_der)):
                break
            p1, p = p, p1
            q0 = q1
            q1 = func(p1, *args)
        else:
            p = p1  # the newest iterate, after the last swap

    zero_der = ~nz_der & failures
    if full_output:
        return NewtonResult(p, ~failures, zero_der)
    if bool(xpm.any(failures)):
        n_failed = int(xpm.sum(failures))
        reason = " (zero derivatives)" if bool(xpm.any(zero_der)) else ""
        warnings.warn(
            f"newton: {n_failed} of {p.size} points did not converge{reason}",
            RuntimeWarning,
            stacklevel=2,
        )
    return p
