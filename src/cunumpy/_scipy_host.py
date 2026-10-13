"""SciPy solvers that ``cupyx.scipy`` lacks, run on the host with callbacks on the device."""

from __future__ import annotations

import copy
import functools
import importlib
from collections.abc import Callable
from typing import Any

from cunumpy import xp
from cunumpy._host import _to_device


def _scipy(name: str, function: str) -> Any:
    try:
        return importlib.import_module(f"scipy.{name}")
    except ImportError as error:
        raise ImportError(
            f"cunumpy.{name}.{function} needs SciPy (pip install scipy)",
        ) from error


def _host(value: Any) -> Any:
    """Copy device arrays in `value` (or a tuple/list of them) to the host."""
    if isinstance(value, (tuple, list)):
        return type(value)(_host(v) for v in value)
    return xp.to_numpy(value) if xp.is_gpu(value) else value


def _device(value: Any) -> Any:
    """Copy NumPy arrays in `value` (a tuple, list or dict of them) to the device."""
    if isinstance(value, dict):  # OptimizeResult, infodict: a copy of the same type
        out = copy.copy(value)
        for key, item in value.items():
            out[key] = _device(item)
        return out
    if isinstance(value, (tuple, list)):
        return type(value)(_device(v) for v in value)
    return _to_device(value)


class _Bridge:
    """Callbacks get arrays on the backend of `x0`; SciPy sees host arrays."""

    def __init__(self, x0: Any) -> None:
        self.device = xp.is_gpu(x0)

    def callback(self, fun: Any, n_arrays: int = 1) -> Any:
        # the first `n_arrays` positional and all keyword arguments of `fun`
        # arrive on the device
        if not (self.device and callable(fun)):
            return fun

        # wraps: SciPy reads the signature (a callback named `intermediate_result`)
        @functools.wraps(fun)
        def call(*args: Any, **kwargs: Any) -> Any:
            args = (*_device(list(args[:n_arrays])), *args[n_arrays:])
            kwargs = {k: _device(v) for k, v in kwargs.items()}  # intermediate_result
            return _host(fun(*args, **kwargs))

        return call

    def result(self, value: Any) -> Any:
        return _device(value) if self.device else value

    def constraints(self, constraints: Any) -> Any:
        if isinstance(constraints, (tuple, list)):
            return type(constraints)(self.constraints(c) for c in constraints)
        if isinstance(constraints, dict):  # {"type": "ineq", "fun": ..., "jac": ...}
            out = dict(constraints)
            for key in ("fun", "jac"):
                out[key] = self.callback(out.get(key))
            return {k: v for k, v in out.items() if v is not None}
        out = copy.copy(constraints)  # NonlinearConstraint, LinearConstraint
        for name, n_arrays in (("fun", 1), ("jac", 1), ("hess", 2)):
            if callable(getattr(out, name, None)):
                setattr(out, name, self.callback(getattr(out, name), n_arrays))
        return out


def fsolve(
    func: Callable[..., Any],
    x0: Any,
    args: tuple = (),
    fprime: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Find a root of a system of equations with :func:`scipy.optimize.fsolve`.

    SciPy (MINPACK) runs on the host. With `x0` on the device, `func` and
    `fprime` get ``x`` on the device, their results are copied to the host,
    and the root comes back on the device: two transfers per evaluation, so
    use it for setup, not in a time loop.

    Parameters
    ----------
    func : callable
        ``func(x, *args)``, the residual, an array of the shape of `x0`.
    x0 : array
        Initial guess; its backend is the backend of the callbacks and result.
    args : tuple, optional
        Extra arguments of `func` and `fprime`, passed unchanged.
    fprime : callable, optional
        ``fprime(x, *args)``, the Jacobian.
    **kwargs
        The other arguments of :func:`scipy.optimize.fsolve`.

    Returns
    -------
    array or tuple
        The root, on the backend of `x0`; with ``full_output=True`` SciPy's
        tuple, with its arrays on that backend.

    See Also
    --------
    root : The same with a choice of method.
    newton : Many independent scalar equations at once, on the device.

    Examples
    --------
    >>> xp.optimize.fsolve(lambda x: x**2 - xp.asarray([4.0, 9.0]), xp.ones(2))
    array([2., 3.])
    """
    optimize = _scipy("optimize", "fsolve")
    bridge = _Bridge(x0)
    out = optimize.fsolve(
        bridge.callback(func),
        _host(x0),
        args=args,
        fprime=bridge.callback(fprime),
        **kwargs,
    )
    return bridge.result(out)


def root(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple = (),
    jac: Any = None,
    callback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Find a root of a system of equations with :func:`scipy.optimize.root`.

    SciPy runs on the host; with `x0` on the device, `fun`, `jac` and
    `callback` get ``x`` on the device and the arrays of the result are
    copied back to the device. See :func:`fsolve`.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args)``, the residual (or ``(residual, jacobian)`` with
        ``jac=True``).
    x0 : array
        Initial guess; its backend is the backend of the callbacks and result.
    args : tuple, optional
        Extra arguments of `fun` and `jac`, passed unchanged.
    jac : callable or bool, optional
        ``jac(x, *args)``, the Jacobian; ``True`` if `fun` returns it too.
    callback : callable, optional
        ``callback(x, f)``, called after each iteration (some methods only).
    **kwargs
        The other arguments of :func:`scipy.optimize.root` (``method``,
        ``tol``, ``options``).

    Returns
    -------
    OptimizeResult
        SciPy's result; ``x``, ``fun`` and the other arrays on the backend of
        `x0`.

    Examples
    --------
    >>> res = xp.optimize.root(lambda x: x**3 - 8.0, xp.ones(1))
    >>> res.success, res.x
    (True, array([2.]))
    """
    optimize = _scipy("optimize", "root")
    bridge = _Bridge(x0)
    out = optimize.root(
        bridge.callback(fun),
        _host(x0),
        args=args,
        jac=bridge.callback(jac),
        callback=bridge.callback(callback, 2),
        **kwargs,
    )
    return bridge.result(out)


def minimize(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple = (),
    jac: Any = None,
    hess: Any = None,
    hessp: Callable[..., Any] | None = None,
    bounds: Any = None,
    constraints: Any = (),
    callback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Minimize a scalar function with :func:`scipy.optimize.minimize`.

    SciPy runs on the host; with `x0` on the device, every callback (`fun`,
    `jac`, `hess`, `hessp`, `callback` and the functions of `constraints`)
    gets its arrays on the device, and the arrays of the result are copied
    back to the device. See :func:`fsolve`.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args)``, a scalar (or ``(value, gradient)`` with
        ``jac=True``).
    x0 : array
        Initial guess; its backend is the backend of the callbacks and result.
    args : tuple, optional
        Extra arguments of the callbacks, passed unchanged.
    jac, hess : callable, str or bool, optional
        Gradient and Hessian, as in SciPy (a callable, a finite-difference
        scheme such as ``"2-point"``, or a ``HessianUpdateStrategy``).
    hessp : callable, optional
        ``hessp(x, p, *args)``, the Hessian times ``p``.
    bounds : sequence or Bounds, optional
        Bounds on ``x``, as in SciPy (host data).
    constraints : dict, constraint object or sequence of them, optional
        As in SciPy; their functions get device arrays like `fun`, their
        bounds and matrices are host data.
    callback : callable, optional
        Called after each iteration, as in SciPy (``callback(intermediate_result)``
        or ``callback(xk)``).
    **kwargs
        The other arguments of :func:`scipy.optimize.minimize` (``method``,
        ``tol``, ``options``).

    Returns
    -------
    OptimizeResult
        SciPy's result; ``x``, ``jac`` and the other arrays on the backend of
        `x0`.

    Examples
    --------
    >>> res = xp.optimize.minimize(lambda x: xp.sum((x - 1.0) ** 2), xp.zeros(2))
    >>> bool(xp.allclose(res.x, 1.0, atol=1e-6))
    True
    """
    optimize = _scipy("optimize", "minimize")
    bridge = _Bridge(x0)
    if bridge.device:
        constraints = bridge.constraints(constraints)
    out = optimize.minimize(
        bridge.callback(fun),
        _host(x0),
        args=args,
        jac=bridge.callback(jac),
        hess=bridge.callback(hess),
        hessp=bridge.callback(hessp, 2),
        bounds=bounds,
        constraints=constraints,
        callback=bridge.callback(callback, 2),  # trust-constr: (xk, state)
        **kwargs,
    )
    return bridge.result(out)


def quad(
    func: Callable[..., Any],
    a: float,
    b: float,
    args: tuple = (),
    **kwargs: Any,
) -> Any:
    """Integrate a scalar function with :func:`scipy.integrate.quad`.

    SciPy (QUADPACK) runs on the host and calls `func` with a Python float.
    `func` may compute on the device and return a 0-d device array: it is
    copied to the host, one synchronization per evaluation. For many
    integrals at once, evaluate the integrand on a fixed grid instead.

    Parameters
    ----------
    func : callable
        ``func(x, *args)``, a scalar.
    a, b : float
        Integration limits (``numpy.inf`` allowed).
    args : tuple, optional
        Extra arguments of `func`, passed unchanged.
    **kwargs
        The other arguments of :func:`scipy.integrate.quad`.

    Returns
    -------
    tuple
        SciPy's result, ``(value, abserr)`` plus the outputs that
        ``full_output`` adds, on the host.

    Examples
    --------
    >>> value, error = xp.integrate.quad(lambda x: x**2, 0.0, 3.0)
    >>> round(value, 12)
    9.0
    """
    integrate = _scipy("integrate", "quad")
    return integrate.quad(
        lambda x, *a: _host(func(x, *a)),
        _host(a),
        _host(b),
        args=args,
        **kwargs,
    )


def odeint(
    func: Callable[..., Any],
    y0: Any,
    t: Any,
    args: tuple = (),
    Dfun: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Integrate a system of ODEs with :func:`scipy.integrate.odeint`.

    SciPy (LSODA) runs on the host; with `y0` on the device, `func` and
    `Dfun` get ``y`` on the device, and the solution comes back on the
    device. See :func:`cunumpy.optimize.fsolve` for the cost.

    Parameters
    ----------
    func : callable
        ``func(y, t, *args)`` (``func(t, y, *args)`` with ``tfirst=True``),
        the derivative of ``y``.
    y0 : array
        Initial condition; its backend is the backend of the callbacks and
        result.
    t : array
        Times at which to return ``y``, starting with the time of `y0`; either
        backend.
    args : tuple, optional
        Extra arguments of `func` and `Dfun`, passed unchanged.
    Dfun : callable, optional
        The Jacobian of `func`, with the same arguments.
    **kwargs
        The other arguments of :func:`scipy.integrate.odeint`.

    Returns
    -------
    array or tuple
        ``y`` at the times `t`, shape ``(len(t), len(y0))``, on the backend of
        `y0`; with ``full_output=True`` also SciPy's info dict.

    Examples
    --------
    >>> y = xp.integrate.odeint(lambda y, t: -y, xp.ones(1), xp.asarray([0.0, 1.0]))
    >>> bool(xp.allclose(y[-1], np.exp(-1.0), rtol=1e-6))
    True
    """
    integrate = _scipy("integrate", "odeint")
    bridge = _Bridge(y0)
    out = integrate.odeint(
        bridge.callback(func, 2),
        _host(y0),
        _host(t),
        args=args,
        Dfun=bridge.callback(Dfun, 2),
        **kwargs,
    )
    return bridge.result(out)
