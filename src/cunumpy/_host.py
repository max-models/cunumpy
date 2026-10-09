"""Run host-only code (SciPy, file readers, external libraries) with arguments of either backend."""

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from cunumpy import xp


def _on_device(arg: Any) -> bool:
    """Whether ``arg`` is (or contains, for tuples and lists) a device array."""
    if isinstance(arg, (tuple, list)):
        return any(_on_device(a) for a in arg)
    return xp.is_gpu(arg)


def _to_host(arg: Any) -> Any:
    """Copy device arrays in ``arg`` (an array, or a tuple/list of them) to the host."""
    if isinstance(arg, (tuple, list)):
        return type(arg)(_to_host(a) for a in arg)
    return xp.to_numpy(arg) if xp.is_gpu(arg) else arg


def _to_device(arg: Any) -> Any:
    """Copy NumPy arrays in ``arg`` (an array, or a tuple/list of them) to the device."""
    if isinstance(arg, (tuple, list)):
        return type(arg)(_to_device(a) for a in arg)
    return xp.to_cupy(arg) if isinstance(arg, np.ndarray) else arg


def host_call(fun: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a host-only function with arguments of either backend.

    Device (CuPy) arrays among the arguments are copied to the host, `fun` runs
    on the NumPy backend, and its array results are copied back to the device,
    once per call (counted by :func:`~cunumpy.profiling.count_transfers`).
    Without device arguments `fun` is called directly, so the result lives
    where the arguments live. See :doc:`/guides/data-movement`.

    Parameters
    ----------
    fun : callable
        The host-only function (a SciPy spline, an external code, ...).
    *args, **kwargs
        Arguments of `fun`: arrays (or tuples/lists of arrays) of either
        backend, or scalars.

    Returns
    -------
    object
        The result of `fun` (an array, a tuple/list of arrays or a scalar),
        with its arrays on the device if any argument was on the device.

    See Also
    --------
    evaluate_on_host : The same for a method.

    Examples
    --------
    >>> xp.host_call(np.cumsum, xp.asarray([1, 2, 3]))
    array([1, 3, 6])
    >>> values = xp.host_call(spline, x_device)  # doctest: +SKIP
    """
    device = _on_device(args) or _on_device(list(kwargs.values()))
    if xp.get_backend() == "numpy" and not device:
        return fun(*args, **kwargs)
    with xp.use_backend("numpy"):
        out = fun(*_to_host(args), **{k: _to_host(v) for k, v in kwargs.items()})
    return _to_device(out) if device else out


def evaluate_on_host(method: Callable) -> Callable:
    """Make a host-only method callable with arguments of either backend.

    Each call goes through :func:`host_call`: device arrays are copied to the
    host, the method runs on the NumPy backend, and array results are copied
    back if any argument was on the device.

    Parameters
    ----------
    method : callable
        The method, ``method(self, *args, **kwargs)``.

    Returns
    -------
    callable
        The wrapped method.

    Examples
    --------
    >>> class Profile:
    ...     @xp.evaluate_on_host
    ...     def cumulative(self, x):
    ...         return np.cumsum(x)
    >>> Profile().cumulative(xp.asarray([1, 2, 3]))
    array([1, 3, 6])
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return host_call(method, self, *args, **kwargs)

    return wrapper


def setup_on_host(init: Callable) -> Callable:
    """Run an ``__init__`` on the NumPy backend.

    The object then holds only host data (NumPy arrays, SciPy splines, floats)
    on either backend; host-only parts of its evaluation can go through
    :func:`host_call` or :func:`evaluate_on_host`.

    Parameters
    ----------
    init : callable
        The ``__init__`` method.

    Returns
    -------
    callable
        The wrapped ``__init__``.

    Examples
    --------
    >>> class Grid:
    ...     @xp.setup_on_host
    ...     def __init__(self, n):
    ...         self.x = xp.linspace(0.0, 1.0, n)
    >>> xp.get_array_backend(Grid(3).x)
    'numpy'
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        with xp.use_backend("numpy"):
            init(self, *args, **kwargs)

    return wrapper
