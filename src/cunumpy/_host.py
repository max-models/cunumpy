"""Run host-only code (SciPy, file readers, external libraries) with arguments of any backend."""

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
    """Call a host-only function with arguments of any backend.

    Device (CuPy) arrays among the arguments are copied to the host, `fun` runs
    on the NumPy backend, and its array results are copied back to the device,
    once per call. Without device arguments `fun` is called directly (on the
    NumPy backend this is a plain call), so the result lives where the
    arguments live. The copies are counted by `xp.profiling.count_transfers()`.

    Parameters
    ----------
    fun
        The host-only function (a SciPy spline, an external code, ...).
    *args, **kwargs
        Arguments of `fun`: arrays (or tuples/lists of arrays) of either
        backend, or scalars.

    Returns
    -------
    The result of `fun` (an array, a tuple/list of arrays or a scalar), with
    arrays on the device if any argument was on the device.

    Examples
    --------
    >>> values = xp.host_call(spline, x)  # x on the device -> values on the device
    """
    device = _on_device(args) or _on_device(list(kwargs.values()))
    if xp.get_backend() == "numpy" and not device:
        return fun(*args, **kwargs)
    with xp.use_backend("numpy"):
        out = fun(*_to_host(args), **{k: _to_host(v) for k, v in kwargs.items()})
    return _to_device(out) if device else out


def evaluate_on_host(method: Callable) -> Callable:
    """Decorator for methods that can only be evaluated on the host (see `host_call`).

    Examples
    --------
    >>> class Equilibrium:
    ...     @xp.evaluate_on_host
    ...     def pressure(self, x):
    ...         return scipy_spline(x)
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return host_call(method, self, *args, **kwargs)

    return wrapper


def setup_on_host(init: Callable) -> Callable:
    """Decorator for an ``__init__`` whose setup is host-only.

    `init` runs on the NumPy backend, so the object holds only host data (NumPy
    arrays, SciPy splines, floats) on either backend. Host-only parts of its
    evaluation can then go through `host_call` or `evaluate_on_host`.
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        with xp.use_backend("numpy"):
            init(self, *args, **kwargs)

    return wrapper
