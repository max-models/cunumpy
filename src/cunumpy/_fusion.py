"""``xp.kernels.fuse``: elementwise functions as one GPU kernel, plain calls on the host.

A chain of elementwise operations (a pressure from density and temperature,
fluxes and limiters of a fluid update, a Maxwellian at many velocities) runs
as one kernel per operation on the GPU, each writing a temporary array: it is
limited by memory bandwidth. ``cupy.fuse`` compiles the whole chain into one
kernel. :func:`fuse` applies it when the function is called with CuPy arrays
and calls the function as it is otherwise, so the same code runs on both
backends::

    @xp.kernels.fuse
    def pressure(rho, T, gamma):
        return (gamma - 1.0) * rho * T

    p = pressure(rho, T, 5.0 / 3.0)

The function must be elementwise: arithmetic, comparisons, ufuncs such as
``xp.exp``, ``xp.sqrt``, ``xp.where`` and reductions that ``cupy.fuse``
supports (``xp.sum`` as the last operation). On the CuPy path the CuPy
backend is active while the function is traced, so ``xp.*`` names resolve to
CuPy's ufuncs. Functions that ``cupy.fuse`` cannot trace (Python control
flow on array values, indexing, wrappers that are not ufuncs) raise when the
fused function is first called with CuPy arrays; test the CuPy path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import array_api_compat
import numpy

from .xp import use_backend

__all__ = ["fuse"]

F = TypeVar("F", bound=Callable[..., Any])

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array


def _cupy_fuse(function: Callable[..., Any], kernel_name: str | None) -> Any:
    import cupy

    return cupy.fuse(kernel_name=kernel_name)(function)


def _typed_scalars(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Give Python scalars the dtype they would take next to the arrays.

    ``cupy.fuse`` types a Python scalar on its own: ``gamma - 1.0`` with
    ``gamma=5/3`` runs in float16. Eagerly, NumPy 2 and CuPy promote it with
    the arrays (float64 arrays: float64), so cast it to that dtype first.
    """
    dtypes = [a.dtype for a in (*args, *kwargs.values()) if hasattr(a, "dtype")]

    def typed(a: Any) -> Any:
        if isinstance(a, (int, float, complex)) and not isinstance(a, bool):
            return numpy.result_type(*dtypes, a).type(a)
        return a

    return (
        tuple(typed(a) for a in args),
        {k: typed(v) for k, v in kwargs.items()},
    )


def fuse(
    function: F | None = None,
    *,
    kernel_name: str | None = None,
) -> F | Callable[[F], F]:
    """Fuse an elementwise function into one kernel when called with CuPy arrays.

    Usable as ``@xp.kernels.fuse`` or ``@xp.kernels.fuse(kernel_name="pressure")``.

    Parameters
    ----------
    function : callable
        The elementwise function.
    kernel_name : str | None
        Name of the generated kernel (shown by profilers); by default the
        function's name.

    Returns
    -------
    callable
        A function with the same signature. If any positional or keyword
        argument is a CuPy array, it calls ``cupy.fuse(function)`` (created on
        first use, with the CuPy backend active) with Python scalars cast to
        the dtype they promote to with the array arguments; otherwise it calls
        `function` itself.
    """
    if function is None:
        return lambda f: fuse(f, kernel_name=kernel_name)

    name = kernel_name if kernel_name is not None else function.__name__
    fused = None

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal fused
        if not any(_is_device_array(a) for a in (*args, *kwargs.values())):
            return function(*args, **kwargs)
        args, kwargs = _typed_scalars(args, kwargs)
        with use_backend("cupy"):
            if fused is None:
                fused = _cupy_fuse(function, name)
            return fused(*args, **kwargs)

    wrapper.__wrapped__ = function
    return wrapper
