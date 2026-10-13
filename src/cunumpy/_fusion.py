"""Elementwise functions as one GPU kernel, plain calls on the host.

On the GPU a chain of elementwise operations runs one kernel per operation,
each writing a temporary array. :func:`~cunumpy.kernels.fuse` compiles the
chain into one kernel with ``cupy.fuse`` when called with CuPy arrays and
calls the function as it is otherwise, so the same code runs on both backends.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import array_api_compat
import numpy

from cunumpy.xp import use_backend

__all__ = ["fuse"]

F = TypeVar("F", bound=Callable[..., Any])

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array


def _cupy_fuse(function: Callable[..., Any], kernel_name: str | None) -> Any:
    import cupy

    return cupy.fuse(kernel_name=kernel_name)(function)


def _typed_scalars(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Cast Python scalars to the dtype they promote to with the array arguments."""
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

    Usable as ``@xp.kernels.fuse`` or ``@xp.kernels.fuse(kernel_name=...)``.
    The function must be elementwise in the sense of ``cupy.fuse``:
    arithmetic, comparisons, ufuncs, ``xp.where`` and supported reductions as
    the last operation; no Python control flow on array values, no indexing.
    The CuPy backend is active while it is traced, so ``xp.exp`` etc. resolve
    to CuPy ufuncs. A function ``cupy.fuse`` cannot trace raises at its first
    call with CuPy arrays: test the CuPy path.

    Parameters
    ----------
    function : callable, optional
        The elementwise function; omitted when used as ``fuse(kernel_name=...)``.
    kernel_name : str, optional
        Name of the kernel in profilers; by default the function's name.

    Returns
    -------
    callable
        A function with the same signature. If any argument is a CuPy array
        it calls the fused kernel (created on first call and reused), with
        Python scalars cast to the dtype they promote to with the arrays;
        otherwise it calls `function` itself. Without `function`, a decorator.

    Examples
    --------
    >>> @xp.kernels.fuse
    ... def pressure(rho, T, gamma):
    ...     return (gamma - 1.0) * rho * T
    >>> pressure(np.array([1.0, 2.0]), np.array([3.0, 3.0]), 2.0)
    array([3., 6.])
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
