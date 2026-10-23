"""Shared implementation for optional framework adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from cunumpy._transfers import _ACTIVE, _describe, _nbytes, _record


def require(module: str, *, extra: str | None = None) -> Any:
    """Import an optional framework and give a consistent installation error."""
    try:
        return import_module(module)
    except ImportError as error:
        extra = module.split(".", 1)[0] if extra is None else extra
        raise ImportError(
            f"{module} interoperability needs {module.split('.', 1)[0]!r}; "
            f"install cunumpy[{extra}]"
        ) from error


def record(source: Any, target: Any) -> None:
    """Record an adapter conversion when transfer profiling is active."""
    if _ACTIVE:
        _record(
            "device_copy",
            f"interop {type(source).__name__} -> {type(target).__name__} "
            f"({_describe(target)})",
            nbytes=_nbytes(target),
        )


def target_backend(backend: str) -> Any:
    """Load a NumPy or CuPy target module."""
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy
        except ImportError as error:
            raise ImportError(
                "CuPy interoperability needs CuPy; install a compatible CuPy release"
            ) from error
        return cupy
    raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")


def cast(array: Any, dtype: Any) -> Any:
    """Cast an array only when a dtype was requested."""
    return array if dtype is None else array.astype(dtype)
