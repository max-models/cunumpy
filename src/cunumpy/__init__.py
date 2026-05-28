# cunumpy/__init__.py
from . import xp
from .xp import get_backend, is_gpu, to_cunumpy, to_cupy, to_numpy

__all__ = ["xp", "to_numpy", "to_cupy", "to_cunumpy", "get_backend", "is_gpu"]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    return getattr(xp.xp, name)
