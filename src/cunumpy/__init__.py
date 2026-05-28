# cunumpy/__init__.py
from . import xp
from .xp import to_cupy, to_numpy

__all__ = ["xp", "to_numpy", "to_cupy"]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    return getattr(xp.xp, name)
