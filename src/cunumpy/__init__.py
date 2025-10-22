# cunumpy/__init__.py
from . import xp

__all__ = ["xp"]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    return getattr(xp.xp, name)
