# cunumpy/__init__.py
from . import xp
from .xp import (get_backend, is_cpu, is_gpu, set_backend, synchronize,
                 to_cunumpy, to_cupy, to_numpy, use_backend)

__all__ = [
    "xp",
    "to_numpy",
    "to_cupy",
    "to_cunumpy",
    "get_backend",
    "is_gpu",
    "is_cpu",
    "use_backend",
    "set_backend",
    "synchronize",
]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    return getattr(xp.xp, name)
