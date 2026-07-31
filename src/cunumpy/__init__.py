# cunumpy/__init__.py
from importlib.metadata import PackageNotFoundError, version

from . import xp
from .xp import (
    get_backend,
    is_cpu,
    is_gpu,
    set_backend,
    synchronize,
    to_cunumpy,
    to_cupy,
    to_numpy,
    use_backend,
)

try:
    __version__ = version("cunumpy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "xp",
    "__version__",
    "to_numpy",
    "to_cupy",
    "to_cunumpy",
    "get_backend",
    "is_gpu",
    "is_cpu",
    "use_backend",
    "set_backend",
    "synchronize",
    "numpy_backend",
    "cupy_backend",
]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    if name == "numpy_backend":
        return xp.numpy_backend
    if name == "cupy_backend":
        return xp.cupy_backend
    return getattr(xp.xp, name)
