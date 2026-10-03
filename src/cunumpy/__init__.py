# cunumpy/__init__.py
import re as _re
import warnings as _warnings
from importlib.metadata import PackageNotFoundError, version

from . import (
    algorithms,
    cuda,
    kernels,
    memory,
    mpi,
    petsc,
    profiling,
    rng,
    xp,
)
from .scipy_backend import scipy
from .xp import (
    as_device_array,
    assert_same_backend,
    cupy_available,
    default_float_dtype,
    get_array_backend,
    get_array_module,
    get_backend,
    is_cpu,
    is_gpu,
    same_backend,
    set_backend,
    synchronize,
    to_cunumpy,
    to_cupy,
    to_numpy,
    use_backend,
)

# Names that were at the top level before cunumpy 0.5, and the submodule each
# moved to. They still resolve (with a DeprecationWarning) until cunumpy 0.6.
_MOVED = {
    **dict.fromkeys(cuda.__all__, "cuda"),
    **dict.fromkeys(kernels.__all__, "kernels"),
    **dict.fromkeys(rng.__all__, "rng"),
    **dict.fromkeys(algorithms.__all__, "algorithms"),
    **dict.fromkeys(mpi.__all__, "mpi"),
    **dict.fromkeys(profiling.__all__, "profiling"),
    **dict.fromkeys(memory.__all__, "memory"),
    "petsc_vec": "petsc",
}
_MOVED.pop("BIT_GENERATORS")  # never was at the top level

# Importing cunumpy.rng loads the module cunumpy.random_streams, which would
# hide the deprecated top-level name random_streams (the generator).
globals().pop("random_streams", None)

try:
    __version__ = version("cunumpy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _re.findall(r"\d+", text.split("+")[0])[:3])


def require_version(minimum: str) -> None:
    """Raise ``ImportError`` if this cunumpy is older than `minimum`.

    For projects that depend on a feature of a given release, as a clearer
    error than an ``AttributeError`` later::

        import cunumpy as xp

        xp.require_version("0.4.0")

    Only the numeric part of the versions is compared (``0.4.0`` and
    ``0.4.0.dev1`` compare equal). Nothing is checked when the installed
    version is unknown (cunumpy not installed as a package).
    """
    if __version__.startswith("0.0.0+unknown"):
        return
    if _version_key(__version__) < _version_key(minimum):
        raise ImportError(
            f"cunumpy {minimum} or newer is required, but {__version__} is "
            "installed: pip install --upgrade cunumpy"
        )


__all__ = [
    "__version__",
    "algorithms",
    "as_device_array",
    "assert_same_backend",
    "cuda",
    "cupy_available",
    "cupy_backend",
    "default_float_dtype",
    "get_array_backend",
    "get_array_module",
    "get_backend",
    "is_cpu",
    "is_gpu",
    "kernels",
    "memory",
    "mpi",
    "numpy_backend",
    "petsc",
    "profiling",
    "require_version",
    "rng",
    "same_backend",
    "scipy",
    "set_backend",
    "synchronize",
    "to_cunumpy",
    "to_cupy",
    "to_numpy",
    "use_backend",
    "xp",
]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy).

    Names moved to a submodule in cunumpy 0.5 (see ``_MOVED``) still resolve,
    with a ``DeprecationWarning``.
    """
    if name == "numpy_backend":
        return xp.numpy_backend
    if name == "cupy_backend":
        return xp.cupy_backend
    submodule = _MOVED.get(name)
    if submodule is not None:
        _warnings.warn(
            f"cunumpy.{name} moved to cunumpy.{submodule}.{name}; the top-level "
            "name is deprecated and will be removed in cunumpy 0.6",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(globals()[submodule], name)
    return getattr(xp.xp, name)
