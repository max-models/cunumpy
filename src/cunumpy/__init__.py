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
from ._scipy_backend import scipy
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
    # the names of cunumpy.mpi that were at the top level (not the later ones)
    **dict.fromkeys(
        (
            "get_mpi_cuda_aware",
            "local_rank",
            "mpi_buffer",
            "mpi_is_cuda_aware",
            "require_cuda_aware_mpi",
            "set_mpi_cuda_aware",
            "synchronize_for_mpi",
        ),
        "mpi",
    ),
    **dict.fromkeys(profiling.__all__, "profiling"),
    **dict.fromkeys(memory.__all__, "memory"),
    "petsc_vec": "petsc",
}
_MOVED.pop("BIT_GENERATORS")  # never was at the top level

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
    """Resolve names that are not in the namespace: cunumpy.<name> -> cunumpy.xp.<name>.

    The public names of the active backend are copied into this namespace (see
    `_sync_backend_namespace`), so this only runs for names missing from the
    backend's ``__all__``, for ``numpy_backend``/``cupy_backend``, and for the
    names moved to a submodule in cunumpy 0.5 (see ``_MOVED``), which still
    resolve with a ``DeprecationWarning``.
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


# `xp.zeros` must be as fast as `numpy.zeros`. A module-level __getattr__ runs
# only after the normal lookup failed, which costs about 3 us per access, so the
# public names of the active backend module are copied into this namespace, and
# replaced whenever the backend changes. cunumpy's own names and the deprecated
# names of _MOVED (e.g. `fuse`, which CuPy also has) are never overwritten.
_OWN_NAMES = frozenset(globals())
_backend_names: dict[int, dict[str, object]] = {}  # id(module) -> names to copy
_switches: dict[tuple[int, int], tuple[tuple[str, ...], dict[str, object]]] = {}
_synced_module: list[int] = [0]  # id of the module whose names are in the namespace


def _names_of(module) -> dict[str, object]:
    names = _backend_names.get(id(module))
    if names is None:
        names = {
            name: getattr(module, name)
            for name in getattr(module, "__all__", ())
            if not name.startswith("_")
            and name not in _OWN_NAMES
            and name not in _MOVED
            and hasattr(module, name)
        }
        _backend_names[id(module)] = names
    return names


def _sync_backend_namespace(module) -> None:
    new = _names_of(module)
    namespace = globals()
    old_id = _synced_module[0]
    if old_id not in _backend_names:
        namespace.update(new)
    else:
        # per pair of modules: the names to drop and the names whose value
        # changes (NumPy and CuPy share dtypes, constants, ...)
        key = (old_id, id(module))
        switch = _switches.get(key)
        if switch is None:
            old = _backend_names[old_id]
            switch = _switches[key] = (
                tuple(old.keys() - new.keys()),
                {k: v for k, v in new.items() if old.get(k, new) is not v},
            )
        stale, changed = switch
        for name in stale:
            namespace.pop(name, None)
        namespace.update(changed)
    _synced_module[0] = id(module)


xp.array_backend.add_listener(_sync_backend_namespace)
