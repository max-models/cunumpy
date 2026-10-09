"""NumPy and CuPy behind one namespace.

``import cunumpy as xp`` gives the functions of the active backend's
``array-api-compat`` module (``xp.zeros``, ``xp.linalg``, ...), replaced when
the backend changes, plus backend selection, array inspection and conversion,
host execution helpers (:func:`host_call`), and :data:`scipy`. Everything else
is in a submodule (:mod:`cunumpy.kernels`, :mod:`cunumpy.cuda`, ...). See
:doc:`/api` for the module layout and :doc:`/guides/backends`.

``xp.numpy_backend`` and ``xp.cupy_backend`` are booleans telling whether the
active backend is NumPy or CuPy (prefer :func:`get_backend` for the name).
"""

# cunumpy/__init__.py
import re as _re
from importlib.metadata import PackageNotFoundError, version

from cunumpy import (
    algorithms,
    arguments,
    cuda,
    kernels,
    memory,
    mpi,
    petsc,
    profiling,
    rng,
    xp,
)
from cunumpy._host import evaluate_on_host, host_call, setup_on_host
from cunumpy._scipy_backend import scipy
from cunumpy._staging import to_host_async
from cunumpy.xp import (
    as_device_array,
    assert_same_backend,
    backend_info,
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

try:
    #: The installed version of cunumpy; ``"0.0.0+unknown"`` without package metadata.
    __version__ = version("cunumpy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _re.findall(r"\d+", text.split("+")[0])[:3])


def require_version(minimum: str) -> None:
    """Raise ``ImportError`` if the installed cunumpy is older than `minimum`.

    A clearer error than a later ``AttributeError`` for code that needs a
    feature of a given release. Only the numeric part is compared (``0.4.0``
    and ``0.4.0.dev1`` are equal); nothing is checked when the installed
    version is unknown (cunumpy not installed as a package, ``xp.__version__``
    is then ``"0.0.0+unknown"``).

    Parameters
    ----------
    minimum : str
        The oldest acceptable version, e.g. ``"0.4.0"``.

    Raises
    ------
    ImportError
        If the installed version is older than `minimum`.

    Examples
    --------
    >>> xp.require_version("0.4.0")
    """
    if __version__.startswith("0.0.0+unknown"):
        return
    if _version_key(__version__) < _version_key(minimum):
        raise ImportError(
            f"cunumpy {minimum} or newer is required, but {__version__} is "
            "installed: pip install --upgrade cunumpy",
        )


__all__ = [
    "__version__",
    "algorithms",
    "arguments",
    "as_device_array",
    "assert_same_backend",
    "backend_info",
    "cuda",
    "cupy_available",
    "cupy_backend",
    "default_float_dtype",
    "evaluate_on_host",
    "get_array_backend",
    "get_array_module",
    "get_backend",
    "host_call",
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
    "setup_on_host",
    "synchronize",
    "to_cunumpy",
    "to_cupy",
    "to_host_async",
    "to_numpy",
    "use_backend",
    "xp",
]


def __getattr__(name: str):
    """Resolve ``numpy_backend``, ``cupy_backend`` and backend names not copied here."""
    if name == "numpy_backend":
        return xp.numpy_backend
    if name == "cupy_backend":
        return xp.cupy_backend
    return getattr(xp.xp, name)


# `xp.zeros` must be as fast as `numpy.zeros`. A module-level __getattr__ runs
# only after the normal lookup failed, which costs about 3 us per access, so the
# public names of the active backend module are copied into this namespace, and
# replaced whenever the backend changes. cunumpy's own names are never overwritten.
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
