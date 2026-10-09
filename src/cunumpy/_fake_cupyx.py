"""The ``cupyx`` package of the fake CuPy: ``cupyx.scipy`` on SciPy, for tests only.

Installed with the fake ``cupy`` (:func:`cunumpy._fake_cupy.install`), so code
written against ``xp.scipy`` runs its CuPy path without a GPU. Each forwarded
SciPy function, class and method behaves like its CuPy counterpart where it
matters for finding host/device bugs:

* array arguments must be fake device arrays: NumPy arrays raise, as they do
  in CuPy;
* array results are fake device arrays, and objects (a fitted spline, a
  factorization) are proxies whose methods follow the same rules;
* ``cupyx.scipy.interpolate`` has only the names CuPy provides
  (:data:`INTERPOLATE`), so code using a SciPy name that CuPy lacks (e.g.
  ``RectBivariateSpline``) fails here as it would on a GPU. The other
  subpackages forward every SciPy name.

``cupyx.empty_pinned`` and ``cupyx.zeros_pinned`` return NumPy arrays, like
CuPy's (which are backed by pinned memory).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np

#: The subpackages of ``cupyx.scipy``; the same as ``_scipy_backend.SUBMODULES``.
SUBPACKAGES = (
    "fft",
    "fftpack",
    "interpolate",
    "linalg",
    "ndimage",
    "signal",
    "sparse",
    "sparse.csgraph",
    "sparse.linalg",
    "spatial",
    "special",
    "stats",
)

#: The names of ``cupyx.scipy.interpolate`` (CuPy reference, stable).
INTERPOLATE = frozenset(
    {
        "Akima1DInterpolator",
        "BPoly",
        "BSpline",
        "BarycentricInterpolator",
        "CloughTocher2DInterpolator",
        "CubicHermiteSpline",
        "CubicSpline",
        "InterpolatedUnivariateSpline",
        "KroghInterpolator",
        "LSQUnivariateSpline",
        "LinearNDInterpolator",
        "NdBSpline",
        "NdPPoly",
        "NearestNDInterpolator",
        "PPoly",
        "PchipInterpolator",
        "RBFInterpolator",
        "RegularGridInterpolator",
        "UnivariateSpline",
        "barycentric_interpolate",
        "interp1d",
        "interpn",
        "krogh_interpolate",
        "make_interp_spline",
        "make_lsq_spline",
        "pchip_interpolate",
        "splantider",
        "splder",
    },
)

_ALLOWED = {"interpolate": INTERPOLATE}


class _Proxy:
    """A SciPy object (spline, interpolator, ...) seen as a CuPy object."""

    def __init__(self, obj: Any, cupy: Any) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_cupy", cupy)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if callable(attr):
            return _device_callable(
                attr, self._cupy, f"{type(self._obj).__name__}.{name}"
            )
        return _to_device(attr, self._cupy)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._obj, name, _to_host(value, self._cupy))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        where = f"{type(self._obj).__name__}.__call__"
        return _device_callable(self._obj, self._cupy, where)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<fake cupyx {self._obj!r}>"


class _ProxyClass:
    """A SciPy class seen as a CuPy class: instances are :class:`_Proxy`."""

    def __init__(self, cls: type, cupy: Any, where: str) -> None:
        self._cls = cls
        self._cupy = cupy
        self._where = where
        self.__name__ = cls.__name__
        self.__doc__ = cls.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _device_callable(self._cls, self._cupy, self._where)(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._cls, name)
        if callable(attr):
            return _device_callable(attr, self._cupy, f"{self._where}.{name}")
        return attr

    def __instancecheck__(self, obj: Any) -> bool:
        return isinstance(obj, _Proxy) and isinstance(obj._obj, self._cls)

    def __repr__(self) -> str:
        return f"<fake {self._where}>"


def _is_scipy_object(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "scipy"


def _to_device(value: Any, cupy: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, cupy) for v in value)
    if _is_scipy_object(value) and not isinstance(value, np.ndarray):
        return _Proxy(value, cupy)
    return cupy._wrap(value)


def _to_host(value: Any, cupy: Any) -> Any:
    if isinstance(value, _Proxy):
        return value._obj
    if isinstance(value, (tuple, list)):
        return type(value)(_to_host(v, cupy) for v in value)
    if isinstance(value, dict):
        return {k: _to_host(v, cupy) for k, v in value.items()}
    return cupy._unwrap(value)


def _device_callable(fun: Any, cupy: Any, where: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        cupy._strict(args, where)
        cupy._strict(tuple(kwargs.values()), where)
        out = fun(*_to_host(args, cupy), **_to_host(kwargs, cupy))
        return _to_device(out, cupy)

    call.__name__ = getattr(fun, "__name__", where)
    call.__doc__ = getattr(fun, "__doc__", None)
    return call


def _subpackage(path: str, cupy: Any) -> types.ModuleType:
    module = types.ModuleType(f"cupyx.scipy.{path}")
    allowed = _ALLOWED.get(path)

    def __getattr__(name: str) -> Any:
        if name.startswith("__") or (allowed is not None and name not in allowed):
            raise AttributeError(
                f"module 'cupyx.scipy.{path}' has no attribute {name!r}"
            )
        attr = getattr(importlib.import_module(f"scipy.{path}"), name, None)
        if attr is None or isinstance(attr, types.ModuleType):
            raise AttributeError(
                f"module 'cupyx.scipy.{path}' has no attribute {name!r}"
            )
        where = f"cupyx.scipy.{path}.{name}"
        if isinstance(attr, type):
            return _ProxyClass(attr, cupy, where)
        if callable(attr):
            return _device_callable(attr, cupy, where)
        return attr

    module.__getattr__ = __getattr__
    return module


def install(cupy: Any) -> None:
    """Install ``cupyx`` and ``cupyx.scipy.*`` into ``sys.modules``, on the fake `cupy`."""
    cupyx = types.ModuleType("cupyx")
    cupyx.__cunumpy_fake__ = True
    cupyx.empty_pinned = lambda shape, dtype=float, order="C": np.empty(
        shape, dtype, order
    )
    cupyx.zeros_pinned = lambda shape, dtype=float, order="C": np.zeros(
        shape, dtype, order
    )
    scipy = types.ModuleType("cupyx.scipy")
    cupyx.scipy = scipy
    modules = {"cupyx": cupyx, "cupyx.scipy": scipy}
    for path in SUBPACKAGES:
        modules[f"cupyx.scipy.{path}"] = _subpackage(path, cupy)
    # parents expose their subpackages as attributes, like real packages
    for name, module in modules.items():
        parent, _, leaf = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], leaf, module)
    sys.modules.update(modules)
