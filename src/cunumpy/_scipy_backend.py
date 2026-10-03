"""SciPy for the active backend: ``xp.scipy`` is SciPy or ``cupyx.scipy``.

Field solvers and fluid codes need more than array functions: sparse matrices
and their iterative solvers, FFTs, special functions, image filters.
``xp.scipy`` forwards to :mod:`scipy` on the NumPy backend and to
:mod:`cupyx.scipy` on the CuPy backend, so this code runs on both::

    A = xp.scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    x, info = xp.scipy.sparse.linalg.cg(A, b)
    phi_k = xp.scipy.fft.rfftn(rho)
    f = xp.scipy.special.erf(v / v_th)

The module is resolved at every attribute access, so switching the backend
(:func:`cunumpy.set_backend`) takes effect immediately; imports are cached by
Python, so the lookup is cheap. Neither SciPy nor CuPy is imported until a
name is used.

``cupyx.scipy`` covers only part of SciPy. A name that the active backend's
module does not have raises ``AttributeError`` saying which backend lacks it;
:meth:`ScipyNamespace.available` checks a name without raising. Keyword
arguments can differ as well: SciPy's ``cg`` takes ``rtol`` since SciPy 1.12,
CuPy's still takes ``tol``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from .xp import get_backend

__all__ = ["SUBMODULES", "ScipyNamespace", "scipy"]

#: The SciPy subpackages forwarded by ``xp.scipy``: those that ``cupyx.scipy``
#: provides as well.
SUBMODULES = (
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

_ROOTS = {"numpy": "scipy", "cupy": "cupyx.scipy"}
_INSTALL = {
    "numpy": "SciPy is not installed (pip install scipy)",
    "cupy": "CuPy is not installed or does not provide it",
}


class ScipyNamespace:
    """SciPy (sub)package of the active backend; see :data:`cunumpy.scipy`.

    Parameters
    ----------
    path : str
        Dotted path below the SciPy root, e.g. ``"sparse.linalg"``; empty for
        the root itself.
    """

    def __init__(self, path: str = "") -> None:
        if path and path not in SUBMODULES:
            raise ValueError(
                f"scipy.{path} is not forwarded; forwarded subpackages: "
                f"{', '.join(SUBMODULES)}"
            )
        self._path = path
        self._children: dict[str, ScipyNamespace] = {}

    def __repr__(self) -> str:
        return f"<cunumpy scipy namespace {self._name('<backend>')}>"

    def _name(self, root: str) -> str:
        return f"{root}.{self._path}" if self._path else root

    def resolve(self) -> ModuleType:
        """The module this namespace stands for on the active backend.

        Returns
        -------
        ModuleType
            E.g. ``scipy.sparse.linalg`` or ``cupyx.scipy.sparse.linalg``.

        Raises
        ------
        ImportError
            If SciPy (NumPy backend) or CuPy (CuPy backend) is not installed.
        """
        backend = get_backend()
        name = self._name(_ROOTS[backend])
        try:
            return importlib.import_module(name)
        except ImportError as error:
            raise ImportError(
                f"xp.scipy on the {backend} backend needs {name}: {_INSTALL[backend]}"
            ) from error

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        child = f"{self._path}.{name}" if self._path else name
        if child in SUBMODULES:
            if name not in self._children:
                self._children[name] = ScipyNamespace(child)
            return self._children[name]
        module = self.resolve()
        try:
            return getattr(module, name)
        except AttributeError:
            other = "cupy" if get_backend() == "numpy" else "numpy"
            raise AttributeError(
                f"{module.__name__} has no attribute {name!r}: it is not available "
                f"on the {get_backend()} backend (it may exist in "
                f"{self._name(_ROOTS[other])})"
            ) from None

    def available(self, name: str) -> bool:
        """Whether `name` exists in this namespace on the active backend.

        Never raises; False also if the backend's SciPy is not installed.
        """
        child = f"{self._path}.{name}" if self._path else name
        if child in SUBMODULES:
            try:
                ScipyNamespace(child).resolve()
            except ImportError:
                return False
            return True
        try:
            return hasattr(self.resolve(), name)
        except ImportError:
            return False

    def __dir__(self) -> list[str]:
        prefix = f"{self._path}." if self._path else ""
        children = [
            s[len(prefix) :]
            for s in SUBMODULES
            if s.startswith(prefix) and "." not in s[len(prefix) :]
        ]
        try:
            names = dir(self.resolve())
        except ImportError:
            names = []
        return sorted(set(names) | set(children) | {"available", "resolve"})


#: SciPy of the active backend, available as ``xp.scipy``.
scipy = ScipyNamespace()
