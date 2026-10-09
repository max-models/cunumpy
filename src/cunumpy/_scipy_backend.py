"""SciPy for the active backend: ``xp.scipy`` is SciPy or ``cupyx.scipy``.

``xp.scipy`` forwards to :mod:`scipy` on the NumPy backend and to
:mod:`cupyx.scipy` on the CuPy backend (sparse matrices and solvers, FFTs,
special functions, ...), so this code runs on both::

    A = xp.scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    x, info = xp.scipy.sparse.linalg.cg(A, b)
    f = xp.scipy.special.erf(v / v_th)

Names are resolved at every access, so a backend switch takes effect at once;
nothing is imported until a name is used, and SciPy is not a dependency of
cunumpy. ``cupyx.scipy`` covers only part of SciPy, and keyword arguments can
differ (SciPy's ``cg`` takes ``rtol``, CuPy's ``tol``). A few missing
functions are filled in by cunumpy on the device (``FILLS``, e.g.
``xp.scipy.linalg.solve_circulant``). A SciPy sparse matrix
moves to the device once with ``xp.scipy.sparse.csr_matrix(host_matrix)`` on
the CuPy backend, and back with ``matrix.get()``. See :doc:`/guides/solvers`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from cunumpy import _linalg
from cunumpy.xp import get_backend

__all__ = ["FILLS", "SUBMODULES", "ScipyNamespace", "scipy"]

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

#: Functions ``cupyx.scipy`` lacks, implemented by cunumpy on the device; used
#: on the CuPy backend only, where the ``cupyx`` module has no such name.
FILLS = {"linalg": {"solve_circulant": _linalg.solve_circulant}}

_ROOTS = {"numpy": "scipy", "cupy": "cupyx.scipy"}
_INSTALL = {
    "numpy": "SciPy is not installed (pip install scipy)",
    "cupy": "CuPy is not installed or does not provide it",
}


class ScipyNamespace:
    """A SciPy (sub)package of the active backend; see :data:`cunumpy.scipy`.

    Attribute access forwards to the module of the active backend. A name the
    backend's module lacks raises ``AttributeError`` naming the backend, unless
    cunumpy fills it in (``FILLS``, CuPy backend only); a
    missing SciPy (NumPy backend) or CuPy raises ``ImportError``.

    Parameters
    ----------
    path : str, optional
        Dotted path below the SciPy root, one of ``SUBMODULES``, e.g.
        ``"sparse.linalg"``; empty for the root itself.

    Raises
    ------
    ValueError
        If `path` is not a forwarded subpackage.

    Examples
    --------
    >>> xp.scipy.sparse
    <cunumpy scipy namespace <backend>.sparse>
    >>> x, info = xp.scipy.sparse.linalg.cg(A, b)  # doctest: +SKIP
    """

    def __init__(self, path: str = "") -> None:
        if path and path not in SUBMODULES:
            raise ValueError(
                f"scipy.{path} is not forwarded; forwarded subpackages: "
                f"{', '.join(SUBMODULES)}",
            )
        self._path = path
        self._children: dict[str, ScipyNamespace] = {}

    def __repr__(self) -> str:
        return f"<cunumpy scipy namespace {self._name('<backend>')}>"

    def _name(self, root: str) -> str:
        return f"{root}.{self._path}" if self._path else root

    def resolve(self) -> ModuleType:
        """Return the module this namespace stands for on the active backend.

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
                f"xp.scipy on the {backend} backend needs {name}: {_INSTALL[backend]}",
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
            fill = self._fill(name)
            if fill is not None:
                return fill
            other = "cupy" if get_backend() == "numpy" else "numpy"
            raise AttributeError(
                f"{module.__name__} has no attribute {name!r}: it is not available "
                f"on the {get_backend()} backend (it may exist in "
                f"{self._name(_ROOTS[other])})",
            ) from None

    def _fill(self, name: str) -> Any:
        if get_backend() != "cupy":
            return None
        return FILLS.get(self._path, {}).get(name)

    def available(self, name: str) -> bool:
        """Return whether `name` exists in this namespace on the active backend.

        Parameters
        ----------
        name : str
            A function, class or subpackage name.

        Returns
        -------
        bool
            False also if the backend's SciPy is not installed; never raises.

        Examples
        --------
        >>> xp.scipy.special.available("erfcx")  # doctest: +SKIP
        True
        """
        child = f"{self._path}.{name}" if self._path else name
        if child in SUBMODULES:
            try:
                ScipyNamespace(child).resolve()
            except ImportError:
                return False
            return True
        try:
            return hasattr(self.resolve(), name) or self._fill(name) is not None
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
        if get_backend() == "cupy":
            names = [*names, *FILLS.get(self._path, {})]
        return sorted(set(names) | set(children) | {"available", "resolve"})


#: SciPy of the active backend, available as ``xp.scipy``.
scipy = ScipyNamespace()
