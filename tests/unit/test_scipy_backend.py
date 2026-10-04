"""Tests for `xp.scipy`: SciPy or cupyx.scipy, by the active backend."""

import sys
import types

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _scipy_backend as scipy_backend
from cunumpy._scipy_backend import SUBMODULES, ScipyNamespace


@pytest.fixture
def fake_cupyx(monkeypatch):
    """A fake `cupyx.scipy` with `special.erf` and `sparse.linalg.cg`, as backend."""
    modules = {}
    for name in ("cupyx", "cupyx.scipy", "cupyx.scipy.special", "cupyx.scipy.sparse"):
        modules[name] = types.ModuleType(name)
    modules["cupyx.scipy.sparse.linalg"] = types.ModuleType("cupyx.scipy.sparse.linalg")
    modules["cupyx.scipy.special"].erf = "device erf"
    modules["cupyx.scipy.sparse.linalg"].cg = "device cg"
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(scipy_backend, "get_backend", lambda: "cupy")
    return modules


def test_scipy_is_exported():
    assert isinstance(xp.scipy, ScipyNamespace)
    assert "scipy" in xp.__all__
    assert set(dir(xp.scipy)) >= {"sparse", "special", "fft", "available", "resolve"}
    assert set(dir(xp.scipy.sparse)) >= {"linalg", "csgraph"}


def test_unknown_subpackage():
    with pytest.raises(ValueError, match="not forwarded"):
        ScipyNamespace("optimize")
    assert all(ScipyNamespace(name) for name in SUBMODULES)


def test_numpy_backend_forwards_to_scipy():
    pytest.importorskip("scipy")
    import scipy.sparse.linalg
    import scipy.special

    assert xp.scipy.special.erf is scipy.special.erf
    assert xp.scipy.sparse.linalg.cg is scipy.sparse.linalg.cg
    assert xp.scipy.sparse.linalg.resolve() is scipy.sparse.linalg
    assert xp.scipy.sparse.linalg is xp.scipy.sparse.linalg  # cached namespaces

    n = 20
    A = xp.scipy.sparse.diags(
        [np.full(n - 1, -1.0), np.full(n, 2.0), np.full(n - 1, -1.0)],
        [-1, 0, 1],
        format="csr",
    )
    x, info = xp.scipy.sparse.linalg.cg(A, np.ones(n), rtol=1e-12)
    assert info == 0 and np.allclose(A @ x, 1.0)


def test_missing_names_say_which_backend_lacks_them():
    pytest.importorskip("scipy")
    with pytest.raises(AttributeError, match=r"not available on the numpy backend"):
        xp.scipy.special.no_such_function  # noqa: B018
    assert xp.scipy.special.available("erf")
    assert not xp.scipy.special.available("no_such_function")
    assert xp.scipy.available("sparse")
    with pytest.raises(AttributeError):
        xp.scipy.__wrapped__  # noqa: B018 -- dunder names are not forwarded


def test_cupy_backend_forwards_to_cupyx(fake_cupyx):
    assert xp.scipy.special.erf == "device erf"
    assert xp.scipy.sparse.linalg.cg == "device cg"
    assert xp.scipy.special.resolve() is fake_cupyx["cupyx.scipy.special"]
    with pytest.raises(
        AttributeError,
        match=r"cupy backend \(it may exist in scipy\.special\)",
    ):
        xp.scipy.special.erfcx  # noqa: B018
    # a subpackage cupyx does not provide
    assert not xp.scipy.available("ndimage")
    with pytest.raises(ImportError, match="needs cupyx.scipy.ndimage"):
        xp.scipy.ndimage.resolve()


def test_backend_switch_takes_effect_immediately(fake_cupyx, monkeypatch):
    pytest.importorskip("scipy")
    import scipy.special

    special = xp.scipy.special
    assert special.erf == "device erf"
    monkeypatch.setattr(scipy_backend, "get_backend", lambda: "numpy")
    assert special.erf is scipy.special.erf


def test_missing_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)  # import scipy raises ImportError
    monkeypatch.setitem(sys.modules, "scipy.special", None)
    with pytest.raises(
        ImportError,
        match=r"needs scipy.special: SciPy is not installed",
    ):
        xp.scipy.special.erf  # noqa: B018
    assert not xp.scipy.special.available("erf")
