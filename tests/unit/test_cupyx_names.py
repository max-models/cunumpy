"""The fake cupyx.scipy names (_fake_cupyx.NAMES) against the oldest supported and the installed CuPy."""

import types

import pytest

import cunumpy as xp
from cunumpy import _fake_cupyx
from cunumpy._fake_cupyx import NAMES, NAMES_VERSION, compare_names, public_names
from cunumpy.kernel_testing import requires_cupy


def test_names_are_those_of_the_oldest_supported_cupy():
    assert NAMES_VERSION == xp.MIN_CUPY_VERSION
    assert tuple(NAMES) == _fake_cupyx.SUBPACKAGES
    # the reason for the minimum: CuPy 13 has no smoothing splines
    assert {"UnivariateSpline", "NdBSpline"} <= NAMES["interpolate"]


def test_compare_names():
    real = {path: set(names) for path, names in NAMES.items()}
    assert compare_names(real) == ({}, {})
    real["special"] = (real["special"] - {"erf"}) | {"jv"}
    missing, added = compare_names(real)
    assert missing == {"special": ["erf"]}
    assert added == {"special": ["jv"]}


def test_public_names_skip_private_names_and_submodules():
    module = types.ModuleType("cupyx.scipy.example")
    module.solve = lambda: None
    module._helper = lambda: None
    module.annotations = object()  # from __future__ import annotations
    module.linalg = types.ModuleType("cupyx.scipy.example.linalg")
    assert public_names(module) == {"solve"}


@requires_cupy
def test_installed_cupy_has_every_name_of_the_fake():
    """On a GPU: the fake has no name the installed CuPy lacks; on the oldest supported CuPy, the same names."""
    import cupy

    missing, added = compare_names(_fake_cupyx.installed_names())
    assert not missing, f"CuPy {cupy.__version__} lacks names of the fake: {missing}"
    if cupy.__version__ == NAMES_VERSION:
        assert not added, f"CuPy {NAMES_VERSION} has names the fake lacks: {added}"


@requires_cupy
def test_installed_cupy_is_supported():
    import cupy

    assert xp.cupy_version_supported(cupy.__version__)


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("13.6.0", False),
        ("14.0.0", True),
        ("14.0.0rc1", False),
        ("14.2.0", True),
        ("15.0.0a1", True),
    ],
)
def test_cupy_version_supported(version, supported):
    assert xp.cupy_version_supported(version) is supported


def test_old_cupy_warns_once(monkeypatch):
    import sys

    from cunumpy import xp as backend

    old = types.ModuleType("cupy")
    old.__version__ = "13.6.0"
    old.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "cupy", old)
    monkeypatch.setattr(backend, "_CUPY_AVAILABLE_CACHE", None)
    with pytest.warns(
        RuntimeWarning, match=r"CuPy 13\.6\.0 is older than .* \(14\.0\.0\)"
    ):
        assert not backend.cupy_available()
    assert not backend.cupy_available()  # cached: no second warning
