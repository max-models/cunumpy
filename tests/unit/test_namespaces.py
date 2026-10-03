"""Tests for the submodule layout of cunumpy (0.5) and the deprecated top-level names."""

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import cunumpy as xp

SUBMODULES = (
    "algorithms",
    "cuda",
    "kernels",
    "memory",
    "mpi",
    "petsc",
    "profiling",
    "rng",
)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_names_do_not_shadow_numpy(name):
    # `import cunumpy as xp` stands in for numpy: a submodule must not hide a numpy name
    assert not hasattr(np, name)
    assert getattr(xp, name) is importlib.import_module(f"cunumpy.{name}")


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_exports_resolve(name):
    module = getattr(xp, name)
    for attr in getattr(module, "__all__", ()):
        assert hasattr(module, attr), f"cunumpy.{name}.{attr}"


def test_top_level_is_backend_and_numpy_only():
    moved = set(xp._MOVED)
    assert not moved & set(xp.__all__)
    for name in xp.__all__:
        assert hasattr(xp, name)


@pytest.mark.parametrize("name", sorted(xp._MOVED))
def test_moved_names_warn_and_resolve(name):
    submodule = xp._MOVED[name]
    with pytest.warns(DeprecationWarning, match=f"cunumpy.{submodule}.{name}"):
        value = getattr(xp, name)
    assert value is getattr(getattr(xp, submodule), name)


def test_numpy_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert xp.zeros(2).shape == (2,)
        assert xp.random is not None
        assert xp.testing.assert_allclose is not None


def test_unknown_name_raises():
    with pytest.raises(AttributeError):
        _ = xp.no_such_function_in_cunumpy


def test_kernel_testing_keeps_numpy_testing():
    import cunumpy.kernel_testing  # noqa: F401

    assert xp.testing.assert_allclose is not None


def test_testing_alias_is_deprecated():
    # a fresh process: importing cunumpy.testing rebinds xp.testing for the rest of it
    code = (
        "import warnings\n"
        "warnings.simplefilter('error')\n"
        "try:\n"
        "    import cunumpy.testing\n"
        "except DeprecationWarning as w:\n"
        "    assert 'cunumpy.kernel_testing' in str(w)\n"
        "else:\n"
        "    raise SystemExit('no warning')\n"
        "warnings.simplefilter('ignore')\n"
        "import cunumpy.testing, cunumpy.kernel_testing\n"
        "assert cunumpy.testing.assert_kernels_agree is "
        "cunumpy.kernel_testing.assert_kernels_agree\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
