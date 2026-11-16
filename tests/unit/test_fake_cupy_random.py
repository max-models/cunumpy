"""The fake cupy.random: NumPy's random functions behind the CuPy rules."""

from pathlib import Path

from cunumpy.kernel_testing import run_in_fake_cupy_subprocess

SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_dunder_names_are_not_wrapped():
    """A module attribute such as ``__file__`` is not a function.

    Wrapping it broke module introspection: importing astropy after cunumpy
    failed with "'function' object has no attribute 'endswith'".
    """
    code = """
import inspect
import sys
import cunumpy as xp
xp.set_backend("cupy")
random = sys.modules["cupy.random"]
assert not callable(getattr(random, "__file__", None))
assert not hasattr(random, "__path__")
assert inspect.getmodule(random) is random
assert callable(random.standard_normal)
print("ok")
"""
    out = run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout
    assert "ok" in out
