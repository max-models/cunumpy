"""xp.optimize.newton: batched Newton and secant, compared with scipy.optimize.newton."""

from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.kernel_testing import run_in_fake_cupy_subprocess

SRC = str(Path(__file__).resolve().parents[2] / "src")


def cubic(levels):
    return (lambda x: x**3 - 2 * x - levels), (lambda x: 3 * x**2 - 2)


@pytest.mark.parametrize("method", ["newton", "secant"])
def test_matches_scipy_on_an_array(method):
    so = pytest.importorskip("scipy.optimize")
    levels = np.random.default_rng(0).uniform(1.0, 10.0, 500)
    f, fp = cubic(levels)
    kw = {"fprime": fp} if method == "newton" else {}
    with xp.use_backend("numpy"):
        root = xp.optimize.newton(f, np.ones(500), **kw)
    np.testing.assert_allclose(
        root, so.newton(f, np.ones(500), **kw), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(f(root), 0.0, atol=1e-12)


def test_scalar_and_list_starting_points_go_to_the_active_backend():
    with xp.use_backend("numpy"):
        root = xp.optimize.newton(lambda x: x**2 - 2, 1.0, fprime=lambda x: 2 * x)
        assert isinstance(root, np.ndarray) and root.shape == ()
        assert root == pytest.approx(np.sqrt(2))
        roots = xp.optimize.newton(lambda x: x**2 - 2, [1.0, -1.0])
        np.testing.assert_allclose(roots, [np.sqrt(2), -np.sqrt(2)])


def test_x0_is_not_modified():
    x0 = np.ones(3)
    xp.optimize.newton(lambda x: x - 2, x0, fprime=lambda x: np.ones_like(x))
    np.testing.assert_array_equal(x0, 1.0)


def test_full_output_matches_scipy():
    so = pytest.importorskip("scipy.optimize")
    # x**2 + 1 has no real root; x = 0 has a zero derivative from the start, and
    # from x = 1 the first step lands on it
    x0 = np.array([2.5, 0.0, 1.0, 1.3])
    kw = {"fprime": lambda x: 2 * x, "maxiter": 20, "full_output": True}
    f = lambda x: x**2 + 1
    out = xp.optimize.newton(f, x0, **kw)
    with pytest.warns(RuntimeWarning):
        ref = so.newton(f, x0, **kw)
    assert isinstance(out, xp.optimize.NewtonResult)
    np.testing.assert_array_equal(out.converged, ref.converged)
    np.testing.assert_array_equal(out.zero_der, ref.zero_der)
    np.testing.assert_array_equal(out.zero_der, [False, True, True, False])


def test_partial_failure_warns():
    with pytest.warns(RuntimeWarning, match="1 of 2 points did not converge"):
        xp.optimize.newton(
            lambda x: x**2 - np.array([4.0, -1.0]),
            np.array([1.0, 1.0]),
            fprime=lambda x: 2 * x,
            maxiter=20,
        )


def test_tolerances():
    f, fp = cubic(np.array([5.0]))
    loose = xp.optimize.newton(f, np.array([10.0]), fprime=fp, tol=1e-2)
    tight = xp.optimize.newton(f, np.array([10.0]), fprime=fp, tol=0.0, rtol=1e-15)
    assert abs(f(tight)[0]) < abs(f(loose)[0])
    assert abs(f(tight)[0]) < 1e-12


def test_args_are_passed_to_func_and_fprime():
    levels = np.array([1.0, 4.0, 9.0])
    root = xp.optimize.newton(
        lambda x, s: x**2 - s, np.ones(3), fprime=lambda x, s: 2 * x, args=(levels,)
    )
    np.testing.assert_allclose(root, [1.0, 2.0, 3.0])


def test_runs_on_the_device_on_the_fake_cupy():
    code = """
import numpy as np
import cunumpy as xp
xp.set_backend("cupy")
levels = xp.asarray(np.linspace(1.0, 10.0, 50))
for kw in ({"fprime": lambda x: 3 * x**2 - 2}, {}):
    out = xp.optimize.newton(lambda x: x**3 - 2 * x - levels, xp.ones(50), full_output=True, **kw)
    assert xp.is_gpu(out.root) and xp.is_gpu(out.converged)
    assert bool(xp.all(out.converged))
    assert float(xp.max(xp.abs(out.root**3 - 2 * out.root - levels))) < 1e-12
assert xp.is_gpu(xp.optimize.newton(lambda x: x**2 - 2, 1.0))
print("ok")
"""
    assert "ok" in run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout
