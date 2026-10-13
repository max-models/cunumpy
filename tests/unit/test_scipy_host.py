"""SciPy solvers cupyx lacks: `xp.optimize.fsolve/root/minimize`, `xp.integrate`."""

from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.kernel_testing import run_in_fake_cupy_subprocess

optimize = pytest.importorskip("scipy.optimize")
integrate = pytest.importorskip("scipy.integrate")

SRC = str(Path(__file__).resolve().parents[2] / "src")


def residual(x):
    return np.array(
        [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]
    )


def test_exported():
    assert set(xp.optimize.__all__) >= {"fsolve", "root", "minimize", "newton"}
    assert xp.integrate.__all__ == ["odeint", "quad"]
    assert "integrate" in xp.__all__


def test_host_arrays_give_scipy_results():
    x0 = np.zeros(2)
    np.testing.assert_allclose(
        xp.optimize.fsolve(residual, x0), optimize.fsolve(residual, x0)
    )
    _, info, ier, _ = xp.optimize.fsolve(residual, x0, full_output=True)
    assert ier == 1 and info["nfev"] > 0
    res = xp.optimize.root(residual, x0, method="lm")
    np.testing.assert_allclose(res.x, optimize.root(residual, x0, method="lm").x)
    rosen = optimize.rosen
    res = xp.optimize.minimize(
        rosen, np.zeros(3), jac=optimize.rosen_der, method="BFGS"
    )
    np.testing.assert_allclose(res.x, 1.0, atol=1e-5)
    value, _ = xp.integrate.quad(np.exp, 0.0, 1.0)
    assert value == pytest.approx(np.e - 1.0)
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(
        xp.integrate.odeint(lambda y, t: -2.0 * y, np.ones(2), t),
        integrate.odeint(lambda y, t: -2.0 * y, np.ones(2), t),
    )


def test_missing_scipy(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(ImportError, match=r"cunumpy.optimize.fsolve needs SciPy"):
        xp.optimize.fsolve(residual, np.zeros(2))


def test_callbacks_and_results_on_the_device():
    code = """
import numpy as np
import scipy.optimize as so
import cunumpy as xp
xp.set_backend("cupy")

target = xp.asarray([4.0, 9.0])  # device data captured by the callbacks

def on_device(*arrays):
    for a in arrays:
        assert xp.is_gpu(a), type(a)

def f(x):
    on_device(x)
    return x**2 - target

def jac(x):
    on_device(x)
    return xp.diag(2 * x)

x = xp.optimize.fsolve(f, xp.ones(2), fprime=jac)
assert xp.is_gpu(x) and np.allclose(xp.to_numpy(x), [2.0, 3.0])
x, info, ier, msg = xp.optimize.fsolve(f, xp.ones(2), full_output=True)
assert xp.is_gpu(x) and xp.is_gpu(info["fvec"])

seen = []
res = xp.optimize.root(f, xp.ones(2), jac=jac, method="hybr")
assert res.success and xp.is_gpu(res.x) and xp.is_gpu(res.fun)
res = xp.optimize.root(lambda x: (f(x), jac(x)), xp.ones(2), jac=True)
assert np.allclose(xp.to_numpy(res.x), [2.0, 3.0])

def fun(x):
    on_device(x)
    return xp.sum((x - target) ** 2)

def grad(x):
    on_device(x)
    return 2 * (x - target)

def hessp(x, p):
    on_device(x, p)
    return 2 * p

def callback(intermediate_result):
    on_device(intermediate_result.x)
    seen.append(1)

res = xp.optimize.minimize(fun, xp.zeros(2), jac=grad, hessp=hessp, method="trust-ncg", callback=callback)
assert seen and xp.is_gpu(res.x) and np.allclose(xp.to_numpy(res.x), [4.0, 9.0])
cons = [
    {"type": "ineq", "fun": lambda x: (on_device(x), 5.0 - x[1])[1]},
    so.NonlinearConstraint(lambda x: (on_device(x), x[0])[1], -np.inf, 3.0),
    so.LinearConstraint(np.asarray([[1.0, 1.0]]), -np.inf, 7.0),
]
bounds = so.Bounds(np.zeros(2), np.full(2, 10.0))
res = xp.optimize.minimize(fun, xp.ones(2), method="SLSQP", constraints=cons, bounds=bounds)
assert res.success and np.allclose(xp.to_numpy(res.x), [2.0, 5.0], atol=1e-6)
res = xp.optimize.minimize(fun, xp.ones(2), bounds=[(0.0, 3.0), (0.0, 3.0)], method="L-BFGS-B")
assert np.allclose(xp.to_numpy(res.x), [3.0, 3.0])

rate = xp.asarray([1.0, 2.0])
def rhs(y, t):
    on_device(y)
    return -rate * y
t = xp.linspace(0.0, 1.0, 5)
y = xp.integrate.odeint(rhs, xp.ones(2), t)
assert xp.is_gpu(y) and y.shape == (5, 2)
assert np.allclose(xp.to_numpy(y[-1]), np.exp([-1.0, -2.0]), rtol=1e-6)
y = xp.integrate.odeint(lambda t, y: rhs(y, t), xp.ones(2), t, tfirst=True)
assert np.allclose(xp.to_numpy(y[-1]), np.exp([-1.0, -2.0]), rtol=1e-6)

value, error = xp.integrate.quad(lambda s: xp.sum(rate) * s, 0.0, 1.0)
assert abs(value - 1.5) < 1e-12
print("ok")
"""
    assert "ok" in run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout
