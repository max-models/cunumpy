"""The fake cupyx.scipy: SciPy behind the CuPy rules (device arrays in and out)."""

from pathlib import Path

import pytest

from cunumpy import _fake_cupyx
from cunumpy._scipy_backend import SUBMODULES
from cunumpy.kernel_testing import run_in_fake_cupy_subprocess

SRC = str(Path(__file__).resolve().parents[2] / "src")


def run(code):
    pytest.importorskip("scipy")
    return run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}).stdout


def test_same_subpackages_as_xp_scipy():
    assert _fake_cupyx.SUBPACKAGES == SUBMODULES
    assert tuple(_fake_cupyx.NAMES) == SUBMODULES


def test_splines_fit_and_evaluate_on_the_device():
    code = """
import numpy as np
import scipy.interpolate as si
import cunumpy as xp
xp.set_backend("cupy")
x_h = np.linspace(0.0, 1.0, 11)
x = xp.asarray(x_h)
y = xp.sin(3 * x)
spl = xp.scipy.interpolate.UnivariateSpline(x, y, k=3, s=0.0, ext=3)
ref = si.UnivariateSpline(x_h, np.sin(3 * x_h), k=3, s=0.0, ext=3)
assert isinstance(spl, xp.scipy.interpolate.UnivariateSpline)
for nu in (0, 1, 2):
    out = spl(x, nu=nu)
    assert xp.is_gpu(out)
    assert np.allclose(xp.to_numpy(out), ref(x_h, nu=nu), rtol=0, atol=1e-14)
assert xp.is_gpu(spl(0.3)) and spl(0.3).ndim == 0
# a returned spline is a proxy too, and its arrays are device arrays
b = xp.scipy.interpolate.make_interp_spline(x, y, k=3)
assert xp.is_gpu(b.t) and xp.is_gpu(b.derivative(1)(x))
nd = xp.scipy.interpolate.NdBSpline((b.t, b.t), xp.ones((11, 11)), 3)
assert xp.is_gpu(nd(xp.asarray([[0.2, 0.3]]), nu=(1, 0)))
print("ok")
"""
    assert "ok" in run(code)


def test_splines_can_be_copied():
    code = """
import copy
import numpy as np
import cunumpy as xp
xp.set_backend("cupy")
x = xp.linspace(0.0, 1.0, 11)
spl = xp.scipy.interpolate.UnivariateSpline(x, xp.sin(3 * x), k=3, s=0.0, ext=3)
for c in (copy.copy(spl), copy.deepcopy(spl), copy.deepcopy({"spline": spl})["spline"]):
    assert isinstance(c, xp.scipy.interpolate.UnivariateSpline)
    assert xp.is_gpu(c(x))
    assert np.array_equal(xp.to_numpy(c(x)), xp.to_numpy(spl(x)))
# a deep copy is independent of the original
c = copy.deepcopy(spl)
assert c._obj is not spl._obj
print("ok")
"""
    assert "ok" in run(code)


def test_host_arrays_are_rejected_like_cupy():
    code = """
import numpy as np
import cunumpy as xp
xp.set_backend("cupy")
x = xp.linspace(0.0, 1.0, 11)
spl = xp.scipy.interpolate.UnivariateSpline(x, x**2, s=0.0)
for call in (lambda: spl(np.linspace(0.0, 1.0, 3)),
             lambda: xp.scipy.interpolate.UnivariateSpline(np.linspace(0, 1, 11), x, s=0.0),
             lambda: xp.scipy.special.erf(np.zeros(2))):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError("host array accepted")
print("ok")
"""
    assert "ok" in run(code)


def test_interpolate_has_only_the_names_cupy_provides():
    code = """
import cunumpy as xp
xp.set_backend("cupy")
assert not xp.scipy.interpolate.available("RectBivariateSpline")
assert not xp.scipy.interpolate.available("splrep")
assert xp.scipy.interpolate.available("NdBSpline")
try:
    xp.scipy.interpolate.RectBivariateSpline
except AttributeError as error:
    assert "not available on the cupy backend" in str(error)
else:
    raise AssertionError
print("ok")
"""
    assert "ok" in run(code)


def test_every_subpackage_has_only_the_names_cupy_provides():
    code = """
import cunumpy as xp
import cupyx.scipy.linalg
xp.set_backend("cupy")
missing = {
    "special": ("jv", "erfi"),
    "linalg": ("solve_circulant", "inv", "eigh"),
    "sparse.linalg": ("inv",),
}
present = {
    "special": ("yn", "erf", "erfinv", "ndtri"),
    "linalg": ("circulant", "lu_factor", "lu_solve"),
    "sparse": ("csr_matrix", "kron", "bmat", "identity"),
    "sparse.linalg": ("splu", "spsolve", "cg"),
    "signal": ("argrelextrema",),
    "fft": ("rfft", "irfft"),
}
ns = lambda path: eval("xp.scipy." + path)
for path, names in missing.items():
    for name in names:
        assert not hasattr(ns(path).resolve(), name), (path, name)
for path, names in present.items():
    for name in names:
        assert ns(path).available(name), (path, name)
# cunumpy fills in solve_circulant, which raw cupyx lacks
assert not hasattr(cupyx.scipy.linalg, "solve_circulant")
assert xp.scipy.linalg.available("solve_circulant")
print("ok")
"""
    assert "ok" in run(code)


def test_solve_circulant_stays_on_the_device():
    code = """
import numpy as np
import scipy.linalg
import cunumpy as xp
xp.set_backend("cupy")
rng = np.random.default_rng(0)
c_h, b_h = rng.random(6) + 2.0, rng.random((6, 3))
x = xp.scipy.linalg.solve_circulant(xp.asarray(c_h), xp.asarray(b_h))
assert xp.is_gpu(x)
assert np.allclose(xp.to_numpy(x), scipy.linalg.solve_circulant(c_h, b_h))
try:
    xp.scipy.linalg.solve_circulant(xp.asarray([1.0, 1.0, 0.0, 0.0]), xp.ones(4))
except np.linalg.LinAlgError:
    pass
else:
    raise AssertionError("singular matrix accepted")
print("ok")
"""
    assert "ok" in run(code)


def test_other_subpackages_and_pinned_memory():
    code = """
import numpy as np
import cunumpy as xp
import cupyx
xp.set_backend("cupy")
out = xp.scipy.special.erf(xp.zeros(3))
assert xp.is_gpu(out)
assert type(cupyx.empty_pinned((2, 3))) is np.ndarray
assert np.all(cupyx.zeros_pinned(4) == 0)
print("ok")
"""
    assert "ok" in run(code)


def test_uninstall_removes_cupyx():
    code = """
import sys
from cunumpy import _fake_cupy
assert "cupyx.scipy.interpolate" in sys.modules
_fake_cupy.uninstall()
assert not [n for n in sys.modules if n.split(".")[0] in ("cupy", "cupyx")]
print("ok")
"""
    assert "ok" in run(code)
