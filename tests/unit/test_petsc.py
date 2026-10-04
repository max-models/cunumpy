"""Tests for `xp.petsc.petsc_vec`: PETSc vectors sharing the memory of an array."""

import sys
import types

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import petsc


def _petsc():
    petsc4py = pytest.importorskip("petsc4py")
    petsc4py.init()
    from petsc4py import PETSc

    return PETSc


def test_host_vector_shares_memory():
    PETSc = _petsc()
    a = np.zeros(6, dtype=PETSc.ScalarType)
    vec = xp.petsc.petsc_vec(a, comm=PETSc.COMM_SELF)
    assert vec.getSize() == 6 and vec.getType() == "seq"
    vec.set(3.0)
    assert a.tolist() == [3.0] * 6  # PETSc wrote into the array
    a[0] = -1.0
    assert vec.getValue(0) == -1.0  # and reads from it
    assert vec.getAttr("cunumpy_array") is a  # keeps the array alive


def test_multidimensional_arrays_are_unrolled():
    PETSc = _petsc()
    a = np.arange(6, dtype=PETSc.ScalarType).reshape(2, 3)
    vec = xp.petsc.petsc_vec(a, comm=PETSc.COMM_SELF)
    assert vec.getArray().tolist() == a.ravel().tolist()


def test_ksp_solve_writes_into_the_array():
    PETSc = _petsc()
    n = 10
    A = PETSc.Mat().createAIJ([n, n], nnz=3, comm=PETSc.COMM_SELF)
    for i in range(n):
        A.setValue(i, i, 2.0)
        if i > 0:
            A.setValue(i, i - 1, -1.0)
        if i < n - 1:
            A.setValue(i, i + 1, -1.0)
    A.assemble()
    b, x = np.ones(n), np.zeros(n)
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("none")
    ksp.setTolerances(rtol=1e-12)
    ksp.solve(
        xp.petsc.petsc_vec(b, comm=PETSc.COMM_SELF),
        xp.petsc.petsc_vec(x, comm=PETSc.COMM_SELF),
    )
    residual = 2 * x - np.r_[0.0, x[:-1]] - np.r_[x[1:], 0.0] - 1.0
    assert np.abs(residual).max() < 1e-9


def test_rejects_arrays_that_would_need_a_copy():
    PETSc = _petsc()
    other = np.float32 if np.dtype(PETSc.ScalarType) != np.float32 else np.float64
    with pytest.raises(TypeError, match="scalar type"):
        xp.petsc.petsc_vec(np.zeros(4, dtype=other))
    with pytest.raises(ValueError, match="C-contiguous"):
        xp.petsc.petsc_vec(np.zeros((4, 4))[:, 0])
    with pytest.raises(TypeError, match="NumPy or CuPy array"):
        xp.petsc.petsc_vec([0.0, 1.0])


class FakeDeviceArray:
    dtype = np.dtype(np.float64)
    flags = types.SimpleNamespace(c_contiguous=True)


def _fake_petsc(monkeypatch, vec_type=None, error=False):
    class Error(Exception):
        pass

    class Vec:
        destroyed = False

        def createWithDLPack(self, array, comm=None):
            if error:
                raise Error("no CUDA")
            return self

        def getType(self):
            return vec_type

        def destroy(self):
            Vec.destroyed = True

        def setAttr(self, name, value):
            self.attr = (name, value)

    PETSc = types.SimpleNamespace(Vec=Vec, Error=Error, ScalarType=np.float64)
    package = types.ModuleType("petsc4py")
    package.PETSc = PETSc
    monkeypatch.setitem(sys.modules, "petsc4py", package)
    monkeypatch.setitem(sys.modules, "petsc4py.PETSc", PETSc)
    monkeypatch.setattr(
        petsc,
        "_is_device_array",
        lambda a: isinstance(a, FakeDeviceArray),
    )
    return Vec


@pytest.mark.parametrize("vec_type", ["seqcuda", "mpicuda", "seqhip"])
def test_device_arrays_give_device_vectors(monkeypatch, vec_type):
    _fake_petsc(monkeypatch, vec_type)
    array = FakeDeviceArray()
    vec = xp.petsc.petsc_vec(array)
    assert vec.attr == ("cunumpy_array", array)


def test_device_arrays_need_a_gpu_petsc(monkeypatch):
    Vec = _fake_petsc(monkeypatch, "seq")  # PETSc without CUDA made a host vector
    with pytest.raises(RuntimeError, match="created a 'seq' vector for a CuPy array"):
        xp.petsc.petsc_vec(FakeDeviceArray())
    assert Vec.destroyed
    _fake_petsc(monkeypatch, error=True)
    with pytest.raises(RuntimeError, match="built with CUDA or HIP support"):
        xp.petsc.petsc_vec(FakeDeviceArray())


def test_missing_petsc4py(monkeypatch):
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    with pytest.raises(ImportError, match="needs petsc4py"):
        xp.petsc.petsc_vec(np.zeros(3))
