"""Tests for the Julia adapter (skipped without juliacall)."""

import gc

import numpy as np
import pytest

from cunumpy.profiling import assert_no_transfers, count_transfers

juliacall = pytest.importorskip("juliacall")

from cunumpy.interop import julia

jl = juliacall.Main


def test_c_order_arrives_with_reversed_dimensions():
    a = np.arange(6.0).reshape(2, 3)
    with assert_no_transfers():
        array = julia.to_julia(a, copy=False)
        back = julia.from_julia(array, copy=False)
    assert jl.isa(array, jl.Matrix) and tuple(array.shape) == (3, 2)
    jl.seval("A -> (A[3, 1] = -1.0; nothing)")(array)
    assert a[0, 2] == -1.0
    assert back.shape == (2, 3) and back.flags.c_contiguous
    assert np.shares_memory(back, a)


def test_f_order_keeps_its_dimensions():
    a = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    array = julia.to_julia(a, copy=False)
    assert tuple(array.shape) == (2, 3)
    back = julia.from_julia(array, order="F")
    assert back.flags.f_contiguous and np.shares_memory(back, a)
    with pytest.raises(ValueError, match="order must be"):
        julia.from_julia(array, order="K")


def test_strided_and_read_only_arrays_are_copied():
    strided = np.arange(6.0).reshape(2, 3)[:, ::2]
    read_only = np.arange(3.0)
    read_only.flags.writeable = False
    with count_transfers() as counter:
        array = julia.to_julia(strided)
        julia.to_julia(read_only)
        julia.to_julia(np.arange(3.0), copy=True)
    assert counter.count("device_copy") == 3
    np.testing.assert_array_equal(julia.from_julia(array), strided)
    with pytest.raises(ValueError, match="not contiguous or read-only"):
        julia.to_julia(strided, copy=False)


def test_dtypes_and_scalars():
    array = julia.to_julia(np.arange(3, dtype=np.int32), dtype=np.float32)
    assert jl.eltype(array) == jl.Float32
    assert jl.eltype(julia.to_julia(np.ones(2, dtype=bool))) == jl.Bool
    assert julia.from_julia(julia.to_julia(np.float64(3.0))) == 3.0
    assert tuple(julia.to_julia(np.zeros((0, 3))).shape) == (3, 0)
    assert julia.from_julia(array, dtype=np.float64).dtype == np.float64
    with pytest.raises(TypeError, match="no Julia element type"):
        julia.to_julia(np.array(["a"]))


def test_julia_arrays_pass_through():
    array = jl.seval("zeros(2, 2)")
    assert julia.to_julia(array) is array
    assert julia.to_julia(array, copy=True) is not array
    with pytest.raises(TypeError, match="cast Julia arrays in Julia"):
        julia.to_julia(array, dtype=np.float32)


def test_other_julia_arrays_are_collected():
    with count_transfers() as counter:
        result = julia.from_julia(jl.seval("reshape(1:6, 2, 3)"))
    np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])
    assert counter.count("device_copy") == 1
    with pytest.raises(ValueError, match="not a Julia Array"):
        julia.from_julia(jl.seval("view(zeros(3, 4), :, 1:2)"), copy=False)
    with pytest.raises(TypeError, match="expected a Julia array"):
        julia.from_julia(np.ones(2))


def test_the_julia_array_keeps_the_numpy_array_alive():
    a = np.ones(4)
    array = julia.to_julia(a)
    del a
    gc.collect()
    jl.GC.gc()
    np.testing.assert_array_equal(julia.from_julia(array), np.ones(4))


def test_device_arrays_are_not_supported_yet():
    class OnCuda:
        def __dlpack_device__(self):
            return (2, 0)

    with pytest.raises(ValueError, match="Julia cannot take a OnCuda on CUDA device 0"):
        julia.to_julia(OnCuda())
