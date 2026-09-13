"""Compiler-free contracts shared by native Python and Pyodide."""

import numpy as np
import pytest

import cunumpy as xp


@pytest.fixture(autouse=True)
def numpy_backend():
    with xp.use_backend("numpy"):
        yield


def test_array_operations():
    a = xp.arange(6, dtype=xp.float64).reshape(2, 3)
    np.testing.assert_array_equal(xp.sum(a, axis=1), [3, 12])
    np.testing.assert_array_equal(a @ a.T, [[5, 14], [14, 50]])
    np.testing.assert_allclose(xp.linalg.solve(xp.eye(2), xp.ones(2)), [1, 1])
    assert xp.numpy_backend and not xp.cupy_backend
    assert xp.get_backend(a) == "numpy"
    assert xp.is_cpu(a) and not xp.is_gpu(a)
    assert xp.same_backend(a, xp.ones(2))
    xp.assert_same_backend(a, xp.ones(2))
    xp.synchronize()
    xp.set_device(0)


def test_conversions_preserve_identity_and_views():
    a = xp.arange(6)
    view = a[::2]
    assert xp.to_numpy(a) is a
    assert xp.to_cunumpy(a) is a
    assert xp.to_numpy(view) is view
    xp.to_cunumpy(view)[...] = -1
    np.testing.assert_array_equal(a, [-1, 1, -1, 3, -1, 5])
    np.testing.assert_array_equal(xp.to_numpy([1, 2]), [1, 2])
    np.testing.assert_array_equal(xp.to_cunumpy([1, 2]), [1, 2])


@pytest.mark.parametrize("raises", [False, True])
def test_backend_context_restoration(raises):
    import cunumpy.xp as backend

    original = backend.array_backend.xp
    try:
        with xp.use_backend("numpy"):
            with xp.use_backend("numpy"):
                xp.set_backend("numpy")
                if raises:
                    raise RuntimeError("kernel failed")
    except RuntimeError:
        assert raises
    assert backend.array_backend.xp is original
    assert xp.numpy_backend and not xp.cupy_backend


@pytest.mark.parametrize("use_cupy", [None, False])
def test_python_kernel_mutation_aliasing_and_returns(use_cupy):
    a = xp.arange(6, dtype=xp.float64)
    view = a[::2]

    def kernel(values, alias, *, out, scale):
        assert values is alias
        assert out is view
        assert np.shares_memory(values, out)
        out[...] *= scale
        return values, out, float(values.sum()), {"array": values}

    wrapped = xp.PyccelKernel(kernel, use_cupy=use_cupy, outputs=("out",))
    result = wrapped(a, a, out=view, scale=3)
    assert result[0] is a
    assert result[1] is view
    assert result[2] == 27.0
    assert result[3]["array"] is a
    np.testing.assert_array_equal(a, [0, 1, 6, 3, 12, 5])


def test_python_kernel_none_return_and_exception():
    a = xp.zeros(2)

    def fill(out):
        out[...] = 4

    assert xp.PyccelKernel(fill)(a) is None
    np.testing.assert_array_equal(a, [4, 4])

    def fail(out):
        out[0] = 7
        raise ValueError("source kernel failed")

    with pytest.raises(ValueError, match="source kernel failed"):
        xp.PyccelKernel(fail)(a)
    np.testing.assert_array_equal(a, [7, 4])
