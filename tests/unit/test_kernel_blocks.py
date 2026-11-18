"""KernelBlock: the CUDA, Metal and NumPy versions of a scalar block from its Pyccel source."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler


def inline(function):
    """A stand-in for ``pyccel.decorators.inline``: the translation ignores decorators."""
    function.inlined = True
    return function


@inline
def boris(
    vx: float,
    vy: float,
    vz: float,
    kx: float,
    ky: float,
    kz: float,
    tx: float,
    ty: float,
    tz: float,
) -> tuple[float, float, float]:
    """The Boris velocity update (as mpiparticles' ``boris_update``)."""
    return (
        (vx + kx) + (((vy + ky) + (vz + kz) * tx - (vx + kx) * tz) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tz) - ((vz + kz) + (vx + kx) * ty - (vy + ky) * tx) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * ty)) + kx,
        (vy + ky) + (((vz + kz) + (vx + kx) * ty - (vy + ky) * tx) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tx) - ((vx + kx) + (vy + ky) * tz - (vz + kz) * ty) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tz)) + ky,
        (vz + kz) + (((vx + kx) + (vy + ky) * tz - (vz + kz) * ty) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * ty) - ((vy + ky) + (vz + kz) * tx - (vx + kx) * tz) * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tx)) + kz,
    )  # fmt: skip


def left_node(rel: float, num: int, periodic: int) -> int:
    """The left node of a cell (as mpifield's ``left_node``): conditionals and int()."""
    return (
        (num - 1 if int(rel) >= num else int(rel))
        if periodic == 1
        else (
            num - 2  # noqa: FURB136 (min() is a call in Pyccel)
            if (int(rel) if rel >= 0.0 else 0) > num - 2
            else (int(rel) if rel >= 0.0 else 0)
        )
    )


def wrapped(rel: float, num: int) -> float:
    """A chained comparison, ``and`` and a floor division of floats."""
    return rel if 0.0 <= rel < num and not rel > 100.0 else rel - num * (rel // num)


def test_the_cuda_function_shares_its_subexpressions():
    source = xp.kernels.KernelBlock(boris).cuda("my_boris")
    assert source.startswith("__device__ inline void my_boris(double vx")
    assert "double *result2)" in source
    # 2 / (1 + |t|^2) is computed once, as are v + k
    assert source.count("2.0 /") == 1
    assert source.count("(vx + kx)") == 1


def test_the_metal_function_is_float32_with_references():
    source = xp.kernels.KernelBlock(boris).metal()
    assert source.startswith("inline void boris(float vx")
    assert "thread float& result0" in source
    assert "2.0f /" in source and "double" not in source
    assert "metal::floor" in xp.kernels.KernelBlock(wrapped).metal()


@pytest.mark.parametrize("periodic", [0, 1])
def test_the_numpy_version_is_the_block(periodic):
    rel = np.random.default_rng(0).uniform(-3.0, 12.0, 200)
    expected = [left_node(float(r), 9, periodic) for r in rel]
    np.testing.assert_array_equal(
        xp.kernels.KernelBlock(left_node).numpy()(rel, 9, periodic), expected
    )
    assert xp.kernels.KernelBlock(left_node).numpy()(3.7, 9, periodic) == 3
    expected = [wrapped(float(r), 9) for r in rel]
    np.testing.assert_allclose(
        xp.kernels.KernelBlock(wrapped).numpy()(rel, 9), expected, rtol=0, atol=1e-12
    )


def test_the_numpy_version_of_several_results():
    v, k, t = np.random.default_rng(1).normal(size=(3, 3, 50))
    got = np.array(xp.kernels.KernelBlock(boris).numpy()(*v, *k, *t))
    expected = np.array([boris(*v[:, i], *k[:, i], *t[:, i]) for i in range(50)]).T
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
def test_the_cuda_functions_run_like_the_block():
    block, cell = xp.kernels.KernelBlock(boris), xp.kernels.KernelBlock(left_node)
    source = (
        '#include "cunumpy/array_view.cuh"\n'
        + block.cuda()
        + "\n"
        + cell.cuda()
        + """
extern "C" __global__ void run(Array2D<double> v, Array2D<double> k, Array2D<double> t,
                               Array2D<double> out, Array1D<double> rel, Array1D<long long> left) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i >= v.shape[1]) return;
    boris(v(0, i), v(1, i), v(2, i), k(0, i), k(1, i), k(2, i), t(0, i), t(1, i), t(2, i),
          &out(0, i), &out(1, i), &out(2, i));
    left(i) = left_node(rel(i), 9, 0);
}"""
    )
    kernel = xp.kernels.CudaKernel(source, "run")
    rng = np.random.default_rng(2)
    v, k, t = (rng.normal(size=(3, 64)) for _ in range(3))
    rel = rng.uniform(-3.0, 12.0, 64)
    out, left = np.zeros((3, 64)), np.zeros(64, dtype=np.int64)
    emulate_cuda_kernel(kernel, v, k, t, out, rel, left, n_threads=64)
    expected = np.array([boris(*v[:, i], *k[:, i], *t[:, i]) for i in range(64)]).T
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(left, [left_node(float(r), 9, 0) for r in rel])


def not_a_scalar(x: "float[:]") -> float:
    return x


def two_statements(x: float) -> float:
    y = x
    return y


def integer_floor(a: int, b: int) -> int:
    return a // b


def test_what_is_not_a_block():
    with pytest.raises(TypeError, match="scalars"):
        xp.kernels.KernelBlock(not_a_scalar)
    with pytest.raises(TypeError, match="one return"):
        xp.kernels.KernelBlock(two_statements)
    with pytest.raises(NotImplementedError, match="//"):
        xp.kernels.KernelBlock(integer_floor).cuda()
