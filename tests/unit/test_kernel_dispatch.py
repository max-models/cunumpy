"""Tests for `cunumpy.Kernel` (host/CUDA pairs) and `cunumpy.KernelCatalog`.

The host kernels here are plain Python functions (wrapped in `PyccelKernel`), so
the NumPy-side tests run everywhere; the CuPy-side tests need a GPU.
"""

import importlib
import sys
import textwrap

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import CudaKernel, Kernel, KernelCatalog, PyccelKernel

SCALE_CUDA = r"""
extern "C" __global__ void scale(double* x, double factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""


def scale(x, factor, n):
    """Host version of the `scale` kernel."""
    for i in range(n):
        x[i] *= factor


def _skip_without_cupy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def test_kernel_on_numpy():
    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    assert isinstance(kernel.host_kernel, PyccelKernel)  # callables are wrapped
    assert kernel.name == "scale" and kernel.has_cuda

    x = np.ones(4)
    with xp.use_backend("numpy"):
        assert kernel.get_kernel() is kernel.host_kernel
        kernel(x, 3.0, 4, n_threads=4)  # n_threads is ignored by the host kernel
        kernel(x, 2.0, 4)
    assert np.all(x == 6.0)


def test_kernel_arguments():
    with pytest.raises(ValueError, match="missing_cuda"):
        Kernel(scale, missing_cuda="ignore")
    with pytest.raises(TypeError, match="cuda_kernel must be a CudaKernel"):
        Kernel(scale, PyccelKernel(scale))
    assert Kernel(scale, name="other").name == "other"
    assert not Kernel(scale).has_cuda


def test_kernel_on_cupy():
    _skip_without_cupy()
    import cupy as cp

    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    x = cp.ones(300)
    with xp.use_backend("cupy"):
        assert kernel.get_kernel() is kernel.cuda_kernel
        kernel(x, 3, 300, n_threads=300)
        with pytest.raises(ValueError, match="n_threads is required"):
            kernel(x, 3, 300)
    assert cp.all(x == 3.0)


def test_missing_cuda_raises_on_cupy():
    _skip_without_cupy()
    kernel = Kernel(scale, cuda_path="kernels/scale/scale_cuda.cu")
    with (
        xp.use_backend("cupy"),
        pytest.raises(
            NotImplementedError,
            match="No CUDA version of kernel 'scale'.*scale_cuda.cu",
        ),
    ):
        kernel.get_kernel()


def test_missing_cuda_fallback_on_cupy():
    """missing_cuda="fallback" calls the host kernel via PyccelKernel (host copies)."""
    _skip_without_cupy()
    import cupy as cp

    kernel = Kernel(scale, missing_cuda="fallback")
    x = cp.ones(5)
    with xp.use_backend("cupy"):
        with pytest.warns(RuntimeWarning, match="copies its arrays to the host"):
            kernel(x, 4.0, 5)
        kernel(x, 0.5, 5)  # warned only once
    assert cp.all(x == 2.0)


# ---------------------------------------------------------------------------
# KernelCatalog
# ---------------------------------------------------------------------------


@pytest.fixture
def kernel_package(tmp_path, monkeypatch):
    """One folder per kernel: `scale` has a CUDA kernel, `shift` has not."""
    root = tmp_path / "demo_kernel_pkg"
    for name, body in (("scale", "x[i] *= a"), ("shift", "x[i] += a")):
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text("")
        (root / name / f"{name}_kernels.py").write_text(textwrap.dedent(f"""
                def {name}(x, a, n):
                    for i in range(n):
                        {body}
                """))
    (root / "scale" / "scale_cuda.cu").write_text(SCALE_CUDA)
    (root / "not_a_kernel").mkdir()
    (root / "__init__.py").write_text(
        "from cunumpy import KernelCatalog\n\n"
        "catalog = KernelCatalog.from_package(__name__)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("demo_kernel_pkg").catalog
    for module in [m for m in sys.modules if m.startswith("demo_kernel_pkg")]:
        del sys.modules[module]


@pytest.fixture
def kernel_package_factory(kernel_package):
    """Build catalogs of the `kernel_package` package with other options."""
    return lambda **options: KernelCatalog.from_package("demo_kernel_pkg", **options)


def test_catalog_from_package(kernel_package):
    catalog = kernel_package
    assert list(catalog) == ["scale", "shift"] and len(catalog) == 2
    assert "scale" in catalog and "not_a_kernel" not in catalog
    assert catalog.without_cuda == ["shift"]
    assert catalog["scale"].cuda_kernel.name == "scale"
    assert catalog["shift"].cuda_path.name == "shift_cuda.cu"

    x = np.ones(3)
    with xp.use_backend("numpy"):
        catalog["scale"](x, 2.0, 3)
        catalog["shift"](x, 1.0, 3)
    assert np.all(x == 3.0)


def test_catalog_on_cupy(kernel_package):
    _skip_without_cupy()
    import cupy as cp

    x = cp.ones(10)
    with xp.use_backend("cupy"):
        kernel_package["scale"](x, 5.0, 10, n_threads=10)
        with pytest.raises(NotImplementedError, match="shift"):
            kernel_package["shift"](x, 1.0, 10, n_threads=10)
    assert cp.all(x == 5.0)


def test_catalog_register():
    catalog = KernelCatalog()
    kernel = catalog.register(Kernel(scale))
    assert catalog["scale"] is kernel and catalog.without_cuda == ["scale"]
    with pytest.raises(KeyError, match="already registered"):
        catalog.register(Kernel(scale))
    with pytest.raises(TypeError, match="expected a Kernel"):
        catalog.register(scale)
    catalog.register(Kernel(scale), name="scale_again")
    assert dict(catalog).keys() == {"scale", "scale_again"}
    assert KernelCatalog({"x": kernel})["x"] is kernel


# ---------------------------------------------------------------------------
# host kernel options, launch configuration and compile_all
# ---------------------------------------------------------------------------


class Holder:
    """An application object holding an array, as passed to host kernels."""

    def __init__(self, x):
        self.x = x


def scale_holder(holder, factor, n):
    for i in range(n):
        holder.x[i] *= factor


def test_host_options():
    kernel = Kernel(scale, host_options={"outputs": (0,)})
    assert kernel.host_kernel.outputs == (0,)

    with pytest.raises(ValueError, match="host_options are for wrapping"):
        Kernel(PyccelKernel(scale), host_options={"outputs": (0,)})


def test_host_options_in_fallback_on_cupy():
    """object_modules lets the fallback find the device arrays inside objects."""
    _skip_without_cupy()
    import cupy as cp

    kernel = Kernel(
        scale_holder,
        missing_cuda="fallback",
        host_options={"object_modules": (__name__,), "outputs": (0,)},
    )
    holder = Holder(cp.ones(4))
    with xp.use_backend("cupy"), pytest.warns(RuntimeWarning):
        kernel(holder, 3.0, 4)
    assert cp.all(holder.x == 3.0)


def test_catalog_host_options(kernel_package_factory):
    catalog = kernel_package_factory(host_options={"outputs": (0,)})
    assert all(catalog[name].host_kernel.outputs == (0,) for name in catalog)

    catalog = kernel_package_factory(
        host_options=lambda name: {"outputs": (0,) if name == "scale" else ()}
    )
    assert catalog["scale"].host_kernel.outputs == (0,)
    assert catalog["shift"].host_kernel.outputs == ()


def test_kernel_launch_configuration_on_cupy():
    _skip_without_cupy()
    import cupy as cp

    kernel = Kernel(scale, CudaKernel(SCALE_CUDA, "scale"))
    x = cp.ones(300)
    with xp.use_backend("cupy"):
        kernel(x, 2.0, 300, grid=5, block=64)  # 320 threads
        kernel(x, 2.0, 300, n_threads=300, block=32, stream=cp.cuda.Stream.null)
        with pytest.raises(ValueError, match="n_threads is required"):
            kernel(x, 2.0, 300, block=32)
    assert cp.all(x == 4.0)


def test_compile(kernel_package_factory):
    catalog = kernel_package_factory()
    assert not Kernel(scale).compile()  # no CUDA kernel: nothing to compile

    if not xp.cupy_available():
        with pytest.raises(RuntimeError, match="CuPy is not installed"):
            catalog.compile_all()
        return

    assert catalog.compile_all() == ["scale"]
    assert catalog["scale"].cuda_kernel.is_compiled


def test_catalog_include_dirs(kernel_package, kernel_package_factory, tmp_path):
    """By default the source root of the package and the kernel folder."""
    kernel = kernel_package["scale"].cuda_kernel
    scale_dir = tmp_path / "demo_kernel_pkg" / "scale"
    assert kernel.include_dirs == (scale_dir, tmp_path)
    assert kernel.source_dir == scale_dir
    assert kernel.options == (f"-I{scale_dir}", f"-I{tmp_path}")

    extra = tmp_path / "headers"
    kernel = kernel_package_factory(include_dirs=[extra])["scale"].cuda_kernel
    assert kernel.include_dirs == (scale_dir, extra)
    kernel = kernel_package_factory(include_dirs=())["scale"].cuda_kernel
    assert kernel.include_dirs == (scale_dir,)


def test_catalog_kernel_includes_from_the_source_root(tmp_path, monkeypatch):
    """A kernel of `pkg.kernels` can `#include "pkg/common.cuh"`."""
    root = tmp_path / "demo_include_pkg"
    (root / "kernels" / "scale").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "common.cuh").write_text("#define FACTOR 2\n")
    (root / "kernels" / "__init__.py").write_text("")
    (root / "kernels" / "scale" / "__init__.py").write_text("")
    (root / "kernels" / "scale" / "scale_kernels.py").write_text(
        "def scale(x, a, n):\n    pass\n"
    )
    (root / "kernels" / "scale" / "scale_cuda.cu").write_text(
        '#include "demo_include_pkg/common.cuh"\n' + SCALE_CUDA
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        catalog = KernelCatalog.from_package("demo_include_pkg.kernels")
        kernel = catalog["scale"].cuda_kernel
        assert kernel.included_headers == (root / "common.cuh",)
        assert kernel.compile_options()[-1].startswith("-DCUNUMPY_INCLUDE_HASH=0x")
    finally:
        for module in [m for m in sys.modules if m.startswith("demo_include_pkg")]:
            del sys.modules[module]
