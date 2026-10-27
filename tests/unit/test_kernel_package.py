"""KernelPackage: the cache and launch options shared by a package's kernel folders."""

import sys
import textwrap

import pytest

import cunumpy as xp

HOST = """
def add_one(x: "float[:]"):
    for i in range(x.shape[0]):
        x[i] += 1.0
"""


@pytest.fixture
def kernel_folder(tmp_path, monkeypatch):
    """A package ``tiny_pkg`` with one kernel folder ``tiny_pkg.add_one``."""
    folder = tmp_path / "tiny_pkg" / "add_one"
    folder.mkdir(parents=True)
    (tmp_path / "tiny_pkg" / "__init__.py").write_text("")
    (folder / "__init__.py").write_text("")
    (folder / "add_one_pyccel.py").write_text(textwrap.dedent(HOST))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "tiny_pkg.add_one"
    for name in [name for name in sys.modules if name.startswith("tiny_pkg")]:
        del sys.modules[name]


def test_options_share_one_cache():
    package = xp.kernels.KernelPackage("my-sim")
    options = package.options
    assert options["compile_host"] is package.cache
    assert package.cache.cache_env == "MY_SIM_KERNEL_CACHE"
    assert options["host_suffix"] == "_pyccel"
    assert options["dispatch"] == "arrays"
    assert options["block_size"] == 256
    assert options["n_threads_from"] == "last_axis"
    assert repr(package) == "KernelPackage('my-sim')"


def test_env_prefix_reaches_the_cache():
    package = xp.kernels.KernelPackage("my-sim", env_prefix="SIM")
    assert package.cache.openmp_env == "SIM_OPENMP"


def test_from_folder_runs_the_python_version(kernel_folder):
    package = xp.kernels.KernelPackage("tiny-pkg")
    kernel = package.from_folder(kernel_folder, missing_cuda="fallback")
    x = xp.zeros(3)
    with xp.kernels.use_host_kernel_implementation("python"):
        kernel(x)
    assert x.tolist() == [1.0, 1.0, 1.0]


def test_metal_kernel_defaults_and_overrides():
    package = xp.kernels.KernelPackage("my-sim", threadgroup=128)
    kernel = package.metal_kernel("scale", "y[i] = 2 * x[i];", ["x"], ["y"])
    assert kernel.name == "scale"
    assert kernel.threadgroup == (128, 1, 1)
    assert kernel.float64 == "cast"
    assert kernel.init_value == 0
    exact = package.metal_kernel("scale", "y[i] = x[i];", ["x"], ["y"], float64="error")
    assert exact.float64 == "error"


def test_source_roots_find_packages_without_importing():
    (root,) = xp.kernels.source_roots("cunumpy")
    assert (root / "cunumpy" / "__init__.py").is_file()
    with pytest.raises(ModuleNotFoundError):
        xp.kernels.source_roots("no_such_package_here")


def test_shared_memory_fits_the_default_block():
    assert xp.cuda.shared_memory_fits(6144)
    assert not xp.cuda.shared_memory_fits(6145)
    assert xp.cuda.shared_memory_fits(12288, itemsize=4)
