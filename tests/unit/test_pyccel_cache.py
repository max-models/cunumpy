"""PyccelCache: compile once, keep the build on disk, reuse it in later processes."""

import importlib.metadata
import importlib.util
import os
import stat
import sys
import tempfile

import pytest

import cunumpy as xp
from cunumpy import _pyccel_cache

SOURCE = """
def add_one(x: float) -> float:
    return x + 1.0
"""


@pytest.fixture
def make_module(tmp_path):
    """Write `source` as an importable module and return the module."""
    created = []

    def make(source=SOURCE, name="tiny_kernel"):
        folder = tmp_path / f"src{len(created)}"
        folder.mkdir()
        path = folder / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        created.append(name)
        return module

    yield make
    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    cache = xp.kernels.PyccelCache("cunumpy-test")
    monkeypatch.setenv(cache.cache_env, str(tmp_path / "cache"))
    monkeypatch.delenv(cache.openmp_env, raising=False)
    return cache


def test_environment_variables_follow_the_package_name():
    cache = xp.kernels.PyccelCache("tree-d-3d")
    assert (cache.cache_env, cache.openmp_env) == (
        "TREE_D_3D_KERNEL_CACHE",
        "TREE_D_3D_OPENMP",
    )
    custom = xp.kernels.PyccelCache("my.sim", env_prefix="SIM")
    assert custom.cache_env == "SIM_KERNEL_CACHE"
    assert xp.kernels.PyccelCache("my.sim").env_prefix == "MY_SIM"
    with pytest.raises(ValueError):
        xp.kernels.PyccelCache("")


def test_cache_directories(monkeypatch, tmp_path):
    cache = xp.kernels.PyccelCache("pkg")
    monkeypatch.delenv(cache.cache_env, raising=False)
    assert cache.cache_root().parts[-3:] == (".cache", "pkg", "kernels")
    monkeypatch.setenv(cache.cache_env, str(tmp_path / "elsewhere"))
    assert cache.cache_root() == tmp_path / "elsewhere"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    assert cache.fallback_root() == tmp_path / "tmp" / "pkg-kernels"


@pytest.mark.parametrize(
    ("value", "enabled"), [("1", True), ("0", False), ("off", False)]
)
def test_openmp_setting(monkeypatch, value, enabled):
    cache = xp.kernels.PyccelCache("pkg")
    monkeypatch.setenv(cache.openmp_env, value)
    assert cache.openmp_enabled() is enabled


def test_key_changes_with_the_source_and_the_openmp_setting(
    make_module, cache, monkeypatch
):
    pytest.importorskip("pyccel")
    module = make_module()
    key = cache.build_key(module)
    assert cache.build_key(module) == key
    assert cache.build_key(module, "fortran") != key
    monkeypatch.setenv(cache.openmp_env, "0")
    assert cache.build_key(module) != key
    monkeypatch.delenv(cache.openmp_env)
    edited = make_module(SOURCE.replace("1.0", "2.0"), name="tiny_kernel_edited")
    edited.__name__ = module.__name__  # same name, other source
    assert cache.build_key(edited) != key


def test_a_module_calling_the_openmp_runtime_keeps_openmp(
    make_module, cache, monkeypatch
):
    pytest.importorskip("pyccel")
    module = make_module(
        "from pyccel.stdlib.internal.openmp import omp_get_num_threads\n"
    )
    monkeypatch.setenv(cache.openmp_env, "0")
    assert cache._openmp_for(module)


def test_compiles_once_and_reuses_the_build(make_module, cache, monkeypatch):
    pytest.importorskip("pyccel")
    module = make_module()
    compiled = cache.compile(module)
    assert compiled.add_one(1.0) == 2.0
    assert compiled is not module
    (build,) = [p for p in cache.cache_root().iterdir() if not p.name.startswith(".")]
    assert build.name == f"tiny_kernel-{cache.build_key(module)}"

    def no_build(*args, **kwargs):
        raise AssertionError("the cached build should have been loaded")

    # A new instance, as in a later process, loads the stored build.
    later = xp.kernels.PyccelCache("cunumpy-test")
    monkeypatch.setattr(xp.kernels.PyccelCache, "_build", no_build)
    assert later(module).add_one(2.0) == 3.0


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root ignores directory permissions",
)
def test_unwritable_cache_builds_in_the_temp_dir(tmp_path, monkeypatch, make_module):
    pytest.importorskip("pyccel")
    cache = xp.kernels.PyccelCache("cunumpy-test")
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv(cache.cache_env, str(readonly / "kernels"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    module = make_module()
    try:
        assert cache.compile(module).add_one(1.0) == 2.0
        assert any((tmp_path / "tmp" / "cunumpy-test-kernels").iterdir())
        assert cache.compile(module).add_one(2.0) == 3.0  # reuses the fallback build
    finally:
        readonly.chmod(stat.S_IRWXU)


def test_a_failed_build_raises_with_the_compiler_output(make_module, cache):
    pytest.importorskip("pyccel")
    module = make_module(
        "def broken(x: float) -> float:\n    return x.missing_attribute\n"
    )
    with pytest.raises(RuntimeError, match="pyccel"):
        cache.compile(module)
    assert not [p for p in cache.cache_root().iterdir() if not p.name.startswith(".")]


def test_missing_pyccel_is_an_import_error(make_module, cache, monkeypatch):
    module = make_module()

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_pyccel_cache.importlib.metadata, "version", not_installed)
    with pytest.raises(ImportError):
        cache.compile(module)


def test_the_cache_is_a_host_compiler(make_module, cache):
    pytest.importorskip("pyccel")
    module = make_module()
    kernel = xp.kernels.CompiledHostKernel(module, "add_one", cache)
    assert kernel.compiled
    assert kernel(4.0) == 5.0
