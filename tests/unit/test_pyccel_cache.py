"""PyccelCache: compile once, keep the build on disk, reuse it in later processes."""

import importlib.machinery
import importlib.metadata
import importlib.util
import os
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace

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
    # serial builds: these tests are about the cache, and a runner may have no
    # OpenMP runtime (macOS without Homebrew's libomp)
    monkeypatch.setenv(cache.openmp_env, "0")
    return cache


def test_environment_variables_follow_the_package_name():
    cache = xp.kernels.PyccelCache("tree-d-3d")
    assert (cache.cache_env, cache.openmp_env, cache.flags_env) == (
        "TREE_D_3D_KERNEL_CACHE",
        "TREE_D_3D_OPENMP",
        "TREE_D_3D_FLAGS",
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
    monkeypatch.setenv(cache.openmp_env, "1")
    assert cache.build_key(module) != key
    monkeypatch.setenv(cache.openmp_env, "0")
    edited = make_module(SOURCE.replace("1.0", "2.0"), name="tiny_kernel_edited")
    edited.__name__ = module.__name__  # same name, other source
    assert cache.build_key(edited) != key


def test_compiler_flags_change_the_key_and_reach_pyccel(
    make_module, cache, monkeypatch
):
    # no compiler needed: the command is captured instead of run
    pytest.importorskip("pyccel")
    module = make_module()
    monkeypatch.delenv(cache.flags_env, raising=False)
    assert cache.flags() == ""
    key = cache.build_key(module)
    monkeypatch.setenv(cache.flags_env, "  -march=native   -O2 ")
    assert cache.flags() == "-march=native -O2"
    assert cache.build_key(module) != key
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="captured")

    monkeypatch.setattr(_pyccel_cache.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="captured"):
        cache.compile(module)
    assert "--flags=-march=native -O2" in calls[0]


def test_key_changes_with_the_numpy_version(make_module, cache, monkeypatch):
    # a build is compiled against one NumPy's C API
    monkeypatch.setattr(_pyccel_cache, "_pyccel_version", lambda: "test")
    module = make_module()
    monkeypatch.setattr(_pyccel_cache, "_numpy_version", lambda: "2.4.0")
    key = cache.build_key(module)
    monkeypatch.setattr(_pyccel_cache, "_numpy_version", lambda: "2.5.0")
    assert cache.build_key(module) != key


def _fake_build(content):
    """A ``PyccelCache._build`` that writes `content` as the extension module."""

    def build(self, module, language, staging):
        output = staging / "build"
        output.mkdir()
        stem = module.__name__.rsplit(".", 1)[-1]
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        (output / f"{stem}{suffix}").write_bytes(content)
        return output

    return build


def test_a_build_that_cannot_be_imported_is_not_stored(make_module, cache, monkeypatch):
    # no compiler needed: the "build" is a file no interpreter can import
    monkeypatch.setattr(_pyccel_cache, "_pyccel_version", lambda: "test")
    monkeypatch.setattr(xp.kernels.PyccelCache, "_build", _fake_build(b"not a library"))
    with pytest.raises(RuntimeError, match="not stored"):
        cache.compile(make_module())
    assert not list(cache.cache_root().iterdir())


def test_a_build_that_crashes_the_interpreter_names_the_signal(
    make_module, cache, monkeypatch
):
    monkeypatch.setattr(_pyccel_cache, "_pyccel_version", lambda: "test")
    monkeypatch.setattr(xp.kernels.PyccelCache, "_build", _fake_build(b""))
    crashed = SimpleNamespace(returncode=-11, stdout="", stderr="Segmentation fault")
    monkeypatch.setattr(_pyccel_cache.subprocess, "run", lambda *a, **k: crashed)
    with pytest.raises(RuntimeError, match="signal 11(.|\n)*libpython"):
        cache.compile(make_module())
    assert not list(cache.cache_root().iterdir())


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


def test_no_lock_files_are_left_in_the_working_directory(
    make_module, cache, tmp_path, monkeypatch
):
    # Pyccel writes .lock_acquisition.lock (and stc.lock for kernels that use
    # its STC library) to the directory it runs in
    pytest.importorskip("pyccel")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    module = make_module()
    assert cache.compile(module).add_one(1.0) == 2.0
    assert list(work.iterdir()) == []


@pytest.mark.parametrize(("openmp", "flag"), [("1", True), ("0", False)])
def test_pyccel_runs_in_the_staging_folder_with_the_openmp_flag(
    make_module, cache, monkeypatch, openmp, flag
):
    # no compiler needed: the command is captured instead of run
    pytest.importorskip("pyccel")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="captured")

    monkeypatch.setattr(_pyccel_cache.subprocess, "run", fake_run)
    monkeypatch.setenv(cache.openmp_env, openmp)
    with pytest.raises(RuntimeError, match="captured"):
        cache.compile(make_module())
    ((command, kwargs),) = calls
    assert ("--openmp" in command) is flag
    staging = kwargs["cwd"]
    assert staging.is_absolute() and staging.parent == cache.cache_root()
    assert not staging.exists()  # removed after the failed build


def test_a_relative_cache_directory_works(make_module, tmp_path, monkeypatch):
    pytest.importorskip("pyccel")
    cache = xp.kernels.PyccelCache("cunumpy-test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(cache.cache_env, "relative/cache")
    monkeypatch.setenv(cache.openmp_env, "0")
    module = make_module()
    assert cache.compile(module).add_one(1.0) == 2.0
    assert any((tmp_path / "relative" / "cache").iterdir())


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
    monkeypatch.setenv(cache.openmp_env, "0")
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


# A kernel that imports functions of another package: the package `kdeps` with an
# @inline helper, a plain helper that imports from a third module, and a relative import.
DEPENDENCY_FILES = {
    "kdeps/__init__.py": "",
    "kdeps/base.py": """
def scale() -> float:
    return 3.0
""",
    "kdeps/helpers.py": """
from pyccel.decorators import inline

from .base import scale


@inline
def twice(x: float) -> float:
    return 2.0 * x


def add_scaled(x: float, y: float) -> float:
    return x + scale() * y
""",
}

KERNEL_WITH_DEPENDENCIES = """
import numpy as np

from kdeps.helpers import add_scaled, twice


def combine(x: float, y: float) -> float:
    return twice(x) + add_scaled(0.0, y) + np.sqrt(4.0)
"""


@pytest.fixture
def dependency_package(tmp_path, monkeypatch):
    """Write the package `kdeps` on sys.path and return its folder."""
    root = tmp_path / "deps"
    for name, text in DEPENDENCY_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.syspath_prepend(str(root))
    yield root / "kdeps"
    for name in [n for n in sys.modules if n == "kdeps" or n.startswith("kdeps.")]:
        sys.modules.pop(name)


def test_the_dependencies_are_found_in_build_order(
    make_module, cache, dependency_package
):
    pytest.importorskip("pyccel")  # the helpers import pyccel.decorators
    module = make_module(KERNEL_WITH_DEPENDENCIES, name="kernel_with_deps")
    dependencies = cache.dependencies(module)
    # numpy and pyccel are not built with the kernel; base before helpers
    assert list(dependencies) == ["kdeps.base", "kdeps.helpers"]
    assert dependencies["kdeps.base"] == DEPENDENCY_FILES["kdeps/base.py"]
    assert cache.dependencies(make_module(SOURCE)) == {}


def test_a_kernel_builds_with_its_dependencies_and_inlines_them(
    make_module, cache, dependency_package
):
    pytest.importorskip("pyccel")
    module = make_module(KERNEL_WITH_DEPENDENCIES, name="kernel_with_deps")
    assert module.combine(1.0, 2.0) == 10.0  # the Python version
    compiled = cache.compile(module)
    assert compiled.combine(1.0, 2.0) == 10.0
    assert compiled is not module
    (build,) = [p for p in cache.cache_root().iterdir() if not p.name.startswith(".")]
    # only the kernel's extension is kept: it contains the dependencies
    extensions = [p.name for p in build.iterdir() if p.suffix in (".so", ".pyd")]
    assert len(extensions) == 1 and extensions[0].startswith("kernel_with_deps")
    # the @inline helper is inlined, the plain one called
    generated = (build / "__pyccel__" / "kernel_with_deps.c").read_text()
    assert "twice(" not in generated
    assert "add_scaled(" in generated


def test_editing_a_dependency_rebuilds_the_kernel(
    make_module, cache, dependency_package
):
    pytest.importorskip("pyccel")
    module = make_module(KERNEL_WITH_DEPENDENCIES, name="kernel_with_deps")
    key = cache.build_key(module)
    assert cache.compile(module).combine(1.0, 2.0) == 10.0
    (dependency_package / "base.py").write_text(
        DEPENDENCY_FILES["kdeps/base.py"].replace("3.0", "5.0")
    )
    assert cache.build_key(module) != key
    assert cache.compile(module).combine(1.0, 2.0) == 14.0


def test_import_cycles_are_an_error(cache, dependency_package, tmp_path):
    (dependency_package / "base.py").write_text(
        "from kdeps.helpers import twice\n" + DEPENDENCY_FILES["kdeps/base.py"]
    )
    # not executed: Python would refuse the cycle on import
    path = tmp_path / "kernel_with_deps.py"
    path.write_text(KERNEL_WITH_DEPENDENCIES)
    module = ModuleType("kernel_with_deps")
    module.__file__ = str(path)
    with pytest.raises(
        ImportError, match="kdeps.helpers -> kdeps.base -> kdeps.helpers"
    ):
        cache.dependencies(module)


def test_the_rewritten_imports_keep_the_line_numbers(dependency_package):
    source = (
        "import math\n"
        "from kdeps.helpers import (\n"
        "    add_scaled,\n"
        "    twice as double,\n"
        ")\n"
        "from .base import scale\n"
        "x = 1\n"
    )
    rewritten = _pyccel_cache._rewrite_imports(source, "kdeps")
    assert rewritten.splitlines() == [
        "import math",
        "from kdeps__helpers import add_scaled, twice as double",
        "",
        "",
        "",
        "from kdeps__base import scale",
        "x = 1",
    ]
