"""C host kernels (``<name>_c.c``): the signature, the wrapper, the cache and the dispatch.

Builds need a C compiler (``cc``); those tests are skipped without one.
"""

import shutil
import sys
import textwrap
import time

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _c_kernels as c_kernels

COMPILER = shutil.which("cc")
needs_compiler = pytest.mark.skipif(COMPILER is None, reason="no C compiler")

PYCCEL_SOURCE = '''\
def axpy(x: "float[:,:]", y: "float[:,:]", a: float, n: int, flag: bool, out: "int32[:]") -> None:
    """y += a x on the first n rows; out[0] = 1 if flag."""
    for i in range(n):
        for j in range(x.shape[1]):
            y[i, j] += a * x[i, j]
    if flag:
        out[0] = 1
'''

C_SOURCE = """\
#include <stdbool.h>
#include <stdint.h>
#include "scale.h"
void axpy(double *x, const int64_t *x_shape, double *y, const int64_t *y_shape,
          double a, int64_t n, bool flag, int32_t *out, const int64_t *out_shape) {
    (void)y_shape; (void)out_shape;
    const int64_t m = x_shape[1];
    #pragma omp simd
    for (int64_t k = 0; k < n * m; ++k) y[k] += SCALE * a * x[k];
    if (flag) out[0] = 1;
}
"""


def _package(tmp_path, monkeypatch, c_source=C_SOURCE, scale="1.0"):
    """A package ``cpkg`` with one kernel folder ``axpy`` (Pyccel and C versions)."""
    root = tmp_path / "src"
    folder = root / "cpkg" / "kernels" / "axpy"
    folder.mkdir(parents=True)
    (root / "cpkg" / "__init__.py").write_text("")
    (root / "cpkg" / "kernels" / "__init__.py").write_text("")
    (folder / "__init__.py").write_text(
        "import cunumpy as xp\n"
        "PACKAGE = xp.kernels.KernelPackage('cpkg-test')\n"
        "kernel = PACKAGE.from_folder(__name__)\n"
    )
    (folder / "axpy_pyccel.py").write_text(PYCCEL_SOURCE)
    (folder / "axpy_c.c").write_text(c_source)
    (folder / "scale.h").write_text(f"#define SCALE {scale}\n")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("CPKG_TEST_KERNEL_CACHE", str(tmp_path / "cache"))
    for name in [m for m in sys.modules if m == "cpkg" or m.startswith("cpkg.")]:
        monkeypatch.delitem(sys.modules, name)
    from cpkg.kernels.axpy import kernel

    return kernel, folder


def _arguments():
    x = np.arange(12.0).reshape(3, 4)
    return x, np.ones((3, 4)), 2.0, 2, True, np.zeros(1, dtype=np.int32)


# The signature and the wrapper ---------------------------------------------------------#


def test_parameters_from_the_pyccel_annotations():
    def kernel(
        a: "float[:,:,:]",
        b: "int[:]",
        c: "int32[:,:]",  # noqa: F821 -- a Pyccel type
        d: "bool[:]",
        e: float,
        f: int,
        g: "int32",  # noqa: F821 -- a Pyccel type
        h: bool,
    ) -> None:
        pass

    parameters = c_kernels.c_parameters(kernel)
    assert [(p.name, p.c_type, p.ndim) for p in parameters] == [
        ("a", "double", 3),
        ("b", "int64_t", 1),
        ("c", "int32_t", 2),
        ("d", "bool", 1),
        ("e", "double", 0),
        ("f", "int64_t", 0),
        ("g", "int32_t", 0),
        ("h", "bool", 0),
    ]
    source = c_kernels.wrapper_source("kernel", "kernel_c", parameters)
    assert "void kernel(double *a, const int64_t *a_shape, int64_t *b," in source
    assert "PyInit_kernel_c" in source


@pytest.mark.parametrize(
    "annotation", ['"complex[:]"', '"float[:,:](order=F)"', '"float[]"']
)
def test_annotations_without_a_c_form_are_refused(annotation):
    namespace = {}
    exec(f"def kernel(x: {annotation}) -> None:\n    pass\n", namespace)  # noqa: S102
    with pytest.raises(NotImplementedError):
        c_kernels.c_parameters(namespace["kernel"])


def test_return_values_and_keyword_only_parameters_are_refused():
    def returns(x: float) -> float:
        return x

    def keyword(*, x: float) -> None:
        pass

    for function in (returns, keyword):
        with pytest.raises(NotImplementedError):
            c_kernels.c_parameters(function)


# Builds -------------------------------------------------------------------------------#


@needs_compiler
def test_the_c_version_is_chosen_and_agrees_with_the_python_one(tmp_path, monkeypatch):
    kernel, _ = _package(tmp_path, monkeypatch)
    host = kernel.host_kernel.kernel
    assert "c" in kernel.implementations
    args = _arguments()
    kernel(*args)
    assert kernel.selected() == "c"
    expected = _arguments()
    host.get("python")(*expected)
    np.testing.assert_array_equal(args[1], expected[1])
    assert args[5][0] == 1 == expected[5][0]


@needs_compiler
def test_bad_arguments_raise(tmp_path, monkeypatch):
    kernel, _ = _package(tmp_path, monkeypatch)
    c = kernel.host_kernel.kernel.get("c")
    x, y, a, n, flag, out = _arguments()
    with pytest.raises(TypeError, match="element type"):
        c(x, y, a, n, flag, np.zeros(1))
    with pytest.raises(TypeError, match="C-contiguous"):
        c(x[:, ::2], y, a, n, flag, out)
    with pytest.raises(TypeError, match="dimensions"):
        c(x.ravel(), y, a, n, flag, out)
    with pytest.raises(TypeError, match="takes 6 arguments"):
        c(x, y)


@needs_compiler
def test_builds_are_cached_and_keyed_on_the_headers(tmp_path, monkeypatch):
    kernel, folder = _package(tmp_path, monkeypatch)
    cache = xp.kernels.CCache("cpkg-test")
    reference = kernel.host_kernel.kernel.get("python")
    first = cache.compile(folder / "axpy_c.c", "axpy", reference, ())
    builds = list(cache.cache_root().iterdir())
    assert len(builds) == 1

    def no_build(*args, **kwargs):
        raise AssertionError("rebuilt")

    monkeypatch.setattr(cache, "_build", no_build)
    second = cache.compile(folder / "axpy_c.c", "axpy", reference, ())
    assert second(*_arguments()) is None
    assert first is not None
    # an included header changes the build
    (folder / "scale.h").write_text("#define SCALE 3.0\n")
    monkeypatch.setattr(cache, "_build", c_kernels.CCache._build.__get__(cache))
    tripled = cache.compile(folder / "axpy_c.c", "axpy", reference, ())
    args = _arguments()
    tripled(*args)
    assert args[1][0, 1] == 1.0 + 3.0 * 2.0 * 1.0
    assert len(list(cache.cache_root().iterdir())) == 2


@needs_compiler
def test_the_flags_and_openmp_are_part_of_the_key(tmp_path, monkeypatch):
    _, folder = _package(tmp_path, monkeypatch)
    cache = xp.kernels.CCache("cpkg-test")
    source = folder / "axpy_c.c"
    keys = {cache.build_key(source, "wrapper", ())}
    monkeypatch.setenv("CPKG_TEST_C_FLAGS", "-O2")
    keys.add(cache.build_key(source, "wrapper", ()))
    assert cache.flags()[0] == "-O2"
    monkeypatch.setenv("CPKG_TEST_OPENMP", "0")
    assert "-fopenmp" not in cache.flags()
    keys.add(cache.build_key(source, "wrapper", ()))
    assert len(keys) == 3


@needs_compiler
def test_a_failing_build_falls_back_to_pyccel(tmp_path, monkeypatch):
    kernel, _ = _package(tmp_path, monkeypatch, c_source="this is not C\n")
    host = kernel.host_kernel.kernel
    assert not host.available("c")
    assert "building the C kernel" in str(host.errors["c"])
    args = _arguments()
    kernel(*args)
    assert kernel.selected() != "c"
    assert args[5][0] == 1


def test_without_a_compiler_the_c_version_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("CC", "cunumpy-no-such-compiler")
    kernel, _ = _package(tmp_path, monkeypatch)
    assert not kernel.host_kernel.kernel.available("c")
    assert kernel.selected() != "c"


@needs_compiler
def test_a_call_costs_about_a_microsecond(tmp_path, monkeypatch):
    kernel, _ = _package(tmp_path, monkeypatch)
    c = kernel.host_kernel.kernel.get("c")
    args = (np.zeros((1, 1)), np.zeros((1, 1)), 1.0, 1, False, np.zeros(1, np.int32))
    c(*args)
    start = time.perf_counter()
    for _ in range(2000):
        c(*args)
    assert (time.perf_counter() - start) / 2000 < 20e-6  # ctypes would be ~15 us


def test_the_kernel_source_is_documented_in_the_module():
    assert "double *x, const int64_t *x_shape" in textwrap.dedent(c_kernels.__doc__)
