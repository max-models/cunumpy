"""Tests for `cunumpy.PyccelKernel`.

Two groups of tests live here:

* conversion tests, which use plain Python callables as stand-in kernels and
  run everywhere;
* end-to-end tests, which compile `pyccel_kernels.py` with pyccel and drive the
  compiled kernels through `PyccelKernel`. They are skipped when pyccel (or a
  working compiler) is unavailable.

The CuPy-side assertions can only be exercised on a machine with a GPU; the
NumPy-side assertions run everywhere.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import PyccelKernel

KERNEL_SOURCE = Path(__file__).parent / "pyccel_kernels.py"


def _skip_without_cupy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    """Compile `pyccel_kernels.py` with pyccel and return the compiled module.

    The source is copied into a temporary directory first so that pyccel's
    build artefacts (`__pyccel__/`) never land in the repository.
    """
    pyccel = pytest.importorskip("pyccel", reason="pyccel is not installed")

    import importlib.util
    import sys

    build_dir = tmp_path_factory.mktemp("pyccel_build")
    source = build_dir / KERNEL_SOURCE.name
    shutil.copy(KERNEL_SOURCE, source)

    spec = importlib.util.spec_from_file_location(source.stem, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[source.stem] = module
    spec.loader.exec_module(module)

    try:
        return pyccel.epyccel(module, language="c")
    except Exception as exc:  # noqa: BLE001 - no compiler / broken toolchain
        pytest.skip(f"pyccel could not compile the example kernels: {exc}")
    finally:
        sys.modules.pop(source.stem, None)


# ---------------------------------------------------------------------------
# Conversion behaviour (no pyccel needed)
# ---------------------------------------------------------------------------


def test_name_kernel_and_repr():
    def my_kernel(x):
        return x

    wrapped = PyccelKernel(my_kernel, use_cupy=False)

    assert wrapped.name == "my_kernel"
    assert wrapped.kernel is my_kernel
    assert wrapped.use_cupy is False
    assert repr(wrapped) == "PyccelKernel(kernel='my_kernel', use_cupy=False)"


def test_numpy_backend_calls_kernel_unchanged():
    """On the NumPy backend the arguments must reach the kernel untouched."""
    seen = {}

    def kernel(x, *, scale):
        seen["x"] = x
        seen["scale"] = scale
        return x * scale

    arr = np.arange(4, dtype=float)
    with xp.use_backend("numpy"):
        result = PyccelKernel(kernel)(arr, scale=2.0)

    assert seen["x"] is arr
    assert seen["scale"] == 2.0
    assert np.array_equal(result, arr * 2.0)


def test_use_cupy_follows_active_backend():
    wrapped = PyccelKernel(lambda: None)

    with xp.use_backend("numpy"):
        assert wrapped.use_cupy is False

    _skip_without_cupy()

    with xp.use_backend("cupy"):
        assert wrapped.use_cupy is True


def test_explicit_use_cupy_overrides_backend():
    wrapped = PyccelKernel(lambda: None, use_cupy=False)

    _skip_without_cupy()

    with xp.use_backend("cupy"):
        assert wrapped.use_cupy is False


def test_kernel_receives_numpy_arrays_when_given_cupy_arrays():
    _skip_without_cupy()

    seen = {}

    def kernel(x, y):
        seen["types"] = (type(x), type(y))

    gpu = xp.to_cupy(np.arange(4, dtype=float))
    PyccelKernel(kernel)(gpu, y=gpu)

    assert seen["types"] == (np.ndarray, np.ndarray)


def test_inplace_updates_are_copied_back_to_device():
    _skip_without_cupy()

    def kernel(out):
        out[:] = 42.0

    gpu = xp.to_cupy(np.zeros(5))
    PyccelKernel(kernel)(gpu)

    assert np.array_equal(xp.to_numpy(gpu), np.full(5, 42.0))


def test_returned_arrays_are_moved_back_to_device():
    _skip_without_cupy()

    def kernel(x):
        return x + 1.0, x.sum(), None

    gpu = xp.to_cupy(np.arange(3, dtype=float))
    arr, total, nothing = PyccelKernel(kernel)(gpu)

    assert xp.is_gpu(arr)
    assert np.array_equal(xp.to_numpy(arr), np.arange(3, dtype=float) + 1.0)
    assert total == 3.0
    assert nothing is None


def test_nested_containers_are_converted_and_written_back():
    _skip_without_cupy()

    def kernel(pair, mapping):
        pair[0][:] = 1.0
        pair[1][0][:] = 2.0
        mapping["a"][:] = 3.0
        assert all(isinstance(a, np.ndarray) for a in (pair[0], pair[1][0]))
        assert isinstance(mapping["a"], np.ndarray)

    first = xp.to_cupy(np.zeros(3))
    second = xp.to_cupy(np.zeros(3))
    third = xp.to_cupy(np.zeros(3))

    PyccelKernel(kernel)((first, [second]), {"a": third})

    assert np.array_equal(xp.to_numpy(first), np.full(3, 1.0))
    assert np.array_equal(xp.to_numpy(second), np.full(3, 2.0))
    assert np.array_equal(xp.to_numpy(third), np.full(3, 3.0))


def test_object_attributes_are_converted_when_module_is_listed():
    """Instances of listed modules are traversed; others are passed through."""
    _skip_without_cupy()

    class Container:
        def __init__(self, data):
            self.data = data
            self.label = "unchanged"

    holder = Container(xp.to_cupy(np.zeros(3)))

    def kernel(obj):
        assert isinstance(obj.data, np.ndarray)
        assert obj.label == "unchanged"
        obj.data[:] = 7.0

    # `Container` is defined in this test module, so use its own module name.
    PyccelKernel(kernel, object_modules=(Container.__module__,))(holder)

    assert xp.is_gpu(holder.data), "the caller's object must keep its device array"
    assert np.array_equal(xp.to_numpy(holder.data), np.full(3, 7.0))


def test_aliased_arguments_stay_a_single_host_array():
    """One device array passed twice must become one host array, so the
    kernel's in-place updates are not lost when copying back."""
    _skip_without_cupy()

    def kernel(a, b):
        assert a is b
        a += 1.0
        b += 1.0

    shared = xp.to_cupy(np.zeros(3))
    PyccelKernel(kernel)(shared, shared)

    assert np.array_equal(xp.to_numpy(shared), np.full(3, 2.0))


def test_aliasing_between_object_attribute_and_argument():
    _skip_without_cupy()

    class Container:
        def __init__(self, data):
            self.data = data

    holder = Container(xp.to_cupy(np.zeros(3)))

    def kernel(arr, obj):
        assert arr is obj.data
        obj.data[:] = 5.0

    PyccelKernel(kernel, object_modules=(Container.__module__,))(holder.data, holder)

    assert np.array_equal(xp.to_numpy(holder.data), np.full(3, 5.0))


def test_reference_cycles_do_not_recurse_forever():
    _skip_without_cupy()

    cyclic = [xp.to_cupy(np.zeros(2))]
    cyclic.append(cyclic)

    def kernel(items):
        assert items[1] is items, "the cycle must be preserved"
        items[0][:] = 4.0

    PyccelKernel(kernel)(cyclic)

    assert np.array_equal(xp.to_numpy(cyclic[0]), np.full(2, 4.0))


def test_dict_argument_is_not_mistaken_for_a_device_array():
    """Dict-like objects expose `.get`; detection must not rely on that."""
    seen = {}

    def kernel(cfg, arr):
        seen["cfg"] = cfg
        arr[:] = 8.0

    arr = np.zeros(2)
    PyccelKernel(kernel)({"n": 3}, arr)

    assert seen["cfg"] == {"n": 3}
    assert np.array_equal(arr, np.full(2, 8.0))


def test_unlisted_objects_are_passed_through_untouched():
    _skip_without_cupy()

    class Container:
        def __init__(self, data):
            self.data = data

    holder = Container(xp.to_cupy(np.zeros(3)))

    def kernel(obj):
        assert obj is holder
        assert xp.is_gpu(obj.data)

    PyccelKernel(kernel)(holder)


# ---------------------------------------------------------------------------
# End-to-end with real pyccel-compiled kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compiled_axpy(kernels, backend):
    if backend == "cupy":
        _skip_without_cupy()

    with xp.use_backend(backend):
        a = 2.5
        x = xp.asarray(np.arange(6, dtype=np.float64))
        y = xp.asarray(np.ones(6, dtype=np.float64))
        out = xp.zeros(6, dtype=np.float64)

        PyccelKernel(kernels.axpy)(a, x, y, out)

        assert np.allclose(xp.to_numpy(out), a * np.arange(6) + 1.0)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compiled_scale_inplace(kernels, backend):
    if backend == "cupy":
        _skip_without_cupy()

    with xp.use_backend(backend):
        x = xp.asarray(np.arange(4, dtype=np.float64))

        PyccelKernel(kernels.scale_inplace)(x, 3.0)

        assert xp.get_backend(x) == backend
        assert np.allclose(xp.to_numpy(x), 3.0 * np.arange(4))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compiled_dot_returns_scalar(kernels, backend):
    if backend == "cupy":
        _skip_without_cupy()

    with xp.use_backend(backend):
        x = xp.asarray(np.arange(5, dtype=np.float64))
        y = xp.asarray(np.arange(5, dtype=np.float64))

        result = PyccelKernel(kernels.dot)(x, y)

        assert isinstance(result, float)
        assert result == pytest.approx(float(np.arange(5) @ np.arange(5)))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compiled_matvec(kernels, backend):
    if backend == "cupy":
        _skip_without_cupy()

    mat_np = np.arange(6, dtype=np.float64).reshape(3, 2)
    vec_np = np.array([1.0, 2.0])

    with xp.use_backend(backend):
        mat = xp.asarray(mat_np)
        vec = xp.asarray(vec_np)
        out = xp.zeros(3, dtype=np.float64)

        PyccelKernel(kernels.matvec)(mat, vec, out)

        assert np.allclose(xp.to_numpy(out), mat_np @ vec_np)


def test_compiled_kernel_matches_pure_python(kernels):
    """The compiled kernel and its Python original must agree."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pyccel_kernels_ref", KERNEL_SOURCE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    x = np.random.rand(10)
    y = np.random.rand(10)

    expected = np.zeros(10)
    reference.axpy(1.5, x, y, expected)

    got = np.zeros(10)
    PyccelKernel(kernels.axpy, use_cupy=False)(1.5, x, y, got)

    assert np.allclose(got, expected)
