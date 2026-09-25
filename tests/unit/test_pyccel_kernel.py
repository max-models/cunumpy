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
    assert wrapped.outputs is None
    assert repr(wrapped) == (
        "PyccelKernel(kernel='my_kernel', use_cupy=False, outputs=None)"
    )


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
# Custom is_array predicate
# ---------------------------------------------------------------------------


class _HostArrayLike:
    """Stand-in for a custom host array type that isn't a `np.ndarray`."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


def test_default_is_array_ignores_custom_array_like_return_value():
    _skip_without_cupy()

    def kernel():
        return _HostArrayLike(np.ones(3))

    result = PyccelKernel(kernel, use_cupy=True)()
    assert isinstance(result, _HostArrayLike)
    assert xp.is_cpu(result.data)  # not moved back: not a np.ndarray


def test_custom_is_array_moves_custom_return_value_back_to_device():
    _skip_without_cupy()

    def kernel():
        return _HostArrayLike(np.ones(3))

    wrapped = PyccelKernel(
        kernel,
        use_cupy=True,
        is_array=lambda v: isinstance(v, _HostArrayLike),
    )
    result = wrapped()
    assert xp.is_gpu(result)
    assert wrapped.is_array is wrapped._is_array


# ---------------------------------------------------------------------------
# Declared outputs
# ---------------------------------------------------------------------------


def test_outputs_restricts_write_back_to_declared_arguments():
    """Inputs must not be copied back, even if the kernel writes to them."""
    _skip_without_cupy()

    def kernel(x, y, out):
        x[:] = 999.0  # a stray write to an input
        out[:] = x[0] + y[0]

    x = xp.to_cupy(np.ones(3))
    y = xp.to_cupy(np.full(3, 2.0))
    out = xp.to_cupy(np.zeros(3))

    PyccelKernel(kernel, outputs=(2,))(x, y, out)

    assert np.array_equal(xp.to_numpy(x), np.ones(3))
    assert np.array_equal(xp.to_numpy(out), np.full(3, 1001.0))


def test_outputs_accepts_negative_indices_and_keyword_names():
    _skip_without_cupy()

    def kernel(x, out):
        out[:] = 7.0

    positional = xp.to_cupy(np.zeros(2))
    PyccelKernel(kernel, outputs=(-1,))(xp.to_cupy(np.ones(2)), positional)
    assert np.array_equal(xp.to_numpy(positional), np.full(2, 7.0))

    keyword = xp.to_cupy(np.zeros(2))
    PyccelKernel(kernel, outputs=("out",))(xp.to_cupy(np.ones(2)), out=keyword)
    assert np.array_equal(xp.to_numpy(keyword), np.full(2, 7.0))


def test_empty_outputs_copies_nothing_back():
    _skip_without_cupy()

    def kernel(out):
        out[:] = 5.0

    arr = xp.to_cupy(np.zeros(2))
    PyccelKernel(kernel, outputs=())(arr)

    assert np.array_equal(xp.to_numpy(arr), np.zeros(2))


def test_outputs_traverse_nested_containers_and_objects():
    _skip_without_cupy()

    class Container:
        def __init__(self, data):
            self.data = data

    nested = xp.to_cupy(np.zeros(2))
    held = xp.to_cupy(np.zeros(2))

    def kernel(inp, pack, obj):
        pack[0]["m"][:] = 1.0
        obj.data[:] = 2.0

    PyccelKernel(kernel, object_modules=(Container.__module__,), outputs=(1, 2))(
        xp.to_cupy(np.ones(2)), [{"m": nested}], Container(held)
    )

    assert np.array_equal(xp.to_numpy(nested), np.full(2, 1.0))
    assert np.array_equal(xp.to_numpy(held), np.full(2, 2.0))


def test_array_aliased_into_an_output_is_written_back():
    """An input that is also the output must still come back."""
    _skip_without_cupy()

    def kernel(inp, out):
        out[:] = 6.0

    shared = xp.to_cupy(np.zeros(2))
    PyccelKernel(kernel, outputs=(1,))(shared, shared)

    assert np.array_equal(xp.to_numpy(shared), np.full(2, 6.0))


def test_misdeclared_output_index_raises():
    """`use_cupy=True` exercises the conversion path without needing a GPU:
    NumPy arguments need no conversion, so nothing is sent to a device."""
    wrapped = PyccelKernel(lambda out: None, use_cupy=True, outputs=(5,))

    with pytest.raises(IndexError, match="positional argument"):
        wrapped(np.zeros(2))


def test_misdeclared_output_name_raises():
    wrapped = PyccelKernel(lambda out: None, use_cupy=True, outputs=("nope",))

    with pytest.raises(KeyError, match="no such keyword argument"):
        wrapped(np.zeros(2))


@pytest.mark.parametrize("bad", [5, "out", (None,), (1.5,), (True,)])
def test_invalid_outputs_rejected_at_construction(bad):
    with pytest.raises(TypeError):
        PyccelKernel(lambda: None, outputs=bad)


def test_outputs_is_ignored_on_the_numpy_path():
    """Without conversion there is no copy-back to skip: the kernel writes
    straight into the caller's arrays."""
    arr = np.zeros(2)

    with xp.use_backend("numpy"):
        PyccelKernel(lambda out: out.__setitem__(slice(None), 4.0), outputs=())(arr)

    assert np.array_equal(arr, np.full(2, 4.0))


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


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_compiled_axpy_with_declared_output(kernels, backend):
    """The `outputs=` form from the issue: only `out` is copied back."""
    if backend == "cupy":
        _skip_without_cupy()

    axpy = PyccelKernel(kernels.axpy, outputs=(3,))

    with xp.use_backend(backend):
        x = xp.asarray(np.arange(6, dtype=np.float64))
        y = xp.asarray(np.ones(6, dtype=np.float64))
        out = xp.zeros(6, dtype=np.float64)

        axpy(2.5, x, y, out)

        assert np.allclose(xp.to_numpy(out), 2.5 * np.arange(6) + 1.0)
        # The inputs are untouched either way, but assert it explicitly since
        # they are the arrays whose copy-back we skipped.
        assert np.allclose(xp.to_numpy(x), np.arange(6))
        assert np.allclose(xp.to_numpy(y), np.ones(6))


def test_compiled_scale_inplace_needs_its_argument_declared(kernels):
    """`scale_inplace` writes to argument 0; declaring no outputs loses that
    update on the GPU, which is exactly what the declaration is for."""
    _skip_without_cupy()

    with xp.use_backend("cupy"):
        declared = xp.asarray(np.arange(4, dtype=np.float64))
        PyccelKernel(kernels.scale_inplace, outputs=(0,))(declared, 3.0)
        assert np.allclose(xp.to_numpy(declared), 3.0 * np.arange(4))

        undeclared = xp.asarray(np.arange(4, dtype=np.float64))
        PyccelKernel(kernels.scale_inplace, outputs=())(undeclared, 3.0)
        assert np.allclose(xp.to_numpy(undeclared), np.arange(4))


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
