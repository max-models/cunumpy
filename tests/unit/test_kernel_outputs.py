"""Kernel outputs by name, from annotations and from a kernel folder; struct fields in tests."""

import importlib
import sys
import textwrap
from typing import Final

import numpy as np
import pytest

from cunumpy.kernel_testing import _collect_arrays
from cunumpy.kernels import Kernel, PyccelKernel, outputs_from_annotations


def solve(a, b, out):
    out[:] = a + b


def host_arrays(kernel, args, kwargs=None):
    """The ids of the arrays the declared outputs of `kernel` reach."""
    return kernel._output_host_arrays(list(args), dict(kwargs or {}))


def test_output_name_finds_a_positional_argument():
    a, b, out = np.zeros(2), np.zeros(2), np.zeros(2)
    kernel = PyccelKernel(solve, outputs=("out",))
    assert host_arrays(kernel, (a, b, out)) == {id(out)}
    assert host_arrays(kernel, (a, b), {"out": out}) == {id(out)}


def test_output_index_finds_a_keyword_argument():
    a, b, out = np.zeros(2), np.zeros(2), np.zeros(2)
    kernel = PyccelKernel(solve, outputs=(2,))
    assert host_arrays(kernel, (a, b, out)) == {id(out)}
    assert host_arrays(kernel, (a, b), {"out": out}) == {id(out)}


def test_unknown_parameter_names_keep_the_old_rules():
    def hidden(*args):  # no named parameters, like a compiled kernel
        pass

    a, out = np.zeros(2), np.zeros(2)
    with pytest.raises(KeyError, match="declared by index"):
        host_arrays(PyccelKernel(hidden, outputs=("out",)), (a, out))
    with pytest.raises(IndexError, match="declared by name"):
        host_arrays(PyccelKernel(hidden, outputs=(1,)), (a,), {"out": out})
    named = PyccelKernel(hidden, outputs=("out",), parameters=["a", "out"])
    assert host_arrays(named, (a, out)) == {id(out)}


def test_unknown_name_is_reported():
    kernel = PyccelKernel(solve, outputs=("missing",))
    with pytest.raises(KeyError, match="no argument of that name"):
        host_arrays(kernel, (np.zeros(1),) * 3)


def test_outputs_from_annotations():
    def push(
        markers: "float[:, :]",
        weights: "Final[float[:]]",
        dt: "float",
        n: int,
        args: "MarkerArgs",  # noqa: F821
        scratch: Final[np.ndarray],
    ):
        pass

    assert outputs_from_annotations(push) == ("markers", "args")

    def unannotated(a, b):
        pass

    assert outputs_from_annotations(unannotated) is None
    assert outputs_from_annotations(len) is None  # no signature to read


@pytest.fixture
def folder_kernel(tmp_path, monkeypatch):
    folder = tmp_path / "output_pkg" / "scale"
    folder.mkdir(parents=True)
    (tmp_path / "output_pkg" / "__init__.py").write_text("")
    (folder / "__init__.py").write_text("")
    (folder / "scale_kernels.py").write_text(
        textwrap.dedent("""
            def scale(x: "float[:]", factor: "Final[float[:]]", a: "float", n: "int"):
                for i in range(n):
                    x[i] *= factor[i] * a
        """),
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "output_pkg.scale"
    for module in [m for m in sys.modules if m.startswith("output_pkg")]:
        del sys.modules[module]
    importlib.invalidate_caches()


def test_from_folder_reads_outputs_from_annotations(folder_kernel):
    kernel = Kernel.from_folder(folder_kernel, outputs="annotations")
    assert kernel.host_kernel.outputs == ("x",)
    x, factor = np.ones(3), np.full(3, 2.0)
    assert host_arrays(kernel.host_kernel, (x, factor, 3.0, 3)) == {id(x)}


def test_from_folder_outputs_by_name_use_the_host_parameters(folder_kernel):
    kernel = Kernel.from_folder(folder_kernel, outputs=("x",))
    x, factor = np.ones(3), np.full(3, 2.0)
    assert host_arrays(kernel.host_kernel, (x, factor, 3.0, 3)) == {id(x)}
    assert host_arrays(kernel.host_kernel, (), {"x": x}) == {id(x)}


def test_host_options_take_precedence_over_outputs(folder_kernel):
    kernel = Kernel.from_folder(
        folder_kernel, outputs="annotations", host_options={"outputs": ("factor",)}
    )
    assert kernel.host_kernel.outputs == ("factor",)


def test_from_folder_rejects_an_unknown_outputs_string(folder_kernel):
    with pytest.raises(ValueError, match="annotations"):
        Kernel.from_folder(folder_kernel, outputs="all")


# --- comparing only some fields of a struct argument -------------------------


class Markers:
    def __init__(self):
        self.positions = np.zeros((3, 2))
        self.buffer = np.ones(5)
        self.n = 3


def test_collect_arrays_by_parameter_name_and_field():
    markers, weights = Markers(), np.zeros(3)
    args = (markers, weights)
    names = ["markers", "weights"]

    both = _collect_arrays(args, ("markers",), names)
    assert set(both) == {"argument 0.positions", "argument 0.buffer"}

    only = _collect_arrays(args, ("markers.positions", "weights"), names)
    assert set(only) == {"argument 0.positions", "argument 1"}
    assert only["argument 0.positions"] is markers.positions

    assert set(_collect_arrays(args, ("0.buffer",), names)) == {"argument 0.buffer"}
    assert set(_collect_arrays(args, (-1,), names)) == {"argument 1"}


def test_collect_arrays_field_and_name_errors():
    args = (Markers(),)
    with pytest.raises(KeyError, match="no array field 'velocities'"):
        _collect_arrays(args, ("markers.velocities",), ["markers"])
    with pytest.raises(KeyError, match="not a parameter"):
        _collect_arrays(args, ("particles",), ["markers"])
    with pytest.raises(KeyError, match="names are unknown|unknown"):
        _collect_arrays(args, ("markers",), None)
