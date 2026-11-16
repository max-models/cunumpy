"""Tests for nested containers and scalar objectives in the framework adapters."""

from collections import OrderedDict, namedtuple

import numpy as np
import pytest

from cunumpy.interop._common import is_array, scalar_value, tree_map

Pair = namedtuple("Pair", "a b")


def _tree():
    return {
        "fields": [np.arange(3.0), (np.ones(2), "label")],
        "pair": Pair(np.zeros(1), 2.5),
        "ordered": OrderedDict(x=np.float64(1.0)),
        "none": None,
    }


def test_tree_map_keeps_structure_and_non_arrays():
    result = tree_map(lambda a: ("converted", a.shape), _tree(), is_array)
    assert result["fields"][0] == ("converted", (3,))
    assert isinstance(result["fields"][1], tuple)
    assert result["fields"][1][1] == "label"
    assert isinstance(result["pair"], Pair) and result["pair"].b == 2.5
    assert isinstance(result["ordered"], OrderedDict)
    assert result["ordered"]["x"] == ("converted", ())
    assert result["none"] is None


def test_tree_map_keeps_list_and_tuple_subclasses():
    class Items(list):
        pass

    result = tree_map(lambda a: 0, Items([np.ones(1)]), is_array)
    assert type(result) is Items and result == [0]


def test_scalar_value():
    assert scalar_value(1.5) == 1.5
    assert scalar_value(np.ones((1, 1))).shape == ()
    with pytest.raises(ValueError, match="one element"):
        scalar_value(np.ones(2))
    with pytest.raises(TypeError, match="a number or an array"):
        scalar_value("x")


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    from cunumpy.interop import jax as adapter

    enabled = jax.config.read("jax_enable_x64")
    adapter.enable_x64()
    yield jax, adapter
    jax.config.update("jax_enable_x64", enabled)


def test_jax_tree_round_trip(jax_x64):
    jax, adapter = jax_x64
    tree = _tree()
    value = adapter.tree_to_jax(tree)
    assert isinstance(value["fields"][0], jax.Array)
    assert value["fields"][0].dtype == np.float64
    assert isinstance(value["pair"], Pair) and value["pair"].b == 2.5
    assert value["fields"][1][1] == "label" and value["none"] is None
    back = adapter.tree_from_jax(value)
    np.testing.assert_array_equal(back["fields"][0], tree["fields"][0])
    assert isinstance(back["fields"][0], np.ndarray)
    assert back["pair"].b == 2.5


def test_jax_tree_dtype_and_scalar(jax_x64):
    jax, adapter = jax_x64
    value = adapter.tree_to_jax([np.arange(2.0)], dtype="float32")
    assert value[0].dtype == np.float32
    objective = adapter.scalar(np.array([0.5]))
    assert isinstance(objective, jax.Array) and objective.shape == ()
    assert adapter.scalar(2.0).dtype == np.float64
    assert adapter.scalar(2.0, dtype="float32").dtype == np.float32
    adapter.enable_x64(False)
    with pytest.raises(TypeError, match="float64 into float32"):
        adapter.scalar(2.0)


def test_torch_tree_round_trip_and_scalar():
    torch = pytest.importorskip("torch")
    from cunumpy.interop import torch as adapter

    tree = _tree()
    value = adapter.tree_to_torch(tree)
    assert isinstance(value["fields"][0], torch.Tensor)
    assert value["fields"][0].dtype == torch.float64
    assert isinstance(value["pair"], Pair) and value["fields"][1][1] == "label"
    tensor = torch.ones(2, requires_grad=True)
    back = adapter.tree_from_torch({"t": tensor, "n": 3})
    assert isinstance(back["t"], np.ndarray) and back["n"] == 3
    objective = adapter.scalar(np.float64(1.5))
    assert objective.shape == () and objective.dtype == torch.float64
    assert adapter.scalar(1, dtype=torch.float32).dtype == torch.float32


def test_mlx_tree_round_trip_and_scalar():
    mx = pytest.importorskip("mlx.core")
    from cunumpy.interop import mlx as adapter

    value = adapter.tree_to_mlx(
        {"a": np.arange(3, dtype=np.float32), "s": "x"}, device=mx.cpu
    )
    assert isinstance(value["a"], mx.array) and value["s"] == "x"
    back = adapter.tree_from_mlx(value)
    np.testing.assert_array_equal(back["a"], np.arange(3, dtype=np.float32))
    objective = adapter.scalar(1.5, device=mx.cpu)
    assert objective.shape == () and objective.dtype == mx.float64
