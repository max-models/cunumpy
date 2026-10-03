"""Tests for `xp.rng.random_streams`: one seeded generator per process and backend."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.rng import BIT_GENERATORS, RandomStreams, random_streams


@pytest.fixture
def streams():
    return RandomStreams()


def _draws(streams):
    return (
        streams.random((4, 2)),
        streams.normal(1.0, 2.0, 5),
        streams.uniform(-1.0, 1.0, 3),
        streams.standard_normal(2),
    )


def test_exported():
    assert isinstance(random_streams, RandomStreams)
    assert "random_streams" in xp.rng.__all__
    assert repr(RandomStreams()) == "RandomStreams(not seeded)"


def test_same_seed_and_rank_give_the_same_draws(streams):
    streams.seed(42, rank=0)
    first = _draws(streams)
    streams.seed(42, rank=0)
    for a, b in zip(first, _draws(streams)):
        np.testing.assert_array_equal(a, b)
    assert "spawn_key=(0,)" in repr(streams)


def test_ranks_and_seeds_give_different_streams(streams):
    streams.seed(42, rank=0)
    rank0 = streams.random(8)
    streams.seed(42, rank=1)
    rank1 = streams.random(8)
    streams.seed(43, rank=0)
    other = streams.random(8)
    assert not np.array_equal(rank0, rank1) and not np.array_equal(rank0, other)


def test_seed_also_seeds_numpy_global_state(streams):
    streams.seed(7)
    a = np.random.random(3)
    streams.seed(7)
    np.testing.assert_array_equal(np.random.random(3), a)


@pytest.mark.parametrize("name", BIT_GENERATORS)
def test_bit_generators(streams, name):
    streams.seed(42, bit_generator=name)
    assert type(streams.generator("numpy").bit_generator).__name__ == name
    first = streams.random(4)
    streams.seed(42, bit_generator=name)
    np.testing.assert_array_equal(streams.random(4), first)


def test_unknown_bit_generator_and_backend(streams):
    with pytest.raises(ValueError, match="Unknown bit generator"):
        streams.seed(42, bit_generator="Mersenne")
    with pytest.raises(ValueError, match="backend must be"):
        streams.generator("jax")


def test_components_share_the_process_generator_unless_seeded(streams):
    streams.seed(1)
    assert streams.make_generator() is streams.generator()
    own = streams.make_generator(123)
    np.testing.assert_array_equal(own.random(3), np.random.default_rng(123).random(3))
    assert isinstance(streams.make_generator(backend="numpy"), np.random.Generator)


def test_unseeded_streams_still_work(streams):
    assert streams.random(3).shape == (3,)


class _MinimalGenerator:
    """Only ``random`` and ``standard_normal``, like some CuPy versions."""

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def random(self, size=None):
        return self._rng.random(size)

    def standard_normal(self, size=None):
        return self._rng.standard_normal(size)


def test_normal_and_uniform_fall_back(streams):
    rng = _MinimalGenerator()
    values = streams.uniform(2.0, 3.0, 1000, rng=rng)
    assert values.min() >= 2.0 and values.max() < 3.0
    assert abs(streams.normal(5.0, 0.1, 1000, rng=rng).mean() - 5.0) < 0.05


def test_cupy_generator_on_gpu(streams):
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    with xp.use_backend("cupy"):
        streams.seed(42, rank=3)
        first = streams.normal(0.0, 1.0, 100)
        assert isinstance(first, cp.ndarray)
        streams.seed(42, rank=3)
        cp.testing.assert_array_equal(streams.normal(0.0, 1.0, 100), first)
