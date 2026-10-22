"""cunumpy's scope-profiler regions: off by default, recorded once the
application enables the ``"cunumpy"`` tag."""

import numpy as np
import pytest
import scope_profiler as sp

import cunumpy as xp


@pytest.fixture
def profiler():
    """A fresh scope-profiler state, reset again afterwards so that no
    configuration leaks into other tests."""
    sp.reset()
    yield
    sp.reset()


def _cunumpy_regions(**setup_options):
    sp.setup(output=None, **setup_options)
    keys = np.asarray([2, 0, 1, 0])
    with xp.use_backend("numpy"):
        xp.algorithms.sort_by_key(keys, np.arange(4.0))
        xp.kernels.Kernel(lambda a: None, name="noop")(keys)
    results = sp.finalize(verbose=False, return_results=True)
    return [name for name in results.region_names if name.startswith("cunumpy.")]


def test_regions_are_off_by_default(profiler):
    assert _cunumpy_regions() == []


@pytest.mark.parametrize("tags", [["cunumpy"], ["*"]])
def test_enabled_tag_records_regions(profiler, tags):
    regions = _cunumpy_regions(enable_tags=tags)
    assert sorted(regions) == ["cunumpy.kernel.noop", "cunumpy.sort_by_key"]
