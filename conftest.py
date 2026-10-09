"""Doctest setup: examples assume ``import cunumpy as xp`` and ``import numpy as np``."""

import numpy as np
import pytest

import cunumpy as xp


@pytest.fixture(autouse=True)
def _doctest_namespace(request, doctest_namespace):
    if not isinstance(request.node, pytest.DoctestItem):
        yield  # leave the backend of the regular tests alone
        return
    doctest_namespace["np"] = np
    doctest_namespace["xp"] = xp
    with xp.use_backend("numpy"):
        yield
