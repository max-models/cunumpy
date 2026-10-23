"""Tests for optional framework interoperability."""

import types

import numpy as np
import pytest

from cunumpy.interop import dlpack
from cunumpy.interop._common import require, target_backend


def test_dlpack_helpers_use_array_protocol(monkeypatch):
    class Source:
        def __dlpack__(self):
            return "capsule"

    assert dlpack.to_dlpack(Source()) == "capsule"

    target = types.SimpleNamespace(
        from_dlpack=lambda source: np.array([source.__dlpack__()]),
    )
    monkeypatch.setattr(
        "cunumpy.interop.dlpack.target_backend", lambda backend: target
    )
    result = dlpack.from_dlpack(Source())
    np.testing.assert_array_equal(result, ["capsule"])


def test_dlpack_rejects_objects_without_protocol():
    with pytest.raises(TypeError, match="DLPack protocol"):
        dlpack.to_dlpack(object())


def test_optional_dependency_error_is_actionable():
    with pytest.raises(ImportError, match="cunumpy\\[not-installed-framework\\]"):
        require("not-installed-framework")


def test_target_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be"):
        target_backend("jax")
