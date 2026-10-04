"""CUDA CI must fail if its hardware is unavailable rather than skip coverage."""

import os

import pytest


def pytest_configure(config):
    if os.environ.get("CUNUMPY_REQUIRE_CUDA", "").lower() not in {"1", "true", "yes"}:
        return
    import cunumpy as xp

    try:
        xp.set_backend("cupy", strict=True)
        import cupy as cp

        if getattr(cp, "__cunumpy_fake__", False):
            raise RuntimeError("fake CuPy does not provide CUDA hardware coverage")
        cp.zeros(1).get()  # exercise allocation, execution and synchronization
    except Exception as error:
        raise pytest.UsageError(
            f"CUDA CI requires a usable real GPU: {error}"
        ) from error
