"""Tests for the submodule layout of cunumpy and the backend names of the top level."""

import importlib
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import cunumpy as xp

SUBMODULES = (
    "algorithms",
    "arguments",
    "cuda",
    "kernels",
    "memory",
    "mpi",
    "petsc",
    "profiling",
    "rng",
)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_names_do_not_shadow_numpy(name):
    # `import cunumpy as xp` stands in for numpy: a submodule must not hide a numpy name
    assert not hasattr(np, name)
    assert getattr(xp, name) is importlib.import_module(f"cunumpy.{name}")


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_exports_resolve(name):
    module = getattr(xp, name)
    for attr in getattr(module, "__all__", ()):
        assert hasattr(module, attr), f"cunumpy.{name}.{attr}"


def test_top_level_is_backend_and_numpy_only():
    for name in xp.__all__:
        assert hasattr(xp, name)


def test_kernel_classes_are_in_kernels_and_arguments():
    import cunumpy._cuda_kernel as impl

    assert xp.kernels.CudaKernel is impl.CudaKernel
    assert xp.kernels.PyccelKernel is not None
    assert xp.arguments.CudaStructArguments is impl.CudaStructArguments
    for name in ("CudaKernel", "CudaStruct"):
        assert not hasattr(xp.cuda, name)
        assert name not in vars(xp)


def test_numpy_names_do_not_warn():
    with warnings.catch_warnings(), xp.use_backend("numpy"):
        warnings.simplefilter("error")
        assert xp.zeros(2).shape == (2,)
        assert xp.random is not None
        assert xp.testing.assert_allclose is not None


def test_unknown_name_raises():
    with pytest.raises(AttributeError):
        _ = xp.no_such_function_in_cunumpy
    with pytest.raises(AttributeError):
        _ = xp.cuda.no_such_name


def test_kernel_testing_keeps_numpy_testing():
    import cunumpy.kernel_testing  # noqa: F401

    with xp.use_backend("numpy"):
        assert xp.testing is np.testing


def test_backend_names_are_plain_attributes():
    # copied into the namespace: no module __getattr__ call per access
    assert vars(xp)["zeros"] is xp.xp.xp.zeros
    assert "zeros" in dir(xp)
    with xp.use_backend("numpy"):
        assert vars(xp)["testing"] is np.testing


def test_backend_names_never_hide_cunumpy_names():
    assert xp.cuda is importlib.import_module("cunumpy.cuda")
    assert xp.scipy is importlib.import_module("cunumpy._scipy_backend").scipy


def test_switching_the_backend_replaces_the_names():
    # the fake CuPy of cunumpy._fake_cupy stands in for a GPU, in a fresh process
    code = (
        "import cunumpy as xp\n"
        "numpy_zeros = xp.zeros\n"
        "with xp.use_backend('cupy'):\n"
        "    assert xp.get_backend() == 'cupy'\n"
        "    assert xp.zeros is xp.xp.xp.zeros is not numpy_zeros\n"
        "    assert type(xp.zeros(2)).__module__.startswith('cupy')\n"
        "assert xp.zeros is numpy_zeros\n"
        "xp.set_backend('cupy')\n"
        "assert type(xp.arange(3)).__module__.startswith('cupy')\n"
        "xp.set_backend('numpy')\n"
        "assert type(xp.arange(3)).__module__ == 'numpy'\n"
    )
    env = {**os.environ, "CUNUMPY_FAKE_CUPY": "1", "CUNUMPY_BACKEND": "numpy"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_listener_runs_only_when_the_module_changes():
    calls = []
    xp.xp.array_backend.add_listener(calls.append)
    try:
        assert calls == [xp.xp.xp]  # called once when added
        active = xp.get_backend()
        with xp.use_backend(active):
            pass
        xp.set_backend(active)
        assert len(calls) == 1
    finally:
        xp.xp.array_backend._listeners.remove(calls.append)
