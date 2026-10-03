"""Tests for binding MPI ranks to GPUs: `local_rank`, `bind_local_device` and
`synchronize_for_mpi`. None of them needs MPI; the GPU parts need a GPU."""

import numpy as np
import pytest

import cunumpy as xp
from cunumpy.xp import _LOCAL_RANK_VARIABLES


@pytest.fixture
def clean_env(monkeypatch):
    """No launcher variables set."""
    for variable in _LOCAL_RANK_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def test_local_rank_default(clean_env):
    assert xp.mpi.local_rank() == 0


@pytest.mark.parametrize("variable", _LOCAL_RANK_VARIABLES)
def test_local_rank_from_launcher(clean_env, variable):
    clean_env.setenv(variable, "3")
    assert xp.mpi.local_rank() == 3


def test_local_rank_order_and_invalid_values(clean_env):
    clean_env.setenv("SLURM_LOCALID", "5")
    clean_env.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")
    assert xp.mpi.local_rank() == 2  # the MPI launcher's value wins over Slurm's

    clean_env.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "not a number")
    assert xp.mpi.local_rank() == 5  # invalid values are skipped


def test_bind_local_device_on_numpy(clean_env):
    with xp.use_backend("numpy"):
        assert xp.cuda.bind_local_device() is None


def test_bind_local_device_on_cupy(clean_env):
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    count = xp.cuda.device_count()
    previous = cp.cuda.runtime.getDevice()
    clean_env.setenv("OMPI_COMM_WORLD_LOCAL_RANK", str(count + 1))
    try:
        with xp.use_backend("cupy"):
            device = xp.cuda.bind_local_device()
        assert device == (count + 1) % count
        assert cp.cuda.runtime.getDevice() == device
    finally:
        cp.cuda.Device(previous).use()


def test_synchronize_for_mpi_host_arrays():
    # host buffers and None: nothing to wait for (and no CuPy needed)
    xp.mpi.synchronize_for_mpi(np.zeros(3), None)
    xp.mpi.synchronize_for_mpi()


def test_synchronize_for_mpi_device_arrays():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")
    import cupy as cp

    x = cp.zeros(1_000_000)
    x += 1  # a kernel still running on the current stream
    xp.mpi.synchronize_for_mpi(np.zeros(1), x)
    assert cp.cuda.get_current_stream().done  # nothing pending any more
    assert float(x[0]) == 1.0
