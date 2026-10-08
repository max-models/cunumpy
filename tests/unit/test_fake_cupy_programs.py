"""Tests for the helpers that run whole CuPy-backend programs on the fake CuPy.

`device_backend_available`, `requires_device_backend`, `fake_cupy_session` and
`run_in_fake_cupy_subprocess` of `cunumpy.kernel_testing`.
"""

from pathlib import Path

import pytest

import cunumpy as xp
from cunumpy.kernel_testing import (
    device_backend_available,
    emulation_compiler,
    fake_cupy_active,
    fake_cupy_session,
    run_in_fake_cupy_subprocess,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")


def run(code, **kwargs):
    return run_in_fake_cupy_subprocess(code, env={"PYTHONPATH": SRC}, **kwargs)


def test_device_backend_available():
    assert device_backend_available() == (fake_cupy_active() or xp.cupy_available())
    result = run(
        "from cunumpy.kernel_testing import device_backend_available, requires_device_backend\n"
        "assert device_backend_available()\n"
        "assert not requires_device_backend.args[0]\n"
        "print('available')"
    )
    assert "available" in result.stdout


@pytest.mark.skipif(fake_cupy_active(), reason="the fake CuPy is active here")
def test_fake_cupy_session_needs_the_fake_cupy():
    with pytest.raises(RuntimeError, match="CUNUMPY_FAKE_CUPY"), fake_cupy_session():
        pass


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
def test_fake_cupy_session_runs_a_cupy_program():
    code = r"""
import numpy as np
import cunumpy as xp
from cunumpy.kernel_testing import fake_cupy_session
from cunumpy.kernels import CudaKernel, KernelCatalog, Kernel

scale = CudaKernel('''
extern "C" __global__ void scale(double* x, double f, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= f;
}''', "scale")
kernel = Kernel(lambda x, f, n: None, scale)
with fake_cupy_session():
    assert xp.get_backend() == "cupy"
    KernelCatalog({"scale": kernel}).compile_all()
    x = xp.arange(4.0)
    kernel(x, 3.0, 4, n_threads=4)
    print(xp.to_numpy(x).tolist())
assert xp.get_backend() == "numpy"
"""
    assert "[0.0, 3.0, 6.0, 9.0]" in run(code).stdout


def test_child_runs_serially_on_the_fake_cupy(monkeypatch):
    monkeypatch.setenv("OMPI_MCA_test_variable", "1")
    result = run(
        "import os\n"
        "assert os.environ['CUNUMPY_FAKE_CUPY'] == '1'\n"
        "assert os.environ['OMP_NUM_THREADS'] == '1'\n"
        "assert 'OMPI_MCA_test_variable' not in os.environ\n"
        "from cunumpy.kernel_testing import fake_cupy_active\n"
        "assert fake_cupy_active()\n"
        "print('serial')"
    )
    assert result.returncode == 0
    assert "serial" in result.stdout


def test_a_failing_child_fails_the_test_with_the_end_of_its_output():
    code = "print('\\n'.join(f'line {i}' for i in range(100)))\nraise SystemExit(3)"
    with pytest.raises(pytest.fail.Exception) as info:
        run(code)
    message = str(info.value)
    assert "exit code 3" in message
    assert "line 99" in message and "line 49" not in message


def test_a_crashing_child_reports_the_signal():
    code = "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)"
    with pytest.raises(pytest.fail.Exception, match="signal SIGSEGV"):
        run(code)


def test_a_child_that_hangs_times_out():
    with pytest.raises(pytest.fail.Exception, match="timed out"):
        run("import time\nprint('started', flush=True)\ntime.sleep(60)", timeout=2)
