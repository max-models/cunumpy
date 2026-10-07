"""With CUNUMPY_REQUIRE_CUDA=1 the GPU markers of kernel_testing fail (as errors) instead of skipping."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cunumpy as xp

pytestmark = pytest.mark.skipif(
    xp.cupy_available(), reason="checks the behavior without a GPU"
)

TESTS = """
import pytest
from cunumpy.kernel_testing import backend, requires_cupy


@requires_cupy
def test_marked():
    pass


def test_backend(backend):
    pass
"""


def run(tmp_path, require):
    (tmp_path / "test_gpu.py").write_text(TESTS)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), env.get("PYTHONPATH", "")) if p
    )
    env.pop("CUNUMPY_REQUIRE_CUDA", None)
    env.pop("CUNUMPY_FAKE_CUPY", None)
    env.pop("CUNUMPY_BACKEND", None)
    if require:
        env["CUNUMPY_REQUIRE_CUDA"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rA", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_markers_skip_without_a_gpu(tmp_path):
    result = run(tmp_path, require=False)
    assert result.returncode == 0, result.stdout
    assert "SKIPPED" in result.stdout and "ERROR" not in result.stdout


def test_markers_error_when_a_gpu_is_required(tmp_path):
    result = run(tmp_path, require=True)
    assert result.returncode != 0, result.stdout
    assert "CUNUMPY_REQUIRE_CUDA is set" in result.stdout
    # the condition of the marker raises while the test is set up: an error
    assert "ERROR test_gpu.py::test_marked" in result.stdout
    assert "ERROR test_gpu.py::test_backend[cupy]" in result.stdout
    assert "PASSED test_gpu.py::test_backend[numpy]" in result.stdout
