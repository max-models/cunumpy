"""Device implementation configuration without requiring a CUDA installation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cunumpy as xp


def test_device_setting_validation_and_context_restoration():
    previous = xp.kernels.get_device_kernel_implementation()
    assert xp.kernels.DEVICE_IMPLEMENTATIONS == ("cuda",)
    with xp.kernels.use_device_kernel_implementation(None):
        assert xp.kernels.get_device_kernel_implementation() is None
        xp.kernels.set_device_kernel_implementation("cuda")
        for invalid in ("cupy", "numba", "", "CUDA"):
            with pytest.raises(ValueError, match="device kernel implementation"):
                xp.kernels.set_device_kernel_implementation(invalid)
            assert xp.kernels.get_device_kernel_implementation() == "cuda"
        with (
            pytest.raises(ValueError, match="device kernel implementation"),
            xp.kernels.use_device_kernel_implementation("unsupported"),
        ):
            pytest.fail("invalid context was entered")
        assert xp.kernels.get_device_kernel_implementation() == "cuda"
        with (
            pytest.raises(RuntimeError, match="body failed"),
            xp.kernels.use_device_kernel_implementation(None),
        ):
            assert xp.kernels.get_device_kernel_implementation() is None
            raise RuntimeError("body failed")
        assert xp.kernels.get_device_kernel_implementation() == "cuda"
    assert xp.kernels.get_device_kernel_implementation() == previous


@pytest.mark.parametrize("value", [None, "", "cuda", " CuDa ", "cupy"])
def test_device_environment_is_read_at_import(value):
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(xp.__file__).parents[1]),
        "CUNUMPY_BACKEND": "numpy",
        "CUNUMPY_HOST_KERNEL_IMPLEMENTATION": "numpy",
    }
    env.pop("CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION", None)
    if value is not None:
        env["CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION"] = value
    expected = value.strip().lower() if value else None
    code = (
        "import os, cunumpy as xp\n"
        f"assert xp.kernels.get_device_kernel_implementation() == {expected!r}\n"
        "assert xp.kernels.get_host_kernel_implementation() == 'numpy'\n"
        "assert xp.get_backend() == 'numpy'\n"
        "os.environ['CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION'] = 'unsupported'\n"
        f"assert xp.kernels.get_device_kernel_implementation() == {expected!r}\n"
        "kernel = xp.kernels.Kernel(lambda: None, missing_cuda='fallback')\n"
        "if xp.kernels.get_device_kernel_implementation() == 'cuda':\n"
        "    try:\n"
        "        kernel.selected(device=True)\n"
        "    except LookupError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('explicit CUDA allowed host fallback')\n"
        "xp.kernels.set_device_kernel_implementation(None)\n"
        "assert xp.kernels.get_device_kernel_implementation() is None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if value == "cupy":
        assert result.returncode != 0
        assert "ValueError: device kernel implementation" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
