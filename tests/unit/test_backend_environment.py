"""Backend selection from CuNumpy's prefixed environment setting at import."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cunumpy


@pytest.mark.parametrize(
    "backend,legacy,expected",
    [
        (None, None, "numpy"),
        ("numpy", None, "numpy"),
        ("cupy", None, "cupy"),
        ("CuPy", None, "cupy"),
        (None, "cupy", "numpy"),
        ("numpy", "cupy", "numpy"),
        ("cupy", "numpy", "cupy"),
    ],
)
def test_prefixed_backend_is_selected_at_import(backend, legacy, expected):
    env = {**os.environ, "CUNUMPY_FAKE_CUPY": "1"}
    env["PYTHONPATH"] = str(Path(cunumpy.__file__).parents[1])
    env.pop("CUNUMPY_BACKEND", None)
    env.pop("ARRAY_BACKEND", None)
    if backend is not None:
        env["CUNUMPY_BACKEND"] = backend
    if legacy is not None:
        env["ARRAY_BACKEND"] = legacy
    code = (
        "import os, cunumpy as xp\n"
        f"assert xp.get_backend() == {expected!r}\n"
        "os.environ['CUNUMPY_BACKEND'] = 'numpy'\n"
        f"assert xp.get_backend() == {expected!r}\n"
        "xp.set_backend('numpy')\n"
        "assert xp.get_backend() == 'numpy'\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
