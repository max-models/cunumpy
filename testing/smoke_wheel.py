"""Run with an isolated Python from a fresh venv containing the release wheel."""

import os
import sys
from importlib.resources import files
from pathlib import Path

os.environ["CUNUMPY_FAKE_CUPY"] = "1"
os.environ["CUNUMPY_BACKEND"] = "cupy"

import numpy as np

import cunumpy as xp


def main():
    # Catch accidental imports from the checkout or another environment.
    assert Path(xp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    package = files("cunumpy")
    for asset in (
        "py.typed",
        "__init__.pyi",
        "LLM_GUIDE.md",
        "cuda/include/cunumpy/array_view.cuh",
    ):
        assert package.joinpath(asset).is_file(), f"Missing wheel asset: {asset}"

    expected = np.array(np.arange(12.0).reshape(3, 4), order="F")
    device = xp.to_cupy(expected.copy(order="K"))
    host = xp.to_numpy(device)
    assert host.flags.f_contiguous
    np.testing.assert_array_equal(host, expected)

    def scale(array):
        assert isinstance(array, np.ndarray) and array.flags.f_contiguous
        array *= 2

    xp.kernels.PyccelKernel(scale, outputs=(0,))(device)
    np.testing.assert_array_equal(xp.to_numpy(device), 2 * expected)
    with xp.use_backend("numpy", strict=True):
        np.testing.assert_array_equal(xp.arange(3), np.arange(3))
    print("Installed-wheel smoke test passed")


if __name__ == "__main__":
    main()
