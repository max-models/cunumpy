"""Explicit interoperability with scientific machine-learning libraries and Julia.

The adapters are optional and import their framework lazily. They do not
participate in CuNumpy backend selection.
"""

from cunumpy.interop import dlpack, jax, julia, mlx, torch

__all__ = ["dlpack", "jax", "julia", "mlx", "torch"]
