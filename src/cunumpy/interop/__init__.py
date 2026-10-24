"""Explicit interoperability with scientific machine-learning libraries.

The adapters are optional and import their framework lazily. They do not
participate in CuNumpy backend selection.
"""

from cunumpy.interop import dlpack, jax, mlx, torch

__all__ = ["dlpack", "jax", "mlx", "torch"]
