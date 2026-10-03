"""Kernels that run on either backend: dispatch, host implementations, fusion.

A :class:`Kernel` pairs a host implementation (Pyccel, numba, NumPy or plain
Python) with an optional CUDA version and runs the one matching the arrays it
is given; a :class:`KernelCatalog` loads every kernel of a package::

    import cunumpy as xp

    push = xp.kernels.Kernel.from_folder("my_code.kernels.push")
    push(positions, velocities, dt)

Which host implementation runs is set with :func:`set_kernel_implementation`
or :func:`use_kernel_implementation`. :func:`as_kernel_array` and
:func:`kernel_output` bring the arguments of a kernel to the side of its main
array. :func:`fuse` turns an elementwise function into one CuPy kernel.

The CUDA-only classes (:class:`~cunumpy.cuda.CudaKernel`, ...) are in
:mod:`cunumpy.cuda`; the pytest helpers for kernel pairs are in
:mod:`cunumpy.kernel_testing`.
"""

from .cuda_kernel import PyccelStructArguments
from .dispatch import Kernel, KernelCatalog
from .fusion import fuse
from .kernel import (
    HOST_IMPLEMENTATIONS,
    CompiledHostKernel,
    HostImplementations,
    KernelArguments,
    PyccelKernel,
    get_kernel_implementation,
    resolve_host_args,
    set_kernel_implementation,
    use_kernel_implementation,
)
from .xp import as_kernel_array, kernel_output

__all__ = [
    "HOST_IMPLEMENTATIONS",
    "CompiledHostKernel",
    "HostImplementations",
    "Kernel",
    "KernelArguments",
    "KernelCatalog",
    "PyccelKernel",
    "PyccelStructArguments",
    "as_kernel_array",
    "fuse",
    "get_kernel_implementation",
    "kernel_output",
    "resolve_host_args",
    "set_kernel_implementation",
    "use_kernel_implementation",
]
