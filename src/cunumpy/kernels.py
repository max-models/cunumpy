"""Kernels that run on either backend: dispatch, host implementations, fusion.

A :class:`Kernel` pairs a host implementation (Pyccel, numba, NumPy or plain
Python, wrapped in a :class:`PyccelKernel`) with an optional CUDA version (a
:class:`CudaKernel`) and runs the one matching the arrays it
is given; a :class:`KernelCatalog` loads every kernel of a package::

    push = xp.kernels.Kernel.from_folder("my_code.kernels.push")
    push(positions, velocities, dt)

Which host implementation runs is set with :func:`set_host_kernel_implementation`
or :func:`use_host_kernel_implementation` (one of ``HOST_IMPLEMENTATIONS``);
device dispatch can require CUDA with :func:`set_device_kernel_implementation`
or :func:`use_device_kernel_implementation` (``DEVICE_IMPLEMENTATIONS``).
:func:`as_kernel_array` and :func:`kernel_output` (several outputs at once:
:func:`kernel_outputs`) bring the arguments of a kernel to the side of its
main array. :func:`fuse` turns an elementwise
function into one CuPy kernel. See :doc:`/kernels/overview`.

A :class:`PyccelCache` compiles Pyccel kernel modules on first use and keeps
the builds on disk, keyed on their source, for later processes: pass it as
``compile_host`` (see :doc:`/kernels/dispatch`). A :class:`KernelPackage` holds
the cache and the launch options shared by the kernel folders of one package;
:func:`source_roots` finds the headers of other packages for ``include_dirs``.

A :class:`MetalKernel` runs a Metal Shading Language kernel on the GPU of an
Apple silicon Mac through MLX (``pip install 'cunumpy[metal]'``), on NumPy
float32 arrays.

Argument objects for CUDA kernels (:class:`~cunumpy.arguments.CudaStruct`, ...)
are in :mod:`cunumpy.arguments`, the device runtime (streams, devices, debug
mode) in :mod:`cunumpy.cuda`, and the pytest helpers for kernel pairs in
:mod:`cunumpy.kernel_testing`.
"""

from cunumpy._cuda_kernel import CudaKernel, CudaKernelVariants
from cunumpy._dispatch import Kernel, KernelCatalog
from cunumpy._fusion import fuse
from cunumpy._kernel import (
    DEVICE_IMPLEMENTATIONS,
    HOST_IMPLEMENTATIONS,
    CompiledHostKernel,
    HostImplementations,
    PyccelKernel,
    as_kernel_array,
    get_device_kernel_implementation,
    get_host_kernel_implementation,
    kernel_output,
    kernel_outputs,
    outputs_from_annotations,
    set_device_kernel_implementation,
    set_host_kernel_implementation,
    use_device_kernel_implementation,
    use_host_kernel_implementation,
)
from cunumpy._kernel_package import KernelPackage, source_roots
from cunumpy._metal_kernel import MetalKernel, metal_available
from cunumpy._pyccel_cache import PyccelCache

__all__ = [
    "DEVICE_IMPLEMENTATIONS",
    "HOST_IMPLEMENTATIONS",
    "CompiledHostKernel",
    "CudaKernel",
    "CudaKernelVariants",
    "HostImplementations",
    "Kernel",
    "KernelCatalog",
    "KernelPackage",
    "MetalKernel",
    "PyccelCache",
    "PyccelKernel",
    "as_kernel_array",
    "fuse",
    "get_device_kernel_implementation",
    "get_host_kernel_implementation",
    "kernel_output",
    "kernel_outputs",
    "metal_available",
    "outputs_from_annotations",
    "set_device_kernel_implementation",
    "set_host_kernel_implementation",
    "source_roots",
    "use_device_kernel_implementation",
    "use_host_kernel_implementation",
]
