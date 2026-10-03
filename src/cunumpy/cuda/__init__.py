"""CUDA-only parts of cunumpy: writing and launching CUDA kernels, and the GPU.

Everything here is only useful with CuPy and a CUDA device. The kernel classes
(:class:`CudaKernel`, :class:`CudaStruct`, ...) need CuPy to launch; the device
functions (:func:`set_device`, :func:`memory_info`, :func:`stream`, ...) do
nothing (or return ``None``/``0``) on the NumPy backend::

    import cunumpy as xp

    kernel = xp.cuda.CudaKernel(source, "push")
    with xp.cuda.stream():
        kernel(positions, velocities, dt, n_threads=n)

The CUDA headers shipped with cunumpy (``cunumpy/atomic.cuh``,
``cunumpy/random.cuh``, ...) are in :func:`cuda_include_dir`.

Backend-neutral kernel tools (:class:`~cunumpy.kernels.Kernel`,
:class:`~cunumpy.kernels.PyccelKernel`, ...) are in :mod:`cunumpy.kernels`.

Importing this module makes ``xp.cuda`` refer to it instead of ``cupy.cuda``;
use ``import cupy; cupy.cuda`` for CuPy's module.
"""

from .._cuda_kernel import (
    DEBUG_OPTIONS,
    CudaArguments,
    CudaKernel,
    CudaKernelVariants,
    CudaParameter,
    CudaStruct,
    CudaStructArguments,
    CudaStructValue,
    ctype_of,
    cuda_include_dir,
    cuda_kernel_names,
    include_hash,
    parse_cuda_signature,
    resolve_includes,
    write_cuda_header,
)
from .._device import (
    DEFAULT_SHARED_MEMORY_PER_BLOCK,
    bind_local_device,
    cuda_debug,
    device_count,
    free_memory,
    get_cuda_debug,
    max_shared_memory_per_block,
    memory_info,
    pin_memory,
    set_cuda_debug,
    set_device,
    set_device_for_rank,
    stream,
)

__all__ = [
    "DEBUG_OPTIONS",
    "DEFAULT_SHARED_MEMORY_PER_BLOCK",
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructArguments",
    "CudaStructValue",
    "bind_local_device",
    "ctype_of",
    "cuda_debug",
    "cuda_include_dir",
    "cuda_kernel_names",
    "device_count",
    "free_memory",
    "get_cuda_debug",
    "include_hash",
    "max_shared_memory_per_block",
    "memory_info",
    "parse_cuda_signature",
    "pin_memory",
    "resolve_includes",
    "set_cuda_debug",
    "set_device",
    "set_device_for_rank",
    "stream",
    "write_cuda_header",
]
