"""CUDA devices, memory, streams, debug mode and CUDA source tools.

The device functions (:func:`set_device`, :func:`memory_info`, :func:`stream`,
...) do nothing, or return ``None`` or ``0``, on the NumPy backend, so the same
code runs on both::

    with xp.cuda.stream():
        kernel(positions, velocities, dt, n_threads=n)

:func:`create_stream` and :func:`create_event` return synchronous
:class:`HostStream` and :class:`HostEvent` objects on NumPy. The CUDA headers
shipped with cunumpy (``cunumpy/atomic.cuh``, ...) are in
:func:`cuda_include_dir`; :func:`parse_cuda_signature` and the other source
tools inspect CUDA sources. The kernel classes are in :mod:`cunumpy.kernels`,
the argument objects in :mod:`cunumpy.arguments`. See :doc:`/guides/gpu-devices`.

Importing this module makes ``xp.cuda`` refer to it instead of ``cupy.cuda``;
use ``import cupy; cupy.cuda`` for CuPy's module.
"""

from cunumpy._cuda_kernel import (
    DEBUG_OPTIONS,
    CudaParameter,
    ctype_of,
    cuda_include_dir,
    cuda_kernel_names,
    include_hash,
    parse_cuda_signature,
    resolve_includes,
)
from cunumpy._device import (
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
from cunumpy._streams import (
    HostEvent,
    HostStream,
    create_event,
    create_stream,
    record_event,
    wait_event,
)

__all__ = [
    "DEBUG_OPTIONS",
    "DEFAULT_SHARED_MEMORY_PER_BLOCK",
    "CudaParameter",
    "HostEvent",
    "HostStream",
    "bind_local_device",
    "create_event",
    "create_stream",
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
    "record_event",
    "resolve_includes",
    "set_cuda_debug",
    "set_device",
    "set_device_for_rank",
    "stream",
    "wait_event",
]
