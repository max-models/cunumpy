"""CUDA device runtime, reusable streams/events, and CUDA source tools.

The device functions (:func:`set_device`, :func:`memory_info`, :func:`stream`,
...) do nothing (or return ``None``/``0``) on the NumPy backend::

    import cunumpy as xp

    kernel = xp.kernels.CudaKernel(source, "push")
    with xp.cuda.stream():
        kernel(positions, velocities, dt, n_threads=n)

The CUDA headers shipped with cunumpy (``cunumpy/atomic.cuh``,
``cunumpy/random.cuh``, ...) are in :func:`cuda_include_dir`;
:func:`parse_cuda_signature` and the other source tools inspect CUDA sources.

Reusable :func:`create_stream` and :func:`create_event` return synchronous host
equivalents on NumPy, supporting the same recording and completion interface.

The kernel classes (:class:`~cunumpy.kernels.CudaKernel`,
:class:`~cunumpy.kernels.PyccelKernel`, :class:`~cunumpy.kernels.Kernel`, ...)
are in :mod:`cunumpy.kernels`, the argument objects
(:class:`~cunumpy.arguments.CudaStructArguments`, ...) in :mod:`cunumpy.arguments`.

Importing this module makes ``xp.cuda`` refer to it instead of ``cupy.cuda``;
use ``import cupy; cupy.cuda`` for CuPy's module.
"""

import importlib as _importlib
import warnings as _warnings

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

# Names that were in this module before they moved to cunumpy.kernels and
# cunumpy.arguments. They still resolve (with a DeprecationWarning) until cunumpy 0.6.
_MOVED = {
    "CudaKernel": "kernels",
    "CudaKernelVariants": "kernels",
    "CudaArguments": "arguments",
    "CudaStruct": "arguments",
    "CudaStructArguments": "arguments",
    "CudaStructValue": "arguments",
    "write_cuda_header": "arguments",
}


def __getattr__(name: str):
    submodule = _MOVED.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warnings.warn(
        f"cunumpy.cuda.{name} moved to cunumpy.{submodule}.{name}; the old name "
        "is deprecated and will be removed in cunumpy 0.6",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_importlib.import_module(f"cunumpy.{submodule}"), name)
