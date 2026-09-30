# cunumpy/__init__.py
from importlib.metadata import PackageNotFoundError, version

from . import xp
from .cuda_kernel import (
    DEBUG_OPTIONS,
    CudaArguments,
    CudaKernel,
    CudaKernelVariants,
    CudaParameter,
    CudaStruct,
    CudaStructValue,
    ctype_of,
    cuda_kernel_names,
    include_hash,
    parse_cuda_signature,
    resolve_includes,
)
from .dispatch import Kernel, KernelCatalog
from .kernel import KernelArguments, PyccelKernel, resolve_host_args
from .transfers import (
    TransferCounter,
    TransferEvent,
    assert_no_transfers,
    count_transfers,
)
from .xp import (
    Timing,
    as_device_array,
    assert_same_backend,
    bind_local_device,
    cuda_debug,
    cupy_available,
    default_float_dtype,
    device_count,
    free_memory,
    get_array_backend,
    get_array_module,
    get_backend,
    get_cuda_debug,
    get_rng,
    is_cpu,
    is_gpu,
    local_rank,
    memory_info,
    mpi_is_cuda_aware,
    nvtx_range,
    pin_memory,
    require_cuda_aware_mpi,
    same_backend,
    set_backend,
    set_cuda_debug,
    set_device,
    set_device_for_rank,
    stream,
    synchronize,
    synchronize_for_mpi,
    timed_region,
    to_cunumpy,
    to_cupy,
    to_numpy,
    use_backend,
)

try:
    __version__ = version("cunumpy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEBUG_OPTIONS",
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructValue",
    "Kernel",
    "KernelArguments",
    "KernelCatalog",
    "PyccelKernel",
    "Timing",
    "TransferCounter",
    "TransferEvent",
    "__version__",
    "as_device_array",
    "assert_no_transfers",
    "assert_same_backend",
    "bind_local_device",
    "count_transfers",
    "ctype_of",
    "cuda_debug",
    "cuda_kernel_names",
    "cupy_available",
    "cupy_backend",
    "default_float_dtype",
    "device_count",
    "free_memory",
    "get_array_backend",
    "get_array_module",
    "get_backend",
    "get_cuda_debug",
    "get_rng",
    "include_hash",
    "is_cpu",
    "is_gpu",
    "local_rank",
    "memory_info",
    "mpi_is_cuda_aware",
    "numpy_backend",
    "nvtx_range",
    "parse_cuda_signature",
    "pin_memory",
    "require_cuda_aware_mpi",
    "resolve_host_args",
    "resolve_includes",
    "same_backend",
    "set_backend",
    "set_cuda_debug",
    "set_device",
    "set_device_for_rank",
    "stream",
    "synchronize",
    "synchronize_for_mpi",
    "timed_region",
    "to_cunumpy",
    "to_cupy",
    "to_numpy",
    "use_backend",
    "xp",
]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    if name == "numpy_backend":
        return xp.numpy_backend
    if name == "cupy_backend":
        return xp.cupy_backend
    return getattr(xp.xp, name)
