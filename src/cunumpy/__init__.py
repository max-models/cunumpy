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
    pin_memory,
    same_backend,
    set_backend,
    set_cuda_debug,
    set_device,
    set_device_for_rank,
    stream,
    synchronize,
    synchronize_for_mpi,
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
    "numpy_backend",
    "parse_cuda_signature",
    "pin_memory",
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
