# cunumpy/__init__.py
from importlib.metadata import PackageNotFoundError, version

from . import xp
from .cuda_kernel import (
    CudaArguments,
    CudaKernel,
    CudaKernelVariants,
    CudaParameter,
    CudaStruct,
    CudaStructValue,
    ctype_of,
    parse_cuda_signature,
)
from .dispatch import Kernel, KernelCatalog
from .kernel import PyccelKernel
from .mirror import DeviceMirror
from .xp import (
    assert_same_backend,
    bind_local_device,
    cuda_include_dir,
    cupy_available,
    default_float_dtype,
    device_count,
    free_memory,
    get_array_backend,
    get_array_module,
    get_backend,
    get_rng,
    is_cpu,
    is_gpu,
    local_rank,
    memory_info,
    pin_memory,
    same_backend,
    set_backend,
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
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructValue",
    "DeviceMirror",
    "Kernel",
    "KernelCatalog",
    "PyccelKernel",
    "__version__",
    "assert_same_backend",
    "bind_local_device",
    "ctype_of",
    "cuda_include_dir",
    "cupy_available",
    "cupy_backend",
    "default_float_dtype",
    "device_count",
    "free_memory",
    "get_array_backend",
    "get_array_module",
    "get_backend",
    "get_rng",
    "is_cpu",
    "is_gpu",
    "local_rank",
    "memory_info",
    "numpy_backend",
    "parse_cuda_signature",
    "pin_memory",
    "same_backend",
    "set_backend",
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
