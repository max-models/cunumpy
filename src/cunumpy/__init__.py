# cunumpy/__init__.py
import re as _re
from importlib.metadata import PackageNotFoundError, version

from . import xp
from .cuda_kernel import (
    DEBUG_OPTIONS,
    CudaArguments,
    CudaKernel,
    CudaKernelVariants,
    CudaParameter,
    CudaStruct,
    CudaStructArguments,
    CudaStructValue,
    PyccelStructArguments,
    ctype_of,
    cuda_include_dir,
    cuda_kernel_names,
    include_hash,
    parse_cuda_signature,
    resolve_includes,
    write_cuda_header,
)
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
from .mirror import DeviceMirror
from .petsc import petsc_vec
from .philox import (
    philox4x32_10,
    philox_normal,
    philox_normal2,
    philox_uniform,
    philox_uniform2,
)
from .random_streams import RandomStreams, random_streams
from .scipy_backend import scipy
from .staging import HostStaging, StagedCopy
from .transfers import (
    TransferCounter,
    TransferEvent,
    assert_no_transfers,
    count_transfers,
)
from .xp import (
    DEFAULT_SHARED_MEMORY_PER_BLOCK,
    Timing,
    as_device_array,
    as_kernel_array,
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
    get_mpi_cuda_aware,
    get_rng,
    is_cpu,
    is_gpu,
    kernel_output,
    local_rank,
    max_shared_memory_per_block,
    memory_info,
    mpi_buffer,
    mpi_is_cuda_aware,
    nvtx_range,
    pin_memory,
    require_cuda_aware_mpi,
    same_backend,
    segment_sum,
    set_backend,
    set_cuda_debug,
    set_device,
    set_device_for_rank,
    set_mpi_cuda_aware,
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


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _re.findall(r"\d+", text.split("+")[0])[:3])


def require_version(minimum: str) -> None:
    """Raise ``ImportError`` if this cunumpy is older than `minimum`.

    For projects that depend on a feature of a given release, as a clearer
    error than an ``AttributeError`` later::

        import cunumpy as xp

        xp.require_version("0.4.0")

    Only the numeric part of the versions is compared (``0.4.0`` and
    ``0.4.0.dev1`` compare equal). Nothing is checked when the installed
    version is unknown (cunumpy not installed as a package).
    """
    if __version__.startswith("0.0.0+unknown"):
        return
    if _version_key(__version__) < _version_key(minimum):
        raise ImportError(
            f"cunumpy {minimum} or newer is required, but {__version__} is "
            "installed: pip install --upgrade cunumpy"
        )


__all__ = [
    "DEBUG_OPTIONS",
    "DEFAULT_SHARED_MEMORY_PER_BLOCK",
    "HOST_IMPLEMENTATIONS",
    "CompiledHostKernel",
    "CudaArguments",
    "CudaKernel",
    "CudaKernelVariants",
    "CudaParameter",
    "CudaStruct",
    "CudaStructArguments",
    "CudaStructValue",
    "DeviceMirror",
    "HostImplementations",
    "HostStaging",
    "Kernel",
    "KernelArguments",
    "KernelCatalog",
    "PyccelKernel",
    "PyccelStructArguments",
    "RandomStreams",
    "StagedCopy",
    "Timing",
    "TransferCounter",
    "TransferEvent",
    "__version__",
    "as_device_array",
    "as_kernel_array",
    "assert_no_transfers",
    "assert_same_backend",
    "bind_local_device",
    "count_transfers",
    "ctype_of",
    "cuda_debug",
    "cuda_include_dir",
    "cuda_kernel_names",
    "cupy_available",
    "cupy_backend",
    "default_float_dtype",
    "device_count",
    "free_memory",
    "fuse",
    "get_array_backend",
    "get_array_module",
    "get_backend",
    "get_cuda_debug",
    "get_kernel_implementation",
    "get_mpi_cuda_aware",
    "get_rng",
    "include_hash",
    "is_cpu",
    "is_gpu",
    "kernel_output",
    "local_rank",
    "max_shared_memory_per_block",
    "memory_info",
    "mpi_buffer",
    "mpi_is_cuda_aware",
    "numpy_backend",
    "nvtx_range",
    "parse_cuda_signature",
    "petsc_vec",
    "philox4x32_10",
    "philox_normal",
    "philox_normal2",
    "philox_uniform",
    "philox_uniform2",
    "pin_memory",
    "random_streams",
    "require_cuda_aware_mpi",
    "require_version",
    "resolve_host_args",
    "resolve_includes",
    "same_backend",
    "scipy",
    "segment_sum",
    "set_backend",
    "set_cuda_debug",
    "set_device",
    "set_device_for_rank",
    "set_kernel_implementation",
    "set_mpi_cuda_aware",
    "stream",
    "synchronize",
    "synchronize_for_mpi",
    "timed_region",
    "to_cunumpy",
    "to_cupy",
    "to_numpy",
    "use_backend",
    "use_kernel_implementation",
    "write_cuda_header",
    "xp",
]


def __getattr__(name: str):
    """Set cunumpy.<name> to cunumpy.xp.<name> (NumPy/CuPy)."""
    if name == "numpy_backend":
        return xp.numpy_backend
    if name == "cupy_backend":
        return xp.cupy_backend
    return getattr(xp.xp, name)
