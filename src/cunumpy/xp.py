from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generator, Literal

import array_api_compat
import array_api_compat.numpy as np

BackendType = Literal["numpy", "cupy"]


_CUPY_AVAILABLE_CACHE = None


def cupy_available() -> bool:
    """Check if CuPy is available and functional."""
    global _CUPY_AVAILABLE_CACHE
    if _CUPY_AVAILABLE_CACHE is not None:
        return _CUPY_AVAILABLE_CACHE

    try:
        import cupy as cp

        # Check if a GPU is available
        _CUPY_AVAILABLE_CACHE = cp.is_available()
        return _CUPY_AVAILABLE_CACHE
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        _CUPY_AVAILABLE_CACHE = False
        return False


class ArrayBackend:
    """Holds the process-wide active backend (NumPy or CuPy).

    Not thread-safe: `set_backend`/`use_backend` mutate this single shared
    instance in place, so concurrent code (threads, async tasks) switching
    backends independently will race. Safe for the typical single-threaded
    script/notebook usage this library targets.
    """

    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        if backend.lower() not in ("numpy", "cupy"):
            raise ValueError("Array backend must be either 'numpy' or 'cupy'.")

        self._backend: BackendType = "cupy" if backend.lower() == "cupy" else "numpy"
        self._xp: ModuleType = np  # Placeholder

        # Import numpy/cupy
        self._xp = self._load_backend(self._backend, verbose)

    def _load_backend(self, backend: BackendType, verbose: bool = False) -> ModuleType:
        if backend == "cupy":
            if cupy_available():
                import array_api_compat.cupy as cp

                self._backend = "cupy"
                return cp
            else:
                if verbose:
                    print(
                        "CuPy not available or not functional. Falling back to NumPy."
                    )
                self._backend = "numpy"
                return np
        self._backend = "numpy"
        return np

    def __repr__(self) -> str:
        return f"ArrayBackend(backend={self._backend!r}, module={self._xp.__name__!r})"

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @contextmanager
    def use_backend(self, backend: BackendType) -> Generator[None, None, None]:
        """Temporarily change the backend."""
        old_backend = self._backend
        old_xp = self._xp

        self._backend = backend
        self._xp = self._load_backend(backend)

        try:
            yield
        finally:
            self._backend = old_backend
            self._xp = old_xp


array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)


def use_backend(backend: BackendType) -> Generator[None, None, None]:
    """Temporarily change the backend."""
    return array_backend.use_backend(backend)


def set_backend(backend: BackendType) -> None:
    """Set the backend globally."""
    array_backend._backend = backend
    array_backend._xp = array_backend._load_backend(backend)


def _cupy_backend() -> bool:
    """Check if the active global backend is CuPy."""
    return array_backend.backend == "cupy"


def _numpy_backend() -> bool:
    """Check if the active global backend is NumPy."""
    return array_backend.backend == "numpy"


def set_device(device_id: int) -> None:
    """Select the active CUDA device for the current process (no-op on NumPy)."""
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.cuda.Device(device_id).use()


def device_count() -> int:
    """Number of visible CUDA devices.

    Returns 0 on the NumPy backend, or if CuPy/CUDA is not available.
    Independent of the currently active backend -- this reports what
    hardware is visible, not what `xp.xp` currently dispatches to.
    """
    if not cupy_available():
        return 0

    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - tolerate any driver/runtime failure
        return 0


def set_device_for_rank(rank: int, devices_per_node: int | None = None) -> int:
    """Select a CUDA device for an MPI rank, round-robin across the node.

    Convenience for one-rank-per-GPU codes: computes
    ``device_id = rank % devices_per_node`` and calls `set_device()` with
    it. `devices_per_node` defaults to `device_count()`. Returns the
    selected device id, or 0 as a no-op if there are no visible devices.

    This assumes ranks map to devices in contiguous blocks per node (i.e.
    local rank == ``rank % devices_per_node``); codes with a different
    rank-to-device layout should call `set_device()` directly instead.
    """
    n = devices_per_node if devices_per_node is not None else device_count()
    if n == 0:
        return 0

    device_id = rank % n
    set_device(device_id)
    return device_id


def memory_info() -> tuple[int, int] | None:
    """Return `(free, total)` bytes of memory on the active CUDA device.

    Returns `None` on the NumPy backend. Queries the CUDA runtime directly,
    so it reflects the whole device rather than just CuPy's memory pool.
    """
    if array_backend.backend != "cupy":
        return None

    import cupy as cp

    return cp.cuda.runtime.memGetInfo()


def free_memory() -> None:
    """Release all free blocks held by CuPy's memory pools (no-op on NumPy).

    CuPy caches freed device (and pinned host) memory in pools rather than
    returning it to the driver/OS immediately, which can look like a leak
    in long-running processes. Call this to give it back.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def pin_memory(array: Any) -> Any:
    """Copy a host array into pinned (page-locked) CUDA host memory.

    Pinned memory transfers to/from the GPU faster than regular pageable
    memory, since the driver can DMA it directly. `array` must already be
    on the host (use `to_numpy()` first if it may be on the GPU). Raises
    `ImportError` if CuPy is not available.
    """
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import cupy as cp

    array = np.asarray(array)
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    pinned = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    pinned[...] = array
    return pinned


@contextmanager
def stream() -> Generator[Any, None, None]:
    """Context manager for a CUDA stream, to overlap transfers and compute.

    On the CuPy backend, operations issued inside the block are enqueued on
    a new, non-blocking stream rather than the default one. Call
    `xp.synchronize()` (or the yielded stream's own `.synchronize()`) before
    reading results computed inside the block. No-op on the NumPy backend,
    where it yields `None`.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        with cp.cuda.Stream(non_blocking=True) as s:
            yield s
    else:
        yield None


def get_rng(seed: int | None = None) -> Any:
    """Return a random Generator matching the active backend.

    NumPy and CuPy both provide `default_rng(seed)`, returning a
    `Generator` with a largely-compatible distribution API, but picking the
    right one requires branching on the backend -- this does that for you.
    """
    if array_backend.backend == "cupy":
        import cupy as cp

        return cp.random.default_rng(seed)

    import numpy as numpy_raw

    return numpy_raw.random.default_rng(seed)


def default_float_dtype() -> Any:
    """Return the active backend's `float64` dtype object.

    NumPy and CuPy resolve Python `int`/`float` literals and the bare
    `dtype=float` spelling to a platform- or backend-dependent default
    (e.g. NumPy's default integer width differs between Windows and
    Linux/macOS). Use ``dtype=xp.default_float_dtype()`` instead of
    ``dtype=float`` when a specific, portable precision matters.
    """
    return array_backend.xp.float64


def synchronize() -> None:
    """Wait for all kernels in all streams on current device to complete."""
    if array_backend.backend == "cupy":
        try:
            import cupy as cp

            cp.cuda.Device().synchronize()
        except ImportError:
            pass
        except AttributeError as e:
            warnings.warn(
                f"CuPy synchronize() failed unexpectedly, this may indicate a "
                f"CuPy API mismatch: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array to a NumPy array."""
    if get_backend(array) == "cupy":
        return array.get()

    return np.asarray(array)


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array."""
    if not cupy_available():
        raise ImportError("CuPy is not available or not functional.")

    import array_api_compat.cupy as cp

    return cp.asarray(array)


def to_cunumpy(array: Any) -> Any:
    """Convert an array to the currently active backend."""
    if array_backend.backend == "cupy" and cupy_available():
        return to_cupy(array)
    return to_numpy(array)


def get_backend(array: Any) -> BackendType:
    """Return 'cupy' or 'numpy' depending on the array type."""
    return "cupy" if array_api_compat.is_cupy_array(array) else "numpy"


def get_array_module(array: Any) -> ModuleType:
    """Return the array-api-compat module matching `array`'s own backend.

    Unlike `xp.xp`, which reflects the process-wide active backend, this
    dispatches on the array itself -- useful for writing functions that
    operate correctly regardless of what `set_backend`/`use_backend` last
    selected. Mirrors `cupy.get_array_module`, but also works in Pyodide
    (where `cupy` cannot be imported) and returns array-api-compat modules
    for standard-conformant behavior, consistent with `xp.xp`.
    """
    if get_backend(array) == "cupy":
        import array_api_compat.cupy as cp

        return cp
    return np


def is_gpu(array: Any) -> bool:
    """Check if the array is stored on a GPU (CuPy)."""
    return get_backend(array) == "cupy"


def is_cpu(array: Any) -> bool:
    """Check if the array is stored on a CPU (NumPy)."""
    return get_backend(array) == "numpy"


def same_backend(*arrays: Any) -> bool:
    """Return True if all given arrays live on the same backend.

    Trivially True for zero or one array.
    """
    if len(arrays) <= 1:
        return True
    backends = {get_backend(array) for array in arrays}
    return len(backends) == 1


def assert_same_backend(*arrays: Any) -> None:
    """Raise TypeError if the given arrays don't all live on the same backend.

    Mixing NumPy and CuPy arrays in an operation typically fails with a
    confusing, backend-internal error (e.g. a CuPy kernel dispatch error
    complaining about an "unsupported type"). Call this upfront to fail
    with a clear message instead. Use `to_cunumpy()`/`to_numpy()`/`to_cupy()`
    to align mismatched arrays onto one backend first.
    """
    if not same_backend(*arrays):
        backends = [get_backend(array) for array in arrays]
        raise TypeError(
            f"Arrays are on mismatched backends: {backends}. Use "
            "xp.to_cunumpy()/xp.to_numpy()/xp.to_cupy() to align them first."
        )


# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    import numpy as xp  # noqa: F401 - type-checker-only alias for autocompletion
else:
    # Use module-level __getattr__ for dynamic xp (Python 3.7+)
    def __getattr__(name):
        if name == "xp":
            return array_backend.xp
        if name == "numpy_backend":
            return _numpy_backend()
        if name == "cupy_backend":
            return _cupy_backend()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
