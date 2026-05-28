import os
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generator, Literal

import numpy as np

BackendType = Literal["numpy", "cupy"]


class ArrayBackend:
    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        assert backend.lower() in [
            "numpy",
            "cupy",
        ], "Array backend must be either 'numpy' or 'cupy'."

        self._backend: BackendType = "cupy" if backend.lower() == "cupy" else "numpy"
        self._xp: ModuleType = np  # Placeholder

        # Import numpy/cupy
        self._xp = self._load_backend(self._backend, verbose)

    def _load_backend(self, backend: BackendType, verbose: bool = False) -> ModuleType:
        if backend == "cupy":
            try:
                import cupy as cp

                return cp
            except ImportError:
                if verbose:
                    print("CuPy not available.")
                return np
        import numpy as np_mod

        return np_mod

    def __init_post__(self, verbose: bool = False) -> None:
        # This is now redundant but kept for compatibility if called
        self._xp = self._load_backend(self._backend, verbose)
        assert isinstance(self._xp, ModuleType)
        if verbose:
            print(f"Using {self._xp.__name__} backend.")

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


# TODO: Make this configurable via environment variable or config file.
array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)
# Re-run initialization logic properly after backend selection
array_backend.__init_post__(verbose=False)


def use_backend(backend: BackendType) -> Generator[None, None, None]:
    """Temporarily change the backend."""
    return array_backend.use_backend(backend)


def set_backend(backend: BackendType) -> None:
    """Set the backend globally."""
    array_backend._backend = backend
    array_backend._xp = array_backend._load_backend(backend)


def synchronize() -> None:
    """Wait for all kernels in all streams on current device to complete."""
    if array_backend.backend == "cupy":
        try:
            import cupy as cp

            cp.cuda.Device().synchronize()
        except (ImportError, AttributeError):
            pass


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array to a NumPy array."""
    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)


def to_cupy(array: Any) -> Any:
    """Convert an array to a CuPy array."""
    try:
        import cupy as cp

        return cp.asarray(array)
    except ImportError:
        raise ImportError("CuPy is not available.")


def to_cunumpy(array: Any) -> Any:
    """Convert an array to the currently active backend."""
    if array_backend.backend == "cupy":
        return to_cupy(array)
    return to_numpy(array)


def get_backend(array: Any) -> BackendType:
    """Return 'cupy' or 'numpy' depending on the array type."""
    module = getattr(type(array), "__module__", "")
    return "cupy" if "cupy" in module else "numpy"


def is_gpu(array: Any) -> bool:
    """Check if the array is stored on a GPU (CuPy)."""
    return get_backend(array) == "cupy"


def is_cpu(array: Any) -> bool:
    """Check if the array is stored on a CPU (NumPy)."""
    return get_backend(array) == "numpy"


# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    import numpy as xp
else:
    # Use module-level __getattr__ for dynamic xp (Python 3.7+)
    def __getattr__(name):
        if name == "xp":
            return array_backend.xp
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
