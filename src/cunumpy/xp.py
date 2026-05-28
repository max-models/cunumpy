import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

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

        # Import numpy/cupy
        if self.backend == "cupy":
            try:
                import cupy as cp

                self._xp = cp
            except ImportError:
                if verbose:
                    print("CuPy not available.")
                self._backend = "numpy"

        if self.backend == "numpy":
            import numpy as np

            self._xp = np

        assert isinstance(self.xp, ModuleType)

        if verbose:
            print(f"Using {self.xp.__name__} backend.")

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        return self._xp


# TODO: Make this configurable via environment variable or config file.
array_backend = ArrayBackend(
    backend=(
        "cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy"
    ),
    verbose=False,
)


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
    xp = array_backend.xp
