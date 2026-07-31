# Changelog

All notable changes to the `cunumpy` library are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-07-31

### Added
- `xp.set_device(device_id)`: Select the active CUDA device for the current process (no-op on the NumPy backend).
- `xp.__version__`: Reports the installed `cunumpy` package version.

### Fixed
- `ArrayBackend` no longer silently reports `"cupy"` as the active backend when CuPy was requested but is unavailable; it now correctly falls back to reporting `"numpy"` so `xp.cupy_backend`/`xp.numpy_backend` reflect what actually loaded.
- `to_numpy()` no longer misdetects CPU objects that merely expose a `.get` method (e.g. dict-like objects) as CuPy arrays; it now checks `get_backend()` instead of `hasattr(array, "get")`.
- Invalid backend names now raise `ValueError` instead of relying on a bare `assert`, which was previously stripped under `python -O`.

### Changed
- Removed the redundant `ArrayBackend.__init_post__` double-initialization path.
- `ArrayBackend` documents that it is not thread-safe (global mutable backend state).
- CI now runs the test suite across a Python 3.8/3.10/3.13 matrix instead of only 3.10, and `ruff` is now an enforced check rather than advisory.
- Added [`array-api-compat`](https://github.com/data-apis/array-api-compat) as a core dependency:
    - `get_backend()`/`is_gpu()`/`is_cpu()` now use `array_api_compat.is_cupy_array()` instead of sniffing `type(array).__module__` for the substring `"cupy"`.
    - The active backend module (`xp.xp`) and array conversions (`to_numpy()`, `to_cupy()`) now resolve through `array_api_compat.numpy`/`array_api_compat.cupy` instead of the raw modules, for standard-conformant behavior across backends (e.g. `xp.xp.__name__` is now `"array_api_compat.numpy"`/`"array_api_compat.cupy"` rather than `"numpy"`/`"cupy"`). Device/synchronization control (`set_device()`, `synchronize()`, `cupy_available()`) still uses raw `cupy`, since CUDA device management isn't part of the Array API standard.

## [0.1.2] - 2026-05-27

### Added
- **Backend Management Helpers**:
    - `xp.to_numpy(arr)`: Explicitly move an array to CPU (NumPy).
    - `xp.to_cupy(arr)`: Explicitly move an array to GPU (CuPy).
    - `xp.to_cunumpy(arr)`: Normalize an array to the currently active backend.
    - `xp.get_backend(arr)`: Identify if an array is `"numpy"` or `"cupy"`.
    - `xp.is_gpu(arr)` & `xp.is_cpu(arr)`: Quick boolean checks for array residence.
- **Global Backend Control**:
    - `xp.set_backend(name)`: Globally switch the active backend at runtime.
    - `xp.use_backend(name)`: Context manager for temporary, scoped backend switching.
    - `xp.numpy_backend` & `xp.cupy_backend`: Boolean properties to check the globally active backend.
- **Synchronization**:
    - `xp.synchronize()`: Blocks until GPU operations are complete (no-op on CPU). Essential for accurate benchmarking.
- **Developer Experience**:
    - Added `isort` configuration to `pyproject.toml` with `black` profile compatibility.
    - Reorganized test suite into specialized files: `test_numpy.py`, `test_cupy.py`, and `test_cunumpy.py`.

### Changed
- **Dynamic Dispatch Architecture**: Refactored `src/cunumpy/xp.py` to use module-level `__getattr__`. This ensures that `cunumpy.<op>` calls always resolve to the currently active backend module, enabling seamless runtime switching via `set_backend`.
- **Type Safety**: Updated `src/cunumpy/__init__.pyi` stubs to provide full IDE autocompletion and type-checking for all new API methods.
- **Documentation**: 
    - Simplified `README.md` and documentation to exclusively focus on PyPI installation (`pip install cunumpy`).
    - Enhanced `quickstart.md` and `api.md` with usage examples for the new backend control and synchronization features.
- **CI/CD**: Restricted GitHub Pages documentation deployment to the `devel` branch only.

### Fixed
- Improved `ArrayBackend` initialization to fallback gracefully to NumPy if CuPy is requested but not installed.
- Fixed symbol accessibility tests to correctly handle custom helper methods in the `cunumpy` namespace.

## [0.1.1] - 2026-05-27

### Added
- **Type Inspection Support**: Added `.pyi` stub files to enable proper symbol inspection and autocompletion for Pylance/VS Code.
- **CI/CD**: Added GitHub Action for automated publishing to PyPI.

### Changed
- Improved import structure in `src/cunumpy/__init__.py` to allow direct access to `xp`.

## [0.1.0] - 2026-05-27

### Added
- **Initial Core Functionality**:
    - `ArrayBackend` class for dispatching between NumPy and CuPy.
    - Environment variable support (`ARRAY_BACKEND`) for selecting the backend.
    - Basic module structure and `src/cunumpy/xp.py`.
- **Project Scaffold**:
    - Documentation setup with Sphinx and nbsphinx.
    - Initial unit tests and testing infrastructure.
    - Basic tutorial notebook and README instructions.
- **Metadata**: Initial configuration of `pyproject.toml` and project metadata.
