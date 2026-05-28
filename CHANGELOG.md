# Changelog

All notable changes to the `cunumpy` library are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
