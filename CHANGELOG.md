# Changelog

All notable changes to the `cunumpy` library are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Support for Python 3.8 and 3.9 (both end-of-life); `cunumpy` now requires Python 3.10 or newer.

### Changed
- Python 3.14 is supported.
- CI now tests every supported Python version (3.10, 3.11, 3.12, 3.13 and 3.14) instead of 3.8/3.10/3.13.
- `CudaKernel.compile()` passes `compile_options()` to CuPy: the given `options` plus `-DCUNUMPY_INCLUDE_HASH=0x<hash>` when the source includes header files, so CuPy's kernel cache (keyed on source and options only) is invalidated when an included header changes. `options` still returns the options as given.
- `KernelCatalog.from_package(..., include_dirs=None)` is now an explicit keyword; by default the source root of the top-level package (the directory containing it) is an include directory of every CUDA kernel, in addition to the kernel's own folder, so kernels can `#include "my_pkg/common.cuh"`.

### Added
- `xp.CudaKernel`: Wraps a CUDA C kernel (`cupy.RawKernel`, compiled lazily with NVRTC) so it can be called with the same arguments as the host kernel it mirrors, plus `n_threads`. The `extern "C" __global__` signature is parsed once and every call is checked against it: argument count, array dtypes (host arrays raise, they are never copied), and scalars (Python scalars are cast to the declared C types with range checks; lossy or mismatching scalars raise instead of reaching the kernel as silently wrong values). Supports `block_size`, NVRTC `options`, `include_dirs`, `shared_mem`, `stream`, `CudaKernel.from_file()` (`<name>_cuda.cu`), `compile()` and `prepare_args()`; `check_signature=False` skips the checks.
- `xp.CudaArguments`: Base class for argument objects that are flattened into several CUDA kernel arguments; any object with a `__cuda_args__()` method is flattened.
- `xp.parse_cuda_signature(source, name)` and `xp.CudaParameter`: Parse the parameters of a `__global__` function.
- `xp.Kernel`: A host kernel (`PyccelKernel`) and its CUDA counterpart, calling the one matching the active backend. Without a CUDA kernel on the CuPy backend it raises `NotImplementedError` (`missing_cuda="raise"`, default) or falls back to the host kernel with host copies (`missing_cuda="fallback"`).
- `xp.KernelCatalog`: Read-only mapping of `Kernel`s; `KernelCatalog.from_package()` collects them from a package with one folder per kernel (`name/name_kernels.py`, `name/name_cuda.cu`); `without_cuda` lists the kernels still to port.
- `xp.CudaStruct` and `xp.CudaStructValue`: C structs passed to CUDA kernels by value. A `CudaStruct` is defined once from `(field, C type)` pairs; it provides the C `declaration`, the NumPy `dtype` with the C memory layout, and packs values (device arrays as addresses, scalars checked and cast) into a `CudaStructValue` that is passed as one kernel argument. `CudaKernel(..., structs=[...])` checks struct parameters and that a struct definition in the source matches.
- `CudaKernel(..., template_args=...)`: Instantiate C++ function templates (e.g. `template_args=(np.float64, 3)` for `name<double, 3>`); the template parameters are substituted into the checked signature.
- `xp.CudaKernelVariants`: Creates and caches one `CudaKernel` per variant key for generated kernel sources (e.g. per dimension and dtype); `compile_all()` compiles given and existing variants.
- `xp.ctype_of(dtype)`: The C type of a NumPy dtype, e.g. for generating CUDA source.
- 1D to 3D launches: `CudaKernel` accepts a tuple `block_size`, and calls take `n_threads` as an integer or tuple, or an explicit `grid`, plus a per-call `block`; `launch_shape()` returns the `(grid, block)` of a call. Dynamic shared memory (`shared_mem`) and `stream` are passed through by `Kernel` as well.
- Compiling at setup: `CudaKernel.compile()` and `is_compiled`, `Kernel.compile()`, and `KernelCatalog.compile_all()`.
- `Kernel(..., host_options=...)` and `KernelCatalog.from_package(..., host_options=...)`: `PyccelKernel` options (e.g. `object_modules`, `outputs`) for the host kernels, for all kernels or per kernel name; needed for the fallback to find device arrays inside application objects.
- `xp.local_rank()`: The node-local rank from the MPI launcher's environment (Open MPI, MVAPICH2, Intel MPI/MPICH, PMI, Cray PALS, Slurm, `LOCAL_RANK`), available before `MPI_Init`.
- `xp.bind_local_device()`: Selects the GPU `local_rank() % device_count()` and creates its context, before `MPI_Init`, for one-rank-per-GPU MPI programs.
- `xp.synchronize_for_mpi(*arrays)`: Waits for pending work on the current stream before MPI uses device buffers (no-op for host buffers and on the NumPy backend).
- Header-aware compile cache: `xp.resolve_includes(source, include_dirs, base_dir=None)` lists the `#include "..."` files of a CUDA source recursively (resolved relative to the including file, then in `include_dirs`; cycles, system headers and missing files are ignored), and `xp.include_hash(paths)` is a short digest of their contents. `CudaKernel` exposes `included_headers`, `include_dirs`, `source_dir` (set by `from_file`, or the new `source_dir` keyword) and `compile_options()`.
- `cunumpy.testing`: Helpers for testing host/CUDA kernel pairs with pytest (pytest is imported only when its objects are used, never by `import cunumpy`). `requires_cupy` is a `skipif` marker for tests that need a GPU, `BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]` parametrizes a test over the backends, and the `backend` fixture runs a test once per backend with that backend active.
- `cunumpy.testing.assert_kernels_agree(kernel, make_args, *, n_threads=..., rtol=1e-12, atol=0.0, n_calls=1, outputs=None, seed=0)`: Builds the arguments with `make_args(backend, seed)` on the NumPy and the CuPy backend, runs the host and the CUDA kernel of a `Kernel`, and compares the arrays they wrote (the declared `outputs`, or every array argument, including arrays held by argument objects) with `numpy.testing.assert_allclose`, naming the differing argument. Skips the test without a GPU and returns the host arrays.
- `KernelCatalog.parity_cases()`: The `(name, kernel)` pairs of the kernels with a CUDA version, so one test parametrised with them and `assert_kernels_agree` covers a whole catalog.
- `cunumpy.testing.device_function_kernel(header_source, signature, *, name=None, includes=(), n_threads_param="n")`: Generates an elementwise `extern "C" __global__` wrapper around a `__device__` function given its C prototype (pointer parameters are shared, scalar parameters become per-thread arrays, the return value goes into `out`), so device helpers can be tested from Python against their host versions without a hand-written test kernel.
- `xp.as_device_array(value, dtype=None, ndim=None, *, name=None)`: The "reference or copy once" rule for building CUDA argument objects: a CuPy array that already has the requested dtype and is C-contiguous is returned unchanged, anything else (tuples, lists, host arrays, other dtypes, non-contiguous views) becomes one C-contiguous device copy. Raises `RuntimeError` on the NumPy backend, so host data is never copied to the device implicitly, and `ValueError` if `ndim` does not match.
- `CudaKernel` pointer parameters and `CudaStruct` pointer fields now also reject non-C-contiguous arrays with `TypeError`: a kernel reads a pointer as a flat buffer, so a view such as `a[:, 0:3]` would give silently wrong results.
- `xp.KernelArguments` and `xp.resolve_host_args(args, kwargs=None)`: Argument objects with a host and a device form. An object whose type defines `__host_args__()` is replaced by its result (the object the host kernel receives, e.g. a Pyccel class of NumPy arrays) by `Kernel` on the host path and by `PyccelKernel` (so the `missing_cuda="fallback"` path works too); on the CuPy backend `CudaKernel` flattens the same object via `__cuda_args__()`. Owners can expose one `kernel_args` property that builds each form lazily, so call sites never branch on the backend and CPU runs never build device arguments.

## [0.2.0] - 2026-09-28

### Changed
- `xp.get_backend()` now returns the active global backend, mirroring `xp.set_backend()`. Use the new `xp.get_array_backend(array)` to inspect an individual array. Calls to the former `xp.get_backend(array)` must be updated.
- Expanded the README and documentation with backend selection, array conversion, GPU controls, kernel adaptation, and Pyodide examples.

### Fixed
- GPU shape round-trip coverage now creates a zero-dimensional NumPy array for the scalar case instead of calling `.astype()` on a Python float.
- GPU tests now account for CuPy's retained split memory blocks and provide an array conversion method on the custom host-array test fixture.
- `set_backend()` and `use_backend()` now reject unsupported backend names with `ValueError` without changing the active selection.

### Added
- `xp.get_array_module(array)`: Return the array-api-compat module (`numpy`/`cupy`) matching a given array's own backend, regardless of the process-wide active backend. Mirrors `cupy.get_array_module` and works in Pyodide.
- `xp.get_rng(seed=None)`: Return a `numpy.random.Generator`/`cupy.random.Generator` matching the active backend, without having to branch on the backend yourself.
- `xp.device_count()`: Number of visible CUDA devices (`0` without a functional CuPy/CUDA install), independent of the currently active backend.
- `xp.set_device_for_rank(rank, devices_per_node=None)`: Convenience for one-MPI-rank-per-GPU codes; selects `rank % devices_per_node` (defaulting `devices_per_node` to `device_count()`) via `set_device()` and returns the chosen device id.
- `xp.memory_info()`: `(free, total)` bytes of memory on the active CUDA device, or `None` on the NumPy backend.
- `xp.free_memory()`: Release all free blocks held by CuPy's device and pinned-host memory pools (no-op on the NumPy backend).
- `xp.default_float_dtype()`: Return the active backend's `float64` dtype object, for pinning a portable float precision instead of the backend/platform-dependent `dtype=float`.
- `xp.stream()`: Context manager for a CUDA stream, to overlap host/device transfers with compute (no-op, yielding `None`, on the NumPy backend).
- `xp.pin_memory(array)`: Copy a host array into pinned (page-locked) CUDA host memory for faster transfers.
- `PyccelKernel(..., is_array=...)`: Extension point overriding the default `isinstance(value, np.ndarray)` check used to decide which host values returned by (or reachable from a declared output of) the wrapped kernel are converted back to the device -- for kernels that return/mutate a NumPy subclass or other custom host array type.
- Pyodide NumPy support documentation and CI that installs the built wheel in Pyodide's WebAssembly runtime and runs compiler-free tests for arrays, conversions, contexts, and Python kernels without CuPy or Pyccel imports.
- `test-compiled` extra for native Pyccel tests. The `test` extra is now compiler-free; `dev` continues to include compiled-test dependencies.
- `xp.same_backend(*arrays)`: Return `True` if all given arrays live on the same backend.
- `xp.assert_same_backend(*arrays)`: Raise `TypeError` with a clear message if the given arrays don't all live on the same backend, instead of letting a mixed-backend operation fail with a confusing backend-internal error.
- `xp.PyccelKernel`: Wraps a kernel compiled with [pyccel](https://github.com/pyccel/pyccel) (which only accepts NumPy arrays) so it can be called with CuPy arrays. Arguments are copied to the host before the call, in-place kernel updates are copied back to the device, and returned arrays are moved back to the device. On the NumPy backend the kernel is called directly, without conversion. Tuples, lists and dicts are traversed recursively; pass `object_modules=("your_package.",)` to also traverse the attributes of your own objects. Pass `outputs=(5,)` (indices for positional arguments, names for keyword arguments) to declare which arguments the kernel writes to, so only those are copied back to the device instead of every converted array; `outputs=()` declares none. Conversion is identity-aware: an array reachable by several paths (passed twice, or both directly and as an object attribute) becomes a single host array, so the kernel sees the aliasing the caller intended and in-place updates are not lost; reference cycles are handled rather than recursed into.

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
