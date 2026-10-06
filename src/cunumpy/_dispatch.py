"""Pairs of host and CUDA kernels, chosen by the backend or by the arguments.

A :class:`Kernel` holds a host kernel (a :class:`~cunumpy.kernels.PyccelKernel`, e.g. a
Pyccel-compiled function) and, optionally, its 1:1 corresponding CUDA kernel
(:class:`~cunumpy.kernels.CudaKernel`). It calls the host kernel on the NumPy backend and
the CUDA kernel on the CuPy backend (or, with ``dispatch="arrays"``, the CUDA
kernel for device arguments and the host kernel for host arguments), so a code
base can port its kernels to CUDA one by one.

:class:`KernelCatalog` collects such pairs from a package laid out with one
folder per kernel::

    my_kernels/
    ├── __init__.py              # catalog = KernelCatalog.from_package(__name__)
    ├── push/
    │   ├── push_kernels.py      # def push(...): ...   (host kernel)
    │   └── push_cuda.cu         # __global__ void push(...)   (CUDA kernel)
    └── deposit/
        └── deposit_kernels.py   # not ported yet
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import array_api_compat

from cunumpy._cuda_kernel import CudaKernel, _compile_in_threads
from cunumpy._kernel import (
    CompiledHostKernel,
    HostImplementations,
    PyccelKernel,
    get_device_kernel_implementation,
)
from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _record
from cunumpy.xp import get_backend

__all__ = ["Kernel", "KernelCatalog"]

_MISSING_CUDA = ("raise", "fallback")
_DISPATCH = ("backend", "arrays")

# substituted in tests that have no GPU
_is_device_array = array_api_compat.is_cupy_array


def _on_device(arg: Any) -> bool:
    """Whether a kernel argument lives on the GPU.

    A CuPy array, or a CUDA argument object (one with ``__cuda_args__``, e.g. a
    :class:`~cunumpy.arguments.CudaArguments` or a struct value).
    """
    return _is_device_array(arg) or callable(getattr(type(arg), "__cuda_args__", None))


#: Longest name a Fortran compiler accepts. pyccel names the wrapper module of a
#: host kernel module ``bind_c_<module>``, so a kernel module name longer than
#: 63 - len("bind_c_") characters cannot be compiled with the Fortran backend.
FORTRAN_NAME_LIMIT = 63


def _pyccel_stub_parameters(function: Any) -> list[str] | None:
    """Parameter names of a pyccel-compiled function, from its ``.pyi`` stub.

    pyccel writes ``__pyccel__/<module>.pyi`` next to the compiled extension
    module; the compiled function itself has no Python signature. Returns
    None if the stub or the function is not found.
    """
    module = getattr(function, "__self__", None)  # extension functions: the module
    if not isinstance(module, ModuleType):
        name = getattr(function, "__module__", None)
        module = sys.modules.get(name) if isinstance(name, str) else None
    file = getattr(module, "__file__", None)
    func_name = getattr(function, "__name__", None)
    if not file or not func_name:
        return None
    path = Path(file)
    stub = path.parent / "__pyccel__" / (path.name.split(".")[0] + ".pyi")
    if not stub.is_file():
        return None
    try:
        tree = ast.parse(stub.read_text(), filename=str(stub))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [a.arg for a in node.args.posonlyargs + node.args.args]
    return None


def _import_loader(module: str, name: str) -> Callable[[], Callable[..., Any]]:
    """Loads function `name` of `module` on first use (the import may fail)."""
    return lambda: getattr(importlib.import_module(module), name)


def _fallback_loader(
    host_fallback: (
        Mapping[str, Callable[..., Any]]
        | Callable[[str], Callable[..., Any] | None]
        | None
    ),
    name: str,
) -> dict[str, Callable[[], Callable[..., Any]]]:
    """`from_package`'s `host_fallback` for kernel `name`, as its NumPy implementation."""
    fallback = (
        host_fallback(name)
        if callable(host_fallback)
        else (host_fallback or {}).get(name)
    )
    return {} if fallback is None else {"numpy": lambda: fallback}


def _positional_parameters(function: Any) -> list[str] | None:
    """Names of the positional parameters of `function`, or None without a signature."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return [p.name for p in signature.parameters.values() if p.kind in positional]


def _source_root(package: str) -> Path:
    """The directory containing the top-level package of `package`."""
    top = importlib.import_module(package.partition(".")[0])
    if top.__file__ is not None:
        return Path(top.__file__).parent.parent
    return Path(next(iter(top.__path__))).parent  # namespace package


class Kernel:
    """A host kernel and its CUDA counterpart; calls the one matching the backend.

    Parameters
    ----------
    host_kernel : PyccelKernel | callable
        The host kernel, called on the NumPy backend. A plain callable is
        wrapped in a :class:`~cunumpy.kernels.PyccelKernel`.
    cuda_kernel : CudaKernel | None
        The CUDA kernel, called on the CuPy backend; None if it has not been
        written yet.
    name : str | None
        Name of the kernel; defaults to the name of the host kernel.
    missing_cuda : {"raise", "fallback"}
        What happens on the CuPy backend if there is no CUDA kernel: ``"raise"``
        raises ``NotImplementedError``; ``"fallback"`` calls the host kernel
        through :class:`~cunumpy.kernels.PyccelKernel`, which copies the arrays to the
        host and back at every call (a warning is emitted once).
    cuda_path : str | Path | None
        Where the CUDA kernel is expected, for the error message if it is missing.
    host_options : Mapping[str, Any] | None
        Keyword arguments for the :class:`~cunumpy.kernels.PyccelKernel` that wraps a
        plain callable `host_kernel`, e.g. ``{"object_modules": ("my_pkg.",),
        "outputs": (2,)}``; they matter for the fallback on the CuPy backend.
        Not allowed if `host_kernel` already is a ``PyccelKernel``.
    dispatch : {"backend", "arrays"}
        How a call chooses between the two kernels. ``"backend"`` (default):
        the CUDA kernel on the CuPy backend, the host kernel on the NumPy
        backend. ``"arrays"``: the CUDA kernel if any top-level argument lives
        on the GPU (a CuPy array, or a device-only argument object such as a
        :class:`~cunumpy.arguments.CudaArguments` or a struct value), else the host
        kernel, whatever the backend. Use ``"arrays"`` when a code deliberately
        hands host arrays to kernels while CuPy is active (diagnostics, MPI
        staging, CPU fallbacks): the host kernel then runs on the host arrays
        instead of the CUDA kernel rejecting them.

    Notes
    -----
    Both kernels take the same arguments, except that the CUDA kernel gets the
    launch shape (``n_threads`` or ``grid``) and argument objects in their CUDA
    form (see :class:`~cunumpy.arguments.CudaArguments` and :class:`~cunumpy.arguments.CudaStruct`).
    Argument objects are passed as they are: the caller passes the host
    argument object (e.g. a pyccel class) on the host and the CUDA one (a
    :class:`~cunumpy.arguments.CudaStructArguments`) on the device; see
    :doc:`/kernels/arguments`.
    """

    def __init__(
        self,
        host_kernel: PyccelKernel | Callable[..., Any],
        cuda_kernel: CudaKernel | None = None,
        *,
        name: str | None = None,
        missing_cuda: str = "raise",
        cuda_path: str | Path | None = None,
        host_options: Mapping[str, Any] | None = None,
        dispatch: str = "backend",
        test_args: str | None = None,
    ) -> None:
        if dispatch not in _DISPATCH:
            raise ValueError(f"dispatch must be one of {_DISPATCH}, got {dispatch!r}")
        self._dispatch = dispatch
        if missing_cuda not in _MISSING_CUDA:
            raise ValueError(
                f"missing_cuda must be one of {_MISSING_CUDA}, got {missing_cuda!r}",
            )
        if cuda_kernel is not None and not isinstance(cuda_kernel, CudaKernel):
            raise TypeError(
                "cuda_kernel must be a CudaKernel or None, "
                f"got {type(cuda_kernel).__name__}",
            )
        if not isinstance(host_kernel, PyccelKernel):
            host_kernel = PyccelKernel(host_kernel, **(host_options or {}))
        elif host_options:
            raise ValueError(
                "host_options are for wrapping a plain callable; configure the "
                "given PyccelKernel directly",
            )
        self._host_kernel = host_kernel
        self._cuda_kernel = cuda_kernel
        self._name = name if name is not None else host_kernel.name
        self._missing_cuda = missing_cuda
        self._cuda_path = None if cuda_path is None else Path(cuda_path)
        self._test_args_module = test_args
        self._test_args: ModuleType | None = None
        self._warned = False

    @classmethod
    def from_folder(
        cls,
        package: str,
        *,
        host_suffix: str = "_kernels",
        cuda_suffix: str = "_cuda.cu",
        test_args_suffix: str | None = "_test_args",
        check_name_length: bool = True,
        missing_cuda: str = "raise",
        host_options: Mapping[str, Any] | None = None,
        include_dirs: Sequence[str | Path] | None = None,
        dispatch: str = "backend",
        compile_host: Callable[[Any], Any] | None = None,
        extra_implementations: (
            Mapping[str, Callable[[], Callable[..., Any]]] | None
        ) = None,
        **cuda_options: Any,
    ) -> Kernel:
        """The kernel of one kernel folder, the unit of :meth:`KernelCatalog.from_package`.

        `package` is the dotted name of the folder ``<name>``. Every file of the
        folder that defines a version of the kernel (a function ``<name>``, or
        the ``__global__`` function ``<name>``) is one implementation:

        * ``<name><host_suffix>.py``: ``"pyccel"``, compiled with `compile_host`
          (as it is without one), and ``"python"``, the same file uncompiled;
        * ``<name>_numba.py``: ``"numba"`` (the function decorated there, e.g.
          with ``numba.njit``; unavailable if the import fails);
        * ``<name>_numpy.py``: ``"numpy"``;
        * ``<name><cuda_suffix>``: the CUDA kernel.

        The host implementations form a :class:`~cunumpy.kernels.HostImplementations`:
        a call runs the one set with :func:`~cunumpy.kernels.set_host_kernel_implementation`,
        or by default the first available of pyccel, numba and NumPy. The
        folder's own ``__init__.py`` can declare its kernel with this method, so
        that the kernel is imported from where it is written::

            # my_sim/kernels/push/__init__.py
            kernel = xp.kernels.Kernel.from_folder(
                __name__, host_suffix="_pyccel", compile_host=compile, dispatch="arrays"
            )

            # anywhere in the code
            from my_sim.kernels.push import kernel as push

        Nothing is compiled here: the CUDA source is parsed, the host module
        imported.

        Parameters
        ----------
        package : str
            Dotted name of the kernel folder (``__name__`` in its ``__init__.py``).
        host_suffix, cuda_suffix, test_args_suffix, check_name_length
            As for :meth:`KernelCatalog.from_package`.
        missing_cuda, host_options, dispatch
            Passed on to :class:`Kernel`.
        include_dirs : Sequence[str | Path] | None
            Include directories of the CUDA kernel, in addition to the folder
            itself; by default the source root of the top-level package (the
            directory containing it).
        compile_host : Callable | None
            Returns the compiled form of the ``<name><host_suffix>`` module
            (cunumpy does not compile anything itself, e.g. a wrapper around
            ``pyccel.epyccel`` with a cache), called on first use of the pyccel
            implementation; if it raises, that implementation is unavailable.
        extra_implementations : Mapping | None
            Loaders of host implementations that are not files of the folder,
            by implementation name, e.g. ``{"numpy": lambda: push_numpy}``.
        **cuda_options
            Passed on to :meth:`CudaKernel.from_file`, e.g. ``block_size`` or
            ``n_threads_from``.

        Raises
        ------
        FileNotFoundError
            If the folder has no host kernel module.
        """
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError(f"{package!r} is not a package (kernel folder)")
        folder = Path(next(iter(spec.submodule_search_locations)))
        name = package.rpartition(".")[2]
        if not (folder / f"{name}{host_suffix}.py").is_file():
            raise FileNotFoundError(
                f"kernel folder {folder} has no host kernel {name}{host_suffix}.py",
            )
        wrapper = f"bind_c_{name}{host_suffix}"
        if check_name_length and len(wrapper) > FORTRAN_NAME_LIMIT:
            warnings.warn(
                f"kernel {name!r}: the module name {name}{host_suffix} gives the "
                f"pyccel Fortran wrapper module {wrapper!r} ({len(wrapper)} "
                f"characters), longer than Fortran's limit of "
                f"{FORTRAN_NAME_LIMIT}; shorten the kernel name to at most "
                f"{FORTRAN_NAME_LIMIT - len('bind_c_') - len(host_suffix)} "
                "characters or compile with the C backend",
                stacklevel=2,
            )
        test_args = None
        if (
            test_args_suffix is not None
            and (folder / f"{name}{test_args_suffix}.py").is_file()
        ):
            test_args = f"{package}.{name}{test_args_suffix}"
        module = importlib.import_module(f"{package}.{name}{host_suffix}")
        python = getattr(module, name)
        loaders: dict[str, Callable[[], Callable[..., Any]]] = {
            "python": lambda: python,
            "pyccel": (
                (lambda: getattr(compile_host(module), name))
                if compile_host is not None
                else (lambda: python)
            ),
        }
        for implementation in ("numba", "numpy"):
            if (folder / f"{name}_{implementation}.py").is_file():
                loaders[implementation] = _import_loader(
                    f"{package}.{name}_{implementation}",
                    name,
                )
        loaders.update(extra_implementations or {})
        host = HostImplementations(name, loaders)
        if include_dirs is None:
            include_dirs = (_source_root(package),)
        cuda_options["include_dirs"] = tuple(include_dirs)
        cuda_path = folder / f"{name}{cuda_suffix}"
        cuda_kernel = (
            CudaKernel.from_file(cuda_path, name, **cuda_options)
            if cuda_path.is_file()
            else None
        )
        return cls(
            host,
            cuda_kernel,
            name=name,
            missing_cuda=missing_cuda,
            cuda_path=cuda_path,
            host_options=host_options,
            dispatch=dispatch,
            test_args=test_args,
        )

    def __repr__(self) -> str:
        return (
            f"Kernel(name={self._name!r}, cuda={self._cuda_kernel is not None}, "
            f"missing_cuda={self._missing_cuda!r})"
        )

    @property
    def name(self) -> str:
        """Name of the kernel."""
        return self._name

    @property
    def host_kernel(self) -> PyccelKernel:
        """The host kernel."""
        return self._host_kernel

    @property
    def cuda_kernel(self) -> CudaKernel | None:
        """The CUDA kernel, or None if there is none."""
        return self._cuda_kernel

    @property
    def has_cuda(self) -> bool:
        """Whether there is a CUDA kernel."""
        return self._cuda_kernel is not None

    @property
    def missing_cuda(self) -> str:
        """What happens on the CuPy backend without CUDA kernel.

        ``"raise"`` or ``"fallback"``, see :class:`Kernel`.
        """
        return self._missing_cuda

    @property
    def cuda_path(self) -> Path | None:
        """Where the CUDA kernel is expected, if known."""
        return self._cuda_path

    @property
    def dispatch(self) -> str:
        """How calls choose a kernel: ``"backend"`` or ``"arrays"``."""
        return self._dispatch

    @property
    def test_args_module(self) -> str | None:
        """Dotted name of the module with the test arguments of this kernel, or None.

        Set by :meth:`KernelCatalog.from_package` for a kernel folder that
        contains ``<name>_test_args.py``; see :func:`cunumpy.kernel_testing.check_parity`.
        """
        return self._test_args_module

    @property
    def test_args(self) -> ModuleType | None:
        """The test-arguments module, imported on first access, or None."""
        if self._test_args is None and self._test_args_module is not None:
            self._test_args = importlib.import_module(self._test_args_module)
        return self._test_args

    def host_parameters(self) -> list[str] | None:
        """The parameter names of the host kernel, or None if they are unknown.

        Read from the Python function (for a :class:`~cunumpy.kernels.CompiledHostKernel`,
        its uncompiled Python version), or, for a pyccel-compiled function
        without a Python signature, from the ``__pyccel__/<module>.pyi`` stub
        pyccel writes next to the extension module. None if neither is
        available.
        """
        function = self._host_kernel.kernel
        if isinstance(function, (CompiledHostKernel, HostImplementations)):
            function = function.python
        parameters = _positional_parameters(function)
        return (
            parameters if parameters is not None else _pyccel_stub_parameters(function)
        )

    def check_signature(self) -> None:
        """Check that all implementations of the kernel take the same parameters.

        Compares the parameter names of the host function with those of the
        parsed ``__global__`` signature, with those of the other host
        implementations of a :class:`~cunumpy.kernels.HostImplementations` (numba,
        NumPy; nothing is compiled, an implementation that fails to import is
        skipped) and with the fallback of a :class:`~cunumpy.kernels.CompiledHostKernel`.
        A side is skipped without a CUDA kernel, without a parsed CUDA signature
        (``check_signature=False``), or without a Python signature.

        Raises
        ------
        ValueError
            If the names or their order differ; the message shows both lists.
        """
        host = self.host_parameters()
        if host is None:
            return
        if self._cuda_kernel is not None and self._cuda_kernel.signature is not None:
            cuda = [param.name for param in self._cuda_kernel.signature]
            if host != cuda:
                raise ValueError(
                    f"kernel {self._name!r}: the host kernel takes "
                    f"({', '.join(host)}), the CUDA kernel ({', '.join(cuda)})",
                )
        function = self._host_kernel.kernel
        others: dict[str, Any] = {}
        if isinstance(function, CompiledHostKernel) and function.fallback is not None:
            others["fallback"] = function.fallback
        elif isinstance(function, HostImplementations):
            for name in ("numba", "numpy"):
                if function.available(name):
                    # a numba dispatcher keeps the Python function in py_func
                    implementation = function.get(name)
                    others[name] = getattr(implementation, "py_func", implementation)
        for name, implementation in others.items():
            parameters = _positional_parameters(implementation)
            if parameters is not None and parameters != host:
                which = "its fallback" if name == "fallback" else f"the {name} version"
                raise ValueError(
                    f"kernel {self._name!r}: the host kernel takes "
                    f"({', '.join(host)}), {which} ({', '.join(parameters)})",
                )

    @property
    def implementations(self) -> tuple[str, ...]:
        """Names of the implementations, e.g. ``("pyccel", "numpy", "python", "cuda")``.

        A host kernel that is not a :class:`~cunumpy.kernels.HostImplementations` counts
        as ``"host"``.
        """
        function = self._host_kernel.kernel
        host = (
            function.names if isinstance(function, HostImplementations) else ("host",)
        )
        return (*host, "cuda") if self._cuda_kernel is not None else host

    def selected(self, device: bool = False) -> str:
        """The implementation a call with host (or `device`) arguments runs now.

        For host arguments: the setting of
        :func:`~cunumpy.kernels.set_host_kernel_implementation` or the default (loads it), or
        ``"host"`` for a host kernel that is not a
        :class:`~cunumpy.kernels.HostImplementations`. For device arguments ``"cuda"``,
        or ``"host"`` if there is no CUDA kernel and ``missing_cuda="fallback"``.
        Explicit CUDA selection rejects missing CUDA implementations with
        ``LookupError``, including when host fallback is configured.
        Useful to check that a run does not use a slow path.
        """
        if device:
            return "cuda" if self._device_kernel() is self._cuda_kernel else "host"
        function = self._host_kernel.kernel
        return (
            function.selected() if isinstance(function, HostImplementations) else "host"
        )

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """The kernel for the active backend.

        Call this once at setup to fail early if a CUDA kernel is missing.

        Raises
        ------
        LookupError
            On the CuPy backend, if CUDA is explicitly selected but missing.
        NotImplementedError
            On the CuPy backend, if there is no CUDA kernel and
            ``missing_cuda="raise"``.
        """
        if get_backend() != "cupy":
            return self._host_kernel
        return self._device_kernel()

    def _device_kernel(self) -> PyccelKernel | CudaKernel:
        """Resolve the device selection, honoring fallback only in automatic mode."""
        if self._cuda_kernel is not None:
            return self._cuda_kernel
        if get_device_kernel_implementation() == "cuda":
            raise LookupError(
                f"kernel {self._name!r} has no 'cuda' implementation "
                "(explicitly selected device implementation)",
            )
        if self._missing_cuda == "raise":
            expected = (
                "" if self._cuda_path is None else f" (expected {self._cuda_path})"
            )
            raise NotImplementedError(
                f"No CUDA version of kernel {self._name!r}{expected}.",
            )
        if not self._warned:
            warnings.warn(
                f"No CUDA version of kernel {self._name!r}: calling the host kernel, "
                "which copies its arrays to the host and back at every call.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True
        return self._host_kernel

    def compile(self) -> bool:
        """Compile the CUDA kernel now, if there is one.

        Returns
        -------
        bool
            Whether there is a CUDA kernel (and it was compiled).
        """
        if self._cuda_kernel is None:
            return False
        self._cuda_kernel.compile()
        return True

    def __call__(
        self,
        *args: Any,
        n_threads: int | Sequence[int] | None = None,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
        shared_mem: int = 0,
        stream: Any = None,
    ) -> Any:
        """Call the kernel for the active backend.

        Parameters
        ----------
        *args
            Kernel arguments, as the selected kernel takes them.
        n_threads, grid, block, shared_mem, stream
            Launch configuration of the CUDA kernel, see
            :meth:`CudaKernel.__call__ <cunumpy.kernels.CudaKernel.__call__>`;
            Thread counts are inferred from array shapes by default;
            `n_threads` or `grid` overrides that choice.
            Ignored by the host kernel.
        """
        if self._dispatch == "arrays":
            on_device = any(_on_device(arg) for arg in args)
            kernel = self._device_kernel() if on_device else self._host_kernel
        else:
            on_device = get_backend() == "cupy"
            kernel = self.get_kernel()
        if kernel is self._host_kernel:
            if _COUNTERS and self._cuda_kernel is None and on_device:
                _record(
                    "fallback",
                    f"Kernel {self._name!r} has no CUDA kernel: host kernel "
                    "called on the CuPy backend",
                )
            if (
                self._dispatch == "arrays"
                and not on_device
                and kernel._use_cupy is not True
            ):
                # only host arguments, by the choice above: nothing to convert,
                # even while CuPy is the active backend
                return kernel.kernel(*args)
            return kernel(*args)
        if n_threads is None and grid is None and kernel.n_threads_from is None:
            raise ValueError(
                f"{self._name}: n_threads is required to launch the CUDA kernel "
                "(or pass grid, or set cuda_kernel.n_threads_from)",
            )
        return kernel(
            *args,
            n_threads=n_threads,
            grid=grid,
            block=block,
            shared_mem=shared_mem,
            stream=stream,
        )


class KernelCatalog(Mapping):
    """Kernels by name.

    A read-only mapping from names to :class:`Kernel` objects, usually built with
    :meth:`from_package`; :meth:`register` adds kernels one by one.

    Parameters
    ----------
    kernels : Mapping[str, Kernel] | None
        Initial kernels.
    """

    def __init__(self, kernels: Mapping[str, Kernel] | None = None) -> None:
        self._kernels: dict[str, Kernel] = {}
        for name, kernel in (kernels or {}).items():
            self.register(kernel, name=name)

    @classmethod
    def from_package(
        cls,
        package: str,
        *,
        host_suffix: str = "_kernels",
        cuda_suffix: str = "_cuda.cu",
        test_args_suffix: str | None = "_test_args",
        check_name_length: bool = True,
        missing_cuda: str = "raise",
        host_options: (
            Mapping[str, Any] | Callable[[str], Mapping[str, Any]] | None
        ) = None,
        include_dirs: Sequence[str | Path] | None = None,
        dispatch: str = "backend",
        compile_host: Callable[[Any], Any] | None = None,
        host_fallback: (
            Mapping[str, Callable[..., Any]]
            | Callable[[str], Callable[..., Any] | None]
            | None
        ) = None,
        **cuda_options: Any,
    ) -> KernelCatalog:
        """Collect the kernels of a package with one folder per kernel.

        For every subfolder ``<name>`` of the package that contains the module
        ``<name><host_suffix>.py``, the function ``<name>`` of that module is the
        host kernel, and ``<name><cuda_suffix>`` in the same folder, if present,
        is the CUDA kernel (with a ``__global__`` function ``<name>``). Other
        ``__global__`` functions in that file are ignored by the catalog; load
        them with :meth:`CudaKernel.all_from_file
        <cunumpy.kernels.CudaKernel.all_from_file>`.

        Parameters
        ----------
        package : str
            Full name of the package, e.g. ``__name__`` in its ``__init__.py``.
        host_suffix : str
            Module name suffix of the host kernels.
        cuda_suffix : str
            File name suffix of the CUDA kernels.
        test_args_suffix : str | None
            Module name suffix of the test arguments: ``<name><test_args_suffix>.py``
            in the kernel's folder, if present, is recorded as
            :attr:`Kernel.test_args_module` (imported only when a test asks for
            it, see :func:`cunumpy.kernel_testing.check_parity`). None disables this.
        check_name_length : bool
            Warn about a kernel whose module name is too long for the Fortran
            backend of pyccel: the wrapper module ``bind_c_<name><host_suffix>``
            must fit :data:`FORTRAN_NAME_LIMIT` (63) characters, so with the
            default suffix a kernel name has at most 48 characters.
        missing_cuda : {"raise", "fallback"}
            Passed on to every :class:`Kernel`.
        host_options : Mapping | Callable[[str], Mapping] | None
            Keyword arguments for the :class:`~cunumpy.kernels.PyccelKernel` wrapping each
            host kernel (see :class:`Kernel`): the same for all kernels, or a
            function of the kernel name, e.g. to declare per-kernel ``outputs``.
        include_dirs : Sequence[str | Path] | None
            Include directories of the CUDA kernels, in addition to each
            kernel's own folder. By default the source root of the top-level
            package (the directory containing it), so that a kernel in
            ``my_pkg.kernels`` can ``#include "my_pkg/common.cuh"``.
        dispatch : {"backend", "arrays"}
            Passed on to every :class:`Kernel`: choose the CUDA kernel by the
            active backend or by where the arguments live.
        compile_host : Callable | None
            Compiles a host kernel module, e.g.
            a function wrapping ``pyccel.epyccel`` (cunumpy does not compile
            anything itself). Each host kernel then is a
            :class:`~cunumpy.kernels.CompiledHostKernel`: compiled on its first
            call, falling back to
            `host_fallback`, or to the uncompiled Python function with a
            warning, if compilation fails. By default the Python function is
            called as it is.
        host_fallback : Mapping | Callable | None
            The fallback per kernel name, for a failed compilation (e.g. a
            vectorized NumPy implementation with the same arguments): a mapping
            from names to callables, or a function of the name returning a
            callable or None.
        **cuda_options
            Passed on to :meth:`CudaKernel.from_file`, e.g. ``block_size`` or
            ``structs``.
        """
        root = Path(importlib.import_module(package).__file__).parent
        if include_dirs is None:
            include_dirs = (_source_root(package),)
        kernels = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            name = folder.name
            if not (folder / f"{name}{host_suffix}.py").is_file():
                continue
            kernels[name] = Kernel.from_folder(
                f"{package}.{name}",
                host_suffix=host_suffix,
                cuda_suffix=cuda_suffix,
                test_args_suffix=test_args_suffix,
                check_name_length=check_name_length,
                missing_cuda=missing_cuda,
                host_options=(
                    host_options(name) if callable(host_options) else host_options
                ),
                include_dirs=include_dirs,
                dispatch=dispatch,
                compile_host=compile_host,
                extra_implementations=_fallback_loader(host_fallback, name),
                **cuda_options,
            )
        return cls(kernels)

    def register(self, kernel: Kernel, name: str | None = None) -> Kernel:
        """Add a kernel under `name` (by default its own name) and return it."""
        if not isinstance(kernel, Kernel):
            raise TypeError(f"expected a Kernel, got {type(kernel).__name__}")
        name = kernel.name if name is None else name
        if name in self._kernels:
            raise KeyError(f"a kernel named {name!r} is already registered")
        self._kernels[name] = kernel
        return kernel

    def __getitem__(self, name: str) -> Kernel:
        return self._kernels[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._kernels)

    def __len__(self) -> int:
        return len(self._kernels)

    def __repr__(self) -> str:
        return (
            f"KernelCatalog({len(self)} kernels, {len(self.without_cuda)} without CUDA)"
        )

    def __str__(self) -> str:
        return self.summary()

    def summary(self, max_missing: int = 10) -> str:
        """One line on the porting status, e.g. for a ``--status`` command.

        Parameters
        ----------
        max_missing : int
            How many of the kernels without CUDA kernel to name; the rest is
            shortened to ``...``.

        Returns
        -------
        str
            ``"CUDA kernels: 3 of 60 (missing: a, b, c)"``; without the
            parenthesis if every kernel has a CUDA kernel.
        """
        text = f"CUDA kernels: {len(self.with_cuda)} of {len(self)}"
        missing = self.without_cuda
        if missing:
            names = missing[:max_missing]
            if len(missing) > max_missing:
                names.append("...")
            text += f" (missing: {', '.join(names)})"
        return text

    @property
    def with_cuda(self) -> list[str]:
        """Names of the kernels with a CUDA kernel."""
        return [name for name, kernel in self._kernels.items() if kernel.has_cuda]

    @property
    def without_cuda(self) -> list[str]:
        """Names of the kernels without a CUDA kernel, i.e. still to port."""
        return [name for name, kernel in self._kernels.items() if not kernel.has_cuda]

    def parity_cases(self) -> list[tuple[str, Kernel]]:
        """The ``(name, kernel)`` pairs of the kernels that have a CUDA kernel.

        For a parametrised parity test of the whole catalog with
        :func:`cunumpy.kernel_testing.assert_kernels_agree`::

            @pytest.mark.parametrize("name, kernel", catalog.parity_cases())
            def test_parity(name, kernel):
                assert_kernels_agree(kernel, make_args[name], n_threads=1000)
        """
        return [
            (name, kernel) for name, kernel in self._kernels.items() if kernel.has_cuda
        ]

    def check_signatures(self) -> None:
        """Check every kernel's host and CUDA parameters (:meth:`Kernel.check_signature`).

        A cheap test for a ported package: call it once, e.g. in a unit test,
        to catch a CUDA kernel whose parameters are missing, extra or in
        another order than those of its host kernel.

        Raises
        ------
        ValueError
            Listing every kernel whose two signatures differ.
        """
        problems = []
        for kernel in self._kernels.values():
            try:
                kernel.check_signature()
            except ValueError as error:
                problems.append(str(error))
        if problems:
            raise ValueError(
                "host and CUDA kernels take different parameters:\n  "
                + "\n  ".join(problems),
            )

    def compile_all(self, jobs: int | None = 1) -> list[str]:
        """Compile all CUDA kernels now, e.g. at setup instead of in the first step.

        Compilation is cached on disk by CuPy, so after the first run this mostly
        loads the compiled kernels.

        Parameters
        ----------
        jobs : int | None
            Number of kernels compiled at a time. With ``jobs > 1`` the kernels
            are compiled in threads (NVRTC compilation releases the GIL, and
            all threads use the current device); None uses the number of CPUs.
            All kernels are compiled even if one fails; the first error is
            raised afterwards.

        Returns
        -------
        list[str]
            Names of the compiled kernels.

        Raises
        ------
        RuntimeError
            If CuPy or a GPU is not available (and there are CUDA kernels).
        """
        return _compile_in_threads(
            {
                name: kernel.compile
                for name, kernel in self._kernels.items()
                if kernel.has_cuda
            },
            jobs,
        )
