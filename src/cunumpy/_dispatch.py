"""Pairs of host and CUDA kernels, chosen by the backend or by the arguments.

A :class:`~cunumpy.kernels.Kernel` holds a host kernel (a
:class:`~cunumpy.kernels.PyccelKernel`) and, optionally, its CUDA counterpart
(a :class:`~cunumpy.kernels.CudaKernel`), so a code base can port its kernels
to CUDA one by one. :class:`~cunumpy.kernels.KernelCatalog` collects such
pairs from a package with one folder per kernel (see :doc:`/kernels/dispatch`)::

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
    outputs_from_annotations,
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
    """Whether `arg` is a CuPy array or a CUDA argument object (``__cuda_args__``)."""
    return _is_device_array(arg) or callable(getattr(type(arg), "__cuda_args__", None))


#: Longest name a Fortran compiler accepts. Pyccel names the wrapper module of a
#: host kernel module ``bind_c_<module>``, so a module name longer than
#: ``63 - len("bind_c_")`` characters cannot be compiled with the Fortran backend.
FORTRAN_NAME_LIMIT = 63


def _pyccel_stub_parameters(function: Any) -> list[str] | None:
    """Read the parameters of a Pyccel-compiled function from its ``.pyi`` stub."""
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
    """Return a loader of function `name` of `module` (imported on first use)."""
    return lambda: getattr(importlib.import_module(module), name)


def _fallback_loader(
    host_fallback: (
        Mapping[str, Callable[..., Any]]
        | Callable[[str], Callable[..., Any] | None]
        | None
    ),
    name: str,
) -> dict[str, Callable[[], Callable[..., Any]]]:
    """Return the `host_fallback` of kernel `name` as its ``"numpy"`` loader."""
    fallback = (
        host_fallback(name)
        if callable(host_fallback)
        else (host_fallback or {}).get(name)
    )
    return {} if fallback is None else {"numpy": lambda: fallback}


def _positional_parameters(function: Any) -> list[str] | None:
    """Return the positional parameter names of `function`, or None if unknown."""
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
    """Return the directory containing the top-level package of `package`."""
    top = importlib.import_module(package.partition(".")[0])
    if top.__file__ is not None:
        return Path(top.__file__).parent.parent
    return Path(next(iter(top.__path__))).parent  # namespace package


class Kernel:
    """A host kernel and its CUDA counterpart; a call runs the one matching the backend.

    See :doc:`/kernels/dispatch` for the porting workflow.

    Parameters
    ----------
    host_kernel : PyccelKernel or callable
        The host kernel. A plain callable is wrapped in a
        :class:`~cunumpy.kernels.PyccelKernel`.
    cuda_kernel : CudaKernel, optional
        The CUDA kernel; None if it has not been written yet.
    name : str, optional
        Name of the kernel; by default the name of the host kernel.
    missing_cuda : {"raise", "fallback"}, optional
        Behavior on the device without CUDA kernel: ``"raise"`` (default)
        raises ``NotImplementedError``; ``"fallback"`` calls the host kernel,
        which copies the arrays to the host and back at every call (a
        ``RuntimeWarning`` is emitted once).
    cuda_path : str or Path, optional
        Where the CUDA kernel is expected, named in the error if it is missing.
    host_options : mapping, optional
        Keyword arguments for the :class:`~cunumpy.kernels.PyccelKernel`
        wrapping a plain callable, e.g. ``{"outputs": (2,)}``; they matter
        for the fallback. Not allowed with a `host_kernel` that already is a
        ``PyccelKernel``.
    dispatch : {"backend", "arrays"}, optional
        ``"backend"`` (default): the CUDA kernel on the CuPy backend, the
        host kernel on the NumPy backend. ``"arrays"``: the CUDA kernel if
        any top-level argument is on the GPU (a CuPy array or an object with
        ``__cuda_args__``), else the host kernel called directly, whatever the
        backend; for codes that hand host arrays to kernels while CuPy is
        active (diagnostics, MPI staging).
    test_args : str, optional
        Dotted name of the test-arguments module of the kernel (see
        :func:`~cunumpy.kernel_testing.check_parity`).

    Raises
    ------
    ValueError
        If `dispatch` or `missing_cuda` is unknown, or `host_options` is
        given with a ``PyccelKernel``.
    TypeError
        If `cuda_kernel` is not a :class:`~cunumpy.kernels.CudaKernel`.

    Notes
    -----
    Both kernels take the same arguments, except that the CUDA kernel gets the
    launch size and argument objects in their CUDA form: the caller passes
    the host object (e.g. a Pyccel class) on the host and the CUDA one (e.g. a
    :class:`~cunumpy.arguments.CudaStructArguments`) on the device; see
    :doc:`/kernels/arguments`.

    Examples
    --------
    >>> def scale(x, out):
    ...     out[:] = 2 * x
    >>> kernel = xp.kernels.Kernel(scale)
    >>> out = np.empty(3)
    >>> kernel(np.arange(3.0), out)
    >>> out
    array([0., 2., 4.])
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
        if host_kernel._parameters is None:
            # compiled kernels have no Python signature: name the parameters
            # from the host function, so that outputs may be given by name
            host_kernel._parameters = self.host_parameters
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
        outputs: Sequence[int | str] | str | None = None,
        include_dirs: Sequence[str | Path] | None = None,
        dispatch: str = "backend",
        compile_host: Callable[[Any], Any] | None = None,
        extra_implementations: (
            Mapping[str, Callable[[], Callable[..., Any]]] | None
        ) = None,
        **cuda_options: Any,
    ) -> Kernel:
        """Build the kernel of one kernel folder.

        Each version of ``<name>`` in the folder is an implementation:
        ``<name><host_suffix>.py`` gives ``"pyccel"`` (compiled with
        `compile_host` on first use) and ``"python"`` (uncompiled),
        ``<name>_numba.py`` gives ``"numba"``, ``<name>_numpy.py`` gives
        ``"numpy"`` and ``<name><cuda_suffix>`` the CUDA kernel. The host ones
        form a :class:`~cunumpy.kernels.HostImplementations`. Nothing is
        compiled here: the host module is imported, the CUDA source parsed.

        Parameters
        ----------
        package : str
            Dotted name of the kernel folder (``__name__`` in its ``__init__.py``).
        host_suffix : str, optional
            Module name suffix of the host kernel.
        cuda_suffix : str, optional
            File name suffix of the CUDA kernel.
        test_args_suffix : str or None, optional
            Module name suffix of the test arguments; None disables them.
        check_name_length : bool, optional
            Warn if the module name is too long for Pyccel's Fortran backend.
        missing_cuda : {"raise", "fallback"}, optional
            Passed on to :class:`~cunumpy.kernels.Kernel`.
        host_options : mapping, optional
            Passed on to :class:`~cunumpy.kernels.Kernel`.
        outputs : sequence of int or str, or "annotations", optional
            The arguments the host kernel writes to, copied back to the device
            on the fallback path. ``"annotations"`` takes every parameter that
            is not ``Final`` or a scalar (see
            :func:`~cunumpy.kernels.outputs_from_annotations`). An
            ``"outputs"`` entry of `host_options` takes precedence.
        include_dirs : sequence of str or Path, optional
            Include directories of the CUDA kernel besides the folder; by
            default the directory containing the top-level package.
        dispatch : {"backend", "arrays"}, optional
            Passed on to :class:`~cunumpy.kernels.Kernel`.
        compile_host : callable, optional
            Returns the compiled form of the host module (e.g. a wrapper of
            ``pyccel.epyccel`` with a cache); cunumpy compiles nothing itself.
            If it raises, the ``"pyccel"`` implementation is unavailable.
        extra_implementations : mapping, optional
            Loaders of further host implementations by name, e.g.
            ``{"numpy": lambda: push_numpy}``.
        **cuda_options
            Passed on to :meth:`CudaKernel.from_file
            <cunumpy.kernels.CudaKernel.from_file>`, e.g. ``block_size``.

        Returns
        -------
        Kernel
            The kernel of the folder.

        Raises
        ------
        ModuleNotFoundError
            If `package` is not a package.
        FileNotFoundError
            If the folder has no host kernel module.

        See Also
        --------
        KernelCatalog.from_package : All kernel folders of a package.

        Examples
        --------
        >>> # my_sim/kernels/push/__init__.py
        >>> kernel = xp.kernels.Kernel.from_folder(  # doctest: +SKIP
        ...     __name__, host_suffix="_pyccel", dispatch="arrays"
        ... )
        >>> kernel.implementations  # doctest: +SKIP
        ('pyccel', 'python', 'cuda')
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
        if outputs is not None and "outputs" not in (host_options or {}):
            declared = (
                outputs_from_annotations(python)
                if isinstance(outputs, str) and outputs == "annotations"
                else outputs
            )
            if isinstance(outputs, str) and outputs != "annotations":
                raise ValueError(
                    f"outputs must be a sequence of names/indices or 'annotations', "
                    f"got {outputs!r}",
                )
            if declared is not None:
                host_options = {**(host_options or {}), "outputs": declared}
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
        """Behavior on the device without CUDA kernel: ``"raise"`` or ``"fallback"``."""
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
        """Dotted name of the test-arguments module of the kernel, or None."""
        return self._test_args_module

    @property
    def test_args(self) -> ModuleType | None:
        """The test-arguments module, imported on first access, or None."""
        if self._test_args is None and self._test_args_module is not None:
            self._test_args = importlib.import_module(self._test_args_module)
        return self._test_args

    def host_parameters(self) -> list[str] | None:
        """Return the parameter names of the host kernel.

        Read from the Python function (the uncompiled version of a
        :class:`~cunumpy.kernels.CompiledHostKernel`), or else from the
        ``__pyccel__/<module>.pyi`` stub Pyccel writes next to a compiled module.

        Returns
        -------
        list of str or None
            The names, or None if neither source is available.
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

        Compares the host parameter names with the parsed ``__global__``
        signature, the numba and NumPy implementations of a
        :class:`~cunumpy.kernels.HostImplementations` (nothing is compiled;
        one that fails to import is skipped) and the fallback of a
        :class:`~cunumpy.kernels.CompiledHostKernel`. A side without a
        signature is skipped.

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
    def implementations(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The names of the implementations, e.g. ``("pyccel", "python", "cuda")``.

        A host kernel that is not a :class:`~cunumpy.kernels.HostImplementations`
        counts as ``"host"``.
        """
        function = self._host_kernel.kernel
        host = (
            function.names if isinstance(function, HostImplementations) else ("host",)
        )
        return (*host, "cuda") if self._cuda_kernel is not None else host

    def selected(self, device: bool = False) -> str:
        """Return the implementation a call runs now, e.g. to rule out a slow path.

        Parameters
        ----------
        device : bool, optional
            Ask for device arguments instead of host arguments.

        Returns
        -------
        str
            For host arguments the implementation chosen by
            :func:`~cunumpy.kernels.set_host_kernel_implementation` or the
            default (loaded now), or ``"host"`` for a plain host kernel. For
            device arguments ``"cuda"``, or ``"host"`` for the fallback.

        Raises
        ------
        LookupError
            With `device`, if CUDA is explicitly selected but missing.
        NotImplementedError
            With `device`, if there is no CUDA kernel and
            ``missing_cuda="raise"``.

        Examples
        --------
        >>> xp.kernels.Kernel(lambda x: None, name="noop").selected()
        'host'
        """
        if device:
            return "cuda" if self._device_kernel() is self._cuda_kernel else "host"
        function = self._host_kernel.kernel
        return (
            function.selected() if isinstance(function, HostImplementations) else "host"
        )

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """Return the kernel for the active backend.

        Call this once at setup to fail early if a CUDA kernel is missing.

        Returns
        -------
        PyccelKernel or CudaKernel
            The host kernel on the NumPy backend; on the CuPy backend the CUDA
            kernel, or the host kernel with ``missing_cuda="fallback"``.

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
        """Call the host or CUDA kernel, as chosen by `dispatch`.

        Parameters
        ----------
        *args
            Kernel arguments, passed as they are.
        n_threads, grid, block, shared_mem, stream
            Launch configuration of the CUDA kernel, see
            :meth:`CudaKernel.__call__ <cunumpy.kernels.CudaKernel.__call__>`;
            by default inferred from the arguments. Ignored by the host kernel.

        Returns
        -------
        object
            What the selected kernel returns.

        Raises
        ------
        ValueError
            If the CUDA kernel runs without `n_threads`, `grid` or
            ``n_threads_from``.
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
    """A read-only mapping from names to :class:`~cunumpy.kernels.Kernel` objects.

    Usually built with :meth:`from_package`; :meth:`register` adds kernels one
    by one. ``str(catalog)`` is :meth:`summary`. See :doc:`/kernels/dispatch`.

    Parameters
    ----------
    kernels : mapping of str to Kernel, optional
        Initial kernels.

    Examples
    --------
    >>> kernel = xp.kernels.Kernel(lambda x: None, name="push")
    >>> catalog = xp.kernels.KernelCatalog({"push": kernel})
    >>> print(catalog)
    CUDA kernels: 0 of 1 (missing: push)
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
        outputs: (
            Sequence[int | str]
            | str
            | Callable[[str], Sequence[int | str] | str | None]
            | None
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

        Every subfolder ``<name>`` containing ``<name><host_suffix>.py`` gives
        a kernel (see :meth:`Kernel.from_folder
        <cunumpy.kernels.Kernel.from_folder>`): the function ``<name>`` of that
        module is the host kernel, the ``__global__`` function ``<name>`` of
        ``<name><cuda_suffix>``, if present, the CUDA kernel. Other
        ``__global__`` functions are ignored; load them with
        :meth:`CudaKernel.all_from_file <cunumpy.kernels.CudaKernel.all_from_file>`.

        Parameters
        ----------
        package : str
            Full name of the package, e.g. ``__name__`` in its ``__init__.py``.
        host_suffix : str, optional
            Module name suffix of the host kernels.
        cuda_suffix : str, optional
            File name suffix of the CUDA kernels.
        test_args_suffix : str or None, optional
            Module name suffix of the test arguments: a module
            ``<name><test_args_suffix>.py`` in a kernel folder is recorded as
            :attr:`Kernel.test_args_module
            <cunumpy.kernels.Kernel.test_args_module>`. None disables this.
        check_name_length : bool, optional
            Warn about a kernel whose Pyccel Fortran wrapper module
            ``bind_c_<name><host_suffix>`` exceeds 63 characters (kernel names
            of at most 48 characters with the default suffix).
        missing_cuda : {"raise", "fallback"}, optional
            Passed on to every :class:`~cunumpy.kernels.Kernel`.
        host_options : mapping or callable, optional
            :class:`~cunumpy.kernels.PyccelKernel` options of the host kernels,
            the same for all or a function of the kernel name.
        outputs : sequence, "annotations" or callable, optional
            The arguments every host kernel writes to, or a function of the
            kernel name returning them (see :meth:`Kernel.from_folder
            <cunumpy.kernels.Kernel.from_folder>`).
        include_dirs : sequence of str or Path, optional
            Include directories of the CUDA kernels besides each kernel's
            folder; by default the directory containing the top-level package,
            so ``#include "my_pkg/common.cuh"`` works.
        dispatch : {"backend", "arrays"}, optional
            Passed on to every :class:`~cunumpy.kernels.Kernel`.
        compile_host : callable, optional
            Compiles a host kernel module (e.g. a wrapper of ``pyccel.epyccel``);
            cunumpy compiles nothing itself. Each host kernel then is compiled
            on first use and falls back to `host_fallback`, or to the Python
            function with a warning, if compilation fails.
        host_fallback : mapping or callable, optional
            The fallback for a failed compilation by kernel name (e.g. a
            vectorized NumPy version): a mapping or a function of the name.
        **cuda_options
            Passed on to :meth:`CudaKernel.from_file
            <cunumpy.kernels.CudaKernel.from_file>`, e.g. ``block_size``.

        Returns
        -------
        KernelCatalog
            The kernels, by folder name.

        Examples
        --------
        >>> # my_kernels/__init__.py
        >>> catalog = xp.kernels.KernelCatalog.from_package(__name__)  # doctest: +SKIP
        >>> catalog["push"](x, out, n_threads=len(x))  # doctest: +SKIP
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
                outputs=outputs(name) if callable(outputs) else outputs,
                include_dirs=include_dirs,
                dispatch=dispatch,
                compile_host=compile_host,
                extra_implementations=_fallback_loader(host_fallback, name),
                **cuda_options,
            )
        return cls(kernels)

    def register(self, kernel: Kernel, name: str | None = None) -> Kernel:
        """Add a kernel to the catalog.

        Parameters
        ----------
        kernel : Kernel
            The kernel.
        name : str, optional
            Its name in the catalog; by default the kernel's own name.

        Returns
        -------
        Kernel
            `kernel` itself.

        Raises
        ------
        KeyError
            If a kernel of that name is already registered.
        """
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
        """Return one line on the porting status, e.g. for a ``--status`` command.

        Parameters
        ----------
        max_missing : int, optional
            How many kernels without CUDA kernel to name (default 10); the rest is
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
        """The names of the kernels with a CUDA kernel."""
        return [name for name, kernel in self._kernels.items() if kernel.has_cuda]

    @property
    def without_cuda(self) -> list[str]:
        """The names of the kernels without a CUDA kernel, i.e. still to port."""
        return [name for name, kernel in self._kernels.items() if not kernel.has_cuda]

    def parity_cases(self) -> list[tuple[str, Kernel]]:
        """Return the ``(name, kernel)`` pairs of the kernels with a CUDA kernel.

        For a parametrised parity test of the whole catalog with
        :func:`~cunumpy.kernel_testing.assert_kernels_agree`::

            @pytest.mark.parametrize("name, kernel", catalog.parity_cases())
            def test_parity(name, kernel):
                assert_kernels_agree(kernel, make_args[name], n_threads=1000)

        Returns
        -------
        list of tuple of (str, Kernel)
            The pairs, in catalog order.
        """
        return [
            (name, kernel) for name, kernel in self._kernels.items() if kernel.has_cuda
        ]

    def check_signatures(self) -> None:
        """Check the parameters of every kernel, e.g. in a unit test.

        Runs :meth:`Kernel.check_signature
        <cunumpy.kernels.Kernel.check_signature>` on every kernel.

        Raises
        ------
        ValueError
            Listing every kernel whose implementations take different parameters.
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

        CuPy caches compilations on disk, so later runs mostly load them. All
        kernels are compiled even if one fails; the first error is raised
        afterwards.

        Parameters
        ----------
        jobs : int or None, optional
            Number of kernels compiled at a time, in threads (NVRTC releases
            the GIL; all threads use the current device); None uses the number
            of CPUs.

        Returns
        -------
        list of str
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
