"""Pairs of host and CUDA kernels, chosen by the active backend.

A :class:`Kernel` holds a host kernel (a :class:`~cunumpy.PyccelKernel`, e.g. a
Pyccel-compiled function) and, optionally, its 1:1 corresponding CUDA kernel
(:class:`~cunumpy.CudaKernel`). It calls the host kernel on the NumPy backend and
the CUDA kernel on the CuPy backend, so a code base can port its kernels to CUDA
one by one.

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

import importlib
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .cuda_kernel import CudaKernel
from .kernel import PyccelKernel, resolve_host_args
from .transfers import _ACTIVE as _COUNTERS
from .transfers import _record
from .xp import get_backend

__all__ = ["Kernel", "KernelCatalog"]

_MISSING_CUDA = ("raise", "fallback")


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
        wrapped in a :class:`~cunumpy.PyccelKernel`.
    cuda_kernel : CudaKernel | None
        The CUDA kernel, called on the CuPy backend; None if it has not been
        written yet.
    name : str | None
        Name of the kernel; defaults to the name of the host kernel.
    missing_cuda : {"raise", "fallback"}
        What happens on the CuPy backend if there is no CUDA kernel: ``"raise"``
        raises ``NotImplementedError``; ``"fallback"`` calls the host kernel
        through :class:`~cunumpy.PyccelKernel`, which copies the arrays to the
        host and back at every call (a warning is emitted once).
    cuda_path : str | Path | None
        Where the CUDA kernel is expected, for the error message if it is missing.
    host_options : Mapping[str, Any] | None
        Keyword arguments for the :class:`~cunumpy.PyccelKernel` that wraps a
        plain callable `host_kernel`, e.g. ``{"object_modules": ("my_pkg.",),
        "outputs": (2,)}``; they matter for the fallback on the CuPy backend.
        Not allowed if `host_kernel` already is a ``PyccelKernel``.

    Notes
    -----
    Both kernels take the same arguments, except that the CUDA kernel gets the
    launch shape (``n_threads`` or ``grid``) and argument objects in their CUDA
    form (see :class:`~cunumpy.CudaArguments` and :class:`~cunumpy.CudaStruct`).
    An argument object implementing :class:`~cunumpy.KernelArguments` is
    replaced by its ``__host_args__()`` on the host path and flattened via
    ``__cuda_args__()`` on the CUDA path, so the call site is the same on both
    backends.
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
    ) -> None:
        if missing_cuda not in _MISSING_CUDA:
            raise ValueError(
                f"missing_cuda must be one of {_MISSING_CUDA}, got {missing_cuda!r}"
            )
        if cuda_kernel is not None and not isinstance(cuda_kernel, CudaKernel):
            raise TypeError(
                "cuda_kernel must be a CudaKernel or None, "
                f"got {type(cuda_kernel).__name__}"
            )
        if not isinstance(host_kernel, PyccelKernel):
            host_kernel = PyccelKernel(host_kernel, **(host_options or {}))
        elif host_options:
            raise ValueError(
                "host_options are for wrapping a plain callable; configure the "
                "given PyccelKernel directly"
            )
        self._host_kernel = host_kernel
        self._cuda_kernel = cuda_kernel
        self._name = name if name is not None else host_kernel.name
        self._missing_cuda = missing_cuda
        self._cuda_path = None if cuda_path is None else Path(cuda_path)
        self._warned = False

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

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """The kernel for the active backend.

        Call this once at setup to fail early if a CUDA kernel is missing.

        Raises
        ------
        NotImplementedError
            On the CuPy backend, if there is no CUDA kernel and
            ``missing_cuda="raise"``.
        """
        if get_backend() != "cupy":
            return self._host_kernel
        if self._cuda_kernel is not None:
            return self._cuda_kernel
        if self._missing_cuda == "raise":
            expected = (
                "" if self._cuda_path is None else f" (expected {self._cuda_path})"
            )
            raise NotImplementedError(
                f"No CUDA version of kernel {self._name!r}{expected}."
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
            Kernel arguments. Objects implementing
            :class:`~cunumpy.KernelArguments` are resolved per backend (see
            :func:`~cunumpy.resolve_host_args`).
        n_threads, grid, block, shared_mem, stream
            Launch configuration of the CUDA kernel, see
            :meth:`CudaKernel.__call__ <cunumpy.CudaKernel.__call__>`;
            `n_threads` (or `grid`) is required when the CUDA kernel is called.
            Ignored by the host kernel.
        """
        kernel = self.get_kernel()
        if kernel is self._host_kernel:
            if _COUNTERS and self._cuda_kernel is None and get_backend() == "cupy":
                _record(
                    "fallback",
                    f"Kernel {self._name!r} has no CUDA kernel: host kernel "
                    "called on the CuPy backend",
                )
            args, _ = resolve_host_args(args)
            return kernel(*args)
        if n_threads is None and grid is None:
            raise ValueError(
                f"{self._name}: n_threads is required to launch the CUDA kernel "
                "(or pass grid)"
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
        missing_cuda: str = "raise",
        host_options: (
            Mapping[str, Any] | Callable[[str], Mapping[str, Any]] | None
        ) = None,
        include_dirs: Sequence[str | Path] | None = None,
        **cuda_options: Any,
    ) -> KernelCatalog:
        """Collect the kernels of a package with one folder per kernel.

        For every subfolder ``<name>`` of the package that contains the module
        ``<name><host_suffix>.py``, the function ``<name>`` of that module is the
        host kernel, and ``<name><cuda_suffix>`` in the same folder, if present,
        is the CUDA kernel (with a ``__global__`` function ``<name>``).

        Parameters
        ----------
        package : str
            Full name of the package, e.g. ``__name__`` in its ``__init__.py``.
        host_suffix : str
            Module name suffix of the host kernels.
        cuda_suffix : str
            File name suffix of the CUDA kernels.
        missing_cuda : {"raise", "fallback"}
            Passed on to every :class:`Kernel`.
        host_options : Mapping | Callable[[str], Mapping] | None
            Keyword arguments for the :class:`~cunumpy.PyccelKernel` wrapping each
            host kernel (see :class:`Kernel`): the same for all kernels, or a
            function of the kernel name, e.g. to declare per-kernel ``outputs``.
        include_dirs : Sequence[str | Path] | None
            Include directories of the CUDA kernels, in addition to each
            kernel's own folder. By default the source root of the top-level
            package (the directory containing it), so that a kernel in
            ``my_pkg.kernels`` can ``#include "my_pkg/common.cuh"``.
        **cuda_options
            Passed on to :meth:`CudaKernel.from_file`, e.g. ``block_size`` or
            ``structs``.
        """
        root = Path(importlib.import_module(package).__file__).parent
        if include_dirs is None:
            include_dirs = (_source_root(package),)
        cuda_options["include_dirs"] = tuple(include_dirs)
        kernels = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            name = folder.name
            if not (folder / f"{name}{host_suffix}.py").is_file():
                continue
            module = importlib.import_module(f"{package}.{name}.{name}{host_suffix}")
            cuda_path = folder / f"{name}{cuda_suffix}"
            cuda_kernel = (
                CudaKernel.from_file(cuda_path, name, **cuda_options)
                if cuda_path.is_file()
                else None
            )
            kernels[name] = Kernel(
                getattr(module, name),
                cuda_kernel,
                name=name,
                missing_cuda=missing_cuda,
                cuda_path=cuda_path,
                host_options=(
                    host_options(name) if callable(host_options) else host_options
                ),
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

    @property
    def without_cuda(self) -> list[str]:
        """Names of the kernels without a CUDA kernel, i.e. still to port."""
        return [name for name, kernel in self._kernels.items() if not kernel.has_cuda]

    def parity_cases(self) -> list[tuple[str, Kernel]]:
        """The ``(name, kernel)`` pairs of the kernels that have a CUDA kernel.

        For a parametrised parity test of the whole catalog with
        :func:`cunumpy.testing.assert_kernels_agree`::

            @pytest.mark.parametrize("name, kernel", catalog.parity_cases())
            def test_parity(name, kernel):
                assert_kernels_agree(kernel, make_args[name], n_threads=1000)
        """
        return [
            (name, kernel) for name, kernel in self._kernels.items() if kernel.has_cuda
        ]

    def compile_all(self) -> list[str]:
        """Compile all CUDA kernels now, e.g. at setup instead of in the first step.

        Compilation is cached on disk by CuPy, so after the first run this mostly
        loads the compiled kernels.

        Returns
        -------
        list[str]
            Names of the compiled kernels.

        Raises
        ------
        RuntimeError
            If CuPy or a GPU is not available (and there are CUDA kernels).
        """
        return [name for name, kernel in self._kernels.items() if kernel.compile()]
