"""The kernel settings shared by every kernel folder of one package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from cunumpy._dispatch import Kernel
from cunumpy._metal_kernel import MetalKernel
from cunumpy._pyccel_cache import PyccelCache


def source_roots(*packages: str) -> tuple[Path, ...]:
    """Return the directories containing `packages`, for ``#include "<package>/..."``.

    A CUDA kernel that includes a header of another package passes these as
    ``include_dirs``; the packages are found without importing them.

    Parameters
    ----------
    *packages : str
        Top-level package names.

    Returns
    -------
    tuple of Path
        The parent directory of each package, in order.

    Raises
    ------
    ModuleNotFoundError
        If a package is not installed.

    Examples
    --------
    >>> (xp.kernels.source_roots("cunumpy")[0] / "cunumpy").is_dir()
    True
    """
    roots = []
    for package in packages:
        spec = find_spec(package)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(package)
        roots.append(Path(spec.origin).resolve().parents[1])
    return tuple(roots)


class KernelPackage:
    """The kernel settings of a package with one folder per kernel.

    A package whose kernels live in ``<package>/kernels/<name>/``
    (``<name>_pyccel.py``, ``<name>_numpy.py``, ``<name>_cuda.cu`` and
    ``<name>_metal.py``) declares one instance, e.g. in
    ``<package>/kernels/__init__.py``, and builds every kernel with it, so the
    folders share one build cache and one set of launch options. Nothing is
    compiled when a kernel is built: the Pyccel version on its first host call
    (by :attr:`cache`), the CUDA version on its first launch.

    Parameters
    ----------
    name : str
        The package name, for the build cache (see
        :class:`~cunumpy.kernels.PyccelCache`).
    env_prefix : str, optional
        Prefix of the cache's environment variables, passed on to
        :class:`~cunumpy.kernels.PyccelCache`.
    host_suffix : str, optional
        Module name suffix of the host kernels. Default ``"_pyccel"``.
    dispatch : {"arrays", "backend"}, optional
        How a kernel picks its version. Default ``"arrays"``: CUDA for CuPy
        arguments, the host version for NumPy ones, also while CuPy is active.
    block_size : int, optional
        CUDA threads per block. Default 256.
    n_threads_from : str or callable, optional
        The CUDA launch size of a call without ``n_threads``. Default
        ``"last_axis"``: the last axis of the first array, the marker count of
        a component-major ``(ncomp, N)`` or ``(N,)`` marker array.
    threadgroup : int, optional
        Threads per threadgroup of the Metal kernels. Default 256.

    Examples
    --------
    In ``my_sim/kernels/__init__.py``::

        KERNEL_PACKAGE = xp.kernels.KernelPackage("my-sim")

    in ``my_sim/kernels/push/__init__.py``::

        from my_sim.kernels import KERNEL_PACKAGE

        kernel = KERNEL_PACKAGE.from_folder(__name__)

    and in ``my_sim/kernels/push/push_metal.py``::

        _KERNEL = KERNEL_PACKAGE.metal_kernel("push", SOURCE, ["x", "v"], ["x_new"])

    >>> package = xp.kernels.KernelPackage("my-sim")
    >>> package.cache.cache_env
    'MY_SIM_KERNEL_CACHE'
    >>> package.options["n_threads_from"]
    'last_axis'
    """

    def __init__(
        self,
        name: str,
        *,
        env_prefix: str | None = None,
        host_suffix: str = "_pyccel",
        dispatch: str = "arrays",
        block_size: int = 256,
        n_threads_from: str | Callable[[tuple[Any, ...]], int] = "last_axis",
        threadgroup: int = 256,
    ) -> None:
        self.name = name
        #: The on-disk build cache of the package's Pyccel kernels.
        self.cache = PyccelCache(name, env_prefix=env_prefix)
        self.host_suffix = host_suffix
        self.dispatch = dispatch
        self.block_size = block_size
        self.n_threads_from = n_threads_from
        self.threadgroup = threadgroup

    def __repr__(self) -> str:
        return f"KernelPackage({self.name!r})"

    @property
    def options(self) -> dict[str, Any]:
        """The options :meth:`from_folder` passes to ``Kernel.from_folder``.

        The same options fit :meth:`KernelCatalog.from_package
        <cunumpy.kernels.KernelCatalog.from_package>`.
        """
        return {
            "host_suffix": self.host_suffix,
            "compile_host": self.cache,
            "dispatch": self.dispatch,
            "block_size": self.block_size,
            "n_threads_from": self.n_threads_from,
        }

    def from_folder(self, package: str, **options: Any) -> Kernel:
        """Build the kernel of one kernel folder with the package's options.

        Parameters
        ----------
        package : str
            Dotted name of the kernel folder (``__name__`` in its ``__init__.py``).
        **options
            Passed on to :meth:`Kernel.from_folder
            <cunumpy.kernels.Kernel.from_folder>`, replacing the package's
            :attr:`options` of the same name (e.g. ``n_threads_from`` for a
            kernel whose first array is not a per-marker array, or
            ``include_dirs=xp.kernels.source_roots(...)``).

        Returns
        -------
        Kernel
        """
        return Kernel.from_folder(package, **{**self.options, **options})

    def metal_kernel(
        self,
        name: str,
        source: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        header: str = "",
        **options: Any,
    ) -> MetalKernel:
        """Build the Metal version of a kernel: float32 on the GPU, outputs zeroed.

        The GPU of an Apple-silicon Mac has no float64, so by default float64
        arguments are cast (``float64="cast"``, a relative error of about 1e-5
        against the CPU versions) and the outputs start at zero
        (``init_value=0``), as the accumulating kernels need.

        Parameters
        ----------
        name, source, inputs, outputs, header
            As for :class:`~cunumpy.kernels.MetalKernel`.
        **options
            Passed on to :class:`~cunumpy.kernels.MetalKernel`, replacing
            ``threadgroup=``:attr:`threadgroup`, ``float64="cast"`` and
            ``init_value=0``.

        Returns
        -------
        MetalKernel
        """
        options = {
            "threadgroup": self.threadgroup,
            "float64": "cast",
            "init_value": 0,
            **options,
        }
        return MetalKernel(
            source, inputs=inputs, outputs=outputs, name=name, header=header, **options
        )
