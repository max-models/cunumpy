"""CUDA kernels written in Python: ``cupyx.jit`` and numba-cuda, launched like a CudaKernel.

A :class:`~cunumpy.kernels.CupyJitKernel` wraps a ``cupyx.jit.rawkernel``, a
:class:`~cunumpy.kernels.NumbaCudaKernel` a ``numba.cuda.jit`` kernel. Both take
the launch keywords of :class:`~cunumpy.kernels.CudaKernel` (``n_threads``,
``grid``, ``block``, ``shared_mem``, ``stream``, the launch size inferred from
the arguments) and follow its rules where a JIT allows: device arrays only (a
NumPy array raises instead of being copied), argument objects
(``__cuda_args__``) flattened. They are compiled by their JIT for the argument
types of the first call, so there is no signature to check against; the number
of arguments is checked against the Python function.

The function is loaded on first use (from a loader, e.g. an import of
``<name>_cupy_jit.py``), so a kernel can be created without CuPy or numba; a
failed load is kept in :attr:`~cunumpy.kernels.CupyJitKernel.error`. See
:doc:`/kernels/python-kernels`.
"""

from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np

from cunumpy._cuda_kernel import (
    CudaStructArguments,
    CudaStructValue,
    _is_capturing,
    _is_device_array,
    _LaunchConfig,
)
from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _record_sync

__all__ = ["CupyJitKernel", "NumbaCudaKernel", "PythonDeviceKernel"]

_NOT_LOADED = object()


class PythonDeviceKernel(_LaunchConfig):
    """Base of the CUDA kernels written in Python and compiled by a JIT.

    Parameters
    ----------
    function : callable, optional
        The kernel: a Python function (compiled by the JIT on first use) or a
        kernel the JIT already made (e.g. decorated with ``@jit.rawkernel()``
        or ``@cuda.jit``). Give this or `loader`.
    name : str, optional
        Name of the kernel; by default the name of the function. Required
        with `loader`.
    loader : callable, optional
        Returns `function` when the kernel is first used, e.g. by importing
        its module; a failure is kept in :attr:`error`.
    block_size : int or sequence of int, optional
        Threads per block, as for :class:`~cunumpy.kernels.CudaKernel`.
    n_threads_from : {"auto", "first_array", "last_axis"}, callable or None, optional
        Launch size when a call gives neither `n_threads` nor `grid`, as for
        :class:`~cunumpy.kernels.CudaKernel`.
    debug : bool, optional
        Synchronize after every launch and raise a ``RuntimeError`` naming the
        kernel. None (default) follows :func:`~cunumpy.cuda.set_cuda_debug`.

    Raises
    ------
    TypeError
        If not exactly one of `function` and `loader` is given, or `loader`
        is given without `name`.
    """

    #: The name of the implementation, as in :data:`~cunumpy.kernels.DEVICE_IMPLEMENTATIONS`.
    implementation = ""
    #: What a user installs to run the kernel, named in load errors.
    requirement = ""

    @staticmethod
    def _python_function(function: Any) -> Any:
        """Return the Python function of a JIT kernel (or `function` itself)."""
        return function

    def __init__(
        self,
        function: Callable[..., Any] | None = None,
        name: str | None = None,
        *,
        loader: Callable[[], Callable[..., Any]] | None = None,
        block_size: int | Sequence[int] = 128,
        n_threads_from: Callable[[tuple[Any, ...]], Any] | str | None = "auto",
        debug: bool | None = None,
    ) -> None:
        if (function is None) == (loader is None):
            raise TypeError("pass exactly one of function and loader")
        if function is not None and not callable(function):
            raise TypeError(f"function must be callable, got {type(function).__name__}")
        if name is None:
            if function is None:
                raise TypeError("name is required with a loader")
            python = self._python_function(function)
            name = getattr(python, "__name__", None) or type(function).__name__
        self._name = name
        self._init_launch(block_size, n_threads_from, debug)
        self._loader = loader
        self._loaded: Any = _NOT_LOADED if function is None else function
        self._kernel: Any = None
        self._error: Exception | None = None
        self._parameters: list[str] | None | object = _NOT_LOADED

    # -- the JIT, per subclass ------------------------------------------------

    @staticmethod
    def _is_jitted(function: Any) -> bool:
        raise NotImplementedError

    def _jit(self, function: Callable[..., Any]) -> Any:
        raise NotImplementedError

    def _launch(
        self,
        kernel: Any,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        values: tuple[Any, ...],
        shared_mem: int,
        stream: Any,
    ) -> None:
        raise NotImplementedError

    # -- loading --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the kernel."""
        return self._name

    @property
    def expression(self) -> str:
        """Name of the kernel, as in error messages of :class:`~cunumpy.kernels.CudaKernel`."""
        return self._name

    @property
    def error(self) -> Exception | None:
        """Why the kernel could not be loaded, or None."""
        if self._loaded is _NOT_LOADED and self._error is None:
            self.available()
        return self._error

    def _load(self) -> Any:
        """Return the function, loading it on first use (a failure is kept)."""
        if self._loaded is _NOT_LOADED:
            if self._error is not None:
                raise self._error
            try:
                function = self._loader()  # type: ignore[misc]
                if not callable(function):
                    raise TypeError(
                        f"the loader of {self.implementation} kernel {self._name!r} "
                        f"returned {type(function).__name__}, not a function",
                    )
            except Exception as error:
                self._error = error
                raise
            self._loaded = function
        return self._loaded

    def available(self) -> bool:
        """Load the function, if not done yet, and return whether that worked.

        Returns
        -------
        bool
            False if the loader failed (see :attr:`error`).
        """
        try:
            self._load()
        except Exception:  # noqa: BLE001 -- kept in error
            return False
        return True

    @property
    def function(self) -> Any:
        """The kernel as given or loaded: a Python function or a JIT kernel."""
        return self._load()

    @property
    def python_function(self) -> Callable[..., Any]:
        """The Python function of the kernel (unwrapped from the JIT kernel)."""
        return self._python_function(self._load())

    def parameters(self) -> list[str] | None:
        """Return the parameter names of the Python function.

        Returns
        -------
        list of str or None
            The names, or None if unknown (no Python signature, or variable
            arguments).

        Raises
        ------
        Exception
            What the loader raised, if the kernel cannot be loaded.
        """
        if self._parameters is _NOT_LOADED:
            names: list[str] | None
            try:
                signature = inspect.signature(self.python_function)
            except (TypeError, ValueError):
                names = None
            else:
                kinds = {p.kind for p in signature.parameters.values()}
                if kinds - {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }:
                    names = None
                else:
                    names = list(signature.parameters)
            self._parameters = names
        return self._parameters  # type: ignore[return-value]

    @property
    def is_compiled(self) -> bool:
        """Whether the JIT kernel exists (its machine code is made per argument types)."""
        return self._kernel is not None

    def compile(self) -> Any:
        """Load the function and wrap it in the JIT, if not done yet.

        The JIT compiles machine code at the first launch with each set of
        argument types, so this checks the import and the JIT, not the code.

        Returns
        -------
        object
            The JIT kernel.

        Raises
        ------
        RuntimeError
            If the kernel cannot be loaded or the JIT is not installed; the
            cause is chained.
        """
        if self._kernel is None:
            try:
                function = self._load()
                self._kernel = (
                    function if self._is_jitted(function) else self._jit(function)
                )
            except Exception as error:
                raise RuntimeError(
                    f"{self.implementation} kernel {self._name!r} cannot be loaded "
                    f"({type(error).__name__}: {error}); it needs {self.requirement}",
                ) from error
        return self._kernel

    # -- launching ------------------------------------------------------------

    def _flatten(self, value: Any, label: str, values: list[Any]) -> None:
        """Append the kernel arguments of `value` to `values`, checking each."""
        if _is_device_array(value):
            values.append(value)
        elif isinstance(value, np.ndarray):
            raise TypeError(
                f"{self.implementation} kernel {self._name!r}: {label} is a NumPy "
                "array; device kernels never copy, pass a CuPy array "
                "(xp.asarray, xp.to_cunumpy)",
            )
        elif isinstance(value, (CudaStructValue, CudaStructArguments, np.void)):
            raise TypeError(
                f"{self.implementation} kernel {self._name!r}: {label} is a C "
                "struct, which only CUDA C kernels (CudaKernel) take by value; pass "
                "its fields as separate arguments or in a CudaArguments",
            )
        elif callable(getattr(type(value), "__cuda_args__", None)):
            for j, item in enumerate(value.__cuda_args__()):
                self._flatten(item, f"{label}[{j}]", values)
        else:
            values.append(value)

    def prepare_args(self, *args: Any) -> tuple[Any, ...]:
        """Return the arguments as the JIT kernel gets them.

        Argument objects with ``__cuda_args__()`` are flattened (recursively);
        device arrays and scalars are passed as they are.

        Parameters
        ----------
        *args
            Kernel arguments.

        Returns
        -------
        tuple
            The flattened arguments.

        Raises
        ------
        TypeError
            For a NumPy array (never copied to the device), a C struct (only
            :class:`~cunumpy.kernels.CudaKernel` takes structs by value), or a
            number of arguments the Python function does not take.
        """
        values: list[Any] = []
        for index, arg in enumerate(args):
            self._flatten(arg, f"argument {index}", values)
        parameters = self.parameters()
        if parameters is not None and len(values) != len(parameters):
            raise TypeError(
                f"{self.implementation} kernel {self._name!r} takes "
                f"{len(parameters)} arguments ({', '.join(parameters)}), got "
                f"{len(values)}"
                + (" after flattening" if len(values) != len(args) else ""),
            )
        return tuple(values)

    def _synchronize(
        self, stream: Any, grid: tuple[int, ...], block: tuple[int, ...]
    ) -> None:
        """Wait for the launch and re-raise an error naming this kernel (debug mode)."""
        if stream is None:
            import cupy as cp

            stream = cp.cuda.get_current_stream()
        if _is_capturing(stream):
            return
        if _COUNTERS:
            _record_sync(f"debug synchronization after kernel {self._name!r}")
        try:
            stream.synchronize()
        except Exception as error:
            raise RuntimeError(
                f"CUDA error after launching {self.implementation} kernel "
                f"{self._name!r} with grid {grid} and block {block}: {error}",
            ) from error

    def __call__(
        self,
        *args: Any,
        n_threads: int | Sequence[int] | None = None,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
        shared_mem: int = 0,
        stream: Any = None,
    ) -> None:
        """Launch the kernel (asynchronously, unless in debug mode).

        Parameters
        ----------
        *args
            Kernel arguments: CuPy arrays, scalars and argument objects, see
            :meth:`prepare_args`.
        n_threads : int or sequence of int, optional
            Number of threads in 1 to 3 dimensions.
        grid : int or sequence of int, optional
            Number of blocks, instead of `n_threads`.
        block : int or sequence of int, optional
            Block shape for this call, instead of :attr:`block_size`.
        shared_mem : int, optional
            Dynamic shared memory per block in bytes.
        stream : cupy.cuda.Stream, optional
            Stream to launch on; the current CuPy stream by default.

        Raises
        ------
        RuntimeError
            If the kernel cannot be loaded, or in debug mode a CUDA error
            found by synchronizing after the launch.
        TypeError
            For arguments :meth:`prepare_args` rejects.
        """
        grid_shape, block_shape = self.launch_shape(
            n_threads, grid=grid, block=block, args=args
        )
        shared_mem = operator.index(shared_mem)
        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        kernel = self.compile()
        values = self.prepare_args(*args)
        if 0 in grid_shape:
            return
        self._launch(kernel, grid_shape, block_shape, values, shared_mem, stream)
        if self.debug_active():
            self._synchronize(stream, grid_shape, block_shape)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self._name!r}, block_size={self.block_size})"
        )


class CupyJitKernel(PythonDeviceKernel):
    """A CUDA kernel written in Python with ``cupyx.jit``, launched like a CudaKernel.

    The kernel uses CuPy's JIT (``cupyx.jit.threadIdx``, ``jit.grid``,
    ``jit.atomic_add``, ...) and runs on CuPy's streams and memory. It is
    compiled for the argument types of each new call signature. In a kernel
    folder it is ``<name>_cupy_jit.py`` (see :doc:`/kernels/python-kernels`).

    Parameters
    ----------
    function : callable, optional
        The kernel: a Python function, or one decorated with
        ``@cupyx.jit.rawkernel()``. Give this or `loader`.
    name : str, optional
        Name of the kernel; by default the name of the function.
    loader : callable, optional
        Returns `function` on first use.
    block_size : int or sequence of int, optional
        Threads per block (default 128).
    n_threads_from : {"auto", "first_array", "last_axis"}, callable or None, optional
        Launch size when a call gives neither `n_threads` nor `grid`.
    debug : bool, optional
        Synchronize after every launch; None follows the global setting.

    See Also
    --------
    NumbaCudaKernel : The same with numba-cuda.
    CudaKernel : A kernel in CUDA C.

    Examples
    --------
    >>> from cupyx import jit  # doctest: +SKIP
    >>> def scale(x, a, out):
    ...     i = jit.grid(1)
    ...     if i < x.size:
    ...         out[i] = a * x[i]
    >>> kernel = xp.kernels.CupyJitKernel(scale)
    >>> kernel.launch_shape(1000)
    ((8,), (128,))
    >>> kernel(x, 2.0, out)  # doctest: +SKIP
    """

    implementation = "cupy_jit"
    requirement = "CuPy (cupyx.jit) and a GPU"

    def __init__(
        self,
        function: Callable[..., Any] | None = None,
        name: str | None = None,
        *,
        loader: Callable[[], Callable[..., Any]] | None = None,
        block_size: int | Sequence[int] = 128,
        n_threads_from: Callable[[tuple[Any, ...]], Any] | str | None = "auto",
        debug: bool | None = None,
    ) -> None:
        super().__init__(
            function,
            name,
            loader=loader,
            block_size=block_size,
            n_threads_from=n_threads_from,
            debug=debug,
        )

    @staticmethod
    def _is_jitted(function: Any) -> bool:
        return type(function).__module__.startswith("cupyx.jit")

    @staticmethod
    def _python_function(function: Any) -> Any:
        if type(function).__module__.startswith("cupyx.jit"):
            return getattr(function, "_func", function)
        return function

    def _jit(self, function: Callable[..., Any]) -> Any:
        from cupyx import jit

        return jit.rawkernel()(function)

    def _launch(
        self,
        kernel: Any,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        values: tuple[Any, ...],
        shared_mem: int,
        stream: Any,
    ) -> None:
        with stream if stream is not None else nullcontext():
            kernel(grid, block, values, shared_mem=shared_mem)


class NumbaCudaKernel(PythonDeviceKernel):
    """A CUDA kernel written in Python with numba-cuda, launched like a CudaKernel.

    The kernel uses ``numba.cuda`` (``cuda.grid``, ``cuda.atomic.add``, ...) and
    takes CuPy arrays (``__cuda_array_interface__``, no copies); it is launched
    on CuPy's current stream (or `stream`), so it is ordered with CuPy's work.
    Install ``numba-cuda`` (e.g. ``pip install 'numba-cuda[cu12]'``). In a
    kernel folder it is ``<name>_numba_cuda.py`` (see
    :doc:`/kernels/python-kernels`).

    With ``NUMBA_ENABLE_CUDASIM=1`` numba's simulator runs the kernel on the
    CPU, also on the arrays of the fake CuPy (:mod:`cunumpy._fake_cupy`): a
    port can be tested without a GPU.

    Parameters
    ----------
    function : callable, optional
        The kernel: a Python function, or one decorated with ``@cuda.jit``.
        Give this or `loader`.
    name : str, optional
        Name of the kernel; by default the name of the function.
    loader : callable, optional
        Returns `function` on first use.
    block_size : int or sequence of int, optional
        Threads per block (default 128).
    n_threads_from : {"auto", "first_array", "last_axis"}, callable or None, optional
        Launch size when a call gives neither `n_threads` nor `grid`.
    debug : bool, optional
        Synchronize after every launch; None follows the global setting.

    See Also
    --------
    CupyJitKernel : The same with ``cupyx.jit``.
    CudaKernel : A kernel in CUDA C.

    Examples
    --------
    >>> from numba import cuda  # doctest: +SKIP
    >>> def scale(x, a, out):
    ...     i = cuda.grid(1)
    ...     if i < x.size:
    ...         out[i] = a * x[i]
    >>> kernel = xp.kernels.NumbaCudaKernel(scale)
    >>> kernel.launch_shape(1000)
    ((8,), (128,))
    >>> kernel(x, 2.0, out)  # doctest: +SKIP
    """

    implementation = "numba_cuda"
    requirement = "numba-cuda (pip install 'numba-cuda[cu12]') and a GPU"

    def __init__(
        self,
        function: Callable[..., Any] | None = None,
        name: str | None = None,
        *,
        loader: Callable[[], Callable[..., Any]] | None = None,
        block_size: int | Sequence[int] = 128,
        n_threads_from: Callable[[tuple[Any, ...]], Any] | str | None = "auto",
        debug: bool | None = None,
    ) -> None:
        super().__init__(
            function,
            name,
            loader=loader,
            block_size=block_size,
            n_threads_from=n_threads_from,
            debug=debug,
        )

    @staticmethod
    def _is_jitted(function: Any) -> bool:
        return type(function).__module__.startswith("numba.cuda")

    @staticmethod
    def _python_function(function: Any) -> Any:
        if type(function).__module__.startswith("numba.cuda"):
            return getattr(function, "py_func", function)
        return function

    def _jit(self, function: Callable[..., Any]) -> Any:
        from numba import cuda

        return cuda.jit(function)

    def _launch(
        self,
        kernel: Any,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        values: tuple[Any, ...],
        shared_mem: int,
        stream: Any,
    ) -> None:
        from numba import cuda

        if stream is None:
            import cupy as cp

            stream = cp.cuda.get_current_stream()
        pointer = getattr(stream, "ptr", None)
        if pointer is None:  # a numba stream
            numba_stream = stream
        else:
            # CuPy's null stream is the legacy default stream, numba's stream 0
            numba_stream = cuda.external_stream(pointer) if pointer else 0
        kernel[grid, block, numba_stream, shared_mem](*values)
