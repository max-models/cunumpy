"""Host kernels: calling them with CuPy arrays, and choosing their implementation.

:class:`~cunumpy.kernels.PyccelKernel` wraps a host kernel (a
`Pyccel <https://github.com/pyccel/pyccel>`_-compiled function or any callable
taking NumPy arrays) so that it can be called with CuPy arrays: inputs are
copied to the host, written arrays copied back. On the NumPy backend the
kernel is called directly. This module neither imports Pyccel nor compiles
anything. It also holds the host/device implementation settings,
:class:`~cunumpy.kernels.HostImplementations`,
:class:`~cunumpy.kernels.CompiledHostKernel` and the argument helpers
:func:`~cunumpy.kernels.as_kernel_array` and
:func:`~cunumpy.kernels.kernel_output`. See :doc:`/kernels/pyccel-kernel`.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType, TracebackType
from typing import Any

import array_api_compat
import numpy as np

from cunumpy._transfers import _ACTIVE as _COUNTERS
from cunumpy._transfers import _describe, _is_device_copy, _nbytes, _record
from cunumpy.xp import _cupy_backend, _to_cupy, _to_numpy, is_gpu, to_cupy, to_numpy

__all__ = ["CompiledHostKernel", "PyccelKernel"]


# The conversions between device and host arrays, as module attributes so that
# tests can substitute a fake device array type without a GPU. They bypass the
# per-array transfer counting: a call that converts is counted once, as a
# ``kernel_conversion`` event, in :meth:`PyccelKernel.__call__`.
_is_device_array = array_api_compat.is_cupy_array
_device_to_host = _to_numpy
_host_to_device = _to_cupy


class PyccelKernel:
    """Call a host kernel (Pyccel-compiled or any NumPy callable) with CuPy arrays.

    When conversion is needed, CuPy arrays in the arguments (also inside
    lists, tuples, dicts and objects from `object_modules`) are copied to the
    host, the kernel runs, the arrays it may have written are copied back into
    the CuPy arrays, and returned NumPy arrays are moved to the device. A
    device array passed several times is copied once. Otherwise the kernel is
    called directly. See :doc:`/kernels/pyccel-kernel`.

    Parameters
    ----------
    kernel : callable
        The host kernel, taking NumPy arrays.
    use_cupy : bool, optional
        Force conversion on or off. By default (None) it is decided per call:
        on when the backend is CuPy or a CuPy array is passed.
    object_modules : sequence of str, optional
        Module prefixes (e.g. ``("my_project.",)``) whose instances are
        shallow-copied and searched attribute by attribute for arrays. Other
        objects are passed unchanged.
    is_array : callable, optional
        Predicate for host values to move back to the device (returned, or
        reachable from a declared output). Default:
        ``isinstance(value, np.ndarray)``.
    outputs : sequence of int or str, optional
        The arguments the kernel writes: indices (negative from the end) for
        positional arguments, names for keyword arguments. Only these are
        copied back. A name also finds a positional argument, and an index a
        keyword argument, only when the parameter names are known (Python
        signature or `parameters`). ``()`` means no output; None (default)
        copies back every converted array.
    parameters : sequence of str or callable, optional
        Names of the positional parameters of `kernel`, or a function returning
        them (or None), to resolve `outputs`; needed for compiled kernels,
        which have no Python signature. Default: from the signature.

    Raises
    ------
    TypeError
        If `outputs` is a bare int or str, or has entries of another type.

    See Also
    --------
    outputs_from_annotations : Read `outputs` from the annotations.

    Notes
    -----
    A call raises IndexError or KeyError if a declared output does not match
    an argument of that call.

    Examples
    --------
    >>> def scale(a, x, out):
    ...     out[:] = a * x
    >>> kernel = xp.kernels.PyccelKernel(scale, outputs=(2,))
    >>> out = np.zeros(3)
    >>> kernel(2.0, np.arange(3.0), out)  # NumPy or CuPy arrays
    >>> out
    array([0., 2., 4.])
    """

    def __init__(
        self,
        kernel: Callable[..., Any],
        use_cupy: bool | None = None,
        object_modules: Sequence[str] = (),
        is_array: Callable[[Any], bool] | None = None,
        outputs: Sequence[int | str] | None = None,
        parameters: Sequence[str] | Callable[[], Sequence[str] | None] | None = None,
    ) -> None:
        self._kernel = kernel
        self._parameters = parameters
        self._use_cupy = use_cupy
        self._object_modules = tuple(object_modules)
        self._is_array = is_array or (lambda value: isinstance(value, np.ndarray))

        if outputs is None:
            self._outputs: tuple[int | str, ...] | None = None
        else:
            if isinstance(outputs, (int, str)):
                raise TypeError(
                    "outputs must be a sequence of argument indices/names, "
                    f"not a bare {type(outputs).__name__} "
                    f"(did you mean outputs=({outputs!r},)?)",
                )
            for entry in outputs:
                if not isinstance(entry, (int, str)) or isinstance(entry, bool):
                    raise TypeError(
                        "outputs entries must be argument indices (int) or "
                        f"names (str), got {entry!r}",
                    )
            self._outputs = tuple(outputs)

    def __repr__(self) -> str:
        return (
            f"PyccelKernel(kernel={self.name!r}, use_cupy={self.use_cupy!r}, "
            f"outputs={self._outputs!r})"
        )

    def _parameter_names(self) -> Sequence[str] | None:
        """Return the names of the positional parameters, or None if unknown."""
        source = self._parameters
        if source is not None:
            return source() if callable(source) else tuple(source)
        function = getattr(self._kernel, "python", self._kernel)
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return None
        positional = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        names = [p.name for p in signature.parameters.values() if p.kind in positional]
        return names or None

    def _convert_to_numpy(
        self,
        value: Any,
        converted: list[tuple[Any, np.ndarray]],
        memo: dict[int, Any],
    ) -> Any:
        """Replace CuPy arrays in `value` by host copies, once each (`memo` by id)."""
        key = id(value)
        if key in memo:
            return memo[key]

        if _is_device_array(value):
            value_np = _device_to_host(value)
            if _COUNTERS:
                _record(
                    "to_host",
                    f"PyccelKernel {self.name!r} input ({_describe(value)})",
                    nbytes=_nbytes(value_np),
                )
            memo[key] = value_np
            converted.append((value, value_np))
            return value_np

        if isinstance(value, tuple):
            # A tuple cannot be memoized before its items are converted, but it
            # can only take part in a cycle through a mutable container, and
            # those are memoized before they are filled in below.
            value_np = tuple(
                self._convert_to_numpy(item, converted, memo) for item in value
            )
            memo[key] = value_np
            return value_np

        if isinstance(value, list):
            value_np = []
            memo[key] = value_np
            value_np.extend(
                self._convert_to_numpy(item, converted, memo) for item in value
            )
            return value_np

        if isinstance(value, dict):
            value_np = {}
            memo[key] = value_np
            for k, v in value.items():
                value_np[k] = self._convert_to_numpy(v, converted, memo)
            return value_np

        if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
            self._object_modules,
        ):
            # Shallow-copy the object so the caller's instance keeps pointing at
            # its device arrays; only the copy holds the host views.
            value_np = copy.copy(value)
            memo[key] = value_np
            for name, attr in vars(value).items():
                setattr(value_np, name, self._convert_to_numpy(attr, converted, memo))
            return value_np

        return value

    def _convert_from_numpy(self, value: Any) -> Any:
        """Move host arrays returned by the kernel back to the device."""
        if self._is_array(value):
            return self._copy_to_device(value)
        if isinstance(value, tuple):
            return tuple(self._convert_from_numpy(item) for item in value)
        if isinstance(value, list):
            return [self._convert_from_numpy(item) for item in value]
        return value

    def _copy_to_device(self, value: Any) -> Any:
        result = _host_to_device(value)
        if _COUNTERS:
            _record(
                "to_device",
                f"PyccelKernel {self.name!r} output ({_describe(value)})",
                nbytes=_nbytes(value),
            )
        return result

    def _collect_host_arrays(self, value: Any, found: set[int], seen: set[int]) -> None:
        """Record the id of every host array reachable from `value`."""
        if self._is_array(value):
            found.add(id(value))
            return

        if id(value) in seen:
            return

        if isinstance(value, (tuple, list)):
            seen.add(id(value))
            for item in value:
                self._collect_host_arrays(item, found, seen)
            return

        if isinstance(value, dict):
            seen.add(id(value))
            for item in value.values():
                self._collect_host_arrays(item, found, seen)
            return

        if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
            self._object_modules,
        ):
            seen.add(id(value))
            for attr in vars(value).values():
                self._collect_host_arrays(attr, found, seen)

    def _output_host_arrays(
        self,
        args_np: list[Any],
        kwargs_np: dict[str, Any],
    ) -> set[int]:
        """Return the ids of the host arrays reachable from the declared outputs."""
        found: set[int] = set()
        seen: set[int] = set()

        names = self._parameter_names() if self._outputs else None
        for entry in self._outputs or ():
            if isinstance(entry, int):
                index = entry + len(args_np) if entry < 0 else entry
                if 0 <= index < len(args_np):
                    self._collect_host_arrays(args_np[index], found, seen)
                    continue
                if (
                    names is not None
                    and 0 <= entry < len(names)
                    and names[entry] in kwargs_np
                ):
                    self._collect_host_arrays(kwargs_np[names[entry]], found, seen)
                    continue
                raise IndexError(
                    f"{self.name}() was declared with output argument "
                    f"{entry}, but was called with {len(args_np)} "
                    "positional argument(s)"
                    + (
                        ""
                        if names is not None
                        else ". Note that an output passed as a keyword must "
                        "be declared by name, not by index (the parameter "
                        "names of the kernel are unknown)."
                    ),
                )
            if entry in kwargs_np:
                self._collect_host_arrays(kwargs_np[entry], found, seen)
                continue
            if (
                names is not None
                and entry in names
                and names.index(entry)
                < len(
                    args_np,
                )
            ):
                self._collect_host_arrays(args_np[names.index(entry)], found, seen)
                continue
            raise KeyError(
                f"{self.name}() was declared with output argument "
                f"{entry!r}, but "
                + (
                    "no argument of that name was passed"
                    if names is not None
                    else "no such keyword argument was passed. Note that an "
                    "output passed positionally must be declared by index, not "
                    "by name (the parameter names of the kernel are unknown)."
                ),
            )

        return found

    def _contains_cupy(self, value: Any, seen: set[int] | None = None) -> bool:
        """Return whether `value` holds a CuPy array (traversed like the conversion)."""
        if _is_device_array(value):
            return True

        if seen is None:
            seen = set()
        if id(value) in seen:
            return False

        if isinstance(value, (tuple, list)):
            seen.add(id(value))
            return any(self._contains_cupy(item, seen) for item in value)

        if isinstance(value, dict):
            seen.add(id(value))
            return any(self._contains_cupy(item, seen) for item in value.values())

        if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
            self._object_modules,
        ):
            seen.add(id(value))
            return any(self._contains_cupy(attr, seen) for attr in vars(value).values())

        return False

    def _needs_conversion(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self._use_cupy is not None:
            return self._use_cupy
        if _cupy_backend():
            return True
        # The backend is NumPy, but individual CuPy arrays may still have been
        # passed in explicitly.
        return any(self._contains_cupy(value) for value in (*args, *kwargs.values()))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._needs_conversion(args, kwargs):
            return self._kernel(*args, **kwargs)

        # Convert CuPy arrays in args/kwargs to NumPy arrays on the host. The
        # memo is shared across args and kwargs so that an array passed more
        # than once stays a single array on the host too.
        converted: list[tuple[Any, np.ndarray]] = []
        memo: dict[int, Any] = {}
        args_np = [self._convert_to_numpy(x, converted, memo) for x in args]
        kwargs_np = {
            k: self._convert_to_numpy(v, converted, memo) for k, v in kwargs.items()
        }

        # Which arrays the kernel may have written to is resolved before the
        # call, so a mis-declared output is reported even if the kernel itself
        # would have raised first.
        writeable = (
            None
            if self._outputs is None
            else self._output_host_arrays(args_np, kwargs_np)
        )

        if _COUNTERS and converted:
            _record(
                "kernel_conversion",
                f"PyccelKernel {self.name!r}: {len(converted)} device array(s) "
                "copied to the host",
            )

        result = self._kernel(*args_np, **kwargs_np)

        # Copy in-place kernel updates back to the device arrays.
        for device_array, host_array in converted:
            if writeable is None or id(host_array) in writeable:
                device_array[...] = self._copy_to_device(host_array)

        return self._convert_from_numpy(result)

    @property
    def name(self) -> str:
        """The name of the wrapped kernel."""
        return getattr(self._kernel, "__name__", type(self._kernel).__name__)

    @property
    def kernel(self) -> Callable[..., Any]:
        """The wrapped kernel."""
        return self._kernel

    @property
    def use_cupy(self) -> bool:
        """Whether calls currently convert between device and host arrays."""
        if self._use_cupy is not None:
            return self._use_cupy
        return _cupy_backend()

    @property
    def object_modules(self) -> tuple[str, ...]:
        """The module prefixes whose instances are searched for arrays."""
        return self._object_modules

    @property
    def is_array(self) -> Callable[[Any], bool]:
        """The predicate for host values to move back to the device."""
        return self._is_array

    @property
    def outputs(self) -> tuple[int | str, ...] | None:
        """The declared output arguments, or None if every array is copied back."""
        return self._outputs


#: The names of the host implementations a kernel can have, see
#: :class:`~cunumpy.kernels.HostImplementations`.
HOST_IMPLEMENTATIONS = ("pyccel", "numba", "numpy", "python")


def _check_implementation(name: str | None) -> str | None:
    if name is not None and name not in HOST_IMPLEMENTATIONS:
        raise ValueError(
            f"kernel implementation must be one of {HOST_IMPLEMENTATIONS} or None, "
            f"got {name!r}",
        )
    return name


_KERNEL_IMPLEMENTATION: str | None = _check_implementation(
    os.environ.get("CUNUMPY_HOST_KERNEL_IMPLEMENTATION", "").strip().lower() or None,
)


def set_host_kernel_implementation(name: str | None) -> None:
    """Choose the host implementation every kernel runs, like ``set_backend``.

    A kernel that lacks the chosen implementation, or cannot load it (e.g.
    Pyccel failed to compile), raises instead of running another one. Device
    calls are not affected. The setting is global, not per thread;
    ``CUNUMPY_HOST_KERNEL_IMPLEMENTATION`` sets it at import.

    Parameters
    ----------
    name : str or None
        One of :data:`~cunumpy.kernels.HOST_IMPLEMENTATIONS` (``"python"`` is
        the uncompiled source of the Pyccel version), or None for the default
        rule of :class:`~cunumpy.kernels.HostImplementations`.

    Raises
    ------
    ValueError
        If `name` is not a known implementation.

    Examples
    --------
    >>> xp.kernels.set_host_kernel_implementation("numpy")
    >>> xp.kernels.get_host_kernel_implementation()
    'numpy'
    >>> xp.kernels.set_host_kernel_implementation(None)
    """
    global _KERNEL_IMPLEMENTATION
    _KERNEL_IMPLEMENTATION = _check_implementation(name)


def get_host_kernel_implementation() -> str | None:
    """Return the host implementation that is set, or None for the default.

    Returns
    -------
    str or None
        The name set with :func:`~cunumpy.kernels.set_host_kernel_implementation`.

    Examples
    --------
    >>> xp.kernels.get_host_kernel_implementation() is None
    True
    """
    return _KERNEL_IMPLEMENTATION


@contextmanager
def use_host_kernel_implementation(name: str | None) -> Iterator[None]:
    """Choose the host implementation inside a ``with`` block, like ``use_backend``.

    For tests and benchmarks, e.g. to run the code path of a machine without
    Pyccel. The setting is global, not per thread; the previous one is
    restored on exit.

    Parameters
    ----------
    name : str or None
        As for :func:`~cunumpy.kernels.set_host_kernel_implementation`.

    Yields
    ------
    None

    Examples
    --------
    >>> with xp.kernels.use_host_kernel_implementation("python"):
    ...     xp.kernels.get_host_kernel_implementation()
    'python'
    """
    global _KERNEL_IMPLEMENTATION
    previous = _KERNEL_IMPLEMENTATION
    set_host_kernel_implementation(name)
    try:
        yield
    finally:
        _KERNEL_IMPLEMENTATION = previous


#: The names of the device implementations a kernel can have.
DEVICE_IMPLEMENTATIONS = ("cuda",)


def _check_device_implementation(name: str | None) -> str | None:
    if name is not None and name not in DEVICE_IMPLEMENTATIONS:
        raise ValueError(
            f"device kernel implementation must be one of {DEVICE_IMPLEMENTATIONS} "
            f"or None, got {name!r}",
        )
    return name


_DEVICE_KERNEL_IMPLEMENTATION = _check_device_implementation(
    os.environ.get("CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION", "").strip().lower() or None,
)


def set_device_kernel_implementation(name: str | None) -> None:
    """Choose the device implementation for dispatched kernels.

    With ``"cuda"``, a :class:`~cunumpy.kernels.Kernel` without a CUDA
    implementation raises :class:`LookupError` on device arguments, even with
    ``missing_cuda="fallback"``. Does not switch the array backend or affect
    host calls or direct :class:`~cunumpy.kernels.CudaKernel` calls. Global,
    not per thread; ``CUNUMPY_DEVICE_KERNEL_IMPLEMENTATION`` sets it at import.

    Parameters
    ----------
    name : str or None
        ``"cuda"`` (see :data:`~cunumpy.kernels.DEVICE_IMPLEMENTATIONS`), or
        None for automatic selection.

    Raises
    ------
    ValueError
        If `name` is not supported; the setting is unchanged.

    Examples
    --------
    >>> xp.kernels.set_device_kernel_implementation("cuda")
    >>> xp.kernels.get_device_kernel_implementation()
    'cuda'
    >>> xp.kernels.set_device_kernel_implementation(None)
    """
    global _DEVICE_KERNEL_IMPLEMENTATION
    _DEVICE_KERNEL_IMPLEMENTATION = _check_device_implementation(name)


def get_device_kernel_implementation() -> str | None:
    """Return the requested device implementation, or None for automatic selection.

    Returns
    -------
    str or None
        The setting; None in automatic mode even when calls use CUDA.

    Examples
    --------
    >>> xp.kernels.get_device_kernel_implementation() is None
    True
    """
    return _DEVICE_KERNEL_IMPLEMENTATION


@contextmanager
def use_device_kernel_implementation(name: str | None) -> Iterator[None]:
    """Choose the device implementation inside a ``with`` block.

    The setting is global, not per thread; the previous one is restored on
    exit, also after an exception.

    Parameters
    ----------
    name : str or None
        As for :func:`~cunumpy.kernels.set_device_kernel_implementation`.

    Yields
    ------
    None

    Examples
    --------
    >>> with xp.kernels.use_device_kernel_implementation("cuda"):
    ...     xp.kernels.get_device_kernel_implementation()
    'cuda'
    """
    global _DEVICE_KERNEL_IMPLEMENTATION
    previous = _DEVICE_KERNEL_IMPLEMENTATION
    set_device_kernel_implementation(name)
    try:
        yield
    finally:
        _DEVICE_KERNEL_IMPLEMENTATION = previous


class HostImplementations:
    """Hold the host implementations of one kernel and call the selected one.

    Each implementation is loaded on first use (a Pyccel build, an import); a
    failed load is remembered in `errors`. A call runs the implementation set
    with :func:`~cunumpy.kernels.set_host_kernel_implementation`, or else the
    first available of ``"pyccel"``, ``"numba"`` and ``"numpy"``, and as a
    last resort ``"python"`` with a :class:`RuntimeWarning` (once). Built by
    :meth:`Kernel.from_folder <cunumpy.kernels.Kernel.from_folder>`.

    Parameters
    ----------
    name : str
        Name of the kernel.
    loaders : mapping of str to callable
        For each implementation name (from
        :data:`~cunumpy.kernels.HOST_IMPLEMENTATIONS`), a function returning
        the implementation or raising if it is unavailable. ``"python"`` is
        required.

    Attributes
    ----------
    errors : dict of str to Exception
        The load errors, by implementation name.

    Raises
    ------
    ValueError
        If `loaders` has an unknown name or no ``"python"`` entry.

    Examples
    --------
    >>> def push(x):
    ...     return x + 1
    >>> host = xp.kernels.HostImplementations(
    ...     "push", {"numpy": lambda: push, "python": lambda: push}
    ... )
    >>> host.selected(), host(1)
    ('numpy', 2)
    """

    def __init__(
        self,
        name: str,
        loaders: Mapping[str, Callable[[], Callable[..., Any]]],
    ) -> None:
        unknown = set(loaders) - set(HOST_IMPLEMENTATIONS)
        if unknown:
            raise ValueError(
                f"kernel {name!r}: unknown implementations {sorted(unknown)}, "
                f"expected names from {HOST_IMPLEMENTATIONS}",
            )
        if "python" not in loaders:
            raise ValueError(f"kernel {name!r}: the 'python' implementation is needed")
        self.__name__ = name
        self._loaders = dict(loaders)
        self._loaded: dict[str, Callable[..., Any]] = {}
        self.errors: dict[str, BaseException] = {}
        self._default: str | None = None

    def __repr__(self) -> str:
        return f"HostImplementations({self.__name__!r}, {list(self.names)})"

    @property
    def names(self) -> tuple[str, ...]:
        """The implementations this kernel has, loaded or not."""
        return tuple(name for name in HOST_IMPLEMENTATIONS if name in self._loaders)

    @property
    def python(self) -> Callable[..., Any]:
        """The uncompiled Python function (for signatures and reference results)."""
        return self.get("python")

    def available(self, name: str) -> bool:
        """Return whether implementation `name` exists and loads (loads it now).

        Parameters
        ----------
        name : str
            Implementation name.

        Returns
        -------
        bool
            False if it is missing or failed to load (see `errors`).
        """
        if name not in self._loaders:
            return False
        try:
            self.get(name)
        except Exception:  # noqa: BLE001 -- recorded in errors
            return False
        return True

    def get(self, name: str) -> Callable[..., Any]:
        """Return implementation `name`, loading it on first use.

        Parameters
        ----------
        name : str
            Implementation name.

        Returns
        -------
        callable
            The implementation.

        Raises
        ------
        LookupError
            If the kernel has no such implementation, or it failed to load
            (the load error is the cause).
        """
        function = self._loaded.get(name)
        if function is not None:
            return function
        if name not in self._loaders:
            raise LookupError(
                f"kernel {self.__name__!r} has no {name!r} implementation "
                f"(it has {', '.join(self.names)})",
            )
        if name in self.errors:
            raise LookupError(
                f"the {name!r} implementation of kernel {self.__name__!r} is "
                f"unavailable: {self.errors[name]!r}",
            ) from self.errors[name]
        try:
            function = self._loaders[name]()
        except Exception as error:
            self.errors[name] = error
            raise LookupError(
                f"the {name!r} implementation of kernel {self.__name__!r} is "
                f"unavailable: {error!r}",
            ) from error
        self._loaded[name] = function
        return function

    def selected(self) -> str:
        """Return the name of the implementation a call runs now.

        Loads the default on first use and keeps it.

        Returns
        -------
        str
            The name set with
            :func:`~cunumpy.kernels.set_host_kernel_implementation`, or the
            default.
        """
        if _KERNEL_IMPLEMENTATION is not None:
            return _KERNEL_IMPLEMENTATION
        if self._default is None:
            for name in ("pyccel", "numba", "numpy"):
                if self.available(name):
                    self._default = name
                    break
            else:
                warnings.warn(
                    f"Kernel {self.__name__!r}: no compiled or NumPy implementation "
                    f"is available ({self.errors!r}); running the uncompiled "
                    "Python version.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._default = "python"
        return self._default

    def build(self) -> str:
        """Load the default implementation now, e.g. to compile at setup.

        Returns
        -------
        str
            The name of the selected implementation.
        """
        return self.selected()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get(self.selected())(*args, **kwargs)


class CompiledHostKernel:
    """Call a host kernel that is compiled on its first call, with a fallback.

    cunumpy compiles nothing itself: `compiler` is the caller's function, e.g. a
    wrapper around ``pyccel.epyccel`` with a cache, or one importing modules
    compiled ahead of time. Used by ``KernelCatalog.from_package(...,
    compile_host=...)``.

    Parameters
    ----------
    module : str or module
        The module that defines the kernel, or its name.
    name : str
        Name of the kernel function in `module`.
    compiler : callable
        Takes the module and returns its compiled form (same function names).
    fallback : callable, optional
        Called when compilation fails, e.g. a NumPy version with the same
        arguments. Without one, the uncompiled Python function runs, with a
        :class:`RuntimeWarning` (once).

    Attributes
    ----------
    error : Exception or None
        The exception of a failed build.

    Examples
    --------
    >>> def no_compiler(module):
    ...     raise RuntimeError("no Pyccel")
    >>> kernel = xp.kernels.CompiledHostKernel(
    ...     "math", "sqrt", no_compiler, fallback=np.sqrt
    ... )
    >>> kernel(4.0), kernel.compiled
    (np.float64(2.0), False)
    """

    def __init__(
        self,
        module: str | ModuleType,
        name: str,
        compiler: Callable[[ModuleType], ModuleType],
        fallback: Callable[..., Any] | None = None,
    ) -> None:
        self._module = module
        self.__name__ = name
        self._compiler = compiler
        self._fallback = fallback
        self._compiled: Callable[..., Any] | None = None
        self._built = False
        self.error: BaseException | None = None
        self._warned = False

    def __repr__(self) -> str:
        state = (
            "not built"
            if not self._built
            else ("compiled" if self._compiled else "failed")
        )
        return f"CompiledHostKernel({self.__name__!r}, {state})"

    @property
    def module(self) -> ModuleType:
        """The (uncompiled) module that defines the kernel."""
        if isinstance(self._module, str):
            self._module = importlib.import_module(self._module)
        return self._module

    @property
    def python(self) -> Callable[..., Any]:
        """The uncompiled Python function (for signatures and reference results)."""
        return getattr(self.module, self.__name__)

    @property
    def fallback(self) -> Callable[..., Any] | None:
        """What runs when compilation fails, or None for the Python function."""
        return self._fallback

    def build(self) -> Callable[..., Any] | None:
        """Compile now, once; a failure is kept in `error`.

        Returns
        -------
        callable or None
            The compiled function, or None if compilation failed.
        """
        if not self._built:
            self._built = True
            try:
                self._compiled = getattr(self._compiler(self.module), self.__name__)
            except Exception as error:  # noqa: BLE001 -- no Pyccel or no compiler
                self.error = error
        return self._compiled

    @property
    def compiled(self) -> bool:
        """Whether the compiled version is available (compiles on first access)."""
        return self.build() is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kernel = self.build()
        if kernel is None:
            if self._fallback is not None:
                kernel = self._fallback
            else:
                if not self._warned:
                    warnings.warn(
                        f"Kernel {self.__name__!r} could not be compiled "
                        f"({self.error!r}); running its uncompiled Python version.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned = True
                kernel = self.python
        return kernel(*args, **kwargs)


def _c_ordered_with_gaps(array: np.ndarray) -> bool:
    """Return whether `array` is in C order, possibly with gaps (e.g. ``a[:, :n]``)."""
    extent = array.itemsize
    for length, stride in reversed(list(zip(array.shape, array.strides, strict=True))):
        if length == 1:
            continue
        if stride < extent:
            return False
        extent = stride * length
    return True


def _with_unit_axis_strides(array: np.ndarray) -> np.ndarray:
    """Return `array`, or a view, with C-order strides on its length-1 axes."""
    if array.size == 0:
        return array
    strides = list(array.strides)
    extent = array.itemsize
    for axis in reversed(range(array.ndim)):
        if array.shape[axis] == 1:
            strides[axis] = max(strides[axis], extent)
        else:
            extent = strides[axis] * array.shape[axis]
    if tuple(strides) == array.strides:
        return array
    return np.lib.stride_tricks.as_strided(
        array, strides=strides, writeable=array.flags.writeable
    )


def as_kernel_array(
    value: Any, like: Any, dtype: Any = None, *, strided: bool = False
) -> Any:
    """Return `value` as an input array for the kernel chosen for `like`.

    For a :class:`~cunumpy.kernels.Kernel` with ``dispatch="arrays"``: pass the
    main array (e.g. the grid) as `like` so that all arguments end up on one
    side. Returns `value` itself if it already fits, else one copy (moved
    across if needed, counted by ``count_transfers()``). For arrays the kernel
    writes, use :func:`~cunumpy.kernels.kernel_output`.

    Parameters
    ----------
    value : array_like
        The array to convert.
    like : array
        A CuPy array for a device kernel, anything else for a host kernel.
    dtype : dtype, optional
        Required dtype; by default any.
    strided : bool, optional
        Also take a NumPy array unchanged when it is in C order with gaps
        (positive strides, each at least the extent of the next axis), such as
        ``storage[:, :n]``. Pyccel's wrappers accept these without a copy; a
        host kernel that needs contiguous memory must not use it. Default
        False. CuPy arrays are always made C-contiguous.

    Returns
    -------
    array
        A C-contiguous (or, with `strided`, C-ordered) array on the side of
        `like`, with `dtype`.

    Examples
    --------
    >>> grid = np.zeros((2, 3))
    >>> xp.kernels.as_kernel_array(grid, like=grid) is grid
    True
    >>> xp.kernels.as_kernel_array([1, 2], like=grid, dtype=float)
    array([1., 2.])
    """
    # fast path: a C-contiguous NumPy array of the dtype for a host kernel, the
    # common case, without the conversions below (subclasses take the full path)
    if (
        type(value) is np.ndarray
        and (type(like) is np.ndarray or not is_gpu(like))
        and (dtype is None or value.dtype == dtype)
        and value.flags.c_contiguous
        and value.ndim > 0
        and (value.ndim == 1 or 1 not in value.shape)
    ):
        return value
    if is_gpu(like):
        import cupy

        if not is_gpu(value):
            value = to_cupy(value)
        result = cupy.ascontiguousarray(value, dtype=dtype)
        if _COUNTERS and _is_device_copy(value, result):
            _record(
                "device_copy",
                f"as_kernel_array({_describe(value)})",
                nbytes=_nbytes(result),
            )
        return result
    value = to_numpy(value)
    if (
        strided
        and value.ndim > 0
        and (dtype is None or value.dtype == np.dtype(dtype))
        and _c_ordered_with_gaps(value)
    ):
        return _with_unit_axis_strides(value)
    return _with_unit_axis_strides(np.ascontiguousarray(value, dtype=dtype))


def _same_memory(buffer: Any, out: Any) -> bool:
    """Return whether `buffer` is `out` or a view of exactly its entries."""
    if buffer is out:
        return True
    return (
        isinstance(buffer, np.ndarray)
        and isinstance(out, np.ndarray)
        and buffer.shape == out.shape
        and buffer.dtype == out.dtype
        and buffer.ctypes.data == out.ctypes.data
        and all(
            length == 1 or a == b
            for length, a, b in zip(out.shape, buffer.strides, out.strides, strict=True)
        )
    )


class _KernelOutput:
    """The context manager of :func:`kernel_output` (a class: cheaper than a generator)."""

    __slots__ = ("_buffer", "_out")

    def __init__(self, out: Any, buffer: Any) -> None:
        self._out = out
        self._buffer = buffer

    def __enter__(self) -> Any:
        return self._buffer

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            return
        buffer, out = self._buffer, self._out
        if not _same_memory(buffer, out):
            if is_gpu(out) and not is_gpu(buffer):
                out[...] = to_cupy(buffer)
            elif not is_gpu(out) and is_gpu(buffer):
                out[...] = to_numpy(buffer)
            else:
                out[...] = buffer


def kernel_output(
    out: Any, like: Any, dtype: Any = None, *, strided: bool = False
) -> _KernelOutput:
    """Provide the output buffer for the kernel chosen for `like`, copied into `out`.

    The buffer is `out` itself if :func:`~cunumpy.kernels.as_kernel_array`
    takes it unchanged, else a converted copy whose contents are written into
    `out` (on its own side) when the block ends without an error.

    Parameters
    ----------
    out : array
        The array the kernel should write.
    like : array
        A CuPy array for a device kernel, anything else for a host kernel.
    dtype : dtype, optional
        Required dtype; by default any.
    strided : bool, optional
        As for :func:`~cunumpy.kernels.as_kernel_array`. Default False.

    Returns
    -------
    context manager
        Yields the buffer to pass to the kernel.

    Examples
    --------
    >>> out = np.zeros(2, dtype=np.float32)
    >>> with xp.kernels.kernel_output(out, like=out, dtype=float) as buffer:
    ...     buffer[:] = 1.5  # the kernel writes a float64 copy
    >>> out
    array([1.5, 1.5], dtype=float32)
    """
    return _KernelOutput(out, as_kernel_array(out, like, dtype, strided=strided))


_SCALAR_ANNOTATIONS = {"int", "float", "bool", "complex", "str"}


def outputs_from_annotations(function: Callable[..., Any]) -> tuple[str, ...] | None:
    """Return the parameters a kernel may write to, read from its annotations.

    A parameter is an output unless it is annotated ``Final`` (Pyccel's mark
    for read-only, e.g. ``"Final[float[:]]"``), ``const``, or a scalar type
    (``int``, ``float``, ``bool``, ``complex``, ``str``). Use the result as the
    `outputs` of a :class:`~cunumpy.kernels.PyccelKernel`.

    Parameters
    ----------
    function : callable
        The Python kernel function.

    Returns
    -------
    tuple of str or None
        The parameter names, in order, or None if `function` has no signature
        or no annotated parameter.

    Examples
    --------
    >>> def push(x: "Final[float[:]]", dt: float, v: "float[:]"):
    ...     v[:] += dt * x
    >>> xp.kernels.outputs_from_annotations(push)
    ('v',)
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return None
    if all(p.annotation is inspect.Parameter.empty for p in parameters):
        return None
    outputs = []
    for p in parameters:
        if isinstance(p.annotation, str):
            text = p.annotation
        elif isinstance(p.annotation, type):
            text = p.annotation.__name__
        else:
            text = str(p.annotation)
        text = text.replace("typing.", "").strip().strip("'\"")
        if p.annotation is inspect.Parameter.empty:
            outputs.append(p.name)  # nothing is known: assume it may be written
        elif (
            text.startswith(("Final[", "const ", "Final "))
            or text in _SCALAR_ANNOTATIONS
        ):
            continue
        else:
            outputs.append(p.name)
    return tuple(outputs)
