"""Capture a step of CUDA work once and replay it as a CUDA graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cunumpy import _fake_cupy
from cunumpy.xp import array_backend

__all__ = ["CudaGraph", "capture_graph"]


class CudaGraph:
    """A captured CUDA graph, or on the host the function that was captured.

    Created by :func:`~cunumpy.cuda.capture_graph`. :meth:`launch` replays the
    kernels and copies of the captured call with one launch on CUDA; on the
    NumPy backend (and the fake CuPy) it calls the function again, so the same
    code runs on both.

    Parameters
    ----------
    graph : cupy.cuda.Graph or None
        The captured graph; None on the host.
    function : callable
        The captured function, called by :meth:`launch` on the host.
    args : tuple
        Its positional arguments.
    kwargs : dict
        Its keyword arguments.
    """

    def __init__(
        self,
        graph: Any,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._graph = graph
        self._function = function
        self._args = args
        self._kwargs = kwargs

    @property
    def captured(self) -> bool:
        """Whether this is a CUDA graph (False on the host)."""
        return self._graph is not None

    def launch(self, stream: Any = None) -> None:
        """Replay the captured work (asynchronously on CUDA).

        Parameters
        ----------
        stream : cupy.cuda.Stream, optional
            Stream to launch on; the current stream by default. Ignored on the
            host.
        """
        if self._graph is None:
            self._function(*self._args, **self._kwargs)
        else:
            self._graph.launch(stream)

    __call__ = launch

    def __repr__(self) -> str:
        kind = "CUDA graph" if self.captured else "host"
        name = getattr(self._function, "__name__", repr(self._function))
        return f"CudaGraph({name!r}, {kind})"


def _real_cupy() -> bool:
    return array_backend.backend == "cupy" and not _fake_cupy.is_active()


def capture_graph(
    function: Callable[..., Any],
    *args: Any,
    warmup: bool = True,
    **kwargs: Any,
) -> CudaGraph:
    """Capture the CUDA work of ``function(*args, **kwargs)`` into a graph.

    A time step of a GPU code is often many small kernels, and launching each
    from Python costs more than running it. A CUDA graph records the launches
    and copies once; :meth:`CudaGraph.launch` then replays all of them with a
    single launch, on the same arrays. Only work on the GPU is recorded:
    Python code in `function` runs once, at the capture, so the step must not
    depend on host values that change (pass them in device arrays), and must not
    read device values on the host (``float(x)``, ``bool(x.any())``, ``check_finite``
    kernels), which fails during the capture. Allocate the arrays of the step
    before (e.g. with the ``out=`` arguments of :mod:`cunumpy.algorithms`).

    On the NumPy backend (and the fake CuPy) nothing is captured: the returned
    object calls `function` at every launch, so the same code runs on both.

    Parameters
    ----------
    function : callable
        The step to capture.
    *args
        Its positional arguments.
    warmup : bool, optional
        Call `function` once before the capture (True by default), so that its
        kernels are compiled and the memory pool has the blocks it needs; both
        must not happen during a capture. On CUDA this runs the step once.
    **kwargs
        Its keyword arguments.

    Returns
    -------
    CudaGraph
        Replay with ``graph.launch()``.

    Examples
    --------
    >>> x = xp.zeros(3)
    >>> def step():
    ...     x[...] += 1.0
    >>> graph = xp.cuda.capture_graph(step, warmup=False)
    >>> for _ in range(3):
    ...     graph.launch()
    >>> x
    array([3., 3., 3.])
    """
    if not _real_cupy():
        if warmup:
            function(*args, **kwargs)
        return CudaGraph(None, function, args, kwargs)

    import cupy as cp

    if warmup:
        function(*args, **kwargs)
    cp.cuda.Device().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            function(*args, **kwargs)
        finally:
            graph = stream.end_capture()
    return CudaGraph(graph, function, args, kwargs)
