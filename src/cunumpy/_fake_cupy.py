"""A strict stand-in for CuPy on machines without a GPU, for tests only.

With the real CuPy absent, the CuPy code paths of a program (argument objects
built from device arrays, host/device conversions, MPI buffers, backend
branches) cannot run in CI at all. This module installs a fake ``cupy``
package whose arrays live in host memory but behave like CuPy arrays where it
matters for finding host/device bugs:

* ``cupy.ndarray`` is **not** a ``numpy.ndarray``; ``numpy.asarray(a)`` raises
  (use ``.get()``), so compiled host kernels reject these arrays exactly as
  they reject real CuPy arrays;
* ``cupy.<function>`` rejects NumPy arrays and Python lists as array arguments,
  like CuPy does; mixing CuPy and NumPy arrays in arithmetic raises;
* reductions and scalar indexing return 0-d arrays, not Python scalars;
* arrays have ``data.ptr``, ``device`` and ``__cuda_array_interface__``, so
  :class:`~cunumpy.cuda.CudaStruct` packing and the argument checks of
  :class:`~cunumpy.cuda.CudaKernel` work;
* CUDA kernels cannot run: ``RawKernel`` and friends raise
  ``NotImplementedError`` when called, and :func:`cunumpy.kernel_testing.requires_cupy`
  skips tests while the fake is active.

Activate it before CuPy or cunumpy's backend is first used, either with the
environment variable ``CUNUMPY_FAKE_CUPY=1`` (read when cunumpy is imported)
or by calling :func:`install` first thing in a test session (e.g. in
``conftest.py``). Then ``ARRAY_BACKEND=cupy`` or ``xp.set_backend("cupy")``
selects the fake like the real thing.

Never installed with a real CuPy present (``install`` raises), never shipped
as a top-level ``cupy`` package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

__all__ = ["install", "is_active", "uninstall"]

_IMPLEMENTATION = Path(__file__).with_name("_fake_cupy_impl.py")
_SUBMODULES = ("cupy.cuda", "cupy.cuda.device", "cupy.cuda.runtime", "cupy.linalg")


def is_active() -> bool:
    """Whether the fake CuPy is the ``cupy`` module of this process."""
    module = sys.modules.get("cupy")
    return bool(getattr(module, "__cunumpy_fake__", False))


def install() -> types.ModuleType:
    """Install the fake ``cupy`` package into ``sys.modules`` and return it.

    Idempotent. Raises ``RuntimeError`` if the real CuPy has been imported
    already, and ``RuntimeError`` if cunumpy has already decided that CuPy is
    unavailable (import cunumpy after installing, or set the environment
    variable instead).

    Returns
    -------
    types.ModuleType
        The fake ``cupy`` module.
    """
    existing = sys.modules.get("cupy")
    if existing is not None:
        if getattr(existing, "__cunumpy_fake__", False):
            return existing
        raise RuntimeError(
            "the real CuPy is already imported; the fake CuPy must be installed "
            "before CuPy (set CUNUMPY_FAKE_CUPY=1 or call install() first)"
        )
    xp = sys.modules.get("cunumpy.xp")
    if xp is not None and getattr(xp, "_CUPY_AVAILABLE_CACHE", None) is not None:
        raise RuntimeError(
            "cunumpy has already checked for CuPy; install the fake CuPy before "
            "the first backend use (set CUNUMPY_FAKE_CUPY=1 in the environment)"
        )
    module = types.ModuleType("cupy")
    module.__file__ = str(_IMPLEMENTATION)
    module.__cunumpy_fake__ = True
    code = compile(_IMPLEMENTATION.read_text(), str(_IMPLEMENTATION), "exec")
    sys.modules["cupy"] = module
    try:
        exec(code, module.__dict__)  # noqa: S102 - our own file, classes named cupy.*
    except BaseException:
        uninstall()
        raise
    return module


def uninstall() -> None:
    """Remove the fake ``cupy`` package from ``sys.modules`` (no-op otherwise)."""
    if not is_active():
        return
    for name in [n for n in sys.modules if n == "cupy" or n.startswith("cupy.")]:
        del sys.modules[name]
    for name in [n for n in sys.modules if n.startswith("array_api_compat.cupy")]:
        del sys.modules[name]
