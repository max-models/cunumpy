"""Pytest helpers for testing host/CUDA kernel pairs.

Not imported by ``import cunumpy``; pytest is imported only when one of the
pytest objects below is used, so e.g. :func:`device_function_kernel` works
without it. See :doc:`/kernels/testing`.

* Compare host and CUDA kernels: :func:`assert_kernels_agree`, and for a
  catalog :func:`parity_cases` / :func:`check_parity`.
* Run on both backends (built on first access):

  ``BACKENDS``
      ``["numpy", pytest.param("cupy", marks=requires_cupy)]``, to
      parametrize a test over the backends.
  ``backend``
      A pytest fixture that runs the test once per backend, with it active.
  ``requires_cupy``
      A ``skipif`` marker for tests that launch CUDA kernels (needs a GPU,
      not the fake CuPy).
  ``requires_device_backend``
      A ``skipif`` marker for tests that need a GPU or the fake CuPy, see
      :func:`device_backend_available`.

  With ``CUNUMPY_REQUIRE_CUDA=1`` (:func:`cuda_required`) they fail
  instead of skipping.
* Test ``__device__`` helpers: :func:`device_function_kernel`.
* Run CUDA kernels on the CPU, without a GPU (from ``cunumpy._emulation``):
  :func:`emulate_cuda_kernel`, :func:`emulated_launches`,
  :func:`compile_for_emulation`, :func:`emulation_compiler` and
  :func:`emulation_cache_dir`.
* Run CuPy-backend code on the fake CuPy, a strict host stand-in for CuPy
  (also installed by ``CUNUMPY_FAKE_CUPY=1``): :func:`install_fake_cupy`,
  :func:`fake_cupy_active`, :func:`fake_cupy_session`,
  :func:`run_in_fake_cupy_subprocess` and :func:`host_buffer` (the NumPy
  array behind a fake CuPy array, not a copy).

Examples
--------
One parametrized test covers every kernel of a catalog::

    from cunumpy.kernel_testing import parity_cases, check_parity

    @pytest.mark.parametrize("kernel", parity_cases(catalog))
    def test_parity(kernel):
        check_parity(kernel)
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import array_api_compat
import numpy as np

from cunumpy import _fake_cupy
from cunumpy._cuda_kernel import (
    CudaKernel,
    CudaParameter,
    CudaStructArguments,
    CudaStructValue,
    _parse_parameter,
    _split_top_level,
    _strip_comments,
)
from cunumpy._dispatch import Kernel
from cunumpy._emulation import (
    compile_for_emulation,
    emulate_cuda_kernel,
    emulated_launches,
    emulation_cache_dir,
    emulation_compiler,
)
from cunumpy._fake_cupy import host_buffer
from cunumpy.xp import cupy_available, get_backend, to_numpy, use_backend

# the pytest objects are created on first access, see __getattr__
__all__ = [
    "BACKENDS",  # noqa: F822
    "assert_kernels_agree",
    "backend",  # noqa: F822
    "check_parity",
    "compile_for_emulation",
    "cuda_required",
    "device_backend_available",
    "device_function_kernel",
    "emulate_cuda_kernel",
    "emulated_launches",
    "emulation_cache_dir",
    "emulation_compiler",
    "fake_cupy_active",
    "fake_cupy_session",
    "host_buffer",
    "install_fake_cupy",
    "parity_cases",
    "requires_cupy",  # noqa: F822
    "requires_device_backend",  # noqa: F822
    "run_in_fake_cupy_subprocess",
]

SKIP_REASON = "CuPy/GPU not available"
DEVICE_SKIP_REASON = (
    "neither a GPU nor the fake CuPy (CUNUMPY_FAKE_CUPY=1) is available"
)
FAKE_SKIP_REASON = "the fake CuPy cannot run CUDA kernels"


def fake_cupy_active() -> bool:
    """Tell whether the fake CuPy stands in for CuPy in this process.

    Returns
    -------
    bool
        True after :func:`install_fake_cupy` or with ``CUNUMPY_FAKE_CUPY=1``.

    Examples
    --------
    >>> from cunumpy.kernel_testing import fake_cupy_active
    >>> fake_cupy_active()  # doctest: +SKIP
    False
    """
    return _fake_cupy.is_active()


def install_fake_cupy() -> Any:
    """Install the fake CuPy, a strict host stand-in for CuPy, in this process.

    Its arrays live in host memory but behave like CuPy arrays (e.g.
    ``numpy.asarray`` rejects them); CUDA kernels cannot run on it except in
    :func:`emulated_launches`. Call it before the first backend use (e.g. at
    the top of ``conftest.py``), or set ``CUNUMPY_FAKE_CUPY=1`` instead.
    Calling it again is a no-op.

    Returns
    -------
    module
        The fake ``cupy`` module.

    Raises
    ------
    RuntimeError
        If the real CuPy was imported already, or cunumpy already checked
        for CuPy.

    Examples
    --------
    >>> from cunumpy.kernel_testing import install_fake_cupy
    >>> install_fake_cupy()  # doctest: +SKIP
    """
    return _fake_cupy.install()


def _can_launch() -> bool:
    """Whether CUDA kernels can run: a functional CuPy that is not the fake."""
    return cupy_available() and not fake_cupy_active()


def cuda_required() -> bool:
    """Tell whether ``CUNUMPY_REQUIRE_CUDA`` demands a real GPU (the CI guard).

    If so, tests that need a GPU fail instead of being skipped (``requires_cupy``,
    ``requires_device_backend``, :func:`assert_kernels_agree`, the ``cupy``
    run of the ``backend`` fixture), so that a GPU CI job cannot pass
    silently because CuPy or the driver is broken.

    Returns
    -------
    bool
        True if ``CUNUMPY_REQUIRE_CUDA`` is ``1``, ``true`` or ``yes``.

    Examples
    --------
    >>> from cunumpy.kernel_testing import cuda_required
    >>> cuda_required()  # doctest: +SKIP
    False
    """
    return os.environ.get("CUNUMPY_REQUIRE_CUDA", "").lower() in {"1", "true", "yes"}


def _skip_or_fail(reason: str) -> None:
    """``pytest.skip``, or ``pytest.fail`` if a GPU is required."""
    if cuda_required():
        _pytest().fail(f"CUNUMPY_REQUIRE_CUDA is set, but {reason}", pytrace=False)
    _pytest().skip(reason)


def cuda_gate() -> bool:
    """Return the ``requires_cupy`` condition under ``CUNUMPY_REQUIRE_CUDA``.

    Returns
    -------
    bool
        False (do not skip) if CUDA kernels can run; otherwise the test fails.
    """
    if not _can_launch():
        reason = FAKE_SKIP_REASON if fake_cupy_active() else SKIP_REASON
        _pytest().fail(f"CUNUMPY_REQUIRE_CUDA is set, but {reason}", pytrace=False)
    return False


def device_backend_available() -> bool:
    """Tell whether a CuPy-backend program can run here: a GPU, or the fake CuPy.

    Unlike ``requires_cupy`` (CUDA kernels can be launched), this is also true
    on the fake CuPy, where launches only work inside :func:`emulated_launches`
    (see :func:`fake_cupy_session`). The marker ``requires_device_backend``
    skips tests where it is false.

    Returns
    -------
    bool
        True with a working CuPy or with the fake CuPy active.

    Examples
    --------
    >>> from cunumpy.kernel_testing import device_backend_available
    >>> device_backend_available()  # doctest: +SKIP
    True
    """
    return fake_cupy_active() or cupy_available()


def device_backend_gate() -> bool:
    """Return the ``requires_device_backend`` condition under ``CUNUMPY_REQUIRE_CUDA``.

    Returns
    -------
    bool
        False (do not skip) if a device backend is available; otherwise the
        test fails.
    """
    if not device_backend_available():
        _pytest().fail(
            f"CUNUMPY_REQUIRE_CUDA is set, but {DEVICE_SKIP_REASON}", pytrace=False
        )
    return False


@contextlib.contextmanager
def fake_cupy_session(
    *,
    compiler: str | None = None,
    options: Sequence[str] = (),
) -> Iterator[None]:
    """Run a CuPy-backend program on the CPU, on the fake CuPy.

    Activates the CuPy backend (the fake CuPy) and emulates every CUDA launch
    and compilation in the block with :func:`emulated_launches`.

    Parameters
    ----------
    compiler : str, optional
        The C++ compiler of the emulation; default :func:`emulation_compiler`.
    options : sequence of str, optional
        Extra compiler options, e.g. ``("-ffp-contract=off",)``.

    Yields
    ------
    None
        The block runs on the fake CuPy with emulated launches.

    Raises
    ------
    RuntimeError
        If the fake CuPy is not active (:func:`install_fake_cupy`,
        ``CUNUMPY_FAKE_CUPY=1``; or use :func:`run_in_fake_cupy_subprocess`).

    Examples
    --------
    >>> from cunumpy.kernel_testing import fake_cupy_session
    >>> with fake_cupy_session():  # doctest: +SKIP
    ...     sim.run()  # kernels compiled and launched, all on the CPU
    """
    if not fake_cupy_active():
        raise RuntimeError(
            "fake_cupy_session() needs the fake CuPy: set CUNUMPY_FAKE_CUPY=1 or call "
            "install_fake_cupy() before cunumpy is used, or use "
            "run_in_fake_cupy_subprocess()",
        )
    with (
        use_backend("cupy", strict=True),
        emulated_launches(compiler=compiler, options=options),
    ):
        yield


#: Prefixes of the environment variables through which an MPI launcher (Open
#: MPI, MPICH/Hydra, Intel MPI, Slurm, ...) hands a process its place in the job.
MPI_LAUNCHER_PREFIXES = (
    "OMPI_",
    "PMIX_",
    "PMI_",
    "HYDRA_",
    "MPIR_",
    "I_MPI_",
    "SLURM_",
    "MV2_",
    "MPI_LOCALRANKID",
    "ALPS_APP_PE",
    "PALS_",
)


def _tail(text: str, lines: int = 50) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_in_fake_cupy_subprocess(
    code: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run code in a serial child process on the fake CuPy; fail the test if it fails.

    The fake CuPy must be installed before anything imports cunumpy, so a test
    process that already uses cunumpy cannot switch to it. The child runs
    ``python -X faulthandler -c code`` with ``CUNUMPY_FAKE_CUPY=1``,
    ``OMP_NUM_THREADS=1``, ``MAYBEMPI=0`` and without the variables of an MPI
    launcher, so it does not join the parent's MPI job. Under MPI only rank 0
    starts the child; the other ranks skip the test.

    Parameters
    ----------
    code : str
        Python source to run.
    env : mapping of str to str, optional
        Additional environment variables for the child (e.g. ``PYTHONPATH``).
    timeout : float, optional
        Seconds after which the child is killed and the test fails.

    Returns
    -------
    subprocess.CompletedProcess
        The finished child (exit code 0), with its stdout and stderr.

    Notes
    -----
    If the child fails or times out, the test fails (``pytest.fail``) with the
    exit code or signal (e.g. ``SIGSEGV``) and the last 50 lines of its stdout
    and stderr.

    Examples
    --------
    >>> from cunumpy.kernel_testing import run_in_fake_cupy_subprocess
    >>> run_in_fake_cupy_subprocess("import my_sim; my_sim.check()")  # doctest: +SKIP
    """
    pytest = _pytest()
    from cunumpy.mpi import get_mpi

    if get_mpi().COMM_WORLD.Get_rank() != 0:
        pytest.skip("serial check in a child process, runs on MPI rank 0")
    child_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(MPI_LAUNCHER_PREFIXES)
    }
    child_env.update(CUNUMPY_FAKE_CUPY="1", OMP_NUM_THREADS="1", MAYBEMPI="0")
    child_env.update(env or {})
    command = [sys.executable, "-X", "faulthandler", "-c", code]
    try:
        result = subprocess.run(
            command,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        out, err = (
            t.decode(errors="replace") if isinstance(t, bytes) else (t or "")
            for t in (error.stdout, error.stderr)
        )
        pytest.fail(
            f"child process timed out after {timeout} s\n"
            f"--- stdout (end) ---\n{_tail(out)}\n--- stderr (end) ---\n{_tail(err)}",
            pytrace=False,
        )
    if result.returncode != 0:
        rc = result.returncode
        try:
            how = f"signal {signal.Signals(-rc).name}" if rc < 0 else f"exit code {rc}"
        except ValueError:
            how = f"signal {-rc}"
        pytest.fail(
            f"child process failed with {how}\n"
            f"--- stdout (end) ---\n{_tail(result.stdout)}\n"
            f"--- stderr (end) ---\n{_tail(result.stderr)}",
            pytrace=False,
        )
    return result


# pytest objects, built on first use so that importing this module does not
# import pytest (see __getattr__ below)
_LAZY: dict[str, Any] = {}


def _pytest() -> Any:
    try:
        import pytest
    except ImportError:  # pragma: no cover - pytest is installed in the test suite
        raise ImportError(
            "cunumpy.kernel_testing needs pytest for this feature: pip install pytest",
        ) from None
    return pytest


def _build_lazy() -> None:
    pytest = _pytest()
    if cuda_required() and not _can_launch():
        # a string condition is evaluated when the test is set up; it fails the
        # test (instead of skipping it) because there is no real GPU. With a
        # working GPU the marker is the ordinary one below.
        requires_cupy = pytest.mark.skipif(
            "__import__('cunumpy.kernel_testing', fromlist=['_']).cuda_gate()",
            reason=SKIP_REASON,
        )
    else:
        requires_cupy = pytest.mark.skipif(
            not _can_launch(),
            reason=FAKE_SKIP_REASON if fake_cupy_active() else SKIP_REASON,
        )
    if cuda_required() and not device_backend_available():
        requires_device_backend = pytest.mark.skipif(
            "__import__('cunumpy.kernel_testing', fromlist=['_']).device_backend_gate()",
            reason=DEVICE_SKIP_REASON,
        )
    else:
        requires_device_backend = pytest.mark.skipif(
            not device_backend_available(), reason=DEVICE_SKIP_REASON
        )
    backends = ["numpy", pytest.param("cupy", marks=requires_cupy)]

    @pytest.fixture(params=backends)
    def backend(request):
        """Run the test once per backend, with that backend active.

        With ``CUNUMPY_REQUIRE_CUDA`` set, the ``cupy`` run fails if the CuPy
        backend cannot be activated, instead of silently running on NumPy.

        Parameters
        ----------
        request : pytest.FixtureRequest
            The pytest request; ``request.param`` is the backend.

        Yields
        ------
        str
            The active backend, ``"numpy"`` or ``"cupy"``.
        """
        with use_backend(request.param, strict=cuda_required()):
            yield request.param

    _LAZY.update(
        requires_cupy=requires_cupy,
        requires_device_backend=requires_device_backend,
        BACKENDS=backends,
        backend=backend,
    )


def __getattr__(name: str) -> Any:
    if name in ("requires_cupy", "requires_device_backend", "BACKENDS", "backend"):
        if not _LAZY:
            _build_lazy()
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# assert_kernels_agree
# ---------------------------------------------------------------------------


def _is_array(value: Any) -> bool:
    return array_api_compat.is_numpy_array(value) or array_api_compat.is_cupy_array(
        value,
    )


def _arrays_in(value: Any, name: str, found: dict[str, Any], depth: int) -> None:
    """Record the arrays in `value` under `name`, looking `depth` levels deep."""
    if _is_array(value):
        found[name] = value
    elif depth == 0:
        return
    elif isinstance(value, (CudaStructArguments, CudaStructValue)) and hasattr(
        value,
        "struct",
    ):
        # by field name, like the attributes of the host argument object; the
        # fields of a CudaStructArguments may be properties (not in vars())
        for field in value.struct.fields:
            item = (
                value[field.name]
                if isinstance(value, CudaStructValue)
                else getattr(value, field.name)
            )
            _arrays_in(item, f"{name}.{field.name}", found, depth - 1)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _arrays_in(item, f"{name}[{i}]", found, depth - 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _arrays_in(item, f"{name}[{key!r}]", found, depth - 1)
    elif hasattr(value, "__dict__"):
        for attr, item in vars(value).items():
            _arrays_in(item, f"{name}.{attr}", found, depth - 1)


def _resolve_output(
    entry: Any,
    n_args: int,
    parameters: Sequence[str] | None,
) -> tuple[int, str | None]:
    """Return the argument index and field filter (or None) of an `outputs` entry."""
    if isinstance(entry, bool) or not isinstance(entry, (int, str)):
        raise TypeError(
            "outputs entries must be argument indices (int) or names (str, "
            f"optionally 'name.field'), got {entry!r}",
        )
    field = None
    if isinstance(entry, str):
        head, _, field = entry.partition(".")
        field = field or None
        if head.lstrip("-").isdigit():
            entry = int(head)
        elif parameters is not None and head in parameters:
            entry = list(parameters).index(head)
        else:
            known = "unknown" if parameters is None else sorted(parameters)
            raise KeyError(
                f"output {head!r} is not a parameter of the kernel (parameters: "
                f"{known}); give an index if the names are unknown",
            )
    index = entry + n_args if entry < 0 else entry
    if not 0 <= index < n_args:
        raise IndexError(
            f"output argument {entry} does not exist: there are {n_args} arguments",
        )
    return index, field


def _collect_arrays(
    args: Sequence[Any],
    outputs: Sequence[int | str] | None = None,
    parameters: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the arrays among `args` (or the arguments `outputs`), by argument name."""
    entries = range(len(args)) if outputs is None else outputs
    found: dict[str, Any] = {}
    for entry in entries:
        index, field = _resolve_output(entry, len(args), parameters)
        local: dict[str, Any] = {}
        _arrays_in(args[index], f"argument {index}", local, depth=2)
        if field is not None:
            prefix = f"argument {index}.{field}"
            local = {
                name: array
                for name, array in local.items()
                if name == prefix or name.startswith((prefix + ".", prefix + "["))
            }
            if not local:
                raise KeyError(
                    f"output {entry!r}: argument {index} has no array field {field!r}",
                )
        found.update(local)
    return found


def _compare_results(
    host: dict[str, Any],
    device: dict[str, Any],
    rtol: float,
    atol: float,
    kernel_name: str = "kernel",
) -> None:
    """Assert that the arrays of `device` match those of `host`, by name."""
    if host.keys() != device.keys():
        raise AssertionError(
            f"{kernel_name}: the host and CUDA calls do not have the same array "
            f"arguments: host {sorted(host)}, CUDA {sorted(device)}",
        )
    for name, expected in host.items():
        np.testing.assert_allclose(
            to_numpy(device[name]),
            to_numpy(expected),
            rtol=rtol,
            atol=atol,
            err_msg=f"{kernel_name}: {name} differs between the host and CUDA kernels",
        )


def assert_kernels_agree(
    kernel: Kernel,
    make_args: Callable[[str, int], Sequence[Any]],
    *,
    n_threads: int | Sequence[int] | Callable[[tuple[Any, ...]], Any] | None = None,
    grid: int | Sequence[int] | None = None,
    block: int | Sequence[int] | None = None,
    rtol: float = 1e-12,
    atol: float = 0.0,
    n_calls: int = 1,
    outputs: Sequence[int | str] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Check that the host and CUDA versions of a kernel compute the same.

    For each backend, ``"numpy"`` then ``"cupy"``, the backend is activated,
    the arguments are built with ``make_args(backend, seed)``, the kernel is
    called `n_calls` times, and the arrays among the arguments are collected.
    The CUDA results are copied to the host and compared with
    ``numpy.testing.assert_allclose``.

    Parameters
    ----------
    kernel : Kernel
        A :class:`~cunumpy.kernels.Kernel` with a CUDA version.
    make_args : callable
        ``make_args(backend, seed)`` returns the positional arguments of the
        kernel (a tuple or list). It runs with the backend active, so arrays
        made through cunumpy land on it. NumPy and CuPy generators differ for
        one seed: build random data with ``numpy.random.default_rng(seed)``
        and convert it with :func:`~cunumpy.to_cunumpy`.
    n_threads, grid, block : int or tuple of int, optional
        Launch configuration of the CUDA kernel, see
        :meth:`CudaKernel.__call__ <cunumpy.kernels.CudaKernel.__call__>`.
        `n_threads` may also be a function of the argument tuple, e.g.
        ``lambda args: args[0].shape[0]``. Without `n_threads` and `grid`,
        the kernel's ``n_threads_from`` is used.
    rtol, atol : float, optional
        Tolerances of ``numpy.testing.assert_allclose``.
    n_calls : int, optional
        How many times the kernel is called on each backend (e.g. to test a
        kernel that accumulates).
    outputs : sequence of int or str, optional
        The arguments to compare: indices (negative from the end) or
        parameter names; ``"markers.positions"`` compares only that field of
        a struct or argument object. Default: the host kernel's ``outputs``,
        else every argument. Arrays inside tuple, list, dict or object
        arguments are compared one level deep (plus container attributes).
    seed : int, optional
        Passed to `make_args` on both backends.

    Returns
    -------
    dict of str to numpy.ndarray
        The host-call arrays by argument name (``"argument 0"``,
        ``"argument 1.x"``), for further checks.

    Raises
    ------
    ValueError
        If `kernel` has no CUDA version or `n_calls` is less than 1.
    TypeError
        If `kernel` is not a ``Kernel``, or no launch size is given or
        configured.
    AssertionError
        If an array differs; the message names the argument.

    Notes
    -----
    Skipped with ``pytest.skip`` without a GPU or on the fake CuPy; with
    ``CUNUMPY_REQUIRE_CUDA=1`` it fails instead. See :doc:`/kernels/testing`.

    Examples
    --------
    >>> from cunumpy.kernel_testing import assert_kernels_agree
    >>> def make_args(backend, seed):
    ...     x = xp.to_cunumpy(np.random.default_rng(seed).random(1000))
    ...     return (x, 2.0, x.size)
    >>> assert_kernels_agree(catalog["scale"], make_args, n_threads=1000)  # doctest: +SKIP
    """
    if not isinstance(kernel, Kernel):
        raise TypeError(f"expected a Kernel, got {type(kernel).__name__}")
    if not kernel.has_cuda:
        raise ValueError(f"kernel {kernel.name!r} has no CUDA version")
    if n_threads is None and grid is None and kernel.cuda_kernel.n_threads_from is None:
        raise TypeError("n_threads (or grid) is required to launch the CUDA kernel")
    if n_calls < 1:
        raise ValueError(f"n_calls must be at least 1, got {n_calls}")
    if outputs is None:
        outputs = kernel.host_kernel.outputs
    if fake_cupy_active():
        _skip_or_fail(FAKE_SKIP_REASON)
    if not cupy_available():
        _skip_or_fail(SKIP_REASON)

    parameters = kernel.host_parameters()
    results = {}
    for backend in ("numpy", "cupy"):
        with use_backend(backend):
            if get_backend() != backend:  # pragma: no cover - cupy_available() lied
                raise RuntimeError(f"could not activate the {backend} backend")
            args = tuple(make_args(backend, seed))
            launch = n_threads(args) if callable(n_threads) else n_threads
            for _ in range(n_calls):
                kernel(*args, n_threads=launch, grid=grid, block=block)
            results[backend] = _collect_arrays(args, outputs, parameters)

    host = {name: to_numpy(a) for name, a in results["numpy"].items()}
    _compare_results(host, results["cupy"], rtol, atol, kernel.name)
    return host


# ---------------------------------------------------------------------------
# parity tests from <name>_test_args.py modules
# ---------------------------------------------------------------------------

#: Module-level names a ``<name>_test_args.py`` module may define, and the
#: keyword of :func:`assert_kernels_agree` each one sets.
TEST_ARGS_SETTINGS = {
    "N_THREADS": "n_threads",
    "GRID": "grid",
    "BLOCK": "block",
    "RTOL": "rtol",
    "ATOL": "atol",
    "N_CALLS": "n_calls",
    "OUTPUTS": "outputs",
    "SEED": "seed",
}


def parity_cases(catalog: Any) -> list[Any]:
    """Return the kernels of a catalog with a CUDA version, as pytest parameters.

    A kernel without a test-arguments module (``<name>_test_args.py`` in its
    folder, see :attr:`Kernel.test_args_module
    <cunumpy.kernels.Kernel.test_args_module>`) is marked ``skip`` with a
    reason naming the missing file, so the report shows which kernels still
    lack a parity test.

    Parameters
    ----------
    catalog : KernelCatalog
        A :class:`~cunumpy.kernels.KernelCatalog` (anything with
        ``parity_cases()``).

    Returns
    -------
    list of pytest.param
        One ``pytest.param(kernel, id=name)`` per kernel of
        ``catalog.parity_cases()``.

    See Also
    --------
    check_parity : The test to run for each case.

    Examples
    --------
    >>> from cunumpy.kernel_testing import check_parity, parity_cases
    >>> @pytest.mark.parametrize("kernel", parity_cases(catalog))  # doctest: +SKIP
    ... def test_parity(kernel):
    ...     check_parity(kernel)
    """
    pytest = _pytest()
    cases = []
    for name, kernel in catalog.parity_cases():
        marks = ()
        if kernel.test_args_module is None:
            marks = (
                pytest.mark.skip(
                    reason=f"no test arguments for {name!r}: add {name}_test_args.py "
                    "with make_args(backend, seed) and N_THREADS to its folder",
                ),
            )
        cases.append(pytest.param(kernel, id=name, marks=marks))
    return cases


def check_parity(kernel: Kernel, **overrides: Any) -> dict[str, np.ndarray]:
    """Run :func:`assert_kernels_agree` with the kernel's test-arguments module.

    The module is ``<name>_test_args.py`` in the kernel's folder (see
    :meth:`KernelCatalog.from_package
    <cunumpy.kernels.KernelCatalog.from_package>`). It defines
    ``make_args(backend, seed)`` and, optionally, the settings of
    ``TEST_ARGS_SETTINGS`` as module-level names (``N_THREADS``, ``GRID``,
    ``BLOCK``, ``RTOL``, ``ATOL``, ``N_CALLS``, ``OUTPUTS``, ``SEED``).

    Parameters
    ----------
    kernel : Kernel
        A :class:`~cunumpy.kernels.Kernel` with a CUDA version.
    **overrides
        Keyword arguments of :func:`assert_kernels_agree` that override the
        module's settings.

    Returns
    -------
    dict of str to numpy.ndarray
        The host arrays, as :func:`assert_kernels_agree` returns them.

    Raises
    ------
    ValueError
        If the kernel has no test-arguments module.
    TypeError
        If the module has no callable ``make_args``.

    Examples
    --------
    >>> from cunumpy.kernel_testing import check_parity
    >>> check_parity(catalog["push"], rtol=1e-10)  # doctest: +SKIP
    """
    module = kernel.test_args
    if module is None:
        raise ValueError(
            f"kernel {kernel.name!r} has no test-arguments module: add "
            f"{kernel.name}_test_args.py with make_args(backend, seed) to its folder",
        )
    make_args = getattr(module, "make_args", None)
    if not callable(make_args):
        raise TypeError(f"{module.__name__} must define make_args(backend, seed)")
    settings = {
        keyword: getattr(module, name)
        for name, keyword in TEST_ARGS_SETTINGS.items()
        if hasattr(module, name)
    }
    settings.update(overrides)
    return assert_kernels_agree(kernel, make_args, **settings)


# ---------------------------------------------------------------------------
# device_function_kernel
# ---------------------------------------------------------------------------

_PROTOTYPE = re.compile(
    r"^\s*(?P<result>.+?)\s*\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*;?\s*$",
    re.DOTALL,
)
_FUNCTION_QUALIFIERS = {"__device__", "__host__", "__forceinline__", "inline", "static"}


def _parse_prototype(
    signature: str,
    structs: dict[str, Any] | None = None,
) -> tuple[CudaParameter | None, str, list[tuple[str, CudaParameter]]]:
    """Parse a C prototype into (result or None, name, [(text, parameter)])."""
    match = _PROTOTYPE.match(_strip_comments(signature))
    if match is None:
        raise ValueError(f"cannot parse the function prototype {signature!r}")
    name = match.group("name")
    result_words = [
        w for w in match.group("result").split() if w not in _FUNCTION_QUALIFIERS
    ]
    result_text = " ".join(result_words)
    if result_text == "void":
        result = None
    else:
        try:
            result = _parse_parameter(f"{result_text} result")
        except ValueError:
            raise ValueError(
                f"unsupported return type {result_text!r} of {name!r}: a scalar "
                "type (or void) is required",
            ) from None
        if result.pointer:
            raise ValueError(
                f"{name!r} returns a pointer; only scalar results can be collected",
            )
    params_text = match.group("params").strip()
    params = []
    if params_text not in ("", "void"):
        for text in _split_top_level(params_text):
            text = text.strip()
            by_reference = "&" in text
            param = _parse_parameter(text.replace("&", " "), structs or None)
            if by_reference and param.struct is None:
                raise ValueError(
                    f"unsupported parameter {text!r} of {name!r}: only structs can "
                    "be passed by reference",
                )
            params.append((text, param))
    return result, name, params


def device_function_kernel(
    header_source: str,
    signature: str,
    *,
    name: str | None = None,
    includes: Sequence[str] = (),
    n_threads_param: str = "n",
    out_param: str = "out",
    **kwargs: Any,
) -> CudaKernel:
    """Wrap a ``__device__`` function in an elementwise kernel, for testing.

    Generates an ``extern "C" __global__`` kernel that calls the device
    function once per thread, so that a device helper (e.g. a B-spline
    evaluation) can be run from Python on many inputs and compared with its
    host version. See :doc:`/kernels/testing`.

    Parameters
    ----------
    header_source : str
        CUDA source defining the device function (typically the content of
        its header).
    signature : str
        The C prototype of the device function, e.g.
        ``"int find_span(const double* t, int p, double eta)"``. Pointer,
        scalar and struct parameters (by value or ``const`` reference, for
        the structs in ``structs``) and a scalar or ``void`` return type are
        supported.
    name : str, optional
        Name of the generated kernel; ``"<function>_kernel"`` by default.
    includes : sequence of str, optional
        Headers to ``#include`` before `header_source`: ``"bsplines.cuh"``
        with quotes, ``"<cupy/complex.cuh>"`` with angle brackets. Pass
        ``include_dirs`` for the directories.
    n_threads_param : str, optional
        Name of the last generated parameter, the number of elements.
    out_param : str, optional
        Name of the generated output array parameter.
    **kwargs
        Passed on to :class:`~cunumpy.kernels.CudaKernel`, e.g.
        ``include_dirs``, ``options``, ``block_size`` or ``structs`` (the
        :class:`~cunumpy.arguments.CudaStruct` types of struct parameters).

    Returns
    -------
    CudaKernel
        The wrapper kernel. Its parameters are those of the function, in
        order, then ``R* out`` (unless the function returns ``void``) and
        ``int n``. Pointer and struct parameters are passed unchanged to every
        call; a scalar parameter ``T x`` becomes an array ``const T* x`` and
        thread ``i`` calls the function with ``x[i]``, storing the result in
        ``out[i]``. Threads ``i >= n`` do nothing. Launch it with
        ``n_threads=n``.

    Raises
    ------
    ValueError
        If the prototype cannot be parsed, the return type is not a scalar or
        ``void``, a parameter type is unsupported, or a parameter is named
        like `out_param` or `n_threads_param`.

    Examples
    --------
    >>> from cunumpy.kernel_testing import device_function_kernel, emulate_cuda_kernel
    >>> sq = device_function_kernel(
    ...     "__device__ double sq(double x) { return x * x; }", "double sq(double x)"
    ... )
    >>> [p.name for p in sq.signature]
    ['x', 'out', 'n']
    >>> x, out = np.arange(4.0), np.empty(4)
    >>> emulate_cuda_kernel(sq, x, out, 4, n_threads=4)  # on the CPU, needs a C++ compiler
    >>> out
    array([0., 1., 4., 9.])
    """
    structs = {struct.name: struct for struct in kwargs.get("structs", ())}
    result, function, params = _parse_prototype(signature, structs)
    if name is None:
        name = f"{function}_kernel"
    reserved = {out_param, n_threads_param}
    for _, param in params:
        if param.name in reserved:
            raise ValueError(
                f"the parameter {param.name!r} of {function!r} clashes with the "
                f"generated parameter of that name; pass another out_param or "
                "n_threads_param",
            )

    wrapper_params, call_args = [], []
    for text, param in params:
        if param.pointer:
            wrapper_params.append(text)
            call_args.append(param.name)
        elif param.struct is not None:
            wrapper_params.append(f"{param.ctype} {param.name}")  # by value
            call_args.append(param.name)
        else:
            wrapper_params.append(f"const {param.ctype}* {param.name}")
            call_args.append(f"{param.name}[i]")
    call = f"{function}({', '.join(call_args)})"
    if result is None:
        body = f"{call};"
    else:
        wrapper_params.append(f"{result.ctype}* {out_param}")
        body = f"{out_param}[i] = {call};"
    wrapper_params.append(f"int {n_threads_param}")

    include_lines = "".join(
        f"#include {h}\n" if h.startswith("<") else f'#include "{h}"\n'
        for h in includes
    )
    source = (
        f"{include_lines}{header_source}\n\n"
        f'extern "C" __global__ void {name}({", ".join(wrapper_params)}) {{\n'
        f"    int i = blockDim.x * blockIdx.x + threadIdx.x;\n"
        f"    if (i >= {n_threads_param}) return;\n"
        f"    {body}\n"
        f"}}\n"
    )
    return CudaKernel(source, name, **kwargs)
