"""Helpers for testing host/CUDA kernel pairs with pytest.

A code base that ports its kernels to CUDA one by one needs the same test for
every kernel: build the arguments on both backends, run the host kernel and the
CUDA kernel, and compare what they wrote. This module provides that test
(:func:`assert_kernels_agree`), the pytest markers to parametrize tests over
the backends (:data:`BACKENDS`, :data:`requires_cupy`, the :func:`backend`
fixture), :func:`device_function_kernel`, which wraps a ``__device__``
function in an elementwise ``__global__`` kernel so that device helpers can be
tested from Python without a hand-written test kernel, and
:func:`emulate_cuda_kernel` (from the private module ``cunumpy._emulation``), which runs a CUDA
kernel on the CPU, one thread after another, so that its arithmetic can be
checked against the host kernel in CI without a GPU. Without a GPU, the CuPy
code paths of a program (argument objects, conversions, backend branches) can
still run on the fake CuPy of :mod:`cunumpy._fake_cupy` (:func:`install_fake_cupy`,
or ``CUNUMPY_FAKE_CUPY=1``); :func:`fake_cupy_active` tells whether it is in
use, and ``requires_cupy`` skips the tests that launch kernels then.
:func:`fake_cupy_session` runs a whole CuPy-backend program on it (launches and
compilation emulated, see :func:`emulated_launches`), :data:`requires_device_backend`
skips tests that need a GPU or the fake CuPy, and :func:`run_in_fake_cupy_subprocess`
runs code in a child process on the fake CuPy from a test process that cannot
switch to it.

A catalog's parity tests need no code per kernel when each kernel folder
holds ``<name>_test_args.py`` with ``make_args(backend, seed)`` (and
``N_THREADS``): :func:`parity_cases` and :func:`check_parity` drive
:func:`assert_kernels_agree` from these modules.

The module imports pytest only when one of its pytest objects is used, so it
can be imported (e.g. for :func:`device_function_kernel`) without pytest, and
``import cunumpy`` never imports pytest.

Examples
--------
One parametrised test covers every kernel of a catalog that has a CUDA
version::

    import pytest
    from cunumpy.kernel_testing import assert_kernels_agree

    from my_kernels import catalog


    def make_args(backend, seed):
        rng = xp.rng.get_rng(seed)  # the backend is active: arrays land on it
        x = rng.random(1000)
        return (x, 2.0, x.size)


    @pytest.mark.parametrize("name, kernel", catalog.parity_cases())
    def test_parity(name, kernel):
        assert_kernels_agree(kernel, make_args, n_threads=1000)
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
    """Whether the fake CuPy (:mod:`cunumpy._fake_cupy`) stands in for CuPy."""
    return _fake_cupy.is_active()


def install_fake_cupy() -> Any:
    """Install the fake CuPy for this process; see :mod:`cunumpy._fake_cupy`.

    Call it before the first backend use (e.g. at the top of ``conftest.py``),
    or set ``CUNUMPY_FAKE_CUPY=1`` in the environment instead. Returns the
    fake ``cupy`` module.
    """
    return _fake_cupy.install()


def _can_launch() -> bool:
    """Whether CUDA kernels can run: a functional CuPy that is not the fake."""
    return cupy_available() and not fake_cupy_active()


def cuda_required() -> bool:
    """Whether ``CUNUMPY_REQUIRE_CUDA`` demands a real GPU (the CI guard).

    Then tests that need a GPU fail instead of being skipped where there is
    none (:data:`requires_cupy`, :func:`assert_kernels_agree`, the ``cupy``
    parameter of :func:`backend`), so that a CI job on a GPU machine cannot
    pass silently because CuPy or the driver is broken.
    """
    return os.environ.get("CUNUMPY_REQUIRE_CUDA", "").lower() in {"1", "true", "yes"}


def _skip_or_fail(reason: str) -> None:
    """``pytest.skip``, or ``pytest.fail`` if a GPU is required."""
    if cuda_required():
        _pytest().fail(f"CUNUMPY_REQUIRE_CUDA is set, but {reason}", pytrace=False)
    _pytest().skip(reason)


def cuda_gate() -> bool:
    """Condition of ``requires_cupy`` under ``CUNUMPY_REQUIRE_CUDA``: fail, or False."""
    if not _can_launch():
        reason = FAKE_SKIP_REASON if fake_cupy_active() else SKIP_REASON
        _pytest().fail(f"CUNUMPY_REQUIRE_CUDA is set, but {reason}", pytrace=False)
    return False


def device_backend_available() -> bool:
    """Whether a CuPy-backend program can run here: a GPU, or the fake CuPy.

    Unlike :data:`requires_cupy` (CUDA kernels can be launched), this is also
    True on the fake CuPy, where launches only work inside
    :func:`emulated_launches` (see :func:`fake_cupy_session`). The marker
    :data:`requires_device_backend` skips tests that need it.
    """
    return fake_cupy_active() or cupy_available()


def device_backend_gate() -> bool:
    """Condition of ``requires_device_backend`` under ``CUNUMPY_REQUIRE_CUDA``."""
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
    and compilation in the block (:func:`emulated_launches`, which takes
    `compiler` and `options`)::

        with fake_cupy_session():
            sim.run()  # kernels compiled up front and launched, all on the CPU

    Raises
    ------
    RuntimeError
        If the fake CuPy is not active (:func:`install_fake_cupy`,
        ``CUNUMPY_FAKE_CUPY=1``; or use :func:`run_in_fake_cupy_subprocess`).
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
    """Run `code` in a serial child Python process on the fake CuPy; fail the test if it fails.

    The fake CuPy must be installed before anything imports cunumpy, so a test
    process that already uses cunumpy cannot switch to it; the child starts
    with ``CUNUMPY_FAKE_CUPY=1``. It runs ``python -X faulthandler -c code``
    (a crash prints the Python traceback) with ``OMP_NUM_THREADS=1`` and
    without the variables of an MPI launcher, so it does not join the MPI job
    of the parent (``MAYBEMPI=0``). Under MPI only rank 0 starts the child and
    the other ranks skip the test: concurrent children are not needed for a
    serial check, and have crashed external libraries.

    Parameters
    ----------
    code : str
        Python source to run.
    env : Mapping[str, str] | None
        Additional environment variables for the child (e.g. ``PYTHONPATH``).
    timeout : float | None
        Seconds after which the child is killed and the test fails.

    Returns
    -------
    subprocess.CompletedProcess
        The finished child (exit code 0), with its stdout and stderr.
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
    """The argument index and the field filter (or None) of an `outputs` entry."""
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
    """The arrays among `args` (or among the arguments `outputs`), by name.

    An argument that is an array is named ``"argument <i>"``; arrays found one
    level deep, in a tuple, list or dict argument or in the attributes of an
    argument object (e.g. a ``CudaArguments`` object), are named
    ``"argument <i>[<j>]"`` or ``"argument <i>.<attribute>"``, and arrays in a
    container attribute of an object ``"argument <i>.<attribute>[<j>]"``. A
    :class:`~cunumpy.arguments.CudaStructArguments` object or a struct value is read
    through its struct fields, ``"argument <i>.<field>"``, so that its arrays
    get the names of the attributes of the host argument object it mirrors,
    also when the fields are properties.

    An entry of `outputs` is an argument index, or the name of a parameter (in
    `parameters`), optionally followed by ``.<field>`` to compare only that
    field (attribute) of a struct or argument object, e.g. ``"markers.positions"``.
    """
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
    """Compare the arrays of `device` with those of `host`, by name.

    Raises
    ------
    AssertionError
        If the two do not hold the same names, or an array differs (the
        message names the argument).
    """
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
    """Check that the host and CUDA versions of `kernel` compute the same.

    For each backend, ``"numpy"`` then ``"cupy"``, the backend is activated
    with :func:`~cunumpy.use_backend`, the arguments are built with
    ``make_args(backend, seed)``, the kernel is called `n_calls` times, and
    the arrays among the arguments are collected. The arrays written by the
    CUDA kernel are then copied to the host and compared with those of the host
    kernel using ``numpy.testing.assert_allclose``.

    Parameters
    ----------
    kernel : Kernel
        A kernel with a CUDA version (``kernel.has_cuda``).
    make_args : Callable[[str, int], Sequence]
        ``make_args(backend, seed)`` returns the positional arguments of the
        kernel, as a tuple or list. It is called with the backend (``"numpy"``
        or ``"cupy"``) active, so arrays created through ``cunumpy`` (e.g. with
        ``xp.zeros`` or ``xp.rng.get_rng(seed)``) land on that backend; NumPy and
        CuPy random generators do not produce the same sequence from one seed,
        so build random data on the host with ``numpy.random.default_rng(seed)``
        and convert it with :func:`~cunumpy.to_cunumpy`. Kernels take positional
        arguments only.
    n_threads, grid, block
        Launch configuration of the CUDA kernel, see
        :meth:`CudaKernel.__call__ <cunumpy.kernels.CudaKernel.__call__>`. `n_threads`
        may also be a function of the tuple of arguments, e.g.
        ``lambda args: args[0].shape[0]``. Omitted sizes use the CUDA kernel's
        shape-based default or its configured ``n_threads_from``; explicit sizes
        are required when inference is disabled.
    rtol, atol : float
        Tolerances of ``numpy.testing.assert_allclose``.
    n_calls : int
        How many times the kernel is called on each backend (e.g. to test a
        kernel that accumulates).
    outputs : Sequence[int | str] | None
        The arguments to compare, like ``PyccelKernel(outputs=...)``: indices
        (negative indices count from the end) or parameter names. A name with
        ``.<field>`` (``"markers.positions"``) compares only that field of a
        struct or argument object, leaving the other fields (e.g. buffers the
        two kernels fill differently) out. By default the ``outputs``
        declared by the host kernel are used, and if it declares none, every
        argument. An argument that is an array is compared; for a tuple, list,
        dict or object argument (e.g. a ``CudaArguments`` object), the arrays
        it holds are compared (one level deep, plus arrays in a container
        attribute of an object).
    seed : int
        Passed to `make_args` on both backends.

    Returns
    -------
    dict[str, numpy.ndarray]
        The arrays of the host call by argument name (``"argument 0"``,
        ``"argument 1.x"``), for further checks.

    Raises
    ------
    ValueError
        If `kernel` has no CUDA version.
    AssertionError
        If an array differs; the message names the argument.

    Notes
    -----
    The test is skipped with ``pytest.skip`` if CuPy or a GPU is not available,
    or if the fake CuPy is active; with ``CUNUMPY_REQUIRE_CUDA=1`` it fails instead.
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
    """The kernels of a catalog with a CUDA version, as pytest parameters.

    One ``pytest.param(kernel, id=name)`` per kernel of
    ``catalog.parity_cases()``. A kernel without a test-arguments module
    (:attr:`Kernel.test_args_module <cunumpy.kernels.Kernel.test_args_module>`, from
    ``<name>_test_args.py`` in its folder) is marked ``skip`` with a reason
    naming the missing file, so the report shows which kernels still lack
    their parity test::

        @pytest.mark.parametrize("kernel", parity_cases(catalog))
        def test_parity(kernel):
            check_parity(kernel)
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

    The module (``<name>_test_args.py`` in the kernel's folder, see
    :meth:`KernelCatalog.from_package <cunumpy.kernels.KernelCatalog.from_package>`)
    defines ``make_args(backend, seed)`` and, as module-level names, the
    launch and comparison settings of :data:`TEST_ARGS_SETTINGS`:
    ``N_THREADS`` (an integer, a tuple, or a function of the argument tuple),
    or ``GRID``, plus optionally ``BLOCK``, ``RTOL``, ``ATOL``, ``N_CALLS``,
    ``OUTPUTS`` and ``SEED``. Keyword arguments override them.

    Returns
    -------
    dict[str, numpy.ndarray]
        The host arrays, as :func:`assert_kernels_agree` returns them.

    Raises
    ------
    ValueError
        If the kernel has no test-arguments module.
    TypeError
        If the module has no callable ``make_args``.
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
    """Parse a C function prototype into (result, name, [(text, parameter)]).

    The result is None for a ``void`` function; each parameter is its original
    text together with its parsed form. A struct of `structs` may be taken by
    value or by (const) reference.
    """
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

    Generates an ``extern "C" __global__`` kernel that calls the device function
    once per thread, so that a device helper (e.g. a B-spline evaluation) can be
    run from Python on many inputs at once and compared with its host version.

    Parameters
    ----------
    header_source : str
        CUDA source defining the device function (typically the content of the
        header, or ``#include`` directives, see `includes`).
    signature : str
        The C prototype of the device function, e.g.
        ``"int find_span(const double* t, int p, double eta)"``. Pointer
        parameters, scalar parameters of the types :class:`CudaKernel` supports,
        struct parameters (by value or by ``const`` reference, for the structs
        passed in ``structs``) and a scalar or ``void`` return type are
        supported.
    name : str | None
        Name of the generated kernel; ``"<function>_kernel"`` by default.
    includes : Sequence[str]
        Headers to include before `header_source`: ``"bsplines.cuh"`` becomes
        ``#include "bsplines.cuh"``, ``"<cupy/complex.cuh>"`` is included with
        angle brackets. Pass ``include_dirs`` for the directories.
    n_threads_param : str
        Name of the generated kernel's last parameter, the number of elements.
    out_param : str
        Name of the generated output array parameter.
    **kwargs
        Passed on to :class:`CudaKernel`, e.g. ``include_dirs``, ``options``,
        ``block_size`` or ``structs`` (the :class:`~cunumpy.arguments.CudaStruct` types
        of struct parameters, whose definitions `header_source` or the
        `includes` must provide).

    Returns
    -------
    CudaKernel
        The wrapper kernel. Its parameters are those of the device function, in
        order, followed by the output array (unless the function returns
        ``void``) and the number of elements:

        * a pointer parameter stays as it is and is passed through unchanged to
          every call (an array shared by all threads);
        * a struct parameter (``DomainArgs d`` or ``const DomainArgs& d``) is
          taken by value and passed through unchanged to every call (pass a
          :class:`~cunumpy.arguments.CudaStructArguments` object or a packed value);
        * a scalar parameter ``T x`` becomes a device array ``const T* x`` of
          length ``n``, and thread ``i`` calls the function with ``x[i]``;
        * the return value of thread ``i`` is stored in ``out[i]``, an array
          ``R* out`` of length ``n`` where ``R`` is the return type;
        * ``int n`` is the number of elements (threads ``i >= n`` do nothing).

        Launch it with ``n_threads=n``.

    Raises
    ------
    ValueError
        If the prototype cannot be parsed, the return type is not a scalar type
        or ``void``, a parameter has an unsupported type, or a parameter is
        named like `out_param` or `n_threads_param`.

    Examples
    --------
    >>> from cunumpy.kernel_testing import device_function_kernel
    >>> sq = device_function_kernel(
    ...     "__device__ double sq(double x) { return x * x; }",
    ...     "double sq(double x)",
    ... )
    >>> sq.signature  # (const double* x, double* out, int n)
    >>> x = cp.arange(10.0); out = cp.empty(10)  # doctest: +SKIP
    >>> sq(x, out, 10, n_threads=10)  # doctest: +SKIP
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
