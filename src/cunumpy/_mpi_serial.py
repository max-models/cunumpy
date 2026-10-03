"""MPI only when launched under MPI, and a serial stand-in otherwise (see :mod:`cunumpy.mpi`).

Importing ``mpi4py.MPI`` calls ``MPI_Init``, which can take close to a second
and makes every collective cost something, even on one process. A plain
``python script.py`` should therefore not touch MPI, even when mpi4py is
installed. :func:`launched_under_mpi` tells, from the environment the launcher
sets up and without importing mpi4py, whether the process was started by
``mpirun``/``mpiexec``/``srun``; :func:`get_mpi` returns ``mpi4py.MPI`` then,
and :class:`SerialMPI` otherwise, so that one code path serves both::

    MPI = xp.mpi.get_mpi()
    comm = MPI.COMM_WORLD
    comm.Allreduce(MPI.IN_PLACE, rho, op=MPI.SUM)   # a no-op on one process
    total = comm.allreduce(local_total)             # local_total itself

:class:`SerialComm` behaves like a communicator of size 1: collectives return
or copy what they would on one rank (never ``None`` in place of a value), and
methods it does not implement raise ``AttributeError`` instead of silently
doing nothing.

This module does not depend on the rest of cunumpy (arrays are handled by duck
typing), so that it could become a package of its own.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import warnings
from types import MappingProxyType
from typing import Any

# Per-rank variables exported by the process managers behind common launchers.
# Each is set only for processes started *by* a launcher. SLURM_PROCID is
# deliberately absent: it is also set for the batch script of a plain `sbatch`
# job, which is not an MPI launch (`srun` exports the PMI/PMIx variables).
_LAUNCHER_VARIABLES = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI (and derivatives)
    "PMI_RANK",  # MPICH, Intel MPI, MS-MPI, Cray, srun --mpi=pmi2
    "PMIX_RANK",  # PMIx: srun --mpi=pmix, Open MPI 5
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Hydra (mpiexec.hydra)
    "ALPS_APP_PE",  # Cray ALPS aprun
    "PALS_RANKID",  # Cray PALS
)

# Node-local rank of the process, as exported by common MPI launchers. They are
# set before ``MPI_Init``, so the device can be chosen before MPI starts.
_LOCAL_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_LOCAL_RANK",  # Open MPI
    "MV2_COMM_WORLD_LOCAL_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Intel MPI, MPICH (Hydra)
    "PMI_LOCAL_RANK",  # MPICH / PMI
    "PALS_LOCAL_RANKID",  # Cray PALS
    "SLURM_LOCALID",  # Slurm (srun)
    "LOCAL_RANK",  # torchrun and others
)

#: Environment variable that forces the decision of :func:`launched_under_mpi`
#: (``1``/``true``/``yes``/``on`` or ``0``/``false``/``no``/``off``).
OVERRIDE_VARIABLE = "CUNUMPY_MPI"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _env_flag(name: str) -> bool | None:
    """The boolean value of the environment variable `name`, or None if unset/unknown."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def local_rank() -> int:
    """Rank of this process within its node, from the MPI launcher's environment.

    Reads the node-local rank that common launchers export (Open MPI, MVAPICH2,
    Intel MPI/MPICH, PMI, Cray PALS, Slurm, ``LOCAL_RANK``). These variables are
    set before ``MPI_Init``, so this works before MPI is initialized, and
    without importing ``mpi4py``. Returns 0 if none is set (e.g. a serial run).
    """
    for variable in _LOCAL_RANK_VARIABLES:
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def launched_under_mpi() -> bool:
    """Whether this process was started by an MPI launcher (without importing mpi4py).

    True if a per-rank variable of a common launcher is set (Open MPI, MPICH,
    Intel MPI, PMIx/``srun``, MVAPICH2, Hydra, Cray ALPS/PALS), or if mpi4py is
    already imported and MPI initialized (then using it costs nothing more).
    ``CUNUMPY_MPI=1``/``0`` overrides the detection, e.g. for a launcher whose
    variables are not known here.
    """
    override = _env_flag(OVERRIDE_VARIABLE)
    if override is not None:
        return override
    if any(variable in os.environ for variable in _LAUNCHER_VARIABLES):
        return True
    # only look at mpi4py if the application imported it: importing it here
    # is what must be avoided
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None:
        try:
            return bool(mpi.Is_initialized())
        except AttributeError:
            return False
    return False


_AUTO_MPI: Any = None  # the result of get_mpi(None), decided once


def get_mpi(use_mpi: bool | None = None) -> Any:
    """``mpi4py.MPI`` for an MPI run, else the serial stand-in (a :class:`SerialMPI`).

    Parameters
    ----------
    use_mpi : bool | None
        ``None`` (the default) decides with :func:`launched_under_mpi`, once
        per process; ``True`` imports mpi4py (``ImportError`` if it is not
        installed); ``False`` returns the stand-in without importing it.

    Returns
    -------
    module or SerialMPI
        ``mpi4py.MPI``, or the one :class:`SerialMPI` object, which has the
        attributes of the module that a serial run needs.
        ``isinstance(MPI, xp.mpi.SerialMPI)`` tells which one it is.

    Warns
    -----
    RuntimeWarning
        Launched under MPI but mpi4py is not installed: every rank then runs
        as if it were alone, with the same rank 0.
    """
    global _AUTO_MPI
    if use_mpi is True:
        from mpi4py import MPI

        return MPI
    if use_mpi is False:
        return _SERIAL_MPI
    if _AUTO_MPI is None:
        if launched_under_mpi():
            try:
                from mpi4py import MPI
            except ImportError:
                warnings.warn(
                    "launched under an MPI launcher, but mpi4py is not installed: "
                    "every process runs serially as rank 0 of 1 (pip install mpi4py)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                MPI = _SERIAL_MPI
            _AUTO_MPI = MPI
        else:
            _AUTO_MPI = _SERIAL_MPI
    return _AUTO_MPI


class _Constant:
    """A named placeholder for an MPI constant (an op, a datatype, ``IN_PLACE``, ...)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"SerialMPI.{self.name}"


class _Datatype(_Constant):
    """Placeholder for an MPI datatype (``isinstance(t, MPI.Datatype)`` holds)."""

    __slots__ = ()


class _Op(_Constant):
    """Placeholder for an MPI reduction operation."""

    __slots__ = ()


class _Null(_Constant):
    """A null handle (``COMM_NULL``, ``DATATYPE_NULL``, ...): false, like mpi4py's."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False


_IN_PLACE = _Constant("IN_PLACE")
_COMM_NULL = _Null("COMM_NULL")


def _buffer(spec: Any) -> Any:
    """The array of an mpi4py buffer specification (``buf`` or ``[buf, ...]``)."""
    if isinstance(spec, (list, tuple)):
        return spec[0]
    return spec


def _displacement(spec: Any) -> int:
    """The displacement of rank 0 in a vector buffer spec ``[buf, counts, displs, type]``."""
    if isinstance(spec, (list, tuple)) and len(spec) >= 3:
        displs = spec[2]
        if displs is not None and not isinstance(displs, _Constant):
            return int(displs[0])
    return 0


def _copy(source: Any, target: Any, offset: int = 0) -> None:
    """Copy the elements of the array `source` into `target`, starting at `offset`.

    Both are flattened (C order); NumPy and CuPy arrays mix (a device source is
    copied to the host with ``.get()`` for a host target).
    """
    if source is _IN_PLACE or source is None or target is None:
        return
    if hasattr(source, "get") and not hasattr(target, "get"):
        source = source.get()
    if not target.flags.c_contiguous:
        raise ValueError("the receive buffer must be C-contiguous")
    flat = target.reshape(-1)
    source = source.reshape(-1)
    if offset + source.size > flat.size:
        raise ValueError(
            f"receive buffer too small: {flat.size} elements for {source.size} "
            f"at offset {offset}"
        )
    flat[offset : offset + source.size] = source


def _check_rank(rank: int, what: str) -> None:
    if rank not in (0, SerialMPI.PROC_NULL, SerialMPI.ANY_SOURCE):
        raise ValueError(f"{what}={rank}: a serial communicator has only rank 0")


class SerialRequest:
    """A completed request, returned by the non-blocking calls of :class:`SerialComm`."""

    def __init__(self, result: Any = None) -> None:
        self._result = result

    def Wait(self, status: Any = None) -> None:
        return None

    def Test(self, status: Any = None) -> bool:
        return True

    def wait(self, status: Any = None) -> Any:
        return self._result

    def test(self, status: Any = None) -> tuple[bool, Any]:
        return True, self._result

    def Free(self) -> None:
        return None

    def Cancel(self) -> None:
        return None

    @staticmethod
    def Waitall(requests: Any, statuses: Any = None) -> None:
        return None

    @staticmethod
    def waitall(requests: Any, statuses: Any = None) -> list[Any]:
        return [request.wait() for request in requests]

    @staticmethod
    def Testall(requests: Any, statuses: Any = None) -> bool:
        return True

    @staticmethod
    def Waitany(requests: Any, status: Any = None) -> int:
        return 0 if requests else SerialMPI.UNDEFINED


class SerialPrequest(SerialRequest):
    """A persistent request (``Send_init``/``Recv_init``), for ``Startall``/``Waitall``."""

    def Start(self) -> None:
        return None

    @staticmethod
    def Startall(requests: Any) -> None:
        return None


class SerialComm:
    """A communicator of size 1, with the mpi4py ``Comm`` methods a serial run needs.

    Collectives return (object methods) or copy (buffer methods) what they
    would on one rank: ``allreduce(x)`` is ``x``, ``gather(x)`` is ``[x]``,
    ``Allreduce(send, recv)`` copies `send` into `recv` (nothing with
    ``IN_PLACE``), ``Bcast`` does nothing. Point-to-point calls are only
    supported to and from rank 0 itself (``sendrecv``, ``Sendrecv``) or
    ``PROC_NULL``. Buffers may be NumPy or CuPy arrays, or mpi4py buffer specs
    (``[array, MPI.DOUBLE]``). Other methods raise ``AttributeError``.
    """

    rank = 0
    size = 1

    def __init__(self, name: str = "COMM_WORLD") -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"SerialComm({self._name})"

    # ---------------------------------------------------------------- queries
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Get_name(self) -> str:
        return self._name

    def Is_inter(self) -> bool:
        return False

    def Is_intra(self) -> bool:
        return True

    # -------------------------------------------------- communicator creation
    def Dup(self, info: Any = None) -> SerialComm:
        return SerialComm(self._name)

    Clone = Dup

    def Split(self, color: int = 0, key: int = 0) -> Any:
        if color == SerialMPI.UNDEFINED:
            return SerialMPI.COMM_NULL
        return SerialComm(self._name)

    def Free(self) -> None:
        return None

    def Abort(self, errorcode: int = 0) -> None:
        raise SystemExit(errorcode)

    # ------------------------------------------------------ synchronization
    def Barrier(self) -> None:
        return None

    barrier = Barrier

    def Ibarrier(self) -> SerialRequest:
        return SerialRequest()

    # ------------------------------------------------- collectives, objects
    def bcast(self, obj: Any, root: int = 0) -> Any:
        _check_rank(root, "root")
        return obj

    def reduce(self, sendobj: Any, op: Any = None, root: int = 0) -> Any:
        _check_rank(root, "root")
        return sendobj

    def allreduce(self, sendobj: Any, op: Any = None) -> Any:
        return sendobj

    def scan(self, sendobj: Any, op: Any = None) -> Any:
        return sendobj

    def exscan(self, sendobj: Any, op: Any = None) -> None:
        return None  # undefined on rank 0, None in mpi4py

    def gather(self, sendobj: Any, root: int = 0) -> list[Any]:
        _check_rank(root, "root")
        return [sendobj]

    def allgather(self, sendobj: Any) -> list[Any]:
        return [sendobj]

    def scatter(self, sendobj: Any, root: int = 0) -> Any:
        _check_rank(root, "root")
        items = list(sendobj)
        if len(items) != 1:
            raise ValueError(f"scatter on 1 process needs 1 item, got {len(items)}")
        return items[0]

    def alltoall(self, sendobj: Any) -> list[Any]:
        items = list(sendobj)
        if len(items) != 1:
            raise ValueError(f"alltoall on 1 process needs 1 item, got {len(items)}")
        return items

    def sendrecv(
        self,
        sendobj: Any,
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = 0,
        recvtag: int = 0,
        status: Any = None,
    ) -> Any:
        _check_rank(dest, "dest")
        _check_rank(source, "source")
        if source == SerialMPI.PROC_NULL:
            return None
        return sendobj if dest != SerialMPI.PROC_NULL else None

    def ibcast(self, obj: Any, root: int = 0) -> SerialRequest:
        return SerialRequest(self.bcast(obj, root))

    def iallreduce(self, sendobj: Any, op: Any = None) -> SerialRequest:
        return SerialRequest(sendobj)

    # ------------------------------------------------- collectives, buffers
    def Bcast(self, buf: Any, root: int = 0) -> None:
        _check_rank(root, "root")

    def Reduce(self, sendbuf: Any, recvbuf: Any, op: Any = None, root: int = 0) -> None:
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Allreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Scan(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Exscan(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        return None  # the receive buffer of rank 0 is undefined

    def Gather(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Gatherv(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf), _displacement(recvbuf))

    def Allgather(self, sendbuf: Any, recvbuf: Any) -> None:
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Allgatherv(self, sendbuf: Any, recvbuf: Any) -> None:
        _copy(_buffer(sendbuf), _buffer(recvbuf), _displacement(recvbuf))

    def Scatter(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        _check_rank(root, "root")
        if recvbuf is not _IN_PLACE:
            _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Scatterv(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        _check_rank(root, "root")
        if recvbuf is _IN_PLACE:
            return
        source = _buffer(sendbuf).reshape(-1)
        start = _displacement(sendbuf)
        target = _buffer(recvbuf)
        _copy(source[start : start + target.size], target)

    def Alltoall(self, sendbuf: Any, recvbuf: Any) -> None:
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Sendrecv(
        self,
        sendbuf: Any,
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = 0,
        recvtag: int = 0,
        status: Any = None,
    ) -> None:
        _check_rank(dest, "dest")
        _check_rank(source, "source")
        if SerialMPI.PROC_NULL in (dest, source):
            return
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Ibcast(self, buf: Any, root: int = 0) -> SerialRequest:
        self.Bcast(buf, root)
        return SerialRequest()

    def Iallreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> SerialRequest:
        self.Allreduce(sendbuf, recvbuf, op)
        return SerialRequest()

    def Iallgather(self, sendbuf: Any, recvbuf: Any) -> SerialRequest:
        self.Allgather(sendbuf, recvbuf)
        return SerialRequest()


_COMM_WORLD = SerialComm("COMM_WORLD")
_COMM_SELF = SerialComm("COMM_SELF")


class SerialStatus:
    """Stand-in for ``MPI.Status`` (source 0, tag 0)."""

    source = 0
    tag = 0
    error = 0

    def Get_source(self) -> int:
        return 0

    def Get_tag(self) -> int:
        return 0

    def Get_count(self, datatype: Any = None) -> int:
        return 0


class SerialMPI:
    """Stand-in for the ``mpi4py.MPI`` module in a serial run (see :func:`get_mpi`).

    :func:`get_mpi` returns one instance; ``isinstance(MPI, SerialMPI)`` tells a
    serial run from an MPI one. ``COMM_WORLD`` and ``COMM_SELF`` are :class:`SerialComm` objects; the
    reduction operations, datatypes and other constants are placeholders that
    :class:`SerialComm` accepts. ``Is_initialized()`` is False: MPI itself is
    never started.
    """

    COMM_WORLD = _COMM_WORLD
    COMM_SELF = _COMM_SELF
    COMM_NULL = _COMM_NULL
    Comm = Intracomm = SerialComm
    Request = SerialRequest
    Status = SerialStatus

    IN_PLACE = _IN_PLACE
    BOTTOM = _Constant("BOTTOM")
    DATATYPE_NULL = _Null("DATATYPE_NULL")
    REQUEST_NULL = _Null("REQUEST_NULL")
    OP_NULL = _Null("OP_NULL")
    PROC_NULL = -2
    ANY_SOURCE = -1
    ANY_TAG = -1
    ROOT = -3
    UNDEFINED = -32766
    SUCCESS = 0

    Datatype = _Datatype
    Op = _Op
    Prequest = SerialPrequest

    # reduction operations
    SUM = _Op("SUM")
    PROD = _Op("PROD")
    MAX = _Op("MAX")
    MIN = _Op("MIN")
    LAND = _Op("LAND")
    LOR = _Op("LOR")
    LXOR = _Op("LXOR")
    BAND = _Op("BAND")
    BOR = _Op("BOR")
    BXOR = _Op("BXOR")
    MAXLOC = _Op("MAXLOC")
    MINLOC = _Op("MINLOC")
    REPLACE = _Op("REPLACE")

    # datatypes
    BYTE = _Datatype("BYTE")
    CHAR = _Datatype("CHAR")
    BOOL = _Datatype("BOOL")
    C_BOOL = _Datatype("C_BOOL")
    INT = _Datatype("INT")
    LONG = _Datatype("LONG")
    LONG_LONG = _Datatype("LONG_LONG")
    UNSIGNED = _Datatype("UNSIGNED")
    UNSIGNED_LONG = _Datatype("UNSIGNED_LONG")
    INT8_T = _Datatype("INT8_T")
    INT16_T = _Datatype("INT16_T")
    INT32_T = _Datatype("INT32_T")
    INT64_T = _Datatype("INT64_T")
    UINT8_T = _Datatype("UINT8_T")
    UINT16_T = _Datatype("UINT16_T")
    UINT32_T = _Datatype("UINT32_T")
    UINT64_T = _Datatype("UINT64_T")
    FLOAT = _Datatype("FLOAT")
    DOUBLE = _Datatype("DOUBLE")
    LONG_DOUBLE = _Datatype("LONG_DOUBLE")
    C_FLOAT_COMPLEX = _Datatype("C_FLOAT_COMPLEX")
    C_DOUBLE_COMPLEX = _Datatype("C_DOUBLE_COMPLEX")
    COMPLEX = _Datatype("COMPLEX")
    DOUBLE_COMPLEX = _Datatype("DOUBLE_COMPLEX")

    # NumPy type characters to datatypes, like mpi4py's (private) MPI._typedict
    _typedict = MappingProxyType({
        "b": INT8_T, "h": INT16_T, "i": INT32_T, "l": LONG, "q": INT64_T,
        "B": UINT8_T, "H": UINT16_T, "I": UINT32_T, "L": UNSIGNED_LONG, "Q": UINT64_T,
        "f": FLOAT, "d": DOUBLE, "g": LONG_DOUBLE, "?": C_BOOL,
        "F": C_FLOAT_COMPLEX, "D": C_DOUBLE_COMPLEX,
    })  # fmt: skip

    def __repr__(self) -> str:
        return "<SerialMPI: serial stand-in for mpi4py.MPI>"

    @staticmethod
    def Wtime() -> float:
        return time.time()

    @staticmethod
    def Wtick() -> float:
        return time.get_clock_info("time").resolution

    @staticmethod
    def Is_initialized() -> bool:
        return False

    @staticmethod
    def Is_finalized() -> bool:
        return False

    @staticmethod
    def Init() -> None:
        return None

    @staticmethod
    def Finalize() -> None:
        return None

    @staticmethod
    def Get_processor_name() -> str:
        return socket.gethostname()

    @staticmethod
    def Query_thread() -> int:
        return 0


_SERIAL_MPI = SerialMPI()
