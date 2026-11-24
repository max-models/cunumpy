"""C host kernels: ``<name>_c.c`` compiled with the system compiler, called like Pyccel's.

A kernel folder may hold a C version of its host kernel next to the Pyccel one,
for loops that the compiler must vectorise (Pyccel's generated C reads arrays
through generic strided accessors, which stops most vectorisation). The C
function takes the arguments of the Pyccel version, in the same order:

=====================  =================================================
Pyccel annotation      C parameters
=====================  =================================================
``"float"``            ``double``
``"int"``, ``"int64"``  ``int64_t``
``"int32"``            ``int32_t``
``"bool"``             ``bool``
``"float[:,:]"``       ``double *x, const int64_t *x_shape`` (C order)
``"int[:]"``           ``int64_t *x, const int64_t *x_shape``
``"int32[:,:,:]"``     ``int32_t *x, const int64_t *x_shape``
``"bool[:]"``          ``bool *x, const int64_t *x_shape``
=====================  =================================================

Arrays must be C-contiguous; the function returns ``void``. A generated
wrapper (CPython fast-call, the buffer protocol, no NumPy headers) is compiled
with the kernel into one extension module, so a call costs about as much as a
Pyccel one. :class:`CCache` builds and caches the modules like
:class:`~cunumpy.kernels.PyccelCache`; the build flags default to
``-O3 -march=native`` (the build key then includes the CPU, so a cache shared
between different machines keeps one build per CPU) and OpenMP.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cunumpy._pyccel_cache import (
    _FALSE,
    _check_importable,
    _load_extension,
    _writable,
)

__all__ = ["CCache", "CParameter", "c_parameters", "wrapper_source"]

#: Pyccel element types: (C type, buffer format kind, item size).
_ELEMENT_TYPES = {
    "float": ("double", "f", 8),
    "float64": ("double", "f", 8),
    "int": ("int64_t", "i", 8),
    "int64": ("int64_t", "i", 8),
    "int32": ("int32_t", "i", 4),
    "bool": ("bool", "b", 1),
}

#: Buffer format characters of each kind (the byte order prefix aside).
_FORMATS = {"f": "d", "i": "bhilqBHILQnN", "b": "?"}

_ANNOTATION = re.compile(r"^\s*(\w+)\s*(?:\[\s*([:,\s]*)\])?\s*$")

#: Default flags of the builds; ``<PREFIX>_C_FLAGS`` replaces them.
DEFAULT_FLAGS = "-O3 -march=native"

#: Format version of the generated wrapper; part of the build key.
WRAPPER_VERSION = "1"


@dataclass(frozen=True)
class CParameter:
    """One parameter of a C kernel: its name, C element type and dimensions (0: scalar)."""

    name: str
    c_type: str
    kind: str
    itemsize: int
    ndim: int


def c_parameters(function: Callable[..., Any]) -> list[CParameter]:
    """Return the C parameters of a kernel, from the Pyccel annotations of ``function``.

    Parameters
    ----------
    function : callable
        The Python form of the Pyccel kernel (its annotations are strings).

    Returns
    -------
    list of CParameter
        One per positional parameter.

    Raises
    ------
    NotImplementedError
        For an annotation the C wrapper does not take (Fortran order, complex,
        keyword-only parameters, a return value).
    """
    signature = inspect.signature(function)
    if signature.return_annotation not in (None, "None", inspect.Signature.empty):
        raise NotImplementedError(f"{function.__name__}: C kernels return nothing")
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise NotImplementedError(
                f"{function.__name__}: parameter {parameter.name} is not positional"
            )
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            raise NotImplementedError(
                f"{function.__name__}: parameter {parameter.name} has no annotation"
            )
        text = (
            annotation
            if isinstance(annotation, str)
            else getattr(annotation, "__name__", str(annotation))
        )
        match = _ANNOTATION.match(text)
        if match is None or match.group(1) not in _ELEMENT_TYPES:
            raise NotImplementedError(
                f"{function.__name__}: annotation {text!r} of {parameter.name} "
                "has no C form"
            )
        c_type, kind, itemsize = _ELEMENT_TYPES[match.group(1)]
        ndim = 0 if match.group(2) is None else match.group(2).count(":")
        if match.group(2) is not None and ndim == 0:
            raise NotImplementedError(f"{function.__name__}: annotation {text!r}")
        parameters.append(CParameter(parameter.name, c_type, kind, itemsize, ndim))
    return parameters


def _prototype(name: str, parameters: Sequence[CParameter]) -> str:
    """Return the C declaration of the kernel function."""
    pieces = []
    for p in parameters:
        if p.ndim:
            pieces += [f"{p.c_type} *{p.name}", f"const int64_t *{p.name}_shape"]
        else:
            pieces.append(f"{p.c_type} {p.name}")
    return f"void {name}({', '.join(pieces) or 'void'})"


def wrapper_source(name: str, module: str, parameters: Sequence[CParameter]) -> str:
    """Return the C source of the extension module ``module`` calling the kernel ``name``.

    Parameters
    ----------
    name : str
        The kernel function in the C source.
    module : str
        The name of the extension module (its ``PyInit_`` function).
    parameters : sequence of CParameter
        The kernel's parameters (:func:`c_parameters`).

    Returns
    -------
    str
        The wrapper's C source.
    """
    count = len(parameters)
    declarations, conversions, releases, call = [], [], [], []
    for index, p in enumerate(parameters):
        if p.ndim:
            declarations.append(
                f"    Py_buffer b{index} = {{0}};\n    int64_t s{index}[{p.ndim}];\n"
            )
            conversions.append(
                f"    if (cunumpy_array(args[{index}], &b{index}, s{index}, {p.ndim}, "
                f"'{p.kind}', {p.itemsize}, \"{p.name}\") < 0) goto done;\n"
            )
            releases.append(f"    if (b{index}.obj) PyBuffer_Release(&b{index});\n")
            call += [f"({p.c_type} *)b{index}.buf", f"s{index}"]
            continue
        declarations.append(f"    {p.c_type} a{index};\n")
        if p.kind == "f":
            conversions.append(
                f"    a{index} = PyFloat_AsDouble(args[{index}]);\n"
                f"    if (a{index} == -1.0 && PyErr_Occurred()) goto done;\n"
            )
        elif p.kind == "i":
            conversions.append(
                f"    {{ long long v = PyLong_AsLongLong(args[{index}]);\n"
                f"      if (v == -1 && PyErr_Occurred()) goto done;\n"
                f"      a{index} = ({p.c_type})v; }}\n"
            )
        else:
            conversions.append(
                f"    {{ int v = PyObject_IsTrue(args[{index}]);\n"
                f"      if (v < 0) goto done;\n"
                f"      a{index} = (bool)v; }}\n"
            )
        call.append(f"a{index}")
    return f"""\
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

{_prototype(name, parameters)};

static int cunumpy_array(PyObject *object, Py_buffer *buffer, int64_t *shape,
                         int ndim, char kind, Py_ssize_t itemsize, const char *name)
{{
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (PyObject_GetBuffer(object, buffer, flags | PyBUF_WRITABLE) < 0) {{
        PyErr_Clear();
        if (PyObject_GetBuffer(object, buffer, flags) < 0) {{
            PyErr_Format(PyExc_TypeError,
                         "{name}: argument %s must be a C-contiguous array", name);
            return -1;
        }}
    }}
    if (buffer->ndim != ndim) {{
        PyErr_Format(PyExc_TypeError, "{name}: argument %s must have %d dimensions, "
                     "got %d", name, ndim, buffer->ndim);
        return -1;
    }}
    const char *format = buffer->format ? buffer->format : "B";
    if (*format == '<' || *format == '=' || *format == '@' || *format == '!' ||
        *format == '>') format++;
    const char *kinds = kind == 'f' ? "d" : (kind == 'i' ? "bhilqBHILQnN" : "?");
    if (buffer->itemsize != itemsize || format[1] != '\\0' ||
        strchr(kinds, format[0]) == NULL) {{
        PyErr_Format(PyExc_TypeError, "{name}: argument %s has element type '%s' of "
                     "%zd bytes", name, format, buffer->itemsize);
        return -1;
    }}
    for (int axis = 0; axis < ndim; ++axis) shape[axis] = (int64_t)buffer->shape[axis];
    return 0;
}}

static PyObject *cunumpy_call(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{{
    (void)self;
    PyObject *result = NULL;
{"".join(declarations)}    if (nargs != {count}) {{
        PyErr_Format(PyExc_TypeError, "{name}() takes {count} arguments, got %zd", nargs);
        goto done;
    }}
{"".join(conversions)}    Py_BEGIN_ALLOW_THREADS
    {name}({", ".join(call)});
    Py_END_ALLOW_THREADS
    result = Py_None;
    Py_INCREF(result);
done:
{"".join(releases)}    return result;
}}

static PyMethodDef cunumpy_methods[] = {{
    {{"{name}", (PyCFunction)(void (*)(void))cunumpy_call, METH_FASTCALL, NULL}},
    {{NULL, NULL, 0, NULL}},
}};

static struct PyModuleDef cunumpy_module = {{
    PyModuleDef_HEAD_INIT, "{module}", NULL, -1, cunumpy_methods,
    NULL, NULL, NULL, NULL,
}};

PyMODINIT_FUNC PyInit_{module}(void) {{ return PyModule_Create(&cunumpy_module); }}
"""


_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _included_sources(source: Path, include_dirs: Sequence[Path]) -> dict[str, str]:
    """Return the text of every ``#include "..."`` reachable from ``source`` (for the build key)."""
    found: dict[str, str] = {}
    pending = [source]
    while pending:
        current = pending.pop()
        text = current.read_text()
        for include in _INCLUDE.findall(text):
            for directory in (current.parent, *include_dirs):
                path = Path(directory) / include
                if path.is_file():
                    key = str(path.resolve())
                    if key not in found:
                        found[key] = path.read_text()
                        pending.append(path)
                    break
    return found


def _cpu_identity() -> str:
    """Return the model and instruction-set flags of this CPU (for ``-march=native`` builds)."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.machine() + platform.processor()
    model = re.search(r"^model name\s*:\s*(.*)$", text, re.MULTILINE)
    flags = re.search(r"^(?:flags|Features)\s*:\s*(.*)$", text, re.MULTILINE)
    return (model.group(1) if model else "") + (flags.group(1) if flags else "")


class CCache:
    """Compile C host kernels on first use and keep the builds on disk.

    The builds live next to the Pyccel ones (``<PREFIX>_KERNEL_CACHE``, default
    ``~/.cache/<name>/kernels``), under ``c/``. ``<PREFIX>_C_FLAGS`` replaces
    the default flags :data:`DEFAULT_FLAGS`; ``<PREFIX>_OPENMP=0`` builds
    without OpenMP; ``CC`` names the compiler (``cc`` by default). A build is
    imported once in a subprocess before it is published, as Pyccel builds are.

    Parameters
    ----------
    name : str
        The package name.
    env_prefix : str, optional
        Prefix of the environment variables; by default `name` in upper case
        with ``_`` for other characters.
    """

    def __init__(self, name: str, *, env_prefix: str | None = None) -> None:
        if not name:
            raise ValueError("CCache needs a package name")
        self.name = name
        self.env_prefix = (
            env_prefix
            if env_prefix is not None
            else re.sub(r"[^0-9A-Za-z]", "_", name).upper()
        )

    @property
    def flags_env(self) -> str:
        """Name of the environment variable replacing the default flags."""
        return f"{self.env_prefix}_C_FLAGS"

    def compiler(self) -> list[str]:
        """Return the compiler command (``CC``, default ``cc``)."""
        return shlex.split(os.environ.get("CC", "cc"))

    def flags(self) -> list[str]:
        """Return the optimisation flags (``<PREFIX>_C_FLAGS`` or :data:`DEFAULT_FLAGS`) and OpenMP."""
        flags = shlex.split(os.environ.get(self.flags_env, DEFAULT_FLAGS))
        if self.openmp_enabled() and sys.platform != "darwin":
            flags.append("-fopenmp")
        return flags

    def openmp_enabled(self) -> bool:
        """Whether builds use OpenMP (``<PREFIX>_OPENMP=0`` turns it off)."""
        return os.environ.get(f"{self.env_prefix}_OPENMP", "1").lower() not in _FALSE

    def cache_root(self) -> Path:
        """Directory holding the builds."""
        default = Path.home() / ".cache" / self.name / "kernels"
        return Path(os.environ.get(f"{self.env_prefix}_KERNEL_CACHE", default)) / "c"

    def fallback_root(self) -> Path:
        """Directory for builds when :meth:`cache_root` is not writable."""
        return Path(tempfile.gettempdir()) / f"{self.name}-kernels" / "c"

    def _compiler_version(self) -> str:
        try:
            result = subprocess.run(
                [*self.compiler(), "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"no C compiler ({' '.join(self.compiler())})"
            ) from error
        return result.stdout.splitlines()[0] if result.stdout else ""

    def build_key(
        self, source: Path, wrapper: str, include_dirs: Sequence[Path]
    ) -> str:
        """Return a hash of what the build depends on.

        The sources (the kernel, the headers it includes, the wrapper), the
        compiler and its flags, the Python version, and the CPU when the flags
        tune for it.

        Parameters
        ----------
        source : Path
            The C source.
        wrapper : str
            The generated wrapper source.
        include_dirs : sequence of Path
            The include directories.

        Returns
        -------
        str
            The first 16 hexadecimal digits of a SHA-256.
        """
        flags = self.flags()
        parts = [
            WRAPPER_VERSION,
            source.read_text(),
            wrapper,
            *sorted(_included_sources(source, include_dirs).values()),
            " ".join(self.compiler()),
            self._compiler_version(),
            " ".join(flags),
            sys.version,
            sysconfig.get_config_var("EXT_SUFFIX") or "",
        ]
        if any("native" in flag for flag in flags):
            parts.append(_cpu_identity())
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]

    def _build(
        self,
        source: Path,
        wrapper: str,
        stem: str,
        include_dirs: Sequence[Path],
        staging: Path,
    ) -> Path:
        build = staging / "build"
        build.mkdir()
        wrapper_path = build / f"{stem}_wrapper.c"
        wrapper_path.write_text(wrapper)
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        link = (
            ["-bundle", "-undefined", "dynamic_lookup"]
            if sys.platform == "darwin"
            else ["-shared"]
        )
        command = [
            *self.compiler(),
            *self.flags(),
            "-fPIC",
            *link,
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{source.parent}",
            *(f"-I{directory}" for directory in include_dirs),
            str(source),
            str(wrapper_path),
            "-o",
            str(build / f"{stem}{suffix}"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"building the C kernel {source.name} failed:\n{' '.join(command)}\n"
                f"{result.stdout}{result.stderr}"
            )
        return build

    def compile(
        self,
        source: str | Path,
        name: str,
        reference: Callable[..., Any],
        include_dirs: Sequence[str | Path] = (),
    ) -> Callable[..., Any]:
        """Return the compiled kernel ``name`` of ``source``, building it on first use.

        Parameters
        ----------
        source : str or Path
            The C source (``<name>_c.c``).
        name : str
            The kernel function in it.
        reference : callable
            The Pyccel version, whose annotations give the arguments.
        include_dirs : sequence of str or Path, optional
            Directories for ``#include "..."`` besides the source's own.

        Returns
        -------
        callable
            The kernel, called with the arguments of `reference`.

        Raises
        ------
        NotImplementedError
            If an annotation has no C form (see :func:`c_parameters`).
        RuntimeError
            With the compiler output if the build fails.
        PermissionError
            If neither the cache nor the fallback directory is writable.
        """
        source = Path(source).resolve()
        include_dirs = [Path(directory) for directory in include_dirs]
        parameters = c_parameters(reference)
        stem = f"{name}_c"
        wrapper = wrapper_source(name, stem, parameters)
        directory = f"{stem}-{self.build_key(source, wrapper, include_dirs)}"
        roots = [self.cache_root(), self.fallback_root()]
        for root in roots:
            if (root / directory).is_dir():
                extension = _load_extension(root / directory, stem)
                if extension is not None:
                    return getattr(extension, name)
        target = next((root / directory for root in roots if _writable(root)), None)
        if target is None:
            raise PermissionError(f"No writable kernel cache among {roots}")
        staging = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent))
        try:
            build = self._build(source, wrapper, stem, include_dirs, staging)
            _check_importable(build, stem)
            try:
                os.rename(build, target)
            except OSError:
                pass  # another process published the same build first
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        extension = _load_extension(target, stem)
        if extension is None:
            raise ImportError(f"compiled C kernel {stem} not found in {target}")
        return getattr(extension, name)

    def __repr__(self) -> str:
        return f"CCache({self.name!r}, env_prefix={self.env_prefix!r})"
