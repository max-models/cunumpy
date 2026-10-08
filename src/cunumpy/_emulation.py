"""Run a CUDA kernel on the CPU, one thread after another, for tests without a GPU.

A ported kernel is usually checked against its host version on a GPU
(:func:`cunumpy.kernel_testing.assert_kernels_agree`). Without one, CI cannot run that
check, and the kernel's index and weight arithmetic go untested.
:func:`emulate_cuda_kernel` closes that gap: it compiles the kernel source as
C++ with the CUDA built-ins replaced by plain C++ (``threadIdx``, ``blockIdx``,
``blockDim``, ``gridDim``, ``atomicAdd``, ...) into a shared library, and calls
the kernel once per thread, serially, on the NumPy arguments, so the call looks
like a launch::

    from cunumpy.kernel_testing import emulate_cuda_kernel

    y = np.zeros(1000)
    emulate_cuda_kernel(axpy, 2.0, x, y, 1000, n_threads=1000)
    np.testing.assert_allclose(y, 2.0 * x)

Arguments follow the kernel signature: NumPy arrays for pointer and array view
parameters (``Array1D<T>`` to ``Array16D<T>``, any strides), Python or NumPy
scalars for scalar parameters (cast and checked like in a launch). The kernel
writes into the given arrays.

Each kernel is compiled once: the library takes the arguments at run time
(``cunumpy_launch(args, grid, block, shared_mem)``, loaded with ctypes), so
launches with other values and array sizes reuse it. Libraries are cached in
the process and on disk (:func:`emulation_cache_dir`), keyed on the generated
source, the included headers, the compiler and its options.
:func:`compile_for_emulation` builds one without a launch. The launch runs in
the Python process: ``__trap()`` (a failed bounds check) is reported as a
``RuntimeError``, but a segmentation fault ends the process.

Block shared memory and ``__syncthreads`` are emulated: ``__shared__``
variables (and ``extern __shared__`` arrays, sized by ``shared_mem``) are one
copy per block, and in a kernel that calls ``__syncthreads`` the threads of a
block run as coroutines (POSIX ``ucontext``, each with its own stack): every
thread runs to its next barrier before any thread continues past it, as on a
GPU. Per-block deposits, shared-memory reductions and tiled kernels therefore
work.

Struct parameters take, for each struct, a mapping from field name to value, a
:class:`~cunumpy.arguments.CudaStructValue`, or any object with an attribute
per field (e.g. a :class:`~cunumpy.arguments.CudaStructArguments` or a host
argument class); the arrays of the fields are updated in place like the others.

:func:`emulated_launches` makes every :class:`~cunumpy.kernels.CudaKernel` launch
(and compilation) in a block run through the emulation, on the arrays of the
fake CuPy (:mod:`cunumpy._fake_cupy`), so that code that launches kernels (a
:class:`~cunumpy.kernels.Kernel` on the CuPy backend, a propagator) can be
tested without a GPU.

What it does not emulate: concurrency between the barriers (threads run one
after another, so atomics are plain additions and races never show), warp
intrinsics (``__shfl_*``, ``__syncwarp``, ``__ballot_sync``, ...; a kernel or an
included header using them is refused), ``<cupy/complex.cuh>``, and inline PTX:
``asm(...)`` and ``asm volatile(...)`` compile, but trap when reached. Use a
GPU for those.

Floating point: like NVRTC (``--fmad=true`` by default) the C++ compiler may
fuse ``a * b + c`` into one fused multiply-add, so results can differ from
NumPy's in the last bit; compare with a tolerance of a few ulp, or pass
``options=("-ffp-contract=off",)`` for NumPy's rounding.

It needs a C++17 compiler: ``CXX`` from the environment, else ``c++``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import weakref
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from cunumpy import _fake_cupy
from cunumpy._cuda_kernel import (
    CudaKernel,
    CudaParameter,
    CudaStructArguments,
    CudaStructValue,
    _scalar_checker,
    _strip_comments,
    cuda_include_dir,
)

__all__ = [
    "compile_for_emulation",
    "emulate_cuda_kernel",
    "emulated_launches",
    "emulation_cache_dir",
    "emulation_compiler",
]

# CUDA constructs that serial emulation would get wrong
_UNSUPPORTED = {
    r"__syncwarp": "warp synchronization (__syncwarp)",
    r"__shfl\w*": "warp shuffles",
    r"__ballot_sync|__any_sync|__all_sync": "warp votes",
}

_STUBS = r"""
// --- cunumpy emulation: CUDA built-ins as plain C++, one thread at a time ---
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __launch_bounds__(...)
// block shared memory: one copy for the block that runs (blocks run one after
// another); `extern __shared__` arrays point into a buffer of shared_mem bytes
#define __shared__ static
static void* cunumpy_dynamic_shared = 0;
struct cunumpy_dim3 { unsigned int x, y, z; };
static cunumpy_dim3 threadIdx, blockIdx, blockDim, gridDim;
static const int warpSize = 32;
// __trap() ends the launch: cunumpy_launch returns 1 (in a coroutine, the hook
// returns to the scheduler first)
static jmp_buf cunumpy_trap_jump;
static int cunumpy_trapped = 0;
static void (*cunumpy_trap_hook)() = 0;
[[noreturn]] inline void __trap() {
    cunumpy_trapped = 1;
    fflush(stdout);
    if (cunumpy_trap_hook) cunumpy_trap_hook();
    longjmp(cunumpy_trap_jump, 1);
}
template <class T> inline T __ldg(const T* p) { return *p; }
inline double rsqrt(double v) { return 1.0 / sqrt(v); }
inline float rsqrtf(float v) { return 1.0f / sqrtf(v); }
using std::min;
using std::max;
template <class T> inline T cunumpy_atomic_op_add(T* p, T v) { T o = *p; *p += v; return o; }
inline double atomicAdd(double* p, double v) { return cunumpy_atomic_op_add(p, v); }
inline float atomicAdd(float* p, float v) { return cunumpy_atomic_op_add(p, v); }
inline int atomicAdd(int* p, int v) { return cunumpy_atomic_op_add(p, v); }
inline unsigned int atomicAdd(unsigned int* p, unsigned int v) { return cunumpy_atomic_op_add(p, v); }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return cunumpy_atomic_op_add(p, v); }
inline int atomicSub(int* p, int v) { int o = *p; *p -= v; return o; }
inline unsigned int atomicSub(unsigned int* p, unsigned int v) { unsigned int o = *p; *p -= v; return o; }
template <class T> inline T atomicExch(T* p, T v) { T o = *p; *p = v; return o; }
template <class T> inline T atomicMin(T* p, T v) { T o = *p; if (v < o) *p = v; return o; }
template <class T> inline T atomicMax(T* p, T v) { T o = *p; if (v > o) *p = v; return o; }
inline unsigned long long atomicCAS(unsigned long long* p, unsigned long long c, unsigned long long v) { unsigned long long o = *p; if (o == c) *p = v; return o; }
inline int atomicCAS(int* p, int c, int v) { int o = *p; if (o == c) *p = v; return o; }
inline long long __double_as_longlong(double v) { long long r; memcpy(&r, &v, sizeof r); return r; }
inline double __longlong_as_double(long long v) { double r; memcpy(&r, &v, sizeof r); return r; }
// inline PTX cannot run on the CPU: `asm(...)` and `asm volatile(...)` trap
static int cunumpy_inline_asm;
#define cunumpy_inline_asm(...) __trap()
#define asm cunumpy_inline_asm
#define volatile(...) , __trap()
// ---------------------------------------------------------------------------
"""

_COROUTINES = r"""
// --- __syncthreads: the threads of a block are coroutines (ucontext); each
// runs until the next barrier, then the scheduler resumes the next one ---
#include <ucontext.h>
struct cunumpy_thread { ucontext_t ctx; char* stack; int done; };
static ucontext_t cunumpy_scheduler;
static cunumpy_thread* cunumpy_threads = 0;
static unsigned int cunumpy_current = 0;
#define CUNUMPY_STACK_BYTES (256 * 1024)
inline void __syncthreads() {
    swapcontext(&cunumpy_threads[cunumpy_current].ctx, &cunumpy_scheduler);
}
static void cunumpy_coroutine_trap() {
    cunumpy_threads[cunumpy_current].done = 1;
    setcontext(&cunumpy_scheduler);
}
"""

_RUN_SERIAL = r"""
static void cunumpy_run() {{
    for (unsigned int bz = 0; bz < gridDim.z; ++bz)
    for (unsigned int by = 0; by < gridDim.y; ++by)
    for (unsigned int bx = 0; bx < gridDim.x; ++bx)
    for (unsigned int tz = 0; tz < blockDim.z; ++tz)
    for (unsigned int ty = 0; ty < blockDim.y; ++ty)
    for (unsigned int tx = 0; tx < blockDim.x; ++tx) {{
        blockIdx = {{bx, by, bz}};
        threadIdx = {{tx, ty, tz}};
        {call};
    }}
}}
"""

_RUN_COROUTINES = r"""
static void cunumpy_entry() {{
    {call};
    cunumpy_threads[cunumpy_current].done = 1;
}}
static void cunumpy_run() {{
    const unsigned int n = blockDim.x * blockDim.y * blockDim.z;
    cunumpy_threads = (cunumpy_thread*)calloc(n, sizeof(cunumpy_thread));
    for (unsigned int t = 0; t < n; ++t)
        cunumpy_threads[t].stack = (char*)malloc(CUNUMPY_STACK_BYTES);
    cunumpy_trap_hook = cunumpy_coroutine_trap;
    for (unsigned int bz = 0; bz < gridDim.z; ++bz)
    for (unsigned int by = 0; by < gridDim.y; ++by)
    for (unsigned int bx = 0; bx < gridDim.x; ++bx) {{
        blockIdx = {{bx, by, bz}};
        for (unsigned int t = 0; t < n; ++t) {{
            cunumpy_thread* th = &cunumpy_threads[t];
            getcontext(&th->ctx);
            th->ctx.uc_stack.ss_sp = th->stack;
            th->ctx.uc_stack.ss_size = CUNUMPY_STACK_BYTES;
            th->ctx.uc_link = &cunumpy_scheduler;
            makecontext(&th->ctx, (void (*)())cunumpy_entry, 0);
            th->done = 0;
        }}
        // rounds: every thread runs to its next __syncthreads (or its end)
        // before any thread continues past it
        for (int running = 1; running;) {{
            running = 0;
            for (unsigned int t = 0; t < n; ++t) {{
                if (cunumpy_threads[t].done) continue;
                cunumpy_current = t;
                threadIdx = {{t % blockDim.x, (t / blockDim.x) % blockDim.y,
                              t / (blockDim.x * blockDim.y)}};
                swapcontext(&cunumpy_scheduler, &cunumpy_threads[t].ctx);
                if (cunumpy_trapped) goto done;
                if (!cunumpy_threads[t].done) running = 1;
            }}
        }}
    }}
done:
    for (unsigned int t = 0; t < n; ++t) free(cunumpy_threads[t].stack);
    free(cunumpy_threads);
    cunumpy_threads = 0;
    cunumpy_trap_hook = 0;
}}
"""

# the entry point of the library: the arguments are pointers to the values
# (scalars), to the data (pointers) or to {data, shape..., strides...} as 64-bit
# integers (array views)
_LAUNCH = r"""
{globals}
{run}
extern "C" int cunumpy_launch(void** cunumpy_args, const unsigned int* g,
                              const unsigned int* b, unsigned long long shared_mem) {{
{inits}
    cunumpy_trapped = 0;
    cunumpy_dynamic_shared = calloc(shared_mem + 16, 1);
    gridDim = {{g[0], g[1], g[2]}};
    blockDim = {{b[0], b[1], b[2]}};
    if (setjmp(cunumpy_trap_jump) == 0) cunumpy_run();
    free(cunumpy_dynamic_shared);
    cunumpy_dynamic_shared = 0;
    fflush(stdout);
    return cunumpy_trapped;
}}
"""


def emulation_compiler() -> str | None:
    """The C++ compiler :func:`emulate_cuda_kernel` uses, or None if there is none."""
    compiler = os.environ.get("CXX")
    if compiler and shutil.which(compiler):
        return shutil.which(compiler)
    return shutil.which("c++")


def _scalar_value(param: CudaParameter, index: int, value: Any) -> np.ndarray:
    """`value` as a 0-d array of the C type, checked and cast like a scalar argument."""
    if np.dtype(param.dtype).kind not in "biuf":
        raise NotImplementedError(f"emulation does not support {param.ctype} scalars")
    return np.asarray(_scalar_checker(param, index)(value), dtype=param.dtype)


def _code(kernel: CudaKernel) -> str:
    """The kernel source and the headers it includes, without comments."""
    texts = [kernel.source]
    for header in kernel.included_headers:
        try:
            texts.append(Path(header).read_text(errors="replace"))
        except OSError:
            pass
    return _strip_comments("\n".join(texts))


_EXTERN_SHARED = re.compile(
    r"extern\s+__shared__\s+(?:__align__\(\s*\d+\s*\)\s+)?([\w:<>\s]+?)\s+(\w+)\s*\[\s*\]\s*;",
)


def _check_supported(kernel: CudaKernel) -> None:
    code = _code(kernel)
    for pattern, what in _UNSUPPORTED.items():
        if re.search(pattern, code):
            raise NotImplementedError(
                f"kernel {kernel.name!r} uses {what}, which emulation cannot run "
                "correctly one thread at a time; test it on a GPU",
            )


def _host(value: Any) -> Any:
    """The NumPy buffer of a fake CuPy array; anything else as it is."""
    if _fake_cupy.is_active():
        try:
            return _fake_cupy.host_buffer(value)
        except TypeError:
            pass
    return value


def _field_value(struct_name: str, value: Any, field: CudaParameter) -> Any:
    """The value of the struct field `field` in the struct argument `value`."""
    try:
        if isinstance(value, (CudaStructValue, Mapping)):
            return value[field.name]
        return getattr(value, field.name)
    except (KeyError, AttributeError):
        raise TypeError(
            f"struct argument {struct_name!r} has no value for the field "
            f"{field.name!r} ({type(value).__name__})",
        ) from None


def _declaration(param: CudaParameter, name: str) -> str:
    if param.view_ndim is not None:
        return f"{param.ctype} {name}"
    return f"{param.ctype}{'*' if param.pointer else ''} {name}"


# wrapper kernels of the kernels with struct parameters, see _struct_wrapper
_WRAPPERS: weakref.WeakKeyDictionary[CudaKernel, CudaKernel] = (
    weakref.WeakKeyDictionary()
)


def _struct_wrapper(kernel: CudaKernel) -> CudaKernel:
    """A kernel that takes every struct field of `kernel` as a parameter.

    The wrapper kernel (the source of `kernel` plus a ``__global__`` function with
    one parameter per field) rebuilds the structs and calls `kernel`, so that the
    emulation only has to deal with arrays and scalars. Its arguments are
    :func:`_struct_fields`. Made once per kernel.
    """
    wrapper = _WRAPPERS.get(kernel)
    if wrapper is not None:
        return wrapper
    params, builds, call = [], [], []
    for p in kernel.signature:
        if p.struct is None:
            params.append(_declaration(p, p.name))
            call.append(p.name)
            continue
        names = [f"{p.name}__{f.name}" for f in p.struct.fields]
        params += [_declaration(f, n) for f, n in zip(p.struct.fields, names)]
        builds.append(f"    {p.struct.name} {p.name}_struct{{{', '.join(names)}}};")
        call.append(f"{p.name}_struct")
    name = f"emulated_{kernel.name}"
    source = (
        kernel.source
        + f'\nextern "C" __global__ void {name}({", ".join(params)}) {{\n'
        + "\n".join(builds)
        + f"\n    {kernel.expression}({', '.join(call)});\n}}\n"
    )
    wrapper = CudaKernel(
        source,
        name,
        block_size=kernel.block_size,
        options=[o for o in kernel.options if not o.startswith("-I")],
        include_dirs=kernel.include_dirs,
        source_dir=kernel.source_dir,
        n_threads_from=kernel.n_threads_from,
    )
    _WRAPPERS[kernel] = wrapper
    return wrapper


def _struct_fields(kernel: CudaKernel, args: Sequence[Any]) -> list[Any]:
    """The arguments of :func:`_struct_wrapper`: each struct replaced by its fields."""
    flat = []
    for p, value in zip(kernel.signature, args):
        if p.struct is None:
            flat.append(value)
        else:
            flat += [_field_value(p.name, value, f) for f in p.struct.fields]
    return flat


def _emulation_source(kernel: CudaKernel) -> tuple[str, bool]:
    """The CUDA stubs and the kernel source as C++, without the launch function.

    Returns
    -------
    tuple[str, bool]
        The source, and whether the threads of a block run as coroutines (the
        kernel calls ``__syncthreads``).
    """
    kernel_source = kernel.source.replace('extern "C"', "")
    kernel_source = _EXTERN_SHARED.sub(
        r"\1* \2 = (\1*)cunumpy_dynamic_shared;",
        kernel_source,
    )
    coroutines = "__syncthreads" in _code(kernel)
    if coroutines:
        # ucontext needs _XOPEN_SOURCE before any system header (and macOS then
        # hides the rest of its C library, which GCC's <cstdlib> needs)
        source = (
            "#define _XOPEN_SOURCE 700\n"
            "#ifdef __APPLE__\n#define _DARWIN_C_SOURCE\n#endif\n"
            + _STUBS
            + _COROUTINES
        )
    else:
        source = _STUBS + "inline void __syncthreads() {}\n"
    return source + kernel_source, coroutines


def _element_type(param: CudaParameter) -> str:
    """The C type of the elements of a pointer or array view parameter."""
    if param.dtype is None:
        return "unsigned char"
    if param.view_ndim is not None:
        return re.match(r"C?Array\d+D<(.*)>", param.ctype).group(1)
    return param.ctype


def _library_source(kernel: CudaKernel) -> str:
    """The C++ source of the emulation library of `kernel` (see :data:`_LAUNCH`)."""
    source, coroutines = _emulation_source(kernel)
    globals_, inits, call_args = [], [], []
    for i, param in enumerate(kernel.signature):
        name = f"cunumpy_arg{i}"
        if param.view_ndim is not None:
            element = _element_type(param)
            n = param.view_ndim
            shape = ", ".join(f"m[{1 + k}]" for k in range(n))
            members = f"{{{shape}}}"
            if not param.contiguous:
                strides = ", ".join(f"m[{1 + n + k}]" for k in range(n))
                members += f", {{{strides}}}"
            globals_.append(f"static {param.ctype} {name};")
            inits.append(
                f"    {{ const long long* m = (const long long*)cunumpy_args[{i}];\n"
                f"      {name} = {param.ctype}{{({element}*)(intptr_t)m[0], {members}}}; }}",
            )
        elif param.pointer:
            element = _element_type(param)
            globals_.append(f"static {element}* {name};")
            inits.append(f"    {name} = ({element}*)cunumpy_args[{i}];")
        else:
            globals_.append(f"static {param.ctype} {name};")
            inits.append(f"    {name} = *(const {param.ctype}*)cunumpy_args[{i}];")
        call_args.append(name)
    run = _RUN_COROUTINES if coroutines else _RUN_SERIAL
    return source + _LAUNCH.format(
        globals="\n".join(globals_),
        run=run.format(call=f"{kernel.expression}({', '.join(call_args)})"),
        inits="\n".join(inits),
    )


def _compile_command(
    kernel: CudaKernel,
    compiler: str,
    options: Sequence[str],
) -> list[str]:
    """The compiler, include directories and options for the emulation of `kernel`."""
    include_dirs = [cuda_include_dir(), *map(str, kernel.include_dirs)]
    if kernel.source_dir is not None:
        include_dirs.insert(0, str(kernel.source_dir))
    defines = [o for o in kernel.options if o.startswith("-D")]
    return [
        compiler,
        "-std=c++17",
        "-w",
        *(f"-I{d}" for d in include_dirs),
        *defines,
        *options,
    ]


def emulation_cache_dir() -> Path | None:
    """Where the emulation libraries are cached on disk, or None for no disk cache.

    ``CUNUMPY_EMULATION_CACHE`` sets the directory (``0`` or empty: no disk
    cache); the default is ``$XDG_CACHE_HOME/cunumpy/emulation``, else
    ``~/.cache/cunumpy/emulation``.
    """
    value = os.environ.get("CUNUMPY_EMULATION_CACHE")
    if value is not None:
        if value.strip().lower() in {"", "0", "false", "no", "off"}:
            return None
        return Path(value).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "cunumpy" / "emulation"


class _Library:
    """A loaded emulation library; launches of it are serialized (global state)."""

    def __init__(self, path: Path) -> None:
        self._dll = ctypes.CDLL(str(path))
        self.launch = self._dll.cunumpy_launch
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_ulonglong,
        ]
        self.launch.restype = ctypes.c_int
        self.lock = threading.Lock()


# libraries loaded in this process, and one lock per key so that a kernel is
# built once even when compile_all() builds in threads
_LIBRARIES: dict[str, _Library] = {}
_BUILDING: dict[str, threading.Lock] = {}
_BUILDING_LOCK = threading.Lock()


def _build(command: list[str], source: str, out: Path, kernel: CudaKernel) -> None:
    with tempfile.TemporaryDirectory(prefix="cunumpy-emulation-") as tmp:
        cpp = Path(tmp) / "kernel.cpp"
        cpp.write_text(source)
        built = subprocess.run(
            [*command, str(cpp), "-o", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
    if built.returncode:
        raise RuntimeError(
            f"kernel {kernel.name!r} does not compile for emulation:\n{built.stderr[:4000]}",
        )


def _library(kernel: CudaKernel, compiler: str, options: Sequence[str]) -> _Library:
    """The emulation library of `kernel`: from this process, the disk cache, or built."""
    command = [*_compile_command(kernel, compiler, options), "-O1", "-shared", "-fPIC"]
    source = _library_source(kernel)
    digest = hashlib.sha256()
    for part in (source, _code(kernel), *command):
        digest.update(part.encode())
        digest.update(b"\0")
    key = digest.hexdigest()[:32]
    library = _LIBRARIES.get(key)
    if library is not None:
        return library
    with _BUILDING_LOCK:
        lock = _BUILDING.setdefault(key, threading.Lock())
    with lock:
        library = _LIBRARIES.get(key)
        if library is not None:
            return library
        cache = emulation_cache_dir()
        if cache is not None:
            try:
                cache.mkdir(parents=True, exist_ok=True)
            except OSError:
                cache = None
        if cache is not None and os.access(cache, os.W_OK):
            path = cache / f"{key}.so"
            if not path.exists():
                partial = cache / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
                try:
                    _build(command, source, partial, kernel)
                    os.replace(partial, path)
                finally:
                    partial.unlink(missing_ok=True)
            library = _Library(path)
        else:
            with tempfile.TemporaryDirectory(prefix="cunumpy-emulation-") as tmp:
                path = Path(tmp) / f"{key}.so"
                _build(command, source, path, kernel)
                library = _Library(path)  # loaded: the file may go
        _LIBRARIES[key] = library
    return library


# per kernel: whether it passed _check_supported, and its libraries by
# (compiler, options), so that a launch does not hash the source and headers
_SUPPORTED: weakref.WeakSet[CudaKernel] = weakref.WeakSet()
_PREPARED: weakref.WeakKeyDictionary[CudaKernel, dict[tuple, _Library]] = (
    weakref.WeakKeyDictionary()
)


def _check_supported_once(kernel: CudaKernel) -> None:
    if kernel not in _SUPPORTED:
        _check_supported(kernel)
        _SUPPORTED.add(kernel)


def _prepared(
    kernel: CudaKernel,
    compiler: str,
    options: Sequence[str],
    *,
    refresh: bool = False,
) -> _Library:
    """The library of `kernel`, looked up once per kernel object (or again with `refresh`)."""
    libraries = _PREPARED.setdefault(kernel, {})
    key = (compiler, tuple(options))
    if refresh or key not in libraries:
        libraries[key] = _library(kernel, compiler, options)
    return libraries[key]


def _resolve_compiler(compiler: str | None) -> str:
    compiler = compiler or emulation_compiler()
    if compiler is None:
        raise RuntimeError("emulation needs a C++ compiler (set CXX or install c++)")
    return compiler


# syntax checks that passed, for kernels without a parsed signature
_CHECKED: set[tuple[str, ...]] = set()


def _check_syntax(kernel: CudaKernel, compiler: str, options: Sequence[str]) -> None:
    command = [*_compile_command(kernel, compiler, options), "-fsyntax-only"]
    key = (_code(kernel), kernel.expression, *command)
    if key in _CHECKED:
        return
    source, _ = _emulation_source(kernel)
    # the address instantiates a kernel template
    source += f"\nstatic void cunumpy_check() {{ (void)&{kernel.expression}; }}\n"
    with tempfile.TemporaryDirectory(prefix="cunumpy-emulation-") as tmp:
        cpp = Path(tmp) / "kernel.cpp"
        cpp.write_text(source)
        built = subprocess.run(
            [*command, str(cpp)],
            capture_output=True,
            text=True,
            check=False,
        )
    if built.returncode:
        raise RuntimeError(
            f"kernel {kernel.name!r} does not compile for emulation:\n{built.stderr[:4000]}",
        )
    _CHECKED.add(key)


def compile_for_emulation(
    kernel: CudaKernel,
    *,
    compiler: str | None = None,
    options: Sequence[str] = (),
) -> None:
    """Build the emulation library of `kernel` now, without a launch.

    :func:`emulate_cuda_kernel` builds it on the first launch otherwise; this
    surfaces compile errors early, like :meth:`CudaKernel.compile
    <cunumpy.kernels.CudaKernel.compile>` does on a GPU (inside
    :func:`emulated_launches`, ``compile()`` calls this). The library is cached
    (see :func:`emulate_cuda_kernel`). A kernel without a parsed signature is
    only checked for syntax and type errors (``-fsyntax-only``).

    Parameters
    ----------
    kernel : CudaKernel
        The kernel.
    compiler : str | None
        C++ compiler; by default :func:`emulation_compiler`.
    options : Sequence[str]
        Additional compiler options, as for :func:`emulate_cuda_kernel`.

    Raises
    ------
    NotImplementedError
        If the kernel uses what the emulation cannot run (warp intrinsics).
    RuntimeError
        If there is no C++ compiler, or the kernel does not compile.
    """
    _check_supported(kernel)
    compiler = _resolve_compiler(compiler)
    if kernel.signature is None:
        _check_syntax(kernel, compiler, options)
        return
    if any(p.struct is not None for p in kernel.signature):
        kernel = _struct_wrapper(kernel)
    _prepared(kernel, compiler, options, refresh=True)


def _pack(
    params: Sequence[CudaParameter],
    args: Sequence[Any],
) -> tuple[Any, list[Any], list[tuple[np.ndarray, np.ndarray]]]:
    """The ``void*`` arguments of ``cunumpy_launch``.

    Returns the argument array, the objects it points into (to keep alive), and
    the (array, copy) pairs whose copy must be written back after the launch.
    """
    argv = (ctypes.c_void_p * max(len(params), 1))()
    keep: list[Any] = []
    write_back: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (param, value) in enumerate(zip(params, args)):
        if not (param.pointer or param.view_ndim is not None):
            scalar = _scalar_value(param, i, value)
            keep.append(scalar)
            argv[i] = scalar.ctypes.data
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"argument {i} ({param.name}) must be a NumPy array, got "
                f"{type(value).__name__}",
            )
        if param.dtype is not None and value.dtype != param.dtype:
            raise TypeError(
                f"argument {i} ({param.name}) must have dtype "
                f"{np.dtype(param.dtype)}, got {value.dtype}",
            )
        if param.view_ndim is not None and value.ndim != param.view_ndim:
            raise TypeError(
                f"argument {i} ({param.name}) must be a {param.view_ndim}D "
                f"array, got {value.ndim}D",
            )
        if param.contiguous and not value.flags.c_contiguous:
            # as on the GPU: a copy would drop what the kernel writes
            raise TypeError(
                f"argument {i} ({param.name}) must be C-contiguous for {param.ctype}",
            )
        # the kernel works on the array itself, unless it is not writeable or
        # its layout cannot be passed (a pointer needs C order, a view strides
        # in whole elements): then on a copy, written back afterwards
        strided = param.view_ndim is not None and not param.contiguous
        usable = (
            value.flags.writeable
            and value.flags.aligned
            and (
                value.flags.c_contiguous
                or (strided and all(s % value.itemsize == 0 for s in value.strides))
            )
        )
        array = value if usable else np.ascontiguousarray(value)
        if array is not value and value.flags.writeable:
            write_back.append((value, array))
        keep.append(array)
        if param.view_ndim is None:
            argv[i] = array.ctypes.data
            continue
        meta = np.array(
            [
                array.ctypes.data,
                *array.shape,
                *(s // array.itemsize for s in array.strides),
            ],
            dtype=np.int64,
        )
        keep.append(meta)
        argv[i] = meta.ctypes.data
    return argv, keep, write_back


def emulate_cuda_kernel(
    kernel: CudaKernel,
    *args: Any,
    n_threads: int | Sequence[int] | None = None,
    grid: int | Sequence[int] | None = None,
    block: int | Sequence[int] | None = None,
    compiler: str | None = None,
    options: Sequence[str] = (),
    shared_mem: int = 0,
) -> None:
    """Run `kernel` on the CPU, serially, as if it were launched with `args`.

    The kernel is compiled once (for each `compiler` and `options`) into a
    shared library that takes the arguments at run time, so later launches,
    with any argument values and array sizes, reuse it. Libraries are cached in
    the process and on disk (:func:`emulation_cache_dir`). The launch runs in
    this process, on the arrays themselves.

    Parameters
    ----------
    kernel : CudaKernel
        The kernel; its signature must be parsed (the default).
    *args
        The kernel arguments with NumPy arrays in place of CuPy arrays (arrays of
        the fake CuPy are used through their host buffer). Arrays are updated in
        place with what the kernel wrote. A struct parameter takes a mapping of
        field names to values, a :class:`~cunumpy.arguments.CudaStructValue`, or
        an object with an attribute per field.
    n_threads, grid, block
        Launch shape, as for :meth:`CudaKernel.__call__`.
    compiler : str | None
        C++ compiler; by default :func:`emulation_compiler`.
    options : Sequence[str]
        Additional compiler options, e.g. ``("-DMY_FLAG=1",)``. ``-D`` options
        of the kernel are passed on as well.
    shared_mem : int
        Dynamic shared memory per block in bytes, for ``extern __shared__``
        arrays, as in a launch.

    Raises
    ------
    NotImplementedError
        If the kernel (or a header it includes) uses warp intrinsics, or has
        complex scalars.
    TypeError
        If an argument does not match its parameter (dtype, dimensions, a
        scalar that does not fit).
    RuntimeError
        If there is no C++ compiler, the kernel does not compile, or it calls
        ``__trap()`` (a failed bounds check, inline ``asm``). Other crashes
        (e.g. a segmentation fault from an unchecked out-of-bounds index) end
        the process, as the launch runs in it.
    """
    if kernel.signature is None:
        raise TypeError(
            "emulation needs a parsed kernel signature (check_signature=True)",
        )
    _check_supported_once(kernel)
    params = kernel.signature
    if len(args) != len(params):
        raise TypeError(
            f"kernel {kernel.name!r} takes {len(params)} arguments, got {len(args)}",
        )
    if any(p.struct is not None for p in params):
        return emulate_cuda_kernel(
            _struct_wrapper(kernel),
            *_struct_fields(kernel, args),
            n_threads=n_threads,
            grid=grid,
            block=block,
            compiler=compiler,
            options=options,
            shared_mem=shared_mem,
        )
    if shared_mem < 0:
        raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
    args = tuple(_host(a) for a in args)
    compiler = _resolve_compiler(compiler)
    grid_shape, block_shape = kernel.launch_shape(
        n_threads, grid=grid, block=block, args=args
    )
    grid_shape = tuple(grid_shape) + (1,) * (3 - len(grid_shape))
    block_shape = tuple(block_shape) + (1,) * (3 - len(block_shape))
    argv, keep, write_back = _pack(params, args)
    library = _prepared(kernel, compiler, options)
    with library.lock:
        trapped = library.launch(
            argv,
            (ctypes.c_uint * 3)(*grid_shape),
            (ctypes.c_uint * 3)(*block_shape),
            int(shared_mem),
        )
    del keep
    for value, copy in write_back:
        value[...] = copy
    if trapped:
        raise RuntimeError(
            f"kernel {kernel.name!r} crashed in emulation: it called __trap() "
            "(e.g. a failed bounds check or inline asm, which is not emulated)",
        )


@contextmanager
def emulated_launches(
    *,
    compiler: str | None = None,
    options: Sequence[str] = (),
) -> Generator[None, None, None]:
    """Run every :class:`~cunumpy.kernels.CudaKernel` launch in the block on the CPU.

    With the fake CuPy (:func:`cunumpy.kernel_testing.install_fake_cupy`)
    CUDA kernels cannot run. Inside this block a launch is emulated instead
    (:func:`emulate_cuda_kernel`), on the host buffers of the fake CuPy arrays
    it is given, so code that launches kernels runs on a machine without a GPU
    and its device arrays hold the results afterwards::

        with emulated_launches():
            propagator(dt)  # CuPy backend: the CUDA kernels run on the CPU

    The launch arguments are those of :meth:`CudaKernel.__call__`
    (`stream` is ignored). Argument objects are flattened like in a launch;
    struct values and :class:`~cunumpy.arguments.CudaStructArguments` objects
    stay one argument and are read through their fields. Arrays must be arrays
    of the fake CuPy or NumPy arrays.

    No CUDA compilation happens in the block either: :meth:`CudaKernel.compile`
    (and so ``recompile()``, ``CudaKernelVariants.compile_all()`` and
    ``KernelCatalog.compile_all()``) builds the emulation library instead
    (:func:`compile_for_emulation`), so compile errors still show up before
    the first launch, and returns None instead of a ``cupy.RawKernel``. Kernels
    the emulation cannot run (warp intrinsics) are skipped by ``compile()``;
    launching them raises. This holds on a real GPU too: the block means "no
    CUDA here".

    Launches are serial, so use it on small problems. Each kernel is compiled
    once (see :func:`emulate_cuda_kernel`). The limits of
    :func:`emulate_cuda_kernel` apply.

    Parameters
    ----------
    compiler : str | None
        C++ compiler; by default :func:`emulation_compiler`.
    options : Sequence[str]
        Additional compiler options for every launch, e.g.
        ``("-ffp-contract=off",)``.
    """
    original_call = CudaKernel.__call__
    original_compile = CudaKernel.compile

    def launch(
        self: CudaKernel,
        *args: Any,
        n_threads: int | Sequence[int] | None = None,
        grid: int | Sequence[int] | None = None,
        block: int | Sequence[int] | None = None,
        shared_mem: int = 0,
        stream: Any = None,
    ) -> None:
        flat: list[Any] = []
        for arg in args:
            if isinstance(arg, (CudaStructValue, CudaStructArguments)):
                flat.append(arg)
            elif hasattr(arg, "__cuda_args__"):
                flat.extend(arg.__cuda_args__())
            else:
                flat.append(arg)
        emulate_cuda_kernel(
            self,
            *flat,
            n_threads=n_threads,
            grid=grid,
            block=block,
            compiler=compiler,
            options=options,
            shared_mem=shared_mem,
        )

    def compile(self: CudaKernel, *, log_stream: Any = None) -> None:
        try:
            compile_for_emulation(self, compiler=compiler, options=options)
        except NotImplementedError:
            pass  # GPU only; a launch raises

    CudaKernel.__call__ = launch  # type: ignore[method-assign]
    CudaKernel.compile = compile  # type: ignore[method-assign]
    try:
        yield
    finally:
        CudaKernel.__call__ = original_call  # type: ignore[method-assign]
        CudaKernel.compile = original_compile  # type: ignore[method-assign]
