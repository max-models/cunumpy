"""Run a CUDA kernel on the CPU, one thread after another, for tests without a GPU.

A ported kernel is usually checked against its host version on a GPU
(:func:`cunumpy.kernel_testing.assert_kernels_agree`). Without one, CI cannot run that
check, and the kernel's index and weight arithmetic go untested.
:func:`emulate_cuda_kernel` closes that gap: it compiles the kernel source as
C++ with the CUDA built-ins replaced by plain C++ (``threadIdx``, ``blockIdx``,
``blockDim``, ``gridDim``, ``atomicAdd``, ...), calls the kernel once per
thread, serially, on copies of the NumPy arguments, and copies the arrays back,
so the call looks like a launch::

    from cunumpy.kernel_testing import emulate_cuda_kernel

    y = np.zeros(1000)
    emulate_cuda_kernel(axpy, 2.0, x, y, 1000, n_threads=1000)
    np.testing.assert_allclose(y, 2.0 * x)

Arguments follow the kernel signature: NumPy arrays for pointer and array view
parameters (``Array1D<T>`` to ``Array16D<T>``; any strides, they are passed as
contiguous copies), Python or NumPy scalars for scalar parameters (cast and
checked like in a launch). Arrays are written back into the given arrays.

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
in a block run through the emulation, on the arrays of the fake CuPy
(:mod:`cunumpy._fake_cupy`), so that code that launches kernels (a
:class:`~cunumpy.kernels.Kernel` on the CuPy backend, a propagator) can be
tested without a GPU.

What it does not emulate: concurrency between the barriers (threads run one
after another, so atomics are plain additions and races never show), warp
intrinsics (``__shfl_*``, ``__syncwarp``, ``__ballot_sync``, ...; a kernel or an
included header using them is refused), and ``<cupy/complex.cuh>``. Use a GPU
for those.

Floating point: like NVRTC (``--fmad=true`` by default) the C++ compiler may
fuse ``a * b + c`` into one fused multiply-add, so results can differ from
NumPy's in the last bit; compare with a tolerance of a few ulp, or pass
``options=("-ffp-contract=off",)`` for NumPy's rounding.

It needs a C++17 compiler: ``CXX`` from the environment, else ``c++``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
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

__all__ = ["emulate_cuda_kernel", "emulated_launches", "emulation_compiler"]

# CUDA constructs that serial emulation would get wrong
_UNSUPPORTED = {
    r"__syncwarp": "warp synchronization (__syncwarp)",
    r"__shfl\w*": "warp shuffles",
    r"__ballot_sync|__any_sync|__all_sync": "warp votes",
}

_STUBS = r"""
// --- cunumpy emulation: CUDA built-ins as plain C++, one thread at a time ---
#include <math.h>
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
inline void __trap() { abort(); }
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
static void cunumpy_read(const char* path, void* data, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f || fread(data, 1, bytes, f) != bytes) { perror(path); exit(2); }
    fclose(f);
}
static void cunumpy_write(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes) { perror(path); exit(2); }
    fclose(f);
}
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
"""

_MAIN_SERIAL = r"""
{globals}
int main() {{
{inits}
    cunumpy_dynamic_shared = calloc({shared_mem} + 16, 1);
    gridDim = {{{gx}u, {gy}u, {gz}u}};
    blockDim = {{{bx}u, {by}u, {bz}u}};
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
{writes}
    return 0;
}}
"""

_MAIN_COROUTINES = r"""
{globals}
static void cunumpy_entry() {{
    {call};
    cunumpy_threads[cunumpy_current].done = 1;
}}
int main() {{
{inits}
    cunumpy_dynamic_shared = calloc({shared_mem} + 16, 1);
    gridDim = {{{gx}u, {gy}u, {gz}u}};
    blockDim = {{{bx}u, {by}u, {bz}u}};
    const unsigned int n = blockDim.x * blockDim.y * blockDim.z;
    cunumpy_threads = (cunumpy_thread*)calloc(n, sizeof(cunumpy_thread));
    for (unsigned int t = 0; t < n; ++t)
        cunumpy_threads[t].stack = (char*)malloc(CUNUMPY_STACK_BYTES);
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
                if (!cunumpy_threads[t].done) running = 1;
            }}
        }}
    }}
{writes}
    return 0;
}}
"""


def emulation_compiler() -> str | None:
    """The C++ compiler :func:`emulate_cuda_kernel` uses, or None if there is none."""
    compiler = os.environ.get("CXX")
    if compiler and shutil.which(compiler):
        return shutil.which(compiler)
    return shutil.which("c++")


def _scalar_literal(param: CudaParameter, index: int, value: Any) -> str:
    """A C++ literal of `value`, checked and cast like a scalar kernel argument."""
    cast = _scalar_checker(param, index)(value)
    kind = np.dtype(param.dtype).kind
    if kind == "b":
        return "true" if cast else "false"
    if kind in "iu":
        return f"(({param.ctype}){int(cast)}{'ULL' if kind == 'u' else 'LL'})"
    if kind == "f":
        return f"(({param.ctype}){float(cast).hex()})"  # exact
    raise NotImplementedError(f"emulation does not support {param.ctype} scalars")


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


def _flatten_structs(
    kernel: CudaKernel,
    args: Sequence[Any],
) -> tuple[CudaKernel, list[Any]]:
    """A kernel that takes every struct field as a parameter, and its arguments.

    The wrapper kernel (the source of `kernel` plus a ``__global__`` function with
    one parameter per field) rebuilds the structs and calls `kernel`, so that the
    emulation only has to deal with arrays and scalars.
    """
    params, builds, call, flat = [], [], [], []
    for p, value in zip(kernel.signature, args):
        if p.struct is None:
            params.append(_declaration(p, p.name))
            call.append(p.name)
            flat.append(value)
            continue
        names = [f"{p.name}__{f.name}" for f in p.struct.fields]
        params += [_declaration(f, n) for f, n in zip(p.struct.fields, names)]
        builds.append(f"    {p.struct.name} {p.name}_struct{{{', '.join(names)}}};")
        call.append(f"{p.name}_struct")
        flat += [_field_value(p.name, value, f) for f in p.struct.fields]
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
    return wrapper, flat


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
        If there is no C++ compiler, or the kernel does not compile or crashes.
    """
    if kernel.signature is None:
        raise TypeError(
            "emulation needs a parsed kernel signature (check_signature=True)",
        )
    _check_supported(kernel)
    params = kernel.signature
    if len(args) != len(params):
        raise TypeError(
            f"kernel {kernel.name!r} takes {len(params)} arguments, got {len(args)}",
        )
    if any(p.struct is not None for p in params):
        wrapper, flat = _flatten_structs(kernel, args)
        return emulate_cuda_kernel(
            wrapper,
            *flat,
            n_threads=n_threads,
            grid=grid,
            block=block,
            compiler=compiler,
            options=options,
            shared_mem=shared_mem,
        )
    args = tuple(_host(a) for a in args)
    compiler = compiler or emulation_compiler()
    if compiler is None:
        raise RuntimeError("emulation needs a C++ compiler (set CXX or install c++)")
    grid_shape, block_shape = kernel.launch_shape(
        n_threads, grid=grid, block=block, args=args
    )
    grid_shape = tuple(grid_shape) + (1,) * (3 - len(grid_shape))
    block_shape = tuple(block_shape) + (1,) * (3 - len(block_shape))

    with tempfile.TemporaryDirectory(prefix="cunumpy-emulation-") as tmp:
        tmp_path = Path(tmp)
        globals_, inits, call_args, writes, arrays = [], [], [], [], []
        for i, (param, value) in enumerate(zip(params, args)):
            name = f"cunumpy_arg{i}"
            if param.pointer or param.view_ndim is not None:
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
                        f"argument {i} ({param.name}) must be C-contiguous for "
                        f"{param.ctype}",
                    )
                buffer = np.ascontiguousarray(value)
                path = tmp_path / f"{name}.bin"
                buffer.tofile(path)
                ctype = param.ctype if param.dtype is not None else "unsigned char"
                element = (
                    re.match(r"C?Array\d+D<(.*)>", ctype).group(1)
                    if param.view_ndim is not None
                    else ctype
                )
                globals_.append(f"static {element}* {name};")
                inits.append(
                    f"    {name} = ({element}*)malloc({max(buffer.nbytes, 1)});\n"
                    f'    cunumpy_read("{path}", {name}, {buffer.nbytes});',
                )
                if param.view_ndim is not None:
                    shape = ", ".join(f"{n}LL" for n in buffer.shape)
                    strides = ", ".join(
                        f"{s // buffer.itemsize}LL" for s in buffer.strides
                    )
                    members = f"{{{shape}}}"
                    if not param.contiguous:
                        members += f", {{{strides}}}"
                    globals_.append(f"static {ctype} {name}_view;")
                    inits.append(f"    {name}_view = {ctype}{{{name}, {members}}};")
                    call_args.append(f"{name}_view")
                else:
                    call_args.append(name)
                out = tmp_path / f"{name}.out"
                writes.append(f'    cunumpy_write("{out}", {name}, {buffer.nbytes});')
                arrays.append((value, buffer, out))
            else:
                call_args.append(_scalar_literal(param, i, value))

        if shared_mem < 0:
            raise ValueError(f"shared_mem must be non-negative, got {shared_mem}")
        kernel_source = kernel.source.replace('extern "C"', "")
        kernel_source = _EXTERN_SHARED.sub(
            r"\1* \2 = (\1*)cunumpy_dynamic_shared;",
            kernel_source,
        )
        coroutines = "__syncthreads" in _code(kernel)
        if coroutines:
            # ucontext needs _XOPEN_SOURCE before any system header
            source = "#define _XOPEN_SOURCE 700\n" + _STUBS + _COROUTINES
            main = _MAIN_COROUTINES
        else:
            source = _STUBS + "inline void __syncthreads() {}\n"
            main = _MAIN_SERIAL
        source += kernel_source
        source += main.format(
            globals="\n".join(globals_),
            inits="\n".join(inits),
            shared_mem=int(shared_mem),
            gx=grid_shape[0],
            gy=grid_shape[1],
            gz=grid_shape[2],
            bx=block_shape[0],
            by=block_shape[1],
            bz=block_shape[2],
            call=f"{kernel.expression}({', '.join(call_args)})",
            writes="\n".join(writes),
        )
        cpp = tmp_path / "kernel.cpp"
        cpp.write_text(source)
        exe = tmp_path / "kernel"
        include_dirs = [cuda_include_dir(), *map(str, kernel.include_dirs)]
        if kernel.source_dir is not None:
            include_dirs.insert(0, str(kernel.source_dir))
        defines = [o for o in kernel.options if o.startswith("-D")]
        command = [
            compiler,
            "-std=c++17",
            "-O1",
            "-w",
            *(f"-I{d}" for d in include_dirs),
            *defines,
            *options,
            str(cpp),
            "-o",
            str(exe),
        ]
        built = subprocess.run(command, capture_output=True, text=True, check=False)
        if built.returncode:
            raise RuntimeError(
                f"kernel {kernel.name!r} does not compile for emulation:\n{built.stderr[:4000]}",
            )
        ran = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
        if ran.returncode:
            raise RuntimeError(
                f"kernel {kernel.name!r} crashed in emulation (exit {ran.returncode}):\n"
                f"{ran.stdout[-2000:]}{ran.stderr[-2000:]}",
            )
        for value, buffer, out in arrays:
            result = np.fromfile(out, dtype=buffer.dtype).reshape(buffer.shape)
            value[...] = result


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

    Launches are serial and slow (each one compiles the kernel as C++), so use it
    on small problems. The limits of :func:`emulate_cuda_kernel` apply.

    Parameters
    ----------
    compiler : str | None
        C++ compiler; by default :func:`emulation_compiler`.
    options : Sequence[str]
        Additional compiler options for every launch, e.g.
        ``("-ffp-contract=off",)``.
    """
    original = CudaKernel.__call__

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

    CudaKernel.__call__ = launch  # type: ignore[method-assign]
    try:
        yield
    finally:
        CudaKernel.__call__ = original  # type: ignore[method-assign]
