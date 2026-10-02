"""Run a CUDA kernel on the CPU, one thread after another, for tests without a GPU.

A ported kernel is usually checked against its host version on a GPU
(:func:`cunumpy.testing.assert_kernels_agree`). Without one, CI cannot run that
check, and the kernel's index and weight arithmetic go untested.
:func:`emulate_cuda_kernel` closes that gap: it compiles the kernel source as
C++ with the CUDA built-ins replaced by plain C++ (``threadIdx``, ``blockIdx``,
``blockDim``, ``gridDim``, ``atomicAdd``, ...), calls the kernel once per
thread, serially, on copies of the NumPy arguments, and copies the arrays back,
so the call looks like a launch::

    from cunumpy.testing import emulate_cuda_kernel

    y = np.zeros(1000)
    emulate_cuda_kernel(axpy, 2.0, x, y, 1000, n_threads=1000)
    np.testing.assert_allclose(y, 2.0 * x)

Arguments follow the kernel signature: NumPy arrays for pointer and array view
parameters (``Array1D<T>`` to ``Array4D<T>``; any strides, they are passed as
contiguous copies), Python or NumPy scalars for scalar parameters (cast and
checked like in a launch). Arrays are written back into the given arrays.

What it does not emulate: concurrency (threads run one after another, so atomics
are plain additions and races never show), block shared memory and
``__syncthreads`` (a kernel using them is refused, since serial threads would
give wrong results), warp intrinsics (``__shfl_*``, ``__ballot_sync``, ...; not
compiled), structs and ``CudaArguments`` objects (not supported), and
``<cupy/complex.cuh>``. Use it for elementwise, gather, scatter and push
kernels; use a GPU for the rest.

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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_kernel import (
    CudaKernel,
    CudaParameter,
    _scalar_checker,
    _strip_comments,
    cuda_include_dir,
)

__all__ = ["emulate_cuda_kernel", "emulation_compiler"]

# CUDA constructs that serial emulation would get wrong
_UNSUPPORTED = {
    r"__shared__": "block shared memory",
    r"__syncthreads": "block synchronization (__syncthreads)",
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

_MAIN = r"""
int main() {{
{declarations}
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


def _check_supported(kernel: CudaKernel) -> None:
    code = _strip_comments(kernel.source)
    for pattern, what in _UNSUPPORTED.items():
        if re.search(pattern, code):
            raise NotImplementedError(
                f"kernel {kernel.name!r} uses {what}, which emulation cannot run "
                "correctly one thread at a time; test it on a GPU"
            )


def emulate_cuda_kernel(
    kernel: CudaKernel,
    *args: Any,
    n_threads: int | Sequence[int] | None = None,
    grid: int | Sequence[int] | None = None,
    block: int | Sequence[int] | None = None,
    compiler: str | None = None,
    options: Sequence[str] = (),
) -> None:
    """Run `kernel` on the CPU, serially, as if it were launched with `args`.

    Parameters
    ----------
    kernel : CudaKernel
        The kernel; its signature must be parsed (the default).
    *args
        The kernel arguments with NumPy arrays in place of CuPy arrays. Arrays
        are updated in place with what the kernel wrote.
    n_threads, grid, block
        Launch shape, as for :meth:`CudaKernel.__call__`.
    compiler : str | None
        C++ compiler; by default :func:`emulation_compiler`.
    options : Sequence[str]
        Additional compiler options, e.g. ``("-DMY_FLAG=1",)``. ``-D`` options
        of the kernel are passed on as well.

    Raises
    ------
    NotImplementedError
        If the kernel uses block shared memory, ``__syncthreads``, warp
        intrinsics, struct parameters or complex scalars.
    TypeError
        If an argument does not match its parameter (dtype, dimensions, a
        scalar that does not fit).
    RuntimeError
        If there is no C++ compiler, or the kernel does not compile or crashes.
    """
    if kernel.signature is None:
        raise TypeError(
            "emulation needs a parsed kernel signature (check_signature=True)"
        )
    _check_supported(kernel)
    params = kernel.signature
    if len(args) != len(params):
        raise TypeError(
            f"kernel {kernel.name!r} takes {len(params)} arguments, got {len(args)}"
        )
    compiler = compiler or emulation_compiler()
    if compiler is None:
        raise RuntimeError("emulation needs a C++ compiler (set CXX or install c++)")
    grid_shape, block_shape = kernel.launch_shape(n_threads, grid=grid, block=block)
    grid_shape = tuple(grid_shape) + (1,) * (3 - len(grid_shape))
    block_shape = tuple(block_shape) + (1,) * (3 - len(block_shape))

    with tempfile.TemporaryDirectory(prefix="cunumpy-emulation-") as tmp:
        tmp_path = Path(tmp)
        declarations, call_args, writes, arrays = [], [], [], []
        for i, (param, value) in enumerate(zip(params, args)):
            name = f"cunumpy_arg{i}"
            if param.struct is not None:
                raise NotImplementedError(
                    "emulation does not support struct parameters"
                )
            if param.pointer or param.view_ndim is not None:
                if not isinstance(value, np.ndarray):
                    raise TypeError(
                        f"argument {i} ({param.name}) must be a NumPy array, got "
                        f"{type(value).__name__}"
                    )
                if param.dtype is not None and value.dtype != param.dtype:
                    raise TypeError(
                        f"argument {i} ({param.name}) must have dtype "
                        f"{np.dtype(param.dtype)}, got {value.dtype}"
                    )
                if param.view_ndim is not None and value.ndim != param.view_ndim:
                    raise TypeError(
                        f"argument {i} ({param.name}) must be a {param.view_ndim}D "
                        f"array, got {value.ndim}D"
                    )
                buffer = np.ascontiguousarray(value)
                path = tmp_path / f"{name}.bin"
                buffer.tofile(path)
                ctype = param.ctype if param.dtype is not None else "unsigned char"
                element = (
                    re.match(r"Array\dD<(.*)>", ctype).group(1)
                    if param.view_ndim is not None
                    else ctype
                )
                declarations.append(
                    f"    {element}* {name} = ({element}*)malloc({max(buffer.nbytes, 1)});\n"
                    f'    cunumpy_read("{path}", {name}, {buffer.nbytes});'
                )
                if param.view_ndim is not None:
                    shape = ", ".join(f"{n}LL" for n in buffer.shape)
                    strides = ", ".join(
                        f"{s // buffer.itemsize}LL" for s in buffer.strides
                    )
                    declarations.append(
                        f"    {ctype} {name}_view{{{name}, {{{shape}}}, {{{strides}}}}};"
                    )
                    call_args.append(f"{name}_view")
                else:
                    call_args.append(name)
                out = tmp_path / f"{name}.out"
                writes.append(f'    cunumpy_write("{out}", {name}, {buffer.nbytes});')
                arrays.append((value, buffer, out))
            else:
                call_args.append(_scalar_literal(param, i, value))

        source = _STUBS + kernel.source.replace('extern "C"', "")
        source += _MAIN.format(
            declarations="\n".join(declarations),
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
                f"kernel {kernel.name!r} does not compile for emulation:\n{built.stderr[:4000]}"
            )
        ran = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
        if ran.returncode:
            raise RuntimeError(
                f"kernel {kernel.name!r} crashed in emulation (exit {ran.returncode}):\n"
                f"{ran.stdout[-2000:]}{ran.stderr[-2000:]}"
            )
        for value, buffer, out in arrays:
            result = np.fromfile(out, dtype=buffer.dtype).reshape(buffer.shape)
            value[...] = result
