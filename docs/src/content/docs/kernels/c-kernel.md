---
title: "C host kernels"
description: "A <name>_c.c next to the Pyccel version, for loops the compiler must vectorise."
---

A kernel folder may hold a C version of its host kernel, `<name>_c.c`, next to
the Pyccel one. Use it for the loops that decide a program's speed and that
the compiler must vectorise: Pyccel's generated C reads arrays through generic
strided accessors, which stops most vectorisation. In picnic's 3D particle push
(trilinear gather of E and B, Boris rotation, drift; one million markers sorted
by cell, one core with AVX-512) a scalar loop took 57 ns per marker, the Pyccel
version with `#$ omp simd` and `-O3 -march=native` about 20 ns, and the C version
with `#pragma omp simd` 9.9 ns.

## The function

The C function takes the arguments of the Pyccel version, in the same order,
and returns `void`:

| Pyccel annotation | C parameters |
| --- | --- |
| `float` | `double` |
| `int`, `int64` | `int64_t` |
| `int32` | `int32_t` |
| `bool` | `bool` |
| `float[:,:]` | `double *x, const int64_t *x_shape` |
| `int[:]` | `int64_t *x, const int64_t *x_shape` |
| `int32[:,:,:]` | `int32_t *x, const int64_t *x_shape` |
| `bool[:]` | `bool *x, const int64_t *x_shape` |

Arrays arrive C-contiguous, with their shape; a call with a non-contiguous
array, another element type or another number of dimensions raises
`TypeError`.

```c
// my_sim/kernels/axpy/axpy_c.c, next to axpy_pyccel.py:
//   def axpy(x: "float[:]", y: "float[:]", a: float) -> None
#include <stdint.h>

void axpy(double *x, const int64_t *x_shape, double *y, const int64_t *y_shape,
          double a) {
    (void)y_shape;
    #pragma omp simd
    for (int64_t i = 0; i < x_shape[0]; ++i) y[i] += a * x[i];
}
```

## Building and dispatch

`KernelPackage` passes its `c_cache` (a `CCache`) to `Kernel.from_folder` as
`compile_c`, so a folder with `<name>_c.c` gets the implementation `"c"`, which a
call prefers to `"pyccel"`. On first use the source is compiled, with a
generated CPython wrapper, into one extension module and kept in the package's
kernel cache under `c/`; a call then costs about as much as a Pyccel one
(0.3 µs). If the build fails, or there is no compiler, `"c"` is unavailable
(the reason is in `kernel.host_kernel.kernel.errors`) and the call runs the
Pyccel version. `set_host_kernel_implementation("pyccel")` chooses it
explicitly.

- `#include "..."` finds headers next to the source and in the folder's
  `include_dirs` (the source roots, as for the CUDA versions); the headers are
  part of the build key.
- Flags: `-O3 -march=native` and OpenMP by default; `<PREFIX>_C_FLAGS`
  replaces the first two, `<PREFIX>_OPENMP=0` drops OpenMP, `CC` names the
  compiler. With `-march=native` the build key includes the CPU, so a cache
  shared by different machines holds one build per CPU.
- Keep the Pyccel version: it is the reference the tests compare against, and
  the fallback where no C compiler is available (Pyodide).
