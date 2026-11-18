---
title: "Shared kernel blocks"
description: "Small functions of scalars that the kernels of several packages share, in every backend: where they live, how each backend imports them, the rules that keep them fast and safe, and the generated versions."
---

A fused kernel does several steps in one pass over the markers: it gathers the field at a
marker, pushes it and tests it against the walls, keeping everything in registers. The loop of
such a kernel belongs to the code that runs it. The arithmetic inside it does not: the Boris
update, the cell of a position on a grid or a polygon hit test is the same in every code. A
**block** is that arithmetic, written once in the package that owns it and used by the kernels
of every package, in every backend.

## Where a block lives

A block lives in the package that owns its physics or numerics (in the max-models packages:
the particle steps in mpiparticles, the cell and weights of a grid in mpifield, the hit tests
in simgeometry), next to its kernels, in one file per backend:

| Backend | File | How a kernel of another package uses it |
| --- | --- | --- |
| Pyccel | `<package>/kernels/<block>_pyccel.py` | `from <package>.kernels.<block>_pyccel import <function>`; `PyccelCache` builds the module with the kernel ([Code shared between kernels](/cunumpy/kernels/dispatch/#code-shared-between-kernels)) |
| CUDA | `<package>/kernels/<block>.cuh` | `#include "<package>/kernels/<block>.cuh"`, with `include_dirs=xp.kernels.source_roots("<package>")` |
| Metal | a string in `<package>/kernels/<block>.py` | `header=` of `KERNEL_PACKAGE.metal_kernel(...)` |
| NumPy | a function in `<package>/kernels/<block>.py` | an ordinary import |

The fused kernels stay in the codes. A fused combination moves into a package only when a
second code needs exactly that combination.

## Rules

- **Scalars in, scalars out, the same arithmetic everywhere.** A block takes and returns
  numbers, never arrays, and every backend computes the same expressions in the same order, so
  the CUDA and Pyccel versions agree to round-off (and usually bit for bit).
- **Pyccel blocks without local variables.** An `@inline` function is inlined by Pyccel, but
  its local variables are declared at the top of the calling function, outside the `private`
  list of an `#$ omp parallel for`: the threads would share them. Write an `@inline` block as
  one expression per result (conditional expressions for branches); the C compiler computes
  repeated subexpressions once.
- **Blocks with loops are plain functions,** not `@inline` (the polygon hit test): a call keeps
  its locals on the stack, so it is safe under OpenMP. It costs a call per use, which matters
  for a small block in a tight loop; measure it.
- **No `min` and `max` in a Pyccel block:** Pyccel turns them into calls of variadic C
  functions that the compiler cannot inline. Use conditional expressions (and tell ruff:
  `# noqa: FURB136`).
- **A zero term is `-0.0`, not `0.0`.** A block called with a constant argument is inlined with
  it: `x + 0.0` is not an identity in floating point (it turns `-0.0` into `+0.0`), so the
  compiler keeps the addition; `x + -0.0` is folded away. In picnic's r–z push this was a 10%
  difference.
- **Metal is float32.** The Apple GPU has no float64: compute differences (positions relative
  to a cell, displacements) on the host in float64 and keep the float64 values there.
- **Test every fused kernel against the plain path.** In a case where the fused steps are
  exact (uniform fields, so the gather is exact), a fused kernel must give the result of the
  unfused steps of the packages (gather, then push) to round-off, on every backend CI can run:
  Pyccel, NumPy, CUDA on the [emulator](/cunumpy/kernels/testing/), CuPy on the fake CuPy.
- **Fuse only where a profile asks for it,** and profile a change to a block in the kernels that
  use it (3-5 runs per side).

## Generated versions: `KernelBlock`

The Pyccel version of an `@inline` block is enough to write the others: `KernelBlock` translates
it to a CUDA `__device__` function, a Metal function and a NumPy function.

```python
import cunumpy as xp
from mpiparticles.kernels.boris_pyccel import boris_update

block = xp.kernels.KernelBlock(boris_update)
block.cuda("mpiparticles_boris")  # source of a __device__ function (double precision)
block.metal()  # source of a Metal function (float32), for header=
update = block.numpy()  # a NumPy function, elementwise on arrays
vx, vy, vz = update(vx, vy, vz, kx, ky, kz, tx, ty, tz)
```

The block is one `return` of an expression or a tuple of expressions over scalar arguments
(`float`, `int`, `bool`): arithmetic, comparisons (also chained), `and`, `or`, `not`,
conditional expressions, `int(x)` (truncation, as in Python and C), `float(x)`, `sqrt` and
`abs`. One result is the return value; several are written through pointers (CUDA) or
`thread` references (Metal) after the arguments. Subexpressions used more than once are
computed once into temporaries (NumPy does not merge them by itself). `//` of two integers is
refused, since Python floors and C truncates.

On mpiparticles' Boris update the generated CUDA function gives the same results bit for bit
as the hand-written `boris.cuh` (on the emulator), the NumPy function the same results to
3e-15 at the same speed, and the Metal function the same results to float32 precision (through
a C++ stand-in for Metal).
