---
title: "Shared kernel blocks"
description: "Small functions of scalars that the kernels of several packages share, in every backend: where they live, how each backend imports them, and the rules that keep them fast, safe and in step."
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

## One source of truth

The Pyccel version of a block is plain Python as well, so it is the reference for the
others: a test runs the hand-written CUDA (on the emulator), NumPy and Metal versions and
compares them with the Pyccel block evaluated in Python, marker by marker (mpiparticles'
`tests/test_kernels.py` does this for the Boris update). The versions cannot drift apart
without a failing test, and no translator between the languages is needed.
