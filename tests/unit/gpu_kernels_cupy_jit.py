"""cupyx.jit kernels for the GPU tests of test_python_kernels.py (imports CuPy)."""

from cupyx import jit


@jit.rawkernel()
def scale(x, factor, out):
    i = jit.grid(1)
    if i < x.size:
        out[i] = factor * x[i]


@jit.rawkernel()
def histogram(cells, counts):
    i, j = jit.grid(2)
    if i < cells.shape[0] and j < cells.shape[1]:
        jit.atomic_add(counts, cells[i, j], 1)
