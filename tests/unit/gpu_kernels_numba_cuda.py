"""numba-cuda kernels for the GPU tests of test_python_kernels.py (imports numba.cuda)."""

from numba import cuda


@cuda.jit
def scale(x, factor, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = factor * x[i]


@cuda.jit
def histogram(cells, counts):
    i, j = cuda.grid(2)
    if i < cells.shape[0] and j < cells.shape[1]:
        cuda.atomic.add(counts, cells[i, j], 1)
