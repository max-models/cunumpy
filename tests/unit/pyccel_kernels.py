"""Example kernels written for compilation with pyccel.

This module is plain Python annotated with pyccel's array type hints, so it can
be imported and run as-is, or compiled to C/Fortran with
``pyccel.epyccel(pyccel_kernels, language="c")``. The tests in
`test_pyccel_kernel.py` compile it and drive the compiled kernels through
:class:`cunumpy.PyccelKernel`.
"""


def axpy(a: float, x: "float[:]", y: "float[:]", out: "float[:]"):
    """Write ``a * x + y`` into `out` (in-place, no return value)."""
    for i in range(x.shape[0]):
        out[i] = a * x[i] + y[i]


def scale_inplace(x: "float[:]", factor: float):
    """Multiply `x` by `factor`, in place."""
    for i in range(x.shape[0]):
        x[i] = x[i] * factor


def dot(x: "float[:]", y: "float[:]") -> float:
    """Return the dot product of `x` and `y` (a scalar return value)."""
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
    return result


def matvec(mat: "float[:,:]", vec: "float[:]", out: "float[:]"):
    """Write ``mat @ vec`` into `out` (2D input, in-place output)."""
    for i in range(mat.shape[0]):
        acc = 0.0
        for j in range(mat.shape[1]):
            acc += mat[i, j] * vec[j]
        out[i] = acc
