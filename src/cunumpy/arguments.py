"""Argument objects for CUDA kernels: flattened groups and C structs.

A host kernel (e.g. compiled with Pyccel) and its CUDA version each take their
own argument objects. Write the host argument class, and a
:class:`CudaStructArguments` with the same constructor and attributes for the
CUDA kernels; the code that owns the arrays builds the one for its backend.
Kernels receive argument objects as they are; cunumpy never converts one form
into the other::

    class CudaMarkerArguments(xp.arguments.CudaStructArguments):
        struct_name = "MarkerArgs"
        fields = (("markers", "Array2D<double>"), ("n_markers", "int"))

        def __init__(self, markers, n_markers):
            self.markers = markers
            self.n_markers = n_markers
            self.pack()

:class:`CudaArguments` flattens a group into several kernel parameters,
:class:`CudaStruct` defines a C struct passed by value, and
:func:`write_cuda_header` writes struct definitions to a header. The kernel
classes are in :mod:`cunumpy.kernels`. See :doc:`/kernels/arguments`.
"""

from cunumpy._cuda_kernel import (
    CudaArguments,
    CudaStruct,
    CudaStructArguments,
    CudaStructValue,
    write_cuda_header,
)

__all__ = [
    "CudaArguments",
    "CudaStruct",
    "CudaStructArguments",
    "CudaStructValue",
    "write_cuda_header",
]
