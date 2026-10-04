"""Moving data between host and device without stalling, on either backend.

:class:`HostStaging` copies device arrays to the host in the background (for
output), and :class:`DeviceMirror` pairs a host buffer owned by another
library with a device copy::

    import cunumpy as xp

    staging = xp.memory.HostStaging(rho.shape, rho.dtype)
    copy = staging.copy(rho)          # returns at once
    ...
    h5file["rho"] = copy.result()     # a NumPy array

On the NumPy backend both work on host arrays only, so the code is the same on
both backends. Page-locked host memory (:func:`cunumpy.cuda.pin_memory`) is in
:mod:`cunumpy.cuda`.
"""

from cunumpy._mirror import DeviceMirror
from cunumpy._staging import HostStaging, StagedCopy

__all__ = ["DeviceMirror", "HostStaging", "StagedCopy"]
