"""Integration on either backend.

``cupyx.scipy`` has no ``integrate``. :func:`quad` and :func:`odeint` run
SciPy on the host, with callbacks that may compute on the device and results
on the backend of the input. See :doc:`/guides/solvers`.
"""

from cunumpy._scipy_host import odeint, quad

__all__ = ["odeint", "quad"]
