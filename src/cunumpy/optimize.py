"""Root finding on either backend.

SciPy's optimizers run on the host only, and ``cupyx.scipy`` has no
``optimize``. This module has the parts that profit from the device: many
independent problems solved at once, each step one array operation::

    r = xp.optimize.newton(lambda r: psi(r) - levels, r0)

See :doc:`/guides/solvers`.
"""

from cunumpy._optimize import NewtonResult, newton

__all__ = ["NewtonResult", "newton"]
