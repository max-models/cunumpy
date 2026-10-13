"""Root finding on either backend.

SciPy's optimizers run on the host only, and ``cupyx.scipy`` has no
``optimize``. :func:`newton` runs on the device: many independent problems
solved at once, each step one array operation::

    r = xp.optimize.newton(lambda r: psi(r) - levels, r0)

:func:`fsolve`, :func:`root` and :func:`minimize` run SciPy on the host, with
callbacks that get their arrays on the backend of ``x0`` (so code written with
``xp`` that uses device data works unchanged) and results on that backend.

See :doc:`/guides/solvers`.
"""

from cunumpy._optimize import NewtonResult, newton
from cunumpy._scipy_host import fsolve, minimize, root

__all__ = ["NewtonResult", "fsolve", "minimize", "newton", "root"]
