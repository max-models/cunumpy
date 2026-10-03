"""Deprecated alias of :mod:`cunumpy.kernel_testing`.

A submodule named ``testing`` replaced NumPy's ``xp.testing`` once imported, so
``xp.testing.assert_allclose`` failed after any ``import cunumpy.testing``.
Import :mod:`cunumpy.kernel_testing` instead; this alias will be removed in
cunumpy 0.6.
"""

import sys as _sys
import warnings as _warnings

from . import kernel_testing as _kernel_testing

_warnings.warn(
    "cunumpy.testing is deprecated and will be removed in cunumpy 0.6; "
    "import cunumpy.kernel_testing instead",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _kernel_testing
