"""Deprecated alias of :mod:`cunumpy.kernel_testing`, removed in cunumpy 0.6.

A submodule named ``testing`` replaced NumPy's ``xp.testing`` once imported, so
``xp.testing.assert_allclose`` failed after any ``import cunumpy.testing``.
"""

from cunumpy._deprecated import alias_module

alias_module(__name__, "cunumpy.kernel_testing", "cunumpy.kernel_testing")
