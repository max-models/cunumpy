"""Deprecated alias of :mod:`cunumpy._dispatch`, removed in cunumpy 0.6.

The public names are in :mod:`cunumpy.kernels`.
"""

from cunumpy._deprecated import alias_module

alias_module(__name__, "cunumpy._dispatch", "cunumpy.kernels")
