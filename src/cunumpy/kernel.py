"""Deprecated alias of :mod:`cunumpy._kernel`, removed in cunumpy 0.6.

The public names are in :mod:`cunumpy.kernels`.
"""

from cunumpy._deprecated import alias_module

alias_module(__name__, "cunumpy._kernel", "cunumpy.kernels")
