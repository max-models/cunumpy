"""Deprecated alias of :mod:`cunumpy._cuda_kernel`, removed in cunumpy 0.6.

The public names are in :mod:`cunumpy.cuda`.
"""

from ._deprecated import alias_module

alias_module(__name__, "cunumpy._cuda_kernel", "cunumpy.cuda")
