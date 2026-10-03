"""Module aliases kept for one release after a module was renamed."""

import importlib
import sys
import warnings


def alias_module(old: str, new: str, use: str) -> None:
    """Make the module `old` an alias of `new`, with a ``DeprecationWarning``.

    Called from the module `old` itself, which is then replaced in
    ``sys.modules`` by `new`; `use` is the module to import instead.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in cunumpy 0.6; import {use} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules[old] = importlib.import_module(new)
