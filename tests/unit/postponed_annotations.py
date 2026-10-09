"""Pyccel-style annotations in a module with postponed evaluation (PEP 563).

Here the quoted annotation ``'float[:, :]'`` is stored as the string
``"'float[:, :]'"``, quotes included.
"""

from __future__ import annotations

from typing import Final


class Args:
    def __init__(
        self,
        markers: "float[:, :]",  # noqa: UP037 (the quotes are what is tested)
        weights: Final["float[:]"],  # noqa: UP037
        n: int,
        dt: float,
    ): ...
