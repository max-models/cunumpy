"""Gather the columns ``order`` of a component-major array, in parallel (Pyccel)."""

from typing import TypeVar

T = TypeVar(
    "T",
    "float[:,:]",
    "float32[:,:]",  # noqa: F821
    "int[:,:]",
    "int32[:,:]",  # noqa: F821
    "bool[:,:]",
)


def take_columns(source: "T", order: "int[:]", target: "T") -> None:
    """Write ``source[:, order[i]]`` to ``target[:, i]`` for every entry of ``order``.

    The indices must lie in ``[0, source.shape[1])`` (the caller checks them) and
    ``target`` must not overlap ``source``. One loop per component, so every
    thread streams through one contiguous row.
    """
    for component in range(source.shape[0]):
        # fmt: off
        #$ omp parallel for schedule(static)
        # fmt: on
        for marker in range(order.shape[0]):
            target[component, marker] = source[component, order[marker]]
