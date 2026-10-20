"""Gather the columns ``order`` of a component-major array, in NumPy."""

import numpy as np


def take_columns(source: np.ndarray, order: np.ndarray, target: np.ndarray) -> None:
    """Write ``source[:, order[i]]`` to ``target[:, i]`` for every entry of ``order``.

    Parameters
    ----------
    source : numpy.ndarray
        The component-major array, ``(ncomp, n)``.
    order : numpy.ndarray
        The column indices to gather, ``(m,)``, within ``[0, n)``.
    target : numpy.ndarray
        Receives the gathered columns, ``(ncomp, m)``; must not overlap ``source``.
    """
    # one contiguous row at a time: faster than a 2-D gather
    for component in range(source.shape[0]):
        target[component] = source[component][order]
