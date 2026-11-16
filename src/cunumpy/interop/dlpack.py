"""Small explicit helpers around the DLPack array interchange protocol."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import check_copy, finish, target_backend


def to_dlpack(array: Any) -> Any:
    """Return a DLPack capsule or protocol object for `array`.

    Parameters
    ----------
    array : object
        Array implementing the DLPack protocol.

    Returns
    -------
    object
        The DLPack capsule or protocol object.
    """
    try:
        return array.__dlpack__()
    except AttributeError as error:
        raise TypeError(
            f"{type(array).__name__} does not implement the DLPack protocol"
        ) from error


def from_dlpack(
    source: Any, *, backend: str = "numpy", copy: bool | None = None
) -> Any:
    """Construct a NumPy or CuPy array from a DLPack source.

    Parameters
    ----------
    source : object
        DLPack capsule or object implementing the protocol.
    backend : {"numpy", "cupy"}, default="numpy"
        Array library to construct.
    copy : bool or None, optional
        None: share memory when possible; False: raise when a copy is
        needed; True: always copy. Passed on to the target's ``from_dlpack``.

    Returns
    -------
    object
        The constructed array.
    """
    check_copy(copy)
    target = target_backend(backend)
    if not hasattr(target, "from_dlpack"):
        raise TypeError(f"{backend} does not provide from_dlpack")
    kwargs = {} if copy is None else {"copy": copy}
    result = target.from_dlpack(source, **kwargs)
    return finish(
        source, result, copy=None, target=backend, duplicate=lambda array: array.copy()
    )


__all__ = ["from_dlpack", "to_dlpack"]
