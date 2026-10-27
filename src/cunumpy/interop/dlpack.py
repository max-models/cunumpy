"""Small explicit helpers around the DLPack array interchange protocol."""

from __future__ import annotations

from typing import Any

from cunumpy.interop._common import record, target_backend


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


def from_dlpack(source: Any, *, backend: str = "numpy") -> Any:
    """Construct a NumPy or CuPy array from a DLPack source.

    Parameters
    ----------
    source : object
        DLPack capsule or object implementing the protocol.
    backend : {"numpy", "cupy"}, default="numpy"
        Array library to construct.

    Returns
    -------
    object
        The constructed array.
    """
    target = target_backend(backend)
    if not hasattr(target, "from_dlpack"):
        raise TypeError(f"{backend} does not provide from_dlpack")
    result = target.from_dlpack(source)
    record(source, result)
    return result


__all__ = ["from_dlpack", "to_dlpack"]
