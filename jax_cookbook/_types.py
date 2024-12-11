from typing import Any 

from equinox import Module


def is_module(element: Any) -> bool:
    """Return `True` if `element` is an Equinox module."""
    return isinstance(element, Module)


is_none = lambda x: x is None