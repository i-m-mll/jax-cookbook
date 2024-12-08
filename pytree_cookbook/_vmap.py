"""For"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax 
from jaxtyping import PyTree


def vmap_multi(
    func: Callable, 
    in_axes_sequence: Sequence[PyTree[int | Callable[[Any], int] | None]],
    vmap_func: Callable = eqx.filter_vmap,
):
    """Given a sequence of `in_axes`, construct a nested vmap of `func`."""
    func_v = func
    for ax in in_axes_sequence:
        func_v = vmap_func(func_v, in_axes=ax)
    return func_v


def unkwarg_key(func):
    """Converts a final `key` kwarg into an initial positional arg.

    This is useful because many Equinox modules take `key` as a kwarg, and transformations 
    such as `equinox.filter_vmap` don't like kwargs -- but sometimes we want to transform over `key`
    anyway.
    """
    @wraps(func)
    def wrapper(key, *args):
        return func(*args, key=key)
    return wrapper
