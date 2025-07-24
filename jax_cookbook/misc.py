from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
    Set,
)
import copy
from functools import wraps
from itertools import zip_longest, chain
import logging
from typing import Any, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class StrAlwaysLT(str):

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    # def __repr__(self):
    #     return self.replace("'", "")


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)


def unzip2(xys: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples.

    Taken from `jax._src.util`.
    """
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    xs: MutableSequence[T1] = []
    ys: MutableSequence[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def get_unique_label(label: str, invalid_labels: Union[Sequence[str], Set[str]]) -> str:
    """Get a unique string from a base string, while avoiding certain strings.

    Simply appends consecutive integers to the string until a unique string is
    found.
    """
    i = 0
    label_ = label
    while label_ in invalid_labels:
        label_ = f"{label}_{i}"
        i += 1
    return label_


def unique_generator(
    seq: Sequence[T1],
    replace_duplicates: bool = False,
    replace_value: Any = None
) -> Iterable[Optional[T1]]:
    """Yields the first occurrence of sequence entries, in order.

    If `replace_duplicates` is `True`, replaces duplicates with `replace_value`.
    """
    seen = set()
    for item in seq:
        if id(item) not in seen:
            seen.add(id(item))
            yield item
        elif replace_duplicates:
            yield replace_value


def nested_dict_update(dict_, *args, make_copy: bool = True):
    """Source: https://stackoverflow.com/a/3233356/23918276"""
    if make_copy:
        dict_ = copy.deepcopy(dict_)
    for arg in args:
        for k, v in arg.items():
            if isinstance(v, Mapping):
                dict_[k] = nested_dict_update(
                    dict_.get(k, type(v)()),
                    v,
                    make_copy=make_copy,
                )
            else:
                dict_[k] = v
    return dict_


def crop_to_shortest(*, axis: int):
    """Decorator that equalises the length of all array arguments along *axis*.

    When the wrapped function is called, each positional argument that has a
    ``shape`` attribute is inspected; the minimum length along *axis* is
    computed, and every such argument is sliced to this length with
    ``jax.lax.slice_in_dim``.  Non-array arguments (scalars, objects, etc.) are
    passed through untouched.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine minimal size along the axis among array args.
            sizes = [a.shape[axis] for a in args if hasattr(a, "shape")]
            if not sizes:
                # No array arguments; nothing to crop.
                return func(*args, **kwargs)

            min_len = min(sizes)

            def _crop(a): 
                if hasattr(a, "shape") and a.shape[axis] != min_len:
                    return jax.lax.slice_in_dim(a, 0, min_len, axis=axis)
                return a

            cropped_args = tuple(_crop(a) for a in args)
            return func(*cropped_args, **kwargs)

        return wrapper

    return decorator