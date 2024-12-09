"""Tools which did not belong any particular other place.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)
import copy
import difflib
import dis
from functools import wraps
import inspect
from itertools import zip_longest, chain
import logging
from operator import attrgetter
import os
from pathlib import Path, PosixPath
from shutil import rmtree
import subprocess
import textwrap
from time import perf_counter
from types import ModuleType
from typing import Any, Optional, Tuple, TypeAlias, TypeVar, Union

import equinox as eqx
from equinox import Module
from equinox._pretty_print import tree_pp, bracketed
import jax
import jax.numpy as jnp
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import jax.tree as jt
from jaxtyping import Float, Array, PyTree, Shaped

from feedbax._progress import _tqdm_write


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


def get_unique_label(label: str, invalid_labels: Sequence[str] | Set[str]) -> str:
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






