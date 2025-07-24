"""For"""

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Iterable, Optional, Union

import equinox as eqx
import jax 
import jax.tree as jt
from jaxtyping import PyTree


def vmap_multi(
    func: Callable, 
    in_axes_sequence: Iterable[PyTree[Union[int, Optional[Callable[[Any], int]]]]],
    out_axes_sequence: Optional[Iterable[PyTree[Union[int, Optional[Callable[[Any], int]]]]]] = None,
    vmap_func: Callable = eqx.filter_vmap,
):
    """Given a sequence of `in_axes`, construct a nested vmap of `func`.
    
    Arguments:
        func: Function to be transformed.
        in_axes_sequence: Sequence of `in_axes` specifications, as they would be passed to `jax.vmap`.
            For example, to successively map over axes 0, 1, and 2 of a single array argument, 
            pass `in_axes_sequence=(0, 0, 0)`.
        vmap_func: Transformation function to use.
    """
    func_v = func
    
    # if out_axes_sequence is None:
    #     out_axes_sequence = jt.map(lambda axis: eqx.if_array(axis=axis), in_axes_sequence)
    
    # for in_axes, out_axes in zip(in_axes_sequence, out_axes_sequence):
    #     func_v = vmap_func(func_v, in_axes=in_axes, out_axes=out_axes)

    for in_axes in in_axes_sequence:
        func_v = vmap_func(func_v, in_axes=in_axes)

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


def natural_to_sequential_axes(
    axes: Sequence[Optional[int]]
) -> list[Optional[int]]:
    """
    Given a flat sequence of natural‐index axes (ints or None),
    produce the corresponding sequential axes by subtracting,
    for each element, the number of *earlier* axes that were
    both non‐None and strictly less than it.

    Examples
    --------
    >>> natural_to_sequential([3, 0])
    [3, 0]

    >>> natural_to_sequential([4, 3, 2, 1])
    [4, 3, 2, 1]

    >>> natural_to_sequential([3, 5, 1])
    # at j=0: 3 (no preds <3)
    # at j=1: 5-1=4 (one pred 3<5)
    # at j=2: 1-0=1 (no pred<1)
    [3, 4, 1]
    """
    non_none_axes = [a for a in axes if a is not None]
    if len(non_none_axes) != len(set(non_none_axes)):
        raise ValueError(
            "Natural indices must be unique; got: "
            f"{axes!r}."
        )
    
    seq: list[Optional[int]] = []
    for j, a_j in enumerate(axes):
        if a_j is None:
            seq.append(None)
        else:
            # count earlier axes that have been peeled away
            peeled = sum(
                1
                for b in axes[:j]
                if b is not None and b < a_j
            )
            seq.append(a_j - peeled)
    return seq


class MultiVmapAxes:
    """Mark that an argument should be vmapped over *multiple* levels.

    Parameters
    ----------
    *axes
        A sequence where **each element** is an `in_axes` spec *for one
        nested vmap level*.  Elements may themselves be arbitrary
        PyTrees (dict/list/tuple) of ints/None.
    natural_indices : bool, default ``False``
        If ``True`` the elements of ``axes`` must be *flat* integers or
        ``None``.  They are interpreted in **natural** coordinates – i.e.
        axis `a_i` of the *original* array – and automatically shifted to
        sequential indices via `a_i - i`.
    """

    axes: tuple[Any, ...]
    natural_indices: bool

    def __init__(self, *axes: Any, natural_indices: bool = False):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])  # allow passing a list/tuple directly
        if natural_indices:
            # All axes must be scalar int/None
            for ax in axes:
                if not (ax is None or isinstance(ax, int)):
                    raise TypeError(
                        "Natural‑index MultiVmapAxes accepts only int/None elements; "
                        f"got element {ax!r}."
                    )
        if len(axes) == 0:
            raise ValueError("MultiVmapAxes must contain at least one axis spec")
        self.axes = tuple(axes)
        self.natural_indices = bool(natural_indices)

    def sequential_axes(self) -> list[Any]:
        """Return a list ready to be slotted per‑level into `in_axes`."""
        if not self.natural_indices:
            return list(self.axes)
        return natural_to_sequential_axes(self.axes)

    # Treat the *wrapper itself* as a PyTree **leaf** so that the recursion
    # sees it as atomic and defers to `sequential_axes` later.
    def tree_flatten(self):
        return (), None  # no children – it's a leaf

    @classmethod
    def tree_unflatten(cls, _: Any, __: Any):  # pragma: no cover – unused
        raise TypeError("MultiVmapAxes should never be unflattened by JAX")

    def __repr__(self) -> str:  # pragma: no cover
        mode = "natural" if self.natural_indices else "sequential"
        return f"MultiVmapAxes({self.axes}, mode={mode})"

    def __iter__(self):  # pragma: no cover – discourage but allow unpacking
        return iter(self.axes)


_AxisSpec = Union[int, None, MultiVmapAxes, PyTree[int]]


def _merge_child_schedules(child_schedules: Sequence[list[Any]], tmpl_container):
    """Pad child schedules so they all have equal length, then merge."""
    n_levels = max(len(s) for s in child_schedules)
    # Pad with None
    pad = lambda s: s + [None] * (n_levels - len(s))
    child_schedules = [pad(s) for s in child_schedules]
    merged_levels: list[Any] = []
    for lvl in range(n_levels):
        if isinstance(tmpl_container, dict):
            merged_levels.append({k: sched[lvl] for k, sched in zip(tmpl_container.keys(), child_schedules)})
        else:  # list or tuple
            merged_levels.append(type(tmpl_container)(sched[lvl] for sched in child_schedules))
    return merged_levels


def _expand(spec: _AxisSpec) -> list[Any]:
    """Recursively expand *any* in_axes spec into per‑level schedules."""
    if isinstance(spec, MultiVmapAxes):
        return spec.sequential_axes()

    # Containers: dict / list / tuple – recurse on each field
    if isinstance(spec, dict):
        child_scheds = {k: _expand(v) for k, v in spec.items()}
        # dict preserves order (py ≥3.7)
        merged = []
        n_levels = max(len(s) for s in child_scheds.values())
        for lvl in range(n_levels):
            merged.append({k: child_scheds[k][lvl] if lvl < len(child_scheds[k]) else None for k in child_scheds})
        return merged

    if isinstance(spec, (list, tuple)):
        child_scheds = [_expand(c) for c in spec]
        merged = _merge_child_schedules(child_scheds, spec)
        return merged

    # Leaf: int / None / callable / object
    return [spec]


def expand_axes_spec(in_axes: _AxisSpec) -> list[PyTree]:
    """Given a single spec which may include `MultiVmapAxes`, expand it into 
    a standard `in_axes_sequence` acceptable by `vmap_multi`.

    For example: 
        expand_axes_spec((MultiVmapAxes(0, 0), MultiVmapAxes(None,))) 
        ⇒ [[0, 0], [None]]
    """
    expanded = _expand(in_axes)
    # Guarantee list (could be tuple), copy for safety
    return list(expanded)