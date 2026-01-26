from __future__ import annotations

"""Utilities for building *nested* `in_axes_sequence`s from specs that may
contain `MultiVmapAxes` wrappers.

Only **two public symbols** are provided:

* **`MultiVmapAxes`** – a lightweight wrapper that marks *one argument’s*
  axis‑mapping schedule when *multiple* nested ``vmap`` calls are
  desired.
* **`expand_in_axes_spec`** – converts a single JAX `in_axes`‐style spec
  (possibly containing nested dicts/tuples **and** `MultiVmapAxes`
  leaves) into a **list of per‑level specs** suitable for feeding into
  existing `vmap_multi`.

Design highlights
-----------------
* If *no* `MultiVmapAxes` objects are encountered, the returned list has
  length 1 – i.e. you get a **single vmap**.
* Each `MultiVmapAxes` can be created in **sequential** or **natural**
  mode.  Natural mode (`natural_indices=True`) only accepts *flat*
  integer/``None`` axes and converts them to sequential by subtracting
  the level index (axis _i_ → `axis_i - i`).
* In sequential mode you may pass **arbitrary PyTree prefixes** for each
  level, e.g.::

      MultiVmapAxes(
          {"foo": 0, "bar": 1},           # level 0
          {"foo": None, "bar": 2},        # level 1
      )

  thereby mapping different sub‑fields at different vmap levels.

* The recursion pads with `None` so that every argument sees the same
  number of levels.  Leading all‑`None` levels are *not* dropped; that’s
  left to the caller (often noop, but preserved here for transparency).

Usage
-----
>>> def add(x, y):
...     return x + y
...
>>> # a sequential two‑level vmap: map x on 0 then 0, y on 1 then None
>>> spec = (MultiVmapAxes(0,0), MultiVmapAxes(1,None))
>>> ia_seq = expand_in_axes_spec(spec)  # → list[tuple]
>>> f_v = vmap_multi(add, ia_seq)

"""
from typing import Any, Iterable, List, Sequence, Union
import jax
from jax import tree_util
from jaxtyping import PyTree

from jax_cookbook._vmap import vmap_multi

__all__ = [
    "MultiVmapAxes",
    "expand_in_axes_spec",
]

# -----------------------------------------------------------------------------
# MultiVmapAxes wrapper
# -----------------------------------------------------------------------------

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

    # ------------------------------ constructor ------------------------------

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

    # ------------------------------ helpers ----------------------------------

    def sequential_axes(self) -> List[Any]:
        """Return a list ready to be slotted per‑level into `in_axes`."""
        if not self.natural_indices:
            return list(self.axes)
        # natural → sequential: subtract level index
        seq: List[Any] = []
        for i, ax in enumerate(self.axes):
            seq.append(None if ax is None else ax - i)
        return seq

    # ------------------------------ pytree ------------------------------------

    # Treat the *wrapper itself* as a PyTree **leaf** so that the recursion
    # sees it as atomic and defers to `sequential_axes` later.

    def tree_flatten(self):
        return (), None  # no children – it's a leaf

    @classmethod
    def tree_unflatten(cls, _: Any, __: Any):  # pragma: no cover – unused
        raise TypeError("MultiVmapAxes should never be unflattened by JAX")

    # ------------------------------ dunder ------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        mode = "natural" if self.natural_indices else "sequential"
        return f"MultiVmapAxes({self.axes}, mode={mode})"

    def __iter__(self):  # pragma: no cover – discourage but allow unpacking
        return iter(self.axes)

# Register the wrapper as a leaf with JAX
jax.tree_util.register_pytree_node_class(MultiVmapAxes)

# -----------------------------------------------------------------------------
# Expansion logic
# -----------------------------------------------------------------------------

_AxisSpec = Union[int, None, MultiVmapAxes, PyTree[int]]


def _expand_leaf(leaf: _AxisSpec) -> List[Any]:
    """Return a *list* of axes for this leaf – one entry per vmap level."""
    if isinstance(leaf, MultiVmapAxes):
        return leaf.sequential_axes()
    # Plain int / None / PyTree spec ⇒ single level
    return [leaf]


def _merge_child_schedules(child_schedules: Sequence[List[Any]], tmpl_container):
    """Pad child schedules so they all have equal length, then merge."""
    n_levels = max(len(s) for s in child_schedules)
    # Pad with None
    pad = lambda s: s + [None] * (n_levels - len(s))
    child_schedules = [pad(s) for s in child_schedules]
    merged_levels: List[Any] = []
    for lvl in range(n_levels):
        if isinstance(tmpl_container, dict):
            merged_levels.append({k: sched[lvl] for k, sched in zip(tmpl_container.keys(), child_schedules)})
        else:  # list or tuple
            merged_levels.append(type(tmpl_container)(sched[lvl] for sched in child_schedules))
    return merged_levels


def _expand(spec: _AxisSpec) -> List[Any]:
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


def expand_in_axes_spec(in_axes: _AxisSpec) -> List[PyTree]:
    """Public wrapper for `_expand`.

    Ensures that the returned value is a *sequence* of per‑level specs
    suitable for passing to ``vmap_multi``.
    """
    expanded = _expand(in_axes)
    # Guarantee list (could be tuple), copy for safety
    return list(expanded)

# -----------------------------------------------------------------------------
# Small sanity check (run as script) – not exhaustive unit test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import numpy as np
    import equinox as eqx

    def add(x, y):
        return x + y

    x = np.arange(6).reshape(2, 3)
    y = x * 10

    # 1) Single‑level vmap – behaves like jax.vmap
    ia1 = expand_in_axes_spec((0, 0))  # → [[0,1]]
    f1 = vmap_multi(add, ia1)
    assert np.array_equal(f1(x, y), x + y)

    # 2) Nested vmap via MultiVmapAxes
    ia2 = expand_in_axes_spec((MultiVmapAxes(0,0), MultiVmapAxes(None,0)))
    f2 = vmap_multi(add, ia2)
    out2 = f2(x, y)
    assert out2.shape == (2, 3) and np.array_equal(out2, x + y)

    print("All sanity checks passed.")