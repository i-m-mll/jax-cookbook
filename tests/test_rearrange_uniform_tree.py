# tests/test_rearrange_levels.py
"""
! Note that some of these tests are not really correct, I think, due to the reference
! implementations involve `jt.transpose`, which (as of Aug 2025) the models still have
! trouble with, possibly because of a lack of examples and official documentation.
"""

from __future__ import annotations

import importlib
import itertools
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu

import jax_cookbook.tree as tu
from jax_cookbook import LDict, LDictConstructor

# Public entrypoints
rearrange_levels = tu.rearrange_uniform_tree


# --- helpers --------------------------------------------------------------------


def _as_descriptor_from_spec(x: Any) -> Any:
    """Mirror user-facing spec normalization (string -> LDict.of, etc.)."""
    if x is Ellipsis:
        return Ellipsis
    if isinstance(x, str):
        if LDict is None:
            raise RuntimeError("String specs require LDict; not available in this env.")
        return LDict.of(x)
    if LDictConstructor is not None and isinstance(x, LDictConstructor):
        return x
    if isinstance(x, type):
        return x
    raise TypeError(f"Unsupported spec: {x!r}")


def _level_key(x: Any) -> tuple[str, Any]:
    if LDictConstructor is not None and isinstance(x, LDictConstructor):
        return ("ldict", x.label)
    if isinstance(x, type):
        return ("type", x)
    raise TypeError(f"Unsupported level descriptor: {x!r}")


def _descriptor_predicate(descriptor: Any) -> Callable[[Any], bool]:
    """Returns a predicate that matches a whole level."""
    if LDictConstructor is not None and isinstance(descriptor, LDictConstructor):
        return LDict.is_of(descriptor.label)
    if isinstance(descriptor, type):
        return lambda node: isinstance(node, descriptor)
    raise TypeError(f"Unsupported descriptor: {descriptor!r}")


def _expand_spec_to_target(spec: Sequence[Any], current: list[Any]) -> list[Any]:
    """Replicate the target-expansion logic (with Ellipsis)."""
    if not current:
        return []
    spec_norm = [_as_descriptor_from_spec(x) for x in spec]
    if sum(1 for x in spec_norm if x is Ellipsis) > 1:
        raise ValueError("Multiple ellipses not allowed")

    cur_keys = [_level_key(x) for x in current]
    specified = [x for x in spec_norm if x is not Ellipsis]

    # duplicates
    seen, dups = set(), []
    for x in specified:
        k = _level_key(x)
        if k in seen:
            dups.append(x)
        seen.add(k)
    if dups:
        raise ValueError("Duplicate levels in spec")

    # unknowns
    unknown = [x for x in specified if _level_key(x) not in cur_keys]
    if unknown:
        raise ValueError("Unknown level(s) in spec")

    if Ellipsis in spec_norm:
        before, after, placed = [], [], set()
        side = "before"
        for x in spec_norm:
            if x is Ellipsis:
                side = "after"
                continue
            (before if side == "before" else after).append(x)
            placed.add(_level_key(x))
        middle = [x for x in current if _level_key(x) not in placed]
        return before + middle + after
    else:
        placed = {_level_key(x) for x in spec_norm}
        rest = [x for x in current if _level_key(x) not in placed]
        return list(spec_norm) + rest


def _build_uniform_tree(levels: Sequence[Any], sizes: Sequence[int]) -> Any:
    """Build a uniform tree given a sequence of level descriptors (outer->inner)
    and per-level sizes. Leaves are unique integers to make permutations visible."""
    assert len(levels) == len(sizes)
    counter = {"i": 0}

    def make_leaf():
        i = counter["i"]
        counter["i"] += 1
        return i

    def build_at(d: int) -> Any:
        if d == len(levels):
            return make_leaf()
        desc = levels[d]
        n = sizes[d]
        if LDictConstructor is not None and isinstance(desc, LDictConstructor):
            cons = desc  # LDict.of(label)
            keys = [f"k{j}" for j in range(n)]
            return cons({k: build_at(d + 1) for j, k in enumerate(keys)})
        if desc is dict:
            keys = [f"k{j}" for j in range(n)]
            return {k: build_at(d + 1) for j, k in enumerate(keys)}
        if desc is list:
            return [build_at(d + 1) for _ in range(n)]
        if desc is tuple:
            return tuple(build_at(d + 1) for _ in range(n))
        raise TypeError(f"Unsupported container type in test builder: {desc!r}")

    return build_at(0)


class _W:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


def _cut_to_front_via_transpose(tree: Any, descriptor: Any, *, is_leaf=None) -> Any:
    """Reference 'cut' implemented with jax.tree_transpose, respecting user is_leaf."""
    outer_td = jt.structure(tree, is_leaf=_descriptor_predicate(descriptor))
    wrapped = tree if is_leaf is None else jt.map(lambda x: _W(x), tree, is_leaf=is_leaf)
    transposed = jt.transpose(outer_td, None, wrapped)
    if is_leaf is None:
        return transposed
    return jt.map(lambda x: x.x if isinstance(x, _W) else x, transposed)


def _rearrange_reference(tree: Any, target: list[Any], *, is_leaf=None) -> Any:
    """Independent reference: realise target by a sequence of cuts."""
    # Discover current levels using the SUT’s helper to avoid re-deriving it here.
    current = tu.tree_level_types(tree, is_leaf=is_leaf)
    levels = list(current)
    placed: list[Any] = []

    def key(x):
        return _level_key(x)

    for i, want in enumerate(target):
        if key(levels[i]) == key(want):
            placed.append(want)
            continue

        # bring want to front
        tree = _cut_to_front_via_transpose(tree, want, is_leaf=is_leaf)
        idx = next(j for j, d in enumerate(levels) if key(d) == key(want))
        levels = [levels[idx]] + levels[:idx] + levels[idx + 1 :]

        # slide it behind the already-placed prefix
        for prev in reversed(placed):
            tree = _cut_to_front_via_transpose(tree, prev, is_leaf=is_leaf)
            jdx = next(j for j, d in enumerate(levels) if key(d) == key(prev))
            levels = [levels[jdx]] + levels[:jdx] + levels[jdx + 1 :]

        placed.append(want)

    return tree


def _flatten(tree: Any, *, is_leaf=None):
    leaves, _ = jt.flatten(tree, is_leaf=is_leaf)
    return leaves


# --- parametrisations -----------------------------------------------------------


def _ld(label: str):
    if LDict is None:
        pytest.skip("LDict not available in this environment")
    return LDict.of(label)


LD_FOO = _ld("foo") if LDict is not None else None
LD_BAR = _ld("bar") if LDict is not None else None
LD_BAZ = _ld("baz") if LDict is not None else None


# --- tests ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "levels,sizes,spec",
    [
        # simple two-level: LDict 'foo' outside list
        pytest.param(
            [_ld("foo"), list],
            [2, 3],
            [list, "foo"],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
        # three-level: tuple, LDict 'foo', list → bring LDict outermost; ellipsis keeps the rest
        pytest.param(
            [tuple, _ld("foo"), list],
            [2, 3, 4],
            ["foo", ...],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
        # mix of builtin types only: dict, list, tuple
        ([dict, list, tuple], [2, 3, 4], [tuple, ..., dict]),
        # three LDicts
        pytest.param(
            [_ld("foo"), _ld("bar"), _ld("baz")],
            [2, 3, 2],
            ["bar", "foo", "baz"],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
    ],
)
def test_matches_reference(levels, sizes, spec):
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=None)
    target = _expand_spec_to_target(spec, current)

    got = rearrange_levels(tree, spec, is_leaf=None)
    ref = _rearrange_reference(tree, target, is_leaf=None)

    assert jt.structure(got) == jt.structure(ref)
    assert _flatten(got) == _flatten(ref)


@pytest.mark.parametrize(
    "levels,sizes,spec,is_leaf_pred",
    [
        # is_leaf at inner LDict('bar'): only outer reorders should occur
        pytest.param(
            [tuple, _ld("foo"), _ld("bar"), list],
            [2, 3, 2, 5],
            ["foo", ..., tuple],
            lambda x: isinstance(x, dict)
            and getattr(x, "_ldict_label", None) == "bar",  # best-effort if LDict stores a marker
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
        # is_leaf at dict: freeze inner parts, reorder outer two
        ([dict, list, tuple], [2, 3, 4], [list, dict, ...], lambda x: isinstance(x, dict)),
    ],
)
def test_is_leaf_boundary_respected(levels, sizes, spec, is_leaf_pred):
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=is_leaf_pred)
    target = _expand_spec_to_target(spec, current)

    got = rearrange_levels(tree, spec, is_leaf=is_leaf_pred)
    ref = _rearrange_reference(tree, target, is_leaf=is_leaf_pred)

    assert jt.structure(got) == jt.structure(ref)
    assert _flatten(got, is_leaf=is_leaf_pred) == _flatten(ref, is_leaf=is_leaf_pred)


def test_idempotent_when_already_ordered():
    levels, sizes = [list, tuple, dict], [2, 3, 4]
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=None)
    spec = current  # already ordered, but use exact descriptor objects

    out = rearrange_levels(tree, spec, is_leaf=None)
    assert jt.structure(out) == jt.structure(tree)
    assert _flatten(out) == _flatten(tree)


@pytest.mark.parametrize(
    "levels,sizes",
    [
        ([list, tuple, dict], [2, 3, 4]),
        pytest.param(
            [_ld("foo"), list, tuple],
            [2, 3, 4],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
    ],
)
def test_invertibility(levels, sizes):
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=None)

    # pick a nontrivial permutation of axes
    p = list(range(len(current)))
    if len(p) >= 3:
        p = [1, 2, 0, *p[3:]]  # rotate first three
    elif len(p) == 2:
        p = [1, 0]
    else:
        p = p

    perm_spec = [current[i] for i in p]
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    inv_spec = [perm_spec[i] for i in inv]

    t1 = rearrange_levels(tree, perm_spec)
    t2 = rearrange_levels(t1, inv_spec)

    assert jt.structure(t2) == jt.structure(tree)
    assert _flatten(t2) == _flatten(tree)


def test_duplicate_and_unknown_errors():
    levels, sizes = [list, tuple, dict], [2, 3, 4]
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=None)

    # duplicate
    dup = [list, list]
    with pytest.raises(ValueError):
        rearrange_levels(tree, dup)

    # unknown (use type not present)
    class Fake: ...

    with pytest.raises(ValueError):
        rearrange_levels(tree, [Fake])


def test_multiple_ellipsis_error():
    levels, sizes = [list, tuple, dict], [2, 3, 4]
    tree = _build_uniform_tree(levels, sizes)
    with pytest.raises(ValueError):
        rearrange_levels(tree, [list, ..., tuple, ...])


def test_empty_containers_are_noops():
    # An empty list at outer level → zero arity; function should just return the tree.
    levels, sizes = [list, tuple], [0, 3]
    tree = _build_uniform_tree(levels, sizes)
    out = rearrange_levels(tree, [tuple, list])
    assert jt.structure(out) == jt.structure(tree)
    assert _flatten(out) == _flatten(tree)


def test_nonuniform_detection():
    # Build a nonuniform tree by changing keys at one inner dict
    levels, sizes = [dict, list], [2, 3]
    tree = _build_uniform_tree(levels, sizes)
    # Corrupt uniformity: change keys of one branch
    # (Pick first outer key and mutate its child mapping shape)
    assert isinstance(tree, dict)
    first_k = next(iter(tree))
    bad = {"X": tree[first_k]["k0"], "Y": tree[first_k]["k1"], "Z": tree[first_k]["k2"]}  # type: ignore[index]
    tree[first_k] = bad  # type: ignore[assignment]

    with pytest.raises(AssertionError):
        rearrange_levels(tree, [list, dict])


@pytest.mark.parametrize(
    "levels,sizes,spec",
    [
        # string → LDict.of
        pytest.param(
            [_ld("foo"), list, _ld("bar")],
            [2, 3, 2],
            ["bar", "foo", ...],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
        # explicit constructor objects
        pytest.param(
            [_ld("foo"), list, _ld("bar")],
            [2, 2, 2],
            [_ld("bar"), list, _ld("foo")],
            marks=pytest.mark.skipif(LDict is None, reason="LDict"),
        ),
        # type objects only
        ([list, tuple, dict], [2, 3, 4], [dict, list, tuple]),
        # leading/trailing ellipsis
        ([list, tuple, dict], [2, 3, 4], [..., dict]),
        ([list, tuple, dict], [2, 3, 4], [list, ...]),
    ],
)
def test_spec_forms(levels, sizes, spec):
    tree = _build_uniform_tree(levels, sizes)
    current = tu.tree_level_types(tree, is_leaf=None)
    target = _expand_spec_to_target(spec, current)
    got = rearrange_levels(tree, spec)

    ref = _rearrange_reference(tree, target)
    assert jt.structure(got) == jt.structure(ref)
    assert _flatten(got) == _flatten(ref)


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_all_permutations_small(L):
    # Exhaustively check all permutations of level orders for small L.
    # Use builtin types to avoid dependency on LDict.
    levels = [list, tuple, dict, list][:L]
    sizes = [2, 2, 2, 2][:L]
    tree = _build_uniform_tree(levels, sizes)

    current = tu.tree_level_types(tree, is_leaf=None)
    perms = list(itertools.permutations(range(L)))

    for perm in perms:
        target = [current[i] for i in perm]
        got = rearrange_levels(tree, target)
        ref = _rearrange_reference(tree, target)
        assert _flatten(got) == _flatten(ref)
        assert jt.structure(got) == jt.structure(ref)
