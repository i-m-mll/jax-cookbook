from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest

import jax_cookbook.tree as jtree
from jax_cookbook import LDict


def test_filter_spec_leaves_expands_selected_subtree():
    tree = {
        "params": {
            "w": jnp.array([1.0, 2.0]),
            "b": jnp.array(3.0),
        },
        "static": "kept",
    }

    spec = jtree.filter_spec_leaves(tree, lambda t: t["params"])

    assert spec == {
        "params": {
            "w": True,
            "b": True,
        },
        "static": False,
    }


def test_filter_spec_leaves_respects_is_leaf_boundary():
    tree = {
        "params": LDict.of("layer")(
            {
                "one": {"w": 1},
                "two": {"w": 2},
            }
        ),
        "static": 0,
    }
    spec = jtree.filter_spec_leaves(
        tree,
        lambda t: t["params"],
        is_leaf=LDict.is_of("layer"),
    )

    assert spec["params"] is True
    assert spec["static"] is False


def test_array_set_preserves_none_leaf_structure():
    tree = {
        "x": jnp.array([[0, 1], [2, 3]]),
        "none": None,
        "label": "old",
    }
    values = {
        "x": jnp.array([9, 8]),
        "none": "now-set",
        "label": "new",
    }

    out = jtree.array_set(tree, values, 1)

    assert jnp.array_equal(out["x"], jnp.array([[0, 1], [9, 8]]))
    assert out["none"] == "now-set"
    assert out["label"] == "new"


def test_rearrange_uniform_tree_preserves_ldict_labels_and_keys():
    tree = LDict.of("outer")(
        {
            "o1": LDict.of("inner")({"i1": 1, "i2": 2}),
            "o2": LDict.of("inner")({"i1": 3, "i2": 4}),
        }
    )

    out = jtree.rearrange_uniform_tree(tree, ["inner", "outer"])

    assert isinstance(out, LDict)
    assert out.label == "inner"
    assert list(out.keys()) == ["i1", "i2"]
    assert all(isinstance(child, LDict) for child in out.values())
    assert [child.label for child in out.values()] == ["outer", "outer"]
    assert list(out["i1"].keys()) == ["o1", "o2"]
    assert out["i1"]["o1"] == 1
    assert out["i2"]["o2"] == 4


def test_rearrange_uniform_tree_accepts_constructor_and_ellipsis_specs():
    tree = LDict.of("outer")(
        {
            "o1": [LDict.of("inner")({"i1": 1, "i2": 2})],
            "o2": [LDict.of("inner")({"i1": 3, "i2": 4})],
        }
    )

    out = jtree.rearrange_uniform_tree(tree, [LDict.of("inner"), ..., "outer"])

    assert out.label == "inner"
    assert list(out.keys()) == ["i1", "i2"]
    assert out["i1"].label == "outer"
    assert out["i1"]["o1"] == [1]
    assert out["i2"]["o2"] == [4]


def test_rearrange_uniform_tree_missing_label_error():
    tree = LDict.of("outer")({"o1": LDict.of("inner")({"i1": 1})})

    with pytest.raises(ValueError, match="Unknown level"):
        jtree.rearrange_uniform_tree(tree, ["missing"])


def test_rearrange_uniform_tree_rejects_uneven_branches():
    tree = LDict.of("outer")(
        {
            "o1": LDict.of("inner")({"i1": 1, "i2": 2}),
            "o2": LDict.of("inner")({"i1": 3}),
        }
    )

    with pytest.raises(AssertionError, match="Uniformity mismatch"):
        jtree.rearrange_uniform_tree(tree, ["inner", "outer"])


def test_rearrange_uniform_tree_keeps_non_ldict_leaves():
    tree = LDict.of("outer")(
        {
            "o1": [None, "a"],
            "o2": [None, "b"],
        }
    )

    out = jtree.rearrange_uniform_tree(
        tree,
        [list, "outer"],
        is_leaf=lambda x: x is None,
    )

    assert out[0]["o1"] is None
    assert out[0]["o2"] is None
    assert out[1]["o1"] == "a"
    assert out[1]["o2"] == "b"


def test_call_with_keys_uses_key_selector_and_preserves_other_leaves():
    shared_key = jr.PRNGKey(123)

    def per_leaf(*, key):
        return jr.key_data(key)

    def shared(*, key):
        return jr.key_data(key)

    shared.use_shared = True  # type: ignore[attr-defined]

    tree = {
        "per_leaf": per_leaf,
        "shared": shared,
        "static": "kept",
    }

    def choose_key(fn: Callable, per_leaf_key):
        return shared_key if getattr(fn, "use_shared", False) else per_leaf_key

    out = jtree.call_with_keys(tree, key=jr.PRNGKey(0), key_fn=choose_key)

    assert out["static"] == "kept"
    assert jnp.array_equal(out["shared"], jr.key_data(shared_key))
    assert not jnp.array_equal(out["per_leaf"], jr.key_data(shared_key))


def test_call_with_keys_exclude_leaves_callable_unchanged():
    def fn(*, key):
        return key

    out = jtree.call_with_keys(
        {"fn": fn},
        key=jr.PRNGKey(0),
        exclude=lambda x: x is fn,
        is_leaf=lambda x: isinstance(x, Callable),
    )

    assert out["fn"] is fn
