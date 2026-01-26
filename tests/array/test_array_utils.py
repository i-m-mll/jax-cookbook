import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
import chex

from jax_cookbook._array import ArrayLikeWrapper, part_by_idx, unwrap_arraylikes, unwrap_arraylikes_and_labels
from jax_cookbook.tree import array_set


def test_unwrap_arraylikes_and_labels_with_str_label():
    tree = [ArrayLikeWrapper(jnp.zeros((2,)), label="foo")]
    vals, labels = unwrap_arraylikes_and_labels(tree)
    chex.assert_trees_all_close(vals, [jnp.zeros((2,))])
    assert labels == ["foo"]


def test_unwrap_arraylikes_and_labels_with_sequence_label():
    tree = [ArrayLikeWrapper(jnp.zeros((2,)), label=["a", "b"])]
    _, labels_short = unwrap_arraylikes_and_labels(tree, label_fmt="short")
    _, labels_full = unwrap_arraylikes_and_labels(tree, label_fmt="full")
    assert labels_short == ["a"]
    assert labels_full == ["a/b"]


def test_unwrap_arraylikes():
    tree = [ArrayLikeWrapper(jnp.array([1, 2])), "x"]
    out = unwrap_arraylikes(tree)
    assert jnp.array_equal(out[0], jnp.array([1, 2]))
    assert out[1] == "x"


def test_part_by_idx():
    arr = jnp.arange(6)
    selected, remainder = part_by_idx(arr, jnp.array([1, 3, 5]), axis=0)
    assert jnp.array_equal(selected, jnp.array([1, 3, 5]))
    assert jnp.array_equal(remainder, jnp.array([0, 2, 4]))


def test_array_set_updates_arrays_only():
    tree = (jnp.zeros((3, 2)), {"x": jnp.ones((3, 2))}, "keep")
    values = (jnp.full((2,), 5.0), {"x": jnp.full((2,), 7.0)}, "ignored")
    out = array_set(tree, values, idx=1)

    expected0 = tree[0].at[1].set(values[0])
    expected1 = tree[1]["x"].at[1].set(values[1]["x"])

    chex.assert_trees_all_close(out[0], expected0)
    chex.assert_trees_all_close(out[1]["x"], expected1)
    assert out[2] == "keep"


def test_array_set_raises_on_missing_array_values():
    tree = (jnp.zeros((2, 2)),)
    values = (None,)
    with pytest.raises(ValueError):
        array_set(tree, values, idx=0)
