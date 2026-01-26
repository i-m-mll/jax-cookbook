import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax
import jax.numpy as jnp

import jax_cookbook.tree as jct


def test_take_and_take_multi():
    tree = [jnp.arange(6).reshape(2, 3), jnp.ones((2, 3))]
    out = jct.take(tree, [0], axis=0)
    assert out[0].shape == (1, 3)

    out2 = jct.take_multi(tree, indices=[0, 1], axes=[0, 1])
    assert out2[0].shape == ()


def test_array_set_scalar():
    tree = [jnp.zeros((3, 2)), jnp.ones((3, 2))]
    out = jct.array_set_scalar(tree, 5.0, idx=1, axis=0)
    assert jnp.allclose(out[0][1], 5.0)
    assert jnp.allclose(out[1][1], 5.0)


def test_random_split_like_tree():
    tree = [1, 2, 3]
    key = jax.random.PRNGKey(0)
    out = jct.random_split_like_tree(key, tree)
    assert len(out) == 3

    treedef = jax.tree_util.tree_structure(tree)
    out2 = jct._random_split_like_treedef(key, treedef)
    assert len(out2) == 3


def test_stack_and_unstack_roundtrip():
    a = [jnp.array([1, 2]), jnp.array([3, 4])]
    b = [jnp.array([5, 6]), jnp.array([7, 8])]
    stacked = jct.stack([a, b], axis=0)
    assert stacked[0].shape == (2, 2)

    unstacked = jct.unstack(stacked, axis=0)
    assert len(unstacked) == 2
    assert jnp.array_equal(unstacked[0][0], a[0])


def test_concatenate():
    a = [jnp.array([1, 2]), jnp.array([3, 4])]
    b = [jnp.array([5, 6]), jnp.array([7, 8])]
    out = jct.concatenate([a, b], axis=0)
    assert jnp.array_equal(out[0], jnp.array([1, 2, 5, 6]))


def test_stack_subtrees():
    tree = {"a": [jnp.array([1, 2]), jnp.array([3, 4])], "b": 0}

    def is_subtree(x):
        return isinstance(x, list)

    out = jct.stack_subtrees(tree, is_subtree, axis=0)
    assert jnp.array_equal(out["a"], jnp.array([[1, 2], [3, 4]]))


def test_stack_inner():
    tree = [[jnp.array([1, 2]), jnp.array([3, 4])], [jnp.array([5, 6]), jnp.array([7, 8])]]
    out = jct.stack_inner(tree)
    assert out[0].shape == (2, 2)


def test_array_bytes_duplicates():
    arr = jnp.zeros((2, 2))
    tree = [arr, arr]
    assert jct.array_bytes(tree, duplicates=False) == arr.nbytes
    assert jct.array_bytes(tree, duplicates=True) == arr.nbytes * 2


def test_struct_bytes():
    tree = [jax.ShapeDtypeStruct((2, 3), jnp.float32)]
    assert jct.struct_bytes(tree) == 2 * 3 * 4


def test_infer_batch_size_and_exclude():
    tree = [jnp.zeros((4, 2)), jnp.ones((4, 3))]
    assert jct.infer_batch_size(tree) == 4

    tree2 = [jnp.zeros((4, 2)), jnp.ones((5, 3))]
    with pytest.raises(ValueError):
        jct.infer_batch_size(tree2)


def test_leaves_of_type():
    tree = [1, "x", 2]
    out = jct.leaves_of_type(int, tree)
    assert out == [1, 2]
