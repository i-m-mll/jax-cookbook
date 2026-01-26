import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx

import jax_cookbook.tree as jct


class Box:
    def __init__(self, x):
        self.x = x


def test_filter_wrap_applies_to_arrays_only():
    tree = [jnp.array([1, 2]), "meta"]

    @jct.filter_wrap(eqx.is_array)
    def f(t):
        return jt.map(lambda x: x + 1, t)

    out = f(tree)
    assert jnp.array_equal(out[0], jnp.array([2, 3]))
    assert out[1] == "meta"


def test_filter_map():
    tree = [1, "x", 2]
    out = jct.filter_map(lambda x: x + 1, tree, lambda x: isinstance(x, int))
    assert out == [2, "x", 3]


def test_filter_spec_leaves():
    tree = {"a": [1, 2], "b": 3}

    def leaf_func(t):
        return t["a"][1]

    spec = jct.filter_spec_leaves(tree, leaf_func)
    assert spec == {"a": [False, True], "b": False}


def test_first_leaf_and_shape():
    tree = [jnp.zeros((2, 3)), jnp.ones((1,))]
    assert jct.first_leaf(tree).shape == (2, 3)
    assert jct.first_leaf_shape(tree) == (2, 3)


def test_shapes():
    tree = [jnp.zeros((2, 3)), "x"]
    out = jct.shapes(tree)
    assert out == [(2, 3), "x"]


def test_get_ensemble():
    def fn(x, *, key):
        return x + key[0]

    import jax.random as jr

    key = jr.PRNGKey(0)
    out = jct.get_ensemble(fn, 1.0, n=3, key=key)
    assert out.shape == (3,)


def test_map_infers_is_leaf():
    class MyObj:
        def __init__(self, x):
            self.x = x

    def f(x: MyObj):
        return x.x + 1

    tree = [MyObj(1), MyObj(2)]
    out = jct._map(f, tree)
    assert out == [2, 3]


def test_sum_squares_and_sum_n_features():
    tree = [jnp.array([1.0, 2.0]), jnp.array([3.0])]
    assert jnp.allclose(jct.sum_squares(tree), 1 + 4 + 9)
    assert jct.sum_n_features([jnp.zeros((2, 3)), jnp.zeros((4, 1))]) == 4


def test_call_tree_of_callables():
    tree = [lambda x: x + 1, "meta", lambda x: x * 2]
    out = jct.call(tree, 3)
    assert out == [4, "meta", 6]

    out2 = jct.call(tree, 3, exclude=lambda f: f is tree[0])
    assert out2[0] is tree[0]
    assert out2[2] == 6
