import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

from types import SimpleNamespace

import jax.tree_util as jtu

from jax_cookbook._where import (
    NodePath,
    get_where_str,
    where_attr_strs_to_func,
    where_func_to_paths,
    where_func_to_strs,
)


class Obj:
    def __init__(self):
        self.foo = SimpleNamespace(bar=SimpleNamespace(baz=1))
        self.x = 2


def test_where_func_to_strs_simple_attrs():
    def where(t):
        return {"a": t.foo.bar, "b": t.x}

    out = where_func_to_strs(where)
    assert out == {"a": "foo.bar", "b": "x"}


def test_where_attr_strs_to_func():
    tree = {"a": "foo.bar", "b": "x"}
    where = where_attr_strs_to_func(tree)
    obj = Obj()
    out = where(obj)
    assert out["a"] == obj.foo.bar
    assert out["b"] == obj.x


def test_where_func_to_paths():
    tree = {"a": [1, 2], "b": {"c": 3}}

    def where(t):
        return t["a"][1]

    node_path = where_func_to_paths(where, tree)
    assert isinstance(node_path, NodePath)
    # Compare with JAX's path for that leaf.
    target_path = next(p for p, leaf in jtu.tree_leaves_with_path(tree) if leaf == 2)
    assert tuple(node_path) == target_path


def test_get_where_str():
    def where(x):
        return x.foo.bar

    assert get_where_str(where) == "foo.bar"
