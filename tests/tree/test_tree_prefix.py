import pytest

pytest.importorskip("jax")

import jax_cookbook.tree as jct


def test_prefix_expand_simple():
    prefix = {"a": 1}
    tree = {"a": [1, 2, 3]}
    out = jct.prefix_expand_simple(prefix, tree)
    assert out == {"a": [1, 1, 1]}


def test_prefix_expand_nested():
    tree1 = {"a": 1}
    tree2 = {"a": {"x": 0, "y": 1}}
    out = jct.prefix_expand(tree1, tree2)
    assert out == {"a": {"x": 1, "y": 1}}
