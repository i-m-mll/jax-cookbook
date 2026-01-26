import pytest

from collections import defaultdict

from jax_cookbook.tree import expand_split_keys


def test_expand_split_keys_basic():
    tree = {"a.b": 1, "a.c": 2, "d": 3}
    out = expand_split_keys(tree)
    assert out == {"a": {"b": 1, "c": 2}, "d": 3}


def test_expand_split_keys_preserves_defaultdict():
    tree = defaultdict(int, {"a.b": 1, "a.c": 2})
    out = expand_split_keys(tree)
    assert isinstance(out, defaultdict)
    assert out.default_factory is tree.default_factory
    assert out == {"a": {"b": 1, "c": 2}}


def test_expand_split_keys_disable_preserve_type():
    tree = defaultdict(int, {"a.b": 1})
    out = expand_split_keys(tree, preserve_type=False)
    assert isinstance(out, dict)
    assert out == {"a": {"b": 1}}
