import pytest

pytest.importorskip("jax")

import jax.tree_util as jtu

import jax_cookbook.tree as jct


def test_node_key_to_value_and_get_child():
    tree = {"a": [1, 2]}
    path, leaf = next(iter(jtu.tree_leaves_with_path(tree)))
    key0 = path[0]
    assert jct.node_key_to_value(key0) == "a"
    child = jct.get_child_node_given_key(tree, key0)
    assert child == [1, 2]

    # SequenceKey
    key1 = path[1]
    assert jct.node_key_to_value(key1) == 0
    child2 = jct.get_child_node_given_key(child, key1)
    assert child2 == 1

    # GetAttrKey via namedtuple
    from collections import namedtuple

    NT = namedtuple("NT", ["x", "y"])
    tree2 = NT(1, 2)
    path2, _ = next(iter(jtu.tree_leaves_with_path(tree2)))
    attr_key = path2[0]
    assert jct.node_key_to_value(attr_key) == "x"
    assert jct.get_child_node_given_key(tree2, attr_key) == 1


def test_leaves_with_annotated_path():
    tree = {"a": [1, 2]}
    out = jct.leaves_with_annotated_path(tree)
    assert len(out) == 2


def test_labels_and_key_tuples():
    tree = {"a": [1, 2]}
    labels = jct.labels(tree)
    assert labels == {"a": ["a_0", "a_1"]}

    keys = jct.key_tuples(tree, keys_to_strs=True)
    assert keys == {"a": [("a", "0"), ("a", "1")]}


def test_paths_of_equal_leaves_and_labels():
    tree = [1, 2, 1]
    paths = jct.paths_of_equal_leaves(tree)
    assert paths[0] == {(jtu.SequenceKey(2),)}
    labels = jct.labels_of_equal_leaves(tree)
    assert "2" in labels[0]


def test_collect_aux_data():
    tree = [1, 2, 3]
    treedef = jtu.tree_structure(tree)
    aux = jct.collect_aux_data(treedef, list)
    assert aux
