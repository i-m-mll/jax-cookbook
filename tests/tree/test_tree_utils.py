import pytest

pytest.importorskip("jax")

import jax.tree as jt
import jax.tree_util as jtu

import jax_cookbook.tree as jct


def test_make_named_tuple_subclass():
    NT = jct.make_named_tuple_subclass("Foo")
    t = NT((1, 2))
    leaves, treedef = jt.flatten(t)
    assert leaves == [1, 2]
    rebuilt = jt.unflatten(treedef, leaves)
    assert isinstance(rebuilt, NT)


def test_make_named_dict_subclass():
    ND = jct.make_named_dict_subclass("Bar")
    d = ND({"a": 1, "b": 2})
    leaves, treedef = jt.flatten(d)
    rebuilt = jt.unflatten(treedef, leaves)
    assert isinstance(rebuilt, ND)


def test_move_level_to_outside_example():
    tree = ([(1, 2), (3, 4)], [(5, 6), (7, 8)])
    out = jct.move_level_to_outside(tree, list)
    assert out == [((1, 2), (5, 6)), ((3, 4), (7, 8))]


def test_n_unique_strs():
    out = list(jct._n_unique_strs(5))
    assert out == ["a", "b", "c", "d", "e"]


def test_tree_level_types_and_first_path_structure():
    tree = {"a": [1, 2]}
    types = jct.tree_level_types(tree)
    assert types[:2] == [dict, list]
    nodes, treedefs = jct.first_path_node_structure(tree)
    assert len(nodes) == len(treedefs)


def test_internal_order_and_permutation_helpers():
    probe_nodes = [[1, 2], {"a": 1}]
    target = [list, dict]
    order = jct._order_from_target(probe_nodes, target)
    assert order == [0, 1]

    tree = [[1, 2], [3, 4]]
    treedefs = jct.first_path_node_structure(tree)[1]
    composed = jct._compose_treedefs_in_order(treedefs, [0, 1])
    assert composed.num_leaves == jtu.tree_structure(tree).num_leaves

    idx = jct._leaf_permutation_for_axes([2, 2], [1, 0])
    assert list(idx) == [0, 2, 1, 3]
