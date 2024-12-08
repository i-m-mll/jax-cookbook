from collections.abc import Callable
from operator import attrgetter

import jax
import jax.tree as jt 
import jax.tree_util as jtu
from jaxtyping import PyTree
import equinox as eqx


def _get_where_str(where_func: Callable) -> str:
    """
    Returns a string representation of the (nested) attributes accessed by a function.

    Only works for functions that take a single argument, and return the argument
    or a single (nested) attribute accessed from the argument.
    """
    bytecode = dis.Bytecode(where_func)
    return ".".join(instr.argrepr for instr in bytecode if instr.opname == "LOAD_ATTR")


class _NodeWrapper:
    def __init__(self, value):
        self.value = value


class NodePath:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        return iter(self.path)


def where_func_to_paths(
    where: Callable[[PyTree[..., 'T']], PyTree[..., 'S']], 
    tree: PyTree[..., 'T']
) -> PyTree[NodePath, 'S']:
    """
    Similar to `_get_where_str`, but:

    - returns node paths, not strings;
    - works for `where` functions that return arbitrary PyTrees of nodes;
    - works for arbitrary node access (e.g. dict keys, sequence indices)
      and not just attribute access.

    Limitations:

    - requires a PyTree argument;
    - assumes the same object does not appear as multiple nodes in the tree;
    - if `where` specifies a node that is a subtree, it cannot also specify a node
      within that subtree.

    See [this issue](https://github.com/i-m-mll/feedbax/issues/14).
    """
    tree = eqx.tree_at(where, tree, replace_fn=lambda x: _NodeWrapper(x))
    id_tree = jt.map(id, tree, is_leaf=lambda x: isinstance(x, _NodeWrapper))
    node_ids = where(id_tree)

    paths_by_id = {
        leaf_id: path for path, leaf_id in jtu.tree_leaves_with_path(
            jt.map(
                lambda x: x if x in jt.leaves(node_ids) else None,
                id_tree,
            )
        )
    }

    paths = jt.map(lambda node_id: NodePath(paths_by_id[node_id]), node_ids)

    return paths


class _WhereStrConstructor:

    def __init__(self, label: str = ""):
        self.label = label

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            key = f"'{key}'"
        elif isinstance(key, type):
            key = key.__name__
        return _WhereStrConstructor("".join([self.label, f"[{key}]"]))

    def __getattr__(self, name: str):
        sep = "." if self.label else ""
        return _WhereStrConstructor(sep.join([self.label, name]))


def _get_where_str_constructor_label(x: _WhereStrConstructor) -> str:
    return x.label


def where_func_to_labels(where: Callable) -> PyTree[str]:
    """Also similar to `_get_where_str` and `where_func_to_paths`, but:

    - Avoids complicated logic of parsing bytecode, or traversing pytrees;
    - Works for `where` functions that return arbitrary PyTrees of node references;
    - Runs significantly (10+ times) faster than the other solutions.
    """

    try:
        return jt.map(_get_where_str_constructor_label, where(_WhereStrConstructor()))
    except TypeError:
        raise TypeError("`where` must return a PyTree of node references")


def attr_str_tree_to_where_func(tree: PyTree[str]) -> Callable:
    """Reverse transformation to `where_func_to_labels`.

    Takes a PyTree of strings describing attribute accesses, and returns a function
    that returns a PyTree of attributes.
    """
    getters = jt.map(lambda s: attrgetter(s), tree)

    def where_func(obj):
        return jt.map(lambda g: g(obj), getters)

    return where_func