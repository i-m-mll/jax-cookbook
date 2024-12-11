# JAX Cookbook

I often find myself manipulating PyTrees in ways not directly provided by the [JAX API](https://jax.readthedocs.io/en/latest/jax.html), or by any other single package I've found.

This is a collection of some of the patterns I've found repeatedly useful.

If you're unfamiliar with Equinox, I hope you'll [check it out](https://docs.kidger.site/equinox/). Its central feature is `equinox.Module`, an elegant way to represent your models as nested, callable dataclasses. 

TODO: Link to feedbax docs? (Did I describe this in more detail?)

## Installation

TODO: `pip install jax-cookbook`

## Usage

TODO: mkdocs

### Filter-combine decorator

By decorating a function whose first argument is a PyTree with `filter_wrap`, we can ensure the function is only applied to leaves that satisfy a certain condition.

```python
tree = [jnp.zeros((3, 4)), 'smeeth']

tree_flat = jt.map(jnp.ravel, tree)  
# TypeError: Argument 'smeeth' of type <class 'str> is not a valid JAX type

@filter_wrap(eqx.is_array)
def flatten_leaves(tree: PyTree[Array]) -> PyTree[Array]:
    return jt.map(jnp.ravel, tree)
    
tree_flat = flatten_array_leaves(tree)  # [jnp.zeros((12,)), 'smeeth']
```

Note that we can type annotate the decorated function as operating on a PyTree of arrays, since all of the tree's non-array leaves will be `None`.

Several of the other functions in this cookbook are wrapped this way, since it is a common pattern to operate only on leaves of a certain type. 

### Where-function parsing and construction

Sometimes we define a function which constructs a PyTree from nodes of another PyTree. A typical case is a function which given a model PyTree, selects those nodes whose array leaves should be trainable. 

```python
where_trainable = lambda model: (
    (
        model.hidden.layer1, 
        model.hidden.layer3,
    ),
    model.linear_out,
)
```

For example, `equinox.tree_at` uses such a "where-function" to select nodes of a PyTree to be updated out-of-place:

```python
updated_tree = equinox.tree_at(
    lambda tree: (tree.foo, tree.subtree.bar),
    tree_to_update,
    (new_foo, new_bar),
)
```

Importantly, while a where-function looks like a PyTree of data -- like a set of addresses we want to access -- it is *not* itself a PyTree. Being a function, it cannot be [serialised as a PyTree](https://docs.kidger.site/equinox/examples/serialisation/). And we probably shouldn't [pickle](https://docs.python.org/3/library/pickle.html) it. 

The cookbook provides a couple of functions that may be useful, here. The first is `where_func_to_labels`, which takes a where-function and returns its representation as a PyTree of strings:

```python
where_trainable_strs = where_func_to_labels(where_trainable)

# where_trainable_strs == (('hidden.layer1', 'hidden.layer3'), 'linear_out')
```

Similarly, `where_attr_strs_to_func` takes a PyTree of strings *representing attribute accesses*, and returns its representation as a where-function:

```python
where_trainable_parsed = where_attr_strs_to_func(where_trainable_strs)

# for all x, where_trainable_parsed(x) is where_trainable(x) 
```

Currently, this only works for where-functions based on attribute accesses, such as those where-functions you'd typically use to select model parameters in a `equinox.Module`-based PyTree. However, maybe you have a where-function which refers to indices of sequences, or keys of dicts, and so on. If you only need to convert it to a string representation, then `where_func_to_labels` may work; but if you need to generate a working where-function from strings, the strings cannot contain indexing notation or other kinds of non-attribute access. 