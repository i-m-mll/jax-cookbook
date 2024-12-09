# JAX Cookbook

While working on other projects, I've often found myself wanting to manipulate PyTrees in ways that are not directly implemented in JAX, or in any other single package or repository that I've been able to find.

This is a collection of some of the patterns I've found useful.

If you're unfamiliar with Equinox, I'm excited for you to [check it out](https://docs.kidger.site/equinox/). The central feature is `equinox.Module`, which is an elegant way to represent your models as nested, callable dataclasses. 

TODO: Link to feedbax docs? Did I describe this in more detail 

## Installation

`pip install jax-cookbook`

## Usage

TODO: mkdocs

### Where-function parsing and construction

Sometimes, we define a function which constructs a PyTree from nodes of another PyTree. A typical case is a function which given a model PyTree, selects those nodes whose array leaves should be trainable. 

```python
where_trainable = lambda model: (
    (model.hidden.layer1, model.hidden.layer3),
    model.linear_out,
)
```

For example, `equinox.tree_at` uses such a "where-function" to select nodes of a PyTree to be updated out-of-place:

```python
updated_tree = equinox.tree_at(
    lambda tree: (tree.foo, tree.bar),
    tree_to_update,
    (new_foo, new_bar),
)
```

Importantly, while a where-function looks like a PyTree of data -- like a set of addresses we want to access -- it is *not* itself a PyTree. Being a function, it cannot be [serialised as a PyTree](https://docs.kidger.site/equinox/examples/serialisation/) -- we probably shouldn't [pickle](https://docs.python.org/3/library/pickle.html) it. 

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

Currently, this only works for where-functions based on attribute accesses. If you only need a string representation of the where-function, then `where_func_to_labels` will generate it in any case; but if you need to generate a working where-function from strings, the strings cannot contain indexing notation or other kinds of non-attribute access. 