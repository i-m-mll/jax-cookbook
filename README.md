# JAX Cookbook

I often find myself manipulating PyTrees in a number of ways not directly provided by the [JAX API](https://jax.readthedocs.io/en/latest/jax.html), or by any other single package I've found.

This is a collection of some of the patterns I've found repeatedly useful.

If you're unfamiliar with Equinox, I hope you'll [check it out](https://docs.kidger.site/equinox/). Its central feature is `equinox.Module`, an elegant way to represent your models as nested, callable dataclasses. 

TODO: Link to feedbax docs? (Did I describe this there in more detail?)

## Installation

TODO: `pip install jax-cookbook`

## Usage

TODO: mkdocs

### Filter-combine decorator

By decorating a function whose first argument is a PyTree with `filter_wrap`, the function is only applied to leaves that satisfy a certain condition.

```python
import jax.numpy as jnp
import jax.tree as jt


tree = [jnp.zeros((3, 4)), 'smeeth']

tree_flat = jt.map(jnp.ravel, tree)  
# TypeError: Argument 'smeeth' of type <class 'str> is not a valid JAX type

@filter_wrap(eqx.is_array)
def flatten_leaves(tree: PyTree[Array]) -> PyTree[Array]:
    return jt.map(jnp.ravel, tree)
    
tree_flat = flatten_array_leaves(tree)  # [jnp.zeros((12,)), 'smeeth']
```

Note that we can type annotate the decorated function as operating on a PyTree of arrays, as all of the original tree's non-array leaves will be passed as`None`.

Several of the other functions in this cookbook are wrapped this way, since it is a common pattern to operate only on leaves of a certain type. 



