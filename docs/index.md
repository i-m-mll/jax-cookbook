# JAX Cookbook

A small collection of PyTree utilities and Equinox-friendly helpers I reach for in JAX projects.

## Install

```
pip install jax-cookbook
```

## Quickstart

```python
import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx

import jax_cookbook.tree as jct


@jct.filter_wrap(eqx.is_array)
def flatten_leaves(tree):
    return jt.map(jnp.ravel, tree)


tree = [jnp.zeros((3, 4)), "meta"]
flattened = flatten_leaves(tree)
# [jnp.zeros((12,)), "meta"]
```

Browse the API pages for the full set of tree, vmap, and functional helpers.
