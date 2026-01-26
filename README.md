# JAX Cookbook

A small collection of PyTree utilities and Equinox-friendly helpers that I reach for in JAX projects.

If you're unfamiliar with Equinox, check out https://docs.kidger.site/equinox/.

## Installation

```bash
pip install jax-cookbook
```

## Usage

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

## Docs

Build the docs locally with MkDocs:

```bash
mkdocs serve
```
