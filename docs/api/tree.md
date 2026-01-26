# PyTree operations

::: jax_cookbook.tree.filter_wrap
::: jax_cookbook.tree.filter_map
::: jax_cookbook.tree.leaves_of_type
::: jax_cookbook.tree.call

::: jax_cookbook.tree.prefix_expand
::: jax_cookbook.tree.prefix_expand_simple

## Paths and labels

::: jax_cookbook.tree.labels
::: jax_cookbook.tree.key_tuples
::: jax_cookbook.tree.leaves_with_annotated_path

## Zipping

::: jax_cookbook.tree.zip_
::: jax_cookbook.tree.zip_named
::: jax_cookbook.tree.unzip
::: jax_cookbook.tree.map_unzip

## Equality

::: jax_cookbook.tree.paths_of_equal_leaves
::: jax_cookbook.tree.labels_of_equal_leaves

## Utilities

!!! note "Rearranging levels"
    `rearrange_uniform_tree` assumes the tree is uniform at each level (respecting
    `is_leaf`). If a level type appears more than once (e.g. two `list` levels), each
    occurrence in the spec matches the next unused level of that type, from the outside in.

::: jax_cookbook.tree.rearrange_uniform_tree
::: jax_cookbook.tree.tree_level_types
::: jax_cookbook.tree.expand_split_keys
