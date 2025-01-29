"""Recipes for working with PyTrees in JAX + Equinox.

:copyright: Copyright 2024 by MLL <mll@mll.bio>
:license: Apache 2.0, see LICENSE for details.
"""

from typing import Any

from equinox import Module

# from .tree import (
#     get_ensemble,
#     random_split_like_tree,
#     filter_wrap,
#     leaves_of_type,
#     make_named_dict_subclass,
#     make_named_tuple_subclass,
#     move_level_to_outside,
#     tree_array_bytes,
#     tree_call,
#     tree_concatenate,
#     tree_infer_batch_size,
#     tree_key_tuples,
#     tree_labels,
#     tree_labels_of_equal_leaves,
#     tree_map_tqdm,
#     tree_map_unzip,
#     tree_prefix_expand,
#     tree_set,
#     tree_set_scalar,
#     tree_stack,
#     tree_struct_bytes,
#     tree_take,
#     tree_take_multi,
#     tree_unstack,
#     tree_unzip,
#     tree_zip,
# )

from ._io import save, load, load_with_hyperparameters

from ._vmap import (
    unkwarg_key,
    vmap_multi, 
)

from ._func import (
    allf,
    anyf,
    compose,
    notf,
    identity,
    is_not_type,
    is_type,
    
)

from ._where import (
    where_attr_strs_to_func,
    where_func_to_strs,
)

from ._types import (
    is_module,
    is_none,
)