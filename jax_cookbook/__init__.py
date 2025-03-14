"""Some recipes in JAX + Equinox.

:copyright: Copyright 2024-2025 by MLL
:license: Apache 2.0, see LICENSE for details.
"""

from typing import Any

from equinox import Module

from ._io import save, load, load_with_hyperparameters, arrays_to_lists

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