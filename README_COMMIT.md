# Commit: [develop] Improve tree utilities and fix edge cases

## Overview

This commit fixes several edge-case bugs in tree utilities and adds better validation. The `array_set` function now correctly preserves non-array leaves, `expand_split_keys` can preserve mapping types like `defaultdict`, and `vmap_multi` properly supports output axis specifications.

## Changes

### Tree Utilities (`jax_cookbook/tree.py`)

**`array_set` fix**: Previously, non-array leaves were being replaced by values from the update tree. Now they are correctly left unchanged, which matches the documented behavior and prevents unexpected data loss.

**`expand_split_keys` improvements**: Added `preserve_type` parameter (default `True`) that maintains mapping types during key expansion. This is important for `defaultdict` and custom mapping subclasses. The helper `_rebuild_mapping` handles reconstruction.

**Uniform tree validation**: Added `_assert_uniform_tree` helper that validates trees have consistent structure at each level. This is now used by `move_level_to_outside` to fail fast with a clear error rather than producing incorrect results on malformed trees.

**`_expand_spec_to_target` fix**: Improved handling of duplicate level keys in rearrangement specs. The previous set-based approach could silently mishandle specs with repeated keys; now uses consumption-based matching.

### Array Utilities (`jax_cookbook/_array.py`)

**`ArrayLikeWrapper` validation**: Fixed confusing validation logic for `axes_names` with ellipsis. Now raises a clear error when ellipsis is provided but the length already matches array rank.

**`unwrap_arraylikes_and_labels`**: Fixed label extraction to properly handle both string labels and sequence-of-string labels with the format options (short/medium/full).

### Vmap Utilities (`jax_cookbook/_vmap.py`)

**`vmap_multi`**: The `out_axes_sequence` parameter was documented but non-functional. Now properly applies output axis specifications at each vmap level.

### Documentation

Rewrote `docs/index.md` from the MkDocs template to actual library documentation with install instructions and a quickstart example. Updated API docs for clarity.

### Tests

Added test files for the new/fixed functionality:
- `test_array_utils.py`
- `test_expand_split_keys.py`
- `test_func_utils.py`
- `test_vmap_multi.py`

## Rationale

These fixes address real bugs encountered in use:
- `array_set` silently corrupting non-array data was a subtle bug that could cause hard-to-trace issues
- `defaultdict` losing its type when processed by `expand_split_keys` broke code relying on the default factory
- Uniform tree validation catches user errors early rather than producing wrong results

The documentation rewrite makes the library actually approachable for new users.

## Files Changed

- `jax_cookbook/tree.py` - Core tree utility fixes and validation
- `jax_cookbook/_array.py` - ArrayLikeWrapper and label handling fixes
- `jax_cookbook/_vmap.py` - vmap_multi out_axes support
- `jax_cookbook/_ldict.py` - Minor addition
- `docs/*.md` - Documentation rewrite
- `mkdocs.yml` - Doc config updates
- `pyproject.toml`, `uv.lock` - Dependency updates
- `tests/test_*.py` - New test coverage
