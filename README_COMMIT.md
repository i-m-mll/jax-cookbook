Changes in this commit:
- Reorganized tests into module-based directories and expanded coverage across jax_cookbook (tree, io, misc, progress, vmap, where, etc.).
- Fixed several behaviors uncovered by tests: mixed-leaf array unwrapping, is_not_type logic, JSON dumping in save, and move_level_to_outside behavior.
- Updated docs/config references to the new GitHub username and docs URL; moved the MultiVmapAxes note into docs/notes.

Tests:
- uv run pytest -q
