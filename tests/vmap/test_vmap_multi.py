import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax.numpy as jnp
import chex

from jax_cookbook import vmap_multi


def test_vmap_multi_out_axes_changes_axis_position():
    def f(x):
        return x * 2

    x = jnp.arange(6).reshape(2, 3)
    f_v = vmap_multi(f, [0], out_axes_sequence=[1])
    out = f_v(x)

    expected = jnp.swapaxes(x * 2, 0, 1)
    chex.assert_trees_all_close(out, expected)


def test_vmap_multi_out_axes_length_mismatch_raises():
    def f(x):
        return x

    with pytest.raises(ValueError):
        vmap_multi(f, [0, 0], out_axes_sequence=[0])
