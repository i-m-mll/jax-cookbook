import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax.numpy as jnp

from jax_cookbook._vmap import MultiVmapAxes, expand_axes_spec, natural_to_sequential_axes, unkwarg_key


def test_natural_to_sequential_axes():
    assert natural_to_sequential_axes([3, 0]) == [3, 0]
    assert natural_to_sequential_axes([3, 5, 1]) == [3, 4, 1]
    with pytest.raises(ValueError):
        natural_to_sequential_axes([1, 1])


def test_multivmapaxes_sequential_and_natural():
    m = MultiVmapAxes(0, None)
    assert m.sequential_axes() == [0, None]

    m2 = MultiVmapAxes(2, None, natural_indices=True)
    assert m2.sequential_axes() == [2, None]


def test_expand_axes_spec_simple():
    spec = (MultiVmapAxes(0, 0), MultiVmapAxes(1, None))
    expanded = expand_axes_spec(spec)
    assert expanded == [(0, 1), (0, None)]


def test_unkwarg_key():
    def f(x, *, key):
        return x + key

    g = unkwarg_key(f)
    assert g(3, 4) == 7
