import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from jax_cookbook._array import ArrayLikeWrapper, MaskedArray


def test_masked_array_unwrap():
    data = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([True, False, True])
    ma = MaskedArray(data=data, mask=mask)
    out = ma.unwrap(invalid_value=-1.0)
    assert jnp.array_equal(out, jnp.array([1.0, -1.0, 3.0]))


def test_arraylikewrapper_axes_names_validation():
    arr = jnp.zeros((2, 3))
    ArrayLikeWrapper(arr, axes_names=["a", "b"])  # ok
    with pytest.raises(ValueError):
        ArrayLikeWrapper(arr, axes_names=["a"])  # too short, no ellipsis
    with pytest.raises(ValueError):
        ArrayLikeWrapper(arr, axes_names=["a", "b", "c"])  # too long
    with pytest.raises(ValueError):
        ArrayLikeWrapper(arr, axes_names=["a", "b", ...])  # ellipsis not allowed when len matches


def test_arraylikewrapper_label_type_validation():
    arr = jnp.zeros((2,))
    ArrayLikeWrapper(arr, label="x")
    ArrayLikeWrapper(arr, label=["x", "y"])
    with pytest.raises(TypeError):
        ArrayLikeWrapper(arr, label=123)
