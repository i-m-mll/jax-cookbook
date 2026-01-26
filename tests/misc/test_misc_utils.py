import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from jax_cookbook import misc


def test_interleave_unequal():
    out = list(misc.interleave_unequal([1, 2], ["a"], [True, False, None]))
    assert out == [1, "a", True, 2, False]


def test_unzip2():
    xs, ys = misc.unzip2([(1, "a"), (2, "b")])
    assert xs == (1, 2)
    assert ys == ("a", "b")


def test_str_always_lt():
    a = misc.StrAlwaysLT("a")
    b = misc.StrAlwaysLT("b")
    assert (a < b) is True
    assert (a > b) is False


def test_get_unique_label():
    assert misc.get_unique_label("x", {"x", "x_0"}) == "x_1"


def test_unique_generator():
    a = object()
    b = object()
    out = list(misc.unique_generator([a, a, b, b], replace_duplicates=True, replace_value=None))
    assert out == [a, None, b, None]


def test_nested_dict_update_make_copy():
    base = {"a": {"b": 1}, "c": 2}
    upd = {"a": {"d": 3}}
    out = misc.nested_dict_update(base, upd, make_copy=True)
    assert out == {"a": {"b": 1, "d": 3}, "c": 2}
    assert base == {"a": {"b": 1}, "c": 2}


def test_crop_to_shortest():
    @misc.crop_to_shortest(axis=0)
    def f(x, y):
        return x + y

    x = jnp.ones((5,))
    y = jnp.arange(3)
    out = f(x, y)
    assert out.shape == (3,)


def test_construct_tuple_like():
    from collections import namedtuple

    NT = namedtuple("NT", ["a", "b"])
    nt = misc.construct_tuple_like(NT, [1, 2])
    assert nt == NT(1, 2)


def test_deep_merge_preserves_types():
    base = {"a": {"b": 1}, "c": 2}
    over = {"a": {"d": 3}}
    out = misc.deep_merge(base, over)
    assert out == {"a": {"b": 1, "d": 3}, "c": 2}


def test_split_by():
    x = jnp.arange(6)
    a, b, c = misc.split_by(x, [2, 2, 2])
    assert jnp.array_equal(a, jnp.array([0, 1]))
    assert jnp.array_equal(c, jnp.array([4, 5]))


def test_fname():
    def f():
        pass

    assert misc._fname(f) == "f"
    assert misc._fname(object()) == "object"


def test_moving_avg():
    x = jnp.array([1, 2, 3, 4])
    out = misc.moving_avg(x, 2)
    assert jnp.allclose(out, jnp.array([1.5, 2.5, 3.5]))


def test_softmin():
    x = jnp.array([[0.0, 1.0]])
    out = misc.softmin(x, tau=1.0, axis=-1)
    assert out.shape == (1,)


def test_mse_tree_map():
    x = (jnp.array([1.0, 2.0]), jnp.array([3.0]))
    y = (jnp.array([1.0, 0.0]), jnp.array([5.0]))
    out = misc.mse(x, y)
    assert jnp.allclose(out[0], jnp.array(2.0))
    assert jnp.allclose(out[1], jnp.array(4.0))


def test_nan_safe_mse():
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.0, jnp.nan, 5.0])
    out = misc.nan_safe_mse(preds, targets)
    assert jnp.allclose(out, ((1 - 1) ** 2 + (3 - 5) ** 2) / 2)


def test_window_take_pad_and_clip():
    arr = jnp.arange(12).reshape(3, 4)  # axis 0 len=3, axis 1 len=4
    idxs = jnp.array([0, 2, 1])
    out_pad = misc.window_take(arr, idxs, bounds=(-1, 2), axis_p=1, axis_q=0, mode="pad")
    out_clip = misc.window_take(arr, idxs, bounds=(-1, 2), axis_p=1, axis_q=0, mode="clip")
    assert out_pad.shape == (3, 3)
    assert out_clip.shape == (3, 3)


def test_window_take_invalid_axes():
    arr = jnp.arange(6).reshape(2, 3)
    idxs = jnp.array([0, 1])
    with pytest.raises(ValueError):
        misc.window_take(arr, idxs, bounds=(0, 1), axis_p=0, axis_q=0)
