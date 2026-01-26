import pytest

from jax_cookbook._func import allf, anyf, notf, wrap_to_accept_var_kwargs


def test_allf_includes_first_predicate():
    is_even = lambda x: x % 2 == 0
    gt_zero = lambda x: x > 0
    assert allf(is_even, gt_zero)(2) is True
    assert allf(is_even, gt_zero)(-2) is False
    assert allf(is_even, gt_zero)(3) is False


def test_anyf_includes_first_predicate():
    is_even = lambda x: x % 2 == 0
    gt_zero = lambda x: x > 0
    assert anyf(is_even, gt_zero)(2) is True
    assert anyf(is_even, gt_zero)(-2) is True
    assert anyf(is_even, gt_zero)(-3) is False


def test_notf_negates():
    is_even = lambda x: x % 2 == 0
    assert notf(is_even)(2) is False
    assert notf(is_even)(3) is True


def test_wrap_to_accept_var_kwargs_drops_unknown_by_default():
    def f(a, b=1):
        return a + b

    wrapped = wrap_to_accept_var_kwargs(f)
    assert wrapped(3, c=10) == 4


def test_wrap_to_accept_var_kwargs_raises_when_strict():
    def f(a, b=1):
        return a + b

    wrapped = wrap_to_accept_var_kwargs(f, strict=True)
    with pytest.raises(TypeError):
        wrapped(3, c=10)
