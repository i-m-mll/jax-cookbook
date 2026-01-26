import pytest

from jax_cookbook._func import (
    bundle,
    compose,
    compose_,
    fingerprint_details,
    identity,
    is_not_type,
    is_type,
)


def test_compose_left_to_right():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    h = compose_(f, g)
    assert h(3) == 8  # (3+1)*2


def test_compose_then_typing():
    def f(x: int) -> int:
        return x + 1

    def g(x: int) -> int:
        return x * 2

    h = compose(f).then(g)
    assert h(3) == 8


def test_bundle():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    b = bundle(f, g)
    assert b(3) == (4, 6)


def test_is_type_and_is_not_type():
    pred = is_type(int, str)
    assert pred(1) is True
    assert pred(1.0) is False

    npred = is_not_type(int, str)
    assert npred(1.0) is True
    assert npred("x") is False


def test_identity():
    obj = object()
    assert identity(obj) is obj


def test_fingerprint_details_returns_consistent_keys():
    def f(x):
        return x + 1

    digest, details = fingerprint_details(f)
    assert isinstance(digest, str)
    assert isinstance(details, dict)
    assert all(isinstance(k, str) for k in details.keys())
