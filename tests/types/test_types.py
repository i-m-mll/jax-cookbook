import pytest

pytest.importorskip("equinox")

import equinox as eqx

from jax_cookbook._types import is_module, is_none


class Mod(eqx.Module):
    x: int


def test_is_module():
    assert is_module(Mod(1)) is True
    assert is_module(1) is False


def test_is_none():
    assert is_none(None) is True
    assert is_none(0) is False
