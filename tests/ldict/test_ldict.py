import pytest

pytest.importorskip("jax")

import jax.tree as jt

from jax_cookbook._ldict import LDict
from jax_cookbook._ldict import is_ldict_of


def test_ldict_basic_mapping_and_label():
    ld = LDict("foo", {"a": 1, "b": 2})
    assert ld.label == "foo"
    assert ld["a"] == 1
    assert list(ld.keys()) == ["a", "b"]


def test_ldict_repr_multiline():
    ld = LDict("foo", {"a": 1, "b": {"c": 2}})
    s = repr(ld)
    assert "LDict.of('foo')" in s


def test_ldict_pytree_roundtrip():
    ld = LDict("foo", {"a": 1, "b": 2})
    leaves, treedef = jt.flatten(ld)
    rebuilt = jt.unflatten(treedef, leaves)
    assert isinstance(rebuilt, LDict)
    assert rebuilt.label == "foo"
    assert dict(rebuilt.items()) == {"a": 1, "b": 2}


def test_ldict_fromkeys_and_of():
    ld = LDict.fromkeys("bar", ["x", "y"], 0)
    assert isinstance(ld, LDict)
    assert ld.label == "bar"
    cons = LDict.of("baz")
    ld2 = cons(x=1, y=2)
    assert ld2.label == "baz"


def test_ldict_is_of_and_predicate():
    cons = LDict.of("foo")
    ld = cons(a=1)
    assert LDict.is_of("foo")(ld) is True
    assert cons.predicate(ld) is True
    assert LDict.is_of("bar")(ld) is False
    assert is_ldict_of("foo")(ld) is True


def test_ldict_merge_ops():
    ld = LDict("foo", {"a": 1})
    out = ld | {"b": 2}
    assert isinstance(out, LDict)
    assert out.label == "foo"
    assert dict(out.items()) == {"a": 1, "b": 2}

    left = {"z": 0}
    out2 = left | ld
    assert isinstance(out2, dict)
    assert out2 == {"z": 0, "a": 1}

    out3 = ld | LDict("foo", {"a": 2})
    assert out3["a"] == 2
