import pytest

pytest.importorskip("jax")

import jax_cookbook.tree as jct


def test_zip_and_unzip():
    a = [1, 2]
    b = [3, 4]
    zipped = jct.zip_(a, b)
    assert zipped == [(1, 3), (2, 4)]

    unzipped = jct.unzip(zipped)
    assert unzipped == ([1, 2], [3, 4])


def test_zip_named():
    out, LeafTuple = jct.zip_named(a=[1, 2], b=[3, 4])
    assert isinstance(out[0], LeafTuple)
    assert out[0].a == 1 and out[0].b == 3


def test_map_unzip():
    def f(x):
        return x, x + 1

    ys, zs = jct.map_unzip(f, [1, 2])
    assert ys == [1, 2]
    assert zs == [2, 3]
