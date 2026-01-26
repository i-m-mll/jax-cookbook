import json
import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax.numpy as jnp
import numpy as np
import equinox as eqx

from jax_cookbook._io import (
    arrays_to_lists,
    json_dump,
    load,
    load_with_hyperparameters,
    save,
    save_old,
)


class M(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray


def setup_func(n: int, *, key, **_):
    del key
    return M(w=jnp.zeros((n,)), b=jnp.ones((n,)))


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "model.eqx"
    tree = M(w=jnp.arange(3.0), b=jnp.ones((3,)))
    save(path, tree, hyperparameters={"n": 3})
    loaded = load(path, setup_func)
    assert jnp.array_equal(loaded.w, tree.w)
    assert jnp.array_equal(loaded.b, tree.b)


def test_load_with_missing_hyperparameters(tmp_path):
    path = tmp_path / "model.eqx"
    tree = M(w=jnp.arange(4.0), b=jnp.ones((4,)))
    save(path, tree, hyperparameters={"n": 4})
    loaded, hps = load_with_hyperparameters(
        path,
        setup_func,
        missing_hyperparameters={"n": 99, "extra": {"x": 1}},
    )
    assert hps["n"] == 4
    assert "extra" in hps
    assert jnp.array_equal(loaded.w, tree.w)


def test_save_old_compatible(tmp_path):
    path = tmp_path / "model_old.eqx"
    tree = M(w=jnp.arange(2.0), b=jnp.ones((2,)))
    save_old(path, tree, hyperparameters={"n": 2})
    loaded = load(path, setup_func)
    assert jnp.array_equal(loaded.w, tree.w)


def test_arrays_to_lists():
    tree = {"a": jnp.array([1, 2]), "b": np.array([3, 4]), "c": "x"}
    out = arrays_to_lists(tree)
    assert out["a"] == [1, 2]
    assert out["b"] == [3, 4]
    assert out["c"] == "x"


def test_json_dump(tmp_path):
    path = tmp_path / "hps.json"
    with path.open("wb") as f:
        json_dump(f, {"a": 1})
    with path.open("rb") as f:
        data = json.loads(f.read().decode())
    assert data == {"a": 1}
