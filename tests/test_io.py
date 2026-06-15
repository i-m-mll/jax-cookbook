import equinox as eqx
import jax
import jax.numpy as jnp

from jax_cookbook._io import load_with_hyperparameters, save


class TinyTree(eqx.Module):
    value: jax.Array


def test_save_load_round_trip_with_default_none_hyperparameters(tmp_path):
    path = tmp_path / "tree.eqx"
    tree = TinyTree(jnp.array([1.0, 2.0]))

    save(path, tree)

    def setup_func(*, key):
        del key
        return TinyTree(jnp.zeros((2,)))

    loaded, hps = load_with_hyperparameters(path, setup_func)

    assert hps == {}
    assert jnp.array_equal(loaded.value, tree.value)
