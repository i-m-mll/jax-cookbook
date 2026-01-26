import pytest

pytest.importorskip("rich")
pytest.importorskip("tqdm")
pytest.importorskip("jax")

import jax.numpy as jnp

import jax_cookbook.progress as prog


def test_retain_release_progress():
    p = prog.retain_progress(transient=True)
    assert p is not None
    prog.release_progress()
    assert prog._RC == 0


def test_progress_task_context():
    with prog.progress_task("test", total=3) as advance:
        advance(1)
        advance(2)


def test_progress_piter():
    data = [1, 2, 3]
    with prog.progress_piter(data, description="test") as (it, update):
        for i, x in enumerate(it):
            update.subdescription(f"{i}")
            assert x in data


def test_piter():
    data = [1, 2, 3]
    out = list(prog.piter(data, description="test"))
    assert out == data


def test_map_rich():
    tree = [jnp.array([1, 2]), jnp.array([3, 4])]
    out = prog.map_rich(lambda x: x + 1, tree, description="test")
    assert jnp.array_equal(out[0], jnp.array([2, 3]))


def test_map_tqdm():
    tree = [1, 2, 3]
    out = prog.map_tqdm(lambda x: x + 1, tree)
    assert out == [2, 3, 4]


def test_per_task_rate_helpers():
    from types import SimpleNamespace

    task = SimpleNamespace(id=1, completed=0, fields={"eta_halflife": 0.1}, total=10)
    rate = prog._PerTaskExpDecayRate()
    assert rate.update_and_get_rate(task) is None
    task.completed = 1
    r = rate.update_and_get_rate(task)
    assert r is None or r >= 0

    win = prog._PerTaskWindowRate(window_s=0.1)
    assert win.update_and_get_rate(task) is None


def test_progress_columns_render():
    from types import SimpleNamespace

    task = SimpleNamespace(id=1, completed=0, fields={"eta_halflife": 0.1}, total=5)
    rate = prog._PerTaskExpDecayRate()
    speed_col = prog.PerTaskSpeedColumn(rate)
    eta_col = prog.PerTaskETAColumn(rate)
    assert "it/s" in str(speed_col.render(task))
    assert str(eta_col.render(task))


def test_display_rich_text_themes_runs():
    prog.display_rich_text_themes()
