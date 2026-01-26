from jax_cookbook._print import highlight_string_diff, print_trees_side_by_side


def test_highlight_string_diff_contains_ansi():
    out = highlight_string_diff({"a": 1}, {"a": 2})
    assert "\033" in out


def test_print_trees_side_by_side(capsys):
    print_trees_side_by_side({"a": 1}, {"a": 2}, column_width=10)
    captured = capsys.readouterr().out
    assert "|" in captured
