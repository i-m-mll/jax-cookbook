# test_hash_callable.py
import pytest
from jax_cookbook._func import hash_callable

# ────────────────────────────── helpers ──────────────────────────────

# Simple decorator factory so we can vary its argument
def multiply(k: int):
    def decorator(fn):
        def wrapper(*a, **kw):
            return k * fn(*a, **kw)
        return wrapper
    return decorator


# Callable-class whose behaviour depends on both a numeric factor
# *and* an optional nested callable stored in its instance state.
class Multiplier:
    def __init__(self, factor, func=lambda x: x):
        self.factor = factor
        self.func = func

    def __call__(self, x):
        return self.factor * self.func(x)


# ─────────────────────────── test batteries ──────────────────────────
def test_whitespace_and_parentheses_are_ignored():
    # Only cosmetic differences
    def f1(x):
        return (x + 1)

    def f2(x):
        return x+1

    assert hash_callable(f1) == hash_callable(f2)


def test_renaming_bound_locals_is_ignored():
    # Same semantics, different local variable names
    def make_a():
        alpha = 10
        return lambda: alpha + 1

    def make_b():
        beta = 10
        return lambda: beta + 1

    assert hash_callable(make_a()) == hash_callable(make_b())


def test_closure_cell_value_changes_hash():
    CONST = 1

    def make(val):
        local = val
        return lambda: (local, CONST)

    h1 = hash_callable(make(11))
    h2 = hash_callable(make(22))

    assert h1 != h2, "Changing captured local should change hash"


def test_referenced_global_name_changes_hash_even_if_value_same():
    GLOBAL_X = 1234
    GLOBAL_Y = 1234  # same value, different identifier

    def make_using_x():
        local = 0
        return lambda: (local, GLOBAL_X)

    def make_using_y():
        local = 0
        return lambda: (local, GLOBAL_Y)

    assert hash_callable(make_using_x()) != hash_callable(make_using_y())


def test_decorator_presence_changes_hash():
    def undecorated(z):               # noqa: E302
        return z + 1

    @multiply(2)
    def decorated(z):                 # noqa: E302
        return z + 1

    assert hash_callable(undecorated) != hash_callable(decorated)


def test_decorator_argument_changes_hash():
    @multiply(2)
    def f_two(x):                     # noqa: E302
        return x - 1

    @multiply(3)
    def f_three(x):                   # noqa: E302
        return x - 1

    assert hash_callable(f_two) != hash_callable(f_three)


def test_instance_state_changes_hash():
    m1 = Multiplier(2)
    m2 = Multiplier(3)  # different factor

    assert hash_callable(m1) != hash_callable(m2)


def test_callable_inside_instance_state_changes_hash():
    m1 = Multiplier(2, func=lambda t: t + 1)
    m2 = Multiplier(2, func=lambda t: t + 2)  # nested callable differs

    assert hash_callable(m1) != hash_callable(m2)


def test_defaults_that_contain_callables_recursed():
    def outer(fn=lambda y: y + 1):    # noqa: E302
        return fn(5)

    def outer_alt(fn=lambda y: y + 2):  # default lambda differs  # noqa: E302
        return fn(5)

    assert hash_callable(outer) != hash_callable(outer_alt)