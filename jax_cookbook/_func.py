from collections.abc import Callable
from functools import reduce


def anyf(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """Returns a function that returns the logical union of boolean functions.

    This is useful when we want to satisfy any of a number of `is_leaf`-like conditions
    without writing another ugly lambda. For example:

        `is_leaf=lambda x: is_module(x) or eqx.is_array(x)`

    becomes `is_leaf=anyf(is_module, eqx.is_array)`.
    """
    return lambda *args, **kwargs: any(f(*args, **kwargs) for f in funcs)


def allf(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """Returns a function that returns the logical intersection of boolean functions."""
    return lambda *args, **kwargs: all(f(*args, **kwargs) for f in funcs)


def notf(func: Callable[..., bool]) -> Callable[..., bool]:
    """Returns a function that returns the negation of the input function."""
    return lambda *args, **kwargs: not func(*args, **kwargs)


def compose(*funcs):
    """Compose a sequence of functions from left to right.
    
    Args:
        *funcs: Functions to compose, applied left to right (first to last)
        
    Returns:
        Composite function, whose arguments are those of the first function in `funcs`,
        and whose returns are those of the last.
    """
    def composite(f, g):
        return lambda *args, **kwargs: g(f(*args, **kwargs))
    
    return reduce(composite, funcs)


def is_type(*types) -> Callable[..., bool]:
    """Returns a function that returns `True` if the input is an instance of any of the given types."""
    return lambda x: any(isinstance(x, t) for t in types)


def is_not_type(*types) -> Callable[..., bool]:
    """Returns a function that returns `True` if the input is not an instance of any of the given types."""
    return lambda x: not is_type(*types)(x)


def idf(x):
    """The identity function."""
    return x