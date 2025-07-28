from collections.abc import Callable
from functools import reduce
import hashlib
import inspect
from operator import not_
import pickle
import re
import types
from typing import Any


def anyf(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """Returns a function that returns the logical union of boolean functions.

    This is useful when we want to satisfy any of a number of `is_leaf`-like conditions
    without writing another ugly lambda. For example:

        `is_leaf=lambda x: is_module(x) or eqx.is_array(x)`

    becomes `is_leaf=anyf(is_module, eqx.is_array)`.
    """
    return lambda *args, **kwargs: any(
        f(*args, **kwargs) for f in funcs
    )


def allf(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """Returns a function that returns the logical intersection of boolean functions."""
    return lambda *args, **kwargs: all(
        f(*args, **kwargs) for f in funcs
    )


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
    if len(funcs) == 1:
        return funcs[0]
    
    def composite(f, g):
        return lambda *args, **kwargs: g(f(*args, **kwargs))
    
    return reduce(composite, funcs)

    # return reduce(lambda f, g: lambda x: g(f(x)), funcs)


def is_type(*types) -> Callable[..., bool]:
    """Returns a function that returns `True` if the input is an instance of any of the given types."""
    return lambda x: any(
        isinstance(x, t) for t in types
    )


def is_not_type(*types) -> Callable[..., bool]:
    """Returns a function that returns `True` if the input is not an instance of any of the given types."""
    return compose(not_, is_type(*types))


def identity(x):
    """The identity function."""
    return x


# hash_callable.py
import inspect
import hashlib
import pickle
import types
import re
from collections import OrderedDict
from typing import Any, Dict, Set, Tuple

# --------------------------------------------------------------------------- #
# -- JAX tree.leaves helper (safe if JAX absent)                              #
# --------------------------------------------------------------------------- #
try:
    from jax.tree_util import leaves as _jax_leaves
except Exception:  # pragma: no cover
    def _jax_leaves(x):            # type: ignore[return-value]
        raise TypeError            # treat anything as non-PyTree


def _walk(v: Any) -> list[Any]:
    """
    Yield *v* itself **plus** all its PyTree leaves (if any),
    with duplicates removed (by object identity).
    """
    items = [v]
    try:
        items.extend(_jax_leaves(v))
    except Exception:
        pass

    uniq, seen = [], set()
    for obj in items:
        oid = id(obj)
        if oid not in seen:
            seen.add(oid)
            uniq.append(obj)
    return uniq


# --------------------------------------------------------------------------- #
# -- utility helpers                                                          #
# --------------------------------------------------------------------------- #
def _norm_code(co: types.CodeType) -> types.CodeType:
    """Strip line-number metadata so identical logic hashes the same."""
    try:
        kwargs = {"co_firstlineno": 0}
        if hasattr(co, "co_linetable"):
            kwargs["co_linetable"] = b""
        elif hasattr(co, "co_lnotab"):
            kwargs["co_lnotab"] = b""
        return co.replace(**kwargs)
    except Exception:                                # pragma: no cover
        return co


def _bytes_for_constant(obj: Any) -> bytes:
    """Deterministic bytes for a non-callable constant."""
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:                                # pragma: no cover
        return repr(obj).encode()


# --------------------------------------------------------------------------- #
# -- private collector ------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _collect_parts(
    obj: Any,
    parts: OrderedDict[str, bytes] | None = None,
    *,
    _memo_obj: Set[int] | None = None,
    _memo_code: Set[int] | None = None,
) -> OrderedDict[str, bytes]:
    """
    Populate *parts* with section-key → raw-bytes for everything that
    influences the behaviour of *obj* (callable or code object).
    """
    if parts is None:
        parts = OrderedDict()
    if _memo_obj is None:
        _memo_obj = set()
    if _memo_code is None:
        _memo_code = set()

    # -- code objects ---------------------------------------------------- #
    if isinstance(obj, types.CodeType):
        if id(obj) in _memo_code:
            return parts
        _memo_code.add(id(obj))

        co = _norm_code(obj)
        parts[f"code:{id(co)}"] = co.co_code

        for idx, const in enumerate(co.co_consts):
            if isinstance(const, types.CodeType) or callable(const):
                for leaf in _walk(const):
                    _collect_parts(
                        leaf, parts, _memo_obj=_memo_obj, _memo_code=_memo_code
                    )
            else:
                parts[f"const:{id(co)}:{idx}"] = _bytes_for_constant(const)
        return parts

    # non-callables do nothing
    if not callable(obj):
        return parts

    # -- callables (functions, methods, callable instances) -------------- #
    if id(obj) in _memo_obj:
        return parts
    _memo_obj.add(id(obj))

    if isinstance(obj, types.FunctionType):
        func, inst_state = obj, None
    elif inspect.ismethod(obj):
        func, inst_state = obj.__func__, None
    elif hasattr(obj, "__call__"):
        bound = obj.__call__
        func = getattr(bound, "__func__", bound)
        inst_state = getattr(obj, "__dict__", None)
    else:                                          # pragma: no cover
        raise TypeError(f"{obj!r} is not a pure-Python callable")

    # 1) byte-code + nested constants ------------------------------------ #
    _collect_parts(func.__code__, parts, _memo_obj=_memo_obj, _memo_code=_memo_code)

    # 2) decorators ------------------------------------------------------- #
    deco_bytes = b""
    try:
        src, _ = inspect.getsourcelines(func)
        pre_def = []
        for line in src:
            s = line.lstrip()
            if s.startswith("@"):
                pre_def.append(line)
            elif s.startswith(("def ", "async def ")):
                break
            else:
                break
        if pre_def:
            deco_bytes = re.sub(r"\s+", "", "".join(pre_def)).encode()
    except (OSError, IOError):
        pass
    parts[f"decorators:{id(func)}"] = deco_bytes

    # 3) defaults --------------------------------------------------------- #
    defaults_seq = (func.__defaults__ or ()) + tuple(
        (getattr(func, "__kwdefaults__", {}) or {}).values()
    )
    if defaults_seq:
        blob = bytearray()
        for v in defaults_seq:
            for leaf in _walk(v):
                _collect_parts(
                    leaf, parts, _memo_obj=_memo_obj, _memo_code=_memo_code
                )
            if not callable(v):
                blob.extend(_bytes_for_constant(v))
        parts[f"defaults:{id(func)}"] = bytes(blob)

    # 4) closure cells (include names only when >1 cell) ------------------ #
    if func.__closure__:
        blob = bytearray()
        include_names = len(func.__closure__) > 1
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            if include_names:
                blob.extend(name.encode())
            val = cell.cell_contents
            for leaf in _walk(val):
                _collect_parts(
                    leaf, parts, _memo_obj=_memo_obj, _memo_code=_memo_code
                )
            if not callable(val):
                blob.extend(_bytes_for_constant(val))
        parts[f"closure:{id(func)}"] = bytes(blob)

    # 5) referenced globals ---------------------------------------------- #
    if func.__code__.co_names:
        blob = bytearray()
        g = func.__globals__
        for name in sorted(func.__code__.co_names):
            if name in g:
                blob.extend(name.encode())
                val = g[name]
                for leaf in _walk(val):
                    _collect_parts(
                        leaf, parts, _memo_obj=_memo_obj, _memo_code=_memo_code
                    )
                if not callable(val):
                    blob.extend(_bytes_for_constant(val))
        parts[f"globals:{id(func)}"] = bytes(blob)

    # 6) instance state (callable-class) ---------------------------------- #
    if inst_state:
        blob = bytearray()
        for k in sorted(inst_state):
            blob.extend(k.encode())
            v = inst_state[k]
            for leaf in _walk(v):
                _collect_parts(
                    leaf, parts, _memo_obj=_memo_obj, _memo_code=_memo_code
                )
            if not callable(v):
                blob.extend(_bytes_for_constant(v))
        parts[f"state:{id(obj)}"] = bytes(blob)

    return parts


# --------------------------------------------------------------------------- #
# -- public helpers ---------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def hash_callable(obj: Any) -> str:
    """Return a single SHA-256 hex digest for *obj*’s behaviour."""
    parts = _collect_parts(obj)
    return hashlib.sha256(b"".join(parts.values())).hexdigest()


def fingerprint_details(obj: Any) -> Tuple[str, OrderedDict[str, str]]:
    """
    Return (**digest**, **details**).  *details* maps section-keys to their
    individual SHA-256 digests, useful for diffing.
    """
    parts = _collect_parts(obj)
    digest = hashlib.sha256(b"".join(parts.values())).hexdigest()
    details = OrderedDict((k, hashlib.sha256(v).hexdigest()) for k, v in parts.items())
    return digest, details