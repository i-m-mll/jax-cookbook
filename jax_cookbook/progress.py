import atexit
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

import jax.tree as jt
from jaxtyping import PyTree
from rich import get_console
from rich.progress import (
    BarColumn,
    # IterationSpeedColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.auto import tqdm as _tqdm

from jax_cookbook.tree import HR

S = TypeVar("S")
T = TypeVar("T")


tqdm_mode = os.environ.get("FEEDBAX_TQDM", "auto")
_tqdm_write = partial(_tqdm.write, file=sys.stdout, end="")


# class ProgressBottom(Progress):
#     """Render tasks bottom-up: the first-added task appears at the bottom."""

#     # Rich builds the table from a list of Task objects; we just reverse that list.
#     def make_tasks_table(self, tasks: Iterable[Task]):  # type: ignore[override]
#         # Convert to list and reverse the order before delegating to parent
#         tasks_list = list(tasks)[::-1]
#         for t in tasks_list:
#             t.fields.setdefault("subdescription", "")
#         return super().make_tasks_table(tasks_list)


_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn(
        "{task.fields[subdescription]}",
        style="dim",
        justify="right",
    ),
    # IterationSpeedColumn(),
]
_PROG: Optional[Progress] = None
_RC = 0  # refcount of active users


def _start_progress(*, redirect_print: bool = True, **kwargs) -> Progress:
    global _PROG
    if _PROG is None:
        _PROG = Progress(
            *_COLUMNS,
            console=get_console(),  # same console as your RichHandler
            redirect_stdout=redirect_print,
            redirect_stderr=redirect_print,
            **kwargs,
        )
        _PROG.start()
    return _PROG


def _stop_progress() -> None:
    global _PROG
    if _PROG is not None:
        _PROG.stop()  # transient=True clears the area
        _PROG = None


@atexit.register
def _shutdown_progress():
    # Safety: ensure clean teardown on process exit.
    _stop_progress()


def retain_progress(
    *,
    transient: bool = True,  # clear area when last task finishes
    redirect_print: bool = True,  # keep prints/logs above bars
    refresh_per_second: float = 10,
    speed_estimate_period: float = 1.0,  # smoothing window for speed estimate
) -> Progress:
    """Acquire the shared Progress (starting it if needed)."""
    global _RC
    prog = _start_progress(
        transient=transient,
        redirect_print=redirect_print,
        refresh_per_second=refresh_per_second,
        speed_estimate_period=speed_estimate_period,
    )
    _RC += 1
    return prog


def release_progress() -> None:
    """Release a previous retain; stops/clears when the last user leaves."""
    global _RC
    _RC = max(0, _RC - 1)
    if _RC == 0:
        _stop_progress()


@contextmanager
def progress_session():
    """Keep the global Progress alive for the duration of this context."""
    retain_progress()  # increments the refcount; starts the hub if needed
    try:
        yield  # tasks can be created/removed freely inside
    finally:
        release_progress()  # decrements; stops hub at the very end (one newline)


@contextmanager
def progress_task(
    description: str,
    total: Optional[int] = None,
    *,
    completed: int = 0,
    transient: bool = True,
    redirect_print: bool = True,
    refresh_per_second: float = 10,
    speed_estimate_period: float = 1.0,  # smoothing window for speed estimate
) -> Iterator[Callable[[int], None]]:
    """
    Create a shared task, yield an `advance(n=1)` function; task is removed automatically.
    """
    prog = retain_progress(
        transient=transient,
        redirect_print=redirect_print,
        refresh_per_second=refresh_per_second,
        speed_estimate_period=speed_estimate_period,
    )
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription=""
    )
    try:
        yield lambda n=1: prog.advance(task_id, n)
    finally:
        prog.remove_task(task_id)
        release_progress()


@dataclass
class PiterUpdate:
    _update: Callable[..., None]
    _advance: Callable[[int], None]

    # Convenience methods you can call from *inside* your loop body:
    def subdescription(self, text: object) -> None:
        self._update(subdescription=str(text))

    def description(self, text: object) -> None:
        self._update(description=str(text))

    def advance(self, n: int = 1) -> None:
        self._advance(n)

    def completed(self, n: int) -> None:
        self._update(completed=int(n))

    def total(self, n: int) -> None:
        self._update(total=int(n))


@contextmanager
def progress_piter(
    iterable: Iterable[T],
    *,
    description: str = "Working…",
    total: Optional[int] = None,
    completed: int = 0,
    auto_advance: bool = True,  # auto-advance after each yielded item
):
    """
    Context manager yielding (iterator, update).

    Use `update.subdesc(...)` or `update.desc(...)` *inside* your loop body.
    """
    if total is None:
        try:
            total = len(iterable)  # may fail for generators
        except Exception:
            total = None

    prog = retain_progress()
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription=""
    )
    try:

        def _update(**kwargs):
            prog.update(task_id, **kwargs)

        def _advance(n=1):
            prog.advance(task_id, n)

        updater = PiterUpdate(_update=_update, _advance=_advance)

        def _iter() -> Iterator[T]:
            i = completed
            for item in iterable:
                yield item
                if auto_advance:
                    i += 1
                    _advance(1)

        yield _iter(), updater
    finally:
        prog.remove_task(task_id)
        release_progress()


def piter(
    iterable: Iterable[T],
    *,
    description: str = "Working…",
    total: Optional[int] = None,
    completed: int = 0,
    right: Optional[Callable[[T, int], str]] = None,  # label function(item, index)
    transient: bool = True,
    redirect_print: bool = True,
    refresh_per_second: float = 10,
    speed_estimate_period: float = 1.0,  # smoothing window for speed estimate
) -> Iterator[T]:
    """
    Iterator wrapper sharing the global Progress.
    """
    if total is None:
        try:
            total = len(iterable)  # may fail (generators)
        except Exception:
            total = None

    prog = retain_progress(
        transient=transient,
        redirect_print=redirect_print,
        refresh_per_second=refresh_per_second,
        speed_estimate_period=speed_estimate_period,
    )
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription=""
    )
    try:
        for i, item in enumerate(iterable):
            if right is not None:
                try:
                    prog.update(task_id, subdescription=right(item, i))
                except Exception:
                    pass  # don't break progress on label errors
            yield item
            prog.advance(task_id, 1)
    finally:
        prog.remove_task(task_id)
        release_progress()


def map_rich(
    f: Callable[..., S],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    description: str = "Processing tree leaves",
    labels: Optional[PyTree[str, "T"]] = None,
    verbose: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO,
    **kwargs,  # for `retain_progress`
) -> PyTree[S, "T"]:
    """Adds a Rich progress task to a shared global Progress (stacks with others)."""
    n_leaves = len(jt.leaves(tree, is_leaf=is_leaf))

    # We still grab the global console for fallback logging
    console = get_console()

    # NEW: acquire the shared Progress instead of creating a local one
    prog = retain_progress(**kwargs)
    task_id = prog.add_task(description, total=n_leaves, subdescription="")
    try:
        # Prepare labels tree (match your tqdm version)
        if labels is None:
            labels = jt.map(lambda _: None, tree, is_leaf=is_leaf)

        def _log(msg: str):
            if verbose:
                if logger is not None:
                    logger.log(log_level, msg)
                else:
                    console.log(msg)

        def _f(leaf, leaf_label, *rest_args):
            if leaf_label is not None:
                prog.update(task_id, subdescription=leaf_label)
            if verbose:
                _log(f"Processing leaf: {leaf_label}")

            result = f(leaf, *rest_args)

            prog.advance(task_id, 1)
            return result

        return jt.map(_f, tree, labels, *rest, is_leaf=is_leaf)
    finally:
        # Remove just this task (so it vanishes immediately)
        prog.remove_task(task_id)
        # Release the shared Progress; if this was the last user, it stops (and clears if transient)
        release_progress()


# TODO: Use a host callback so this can be wrapped in JAX transformations.
# See https://github.com/jeremiecoullon/jax-tqdm for a similar example.
# (Currently I only use this function when `f` is a `TaskTrainer`.)
def map_tqdm(
    f: Callable[..., S],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    label: Optional[str] = None,
    labels: Optional[PyTree[str, "T"]] = None,
    verbose: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[S, "T"]:
    """Adds a tqdm progress bar to `tree_map`.

    Arguments:
        f: The function to map over the tree.
        tree: The PyTree to map over.
        *rest: Additional arguments to `f`, as PyTrees with the same structure as `tree`.
        label: A single label to be displayed irrespective of the leaf being processed.
            Overridden by `labels`.
        labels: A PyTree of labels for the leaves of `tree`, to be displayed on the
            progress bar.
        is_leaf: A function that returns `True` for leaves of `tree`.
    """
    n_leaves = len(jt.leaves(tree, is_leaf=is_leaf))
    pbar = _tqdm(total=n_leaves, desc=label)

    def _f(leaf, label, *rest):
        if label is not None:
            pbar.set_description(f"Processing leaf: {label}")
        if verbose:
            _tqdm_write(f"Processing leaf: {label}\n\n")
        result = f(leaf, *rest)
        if verbose:
            _tqdm_write(f"\n{HR}\n")
        else:
            _tqdm_write("\n")
        pbar.update(1)
        return result

    if labels is None:
        pbar.set_description("Processing tree leaves")
        labels = jt.map(lambda _: None, tree, is_leaf=is_leaf)
    return jt.map(_f, tree, labels, *rest, is_leaf=is_leaf)
