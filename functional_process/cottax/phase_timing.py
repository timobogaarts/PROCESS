"""Where a row's wall clock goes: tracing, lowering, compiling, or solving.

**Why this exists.** `_audit/optimise_design.md` §24 and §25 established that this port's
runtime is dominated by *compilation* and not by arithmetic -- one schedule lowers to
33 935 MLIR lines (132 125 for a tokamak), a single configuration peaks at 4.2 GB,
and `low_aspect_ratio_DEMO`'s 500 SQP iterations cost about 15 s of its ~160 s row.
Every one of those numbers was taken by hand, once, on one configuration, with a
throwaway script. A column on the table is the difference between knowing that and
re-deriving it.

**And one of them was wrong for a month.** `run_cold_matrix` printed
`"PROCESS: 16 VMCON iterations in 46.0 s"` from `reference.solve_seconds`, a field
*recorded on the pickled reference* when it was first computed -- not time spent in this
run. Reading it as elapsed made PROCESS look like a third of every row when the cached
load actually costs 4.6 s. A phase table that measures rather than reports is the fix.

How the split is taken
----------------------
Three jax entry points, patched once by `install()`:

- `partial_eval.trace_to_jaxpr_dynamic` -- **tracing**, python to jaxpr
- `mlir.lower_jaxpr_to_module` -- **lowering**, jaxpr to StableHLO
- `compiler.backend_compile_and_load` -- **compiling**, XLA proper

`solve` is then the remainder: total wall minus the three, minus whatever the caller
scopes explicitly (`assemble`, `process`). It is a residual on purpose -- a fourth patch
point would have to name every dispatch path, and the residual cannot silently lose time
the way an enumeration can.

**Times are exclusive, not inclusive.** Lowering happens *inside* a trace on a nested
`jit`, so accumulating naively double-counts. Each scope subtracts the time its children
consumed, so the four figures sum to the row's own wall clock and can be read as a
partition. That is what `_stack` is for, and it is why `phase` is a context manager
rather than a decorator: the nesting has to be observable.

Not thread-safe, and deliberately not: every harness in this tree is single-threaded, and
a lock here would be a second thing to be wrong about.
"""

import collections
import contextlib
import time

_totals: collections.defaultdict = collections.defaultdict(float)
_stack: list = []
_installed = False


@contextlib.contextmanager
def phase(name: str):
    """Attribute this block's *exclusive* wall time to `name`.

    Re-entrant and nesting-aware: an inner `phase` subtracts itself from every outer one,
    so `totals()` partitions the wall clock instead of over-counting it.
    """
    _stack.append(0.0)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        children = _stack.pop()
        _totals[name] += elapsed - children
        if _stack:
            _stack[-1] += elapsed


def install() -> bool:
    """Patch jax's trace/lower/compile entry points. Idempotent; returns whether it ran.

    Returns `False` on a jax whose internals have moved rather than raising: a phase
    breakdown is a diagnostic, and a matrix pass that refused to run because a private
    name changed would be a worse trade than a pass with an empty `trace` column. The
    caller reports which it got -- see `run_cold_matrix.render`'s timing block.
    """
    global _installed  # module-level patch, installed exactly once
    if _installed:
        return True
    try:
        # Private jax names, knowingly: there is no public hook for the three phases,
        # and `install` returns False rather than raising when they move.
        from jax._src import compiler  # noqa: PLC0415, PLC2701
        from jax._src.interpreters import mlir, partial_eval  # noqa: PLC0415, PLC2701
    except ImportError:
        return False

    for module, attr, name in (
        (partial_eval, "trace_to_jaxpr_dynamic", "trace"),
        (mlir, "lower_jaxpr_to_module", "lower"),
        (compiler, "backend_compile_and_load", "compile"),
    ):
        original = getattr(module, attr, None)
        if original is None:
            return False

        def wrapper(*args, _original=original, _name=name, **kwargs):
            with phase(_name):
                return _original(*args, **kwargs)

        setattr(module, attr, wrapper)
    _installed = True
    return True


def reset() -> None:
    """Forget every accumulated total. Call once per measurement, not per process."""
    _totals.clear()
    _stack.clear()


def totals() -> dict[str, float]:
    """Exclusive seconds per phase, for the phases that actually ran."""
    return dict(_totals)


def split(total: float) -> dict[str, float]:
    """`totals()` plus `solve`, the residual of `total` after the measured phases.

    `total` is the caller's own wall clock for the same span. `solve` can come out
    slightly negative if the span and the patches disagree about their boundaries by a
    few microseconds; it is clamped at zero rather than printed as a negative time,
    because a negative duration in a table reads as a defect in the thing measured
    rather than in the measuring.
    """
    measured = totals()
    accounted = sum(measured.values())
    return {**measured, "solve": max(total - accounted, 0.0)}
