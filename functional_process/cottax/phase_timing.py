"""Where a row's wall clock goes: tracing, lowering, compiling, or solving."""

import collections
import contextlib
import time

_totals: collections.defaultdict = collections.defaultdict(float)
_stack: list = []
_installed = False


@contextlib.contextmanager
def phase(name: str):
    """Attribute this block's *exclusive* wall time to `name`."""
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
    """Patch jax's trace/lower/compile entry points."""
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
    """`totals()` plus `solve`, the residual of `total` after the measured phases."""
    measured = totals()
    accounted = sum(measured.values())
    return {**measured, "solve": max(total - accounted, 0.0)}
