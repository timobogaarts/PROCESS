"""The optimiser's evaluation run eagerly: what the compiled evaluation costs without the
compiler around it.

`eager_graph(graph)` is `graph` with nothing compiled across nodes:

- every function node's body compiled on its own (`NodeJit`: one `jax.jit` per body,
  its non-array arguments static), so each node is one call -- what any eager graph
  runner pays, without jax's op-by-op dispatch inside a body, which is a property of
  writing the bodies in jax and not of the graph;
- every Picard and Newton solve stepped from Python (`EagerPicard`, `EagerNewton`):
  optimistix's own solver -- the same `init` / `step` / `terminate` / `postprocess`, in
  the order `optimistix._iterate` runs them -- with the loop a Python `while` instead of
  a compiled `while_loop`, so every iterate is a walk of the solve's interior.

cottax walks the graph in Python as it always does (`run`); the top's driver is not run,
the optimiser's problem being read as gaps (`bench.block`), as `iteration.py` reads it.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cottax.core import Plan
from cottax.assemblies.jax import FixedPointIterator, RootFinder
from cottax.consumers.execution.interface import DriverImplementation
from cottax.core.driven import Start, Steps
from cottax.core.realisation import (
    Implement,
    ImplementedDriven,
    ImplementedFunction,
    Unimplement,
)
from cottax.core.frozen import OrderedMap
from cottax.relational.consumers.execution.jax_solvers import Newton, Picard
from cottax.assemblies.jax import canonical
import contextlib
import types

import lineax as lx
import optimistix as optx
import optimistix._solver.newton_chord as _newton_chord

STEPS: list[tuple[str, int]] = []
"""`(solver, steps)` of every eager solve since it was last cleared."""

_JITTED: dict = {}
_STRUCTS: dict = {}


@dataclasses.dataclass(frozen=True)
class NodeJit:
    """A function node's body, compiled on its own the first time it is called: an
    array argument traced, anything else (an integer, a table) static."""

    fn: object

    def __call__(self, *args):
        jitted = _JITTED.get(self.fn)
        if jitted is None:
            static = tuple(
                i
                for i, a in enumerate(args)
                if not (leaves := jax.tree.leaves(a))
                or not all(isinstance(x, (jax.Array, np.ndarray)) for x in leaves)
            )
            jitted = _JITTED[self.fn] = jax.jit(self.fn, static_argnums=static)
        return jitted(*args)


class _Solution:
    def __init__(self, value, steps):
        self.value, self.stats = value, {"num_steps": steps}


class _NoAux(eqx.Module):
    """`fn` returning no auxiliary output, as optimistix wraps it. A module, equal across
    calls for the same `fn`, so what optimistix compiles inside a step (lineax's
    `linear_solve` of Newton's Jacobian) is compiled once and not on every evaluation."""

    fn: object

    def __call__(self, y, args):
        return self.fn(y, args), None


def _stepped(kind, solver, fn, y0, args, max_steps):
    """`optimistix._iterate._iterate` with its `while_loop` a Python loop."""
    f = _NoAux(fn)
    tags, options = frozenset(), {}
    y = jax.tree.map(jnp.asarray, tuple(y0))
    key = (
        args[0],
        jax.tree.structure(y),
        tuple(jnp.shape(v) for v in jax.tree.leaves(y)),
    )
    if (structs := _STRUCTS.get(key)) is None:
        structs = _STRUCTS[key] = eqx.filter_eval_shape(f, y, args)
    f_struct, aux_struct = structs
    state = solver.init(f, y, args, options, f_struct, aux_struct, tags)
    done, result = solver.terminate(f, y, args, options, state, tags)
    steps = 0
    while not bool(done) and result == optx.RESULTS.successful:
        if steps == max_steps:
            raise RuntimeError(f"eager {kind}: no convergence in {max_steps} steps")
        new_y, state, _ = solver.step(f, y, args, options, state, tags)
        done, result = solver.terminate(f, y, args, options, state, tags)
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(y)):
            raise RuntimeError(f"eager {kind}: a non-finite iterate")
        y, steps = new_y, steps + 1
    if result != optx.RESULTS.successful:
        raise RuntimeError(f"eager {kind}: {result}")
    y, _, _ = solver.postprocess(f, y, None, args, options, state, tags, result)
    STEPS.append((kind, steps))
    return _Solution(y, steps)


def _update(u, args):
    """Picard's function, `u <- g(u)`: each unknown's relation's right side, as
    cottax's own Picard reads it (`canonical_nlp.next_values`)."""
    posed, context = args
    condition = posed.condition
    value = dict(zip(condition.unknowns, u)) | dict(
        zip(condition.reads, posed.conditions(u, context))
    )
    update = {r.lhs: r.rhs for r in condition.relations}
    return tuple(value[update[unknown]] for unknown in condition.unknowns)


def _residuals(u, args):
    posed, context = args
    return canonical(posed.condition, u, posed.conditions(u, context)).equalities


class _Stepped(DriverImplementation):
    """What cottax's `Optimistix` drivers are, the solve stepped from Python: from a
    `Start`, reporting `Steps`, at the same tolerances and budget."""

    needs = frozenset({Start})
    provides = frozenset({Steps})

    rtol: float = 1e-4
    atol: float = 1e-4
    max_steps: int = 256

    def __call__(self, posed, context, data):
        if _traced((context, data[Start])):
            # Under a trace (what runs once before the timed evaluation, inside
            # `study.solve`), the compiled solve: a Python loop cannot branch on a tracer.
            solution = self.compiled(posed, context, data)
        else:
            solution = self.solve(posed, context, data)
        steps = {Steps: (solution.stats["num_steps"],)}
        return tuple(solution.value), steps if Steps in posed.reports else {}


def _traced(tree) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


class EagerPicard(_Stepped, DriverImplementation[FixedPointIterator]):
    """cottax's `Picard` (`optx.FixedPointIteration`), its loop stepped from Python."""

    def solve(self, posed, context, data):
        solver = optx.FixedPointIteration(rtol=self.rtol, atol=self.atol)
        return _stepped(
            "picard", solver, _update, data[Start], (posed, context), self.max_steps
        )

    def compiled(self, posed, context, data):
        solver = optx.FixedPointIteration(rtol=self.rtol, atol=self.atol)
        return optx.fixed_point(
            _update, solver, data[Start], args=(posed, context), max_steps=self.max_steps
        )


@contextlib.contextmanager
def _linear_solve_uncompiled():
    """optimistix's Newton step with lineax's `linear_solve` run as plain Python.
    Compiled, the step is traced once; stepped eagerly, it hands `linear_solve` (an
    `eqx.filter_jit`) a fresh `jax.linearize` closure every step, which recompiles it
    every step -- an artefact of stepping a compiled solver, not a cost of eager
    execution."""
    proxy = types.ModuleType("lineax_uncompiled")
    proxy.__dict__.update(lx.__dict__)
    proxy.linear_solve = lx.linear_solve.__wrapped__
    _newton_chord.lx = proxy
    try:
        yield
    finally:
        _newton_chord.lx = lx


class EagerNewton(_Stepped, DriverImplementation[RootFinder]):
    """cottax's `Newton` (`optx.Newton`), its loop stepped from Python, its linear
    solve uncompiled."""

    def compiled(self, posed, context, data):
        solver = optx.Newton(rtol=self.rtol, atol=self.atol)
        return optx.root_find(
            _residuals,
            solver,
            data[Start],
            args=(posed, context),
            max_steps=self.max_steps,
        )

    def solve(self, posed, context, data):
        solver = optx.Newton(rtol=self.rtol, atol=self.atol)
        with _linear_solve_uncompiled():
            return _stepped(
                "newton",
                solver,
                _residuals,
                data[Start],
                (posed, context),
                self.max_steps,
            )


EAGER = {Picard: EagerPicard, Newton: EagerNewton}


def eager_graph(graph, node_jit: bool = True, solvers: bool = True):
    """`graph` with, if `solvers`, every Picard and Newton stepped from Python and, if
    `node_jit`, every function body compiled on its own; anything else (the top's
    driver) as it was."""
    swapped = {}
    for name, node in graph.definitions.items():
        impl = getattr(node, "implementation", None)
        if solvers and isinstance(node, ImplementedDriven) and type(impl) in EAGER:
            swapped[name] = EAGER[type(impl)](
                rtol=impl.rtol, atol=impl.atol, max_steps=impl.max_steps
            )
        elif node_jit and isinstance(node, ImplementedFunction):
            swapped[name] = NodeJit(impl)
    names = tuple(swapped)
    return (Plan(graph) + Unimplement(names) + Implement(OrderedMap(swapped))).graph
