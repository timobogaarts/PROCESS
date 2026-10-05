"""One configuration solved under one architecture: the policy the port's studies share.

`problem.problem_model` is what the input file states; this is how it is answered.

- **The graph**: the configuration's model, and the architecture's op over it: `MDA` the
  machine with each cycle closed (`cx.mda`), `MDF` / `IDF` / `SAND` the problem at
  `problem.TOP` with the architecture around it. Every cycle is cut by `RULE`.
- **The drivers**: every problem answered by the port's own algorithm for its shape
  (`drivers_for`): Picard for a fixed point, a seeded Newton for a root find, VMCON for
  the optimisation.
- **The values**: every boundary input from the configuration's state (`truth`), a
  given value where the model gives one, and a start from the value of the place it
  starts -- the state's, else `cold_state`'s, one Gauss-Seidel sweep of the machine in
  binding order, which is how PROCESS itself starts. An integer stays a Python `int`,
  so a count sizes an array and is part of the jit key.
"""

import dataclasses
import functools
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import cottax.relational_pytree as cx
from cottax.core.frozen import OrderedMap
from cottax.core import is_problem
from cottax.core import Plan, Rename
from cottax.core.driven import is_driven
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface_types import ConditionMap, Posed
from cottax.relational_pytree import RootFinding
from cottax.relational.consumers.execution.jax_solvers import Newton, Picard
from cottax.pytree import unminted
from cottax.core import GaussSeidel, GaussSeidelMinimal, GraphWithValues
from cottax.relational import (
    Eq,
    Le,
    is_equalities,
    is_fixed_point,
)

from functional_process.configurations import Configuration, Optimisation
from functional_process.cottax.architectures.contracts import (
    Converged,
    PortFixedPointIteration,
    PortMinimisation,
    PortRootFinding,
    Start,
    Status,
    Steps,
)
from functional_process.cottax.architectures.drivers import SweepDriver, VmconDriver
from functional_process.cottax.input.native import state_of
from functional_process.cottax.models.constraints import MERIT, held
from functional_process.cottax.problem import TOP, problem_model
from functional_process.cottax.truth import ground_truth

RULE = GaussSeidelMinimal(18)
"""How every cycle is cut: a Gauss-Seidel sweep in the order that cuts the fewest
variables, searched exactly up to 18 nodes. One Picard iterate of a closure is then one
sweep of its cycle, which is how PROCESS's own idempotence loop runs."""

GIVEN_STARTS = {
    ".pf_coil.ind_pf_cs_plasma_mutual": 1.0,
    ".pf_coil.n_pf_coil_turns": 100.0,
}
"""A start given its value rather than read off the state, keyed by the quantity it
starts: a dataclass default of zero is no starting guess for these."""

SUPPLIED_STARTS = {
    ".stellarator.wp_width_r_min": ".stellarator.wp_width_r_min_guess",
}
"""A start read off a place a node computes, where one does, off the solve's cycle."""


# ---------------------------------------------------------------- the model


def machine_model(configuration: Configuration):
    """The machine, every constraint body and the merit, nothing held: what an MDA
    evaluates at the file's design."""
    problem = configuration.problem
    constraints = (
        *problem.equalities,
        *getattr(problem, "inequalities", ()),
        *getattr(problem, "evaluated", ()),
    )
    bodies = {held(c).segments[-1].name: c for c in constraints}
    if isinstance(problem, Optimisation):
        bodies["merit"] = problem.minimise or problem.maximise
    return cx.declare(configuration.machine, bodies)


# ---------------------------------------------------------------- the drivers


def picard():
    """cottax's Picard at the port's tolerances."""
    return Picard(rtol=1e-6, atol=1e-8)


def drivers_for(graph, fixed_point=picard, skip=()):
    """`{problem: (contract, implementation)}` for every problem in `graph` not yet
    driven and not in `skip`, by its shape."""
    chosen = {}
    for name, node in graph.definitions.items():
        if not is_problem(node) or is_driven(node) or name in skip:
            continue
        if is_fixed_point(node):
            chosen[name] = (PortFixedPointIteration(), fixed_point())
        elif is_equalities(node):
            chosen[name] = (PortRootFinding(), Newton())
        else:
            raise TypeError(f"{name.spelling}: no default driver for {node!r}")
    return chosen


def driven(plan, chosen):
    """`plan` with each problem assigned its contract and implemented, then the starts
    `SUPPLIED_STARTS` names read off the nodes that compute them."""
    for name, (contract, implementation) in chosen.items():
        plan = plan + cx.assign(plan.graph, name, contract)
    plan = plan + cx.implement({name: impl for name, (_c, impl) in chosen.items()})
    return _supplied(plan)


def starts(graph) -> dict:
    """`{start place: the unknown it starts}` over every driven problem in `graph`."""
    pairs = {}
    for node in graph.definitions.values():
        if is_driven(node) and Start in node.contract.takes:
            slot = node.data[node.contract.takes.index(Start)]
            pairs.update(zip(slot, node.unknowns, strict=True))
    return pairs


def _supplied(plan):
    graph = plan.graph
    owners = graph.graph.owners
    by_spelling = {v.spelling: v for v in owners}
    for name, node in graph.definitions.items():
        if not (is_driven(node) and Start in node.contract.takes):
            continue
        slot = node.data[node.contract.takes.index(Start)]
        onto = []
        for start, unknown in zip(slot, node.unknowns, strict=True):
            target = by_spelling.get(SUPPLIED_STARTS.get(unknown.spelling, ""))
            if target is not None and owners[target] not in graph.graph.descendants([
                name
            ]):
                onto.append((start, target))
        if onto:
            plan = plan + Rename(name, tuple(onto))
    return plan


# ---------------------------------------------------------------- the values


def _place(var):
    """The place `var` is a copy of, every mint stripped."""
    while (stripped := unminted(var)) != var:
        var = stripped
    return var


def _typed(value):
    """An integer as a Python `int` and an integer table as nested tuples of them
    (static under jit: it may size or index an array), a number as an array (traced),
    anything else (a table of the machine) as it is."""
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, jax.Array):
        return value
    try:
        array = np.asarray(value)
    except ValueError:  # a ragged table: not a number
        return value
    if array.dtype.kind in "iu":
        return _nested(array.tolist())
    return array if array.dtype.kind in "bfc" else value


def _nested(x):
    return tuple(map(_nested, x)) if isinstance(x, list) else x


def values(graph, state, given=OrderedMap(), cold=None) -> OrderedMap:
    """A value at every boundary input of `graph`: a given one, else the state's at the
    place (a start: at the unknown it starts), else `cold`'s, else `0.0`."""
    started = starts(graph)
    cold = cold or {}
    out = {}
    for var in graph.graph.boundary_inputs:
        if var in given:
            out[var] = given[var]
            continue
        source = started.get(var, var)
        try:
            value = ground_truth(state, source)
        except (AttributeError, KeyError):
            value = cold.get(_place(source), 0.0)
        if var in started and (fixed := GIVEN_STARTS.get(_place(source).spelling)):
            value = jnp.full_like(jnp.asarray(value, dtype=float), fixed)
        out[var] = _typed(value)
    return OrderedMap(out)


@eqx.filter_jit
def _run(graph, inputs):
    return run(GraphWithValues(graph, inputs))


@functools.cache
def proved(graph) -> CallableGraph:
    """`graph` proved callable, once: the proof walks every node, which costs several
    times a warm run of the graph itself."""
    return CallableGraph(graph)


def solve(graph, inputs) -> OrderedMap:
    """`run` of `graph` at `inputs`, compiled once per graph: an array input is traced,
    anything else (an integer, a table) is part of the compile key."""
    return _run(proved(graph), inputs)


# ---------------------------------------------------------------- the MDA


def cold_state(configuration: Configuration) -> dict:
    """`{place: value}` after one pass of the machine in binding order -- each cycle cut
    Gauss-Seidel in binding order, each fixed point applied once (`SweepDriver`), each
    root find solved -- from the state and zeros shaped as the MDA's own values. What a
    cut copy the state does not hold starts from."""
    state = state_of(configuration)
    model = machine_model(configuration)
    mda = mda_graph(model, RULE)
    shaped = eqx.filter_eval_shape(
        lambda v: run(GraphWithValues(CallableGraph(mda), v)),
        values(mda, state, model.given),
    )
    zeros = {
        _place(k): jax.tree.map(
            lambda s: jnp.zeros(s.shape, s.dtype) if hasattr(s, "shape") else s, v
        )
        for k, v in shaped.items()
    }
    sweep = mda_graph(model, GaussSeidel(), fixed_point=SweepDriver)
    out = solve(sweep, values(sweep, state, model.given, zeros))
    return {_place(k): v for k, v in out.items()}


def mda_plan(model, rule=RULE, fixed_point=picard) -> Plan:
    """`model` with every cycle closed by `rule` and every problem driven, as the plan
    that does it."""
    plan = Plan(model.graph) + cx.mda(rule)
    return driven(plan, drivers_for(plan.graph, fixed_point))


def mda_graph(model, rule=RULE, fixed_point=picard):
    """`mda_plan`'s graph."""
    return mda_plan(model, rule, fixed_point).graph


def conditions(configuration: Configuration, out) -> dict:
    """`objf`, `max_eq` and `min_ie` off a run's values."""
    problem = configuration.problem
    number = lambda c: float(np.asarray(out[held(c)]))  # noqa: E731
    equalities = [number(c) for c in problem.equalities]
    others = getattr(problem, "inequalities", None) or getattr(problem, "evaluated", ())
    inequalities = [number(c) for c in others]
    objf = None
    if isinstance(problem, Optimisation):
        merit = float(np.asarray(out[MERIT]))
        objf = merit if problem.minimise is not None else -merit
    return {
        "objf": objf,
        "max_eq": max(map(abs, equalities)) if equalities else 0.0,
        "min_ie": min(inequalities) if inequalities else None,
    }


# ---------------------------------------------------------------- the blocks

ARMS = ("MDA", "MDF", "IDF", "SAND")

MAX_ITER = 500
"""SQP iterations an optimising arm allows itself."""

TOLERANCE = 1.0e-6
"""The convergence parameter below which a trace's last iterate counts as converged:
`VmconDriver`'s own reading of PROCESS's `epsvmc` when none is set."""


def arms(configuration: Configuration) -> tuple[str, ...]:
    """The architectures `configuration` can be solved under: a root find has no
    optimisation to distribute, so an MDA and an MDF only."""
    return ARMS if isinstance(configuration.problem, Optimisation) else ARMS[:2]


def _architecture(arm: str, rule=RULE):
    return {"MDF": cx.mdf, "IDF": cx.idf, "SAND": cx.sand}[arm](TOP, rule)


def _is_bound(node, r) -> bool:
    fixed = lambda side: side is None or side in node.parameters  # noqa: E731
    return r.op is Le and (
        (r.rhs in node.unknowns and fixed(r.lhs))
        or (r.lhs in node.unknowns and fixed(r.rhs))
    )


def condition_scale(node, env, floor=1e-12) -> tuple:
    """`((condition, 1 / |value|), ...)` for every relation of `node` whose held side is
    a copy of one of its unknowns, measured in `env` (an MDA at the same design): the
    coupling conditions an SQP reads in the unknowns' own units. A constraint or the
    merit is left at 1."""
    unknowns = {_place(u): u for u in node.unknowns}
    scales = []
    for r in node.relations:
        if _is_bound(node, r):
            continue
        condition = r.lhs if r.lhs is not None else r.rhs
        if condition.spelling.startswith(".constraints.c"):
            continue
        unknown = unknowns.get(_place(condition))
        if unknown is None or unknown not in env:
            continue
        magnitude = float(np.max(np.abs(np.asarray(env[unknown], dtype=float))))
        usable = np.isfinite(magnitude) and magnitude > floor
        scales.append((condition, 1.0 / magnitude if usable else 1.0))
    return tuple(scales)


def block_plan(
    configuration: Configuration,
    arm: str,
    env,
    trace: list,
    rule=RULE,
    optimiser=VmconDriver,
    reports=True,
    *,
    model=None,
) -> Plan:
    """The problem under `arm`, every cycle cut by `rule` and every problem driven, as
    the plan that does it: the top by `optimiser` (or, for a root find, cottax's
    `Newton`), recording its iterates into `trace`. `optimiser` is a driver class, or a
    partial of one, taking VMCON's fields. `reports`: whether the top's contract asks
    its steps, verdict and status back. `env` (an MDA at the design) scales the
    coupling conditions; `{}` leaves them at 1. `model` is `problem_model`'s, if made."""
    model = problem_model(configuration) if model is None else model
    plan = Plan(model.graph) + _architecture(arm, rule)
    top = plan.graph[TOP]
    chosen = drivers_for(plan.graph, skip=(TOP,))
    if isinstance(configuration.problem, Optimisation):
        held_relations = [r for r in top.relations if not _is_bound(top, r)]
        chosen[TOP] = (
            PortMinimisation(reports=(Steps, Converged, Status) if reports else ()),
            optimiser(
                n_equality=sum(r.op is Eq for r in held_relations),
                n_inequality=sum(r.op is Le for r in held_relations),
                condition_scale=condition_scale(top, env),
                callback=_recorder(trace),
                max_iter=MAX_ITER,
            ),
        )
    else:
        chosen[TOP] = (RootFinding(steps=reports), Newton())
    return driven(plan, chosen)


def block_graph(configuration: Configuration, arm: str, env, trace: list, **how):
    """`(problem_model, block_plan's graph)`; `how` is `block_plan`'s."""
    model = problem_model(configuration)
    return model, block_plan(configuration, arm, env, trace, model=model, **how).graph


def _recorder(trace):
    def record(i, result, _x, convergence):
        trace.append((
            i,
            float(convergence),
            float(np.asarray(result.f)),
            float(np.max(np.abs(result.eq))) if len(result.eq) else 0.0,
            float(np.min(result.ie)) if len(result.ie) else 0.0,
        ))

    return record


def block_values(graph, state, given, env, design, nested_given: bool) -> OrderedMap:
    """A value at every boundary input of a block arm: a design variable's start from
    the state, every other start and every cut copy from `env` (an MDA at that design),
    anything else from the state, else `env`. `GIVEN_STARTS` applies to the starts of
    problems nested under the top where `nested_given`."""
    started = starts(graph)
    top_starts = set(graph[TOP].data[graph[TOP].contract.takes.index(Start)])
    out = {}
    for var in graph.graph.boundary_inputs:
        if var in given:
            out[var] = given[var]
            continue
        unknown = started.get(var)
        source = var if unknown is None else unknown
        coupling = unknown is not None and unknown not in design
        cut = unknown is None and unminted(var) != var and var in env
        if (coupling or cut) and source in env:
            value = env[source]
        else:
            try:
                value = ground_truth(state, source)
            except (AttributeError, KeyError):
                value = env.get(source, 0.0)
        if (
            nested_given
            and unknown is not None
            and var not in top_starts
            and (fixed := GIVEN_STARTS.get(_place(unknown).spelling))
        ):
            value = jnp.full_like(jnp.asarray(value, dtype=float), fixed)
        out[var] = _typed(value)
    return OrderedMap(out)


@dataclasses.dataclass(frozen=True)
class Assembled:
    """One arm built and valued, not yet solved: what `answer` solves."""

    arm: str
    graph: object
    """The driven graph the arm runs."""
    inputs: OrderedMap
    """A value at every boundary input of `graph`."""
    env: OrderedMap
    """The MDA at the configuration's design: every place's value."""
    trace: list
    """The top's iterates, `(i, convergence, objf, max|eq|, min ie)`, of the last
    solve (an optimising arm's)."""


def assemble(
    configuration: Configuration,
    arm: str,
    cold=None,
    *,
    rule=RULE,
    optimiser=VmconDriver,
    state=None,
) -> Assembled:
    """`arm` of `configuration` built, every cycle cut by `rule`, the top answered by
    `optimiser`, and valued from `state` (default the configuration's) over `cold`:
    the MDA solved once, the block not yet."""
    if arm not in arms(configuration):
        raise ValueError(f"{configuration.name} has no {arm} arm: {arms(configuration)}")
    cold = cold_state(configuration) if cold is None else cold
    state = state_of(configuration) if state is None else state
    model = machine_model(configuration)
    mda = mda_graph(model, rule)
    mda_inputs = values(mda, state, model.given, cold)
    env = solve(mda, mda_inputs)
    if arm == "MDA":
        return Assembled(arm, mda, mda_inputs, env, [])
    trace: list = []
    block_model, graph = block_graph(
        configuration, arm, env, trace, rule=rule, optimiser=optimiser
    )
    inputs = block_values(
        graph,
        state,
        block_model.given,
        env,
        set(configuration.problem.design),
        nested_given=arm in ("MDF", "IDF"),
    )
    return Assembled(arm, graph, inputs, env, trace)


def answer(configuration: Configuration, assembled: Assembled) -> dict:
    """`assembled` solved: `status`, `iterations`, `objf`, `max_eq`, `min_ie`,
    `seconds` and the design at the answer, `x`. The MDA arm is its analysis, run
    again."""
    graph, trace = assembled.graph, assembled.trace
    trace.clear()
    problem_ = configuration.problem
    started = time.perf_counter()
    try:
        out = solve(graph, assembled.inputs)
    except eqx.EquinoxRuntimeError:
        # cottax's Newton raises when it does not converge: the row says so.
        if isinstance(configuration.problem, Optimisation) or assembled.arm == "MDA":
            raise
        return {
            "status": "not-converged",
            "iterations": None,
            "seconds": time.perf_counter() - started,
            "x": (),
        }
    seconds = time.perf_counter() - started
    if assembled.arm == "MDA":
        return {
            "status": "evaluated",
            "iterations": 0,
            "seconds": seconds,
            "x": (),
            **conditions(configuration, out),
        }
    top = graph[TOP]
    reported = dict(zip(top.contract.reports, (s[0] for s in top.reports), strict=True))
    x = tuple(float(np.asarray(out[u])) for u in problem_.design)
    if isinstance(problem_, Optimisation):
        iterations, _f, _eq, _ie = (
            (len(trace), *trace[-1][2:]) if trace else (0, None, None, None)
        )
        status = (
            "no-step"
            if not trace
            else "converged"
            if trace[-1][1] <= TOLERANCE
            else f"cap({MAX_ITER})"
            if len(trace) >= MAX_ITER
            else "stopped"
        )
        if trace:
            result = {
                "objf": trace[-1][2],
                "max_eq": trace[-1][3],
                "min_ie": trace[-1][4],
            }
        else:
            result = conditions(configuration, out)
    else:
        iterations = int(np.asarray(out[reported[Steps]]))
        status = "converged"
        result = conditions(configuration, out)
    return {
        "status": status,
        "iterations": iterations,
        **result,
        "seconds": seconds,
        "x": x,
    }


def study(configuration: Configuration, arm: str, cold=None, **how) -> dict:
    """`configuration` solved under `arm`, cold: `answer(assemble(...))`. `how` is
    `assemble`'s `rule`, `optimiser` and `state`."""
    return answer(configuration, assemble(configuration, arm, cold, **how))


# ---------------------------------------------------------------- one problem, posed


def before(graph, inputs, problem=TOP) -> dict:
    """`inputs` and every place computed before `problem`'s cycle, run once: what the
    cycle reads from outside. `problem` is outermost."""
    cycle = set(graph.cycle_of(problem))
    upstream = [n for n in graph.graph.ancestors(cycle) if n not in cycle]
    env = dict(inputs)
    if upstream:
        sub = graph.subgraph(upstream)
        env.update(
            solve(sub, OrderedMap((v, inputs[v]) for v in sub.graph.boundary_inputs))
        )
    return env


class Interior(ConditionMap):
    """A problem's conditions as a function of its unknowns at a fixed context: one run
    of its interior (`graph.interior(problem)`, proved callable), the conditions read
    off every place. A value: it hashes and compares by its graph and places."""

    interior: CallableGraph
    unknowns: tuple
    context_places: tuple
    """What the interior reads from outside, the unknowns aside, then what the condition
    reads that the interior does not compute (a parameter): where `context` goes."""
    conditions: tuple
    """The condition's reads, its conditions then its parameters."""

    @classmethod
    def of(cls, graph, problem) -> "Interior":
        node = graph[problem]
        interior = graph.interior(problem)
        solved = set(node.unknowns)
        produced = set(interior.graph.owned_variables)
        fixed = tuple(
            dict.fromkeys(
                p
                for p in (*interior.graph.boundary_inputs, *node.condition.reads)
                if p not in solved and p not in produced
            )
        )
        return cls(CallableGraph(interior), node.unknowns, fixed, node.condition.reads)

    def __call__(self, unknowns, context) -> tuple:
        given = dict(zip(self.context_places, context, strict=True))
        given.update(zip(self.unknowns, unknowns, strict=True))
        inputs = self.interior.graph.graph.boundary_inputs
        ran = run(GraphWithValues(self.interior, OrderedMap((p, given[p]) for p in inputs)))
        return tuple(ran[p] if p in ran else given[p] for p in self.conditions)


def posed(graph, inputs, problem=TOP) -> tuple:
    """`(posed, context, data)`: what a run of `graph` at `inputs` hands `problem`'s
    driver, without solving it -- what runs before it once, then the problem posed as
    a function of its unknowns (`Interior`). `problem` is outermost."""
    env = before(graph, inputs, problem)
    node = graph[problem]
    conditions_ = Interior.of(graph, problem)
    fixed = conditions_.context_places
    data = {
        role: tuple(env[p] for p in slot)
        for role, slot in zip(node.contract.takes, node.data, strict=True)
    }
    return (
        Posed(node.condition, node.contract.reports, conditions_),
        tuple(env[p] for p in fixed),
        data,
    )


# ---------------------------------------------------------------- OpenMDAO


def openmdao(configuration: Configuration, arm: str = "MDF", cold=None, *, rule=RULE):
    """`arm` of `configuration` as cottax's OpenMDAO export: the top's cycle of the graph
    `assemble` builds -- what every iteration re-runs -- with what it reads from outside
    run once and handed in; the top answered by OpenMDAO's SLSQP in units of its
    design's starts (PROCESS's `x / x_start`), a solve by OpenMDAO's Newton or
    Gauss-Seidel. `.problem()` builds the `om.Problem`; `.values_in(problem)` reads it
    back."""
    from cottax.relational_pytree import ConstrainedMinimiser  # noqa: PLC0415
    from cottax.relational.consumers.openmdao import SOLVERS  # noqa: PLC0415
    from cottax.consumers.openmdao import (  # noqa: PLC0415
        OMSLSQP,
        OpenMDAOExport,
        OpenMDAOGraph,
        Sizing,
    )

    if arm not in arms(configuration) or arm == "MDA":
        raise ValueError(f"{configuration.name}: no {arm} arm to export")
    cold = cold_state(configuration) if cold is None else cold
    state = state_of(configuration)
    model = machine_model(configuration)
    mda = mda_graph(model, rule)
    env = solve(mda, values(mda, state, model.given, cold))
    block_model, graph = block_graph(
        configuration, arm, env, [], rule=rule, reports=False
    )
    inputs = block_values(
        graph,
        state,
        block_model.given,
        env,
        set(configuration.problem.design),
        nested_given=arm in ("MDF", "IDF"),
    )
    solvers = OrderedMap({
        **dict(SOLVERS),
        ConstrainedMinimiser: OMSLSQP(scaled=True),
    })
    env = before(graph, inputs)
    cycle = graph.subgraph(graph.cycle_of(TOP))
    outside = OrderedMap((v, env[v]) for v in cycle.graph.boundary_inputs)
    # Every input is held while SLSQP moves the design, which the top owns: no
    # derivative is asked in one, so none is taken.
    return OpenMDAOExport(
        Sizing(GraphWithValues(OpenMDAOGraph(cycle), outside)),
        solvers=solvers,
        constants=tuple(outside),
    )
