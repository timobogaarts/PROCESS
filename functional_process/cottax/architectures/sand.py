"""The optimisation problem this port states, and its **SAND** assembly (Simultaneous
ANalysis and Design).

The problem is two kinds of node, both stated the way the models are:

- one **constraint** declaration per active `icc` (`models/constraints.py`): a body
  owning `.constraints.c<id>` at `Constraint<id>`, and beside it -- as part of
  the same declaration -- the requirement that holds it against zero,
  `^require.Constraint<id>`, `c = 0` for the first `n_equality` and `c <= 0` for the
  rest. That is what makes a constraint a constraint, and it is declared where the
  constraint is computed;
- one **objective** node owning `.numerics.objf`, and `Optimise(objf, design)` at
  `Opt`, with no constraints of its own.

An architecture (`SAND(rule, closing, OPT)`, `MDF(...)`, `IDF(...)`) absorbs every requirement
the design reaches, cuts what is still cyclic, and combines or nests every problem on
the optimiser's cycle its own way.
"""

import dataclasses
import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.interfaces import (
    Call,
    Delete,
    Drive,
    Eq,
    Insert,
    Le,
    Plan,
    RunnableGraph,
    Schedule,
    is_fixed_point,
    is_optimise,
    is_problem,
    requirements,
)
from cottax.interfaces.pytree_namespace_module import to_graph
from cottax.interfaces.statements import Optimise
from cottax.mdao_architectures import SAND
from cottax.pytree.mint import unminted
from cottax.pytree.path import NodePath, PathMap, VarPath
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey, SequenceKey

from functional_process.cottax.architectures.mda import (
    assign_drivers,
    default_drivers,
    relation_counts,
)
from functional_process.cottax.models.constraints import (
    SWITCH_PARAMETER_NAMES,
    condition_place,
    constraint_declaration,
    constraint_place,
    requirement_place,
)
from functional_process.cottax.models.objectives import objective_node, objective_place
from functional_process.cottax.queries import declared
from functional_process.models import constraints as ported_constraints
from functional_process.models import objectives as ported_objectives
from functional_process.vocabulary import (
    AREAS,
    ITERATION_VARIABLES,
    FiguresOfMerit,
)

OPT = NodePath((GetAttrKey("Opt"),))
"""Where the run's `Optimise` binds."""


def switch_values_for(data, icc, i_figure_merit):
    """Static switch arguments for one run, read off its **initialised** `DataStructure`
    -- `init_process`'s answer to the file plus PROCESS's own defaults, so no default is
    transcribed here.
    """
    needed = set()
    for cid in icc:
        fn = getattr(ported_constraints, f"constraint_{cid}", None)
        if fn is None:
            # `constraint_declaration` raises on this, with the message naming the id;
            # a second, earlier copy of the refusal here would just shadow it.
            continue
        needed |= set(inspect.signature(fn).parameters) & set(SWITCH_PARAMETER_NAMES)
    merit = FiguresOfMerit(abs(int(i_figure_merit)))
    metric = ported_objectives.OBJECTIVE_METRICS[merit]
    needed |= set(inspect.signature(metric).parameters) & set(SWITCH_PARAMETER_NAMES)
    areas = AREAS
    values = {}
    for name in sorted(needed):
        hits = [a for a in areas if hasattr(getattr(data, a), name)]
        if len(hits) != 1:
            raise ValueError(
                f"switch {name!r} is "
                + (
                    "in no `DataStructure` area"
                    if not hits
                    else f"ambiguous across `DataStructure` areas {hits}"
                )
                + " -- it cannot be read mechanically, pass `switch_values` by hand"
            )
        values[name] = int(getattr(getattr(data, hits[0]), name))
    return values


def iteration_variable_path(ixc_id: int) -> VarPath:
    """The `VarPath` of PROCESS iteration variable `ixc_id`."""
    iteration_variable = ITERATION_VARIABLES[ixc_id]
    keys = (
        GetAttrKey(iteration_variable.module),
        GetAttrKey(iteration_variable.target_name or iteration_variable.name),
    )
    if iteration_variable.array_index is not None:
        keys = (*keys, SequenceKey(iteration_variable.array_index))
    return VarPath(keys)


def design_bounds(ixc):
    """`((VarPath, lower, upper), ...)` from `ITERATION_VARIABLES`' own defaults, ready
    for `VmconDriver.bounds`.
    """
    return tuple(
        (
            iteration_variable_path(i),
            float(ITERATION_VARIABLES[i].lower_bound),
            float(ITERATION_VARIABLES[i].upper_bound),
        )
        for i in ixc
    )


# ---------------------------------------------------------------- stating the problem


def constraint_declarations(graph, icc, n_equality, switch_values=None, omit=()):
    """`({Constraint<id>: declaration}, equalities, inequalities, omitted)` for the
    active constraints -- each declaration a body and the requirement beside it, the
    first `n_equality` of `icc` held by `Eq` and the rest by `Le`.

    Raises
    ------
    ValueError
        If a constraint is active but cannot be assembled against this graph.
    """
    variables = graph.graph.variables
    declarations, equalities, inequalities, omitted = {}, [], [], {}
    for position, cid in enumerate(icc):
        if cid in omit:
            omitted[cid] = "omitted by the caller"
            continue
        holds = Eq if position < n_equality else Le
        try:
            declarations[constraint_place(cid)] = constraint_declaration(
                variables, cid, holds, switch_values
            )
        except ValueError as e:
            raise ValueError(
                f"{e} Pass `omit={{{cid}}}` to leave it out on purpose and have it "
                f"reported"
            ) from e
        (equalities if position < n_equality else inequalities).append(
            condition_place(cid)
        )
    return declarations, tuple(equalities), tuple(inequalities), omitted


def condition_nodes(graph, icc, n_equality, switch_values=None, omit=()):
    """The constraint **bodies** alone, plus the equality/inequality split -- what an
    arm that answers its conditions outside the graph takes.

    A declaration is a body and a requirement; this is the first of the two, so the
    graph carries the computations and nothing is held of them.
    """
    declarations, equalities, inequalities, omitted = constraint_declarations(
        graph, icc, n_equality, switch_values, omit
    )
    nodes = {
        name: declaration.node_definitions[0]
        for name, declaration in declarations.items()
    }
    return nodes, equalities, inequalities, omitted


def objective_entry(graph, i_figure_merit, switch_values=None):
    """`({Objective: definition}, .numerics.objf)` for this run's figure of merit,
    or `({}, None)` where the run states none.
    """
    if i_figure_merit is None:
        return {}, None
    from functional_process.cottax.input.indat import (
        objective_selection,  # noqa: PLC0415
    )

    name, definition = objective_node(
        graph.graph.variables, objective_selection(i_figure_merit), switch_values
    )
    return {name: definition}, objective_place()


def condition_graph(graph, icc, n_equality, i_figure_merit, switch_values=None, omit=()):
    """`graph` with one node per active constraint and, where `i_figure_merit` says so,
    one for the figure of merit. No requirement and no `Optimise`: what the conditions
    are, without what is asked of them.
    """
    nodes, equalities, inequalities, omitted = condition_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    objective_nodes, objective = objective_entry(graph, i_figure_merit, switch_values)
    nodes.update(objective_nodes)
    return (
        (Plan(graph) + Insert(PathMap(nodes.items()))).graph,
        {
            "equalities": equalities,
            "inequalities": inequalities,
            "objective": objective,
            "omitted": omitted,
        },
    )


def unanswered_requirements(graph, optimiser=OPT):
    """The requirements `optimiser`'s design does not reach, in binding order.

    An architecture absorbs the ones it does reach; one it does not is a constraint on
    something the design cannot move, and `ExecutableGraph` refuses it. Found here so
    an assembly can drop it deliberately and report it.
    """
    reached = set(graph.graph.reach(graph[optimiser].unknowns))
    owners = graph.graph.owners
    return tuple(
        r
        for r in requirements(graph.definitions)
        if not all(owners.get(c) in reached for c in graph[r].conditions)
    )


def problem_graph(
    graph,
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    switch_values=None,
    omit=(),
):
    """`(graph, Opt, report)`: one declaration per active constraint -- its body and
    the requirement beside it -- the objective node, and `Optimise(objf, design)`.

    One `to_graph` over the declarations, not two `Insert`s: a constraint's requirement
    is part of the declaration that computes it, so nothing states it separately and
    the graph's nesting is kept.

    A requirement the design does not reach is **dropped** here and reported under
    `report["external"]`: no architecture can absorb it and the proof refuses it. An
    *equality* among them would change what feasible means, so that is refused instead.

    The rest are left standing: an architecture (`MDF` / `IDF` / `SAND`) absorbs every
    requirement the design reaches as its own first step and reads the cycles off the
    graph that leaves, so nothing here has to absorb them first to put a problem the
    design reaches only through a constraint on the optimiser's cycle.
    `report["required"]` names the ones it will take.

    Raises
    ------
    ValueError
        If an equality constraint reads nothing the design reaches, or the graph
        already held a requirement this assembly did not state.
    """
    declarations, equalities, inequalities, omitted = constraint_declarations(
        graph, icc, n_equality, switch_values, omit
    )
    objective_nodes, objective = objective_entry(graph, i_figure_merit, switch_values)
    report = {
        "equalities": equalities,
        "inequalities": inequalities,
        "objective": objective,
        "omitted": omitted,
    }
    design = tuple(iteration_variable_path(i) for i in ixc)
    stated = to_graph(
        graph,
        {**declarations, **objective_nodes, OPT: Optimise(objective, design)},
    )

    mine = {requirement_place(cid): cid for cid in icc if cid not in omit}
    unanswered = unanswered_requirements(stated)
    if stray := [r for r in unanswered if r not in mine]:
        raise ValueError(
            f"{[r.spelling for r in stray]} hold(s) against something the design does "
            f"not reach, and this assembly did not state them -- absorb them into an "
            f"optimiser of their own, or drop them before stating this problem"
        )
    external = {mine[r]: r for r in unanswered}
    if equalities := [cid for cid in icc[:n_equality] if cid in external]:
        raise ValueError(
            f"equality constraint(s) {equalities} read nothing the design reaches -- "
            f"omitting an equality changes what 'feasible' means, so this assembly is "
            f"refused rather than reduced"
        )
    if external:
        stated = Delete(tuple(external.values())).apply(stated)
        gone = {condition_place(cid) for cid in external}
        report["inequalities"] = tuple(
            c for c in report["inequalities"] if c not in gone
        )
        for cid in external:
            report["omitted"][cid] = (
                "reads nothing the design reaches -- constant over it, so no unknown "
                "of the optimiser can move it"
            )
    report["design"] = design
    report["external"] = external
    report["required"] = requirements(stated.definitions)
    return stated, OPT, report


# ---------------------------------------------------------------- degenerate blocks


@dataclasses.dataclass(frozen=True)
class FixedPointResidual:
    """One fixed point's residual Jacobian `d(g(u) - u)/du` at a point -- or the reason
    it could not be formed.
    """

    problem: NodePath
    jacobian: np.ndarray | None
    """`(n, n)` over the block's unknowns, flattened and concatenated, with the identity
    subtracted; `None` exactly when `undetectable` is set.
    """
    undetectable: str | None
    """The exception that stopped the measurement, `f"{type}: {message}"`, or `None`."""

    @property
    def rank(self) -> int | None:
        """Numerical rank of `jacobian`, or `None` if it could not be formed."""
        if self.jacobian is None:
            return None
        return int(np.linalg.matrix_rank(self.jacobian))

    @property
    def columns(self) -> int | None:
        """How many flattened unknowns the block owns -- the rank a well-posed residual
        block has.
        """
        return None if self.jacobian is None else int(self.jacobian.shape[1])

    @property
    def degenerate(self) -> bool:
        """`True` only when the residual is *identically* zero here -- the strongest
        form of rank deficiency, and the only one `assemble` can act on by dropping
        the problem.
        """
        return self.jacobian is not None and bool(np.allclose(self.jacobian, 0.0))


def fixed_point_residuals(graph, env, problems=None):
    """`d(g(u) - u)/du` for every fixed point in `graph`, differentiated at `env`."""
    if problems is None:
        problems = tuple(n for n in declared(graph) if is_fixed_point(graph[n]))
    residuals = []
    for problem in problems:
        definition = graph[problem]
        # `conditions`, not `.reads`: a driven statement also reads its `Start` places,
        # and those are driver data, not conditions.
        owns, reads = definition.unknowns, definition.conditions
        producers = {r: graph.graph.owners[r] for r in reads if r in graph.graph.owners}
        inside = graph.graph.ancestors(set(producers.values()))
        body = Call(graph.subgraph([n for n in inside if n not in declared(graph)]))

        def residual(flat, _body=body, _owns=owns, _reads=reads, _unravel=None):
            values = dict(env)
            values.update(zip(_owns, _unravel(flat), strict=True))
            out = _body(values)
            return jnp.concatenate([jnp.ravel(jnp.asarray(out[r])) for r in _reads])

        try:
            start, unravel = ravel_pytree([jnp.asarray(env[v]) for v in owns])
            # `np.array`, not `np.asarray`: a JAX array converts to a **read-only** view,
            # and the identity subtraction below is in place.
            jacobian = np.array(
                # Jitted: eagerly this is six `jacfwd`s over acyclic bodies, one XLA
                # compile per `jnp` primitive. `env` is a closure, not an argument, so
                # no `VarPath` is flattened and no antichain question arises.
                eqx.filter_jit(
                    jax.jacfwd(functools.partial(residual, _unravel=unravel))
                )(start),
                dtype=float,
            ).reshape(-1, start.size)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            reason = f"{type(error).__name__}: {error}"
            residuals.append(FixedPointResidual(problem, None, reason))
            continue
        if jacobian.shape[0] != jacobian.shape[1]:
            # A fixed point's conditions are `g(u)`, one per unknown and of the same
            # shape, so this is square by construction -- and if it ever is not, the
            # identity below would be nonsense. Recorded as the block's own reason
            # rather than raised, because one malformed block must not stop the other
            # measurements.
            residuals.append(
                FixedPointResidual(
                    problem,
                    None,
                    f"ValueError: residual is {jacobian.shape}, not square -- "
                    f"`conditions` and `unknowns` do not correspond element for element",
                )
            )
            continue
        jacobian -= np.eye(start.size)
        residuals.append(FixedPointResidual(problem, jacobian, None))
    return tuple(residuals)


def degenerate_fixed_points(graph, env, problems=None):
    """Fixed points whose residual `g(u) - u` is *structurally* zero here."""
    measured = fixed_point_residuals(graph, env, problems)
    undetectable = [r for r in measured if r.undetectable is not None]
    if undetectable:
        detail = "; ".join(
            f"{r.problem.spelling} ({r.undetectable})" for r in undetectable
        )
        raise ValueError(
            f"cannot tell whether {len(undetectable)} of {len(measured)} fixed "
            f"point(s) are degenerate, so cannot report the rest healthy either: "
            f"{detail}"
        )
    return tuple(r.problem for r in measured if r.degenerate)


def array_valued_problems(graph, env, problems=None):
    """Declared problems owning a **non-scalar** unknown at `env`'s own values -- the
    ones today's SAND layer cannot absorb, detected rather than listed.
    """
    if problems is None:
        problems = tuple(n for n in declared(graph) if is_fixed_point(graph[n]))
    return tuple(
        problem
        for problem in problems
        if any(
            unknown in env and jnp.ndim(jnp.asarray(env[unknown])) > 0
            for unknown in graph[problem].unknowns
        )
    )


# ---------------------------------------------------------------- the architecture


def sand_graph(graph, keep=(), scheme=None):
    """`graph` with every problem on the optimiser's cycle folded into the optimiser,
    which keeps its name: `cottax.mdao_architectures.SAND`, handed `scheme`'s rule
    (`mda.SCHEME`) to open what is still cyclic first. A problem the design does not
    reach stays a solve of its own.

    `keep` -- problems left nested rather than folded -- has no counterpart since the
    architecture's per-name levels went (cottax 2026-09-24): a final of `Monolithic`
    with a `keep` field is the way to say it, and nothing in this port asks for one.
    """
    if keep:
        raise NotImplementedError(
            f"keep={tuple(p.spelling for p in keep)}: SAND with problems left nested is "
            f"a `Monolithic` final of its own now, and none is written"
        )
    from functional_process.cottax.architectures.mda import SCHEME

    scheme = SCHEME if scheme is None else scheme
    return (Plan(graph) + SAND(scheme.rule, scheme.closing, OPT)).graph


def residual_condition_scales(drive, env, floor=1e-12):
    """`((condition, factor), ...)` for exactly the coupling conditions of a folded
    problem, ready for `VmconDriver.condition_scale`.
    """

    def place(path):
        while (stripped := unminted(path)) != path:
            path = stripped
        return path

    statement = drive.statement
    # Keyed the way a driver reads a condition_scale: one place per stacked entry, the
    # objectives and then the side each relation names -- its left, or its right where
    # the left is zero (`drivers.condition_places`).
    stacked = statement.objectives + tuple(
        r.lhs if r.lhs is not None else r.rhs for r in statement.relations
    )
    unknowns = {place(v): v for v in drive.unknowns}
    scales = []
    for condition in stacked:
        if condition.spelling.startswith((".constraints.c", ".numerics.objf")):
            continue
        unknown = unknowns.get(place(condition))
        if unknown is None or unknown not in env:
            continue
        # The largest element for an array-valued unknown (a scheme cut copies whole
        # profiles); the scale is one factor per condition, so one number per unknown.
        magnitude = float(np.max(np.abs(np.asarray(env[unknown], dtype=float))))
        usable = np.isfinite(magnitude) and magnitude > floor
        scales.append((condition, 1.0 / magnitude if usable else 1.0))
    return tuple(scales)


def sand_schedule(
    graph,
    problem_name=None,
    driver=None,
    bounds=(),
    callback=None,
    condition_scale=(),
    max_iter=None,
    nest=False,
    inner_drivers=None,
    optimiser=None,
):
    """A `Schedule` for `graph`'s single optimise statement, answered by `driver`."""
    # `is_problem` first: a shape predicate is asked of a condition node only.
    optimise = next(
        p for p, d in graph.definitions.items() if is_problem(d) and is_optimise(d)
    )
    drivers = default_drivers(
        graph,
        bounds=bounds,
        callback=callback,
        condition_scale=condition_scale,
        max_iter=max_iter,
        **({} if optimiser is None else {"optimiser": optimiser}),
    )
    if driver is not None:
        drivers[optimise] = driver
    drivers.update(inner_drivers or {})
    # Drivers go into the graph (`Assign`), and the schedule reads them from there.
    assigned = assign_drivers(graph, drivers)
    # Nesting is an op on the *graph*: which statement's iteration answers which is
    # recorded in `Graph.within`, and `ExecutableGraph` reads it.
    if nest:
        from functional_process.cottax.queries import nested_inside  # noqa: PLC0415

        assigned = nested_inside(assigned, optimise)
    return Schedule(RunnableGraph(assigned))


def sand_shape(schedule: Schedule) -> dict:
    """The one `Drive`'s size, for reporting: how much of the graph is actually inside
    the solved block and how much still runs as ordinary `Call` steps.
    """
    # The `Drive` whose problem is the optimise statement, not the first one: a problem
    # outside the optimiser's cycle (IDF leaves the disciplines' own solves where they
    # are) is its own top-level `Drive`, and may be scheduled before it.
    drive = next(
        step
        for step in schedule.steps
        if isinstance(step, Drive)
        if is_optimise(step.statement)
    )
    n_equality, n_inequality = relation_counts(drive.statement)
    return {
        "drive_nodes": len(drive.nodes),
        "unknowns": len(drive.unknowns),
        "conditions": len(drive.conditions),
        "context": len(drive.context),
        "design": len(drive.statement.unknowns),
        "equalities": n_equality,
        "inequalities": n_inequality,
        "schedule_steps": len(schedule.steps),
        "drive": drive,
    }


def assemble(
    reference,
    driven,
    env,
    omit=(),
    switch_values=None,
    keep=(),
    drop_arrays=True,
    graph=None,
    scheme=None,
):
    """The SAND graph for `reference`'s own `ixc`/`icc`/`i_figure_merit`.

    Built from `graph` (default: `graph_for()`), since SAND cuts the cycles itself and
    refuses a closure cut beforehand: `SAND(...).resolution` of the problem graph.
    `driven` and `env` are `evaluate.mda_env`'s over that graph cut by `scheme`
    (default: `mda.SCHEME`), and the degenerate and array-valued problems are measured
    on them. Not on the resolution's `opened`: there the requirements are already in
    the optimiser, so a closure's cycle runs through `.Opt` and pulls the constraint
    nodes into the fixed point's body, whose reads `env` does not hold. `driven`'s
    closures are the resolution's (same scheme, same graph), which is checked. A
    dropped problem is deleted from `opened` and the rest placed by SAND's own
    `combining`, so a dropped closure's copies stand frozen at the seed, as deleting
    it from `driven` did.

    `drop_arrays`: delete every fixed point over a non-scalar unknown, leaving its copy
    frozen at the seed -- what every reference SAND row was measured with. `False`
    folds them into the optimiser like any other (both SQP drivers ravel their
    unknowns), which a scheme's cuts need: a Jacobi cut copies whole profiles.

    Raises
    ------
    ValueError
        If an equality constraint reads nothing the design reaches.
    """
    from functional_process.cottax.architectures.evaluate import without_excluded
    from functional_process.cottax.architectures.mda import SCHEME
    from functional_process.cottax.input.indat import graph_for

    scheme = SCHEME if scheme is None else scheme
    if keep:
        raise NotImplementedError(
            f"keep={tuple(p.spelling for p in keep)}: SAND with problems left nested is "
            f"a `Monolithic` final of its own now, and none is written"
        )
    with_problem, _name, report = problem_graph(
        without_excluded(graph if graph is not None else graph_for()),
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        switch_values=switch_values,
        omit=omit,
    )
    architecture = SAND(scheme.rule, scheme.closing, OPT)
    resolution = architecture.resolution(with_problem)
    opened = resolution.opened
    cut = {p for p in declared(driven) if p not in set(declared(with_problem))}
    if cut != {c.at for c in resolution.closures}:
        raise ValueError(
            f"`driven` was cut differently from SAND's own cutting: "
            f"{sorted(p.spelling for p in cut)} against "
            f"{sorted(c.at.spelling for c in resolution.closures)} -- pass the "
            f"`graph` and `scheme` `driven` was built from"
        )
    degenerate = degenerate_fixed_points(driven, env)
    array_valued = (
        array_valued_problems(
            driven,
            env,
            tuple(p for p in declared(driven) if p not in set(degenerate)),
        )
        if drop_arrays
        else ()
    )
    dropped = tuple(degenerate) + tuple(array_valued)
    if dropped:
        opened = Delete(dropped).apply(opened)
        closures = tuple(c.at for c in resolution.closures if c.at not in set(dropped))
        combined = opened
        for op in architecture.combining(opened, closures):
            combined = op.apply(combined)
    else:
        combined = resolution.graph
    report["degenerate"] = degenerate
    report["array_valued"] = array_valued
    return combined, report


__all__ = [
    "OPT",
    "FixedPointResidual",
    "array_valued_problems",
    "assemble",
    "condition_graph",
    "condition_nodes",
    "constraint_declarations",
    "degenerate_fixed_points",
    "design_bounds",
    "fixed_point_residuals",
    "iteration_variable_path",
    "objective_entry",
    "problem_graph",
    "requirement_place",
    "residual_condition_scales",
    "sand_graph",
    "sand_schedule",
    "sand_shape",
    "switch_values_for",
    "unanswered_requirements",
]
