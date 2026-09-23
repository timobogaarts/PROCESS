"""Consistency solved where it arises: an equality constraint closed by a root find
over one variable, **inside** the MDA, and the optimiser left with the objective, the
inequalities and the design variables that remain.

MDF hands every condition to the optimiser, which owns every design variable, so the
`Optimise` closes one cycle over most of the machine. An equality need not be closed
that way. A requirement `cond = 0` `Determine`d by one variable closes a cycle of its
own, and the size of that cycle depends on the variable chosen:
the power balance (`^cond.constraints.c2`) closed by `hfact` is three nodes, since
`hfact` enters the confinement time and nothing else; closed by the density it is 27,
and by `rmajor` 24. Which variable closes which equality is the caller's choice
(`configurations.kinds.PAIRINGS`); this module is the recipe that applies it:

    raw -> cut -> constraint and objective nodes (`mdf.mdf_graph`)
        -> `Insert` a `Requirement(cond, Eq)` at `.Close.<cond>` and `Determine` it by
           the variable that closes it -- a root find over one unknown
        -> every declared statement on its cycle `Absorb`ed into it, which keeps its
           name (`flattened`; `flatten=False` nests them inside it, Picard-driven)
        -> `queries.nested_inside(each root find)`
        -> drivers: `mda.default_drivers`, and `SafeguardedNewtonDriver` on each root
           find (capped, backtracked, Broyden-updated) -- or, nested and one unknown,
           `bracketed()`: `BracketedRootDriver`, a bracketed root that converges from
           any start, the Picards re-converged inside every residual evaluation

The result is an `mdf.Mdf` whose schedule carries the root finds, so `mdf.seed`,
`mdf.prime`, `mdf.solve` and `evaluate.run_schedule` work on it unchanged: the record
is built with `n_equality = 0`, the design is `ixc` minus the closing variables, and
the outer optimiser sees the objective and the inequalities only.
"""

import dataclasses

import numpy as np
from cottax.execution.drivers.kinds import Converged, Steps
from cottax.interfaces import (
    Absorb,
    Determine,
    Eq,
    ExecutableGraph,
    Graph,
    Insert,
    Plan,
    RunnableGraph,
    Schedule,
    is_bare_condition,
    is_problem,
    shape_of,
)
from cottax.interfaces.statements import Optimise, Requirement
from cottax.pytree.path import NodePath, PathMap, VarPath
from cottax.visualization.sequencing import Solve, entries, problem_types, walk
from jax.tree_util import GetAttrKey

from functional_process.configurations import kinds
from functional_process.cottax.architectures import mdf, sand
from functional_process.cottax.architectures.drivers import (
    BracketedRootDriver,
    SafeguardedNewtonDriver,
    condition_scale,
)
from functional_process.cottax.architectures.evaluate import (
    mda_schedule,
    without_excluded,
)
from functional_process.cottax.architectures.mda import (
    SCHEME,
    assign_drivers,
    cut_graph,
    default_drivers,
    guess_sources,
)
from functional_process.cottax.architectures.session import Session, open_session
from functional_process.cottax.input.indat import graph_for
from functional_process.cottax.queries import nested_inside

NEWTON_TOL = 1e-10
"""`rtol = atol` of the Newton on a closing root find. The port's default
(`SeededNewtonDriver`, 1e-4) would leave the equalities four orders looser than the
MDF's own `max|eq|`."""

PLACE = NodePath((GetAttrKey("Close"),))
"""Where a requirement closing an equality binds: `.Close.c2`, `.Close.c16`."""


def place_for(cond: VarPath) -> NodePath:
    """`.Close.<the condition's last component>`."""
    return NodePath((*PLACE.segments, GetAttrKey(cond.spelling.split(".")[-1])))


# ---------------------------------------------------------------- the drivers


def newton(**kwargs) -> mdf.MdfNewtonDriver:
    """The undamped Newton (`mdf.MdfNewtonDriver`), reporting its verdict, at
    `NEWTON_TOL`.
    """
    return mdf.MdfNewtonDriver(**{"rtol": NEWTON_TOL, "atol": NEWTON_TOL, **kwargs})


def safeguarded(**kwargs) -> SafeguardedNewtonDriver:
    """The default driver on a closing root find: capped, backtracked, Broyden-updated,
    at `NEWTON_TOL`.
    """
    return SafeguardedNewtonDriver(**{"rtol": NEWTON_TOL, "atol": NEWTON_TOL, **kwargs})


def safeguarded_newton(**kwargs) -> SafeguardedNewtonDriver:
    """The exact-Jacobian variant of `safeguarded`."""
    return safeguarded(jacobian="newton", **kwargs)


def bracketed(**kwargs) -> BracketedRootDriver:
    """The globally convergent driver on a closing root find of **one** unknown:
    bracket the root, then Newton inside the bracket with bisection as the fallback
    (`BracketedRootDriver`), at `NEWTON_TOL`. For `close(flatten=False)`, where the
    root find keeps its one unknown and the cycle's fixed points are nested inside it
    and re-converged per residual evaluation (the Picards inside the root) -- a
    flattened problem has the cut copies as unknowns too, and `accepts` refuses it.
    `close` fills `bounds` for the closing variable from the session's reference when
    none are given, as the first bracket tried.
    """
    return BracketedRootDriver(**{"rtol": NEWTON_TOL, "atol": NEWTON_TOL, **kwargs})


def with_bounds(driver, var: VarPath, bounds) -> object:
    """`driver` with `var`'s `(lower, upper)` out of `bounds` (`((VarPath, lo, hi),
    ...)`, `Session.reference.bounds`) added, where it is a `BracketedRootDriver`
    without one; any other driver, or one already bounding `var`, unchanged.
    """
    if not isinstance(driver, BracketedRootDriver) or driver.bracket_for(var):
        return driver
    for bound_var, lo, hi in bounds:
        if bound_var == var:
            return dataclasses.replace(
                driver, bounds=(*driver.bounds, (var, float(lo), float(hi)))
            )
    return driver


# ---------------------------------------------------------------- the ops


def closed_at(graph: Graph, cond: VarPath, var: VarPath) -> Graph:
    """`graph` with `cond` asserted at `place_for(cond)` and determined by `var`: a
    requirement, and the `Determine` that makes it a root find over one unknown.

    `Eq` even where `cond` is one of the file's inequalities -- naming one here is an
    operating choice, *run exactly at the limit*, and the normalised residual every
    constraint node owns is the same scalar either way.
    """
    place = place_for(cond)
    return (
        Plan(graph)
        + Insert(PathMap(((place, Requirement(cond, Eq)),)))
        + Determine(place, (var,))
    ).graph


def cycle_of(graph: Graph, cond: VarPath, var: VarPath) -> tuple[NodePath, ...]:
    """The component closing `cond` by `var` creates: the root find and every node its
    iteration would re-run.
    """
    place = place_for(cond)
    return next(c for c in closed_at(graph, cond, var).graph.components if place in c)


def flattened(graph: Graph, place: NodePath) -> tuple[Graph, NodePath]:
    """Every declared statement on `place`'s cycle folded into it -- `Absorb`, so the
    root find keeps its name and binding -- giving one square problem over `(var, the
    cut copies)`. Returns the graph and that problem's path, which is `place`.

    The alternative to `nested_inside`: nested, every evaluation of the root find's
    residual re-converges each Picard inside it, and the Newton's derivative goes
    through their implicit adjoints; combined, one Newton over a few unknowns
    evaluates the body once per step with no loop inside the loop. Which is cheaper
    depends on the driver -- see `SafeguardedNewtonDriver`.
    """
    component = next(c for c in graph.graph.components if place in c)
    inner = tuple(
        n
        for n in component
        if n != place and (is_problem(graph[n]) or is_bare_condition(graph[n]))
    )
    if not inner:
        return graph, place
    return (Plan(graph) + Absorb(place, inner)).graph, place


# ---------------------------------------------------------------- the closed graph


@dataclasses.dataclass(frozen=True)
class Closed:
    """The architecture: an `Mdf` whose schedule carries the closing root finds."""

    session: Session
    """The configuration this closes, in its solve environment."""
    problem: mdf.Mdf
    pairings: dict
    """`{condition: variable}` -- what closes what."""
    places: dict
    """`{condition: NodePath}` -- the problem each root find is answered by. Two
    conditions flattened onto one cycle share a place.
    """
    design: tuple[VarPath, ...]
    """`ixc` minus the closing variables: what the outer optimiser keeps."""
    flat: bool = True
    """Whether the declared problems on each cycle were folded into its root find."""

    @property
    def graph(self) -> Graph:
        """The driven graph the schedule runs -- `problem.graph` is the same structure
        without its drivers, as `mdf.assemble` keeps it, and so without the `^guess.*`
        ports `Assign` mints.
        """
        return self.problem.eager.executable.graph

    @property
    def report(self) -> dict:
        """`mdf_graph`'s report plus `closing`, `closing_problems` and `flattened`."""
        return self.problem.report

    @property
    def closing(self) -> tuple[VarPath, ...]:
        """The closing variables, in `pairings` order."""
        return tuple(self.pairings.values())

    def unknowns(self, cond: VarPath) -> tuple[VarPath, ...]:
        """The unknowns of the problem answering `cond`, the closing variable first."""
        return tuple(self.graph[self.places[cond]].unknowns)

    def start_port(self, var: VarPath) -> VarPath:
        """The `^guess.*` port a closing problem starts `var` from.

        Raises
        ------
        KeyError
            If no closing problem owns `var`.
        """
        for port, unknown in guess_sources(self.graph).items():
            if unknown == var:
                return port
        raise KeyError(f"{var.spelling} is not an unknown of a closing problem")

    def root_find_reports(self, out) -> dict:
        """`{condition: (steps, converged, residual, unknown)}` out of a run's env."""
        return {
            cond: (
                int(np.asarray(mdf.verdict(out, Steps, self.graph[place]))),
                bool(np.asarray(mdf.verdict(out, Converged, self.graph[place]))),
                float(np.asarray(out[cond])),
                float(np.asarray(out[self.pairings[cond]])),
            )
            for cond, place in self.places.items()
        }


def _session(what) -> Session:
    return what if isinstance(what, Session) else open_session(what)


def _resolve(spelling: str, candidates, what: str) -> VarPath:
    for v in candidates:
        if v.spelling == spelling:
            return v
    raise KeyError(
        f"{spelling} is not {what} of this configuration -- the choices are "
        f"{sorted(v.spelling for v in candidates)}"
    )


def close(
    session,
    pairings: dict[str, str] | None = None,
    *,
    flatten: bool = True,
    driver=None,
    rewrite=None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> Closed:
    """Close each condition in `pairings` by the variable it names, inside the MDA.

    `session`: a `session.Session`, or what `open_session` takes (a configuration or
    its name). `pairings`: `{condition spelling: variable spelling}`, the condition
    one of the file's **active conditions** -- an equality *or* an inequality -- and
    the variable **any boundary input of the problem graph**, not only one of the
    file's `ixc` (the power balance `c2` closed by the heating power
    `p_hcd_primary_extra_heat_mw`, an input PROCESS holds at 75 MW); default
    `kinds.PAIRINGS["one"]`.

    Naming an inequality drives its residual to exactly zero with the same root-find
    machinery an equality gets -- legitimate because `sand.condition_nodes` gives
    every active constraint, equality or inequality, the same one scalar
    (`^cond.constraints.c<n>`, the normalised residual, index 1 of `(residual,
    normalised_residual, value, bound)`), so the zero a closing root find drives it to
    is the same point where the original `<=` relation would sit tight -- an operating
    choice (run exactly at the limit) stated as a structural one. Once named this way
    the condition leaves `report["inequalities"]` (it is no longer a free condition
    anything else can read as a constraint, the same way a closed equality is not
    carried as one) and is listed in `report["closed_inequalities"]`.

    `flatten`: the declared problems on a root find's cycle are combined into it
    (`flattened`); otherwise nested inside it and driven by Picard. Root finds sharing
    a cycle are flattened together into one square problem over their closing
    variables, and refused when `flatten` is off; root finds on cycles of their own
    keep their own problems. `driver`: the driver assigned to every closing problem,
    default `safeguarded()`; a `bracketed()` driver is given the closing variable's
    bounds from the reference (`with_bounds`) and, answering one unknown only, wants
    `flatten=False`. `rewrite`: a `Graph -> Graph` applied to the problem graph -- the
    cut MDA with the condition and objective nodes -- before any root find is inserted,
    so a study's own ops (a `Rewire` of what the cost node reads, a `Redefine` of a
    limit) are part of the closed graph and of the report (`report["rewrite"]`, its
    name). `bounds`: `{variable spelling: (lower, upper)}` for a closing variable the
    reference has no bounds for, what `with_bounds` gives a `bracketed()` driver and
    what the report carries as `report["closing_bounds"]`.

    Raises
    ------
    KeyError
        If a condition is not active in this file, or a variable is not a boundary
        input of the problem graph.
    ValueError
        If one variable is named for two conditions, or two root finds share a cycle
        and `flatten` is off.
    """
    if pairings is None:
        pairings = kinds.PAIRINGS["one"]
    live = _session(session)
    ref = live.reference
    raw = without_excluded(
        live.machine_graph if live.machine_graph is not None else graph_for()
    )
    scheme = SCHEME if live.scheme is None else live.scheme
    graph, _conditions, _n, report = mdf.mdf_graph(
        cut_graph(raw, scheme), ref.icc, ref.n_equality, ref.i_figure_merit, live.switch_values
    )
    if rewrite is not None:
        graph = rewrite(graph)
    design = tuple(sand.iteration_variable_path(i) for i in ref.ixc)
    equalities = tuple(report["equalities"])
    inequalities = tuple(report["inequalities"])
    inputs = tuple(graph.graph.boundary_inputs)
    chosen = {
        _resolve(c, equalities + inequalities, "an active condition"): _resolve(
            v, inputs, "a boundary input of the problem graph"
        )
        for c, v in pairings.items()
    }
    if len(set(chosen.values())) != len(chosen):
        raise ValueError("one variable cannot close two conditions")
    closed_inequalities = tuple(c for c in chosen if c in inequalities)
    remaining = tuple(c for c in inequalities if c not in chosen)
    places: dict = {}
    for cond, var in chosen.items():
        places[cond] = place_for(cond)
        graph = closed_at(graph, cond, var)
    components = {
        cond: next(c for c in graph.graph.components if p in c)
        for cond, p in places.items()
    }
    # The conditions grouped by cycle: two root finds on one cycle are one group,
    # flattened into one square problem; a root find on a cycle of its own is a group
    # of one and keeps its own problem (`close` maps every place onto the one combined
    # problem when any two share a cycle, which is right for two pairings and wrong
    # for three).
    groups: list[list] = []
    for cond, component in components.items():
        for group in groups:
            if set(component) & set(components[group[0]]):
                if not flatten:
                    raise ValueError(
                        f"the root find for {cond.spelling} shares a cycle with another -- "
                        f"pick pairings whose cycles are disjoint, or `flatten` them into "
                        f"one square problem"
                    )
                group.append(cond)
                break
        else:
            groups.append([cond])
    if flatten:
        for group in groups:
            first_cond = group[0]
            graph, combined = flattened(graph, places[first_cond])
            for cond in group:
                places[cond] = combined
    for place in dict.fromkeys(places.values()):
        graph = nested_inside(graph, place)
    given = {
        v: (float(lo), float(hi)) for v, lo, hi in ref.bounds
    }
    for spelling, (lo, hi) in (bounds or {}).items():
        given[_resolve(spelling, inputs, "a boundary input of the problem graph")] = (
            float(lo),
            float(hi),
        )
    all_bounds = tuple((v, lo, hi) for v, (lo, hi) in given.items())
    drivers = default_drivers(graph)
    for cond, place in places.items():
        drivers[place] = (
            safeguarded()
            if driver is None
            else with_bounds(driver, chosen[cond], all_bounds)
        )
    assigned = assign_drivers(graph, drivers)
    schedule = Schedule(RunnableGraph(assigned))
    kept = tuple(v for v in design if v not in set(chosen.values()))
    report = dict(
        report,
        inequalities=remaining,
        closed_inequalities=tuple(c.spelling for c in closed_inequalities),
        closing={c.spelling: v.spelling for c, v in chosen.items()},
        closing_problems={
            p.spelling: tuple(u.spelling for u in assigned[p].unknowns)
            for p in dict.fromkeys(places.values())
        },
        closing_bounds={
            v.spelling: given[v] for v in chosen.values() if v in given
        },
        rewrite=None if rewrite is None else getattr(rewrite, "__name__", repr(rewrite)),
        flattened=flatten,
        blocks=len(assigned.graph.components),
        driven_blocks=sum(1 for t in problem_types(assigned) if t is not None),
    )
    problem = mdf.Mdf(
        graph=graph,
        eager=schedule,
        traceable=schedule,
        design=kept,
        conditions=(report["objective"], *remaining),
        n_equality=0,
        n_inequality=len(remaining),
        report=report,
        raw=raw,
    )
    return Closed(
        session=live,
        problem=problem,
        pairings=chosen,
        places=places,
        design=kept,
        flat=flatten,
    )


# ---------------------------------------------------------------- seeding


def seed(built: Closed, data, design_values=None, closing_values=None) -> dict:
    """`mdf.seed` for the closed problem, then every other unknown of a closing
    problem -- the cut copies `flattened` folded into it -- started from the plain MDA
    at that design (`copies_from_mda`).

    `design_values`: one value per `built.design`, as `mdf.seed` takes them.
    `closing_values`: `{closing variable: value}`, written to its `^guess.*` port.

    `mdf.seed` alone grounds a cut copy's start from `data`, where PROCESS's cold
    value of a fusion rate is `0.0`: a Picard evaluates the map from there and
    converges, but a Newton over the absorbed copy has its scale set by that
    start and cannot. IDF and SAND seed their copies the same way
    (`session.solve_block`: design variables from the file, coupling copies from an
    MDA at that design).
    """
    env = mdf.seed(built.problem, data, design_values)
    for named, value in (closing_values or {}).items():
        var = (
            _resolve(named, built.closing, "a closing variable")
            if isinstance(named, str)
            else named
        )
        env[built.start_port(var)] = mdf._not_weak(value)
    env.update(copies_from_mda(built, env))
    return env


def copies_from_mda(built: Closed, env) -> dict:
    """`{start port: value}` for every unknown of a closing problem that is not a
    closing variable, from the plain MDA (`evaluate.mda_schedule`) run at `env`'s
    design -- the closing variables at the values their own start ports hold.

    Raises
    ------
    KeyError
        If `env` lacks one of the MDA's inputs.
    """
    scheme = SCHEME if built.session.scheme is None else built.session.scheme
    _driven, _runnable, schedule, run = mda_schedule(built.session.machine_graph, scheme)
    closing = set(built.closing)
    at_port = {var: built.start_port(var) for var in closing}
    inputs = {}
    for var in schedule.inputs:
        if var in env:
            inputs[var] = env[var]
        elif var in at_port and at_port[var] in env:
            inputs[var] = env[at_port[var]]
        else:
            raise KeyError(
                f"no value for the MDA's input {var.spelling} -- `seed` fills this "
                f"env from `mdf.seed`, and a closing variable from its `^guess.*` port"
            )
    out = run(PathMap(inputs))
    return {
        port: out[unknown]
        for port, unknown in guess_sources(built.graph).items()
        if unknown not in closing and unknown in out
    }


# ---------------------------------------------------------------- as structure

OPTIMISE = NodePath((GetAttrKey("Opt"),))
"""Where `nested_blocking` binds the outer `Optimise`."""


def nested_blocking(built: Closed, driver=None) -> ExecutableGraph:
    """The architecture as structure: the `Optimise` over the kept design inserted and
    `nested_inside` it, so the root finds and the MDA's fixed points are answered
    inside its iteration. For the picture -- `mdf.solve` drives the optimiser from
    outside the graph, as `mdf.assemble` does.
    """
    graph = built.problem.graph  # undriven: `Assign` refuses a problem already driven
    report = built.report
    node = Optimise(
        report["objective"],
        built.design,
        inequalities=tuple(report["inequalities"]),
    )
    # A lift leaves its inequality as a `Relation` rather than a node (`lift.lift`),
    # so the ones this problem carries are stated here, where the optimiser is.
    if lifted := tuple(report.get("lift_relations", ())):
        node = dataclasses.replace(node, relations=node.relations + lifted)
    with_problem = (Plan(graph) + Insert(PathMap(((OPTIMISE, node),)))).graph
    with_problem = nested_inside(with_problem, OPTIMISE)
    drivers = default_drivers(with_problem)
    for cond, place in built.places.items():
        drivers[place] = (
            safeguarded()
            if driver is None
            else with_bounds(
                driver, built.pairings[cond], built.session.reference.bounds
            )
        )
    return ExecutableGraph(assign_drivers(with_problem, drivers))


def describe(graph) -> list[str]:
    """One line per driven or cyclic entry at every depth: its size and its problem."""
    if isinstance(graph, ExecutableGraph):
        graph = graph.graph
    lines = []
    for within, entry in walk(entries(graph)):
        if not isinstance(entry, Solve) and len(entry.nodes) == 1:
            continue
        label = (
            f"{shape_of(graph[entry.problem])} {entry.problem.spelling}"
            if isinstance(entry, Solve)
            else type(entry).__name__.lower()
        )
        lines.append("  " * len(within) + f"{len(entry.nodes)} nodes, {label}")
    return lines


__all__ = [
    "NEWTON_TOL",
    "OPTIMISE",
    "PLACE",
    "BracketedRootDriver",
    "Closed",
    "SafeguardedNewtonDriver",
    "bracketed",
    "close",
    "closed_at",
    "condition_scale",
    "copies_from_mda",
    "cycle_of",
    "describe",
    "flattened",
    "nested_blocking",
    "newton",
    "place_for",
    "safeguarded",
    "safeguarded_newton",
    "seed",
    "with_bounds",
]
