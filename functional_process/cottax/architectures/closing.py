"""Consistency solved where it arises: an equality constraint closed by a root find
over one variable, **inside** the MDA, and the optimiser left with the objective, the
inequalities and the design variables that remain.

MDF hands every condition to the optimiser, which owns every design variable, so the
`Optimise` closes one cycle over most of the machine. An equality need not be closed
that way. `RootFind(conditions=(cond,), unknowns=(var,))` inserted into the graph
closes a cycle of its own, and the size of that cycle depends on the variable chosen:
the power balance (`^cond.constraints.c2`) closed by `hfact` is three nodes, since
`hfact` enters the confinement time and nothing else; closed by the density it is 27,
and by `rmajor` 24. Which variable closes which equality is the caller's choice
(`configurations.kinds.PAIRINGS`); this module is the recipe that applies it:

    raw -> cut -> constraint and objective nodes (`mdf.mdf_graph`)
        -> `Insert` a `RootFind` per pairing, at `.Close.<cond>`
        -> every declared problem on its cycle `Residualise`d and `Combine`d into it
           (`flattened`; `flatten=False` keeps them nested inside it, Picard-driven)
        -> `queries.nested_inside(each root find)`
        -> drivers: `mda.default_drivers`, and `SafeguardedNewtonDriver` on each root
           find (capped, backtracked, Broyden-updated)

The result is an `mdf.Mdf` whose schedule carries the root finds, so `mdf.seed`,
`mdf.prime`, `mdf.solve` and `evaluate.run_schedule` work on it unchanged: the record
is built with `n_equality = 0`, the design is `ixc` minus the closing variables, and
the outer optimiser sees the objective and the inequalities only.
"""

import dataclasses

import numpy as np
from cottax.answerable import AnswerableGraph
from cottax.evaluation.schedule import Schedule
from cottax.graph import Graph
from cottax.names import PathMap
from cottax.plan import Insert, Plan
from cottax.problem import (
    ConditionalNode,
    Converged,
    Optimise,
    RootFind,
    Steps,
    shape_of,
    unknowns_of,
)
from cottax.rewrites import Combine, Residualise
from cottax.spec import NodePath, VarPath
from cottax.visualization.sequencing import Solve, entries, problem_types, walk
from jax.tree_util import GetAttrKey

from functional_process.configurations import kinds
from functional_process.cottax.architectures import mdf, sand
from functional_process.cottax.architectures.drivers import (
    SafeguardedNewtonDriver,
    condition_scale,
)
from functional_process.cottax.architectures.evaluate import (
    mda_schedule,
    without_excluded,
)
from functional_process.cottax.architectures.mda import (
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
"""Where a `RootFind` closing an equality binds: `.Close.c2`, `.Close.c16`."""


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


# ---------------------------------------------------------------- the ops


def cycle_of(graph: Graph, cond: VarPath, var: VarPath) -> tuple[NodePath, ...]:
    """The component `RootFind((cond,), (var,))` creates when inserted at
    `place_for(cond)`: the root find and every node its iteration would re-run.
    """
    place = place_for(cond)
    with_problem = (
        Plan(graph) + Insert(PathMap(((place, RootFind((cond,), (var,))),)))
    ).graph
    return next(c for c in with_problem.graph.components if place in c)


def flattened(graph: Graph, place: NodePath) -> tuple[Graph, NodePath]:
    """Every declared problem on `place`'s cycle folded into it: each `Residualise`d
    (its `u = g(u)` becomes `g(u) - u = 0`) and all of them `Combine`d with the root
    find into one square problem over `(var, the cut copies)`. Returns the graph and
    the combined problem's path (`^problem.Close.c2`).

    The alternative to `nested_inside`: nested, every evaluation of the root find's
    residual re-converges each Picard inside it, and the Newton's derivative goes
    through their implicit adjoints; combined, one Newton over a few unknowns
    evaluates the body once per step with no loop inside the loop. Which is cheaper
    depends on the driver -- see `SafeguardedNewtonDriver`.
    """
    component = next(c for c in graph.graph.components if place in c)
    inner = [
        n for n in component if n != place and isinstance(graph[n], ConditionalNode)
    ]
    if not inner:
        return graph, place
    plan = Plan(graph)
    for n in inner:
        if any(not r.against_zero for r in graph[n].relations):
            plan += Residualise(n)  # a root find is already against zero
    combine = Combine(place, (place, *inner))
    return (plan + combine).graph, combine.problem


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
        return self.problem.eager.answerable.graph

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
        return tuple(unknowns_of(self.graph[self.places[cond]]))

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
                int(np.asarray(out[Steps.name_for(place)])),
                bool(np.asarray(out[Converged.name_for(place)])),
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
) -> Closed:
    """Close each equality in `pairings` by the variable it names, inside the MDA.

    `session`: a `session.Session`, or what `open_session` takes (a configuration or
    its name). `pairings`: `{condition spelling: variable spelling}`, the condition an
    active equality of the file and the variable one of its design variables; default
    `kinds.PAIRINGS["one"]`. `flatten`: the declared problems on a root find's cycle
    are combined into it (`flattened`); otherwise nested inside it and driven by
    Picard. Two root finds on one cycle are flattened together into one square
    problem over both closing variables, and refused when `flatten` is off. `driver`:
    the driver assigned to every closing problem, default `safeguarded()`.

    Raises
    ------
    ValueError
        If one variable is named for two equalities, or two root finds share a cycle
        and `flatten` is off.
    """
    live = _session(session)
    ref = live.reference
    raw = without_excluded(
        live.machine_graph if live.machine_graph is not None else graph_for()
    )
    cut = cut_graph if live.cut is None else live.cut
    graph, _conditions, _n, report = mdf.mdf_graph(
        cut(raw), ref.icc, ref.n_equality, ref.i_figure_merit, live.switch_values
    )
    design = tuple(sand.iteration_variable_path(i) for i in ref.ixc)
    equalities = tuple(report["equalities"])
    chosen = {
        _resolve(c, equalities, "an equality"): _resolve(v, design, "a design variable")
        for c, v in (pairings if pairings is not None else kinds.PAIRINGS["one"]).items()
    }
    if len(set(chosen.values())) != len(chosen):
        raise ValueError("one variable cannot close two equalities")
    places: dict = {}
    for cond, var in chosen.items():
        place = place_for(cond)
        places[cond] = place
        graph = (
            Plan(graph) + Insert(PathMap(((place, RootFind((cond,), (var,))),)))
        ).graph
    components = {
        cond: next(c for c in graph.graph.components if p in c)
        for cond, p in places.items()
    }
    seen: set = set()
    shared = False
    for cond, component in components.items():
        if seen & set(component):
            if not flatten:
                raise ValueError(
                    f"the root find for {cond.spelling} shares a cycle with another -- "
                    f"pick pairings whose cycles are disjoint, or `flatten` them into "
                    f"one square problem"
                )
            shared = True
        seen |= set(component)
    # Whatever declared problem sits on a root find's cycle (a cut fixed point, a
    # model's own solve) is either folded into it or answered inside its iteration.
    if flatten:
        for cond, place in list(places.items()):
            if shared and place not in graph.nodes:
                continue  # already folded into the first root find's problem
            graph, places[cond] = flattened(graph, place)
        if shared:
            combined = next(p for p in places.values() if p in graph.nodes)
            places = dict.fromkeys(places, combined)
    for place in set(places.values()):
        graph = nested_inside(graph, place)
    drivers = default_drivers(graph)
    for place in set(places.values()):
        drivers[place] = safeguarded() if driver is None else driver
    assigned = assign_drivers(graph, drivers)
    schedule = Schedule(AnswerableGraph(assigned))
    kept = tuple(v for v in design if v not in set(chosen.values()))
    report = dict(
        report,
        closing={c.spelling: v.spelling for c, v in chosen.items()},
        closing_problems={
            p.spelling: tuple(u.spelling for u in unknowns_of(assigned[p]))
            for p in dict.fromkeys(places.values())
        },
        flattened=flatten,
        blocks=len(assigned.graph.components),
        driven_blocks=sum(1 for t in problem_types(assigned) if t is not None),
    )
    problem = mdf.Mdf(
        graph=graph,
        eager=schedule,
        traceable=schedule,
        design=kept,
        conditions=(report["objective"], *report["inequalities"]),
        n_equality=0,
        n_inequality=len(report["inequalities"]),
        report=report,
        raw=None if cut is cut_graph else raw,
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
    converges, but a Newton over the residualised copy has its scale set by that
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
    cut = cut_graph if built.session.cut is None else built.session.cut
    _driven, _runnable, schedule, run = mda_schedule(built.session.machine_graph, cut)
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


def nested_blocking(built: Closed, driver=None) -> AnswerableGraph:
    """The architecture as structure: the `Optimise` over the kept design inserted and
    `nested_inside` it, so the root finds and the MDA's fixed points are answered
    inside its iteration. For the picture -- `mdf.solve` drives the optimiser from
    outside the graph, as `mdf.assemble` does.
    """
    graph = built.problem.graph  # undriven: `Assign` refuses a problem already driven
    report = built.report
    node = Optimise(
        objective=report["objective"],
        unknowns=built.design,
        equalities=(),
        inequalities=tuple(report["inequalities"]),
    )
    with_problem = (Plan(graph) + Insert(PathMap(((OPTIMISE, node),)))).graph
    with_problem = nested_inside(with_problem, OPTIMISE)
    drivers = default_drivers(with_problem)
    for place in set(built.places.values()):
        drivers[place] = safeguarded() if driver is None else driver
    return AnswerableGraph(assign_drivers(with_problem, drivers))


def describe(graph) -> list[str]:
    """One line per driven or cyclic entry at every depth: its size and its problem."""
    if isinstance(graph, AnswerableGraph):
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
    "Closed",
    "SafeguardedNewtonDriver",
    "close",
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
]
