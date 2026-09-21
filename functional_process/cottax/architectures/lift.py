"""Lifting a sizing choice out of a model into the optimiser, as a graph rewrite.

A sizing rule inside a model -- "the winding pack is as wide as the critical current
requires" -- is an inequality from PROCESS's own constraint catalogue held at zero
slack by a root find. Deterministically that is reduced-space elimination and costs
nothing. Under uncertainty it re-sizes a *build* quantity in every belief sample,
which violates non-anticipativity (`~/jaxgraph`, `plans/handoff_2026-09-17.md`,
narrative item 2; the decision on the winding pack is item 1 of "Decisions on the 14
sizing choices"). The lift is the rewrite that takes the rule apart, in cottax's own
ops and one node composed by hand:

    Undrive(problem)        the algorithm goes first, deliberately (a driven problem)
    Unnest(...)             a nesting either way, since the cycle is about to open
    Undetermine(problem)    the unknown becomes a boundary input -- the design
                            variable the optimiser will own -- and the statement a
                            requirement asserting `residual = 0`
    Delete((problem,))      the requirement: a graph that asserts what it does not
                            enforce is not one a `Schedule` runs, and the inequality
                            is enforced outside it, beside PROCESS's own constraints
    Insert(.Lift.<name>)    an `ImplementedFunction` reading the residual (which its
                            body still computes -- the producer stays, as it does
                            under a `Cut`) and owning `^cond.lift.<unknown>` =
                            `sign * residual / scale`, `sign` chosen so that the
                            *safe* side is negative: the `<= 0` spelling every
                            `Optimise` here reads, `sand.optimise_graph` /
                            `closing.nested_blocking` / `ouu.outer` alike

**Why the last step is composed and not a cottax op.** The relation `Eq` becomes `Le`
on the residual; cottax has `Le` as a relation symbol and `Compare` to mint a gap
between two *places*, but no op that restates a one-sided relation with a sign, and
no place that is zero to compare against. `Compare(place, ((residual, residual),),
compare=negate)` would spell it but names the condition `^cond.^cond...`, a minted
name minted again. So the inequality is one inserted node, said here, and the
`Relation((condition,), (), Le)` a caller would state in an `Optimise` is what
`report["relation"]` carries. `unlift` is not provided: the inverse is a fresh
`Insert` of the problem, and nothing needs it.

**The safe side is the modeller's.** `safe="above"`: a positive residual is
admissible, so the condition is `-residual <= 0`; `safe="below"`: `residual <= 0`.
Which side is safe is not in the graph -- it is what the rule *meant*, and the caller
says it. `lift_winding_pack` establishes it for the coil and
`tests/architectures/test_lift.py` pins it numerically (a wider pack satisfies).

`applied` is the `Closed`-level hook `ouu.two_stage(lifts=...)` uses: every lift on
both of the closed problem's graphs, its `Mdf` rebuilt (design, conditions, schedule,
report), and the seeding env extended with each lifted unknown at the root the
un-lifted problem found -- since a place that was owned has no cold value in the
input file (`native` raises on `.stellarator.wp_width_r_min`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

from cottax.pytree.executable import ExecutableGraph  # noqa: E402
from cottax.execution import RunnableGraph
from cottax.execution.schedule import Schedule  # noqa: E402
from cottax.pytree.names import MintKey, PathMap, prefix_path  # noqa: E402
from cottax.pytree.nodes import ImplementedFunction  # noqa: E402
from cottax.pytree.plan import Delete, Insert, Plan, Unnest  # noqa: E402
from cottax.pytree.problem import (  # noqa: E402
    ConditionalNode,
    Driven,
    Le,
    Relation,
    conditions_of,
    shape_of,
    unknowns_of,
)
from cottax.pytree.rewrites import Undetermine, Undrive  # noqa: E402
from cottax.pytree.spec import NodePath, VarPath  # noqa: E402
from cottax.visualization.sequencing import problem_types  # noqa: E402
from jax.tree_util import GetAttrKey  # noqa: E402

from functional_process.configurations.kinds import KINDS, Kind  # noqa: E402
from functional_process.cottax.architectures import closing, mdf  # noqa: E402
from functional_process.cottax.queries import declared  # noqa: E402
from functional_process.models.stellarator.coils.coils import (  # noqa: E402
    pchip_interp,
)
from functional_process.vocabulary.iteration_variables import (  # noqa: E402
    ITERATION_VARIABLES,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from cottax.pytree.graph import Graph

COND = MintKey("cond")
SAFE = ("above", "below")
"""Which sign of the residual is admissible: `above` -- `residual >= 0` is safe, so
the condition is `-residual <= 0`; `below` -- `residual <= 0` is safe, the condition
is `residual <= 0`."""
PLACE = NodePath((GetAttrKey("Lift"),))
"""Where the lifted inequality's node binds: `.Lift.wp_width_r_min`, beside
`closing.PLACE`'s `.Close.c2`."""
LIFT = GetAttrKey("lift")
"""The namespace under `^cond` a lifted condition is named in: `^cond.lift.<unknown>`.
Not `^cond.<unknown>`, which is the residual's own name (an `ImplicitFunction` mints
`^cond.u` for the residual of `u`)."""


def condition_for(unknown: VarPath) -> VarPath:
    """`^cond.lift.<unknown>`: `.stellarator.wp_width_r_min` ->
    `^cond.lift.stellarator.wp_width_r_min`.
    """
    return prefix_path(VarPath((LIFT, *unknown.segments)), COND)


def place_for(unknown: VarPath) -> NodePath:
    """`.Lift.<the unknown's last component>`."""
    return NodePath((*PLACE.segments, GetAttrKey(unknown.spelling.split(".")[-1])))


@dataclasses.dataclass(frozen=True)
class SignedResidual:
    """The lifted inequality's body: `sign * residual / scale`, `scale` absent = 1.
    A frozen dataclass and not a closure, so a graph built twice has one body and one
    jit key (cottax's `Pairwise` rule).
    """

    sign: float

    def __call__(self, residual, scale=None):
        """The condition: negative on the safe side."""
        value = self.sign * residual
        return value if scale is None else value / scale


# ---------------------------------------------------------------- the generic lift


def _one_unknown(graph: Graph, problem: NodePath, unknown: VarPath) -> ConditionalNode:
    node = graph[problem]
    if not isinstance(node, ConditionalNode):
        raise TypeError(
            f"{problem.spelling} is a {type(node).__name__} and determines nothing, "
            f"so there is no sizing choice to lift"
        )
    owned = tuple(unknowns_of(node))
    if unknown not in owned:
        raise KeyError(
            f"{unknown.spelling} is not an unknown of {problem.spelling}, which "
            f"determines {[u.spelling for u in owned]}"
        )
    if len(owned) != 1:
        raise ValueError(
            f"{problem.spelling} determines {len(owned)} unknowns; a lift takes one "
            f"root find apart, so lifting one of several would leave the rest "
            f"over-determined -- `Undetermine` and `Determine` by hand"
        )
    if shape_of(node) != "root-find":
        raise ValueError(
            f"{problem.spelling} is a {shape_of(node)}, not a root find: a lifted "
            f"condition is a residual with a sign, and only a root find states one "
            f"against zero (`Residualise` a fixed point first)"
        )
    return node


def lift(
    graph: Graph,
    problem: NodePath,
    *,
    unknown: VarPath,
    safe: str,
    scale: VarPath | None = None,
) -> tuple[Graph, VarPath, dict]:
    """`graph` with the root find `problem` lifted: its `unknown` a boundary input and
    its residual an inequality `^cond.lift.<unknown> <= 0`, the admissible sign of the
    residual given by `safe` (`SAFE`). `scale`: a place the signed residual is divided
    by, so the condition is dimensionless like PROCESS's normalised residuals
    (`^cond.constraints.c<n>`); `None` leaves it in the residual's own units.

    Returns the graph, the condition's path and a report: the ops applied (as
    `repr`s), the driver dropped, what was unnested, the sign, and `relation`, the
    `Relation((condition,), (), Le)` a caller states in an `Optimise`.

    Raises
    ------
    ValueError
        If `safe` is not one of `SAFE`, or `problem` is not a one-unknown root find.
    KeyError
        If `unknown` is not what `problem` determines.
    TypeError
        If `problem` has a body.
    """
    if safe not in SAFE:
        raise ValueError(f"safe {safe!r}; one of {SAFE}")
    node = _one_unknown(graph, problem, unknown)
    (residual,) = conditions_of(node)
    driver = type(node.driver).__name__ if isinstance(node, Driven) else None
    sign = -1.0 if safe == "above" else 1.0
    condition = condition_for(unknown)
    place = place_for(unknown)
    reads = (residual,) if scale is None else (residual, scale)

    plan = Plan(graph)
    if driver is not None:
        plan += Undrive(problem)
    unnested = []
    # A nesting either way: the problem inside another's iteration, or others inside
    # its own. Its cycle opens with the lift, so neither pair can stand.
    if problem in graph.within:
        plan += Unnest(problem)
        unnested.append(problem.spelling)
    for inner, outer in graph.within.items():
        if outer == problem:
            plan += Unnest(inner)
            unnested.append(inner.spelling)
    plan += Undetermine(problem)
    plan += Delete((problem,))
    plan += Insert(
        PathMap((
            (place, ImplementedFunction(reads, (condition,), SignedResidual(sign))),
        ))
    )
    report = {
        "problem": problem.spelling,
        "shape": "root-find",
        "unknown": unknown.spelling,
        "residual": residual.spelling,
        "condition": condition.spelling,
        "node": place.spelling,
        "safe": safe,
        "sign": sign,
        "scale": None if scale is None else scale.spelling,
        "driver": driver,
        "unnested": tuple(unnested),
        "ops": tuple(repr(op) for op in plan.ops),
        "relation": Relation((condition,), (), Le),
        "composed": (
            "the inequality is one Insert of an ImplementedFunction computing "
            "sign * residual / scale: cottax has no op restating a one-sided relation "
            "with a sign, and Compare needs two places"
        ),
    }
    return plan.graph, condition, report


# ---------------------------------------------------------------- the winding pack

STELLARATOR = GetAttrKey("stellarator")
INTERSECT = "^problem.stellarator.coils.intersect"
WP_WIDTH_R_MIN = VarPath((STELLARATOR, GetAttrKey("wp_width_r_min")))
"""The root find's unknown: the radial winding-pack width where the required current
density curve meets the critical one (m)."""
J_TF_SC_WP = VarPath((STELLARATOR, GetAttrKey("j_tf_sc_wp")))
"""The current density the superconductor carries at the chosen width, MA/m^2: the
`rhs` curve interpolated at `wp_width_r_min`."""
J_TF_SC_WP_MAX = VarPath((STELLARATOR, GetAttrKey("j_tf_sc_wp_max")))
"""What it may carry, MA/m^2: `f_j_tf_wp_critical_max * j_c(B_peak(width), tftmp +
tmargmin)`, the `lhs` curve interpolated at `wp_width_r_min` -- PROCESS's
`j_tf_wp_critical * f_j_tf_wp_critical_max` on this stellarator."""
CURRENT_DENSITY = NodePath((
    STELLARATOR,
    GetAttrKey("coils"),
    GetAttrKey("winding_pack_current_density"),
))
IXC = 140
"""PROCESS's iteration variable for the pack thickness, `dr_tf_wp_with_insulation`."""
BOUNDS = (
    float(ITERATION_VARIABLES[IXC].lower_bound),
    float(ITERATION_VARIABLES[IXC].upper_bound),
)
"""`ITERATION_VARIABLES[140]`'s own bounds, 0.001 .. 2.0 m -- the place's unit, so
unscaled. The lower one sits below the turn-size floor and the curves' sampled range
(`r_coil_minor / 40`, 0.106 m here), where the interpolant is clamped and the lifted
inequality violated; the optimiser never goes there."""


@dataclasses.dataclass(frozen=True)
class WindingPackCurrentDensity:
    """`(j_tf_sc_wp, j_tf_sc_wp_max)` at the chosen width: the two tabulated curves
    `intersect_residual` subtracts, each interpolated (monotone cubic, as the residual
    is) at `wp_width_r_min`.
    """

    def __call__(self, wp_width_r_min, wp_width_r, lhs, rhs):
        """`(j_tf_sc_wp, j_tf_sc_wp_max)`, MA/m^2, at `wp_width_r_min`."""
        return (
            pchip_interp(wp_width_r_min, wp_width_r, rhs),
            pchip_interp(wp_width_r_min, wp_width_r, lhs),
        )


def _stellarator(name: str) -> VarPath:
    return VarPath((STELLARATOR, GetAttrKey(name)))


def lift_winding_pack(graph: Graph) -> tuple[Graph, VarPath, dict]:
    """The one lift the 2026-09-17 decisions call for: `^problem.stellarator.coils.
    intersect` taken apart, `.stellarator.wp_width_r_min` a design variable, and the
    inequality `j_tf_wp <= f_j_tf_wp_critical_max * j_c` -- PROCESS's icc 33 -- in the
    port's spelling, `^cond.lift.stellarator.wp_width_r_min = j_tf_sc_wp /
    j_tf_sc_wp_max - 1 <= 0`, which is `leq(j, f * j_c)`'s normalised residual
    (`models/constraints.constraint_33`) with the two current densities read off the
    coil's own curves at the chosen width.

    **The design variable and PROCESS's ixc 140.** PROCESS iterates
    `dr_tf_wp_with_insulation`, the radial pack thickness (m). On this stellarator
    `winding_pack_post_intersect` sets `dr_tf_wp_with_insulation =
    max(dx_tf_turn_general**2, wp_width_r_min)`: the thickness *is* the root find's
    unknown above the turn-size floor (0.056^2 = 3.1e-3 m against a root of 0.63 m),
    and `j_tf_wp = coilcurrent * 1e6 / (wp_width_r_min**2 / wp_ratio)` follows from
    it. So the port's own place, the unknown, is what is lifted; the bounds are
    `ITERATION_VARIABLES[140]`'s, in the same unit (`BOUNDS`).

    **The safe side.** `intersect_residual = lhs(w) - rhs(w)` with `lhs = f * j_c(w)`
    (rising with width: a wider pack has a lower peak field) and `rhs = I /
    (w^2 / ratio * f_sc)` (falling as 1 / w^2), so the residual is positive on the
    wide side of the root -- a wider pack carries less current density than it may.
    `safe="above"`, and the condition is `-(lhs - rhs) / lhs = rhs / lhs - 1`.

    Two nodes are inserted: `.stellarator.coils.winding_pack_current_density`, owning
    `j_tf_sc_wp` and `j_tf_sc_wp_max` at the chosen width (the second is the
    `scale`), and the lift's own `.Lift.wp_width_r_min`. The report is `lift`'s plus
    `design`: the unknown, `BOUNDS`, `IXC`, and `Kind.BUILD` for `stages.leaves`.

    Raises
    ------
    KeyError
        If the graph has no `^problem.stellarator.coils.intersect` -- a tokamak, or a
        closing that folded it into its root find.
    """
    problem = next((p for p in declared(graph) if p.spelling == INTERSECT), None)
    if problem is None:
        raise KeyError(
            f"{INTERSECT} is not a problem of this graph -- the winding-pack lift is "
            f"the stellarator coil's, and it must still be a root find of its own"
        )
    with_density = (
        Plan(graph)
        + Insert(
            PathMap((
                (
                    CURRENT_DENSITY,
                    ImplementedFunction(
                        reads=(
                            WP_WIDTH_R_MIN,
                            _stellarator("wp_width_r"),
                            _stellarator("lhs"),
                            _stellarator("rhs"),
                        ),
                        owns=(J_TF_SC_WP, J_TF_SC_WP_MAX),
                        fn=WindingPackCurrentDensity(),
                    ),
                ),
            ))
        )
    ).graph
    lifted, condition, report = lift(
        with_density,
        problem,
        unknown=WP_WIDTH_R_MIN,
        safe="above",
        scale=J_TF_SC_WP_MAX,
    )
    report = dict(
        report,
        icc=33,
        current_density_node=CURRENT_DENSITY.spelling,
        current_density=(J_TF_SC_WP.spelling, J_TF_SC_WP_MAX.spelling),
        design={
            "unknown": WP_WIDTH_R_MIN.spelling,
            "bounds": BOUNDS,
            "bounds_from": f"ITERATION_VARIABLES[{IXC}], the place's own unit (m)",
            "ixc": IXC,
            "kind": Kind.BUILD,
        },
    )
    return lifted, condition, report


# ---------------------------------------------------------------- on a closed problem


def _resolve(spelling: str, candidates: Iterable[VarPath]) -> VarPath:
    for v in candidates:
        if v.spelling == spelling:
            return v
    raise KeyError(f"{spelling} is not a boundary input of the lifted graph")


def applied(
    built: closing.Closed,
    lifts: Iterable[Callable[[Graph], tuple[Graph, VarPath, dict]]],
    env: Mapping[VarPath, object],
) -> tuple[closing.Closed, dict]:
    """`built` with every lift in `lifts` applied, and `env` extended for it.

    Each lift (`lift_winding_pack`, or any `graph -> (graph, condition, report)` whose
    report carries `design`) is applied to both the driven graph the schedule runs
    and the undriven one `closing.nested_blocking` draws; the `Mdf` is rebuilt: the
    design plus each lifted unknown, the conditions and `report["inequalities"]`
    plus each condition, one more inequality apiece, a fresh `Schedule`, and
    `report["lifts"]` (condition spelling -> the lift's report). `env` is a
    `closing.seed` env of the un-lifted problem; the un-lifted problem is primed once
    (`mdf.prime`) and each lifted unknown written at the root it found, since a place
    that was owned has no cold value in the file.

    Returns the new `Closed` and the env, its keys those of `env` plus the unknowns.

    Raises
    ------
    ValueError
        If a lift names different conditions on the two graphs.
    """
    lifts = tuple(lifts)
    _primed, roots = mdf.prime(built.problem, env)
    driven, undriven = built.graph, built.problem.graph
    reports: dict = {}
    conditions: list[VarPath] = []
    for one in lifts:
        driven, condition, report = one(driven)
        undriven, same, _ = one(undriven)
        if same != condition:
            raise ValueError(
                f"{one.__name__} lifted {condition.spelling} on the driven graph and "
                f"{same.spelling} on the undriven one"
            )
        reports[condition.spelling] = report
        conditions.append(condition)
    unknowns = tuple(
        _resolve(r["design"]["unknown"], driven.graph.boundary_inputs)
        for r in reports.values()
    )
    schedule = Schedule(RunnableGraph(driven))
    old = built.problem
    report = dict(
        old.report,
        inequalities=tuple(old.report["inequalities"]) + tuple(conditions),
        lifts=reports,
        blocks=len(driven.graph.components),
        driven_blocks=sum(1 for t in problem_types(driven) if t is not None),
    )
    problem = dataclasses.replace(
        old,
        graph=undriven,
        eager=schedule,
        traceable=schedule,
        design=tuple(old.design) + unknowns,
        conditions=tuple(old.conditions) + tuple(conditions),
        n_inequality=old.n_inequality + len(conditions),
        report=report,
    )
    lifted = dataclasses.replace(built, problem=problem, design=problem.design)
    extended = dict(env)
    for unknown in unknowns:
        extended[unknown] = roots[unknown]
    return lifted, extended


def lifted_design(built: closing.Closed) -> dict[VarPath, dict]:
    """`{unknown: design}` over `built.report["lifts"]`: each lifted design
    variable's `bounds`, `ixc` and `kind`, as the lift reported them. Empty for a
    `Closed` nothing was lifted on.
    """
    reports = built.report.get("lifts", {})
    if not reports:
        return {}
    by_spelling = {v.spelling: v for v in built.design}
    return {by_spelling[r["design"]["unknown"]]: r["design"] for r in reports.values()}


def kind_table(built: closing.Closed, kinds: Mapping[str, Kind] = KINDS) -> dict:
    """`kinds` plus each lifted unknown at the kind its lift reported (`Kind.BUILD` for
    the winding pack): what `stages.leaves` takes for the lifted graph, whose new
    boundary input the hand-made table does not know.
    """
    table = dict(kinds)
    for unknown, design in lifted_design(built).items():
        table[unknown.spelling] = design["kind"]
    return table


__all__ = [
    "BOUNDS",
    "COND",
    "CURRENT_DENSITY",
    "INTERSECT",
    "IXC",
    "J_TF_SC_WP",
    "J_TF_SC_WP_MAX",
    "LIFT",
    "PLACE",
    "SAFE",
    "WP_WIDTH_R_MIN",
    "SignedResidual",
    "WindingPackCurrentDensity",
    "applied",
    "condition_for",
    "kind_table",
    "lift",
    "lift_winding_pack",
    "lifted_design",
    "place_for",
]
