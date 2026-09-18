"""Optimisation under uncertainty as a two-stage architecture over the closed MDA.

The formulation the 2026-09-17 handoff settled on ("variant D", `~/jaxgraph`,
`plans/handoff_2026-09-17.md` -- OUU prototype):

- **First stage**, the design `x`: the file's `ixc` minus every place a belief draws
  (`hfact`) minus every closing variable (the density) -- the build and the shared
  operating set-points -- bounded by the file's own bounds. With `te_recourse`, minus
  the electron temperature too; with `lifts`, plus every sizing choice lifted out of
  the models (`lift.lift_winding_pack`: the pack width, its rule a chance constraint).
- **Second stage**, per belief sample: the power balance `c2` closed by the density
  inside the MDA (`closing.close`, `kinds.PAIRINGS["one"]`), and with `te_recourse = K`
  the temperature chosen per sample from a grid of `K` values -- the cheapest feasible
  one, else the least infeasible -- with `stop_gradient` on the choice so the outer
  derivative is the envelope theorem's at the chosen point.
- **Beliefs**: `kinds.BELIEFS` minus `held` (the *build table*: `BUILD_LEAVES`, which
  the stage check found behind every coil and radial-build place, and `ECONOMIC`, at
  their nominal), drawn once as a scrambled Sobol' set (sample-average approximation,
  common random numbers), the nominal point appended as the last row so every batched
  call also answers the nominal design.
- **Chance constraints**: `CVaR_alpha[g_j] <= 0` for every inequality of the file (and
  `c16`, the net power, with `with_c16` -- infeasible on this problem, the handoff
  found; it is priced into the objective instead), the Rockafellar-Uryasev sample
  estimator, the mean of the worst `ceil((1 - alpha) N)` converged samples
  (`lax.top_k`). A sample whose closing Newton did not converge is **infeasible**: it
  enters every CVaR at `FAILED_G` and the objective at `FAILED_F / N`, and the failed
  fraction is itself bounded by `EPS_FAILED`.
- **Objective**: `levelised` -- `sum_i w_i coe_i kwh_i / sum_i w_i kwh_i`, the ratio of
  expectations, so a shortfall of net power is priced. `mean` (E[coe | converged,
  net > 0]) is ill-posed since coe ~ 1 / net and failed at every alpha; `median`,
  `nominal` and `rated` are kept as measures.

**The hoist.** The closed graph is split by `stages.split` at the leaves that vary
per sample (the sampled beliefs, the closing problem's start ports, the temperature
under recourse): the first stage -- 66 of 170 nodes on `stellarator_helias`, the
whole coil and radial build -- runs once per design, and the batched program `vmap`s
the recourse subgraph alone over the samples, reading the first stage's outputs as
constants. `make(hoist=False)` runs the whole closed schedule per sample instead; the
two agree to round-off (`tests/architectures/test_ouu.py`).

**Where the batched program sits.** The statistics are one node of a small graph --
`Statistics`, reading the design places and owning `^cond.ouu.objective`,
`^cond.ouu.cvar.<constraint>` and `^cond.ouu.failed` -- and the outer problem is an
`Optimise` over the design reading them, `nested` on their cycle, so the architecture
is a `Graph` + `Schedule(AnswerableGraph(...))` like the others. The node's body is
the jitted batched program (`Program`), called outside any trace by the driver: a
`jax.vmap` over a `Schedule.run` inside an `ImplementedFunction` inside a second
`Schedule` would trace the batch into every evaluation of the outer graph, and the
warm starts (every sample's next start is its converged closing unknowns at the
**incumbent**) and the fused value + Jacobian + starts are what one compiled call
returns and a condition map cannot carry. So the node memoises per design point, the
`BoxedSlsqpDriver` takes the Jacobian rows from that memo (`jacobian=`) and adopts
the incumbent's starts through it (`warm_start=`), and the outer schedule is run
with `whole=False`, as SAND's is. Both hooks are `outer`'s keywords: `DEFAULT` is the
program's own, `None` the driver's own meaning (no warm start; `jacfwd` through the
node), the control a study measures the warm starts against.

**The rows.** `TwoStage.columns` is what one batched call returns per sample:
coe, net, availability, constructed cost, every constraint under CVaR, the closing
problem's `Steps` and `Converged`, its unknowns, then every *other* driven problem's
`Converged` in the recourse schedule (`verdicts`: a row is valid only if every
driver in it converged -- none report one on `stellarator_helias`, whose other
drivers are Picards), then `extra_columns` (`two_stage(extra_columns=...)`), spellings
the caller wants per sample and the measures ignore. `TwoStage.layout` is the
offsets; `valid_rows`, `per_sample`, `summarise` and `extra_columns` read by them.

**Differentiation.** `jacfwd` of the statistics vector: six inputs, fourteen outputs,
and every loop in the closed MDA carries an implicit derivative (the closing Newton
under `lax.custom_root`, the Picards and the coil root find under optimistix's
implicit adjoint). `chunks` runs the samples through `lax.map` in that many
`jax.checkpoint`ed pieces, which bounds reverse mode's memory to one chunk's.

What stays in `paper_tests/ouu.py`: the CLI, the scaling study, the JSON / CSV /
`.npz` writers, the plots, `sweep_te`, `decompose` and `evidence`'s file handling.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
import operator
import time
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cottax.answerable import AnswerableGraph  # noqa: E402
from cottax.evaluation.schedule import Drive, Schedule  # noqa: E402
from cottax.graph import Graph  # noqa: E402
from cottax.names import MintKey, PathMap, prefix_path  # noqa: E402
from cottax.nodes import ImplementedFunction  # noqa: E402
from cottax.problem import Converged, Optimise, Steps, unknowns_of  # noqa: E402
from cottax.spec import NodePath, VarPath  # noqa: E402
from jax import lax  # noqa: E402
from jax.tree_util import GetAttrKey  # noqa: E402

from functional_process.configurations import kinds  # noqa: E402
from functional_process.configurations.kinds import (  # noqa: E402
    BELIEFS,
    BUILD_LEAVES,
    C16,
    ECONOMIC,
    TE,
    Belief,
    Kind,
)
from functional_process.cottax.architectures import (  # noqa: E402
    beliefs as beliefs_,
)
from functional_process.cottax.architectures import (  # noqa: E402
    closing,
    lift,
    mdf,
    sand,
    stages,
)
from functional_process.cottax.architectures.drivers import (  # noqa: E402
    BoxedSlsqpDriver,
    Status,
)
from functional_process.cottax.architectures.evaluate import run_schedule  # noqa: E402
from functional_process.cottax.architectures.mda import (  # noqa: E402
    assign_drivers,
    guess_sources,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

ALPHA = 0.9
EPS_FAILED = 0.02
"""Largest tolerated fraction of samples whose closed MDA did not converge."""
FAILED_G = 10.0
"""A failed sample is infeasible: it enters every CVaR at this (normalised) violation,
so a design where the closed MDA stops converging cannot look feasible. A constant,
so it carries no gradient; what pulls the design back is the converged samples'."""
G_TOL = 2.0e-3
"""A sample violates a constraint at `g > G_TOL` (normalised: 0.2 % of the limit),
not `g > 0`: a constraint an optimiser holds active sits a hair above zero and would
otherwise read as violated in every sample -- the radial build `c83` at +2e-9 after
SLSQP, and PROCESS's own converged point has `c24` at +1.5e-3 and `c35` at +1.6e-4
(VMCON's slack), so the deterministic design is feasible only to this tolerance."""
FAILED_F = 1.0e3
"""Cost added to `f` per unit failed fraction (and `f` itself where nothing
converged): about ten times the deterministic coe, so a line search rejects a point
whose samples fail rather than reading their absence as a cheaper plant."""

OBJECTIVES = ("mean", "median", "levelised", "nominal", "rated")
"""`rated`: `mean_i(annual cost_i) / (rated energy)`, the expected annual cost per
MWh the plant is rated for (`p_plant_electric_net_required_mw` x nominal
availability) -- finite and differentiable, which `mean` (coe ~ 1 / net) is not; but
it does not price a shortfall, so without `c16` as a constraint it is the wrong one."""
JAC_MODES = ("fwd", "rev")

COE = ".costs.coe"
NET = ".heat_transport.p_plant_electric_net_mw"
AVAIL = ".costs.f_t_plant_available"
CONCOST = ".costs.concost"
RATED = ".constraints.p_plant_electric_net_required_mw"
BUILD_TABLE: tuple[str, ...] = BUILD_LEAVES + ECONOMIC
"""`held` by default: the build leaves and the economic rows at their nominal."""

COND = MintKey("cond")
STATISTICS = NodePath((GetAttrKey("Statistics"),))
"""Where the statistics node binds in the outer graph."""
OPTIMISE = NodePath((GetAttrKey("Ouu"),))
"""Where the outer `Optimise` binds."""


def condition_path(name: str) -> VarPath:
    """`^cond.ouu.<name>`."""
    return prefix_path(VarPath((GetAttrKey("ouu"), GetAttrKey(name))), COND)


def cvar_path(constraint) -> VarPath:
    """`^cond.ouu.cvar.<label>` for a constraint (a `VarPath` or a `RecourseBound`)."""
    return prefix_path(
        VarPath((
            GetAttrKey("ouu"),
            GetAttrKey("cvar"),
            GetAttrKey(label_of(constraint)),
        )),
        COND,
    )


def label_of(constraint) -> str:
    """`c24` for `^cond.constraints.c24`; `f_nd_alpha_thermal_electron_lower` for a
    `RecourseBound`.
    """
    if isinstance(constraint, RecourseBound):
        return f"{constraint.var.spelling.rsplit('.', 1)[-1]}_{constraint.side}"
    return constraint.spelling.rsplit(".", 1)[-1]


# ---------------------------------------------------------------- the problem


@dataclasses.dataclass(frozen=True)
class RecourseBound:
    """A closing variable's bound as a per-sample inequality: the recourse exists only
    where the root the closure found is a physical operating point (an alpha fraction
    below zero delivers the rated power on paper and nowhere else). Normalised by the
    bound's range, so it sits in the CVaR beside the graph's own inequalities. Only
    past one pairing: the density's own bounds are what `c2` needs, not a constraint.
    """

    var: VarPath
    side: str
    lower: float
    upper: float

    @property
    def spelling(self) -> str:
        """`^bound<var>.<side>`, the column's name."""
        return f"^bound{self.var.spelling}.{self.side}"

    def residual(self, value):
        """The normalised violation of this side at `value`: positive outside."""
        span = self.upper - self.lower
        if self.side == "lower":
            return (self.lower - value) / span
        return (value - self.upper) / span


@dataclasses.dataclass(frozen=True)
class TwoStage:
    """The assembled two-stage problem: the closed MDA, its split, the sample set and
    the columns the batched program returns. From `two_stage`; `make` compiles it.
    """

    closed: closing.Closed
    point: PathMap
    """Every schedule input at the nominal: the deterministic start (the file's own
    design, or `design_values`), the closing unknowns' start ports at their roots."""
    nominal_out: dict
    """The un-batched closed MDA's env at the nominal, from `mdf.prime`."""
    var_of: Mapping[str, VarPath]
    """Spelling -> `VarPath`, inputs and outputs alike."""
    beliefs: tuple[Belief, ...]
    """The rows drawn, `dummy` kept, in `kinds.BELIEFS` order."""
    dropped: tuple[str, ...]
    """Belief paths the closed graph does not read."""
    nominal: Mapping[str, np.ndarray]
    """Belief path -> its nominal value."""
    design: tuple[VarPath, ...]
    """The first-stage places, in `ixc` order, every lifted unknown after them."""
    ixc: tuple[int, ...]
    """PROCESS's iteration variable per design place; a lifted unknown's is the one
    its lift names (`lift.IXC`, 140, for the pack width)."""
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    constraints: tuple
    """Under CVaR: the file's inequalities, `c16` with `with_c16`, the
    `RecourseBound`s past one pairing."""
    columns: tuple
    """Per-sample columns: coe, net, f_avail, concost, *constraints, steps,
    converged, *unknowns, *verdicts, *extra."""
    unknowns: tuple[VarPath, ...]
    """The closing problem's unknowns (the closing variable first, then the cut
    copies folded into it)."""
    guesses: tuple[VarPath, ...]
    """Their start ports, parallel."""
    place: NodePath
    """The closing problem, whose `Steps` / `Converged` are two columns."""
    verdicts: tuple[VarPath, ...]
    """Every *other* driven problem's `Converged` in the recourse schedule (the
    `^driver_out.converged^problem.<place>` names of the drivers that report one), so
    a row's validity sees every driver's verdict, not only the closing problem's.
    Empty on a configuration whose other drivers report nothing (the Picards and the
    coil root find on `stellarator_helias`)."""
    extra: tuple[VarPath, ...]
    """Columns the caller asked for (`two_stage(extra_columns=...)`): carried through
    per sample, ignored by `measures`."""
    alpha: float
    n: int
    seed: int
    rated_mw: float
    alpha16: float | None
    te_grid: np.ndarray | None
    """`te_recourse = K`: the temperature is not a design place but chosen per sample
    from these `K` values."""
    objective: str
    pairing: str
    held: tuple[str, ...]
    Theta: np.ndarray
    """Coordinates [N + 1, k], the nominal point as the last row."""
    theta: PathMap
    """The batched env over the sampled paths, [N + 1, ...] each."""
    starts0: np.ndarray
    """[N + 1, unknowns]: every row started from the nominal root."""
    stages: stages.Stages
    first: Schedule
    """The first stage: a function of the design alone."""
    recourse: Schedule
    """The second stage: what the batch runs, the first stage's outputs among its
    inputs."""
    build_s: float

    @property
    def n_g(self) -> int:
        """How many constraints are under CVaR."""
        return len(self.constraints)

    @property
    def m(self) -> int:
        """How many worst samples a CVaR averages."""
        return max(1, math.ceil((1.0 - self.alpha) * self.n))

    @property
    def m16(self) -> int:
        """How many worst samples `c16`'s CVaR averages: `alpha16`'s where one is
        given (a shortfall of the rated power is a softer requirement than a physics
        limit), else `alpha`'s.
        """
        alpha = self.alpha if self.alpha16 is None else self.alpha16
        return max(1, math.ceil((1.0 - alpha) * self.n))

    @property
    def whole(self) -> Schedule:
        """The closed MDA's own schedule -- what `make(hoist=False)` runs per sample."""
        return self.closed.problem.traceable

    @property
    def names(self) -> tuple[str, ...]:
        """The constraints' spellings."""
        return tuple(c.spelling for c in self.constraints)

    @property
    def layout(self) -> dict:
        """Column offsets: `c_coe`, `c_net`, `c_avail`, `c_concost`, `c_g` (a pair),
        `c_steps`, `c_conv`, `c_u` (a pair), `c_verdicts` (a pair, the other drivers'
        `Converged` columns), `verdicts` (their problems' spellings), `c_extra` (a
        pair), `extra` (the spellings asked for), `tiny`.
        """
        n_g = self.n_g
        u0 = 6 + n_g
        v0 = u0 + len(self.unknowns)
        e0 = v0 + len(self.verdicts)
        return {
            "c_coe": 0,
            "c_net": 1,
            "c_avail": 2,
            "c_concost": 3,
            "c_g": (4, 4 + n_g),
            "c_steps": 4 + n_g,
            "c_conv": 5 + n_g,
            "c_u": (u0, v0),
            "c_verdicts": (v0, e0),
            "verdicts": tuple(_problem_of(v) for v in self.verdicts),
            "c_extra": (e0, e0 + len(self.extra)),
            "extra": tuple(v.spelling for v in self.extra),
            "tiny": TINY,
        }

    @property
    def rated_kwh(self) -> float:
        """The rated net energy in `TINY`'s unit: the required net power at the nominal
        availability (an economic input, at its nominal in every table).
        """
        return self.rated_mw * float(np.asarray(self.point[self.var_of[AVAIL]]))


TINY = 1.0e-10 / (1.0e3 * 24.0 * 365.2425)
"""The cost model's `max(kwhpy, 1e-10)` guard, in MW-years of net energy."""


def _closing_ports(
    built: closing.Closed,
) -> tuple[tuple[VarPath, ...], tuple[VarPath, ...]]:
    """The closing problem's unknowns and their start ports, parallel."""
    place = next(iter(built.places.values()))
    unknowns = tuple(unknowns_of(built.graph[place]))
    ports = {u: g for g, u in guess_sources(built.graph).items()}
    return unknowns, tuple(ports[u] for u in unknowns)


def _problem_of(verdict: VarPath) -> str:
    """`^driver_out.converged^problem.Close.c2` -> `^problem.Close.c2`."""
    return "^problem" + verdict.spelling.split("^problem", 1)[1]


def driven_problems(schedule: Schedule) -> tuple[Drive, ...]:
    """Every `Drive` step of `schedule`, at every depth, in schedule order."""
    found: list[Drive] = []
    for step in schedule.steps:
        if isinstance(step, Drive):
            found.append(step)
            if isinstance(step.body, Schedule):
                found.extend(driven_problems(step.body))
    return tuple(found)


def other_verdicts(schedule: Schedule, place: NodePath) -> tuple[VarPath, ...]:
    """The `Converged` report of every driven problem of `schedule` other than
    `place`, for the drivers that report one (`Drive.reports`), in schedule order:
    what a row's validity has to see beside the closing problem's own verdict.
    """
    verdicts = []
    for step in driven_problems(schedule):
        if step.problem == place:
            continue
        converged = Converged.name_for(step.problem)
        if converged in step.reports:
            verdicts.append(converged)
    return tuple(verdicts)


def resolve_columns(
    spellings: Iterable[str], var_of: Mapping[str, VarPath]
) -> tuple[VarPath, ...]:
    """`spellings` as `VarPath`s through `var_of` (the closed graph's inputs and
    outputs by spelling).

    Raises
    ------
    KeyError
        If a spelling is not a variable of the closed graph, naming the spellings
        nearest to it.
    """
    resolved = []
    for spelling in spellings:
        var = var_of.get(spelling)
        if var is None:
            near = difflib.get_close_matches(spelling, list(var_of), n=5, cutoff=0.5)
            raise KeyError(
                f"{spelling!r} is not a variable of the closed graph"
                + (f"; near it: {near}" if near else "")
            )
        resolved.append(var)
    return tuple(resolved)


def two_stage(
    session,
    *,
    pairing: str = "one",
    beliefs: Iterable[Belief] = BELIEFS,
    held: Iterable[str] = BUILD_TABLE,
    alpha: float = ALPHA,
    objective: str = "levelised",
    n: int = 256,
    seed: int = 0,
    te_recourse: int = 0,
    te_range: tuple | None = None,
    with_c16: bool = False,
    alpha16: float | None = None,
    design_values=None,
    closing_values=None,
    flatten: bool = True,
    lifts: Iterable = (),
    extra_columns: Iterable[str] = (),
) -> TwoStage:
    """Assemble the two-stage problem on `session` (a `session.Session`, a
    configuration or its name).

    The closed MDA (`closing.close(session, kinds.PAIRINGS[pairing])`) is seeded and
    primed at the configuration's own cold design (`closing.seed`, `mdf.prime`) --
    or at `design_values` / `closing_values`, as `closing.seed` takes them -- and that
    point is the nominal. The beliefs are `beliefs` minus `held` (spellings), the
    ones the closed graph does not read dropped; the design is the closed problem's
    minus every sampled place minus `TE` under `te_recourse`; the graph is split at
    the sampled leaves and the closing start ports (and `TE`) and the two stages'
    schedules built.

    `lifts`: sizing choices lifted out of the models (`lift.lift_winding_pack`),
    applied to the closed graph (`lift.applied`): each lifted unknown joins the design
    as a first-stage build variable (bounds, `ixc` and kind as its lift reports them)
    and its inequality joins the constraints under CVaR -- a chance constraint at
    `alpha` -- beside the file's own.

    `extra_columns`: spellings of the closed graph's variables (an output such as
    `.physics.p_fusion_total_mw` or `^cond.constraints.c2`, an input too) the caller
    wants per sample beside the measures' own columns: appended last, after the
    closing unknowns and the other drivers' verdicts, at `layout["c_extra"]`, named
    in `layout["extra"]`; `measures` ignores them, `per_sample` and `summarise`
    carry them through under `"extra"`.

    Raises
    ------
    ValueError
        If `objective` or `pairing` is unknown, or `with_c16` is asked of a pairing
        that closes `c16`.
    KeyError
        If an extra column is not a variable of the closed graph.
    """
    began = time.perf_counter()
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r}; one of {OBJECTIVES}")
    if pairing not in kinds.PAIRINGS:
        raise ValueError(f"pairing {pairing!r}; one of {tuple(kinds.PAIRINGS)}")
    if with_c16 and C16 in kinds.PAIRINGS[pairing]:
        raise ValueError(
            f"`with_c16` states c16 as a CVaR constraint; pairing {pairing!r} closes "
            f"it per sample"
        )
    held = tuple(held)
    built = closing.close(session, kinds.PAIRINGS[pairing], flatten=flatten)
    live = built.session
    problem = built.problem
    env = closing.seed(built, live.reference.cold, design_values, closing_values)
    if lifts:
        built, env = lift.applied(built, lifts, env)
        problem = built.problem
    env, primed = mdf.prime(problem, env)
    point = PathMap(mdf._inputs_only(problem, env).items())
    var_of = {v.spelling: v for v in point}
    var_of.update({k.spelling: k for k in primed})

    rows, dropped, nominal = [], [], {}
    for belief in beliefs_.sampled(beliefs, held):
        if belief.path == "dummy":
            rows.append(belief)
            continue
        var = var_of.get(belief.path)
        if var is None or var not in point:
            dropped.append(belief.path)
            continue
        rows.append(belief)
        nominal[belief.path] = np.asarray(point[var])
    rows = tuple(rows)
    uncertain = {var_of[b.path] for b in rows if b.path != "dummy"}

    design = tuple(v for v in built.design if v not in uncertain)
    bounds = {v: (lo, hi) for v, lo, hi in live.reference.bounds}
    ixc_of = {sand.iteration_variable_path(i): i for i in live.reference.ixc}
    for unknown, lifted in lift.lifted_design(built).items():
        bounds[unknown] = tuple(lifted["bounds"])
        ixc_of[unknown] = lifted["ixc"]
    te_grid = None
    if te_recourse:
        te = var_of[TE]
        design = tuple(v for v in design if v != te)
        lo, hi = te_range or bounds[te]
        te_grid = np.linspace(lo, hi, te_recourse)
    x0 = np.array([float(np.asarray(point[v])) for v in design])
    lower = np.array([bounds[v][0] for v in design])
    upper = np.array([bounds[v][1] for v in design])

    constraints = tuple(problem.report["inequalities"])
    if with_c16:
        constraints += (var_of[C16],)
    if len(built.pairings) > 1:
        for var in built.closing:
            if var in bounds:
                lo, hi = bounds[var]
                constraints += (
                    RecourseBound(var, "lower", lo, hi),
                    RecourseBound(var, "upper", lo, hi),
                )
    place = next(iter(built.places.values()))
    unknowns, guesses = _closing_ports(built)
    extra = resolve_columns(extra_columns, var_of)

    u0 = np.array([float(np.asarray(primed[u])) for u in unknowns])
    Theta, theta = sample(rows, var_of, nominal, n, seed)
    starts0 = np.tile(u0, (n + 1, 1))

    # The split: what varies per sample is a belief, a closing start, and the
    # temperature under recourse. The closing ports have no kind in the table (the
    # table sorts the MDA's own leaves), so they are given one here.
    varying = tuple(g.spelling for g in guesses) + ((TE,) if te_recourse else ())
    table = lift.kind_table(built)
    for spelling in varying:
        table.setdefault(spelling, Kind.OPERATING)
    graph = built.graph
    split = stages.split(
        graph,
        stages.leaves(
            graph,
            table,
            sampled=tuple(b.path for b in rows if b.path != "dummy"),
            varying=varying,
        ),
    )
    first = Schedule(AnswerableGraph(stages.first_stage_graph(graph, split)))
    recourse = Schedule(AnswerableGraph(stages.recourse_graph(graph, split)))
    verdicts = other_verdicts(recourse, place)
    columns = (
        var_of[COE],
        var_of[NET],
        var_of[AVAIL],
        var_of[CONCOST],
        *constraints,
        Steps.name_for(place),
        Converged.name_for(place),
        *unknowns,
        *verdicts,
        *extra,
    )

    return TwoStage(
        closed=built,
        point=point,
        nominal_out=primed,
        var_of=var_of,
        beliefs=rows,
        dropped=tuple(dropped),
        nominal=nominal,
        design=design,
        ixc=tuple(ixc_of[v] for v in design),
        x0=x0,
        lower=lower,
        upper=upper,
        constraints=constraints,
        columns=columns,
        unknowns=unknowns,
        guesses=guesses,
        place=place,
        verdicts=verdicts,
        extra=extra,
        alpha=alpha,
        n=n,
        seed=seed,
        rated_mw=float(np.asarray(point[var_of[RATED]])),
        alpha16=alpha16,
        te_grid=te_grid,
        objective=objective,
        pairing=pairing,
        held=held,
        Theta=Theta,
        theta=theta,
        starts0=starts0,
        stages=split,
        first=first,
        recourse=recourse,
        build_s=time.perf_counter() - began,
    )


def sample(beliefs, var_of, nominal, n: int, seed: int) -> tuple[np.ndarray, PathMap]:
    """One scrambled Sobol' set of `n` points over `beliefs` (a dummy column is drawn
    and ignored), the nominal point appended as row `n`: the coordinates and the
    batched env.
    """
    u_q = beliefs_.sobol(beliefs, n, seed)
    Theta = np.vstack([
        beliefs_.coordinates(beliefs, u_q, nominal),
        beliefs_.nominal_coordinates(beliefs, nominal)[None, :],
    ])
    return Theta, beliefs_.theta_env(beliefs, var_of, Theta, nominal)


def fresh_sample(model: TwoStage, seed: int) -> tuple[np.ndarray, PathMap, np.ndarray]:
    """`sample` with another seed, over `model`'s beliefs: `(Theta, theta, starts0)`,
    the starts every row's nominal root as `model.starts0`.
    """
    Theta, theta = sample(model.beliefs, model.var_of, model.nominal, model.n, seed)
    return Theta, theta, np.tile(model.starts0[-1], (model.n + 1, 1))


# ---------------------------------------------------------------- the functions


def make(
    model: TwoStage, jac: str = "fwd", chunks: int | None = None, hoist: bool = True
) -> dict:
    """The jittable functions over `(x_flat, theta, starts)`; `theta` and `starts`
    are arguments rather than closed-over constants so that nothing of size N is baked
    into an executable. Every batched array has N + 1 rows, the nominal point last.

    `hoist`: run the first stage once per design and the recourse schedule per
    sample; `False` runs the whole closed schedule per sample (the un-hoisted
    program, for checking the split).

    Returns `run_batch`, `first`, `statistics`, `statistics_vec`, `statistics_full`,
    `statistics_chunked_vec`, `value_jac_starts` and `layout`.

    Raises
    ------
    ValueError
        If `jac` is not one of `JAC_MODES`.
    """
    if jac not in JAC_MODES:
        raise ValueError(f"jac {jac!r}; one of {JAC_MODES}")
    whole_run = model.whole.run
    first_run, recourse_run = model.first.run, model.recourse.run
    first_inputs = tuple(model.first.inputs)
    recourse_inputs = tuple(model.recourse.inputs)
    point = dict(model.point.items())
    design, guesses, columns = model.design, model.guesses, model.columns
    n_g, m, m16, objective = model.n_g, model.m, model.m16, model.objective
    i16 = next((i for i, c in enumerate(model.constraints) if c.spelling == C16), None)
    layout = model.layout
    c_coe, c_net, c_avail, c_concost = 0, 1, 2, 3
    c_g = slice(*layout["c_g"])
    c_steps, c_conv = layout["c_steps"], layout["c_conv"]
    c_u = slice(*layout["c_u"])
    c_verdicts = slice(*layout["c_verdicts"])
    rated_kwh = model.rated_kwh
    te_var, te_grid = model.var_of[TE], model.te_grid

    def _column(c, env):
        value = c.residual(env[c.var]) if isinstance(c, RecourseBound) else env[c]
        return jnp.asarray(value, dtype=jnp.float64).reshape(())

    def first(x_flat):
        """The first stage at design `x_flat`: what every sample reads as constant."""
        values = dict(point)
        values.update(zip(design, [x_flat[j] for j in range(len(design))], strict=True))
        out = first_run(PathMap((v, values[v]) for v in first_inputs))
        return dict(out.items())

    def at_point(x_flat, theta_row, start_row, te=None, first_env=None):
        values = dict(point)
        values.update(theta_row.items())
        values.update(zip(design, [x_flat[j] for j in range(len(design))], strict=True))
        values.update(
            zip(guesses, [start_row[j] for j in range(len(guesses))], strict=True)
        )
        if te is not None:
            values[te_var] = te
        if first_env is None:
            out = whole_run(PathMap(values))
        else:
            values.update(first_env)
            out = recourse_run(PathMap((v, values[v]) for v in recourse_inputs))
        joined = {**values, **dict(out.items())}
        return jnp.stack([_column(c, joined) for c in columns])

    def converged(rows):
        """Which rows count: the closing problem and every other driver converged,
        and every measured column finite (`valid_rows`, traced).
        """
        finite = jnp.all(jnp.isfinite(rows[:, :c_steps]), axis=1)
        verdicts = jnp.all(rows[:, c_verdicts] > 0.5, axis=1)
        return (rows[:, c_conv] > 0.5) & verdicts & finite

    def choose(rows):
        """One sample's K rows -> the row at its recourse: the cheapest feasible T_e,
        else the least infeasible one.
        """
        conv = converged(rows)
        g = rows[:, c_g]
        feasible = conv & (rows[:, c_net] > 0) & jnp.all(g <= G_TOL, axis=1)
        worst = jnp.where(conv, jnp.max(g, axis=1), FAILED_G)
        score = jnp.where(feasible, rows[:, c_coe], 1.0e6 + worst)
        k = lax.stop_gradient(jnp.argmin(score))
        return rows[k]

    def vmapped(x_flat, theta, starts, first_env):
        if te_grid is None:
            return jax.vmap(
                lambda th, st: at_point(x_flat, th, st, None, first_env), in_axes=(0, 0)
            )(theta, starts)
        # One flat batch over (sample, T_e) pairs -- a vmap inside a vmap trips the
        # solver's error branch at lowering -- then the recourse picks per sample.
        k, n = len(te_grid), starts.shape[0]
        theta_k = jax.tree_util.tree_map(lambda a: jnp.repeat(a, k, axis=0), theta)
        starts_k = jnp.repeat(starts, k, axis=0)
        te_k = jnp.tile(jnp.asarray(te_grid), n)
        rows = jax.vmap(
            lambda th, st, te: at_point(x_flat, th, st, te, first_env), in_axes=(0, 0, 0)
        )(theta_k, starts_k, te_k)
        return jax.vmap(choose)(rows.reshape(n, k, -1))

    def run_batch(x_flat, theta, starts, chunks: int | None = chunks):
        """[N + 1, columns]: every row's closed MDA at design `x_flat`; with `chunks`
        the N samples go through `lax.map` in that many checkpointed pieces and the
        nominal row on its own.

        Raises
        ------
        ValueError
            If N is not a multiple of `chunks`.
        """
        first_env = first(x_flat) if hoist else None
        if not chunks:
            return vmapped(x_flat, theta, starts, first_env)
        n = starts.shape[0] - 1
        if n % chunks:
            raise ValueError(f"N = {n} is not a multiple of chunks = {chunks}")
        size = n // chunks
        theta_s = jax.tree_util.tree_map(
            lambda a: a[:-1].reshape(chunks, size, *a.shape[1:]), theta
        )
        theta_n = jax.tree_util.tree_map(operator.itemgetter(-1), theta)
        starts_s = starts[:-1].reshape(chunks, size, -1)

        @jax.checkpoint
        def body(chunk):
            th, st = chunk
            return vmapped(x_flat, th, st, first_env)

        ys = lax.map(body, (theta_s, starts_s)).reshape(n, -1)
        last = vmapped(
            x_flat,
            jax.tree_util.tree_map(operator.itemgetter(None), theta_n),
            starts[-1:],
            first_env,
        )
        return jnp.concatenate([ys, last], axis=0)

    def measures(y_all):
        """Every objective measure and the constraints' CVaRs from the rows."""
        y, y_nominal = y_all[:-1], y_all[-1]
        w = jnp.where(converged(y), 1.0, 0.0)
        sum_w = jnp.sum(w)
        count = jnp.asarray(y.shape[0], float)
        failed = 1.0 - sum_w / count
        coe, net = y[:, c_coe], y[:, c_net]
        kwh = jnp.maximum(net * y[:, c_avail], TINY)  # proportional to kwhpy
        annual = jnp.where(w > 0, coe, 0.0) * kwh  # finite: coe = 1e9 ann / kwhpy
        levelised = jnp.sum(w * annual) / jnp.maximum(jnp.sum(w * kwh), 1.0e-10)
        rated = jnp.sum(w * annual) / jnp.maximum(sum_w, 1.0) / rated_kwh
        positive = (w > 0) & (net > 0)
        n_pos = jnp.sum(jnp.where(positive, 1.0, 0.0))
        mean = jnp.where(n_pos > 0, _masked_mean(coe, positive), FAILED_F)
        median = _median(coe, w > 0)
        nominal_ok = converged(y_nominal[None, :])[0]
        nominal = jnp.where(nominal_ok, y_nominal[c_coe], FAILED_F)
        g = jnp.where(w[:, None] > 0, y[:, c_g], FAILED_G)  # [N, n_g]
        g_cvar = jnp.mean(lax.top_k(g.T, m)[0], axis=1)
        if i16 is not None and m16 != m:
            g_cvar = g_cvar.at[i16].set(jnp.mean(lax.top_k(g[:, i16], m16)[0]))
        value = {
            "mean": mean,
            "median": median,
            "levelised": levelised,
            "nominal": nominal,
            "rated": rated,
        }[objective]
        f = jnp.where(sum_w > 0, value, FAILED_F) + FAILED_F * failed
        u = jnp.where(w[:, None] > 0, y[:, c_u], jnp.nan)
        conv = w > 0
        diagnostics = {
            "steps_mean": jnp.mean(y[:, c_steps]),
            "steps_max": jnp.max(y[:, c_steps]),
            "coe_nominal": y_nominal[c_coe],
            "net_nominal_mw": y_nominal[c_net],
            "levelised": levelised,
            "rated": rated,
            "mean_coe_where_converged_and_positive_net": mean,
            "median_coe_where_converged": median,
            "mean_concost": _masked_mean(y[:, c_concost], conv),
            "mean_net_mw": _masked_mean(net, conv),
            "p_net_nonpositive": jnp.mean(jnp.where(net <= 0, 1.0, 0.0)),
            "p_violated": jnp.mean(jnp.where(y[:, c_g] > 0, 1.0, 0.0), axis=0),
            "mean_g": jnp.mean(y[:, c_g], axis=0),
        }
        return f, g_cvar, failed, u, diagnostics

    def statistics_full(x_flat, theta, starts):
        y_all = run_batch(x_flat, theta, starts)
        f, g_cvar, failed, u, diagnostics = measures(y_all)
        u_all = jnp.concatenate([u, y_all[-1:, c_u]], axis=0)
        u_next = jnp.where(jnp.isnan(u_all), starts, u_all)
        return f, g_cvar, failed, lax.stop_gradient(u_next), diagnostics

    def statistics(x_flat, theta, starts):
        """`(f, g_cvar, failed_fraction)` at design `x_flat`."""
        f, g_cvar, failed, _u, _d = statistics_full(x_flat, theta, starts)
        return f, g_cvar, failed

    def statistics_vec(x_flat, theta, starts):
        f, g_cvar, failed = statistics(x_flat, theta, starts)
        return jnp.concatenate([f[None], g_cvar, failed[None]])

    def statistics_chunked_vec(x_flat, theta, starts, chunks: int):
        f, g_cvar, failed, _u, _d = measures(run_batch(x_flat, theta, starts, chunks))
        return jnp.concatenate([f[None], g_cvar, failed[None]])

    def value_jac_starts(x_flat, theta, starts):
        """One compiled call for the outer driver: the statistics vector, its
        Jacobian (forward or reverse, `jac`; one trace of the model, primal reused),
        the next starts, diagnostics.
        """

        def twice(x):
            f, g_cvar, failed, u_next, diagnostics = statistics_full(x, theta, starts)
            out = jnp.concatenate([f[None], g_cvar, failed[None]])
            return out, (out, u_next, diagnostics)

        differentiate = jax.jacfwd if jac == "fwd" else jax.jacrev
        jacobian, (out, u_next, diagnostics) = differentiate(twice, has_aux=True)(x_flat)
        return out, jacobian, u_next, diagnostics

    return {
        "run_batch": run_batch,
        "first": first,
        "statistics": statistics,
        "statistics_vec": statistics_vec,
        "statistics_full": statistics_full,
        "statistics_chunked_vec": statistics_chunked_vec,
        "value_jac_starts": value_jac_starts,
        "layout": layout,
        "hoist": hoist,
        "n_g": n_g,
    }


def _masked_mean(v, mask):
    w = jnp.where(mask, 1.0, 0.0)
    return jnp.sum(w * jnp.where(mask, v, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)


def _median(v, mask):
    """The sample median of `v` over `mask` (the two middle values averaged): a sort,
    so differentiable wherever the order is strict.
    """
    s = jnp.sort(jnp.where(mask, v, jnp.inf))
    k = jnp.sum(jnp.where(mask, 1, 0))
    lo = s[jnp.maximum((k - 1) // 2, 0)]
    hi = s[jnp.maximum(k // 2, 0)]
    return jnp.where(k > 0, 0.5 * (lo + hi), FAILED_F)


def jitted(fns: dict, name: str):
    """`jax.jit(fns[name])`, built once and kept in `fns` under `_<name>_jit`."""
    key = f"_{name}_jit"
    if key not in fns:
        fns[key] = jax.jit(fns[name])
    return fns[key]


def sensitivity(model: TwoStage) -> np.ndarray:
    """D closing unknowns / d belief coordinate at the nominal, [unknowns, k], by one
    `jax.jacfwd` of the un-batched closed MDA through the root find's `custom_root`
    -- the linear predictor of the roots `uq.Model.predicted` used. A dummy column
    is zero.
    """
    run = jax.jit(model.whole.run)
    x0 = jnp.asarray(beliefs_.nominal_coordinates(model.beliefs, model.nominal))
    point = dict(model.point.items())

    def roots(x):
        values = dict(point)
        for j, belief in enumerate(model.beliefs):
            var = model.var_of.get(belief.path)
            if var is None:
                continue
            values[var] = beliefs_.value(belief, x[j], model.nominal[belief.path])
        out = run(PathMap(values))
        return jnp.stack([out[u] for u in model.unknowns])

    return np.asarray(jax.jacfwd(roots)(x0))


# ---------------------------------------------------------------- the program


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """What one fused call at one design returned."""

    x: np.ndarray
    values: np.ndarray
    """`[f, *g_cvar, failed]`."""
    jacobian: np.ndarray
    """`[1 + n_g + 1, n_x]`, unscaled."""
    next_starts: np.ndarray
    """[N + 1, unknowns]: every sample's converged closing unknowns, the start where
    it did not converge."""
    diagnostics: dict
    seconds: float


class Program:
    """The batched program bound to one sample set: `theta` fixed, `starts` mutable
    (the warm starts), a memo of `Evaluation`s per design point, and the trace of
    every call. What the `Statistics` node calls and the `BoxedSlsqpDriver` reads its
    Jacobian and warm starts from.
    """

    def __init__(self, fns: dict, theta: PathMap, starts0, kept: int = 3):
        self.fns = fns
        self.theta = theta
        self.starts = jnp.asarray(starts0)
        self.starts0 = np.asarray(starts0)
        self.kept = kept
        self.memo: dict = {}
        self.adopted: dict = {}
        self.trace: list = []
        self.calls = 0

    @staticmethod
    def _key(x) -> bytes:
        return np.asarray(x, dtype=float).tobytes()

    def at(self, x) -> Evaluation:
        """The fused call at `x` from the current starts, memoised per point."""
        key = self._key(x)
        found = self.memo.get(key)
        if found is not None:
            return found
        began = time.perf_counter()
        out, jacobian, u_next, diagnostics = jax.block_until_ready(
            jitted(self.fns, "value_jac_starts")(
                jnp.asarray(x, dtype=float), self.theta, self.starts
            )
        )
        seconds = time.perf_counter() - began
        self.calls += 1
        found = Evaluation(
            x=np.asarray(x, dtype=float).copy(),
            values=np.asarray(out, dtype=float),
            jacobian=np.asarray(jacobian, dtype=float),
            next_starts=u_next,
            diagnostics={k: np.asarray(v).tolist() for k, v in diagnostics.items()},
            seconds=seconds,
        )
        while len(self.memo) >= self.kept:
            del self.memo[next(iter(self.memo))]
        self.memo[key] = found
        n_g = self.fns["n_g"]
        self.trace.append({
            "call": len(self.trace) + 1,
            "f": float(found.values[0]),
            "max_g_cvar": float(found.values[1 : 1 + n_g].max()) if n_g else 0.0,
            "failed_fraction": float(found.values[-1]),
            "x": found.x.tolist(),
            "steps_mean": found.diagnostics["steps_mean"],
            "steps_max": found.diagnostics["steps_max"],
            "model_s": seconds,
        })
        return found

    def value(self, x) -> np.ndarray:
        """`[f, *g_cvar, failed]` at `x`."""
        return self.at(x).values

    def jacobian(self, x) -> np.ndarray:
        """Its Jacobian at `x`, `[1 + n_g + 1, n_x]` -- `BoxedSlsqpDriver.jacobian`."""
        return self.at(x).jacobian

    def adopt(self, x) -> None:
        """Warm-start every sample from its root at `x` -- the incumbent's answer
        (`BoxedSlsqpDriver.warm_start`). A point adopted before is restored from what
        it answered then, even if the memo has moved on.
        """
        key = self._key(x)
        starts = self.adopted.get(key)
        if starts is None:
            starts = self.adopted[key] = self.at(x).next_starts
        self.starts = jnp.asarray(starts)

    def traceable(self, x):
        """The statistics vector at `x` under a trace (a caller's `jacfwd` or
        `eval_shape` through the node) -- the plain program, from the current starts.
        """
        return self.fns["statistics_vec"](x, self.theta, self.starts)

    def rows(self, x, starts=None, theta=None) -> np.ndarray:
        """`run_batch` at `x`: [N + 1, columns], from `starts` (default the current)."""
        return np.asarray(
            jax.block_until_ready(
                jitted(self.fns, "run_batch")(
                    jnp.asarray(x, dtype=float),
                    self.theta if theta is None else theta,
                    self.starts if starts is None else jnp.asarray(starts),
                )
            )
        )

    def forward(self, x) -> np.ndarray:
        """The statistics vector at `x`, the plain jitted program, no memo."""
        return np.asarray(
            jax.block_until_ready(
                jitted(self.fns, "statistics_vec")(
                    jnp.asarray(x, dtype=float), self.theta, self.starts
                )
            )
        )


def _traced(values) -> bool:
    return any(isinstance(v, jax.core.Tracer) for v in values)


@dataclasses.dataclass(frozen=True)
class Statistics:
    """The body of the statistics node: `(*design) -> (f, *cvar, failed - eps)`.

    Outside a trace (the driver's own calls) through the program's memo; under one (a
    caller's `jacfwd`, or `eval_shape` asking the shapes) through the plain program.
    Compares by the program's identity, so one assembly is one jit key.
    """

    program: Program
    eps: float = EPS_FAILED

    def __call__(self, *design):
        """The conditions at `design`, one scalar each."""
        x = jnp.stack([jnp.asarray(d, dtype=jnp.float64).reshape(()) for d in design])
        v = (
            self.program.traceable(x)
            if _traced(design)
            else jnp.asarray(self.program.value(np.asarray(x)))
        )
        return (v[0], *[v[i] for i in range(1, v.shape[0] - 1)], v[-1] - self.eps)


@dataclasses.dataclass(frozen=True)
class Outer:
    """The outer problem as a graph: the statistics node and the `Optimise` over the
    design, driven by a `BoxedSlsqpDriver`, and the program the node calls.
    """

    model: TwoStage
    program: Program
    graph: Graph
    """Undriven: the two nodes."""
    schedule: Schedule
    """`Schedule(AnswerableGraph(the graph with the driver assigned))`."""
    driver: BoxedSlsqpDriver
    objective: VarPath
    cvars: tuple[VarPath, ...]
    failed: VarPath

    @property
    def design(self) -> tuple[VarPath, ...]:
        """The first-stage places the `Optimise` owns, in `ixc` order."""
        return self.model.design

    @property
    def starts(self) -> dict:
        """`{start port: design place}` -- what an env hands the driver."""
        return guess_sources(self.schedule.answerable.graph)

    @property
    def eps(self) -> float:
        """The failed-fraction bound the statistics node subtracts."""
        return self.graph[STATISTICS].fn.eps

    @staticmethod
    def verdict(out, kind) -> object:
        """What the driver said, out of a run's env."""
        return out.get(kind.name_for(OPTIMISE))


class _Default:
    """The sentinel `outer` reads as "the program's own": distinct from `None`, which
    is a choice (no warm start; `jacfwd` through the node).
    """

    def __repr__(self) -> str:
        return "DEFAULT"


DEFAULT = _Default()


def outer(
    model: TwoStage,
    fns: dict | None = None,
    *,
    warm_start=DEFAULT,
    jacobian=DEFAULT,
    **driver_kwargs,
) -> Outer:
    """The outer problem stated as a graph over `model`, the program built from `fns`
    (`make(model)` by default) on `model`'s sample set. `driver_kwargs` go to
    `BoxedSlsqpDriver` (`delta`, `max_iter`, `gtol`, `callback`, ...), bar `eps`, the
    failed-fraction bound (`EPS_FAILED`), which is the node's; the driver's `bounds`
    and `n_inequality` are this function's to set.

    `warm_start` and `jacobian` are the driver's two hooks, `DEFAULT` being the
    program's own: `program.adopt` (every sample started from its root at the
    incumbent) and `program.jacobian` (the rows of the fused call's memo). `None`
    is expressible and means what it means to the driver -- no warm start (every
    call from the starts the program holds), `jax.jacfwd` through the statistics
    node (the plain program under a trace, `Program.traceable`) -- and any other
    callable is handed to the driver as it is.
    """
    fns = make(model) if fns is None else fns
    program = Program(fns, model.theta, model.starts0)
    if warm_start is DEFAULT:
        warm_start = program.adopt
    if jacobian is DEFAULT:
        jacobian = program.jacobian
    objective = condition_path("objective")
    cvars = tuple(cvar_path(c) for c in model.constraints)
    failed = condition_path("failed")
    node = ImplementedFunction(
        reads=model.design,
        owns=(objective, *cvars, failed),
        fn=Statistics(program, driver_kwargs.pop("eps", EPS_FAILED)),
    )
    problem = Optimise(
        objective=objective, unknowns=model.design, inequalities=(*cvars, failed)
    )
    graph = Graph.of({STATISTICS: node, OPTIMISE: problem})
    driver = BoxedSlsqpDriver(
        n_inequality=len(cvars) + 1,
        bounds=tuple(
            (v, float(lo), float(hi))
            for v, lo, hi in zip(model.design, model.lower, model.upper, strict=True)
        ),
        jacobian=jacobian,
        warm_start=warm_start,
        **driver_kwargs,
    )
    driven = assign_drivers(graph, {OPTIMISE: driver})
    return Outer(
        model=model,
        program=program,
        graph=graph,
        schedule=Schedule(AnswerableGraph(driven)),
        driver=driver,
        objective=objective,
        cvars=cvars,
        failed=failed,
    )


def solve(built: Outer, x0=None, starts=None) -> tuple[np.ndarray, dict, float]:
    """Run the outer schedule from `x0` (default the deterministic start) with every
    sample started from `starts` (default the nominal roots): the design at the
    answer, the run's env (the statistics at the answer, the driver's `Steps`,
    `Converged`, `Status`), and the wall time. Walked with the driver eager
    (`whole=False`), as SAND's is.
    """
    model, program = built.model, built.program
    x0 = model.x0 if x0 is None else np.asarray(x0, dtype=float)
    program.starts = jnp.asarray(model.starts0 if starts is None else starts)
    env = {
        port: jnp.asarray(float(x0[model.design.index(place)]))
        for port, place in built.starts.items()
    }
    began = time.perf_counter()
    out = run_schedule(built.schedule, env, whole=False)
    elapsed = time.perf_counter() - began
    x = np.array([float(np.asarray(out[v])) for v in model.design])
    return x, out, elapsed


def report(built: Outer, out) -> dict:
    """The solve in one dict: the design, `f`, every CVaR, the failed fraction, the
    driver's verdict, and the trace's best feasible call.
    """
    model = built.model
    return {
        "x": dict(
            zip(
                [v.spelling for v in model.design],
                [float(np.asarray(out[v])) for v in model.design],
                strict=True,
            )
        ),
        "f": float(np.asarray(out[built.objective])),
        "g_cvar": dict(
            zip(
                model.names,
                [float(np.asarray(out[c])) for c in built.cvars],
                strict=True,
            )
        ),
        "failed_fraction": float(np.asarray(out[built.failed])) + built.eps,
        "steps": int(np.asarray(built.verdict(out, Steps))),
        "converged": bool(np.asarray(built.verdict(out, Converged))),
        "status": int(np.asarray(built.verdict(out, Status))),
        "model_calls": built.program.calls,
        "best_feasible": best_feasible(
            built.program.trace, built.eps, built.driver.gtol
        ),
    }


def best_feasible(
    trace: list, eps: float = EPS_FAILED, tol: float = 1e-3
) -> dict | None:
    """The cheapest model call at which every CVaR constraint held (to `tol`) and the
    failed fraction was within `eps` -- SLSQP's last iterate need not be its best.
    """
    feasible = [
        t for t in trace if t["max_g_cvar"] <= tol and t["failed_fraction"] <= eps
    ]
    return min(feasible, key=operator.itemgetter("f")) if feasible else None


# ---------------------------------------------------------------- evidence


def valid_rows(layout: dict, y: np.ndarray) -> np.ndarray:
    """A row counts if the closing problem converged, every other driver in the
    recourse schedule did too (`layout["c_verdicts"]`), and every measured column
    is finite (the extra columns are not measured, so not asked to be).
    """
    y = np.asarray(y, dtype=float)
    finite = np.all(np.isfinite(y[:, : layout["c_steps"]]), axis=1)
    v0, v1 = layout["c_verdicts"]
    verdicts = np.all(y[:, v0:v1] > 0.5, axis=1)
    return (y[:, layout["c_conv"]] > 0.5) & verdicts & finite


def extra_columns(layout: dict, y: np.ndarray) -> dict[str, np.ndarray]:
    """The extra columns of `y` by name: `{spelling: [rows]}`, empty when none were
    asked for.
    """
    y = np.asarray(y, dtype=float)
    e0, _e1 = layout["c_extra"]
    return {name: y[:, e0 + j] for j, name in enumerate(layout["extra"])}


def evaluate(
    fns: dict,
    x: np.ndarray,
    theta: PathMap,
    starts0: np.ndarray,
    x_from: np.ndarray | None = None,
    steps: int = 8,
    eps: float = EPS_FAILED,
    max_calls: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The batched rows at design `x`: once from the nominal root (`y_cold`), and once
    warm (`y_warm`, what the optimiser saw). Warm means from the converged roots of a
    **continuation** from `x_from` (the deterministic design, where the nominal root
    is a start every sample converges from) in `steps` pieces, each piece's Newton
    started from the previous piece's roots and halved while it loses more than `eps`
    of the samples -- the chain of warm starts the optimiser itself walked. Without
    `x_from`, from the cold pass's own converged roots. A last pass from the converged
    roots either way. Returns `(y_cold, y_warm, timing)`.
    """
    run = jitted(fns, "run_batch")
    layout = fns["layout"]
    u0, u1 = layout["c_u"]
    x = np.asarray(x, float)
    calls = {"n": 0}

    def call(xi, starts):
        began = time.perf_counter()
        y = np.asarray(
            jax.block_until_ready(run(jnp.asarray(xi), theta, jnp.asarray(starts)))
        )
        calls["n"] += 1
        return y, valid_rows(layout, y), time.perf_counter() - began

    y_cold, conv, cold_s = call(x, starts0)
    starts = np.where(conv[:, None], y_cold[:, u0:u1], starts0)
    path_failed = []
    if x_from is not None and not np.allclose(x_from, x):
        x_from = np.asarray(x_from, float)
        y0, conv0, _ = call(x_from, starts0)
        starts = np.where(conv0[:, None], y0[:, u0:u1], starts0)
        base_failed = 1.0 - conv0.mean()
        x_cur, fraction, halvings = x_from.copy(), 1.0 / steps, 0
        while not np.allclose(x_cur, x) and calls["n"] < max_calls:
            x_try = x if fraction >= 1.0 - 1e-12 else x_cur + fraction * (x - x_from)
            if np.linalg.norm(x_try - x_from) >= np.linalg.norm(x - x_from) - 1e-12:
                x_try = x
            y_i, conv_i, _ = call(x_try, starts)
            failed_i = 1.0 - conv_i.mean()
            if failed_i > base_failed + eps and halvings < 6:
                fraction, halvings = fraction / 2, halvings + 1
                continue
            path_failed.append(float(failed_i))
            x_cur = x_try
            starts = np.where(conv_i[:, None], y_i[:, u0:u1], starts)
    y_warm, conv, warm_s = call(x, starts)
    starts = np.where(conv[:, None], y_warm[:, u0:u1], starts)
    y_warm, conv, warm_s = call(x, starts)
    return (
        y_cold,
        y_warm,
        {
            "cold_s": cold_s,
            "warm_s": warm_s,
            "calls": calls["n"],
            "path_failed": path_failed,
            "failed_cold": float(1.0 - valid_rows(layout, y_cold).mean()),
        },
    )


def _percentiles(v, mask) -> dict:
    v = v[mask]
    if v.size == 0:
        return {
            "p5": None,
            "p50": None,
            "p95": None,
            "mean": None,
            "median": None,
            "count": 0,
        }
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return {
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "count": int(v.size),
    }


def summarise(
    model: TwoStage, layout: dict, y_all: np.ndarray, alpha_extra: float = 0.9
) -> dict:
    """Everything the evidence reads off one batched call at one design: the nominal
    row, the objective measures, failure and violation fractions, per-sample
    feasibility (converged, net > 0, every g <= `G_TOL`) and the coe statistics over
    the feasible samples, percentiles, and the constraints' CVaRs (with a failed
    sample at `FAILED_G`, as the optimiser sees them).
    """
    y_all = np.asarray(y_all, dtype=float)
    y, y_nominal = y_all[:-1], y_all[-1]
    c_coe, c_net, c_avail = layout["c_coe"], layout["c_net"], layout["c_avail"]
    g0, g1 = layout["c_g"]
    c_steps = layout["c_steps"]
    conv = valid_rows(layout, y)
    coe, net, g = y[:, c_coe], y[:, c_net], y[:, g0:g1]
    n = y.shape[0]
    positive = conv & (net > 0)
    violates = np.where(conv[:, None], g > G_TOL, True)
    feasible = positive & ~np.any(violates, axis=1)
    kwh = np.maximum(net * y[:, c_avail], layout["tiny"])
    annual = np.where(conv, coe, 0.0) * kwh
    g_masked = np.where(conv[:, None], g, FAILED_G)
    names = list(model.names)

    def cvar(alpha):
        m = max(1, math.ceil((1.0 - alpha) * n))
        worst = -np.sort(-g_masked, axis=0)[:m]
        return dict(zip(names, np.mean(worst, axis=0).tolist(), strict=True))

    nominal_ok = bool(valid_rows(layout, y_nominal[None, :])[0])
    extra = extra_columns(layout, y)
    extra_nominal = extra_columns(layout, y_nominal[None, :])
    return {
        "n": int(n),
        "nominal": {
            "converged": nominal_ok,
            "coe": float(y_nominal[c_coe]),
            "net_mw": float(y_nominal[c_net]),
            "g": dict(zip(names, y_nominal[g0:g1].tolist(), strict=True)),
            "feasible": bool(
                nominal_ok and y_nominal[c_net] > 0 and np.all(y_nominal[g0:g1] <= G_TOL)
            ),
        },
        "failed_fraction": float(1.0 - conv.mean()),
        "p_net_nonpositive": float(np.mean(net <= 0)),
        "feasible_fraction": float(feasible.mean()),
        "any_violation_fraction": float(np.mean(np.any(violates, axis=1))),
        "any_violation_fraction_among_converged": (
            float(np.mean(np.any(g[conv] > G_TOL, axis=1))) if conv.any() else None
        ),
        "p_violated": dict(zip(names, violates.mean(axis=0).tolist(), strict=True)),
        "p_violated_among_converged": (
            dict(zip(names, (g[conv] > G_TOL).mean(axis=0).tolist(), strict=True))
            if conv.any()
            else None
        ),
        "mean_g_among_converged": (
            dict(zip(names, g[conv].mean(axis=0).tolist(), strict=True))
            if conv.any()
            else None
        ),
        "cvar": {
            f"{model.alpha:g}": cvar(model.alpha),
            f"{alpha_extra:g}": cvar(alpha_extra),
        },
        "levelised": (
            float(np.sum(annual[conv]) / max(np.sum(kwh[conv]), 1e-10))
            if conv.any()
            else None
        ),
        "rated": (
            float(np.mean(annual[conv]) / (model.rated_mw * float(y_nominal[c_avail])))
            if conv.any()
            else None
        ),
        "mean_coe_where_converged_and_positive_net": (
            float(np.mean(coe[positive])) if positive.any() else None
        ),
        "median_coe_where_converged": float(np.median(coe[conv]))
        if conv.any()
        else None,
        "coe_where_converged_and_positive_net": _percentiles(coe, positive),
        "coe_where_feasible": _percentiles(coe, feasible),
        "net_mw_where_converged": _percentiles(net, conv),
        "steps_mean": float(np.mean(y[:, c_steps])),
        "steps_max": float(np.max(y[:, c_steps])),
        "extra": {
            name: {
                "nominal": float(extra_nominal[name][0]),
                **_percentiles(values, conv & np.isfinite(values)),
            }
            for name, values in extra.items()
        },
    }


def per_sample(layout: dict, y_all: np.ndarray) -> dict:
    """The columns a histogram draws, per sample: coe, net, g, converged, feasible,
    and `extra` (`{spelling: [N]}`, the columns `two_stage(extra_columns=...)` asked
    for, as they came).
    """
    y = np.asarray(y_all, dtype=float)[:-1]
    g0, g1 = layout["c_g"]
    conv = valid_rows(layout, y)
    g = y[:, g0:g1]
    feasible = conv & (y[:, layout["c_net"]] > 0) & np.all(g <= G_TOL, axis=1)
    return {
        "coe": y[:, layout["c_coe"]],
        "net": y[:, layout["c_net"]],
        "g": g,
        "converged": conv,
        "feasible": feasible,
        "extra": extra_columns(layout, y),
    }


def _active(x, lo, hi, tol) -> str | None:
    """Which bound `x` sits on, within `tol` of the range, or `None`."""
    span = hi - lo
    if hi - x < tol * span:
        return "upper"
    if x - lo < tol * span:
        return "lower"
    return None


def design_table(
    model: TwoStage, x_det: np.ndarray, x_rob: np.ndarray, tol: float = 1e-3
) -> list[dict]:
    """Per `ixc` entry: bounds, both designs, the delta, and which bound is active
    (within `tol` of the range).
    """
    rows = []
    for i, v in enumerate(model.design):
        lo, hi = float(model.lower[i]), float(model.upper[i])
        rows.append({
            "ixc": int(model.ixc[i]),
            "place": v.spelling,
            "lower": lo,
            "upper": hi,
            "deterministic": float(x_det[i]),
            "robust": float(x_rob[i]),
            "delta": float(x_rob[i] - x_det[i]),
            "delta_relative": float(x_rob[i] / x_det[i] - 1.0),
            "active_bound": _active(x_rob[i], lo, hi, tol),
            "deterministic_active_bound": _active(x_det[i], lo, hi, tol),
        })
    return rows


__all__ = [
    "ALPHA",
    "BUILD_TABLE",
    "DEFAULT",
    "EPS_FAILED",
    "FAILED_F",
    "FAILED_G",
    "G_TOL",
    "JAC_MODES",
    "OBJECTIVES",
    "OPTIMISE",
    "STATISTICS",
    "TINY",
    "Evaluation",
    "Outer",
    "Program",
    "RecourseBound",
    "Statistics",
    "TwoStage",
    "best_feasible",
    "condition_path",
    "cvar_path",
    "design_table",
    "driven_problems",
    "evaluate",
    "extra_columns",
    "fresh_sample",
    "jitted",
    "label_of",
    "make",
    "other_verdicts",
    "outer",
    "per_sample",
    "report",
    "resolve_columns",
    "sample",
    "sensitivity",
    "solve",
    "summarise",
    "two_stage",
    "valid_rows",
]
