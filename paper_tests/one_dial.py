"""One dial in, one dial out.

PROCESS decides the confinement factor `hfact`: it is an iteration variable, and the
optimiser picks whatever value makes the power balance close. Nobody picks it in
reality -- it is what the plasma does. This script turns that one quantity from
something we choose into something the world hands us, and reports what the model then
demands in exchange.

The demand is not a matter of taste. `hfact` leaves the unknowns, `c2` (the power
balance) stays, and the graph has one equation more than unknowns: it refuses, naming
the cycle that carries a condition with nothing to drive it. Something must absorb the
equation, and *which* variable is a modelling decision -- on an ignited machine the
density, on a driven one the heating power. The graph does not make that choice; it
makes the choice unavoidable, records it, and prices it (the cycle each candidate
opens).

Then the build is frozen -- the machine exists, its coils are wound -- and `hfact` is
swept. Two curves:

  redesign   PROCESS's own question at every point: re-optimise the whole machine.
             Smooth, always feasible, a *different machine* at each value. PROCESS
             produces this itself (`hfact` is scan variable 4, `process/core/scan.py`).
  built      the same sweep with the design held at PROCESS's answer. The absorbing
             variable moves, and somewhere the curve hits a wall.

The gap between them is the point: same model, two questions, one line of difference in
the problem statement.

Every step below is one named procedure taking a `Plan` and returning a `Plan`, and
records the graph operations it applied. `report(plan)` prints the recipe. The file's
problem is `problem.problem_model`; the closed machine is `study.machine_model` with the
root find as the top of `cx.mdf`; every run is `study.solve`.

    PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
    export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
    $PY paper_tests/one_dial.py --machine stellarator_helias      # ignited: density absorbs
    $PY paper_tests/one_dial.py --machine low_aspect_ratio_DEMO   # driven: heating absorbs
    $PY paper_tests/one_dial.py                                   # both, the showcase pair

The two reach the same place by different routes, and `report(plan)` shows which:
`stellarator_helias` chooses `hfact` (`ixc = 10`, converging to 1.056 inside its bound),
so giving it away is what over-determines the problem; `low_aspect_ratio_DEMO` fixes
`hfact` at 1.1 in the file, so step 1 is a no-op there and the count breaks one step
later, when the build is frozen. It is freezing the design that breaks the equation
count a systems code relies on -- the dial is only what makes it visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import cottax.interfaces as cx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cottax import ExecutableGraph, Function, Minted, NodePath, OrderedMap, VarPath  # noqa: E402
from cottax.core.maths.executable import RequirementUnanswered  # noqa: E402
from cottax.core.frozen import violations  # noqa: E402
from cottax.core.maths.plan import Cut, Determine, Insert  # noqa: E402
from cottax.core.maths.plan import Plan as Ops  # noqa: E402
from cottax.mdao_architectures import CloseFunctionalCycles  # noqa: E402
from jax.tree_util import GetAttrKey  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functional_process import configurations  # noqa: E402
from functional_process.configurations import Optimisation  # noqa: E402
from functional_process.configurations.kinds import CLAIMED_BUILD_OUTPUTS  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.architectures.contracts import (  # noqa: E402
    BracketedRootFinding,
    Converged,
    PortRootFinding,
    Start,
    Steps,
)
from functional_process.cottax.architectures.contracts import Bracket as BracketRole  # noqa: E402
from functional_process.cottax.architectures.drivers import (  # noqa: E402
    BracketedRootDriver,
    SafeguardedNewtonDriver,
)
from functional_process.cottax.input.native import state_of  # noqa: E402
from functional_process.cottax.models.constraints import held  # noqa: E402
from functional_process.cottax.problem import problem_model  # noqa: E402
from paper_tests.common import OUT, process_reference  # noqa: E402

# ---------------------------------------------------------------- the one dial

DIAL = ".physics.hfact"
"""The quantity that stops being ours to choose."""
BALANCE = ".constraints.c2"
"""The power balance: the equation `hfact` was closing."""
DENSITY = ".physics.nd_plasma_electrons_vol_avg"
"""What absorbs it on an ignited machine."""
HEATING = ".current_drive.p_hcd_primary_extra_heat_mw"
"""What absorbs it on a driven one -- a plain input in every PROCESS file."""
HEATING_BRACKET = (-25.0, 1000.0)
"""The bracket the heating closure is answered in (MW), by `BracketedRootDriver`. A Newton
scales its step by the unknown it moves, and two of these files install no heating at
all, so from zero it cannot move; a bracketed root converges from any start. The bottom
is negative because the root is then *at* zero and a bracket has to contain it -- a
negative answer is a world whose balance closes with less heating than none, which is a
reading, not a machine."""

CLOSURE: dict[str, str] = {
    "stellarator_helias": "density",  # ignited (`i_plasma_ignited = 1`), no burn heating
    "helias_5b": "density",  # ignited
    "large_tokamak_nof": "heating",  # driven: 75 MW of ECRH during the burn
    "large_tokamak_eval": "heating",  # driven: 75 MW
    "low_aspect_ratio_DEMO": "heating",  # driven: 10 MW
    "spherical_tokamak_eval": "heating",  # driven by current drive alone
    "st_regression": "heating",  # driven by current drive alone
}
"""Which variable absorbs the power balance, per configuration. **The one modelling
decision this script makes**, and it is made here, in a table a reader can argue with,
rather than inside a solver."""

SHOWCASE: tuple[str, str] = ("stellarator_helias", "low_aspect_ratio_DEMO")
"""The two machines the paper shows -- the two genuinely optimised configurations, and
the same pair as the rest of the section. They reach the same place by different routes,
which is the point of showing both: the stellarator *chooses* `hfact` (`ixc = 10` is
active and converges to 1.056, inside `boundu = 1.2`), so giving it away is what breaks
the equation count; `low_aspect_ratio_DEMO` fixes it in the file at 1.1, so there is
nothing to reclassify and the count breaks one step later, when the build is frozen.

The others are out of the showcase and `CLOSURE` still names their closure so they can
be run: `large_tokamak_eval` and `spherical_tokamak_eval` are evaluation files, not
optimisations; `large_tokamak_nof` is doubly degenerate (its figure of merit is
`0.2 x rmajor` with `rmajor` boxed at its lower bound 8.0, and `hfact` converges to
`boundu(10) = 1.2`), so its design already sits on the wall and there is no curve to
draw -- measured, before it was dropped: margin 2e-6, `c68` holding by 1.3e-5, and the
heating power landing on exactly PROCESS's own 75 MW at the nominal."""

FALLS: tuple[float, ...] = (
    0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30
)
"""How far below PROCESS's own `hfact` the dial is turned, as a fraction of it: dense
near the nominal, because a cost optimum sits *on* its constraints and the wall is
close."""

NET = ".heat_transport.p_plant_electric_net_mw"
"""What the plant delivers: not a condition, the outcome the sweep is about."""

REFUSAL = "the graph refuses: "
"""How a step's `note` marks the refusal it drew, so `report` can print it on a line of
its own. **Which step draws it differs between the two machines, and that difference is
the point**: where `hfact` is an active `ixc` it is giving it away that over-determines
the problem, and where the file fixes `hfact` there is nothing to reclassify and the
count breaks only when the build is frozen."""

HELD = 0.0
"""A condition of this port is a normalised residual, satisfied at `<= 0` -- the same
number for an equality and an inequality (`sand.condition_nodes`). A closed equality
sits at `0` by construction."""

SLACK = 1e-9
"""Below `-SLACK` a condition has room to spend; at or above it, it is tight at the
nominal and cannot *break* -- it was already at its limit there."""


@dataclass(frozen=True)
class Step:
    """One procedure's account of itself: what it did to the graph, and why."""

    name: str
    why: str
    ops: tuple[str, ...]
    nodes: int
    note: str = ""


@dataclass(frozen=True)
class Plan:
    """A design question: the model, plus who decides what.

    `chosen` are the quantities we pick (PROCESS's `ixc`), `given` the ones handed to
    us, `conditions` the relations that must hold. A procedure moves a quantity between
    those and says so in `steps`.
    """

    machine: str
    graph: object
    values: dict
    chosen: tuple[str, ...]
    conditions: tuple[str, ...]
    steps: tuple[Step, ...] = ()

    def having(self, step: Step, **changes) -> Plan:
        """This plan with `step` recorded and `changes` applied."""
        return replace(self, steps=(*self.steps, step), **changes)


# ---------------------------------------------------------------- freezing a build output

BUILT_MINT = Minted("built")
"""The namespace a frozen build output's copy is minted in: `.vacuum.n_vac_pumps_high`
-> `^built.vacuum.n_vac_pumps_high`, as the `Naming` a `Cut` takes."""
COND = Minted("cond")
CHECK = NodePath((GetAttrKey("Built"),))


def built_of(var: VarPath) -> VarPath:
    """`^built.<var>`: what was built, a boundary input of the built machine."""
    return BUILT_MINT(var)


def check_condition_for(var: VarPath) -> VarPath:
    """`^cond.built.<var>`: the capacity inequality of a frozen place."""
    return COND(VarPath((GetAttrKey("built"), *var.segments)))


@dataclass(frozen=True)
class Capacity:
    """`(|required| - |built|) / max|built|`, elementwise then the worst: negative when
    what this world asks for fits in what was built. Magnitudes, since a coil current
    has a sign; an array-valued place reduces to its maximum so the condition is one
    scalar, normalised like PROCESS's own residuals. `sign = -1` asks the other way --
    what the world gives must not fall *short* of what was assumed (a lifetime).
    """

    sign: float = 1.0

    def __call__(self, required, built):
        """The worst normalised gap between what this world asks and what was built."""
        required, built = jnp.abs(jnp.asarray(required)), jnp.abs(jnp.asarray(built))
        return jnp.max(self.sign * (required - built) / jnp.maximum(jnp.max(built), 1e-30))


def freeze(graph, var: VarPath, check: str) -> tuple[object, dict]:
    """`var`'s readers rewired to `^built.<var>` (one `Cut` with the `BUILT_MINT`
    naming), so what the rest of the machine is sized from is a boundary input fixed
    the day the machine was built -- while the producer still computes what *this*
    world would ask for, and one `Insert` states the two against each other as
    `.Built.<name>`, owning the capacity inequality `^cond.built.<var>` (implemented by
    `Capacity`).

    Returns the graph and a record: the ops, the readers rewired, the condition.

    Raises
    ------
    KeyError
        If nothing reads `var`, so there is nothing to freeze.
    """
    readers = tuple(n for n, d in graph.definitions.items() if var in d.reads)
    if not readers:
        raise KeyError(f"nothing reads {var.spelling}, so there is nothing to freeze")
    condition = check_condition_for(var)
    place = NodePath((*CHECK.segments, GetAttrKey(var.spelling.split(".")[-1])))
    plan = (
        Ops(graph)
        + Cut(var, readers, BUILT_MINT)
        + Insert(OrderedMap({place: Function((var, built_of(var)), (condition,))}))
        + cx.implement({place: Capacity(1.0 if check == "above" else -1.0)})
    )
    return plan.graph, {
        "place": var.spelling,
        "built": built_of(var).spelling,
        "readers": tuple(r.spelling for r in readers),
        "check": check,
        "condition": condition,
        "ops": (f"Cut({var.spelling}, {len(readers)} readers, ^built)", f"Insert({place.spelling})"),
    }


# ---------------------------------------------------------------- closing a condition

CLOSE = NodePath((GetAttrKey("Close"),))
"""Where a requirement closing an equality binds: `.Close.c2`."""


def place_for(condition: VarPath) -> NodePath:
    """`.Close.<the condition's last component>`."""
    return NodePath((*CLOSE.segments, GetAttrKey(condition.spelling.split(".")[-1])))


def closed_at(graph, condition: VarPath, var: VarPath) -> Ops:
    """`graph` with `condition = 0` asserted at `place_for(condition)` and determined
    by `var`: a requirement, and the `Determine` that makes it a root find over one
    unknown."""
    place = place_for(condition)
    return (
        Ops(graph)
        + Insert(OrderedMap({place: cx.requirement(condition, cx.Eq)}))
        + Determine(place, (var,))
    )


def cycle_of(graph, condition: VarPath, var: VarPath) -> tuple[NodePath, ...]:
    """The component closing `condition` by `var` creates in `graph`: the root find and
    every node its iteration would re-run."""
    place = place_for(condition)
    return next(c for c in closed_at(graph, condition, var).graph.graph.components if place in c)


def analysis(configuration):
    """The machine with every constraint body and the merit, each of its cycles closed
    (`cx.mda`), nothing driven: what a closure's cycle is priced on."""
    return (Ops(study.machine_model(configuration).graph) + cx.mda(study.RULE)).graph


@dataclass(frozen=True)
class Closed:
    """The machine with `condition` closed by `var`: a root find at `place`, every
    cycle on it closed and nested in it (`cx.mdf` with the root find as its top), the
    rest closed on its own, every problem driven. The design is held -- it is a
    boundary input -- so one run is PROCESS's machine at that design."""

    configuration: object
    graph: object
    """The driven graph."""
    pairings: dict
    """`{condition: variable}` -- what closes what."""
    place: NodePath
    design: tuple
    """The configuration's design minus the closing variable: what is held."""
    report: dict
    """`objective`, `equalities`, `inequalities` (places), `closing` (spellings)."""
    given: OrderedMap
    """The model's given values."""

    @property
    def closing(self) -> tuple[VarPath, ...]:
        """The closing variables, in `pairings` order."""
        return tuple(self.pairings.values())

    def root_find_reports(self, out) -> dict:
        """`{condition: (steps, converged, residual, unknown)}` out of a run's values."""
        node = self.graph[self.place]
        reported = dict(zip(node.driver.reports, (s[0] for s in node.reports), strict=True))
        return {
            cond: (
                int(np.asarray(out[reported[Steps]])),
                bool(np.asarray(out[reported[Converged]])),
                float(np.asarray(out[cond])),
                float(np.asarray(out[var])),
            )
            for cond, var in self.pairings.items()
        }


def close(configuration, condition: str, var: str, *, bracket=None) -> Closed:
    """`configuration`'s machine with `condition` closed by `var` -- any boundary input
    of the machine, not only a design variable. Answered by `SafeguardedNewtonDriver`,
    or with `bracket` (`(lower, upper)`) by `BracketedRootDriver` from that bracket."""
    model = study.machine_model(configuration)
    cond, unknown = _var(model.graph, condition), _var(model.graph, var)
    place = place_for(cond)
    plan = closed_at(model.graph, cond, unknown) + cx.mdf(place, study.RULE)
    if any(len(c) > 1 for c in plan.graph.graph.components if place not in c):
        plan = plan + CloseFunctionalCycles(study.RULE, cx.closing)
    chosen = study.drivers_for(plan.graph, skip=(place,))
    chosen[place] = (
        (PortRootFinding(reports=(Steps, Converged)), SafeguardedNewtonDriver())
        if bracket is None
        else (BracketedRootFinding(reports=(Steps, Converged), bracket=True),
              BracketedRootDriver())
    )
    graph = study.driven(plan, chosen).graph
    problem = configuration.problem
    others = getattr(problem, "inequalities", ()) or getattr(problem, "evaluated", ())
    merit = problem.minimise or problem.maximise
    report = {
        "objective": merit.function_outputs[0],
        "equalities": tuple(held(c) for c in problem.equalities),
        "inequalities": tuple(held(c) for c in others),
        "closing": {cond.spelling: unknown.spelling},
    }
    design = tuple(p for p in problem.design if p != unknown)
    return Closed(configuration, graph, {cond: unknown}, place, design, report, model.given)


# ---------------------------------------------------------------- the steps


def refusal(graph, condition: VarPath) -> str:
    """What the graph says when `condition` still has to hold and nothing is left to
    move it: a requirement holding it against no unknown, and the case cottax reports
    of that node quoted verbatim -- the `RequirementUnanswered` naming it, matched **by
    kind**, since the node is the trial insertion. The trial graph is thrown away --
    this is a question, not an op.

    Raises
    ------
    StopIteration
        If the graph does not refuse, which would mean something can still move it.
    """
    place = place_for(condition)
    trial = (Ops(graph) + Insert(OrderedMap({place: cx.requirement(condition, cx.Eq)}))).graph
    return REFUSAL + next(
        v.message
        for v in violations(ExecutableGraph, trial)
        if isinstance(v, RequirementUnanswered) and v.node == place
    )


def _var(graph, spelling: str) -> VarPath:
    """The one place of `graph` spelled `spelling`.

    Raises
    ------
    KeyError
        If no place of `graph` is spelled that way.
    """
    for v in (*graph.graph.boundary_inputs, *graph.graph.owners):
        if v.spelling == spelling:
            return v
    raise KeyError(f"{spelling} is not a place of this graph")


def read_the_input_file(machine: str) -> Plan:
    """Step 0. PROCESS's own question, verbatim: the design variables are what we
    choose, the constraints are what must hold, everything else is the file's
    (`problem.problem_model`).

    No graph operation -- this is the baseline the paper reproduces.
    """
    configuration = configurations.load(machine)
    problem = configuration.problem
    graph = problem_model(configuration).graph
    chosen = tuple(p.spelling for p in problem.design)
    inequalities = getattr(problem, "inequalities", ())
    conditions = tuple(held(c).spelling for c in (*problem.equalities, *inequalities))
    values = dict(process_reference(machine)["x"])
    # PROCESS's `x` covers its design only, and in three of the seven files the dial is
    # not one: there it is the file's own number, given from the start.
    values.setdefault(DIAL, float(state_of(configuration).get(*DIAL.lstrip(".").split("."))))
    merit = problem.minimise or problem.maximise
    plan = Plan(
        machine=machine,
        graph=graph,
        values=values,
        chosen=chosen,
        conditions=conditions,
    )
    return plan.having(
        Step(
            name="read the input file",
            why="PROCESS's own question, verbatim -- the baseline both curves start from",
            ops=(),
            nodes=len(graph.nodes),
            note=(
                f"{len(chosen)} chosen ({len(chosen)} `ixc`), "
                f"{len(problem.equalities)} equalities and "
                f"{len(inequalities)} inequalities, "
                f"{'minimising' if problem.minimise is not None else 'maximising'} "
                f"{merit.function_outputs[0].spelling}; "
                f"`values` is PROCESS's own converged answer"
            ),
        )
    )


def make_hfact_a_given(plan: Plan) -> Plan:
    """Step 1. `hfact` stops being ours to choose.

    One operation: it leaves `chosen`. Where PROCESS was choosing it -- `ixc = 10`
    active -- the problem is now over-determined and the graph says so; the refusal is
    quoted in the step's note, because that refusal is the whole argument: a
    hand-written code lets you make this choice silently and never records it.

    **Where the file fixes `hfact` instead, this step is honestly a no-op** and says
    that too: nothing was reclassified, nothing refuses yet, and it is
    `freeze_the_build` that breaks the count there. Which step drew the refusal is
    visible in `report`, on its own `REFUSED` line.
    """
    graph = plan.graph
    was_chosen = DIAL in plan.chosen
    # PROCESS's own accounting pairs `c2` with iteration variable 10: where that
    # variable is active, giving it away is what leaves the power balance with nothing
    # to move it, and the graph says so. Where the file fixes `hfact` instead, nothing
    # has been reclassified, nothing refuses yet, and this step is honestly a no-op --
    # `freeze_the_build` is where the count breaks there.
    note = (
        f"{DIAL} leaves `chosen`, and PROCESS's own iteration variable 10 with it. "
        + refusal(graph, _var(graph, BALANCE))
        if was_chosen
        else f"{DIAL} is not an `ixc` in this file: PROCESS fixes it at "
        f"{plan.values[DIAL]:g} and already hands it to us, so there is nothing to "
        f"reclassify and **this step is a no-op**. Nothing refuses yet -- the equation "
        f"count breaks one step later, when the build is frozen"
    )
    return plan.having(
        Step(
            name="make hfact a given",
            why=(
                "nobody picks the confinement factor: it is what the plasma does, so it "
                "is handed to us, not chosen"
            ),
            ops=(),
            nodes=len(graph.nodes),
            note=note,
        ),
        chosen=tuple(c for c in plan.chosen if c != DIAL),
    )


def _closure_ops(var: str) -> tuple[str, ...]:
    """What `close` applies, as the recipe prints it."""
    return (
        f"Determine(Requirement({BALANCE}), ({var},)) at .Close.c2",
        "MDF(.Close.c2)",
    )


def make_density_procedure(plan: Plan) -> Plan:
    """Step 2, ignited machines. The power balance is closed by the plasma density:
    at the operating temperature the plasma sits at whatever density balances its own
    heating.

    One requirement `c2 = 0` `Determine`d by `nd_plasma_electrons_vol_avg`, and every
    closure on its cycle nested in it (`MDF` with the root find as its top). Records
    the cycle size -- the price of this choice against the others.
    """
    configuration = configurations.load(plan.machine)
    built = close(configuration, BALANCE, DENSITY)
    priced = analysis(configuration)
    cycle = cycle_of(priced, _var(priced, BALANCE), _var(priced, DENSITY))
    return plan.having(
        Step(
            name="the density absorbs the power balance",
            why=(
                "an ignited plasma is heated by its own alphas: at the operating "
                "temperature it sits at whatever density balances its losses"
            ),
            ops=_closure_ops(DENSITY),
            nodes=len(built.graph.nodes),
            note=(
                f"the cycle it opens: {len(cycle)} nodes"
                f"{_prices(plan, priced, DENSITY)}. {DENSITY} leaves `chosen` -- it is no "
                f"longer ours"
            ),
        ),
        graph=built,
        chosen=tuple(c for c in plan.chosen if c != DENSITY),
        conditions=tuple(c for c in plan.conditions if c != BALANCE),
    )


def make_heating_procedure(plan: Plan) -> Plan:
    """Step 2, driven machines. The power balance is closed by the auxiliary heating
    power: worse confinement simply means turning the heating up.

    One requirement `c2 = 0` `Determine`d by `p_hcd_primary_extra_heat_mw`, answered
    in `HEATING_BRACKET`. The absorbing variable is a plain input here, not an
    iteration variable -- PROCESS never treats the heating as an unknown, which is
    exactly why this question is awkward to ask of it.
    """
    configuration = configurations.load(plan.machine)
    built = close(configuration, BALANCE, HEATING, bracket=HEATING_BRACKET)
    priced = analysis(configuration)
    cycle = cycle_of(priced, _var(priced, BALANCE), _var(priced, HEATING))
    nominal = float(state_of(configuration).get(*HEATING.lstrip(".").split(".")))
    return plan.having(
        Step(
            name="the heating power absorbs the power balance",
            why=(
                "a driven plasma is heated from outside: worse confinement means "
                "turning the heating up, which is what an operator would do"
            ),
            ops=_closure_ops(HEATING),
            nodes=len(built.graph.nodes),
            note=(
                f"the cycle it opens: {len(cycle)} nodes"
                f"{_prices(plan, priced, HEATING)}. {HEATING} is a plain input in this file "
                f"({nominal:g} MW), never an `ixc`: PROCESS cannot be asked this "
                f"question without restating the problem"
            ),
        ),
        graph=built,
        values={**plan.values, HEATING: nominal},
        conditions=tuple(c for c in plan.conditions if c != BALANCE),
    )


def _prices(plan: Plan, priced, chosen: str) -> str:
    """What the other design variables would have cost, as `cycle_of` counts it on
    `priced`: the three cheapest cycles among them. The price of the choice against the
    others.
    """
    condition = _var(priced, BALANCE)
    rows = []
    for spelling in plan.chosen:
        if spelling == chosen:
            continue
        try:
            cycle = cycle_of(priced, condition, _var(priced, spelling))
        except (KeyError, StopIteration):
            continue
        if len(cycle) > 1:  # a cycle of one is the root find alone: it closes nothing
            rows.append((len(cycle), spelling))
    cheapest = ", ".join(f"{s} {n}" for n, s in sorted(rows)[:3])
    return f" (the other candidates, cheapest first: {cheapest})" if cheapest else ""


def freeze_the_build(plan: Plan) -> Plan:
    """Step 3. The machine is built: the design is held at PROCESS's answer.

    Every build quantity leaves `chosen`. Some sizing rules would otherwise quietly
    re-size the coils at each sweep point; they are cut and held at their built value,
    and the check they performed becomes a condition on the built machine instead
    (`|what this world asks| <= |what was built|`).

    Records both the cuts and -- the part PROCESS cannot answer -- which conditions are
    now *constants given the build*, decided the day the coil was wound, versus which
    still depend on the swept dial (`reach` from the dial, the one varying leaf).
    """
    built = plan.graph
    out = study.solve(built.graph, _inputs(plan))  # the machine as PROCESS built it
    driven = built.graph
    reached = set(driven.graph.reach([_var(driven, DIAL)]))
    owners = driven.graph.owners
    owned = {v.spelling: v for v in owners}
    # Every claimed build output the dial still re-sizes, in spelling order.
    hits = [owned[s] for s in sorted(CLAIMED_BUILD_OUTPUTS)
            if s in owned and owners[owned[s]] in reached]
    values, ops, conditions, held_ = dict(plan.values), [], [], []
    for var in hits:
        # A lifetime must not fall short of what was assumed; every other built
        # quantity must not be exceeded by what this world asks for.
        check = "below" if var.spelling.split(".")[-1].startswith("life_") else "above"
        driven, record = freeze(driven, var, check)
        values[record["built"]] = float(np.asarray(out[var]))
        ops.extend(record["ops"])
        conditions.append(record["condition"])
        held_.append(record["built"])
    report = dict(
        built.report,
        inequalities=(*built.report["inequalities"], *conditions),
        frozen={h: values[h] for h in held_},
        blocks=len(driven.graph.components),
    )
    downstream = [c for c in plan.conditions if owners[_var(built.graph, c)] in reached]
    constant = [c for c in plan.conditions if c not in downstream]
    # Where the file fixed the dial, nothing refused at step 1 and **this** is the step
    # that breaks the equation count: with every design variable held, the power
    # balance has no unknown left. Asked of the graph as the file states it, since the
    # one in hand has `c2` closed already.
    drew = "" if any(REFUSAL in step.note for step in plan.steps) else (
        f". And with every design variable held, {BALANCE} has no unknown left: had it "
        f"not been closed a step earlier, "
        + refusal(read_the_input_file(plan.machine).graph, _var(driven, BALANCE))
    )
    return plan.having(
        Step(
            name="freeze the build",
            why="the machine exists: its coils are wound, and nothing re-sizes them",
            ops=tuple(ops),
            nodes=len(driven.nodes),
            note=(
                f"{len(plan.chosen)} design variables held at PROCESS's answer; "
                f"{len(hits)} build outputs were still being re-sized by the dial and "
                f"are now cut and checked ({', '.join(h.spelling for h in hits)}). "
                f"Of the {len(plan.conditions)} conditions, {len(constant)} are "
                f"constants given the build ({', '.join(c.split('.')[-1] for c in constant)}) "
                f"and {len(downstream)} still depend on the dial "
                f"({', '.join(c.rsplit('.', 1)[-1] for c in downstream)})" + drew
            ),
        ),
        graph=replace(built, graph=driven, report=report),
        values=values,
        chosen=(),
        conditions=(*plan.conditions, *(c.spelling for c in conditions)),
    )


# ---------------------------------------------------------------- turning it


def _inputs(plan: Plan, dial: float | None = None) -> OrderedMap:
    """A value at every boundary input of the built machine: PROCESS's converged design
    at each design place held, the closing variable started from its own value (its
    bracket, where it takes one), every frozen place at what was built, the dial at
    `dial` (default its value), and everything else as `study.values` has it: the
    file's state, the given values, else the cold state. Shared by `freeze_the_build`
    (which measures what was built) and `sweep` (which holds it).
    """
    built = plan.graph
    graph, configuration = built.graph, built.configuration
    node = graph[built.place]
    given = dict(built.given)
    for v in graph.graph.boundary_inputs:
        if v.spelling in plan.values:
            given[v] = jnp.asarray(plan.values[v.spelling], dtype=jnp.float64)
    for role, slot in zip(node.driver.takes, node.data, strict=True):
        for place, unknown in zip(slot, node.unknowns, strict=True):
            if role is Start and unknown.spelling in plan.values:
                given[place] = jnp.asarray(plan.values[unknown.spelling], dtype=jnp.float64)
            elif role is BracketRole:
                given[place] = jnp.asarray(HEATING_BRACKET, dtype=jnp.float64)
    if dial is not None:
        given[_var(graph, DIAL)] = jnp.asarray(float(dial), dtype=jnp.float64)
    return study.values(graph, state_of(configuration), OrderedMap(given),
                        _cold(configuration.name))


_COLD: dict = {}


def _cold(machine: str) -> dict:
    """`study.cold_state`, once per machine."""
    if machine not in _COLD:
        _COLD.clear()
        _COLD[machine] = study.cold_state(configurations.load(machine))
    return _COLD[machine]


def sweep(plan: Plan, dial: str, values) -> object:
    """Step 4. Turn the dial. One solve per value: the absorbing variable moves, every
    condition is evaluated, and the machine is operable where all of them hold.

    Returns a table with, per value: the absorbing variable, the objective, each
    condition's residual, and which one binds first.
    """
    built = plan.graph
    assert dial == DIAL, "the built machine turns the one dial it was frozen for"
    # Not the closed ones: a condition a root find answers sits at zero by
    # construction, and `steps`/`converged` below are what says so.
    conditions = tuple(
        c
        for c in (*built.report["equalities"], *built.report["inequalities"])
        if c not in built.pairings
    )
    absorbing = built.closing
    rows: list[dict] = []
    nominal: dict[str, float] = {}
    for value in values:
        try:
            out = study.solve(built.graph, _inputs(plan, value))
        except Exception as error:  # noqa: BLE001 -- a failed point is a row, not a crash
            rows.append({dial: float(value), "failed": f"{type(error).__name__}: {error}"})
            continue
        residuals = {c.spelling: float(np.asarray(out[c])) for c in conditions}
        if not nominal:  # the first point is PROCESS's own: what held there can break
            nominal = residuals
        # Only a condition that held **with slack** at the nominal can break: one
        # already at its limit there (a constraint the optimiser made active, and every
        # frozen capacity check, which is zero by construction the day it is frozen)
        # was never a margin to spend.
        broken = {
            s: r for s, r in residuals.items() if r > HELD and nominal[s] < -SLACK
        }
        verdicts = built.root_find_reports(out)
        delivered = next((v for v in out if v.spelling == NET), None)
        rows.append({
            dial: float(value),
            **{v.spelling: float(np.asarray(out[v])) for v in absorbing},
            "objf": float(np.asarray(out[built.report["objective"]])),
            NET: None if delivered is None else float(np.asarray(out[delivered])),
            "converged": all(bool(v[1]) for v in verdicts.values()),
            "steps": max(int(v[0]) for v in verdicts.values()),
            "conditions": residuals,
            "binds": max(broken, key=broken.get) if broken else "",
            **({"tight": [s for s, r in residuals.items()
                          if r >= -SLACK and not s.startswith("^cond.built.")],
                "frozen": [s for s in residuals if s.startswith("^cond.built.")]}
               if residuals is nominal else {}),
        })
    return rows


def redesign(plan: Plan, dial: str, values) -> object:
    """The control: PROCESS's own question at every value of the dial -- the whole
    machine re-optimised under MDF (`study.study`), as `Scan` does. Always feasible, a
    different machine each time. Run from the plan *before* `freeze_the_build`.
    """
    configuration = configurations.load(plan.machine)
    problem = configuration.problem
    if not isinstance(problem, Optimisation):
        return [{dial: float(v), "status": "no MDF arm: this file states a root find"}
                for v in values]
    # What is still ours to choose: `plan.chosen`, which is the file's design minus the
    # dial -- so this must be the plan from *before* the build was frozen.
    kept = {p: b for p, b in problem.design.items() if p.spelling in plan.chosen}
    cold = _cold(plan.machine)
    rows, previous = [], {}
    for value in values:
        # The dial out of the design and into the file's own numbers: PROCESS's
        # question at this value of it, which is what `Scan` asks one point at a time.
        # Each point starts from the last one that converged (PROCESS's `Scan` does the
        # same), and `--no-redesign` skips the whole curve.
        stated = {a: dict(fields) for a, fields in configuration.values.items()}
        for spelling, x in (*previous.items(), (dial, float(value))):
            where, field = spelling.lstrip(".").split(".")
            stated.setdefault(where, {})[field] = float(x)
        answer = study.study(
            replace(configuration, values=stated, problem=replace(problem, design=kept)),
            "MDF", cold,
        )
        design = dict(zip([p.spelling for p in kept], answer["x"], strict=True))
        if answer["status"] == "converged":
            # A design variable addressing one element of an array
            # (`f_nd_impurity_electron_array[12]`) is not a `{area: {field: value}}`
            # place, so it stays at the file's value and only the scalars are carried
            # forward.
            previous = {s: x for s, x in design.items() if "[" not in s}
        rows.append({
            dial: float(value),
            "objf": answer["objf"],
            "status": answer["status"],
            "iterations": answer["iterations"],
            "max_eq": answer["max_eq"],
            "min_ie": answer["min_ie"],
            "x": design,
        })
    return rows


def report(plan: Plan) -> str:
    """The recipe as a numbered list: each step, the operations it applied, why, and
    the node count after it. This is what the paper prints.
    """
    lines = [f"one dial in, one dial out -- {plan.machine}", ""]
    for number, step in enumerate(plan.steps):
        lines.append(f"{number}. {step.name}   [{step.nodes} nodes]")
        lines.append(f"      why    {step.why}")
        for op in step.ops or ("(no graph operation)",):
            lines.append(f"      op     {op}")
        if step.note:
            note, _, quoted = step.note.partition(REFUSAL)
            if note.strip():
                lines.extend(_wrapped("      note   ", note.strip()))
            if quoted:
                lines.extend(_wrapped("      REFUSED ", quoted.strip()))
    lines.append("")
    lines.append(f"   chosen     {', '.join(plan.chosen) or '(nothing: the machine is built)'}")
    lines.extend(_wrapped("   conditions ", ", ".join(plan.conditions)))
    return "\n".join(lines)


def _wrapped(label: str, text: str) -> list[str]:
    """`text` wrapped under `label`, continuation lines indented to it."""
    import textwrap  # noqa: PLC0415

    wrapped = textwrap.wrap(text, 96 - len(label)) or [""]
    return [label + wrapped[0]] + [" " * len(label) + line for line in wrapped[1:]]


# ---------------------------------------------------------------- the command line


def recipe(machine: str) -> Plan:
    """The four steps, in order: the whole script in one place."""
    plan = read_the_input_file(machine)
    plan = make_hfact_a_given(plan)
    procedure = (
        make_density_procedure if CLOSURE[machine] == "density" else make_heating_procedure
    )
    return freeze_the_build(procedure(plan))


def margin(rows, dial: str) -> tuple[float, str]:
    """How far the dial fell before a condition that held at the nominal stopped
    holding, as a fraction of the nominal, and which condition it was -- interpolated
    between the last point that worked and the first that did not.
    """
    nominal = rows[0][dial]
    for previous, row in zip(rows, rows[1:], strict=False):
        if row.get("failed"):
            return (nominal - row[dial]) / nominal, "did not solve"
        if not row["converged"]:
            return (nominal - row[dial]) / nominal, "the closure stopped converging"
        if row["binds"]:
            binding = row["binds"]
            before, after = previous["conditions"][binding], row["conditions"][binding]
            crossing = previous[dial] + (row[dial] - previous[dial]) * (
                (HELD - before) / (after - before) if after != before else 0.0
            )
            return (nominal - crossing) / nominal, binding
    return (nominal - rows[-1][dial]) / nominal, ""


def figure(machine: str, built_rows, redesign_rows, dial: str, absorbing: str, path: Path):
    """Three panels, one x-axis and one y-axis each: the objective under both
    questions, the absorbing variable on the built machine, and what the plant then
    delivers -- with the wall marked on all three.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # Okabe-Ito: a blue and a vermillion, separable under every common CVD.
    control, held = "#0072B2", "#D55E00"
    good = [r for r in built_rows if not r.get("failed")]
    x = [r[dial] for r in good]
    fall, binding = margin(built_rows, dial)
    wall = good[0][dial] * (1 - fall)

    fig, (left, right, plant) = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    scan = [r for r in redesign_rows if r.get("objf") is not None]
    if scan:
        left.plot([r[dial] for r in scan], [r["objf"] for r in scan], "-o", ms=4, lw=2,
                  color=control, label="redesign (re-optimised at each value)")
    left.plot(x, [r["objf"] for r in good], "-o", ms=4, lw=2, color=held,
              label="built (the design held)")
    left.set_xlabel(dial)
    left.set_ylabel("objective")
    left.legend(frameon=False, fontsize=9)

    right.plot(x, [r[absorbing] for r in good], "-o", ms=4, lw=2, color=held)
    right.set_xlabel(dial)
    right.set_ylabel(absorbing.rsplit(".", maxsplit=1)[-1])

    plant.plot(x, [r[NET] for r in good], "-o", ms=4, lw=2, color=held)
    plant.axhline(0.0, color="0.6", lw=0.8)
    plant.set_xlabel(dial)
    plant.set_ylabel("net electric power (MW)")

    for axis in (left, right, plant):
        axis.grid(True, lw=0.4, color="0.85")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlim(min(x), max(x))
        if binding:
            # Everything left of the wall is a machine that does not work: the built
            # curve is drawn there only to show how fast it leaves the feasible set.
            axis.axvspan(min(x), wall, color="0.92", zorder=0)
            axis.axvline(wall, color="0.35", lw=1, ls="--")
    label = {"fontsize": 8.5, "color": "0.3",
             "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5}}
    if binding:
        left.text(0.02, 0.96, "not operable", transform=left.transAxes, va="top", **label)
    if scan and len(scan) < len(good):
        # The control is a cold VMCON per point; where it reports `no-step` there is no
        # re-optimised machine to draw, and saying so is better than a curve that stops.
        left.text(0.02, 0.86, f"control: MDF converges only above {min(r[dial] for r in scan):.3f}",
                  transform=left.transAxes, va="top", **label)
        right.text(0.02, 0.04,
                   f"{binding.split('.')[-1]} breaks "
                   + (f"{100 * fall:.1f} % below nominal" if fall > 1e-3
                      else "as soon as the dial moves"),
                   transform=right.transAxes, **label)
    fig.suptitle(f"{machine}: one dial in ({dial}), one dial out ({absorbing})", fontsize=11)
    tight = built_rows[0].get("tight", ())
    if tight:
        fig.supxlabel(
            "tight at the nominal, so never a margin to spend: "
            + ", ".join(t.split(".")[-1] for t in tight),
            fontsize=8, color="0.35",
        )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    """`--machine <name>` for one configuration, or `both` for the showcase pair."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine", default="both",
                        help=f"a configuration name, or `both` for {' and '.join(SHOWCASE)}")
    parser.add_argument("--no-redesign", action="store_true",
                        help="skip the control curve (one MDF solve per point)")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    names = list(SHOWCASE) if args.machine == "both" else [args.machine]
    rows = []
    for machine in names:
        print()
        plan = recipe(machine)
        print(report(plan))
        rows.append(one(plan, args))
        # One machine at a time, and nothing of it kept: several configurations' worth
        # of jitted schedules does not fit in a laptop.
        jax.clear_caches()
        gc.collect()
    if len(rows) > 1:
        print(f"\n{'configuration':24s} {'closed by':44s} {'margin':>8s}  binds first")
        for row in rows:
            slack = row["slack_at_the_nominal"]
            print(f"{row['machine']:24s} {row['closure']:44s} "
                  f"{100 * row['margin_fraction']:7.2f}%  {row['binds'].rsplit('.', 1)[-1]}"
                  + (f" (held by {abs(slack):.0e})" if slack is not None else ""))
    return 0


def one(plan: Plan, args) -> dict:
    """One machine, turned: the sweep, the control curve, the JSON beside it, the
    figure, and the row the summary prints.
    """
    machine = plan.machine
    nominal = plan.values[DIAL]
    values = [nominal * (1 - f) for f in FALLS]
    built_rows = sweep(plan, DIAL, values)
    absorbing = plan.graph.closing[0].spelling
    fall, binding = margin(built_rows, DIAL)
    tight = built_rows[0].get("tight", [])
    frozen = built_rows[0].get("frozen", [])
    slack = built_rows[0]["conditions"].get(binding)
    # The control is asked of the plan before the build was frozen: step 1, where the
    # dial is given and everything else is still ours.
    scan_rows = [] if args.no_redesign else redesign(
        make_hfact_a_given(read_the_input_file(machine)), DIAL, values
    )
    (OUT / f"one_dial_{machine}.json").write_text(json.dumps({
        "machine": machine,
        "recipe": report(plan),
        "dial": DIAL,
        "absorbing": absorbing,
        "nominal": nominal,
        "margin_fraction": fall,
        "binds": binding,
        "slack_at_the_nominal": slack,
        "tight_at_the_nominal": tight,
        "frozen_tight_by_construction": frozen,
        "built": built_rows,
        "redesign": scan_rows,
    }, indent=1, default=float))
    if not args.no_figure:
        figure(machine, built_rows, scan_rows, DIAL, absorbing,
               OUT / f"one_dial_{machine}.png")
    print(f"\n{machine}: {absorbing} absorbs c2; hfact can fall "
          f"{100 * fall:.2f} % before {binding or 'nothing'} breaks"
          + (f" (which held by only {abs(slack):.1e} at the nominal)"
             if slack is not None else ""))
    print(f"   already tight at the nominal, so never a margin to spend: "
          f"{', '.join(t.rsplit('.', 1)[-1] for t in tight) or '(none)'}")
    print(f"   frozen, tight by construction: "
          f"{', '.join(t.rsplit('.', 1)[-1] for t in frozen) or '(none)'}\n")
    return {
        "machine": machine,
        "closure": absorbing,
        "nominal_hfact": nominal,
        "margin_fraction": fall,
        "binds": binding or "(nothing within the swept range)",
        "slack_at_the_nominal": slack,
        "tight_at_the_nominal": tight,
        "frozen_tight_by_construction": frozen,
        "nominal_absorbing": built_rows[0].get(absorbing),
        "steps": len(plan.steps),
    }


if __name__ == "__main__":
    raise SystemExit(main())
