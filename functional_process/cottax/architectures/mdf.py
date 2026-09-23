"""The `Optimise` layer, assembled the other way round: **MDF** (Multidisciplinary
Feasible).
"""

import dataclasses
import time

import jax.numpy as jnp
import numpy as np
from cottax.execution.drivers.kinds import Converged, Start, Steps
from cottax.interfaces import (
    Absorb,
    ConditionMap,
    Drive,
    ExecutableGraph,
    Graph,
    Insert,
    Plan,
    RunnableGraph,
    Schedule,
    is_driven,
)
from cottax.interfaces.statements import RootFind
from cottax.pytree.path import NodePath, PathMap, VarPath
from cottax.visualization.sequencing import problem_types
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey

from functional_process.cottax.architectures import sand
from functional_process.cottax.architectures.drivers import (
    SeededNewtonDriver,
    # `Status` was written here, for `MdfNewtonDriver`, and moved to `drivers` when
    # `VmconDriver` and `SlsqpDriver` started reporting one too: a port naming belongs
    # beside the drivers that write it, not beside the one assembly that first read it.
    # Re-exported (`__all__`) so `mdf.Status` still resolves for every existing caller.
    Status,
)
from functional_process.cottax.architectures.evaluate import (
    cold_state,
    cold_value,
    ground_truth,
    run_schedule,
    without_excluded,
)
from functional_process.cottax.architectures.mda import (
    SCHEME,
    assign_drivers,
    cut_graph,
    default_drivers,
    given_start,
    guess_sources,
)
from functional_process.cottax.input.indat import graph_for
from functional_process.cottax.queries import component_of, nested_inside


@dataclasses.dataclass(frozen=True)
class Mdf:
    """One assembled MDF problem: the graph, the two schedules, and the problem's shape.
    """

    graph: Graph
    """The MDA graph plus one node per active constraint and one for the objective."""
    eager: Schedule
    traceable: Schedule
    design: tuple[VarPath, ...]
    """The run's `ixc`, in PROCESS's own order."""
    conditions: tuple[VarPath, ...]
    """What the outer driver is handed."""
    n_equality: int
    n_inequality: int
    report: dict
    problem_type: str = 'optimise'
    """Which problem this file states -- `Optimise` or `RootFind`. See `assemble`."""
    reported: tuple[VarPath, ...] = ()
    """Conditions assembled but **not driven**: a `RootFind`'s inequalities."""
    raw: Graph | None = None
    """The graph before any cut -- what `seed` asks `evaluate.cold_state` about for a
    copy `data` has no value for."""


def mdf_graph(graph, icc, n_equality, i_figure_merit, switch_values=None, omit=()):
    """`graph` with `sand.condition_graph`'s nodes inserted -- the conditions, with no
    requirement and no `Optimise`: this arm drives the optimiser from outside the graph.
    """
    inserted, report = sand.condition_graph(
        graph, icc, n_equality, i_figure_merit, switch_values, omit
    )
    objective = report["objective"]
    return (
        inserted,
        ((objective,) if objective is not None else ())
        + (*report["equalities"], *report["inequalities"]),
        len(report["inequalities"]),
        report,
    )


def traceable_drivers(drivers):
    """`drivers` with every `SeededNewtonDriver`'s `seed` cleared."""
    return {
        problem: (
            dataclasses.replace(driver, seed=None)
            if isinstance(driver, SeededNewtonDriver)
            else driver
        )
        for problem, driver in drivers.items()
    }


def assemble(
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    graph=None,
    switch_values=None,
    omit=(),
    root_find=False,
    scheme=SCHEME,
):
    """The whole MDF assembly: cut the raw cycles, add the conditions, build both
    schedules. `scheme` is how the cycles are opened (`mda.SCHEME`).
    """
    if root_find and n_equality != len(ixc):
        raise ValueError(
            f"a root find needs one equality per iteration variable, and this file "
            f"states {n_equality} equality constraint(s) against {len(ixc)} iteration "
            f"variable(s) -- PROCESS's own `fsolve` over `evaluate_eq_cons` would be "
            f"the same non-square system, so there is nothing to root-find here"
        )
    raw = without_excluded(graph if graph is not None else graph_for())
    driven = cut_graph(raw, scheme)
    graph, conditions, n_inequality, report = mdf_graph(
        driven,
        icc,
        n_equality,
        None if root_find else i_figure_merit,
        switch_values,
        omit,
    )
    drivers = default_drivers(graph)
    # Two algorithms over one structure means **two graphs** now, not two driver maps:
    # `Assign` puts the algorithm in the graph, so the eager and traceable variants are
    # separate objects. `reassign_drivers` is not needed -- neither graph carries a
    # driver yet, since `cut_graph` is structure only.
    eager_graph = assign_drivers(graph, drivers)
    answerable = RunnableGraph(eager_graph)
    design = tuple(sand.iteration_variable_path(i) for i in ixc)
    eager = Schedule(answerable)
    missing = [d for d in design if d not in eager.inputs]
    if missing:
        raise ValueError(
            f"design variable(s) {[d.spelling for d in missing]} are not boundary "
            f"inputs of the MDA graph -- a node already produces them, so the optimiser "
            f"cannot own them (see `sand.problem_graph` on the same conflict)"
        )
    report["blocks"] = len(eager_graph.graph.components)
    report["driven_blocks"] = sum(1 for t in problem_types(eager_graph) if t is not None)
    driven_conditions, reported = conditions, ()
    if root_find:
        # The equalities alone are driven; the inequalities stay in the graph so they
        # can be read at the answer, exactly as PROCESS reads them there.
        driven_conditions = tuple(report["equalities"])
        reported = tuple(report["inequalities"])
    return Mdf(
        graph=graph,
        eager=eager,
        traceable=Schedule(
            RunnableGraph(assign_drivers(graph, traceable_drivers(drivers)))
        ),
        design=design,
        conditions=driven_conditions,
        n_equality=n_equality,
        n_inequality=0 if root_find else n_inequality,
        report=report,
        problem_type='root-find' if root_find else 'optimise',
        reported=reported,
        raw=raw,
    )


def guess_ports(mdf: Mdf) -> dict:
    """`{guess_port: unknown}` over every problem in `mdf.graph`."""
    # Asked of the **eager schedule's** graph, not `mdf.graph`. Since `Assign` mints the
    # `^guess.*` ports from the driver's own `requires`, a graph with no drivers on it has
    # no start ports at all -- and `mdf.graph` is the cut graph plus the `Optimise`,
    # deliberately undriven so that `Combine` can still join its problems. Asking that one
    # returns nothing, every port falls to `ground_truth`'s `0.0`, and the inner solves
    # start from exactly the cold point `prime` exists to get them off.
    return guess_sources(mdf.eager.executable.graph)


def seed(mdf: Mdf, data, design_values=None):
    """Every schedule input and every inner unknown, read off `data`."""
    env = {}
    starts = guess_ports(mdf)
    shapes = None if mdf.raw is None else cold_state(data, mdf.raw)
    for var in list(mdf.eager.inputs) + list(mdf.eager.unknowns):
        # A `^guess.*` port is grounded from the unknown it starts, not from its own
        # name -- there is no `DataStructure` field spelled that way.
        source = starts.get(var, var)
        try:
            grounded = ground_truth(data, source)
        except (AttributeError, KeyError):
            grounded = cold_value(source, shapes)
        # A `^guess.*` port may be *given* its value instead of read off `data` --
        # `mda.GIVEN_STARTS` for which, and why a cold dataclass default is not a
        # starting guess. Guess ports only; an ordinary input is the machine's own number.
        if var in starts:
            grounded = given_start(source, grounded)
        env[var] = _not_weak(grounded)
    if design_values is not None:
        values = [_not_weak(v) for v in design_values]
        env.update(zip(mdf.design, values, strict=True))
    return env


def _not_weak(value):
    """`jnp.asarray(value)`, with jax's own `weak_type` bit forced off."""
    return jnp.asarray(value, dtype=np.asarray(value).dtype)


def prime(mdf: Mdf, env):
    """Run the MDA once, eagerly, and keep its answer as the inner starting guess."""
    # Jitted, driver by driver: `Mdf.eager`'s `SeededNewtonDriver`s cannot be traced,
    # so the schedule is walked with every `Call` run and every `Drive` body fused and
    # the drivers left eager. 49.3 s / 978 XLA compiles -> 16.4 s / 32 on
    # `large_tokamak_nof` (`_audit/next_steps.md` §24.11).
    out = run_schedule(mdf.eager, _inputs_only(mdf, env))
    primed = dict(env)
    for guess, unknown in guess_ports(mdf).items():
        # Written to the `Start` port, which is where the driver reads it. Writing the
        # unknown's own name would set the *answer* and leave the guess cold.
        primed[guess] = out[unknown]
    return primed, out


def _inputs_only(mdf: Mdf, env):
    """`env` restricted to what the schedule may be handed: its own inputs."""
    inputs = set(mdf.eager.inputs)
    return {var: value for var, value in env.items() if var in inputs}


class MdfNewtonDriver(SeededNewtonDriver):
    """`SeededNewtonDriver` that reports optimistix's verdict instead of raising it."""

    max_steps: int = 256

    @property
    def reports(self):
        """`(Steps, Converged, Status)` -- what `__call__` returns after the design."""
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """The root of `conditions` started from `data[Start]`, then the verdict."""
        import optimistix as optx  # noqa: PLC0415 -- only this arm needs it

        start = data.get(Start)
        if start is None:
            raise ValueError(
                f"MdfNewtonDriver needs a starting value for every design variable "
                f"({', '.join(v.spelling for v in conditions.unknowns)})"
            )
        flat_guess, unravel = ravel_pytree(start)

        def residual(flat, args=None):
            out, _ = ravel_pytree(conditions(*unravel(flat)))
            return out

        solution = optx.root_find(
            residual,
            optx.Newton(rtol=self.rtol, atol=self.atol),
            flat_guess,
            throw=False,
            max_steps=self.max_steps,
        )
        return (
            *unravel(solution.value),
            solution.stats["num_steps"],
            solution.result == optx.RESULTS.successful,
            getattr(solution.result, "_value", jnp.asarray(-1)),
        )


def report_place(node, naming):
    """Where `naming`'s report of a driven statement lands, or `None` where its driver
    does not report one. A report is named from the statement's **first unknown**
    (`^status.<u>`), so this asks the node and never spells a path.
    """
    node = node.node if isinstance(node, Drive) else node
    if not is_driven(node) or naming not in node.driver.reports:
        return None
    return node.reports[node.driver.reports.index(naming)]


def verdict(out, naming, node):
    """What a driver said about its own run, out of the env a solve returned. `node` is
    the driven statement, or the `Drive` answering it.
    """
    place = report_place(node, naming)
    return None if place is None else out.get(place)


def nested_blocking(ixc, icc, n_equality, i_figure_merit, graph=None, scheme=SCHEME, **kwargs):
    """MDF **stated as structure**: `ExecutableGraph(nested_inside(graph + Optimise, the Optimise))`.
    """
    driven = cut_graph(without_excluded(graph if graph is not None else graph_for()), scheme)
    with_problem, problem_name, report = sand.problem_graph(
        driven, ixc, icc, n_equality, i_figure_merit, **kwargs
    )
    # `problem_graph` leaves the requirements standing for an architecture to absorb,
    # and this route states the nesting itself instead of taking one -- so it does the
    # absorb an architecture would do first, or the proof refuses them.
    required = report["required"]
    absorbed = (
        Absorb(problem_name, required).apply(with_problem) if required else with_problem
    )
    nested = nested_inside(absorbed, problem_name)
    return ExecutableGraph(nested), problem_name, report


IN_GRAPH_PLACE = NodePath((GetAttrKey("RootFind"),))
"""Where the in-graph problem binds."""


@dataclasses.dataclass(frozen=True)
class InGraphRootFind:
    """PROCESS's evaluation-mode root find, **stated as a node of the graph**."""

    mdf: Mdf
    """The assembly this states -- a `RootFind` one (`assemble(root_find=True)`)."""
    graph: Graph
    """`mdf.graph` plus the `RootFind`, with every problem's driver `Assign`ed on."""
    blocking: ExecutableGraph
    """`ExecutableGraph(queries.nested_inside(graph, problem))`: the graph proved
    answerable, nested at the problem.
    """
    schedule: Schedule
    problem: NodePath

    def verdict(self, out, naming):
        """What the outer driver said about its own run, out of a run's env."""
        return verdict(out, naming, self.graph[self.problem])

    def steps(self, out) -> int | None:
        """How many Newton steps the outer solve took, from that run's env."""
        return self.verdict(out, Steps)

    def successful(self, out) -> bool | None:
        """Optimistix's own verdict on the outer solve, from that run's env."""
        return self.verdict(out, Converged)

    @property
    def index(self) -> int:
        """Which component of `graph` -- which step of `schedule` -- answers the root
        find.
        """
        return component_of(self.graph, self.problem)

    @property
    def drive(self) -> Drive:
        """The outer `Drive` step -- the surface a caller measures a residual on."""
        return self.schedule.steps[self.index]

    @property
    def block(self) -> tuple:
        """The nodes of the driven component, the problem included."""
        return self.graph.graph.components[self.index]

    @property
    def interior(self) -> Graph:
        """That component one level down -- the block minus the root find
        (`NestingGraph.interior`), a graph whose own `entries` are the next depth.
        """
        return self.graph.interior(self.problem)

    @property
    def design(self) -> tuple[VarPath, ...]:
        """The run's `ixc`, in PROCESS's own order -- what the root find owns."""
        return self.mdf.design

    @property
    def conditions(self) -> tuple[VarPath, ...]:
        """The equalities alone -- what the root find reads, one per design variable."""
        return self.mdf.conditions

    @property
    def reported(self) -> tuple[VarPath, ...]:
        """The inequalities: in the graph, evaluated once at the answer, never driven."""
        return self.mdf.reported


def root_find_node(mdf: Mdf):
    """The root find `mdf` states, as a cottax node: owns `design`, reads `conditions`.
    """
    if not mdf.problem_type == 'root-find':
        raise TypeError(
            f"this MDF states an {mdf.problem_type}, and only the `RootFind` "
            f"arm is stated in-graph here -- an `Optimise` nests just as well "
            f"(`_audit/in_graph_rootfind.md` §1 measures it), but its outer driver is a "
            f"`VmconDriver`, which does not trace, so that is a separate change"
        )
    return RootFind(tuple(mdf.conditions), tuple(mdf.design))


def in_graph_root_find(
    mdf: Mdf,
    place: NodePath = IN_GRAPH_PLACE,
    driver=None,
    traceable: bool = True,
    **kwargs,
) -> InGraphRootFind:
    """State `mdf`'s root find inside the graph and let `ExecutableGraph` decide what it
    drives.
    """
    node = root_find_node(mdf)
    if place in mdf.graph.definitions:
        raise ValueError(
            f"{place!r} is already a node of this graph -- pass `place` to bind the "
            f"root find somewhere else"
        )
    with_problem = (Plan(mdf.graph) + Insert(PathMap([(place, node)]))).graph
    drivers = default_drivers(with_problem)
    if traceable:
        drivers = traceable_drivers(drivers)
    # `default_drivers` already put a `SeededNewtonDriver` on the new `RootFind` -- it
    # dispatches on the problem type and cannot know this one is the outer solve. It is
    # replaced rather than skipped: `MdfNewtonDriver` reports optimistix's verdict
    # instead of raising it, which is what makes a non-converged outer solve a row.
    drivers[place] = driver or MdfNewtonDriver(**kwargs)
    assigned = nested_inside(assign_drivers(with_problem, drivers), place)
    blocking = RunnableGraph(assigned)
    return InGraphRootFind(
        mdf=mdf,
        graph=assigned,
        blocking=blocking,
        schedule=Schedule(blocking),
        problem=place,
    )


def in_graph_inputs(built: InGraphRootFind, env):
    """The env `built.schedule` is handed, out of an `Mdf` seed/prime env."""
    starts = guess_sources(built.graph)
    out, missing = {}, []
    for var in built.schedule.inputs:
        if var in env:
            out[var] = env[var]
        elif var in starts and starts[var] in env:
            # A `^guess.*` port is grounded from the unknown it starts. For the outer
            # root find that unknown is a design variable, whose value `seed` read off
            # the input file's own `ixc`.
            out[var] = env[starts[var]]
        else:
            missing.append(var)
    if missing:
        raise KeyError(
            f"no value for schedule input(s) "
            f"{[v.spelling for v in missing]} -- `seed` then `prime` is what fills "
            f"this env, and a `^guess.*` port is filled from the unknown it starts"
        )
    return out


def _hashable(value) -> bool:
    """Whether `value` can be a key -- asked, never inferred from what it holds."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def in_graph_solve(built: InGraphRootFind, env, whole=None):
    """Run the whole schedule: upstream once, the root find, downstream once."""
    inputs = in_graph_inputs(built, env)
    started = time.perf_counter()
    if _hashable(built.schedule):
        out = run_schedule(built.schedule, inputs, whole=whole)
    else:
        # The walk computes the same values by the same nodes in the same order -- it is
        # the cost that differs, not the answer (`_audit/in_graph_rootfind.md` §6).
        out = dict(built.schedule.run(PathMap(inputs)))
    return (
        tuple(out[var] for var in built.design),
        out,
        time.perf_counter() - started,
    )


def in_graph_shape(built: InGraphRootFind) -> dict:
    """The assembly's shape -- `mdf_shape` plus what the blocking decided."""
    interior = built.interior
    return {
        **mdf_shape(built.mdf),
        "graph_nodes": len(built.graph.nodes),
        "outer_blocks": len(built.graph.graph.components),
        "block": len(built.block),
        "body": len(built.drive.body.nodes),
        "interior_blocks": len(interior.graph.components),
        "interior_driven": sum(1 for t in problem_types(interior) if t is not None),
        "upstream": built.index,
        "downstream": len(built.graph.graph.components) - built.index - 1,
    }


def mdf_shape(mdf: Mdf) -> dict:
    """The assembly's shape, for reporting -- the counterpart of `sand.sand_shape`."""
    return {
        "nodes": len(mdf.graph.nodes),
        "design": len(mdf.design),
        "conditions": len(mdf.conditions),
        "equalities": mdf.n_equality,
        "inequalities": mdf.n_inequality,
        "inner_blocks": mdf.report["blocks"],
        "inner_driven": mdf.report["driven_blocks"],
        "inner_unknowns": len(mdf.eager.unknowns),
        "inner_inputs": len(mdf.eager.inputs),
    }


__all__ = [
    "IN_GRAPH_PLACE",
    "InGraphRootFind",
    "Mdf",
    "MdfNewtonDriver",
    "Status",
    "assemble",
    "in_graph_inputs",
    "in_graph_root_find",
    "in_graph_shape",
    "in_graph_solve",
    "mdf_graph",
    "mdf_shape",
    "nested_blocking",
    "prime",
    "report_place",
    "root_find_node",
    "seed",
    "traceable_drivers",
    "verdict",
]
