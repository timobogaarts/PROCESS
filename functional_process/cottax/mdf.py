"""The `Optimise` layer, assembled the other way round: **MDF** (Multidisciplinary
Feasible).
"""

import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import ConditionMap, Drive, Schedule
from cottax.graph import Graph
from cottax.plan import Insert, Plan
from cottax.problem import (
    Converged,
    DriverOut,
    Equality,
    FixedPoint,
    Inequality,
    Objective,
    Optimise,
    Residual,
    RootFind,
    Start,
    Steps,
)
from cottax.spec import In, NodePath, Out, VarPath
from cottax.tools.path import path_map
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey

from functional_process.cottax import sand
from functional_process.cottax.core.solver.drivers import (
    SeededNewtonDriver,
    # `Status` was written here, for `MdfNewtonDriver`, and moved to `drivers` when
    # `VmconDriver` and `SlsqpDriver` started reporting one too: a port kind belongs
    # beside the drivers that write it, not beside the one assembly that first read it.
    # Re-exported (`__all__`) so `mdf.Status` still resolves for every existing caller.
    Status,
    VmconDriver,
)
from functional_process.cottax.indat import graph_for
from functional_process.cottax.mda import (
    assign_drivers,
    cut_graph,
    default_drivers,
    given_start,
    guess_sources,
)
from functional_process.cottax.mda_harness import _without_excluded
from functional_process.cottax.sand_harness import ground_truth, run_schedule


@dataclasses.dataclass(frozen=True)
class Mdf:
    """One assembled MDF problem: the graph, the two schedules, and the problem's shape.
    """

    graph: Graph
    """The MDA graph plus one `ImplementedFunction` per active constraint and one for
    the objective.
    """
    eager: Schedule
    traceable: Schedule
    design: tuple[VarPath, ...]
    """The run's `ixc`, in PROCESS's own order."""
    conditions: tuple[VarPath, ...]
    """What the outer driver is handed."""
    n_equality: int
    n_inequality: int
    report: dict
    problem_type: type = Optimise
    """Which problem this file states -- `Optimise` or `RootFind`. See `assemble`."""
    reported: tuple[VarPath, ...] = ()
    """Conditions assembled but **not driven**: a `RootFind`'s inequalities."""


def mdf_graph(graph, icc, n_equality, i_figure_merit, switch_values=None, omit=()):
    """`graph` with `sand.constraint_nodes`' and `sand.objective_nodes`' nodes inserted.
    """
    nodes, equalities, inequalities, omitted = sand.constraint_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    objective = None
    if i_figure_merit is not None:
        from functional_process.cottax.indat import objective_selection  # noqa: PLC0415

        objective_built, objective = sand.objective_nodes(
            graph, objective_selection(i_figure_merit), switch_values
        )
        nodes.update(objective_built)
    inserted = (Plan(graph) + Insert(path_map(nodes.items()))).graph
    return (
        inserted,
        ((objective,) if objective is not None else ()) + (*equalities, *inequalities),
        len(inequalities),
        {
            "equalities": equalities,
            "inequalities": inequalities,
            "objective": objective,
            "omitted": omitted,
        },
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
):
    """The whole MDF assembly: cut the raw cycles, add the conditions, build both
    schedules.
    """
    if root_find and n_equality != len(ixc):
        raise ValueError(
            f"a root find needs one equality per iteration variable, and this file "
            f"states {n_equality} equality constraint(s) against {len(ixc)} iteration "
            f"variable(s) -- PROCESS's own `fsolve` over `evaluate_eq_cons` would be "
            f"the same non-square system, so there is nothing to root-find here"
        )
    driven = cut_graph(_without_excluded(graph if graph is not None else graph_for()))
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
    blocking = Blocking.scc(eager_graph)
    design = tuple(sand.iteration_variable_path(i) for i in ixc)
    eager = Schedule(blocking)
    missing = [d for d in design if d not in eager.inputs]
    if missing:
        raise ValueError(
            f"design variable(s) {[d.path_str() for d in missing]} are not boundary "
            f"inputs of the MDA graph -- a node already produces them, so the optimiser "
            f"cannot own them (see `sand.optimise_graph` on the same conflict)"
        )
    report["blocks"] = len(blocking.blocks)
    report["driven_blocks"] = sum(1 for t in blocking.problem_types if t is not None)
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
            Blocking.scc(assign_drivers(graph, traceable_drivers(drivers)))
        ),
        design=design,
        conditions=driven_conditions,
        n_equality=n_equality,
        n_inequality=0 if root_find else n_inequality,
        report=report,
        problem_type=RootFind if root_find else Optimise,
        reported=reported,
    )


def guess_ports(mdf: Mdf) -> dict:
    """`{guess_port: unknown}` over every problem in `mdf.graph`."""
    # Asked of the **eager schedule's** graph, not `mdf.graph`. Since `Assign` mints the
    # `^guess.*` ports from the driver's own `requires`, a graph with no drivers on it has
    # no start ports at all -- and `mdf.graph` is the cut graph plus the `Optimise`,
    # deliberately undriven so that `Combine` can still join its problems. Asking that one
    # returns nothing, every port falls to `ground_truth`'s `0.0`, and the inner solves
    # start from exactly the cold point `prime` exists to get them off.
    return guess_sources(mdf.eager.blocking.graph)


def seed(mdf: Mdf, data, design_values=None):
    """Every schedule input and every inner unknown, read off `data`."""
    env = {}
    starts = guess_ports(mdf)
    for var in list(mdf.eager.inputs) + list(mdf.eager.unknowns):
        # A `^guess.*` port is grounded from the unknown it starts, not from its own
        # name -- there is no `DataStructure` field spelled that way.
        source = starts.get(var, var)
        try:
            grounded = ground_truth(data, source)
        except (AttributeError, KeyError):
            grounded = 0.0
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


def restart(mdf: Mdf, out):
    """The env a next pass starts from: the run's own inputs, every `Start` port
    re-seeded from the unknown its driver converged.
    """
    env = {var: out[var] for var in mdf.eager.inputs if var in out}
    for guess, unknown in guess_ports(mdf).items():
        env[guess] = out[unknown]
    return env


class MdfConditionMap(ConditionMap):
    """`f(*design) -> conditions`, with a whole converged MDA inside every call."""

    schedule: Schedule

    def __call__(self, *design):
        """The conditions at `design`, with the MDA driven to convergence there."""
        if len(design) != len(self.unknowns):
            raise TypeError(
                f"MDF condition map takes {len(self.unknowns)} design variable(s) "
                f"({', '.join(v.path_str() for v in self.unknowns)}), got {len(design)}"
            )
        env = dict(self.context)
        env.update(zip(self.unknowns, design, strict=True))
        at = self.schedule.run(path_map(env))
        return tuple(at[condition] for condition in self.conditions)


def condition_map(mdf: Mdf, env, traceable=True) -> MdfConditionMap:
    """`f(*design) -> conditions` for `mdf`, everything else in `env` closed over."""
    # Restricted to the schedule's own inputs (minus the design, supplied per call):
    # the primed env is also a value store carrying the inner unknowns at their own
    # names, and a `Schedule` refuses a value at an owned name (`_inputs_only`).
    design = set(mdf.design)
    context = {
        var: value for var, value in _inputs_only(mdf, env).items() if var not in design
    }
    # `roles` is cottax's own answer to what this module worked around with
    # `VmconDriver.n_equality`/`n_inequality`: the condition map now carries what each
    # condition *is*, parallel to `conditions`, so the split travels on the driver seam
    # instead of beside it (`_audit/optimise_design.md` §8, closed upstream). MDF's
    # order is the one `mdf_graph` assembles -- objective, equalities, inequalities --
    # and it is spelled here rather than counted by anyone.
    if issubclass(mdf.problem_type, RootFind):
        # Every condition vanishes at the answer, and none of them is an objective or a
        # one-sided bound -- which is precisely `RootFind.condition_roles`.
        roles = (Residual,) * len(mdf.conditions)
    else:
        n_equality = len(mdf.conditions) - 1 - mdf.n_inequality
        roles = (
            (Objective,) + (Equality,) * n_equality + (Inequality,) * mdf.n_inequality
        )
    return MdfConditionMap(
        body=mdf.traceable.subgraph,
        unknowns=mdf.design,
        conditions=mdf.conditions,
        roles=roles,
        context=path_map(context.items()),
        schedule=mdf.traceable if traceable else mdf.eager,
    )


def driver(mdf: Mdf, bounds=(), callback=None, optimiser=VmconDriver, **kwargs):
    """The block's optimiser, its equality/inequality counts read off the assembly."""
    return optimiser(
        n_equality=mdf.n_equality,
        n_inequality=mdf.n_inequality,
        bounds=bounds,
        callback=callback,
        **kwargs,
    )


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
                f"({', '.join(v.path_str() for v in conditions.unknowns)})"
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


def root_find_driver(mdf: Mdf, **kwargs) -> MdfNewtonDriver:
    """The driver for a `RootFind` MDF -- `mdf.problem_type` decides, not the caller."""
    if not issubclass(mdf.problem_type, RootFind):
        raise TypeError(
            f"this MDF states an {mdf.problem_type.__name__}, not a RootFind -- "
            f"`mdf.driver` is the one to build"
        )
    return MdfNewtonDriver(**kwargs)


def solve(mdf: Mdf, env, bounds=(), callback=None, optimiser=None, **kwargs):
    """Drive the outer problem, then re-run the MDA at the answer."""
    conditions = condition_map(mdf, env)
    start = tuple(jnp.asarray(env[var]) for var in mdf.design)
    if optimiser is None and issubclass(mdf.problem_type, RootFind):
        # A `RootFind` takes neither an objective nor a bound nor a per-iterate callback:
        # `bounds` and `callback` are dropped here rather than forwarded, so that a
        # caller passing the `Optimise` arm's arguments gets PROCESS's unbounded
        # `fsolve` semantics and not a silently different problem. The step count comes
        # back through the driver's own reports, in the env this returns (`verdict`).
        optimiser = root_find_driver(mdf, **kwargs)
    if isinstance(optimiser, type):
        optimiser = driver(
            mdf, bounds=bounds, callback=callback, optimiser=optimiser, **kwargs
        )
    optimiser = optimiser or driver(mdf, bounds=bounds, callback=callback, **kwargs)
    started = time.perf_counter()
    # The driver is called directly here, not through a `Drive`, so the driver-data
    # mapping `Drive.role_data` would have built has to be built by hand: `Start` is
    # what `VmconDriver.requires` names, and the design values are what starts it.
    answered = optimiser(conditions, {Start: start})
    elapsed = time.perf_counter() - started
    # `Drive.__call__`'s own split, written out for the same reason the rest of this
    # function is: the driver returns its unknowns and then one value per kind in
    # `reports`, so the design is the first `len(mdf.design)` and the verdict is the
    # rest.
    x, verdict = answered[: len(mdf.design)], answered[len(mdf.design) :]
    at = dict(env)
    at.update(zip(mdf.design, x, strict=True))
    # Through `run_schedule`, not a bare `mdf.eager(...)`, for the reason `prime` a few
    # lines above already goes that way: a direct `Schedule.__call__` dispatches every
    # primitive eagerly and XLA compiles each one as its own module.
    # `_audit/optimise_design.md` §24.8 measured that tail at 268 compiles in an
    # MDF-only process and §24.9 left it alone, to avoid moving a second thing in
    # §24.1's pass. Measured here on a full `run_cold_matrix --native` row
    # (`stellarator_helias`, MDF then SAND, persistent cache OFF): **336 -> 277** XLA
    # compiles for the row, and this call itself **60 -> 1**. The gap to §24.8's 268 is
    # that a row runs SAND too, so jax's in-process cache already holds most of the
    # one-primitive programs by the time this line is reached -- inference from the
    # counts, not separately verified.
    # The row's result is unchanged to the table's precision (`objf 1.21775747` MDF /
    # `1.21775743` SAND, 108/169 iterations, identical `max|eq|`/`min ie`). NOT checked
    # bitwise: `sand_harness.mda_env`'s docstring records the jitted schedule differing
    # from the eager one at `1.1e-13` on 254 of 831 keys, so the reported env may move
    # in its last bits even where the table does not.
    out = run_schedule(mdf.eager, _inputs_only(mdf, at))
    out.update(
        zip(
            (kind.name_for(IN_GRAPH_PLACE) for kind in optimiser.reports),
            verdict,
            strict=True,
        )
    )
    return tuple(x), out, elapsed


def verdict(out, kind: type[DriverOut], place: NodePath = None):
    """What a driver said about its own run, out of the env a solve returned."""
    return out.get(kind.name_for(IN_GRAPH_PLACE if place is None else place))


def evaluation(conditions: MdfConditionMap, start, repeats=5):
    """`(values, compile seconds, jitted median milliseconds)` -- what one MDF condition
    evaluation costs.
    """

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(flat)
    began = time.perf_counter()
    values = compiled(*start)
    jax.block_until_ready(values)
    compile_seconds = time.perf_counter() - began
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        values = compiled(*start)
        jax.block_until_ready(values)
        timings.append(time.perf_counter() - began)
    return (
        np.asarray(values, dtype=float),
        compile_seconds,
        float(np.median(timings) * 1e3),
    )


def jacobian(conditions: MdfConditionMap, start, repeats=5):
    """`(J, compile seconds, jitted median milliseconds)` -- `d(conditions)/d(design)`.
    """

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(jax.jacfwd(flat, argnums=tuple(range(len(start)))))
    began = time.perf_counter()
    columns = compiled(*start)
    jax.block_until_ready(columns[0])
    compile_seconds = time.perf_counter() - began
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        columns = compiled(*start)
        jax.block_until_ready(columns[0])
        timings.append(time.perf_counter() - began)
    full = np.stack([np.asarray(c, dtype=float) for c in columns], axis=1)
    return full, compile_seconds, float(np.median(timings) * 1e3)


def central_difference(conditions: MdfConditionMap, start, relative_step=1e-5):
    """`d(conditions)/d(design)` by central differences **of this same map**."""

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(flat)
    columns = []
    for index, value in enumerate(start):
        step = relative_step * max(abs(float(np.asarray(value))), 1.0)
        forward = list(start)
        backward = list(start)
        forward[index] = jnp.asarray(float(np.asarray(value)) + step)
        backward[index] = jnp.asarray(float(np.asarray(value)) - step)
        high = np.asarray(compiled(*forward), dtype=float)
        low = np.asarray(compiled(*backward), dtype=float)
        columns.append((high - low) / (2.0 * step))
    return np.stack(columns, axis=1)


def inner_residuals(schedule: Schedule, env):
    """How well each driven block is actually solved at `env`, as `[(block, unknown,
    residual, relative)]`.
    """
    rows = []
    for step in schedule.steps:
        if not isinstance(step, Drive):
            continue
        values = step.condition_map(env)(*[env[u] for u in step.unknowns])
        fixed_point = issubclass(step.problem_type, FixedPoint)
        for unknown, value in zip(step.unknowns, values, strict=True):
            current = np.asarray(env[unknown], dtype=float)
            gap = np.asarray(value, dtype=float) - (current if fixed_point else 0.0)
            gap, current = np.broadcast_arrays(gap, current)
            if gap.size == 0:
                continue
            relative = np.abs(gap) / np.maximum(np.abs(current), 1e-30)
            worst = int(np.argmax(relative))
            rows.append((
                step.problem,
                unknown,
                float(gap.reshape(-1)[worst]),
                float(relative.reshape(-1)[worst]),
            ))
    return rows


def nested_blocking(ixc, icc, n_equality, i_figure_merit, graph=None, **kwargs):
    """MDF **stated as structure**: `Blocking.scc(graph + Optimise).nest(the Optimise)`.
    """
    driven = cut_graph(_without_excluded(graph if graph is not None else graph_for()))
    with_problem, problem_name, report = sand.optimise_graph(
        driven, ixc, icc, n_equality, i_figure_merit, **kwargs
    )
    return Blocking.scc(with_problem).nest(problem_name), problem_name, report


IN_GRAPH_PLACE = NodePath((GetAttrKey("RootFind"),))
"""Where the in-graph problem binds."""


@dataclasses.dataclass(frozen=True)
class InGraphRootFind:
    """PROCESS's evaluation-mode root find, **stated as a node of the graph**."""

    mdf: Mdf
    """The assembly this states -- a `RootFind` one (`assemble(root_find=True)`)."""
    graph: Graph
    """`mdf.graph` plus the `RootFind`, with every problem's driver `Assign`ed on."""
    blocking: Blocking
    """`Blocking.scc(graph).nest(problem)`: the SCC blocking, nested at the problem."""
    schedule: Schedule
    problem: NodePath

    def verdict(self, out, kind: type[DriverOut]):
        """What the outer driver said about its own run, out of a run's env."""
        return out.get(kind.name_for(self.problem))

    def steps(self, out) -> int | None:
        """How many Newton steps the outer solve took, from that run's env."""
        return self.verdict(out, Steps)

    def successful(self, out) -> bool | None:
        """Optimistix's own verdict on the outer solve, from that run's env."""
        return self.verdict(out, Converged)

    @property
    def index(self) -> int:
        """Which block of `blocking` the root find is answered at."""
        return self.blocking.index[self.problem]

    @property
    def drive(self) -> Drive:
        """The outer `Drive` step -- the surface a caller measures a residual on."""
        return self.schedule.steps[self.index]

    @property
    def block(self) -> tuple:
        """The nodes `Blocking.scc` put in the driven block, the problem included."""
        return self.blocking.blocks[self.index]

    @property
    def interior(self) -> Blocking:
        """How that block is blocked one level down -- the block minus the root find."""
        return self.blocking.inner[self.index]

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


def root_find_node(mdf: Mdf) -> RootFind:
    """The `RootFind` `mdf` states, as a cottax node: owns `design`, reads `conditions`.
    """
    if not issubclass(mdf.problem_type, RootFind):
        raise TypeError(
            f"this MDF states an {mdf.problem_type.__name__}, and only the `RootFind` "
            f"arm is stated in-graph here -- an `Optimise` nests just as well "
            f"(`_audit/in_graph_rootfind.md` §1 measures it), but its outer driver is a "
            f"`VmconDriver`, which does not trace, so that is a separate change"
        )
    return RootFind(
        inputs=tuple(In(c) for c in mdf.conditions),
        outputs=tuple(Out(v) for v in mdf.design),
    )


def in_graph_root_find(
    mdf: Mdf,
    place: NodePath = IN_GRAPH_PLACE,
    driver=None,
    traceable: bool = True,
    **kwargs,
) -> InGraphRootFind:
    """State `mdf`'s root find inside the graph and let `Blocking.scc` decide what it
    drives.
    """
    node = root_find_node(mdf)
    if place in mdf.graph.definitions:
        raise ValueError(
            f"{place!r} is already a node of this graph -- pass `place` to bind the "
            f"root find somewhere else"
        )
    with_problem = (Plan(mdf.graph) + Insert(path_map([(place, node)]))).graph
    drivers = default_drivers(with_problem)
    if traceable:
        drivers = traceable_drivers(drivers)
    # `default_drivers` already put a `SeededNewtonDriver` on the new `RootFind` -- it
    # dispatches on the problem type and cannot know this one is the outer solve. It is
    # replaced rather than skipped: `MdfNewtonDriver` reports optimistix's verdict
    # instead of raising it, which is what makes a non-converged outer solve a row.
    drivers[place] = driver or MdfNewtonDriver(**kwargs)
    assigned = assign_drivers(with_problem, drivers)
    blocking = Blocking.scc(assigned).nest(place)
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
            f"{[v.path_str() for v in missing]} -- `seed` then `prime` is what fills "
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
        out = dict(built.schedule.run(path_map(inputs)))
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
        "outer_blocks": len(built.blocking.blocks),
        "block": len(built.block),
        "body": len(built.drive.body.nodes),
        "interior_blocks": len(interior.blocks),
        "interior_driven": sum(1 for t in interior.problem_types if t is not None),
        "upstream": built.index,
        "downstream": len(built.blocking.blocks) - built.index - 1,
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
    "MdfConditionMap",
    "MdfNewtonDriver",
    "Status",
    "assemble",
    "central_difference",
    "condition_map",
    "driver",
    "evaluation",
    "in_graph_inputs",
    "in_graph_root_find",
    "in_graph_shape",
    "in_graph_solve",
    "inner_residuals",
    "jacobian",
    "mdf_graph",
    "mdf_shape",
    "nested_blocking",
    "prime",
    "root_find_driver",
    "root_find_node",
    "seed",
    "solve",
    "traceable_drivers",
    "verdict",
]
