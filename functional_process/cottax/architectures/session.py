"""**Assemble once, solve many times** -- one configuration under every architecture.

A `Session` is one `configurations.Configuration` -- a machine, its values and its
problem, stated -- in the port's own solve environment (`native.reference_of`, nothing
from PROCESS), with one *arm* per architecture:

| arm    | recipe                                    | solved by                |
|--------|-------------------------------------------|--------------------------|
| `MDA`  | `mda.cut_graph` + `mda.default_drivers`   | the cut graph with the condition nodes, once |
| `MDF`  | `sand.problem_graph` + `MDF()`            | `evaluate.run_schedule`  |
| `IDF`  | `idf.idf_graph` (`IDF()`)                 | `evaluate.run_schedule`  |
| `SAND` | `sand.assemble` (`SAND()`)                | `evaluate.run_schedule`  |

A root-find configuration's `MDF` is `mdf.assemble(root_find=True)`'s in-graph root
find, solved by `mdf.in_graph_solve`.

Each arm is assembled on its first call (`Session.assemble`, which a caller may
make itself to take the build without solving) and only solved on the next. Every solve
starts **cold**, from the configuration's own values (`Session.reference.cold`),
unless handed another state. A configuration stating a root find (`Problem.root_find`)
has an `MDA` and an `MDF` arm only: PROCESS's own square system,
`mdf.assemble(root_find=True)`.

Every arm answers the same `dict`: the assembly's shape, the driver's `status`, its
`iterations`, `objf`, `max_eq`, `min_ie`, `seconds`, `x` (the design at the answer)
and a `note` where the status needs one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cottax.execution.driver import Gaps
from cottax.interfaces import Le, Plan
from cottax.mdao_architectures import MDF
from jax.flatten_util import ravel_pytree

from functional_process import configurations
from functional_process.cottax.architectures import idf, mdf, sand
from functional_process.cottax.architectures.drivers import (
    VMCON_NON_FINITE,
    Status,
    condition_places,
    non_finite_summary,
)
from functional_process.cottax.architectures.evaluate import (
    ground_truth,
    inputs_only,
    mda_env,
    resolve,
    run_schedule,
    seed_block,
)
from functional_process.cottax.architectures.mda import seed_starts
from functional_process.cottax.input import native
from functional_process.cottax.input.indat import configuration_from_indat, graph_for

ARMS = ("MDA", "MDF", "IDF", "SAND")

SAND_MAX_ITER = 500
"""SQP iterations the SAND and IDF arms allow themselves."""

SAND_TOLERANCE = None
"""`VmconDriver`'s own default."""


# ------------------------------------------------------------------ the result


def _blank(shape=None):
    """One arm's result with every measurement absent."""
    result = {
        "nodes": None,
        "design": None,
        "conditions": None,
        "equalities": None,
        "blocks": None,
        "driven": None,
        "iterations": None,
        "status": "",
        "objf": None,
        "max_eq": None,
        "min_ie": None,
        "seconds": None,
        "note": "",
        "x": (),
    }
    result.update(shape or {})
    return result


def recorder(trace):
    """The `VmconDriver.callback` every arm records its iterates with."""

    def record(i, result, _x, convergence):
        trace.append((
            i,
            float(convergence),
            float(np.asarray(result.f)),
            float(np.max(np.abs(result.eq))) if len(result.eq) else 0.0,
            float(np.min(result.ie)) if len(result.ie) else 0.0,
        ))

    return record


def trace_tail(trace):
    """`(iterations, objf, max|eq|, min ie)` off a callback trace."""
    if not trace:
        return 0, None, None, None
    last = trace[-1]
    return len(trace), last[2], last[3], last[4]


def _status(trace, tolerance, cap):
    """Which of the four ways a solve ended, in one word."""
    if not trace:
        return "no-step"
    epsilon = 1.0e-6 if tolerance is None else tolerance
    if trace[-1][1] <= epsilon:
        return "converged"
    if len(trace) >= cap:
        return f"cap({cap})"
    return "stopped"


def _why_no_step(drive, context, seeded):
    """The conditions that make a first QP infeasible: **violated and constant**.

    Read as **gaps**, the way the SQP answering this block reads them: the seam hands
    the two sides of each relation, and `Gaps` is the level that subtracts.
    """
    unknowns = [jnp.asarray(seeded[u]) for u in drive.unknowns]
    condition_map = Gaps(drive.condition_map(context))

    def stacked(*x):
        objectives, gaps = condition_map(*x)
        return jnp.stack([jnp.asarray(v) for v in (*objectives, *gaps)])

    values = np.asarray(stacked(*unknowns), dtype=float)
    rows = np.asarray(jax.jacfwd(stacked)(*unknowns), dtype=float).reshape(
        len(values), -1
    )
    n_objectives = len(condition_map.objectives)
    symbols = condition_map.symbols
    stuck = []
    for index, (place, value, row) in enumerate(
        zip(condition_places(condition_map), values, rows, strict=True)
    ):
        if index < n_objectives:
            continue  # the objective: never a feasibility question
        away = (
            value > 1e-8
            if symbols[index - n_objectives] is Le
            else abs(value) > 1e-8
        )
        if away and not np.any(row != 0.0):  # noqa: RUF069
            stuck.append((place.spelling, float(value)))
    return stuck


# ------------------------------------------------------------------ the arms


@dataclass
class MdfBuild:
    """Everything about the MDF arm that does not change between solves."""

    problem: object
    in_graph: object = None
    root_find: bool = False
    shape: dict = field(default_factory=dict)


@dataclass
class BlockBuild:
    """Everything about a SAND or IDF arm that does not change between solves: one
    combined problem, its solve schedule, and the trace its driver records into.
    """

    solve_schedule: object
    drive: object
    trace: list
    design_paths: set
    shape: dict = field(default_factory=dict)
    omitted: object = None
    scheme: object = None
    nested: bool = False
    """IDF: the disciplines' own problems are nested inside the block, and their
    `Start` ports are seeded from the MDA too."""


def build_mdf(reference, machine_graph, switch_values, root_find=False, scheme=None):
    """Assemble the MDF arm. `scheme=None` is `mda.SCHEME`."""
    problem = mdf.assemble(
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        graph=machine_graph,
        switch_values=switch_values,
        root_find=root_find,
        **({} if scheme is None else {"scheme": scheme}),
    )
    shape = mdf.mdf_shape(problem)
    build = MdfBuild(
        problem=problem,
        root_find=root_find,
        shape={
            "nodes": shape["nodes"],
            "design": shape["design"],
            "conditions": shape["conditions"],
            "equalities": shape["equalities"],
            "blocks": shape["inner_blocks"],
            "driven": shape["inner_driven"],
        },
    )
    if root_find:
        # Stated in the graph, not driven from outside it (`mdf.in_graph_root_find`):
        # the root find is a problem node the blocking sees, so `Blocking.scc` decides
        # what it drives, and the reported blocks are that interior's.
        build.in_graph = mdf.in_graph_root_find(problem)
        interior = mdf.in_graph_shape(build.in_graph)
        build.shape["blocks"] = interior["interior_blocks"]
        build.shape["driven"] = interior["interior_driven"]
    return build


def solve_mda(build: MdfBuild, cold) -> dict:
    """The MDA at the file's own design: the MDF's inner schedule run once from `cold`,
    which is also what primes the MDF arm. `objf`/`max_eq`/`min_ie` are the conditions
    at that design, not an optimiser's.
    """
    problem = build.problem
    result = _blank(build.shape)
    began = time.perf_counter()
    _primed, out = mdf.prime(problem, mdf.seed(problem, cold))
    elapsed = time.perf_counter() - began
    objective = problem.report["objective"]
    equalities = [float(np.asarray(out[c])) for c in problem.report["equalities"]]
    inequalities = [float(np.asarray(out[c])) for c in problem.report["inequalities"]]
    result.update(
        iterations=0,
        status="evaluated",
        objf=None if objective is None else float(np.asarray(out[objective])),
        max_eq=max(abs(r) for r in equalities) if equalities else 0.0,
        min_ie=min(inequalities) if inequalities else None,
        seconds=elapsed,
        x=tuple(float(np.asarray(out[v])) for v in problem.design),
    )
    return result


def solve_root_find(build: MdfBuild, cold) -> dict:
    """Solve a root-find configuration's `MDF` arm from `cold`: PROCESS's own square
    system, the root find stated in the graph (`mdf.in_graph_root_find`)."""
    problem = build.problem
    result = _blank(build.shape)
    env = mdf.seed(problem, cold)
    env, _primed = mdf.prime(problem, env)
    built = build.in_graph
    x, out, seconds = mdf.in_graph_solve(built, env)
    steps = int(np.asarray(built.steps(out)))
    converged = bool(np.asarray(built.successful(out)))
    residuals = [float(np.asarray(out[c])) for c in problem.conditions]
    # PROCESS's own last act in this mode: every inequality evaluated once at the
    # answer, none of them driven.
    inequalities = [float(np.asarray(out[c])) for c in problem.reported]
    result.update(
        iterations=steps,
        objf=None,
        max_eq=max(abs(r) for r in residuals) if residuals else 0.0,
        min_ie=min(inequalities) if inequalities else None,
        status="converged" if converged else "not-converged",
        seconds=seconds,
        note="" if converged else f"root find: {built.verdict(out, Status)}",
        x=tuple(float(np.asarray(v)) for v in x),
    )
    return result


def _block_build(combined, reference, env, optimiser, omitted, scheme, nested):
    """The solve schedule for one combined problem, its driver recording into `trace`."""
    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    condition_scale = sand.residual_condition_scales(shape["drive"], env)
    trace: list = []
    solve_schedule = sand.sand_schedule(
        combined,
        None,
        bounds=reference.bounds,
        condition_scale=condition_scale,
        callback=recorder(trace),
        max_iter=SAND_MAX_ITER,
        optimiser=optimiser,
    )
    return BlockBuild(
        solve_schedule=solve_schedule,
        drive=sand.sand_shape(solve_schedule)["drive"],
        trace=trace,
        design_paths={sand.iteration_variable_path(i) for i in reference.ixc},
        shape={
            "nodes": shape["drive_nodes"],
            "design": shape["design"],
            "conditions": shape["conditions"],
            "equalities": shape["equalities"],
            "blocks": shape["schedule_steps"],
            "driven": shape["unknowns"],
        },
        omitted=omitted,
        scheme=scheme,
        nested=nested,
    )


def build_sand(reference, machine_graph, switch_values, optimiser=None, scheme=None):
    """Assemble the SAND arm. `scheme=None` is `mda.SCHEME`."""
    driven, env = mda_env(
        reference, graph=machine_graph, **({} if scheme is None else {"scheme": scheme})
    )
    combined, report = sand.assemble(
        reference, driven, env, switch_values=switch_values, drop_arrays=False
    )
    return _block_build(
        combined, reference, env, optimiser, report["omitted"], scheme, nested=False
    )


def build_mdf_block(reference, machine_graph, switch_values, optimiser=None, scheme=None):
    """Assemble the MDF arm **in the graph**: the cut MDA, the constraint and
    objective nodes and the `Optimise`, then `cottax.mdao_architectures.MDF` -- every
    consistency statement nested in the optimiser, every model's own solve inside
    the consistency statement on its cycle. Solved as IDF and SAND are, by
    `solve_block`, so the three arms differ in the architecture and nothing else.
    """
    driven, env = mda_env(
        reference, graph=machine_graph, **({} if scheme is None else {"scheme": scheme})
    )
    with_problem, _optimiser, report = sand.problem_graph(
        driven,
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        switch_values=switch_values,
    )
    nested = (Plan(with_problem) + MDF()).graph
    return _block_build(
        nested, reference, env, optimiser, report["omitted"], scheme, nested=True
    )


def build_idf(reference, machine_graph, switch_values, optimiser=None, scheme=None):
    """Assemble the IDF arm. `scheme=None` is `mda.SCHEME`."""
    _driven, env = mda_env(
        reference, graph=machine_graph, **({} if scheme is None else {"scheme": scheme})
    )
    combined, _problem, report = idf.idf_graph(
        machine_graph if machine_graph is not None else graph_for(),
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        switch_values=switch_values,
        **({} if scheme is None else {"scheme": scheme}),
    )
    return _block_build(
        combined, reference, env, optimiser, report["omitted"], scheme, nested=True
    )


def solve_block(build: BlockBuild, reference, machine_graph, cold) -> dict:
    """Solve an assembled SAND or IDF arm from `cold`: design variables from the file,
    coupling copies from an MDA at that design.
    """
    result = _blank(build.shape)
    solve_schedule, solve_drive = build.solve_schedule, build.drive
    trace, design_paths = build.trace, build.design_paths
    trace.clear()

    stage_env = mda_env(
        reference,
        graph=machine_graph,
        data=cold,
        **({} if build.scheme is None else {"scheme": build.scheme}),
    )[1]
    seeded, _borrowed = seed_block(
        solve_schedule, solve_drive, cold, stage_env, design=design_paths
    )
    if build.nested:
        # The nested problems' own `Start` ports: not the outer drive's unknowns, so
        # `seed_block` leaves them cold, and a Picard from a cold zero does not converge.
        seeded.update(seed_starts(solve_schedule, stage_env, exclude=design_paths))

    context = {}
    for var in solve_drive.context:
        if var in stage_env:
            context[var] = stage_env[var]
        else:
            try:
                context[var] = jnp.asarray(ground_truth(cold, var))
            except (AttributeError, KeyError):
                context[var] = jnp.asarray(0.0)

    # Compiled whole where it traces -- the host SQP is a `jax.pure_callback`, so the
    # steps before and after it compile around it -- and walked step by step where it
    # does not (`run_schedule` records the verdict per schedule). `whole=False` walked
    # every solve schedule: the upstream steps, a host-side Newton and the trailing
    # steps each a dispatch from Python, ~30 ms of a helias_5b warm solve.
    started = time.perf_counter()
    out = run_schedule(solve_schedule, inputs_only(solve_schedule, seeded))
    elapsed = time.perf_counter() - started

    # The driver refuses a non-finite problem and reports `VMCON_NON_FINITE` through
    # its own `Status` port -- read as data out of the env, never caught.
    reported = mdf.verdict(out, Status, solve_drive)
    if reported is not None and int(np.asarray(reported)) == VMCON_NON_FINITE:
        flat_probe, probe_unravel = ravel_pytree(
            tuple(jnp.asarray(seeded[u]) for u in solve_drive.unknowns)
        )
        summary = non_finite_summary(
            Gaps(solve_drive.condition_map(context)), probe_unravel, flat_probe
        )
        result.update(
            status="non-finite",
            seconds=elapsed,
            note=f"refused at the first iterate -- {summary}"
            if summary
            else "refused at the first iterate",
        )
        return result

    iterations, objf, max_eq, min_ie = trace_tail(trace)
    note = ""
    if not trace:
        # Zero iterations is ambiguous between "converged where it stood" and "the
        # first QP had no feasible point"; a violated condition with an identically
        # zero gradient row tells the two apart.
        stuck = _why_no_step(solve_drive, context, seeded)
        note = (
            f"first QP infeasible: {len(stuck)} condition(s) violated with an "
            f"identically zero gradient row, first {stuck[0][0]} at {stuck[0][1]:+.2e}"
            if stuck
            else "no condition is both violated and constant -- converged where it stood"
        )
    result.update(
        iterations=iterations,
        objf=objf,
        max_eq=max_eq,
        min_ie=min_ie,
        status=_status(trace, SAND_TOLERANCE, SAND_MAX_ITER),
        seconds=elapsed,
        note=note,
        x=tuple(
            float(np.asarray(out[sand.iteration_variable_path(i)]))
            for i in reference.ixc
        ),
    )
    return result


# --------------------------------------------------------------- the session


@dataclass
class Session:
    """One configuration, assembled once per arm, solvable any number of times."""

    name: str
    configuration: configurations.Configuration
    reference: object
    machine_graph: object = None
    switch_values: object = None
    root_find: bool = False
    optimiser: object = None
    """The driver **class** every `Optimise` in this session is answered by, or `None`
    for `mda.default_drivers`' own choice.
    """
    scheme: object = None
    """How the raw graph's cycles are opened: `None` for `mda.SCHEME`, or any
    `cottax.mdao_architectures.Scheme` (`Jacobi`, `GaussSeidel`, `GaussSeidelMinimal`).
    """
    builds: dict = field(default_factory=dict)

    @property
    def arms(self) -> tuple[str, ...]:
        """The architectures this file can be solved under."""
        return ("MDA", "MDF") if self.root_find else ARMS

    def _check_arm(self, arm: str) -> None:
        if arm not in self.arms:
            raise ValueError(
                f"{self.name} has no {arm} arm: "
                + (
                    "it states a root find (`i_process_run_mode = -2`), which poses no "
                    "optimisation to distribute over design and coupling"
                    if arm in ARMS
                    else f"the arms are {ARMS}"
                )
            )

    def assemble(self, arm: str):
        """The assembled `arm`, nothing solved: the `MdfBuild` for `MDA` (the cut
        MDA with the condition nodes, run once -- and the root-find `MDF` of a
        configuration stating one), or the `BlockBuild` for `MDF`, `IDF` and `SAND` --
        one optimiser in the graph, the architecture around it. Built on the first
        call and kept in `builds`.

        Raises
        ------
        ValueError
            If `arm` is not one of `Session.arms`.
        """
        self._check_arm(arm)
        key = "MDA" if arm == "MDA" or self.root_find else arm
        build = self.builds.get(key)
        if build is None:
            if key == "MDA":
                build = build_mdf(
                    self.reference,
                    self.machine_graph,
                    self.switch_values,
                    root_find=self.root_find,
                    scheme=self.scheme,
                )
            else:
                builder = {"MDF": build_mdf_block, "IDF": build_idf, "SAND": build_sand}[key]
                build = builder(
                    self.reference,
                    self.machine_graph,
                    self.switch_values,
                    optimiser=self.optimiser,
                    scheme=self.scheme,
                )
            self.builds[key] = build
        return build

    def solve(self, arm: str, cold=None) -> dict:
        """Solve this configuration under `arm`, assembling it (`assemble`) on the
        first call.

        Raises
        ------
        ValueError
            If `arm` is not one of `Session.arms`.
        """
        build = self.assemble(arm)
        cold = self.reference.cold if cold is None else cold
        if arm == "MDA":
            return solve_mda(build, cold)
        if self.root_find:
            return solve_root_find(build, cold)
        return solve_block(build, self.reference, self.machine_graph, cold)

    def mda(self, cold=None) -> dict:
        """The MDA at the file's own design -- see `solve_mda`."""
        return self.solve("MDA", cold)

    def mdf(self, cold=None) -> dict:
        """Solve the MDF arm."""
        return self.solve("MDF", cold)

    def idf(self, cold=None) -> dict:
        """Solve the IDF arm."""
        return self.solve("IDF", cold)

    def sand(self, cold=None) -> dict:
        """Solve the SAND arm."""
        return self.solve("SAND", cold)


def open_session(configuration, optimiser=None, scheme=None) -> Session:
    """A `Session` for one configuration -- a `configurations.Configuration`, a name
    from `configurations.NAMES`, or an `IN.DAT` path (converted on the way in) --
    assembled from it alone, nothing solved.
    """
    if isinstance(configuration, (str, Path)):
        name = str(configuration)
        configuration = (
            configurations.load(name)
            if name in configurations.NAMES
            else configuration_from_indat(str(resolve(name)))
        )
    return Session(
        name=configuration.name,
        configuration=configuration,
        reference=native.reference_of(configuration),
        machine_graph=graph_for(configuration.machine),
        switch_values=dict(configuration.problem.switches),
        root_find=configuration.problem.root_find,
        optimiser=optimiser,
        scheme=scheme,
    )
