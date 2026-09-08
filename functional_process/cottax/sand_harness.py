"""The `Optimise` layer's validation ladder: three stages, three different claims."""

import shutil
import tempfile
import time
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.plan import Delete
from cottax.tools.minting import unminted
from cottax.tools.path import path_map
from cottax.tools.pytree import get_at

from functional_process.cottax._harness.finite_difference import fd_gradient_with_error
from functional_process.cottax.mda import (
    assign_drivers,
    default_drivers,
    cut_graph,
)
from functional_process.cottax.mda_harness import KNOWN_MINT_VALUES, _without_excluded
from functional_process.cottax.sand import (
    array_valued_problems,
    constraints_outside_block,
    degenerate_fixed_points,
    iteration_variable_path,
    optimise_graph,
    sand_graph,
)

REFERENCE_INPUT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "tests/regression/input_files/stellarator_helias.IN.DAT"
)
STELLA_CONF = "stellarator_helias.stella_conf.json"


def _scratch_copy(input_file):
    """`input_file` copied into a fresh directory, with its `.stella_conf.json`."""
    directory = Path(tempfile.mkdtemp())
    shutil.copy(input_file, directory / Path(input_file).name)
    companion = Path(input_file).with_name(STELLA_CONF)
    if companion.exists():
        shutil.copy(companion, directory / STELLA_CONF)
    return str(directory / Path(input_file).name)


@dataclass
class ReferenceRun:
    """One converged PROCESS run, plus the problem it solved and its own cold start."""

    data: object
    models: object
    cold: object
    """The `DataStructure` after `init_process` and **before** any model has run."""
    ixc: list
    icc: list
    n_equality: int
    i_figure_merit: int
    epsfcn: float
    scale: np.ndarray
    xcm: np.ndarray
    converged: dict
    initial: dict
    bounds: tuple
    """`((VarPath, lower, upper), ...)`, from the run's own `numerics.boundl`/`boundu`
    -- the input file's overrides, not `ITERATION_VARIABLES`' table defaults.
    """
    solver_iterations: int
    convergence_parameter: float
    solve_seconds: float


_REFERENCE_CACHE_VERSION = "reference-v2"
"""Bumped when `ReferenceRun`'s *contents* change, so an old pickle can never be read
back under a key whose meaning has moved -- `mda_harness._CACHE_VERSION`'s discipline.
"""


def reference_run(input_file=None, *, use_cache: bool = True) -> ReferenceRun:
    """Run PROCESS in-process to convergence and capture everything the ladder needs."""
    import pickle  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from process.core.solver.iteration_variables import ITERATION_VARIABLES
    from process.main import SingleRun

    from functional_process.cottax.mda_harness import CACHE_DIR, _cache_key  # noqa: PLC0415

    input_file = input_file or REFERENCE_INPUT_FILE
    use_cache = use_cache and not os.environ.get("FP_HARNESS_NO_CACHE")
    cached = (
        Path(CACHE_DIR) / f"{_REFERENCE_CACHE_VERSION}-{_cache_key(input_file)}.pkl"
        if use_cache
        else None
    )
    if cached is not None and cached.exists():
        with cached.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 -- our own file, written just below

    cold = SingleRun(_scratch_copy(input_file), "vmcon").data
    run = SingleRun(_scratch_copy(input_file), "vmcon")
    started = time.perf_counter()
    run.run()
    elapsed = time.perf_counter() - started

    data = run.data
    n = int(data.numerics.n_iteration_variables)
    ixc = [int(i) for i in data.numerics.ixc[:n]]
    m = int(data.numerics.n_equality_constraints) + int(
        data.numerics.n_inequality_constraints
    )

    def value(structure, i):
        """One iteration variable's value, honouring `array_index`."""
        iteration_variable = ITERATION_VARIABLES[i]
        area = getattr(structure, iteration_variable.module)
        field = getattr(area, iteration_variable.target_name or iteration_variable.name)
        if iteration_variable.array_index is None:
            return float(field)
        return float(field[iteration_variable.array_index])

    result = ReferenceRun(
        data=data,
        models=run.models,
        cold=cold,
        ixc=ixc,
        icc=[int(i) for i in data.numerics.icc[:m]],
        n_equality=int(data.numerics.n_equality_constraints),
        i_figure_merit=int(data.numerics.i_figure_merit),
        epsfcn=float(data.numerics.epsfcn),
        scale=np.array(data.numerics.scale[:n], dtype=float),
        xcm=np.array(data.numerics.xcm[:n], dtype=float),
        converged={i: value(data, i) for i in ixc},
        initial={i: value(cold, i) for i in ixc},
        bounds=tuple(
            (
                iteration_variable_path(i),
                float(cold.numerics.boundl[i - 1]),
                float(cold.numerics.boundu[i - 1]),
            )
            for i in ixc
        ),
        solver_iterations=int(data.numerics.n_solver_iterations),
        convergence_parameter=float(data.globals.convergence_parameter),
        solve_seconds=elapsed,
    )
    if cached is not None:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(".partial")
        with partial.open("wb") as handle:
            pickle.dump(replace(result, models=None), handle)
        partial.replace(cached)
    return result


UNWRITTEN_BY_PROCESS = float("nan")
"""What `ground_truth` seeds for a field PROCESS itself leaves `None` -- see its
docstring.
"""


def ground_truth(data, var):
    """`data`'s own value at `var` -- `mda_harness._ground_truth`'s rule, restated here
    with two caveats that matter for this module and not for that one.
    """
    known = KNOWN_MINT_VALUES.get(var.path_str())
    if known is not None:
        return known(data)
    value = get_at(data, unminted(var).keys)
    return UNWRITTEN_BY_PROCESS if value is None else value


_MDA_SCHEDULES: dict = {}
"""`graph -> (driven, runnable, schedule, jitted runner)`, built once per graph."""


def mda_schedule(graph=None):
    """`(driven, runnable, schedule, run)` for `graph` -- the MDA, assembled once."""
    from functional_process.cottax.indat import graph_for  # noqa: PLC0415

    key = graph if graph is not None else graph_for()
    cached = _MDA_SCHEDULES.get(key)
    if cached is None:
        driven = cut_graph(_without_excluded(key))
        blocking = Blocking.scc(driven)
        runnable = assign_drivers(blocking.graph, default_drivers(blocking.graph))
        schedule = Schedule(Blocking.scc(runnable))
        cached = _MDA_SCHEDULES[key] = (
            driven,
            runnable,
            schedule,
            _mda_runner(schedule),
        )
    return cached


def _mda_runner(schedule):
    """`schedule` under `jit`, taking and returning a `PathMap` rather than a `dict`."""

    @eqx.filter_jit
    def run(values):
        return schedule.run(values)

    return run


_SCHEDULE_RUNNERS: dict = {}
"""`(Schedule, fuse_upstream) -> tuple[step-or-jitted-run, ...]`, built once per key."""


def run_schedule(schedule, env, whole=None, fuse_upstream=True):
    """`schedule(env)`, jitted whole where that is possible and part by part where not.
    """
    if stale := [var for var in env if var in schedule._owned]:
        raise ValueError(
            f"value(s) at {sorted(v.path_str() for v in stale)}, which this schedule's "
            f"own nodes produce -- hand the run its inputs and take results from the "
            f"env it returns"
        )
    if whole is False:
        # The caller already knows this schedule holds a host-side driver and does not
        # want the probe. Measured on `large_tokamak_nof`'s SAND solve schedule: the
        # failed attempt is one symbolic trace, no extra XLA compile, ~5 s (62.3 s
        # against 57.2 s). Worth paying once where the answer is unknown, worth skipping
        # where it is not.
        _SCHEDULE_WHOLE.setdefault(schedule, False)
    whole = _SCHEDULE_WHOLE.get(schedule)
    if whole is None:
        whole = _SCHEDULE_WHOLE[schedule] = _mda_runner(schedule)
        try:
            out = dict(whole(path_map(env)))
        except Exception as refusal:  # noqa: BLE001 -- the verdict, not an error
            # A driver that does not trace. Recorded rather than raised: the walk below
            # computes the same values from the same nodes in the same order, so this
            # is a choice of algorithm for one schedule and not a failure of the run.
            whole = _SCHEDULE_WHOLE[schedule] = False
            _SCHEDULE_VERDICT[schedule] = f"{type(refusal).__name__}: {refusal}"
        else:
            _SCHEDULE_VERDICT[schedule] = None
            return out
    if whole is not False:
        return dict(whole(path_map(env)))
    key = (schedule, bool(fuse_upstream))
    runners = _SCHEDULE_RUNNERS.get(key)
    if runners is None:
        runners = _SCHEDULE_RUNNERS[key] = _schedule_runners(
            schedule, fuse_upstream=fuse_upstream
        )
    for runner in runners:
        env = runner(env)
    return env


_SCHEDULE_WHOLE: dict = {}
"""`Schedule -> jitted whole-schedule runner, or `False` where one refused to trace."""

_SCHEDULE_VERDICT: dict = {}
"""`Schedule -> None if it jits whole, else the refusal that sent it to the walk."""


def schedule_verdict(schedule):
    """`(jitted_whole, reason)` for a schedule `run_schedule` has run."""
    return _SCHEDULE_WHOLE.get(schedule) is not False, _SCHEDULE_VERDICT.get(schedule)


def _schedule_runners(schedule, fuse_upstream=True):
    """`schedule.steps` as `Env -> Env` callables: every undriven group one jit, every
    driver eager.
    """
    from cottax.evaluate import Drive  # noqa: PLC0415, Schedule

    upstream_group = _jitted_group if fuse_upstream else _eager_group
    runners, group = [], []
    for step in schedule.steps:
        if isinstance(step, Drive):
            if group:
                runners.append(upstream_group(tuple(group)))
                group = []
            runners.append(_driven_runner(step, fuse_upstream=fuse_upstream))
        else:
            group.append(step)
    if group:
        # The trailing group: everything after the last `Drive`, so no driver reads it.
        # With no `Drive` at all there is nothing to protect and the fusion is free.
        runners.append(_jitted_group(tuple(group)))
    return tuple(runners)


def _eager_group(steps):
    """A run of undriven steps, unfused, as an `Env -> Env` callable."""

    def run(env):
        for step in steps:
            env = step._run(env)
        return env

    return run


def _jitted_group(steps):
    """One `jit` over a run of undriven steps, as an `Env -> Env` callable."""

    @eqx.filter_jit
    def jitted(values):
        env = dict(values)
        for step in steps:
            env = step._run(env)
        return path_map(env)

    def run(env):
        return dict(jitted(path_map(env)))

    return run


def _driven_runner(step, fuse_upstream=True):
    """`Drive.__call__`, with the body's re-run jitted and the driver left eager."""
    from cottax.evaluate import Schedule  # noqa: PLC0415

    body = (
        (lambda env: run_schedule(step.body, env, fuse_upstream=fuse_upstream))
        if isinstance(step.body, Schedule)
        else _jitted_group((step.body,))
    )

    def run(env):
        answered = step.driver(step.condition_map(env), step.role_data(env))
        # **`Drive.__call__`'s own contract, and this used to get it wrong.** A driver
        # returns one value per unknown *and then* one per kind in `driver.reports`
        # (`cottax.problem.AbstractDriver.__call__`), which `Drive` binds as
        # `self.unknowns + self.reports` (`cottax.evaluate.Drive.__call__`). This
        # re-implementation checked and bound `step.unknowns` alone, which was invisible
        # only because every driver reaching it reported nothing. The moment
        # `VmconDriver` gained `(Steps, Converged, Status)` it became a spurious
        # `ValueError`.
        #
        # **It could not have mis-bound a report to an unknown**, and that is worth
        # writing down because it is the failure one expects here and it is not the one
        # available: the reports come *after* the unknowns positionally, so even a
        # truncating `zip` would have paired every unknown with its own value and merely
        # dropped the verdict. The bug's whole reachable surface is the length check --
        # loud, which is why this was caught the first time a driver reported anything
        # rather than quietly answering the wrong question.
        bound = step.unknowns + step.reports
        if len(answered) != len(bound):
            reported = (
                f" and {len(step.reports)} report(s) "
                f"{[v.path_str() for v in step.reports]}"
                if step.reports
                else ""
            )
            raise ValueError(
                f"{type(step.driver).__name__} for block "
                f"{[n.path_str() for n in step.nodes]} returned {len(answered)} "
                f"value(s) for {len(step.unknowns)} unknown(s) "
                f"{[v.path_str() for v in step.unknowns]}{reported}"
            )
        env.update(zip(bound, answered, strict=True))
        return body(env)

    return run


def mda_env(reference, graph=None, data=None):
    """Run the plain MDA schedule seeded from `data` (default `reference.data`); return
    its output env.
    """
    from functional_process.cottax.mda import (  # noqa: PLC0415
        given_start,
        guess_sources,
    )

    data = reference.data if data is None else data
    driven, runnable, schedule, run = mda_schedule(graph)
    # Seeded over the **schedule's own inputs**, which is the driven graph's boundary:
    # `Assign` mints each driver's `^guess.*` `Start` ports into it and `Supply` takes
    # the supplied ones back out, so asking the schedule is what keeps this in step with
    # the algorithm assignment. The old shape here asked `starts_for` of the *undriven*
    # `driven`, which has no `Start` ports at all since the ports moved from
    # `Initialise` to `Assign` -- so no `^guess.*` was ever seeded and the first
    # `Drive.role_data` raised `KeyError` on its own start. A `^guess.*` port is
    # grounded from the unknown it starts (`guess_sources`); there is nothing in `data`
    # spelled `^guess.*`.
    #
    # `cottax.boundary.seeds` answers the same question a different way -- it maps a
    # `Start` port to `unminted(port)`, the place in the caller's structure, where
    # `guess_sources` maps it to the *unknown*, which for a `FixedPointCut` is the
    # minted copy `^hat.X`. On this graph the two disagree by name on three of five
    # starts and agree on every value, because `ground_truth` falls back to `unminted`
    # for a mint with no `KNOWN_MINT_VALUES` entry and none of the entries is spelled
    # `^hat.*`. Checked rather than assumed: `test_sand.py::test_boundary_seeds_agree`.
    guesses = guess_sources(runnable)
    env = {}
    for var in schedule.inputs:
        source = guesses.get(var, var)
        try:
            grounded = ground_truth(data, source)
        except (AttributeError, KeyError):
            grounded = 0.0
        # A `^guess.*` port may be *given* its value rather than read off `data` --
        # `mda.GIVEN_STARTS` for which, and why a cold dataclass default is not a
        # starting guess. Only guess ports: an ordinary input is the machine's own
        # number and the table has no standing over it.
        if var in guesses:
            grounded = given_start(source, grounded)
        env[var] = _strongly_typed(grounded)
    return driven, dict(run(path_map(env)))


def _strongly_typed(value):
    """`value` as a jax array whose `weak_type` is `False`, whatever it came in as."""
    array = jnp.asarray(value)
    return jax.lax.convert_element_type(array, array.dtype)


def assemble(reference, driven, env, omit=(), switch_values=None, keep=()):
    """The SAND graph for `reference`'s own `ixc`/`icc`/`i_figure_merit`."""
    keep = frozenset(keep)
    degenerate = tuple(p for p in degenerate_fixed_points(driven, env) if p not in keep)
    array_valued = tuple(
        p
        for p in array_valued_problems(
            driven, env, tuple(p for p in driven.declared if p not in set(degenerate))
        )
        if p not in keep
    )
    dropped = tuple(degenerate) + tuple(array_valued)
    graph = Delete(dropped).apply(driven) if dropped else driven

    def build(omit_now):
        with_problem, _name, report = optimise_graph(
            graph,
            reference.ixc,
            reference.icc,
            reference.n_equality,
            reference.i_figure_merit,
            switch_values=switch_values,
            omit=omit_now,
        )
        combined, residualised = sand_graph(with_problem, keep=keep)
        return combined, residualised, report

    combined, residualised, report = build(omit)
    # Second pass, only when the first left a constraint outside the problem's own
    # block (`sand.constraints_outside_block` -- a constraint that reads nothing the
    # block produces). Such a `^cond.*` never reaches the condition map, so those
    # constraints are re-assembled as explicit omissions and reported under
    # `report["external"]`; an equality among them would change the problem's very
    # feasibility and is refused instead of omitted. Empty on the stellarator, so its
    # single-pass path is bit-for-bit what it always was.
    external = constraints_outside_block(combined)
    if external:
        equalities = [
            cid for cid in reference.icc[: reference.n_equality] if cid in external
        ]
        if equalities:
            raise ValueError(
                f"equality constraint(s) {equalities} read nothing the SAND block "
                f"produces -- omitting an equality changes what 'feasible' means, so "
                f"this assembly is refused rather than reduced. The missing-producer "
                f"audit names what each read needs."
            )
        combined, residualised, report = build(tuple(omit) + tuple(external))
        for cid in external:
            report["omitted"][cid] = (
                "reads nothing the SAND block produces (every input is a boundary "
                "value or upstream of every unknown) -- constant over the design, "
                "outside the drive, unreachable by its condition map"
            )
    report["external"] = external
    report["degenerate"] = degenerate
    report["array_valued"] = array_valued
    report["residualised"] = residualised
    return combined, report


# ---------------------------------------------------------------- stage A


@dataclass
class StageA:
    """Per-condition comparison of the graph's `^cond.*` against PROCESS's own."""

    rows: list = field(default_factory=list)
    """`(name, port, process_or_None, rel_diff_or_None)`; `None` for a SAND residual,
    which PROCESS has no counterpart for.
    """

    def summary(self):
        lines = [
            "STAGE A -- conditions at PROCESS's converged point",
            f"{'condition':<48s} {'port':>18s} {'PROCESS':>18s} {'rel':>10s}",
        ]
        for name, port, process, rel in self.rows:
            if process is None:
                lines.append(f"{name:<48s} {port:+18.9e} {'(SAND residual)':>18s}")
            else:
                lines.append(f"{name:<48s} {port:+18.9e} {process:+18.9e} {rel:10.2e}")
        exact = sum(1 for _n, _p, q, r in self.rows if q is not None and r < 1e-9)
        total = sum(1 for _n, _p, q, _r in self.rows if q is not None)
        lines.append(f"exact (rel < 1e-9): {exact} of {total}")
        return "\n".join(lines)


def stage_a(reference, condition_map, condition_names, unknowns_start) -> StageA:
    """Evaluate every condition at PROCESS's converged point and diff."""
    from process.core.solver.constraints import constraint_eqns
    from process.core.solver.objectives import objective_function

    values = condition_map(*unknowns_start)
    total = len(reference.icc)
    cc, *_ = constraint_eqns(total, -1, reference.data)
    process = {cid: -float(cc[i]) for i, cid in enumerate(reference.icc)}
    objective = float(objective_function(reference.i_figure_merit, reference.data))

    report = StageA()
    for name, value in zip(condition_names, values, strict=True):
        got = float(np.asarray(value))
        if name.startswith("^cond.constraints.c"):
            expected = process[int(name.rsplit(".c", 1)[1])]
        elif name == "^cond.numerics.objf":
            expected = objective
        else:
            report.rows.append((name, got, None, None))
            continue
        denominator = abs(expected) or 1.0
        report.rows.append((name, got, expected, abs(got - expected) / denominator))
    return report


# ---------------------------------------------------------------- stage B


def reduce_jacobian(
    full, condition_index, design_index, residual_index, coupling_index, coupling_values
):
    """The SAND Jacobian's design-only block, Schur-complemented."""
    column = np.asarray(coupling_values, dtype=float)
    # Exact comparison: it is exactly zero, not a neighbourhood of it, that has no scale.
    column = np.where(column == 0.0, 1.0, np.abs(column))  # noqa: RUF069
    j_ry = full[np.ix_(residual_index, coupling_index)]
    row = np.abs(j_ry * column[None, :]).max(axis=1)
    row = np.where(row == 0.0, 1.0, row)  # noqa: RUF069
    scaled = (j_ry * column[None, :]) / row[:, None]
    return full[np.ix_(condition_index, design_index)] - (
        full[np.ix_(condition_index, coupling_index)] * column[None, :]
    ) @ np.linalg.solve(
        scaled, full[np.ix_(residual_index, design_index)] / row[:, None]
    )


def port_jacobian(condition_map, unknowns_start, repeats=10):
    """`(full Jacobian, compile seconds, jitted median milliseconds)`."""

    def flat(*unknowns):
        return jnp.stack([jnp.asarray(v) for v in condition_map(*unknowns)])

    jacobian = eqx.filter_jit(
        jax.jacfwd(flat, argnums=tuple(range(len(unknowns_start))))
    )
    started = time.perf_counter()
    columns = jacobian(*unknowns_start)
    compile_seconds = time.perf_counter() - started
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        columns = jacobian(*unknowns_start)
        jax.block_until_ready(columns[0])
        timings.append(time.perf_counter() - started)
    full = np.stack([np.asarray(c, dtype=float) for c in columns], axis=1)
    return full, compile_seconds, float(np.median(timings) * 1e3)


def process_jacobian_with_error(reference):
    """PROCESS's own gradients at its converged point, **with a per-cell Richardson
    error bar**, computed column by column.
    """
    from process.core.solver.evaluators import Evaluators

    evaluators = Evaluators(reference.models, reference.data, reference.xcm)
    n = len(reference.xcm)
    m = len(reference.icc)

    def at(index, value):
        probe = np.array(reference.xcm, dtype=float)
        probe[index] = value
        objective, constraints = evaluators.caller.call_models(probe, m)
        return np.concatenate([[objective], np.asarray(constraints, dtype=float)])

    started = time.perf_counter()
    columns, errors = [], []
    for index in range(n):
        derivative, error = fd_gradient_with_error(
            lambda value, _i=index: at(_i, value),
            float(reference.xcm[index]),
            reference.epsfcn,
        )
        columns.append(derivative)
        errors.append(error)
    seconds = time.perf_counter() - started
    # Leave `data` at the unperturbed point, exactly as `fcnvmc2` does on its way out.
    evaluators.caller.call_models(np.array(reference.xcm, dtype=float), m)
    return (
        np.stack(columns, axis=1),
        np.stack(errors, axis=1),
        seconds,
    )


def to_process_spelling(reduced, scale):
    """The reduced Jacobian in PROCESS's own coordinates."""
    return reduced[0] / scale, -reduced[1:] / scale[None, :]
