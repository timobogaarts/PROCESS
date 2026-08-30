"""The `Optimise` layer's validation ladder: three stages, three different claims.

`mda_harness.py` asks *"does the graph reproduce a converged PROCESS run's values?"*.
This module asks the three questions that stack on top of it, and keeps them separate
because they establish genuinely different things (`_audit/optimise_design.md` §5):

- **Stage A -- conditions at the reference point.** Seed the design variables with
  PROCESS's converged values, run the MDA schedule, evaluate every `^cond.*` the
  assembled `Optimise` reads, and diff against PROCESS's own `constraint_eqns` output at
  the same point. *Proves:* the constraint layer's **values** are right when fed a real,
  self-consistent state **through the graph** -- one step beyond
  `mda_constraint_harness.py`, which reads arguments straight off a `DataStructure` and
  therefore cannot catch a mis-wired `In`. *Proves nothing about the optimiser.*

- **Stage B -- the Jacobian.** Compare the port's `jax.jacfwd` against PROCESS's own
  finite differences, **cell by cell**, with a per-cell Richardson error bar from
  `_harness/finite_difference.py` rather than a fixed `rtol`. One subtlety decides
  whether the comparison means anything: **a SAND Jacobian and PROCESS's are not the
  same derivative.** The SAND `ConditionMap` holds the coupling variables fixed as
  unknowns; PROCESS re-converges its Gauss-Seidel loop at every perturbed point, so its
  `cnorm` is a *total* derivative. They are related by one linear solve -- a Schur
  complement -- which `reduce_jacobian` applies. *Proves:* the port's model **and its
  exact derivatives** agree with PROCESS's model and its finite differences, or names
  the cells where they do not. This is the stage that finds defects no value comparison
  can: a quantity can be right everywhere and still have the wrong sensitivity, which is
  exactly what an unwired `In` produces.

- **Stage C -- the solve.** Run `VmconDriver` and see where it lands. C2 starts *at*
  PROCESS's own answer and asks whether the driver stays there; C3 starts from the input
  file's own values and asks where it goes.

What "matches PROCESS" can and cannot mean here
-----------------------------------------------
PROCESS's converged `x` is a point where **PROCESS's finite-difference-approximated** KKT
conditions hold to **its own** tolerance, using a **1 % relative** perturbation
(`data.numerics.epsfcn = 0.01` on the reference run) over a pipeline itself iterated only
to `check_agreement`'s `rtol = 1e-6` in at most 10 Gauss-Seidel passes
(`process/core/caller.py:96-126`). It is not the exact optimum of the stated problem.
*"The port's optimiser lands on PROCESS's `x`"* and *"the port's optimiser solves the
stated problem"* are therefore different claims and can differ by more than either
tolerance. Every report below says which one it is making.

One further caveat that this harness discovered rather than assumed, and that anyone
reading a Stage A/B/C number must know: **PROCESS's converged `DataStructure` is not
internally self-consistent.** `Stellarator.run(output=True)` re-runs `st_build`/`st_coil`
in the opposite order to the solve pass, so `.build.z_tf_inside_half` ends at `7.359`
where the solver used `4.156`; `Buildings.run` then recomputes
`.buildings.a_plant_floor_effective` from it (563075 -> 680433) while
`.heat_transport.p_plant_electric_base_total_mw` keeps the solve-pass number. The port
models the reported (output-pass) arm, so it is self-consistent and PROCESS's stored
value is not -- which shows up as one 17.604 MW offset through the whole AC-power and
cost chain. See `_audit/optimise_design.md` §5.1.
"""

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import schedule_for
from cottax.plan import Delete
from cottax.tools.minting import unminted
from cottax.tools.pytree import get_at

from functional_process._harness.finite_difference import fd_gradient_with_error
from functional_process.mda import (
    assign_drivers,
    default_drivers,
    cut_graph,
)
from functional_process.mda_harness import KNOWN_MINT_VALUES, _without_excluded
from functional_process.sand import (
    array_valued_problems,
    constraints_outside_block,
    degenerate_fixed_points,
    iteration_variable_path,
    optimise_graph,
    sand_graph,
)

REFERENCE_INPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "tests/regression/input_files/stellarator_helias.IN.DAT"
)
STELLA_CONF = "stellarator_helias.stella_conf.json"


def _scratch_copy(input_file):
    """`input_file` copied into a fresh directory, with its `.stella_conf.json`.

    Not a `TemporaryDirectory` context manager: PROCESS re-reads the stellarator preset
    file on **every** model call (`preset_config.load_stellarator_config`, reached from
    `Stellarator.st_new_config`), so the directory must outlive the `with` block or the
    second call raises `FileNotFoundError`. Found the hard way.
    """
    directory = Path(tempfile.mkdtemp())
    shutil.copy(input_file, directory / Path(input_file).name)
    companion = Path(input_file).with_name(STELLA_CONF)
    if companion.exists():
        shutil.copy(companion, directory / STELLA_CONF)
    return str(directory / Path(input_file).name)


@dataclass
class ReferenceRun:
    """One converged PROCESS run, plus the problem it solved and its own cold start.

    Kept as one object because every stage needs several of these together and running
    PROCESS costs ~95 s. `models` is retained so `Evaluators` can be rebuilt for Stage B:
    PROCESS's finite-difference Jacobian is the reference there and it needs the live
    model objects, not just `data`.
    """

    data: object
    models: object
    cold: object
    """The `DataStructure` after `init_process` and **before** any model has run.
    `SingleRun.__init__` already performs that init, so an un-run instance *is* the input
    file's own starting state -- there is no way to recover it from the solved `data`,
    because `set_scaled_iteration_variable` overwrites every iteration variable in place
    on the first model call."""
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
    """`((VarPath, lower, upper), ...)`, from the run's own `numerics.boundl`/`boundu` --
    the input file's overrides, not `ITERATION_VARIABLES`' table defaults."""
    solver_iterations: int
    convergence_parameter: float
    solve_seconds: float


def reference_run(input_file=None) -> ReferenceRun:
    """Run PROCESS in-process to convergence and capture everything the ladder needs."""
    from process.core.solver.iteration_variables import ITERATION_VARIABLES
    from process.main import SingleRun

    input_file = input_file or REFERENCE_INPUT_FILE
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
        """One iteration variable's value, honouring `array_index`.

        An `IterationVariable` may address a single *element* of an array field --
        `target_name` names the array, `array_index` the slot (ID 125/126 are the
        standing case, `f_nd_impurity_electron_array[2]`/`[3]` under the display name
        `f_nd_impurity_electrons(03)`/`(04)`). This used to `float()` whatever `getattr`
        returned, which is the whole array for such a variable, and every
        configuration that declares one -- `large_tokamak_nof` and
        `low_aspect_ratio_DEMO` among the reference files -- failed here with
        `TypeError: only 0-dimensional arrays can be converted to Python scalars`
        before any harness stage could run.
        """
        iteration_variable = ITERATION_VARIABLES[i]
        area = getattr(structure, iteration_variable.module)
        field = getattr(area, iteration_variable.target_name or iteration_variable.name)
        if iteration_variable.array_index is None:
            return float(field)
        return float(field[iteration_variable.array_index])

    return ReferenceRun(
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


UNWRITTEN_BY_PROCESS = float("nan")
"""What `ground_truth` seeds for a field PROCESS itself leaves `None` -- see its
docstring. `nan` rather than `0.0` deliberately: it is the value that cannot be read
without saying so."""


def ground_truth(data, var):
    """`data`'s own value at `var` -- `mda_harness._ground_truth`'s rule, restated here
    with two caveats that matter for this module and not for that one.

    The `unminted` fallback maps `^cond.X -> X`. That is correct for a
    `FixedPointFunction`'s condition (at the fixed point `^cond.X == X`) and **wrong for
    a `RootFind`'s**, whose `^cond.X` is a residual that should be ~0. Nothing seeded
    here is a `RootFind` residual today; a future one would silently get the wrong seed.

    **A field PROCESS never writes reads back as `None`, and seeds as `nan`.** The
    standing case is `.tfcoil.sig_tf_cs_bucked`: `stresscl` assigns it only at
    `i_tf_bucking >= 2` (`process/models/tfcoil/base.py:3235`), so on
    `large_tokamak_eval` (`i_tf_bucking = 1`) it is `None` in PROCESS's own *converged*
    `DataStructure`, and `jnp.asarray(None)` is what stopped the tokamak SAND harness
    at the point where it builds a `Drive`'s context. It reaches the context at all
    because `sand._bind` declares an `In` for every non-switch parameter of a
    constraint, whether or not the statically selected arm consumes it -- c72 takes
    `max(stress_shear_cs_peak, sig_tf_cs_bucked)` only on the bucked-and-wedged arm --
    so this is a **dead read**, not a missing producer
    (`_audit/units/models/tfcoil/superconducting.md` § "`sig_tf_cs_bucked` is a dead
    read, not a gap"; `_audit/optimise_design.md` §11.5 records the `None` itself).

    `nan` is the seed *because* the deadness is the claim being made. A `0.0` would let
    a read that is not actually dead produce a plausible number and be believed; `nan`
    propagates through arithmetic into the condition, and the pre-solve probe already
    stops on a non-finite condition and names it. The claim is therefore re-checked on
    every run rather than asserted once here.

    **The alarm used to be mute through `max`/`min`, and is not any more.** `nan > x`
    is False, so Python's builtin `max(x, nan)` returns `x` and discards the sentinel;
    c72's bucked arm was exactly such a builtin `max`, and `test_sand.py` pinned that
    hole as a measurement rather than an assumption. It was fixed on 2026-08-30 for an
    unrelated and more urgent reason -- the builtin calls `bool()` on `b > a`, so it
    raises `TracerBoolConversionError` under `jit` and made the tokamak MDF problem
    untraceable. `jnp.maximum` fixes the tracing and propagates the sentinel in the
    same stroke, so the seed is now loud on both of c72's arms.

    Fixing the declaration instead -- having `_bind` drop the reads the bound arm
    cannot reach -- is the structural repair, and it is deliberately *not* done here:
    it is measurable (12 dead declared reads on the tokamak, 6 on the stellarator, of
    which four are produced inside the drive) and so it can move the omit set and the
    pinned Stage C numbers. `_audit/next_steps.md` §16 carries the measurement.
    """
    known = KNOWN_MINT_VALUES.get(var.path_str())
    if known is not None:
        return known(data)
    value = get_at(data, unminted(var).keys)
    return UNWRITTEN_BY_PROCESS if value is None else value


def mda_env(reference, graph=None, data=None):
    """Run the plain MDA schedule seeded from `data` (default `reference.data`); return
    its output env.

    `data` exists so a **cold** env can be built the same way as the warm one: pass
    `reference.cold` and every coupling variable comes back at the cold design instead
    of at PROCESS's converged one. That is what `run_sand_harness._seed` hands a cold
    SAND solve, and it is the difference between that solve taking 0 steps and 85.

    Stage A's `Drive` needs a value for every one of its ~340 context variables, and some
    have no `DataStructure` field at all -- a scalar `0.0` placeholder for an
    array-valued one crashes downstream in `plasma_profiles._simpson`. Seeding the SAND
    block from a **completed MDA run's own output env** rather than from the
    `DataStructure` removes that whole class of hole: everything the graph produces is
    grounded by the graph.
    """
    from functional_process.indat import graph_for
    from functional_process.mda import guess_sources

    data = reference.data if data is None else data
    driven = cut_graph(_without_excluded(graph if graph is not None else graph_for()))
    blocking = Blocking.scc(driven)
    runnable = assign_drivers(blocking.graph, default_drivers(blocking.graph))
    schedule = schedule_for(Blocking.scc(runnable))
    # Seeded over the **schedule's own inputs**, which is the driven graph's boundary:
    # `Assign` mints each driver's `^guess.*` `Start` ports into it and `Supply` takes
    # the supplied ones back out, so asking the schedule is what keeps this in step with
    # the algorithm assignment. The old shape here asked `starts_for` of the *undriven*
    # `driven`, which has no `Start` ports at all since the ports moved from
    # `Initialise` to `Assign` -- so no `^guess.*` was ever seeded and the first
    # `Drive.role_data` raised `KeyError` on its own start. A `^guess.*` port is
    # grounded from the unknown it starts (`guess_sources`); there is nothing in `data`
    # spelled `^guess.*`.
    guesses = guess_sources(runnable)
    env = {}
    for var in schedule.inputs:
        source = guesses.get(var, var)
        try:
            env[var] = jnp.asarray(ground_truth(data, source))
        except (AttributeError, KeyError):
            env[var] = jnp.asarray(0.0)
    return driven, schedule(dict(env))


def assemble(reference, driven, env, omit=(), switch_values=None):
    """The SAND graph for `reference`'s own `ixc`/`icc`/`i_figure_merit`.

    `env` is used only to detect the structurally degenerate fixed points, whose problem
    nodes are then dropped: an identity `FixedPointFunction` is a perfectly well-posed
    Picard problem and a rank-deficient SAND equality, and any SQP fails on it. Dropping
    the problem reverts its unknown to an ordinary boundary input, which is the
    structurally honest statement of "nothing here determines this".

    `switch_values` is passed through to `optimise_graph` untouched: `None` keeps
    `sand.REFERENCE_SWITCH_VALUES` (the stellarator reference run, exactly as before);
    any other machine passes `sand.switch_values_for(...)`'s answer for its own file.

    **A `FixedPoint` owning a non-scalar unknown is dropped the same way a degenerate
    one is**, and reported as `report["array_valued"]`: the SAND layer's per-condition
    machinery is scalar (`sand.array_valued_problems` names every seam), so such a
    problem cannot be residualised into the combined `Optimise` today. Deleting it
    freezes its loop-carried unknowns at the env's own (converged-MDA) values -- a
    *reduction of the problem* a reader of any C2/C3 number for that machine must
    know about, which is why it travels in the report rather than in a log line. The
    stellarator has none, so its path is untouched.
    """
    degenerate = degenerate_fixed_points(driven, env)
    array_valued = array_valued_problems(
        driven, env, tuple(p for p in driven.declared if p not in set(degenerate))
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
        combined, residualised = sand_graph(with_problem)
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
    which PROCESS has no counterpart for."""

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
    """Evaluate every condition at PROCESS's converged point and diff.

    PROCESS's numbers come from `constraint_eqns`, whose `cc` entries are
    `-normalised_residual` (`process/core/solver/constraints.py:2007`), so the comparison
    negates them back. The objective comes from `objective_function`, which already
    carries `np.sign(i_figure_merit)` -- the same sign `sand.objective_node` folds into
    the node's `fn`.
    """
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
    """The SAND Jacobian's design-only block, Schur-complemented.

    Partition into design columns `D`, coupling columns `Y`, residual rows `R` and
    condition rows `C`. Holding the residuals at zero,

        dC/dD |total  =  J_CD  -  J_CY (J_RY)^-1 J_RD

    which is the total derivative PROCESS's `fcnvmc2` measures by re-converging its
    Gauss-Seidel loop at every perturbed point. Without this reduction the two Jacobians
    are simply different derivatives and any agreement would be a coincidence.

    **The solve is equilibrated, and it has to be.** `J_RY`'s raw singular values on the
    reference run span `4.9e14` down to `2.1e-15` -- a condition number of `2.4e29`,
    which would make the reduction numerically meaningless. That is **entirely units**,
    not rank deficiency: scaling each coupling column by its own value and each row by
    its largest entry brings the condition number to **12.1**. Since
    `J_RY^-1 = C A^-1 R^-1` exactly, for `A = R^-1 J_RY C` with `R` and `C` diagonal and
    invertible, the identity used here

        J_CY J_RY^-1 J_RD  =  (J_CY C) A^-1 (R^-1 J_RD)

    is algebra, not an approximation -- the same Jacobian, computed in coordinates where
    a `float64` solve means something.

    Parameters
    ----------
    coupling_values :
        The coupling unknowns' own values at the linearisation point, in
        `coupling_index` order. The column scale; an entry that is exactly zero falls
        back to `1.0`.
    """
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
    """`(full Jacobian, compile seconds, jitted median milliseconds)`.

    Timed under `eqx.filter_jit`, not raw. A bare `jax.jacfwd(...)(*u)` called once is
    trace **plus** compile **plus** one execution -- an earlier pass in this project
    reported 5.9 s from exactly that and concluded there was no speed win, which was an
    artifact of the measurement. Compilation is paid once per shape and amortises over a
    whole solve, so the number that matters is the steady state.
    """

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

    `Evaluators.fcnvmc2` is not reused directly because it returns one derivative and no
    error estimate. `_harness/finite_difference.fd_gradient_with_error` reproduces
    PROCESS's exact scheme (relative step `x*(1+/-epsfcn)`, realised denominator) and
    adds the extrapolation `(4/3)|D(h) - D(h/2)|` plus a round-off floor -- the same
    convention every tier-1 gradient test in this project already uses, rather than a
    fixed `rtol` that at `epsfcn = 0.01` would be a coin flip on the function's
    curvature.

    Returns
    -------
    :
        `(gradient, error, seconds)`, both arrays shaped `(1 + m, n)` with the objective
        as row 0 -- **rows are conditions**, unlike `fcnvmc2`'s `cnorm`, which is
        `(n, m)`.
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
    """The reduced Jacobian in PROCESS's own coordinates.

    Two conversions, both exact and both necessary. PROCESS differentiates with respect
    to the **scaled** variable `x * scale` (`iteration_variables.py:348-352`), so every
    column gains a `1/scale`; and its constraint vector is `cc = -normalised_residual`
    (`constraints.py:2007`), so every constraint row gains a minus. The objective row
    gets only the scaling -- `objective_function` already carries the sign.
    """
    return reduced[0] / scale, -reduced[1:] / scale[None, :]
