"""Run the MDF formulation's validation ladder and print the report.

    $PY functional_process/run_mdf_harness.py                   # the stellarator
    $PY functional_process/run_mdf_harness.py --input <IN.DAT>  # any other machine
    $PY functional_process/run_mdf_harness.py --machine         # the tokamak

With no argument this is exactly what it always was: the Helias stellarator,
`tests/regression/input_files/stellarator_helias.IN.DAT`, whose numbers other records
quote. `--input`/`--machine` are spelled exactly as `run_sand_harness.py`'s and
`run_mda_harness.py`'s (whose `input_file` this imports rather than restates).

A non-reference machine differs from the stellarator path in exactly two threaded values,
both derived from its own file and both already parameters of `mdf.assemble`: the graph
(`graph_for(machine_from_indat(...))`) and the static switch values
(`sand.switch_values_for`, read off the cold initialised `DataStructure` instead of
`mdf_graph`'s own default). The reference file keeps `None` for both, which is the
literal code path it has always run -- its numbers are pinned regression evidence
(`_audit/next_steps.md` §16.1) and must not move because a second device exists.

Sibling of `run_sand_harness.py`, deliberately stage for stage: **A** the conditions at
PROCESS's converged point, **B** the Jacobian against PROCESS's own finite differences,
**C** the solve from two starts. Everything PROCESS-side is reused from
`sand_harness.py` (`reference_run`, `stage_a`, `process_jacobian_with_error`,
`to_process_spelling`) so that any difference between the two reports is a difference
between the two *formulations* and not between two harnesses.

Three things this ladder has that SAND's does not, all of them consequences of the
formulation rather than extra diligence:

- **No `reduce_jacobian`.** MDF's Jacobian is already `d(conditions)/d(design)` with the
  MDA converged at every point -- the derivative `Evaluators.fcnvmc2` approximates -- so
  the Schur complement SAND needs in order to be comparable at all does not arise.
- **Stage B0, the check that matters most.** `jax.jacfwd` differentiates *through*
  thirteen driven blocks (`lax.while_loop`s and an `optimistix` root find). Before
  comparing it with PROCESS, it is compared with a **central difference of the port's own
  condition map**, which shares every model and every solver with it and differs only in
  not using autodiff. That isolates "is differentiating through the inner solve correct"
  from "does the port agree with PROCESS".
- **Inner convergence is reported.** An MDF answer is only as good as its inner solve,
  and `PicardDriver` stops at `max_iter` silently. `mdf.inner_residuals` says how far
  each driven block actually got at the point being reported.

One PROCESS run (~95 s) serves all of it, same as `run_sand_harness.py`.
"""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from functional_process import mdf, sand  # noqa: E402
from functional_process.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.run_mda_harness import _resolve, input_file  # noqa: E402
from functional_process.sand_harness import (  # noqa: E402
    process_jacobian_with_error,
    reference_run,
    stage_a,
    to_process_spelling,
)
from process.core.solver.evaluators import Evaluators  # noqa: E402
from process.core.solver.iteration_variables import ITERATION_VARIABLES  # noqa: E402

MAX_ITER = 800
"""`VmconDriver.max_iter`. Higher than `run_sand_harness.py`'s `SAND_MAX_ITER = 500` for
the same reason that one is higher than the driver's own default of 100: **measured**,
not rounded. C2 -- the warm solve, from PROCESS's own converged x -- converges at
**523** SQP iterations (`conv 7.400e-09`, `objf 1.217758052`), and the previous 200
stopped it two-thirds of the way through, where an unconverged VMCON tail is
indistinguishable from oscillation (`_audit/optimise_design.md` §14, and §12.2 for the
identical diagnosis on SAND). 800 is 523 with the same ~1.5x margin SAND's 500 gives its
326. **C3 is not capped and never was**: it stops at 60 because `pyvmcon` raises
`QSPSolverException` there -- raising the cap does not move it, and `_measure` below says
which of the two happened.

**523 and 60 are OSQP numbers.** The driver now passes PROCESS's own CLARABEL
(`_audit/optimise_design.md` §15): C3 cold converges in 67 at `1e-8` and 58 at
PROCESS's `epsvmc`, and the `QSPSolverException` at 60 is gone. C2 warm inverts --
CLARABEL stops at 45 on an `infeasible` QP where OSQP ground through to 523 -- so the
cap is still needed and still means what this says."""

TOLERANCE = 1.0e-8
"""`VmconDriver.tolerance`. **Tighter than PROCESS's own** `epsvmc = 1e-6` on this run,
and kept there so the two Stage C solves are comparable with `run_sand_harness.py`'s,
which uses the driver's own default. PROCESS stops at `2.396e-07`, so a solve reported
here as "not converged" may still be inside PROCESS's own criterion -- the trace's
convergence column is what to read, not the boolean alone."""


def process_evaluation_cost(reference, repeats=3):
    """Median seconds for one `Caller.call_models` -- PROCESS's own objective-and-
    constraint evaluation, and the thing an MDF condition evaluation is the analogue of.

    This is the like-for-like cost comparison the whole rewrite is about: one call here
    is up to ten full pipeline passes (`caller.py:96-126`), one call of an
    `MdfConditionMap` is one compiled program that drives thirteen blocks. Measured, not
    inferred from the solve's wall clock, which also carries the QP subproblems.
    """
    evaluators = Evaluators(reference.models, reference.data, reference.xcm)
    m = len(reference.icc)
    x = np.array(reference.xcm, dtype=float)
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        evaluators.caller.call_models(x, m)
        timings.append(time.perf_counter() - began)
    return float(np.median(timings))


def _why_it_stopped(converged, iterations):
    """Which of the three ways a `VmconDriver` solve can end actually happened.

    "Not converged" is two entirely different events and the boolean cannot tell them
    apart: **the cap** (`iterations == MAX_ITER`, a solve stopped part-way, which is what
    C2's 200 was) and **the driver giving up** (`pyvmcon` raising -- for C3 here, a
    `QSPSolverException`: the QP subproblem itself became infeasible -- which
    `VmconDriver.__call__` catches by design, keeping `e.x` and reporting the failure out
    of band). Raising `MAX_ITER` fixes the first and cannot touch the second, so the
    report has to say which one it is or the next reader re-runs this investigation
    (`_audit/optimise_design.md` §14).
    """
    if converged:
        return "the convergence test passed"
    if iterations >= MAX_ITER:
        return f"it reached the cap of {MAX_ITER} -- raise MAX_ITER and re-measure"
    return (
        f"the driver stopped short at {iterations} of {MAX_ITER}, i.e. `pyvmcon` raised "
        f"and `VmconDriver` kept the point -- a cap raise will NOT help"
    )


def _measure(mdf_problem, data, label, bounds, tolerance):
    """One Stage C solve: seed from `data`, prime the MDA, drive the optimiser."""
    env = mdf.seed(mdf_problem, data)
    began = time.perf_counter()
    env, primed = mdf.prime(mdf_problem, env)
    prime_seconds = time.perf_counter() - began

    trace: list = []

    def record(i, result, _x, convergence, _trace=trace):
        _trace.append((
            i,
            float(convergence),
            float(np.asarray(result.f)),
            float(np.max(np.abs(result.eq))) if len(result.eq) else 0.0,
            float(np.min(result.ie)) if len(result.ie) else 0.0,
        ))

    x, out, seconds = mdf.solve(
        mdf_problem,
        env,
        bounds=bounds,
        callback=record,
        tolerance=tolerance,
        max_iter=MAX_ITER,
    )
    print(
        f"\nSTAGE C {label}: {len(trace)} SQP iterations in {seconds:.1f} s "
        f"(+{prime_seconds:.1f} s priming the MDA)"
    )
    print(f"  {'it':>3s} {'conv':>12s} {'objf':>14s} {'max|eq|':>11s} {'min ie':>12s}")
    for entry in trace[:3] + ([("...",)] if len(trace) > 8 else []) + trace[-5:]:
        if len(entry) == 1:
            print("  ...")
            continue
        print(
            f"  {entry[0]:3d} {entry[1]:12.3e} {entry[2]:14.9f} "
            f"{entry[3]:11.3e} {entry[4]:12.3e}"
        )
    converged = bool(trace and trace[-1][1] <= tolerance)
    print(
        f"  converged: {converged} (tolerance {tolerance:.0e}); "
        f"stopped because {_why_it_stopped(converged, len(trace))}; "
        f"{seconds / max(len(trace), 1) * 1e3:.0f} ms per SQP iteration"
    )
    worst = sorted(mdf.inner_residuals(mdf_problem.eager, out), key=lambda r: -r[3])[:3]
    print("  worst inner-solve residuals at the answer:")
    for _problem, unknown, residual, relative in worst:
        print(f"    {unknown.path_str():<48s} {residual:+12.3e}  rel {relative:8.2e}")
    return x, trace, primed


def main(argv=None):
    """Run the ladder and print each stage's report. `argv` defaults to this process's
    own, so importing and calling `main([...])` is the same thing as the command line.
    """
    argv = sys.argv[1:] if argv is None else argv
    path = input_file(argv)
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    print(f"input file:     {path}")

    reference = reference_run(str(path))
    print(
        f"PROCESS: {reference.solver_iterations} VMCON iterations in "
        f"{reference.solve_seconds:.1f} s, convergence parameter "
        f"{reference.convergence_parameter:.3e}"
    )
    print(
        f"  ixc {reference.ixc}\n  icc {reference.icc} "
        f"(first {reference.n_equality} are equalities)\n"
        f"  i_figure_merit {reference.i_figure_merit}, epsfcn {reference.epsfcn}"
    )

    # The two per-machine values -- see the module docstring. `None` for the reference
    # keeps that path byte-identical to what it always was.
    machine_graph = None if is_reference else graph_for(machine_from_indat(str(path)))
    switch_values = (
        None
        if is_reference
        else sand.switch_values_for(
            reference.cold, reference.icc, reference.i_figure_merit
        )
    )
    if switch_values is not None:
        print(f"  switch values (from the file's own cold init): {switch_values}")

    problem = mdf.assemble(
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        graph=machine_graph,
        switch_values=switch_values,
    )
    print("\nMDF shape:", mdf.mdf_shape(problem))
    if problem.report["omitted"]:
        print(f"CONSTRAINTS OMITTED: {problem.report['omitted']}")

    nested, name, _ = mdf.nested_blocking(
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        graph=machine_graph,
        switch_values=switch_values,
    )
    index = nested.index[name]
    print(
        f"  stated as a nesting: {name.path_str()} answers a block of "
        f"{len(nested.blocks[index])} nodes whose interior is "
        f"{len(nested.inner[index].blocks)} blocks "
        f"({sum(1 for t in nested.inner[index].problem_types if t is not None)} driven) "
        f"-- `Blocking.nest` states it, `schedule_for` cannot run it (see `mdf.py`)"
    )

    # ---------------------------------------------------------------- A
    env = mdf.seed(problem, reference.data)
    env, primed = mdf.prime(problem, env)
    conditions = mdf.condition_map(problem, env)
    start = tuple(jnp.asarray(env[v]) for v in problem.design)
    names = [c.path_str() for c in problem.conditions]
    print()
    print(stage_a(reference, conditions, names, start).summary())
    worst = sorted(mdf.inner_residuals(problem.eager, primed), key=lambda r: -r[3])[:3]
    print("  worst inner-solve residual at PROCESS's converged point:")
    for _problem, unknown, residual, relative in worst:
        print(f"    {unknown.path_str():<48s} {residual:+12.3e}  rel {relative:8.2e}")

    # ---------------------------------------------------------------- cost
    _values, evaluate_compile, evaluate_ms = mdf.evaluation(conditions, start)
    full, compile_seconds, jit_ms = mdf.jacobian(conditions, start)
    call_models_seconds = process_evaluation_cost(reference)
    print(
        f"\nCOST per outer iteration -- one condition evaluation and one Jacobian\n"
        f"  MDF   : evaluate {evaluate_ms:8.3f} ms (compile {evaluate_compile:.1f} s), "
        f"jacfwd {jit_ms:8.3f} ms (compile {compile_seconds:.1f} s)\n"
        f"  PROCESS: call_models {call_models_seconds * 1e3:8.1f} ms, "
        f"fcnvmc2 ~{call_models_seconds * (2 * len(reference.ixc) + 1) * 1e3:8.1f} ms "
        f"({2 * len(reference.ixc) + 1} sweeps)\n"
        f"  ratio  : evaluation {call_models_seconds * 1e3 / evaluate_ms:.0f}x, "
        f"Jacobian "
        f"{call_models_seconds * (2 * len(reference.ixc) + 1) * 1e3 / jit_ms:.0f}x"
    )

    # ---------------------------------------------------------------- B
    began = time.perf_counter()
    finite = mdf.central_difference(conditions, start)
    fd_seconds = time.perf_counter() - began
    scale = np.maximum(np.abs(finite), 1e-30)
    relative = np.abs(full - finite) / scale
    alive = np.abs(finite) > 1e-8 * np.abs(finite).max()
    print(
        f"\nSTAGE B0 -- autodiff through the inner solve against a central difference "
        f"of the same map\n"
        f"  port jacfwd {full.shape}: compile {compile_seconds:.2f} s, jitted median "
        f"{jit_ms:.3f} ms, {int((~np.isfinite(full)).sum())} non-finite cells\n"
        f"  central difference of the same map: {fd_seconds:.1f} s\n"
        f"  worst relative disagreement over the {int(alive.sum())} cells the FD "
        f"resolves: {float(relative[alive].max()):.2e}\n"
        f"  cells above 1e-5: "
        f"{int((relative[alive] > 1e-5).sum())} of {int(alive.sum())}"
    )

    process, process_error, process_seconds = process_jacobian_with_error(reference)
    port_objective, port_constraints = to_process_spelling(full, reference.scale)
    print(
        f"\nSTAGE B -- the same Jacobian against PROCESS's own finite differences\n"
        f"  PROCESS: {5 * len(reference.ixc)} pipeline sweeps with Richardson error "
        f"bars, {process_seconds:.2f} s\n"
        f"  no Schur reduction: both sides are d(conditions)/d(design) with the MDA "
        f"converged\n"
        f"  |port - PROCESS| / |PROCESS|; '*' = outside the FD's own error bar x 4"
    )
    print(f"{'':8s}" + "".join(f"{'x' + str(i):>12s}" for i in reference.ixc))

    def row(label, port, reference_row, error_row):
        def cell(k):
            return abs(port[k] - reference_row[k]) / max(abs(reference_row[k]), 1e-300)

        cells = "".join(
            f"{cell(k):10.2e} "
            + (
                "*"
                if abs(port[k] - reference_row[k]) > 4 * max(error_row[k], 1e-300)
                else " "
            )
            for k in range(len(port))
        )
        print(f"{label:8s}{cells}")

    row("objf", port_objective, process[0], process_error[0])
    for j, cid in enumerate(reference.icc):
        row(f"c{cid}", port_constraints[j], process[1:][j], process_error[1:][j])

    # ---------------------------------------------------------------- C
    for label, data in (
        ("C2 (start at PROCESS's converged x)", reference.data),
        ("C3 (cold start from the IN.DAT values)", reference.cold),
    ):
        x, _trace, _primed = _measure(
            problem, data, label, reference.bounds, tolerance=TOLERANCE
        )
        print(
            f"\n  {'ixc':>5s} {'name':<36s} {'PROCESS':>18s} {'port':>18s} {'rel':>10s}"
        )
        for i, value in zip(reference.ixc, x, strict=True):
            got = float(np.asarray(value))
            expected = reference.converged[i]
            print(
                f"  {i:5d} {ITERATION_VARIABLES[i].name:<36s} "
                f"{expected:18.10g} {got:18.10g} "
                f"{abs(got - expected) / abs(expected):10.2e}"
            )


if __name__ == "__main__":
    main()
