"""Run the MDF formulation's validation ladder and print the report."""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from functional_process.cottax import mdf, sand  # noqa: E402
from functional_process.cottax.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.cottax.run_mda_harness import _resolve, input_file  # noqa: E402
from functional_process.cottax.sand_harness import (  # noqa: E402
    process_jacobian_with_error,
    reference_run,
    stage_a,
    to_process_spelling,
)
from process.core.solver.evaluators import Evaluators  # noqa: E402
from process.core.solver.iteration_variables import ITERATION_VARIABLES  # noqa: E402

MAX_ITER = 800
"""`VmconDriver.max_iter`."""

TOLERANCE = 1.0e-8
"""`VmconDriver.tolerance`."""


def process_evaluation_cost(reference, repeats=3):
    """Median seconds for one `Caller.call_models` -- PROCESS's own objective-and-
    constraint evaluation, and the thing an MDF condition evaluation is the analogue of.
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
    """Which of the three ways a `VmconDriver` solve can end actually happened."""
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
    """Run the ladder and print each stage's report."""
    argv = sys.argv[1:] if argv is None else argv
    path = input_file(argv)
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    print(f"input file:     {path}")

    # `use_cache=False`: `condition_map_cost` needs live `reference.models`, which a
    # cached run does not carry (see `sand_harness.reference_run`).
    reference = reference_run(str(path), use_cache=False)
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
