"""Run the `Optimise` layer's whole validation ladder against
`tests/regression/input_files/stellarator_helias.IN.DAT` and print the report.

    $PY functional_process/run_sand_harness.py

Sibling of `run_mda_harness.py`; see `sand_harness.py`'s module docstring for what each
of the three stages proves and, just as importantly, what it does not.

One PROCESS run (~95 s) serves all three stages: Stage A needs its converged
`DataStructure`, Stage B needs its live model objects to re-run the pipeline for the
finite-difference reference, and Stage C needs both its converged answer (to start C2 at)
and the input file's own cold start (to start C3 at).
"""

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from functional_process import sand  # noqa: E402
from functional_process.sand_harness import (  # noqa: E402
    assemble,
    ground_truth,
    mda_env,
    port_jacobian,
    process_jacobian_with_error,
    reduce_jacobian,
    reference_run,
    stage_a,
    to_process_spelling,
)
from process.core.solver.iteration_variables import ITERATION_VARIABLES  # noqa: E402


def _seed(schedule, drive, base, fallback):
    """Every schedule input and every block unknown, from `base`'s own fields where it
    has one and from `fallback` (a completed MDA env) otherwise.

    The fallback is not a convenience: the SAND block's unknowns include nine **coupling
    variables** that PROCESS never exposes as unknowns and therefore has no starting
    value for, plus a handful of minted `VarPath`s with no `DataStructure` field at all.
    A cold solve is cold in its *design* variables; its coupling guesses come from an MDA
    run, which is exactly what an MDF architecture would hand iteration 0 anyway.
    """
    env, borrowed = {}, []
    for var in list(schedule.inputs) + list(drive.unknowns):
        try:
            env[var] = jnp.asarray(ground_truth(base, var))
            continue
        except (AttributeError, KeyError):
            pass
        if var in fallback:
            env[var] = fallback[var]
            borrowed.append(var)
        else:
            env[var] = jnp.asarray(0.0)
    return env, tuple(borrowed)


def main():
    """Run the three stages and print each one's report."""
    reference = reference_run()
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

    driven, env = mda_env(reference)
    combined, report = assemble(reference, driven, env)
    print(
        f"\ndegenerate fixed points dropped: "
        f"{[d.path_str() for d in report['degenerate']]}"
    )
    print(f"residualised: {[r.path_str() for r in report['residualised']]}")
    if report["omitted"]:
        print(f"CONSTRAINTS OMITTED: {report['omitted']}")

    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    drive = shape["drive"]
    # Residual equalities only; PROCESS's own constraints keep scale 1.0 so the iterates
    # stay comparable. See `VmconDriver.condition_scale`.
    condition_scale = sand.residual_condition_scales(drive, env)
    print("SAND shape:", {k: v for k, v in shape.items() if k != "drive"})

    context = {
        v: (env[v] if v in env else jnp.asarray(ground_truth(reference.data, v)))
        for v in drive.context
    }
    condition_map = drive.condition_map(context)
    start = [
        jnp.asarray(env[v]) if v in env else jnp.asarray(ground_truth(reference.data, v))
        for v in drive.unknowns
    ]
    names = [c.path_str() for c in drive.conditions]

    # ---------------------------------------------------------------- A
    print()
    print(stage_a(reference, condition_map, names, start).summary())

    # ---------------------------------------------------------------- B
    full, compile_seconds, jit_ms = port_jacobian(condition_map, start)
    condition_rows = [
        i
        for i, name in enumerate(names)
        if name == "^cond.numerics.objf" or name.startswith("^cond.constraints.")
    ]
    residual_rows = [i for i in range(len(names)) if i not in condition_rows]
    design = list(range(len(reference.ixc)))
    coupling = list(range(len(reference.ixc), len(start)))
    reduced = reduce_jacobian(
        full,
        condition_rows,
        design,
        residual_rows,
        coupling,
        [float(np.asarray(start[k])) for k in coupling],
    )
    port_objective, port_constraints = to_process_spelling(reduced, reference.scale)
    process, process_error, fd_seconds = process_jacobian_with_error(reference)

    print(f"\nSTAGE B -- Jacobian, port {full.shape} against PROCESS's own")
    print(
        f"  port: compile {compile_seconds:.2f} s, jitted median {jit_ms:.3f} ms, "
        f"{int((~np.isfinite(full)).sum())} non-finite cells"
    )
    print(
        f"  PROCESS: {5 * len(reference.ixc)} pipeline sweeps for the same Jacobian "
        f"with Richardson error bars, {fd_seconds:.2f} s"
    )
    print("  |port - PROCESS| / |PROCESS|; '*' = outside the FD's own error bar x 4")
    print(f"{'':8s}" + "".join(f"{'x' + str(i):>12s}" for i in reference.ixc))

    def row(label, port, reference_row, error_row):
        def relative(k):
            return abs(port[k] - reference_row[k]) / max(abs(reference_row[k]), 1e-300)

        cells = "".join(
            f"{relative(k):10.2e} "
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
    for label, base, starts in (
        ("C2 (start at PROCESS's converged x)", reference.data, reference.converged),
        ("C3 (cold start from the IN.DAT values)", reference.cold, reference.initial),
    ):
        trace: list = []

        def record(i, result, _x, convergence, _trace=trace):
            _trace.append((
                i,
                float(convergence),
                float(np.asarray(result.f)),
                float(np.max(np.abs(result.eq))) if len(result.eq) else 0.0,
                float(np.min(result.ie)) if len(result.ie) else 0.0,
            ))

        solve_schedule = sand.sand_schedule(
            combined,
            None,
            bounds=reference.bounds,
            condition_scale=condition_scale,
            callback=record,
        )
        solve_drive = sand.sand_shape(solve_schedule)["drive"]
        seeded, borrowed = _seed(solve_schedule, solve_drive, base, env)
        started = time.perf_counter()
        out = solve_schedule(dict(seeded))
        elapsed = time.perf_counter() - started
        print(f"\nSTAGE C {label}: {len(trace)} SQP iterations in {elapsed:.1f} s")
        print(f"  ({len(borrowed)} unknown(s)/input(s) seeded from the MDA env)")
        print(
            f"  {'it':>3s} {'conv':>12s} {'objf':>14s} {'max|eq|':>11s} {'min ie':>12s}"
        )
        for entry in trace:
            print(
                f"  {entry[0]:3d} {entry[1]:12.3e} {entry[2]:14.9f} "
                f"{entry[3]:11.3e} {entry[4]:12.3e}"
            )
        print(
            f"\n  {'ixc':>5s} {'name':<36s} {'start':>16s} {'PROCESS':>18s} "
            f"{'port':>18s} {'rel':>10s}"
        )
        for i in reference.ixc:
            var = sand.iteration_variable_path(i)
            got = float(np.asarray(out[var]))
            expected = reference.converged[i]
            print(
                f"  {i:5d} {ITERATION_VARIABLES[i].name:<36s} "
                f"{starts[i]:16.8g} {expected:18.10g} {got:18.10g} "
                f"{abs(got - expected) / abs(expected):10.2e}"
            )


if __name__ == "__main__":
    main()
