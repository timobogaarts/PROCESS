"""Run the `Optimise` layer's whole validation ladder and print the report."""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from cottax.problem import Driven  # noqa: E402
from cottax.tools.minting import unminted  # noqa: E402

from functional_process.cottax import sand  # noqa: E402
from functional_process.cottax.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.cottax.mda import guess_sources  # noqa: E402
from functional_process.cottax.run_mda_harness import _resolve, input_file  # noqa: E402
from functional_process.cottax.sand_harness import (  # noqa: E402
    assemble,
    ground_truth,
    mda_env,
    port_jacobian,
    process_jacobian_with_error,
    reduce_jacobian,
    reference_run,
    run_schedule,
    stage_a,
    to_process_spelling,
)
from process.core.solver.iteration_variables import ITERATION_VARIABLES  # noqa: E402

SAND_MAX_ITER = 500
"""SQP iterations Stage C allows itself, against `VmconDriver`'s own default of 100."""


def _why_no_step(drive, context, seeded):
    """The conditions that make a first QP infeasible: **violated and constant**."""
    unknowns = [jnp.asarray(seeded[u]) for u in drive.unknowns]
    condition_map = drive.condition_map(context)

    def stacked(*x):
        return jnp.stack([jnp.asarray(v) for v in condition_map(*x)])

    values = np.asarray(stacked(*unknowns), dtype=float)
    rows = np.asarray(jax.jacfwd(stacked)(*unknowns), dtype=float).reshape(
        len(values), -1
    )
    # `Driven` forwards `inputs`/`outputs` and nothing else; the split lives on the
    # problem it *has*. Same unwrap `sand.sand_shape` does.
    node = drive.subgraph[drive.problem]
    definition = node.problem if isinstance(node, Driven) else node
    n_equality = len(definition.equalities)
    # **By position, not by membership.** `definition.equalities` and `drive.conditions`
    # name the same nine equalities and compare equal to none of them -- measured, 0 of
    # 30 -- so a `condition in equality` test silently classified every condition as the
    # objective and reported that nothing was stuck. `VmconDriver.n_equality`'s own
    # docstring is the authority for doing it this way instead: `Drive.conditions` is the
    # problem node's `reads` and `Optimise.inputs` is `(objective, *equalities,
    # *inequalities)`, so counts recover the split, which is exactly how the driver
    # itself slices `values` at `[0]`, `[1 : 1 + meq]`, `[1 + meq :]`.
    stuck = []
    for index, (condition, value, row) in enumerate(
        zip(drive.conditions, values, rows, strict=True)
    ):
        if index == 0:
            continue  # the objective: never a feasibility question
        # An equality is away from satisfaction on either side of zero; an inequality
        # only above it -- cottax's sign convention is `g <= 0`, so a positive residual
        # is the violated one.
        away = abs(value) > 1e-8 if index <= n_equality else value > 1e-8
        # Exact comparison is the point: a row that is *identically* zero is a
        # condition no step can move, which is a structural fact and not a
        # tolerance question. A merely small row is a badly conditioned
        # constraint and the QP can still use it.
        if away and not np.any(row != 0.0):  # noqa: RUF069
            stuck.append((condition.path_str(), float(value)))
    return stuck


def _seed(schedule, drive, base, fallback, design=()):
    """Every schedule input and every block unknown: **design** variables from `base`,
    every other unknown from `fallback` (a completed MDA env at the same design).
    """
    design = set(design)
    env, borrowed = {}, []  # the env doubles as a value store; see `_inputs_only`
    # A `^guess.*` input is a *starting value for* an unknown, so every question below
    # -- is it coupling, is it in `fallback`, what does `base` say -- is asked about the
    # unknown it starts, never about the port's own name. `fallback` is an MDA output
    # env, keyed by real paths, and no `DataStructure` field is spelled `^guess.*`.
    guesses = guess_sources(schedule.blocking.graph)
    for var in list(schedule.inputs) + list(drive.unknowns):
        source = guesses.get(var, var)
        # A **cut** (`^hat.*`) is coupling by the same argument as an unknown, and it is
        # not an unknown: `mda.CUTS` opens each SCC by minting a copy of one loop-carried
        # variable, and that copy becomes a schedule *input*. Without this clause it fell
        # to `ground_truth(base, ...)`, which `unminted`s it to the real field and reads
        # the cold `DataStructure`'s dataclass default -- so the cold tokamak solve was
        # handed `n_pf_coil_turns = 0`, `ind_pf_cs_plasma_mutual = 0` and
        # `t_plant_pulse_burn = 1000` (`times_variables.py`'s default) while a completed
        # cold MDA env beside it held 3814.9, 132.7 and 144099. It is the same disease
        # this docstring already diagnoses for unknowns, one mint further out.
        cut = unminted(source) != source and source in fallback
        coupling = cut or (source in drive.unknowns and source not in design)
        if coupling and source in fallback:
            env[var] = fallback[source]
            borrowed.append(source)
            continue
        try:
            env[var] = jnp.asarray(ground_truth(base, source))
            continue
        except (AttributeError, KeyError):
            pass
        if source in fallback:
            env[var] = fallback[source]
            borrowed.append(source)
        else:
            env[var] = jnp.asarray(0.0)
    return env, tuple(borrowed)


def _inputs_only(schedule, env):
    """`env` restricted to what the schedule may be handed: its own inputs."""
    inputs = set(schedule.inputs)
    return {var: value for var, value in env.items() if var in inputs}


def stages(argv: list[str]) -> str:
    """Which of the three stages this invocation runs -- `--stages ABC` by default."""
    if "--stages" not in argv:
        return "ABC"
    index = argv.index("--stages") + 1
    if index >= len(argv) or argv[index].startswith("-"):
        raise SystemExit("--stages needs a subset of ABC, e.g. --stages A")
    asked = argv[index].upper()
    unknown = set(asked) - set("ABC")
    if unknown:
        raise SystemExit(f"--stages: {''.join(sorted(unknown))} is not a stage (ABC)")
    return asked


def main(argv=None):
    """Run the three stages and print each one's report."""
    argv = sys.argv[1:] if argv is None else argv
    path = input_file(argv)
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    asked = stages(argv)
    print(f"input file:     {path}")
    print(f"stages:         {asked}")

    # `use_cache=False`: Stage B rebuilds `Evaluators` from `reference.models`, and a
    # cached run carries `models=None` by design (see `sand_harness.reference_run`). Only
    # Stage B wants them, so a run without it takes the cache and skips PROCESS's own
    # solve -- ~95 s on the stellarator, which is most of a Stage A measurement.
    reference = reference_run(str(path), use_cache="B" not in asked)
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

    # The three per-machine values -- see the module docstring. `None` for the
    # reference keeps that path exactly as it always was (`mda_env`'s own default
    # graph, `sand.REFERENCE_SWITCH_VALUES`): its numbers are pinned regression
    # evidence and must not move because a second device exists.
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

    driven, env = mda_env(reference, graph=machine_graph)
    combined, report = assemble(reference, driven, env, switch_values=switch_values)
    print(
        f"\ndegenerate fixed points dropped: "
        f"{[d.path_str() for d in report['degenerate']]}"
    )
    if report["array_valued"]:
        print(
            f"ARRAY-UNKNOWN PROBLEMS DROPPED (loop-carried values frozen at the "
            f"seed -- the SAND problem is reduced, see `sand_harness.assemble`): "
            f"{[p.path_str() for p in report['array_valued']]}"
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
    if "A" in asked:
        print()
        print(stage_a(reference, condition_map, names, start).summary())

    # ---------------------------------------------------------------- B
    if "B" in asked:
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
                return abs(port[k] - reference_row[k]) / max(
                    abs(reference_row[k]), 1e-300
                )

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
        # Keyed by the assembled conditions, not by position in `reference.icc`: an
        # omitted constraint (`report["omitted"]`) has no port row, while PROCESS's own
        # Jacobian still carries every active constraint in `icc` order. Identical to the
        # old positional loop whenever nothing is omitted -- i.e. on the stellarator.
        assembled = [
            int(names[i].rsplit(".c", 1)[1]) for i in condition_rows[1:]
        ]  # `condition_rows[0]` is the objective
        for j, cid in enumerate(assembled):
            at_icc = reference.icc.index(cid)
            row(
                f"c{cid}",
                port_constraints[j],
                process[1:][at_icc],
                process_error[1:][at_icc],
            )

    # ---------------------------------------------------------------- C
    if "C" in asked:
        # The design variables -- everything else the block solves for is coupling, and
        # is seeded from an MDA run rather than from a `DataStructure` field that a cold
        # run has never written.
        design_paths = {sand.iteration_variable_path(i) for i in reference.ixc}
        for label, base, starts in (
            ("C2 (start at PROCESS's converged x)", reference.data, reference.converged),
            (
                "C3 (cold start from the IN.DAT values)",
                reference.cold,
                reference.initial,
            ),
        ):
            # The coupling unknowns are seeded from an MDA run **at this stage's own
            # design** -- `env` (built from PROCESS's converged state) for C2, a fresh
            # cold MDA for C3. Seeding a cold solve's coupling from a converged run would
            # make "cold" a half-truth; seeding it from the cold `DataStructure`'s zeros
            # made the solve impossible. See `_seed`.
            try:
                stage_env = (
                    env
                    if base is reference.data
                    else mda_env(reference, graph=machine_graph, data=base)[1]
                )
            except Exception as failure:  # noqa: BLE001 -- report the stage, run the next
                print(f"\nSTAGE C {label}: the MDA at this stage's design failed before")
                print(
                    f"  any solve could be seeded: {type(failure).__name__}: {failure}"
                )
                continue
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
                max_iter=SAND_MAX_ITER,
            )
            solve_drive = sand.sand_shape(solve_schedule)["drive"]
            # Which unknowns count as "design" differs by stage, because the stages mean
            # different things. C2's premise is *start where PROCESS ended*, so every
            # unknown PROCESS has a value for should come from PROCESS -- that is the
            # definition of the stage. C3's premise is *start from the input file*, and
            # an input file carries values for design variables only; everything else has
            # to come from an MDA at that design, because the `DataStructure` field
            # behind it holds a dataclass default that no run has ever written.
            from_process = base is reference.data
            seeded, borrowed = _seed(
                solve_schedule,
                solve_drive,
                base,
                stage_env,
                design=set(solve_drive.unknowns) if from_process else design_paths,
            )
            # One probe of the conditions at the seeded start, before any solve. A SAND
            # condition map holds the coupling unknowns fixed at their seed, so a seed
            # the models cannot evaluate shows up here as non-finite conditions -- and
            # handing those to an SQP produces a wander, not an answer. Documenting
            # exactly which conditions are non-finite and stopping is the honest report;
            # a seeding rule that gets a cold start past a genuine model singularity is a
            # separate, recorded decision (`_seed`'s own docstring is the stellarator's
            # precedent). At a healthy start (every C2, and the stellarator's C3) nothing
            # is printed and nothing changes.
            probe_context = {}
            for var in solve_drive.context:
                if var in stage_env:
                    probe_context[var] = stage_env[var]
                else:
                    try:
                        probe_context[var] = jnp.asarray(ground_truth(base, var))
                    except (AttributeError, KeyError):
                        probe_context[var] = jnp.asarray(0.0)
            at_start = solve_drive.condition_map(probe_context)(*[
                jnp.asarray(seeded[u]) for u in solve_drive.unknowns
            ])
            non_finite = [
                (condition.path_str(), float(np.asarray(value)))
                for condition, value in zip(
                    solve_drive.conditions, at_start, strict=True
                )
                if not np.all(np.isfinite(np.asarray(value)))
            ]
            if non_finite:
                print(
                    f"\nSTAGE C {label}: NOT SOLVED -- {len(non_finite)} of "
                    f"{len(solve_drive.conditions)} conditions are non-finite at the "
                    f"seeded start:"
                )
                for name, value in non_finite:
                    print(f"  {name:<56s} {value}")
                continue
            started = time.perf_counter()
            out = run_schedule(
                solve_schedule, _inputs_only(solve_schedule, seeded), whole=False
            )
            elapsed = time.perf_counter() - started
            print(f"\nSTAGE C {label}: {len(trace)} SQP iterations in {elapsed:.1f} s")
            print(f"  ({len(borrowed)} unknown(s)/input(s) seeded from the MDA env)")
            if not trace:
                # `VmconDriver` returns the best point on a `VMCONConvergenceException`
                # rather than propagating it, so zero recorded iterations is ambiguous
                # between "converged where it stood" and "the first QP was infeasible and
                # the start came back untouched". Measured on the tokamak: pyvmcon's
                # `QSPSolverException` ("no feasible solution") from constraint **72**'s
                # constantly-violated, zero-gradient row produced exactly this shape. Say
                # so instead of letting a swallowed failure read as a perfect solve.
                # (This said 68 until 2026-08-30, when the rows were actually measured:
                # c72 is `+5.53e-01` with an identically zero row, while c68 is violated
                # by `+4.95e-02` and *can* move, `|row| 2.95e-02`. 68 was named from a
                # violated-constraint list, which does not look at gradients -- being
                # violated is half the test and the cheaper half.)
                stuck = _why_no_step(solve_drive, probe_context, seeded)
                if stuck:
                    print(
                        f"  NO SQP ITERATION RECORDED, and the reason is measured: "
                        f"{len(stuck)} condition(s) are away from satisfaction with an "
                        f"identically zero gradient row, so no linearised step can "
                        f"reach a feasible point and `pyvmcon`'s first QP has none:"
                    )
                    for name, value in stuck:
                        print(
                            f"    {name:<52s} {value:+.6e}  (constant in every unknown)"
                        )
                else:
                    print(
                        "  NO SQP ITERATION RECORDED, and no condition is both violated "
                        "and constant -- so this is VMCON converging where it stood, "
                        "not a QP that had nowhere to go."
                    )
            print(
                f"  {'it':>3s} {'conv':>12s} {'objf':>14s} {'max|eq|':>11s} "
                f"{'min ie':>12s}"
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
