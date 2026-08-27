"""Run the `Optimise` layer's whole validation ladder and print the report.

    $PY functional_process/run_sand_harness.py                   # the stellarator
    $PY functional_process/run_sand_harness.py --input <IN.DAT>  # any other machine
    $PY functional_process/run_sand_harness.py --machine         # the tokamak

With no argument this is exactly what it always was: the Helias stellarator,
`tests/regression/input_files/stellarator_helias.IN.DAT`, whose numbers other records
quote. `--input`/`--machine` are spelled exactly as `run_mda_harness.py`'s (whose
`input_file` this imports rather than restates): `--machine` is shorthand for
`boundary.TOKAMAK_INPUT_FILE`, the conventional large tokamak.

A non-reference machine differs from the stellarator path in exactly three threaded
values, all derived from its own file: the graph (`graph_for(machine_from_indat(...))`),
the static switch values (`sand.switch_values_for`, read off the cold initialised
`DataStructure` instead of the hand-audited `REFERENCE_SWITCH_VALUES`), and the study
itself, which `reference_run` already reads off the run. Everything else -- the ladder,
the seeding rules, the solve -- is one code path, deliberately: two per-device
harnesses would drift apart exactly the way five switch registrations once did.

Sibling of `run_mda_harness.py`; see `sand_harness.py`'s module docstring for what each
of the three stages proves and, just as importantly, what it does not.

One PROCESS run (~95 s for the stellarator) serves all three stages: Stage A needs its
converged `DataStructure`, Stage B needs its live model objects to re-run the pipeline
for the finite-difference reference, and Stage C needs both its converged answer (to
start C2 at) and the input file's own cold start (to start C3 at).
"""

import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from functional_process import sand  # noqa: E402
from functional_process.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.mda import guess_sources  # noqa: E402
from functional_process.run_mda_harness import _resolve, input_file  # noqa: E402
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

SAND_MAX_ITER = 500
"""SQP iterations Stage C allows itself, against `VmconDriver`'s own default of 100.

**100 is PROCESS's `n_iteration_max`, for PROCESS's own eight-variable problem**, and it
was inherited here rather than chosen. The stellarator's SAND block is a different and
larger problem -- 14 unknowns, 21 conditions -- and it needs more, measured off one
cached PROCESS run with everything else held fixed:

| | SQP iterations | conv | `objf` |
|---|---|---|---|
| C2 (start at PROCESS's `x`) | **326** | `8.8e-11` | 1.217757338 |
| C3 (cold) | **258** | `8.0e-09` | 1.217757452 |

Both land on the known optimum (`_audit/next_steps.md` §11.11's `objf 1.217757336`,
`x109 = 0.0299518`), so what `max_iter = 100` produced was not a wander around a wrong
point but a solve stopped two-thirds of the way through -- reported as "100 iterations,
oscillating around `objf ~ 1.218`", which is what the trace of an unconverged run looks
like from the outside.

**Why the count grew, and why it is not a defect.** Ten `FixedPoint`s dissolved into
ordinary nodes when the topology switches became slots ("A switch selects an occupant"),
which is a correctness improvement -- PROCESS's own body for those was *"x is an input"*
-- and it took the SAND block from 22 unknowns / 16 equalities to 14 / 8. The eight
equalities it removed were nearly-linear `u = g(u)` rows; what is left is the same
problem with its nonlinearity concentrated into fewer conditions. Measured on the same
cached run, same seeds, same cottax: the pre-round-2 graph converges in **131**
iterations, this one in **326**, both on the same point to six digits. Neither the
cottax version nor any condition scale accounts for it -- `_audit/optimise_design.md`
§12 carries the bisect that rules both out, and the sweep showing the largest residual
row's factor moves the count only within this problem's own noise (219-326).

500 rather than 400 leaves headroom over the largest count measured; a solve that needs
more than this is not slow, it is stuck, and the trace the harness prints says so."""


def _seed(schedule, drive, base, fallback, design=()):
    """Every schedule input and every block unknown: **design** variables from `base`,
    every other unknown from `fallback` (a completed MDA env at the same design).

    A SAND solve is cold in its *design* variables. Its **coupling** unknowns are a
    different thing entirely -- quantities PROCESS never exposes as unknowns, because
    its own architecture (MDF) converges them by re-running the whole pipeline before
    every evaluation. They have to start somewhere consistent, and an MDA run at the
    same design is exactly what MDF would hand iteration 0.

    **This function used to say that and do something else, and it cost a cold start.**
    The old rule tried `ground_truth(base, var)` first for *every* unknown and fell back
    to the MDA env only on `AttributeError`/`KeyError`. But every coupling unknown *has*
    a `DataStructure` field -- holding the dataclass default `0.0` in a cold structure --
    so the lookup always succeeded and the fallback was never reached (the harness's own
    report line read `0 unknown(s)/input(s) seeded from the MDA env` on every cold run).
    Twelve of twenty-three unknowns therefore started at exactly zero, which is not a
    cold design but a **physically impossible state**: net electric power `-1.9e6` MW,
    `coe = 1.0e25`.

    What that did to the solve, measured at the cold point, old rule -> new:

    | | as-was | MDA-seeded |
    |---|---|---|
    | non-finite Jacobian cells | 46 / 690 | **0** |
    | condition number VMCON receives | >= 1.1e23 | **2.87e4** |
    | SQP iterations | **0** | **85, converged** |

    The 46 non-finite cells sat in exactly two rows (the objective and `c16`) and came
    from `x ** p` with `0 < p < 1` evaluated at `x == 0` -- value `0`, derivative `+inf`,
    and `inf * 0 = nan` under JVP -- at `buildings.py:282`'s `55.0 * helpow**0.5` and
    three sibling sites in `costs.py`, all reachable only because the cryogenic loads
    were seeded to zero. Those sites are worth fixing in their own right (same defect
    class as `next_steps.md` §9's `sqrt(maximum(...))`), but they are the symptom; this
    is the disease.

    `design` names the unknowns that genuinely come from `base` -- the run's iteration
    variables. Anything else is coupling.
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
        coupling = source in drive.unknowns and source not in design
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
    """`env` restricted to what the schedule may be handed: its own inputs.

    A `_seed` env is also a value store -- it grounds the drive's unknowns at their own
    names, which is what `borrowed` reporting and a reader inspecting the seeded design
    point rely on -- but a `Schedule` refuses a value at a name it owns (cottax's
    owned-name guard: an owned value could only be clobbered unread or, under an
    ordering bug, read stale in silence). So the solve call filters at the door and the
    store keeps its extra names for its other readers -- the same seam, in the same
    style, as `mdf._inputs_only`.
    """
    inputs = set(schedule.inputs)
    return {var: value for var, value in env.items() if var in inputs}


def main(argv=None):
    """Run the three stages and print each one's report. `argv` defaults to this
    process's own, so importing and calling `main([...])` is the same thing as the
    command line.
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
    # Keyed by the assembled conditions, not by position in `reference.icc`: an omitted
    # constraint (`report["omitted"]`) has no port row, while PROCESS's own Jacobian
    # still carries every active constraint in `icc` order. Identical to the old
    # positional loop whenever nothing is omitted -- i.e. on the stellarator.
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
    # The design variables -- everything else the block solves for is coupling, and is
    # seeded from an MDA run rather than from a `DataStructure` field that a cold run
    # has never written.
    design_paths = {sand.iteration_variable_path(i) for i in reference.ixc}
    for label, base, starts in (
        ("C2 (start at PROCESS's converged x)", reference.data, reference.converged),
        ("C3 (cold start from the IN.DAT values)", reference.cold, reference.initial),
    ):
        # The coupling unknowns are seeded from an MDA run **at this stage's own
        # design** -- `env` (built from PROCESS's converged state) for C2, a fresh cold
        # MDA for C3. Seeding a cold solve's coupling from a converged run would make
        # "cold" a half-truth; seeding it from the cold `DataStructure`'s zeros made the
        # solve impossible. See `_seed`.
        try:
            stage_env = (
                env
                if base is reference.data
                else mda_env(reference, graph=machine_graph, data=base)[1]
            )
        except Exception as failure:  # noqa: BLE001 -- report the stage, run the next
            print(f"\nSTAGE C {label}: the MDA at this stage's design failed before")
            print(f"  any solve could be seeded: {type(failure).__name__}: {failure}")
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
        # definition of the stage. C3's premise is *start from the input file*, and an
        # input file carries values for design variables only; everything else has to
        # come from an MDA at that design, because the `DataStructure` field behind it
        # holds a dataclass default that no run has ever written.
        from_process = base is reference.data
        seeded, borrowed = _seed(
            solve_schedule,
            solve_drive,
            base,
            stage_env,
            design=set(solve_drive.unknowns) if from_process else design_paths,
        )
        # One probe of the conditions at the seeded start, before any solve. A SAND
        # condition map holds the coupling unknowns fixed at their seed, so a seed the
        # models cannot evaluate shows up here as non-finite conditions -- and handing
        # those to an SQP produces a wander, not an answer. Documenting exactly which
        # conditions are non-finite and stopping is the honest report; a seeding rule
        # that gets a cold start past a genuine model singularity is a separate,
        # recorded decision (`_seed`'s own docstring is the stellarator's precedent).
        # At a healthy start (every C2, and the stellarator's C3) nothing is printed
        # and nothing changes.
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
            for condition, value in zip(solve_drive.conditions, at_start, strict=True)
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
        out = solve_schedule(_inputs_only(solve_schedule, seeded))
        elapsed = time.perf_counter() - started
        print(f"\nSTAGE C {label}: {len(trace)} SQP iterations in {elapsed:.1f} s")
        print(f"  ({len(borrowed)} unknown(s)/input(s) seeded from the MDA env)")
        if not trace:
            # `VmconDriver` returns the best point on a `VMCONConvergenceException`
            # rather than propagating it, so zero recorded iterations is ambiguous
            # between "converged where it stood" and "the first QP was infeasible and
            # the start came back untouched". Measured on the tokamak: pyvmcon's
            # `QSPSolverException` ("no feasible solution") from constraint 68's
            # constantly-violated, zero-gradient row produced exactly this shape. Say
            # so instead of letting a swallowed failure read as a perfect solve.
            print(
                "  NO SQP ITERATION RECORDED -- either VMCON converged at the start "
                "or its first QP failed (a constantly-violated, zero-gradient "
                "condition makes the QP infeasible) and the driver returned the "
                "start unchanged. The condition values above (Stage A) say which."
            )
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
