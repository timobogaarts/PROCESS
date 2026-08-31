"""**Cold** MDF and cold SAND on every reference configuration, as one table.

    $PY functional_process/run_cold_matrix.py                 # all seven in-scope files
    $PY functional_process/run_cold_matrix.py --input a.IN.DAT --input b.IN.DAT

Why this exists as a module and not as a scratch script
-------------------------------------------------------
`_audit/next_steps.md` §16.1's cold matrix was produced by throwaway scripts that no
longer exist, which is why *"re-run the cold matrix"* kept being a punch-list item
rather than a command. Every number below is one the two ladder harnesses already
compute; what was missing was a runner that walks the configurations, survives the ones
that refuse, and prints the rows side by side.

**This is deliberately not a third harness.** Everything it calls is imported --
`sand_harness.reference_run`/`mda_env`/`assemble`, `run_sand_harness._seed`/
`_inputs_only`/`SAND_MAX_ITER`, `mdf.assemble`/`seed`/`prime`/`solve`,
`run_mdf_harness.MAX_ITER`/`TOLERANCE` -- so a row here and the corresponding Stage C3
of `run_sand_harness.py`/`run_mdf_harness.py` are the same computation on the same code
path. Three harnesses that agree by construction is the same rule
`run_mdf_harness.py`'s docstring states for two.

What it does *not* do is the rest of either ladder: no Stage A, no Jacobian against
PROCESS's finite differences, no warm (C2) solve. Those are per-configuration
investigations; this is the matrix, and its unit of work is a row.

The reference file's path is preserved exactly
----------------------------------------------
`stellarator_helias.IN.DAT` runs with `graph=None`/`switch_values=None` -- the literal
default code path whose numbers other records pin -- and every other machine gets
`graph_for(machine_from_indat(...))` and `sand.switch_values_for(...)`. Same
`is_reference` discriminator, same reasoning, as `run_sand_harness.main`.

What a row costs, and what it does not include
----------------------------------------------
One PROCESS solve (10 s to ~110 s depending on the file), two MDA runs for SAND (the
warm one that assembly and `residual_condition_scales` read, the cold one that seeds the
solve -- both, so that a SAND row here is the same problem `run_sand_harness.py`'s C3
row is), one MDA prime for MDF, and the two solves. A configuration that refuses
assembly costs `machine_from_indat` alone: the refusal is checked *before* PROCESS is
run, because a row that reports "REFUSED" needs nothing from PROCESS and a 100 s solve
for a row that will be one line of text is the difference between a matrix that gets
re-run and one that does not.

A failure is a row, not an exit
-------------------------------
Every phase -- the assembly refusal, the PROCESS run, each formulation's build, each
solve -- is caught per configuration and recorded in the row. The point of the table is
the *whole* row set: a runner that dies on `st_regression` -- whose MDF build hands the
SQP a `nan` derivative row and whose SAND build raises `KeyError` on the objective's own
condition -- tells you nothing about the six rows either side of it, and those two
failures are themselves the most useful thing the run has to say about that file.

What the columns mean
---------------------
`SQP` is the recorded iteration count, and **zero is a real and distinct outcome**: it
means `pyvmcon`'s first QP had no feasible point and `VmconDriver` returned the start
untouched (`run_sand_harness._why_no_step` is the instrument that says why). `max|eq|`
is reported in `VmconDriver.condition_scale`'s units -- for SAND that means the residual
equalities are relative (`sand.residual_condition_scales`' `1/|u|`) and PROCESS's own
constraints keep a factor of 1.0; MDF declares no residual equalities, so its column is
already PROCESS's normalised residual. `min ie` is cottax's sign convention as VMCON
sees it: **negative is violated**. `PROCESS` is `numerics.n_solver_iterations` from the
run that produced the reference -- it is the count for a *converged* solve from the same
`IN.DAT`, and it is the only external scale on this table.

**Every `SQP` count is at `1e-8`**, and saying so is not pedantry. `MDF_TOLERANCE` and
`SAND_TOLERANCE` are imported from -- or default identically to -- the two ladder
harnesses, so the stellarator's 67 (MDF) and 83 (SAND) are exactly the counts those
harnesses already record at that tolerance (`run_mdf_harness.py:85`,
`run_sand_harness.py:99`). The *same two solves* take **58 and 58** at PROCESS's own
`epsvmc = 1e-6` (`_audit/optimise_design.md` §15, rows at `:2295`/`:2301`). Both pairs
are in the records four rows apart, and these counts were once checked against the wrong
pair and read as tree drift. There is no `conv` column to catch that, so the tolerance
is stated here instead. Caps: MDF 800, SAND 500.

Where the boundary values come from
-----------------------------------
Until 2026-08-31 every value in every row came from PROCESS's `DataStructure` after
`init_process` -- the seed. `provider.py` classified that boundary but nothing consumed
it, so "271-360 of 303-397 paths are answered independently" was a property of a
classification and not of a solve. It is now a property of a solve: `_boundary_seed`
copies the seed, hands it to `provider.install`, and every downstream reader
(`mdf.seed`, `mda_env`, `run_sand_harness._seed`) is given that copy instead. The
seeding machinery is untouched -- what moves is its *input*.

Three modes, `--provider` (the default), `--provider-strict` and `--seed`:

- **`--provider`** writes only the independently answered paths the seed *agrees* with,
  so the substitution is inert by construction and the table stays comparable with every
  row ever measured. The number it reports is real anyway: those values are read out of
  the input file and the dataclass defaults, and deleting the seed for them would change
  nothing. Its honesty rests on the disagreements being pinned and named
  (`reference_provider_*.txt`'s `off` rows), not on their being absent.
- **`--provider-strict`** takes the provider at its word at the `off` rows too. That is
  the experiment: a row that moves under it is a boundary value `init.py` supplies and
  the file does not.
- **`--seed`** is the old path exactly, kept so the two can be diffed.
"""

from __future__ import annotations

import copy
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from functional_process import mdf, provider, sand  # noqa: E402
from functional_process.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.run_mda_harness import _resolve  # noqa: E402
from functional_process.run_mdf_harness import MAX_ITER as MDF_MAX_ITER  # noqa: E402
from functional_process.run_mdf_harness import TOLERANCE as MDF_TOLERANCE  # noqa: E402
from functional_process.run_sand_harness import (  # noqa: E402
    SAND_MAX_ITER,
    _inputs_only,
    _seed,
    _why_no_step,
)
from functional_process.sand_harness import (  # noqa: E402
    assemble as sand_assemble,
)
from functional_process.sand_harness import (  # noqa: E402
    ground_truth,
    mda_env,
    reference_run,
)

CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/helias_5b.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/spherical_tokamak_eval.IN.DAT",
    "tests/regression/input_files/st_regression.IN.DAT",
)
"""Every `tests/regression/input_files/*.IN.DAT` except `IFE.IN.DAT`.

`IFE` is `ife == 1`, a whole unported device -- `.ife.*` has no unit in
`unit_registry.md` at all (`_audit/next_steps.md` §20.4) -- so it is not a row that could
become a number by any amount of running. Everything else is a row whether or not it
assembles, and the day this file was written is the argument for that: the two spherical
tokamaks refused on `tf_stress_arm == (0, 1, 0)` at the start of the run and
**assembled by the end of it**, because the `extended_plane_strain` port landed in the
same working tree while the pass was going. A runner whose configuration list encoded
today's verdict would have needed an edit to notice; this one needed a re-run.

Ordered stellarators first, then the four tokamaks, so that a truncated run still has
the rows whose numbers other records quote.
"""

SAND_TOLERANCE = None
"""`VmconDriver`'s own default, which is what `run_sand_harness.py`'s Stage C uses.

Left explicitly `None` rather than set to `MDF_TOLERANCE`: the two formulations' Stage C
solves are each compared against their own harness's published number, and a tolerance
this file chose would make both rows new measurements of a problem nobody has run.
`run_mdf_harness.TOLERANCE`'s own docstring records why MDF's is tighter than PROCESS's.
"""


PROVIDER, PROVIDER_STRICT, SEED_ONLY = "provider", "provider-strict", "seed"
"""The three boundary-value modes; see this module's docstring. `PROVIDER` is the
default because it is the only one that is simultaneously a measurement and inert."""


@dataclass
class Row:
    """One configuration's whole result -- both formulations, or the reason there is
    none.

    Every field defaults to "not measured" rather than to a number, so a row that fell
    over in phase two cannot be read as a row that solved: `render` prints `-` for
    `None` and the `note` carries the refusal or the exception.
    """

    name: str
    assembles: bool | None = None
    note: str = ""
    graph_nodes: int | None = None
    process_iterations: int | None = None
    n_ixc: int | None = None
    n_icc: int | None = None
    mdf: dict = field(default_factory=dict)
    sand: dict = field(default_factory=dict)
    omitted: object = None
    seconds: float = 0.0
    boundary: dict = field(default_factory=dict)
    """`provider.installed`'s counts for this configuration, plus `mode` and the paths
    the provider was allowed to move. Empty when the provider was not consulted."""


def _blank():
    """A formulation result with every measurement absent -- see `Row`."""
    return {
        "built": None,
        "note": "",
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
    }


def _trace_tail(trace):
    """`(iterations, objf, max|eq|, min ie)` off a Stage C callback trace.

    A trace is a list of `(i, convergence, objf, max|eq|, min ie)` tuples recorded by the
    harnesses' own `record` callbacks, so this reads the last entry and nothing else. An
    **empty** trace is not an error and not a zero-iteration success: it is the shape a
    first-QP-infeasible solve leaves behind (`run_sand_harness._why_no_step`), and the
    caller distinguishes those two by asking that function, not by looking here.
    """
    if not trace:
        return 0, None, None, None
    last = trace[-1]
    return len(trace), last[2], last[3], last[4]


def _status(trace, tolerance, cap):
    """Which of the four ways a solve ended, in one word.

    `run_mdf_harness._why_it_stopped` makes the same three-way distinction in a sentence;
    this adds the fourth outcome that only shows up cold -- `no-step`, an empty trace --
    because on this table a blank iteration count and a converged one must not look
    alike. `tolerance` may be `None` (`VmconDriver`'s own default), in which case the
    driver's default is what the convergence column is read against.
    """
    if not trace:
        return "no-step"
    epsilon = 1.0e-6 if tolerance is None else tolerance
    if trace[-1][1] <= epsilon:
        return "converged"
    if len(trace) >= cap:
        return f"cap({cap})"
    return "stopped"


def _recorder(trace):
    """The `VmconDriver.callback` both harnesses use, verbatim.

    `result.eq`/`result.ie` are the values `_Problem.__call__` handed VMCON, which are
    already multiplied by `VmconDriver.condition_scale` (`core/solver/drivers.py:794`).
    That is what makes the `max|eq|` column comparable across configurations and is why
    §19.1 item 6's absolute-`1e-6`-on-physical-units concern does not apply once
    `condition_scale` is passed to the *solve* schedule, as it is below.
    """

    def record(i, result, _x, convergence):
        trace.append((
            i,
            float(convergence),
            float(np.asarray(result.f)),
            float(np.max(np.abs(result.eq))) if len(result.eq) else 0.0,
            float(np.min(result.ie)) if len(result.ie) else 0.0,
        ))

    return record


def _boundary_seed(reference, path, mode):
    """`(cold, counts, moved)` -- the `DataStructure` this configuration's solves seed
    from, with the provider's answers written over the seed's where it has them.

    A **copy** of `reference.cold`, so the run's own PROCESS state is untouched and the
    two modes are comparable within one process. Nothing in `mda.py`/`mdf.py`/`sand.py`
    is involved: they go on reading a `DataStructure` through `ground_truth`, and the
    only thing that changed is which one they are handed. That is deliberate -- the
    seeding machinery is shared with three other harnesses and is not this file's to
    rewrite.

    `SEED_ONLY` returns `reference.cold` itself and no counts, which is the code path
    every row before 2026-08-31 was measured on.

    **The provider is asked about `driven_graph(graph_for(machine_from_indat(...)))`,
    which for `stellarator_helias` is not quite the graph that file's row solves** -- the
    reference file deliberately runs `graph=None`, i.e. `graph_for()`. The two boundaries
    are not guaranteed identical, and the consequence is bounded in the safe direction: a
    path the provider answers that this row does not read is a write nobody reads, and a
    path this row reads that the provider does not answer keeps the seed's value, which
    is the pre-existing behaviour. Worth removing when `boundary` is asked of the solved
    graph rather than of the assembled one; not worth special-casing here.
    """
    if mode == SEED_ONLY:
        return reference.cold, {}, ()
    answers = provider.answers_for(str(path))
    cold = copy.deepcopy(reference.cold)
    counts, moved = provider.install(
        answers, cold, disagreeing=(mode == PROVIDER_STRICT)
    )
    return cold, {**counts, "mode": mode}, moved


def cold_mdf(reference, machine_graph, switch_values, cold):
    """Build MDF for this run and solve it from the input file's own cold values.

    The same three calls `run_mdf_harness._measure` makes for its C3 -- `seed` off
    `reference.cold`, `prime` the MDA once, `solve` -- with the shape reported alongside,
    because on a table the shape is what makes a solve's cost legible.
    """
    result = _blank()
    began = time.perf_counter()
    problem = mdf.assemble(
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        graph=machine_graph,
        switch_values=switch_values,
    )
    shape = mdf.mdf_shape(problem)
    result.update(
        built=True,
        nodes=shape["nodes"],
        design=shape["design"],
        conditions=shape["conditions"],
        equalities=shape["equalities"],
        blocks=shape["inner_blocks"],
        driven=shape["inner_driven"],
    )
    env = mdf.seed(problem, cold)
    env, _primed = mdf.prime(problem, env)
    trace: list = []
    _x, _out, seconds = mdf.solve(
        problem,
        env,
        bounds=reference.bounds,
        callback=_recorder(trace),
        tolerance=MDF_TOLERANCE,
        max_iter=MDF_MAX_ITER,
    )
    iterations, objf, max_eq, min_ie = _trace_tail(trace)
    result.update(
        iterations=iterations,
        objf=objf,
        max_eq=max_eq,
        min_ie=min_ie,
        status=_status(trace, MDF_TOLERANCE, MDF_MAX_ITER),
        seconds=seconds,
        note="" if trace else "first QP infeasible -- the start came back untouched",
    )
    result["_omitted"] = problem.report["omitted"]
    result["_seconds_total"] = time.perf_counter() - began
    return result


def cold_sand(reference, machine_graph, switch_values, cold):
    """Build SAND for this run and solve it from the input file's own cold values.

    `run_sand_harness.main`'s C3 branch, with its two MDA runs kept as they are there and
    for the reason recorded there:

    - the **warm** env (`reference.data`) is what `sand_harness.assemble` reads to find
      the degenerate and array-valued fixed points, and what
      `sand.residual_condition_scales` reads for its `1/|u|` factors;
    - the **cold** env (`reference.cold`) is what `_seed` hands the solve for every
      coupling unknown, since a cold `DataStructure` field holds a dataclass default no
      run has written.

    Building the scales off the cold env instead would be a different problem from the
    one `run_sand_harness.py` reports, and this file's whole claim is that it is not.
    """
    result = _blank()
    began = time.perf_counter()
    driven, env = mda_env(reference, graph=machine_graph)
    combined, report = sand_assemble(reference, driven, env, switch_values=switch_values)
    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    drive = shape["drive"]
    condition_scale = sand.residual_condition_scales(drive, env)
    result.update(
        built=True,
        nodes=shape["drive_nodes"],
        design=shape["design"],
        conditions=shape["conditions"],
        equalities=shape["equalities"],
        blocks=shape["schedule_steps"],
        driven=shape["unknowns"],
    )
    result["_omitted"] = report["omitted"]

    stage_env = mda_env(reference, graph=machine_graph, data=cold)[1]
    trace: list = []
    solve_schedule = sand.sand_schedule(
        combined,
        None,
        bounds=reference.bounds,
        condition_scale=condition_scale,
        callback=_recorder(trace),
        max_iter=SAND_MAX_ITER,
    )
    solve_drive = sand.sand_shape(solve_schedule)["drive"]
    design_paths = {sand.iteration_variable_path(i) for i in reference.ixc}
    seeded, _borrowed = _seed(
        solve_schedule, solve_drive, cold, stage_env, design=design_paths
    )

    # The same pre-solve probe `run_sand_harness.main` runs, for the same reason: a SAND
    # condition map holds the coupling unknowns at their seed, so a seed the models
    # cannot evaluate shows up as non-finite conditions, and handing those to an SQP
    # produces a wander rather than an answer. On this table that has to be a *stated*
    # outcome, not an iteration count.
    probe_context = {}
    for var in solve_drive.context:
        if var in stage_env:
            probe_context[var] = stage_env[var]
        else:
            try:
                probe_context[var] = jnp.asarray(ground_truth(cold, var))
            except (AttributeError, KeyError):
                probe_context[var] = jnp.asarray(0.0)
    at_start = solve_drive.condition_map(probe_context)(*[
        jnp.asarray(seeded[u]) for u in solve_drive.unknowns
    ])
    non_finite = [
        condition.path_str()
        for condition, value in zip(solve_drive.conditions, at_start, strict=True)
        if not np.all(np.isfinite(np.asarray(value)))
    ]
    if non_finite:
        result.update(
            status="non-finite",
            note=(
                f"{len(non_finite)} of {len(solve_drive.conditions)} conditions are "
                f"non-finite at the seeded start, first {non_finite[0]}"
            ),
        )
        result["_seconds_total"] = time.perf_counter() - began
        return result

    started = time.perf_counter()
    solve_schedule(_inputs_only(solve_schedule, seeded))
    elapsed = time.perf_counter() - started
    iterations, objf, max_eq, min_ie = _trace_tail(trace)
    note = ""
    if not trace:
        # Zero iterations is ambiguous between "converged where it stood" and "the first
        # QP had no feasible point", and only the second is a failure. The
        # discriminator -- a condition both away from satisfaction and constant in every
        # unknown -- is measurable, so the row says which it was.
        stuck = _why_no_step(solve_drive, probe_context, seeded)
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
    )
    result["_seconds_total"] = time.perf_counter() - began
    return result


_HEADLINE = 260
"""How much of a refusal the table quotes -- see `_headline`."""


def _headline(refusal) -> str:
    """A refusal in one readable line, and the pointer to the rest of it.

    `indat`'s refusals are *paragraphs* -- deliberately, and they are the right length
    where they are raised, since a reader who hits one is about to decide whether to port
    a 517-line solver. Written into a matrix cell unabridged they crowd out the other six
    rows, which is the one thing this table exists to show at once. So the note carries
    the first `_HEADLINE` characters -- enough for the arm, the switch and the model that
    is missing -- and says where the full text is, rather than choosing between an
    unreadable table and a refusal with no reason.
    """
    text = " ".join(str(refusal).split())
    if len(text) <= _HEADLINE:
        return text
    return f"{text[:_HEADLINE].rstrip()} [...] (run `machine_from_indat` for the rest)"


def run_one(path, mode=PROVIDER) -> Row:
    """One configuration: assembly verdict, PROCESS, cold MDF, cold SAND.

    Nothing here raises. Each of the five phases records what it got and the next one
    runs anyway where it can -- a formulation that fails to build does not stop the other
    from solving, because the two failures are independent evidence. `mode` selects where
    the boundary values come from; see `_boundary_seed` and this module's docstring.

    The provider's answer is built **after** the PROCESS run, because it needs the seed
    to diff against, and its cost is one `cold_state` -- disk-cached, and already paid by
    `boundary.computed_by_process`. A provider failure is a row like any other: the
    configuration falls back to the seed and says so, since a matrix that lost six rows
    to a classifier would be a worse instrument than one that lost a column.
    """
    began = time.perf_counter()
    name = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    row = Row(name=name)
    print(f"\n=== {name} ", "=" * 40, flush=True)

    # Assembly first: a refusal costs a `machine_from_indat` and saves the PROCESS run.
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    machine_graph = None
    switch_values = None
    try:
        machine = machine_from_indat(str(path))
        machine_graph = None if is_reference else graph_for(machine)
        row.assembles = True
        row.graph_nodes = len(
            (machine_graph if machine_graph is not None else graph_for()).nodes
        )
        print(f"  assembles: {type(machine).__name__}, {row.graph_nodes} graph nodes")
    except (NotImplementedError, ValueError) as refusal:
        row.assembles = False
        row.note = _headline(refusal)
        row.seconds = time.perf_counter() - began
        print(f"  ASSEMBLY REFUSED: {row.note}")
        return row

    try:
        reference = reference_run(str(path))
    except Exception as failure:  # noqa: BLE001 -- a row, not an exit
        row.note = f"PROCESS run failed: {type(failure).__name__}: {failure}"
        row.seconds = time.perf_counter() - began
        print(f"  {row.note}")
        traceback.print_exc()
        return row
    row.process_iterations = reference.solver_iterations
    row.n_ixc = len(reference.ixc)
    row.n_icc = len(reference.icc)
    print(
        f"  PROCESS: {reference.solver_iterations} VMCON iterations in "
        f"{reference.solve_seconds:.1f} s, conv "
        f"{reference.convergence_parameter:.2e}; "
        f"{len(reference.ixc)} ixc, {len(reference.icc)} icc "
        f"({reference.n_equality} eq)"
    )
    cold = reference.cold
    try:
        cold, row.boundary, moved = _boundary_seed(reference, path, mode)
    except Exception as failure:  # noqa: BLE001 -- a row, not an exit
        row.boundary = {"mode": f"FAILED: {type(failure).__name__}: {failure}"}
        print(f"  boundary: PROVIDER FAILED -- {row.boundary['mode']}")
        traceback.print_exc()
    else:
        if row.boundary:
            print(
                f"  boundary: {row.boundary['written']} of "
                f"{row.boundary['supplied']} value(s) from the provider, "
                f"{row.boundary['from_process'] + row.boundary['held']} from the seed, "
                f"{row.boundary['nothing']} `None` in both"
                + (f" -- {len(moved)} moved" if moved else "")
            )
            for name, value, seeded in moved:
                print(f"    moved {name}: seed {seeded!r} -> provider {value!r}")

    if not is_reference:
        # Read off the same `DataStructure` the values come from: a switch the provider
        # disagreed with would change the graph, not just a number, and that has to be
        # visible rather than suppressed. None of the pinned `off` rows is a switch.
        switch_values = sand.switch_values_for(
            cold, reference.icc, reference.i_figure_merit
        )

    omitted = []
    for label, run, store in (
        ("MDF", cold_mdf, row.mdf),
        ("SAND", cold_sand, row.sand),
    ):
        try:
            store.update(run(reference, machine_graph, switch_values, cold))
        except Exception as failure:  # noqa: BLE001 -- a row, not an exit
            store.update(_blank())
            store.update(
                built=False,
                status="FAILED",
                note=f"{type(failure).__name__}: {failure}",
            )
            print(f"  {label}: FAILED -- {store['note']}")
            traceback.print_exc()
            continue
        if store.get("_omitted"):
            omitted.append(f"{label} {store['_omitted']}")
        print(
            f"  {label}: {store['iterations']} SQP it, {store['status']}, "
            f"objf {store['objf']}, max|eq| {store['max_eq']}, "
            f"min ie {store['min_ie']}"
            + (f" -- {store['note']}" if store["note"] else "")
        )
    row.omitted = "; ".join(omitted) if omitted else ""
    row.seconds = time.perf_counter() - began
    return row


_COLUMNS = (
    ("configuration", 22, "{}"),
    ("form", 5, "{}"),
    ("graph", 6, "{}"),
    ("nodes", 6, "{}"),
    ("desg", 5, "{}"),
    ("cond", 5, "{}"),
    ("eq", 4, "{}"),
    ("blks", 5, "{}"),
    ("drvn", 5, "{}"),
    ("SQP", 5, "{}"),
    ("status", 11, "{}"),
    ("objf", 15, "{}"),
    ("max|eq|", 10, "{}"),
    ("min ie", 11, "{}"),
    ("PRO", 4, "{}"),
)
"""`(heading, width, format)` per column of `render`'s table.

Widths are fixed rather than computed so that two runs of this file line up when they are
diffed, which is the operation the table is actually for.
"""


def _cell(value, width, numeric=None):
    """One table cell: `-` for a measurement that was not taken, never `0` or `nan`."""
    if value is None:
        return "-".rjust(width)
    if numeric == "g":
        return f"{value:.9g}".rjust(width)
    if numeric == "e":
        return f"{value:.2e}".rjust(width)
    return str(value).rjust(width)


def render(rows) -> str:
    """The table, plus a notes block for anything that did not fit in a cell.

    Two lines per configuration (MDF, SAND) so that the shape columns mean one thing per
    line; a refused configuration gets one line with its reason in the notes. The notes
    block is not decoration -- a `no-step` row's *reason* is the whole content of that
    row, and a table that dropped it would report the same cell for "converged trivially"
    and "the QP had nowhere to go".
    """
    head = " ".join(h.rjust(w) for h, w, _ in _COLUMNS)
    lines = [head, "-" * len(head)]
    notes = []
    for row in rows:
        if not row.assembles:
            lines.append(
                _cell(row.name, 22)
                + " "
                + " ".join(
                    _cell("REFUSED" if i == 0 else None, w)
                    for i, (_h, w, _f) in enumerate(_COLUMNS[1:])
                )
            )
            notes.append(f"{row.name}: ASSEMBLY REFUSED -- {row.note}")
            continue
        if row.process_iterations is None and row.note:
            # A configuration that assembled but has no PROCESS count got as far as the
            # run and failed there, and `run_one` always records why. The `and row.note`
            # guard is not defensive padding: without it a row with nothing to say still
            # emits an empty note line, which reads as a note somebody forgot to write.
            notes.append(f"{row.name}: {row.note}")
        for form, measured in (("MDF", row.mdf), ("SAND", row.sand)):
            store = measured or _blank()
            lines.append(
                " ".join([
                    _cell(row.name, 22),
                    _cell(form, 5),
                    _cell(row.graph_nodes, 6),
                    _cell(store["nodes"], 6),
                    _cell(store["design"], 5),
                    _cell(store["conditions"], 5),
                    _cell(store["equalities"], 4),
                    _cell(store["blocks"], 5),
                    _cell(store["driven"], 5),
                    _cell(store["iterations"], 5),
                    _cell(store["status"] or None, 11),
                    _cell(store["objf"], 15, "g"),
                    _cell(store["max_eq"], 10, "e"),
                    _cell(store["min_ie"], 11, "e"),
                    _cell(row.process_iterations, 4),
                ])
            )
            if store["note"]:
                notes.append(f"{row.name} {form}: {store['note']}")
        if row.omitted:
            notes.append(f"{row.name}: CONSTRAINTS OMITTED -- {row.omitted}")
    out = ["", "COLD MATRIX -- MDF and SAND from each input file's own cold start", ""]
    out.extend(lines)
    out += [
        "",
        (
            "`nodes/blks/drvn` differ by formulation and are not comparable across the "
            "two lines:"
        ),
        "  MDF  nodes = graph + conditions, blks/drvn = the inner MDA's blocking",
        (
            "  SAND nodes = the one Drive's nodes, blks = schedule steps, drvn = the "
            "block's unknowns"
        ),
        (
            "`desg/cond/eq` are the optimiser's own shape and ARE comparable: MDF's "
            "design is PROCESS's `ixc` exactly; SAND's adds the coupling unknowns and "
            "the residual equalities that hold them."
        ),
        (
            "`max|eq|` is in `VmconDriver.condition_scale`'s units (SAND's residuals "
            "relative, PROCESS's own constraints at factor 1.0). `min ie` is VMCON's "
            "sign: NEGATIVE IS VIOLATED."
        ),
        (
            "`PRO` is PROCESS's own VMCON iteration count for a converged solve of the "
            "same file."
        ),
    ]
    out += _boundary_block(rows)
    if notes:
        out += ["", "NOTES"]
        out += [f"  {note}" for note in notes]
    return "\n".join(out)


def _boundary_block(rows) -> list[str]:
    """Where each row's boundary values came from -- the provider or PROCESS's seed.

    A block rather than a column, for one reason: the table's rows are diffed against
    every previous run of this file, and a new column would report a change in every one
    of them. `supplied` excludes the `solver` and `guess` rows (`provider.NOT_SUPPLIED`)
    -- neither is a value a provider could answer -- and `paths` is the raw boundary
    total beside it, because §22.6's published ratios are over that one. `held` is the
    `off` rows the seed still owns in `--provider` mode, `none` the answers that are
    `None` in both (`provider.install`); the five columns after `configuration` sum to
    `supplied`.
    """
    measured = [row for row in rows if row.boundary.get("supplied")]
    if not measured:
        return []
    modes = sorted({str(row.boundary.get("mode")) for row in measured})
    block = [
        "",
        (
            f"BOUNDARY VALUES -- mode {'/'.join(modes)}; `supplied` is every boundary "
            "path but the"
        ),
        (
            "`solver` and `guess` rows, which no provider could answer (`paths` is the "
            "raw total)."
        ),
        "",
        "         configuration  provider   seed   held  none  supplied  paths",
    ]
    for row in measured:
        have = row.boundary
        block.append(
            " ".join([
                _cell(row.name, 22),
                _cell(have["written"], 9),
                _cell(have["from_process"], 6),
                _cell(have["held"], 6),
                _cell(have["nothing"], 5),
                _cell(have["supplied"], 9),
                _cell(have["paths"], 6),
            ])
        )
    return block


OUT = str(Path(__file__).parent / "reference_cold_matrix.txt")
"""Where `main` checkpoints the table.

**Rewritten after every configuration, not once at the end.** A full pass is ten to
fifteen minutes of PROCESS runs and MDA primes, and a run that is interrupted -- a
dropped connection, a `Ctrl-C`, a machine going to sleep -- used to leave nothing at all,
because the table only existed in a list that `print` was going to consume. Four rows on
disk are worth incomparably more than seven rows nobody ever saw, and the rewrite costs a
few hundred bytes of I/O against a row that cost minutes to compute.

Not a *pin*: unlike `reference_cold_start.txt` no test reads this and no meta-test
enforces it. It is the artefact the session's record quotes, and its value is that a
reader can `git diff` two runs of the matrix and see which cell moved.
"""


def checkpoint(rows, out=OUT) -> None:
    """Write the table as it stands. Called after every configuration; see `OUT`.

    Failing to write the checkpoint must never lose the row that was just computed, so
    an `OSError` here is reported and swallowed -- a read-only tree or a full disk is a
    reason to keep going with an in-memory table, not a reason to discard four minutes
    of PROCESS runs.
    """
    try:
        Path(out).write_text(render(rows) + "\n", encoding="utf-8")
    except OSError as failure:  # pragma: no cover -- reported, never fatal
        print(f"  (could not checkpoint to {out}: {failure})", flush=True)


def _mode(argv) -> str:
    """The boundary-value mode named on the command line; `PROVIDER` by default."""
    for flag, mode in (
        ("--seed", SEED_ONLY),
        ("--provider-strict", PROVIDER_STRICT),
        ("--provider", PROVIDER),
    ):
        if flag in argv:
            return mode
    return PROVIDER


def main(argv=None, out=OUT):
    """Walk the configurations, run both formulations cold on each, print the table.

    `--input <path>` may be repeated and replaces the default list entirely; with none
    given every entry of `CONFIGURATIONS` runs. `--out <path>` moves the checkpoint file;
    the table is written there after **each** configuration, so an interrupted run leaves
    every row it finished (see `OUT`). `--seed`/`--provider`/`--provider-strict` choose
    where the boundary values come from (see this module's docstring); the default is
    `--provider`.
    """
    argv = sys.argv[1:] if argv is None else argv
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    paths = [_resolve(p) for p in (chosen or CONFIGURATIONS)]
    mode = _mode(argv)
    print(f"boundary values: {mode}")
    began = time.perf_counter()
    rows: list[Row] = []
    for path in paths:
        rows.append(run_one(path, mode))
        checkpoint(rows, out)
        print(f"  (checkpointed {len(rows)} of {len(paths)} row(s) to {out})")
    print(render(rows))
    print(f"\n{len(rows)} configuration(s) in {time.perf_counter() - began:.0f} s")
    return rows


if __name__ == "__main__":
    main()
