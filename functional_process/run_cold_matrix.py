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
`IN.DAT`.

`PRO objf`, `d objf` and `worst dx` compare the port with **PROCESS's own answer**
--------------------------------------------------------------------------------
Until 2026-08-31 this table carried PROCESS's iteration count and never PROCESS's
*answer*, so every "matches" claim the record made off it compared the port against
**itself** under a different seeding mode -- which is exactly `_audit/next_steps.md`
§17.2's error, and it was repeated twice after §17.2 named it. `reference_run` is
disk-cached, so the three columns cost about `0.01 s` a row.

`PRO objf` is `objective_function(i_figure_merit, reference.data)`; `d objf` its relative
distance from the port's; `worst dx` the largest relative deviation over the `ixc` design
vector (`run_sand_harness.py`'s own `rel` column, worst-cased, with the iteration
variable named in the notes).

**A gap can be right.** `stellarator_helias` reads `1.2149167845171462` against the
port's `1.21775735`, and that 0.23 % is the `+17.604 MW` chain
`mda_harness.EXPLAINED_DISAGREEMENTS` documents, where PROCESS's converged
`DataStructure` is internally inconsistent and the port is the self-consistent side. Such
rows are flagged **EXPLAINED** in the notes (`EXPLAINED_OBJECTIVE_READS`), because a
column that reported them as failures would be worse than no column at all.

The problem type comes from the file, not from a default
--------------------------------------------------------
`large_tokamak_eval` and `spherical_tokamak_eval` state `i_process_run_mode = -2`, and
PROCESS answers that by **root-finding the equalities** with `scipy.optimize.fsolve`,
forming no objective and never examining the inequalities. `Row.root_find` reads that off
the file (`importer.Problem.is_evaluation`) and the MDF arm assembles a `RootFind`
accordingly (`mdf.assemble`). Their `PRO objf` cell reads `none`, not a number: PROCESS
forms none, and `reference.i_figure_merit` is `7` on both only because
`numerics.py:154`'s dataclass default put it there.

**Such a file gets ONE row, and the formulation column is not a choice it has.** MDF and
SAND are two ways of distributing an *optimisation*; a file that states none has nothing
to distribute, and MDF's design vector is `ixc` exactly, so the MDF root find is
PROCESS's own square system rather than one of two readings of it. The second row this
table used to print was a SAND `Optimise` over design *and* coupling against a figure of
merit the file never named -- a number with nothing to compare it to, under a heading
implying there was. `_solve_both` carries the argument; the notes block states it on
every affected row rather than leaving the single line to be read as a missing
measurement.

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

Four modes, `--provider` (the default), `--provider-strict`, `--seed` and `--native`:

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
- **`--native`** has no `DataStructure` in it at all, and no PROCESS run. The env is
  `native.native_state` -- `importer.read_indat`'s values over a vendored table of
  PROCESS's dataclass defaults -- and the problem is `native.native_reference`'s, off
  `indat.problem_from_indat` and the vendored `ITERATION_VARIABLES` bounds. The other
  three modes all *install into* a copy of PROCESS's seed, so they measure how much of
  the boundary need not come from PROCESS; this one measures what happens when none of
  it does.

  **It is strictly weaker than `--provider-strict`, and that is the measurement.** The
  provider answers a `derived` path from the seed; a native state has nothing to fall
  back to and answers it with the bare dataclass default, so `init.py`'s and
  `st_init`'s writes are simply absent. §22.7 measured that taking the provider at its
  word at 13 `off` paths costs four of seven configurations their solve; a native row
  starts from 12-24 wrong paths, not 5-8. A native row that *does* solve is therefore
  worth more than a `--provider` row that does, and a native row that does not is a
  work list keyed to `_audit/init_audit.md`, which is what this mode is for.

  The SAND column carries one caveat -- `native.NativeReference` -- because with no
  converged run there is no warm env for `sand.residual_condition_scales` **or for
  `sand_harness.assemble`'s degeneracy test**, so a native SAND row is a differently
  scaled and sometimes differently *shaped* problem from the same file's `--provider`
  row, while the MDF rows are directly comparable. `_audit/optimise_design.md` §27
  measures both and shows they are the whole of the difference: the two modes' boundary
  values are bit-identical on all seven files.

Seeding and scoring are two axes, and `--compare-process` is the second one
--------------------------------------------------------------------------
Until 2026-09-01 a `--native` row's `PRO`, `PRO objf`, `d objf` and `worst dx` cells were
blank **by construction**, and that was the only remaining reason to run `--provider` at
all. The coupling was never real. `sand_harness.reference_run` is disk-cached and costs
~4.6 s cold and ~0.01 s warm, and scoring a finished solve against PROCESS's converged
answer needs that answer *loaded*, not *used as a seed*.

So `--compare-process` / `--no-compare-process` compose freely with the four seeding
modes. In `run_one` the scoring object is a local named `oracle` and never `reference`;
it reaches `_against_process` and nothing that assembles, seeds or solves. The table
prints the two facts in two places -- a `seed` column for where the start came from, the
`PRO*` group for whether the answer was scored -- so that a filled `PRO objf` beside
`seed nat` reads as what it is.

**`--native --compare-process` is the intended default table.**
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
from jax.flatten_util import ravel_pytree  # noqa: E402

from functional_process import (  # noqa: E402
    mdf,
    native,
    phase_timing,
    provider,
    sand,
)
from functional_process.core.solver.drivers import (  # noqa: E402
    VMCON_NON_FINITE,
    SlsqpDriver,
    Status,
    non_finite_summary,
)
from functional_process.importer import read_indat  # noqa: E402
from functional_process.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
    switch_values_from_indat,
)
from functional_process.mda_harness import EXPLAINED_DISAGREEMENTS  # noqa: E402
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
    run_schedule,
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
NATIVE = "native"
"""The four boundary-value modes; see this module's docstring. `PROVIDER` is the
default because it is the only one that is simultaneously a measurement and inert."""

SEED_LABEL = {
    PROVIDER: "prov",
    PROVIDER_STRICT: "strict",
    SEED_ONLY: "seed",
    NATIVE: "nat",
}
"""What the `seed` column prints. **Seeding and comparison are two axes, not one**, and
this column exists so that no reader can take a filled `PRO objf` cell as evidence that
PROCESS supplied the starting state -- see `run_one`'s `compare` argument."""


@dataclass
class Row:
    """One configuration's whole result -- both formulations, or the reason there is
    none.

    Every field defaults to "not measured" rather than to a number, so a row that fell
    over in phase two cannot be read as a row that solved: `render` prints `-` for
    `None` and the `note` carries the refusal or the exception.
    """

    name: str
    seed_mode: str = PROVIDER
    """Where this row's **starting state** came from -- one of the four boundary modes.
    Printed as the `seed` column, and deliberately independent of `compared`."""
    compared: bool = False
    """Was PROCESS run for this row's `PRO`/`PRO objf`/`d objf`/`worst dx` cells?

    **This is not the seeding question**, and conflating the two is the thing the split
    exists to prevent: a `--native --compare-process` row is seeded with no PROCESS
    object in the path at all and still scored against PROCESS's converged answer,
    because loading that answer costs one disk-cached `reference_run` and nothing about
    *scoring* a solve requires having *started* it from PROCESS."""
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
    timings: dict = field(default_factory=dict)
    """`{formulation: {phase: seconds}}` -- where each arm's wall clock went, from
    `phase_timing`. Empty when the patches did not install (see `phase_timing.install`),
    which `render` reports rather than papering over."""
    boundary: dict = field(default_factory=dict)
    """`provider.installed`'s counts for this configuration, plus `mode` and the paths
    the provider was allowed to move. Empty when the provider was not consulted."""
    omitted_paths: tuple = ()
    """`NATIVE` only: the places this run asked for and the native state could not
    answer, so each was seeded `0.0`. The work list, per configuration."""
    root_find: bool = False
    """Does this file state a root find (`i_process_run_mode = -2`) rather than an
    optimisation? Read off the file's own text (`importer.Problem.is_evaluation`), so it
    is the same answer in every boundary mode including `NATIVE`."""
    process_objf: float | None = None
    """PROCESS's own converged objective, `objective_function(i_figure_merit, data)`.

    `None` in two distinct situations the table must not conflate: a `--native` row (no
    PROCESS run at all, so the `PRO` column is `-` too) and a root-find row (PROCESS
    **forms no objective** in evaluation mode -- `_Fsolve.solve` ends `self.objf = None`
    -- so there is no number to compare against and inventing one would be worse than
    the blank). `render` distinguishes them by `root_find`."""


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
        "dx": None,
        "dx_at": None,
        "dobjf": None,
        "explained": "",
        "withheld": "",
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


@dataclass
class MdfBuild:
    """Everything about an MDF arm that does **not** change between solves.

    The unit `functional_process.session` reuses. `problem` is the assembled `Mdf`,
    `in_graph` its `InGraphRootFind` where the file states one, and `shape` the cells
    `_blank()` fills from the assembly rather than from the answer.
    """

    problem: object
    in_graph: object = None
    root_find: bool = False
    shape: dict = field(default_factory=dict)


def build_mdf(reference, machine_graph, switch_values, root_find=False) -> MdfBuild:
    """Assemble MDF for this configuration -- the half of a row that a *second* solve of
    the same configuration must not repeat.

    Split out of `cold_mdf` on 2026-09-03 for `functional_process.session`, and the
    reason is measured rather than tidy: re-assembling builds a structurally *equal* but
    freshly allocated block, and every memo downstream is keyed on what was built --
    `sand_harness._SCHEDULE_WHOLE`, jax's own executable cache -- so a loop that
    re-assembles re-traces and re-compiles the whole graph on every iteration while a
    loop that does not is 30-75x faster (`_audit/optimise_design.md` §32.2). Nothing here
    reads `cold`, which is exactly why it can be hoisted.

    **That 30-75x is the number §37 moved**, and this docstring is left standing as the
    measurement it was: since the graph became static (§34), the switches stopped being
    partials (§35) and `host_cache._BOUND` was deleted for a module-level jit (§37), a
    re-assembled block is a jax cache *hit*. Hoisting is still right -- it saves the
    assembly itself -- but it is no longer the difference between a second and a minute.
    """
    build = MdfBuild(
        problem=mdf.assemble(
            reference.ixc,
            reference.icc,
            reference.n_equality,
            reference.i_figure_merit,
            graph=machine_graph,
            switch_values=switch_values,
            root_find=root_find,
        ),
        root_find=root_find,
    )
    shape = mdf.mdf_shape(build.problem)
    build.shape = {
        "built": True,
        "nodes": shape["nodes"],
        "design": shape["design"],
        "conditions": shape["conditions"],
        "equalities": shape["equalities"],
        "blocks": shape["inner_blocks"],
        "driven": shape["inner_driven"],
    }
    if root_find:
        # **Stated in the graph, not driven from outside it** (`mdf.in_graph_root_find`,
        # `_audit/in_graph_rootfind.md`). Built here rather than after `prime` because it
        # reads only the assembly: the root find is a problem node the blocking sees, so
        # `Blocking.scc` decides what it drives -- 39 nodes of 259 on
        # `spherical_tokamak_eval`, 35 of 271 on `large_tokamak_eval`, with the two
        # coupled SCCs that fall inside the loop as its `Blocking.inner`. The outer arm
        # re-ran the whole MDA per residual instead -- ~85 % of it for nothing, since
        # 69/117 nodes are upstream and constant during the solve and 152/120 are
        # downstream of the constraints and have no vote in choosing x.
        #
        # This is not a different problem from the ordinary graph-based MDA solve and it
        # should not be written as one: it is the same graph, with the file's own problem
        # declared inside it, decomposed by the same `Blocking.scc` as the rest. Same
        # answer -- `3.635e-09`/`3.292e-12` against PROCESS's `fsolve` x, agreeing with
        # the outer arm to `1.4e-15` -- and one XLA program instead of 856.
        build.in_graph = mdf.in_graph_root_find(build.problem)
        interior = mdf.in_graph_shape(build.in_graph)
        # `interior_*`, not `inner_*`: `in_graph_shape` returns `mdf_shape`'s keys too,
        # and those are the whole MDA's blocking -- which is what this row would report
        # if the formulation had not changed, i.e. exactly the wrong number. The driven
        # block's own interior is the measurement the restructuring exists for.
        build.shape["blocks"] = interior["interior_blocks"]
        build.shape["driven"] = interior["interior_driven"]
    return build


def solve_mdf(build: MdfBuild, reference, cold, optimiser=None) -> dict:
    """Solve an already-assembled MDF from `cold` -- the half of a row that a repeated
    solve *does* repeat.

    The same three calls `run_mdf_harness._measure` makes for its C3 -- `seed` off the
    cold `DataStructure`, `prime` the MDA once, `solve`.

    **A re-seed and a re-prime are cheap and are meant to be repeated**: measured warm on
    `stellarator_helias`, `mdf.seed` plus `mdf.prime` is 10-20 ms with **zero** XLA
    compiles, and a full re-seed from a *different* start is 25-42 ms
    (`_audit/optimise_design.md` §32.2). So a scan point is a call to this function and
    not a reason to rebuild anything.
    """
    problem, root_find = build.problem, build.root_find
    result = _blank()
    began = time.perf_counter()
    result.update(build.shape)
    env = mdf.seed(problem, cold)
    env, _primed = mdf.prime(problem, env)
    if root_find:
        built = build.in_graph
        shape = mdf.in_graph_shape(built)
        x, out, seconds = mdf.in_graph_solve(built, env)
        # The driver's own verdict, out of the env the run returned. `MdfNewtonDriver`
        # reports it through `DriverOut` ports the problem node owns, so there is no
        # results sink to hand in and no `jax.debug.callback` to carry it. Both survived
        # the whole-schedule jit as traced arrays, which is why they are rendered here
        # rather than used raw.
        steps = int(np.asarray(built.steps(out)))
        converged = bool(np.asarray(built.successful(out)))
        # The conditions at the answer come out of `out`, which is now the right place
        # rather than a second-best one: the schedule evaluated them *inside* the driven
        # block, so what is reported is literally what was driven. The outer arm had to
        # rebuild a condition map to say the same thing.
        residuals = [float(np.asarray(out[c])) for c in problem.conditions]
        # PROCESS's own last act in this mode: evaluate every constraint, equalities and
        # inequalities alike, once at the answer (`_Fsolve.solve`). `min ie` is that.
        inequalities = [float(np.asarray(out[c])) for c in problem.reported]
        result.update(
            iterations=steps,
            objf=None,
            max_eq=max(abs(r) for r in residuals) if residuals else 0.0,
            min_ie=min(inequalities) if inequalities else None,
            status="converged" if converged else "not-converged",
            seconds=seconds,
            note=(
                f"RootFind over {len(residuals)} equality/-ies, "
                f"{len(inequalities)} inequality/-ies evaluated at the answer and not "
                f"driven -- PROCESS's own `fsolve` shape. Stated IN the graph: the "
                f"blocking drives {shape['block']} of {shape['nodes']} nodes"
                + ("" if converged else f" ({mdf.verdict(out, mdf.Status)})")
            ),
        )
        result["_x"] = tuple(float(np.asarray(v)) for v in x)
        result["_omitted"] = problem.report["omitted"]
        result["_seconds_total"] = time.perf_counter() - began
        return result
    trace: list = []
    x, _out, seconds = mdf.solve(
        problem,
        env,
        bounds=reference.bounds,
        callback=_recorder(trace),
        tolerance=MDF_TOLERANCE,
        max_iter=MDF_MAX_ITER,
        # `mdf.solve` accepts a *class* here and builds the default driver out of it,
        # so every other argument on this call -- bounds, tolerance, cap, callback --
        # and the equality/inequality counts stay exactly what the VMCON row had. That
        # is what makes the two tables a comparison of solvers rather than of two
        # slightly different problems.
        **({} if optimiser is None else {"optimiser": optimiser}),
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
    result["_x"] = tuple(float(np.asarray(v)) for v in x)
    result["_omitted"] = problem.report["omitted"]
    result["_seconds_total"] = time.perf_counter() - began
    return result


def cold_mdf(
    reference, machine_graph, switch_values, cold, root_find=False, optimiser=None
):
    """Build MDF for this run and solve it from the input file's own cold values.

    `build_mdf` then `solve_mdf`, which is what a matrix row is: assemble the problem
    and solve it once. A caller that wants to solve it *again* calls the two halves --
    see `functional_process.session`, whose whole point is that a second row's worth of
    answer costs the second half only.

    `root_find` states the file's own problem type instead of an `Optimise` for the two
    files whose `i_process_run_mode` is `-2` (`mdf.assemble`; `_run_mode` chooses). The
    row then reports `objf` as `-`, because PROCESS forms none in that mode and neither
    does the port.
    """
    began = time.perf_counter()
    build = build_mdf(reference, machine_graph, switch_values, root_find=root_find)
    result = solve_mdf(build, reference, cold, optimiser=optimiser)
    result["_seconds_total"] = time.perf_counter() - began
    return result


@dataclass
class SandBuild:
    """Everything about a SAND arm that does **not** change between solves.

    `solve_schedule` is built **once**, callback and all, and that is the load-bearing
    part rather than an economy: `sand_harness.run_schedule` memoises its whole-schedule
    jit and its fused runners on the `Schedule` **object**, so a schedule rebuilt per
    solve re-traces and re-compiles everything it holds. The per-solve state that used to
    force a rebuild is the callback's trace, and a list can simply be cleared.
    """

    solve_schedule: object
    drive: object
    trace: list
    design_paths: set
    shape: dict = field(default_factory=dict)
    omitted: object = None


def build_sand(reference, machine_graph, switch_values, optimiser=None) -> SandBuild:
    """Assemble SAND for this configuration, solve schedule included.

    `run_sand_harness.main`'s C3 branch up to the point where `cold` first matters, with
    its warm MDA run kept as it is there and for the reason recorded there: the **warm**
    env (`reference.data`) is what `sand_harness.assemble` reads to find the degenerate
    and array-valued fixed points, and what `sand.residual_condition_scales` reads for
    its `1/|u|` factors. Building the scales off the cold env instead would be a
    different problem from the one `run_sand_harness.py` reports, and this file's whole
    claim is that it is not.

    Split out of `cold_sand` on 2026-09-03 for `functional_process.session` -- see
    `build_mdf` for why re-assembly is the thing a repeated solve must avoid.
    """
    driven, env = mda_env(reference, graph=machine_graph)
    combined, report = sand_assemble(reference, driven, env, switch_values=switch_values)
    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    condition_scale = sand.residual_condition_scales(shape["drive"], env)
    trace: list = []
    solve_schedule = sand.sand_schedule(
        combined,
        None,
        bounds=reference.bounds,
        condition_scale=condition_scale,
        callback=_recorder(trace),
        max_iter=SAND_MAX_ITER,
        optimiser=optimiser,
    )
    return SandBuild(
        solve_schedule=solve_schedule,
        drive=sand.sand_shape(solve_schedule)["drive"],
        trace=trace,
        design_paths={sand.iteration_variable_path(i) for i in reference.ixc},
        shape={
            "built": True,
            "nodes": shape["drive_nodes"],
            "design": shape["design"],
            "conditions": shape["conditions"],
            "equalities": shape["equalities"],
            "blocks": shape["schedule_steps"],
            "driven": shape["unknowns"],
        },
        omitted=report["omitted"],
    )


def solve_sand(build: SandBuild, reference, machine_graph, cold) -> dict:
    """Solve an already-assembled SAND from `cold`.

    The **cold** env (`reference.cold`, or whatever `cold` a caller substitutes) is what
    `_seed` hands the solve for every coupling unknown, since a cold `DataStructure`
    field holds a dataclass default no run has written.

    The trace is cleared rather than replaced, because the schedule closes over it -- see
    `SandBuild`.
    """
    result = _blank()
    began = time.perf_counter()
    result.update(build.shape)
    result["_omitted"] = build.omitted
    solve_schedule, solve_drive = build.solve_schedule, build.drive
    trace, design_paths = build.trace, build.design_paths
    trace.clear()

    stage_env = mda_env(reference, graph=machine_graph, data=cold)[1]
    seeded, _borrowed = _seed(
        solve_schedule, solve_drive, cold, stage_env, design=design_paths
    )

    # The context `_why_no_step` reads below. It used to be built for a *pre-solve
    # probe* as well -- one `host_cache.flat_conditions` call over the seeded start,
    # emitting a `status="non-finite"` row when any condition value was not finite.
    #
    # **That probe is gone, and the guard that replaced it checks strictly more.**
    # `drivers._refuse_non_finite` runs inside `_Problem.__call__` on the first iterate
    # and reads the condition values *and* the Jacobian rows, plus any identically zero
    # Jacobian column, naming every offender. The probe read values alone -- and the
    # case its own docstring records is one a values-only probe cannot see: a cold SAND
    # start where all 30 conditions were **finite in value** and only the derivatives
    # were `nan`. So the probe was the weaker of two checks over the same block.
    #
    # It was also a whole extra program. `flat_conditions` has a different signature and
    # wrapper (`eqx.filter_jit` over the whole `ConditionMap`) from `host_cache.bind`'s
    # plain `jax.jit` over array leaves, so jax sees a **third module computing the same
    # block's values** and compiles it -- 5 161 emitted MLIR lines on the stellarator
    # SAND block against `bind`'s `values` at 5 160, i.e. the same program twice, and
    # 15 161 against 15 160 on the tokamak. That is 23.7 % and 21.7 % of everything the
    # SAND arm emitted, or 1.92 s and 10.75 s of first-call wall clock measured
    # interleaved in one process. [measured, `_audit/optimise_design.md` §31.30.1]
    probe_context = {}
    for var in solve_drive.context:
        if var in stage_env:
            probe_context[var] = stage_env[var]
        else:
            try:
                probe_context[var] = jnp.asarray(ground_truth(cold, var))
            except (AttributeError, KeyError):
                probe_context[var] = jnp.asarray(0.0)

    started = time.perf_counter()
    out = run_schedule(solve_schedule, _inputs_only(solve_schedule, seeded), whole=False)
    elapsed = time.perf_counter() - started

    # **A stated outcome, not an exit -- and read as data, not caught.** The driver
    # refuses a non-finite problem (`drivers._refuse_non_finite`) and reports
    # `VMCON_NON_FINITE` through its own `Status` port, so the verdict arrives in the
    # env this run returned like every other value. Nothing crosses the
    # `jax.pure_callback` boundary as an exception, which is the point: that callback
    # promises a pure function of its inputs, and jax may elide, repeat or reorder it.
    #
    # The names are recomputed here rather than carried, because they are wanted only
    # for a row that already failed: `non_finite_summary` runs the same check eagerly
    # on the host and reports which conditions were non-finite in VALUE and which in
    # DERIVATIVE -- the distinction the deleted pre-solve probe could not draw at all,
    # and the one that cost a full investigation the first time it was met.
    reported = mdf.verdict(out, Status, solve_drive.problem)
    if reported is not None and int(np.asarray(reported)) == VMCON_NON_FINITE:
        flat_probe, probe_unravel = ravel_pytree(
            tuple(jnp.asarray(seeded[u]) for u in solve_drive.unknowns)
        )
        summary = non_finite_summary(
            solve_drive.condition_map(probe_context), probe_unravel, flat_probe
        )
        result.update(
            status="non-finite",
            note=(
                f"refused at the first iterate -- {summary}"
                if summary
                else "refused at the first iterate (the driver reported a non-finite "
                "problem; re-evaluating at the seeded start no longer reproduces it)"
            ),
        )
        result["_seconds_total"] = time.perf_counter() - began
        return result

    result["_x"] = tuple(
        float(np.asarray(out[sand.iteration_variable_path(i)])) for i in reference.ixc
    )
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


def cold_sand(reference, machine_graph, switch_values, cold, optimiser=None):
    """Build SAND for this run and solve it from the input file's own cold values.

    `build_sand` then `solve_sand`, which is what a matrix row is; a caller that solves
    the same configuration more than once calls the two halves
    (`functional_process.session`).
    """
    began = time.perf_counter()
    build = build_sand(reference, machine_graph, switch_values, optimiser=optimiser)
    result = solve_sand(build, reference, machine_graph, cold)
    result["_seconds_total"] = time.perf_counter() - began
    return result


EXPLAINED_OBJECTIVE_READS = {
    ".costs.coe": ".heat_transport.p_plant_electric_base_total_mw",
    ".costs.cdirt": ".heat_transport.p_plant_electric_base_total_mw",
    ".costs.concost": ".heat_transport.p_plant_electric_base_total_mw",
}
"""`objective read -> the `mda_harness.EXPLAINED_DISAGREEMENTS` key that explains a gap
on it`.

Why this table exists at all
----------------------------
The `PRO objf` column below is the first thing on this matrix that compares the port
against **PROCESS's answer** rather than against the port under another seeding mode
(`_audit/next_steps.md` §17.2's error, repeated twice since). The first time it was run
it reported `stellarator_helias` off by 0.23 %, and a column that called that a
regression would be worse than no column: it is the `+17.604 MW` chain
`mda_harness.EXPLAINED_DISAGREEMENTS` already documents at
`.heat_transport.p_plant_electric_base_total_mw`, where **PROCESS's own converged
`DataStructure` is internally inconsistent and the port is the self-consistent side**.
That entry's own last sentence names the tail of the chain -- *"the rest is that delta
through the linear cost accumulation to `.costs.coe`"* -- and every objective metric that
reads a cost total is therefore downstream of it.

Why it is a read map and not a configuration list
-------------------------------------------------
The property is of the **metric**, not of the machine: `objective_metric_6` is `coe/100`
and `objective_metric_7` is `cdirt/1e3` or `concost/1e4`, so any file choosing figure of
merit 6 or 7 inherits the same chain, and a list of today's seven configurations would go
stale the first time an eighth arrived. `_explained_by` asks the assembled objective node
what it reads and looks the answer up here.

**A gap is marked explained only when the design vector agrees**, which is the second
half of the rule and the part that keeps it honest: the chain is a difference in
*evaluating* the objective at a shared point. A row whose `worst dx` has moved is not
this; it is a different answer, and it gets no label.
"""

_UNKNOWN_EXPLANATIONS = sorted(
    set(EXPLAINED_OBJECTIVE_READS.values()) - set(EXPLAINED_DISAGREEMENTS)
)
if _UNKNOWN_EXPLANATIONS:  # pragma: no cover -- a wiring error, checked at import
    raise ValueError(
        f"EXPLAINED_OBJECTIVE_READS points at {_UNKNOWN_EXPLANATIONS}, which "
        f"`mda_harness.EXPLAINED_DISAGREEMENTS` no longer documents -- a row would be "
        f"labelled 'explained' by a record that has been deleted. Re-derive the label "
        f"or drop the entry; do not silence this."
    )
"""The label may only cite a live record.

A matrix cell reading EXPLAINED is a claim that somebody has already chased this
difference and written down why. If the write-up is deleted or renamed and the label
outlives it, the cell becomes an assertion with nothing behind it -- which is strictly
worse than the unlabelled 0.23 % it replaced. Checked at import so the failure lands on
whoever moved the record, not on a reader of a table generated three weeks later.
"""


_EXPLAINED_DX = 1e-4
"""How closely the design vector must agree before an objective gap may be called
explained. Loose on purpose -- the cold solves land within `1e-6`-ish of PROCESS's `x`
when they land at all, so this separates "the same point" from "a different answer"
rather than grading the solve."""


def _explained_by(reference, graph, switch_values):
    """`(key, read)` if this run's objective is downstream of a documented, deliberate
    disagreement, else `None`.

    Asks `sand.objective_nodes` -- the same call `mdf.assemble` makes -- which `VarPath`s
    the run's figure of merit reads, so this cannot drift from what was actually
    assembled the way a hand-kept per-configuration list would.

    **Every node it builds is asked, not just the first.** A maximise run is two nodes
    since §36 -- the metric and a `.ObjectiveNegated` -- and the negation reads
    `^metric.numerics.objf`, which is in no `EXPLAINED_OBJECTIVE_READS` table and simply
    does not match. Asking all of them is what keeps this from depending on which one
    happens to come out first.
    """
    from functional_process.indat import objective_selection  # noqa: PLC0415

    try:
        nodes, _objective = sand.objective_nodes(
            graph if graph is not None else graph_for(),
            objective_selection(reference.i_figure_merit),
            switch_values,
        )
    except Exception:  # noqa: BLE001 -- an unmarked row, never a lost row
        return None
    for node in nodes.values():
        for port in node.inputs:
            read = port.var.path_str()
            if read in EXPLAINED_OBJECTIVE_READS:
                return EXPLAINED_OBJECTIVE_READS[read], read
    return None


def _process_objective(reference, root_find):
    """PROCESS's **own** converged objective, or `None` when it forms none.

    `objective_function(i_figure_merit, data)` is the function PROCESS's solver
    maximises or minimises, read at the converged `DataStructure` -- so this is the
    number the port's `objf` column has to be compared against, and until 2026-08-31
    this table had no column for it at all. Every "matches PROCESS" claim in the record
    before then compared the port against *itself* under a different seeding mode, which
    is exactly `_audit/next_steps.md` §17.2's error.

    `root_find` short-circuits it, and not as an optimisation: a file whose
    `i_process_run_mode` is `-2` is answered by `_Fsolve`, whose `solve` ends
    `self.objf = None` and whose output writer omits the figure-of-merit line entirely.
    `reference.i_figure_merit` is `7` on both such files **because `numerics.py:154`
    defaults it there**, not because the file or the solver ever chose it -- evaluating
    that metric would produce a number PROCESS never formed and print it in a column
    headed "PROCESS".
    """
    if root_find:
        return None
    from process.core.solver.objectives import objective_function  # noqa: PLC0415

    try:
        return float(objective_function(reference.i_figure_merit, reference.data))
    except Exception:  # noqa: BLE001 -- an empty cell, never a lost row
        return None


def _against_process(store, oracle, process_objf, explained=None, ixc=None):
    """Fill a formulation's `dx`/`dobjf`/`explained` -- the port's answer against
    PROCESS's own.

    **`oracle` is PROCESS's run and only PROCESS's run.** On a
    `--native --compare-process` row the solve was seeded and stated from a
    `native.NativeReference` and this object is a separate, disk-cached `ReferenceRun`
    loaded for scoring; the caller passes the right one and this function does not know
    or care which mode produced `store`.

    `ixc` is the **design vector's own order**, which is the order `store["_x"]` was
    written in, and it is compared against `oracle.ixc` rather than assumed equal:
    `native.native_reference` sorts the file's `ixc` and `SingleRun.init` sorts PROCESS's
    (`native_reference`'s docstring records that this is an eighth initialisation source
    and that three of the seven files state `ixc` out of order). They agree on all seven
    today, and a `dx` column computed by zipping two differently ordered ID lists would
    report a per-variable disagreement that is really a permutation -- silently, and in
    the one column that exists to catch silent disagreement. So a mismatch blanks the
    column instead.

    `dx` is the **worst relative deviation over the `ixc` design vector**, the same
    quantity `run_sand_harness.main`'s per-variable table prints in its `rel` column and
    computed the same way (`|port - PROCESS| / |PROCESS|`, `reference.converged` being
    PROCESS's converged value per iteration-variable ID). It is reported as one number
    plus the ID it occurred at, because on a matrix the row is the unit and the full
    table belongs to the per-configuration harness.

    A `NativeReference` has no `converged` and no objective, so a `--native` row without
    `--compare-process` never reaches here at all (`_solve_both` skips the call) and both
    columns stay `-`, exactly as `PRO` does.
    """
    x = store.get("_x")
    converged = getattr(oracle, "converged", None)
    order = list(oracle.ixc) if ixc is None else list(ixc)
    if x and converged and order == list(oracle.ixc):
        rels = [
            abs(got - converged[i]) / max(abs(converged[i]), 1e-300)
            for i, got in zip(order, x, strict=True)
        ]
        if rels:
            worst = int(np.argmax(rels))
            store["dx"] = rels[worst]
            store["dx_at"] = order[worst]
    if process_objf is not None and store.get("objf") is not None:
        store["dobjf"] = abs(store["objf"] - process_objf) / max(
            abs(process_objf), 1e-300
        )
        if explained is not None and store["dx"] is not None:
            if store["dx"] <= _EXPLAINED_DX:
                store["explained"] = explained[0]
            else:
                # The label is **withheld, and the withholding is itself reported.**
                # This row's objective is downstream of a documented chain, so the
                # tempting reading is "explained" -- but the two objectives were
                # evaluated at two different design points, and a difference measured
                # across two points is not evidence about either. Saying "withheld and
                # here is why" is the only honest cell; silently labelling it would
                # repeat §17.2's error in a new place, and silently dropping the
                # explanation would lose the fact that a documented chain is in play.
                store["withheld"] = explained[0]


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


def compares_by_default(mode: str) -> bool:
    """Does `mode` score its rows against PROCESS unless told not to?

    Yes for the three modes that already run PROCESS to build their seed -- the
    comparison is then free. No for `NATIVE`, which is the mode whose whole claim is
    that PROCESS is not in the path, so paying 4.6 s a row for it is a choice the caller
    makes with `--compare-process` rather than one this file makes for them.
    """
    return mode != NATIVE


def run_one(path, mode=PROVIDER, compare=None, optimiser=None) -> Row:
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

    **`NATIVE` takes a different second phase and seeds from no PROCESS object at all.**
    The other three modes need `reference_run` for the seed they start from; a native row
    starts from `native.native_state` and states its problem with
    `native.native_reference`.

    **Seeding and comparison are separate axes, and `compare` is the second one.** Until
    2026-09-01 they were one: a `--native` row had `process_objf = None` *by
    construction*, so `PRO objf`, `d objf` and `worst dx` were blank, and that was the
    only reason `--provider` still existed. It was never a real coupling. `reference_run`
    is disk-cached (~4.6 s a row, and 0.01 s once warm), and scoring a solve against
    PROCESS's converged answer requires *having* that answer, not *having started from*
    it -- so `compare=True` loads it as an **oracle** and nothing else. It is never
    handed to `mdf.assemble`, `mda_env`, `sand_harness.assemble` or `_seed`; the local is
    called `oracle` rather than `reference` precisely so that a future edit that leaks it
    into the solve path has to rename it first.

    `compare` defaults to `compares_by_default(mode)` -- free where PROCESS already ran,
    opt-in on `NATIVE` via `--compare-process`.
    """
    began = time.perf_counter()
    if compare is None:
        compare = compares_by_default(mode)
    name = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    row = Row(name=name, seed_mode=mode, compared=bool(compare))
    # The file's own problem type, read from its text before anything else runs: a file
    # stating `i_process_run_mode = -2` is a **root find over its equalities**, which is
    # what PROCESS answers it with (`scipy.optimize.fsolve`, no objective, the
    # inequalities evaluated once at the answer). It is read here rather than off a
    # `ReferenceRun` so that every boundary mode -- `NATIVE` included, which runs no
    # PROCESS -- gets the same answer from the same place.
    row.root_find = read_indat(str(path)).problem.is_evaluation
    print(f"\n=== {name} ", "=" * 40, flush=True)
    if row.root_find:
        print("  states a ROOT FIND (i_process_run_mode = -2), not an optimisation")

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

    if mode == NATIVE:
        try:
            reference = native.native_reference(str(path))
        except Exception as failure:  # noqa: BLE001 -- a row, not an exit
            row.note = f"native env failed: {type(failure).__name__}: {failure}"
            row.seconds = time.perf_counter() - began
            print(f"  {row.note}")
            traceback.print_exc()
            return row
        cold = reference.cold
        row.n_ixc, row.n_icc = len(reference.ixc), len(reference.icc)
        row.boundary = _native_counts(cold, mode)
        print(
            f"  native: {row.boundary['written']} value(s) from the file and the "
            f"vendored defaults ({row.boundary['indat']} indat / "
            f"{row.boundary['defaults']} defaults), 0 from PROCESS; "
            f"{len(reference.ixc)} ixc, {len(reference.icc)} icc "
            f"({reference.n_equality} eq), i_figure_merit {reference.i_figure_merit}"
        )
        switch_values = None if is_reference else switch_values_from_indat(str(path))
        oracle = None
        if compare:
            # **Scoring only.** Loaded after the native reference is already built and
            # never passed to anything that assembles, seeds or solves -- see this
            # function's docstring on why the name is `oracle`.
            try:
                oracle = reference_run(str(path))
            except Exception as failure:  # noqa: BLE001 -- an empty column, not a lost row
                row.compared = False
                print(
                    f"  compare: PROCESS run failed, PRO columns stay blank -- "
                    f"{type(failure).__name__}: {failure}"
                )
            else:
                row.process_iterations = oracle.solver_iterations
                row.process_objf = _process_objective(oracle, row.root_find)
                said = (
                    "formed no objective (evaluation mode)"
                    if row.root_find
                    else repr(row.process_objf)
                )
                print(
                    f"  compare: PROCESS {said} in {oracle.solver_iterations} "
                    f"iteration(s) -- SCORING ONLY, this row is seeded natively"
                )
        return _solve_both(
            row,
            reference,
            machine_graph,
            switch_values,
            cold,
            began,
            oracle=oracle,
            optimiser=optimiser,
        )

    try:
        reference = reference_run(str(path))
    except Exception as failure:  # noqa: BLE001 -- a row, not an exit
        row.note = f"PROCESS run failed: {type(failure).__name__}: {failure}"
        row.seconds = time.perf_counter() - began
        print(f"  {row.note}")
        traceback.print_exc()
        return row
    row.n_ixc = len(reference.ixc)
    row.n_icc = len(reference.icc)
    # These three modes seed *from* `reference`, so PROCESS ran whatever `compare` says.
    # `compare=False` still suppresses the comparison columns -- the axes are separate in
    # both directions, and `--no-compare-process` is how a reader asks for the port's own
    # numbers with nothing of PROCESS's answer beside them.
    oracle = reference if compare else None
    if oracle is not None:
        row.process_iterations = oracle.solver_iterations
        row.process_objf = _process_objective(oracle, row.root_find)
    print(
        f"  PROCESS: {reference.solver_iterations} "
        f"{'fsolve' if row.root_find else 'VMCON'} iterations in "
        f"{reference.solve_seconds:.1f} s, conv "
        f"{reference.convergence_parameter:.2e}; "
        f"{len(reference.ixc)} ixc, {len(reference.icc)} icc "
        f"({reference.n_equality} eq), objf "
        + (
            "none formed (evaluation mode)"
            if row.root_find
            else ("not compared" if oracle is None else f"{row.process_objf!r}")
        )
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

    return _solve_both(
        row,
        reference,
        machine_graph,
        switch_values,
        cold,
        began,
        oracle=oracle,
        optimiser=optimiser,
    )


def _solve_both(
    row, reference, machine_graph, switch_values, cold, began, oracle=None, optimiser=None
):
    """Cold MDF and cold SAND for one configuration, each a row rather than an exit.

    Factored out of `run_one` when `NATIVE` arrived: the two modes differ entirely in
    *where the four arguments come from* and not at all in what is done with them, and a
    second copy of this loop would be the place a difference crept in unnoticed.

    **An evaluation-mode file gets one arm, not two, and the formulation split is not a
    thing that exists for it.** MDF and SAND are two ways of *distributing an
    optimisation*: MDF hands the optimiser PROCESS's `ixc` and converges the MDA inside
    each evaluation, SAND hands it the design and the coupling together and holds them
    with residual equalities. A file stating `i_process_run_mode = -2` poses no
    optimisation to distribute -- PROCESS root-finds the equalities alone with
    `scipy.optimize.fsolve` and forms no objective at all -- and MDF's design vector *is*
    `ixc`, so the MDF root find poses **exactly** the square system PROCESS poses. A SAND
    row would state a larger square system over design *and* coupling that PROCESS never
    writes down, and reporting it beside PROCESS's answer under a "formulation" heading
    implies a comparison that has no content. So the run is one row, and the row is MDF's
    because MDF is the one that is PROCESS's own problem.

    **`oracle` is the comparison side and `reference` is the seeding side.** They are the
    same object for the three provider/seed modes and *different* objects for
    `--native --compare-process`, which is exactly why they are two parameters: `oracle`
    reaches `_against_process` and nothing else, and `reference` reaches every build and
    every solve. `oracle is None` leaves the `PRO` columns blank.
    """
    arms = [("MDF", cold_mdf, row.mdf, {"root_find": row.root_find})]
    if not row.root_find:
        arms.append(("SAND", cold_sand, row.sand, {}))
    explained = _explained_by(reference, machine_graph, switch_values)
    omitted = []
    for label, run, store, kwargs in arms:
        try:
            # Phase split per *arm*, not per row: MDF and SAND compile different programs
            # (§25 measured 38 635 MLIR lines against 132 125 on one machine), and a row
            # total would average the two into a number describing neither.
            phase_timing.reset()
            arm_began = time.perf_counter()
            store.update(
                run(
                    reference,
                    machine_graph,
                    switch_values,
                    cold,
                    optimiser=optimiser,
                    **kwargs,
                )
            )
            row.timings[label] = phase_timing.split(time.perf_counter() - arm_began)
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
        # The one column group on this table that compares the port with **PROCESS's
        # answer** rather than with the port under another seeding mode. Cheap -- both
        # numbers are already in hand -- and it is the check §17.2 got wrong three times.
        # `oracle`, never `reference`: on a `--native --compare-process` row those are
        # two different objects and only one of them is PROCESS's.
        if oracle is not None:
            _against_process(
                store, oracle, row.process_objf, explained, ixc=reference.ixc
            )
        if store.get("_omitted"):
            omitted.append(f"{label} {store['_omitted']}")
        print(
            f"  {label}: {store['iterations']} SQP it, {store['status']}, "
            f"objf {store['objf']}, max|eq| {store['max_eq']}, "
            f"min ie {store['min_ie']}"
            + (
                f", vs PROCESS: worst dx {store['dx']:.2e} at x{store['dx_at']}"
                if store["dx"] is not None
                else ""
            )
            + (f", d objf {store['dobjf']:.2e}" if store["dobjf"] is not None else "")
            + (f" [EXPLAINED {store['explained']}]" if store["explained"] else "")
            + (f" -- {store['note']}" if store["note"] else "")
        )
    if getattr(cold, "missing", None):
        # Every place the run asked for and the native state could not answer, deduped
        # and in the order asked. `mdf.seed`/`mda_env` already turned each into a `0.0`,
        # so this is the difference between a measured hole and a silent zero.
        seen = list(dict.fromkeys(cold.missing))
        row.boundary["unanswered"] = len(seen)
        row.omitted_paths = tuple(f".{a}.{f}" for a, f in seen)
        print(f"  native: {len(seen)} place(s) unanswered -- {row.omitted_paths}")
    row.omitted = "; ".join(omitted) if omitted else ""
    row.seconds = time.perf_counter() - began
    return row


def _native_counts(state, mode) -> dict:
    """`installed`-shaped counts for a native row, so the boundary block still adds up.

    The columns mean what they meant, with the seed's two gone: `written` is every place
    the state answers, `from_process` is **zero by construction** -- there is no PROCESS
    object in a native run to fall back to -- and `held`/`nothing` are zero for the same
    reason. `unanswered` is filled in after the solve, because it is a property of what
    the run asked for and not of the state.
    """
    sources = state.sources
    return {
        "mode": mode,
        "paths": len(sources),
        "supplied": len(sources),
        "independent": len(sources),
        "written": len(sources),
        "held": 0,
        "nothing": 0,
        "from_process": 0,
        "indat": sum(1 for s in sources.values() if s == "indat"),
        "defaults": sum(1 for s in sources.values() if s == "defaults"),
        "unanswered": 0,
    }


_COLUMNS = (
    ("configuration", 22, "{}"),
    ("seed", 6, "{}"),
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
    ("PRO objf", 15, "{}"),
    ("d objf", 9, "{}"),
    ("worst dx", 9, "{}"),
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
                + _cell(SEED_LABEL.get(row.seed_mode, row.seed_mode), 6)
                + " "
                + " ".join(
                    _cell("REFUSED" if i == 0 else None, w)
                    for i, (_h, w, _f) in enumerate(_COLUMNS[2:])
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
        forms = [("MDF", row.mdf)]
        if not row.root_find:
            # One line, not two, for an evaluation-mode file -- `_solve_both` says why.
            forms.append(("SAND", row.sand))
        for form, measured in forms:
            store = measured or _blank()
            lines.append(
                " ".join([
                    _cell(row.name, 22),
                    _cell(SEED_LABEL.get(row.seed_mode, row.seed_mode), 6),
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
                    _cell(
                        "none" if row.root_find else row.process_objf,
                        15,
                        None if row.root_find else "g",
                    ),
                    _cell(store["dobjf"], 9, "e"),
                    _cell(store["dx"], 9, "e"),
                    _cell(store["max_eq"], 10, "e"),
                    _cell(store["min_ie"], 11, "e"),
                    _cell(row.process_iterations, 4),
                ])
            )
            if store["note"]:
                notes.append(f"{row.name} {form}: {store['note']}")
            if store["dx"] is not None:
                notes.append(
                    f"{row.name} {form}: worst dx {store['dx']:.2e} at "
                    f"ixc {store['dx_at']} (over {row.n_ixc} design variable(s), "
                    f"against PROCESS's own converged x)"
                )
            if row.root_find:
                notes.append(
                    f"{row.name}: this file states a ROOT FIND "
                    f"(`i_process_run_mode = -2`), so it gets ONE row and not two. "
                    f"PROCESS root-finds the equalities alone with "
                    f"`scipy.optimize.fsolve`, forms no objective and never examines "
                    f"the inequalities; MDF's design vector IS PROCESS's `ixc`, so this "
                    f"row poses exactly PROCESS's own square system "
                    f"(`mdf.assemble(root_find=True)`). MDF-against-SAND is a split "
                    f"between two ways of distributing an OPTIMISATION, and this file "
                    f"states none -- a SAND row here would solve a larger square system "
                    f"over design and coupling that PROCESS never writes down. The "
                    f"inequalities are still evaluated once at the answer, exactly as "
                    f"PROCESS evaluates them, and reported through `Mdf.reported`."
                )
                if store["objf"] is not None:
                    # Nothing produces this today -- `mdf_graph` mints no objective node
                    # when the file names no figure of merit -- but a number in this
                    # cell would be `objective_metric_7` at `numerics.py:154`'s DEFAULT,
                    # which is not a metric this file or PROCESS ever chose, and it
                    # would sit beside a `PRO objf` of `none` looking comparable.
                    notes.append(
                        f"{row.name} {form}: UNEXPECTED `objf` "
                        f"{store['objf']:.9g} on a root-find row -- PROCESS forms no "
                        f"objective for this file, so this number compares to nothing."
                    )
            if store["withheld"]:
                notes.append(
                    f"{row.name} {form}: objective gap {store['dobjf']:.2e} is NOT "
                    f"labelled explained, and the reason is measured: the design vector "
                    f"moved {store['dx']:.2e} (ixc {store['dx_at']}), so the two "
                    f"objectives were evaluated at two DIFFERENT points and the "
                    f"difference is not evidence about either. This objective does read "
                    f"a path downstream of "
                    f"`mda_harness.EXPLAINED_DISAGREEMENTS[{store['withheld']!r}]`, so "
                    f"part of the gap may be that chain -- but separating the two needs "
                    f"the port's objective at PROCESS'S OWN x, which is a Stage A "
                    f"measurement (`run_sand_harness.py`) and not a matrix row."
                )
            if store["explained"]:
                notes.append(
                    f"{row.name} {form}: EXPLAINED GAP, not a regression -- the "
                    f"objective differs by {store['dobjf']:.2e} at a design vector "
                    f"agreeing to {store['dx']:.2e}, which is "
                    f"`mda_harness.EXPLAINED_DISAGREEMENTS[{store['explained']!r}]`: "
                    f"PROCESS's own converged `DataStructure` is internally "
                    f"inconsistent there and the port is the self-consistent side."
                )
        if row.omitted:
            notes.append(f"{row.name}: CONSTRAINTS OMITTED -- {row.omitted}")
        if row.omitted_paths:
            notes.append(
                f"{row.name}: UNANSWERED NATIVELY (seeded 0.0) -- "
                + ", ".join(row.omitted_paths)
            )
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
            "`PRO` is PROCESS's own solver iteration count for a converged solve of the "
            "same file."
        ),
        "",
        (
            "TWO INDEPENDENT AXES, and this table keeps them apart on purpose. `seed` "
            "is where the"
        ),
        (
            "STARTING STATE came from; the `PRO*` columns are whether the finished "
            "answer was SCORED"
        ),
        (
            "against PROCESS's. `seed nat` with a filled `PRO objf` means no PROCESS "
            "object was in the"
        ),
        (
            "solve path at all and PROCESS's converged answer was loaded afterwards, "
            "for scoring only"
        ),
        (
            "(`run_one`'s `oracle`, `--compare-process`). A blank `PRO*` group means no "
            "comparison was"
        ),
        (
            "asked for -- it is NOT evidence that the row was seeded natively; read the "
            "`seed` column."
        ),
        "",
        (
            "`PRO objf`/`d objf`/`worst dx` compare the port with PROCESS'S OWN ANSWER, "
            "which is the"
        ),
        (
            "only comparison on this table that is not the port against itself under "
            "another seeding mode:"
        ),
        (
            "  PRO objf  `objective_function(i_figure_merit, converged data)`. `none` "
            "means PROCESS FORMED NO"
        ),
        (
            "            OBJECTIVE -- the file states `i_process_run_mode = -2`, so "
            "`_Fsolve.solve` ends"
        ),
        (
            "            `self.objf = None`. `-` means NO COMPARISON WAS ASKED FOR "
            "(`--no-compare-process`,"
        ),
        (
            "            or `--native` without `--compare-process`) -- a statement "
            "about this column, not"
        ),
        "            about the seed.",
        (
            "  d objf    |port - PROCESS| / |PROCESS| on that objective. A row flagged "
            "EXPLAINED in the"
        ),
        (
            "            notes is a DOCUMENTED disagreement, not a regression "
            "(`EXPLAINED_OBJECTIVE_READS`)."
        ),
        (
            "  worst dx  the largest |port - PROCESS| / |PROCESS| over the `ixc` design "
            "vector -- the same"
        ),
        (
            "            quantity `run_sand_harness.py`'s per-variable table prints as "
            "`rel`. The notes"
        ),
        "            block says which iteration variable it occurred at.",
    ]
    out += _boundary_block(rows)
    out += _timing_block(rows)
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


def _timing_block(rows) -> list[str]:
    """Where each arm's wall clock went -- tracing, lowering, compiling, solving.

    **The point of the column, from `_audit/optimise_design.md` §24/§25**: this port's
    cost is compilation, not arithmetic. One schedule lowers to 33 935 MLIR lines
    (132 125 for a tokamak) and `low_aspect_ratio_DEMO`'s 500 SQP iterations are about
    15 s of a ~160 s row. Those were hand measurements on one configuration; this is the
    same question asked of every arm, every pass.

    `solve` is the **residual** of the arm's wall clock after the three measured phases,
    so the four sum to the total by construction and include the graph assembly, the
    PROCESS reference load and every dispatch -- see `phase_timing.split`. Times are
    exclusive, so lowering inside a nested trace is not counted twice.

    Empty when `phase_timing.install()` found jax's internals moved; the header line says
    so rather than printing a table of zeros.

    **The `model` and `other` columns are UNVERIFIED and must not be quoted.** A direct
    probe of `stellarator_helias` -- wrapping `host_cache.flat_conditions` plus
    `flat_condition_jacobian` and running `run_one` on the same tree -- reports `model`
    4.14 s for the whole row against this block's 6.6 + 4.4 = 11.0 s, and `compile`
    11.1 s against 25.3 s. The probe reproduces itself to three figures across two runs,
    so the disagreement is not noise and is not yet explained. `trace`, `lower` and the
    *shape* of the conclusion (compilation dominates) survive it; the per-arm split does
    not. `_audit/next_steps.md` §28.3 carries it as an open item.

    What the probe does establish, twice: **552 calls each** of values and Jacobian for
    108 + 169 = 277 SQP iterations -- about four evaluations per iteration, i.e. the
    line-search trials §21.1 had to instrument `pyvmcon`'s problem object to see --
    at 13.7 ms and 32.7 ms per call *inclusive*, and roughly 3.75 ms per call once the
    trace/lower/compile inside each is subtracted. So a few hundred iterations at a few
    milliseconds genuinely cannot account for the wall clock, and they do not: the
    arithmetic is seconds and the compiler is tens of seconds.
    """
    timed = [(row, arm, split) for row in rows for arm, split in row.timings.items()]
    if not timed:
        return [
            "",
            "PHASE TIMINGS -- unavailable (jax internals moved; see",
            "`phase_timing.install`).",
        ]
    block = [
        "",
        (
            "PHASE TIMINGS -- seconds per arm, EXCLUSIVE. `model`/`other` UNVERIFIED "
            "-- see `_timing_block`."
        ),
        (
            "`model` is graph evaluation inside the driver, `sqp` the optimiser's own "
            "cost (cvxpy, CLARABEL,"
        ),
        (
            "the line search). `other` is the residual: graph assembly, the cached "
            "PROCESS load, dispatch."
        ),
        "",
        (
            "         configuration  form     trace     lower   compile     model"
            "       sqp     other     total"
        ),
    ]
    for row, arm, split in timed:
        total = sum(split.values())
        block.append(
            " ".join([
                _cell(row.name, 22),
                _cell(arm, 4),
                _cell(f"{split.get('trace', 0.0):.1f}", 9),
                _cell(f"{split.get('lower', 0.0):.1f}", 9),
                _cell(f"{split.get('compile', 0.0):.1f}", 9),
                _cell(f"{split.get('model', 0.0):.1f}", 9),
                _cell(f"{split.get('sqp', 0.0):.1f}", 9),
                _cell(f"{split.get('solve', 0.0):.1f}", 9),
                _cell(f"{total:.1f}", 9),
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


PORT_FILES = (
    "mdf.py",
    "sand.py",
    "sand_harness.py",
    "mda.py",
    "mda_harness.py",
    "indat.py",
    "native.py",
    "provider.py",
    "importer.py",
    "run_cold_matrix.py",
    "core/solver/drivers.py",
    "core/solver/host_cache.py",
)
"""The modules a row's numbers depend on, for the provenance header.

Not every file in the package -- the ones a changed byte in could move a cell. A row is
`machine_from_indat` -> `reference_run` -> the provider or the native state -> `assemble`
-> `solve`, and this is that path's source.

`core/solver/drivers.py`/`host_cache.py` joined the list 2026-09-05: `VmconDriver.fused`
lives in the first and `host_cache.bind`'s three programs in the second, and a byte
changed in either can move a cell (it did not, this time -- see `optimise_design.md`
§40 -- but the header would not have known that without these two names in the list)."""


def provenance(mode=PROVIDER, argv=(), compare=None) -> list[str]:
    """The header every table carries: **which tree state these rows were measured on.**

    Emitted by `checkpoint`, not hand-written on top afterwards -- which is the whole
    change. The previous table's header was a hand-added block, so the first re-run
    silently deleted it and a reader of the new file had no way to know what it was
    measured against. A provenance line that a re-run destroys is worse than none,
    because its absence is invisible.

    The dirty-file list is the load-bearing part. `git rev-parse HEAD` alone is a lie on
    a working tree with uncommitted edits, and every table this port has ever produced
    was produced on one -- including this one. So the header names the commit *and* every
    file of `PORT_FILES` that differs from it, because a row measured against
    `mdf.py + 200 uncommitted lines` is not a row measured against that commit.
    """
    import subprocess  # noqa: PLC0415, S404 -- header only, not a solve path

    def git(*args):
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(Path(__file__).resolve().parent.parent),
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return ""

    head = git("rev-parse", "--short", "HEAD") or "unknown"
    subject = git("log", "-1", "--format=%s") or ""
    changed = {
        line[3:].strip()
        for line in (git("status", "--porcelain") or "").splitlines()
        if line[3:].strip()
    }
    dirty = sorted(
        name for name in PORT_FILES if f"functional_process/{name}" in changed
    )
    when = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Generated by `$PY functional_process/run_cold_matrix.py"
        + ("" if not argv else " " + " ".join(argv))
        + "`; do not hand-edit",
        "# the table -- a re-run overwrites it, this header included.",
        "#",
        f"# MEASURED {when}.",
        f"# SEEDED    `--{mode}` -- where every starting value came from.",
        # A third axis, and it belongs beside the other two rather than only inside the
        # echoed command line: two tables that differ in nothing but this are the point
        # of having the flag, and a reader who cannot see which is which has neither.
        f"# DRIVEN BY {'scipy SLSQP (`--slsqp`)' if '--slsqp' in argv else 'VMCON'} "
        f"-- which optimiser answered every `Optimise`.",
        (
            "# SCORED    "
            + (
                "against PROCESS's converged answer (`PRO`/`PRO objf`/`d objf`/"
                "`worst dx`)."
                if compare
                else "NOT against PROCESS -- the `PRO*` columns are blank by request."
            )
        ),
        (
            "#   These are TWO AXES. A `--native` row that is scored still had no "
            "PROCESS object"
        ),
        (
            "#   in its solve path; the answer is loaded afterwards, for the columns "
            "only."
        ),
        f"# TREE: HEAD {head} ({subject[:88]})",
    ]
    if dirty:
        lines += [
            "#   ...plus UNCOMMITTED edits to "
            + ", ".join(f"`{name}`" for name in dirty)
            + ".",
            (
                "#   These rows were measured on that working tree, NOT on the "
                "commit above."
            ),
        ]
    else:
        lines.append(
            "#   Every file in `PORT_FILES` is clean, so these rows are that commit's."
        )
    lines += [
        "#",
        "# A row is one configuration under BOTH formulations; a failure is a row and",
        "# never an exit, so a `REFUSED`/`FAILED`/`no-step` cell is a measurement with",
        "# its reason in the NOTES block, not a gap in the run.",
        "",
    ]
    return lines


def _enable_compilation_cache(directory: str) -> None:
    """Point jax's persistent compilation cache at `directory`, thresholds lowered.

    **Measured** (`_audit/optimise_design.md` §31.20): the whole seven-configuration
    pass goes **410 s -> 184 s** with every result row **bitwise identical**, and the
    `compile` column reads `0.0` on every arm because `backend_compile_and_load` is
    never reached. The cache is 131 files / 43.5 MB.

    **Both thresholds have to be lowered or almost nothing is cached.** At jax's
    defaults only 4 of 228 programs qualified and 6.0--6.5 s of compilation survived
    (§31.7): `min_compile_time_secs` admits only programs that took over a second, and
    this graph's cost is many medium modules rather than a few huge ones.

    **Opt-in, and it should stay that way for measurement runs.** A cache hit does not
    go through the entry point `phase_timing` patches, so `compile` reads zero and that
    time reappears in the `model` residual -- a cached row's phase table is therefore
    *not* comparable with a published one. It is also **not** a memory lever: peak RSS
    moves 2.750 -> 2.802 GiB across the same pass, because §31.16 established the peak
    is resident executables rather than the compiler's workspace.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", directory)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def _return_freed_memory_to_the_os() -> None:
    """`malloc_trim(0)`, because `jax.clear_caches()` frees memory it does not give back.

    **Measured** (`_audit/optimise_design.md` §31.16, 2026-09-02). At the end of a
    `stellarator_helias` row, `gc.collect()` and `jax.clear_caches()` together change
    RSS by nothing measurable, and `malloc_trim(0)` alone takes it **1.611 GiB ->
    0.648 GiB**. glibc's allocator had returned the freed arenas to its own pool and not
    to the kernel, so the next configuration started 1 GiB deeper than it needed to.

    That matters here and almost nowhere else: a pass died twice on
    `LLVM ERROR: Unable to allocate section memory` at the fifth configuration on a 15 GB
    box, and §31.16 established the peak is **resident compiled executables** -- roughly
    200 bytes per character of pre-optimisation StableHLO the row lowers -- rather than
    the compiler's transient workspace. A warm compilation cache removes *all* compile
    time and only **1.2%** of the peak, so it is not the lever; releasing what the
    previous row is finished with is.

    Best effort: glibc-only, and a `malloc_trim` that is absent or fails is not a reason
    to lose a pass that is otherwise working.
    """
    import ctypes  # noqa: PLC0415

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc, or no symbol -- not fatal
        pass


def checkpoint(rows, out=OUT, mode=PROVIDER, argv=(), compare=None) -> None:
    """Write the table as it stands. Called after every configuration; see `OUT`.

    Failing to write the checkpoint must never lose the row that was just computed, so
    an `OSError` here is reported and swallowed -- a read-only tree or a full disk is a
    reason to keep going with an in-memory table, not a reason to discard four minutes
    of PROCESS runs.
    """
    try:
        Path(out).write_text(
            "\n".join(provenance(mode, argv, compare)) + render(rows) + "\n",
            encoding="utf-8",
        )
    except OSError as failure:  # pragma: no cover -- reported, never fatal
        print(f"  (could not checkpoint to {out}: {failure})", flush=True)


def _mode(argv) -> str:
    """The **seeding** mode named on the command line; `PROVIDER` by default.

    Comparison is `_compare`'s axis, not this one.
    """
    for flag, mode in (
        ("--seed", SEED_ONLY),
        ("--native", NATIVE),
        ("--provider-strict", PROVIDER_STRICT),
        ("--provider", PROVIDER),
    ):
        if flag in argv:
            return mode
    return PROVIDER


def _compare(argv, mode: str) -> bool:
    """Should the rows be **scored against PROCESS**? An axis of its own.

    `--compare-process` and `--no-compare-process` set it explicitly; otherwise
    `compares_by_default(mode)` decides, which is `True` wherever PROCESS already ran to
    build the seed and `False` on `--native`. The pairing this exists for is
    `--native --compare-process`: seeded with no `DataStructure` anywhere in the solve
    path, scored against PROCESS's converged answer, and the two facts reported in two
    different places on the table.
    """
    if "--no-compare-process" in argv:
        return False
    if "--compare-process" in argv:
        return True
    return compares_by_default(mode)


def main(argv=None, out=OUT):
    """Walk the configurations, run both formulations cold on each, print the table.

    `--input <path>` may be repeated and replaces the default list entirely; with none
    given every entry of `CONFIGURATIONS` runs. `--out <path>` moves the checkpoint file;
    the table is written there after **each** configuration, so an interrupted run leaves
    every row it finished (see `OUT`).
    `--seed`/`--provider`/`--provider-strict`/`--native` choose where the boundary values
    come from (see this module's docstring); the default is `--provider`.

    `--cache <dir>` turns on jax's persistent compilation cache, which takes the whole
    pass from 410 s to 184 s with bitwise-identical rows (`_enable_compilation_cache`).
    Off by default, because a cached row's phase table is not comparable with a
    published one.

    `--compare-process`/`--no-compare-process` are the **other** axis: whether the
    finished rows are scored against PROCESS's converged answer. They compose freely with
    the seeding flags, and `--native --compare-process` is the pairing the split was made
    for -- no `DataStructure` anywhere in the solve path, and the `PRO` columns filled
    from a disk-cached run loaded afterwards.

    `--slsqp` answers every `Optimise` with `scipy`'s SLSQP instead of VMCON, changing
    **nothing else**: the same assembly, the same bounds, tolerance and iteration cap,
    the same equality/inequality counts read off the definition, the same seeding and
    scoring. That is what makes the two tables comparable, and why the class rather than
    a built driver is what travels (`mda.default_drivers`' `optimiser`). Write it
    somewhere other than `OUT` -- `reference_slsqp_matrix.txt` is where the published one
    lives -- since a VMCON table and an SLSQP table are not the same measurement.
    """
    argv = sys.argv[1:] if argv is None else argv
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    paths = [_resolve(p) for p in (chosen or CONFIGURATIONS)]
    mode = _mode(argv)
    compare = _compare(argv, mode)
    optimiser = SlsqpDriver if "--slsqp" in argv else None
    # Patch jax's trace/lower/compile entry points before the first graph is built, so no
    # phase is missed. Idempotent, and a `False` costs the timing block, not the run.
    timed = phase_timing.install()
    cache_dir = argv[argv.index("--cache") + 1] if "--cache" in argv else None
    if cache_dir:
        _enable_compilation_cache(cache_dir)
    print(
        f"seeding: {mode}    scored against PROCESS: {compare}    "
        f"optimiser: {'SLSQP' if optimiser else 'VMCON'}    "
        f"phase timing: {'on' if timed else 'UNAVAILABLE (jax internals moved)'}"
    )
    if cache_dir:
        print(
            f"compilation cache: {cache_dir} -- the `compile` column will read 0.0 and "
            f"that time reappears in `model`; this table is NOT comparable with an "
            f"uncached one (see `_enable_compilation_cache`)"
        )
    began = time.perf_counter()
    rows: list[Row] = []
    for path in paths:
        rows.append(run_one(path, mode, compare, optimiser=optimiser))
        checkpoint(rows, out, mode, argv, compare)
        print(f"  (checkpointed {len(rows)} of {len(paths)} row(s) to {out})")
        # Configurations are independent, and jax caches every executable it compiles
        # for the life of the process. A whole pass therefore accumulates all seven
        # graphs' modules, and on 2026-09-01 that stopped fitting: the pass died on the
        # fifth configuration with `LLVM ERROR: Unable to allocate section memory`,
        # twice, while that same configuration run *alone* completes in 158 s at a
        # 4.1 GB peak.
        # `_audit/optimise_design.md` §24 measures one schedule at 33935 MLIR lines, and
        # the same section's `host_cache` change took the SAND Jacobian from 2943 to 6089
        # parameters -- bigger modules, and enough of them to cross the line.
        #
        # Nothing is lost by dropping them: the next configuration is a different graph
        # and would miss every entry anyway, so this frees memory without costing a
        # single cache hit. It is *not* a fix for a leak -- there is no leak, only a
        # cache doing
        # exactly what it promises for longer than this runner needs.
        #
        # **`host_cache._BOUND` used to be cleared here and there is no such memo any
        # more** (`_audit/optimise_design.md` §37). The three programs are module level
        # and jax's own cache holds them, so `jax.clear_caches()` above already releases
        # exactly what that line was for -- one call instead of two, and no port-side
        # cache left to forget about.
        #
        # **This clearing is not what makes a repeated solve slow**, and it is regularly
        # mistaken for it. A *row* is one configuration and the next row is a different
        # graph, so nothing here could have been a hit. What destroys the repeated-solve
        # regime is re-*assembly*, which `functional_process.session` exists to avoid;
        # `_audit/optimise_design.md` §32.2 separates the two and exonerates this line.
        jax.clear_caches()
        _return_freed_memory_to_the_os()
    print(render(rows))
    print(f"\n{len(rows)} configuration(s) in {time.perf_counter() - began:.0f} s")
    return rows


if __name__ == "__main__":
    main()
