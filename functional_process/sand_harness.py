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


_REFERENCE_CACHE_VERSION = "reference-v1"
"""Bumped when `ReferenceRun`'s *contents* change, so an old pickle can never be read
back under a key whose meaning has moved -- `mda_harness._CACHE_VERSION`'s discipline."""


def reference_run(input_file=None, *, use_cache: bool = True) -> ReferenceRun:
    """Run PROCESS in-process to convergence and capture everything the ladder needs.

    **Cached on disk**, keyed exactly as `mda_harness`'s converged runs are -- on the
    input files *and* the state of `process/` -- so a change to either invalidates it.
    The saving is the point: this function performs two `SingleRun`s, of which the
    solve is ~95 s, and `run_cold_matrix` pays it seven times for a ~1900 s pass that is
    mostly PROCESS. `FP_HARNESS_NO_CACHE=1` forces the run.

    **A cached run carries `models=None`**, and that is not an oversight. `models` holds
    live PROCESS `Model` instances, each with its own `self.data`; pickling them and
    re-attaching them to a different `DataStructure` is a correctness question nobody has
    answered, so the cache declines to answer it. Only Stage B needs `models` -- it
    rebuilds `Evaluators` for PROCESS's finite-difference Jacobian
    (`run_mdf_harness.py:107`, `sand_harness.py:548`) -- and both ladder mains therefore
    pass `use_cache=False`. The two callers that do not touch `models`
    (`boundary.py:318`, `run_cold_matrix.py:578`) get the cache for free.
    """
    import pickle  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from process.core.solver.iteration_variables import ITERATION_VARIABLES
    from process.main import SingleRun

    from functional_process.mda_harness import CACHE_DIR, _cache_key  # noqa: PLC0415

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


_MDA_SCHEDULES: dict = {}
"""`graph -> (driven, runnable, schedule, jitted runner)`, built once per graph.

Keyed by the `Graph` itself, which is hashable and frozen, so two `graph_for()` calls
that build equal graphs share one entry -- and, more to the point, one **jit cache**:
`_mda_runner` closes over the `Schedule`, so a fresh `Schedule` object per call would
retrace whatever the cache already held. `run_cold_matrix` calls `mda_env` three times
per configuration (two for SAND, one to prime MDF), and the second and third calls are
what this table is for.

Construction itself is cheap and was never the cost: measured 2026-08-31 at 0.07-0.27 s
per configuration against a 7.8-29 s eager run. It is cached because the jit is, not for
its own sake.
"""


def mda_schedule(graph=None):
    """`(driven, runnable, schedule, run)` for `graph` -- the MDA, assembled once.

    `driven` is the cut graph the SAND layer assembles against; `runnable` is that graph
    with drivers assigned (`guess_sources` is asked of it, not of `driven`, because the
    `^guess.*` ports are minted by `Assign`); `schedule` is what evaluates it; and `run`
    is `schedule` under `equinox.filter_jit`.
    """
    from functional_process.indat import graph_for  # noqa: PLC0415

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
    """`schedule` under `jit`, taking and returning a `PathMap` rather than a `dict`.

    **The dict is what cannot cross the jit boundary, and `PathMap` is the fix.** An
    `Env` is `dict[VarPath, Any]`; jax flattens a dict by *sorting* its keys, and a
    `VarPath` is deliberately unordered (`~/jaxgraph/CLAUDE.md` § Names, "No order"), so
    an env handed to a jitted function raises on the sort before the schedule is
    reached. `cottax.boundary.run` exists to say exactly that, and solves it by taking
    the caller's own pytree instead. `PathMap` solves it a second way, and the one that
    fits here: its paths are aux data (structure jax carries) and only the values are
    children, so a name is a jit cache key rather than something jax orders.

    **Why not `cottax.boundary.run` itself.** `run` ends in
    `collapse(tree, out, exclude=MintKey)` -- the caller's structure back out, minted
    names dropped -- and this harness's callers need those names. `sand.
    residual_condition_scales` looks up `env[unknown]` where a `FixedPointCut`'s unknown
    *is* `^hat.X`, and `degenerate_fixed_points`/`array_valued_problems` differentiate at
    the env's own values including the cut copies. So `run`'s return contract, not its
    jittability, is what rules it out; it stays the right shape for a caller that wants
    its `DataStructure` back. `boundary.seeds` is checked against this module's own
    seeding rather than replacing it -- see `mda_env`.

    Measured on 2026-08-31 (`stellarator_helias`): 805 separate XLA compiles and 14,417
    primitive dispatches eagerly; one compile and one dispatch here.
    """

    @eqx.filter_jit
    def run(values):
        return path_map(schedule(dict(values)))

    return run


_SCHEDULE_RUNNERS: dict = {}
"""`(Schedule, fuse_upstream) -> tuple[step-or-jitted-run, ...]`, built once per key.

Keyed by the `Schedule`, which is a frozen `equinox.Module` over a `Blocking` and
therefore hashable, for the same reason `_MDA_SCHEDULES` is keyed by its `Graph`: the
jitted groups are closures over the steps, so a second `run_schedule` on an equal
schedule must find the *same* callables or it retraces what the cache already holds.
`fuse_upstream` joins the key because the two groupings are different closures over the
same steps and a shared entry would hand one policy's runners to the other's caller.
"""


def run_schedule(schedule, env, whole=None, fuse_upstream=True):
    """`schedule(env)`, jitted whole where that is possible and part by part where not.

    **One jit is tried first, and it is `_mda_runner`'s exactly.** A schedule whose every
    driver traces is one program, one XLA compile, and one dispatch -- which is what
    `mda_env` already gets. `Mdf.eager` is such a schedule *now*: `mdf.traceable_drivers`
    says a `SeededNewtonDriver`'s cold-start fallback raises
    `TracerArrayConversionError` on a traced `start`, and that is **stale** -- `drivers.
    _usable` opens with `isinstance(flat, jax.core.Tracer)`, the one-line upstream fix
    that docstring proposed. Measured on `large_tokamak_nof`: **1 compile, 10.6 s**,
    against 32 and 16.4 s for the walk below (`_audit/next_steps.md` §24.11).

    **The fallback is the SAND solve schedule, and it is not conservatism.** Its `Drive`
    is a `VmconDriver`: a `cvxpy` QP, a `pyvmcon` line search and a Python callback, none
    of it traceable at all. So the driver stays exactly where it is, eager, and
    everything on either side of it is fused: maximal runs of `Call` steps become one
    jitted program each, and a `Drive`'s **body** -- the block re-run after the driver
    converges it, which is `Drive.__call__`'s own last line -- becomes another.
    Everything each step was doing op by op through `evaluate._run_acyclic` (a separate
    ~25 ms XLA compile per `jnp` primitive, `_audit/optimise_design.md` §18.6) is then
    one program.

    **Which of the two a schedule gets is measured, not declared**, because nothing
    structural distinguishes them: a driver's traceability is a property of its body, and
    `Drive` carries no flag for it. The whole-schedule jit is attempted once per schedule
    and the verdict cached; a failure costs one trace (symbolic, no compile) and falls
    back to a walk that computes the same values by the same nodes in the same order.
    A caller reads `schedule_verdict(schedule)` for which it got and why.

    **Where the eager cost actually sits differs between the two, and the split is not
    the one §18.6 assumed** (`_audit/next_steps.md` §24.11 carries the correction).
    Measured on `large_tokamak_nof`, cold, one process each:

    | | undriven `Call` steps | `Drive` steps |
    |---|---|---|
    | `mdf.prime` (239 `Call`, 6 `Drive`) | 35.8 s, 718 compiles | 13.5 s, 260 |
    | SAND solve (92 `Call`, 1 `Drive`) | 2.8 s, 40 compiles | 105.4 s, 729 |

    The SAND `Optimise` fuses nearly the whole graph into one SCC, so its 92 `Call`
    steps are the *leftovers* and its eager cost is inside the `Drive` -- one
    `_run_acyclic` over the whole block on the way out. Jitting only the `Call` runs
    would have bought that schedule 2.8 s of 108, which is why the body is jitted too.

    **A jitted part takes and returns a `PathMap`, for `_mda_runner`'s reason**: an
    `Env` is `dict[VarPath, Any]`, jax flattens a dict by sorting its keys, and a
    `VarPath` is deliberately unordered. A `PathMap`'s paths are aux data, so a name is
    a jit cache key rather than something jax orders. `check_antichain` -- what
    `cottax.boundary.run` would ask, and what `.tfcoil.dcond` failed before §24.4
    narrowed it -- is measured clean on both schedules of all six assembling
    configurations (§24.11), and `path_map` does not ask it anyway.

    The owned-name guard is `Schedule.__call__`'s and is re-asked here, because walking
    the steps directly is what skips it: a value handed in at a name the run computes
    can only be clobbered unread or read stale under an ordering bug.

    Parameters
    ----------
    whole :
        `False` to skip the single-jit probe for a schedule already known to hold a
        host-side driver; `None` (the default) to probe and cache the verdict.
    fuse_upstream :
        `True` -- the default -- fuses **every** undriven group, including those a
        `Drive` reads: one jit over everything the host-side driver does not sit in.
        `False` restores §19.3's grouping, which left every group upstream of a `Drive`
        eager. That was the default until 2026-09-01 and is kept only so the two
        policies can be diffed; `_audit/optimise_design.md` §20 is the measurement that
        retired it, and §20.9 the flip itself.

    Raises
    ------
    ValueError
        If `env` carries a value at a name this schedule's own nodes produce.
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
    """`(jitted_whole, reason)` for a schedule `run_schedule` has run.

    `reason` is `None` where the single jit took, and the driver's own refusal --
    exception type and message -- where it did not. Exposed because "this schedule is
    one XLA program" and "this schedule is a walk with the driver eager" is a 3x
    difference in cold cost with no visible difference in the answer, so a caller that
    reports timings must be able to say which one it measured.
    """
    return _SCHEDULE_WHOLE.get(schedule) is not False, _SCHEDULE_VERDICT.get(schedule)


def _schedule_runners(schedule, fuse_upstream=True):
    """`schedule.steps` as `Env -> Env` callables: every undriven group one jit, every
    driver eager. Split out so the grouping is `_SCHEDULE_RUNNERS`'s value and not
    rebuilt.

    **The whole schedule is fused except the host-side driver itself**, which cannot
    trace at all (`cvxpy`, `pyvmcon`, a Python callback). `fuse_upstream=False` restores
    the earlier policy -- groups a `Drive` reads left eager -- and is kept only so the
    two can be diffed.

    **What that earlier policy was, and why it is no longer the default.** Fusing a run
    of `Call` steps reassociates its arithmetic, so its outputs move by ~1 ulp -- the
    drift §24.4 recorded for `mda_env`. Downstream of the last `Drive` that is harmless.
    *Upstream* of one it moves the values the driver is handed, and an SQP trajectory is
    not a continuous function of them: measured on `stellarator_helias`'s cold SAND solve
    (`_audit/optimise_design.md` §19), the 25 `Call` steps ahead of its `Drive` produce
    two differing values under the group jit, worst `4.4e-16` relative, one of them
    (`.tfcoil.a_tf_turn_steel`) inside the `Drive`'s own context -- and the solve went
    from **90 iterations converged at `1.21775735`** to **108 `stopped`** at
    `max|eq| 2.85e-02`. So §19.3 left those groups eager, and §19 called that a rule.

    **§20 measured that the rule protects a coin flip, so it was retired (2026-09-01).**
    Perturbing `.tfcoil.a_tf_turn_steel` by hand, eager throughout, reproduces the fused
    108-`stopped` run **bit for bit** at `-1` and `-2` ulp and leaves the 90-converged
    run untouched at `+1` and `+2`. The fusion moves that value by exactly `-2` ulp; the
    eager answer was on the tolerant side of one bit and nothing chose it. `SlsqpDriver`
    flips on `+1` too. On `large_tokamak_nof` the same fusion moves only
    `.heat_transport.etath_liq`, which no `Drive` there reads, and both policies agree
    bitwise for all three drivers -- i.e. the protection was contingent on an accident of
    which values a given fusion reassociates, not on anything a policy can state.

    Speed decided nothing either way. §24.11's split -- on `large_tokamak_nof`'s SAND
    solve the undriven steps are **2.8 s of 108** against the `Drive`'s 105.4 s -- says
    the group jit was always a rounding error against the body jit below, which runs
    *after* the driver has converged and cannot move it.
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
    """A run of undriven steps, unfused, as an `Env -> Env` callable.

    Reached only under `_schedule_runners(fuse_upstream=False)`, the retired policy that
    left every group a `Drive` reads eager; see there, and `_audit/optimise_design.md`
    §20 for what retired it. Kept because diffing the two policies is how that section
    was measured and how any successor to it will be.
    """

    def run(env):
        for step in steps:
            env = step(env)
        return env

    return run


def _jitted_group(steps):
    """One `jit` over a run of undriven steps, as an `Env -> Env` callable.

    `dict(values)` inside the trace rather than outside it: `_run_acyclic` writes into
    the env it is handed, and the caller's own dict must not be one a tracer lands in.
    """

    @eqx.filter_jit
    def jitted(values):
        env = dict(values)
        for step in steps:
            env = step(env)
        return path_map(env)

    def run(env):
        return dict(jitted(path_map(env)))

    return run


def _driven_runner(step, fuse_upstream=True):
    """`Drive.__call__`, with the body's re-run jitted and the driver left eager.

    Re-implemented rather than wrapped because the two halves of `Drive.__call__` need
    different treatment and it does not separate them: the driver is host code (`cvxpy`,
    `pyvmcon`, a Python callback, or a `SeededNewtonDriver` whose cold-start fallback
    raises on a traced start), while the body is the block itself and is exactly what a
    jit is for. The condition map the driver iterates is untouched -- `VmconDriver` jits
    it already -- so this changes nothing the driver sees, only the once-per-solve
    `_run_acyclic` on the way out.

    A nested body (`Drive.body` is the interior's own `Schedule` where `Blocking.inner`
    states one) is run through `run_schedule` instead, so its own drivers stay eager
    too. Nothing in this tree builds one today; this is the branch that keeps that from
    being silently wrong when something does.

    **The binding follows `Drive.__call__` exactly, reports included** -- see the comment
    in `run`. Re-implementing a contract is how a copy drifts from it, and this one had:
    it bound `step.unknowns` alone for as long as no driver reaching it reported
    anything.

    Raises
    ------
    ValueError
        If the driver returns a number of values other than one per unknown followed by
        one per reported kind.
    """
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

    **The run is jitted** (2026-08-31, `_mda_runner`), and it is **not** bit-identical
    to the eager one. Measured key by key against the eager envs, warm and cold:

    | | differing keys | worst relative difference |
    |---|---|---|
    | `stellarator_helias` | 254 / 831 | `1.1e-13` |
    | `large_tokamak_nof` | 399 / 1134 | `5.0e-12` |

    **The seeding is not the cause and neither is `_strongly_typed`** -- both were
    isolated: the same schedule run *eagerly* from the normalised env reproduces the old
    env with **0** differing keys, so every difference above is XLA's, not this
    function's. Fusing an evaluation reassociates and contracts its arithmetic, and the
    MDA's drives are iterative, so a last-bit change in a residual moves the step the
    Newton solve takes and lands a few ulp away. The one entry that looks alarming --
    `^cond.stellarator.wp_width_r_min`, `0.0` eagerly and `-2.8e-14` jitted -- is that
    same effect on a *residual*, where a relative measure has nothing to divide by.
    Nothing here approaches the `1e-8` these envs are used at, but a cold SAND solve's
    step count is a discrete function of its seed, so a row moving by one iteration is a
    consequence a reader of any C2/C3 count must know about.
    """
    from functional_process.mda import guess_sources  # noqa: PLC0415

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
            env[var] = _strongly_typed(ground_truth(data, source))
        except (AttributeError, KeyError):
            env[var] = _strongly_typed(0.0)
    return driven, dict(run(path_map(env)))


def _strongly_typed(value):
    """`value` as a jax array whose `weak_type` is `False`, whatever it came in as.

    **A weak type is a distinct jit signature, and warm and cold data differ in exactly
    that.** A Python `float` traces weak; a `numpy.float64` traces strong. PROCESS
    overwrites every iteration variable in place with a numpy scalar on its first model
    call (`set_scaled_iteration_variable`), so `reference.data` hands back strong values
    at the very places `reference.cold` -- the un-run `DataStructure`, still holding the
    input file's own Python floats -- hands back weak ones. Measured 2026-08-31: 13 such
    names on `stellarator_helias` and 26 on `large_tokamak_nof`, every one of them a
    `weak_type` difference and nothing else. Left alone they made the warm and the cold
    env two signatures of one jitted schedule, so `run_cold_matrix`'s two SAND calls per
    configuration paid the ~22 s compile **twice**.

    Normalising is safe because weakness only decides *promotion*, and with
    `jax_enable_x64` on there is nothing to promote to: `float64` is the widest float
    either way, and the env-identity check below is what confirms it rather than this
    paragraph. `lax.convert_element_type` to a value's own dtype is the documented way
    to spend the weakness; it is a no-op on anything already strong.
    """
    array = jnp.asarray(value)
    return jax.lax.convert_element_type(array, array.dtype)


def assemble(reference, driven, env, omit=(), switch_values=None, keep=()):
    """The SAND graph for `reference`'s own `ixc`/`icc`/`i_figure_merit`.

    `keep` names declared problems that are **not** lifted into the SQP -- neither
    residualised nor combined, so their unknowns stay out of `Optimise.design` and their
    residuals out of `Optimise.equalities`. It is forwarded to `sand.sand_graph`, whose
    docstring carries the reasoning; a caller passing it must also pass
    `sand.sand_schedule(nest=True)`, and a kept problem is never dropped as degenerate
    or array-valued here (dropping it would answer a different question from leaving it
    in the graph). Empty -- every published row -- is the path that was always taken.

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
