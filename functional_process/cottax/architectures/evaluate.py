"""Seeding a schedule from an input state, and running it.

Three things every architecture needs before its recipe can be run, and none of them
is part of the recipe:

- **`ground_truth`**: what a `data`-shaped state (PROCESS's `DataStructure` or the
  port's own `native.NativeState`) says at a `VarPath`, including the minted names
  PROCESS never stores (`KNOWN_MINT_VALUES`).
- **`seed_env` / `seed_block` / `cold_state`**: an env for a schedule's inputs and a
  block's unknowns, read off that state -- design variables from the file, coupling
  copies from a completed MDA at the same design.
- **`run_schedule`**: the schedule, jitted whole where every driver traces and walked
  step by step where one does not.

`mda_schedule`/`mda_env` are the plain MDA -- the cut graph with `mda.default_drivers`
on every problem -- assembled once per graph and run from a state.
"""

from __future__ import annotations

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.pytree.executable import ExecutableGraph
from cottax.execution import RunnableGraph
from cottax.execution.schedule import Drive, Schedule
from cottax.pytree.names import PathMap, unminted
from cottax.pytree.plan import Delete
from cottax.pytree.problem import is_fixed_point
from cottax.execution.crossings import get_at

from functional_process.cottax.architectures.drivers import SweepDriver
from functional_process.cottax.architectures.mda import (
    SCHEME,
    assign_drivers,
    cut_graph,
    default_drivers,
    given_start,
    guess_sources,
)
from cottax.mdao_architectures import GaussSeidel
from functional_process.cottax.input.indat import STATED_VALUES, graph_for

ROOT = pathlib.Path(__file__).resolve().parents[3]
"""The repository root: what a relative input-file name is read against."""


def resolve(name: str) -> pathlib.Path:
    """`name` as an absolute path, read relative to the repository root when it is not
    already absolute.
    """
    path = pathlib.Path(name)
    return (path if path.is_absolute() else ROOT / path).resolve()


# ------------------------------------------------------------------ the graph

EXCLUDED_NODE_NAMES: tuple[str, ...] = ()
"""Nodes to drop from a graph before assembly -- **empty, and worth keeping empty.**

Its one entry was `.vacuum.duct_diameter_root_find`, an island `vacuum/namespace.py`
registered on purpose: no real `DataStructure` field backs any of its `VarPath`s and no
other node read what it produced, so every architecture deleted it again here before
solving anything. That made the declared graph and the graph that ran differ by one
solve, for a reason no graph operation accounts for -- a name matched in a tuple. The
island is unregistered now, so nothing needs deleting.

A node that has to be removed before a graph can be assembled is a statement about the
model, not about the harness. Unregister it where it is declared, and this stays empty.
"""


def without_excluded(graph):
    """`graph` minus every node `EXCLUDED_NODE_NAMES` names."""
    to_delete = tuple(
        n for n in graph.nodes if any(name in n.spelling for name in EXCLUDED_NODE_NAMES)
    )
    if not to_delete:
        return graph
    return Delete(to_delete).apply(graph)


# ----------------------------------------------------------- ground truth

KNOWN_MINT_VALUES = {
    # --- the three incoming-value places that replaced three self-loops ---
    #
    # `IonVolAvgTemperature`, `DrTfPlasmaCaseFromInput` and `DeltaEtaStep` each used to
    # read the field they own, which made each of them a `FixedPointFunction` with a
    # cut, a minted `^cond.` copy and a driver. None of the three ever read a previous
    # iterate: the value is PROCESS's *incoming* field -- the IN.DAT value for the
    # first two ("use the input directly" is what `f_temp_plasma_ion_electron <= 0`
    # means, and the second clamps the input against a geometric floor), and an inert
    # read for the third. Each now reads a free `..._in` place instead, and each
    # resolves here to the very field it used to read, which is why no value moves.
    #
    # Nothing new is asked of PROCESS: these are identities, not inversions, and no
    # IN.DAT name, `INPUT_VARIABLES` entry or `DataStructure` field is added.
    ".physics.temp_plasma_ion_vol_avg_kev_in": (
        lambda d: d.physics.temp_plasma_ion_vol_avg_kev
    ),
    ".tfcoil.dr_tf_plasma_case_in": (lambda d: d.tfcoil.dr_tf_plasma_case),
    ".power.delta_eta_in": (lambda d: d.power.delta_eta),
    # `.stellarator.coilcurrent` -- a local in `st_coil` (`process/models/stellarator/
    # coils/calculate.py:46,378`), never stored, but exactly recoverable from two real
    # fields: `calculate.py:276` writes `data.tfcoil.c_tf_total = data.tfcoil.n_tf_coils
    # * coilcurrent * 1.0e6`, so this inverts that same line. This port already
    # implements the identity in the same direction at `coils/quench.py:201`.
    ".stellarator.coilcurrent": (
        lambda d: d.tfcoil.c_tf_total / (d.tfcoil.n_tf_coils * 1.0e6)
    ),
    # `.impurity_radiation.pden_impurity_rad_total_mw` -- an `ImpurityRadiation`
    # instance attribute (`process/models/physics/impurity_radiation.py:667-668,737`),
    # never stored, but inverted from `process/models/physics/radiation_power.py:132`
    # (`pden_plasma_rad_mw = imp_rad.pden_impurity_rad_total_mw + pden_plasma_sync_mw`).
    #
    # **The `pden_impurity_core_rad_total_mw` sibling deliberately gets no entry.** Its
    # analogous inverse (`radiation_power.py:128-129`) is *not* valid: PROCESS clips
    # `.physics.pden_plasma_core_rad_mw` at zero after storing it
    # (`process/models/stellarator/stellarator.py:2153-2155`), so the stored value is
    # not recoverable back through the sum. `pden_plasma_rad_mw` itself is written
    # unclipped (`ibid.:2151`), which is what makes the entry above sound.
    ".impurity_radiation.pden_impurity_rad_total_mw": (
        lambda d: d.physics.pden_plasma_rad_mw - d.physics.pden_plasma_sync_mw
    ),
    # `.physics.pden_plasma_core_rad_mw_unclipped` / `..._outer_...` --
    # `PlasmaRadiationPowers`'s outputs *before* `st_phys`'s zero-clip
    # (`stellarator.py:2152-2159`), which `ClippedRadiationPowers` applies. PROCESS
    # overwrites the field in place, so only the post-clip value is ever stored: these
    # two mints equal the stored field **exactly whenever the clip is inactive**, and
    # are a lower bound on it otherwise -- the same discipline, and the same caveat, as
    # `.stellarator.wp_width_r_min` below. Measured on this run: core `0.0575`, outer
    # `0.0553`, both positive, so both are exact here. A run that clipped would show up
    # as a disagreement on these two mints and *not* on the real fields, which is the
    # right way round: the real fields would still be right.
    ".physics.pden_plasma_core_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_core_rad_mw
    ),
    ".physics.pden_plasma_outer_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_outer_rad_mw
    ),
    # --- the three entries that let the coil island come out of EXCLUDED_NODE_NAMES ---
    #
    # `.stellarator.wp_width_r_min` -- `Intersect`'s unknown, the raw crossing point of
    # `intersect(wp_width_r, lhs, wp_width_r, rhs, ...)`. A local in
    # `winding_pack_total_size` (`process/models/stellarator/coils/
    # calculate.py:452-465`), never stored. But `calculate.py:481,489` write
    # `awp_rad = wp_width_r_min` straight into the real field
    # `data.tfcoil.dr_tf_wp_with_insulation`, *after* the turn-size
    # clamp `wp_width_r_min = max(dx_tf_turn_general**2, wp_width_r_min)`
    # (`calculate.py:465`, ported at `coils/calculate.py:778`). So this identity is
    # exact whenever the clamp is inactive, and a lower bound otherwise. Measured on
    # this run: `dx_tf_turn_general**2 = 3.136e-03` against
    # `dr_tf_wp_with_insulation = 7.170e-01`, i.e. the clamp is inactive by 228x, so the
    # seed is exact here. It is in any case only a `RootFind`'s **starting guess** --
    # `Intersect` re-solves its own residual from it, and the solved answer is then
    # compared back against this same value like any other output. A clamped run would
    # start the Newton slightly low and still converge to the true crossing.
    ".stellarator.wp_width_r_min": (lambda d: d.tfcoil.dr_tf_wp_with_insulation),
    # `.tfcoil.a_tf_wp_no_insulation` / `.tfcoil.a_tf_wp_with_insulation` -- Python
    # locals in `winding_pack_total_size` (`calculate.py:494`/`:498`; the source's own
    # comment on `:493` says "(not global)"), never stored, but written *from* real
    # fields on those same lines:
    #
    #   a_tf_wp_no_insulation   = awp_tor * awp_rad
    #                           = .tfcoil.dx_tf_wp_primary_toroidal
    #                             * .tfcoil.dr_tf_wp_with_insulation
    #   a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2*dx_tf_wp_insulation)
    #                             * (dx_tf_wp_primary_toroidal + 2*dx_tf_wp_insulation)
    #
    # -- `calculate.py:483-491` assign `awp_tor`/`awp_rad` to those two real fields
    # immediately before, so all three right-hand sides are stored values. Independently
    # confirmed on the converged run rather than read off the source alone: PROCESS's own
    # `data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation`
    # (`calculate.py:499`) is reproduced to the last printed digit
    # (`3.0620392270788945e7`) by dividing `.tfcoil.c_tf_total / n_tf_coils` by the
    # reconstruction above.
    #
    # Not circular as a *comparison*: the right-hand sides are PROCESS's stored numbers,
    # so scoring the port's `a_tf_wp_*` against them checks that the port's own resolved
    # `wp_width_r_min` matches PROCESS's, which is exactly what `Intersect` is on the
    # hook for.
    ".tfcoil.a_tf_wp_no_insulation": (
        lambda d: d.tfcoil.dx_tf_wp_primary_toroidal * d.tfcoil.dr_tf_wp_with_insulation
    ),
    ".tfcoil.a_tf_wp_with_insulation": (
        lambda d: (
            (d.tfcoil.dr_tf_wp_with_insulation + 2.0 * d.tfcoil.dx_tf_wp_insulation)
            * (d.tfcoil.dx_tf_wp_primary_toroidal + 2.0 * d.tfcoil.dx_tf_wp_insulation)
        )
    ),
    # --- the two `.tokamak.build` mints (`_audit/units/models/build.md`) --------------
    #
    # Both are values PROCESS computes and then **overwrites in place**, so neither has a
    # `DataStructure` field of its own -- the same shape as the two `_unclipped`
    # radiation mints above, and the same resolution: each is reconstructible from stored
    # fields by an identity read straight off PROCESS's own source.
    #
    # `.tfcoil.dx_tf_wp_conductor_max` in particular is one edge of the build/winding-
    # pack cycle `mda.CUTS` closes.
    ".tfcoil.dx_tf_wp_conductor_max": (
        lambda d: (
            d.tfcoil.dx_tf_wp_primary_toroidal
            - 2.0 * (d.tfcoil.dx_tf_wp_insulation + d.tfcoil.dx_tf_wp_insertion_gap)
        )
    ),
    # `.build.r_tf_outboard_mid_unrippled` -- `process/models/build.py:1901-1909`, the
    # outboard stack-up *before* `:1939` may raise it to satisfy the ripple constraint.
    # Exact whenever the ripple constraint is inactive and a lower bound otherwise,
    # which is the same caveat `.stellarator.wp_width_r_min` carries and the same way
    # round: a run where the constraint bit would disagree on this mint and not on
    # `.build.r_tf_outboard_mid` itself. `13.988666666666669` on
    # `large_tokamak_eval.IN.DAT`.
    ".build.r_tf_outboard_mid_unrippled": (
        lambda d: (
            d.build.r_shld_outboard_outer
            + d.build.dr_shld_blkt_gap
            + d.build.dr_vv_outboard
            + d.build.gapomin
            + d.build.dr_shld_thermal_outboard
            + d.build.dr_tf_shld_gap
            + 0.5 * d.build.dr_tf_outboard
        )
    ),
}

KNOWN_MINT_VALUES.update(STATED_VALUES)

UNWRITTEN_BY_PROCESS = float("nan")
"""What `ground_truth` seeds for a field PROCESS itself leaves `None`."""


def ground_truth(data, var):
    """`data`'s own value at `var`: a `KNOWN_MINT_VALUES` rule where one exists, else
    the field `unminted(var)` names. `None` there is PROCESS's own "never written" and
    comes back as `UNWRITTEN_BY_PROCESS`.
    """
    known = KNOWN_MINT_VALUES.get(var.spelling)
    if known is not None:
        return known(data)
    value = get_at(data, unminted(var).segments)
    return UNWRITTEN_BY_PROCESS if value is None else value


def strongly_typed(value):
    """`value` as a jax array whose `weak_type` is `False`, whatever it came in as."""
    array = jnp.asarray(value)
    return jax.lax.convert_element_type(array, array.dtype)


# ---------------------------------------------------------------- the MDA

_MDA_SCHEDULES: dict = {}
"""`(graph, scheme) -> (driven, runnable, schedule, run)`, built once per key."""


def mda_schedule(graph=None, scheme=SCHEME):
    """`(driven, runnable, schedule, run)` for `graph` -- the MDA, assembled once.

    `scheme` opens the raw graph's cycles and closes each with a consistency statement
    (`cottax.mdao_architectures`): `mda.SCHEME` by default.
    """
    key = (graph if graph is not None else graph_for(), scheme)
    cached = _MDA_SCHEDULES.get(key)
    if cached is None:
        driven = cut_graph(without_excluded(key[0]), scheme)
        runnable = assign_drivers(driven, default_drivers(driven))
        schedule = Schedule(RunnableGraph(runnable))
        cached = _MDA_SCHEDULES[key] = (
            driven,
            runnable,
            schedule,
            jit_schedule(schedule),
        )
    return cached


def jit_schedule(schedule):
    """`schedule` under one `jit`, taking and returning a `PathMap` rather than a
    `dict`. `run_schedule` is the one to call: it falls back to a walk where a driver
    refuses to trace.
    """

    @eqx.filter_jit
    def run(values):
        return schedule.run(values)

    return run


_SCHEDULE_RUNNERS: dict = {}
"""`(Schedule, fuse_upstream) -> tuple[step-or-jitted-run, ...]`, built once per key."""


def run_schedule(schedule, env, whole=None, fuse_upstream=True):
    """`schedule(env)`, jitted whole where that is possible and part by part where not.

    Raises
    ------
    ValueError
        If `env` holds a value the schedule's own nodes produce.
    """
    if stale := [var for var in env if var in schedule._owned]:
        raise ValueError(
            f"value(s) at {sorted(v.spelling for v in stale)}, which this schedule's "
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
        whole = _SCHEDULE_WHOLE[schedule] = jit_schedule(schedule)
        try:
            out = dict(whole(PathMap(env)))
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
        return dict(whole(PathMap(env)))
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
        return PathMap(env)

    def run(env):
        return dict(jitted(PathMap(env)))

    return run


def _driven_runner(step, fuse_upstream=True):
    """`Drive.__call__`, with the body's re-run jitted and the driver left eager."""
    body = (
        (lambda env: run_schedule(step.body, env, fuse_upstream=fuse_upstream))
        if isinstance(step.body, Schedule)
        else _jitted_group((step.body,))
    )

    def run(env):
        answered = step.driver(step.condition_map(env), step.role_data(env))
        # `Drive.__call__`'s own contract: a driver returns one value per unknown and
        # then one per kind in `driver.reports`, bound as `unknowns + reports`.
        bound = step.unknowns + step.reports
        if len(answered) != len(bound):
            reported = (
                f" and {len(step.reports)} report(s) "
                f"{[v.spelling for v in step.reports]}"
                if step.reports
                else ""
            )
            raise ValueError(
                f"{type(step.driver).__name__} for block "
                f"{[n.spelling for n in step.nodes]} returned {len(answered)} "
                f"value(s) for {len(step.unknowns)} unknown(s) "
                f"{[v.spelling for v in step.unknowns]}{reported}"
            )
        env.update(zip(bound, answered, strict=True))
        return body(env)

    return run


def mda_env(reference, graph=None, data=None, scheme=SCHEME):
    """Run the plain MDA schedule seeded from `data` (default `reference.data`); return
    its output env.
    """
    data = reference.data if data is None else data
    driven, runnable, schedule, run = mda_schedule(graph, scheme)
    # Seeded over the schedule's own inputs, which is where `Assign` minted each
    # driver's `^guess.*` port; a guess port is grounded from the unknown it starts,
    # and a copy `data` has no value for from one cold pass of the graph.
    env = seed_env(data, schedule, runnable, cold_state(data, graph))
    return driven, dict(run(PathMap(env)))


def seed_env(data, schedule, runnable, cold=None) -> dict:
    """Every input of `schedule` read off `data`, a `^guess.*` port off the unknown it
    starts. A place `data` does not hold is read off `cold` -- `cold_state`'s
    first-pass values, or `cold_shapes`' zeros -- and failing that starts at the scalar
    `0.0`, PROCESS's own default for a quantity nothing has written.
    """
    guesses = guess_sources(runnable)
    env = {}
    for var in schedule.inputs:
        source = guesses.get(var, var)
        try:
            grounded = ground_truth(data, source)
        except (AttributeError, KeyError):
            grounded = cold_value(source, cold)
        # A `^guess.*` port may be *given* its value rather than read off `data` --
        # `mda.GIVEN_STARTS` for which, and why a cold dataclass default is not a
        # starting guess. Only guess ports: an ordinary input is the machine's own
        # number and the table has no standing over it.
        if var in guesses:
            grounded = given_start(source, grounded)
        env[var] = strongly_typed(grounded)
    return env


def _place(var):
    """`var` with every mint stripped: the place in the caller's structure it copies."""
    place = var
    while (stripped := unminted(place)) != place:
        place = stripped
    return place


def cold_value(var, cold=None):
    """`cold`'s value at the place `var` is a copy of -- a `ShapeDtypeStruct` there is
    read as zeros of that shape -- or the scalar `0.0` when nothing says.
    """
    if cold:
        known = cold.get(_place(var))
        if isinstance(known, jax.ShapeDtypeStruct):
            return jnp.zeros(known.shape, known.dtype)
        if known is not None:
            return known
    return 0.0


_COLD_SHAPES: dict = {}


def cold_shapes(data, graph=None) -> dict:
    """`{variable: ShapeDtypeStruct}` for every value the MDA computes, by
    `jax.eval_shape` -- traced, never run. **What a cut copy is shaped like**: a scheme
    cuts port-internal variables too (`.physics.nd_plasma_electron_profile` is a
    201-point profile PROCESS never names), and a Picard refuses a scalar guess for an
    array unknown. Cached per graph.
    """
    key = graph if graph is not None else graph_for()
    cached = _COLD_SHAPES.get(key)
    if cached is None:
        _driven, runnable, schedule, run = mda_schedule(key)
        env = seed_env(data, schedule, runnable)
        traced = jax.eval_shape(run, PathMap(env))
        cached = _COLD_SHAPES[key] = dict(traced)
    return cached


_COLD_STATES: dict = {}


def cold_state(data, graph=None) -> dict:
    """`{variable: value}` after **one pass of the graph in binding order** -- what a
    scheme's cut copies start from.

    PROCESS starts a solve the same way: every model reads what the models before it
    in call order just wrote and a data-structure default for anything after it. Said
    on the graph, that is the Gauss-Seidel cut in binding order with each fixed point
    applied once (`drivers.SweepDriver`) and each root find solved, from `data` and
    `cold_shapes`' zeros. A Jacobi from zeros instead divides by a copy nothing has
    written yet and never recovers, and starting one recipe from another's converged
    answer would compare nothing. Cached per graph.
    """
    key = graph if graph is not None else graph_for()
    cached = _COLD_STATES.get(key)
    if cached is None:
        sweep = cut_graph(without_excluded(key), GaussSeidel())
        drivers = default_drivers(sweep)
        for problem in list(drivers):
            if is_fixed_point(sweep[problem]):
                drivers[problem] = SweepDriver()
        runnable = assign_drivers(sweep, drivers)
        schedule = Schedule(RunnableGraph(runnable))
        env = seed_env(data, schedule, runnable, cold_shapes(data, key))
        out = jit_schedule(schedule)(PathMap(env))
        cached = _COLD_STATES[key] = dict(out)
    return cached


def inputs_only(schedule, env):
    """`env` restricted to what the schedule may be handed: its own inputs."""
    inputs = set(schedule.inputs)
    return {var: value for var, value in env.items() if var in inputs}


def seed_block(schedule, drive, base, fallback, design=()):
    """Every schedule input and every block unknown: **design** variables from `base`,
    every other unknown from `fallback` (a completed MDA env at the same design).
    """
    design = set(design)
    env, borrowed = {}, []  # the env doubles as a value store; see `_inputs_only`
    # A `^guess.*` input is a *starting value for* an unknown, so every question below
    # -- is it coupling, is it in `fallback`, what does `base` say -- is asked about the
    # unknown it starts, never about the port's own name. `fallback` is an MDA output
    # env, keyed by real paths, and no `DataStructure` field is spelled `^guess.*`.
    guesses = guess_sources(schedule.executable.graph)
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
