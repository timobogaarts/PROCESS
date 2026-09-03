"""Generic `AbstractDriver`s for `cottax.problem.FixedPoint` and
`cottax.problem.Optimise`, local to this port.

`cottax.evaluate.AbstractDriver`'s own docstring names the intended pairing directly:
*"a Newton drives `RootFind`, a Picard `FixedPoint`, an optimiser `Optimise`."*
Written here, in `functional_process`, not in `~/jaxgraph` -- unlike the earlier
`Feasibility`/`to_graph` gaps (genuine core-library holes, fixed upstream), a driver is
exactly the kind of generic, swappable *solver choice* `AbstractDriver` exists to make
pluggable, not core graph machinery; this port is free to bring its own without asking
`cottax` to grow one. The argument is stronger for `VmconDriver` than for
`PicardDriver`: an SQP is a much larger algorithm choice, and the backing solver
(`pyvmcon`) is a **PROCESS** dependency, not a `cottax` one.

**This docstring used to say "nothing in `cottax` implements the other two", and that is
stale**: `~/jaxgraph/src/cottax/drivers/optimistix.py` now carries `NewtonDriver`,
`ScaledNewtonDriver`, `PicardDriver` and `OptimiseDriver`. The one below is worth
comparing against rather than assuming equivalent -- cottax's `PicardDriver` is built on
`optx.fixed_point`, so it has an **implicit adjoint** where this one differentiates
through its own `lax.while_loop`, and that difference is measurable and large
(`_audit/optimise_design.md` §31.27; `PicardDriver.implicit` below is the same change
made opt-in here). It also runs at `rtol=atol=1e-4` with `max_steps=256` against this
class's `1e-6`/`1e-8` and `max_iter=20`.

Every `FixedPointFunction`/`FixedPointCut` block the factory assembles
(`DeltaEtaStep`, `CryoQNucStep`, the two cut cross-node cycles once cut, ...) is a
candidate for `PicardDriver` -- it answers any `FixedPoint`, generically, the same way
`NewtonDriver` answers any `RootFind`. `VmconDriver` answers the single `Optimise`
`functional_process.sand` assembles.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from cottax.drivers import PicardDriver as CottaxPicardDriver
from cottax.evaluate import AbstractDriver, ConditionMap
from cottax.problem import (
    Converged,
    DriverOut,
    FixedPoint,
    Optimise,
    RootFind,
    Start,
    Steps,
)
from cottax.spec import VarPath
from cottax.tools.path import written
from jax.flatten_util import ravel_pytree

from functional_process.core.solver.host_cache import bind


UNSCALABLE_BELOW = 1e-12
"""Magnitude below which a start value cannot condition its own coordinate.

PROCESS's own threshold, from `check_iteration_variable`
(`process/core/solver/iteration_variables.py`), where it is a hard error. Here it selects
`scale = 1` instead, because a coupling unknown may legitimately converge to ~0.
"""


def design_scale(flat_start):
    """`1 / x_start` per coordinate -- PROCESS's conditioning -- with a floor.

    `np.divide(..., where=)` rather than `np.where(cond, 1/x, 1)`: the latter evaluates
    `1/x` everywhere first, so a near-zero start warns before the select discards it.

    The floor is **PROCESS's own**, not a number invented here. `check_iteration_variable`
    rejects any iteration variable with `abs(value) <= 1e-12` outright, so that is where
    PROCESS itself stops believing a value can condition anything.

    This used to test `flat_start != 0.0`, i.e. exact zero only, reasoning that "exactly
    zero has no scale". True, and insufficient: a coupling unknown that is *numerically*
    zero is not exactly zero. `.power.qac` is exactly `0.0` on a seeded env but
    `-3.8e-27` after a solve, so restarting one solve from another's answer handed VMCON
    a scale of `-2.6e+26` and its QP died -- a failure reachable only by restarting, never
    from a cold start, which is why it survived every run until one was tried.

    PROCESS *raises* on such a value. Raising would be wrong here: SAND legitimately owns
    coupling unknowns whose converged value is genuinely ~0. They are not ill-posed,
    merely unscalable, and leaving `scale = 1` degrades to the unscaled problem in exactly
    those coordinates.
    """
    scale = np.ones_like(flat_start)
    np.divide(  # noqa: RUF069
        1.0, flat_start, out=scale, where=np.abs(flat_start) > UNSCALABLE_BELOW
    )
    return scale


def scaled_problem(driver, conditions: ConditionMap, flat_start, unravel):
    """The pieces every SQP driver here needs, built once from a block's `ConditionMap`.

    Returns `(evaluate, jacobian, both, unravel, scale, condition_scale, bounds)`, where
    `evaluate`/`jacobian` take a **scaled** flat design vector and return
    already-scaled values and Jacobian, `both` returns the pair from **one** compiled
    program, `unravel` puts a flat vector back into the block's unknown pytree, and
    `bounds` is `(lower, upper)` in scaled coordinates.

    Extracted so `VmconDriver` and `SlsqpDriver` differ **only** in which solver they
    hand the same problem to. That is the whole point of having two: if one takes a
    step from a point where the other cannot, the difference is the solver's QP
    handling, and if neither can, the difference is the problem. A shared builder is
    what makes that a controlled comparison rather than two implementations that might
    be scaling differently.

    **Both drivers now genuinely go through here.** They did not: `SlsqpDriver` called
    this function while `VmconDriver.__call__`'s `host` carried a second, textually
    identical copy of the same scaling, bounds and chain rule -- the copy this function
    was extracted *from*, left behind when it was extracted. `SlsqpDriver`'s own
    docstring has claimed the shared builder since, so the claim was stale rather than
    wrong in intent. Unifying them is what gives `both` a caller on the path the cold
    matrix actually runs, and it is bitwise-checkable precisely because the two spellings
    were identical expression by expression (`_audit/optimise_design.md` §31.30).

    **`both` exists for compile time, and only VMCON may use it.** `pyvmcon` demands
    value *and* derivative at every point it evaluates, so a fused program costs it
    nothing and saves the whole value-only trace/lower/compile;
    `scipy.optimize.minimize(method="SLSQP")` calls `fun` alone during its line search,
    so fusing there would pay a full Jacobian per trial point. This function therefore
    hands out all three and lets the driver choose -- see `host_cache.bind`. It is
    **not** bitwise and is therefore behind `VmconDriver.fused`, off; that field carries
    the measurement and the row it moved.

    The scaling rules are `VmconDriver`'s, unchanged and deliberately not re-derived
    here -- design variables by `1 / x_start` (PROCESS's own conditioning, from
    `load_iteration_variables`), conditions by `driver.condition_scale`, and bounds by
    the same design factor with the swap a negative scale forces.

    **Takes an already-flat, already-concrete start** (`flat_start`) and the `unravel`
    that goes with it, rather than the `Start` tuple it used to take. `design_scale`
    needs *values*, and since `SlsqpDriver` runs inside a `jax.pure_callback` its start
    is a tracer everywhere except on the host -- so the ravel happens in the driver,
    outside the callback, and the concrete array arrives here from inside it. Passing
    the tuple would have meant re-raveling a pytree the caller had already raveled, and
    a round trip is exactly where a bit goes missing.
    """
    flat_start = np.asarray(flat_start, dtype=float)

    scale = (
        design_scale(flat_start)
        if getattr(driver, "scaled", True)
        else np.ones_like(flat_start)
    )

    by_name = {var: float(factor) for var, factor in driver.condition_scale}
    stray = set(by_name) - set(conditions.conditions)
    if stray:
        raise ValueError(
            f"condition_scale names {written(tuple(stray))}, which this block does not "
            f"read as a condition (it reads {written(conditions.conditions)})"
        )
    condition_scale = np.array(
        [by_name.get(c, 1.0) for c in conditions.conditions], dtype=float
    )

    # **Bound once here, not per call.** `host_cache.bind` partitions and flattens the
    # `ConditionMap` a single time and hands back two callables that take only `flat_x`,
    # so a per-iteration call flattens 313 pytree leaves instead of 5 462 -- 10.58 ms ->
    # 0.73 ms on `stellarator_helias` MDF, bitwise identical
    # (`_audit/optimise_design.md` §31.14). `bind` memoises, so a second solve of the
    # same block is a cache hit rather than a re-trace, which is §24.1's property kept
    # rather than given back. This function still builds no `jax.jit` of its own.
    bound_values, bound_jacobian, bound_both = bind(conditions, unravel)

    def _scale_values(raw):
        return np.asarray(raw, dtype=float) * condition_scale

    def _scale_jacobian(raw):
        # d/dx_scaled = (d/dx) / scale -- one chain-rule factor per column, one
        # `condition_scale` factor per row.
        return np.asarray(raw, dtype=float) * condition_scale[:, None] / scale[None, :]

    def evaluate(x_scaled):
        flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
        return _scale_values(bound_values(flat_x))

    def jacobian(x_scaled):
        flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
        return _scale_jacobian(bound_jacobian(flat_x))

    def both(x_scaled):
        """`(values, jacobian)` from one program -- `evaluate` and `jacobian` fused.

        Deliberately the *same* post-scaling helpers as the two above, so "fused equals
        split" is a claim about `host_cache.bind`'s two jaxprs and not about two
        transcriptions of the chain rule.
        """
        flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
        raw_values, raw_jacobian = bound_both(flat_x)
        return _scale_values(raw_values), _scale_jacobian(raw_jacobian)

    limits = {var: (lo, hi) for var, lo, hi in driver.bounds}
    lower = np.array(
        [limits.get(v, (-np.inf, np.inf))[0] for v in conditions.unknowns], dtype=float
    )
    upper = np.array(
        [limits.get(v, (-np.inf, np.inf))[1] for v in conditions.unknowns], dtype=float
    )
    # A negative scale (a variable starting below zero) swaps which bound is which.
    scaled_lower = np.where(scale > 0, lower * scale, upper * scale)
    scaled_upper = np.where(scale > 0, upper * scale, lower * scale)
    return (
        evaluate,
        jacobian,
        both,
        unravel,
        scale,
        condition_scale,
        (scaled_lower, scaled_upper),
    )


SUMMARY_MARKER = "SUMMARY: "
"""Where `NonFiniteProblemError.summary` starts inside the message.

The one thing a caller reporting this refusal in a single cell may match on. It is here
rather than in that caller because the string it delimits is written here: a marker
owned by the reader and a message owned by the writer is exactly the pairing that goes
stale silently."""

_SUMMARY_HEADS = 3
"""How many names of each kind `summary` quotes before `...`. Enough to recognise the
failure; a table cell cannot hold thirty paths and the full lists are above it in the
message anyway."""


def _first_few(names) -> str:
    """`(a, b, c, ...)`, or `(none)` -- `summary`'s bracket."""
    if not names:
        return "(none)"
    shown = ", ".join(names[:_SUMMARY_HEADS])
    return f"({shown}{', ...' if len(names) > _SUMMARY_HEADS else ''})"


class NonFiniteProblemError(ValueError):
    """What `_refuse_non_finite` raises -- a `ValueError` with a name to catch it by.

    **Why the name exists.** A caller that wants to survive a non-finite start and
    report it as an *outcome* -- `run_cold_matrix.cold_sand`, which must produce a table
    row rather than an exit -- has to tell this refusal apart from every other
    `ValueError` a solve can raise, and there are several with nothing in common but
    their type: `_refuse_inert_objective`'s, `scaled_problem`'s stray-`condition_scale`
    one, `start_from`'s missing-`Start` one, and whatever `pyvmcon`, `cvxpy` or a model
    body raises. Catching `ValueError` around a solve would label all of those
    "non-finite", which is worse than not catching at all: it turns an unrelated defect
    into a plausible-looking measurement.

    A subclass rather than a sentinel string, because a string in a message is a
    contract nobody can see. `ValueError` remains its base so every existing
    `except ValueError` -- and every caller that only wants the message -- is unaffected.

    **The class does not reach a caller through any schedule, and that is why `summary`
    is also in the message.** Measured 2026-09-03, and it was not the expectation: every
    path in this tree that runs a `VmconDriver` puts it inside a `jax.pure_callback`
    under a **compiled** program -- `cottax.evaluate.Schedule.run` and
    `sand_harness.run_schedule` alike, the latter with `whole=False` included, which is
    the call `run_cold_matrix.cold_sand` makes. XLA cannot carry a Python exception, so
    what such a caller sees is `jax.errors.JaxRuntimeError: INTERNAL: CpuCallback error
    calling callback: <the host traceback, as text>`. The class is gone, the attributes
    are gone, and **only the message text is left**.

    So a caller catching the type alone would have caught nothing on the one path that
    matters, and the one-line rendering a reporting caller needs is put **in the
    message**, after `SUMMARY_MARKER`, where it survives being turned into text. The
    type still earns its name: it is what `_refuse_non_finite`'s own tests assert on,
    and it is what a future eager or in-graph driver would deliver.

    The four tuples plus `summary` are set by `_refuse_non_finite`. Empty on an instance
    built any other way -- they are a convenience for a caller *reporting* this, never
    the check itself.
    """

    summary: str = ""
    """The three lists in one line, for a caller with one cell to put them in.

    Rendered here rather than by the caller so that the eager arrival and the compiled
    one -- attribute and parsed-from-text -- are the *same string* and not two
    formattings that can drift apart."""

    bad_values: tuple = ()
    """Conditions whose **value** is not finite, by `path_str()`."""
    bad_rows: tuple = ()
    """Conditions whose **derivative row** holds a non-finite cell, by `path_str()`."""
    zero_columns: tuple = ()
    """Unknowns whose Jacobian column is identically zero, by `path_str()`."""
    n_conditions: int = 0
    """How many conditions the block declares, so a caller can say "3 of 30"."""


def _refuse_non_finite(values, jacobian, conditions: ConditionMap) -> None:
    """Raise if any condition value or derivative is not finite, naming which.

    This is the port's missing analogue of PROCESS's own guard: `constraint_eqns`
    refuses `nan`, `inf` or `|cc| > 9.99e99` before its solver ever sees them
    (`process/core/solver/constraints.py:1997-2002`). Without it a `nan` row reaches
    `cvxpy` and comes back as *"the problem seems to be non-convex"* -- a message that
    points at the Hessian when the fault is in the constraint matrix.

    That is not hypothetical: it cost a full investigation. A cold SAND start produced
    46 non-finite Jacobian cells confined to exactly two rows, and the whole diagnosis
    was reconstructing which rows they were. This message says it outright.

    Values and derivatives are reported separately because they fail separately: every
    one of the 30 conditions was **finite in value** at that point and only the
    derivatives were `nan`, which is precisely what made the failure look mysterious.

    **This is also the only non-finite check the cold matrix now runs.**
    `run_cold_matrix.cold_sand` used to probe the condition *values* at the seeded start
    through `host_cache.flat_conditions` before solving. That probe was strictly weaker
    than this function -- values only, no derivative rows, no identically zero columns,
    no names -- and it was a whole third XLA module over the same block, built for a
    check this one already makes on the first iterate. It is gone; the row now comes
    from catching `NonFiniteProblemError`, whose docstring says why the type has a
    name.

    Raises
    ------
    NonFiniteProblemError
        Naming the non-finite values, the non-finite derivative rows, and any unknown
        whose Jacobian column is identically zero.
    """
    names = [c.path_str() for c in conditions.conditions]
    bad_values = [n for n, v in zip(names, values, strict=True) if not np.isfinite(v)]
    bad_rows = [
        n for n, row in zip(names, jacobian, strict=True) if not np.all(np.isfinite(row))
    ]
    if not bad_values and not bad_rows:
        return
    unknowns = [u.path_str() for u in conditions.unknowns]
    zeroed = [
        u for u, col in zip(unknowns, jacobian.T, strict=True) if not np.any(col != 0.0)
    ]
    summary = "; ".join([
        f"{len(bad_values)}/{len(names)} non-finite in VALUE {_first_few(bad_values)}",
        f"{len(bad_rows)}/{len(names)} non-finite in DERIVATIVE {_first_few(bad_rows)}",
        f"{len(zeroed)} unknown(s) with an all-zero column {_first_few(zeroed)}",
    ])
    refusal = NonFiniteProblemError(
        "the SQP was handed a non-finite problem, so its QP subproblem cannot be "
        "trusted (a solver will usually report this as non-convexity or infeasibility, "
        "which is not what is wrong):\n"
        f"  non-finite condition values:      {bad_values or 'none'}\n"
        f"  non-finite derivative rows:       {bad_rows or 'none'}\n"
        f"  unknowns with an all-zero column: {zeroed or 'none'}\n"
        "A derivative that is `nan` where the value is finite usually means an "
        "unbounded slope evaluated at its boundary -- `x ** p` with `0 < p < 1`, or "
        "`sqrt`, at exactly `0.0` -- reached because an unknown was started there.\n"
        # Last, on its own line, and after a marker: this is the half a caller with one
        # table cell quotes, and the only half that survives being rendered to text by
        # a compiled `pure_callback`. See `NonFiniteProblemError`.
        f"{SUMMARY_MARKER}{summary}"
    )
    # The same lists as fields, so an *eager* caller need not parse anything.
    refusal.summary = summary
    refusal.bad_values = tuple(bad_values)
    refusal.bad_rows = tuple(bad_rows)
    refusal.zero_columns = tuple(zeroed)
    refusal.n_conditions = len(names)
    raise refusal


def _refuse_inert_objective(jacobian, conditions: ConditionMap) -> None:
    """Raise if the objective's gradient row is identically zero at the **start**.

    The numeric twin of `boundary.inert_conditions`, and the guard whose absence cost
    a session (`_audit/optimise_design.md` §26). `st_regression.IN.DAT` states a real
    optimisation -- `i_process_run_mode = 1`, which PROCESS solves in 16 VMCON
    iterations to `objf = -16.5885765` -- and maximises `FUSION_GAIN_Q`, whose metric
    reads `.current_drive.big_q_plasma`. On a *tokamak* graph nothing owns that path;
    only `models/stellarator/heating.py` does. So it was a boundary input frozen at its
    cold `0.0`, the objective was identically zero with an identically zero gradient at
    every iterate, and VMCON quietly solved the **feasibility** problem that leaves
    while reporting `converged` in 4 iterations. `d objf 1.00e+00` and
    `worst dx 9.90e+01` were the only symptoms and both read as ordinary disagreement.

    **Row 0 is the objective**, by `_Problem.__call__`'s own `f=values[0]`, and the
    columns are the design variables. `grad f == 0` deletes the first term of VMCON's
    convergence test exactly, so the test is then over the constraints alone -- which
    is why the run converged and said so.

    **Checked once, at the first evaluation, and not on every call.** A zero objective
    gradient at an *interior* iterate is a legitimate thing for a well-posed problem to
    reach (a stationary point, or an optimum in every design direction); a zero one at
    the starting point is a statement about the problem rather than about where it got
    to. Refusing per-iteration would turn success into a crash.

    Other identically zero rows are *named and not refused on*. A constraint the
    design cannot steer is common and often intended -- `helias_5b`'s equality 11
    compares a radial build against `.physics.rmajor` on a file whose three iteration
    variables are the temperature, the density and `hfact` -- so listing them helps a
    reader while refusing on them would fail working configurations.

    Raises
    ------
    ValueError
        Naming the objective, the design variables, and any other zero rows.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.size == 0 or np.any(jacobian[0] != 0.0):
        return
    names = [c.path_str() for c in conditions.conditions]
    others = [
        n
        for n, row in zip(names[1:], jacobian[1:], strict=True)
        if not np.any(row != 0.0)
    ]
    raise ValueError(
        f"the objective {names[0]} has an identically zero gradient with respect to "
        f"all {jacobian.shape[1]} design variable(s), so this is not an optimisation: "
        "the SQP will solve the feasibility problem that remains and report it as "
        "converged.\n"
        f"  design variables: {[u.path_str() for u in conditions.unknowns]}\n"
        f"  other conditions with an all-zero row: {others or 'none'}\n"
        "The usual cause is a MISSING PRODUCER -- the objective reads a path this "
        "configuration's graph does not own, so it is a boundary input frozen at its "
        "seed while PROCESS computes a live value. "
        "`$PY -m functional_process.boundary --inert --input <IN.DAT>` names the path "
        "without running anything; see `_audit/optimise_design.md` §26."
    )


class Status(DriverOut):
    """The integer code the driver's own solver library stopped with.

    **Port-local, and deliberately not upstream.** `Steps` and `Converged` mean the same
    thing for every iterative algorithm, so `cottax` ships them; a *status code* does not
    -- `optimistix.RESULTS`, `scipy`'s `OptimizeResult.status` and "which `pyvmcon`
    exception fired" are three different alphabets, and a shared kind would imply an
    agreement that does not exist. What a code means is the driver's to document; the
    problem's place is in the name (`^driver_out.status.<place>`), so which alphabet a
    given value is written in is never in doubt.

    An integer and not a string, because it has to survive a trace: `str(int(tracer))`
    is exactly what forced the old `jax.debug.callback`. A caller that wants words
    renders them from the code, which is what it was doing anyway.

    **Lives here, not in `mdf`, since 2026-09-01.** It was written for
    `mdf.MdfNewtonDriver` because that was the first driver to report one; with
    `VmconDriver` and `SlsqpDriver` reporting one too it belongs beside the drivers that
    write it rather than beside the one assembly that first read it. `mdf` re-exports the
    name, so `mdf.Status` still resolves.
    """

    label = "status"


VMCON_CONVERGED = 0
"""`Status` for a `VmconDriver` solve `pyvmcon.solve` returned from normally."""

VMCON_STATUS: dict[str, int] = {
    "VMCONConvergenceException": 1,
    "QSPSolverException": 2,
    "LineSearchConvergenceException": 3,
}
"""`Status` per `pyvmcon` failure, keyed by **exception class name**.

Keyed by name rather than by class so this table can be written without importing
`pyvmcon` at module scope -- `VmconDriver.__call__` imports it inside the call, and has
since long before this table existed, so that a tree without `pyvmcon` still imports this
module.

The three are genuinely different outcomes and the distinction is the whole reason a
`Status` port exists: `1` is the loop running out of `max_iter` (the point is good, the
tolerance was not reached), `2` is `cvxpy` finding the first QP subproblem infeasible
(usually zero steps taken, and `run_sand_harness._why_no_step` is the instrument that
says why), `3` is the line search failing to make progress along an otherwise valid
search direction. A caller that renders one word for all three is discarding exactly the
information this column was added to carry.

**Anything not in this table takes `1`**, the base class's code, because
`QSPSolverException` and `LineSearchConvergenceException` are the only two subclasses
`pyvmcon` defines and a fourth invented upstream would be a base-class failure until
someone measured it and gave it a number.
"""


def _sqp_callback(conditions: ConditionMap, start, host):
    """`jax.pure_callback` around one host-side SQP solve, plus its verdict.

    **The whole point of this function**: `pyvmcon` and `scipy` are NumPy, so a `Drive`
    answered by either used to refuse to trace at all -- `sand_harness.run_schedule`'s
    whole-schedule jit failed with `TracerArrayConversionError: ... traced array with
    shape float64[14]` on the unknowns vector, and every SAND schedule fell back to a
    step-by-step walk with the driver eager. Wrapped like this the solve is one opaque
    host round trip *inside* one XLA program, exactly as `cottax.drivers.SLSQPDriver`
    does it (`~/jaxgraph/src/cottax/drivers/scipy_slsqp.py`, which is the model this
    copies).

    **The static half rides in the closure.** `eqx.partition(conditions, eqx.is_array)`
    sends only the live arrays across the boundary; the names, the `Step` and every `fn`
    stay in `host`'s closure, because a different graph is a different trace and
    therefore a different closure anyway. `host(dyn, flat)` recombines them.

    **`float64` at the boundary, deliberately.** Both solvers are float64 libraries and
    the model runs in whatever precision the caller enabled, so the conversion is this
    wrapper's and happens at its own edge: `float64` out to the solver, `flat_guess`'s
    own dtype back. Without that, `pure_callback`'s dtype canonicalisation is free to
    move a bit, and this problem is measurably sensitive at the last bit
    (`_audit/optimise_design.md` §21.2).

    **`vmap` over this is sequential**: neither library vectorises, so a batch of solves
    is a host-side loop where a batch of jax-native ones would map.

    **There is no JVP, and this is the honest cost.** `jax.pure_callback` defines no
    derivative rule, so differentiating *through* a converged constrained solve is not
    something this wrap provides -- that needs the implicit KKT system. Measured, not
    assumed: `jax.jvp`/`jax.grad` of a `pure_callback` raises
    `ValueError: Pure callbacks do not support JVP. Please use jax.custom_jvp to use
    callbacks while taking gradients.` So a driver nested inside another differentiated
    block fails **loudly**, with a message naming the fix, and not as a silent zero --
    which is a better failure than the pre-wrap one, though a smaller trade than "it
    just works". `pyvmcon.solve` returns the equality and inequality multipliers the KKT
    system needs, so the road is open; it is unwritten because nothing here nests one.
    The same note `cottax`'s own SLSQP driver makes, for the same reason.

    Parameters
    ----------
    conditions :
        The block's `ConditionMap`; partitioned here, recombined inside `host`.
    start :
        The `Start` tuple, raveled here so `host` receives one flat vector.
    host :
        `f(live_conditions, flat_start_float64) -> (x, steps, converged, status)`, all
        four NumPy, `x` in the block's *unscaled* coordinates.

    Returns
    -------
    :
        `(*unknowns, steps, converged, status)` -- `AbstractDriver.__call__`'s contract
        for a driver whose `reports` is `(Steps, Converged, Status)`.
    """
    flat_guess, unravel = ravel_pytree(start)
    dynamic, static = eqx.partition(conditions, eqx.is_array)

    def wrapped(dyn, flat):
        live = eqx.combine(dyn, static)
        x, steps, converged, status = host(live, np.asarray(flat, dtype=np.float64))
        return (
            np.asarray(x, dtype=np.asarray(flat).dtype),
            np.int32(steps),
            np.bool_(converged),
            np.int32(status),
        )

    answer, steps, converged, status = jax.pure_callback(
        wrapped,
        (
            jax.ShapeDtypeStruct(flat_guess.shape, flat_guess.dtype),
            jax.ShapeDtypeStruct((), np.int32),
            jax.ShapeDtypeStruct((), np.bool_),
            jax.ShapeDtypeStruct((), np.int32),
        ),
        dynamic,
        flat_guess,
        vmap_method="sequential",
    )
    return (*unravel(answer), steps, converged, status)


def start_from(data, driver_name: str, conditions: ConditionMap) -> tuple:
    """The `Start` tuple out of a driver's `data` mapping, or a clear refusal.

    `cottax.evaluate.AbstractDriver.__call__` takes `data`, one tuple per kind named in
    `requires`, rather than the bare `start` these drivers used to receive. Going
    through `Drive`, `Start` is guaranteed present two ways over -- `Drive.__check_init__`
    refuses a driver requiring a kind the problem does not declare, and `Drive.role_data`
    raises on a declared port with no value in `env` -- so this only ever fires for a
    driver called **directly**, which the unit tests do.

    The requirement it states is unchanged and still real: there is no shape to guess a
    pytree of unknowns from, so an absent start is an error, not a silent default.
    """
    start = data.get(Start)
    if start is None:
        raise ValueError(
            f"{driver_name} needs a starting value for every unknown "
            f"({', '.join(v.path_str() for v in conditions.unknowns)}) -- supply one "
            f"in env at its `^guess.*` port, or give this driver a `seed`"
        )
    return start


class SlsqpDriver(AbstractDriver):
    """`scipy.optimize.minimize(method="SLSQP")` answering `Optimise`, on exactly the
    problem `VmconDriver` receives.

    **Why a second SQP.** PROCESS solves this design problem from a cold `IN.DAT` in 46
    VMCON iterations; the port's SAND formulation, cold, takes **zero** steps --
    `cvxpy`/OSQP reports a non-convex KKT matrix inside `pyvmcon` and the line search
    never starts. Two explanations fit that equally well: the QP subproblem's handling,
    or the problem's own conditioning at a cold point. A second, independently written
    SQP with its own QP solver, its own line search and its own Hessian update
    (SLSQP maintains a BFGS approximation with a positive-definite safeguard, where
    `pyvmcon` builds its QP directly) **separates them**: if SLSQP steps where VMCON
    cannot, it is the solver; if neither steps, it is the problem. That experiment is
    the reason this class exists, and it is worth more than either solver's convergence
    record.

    Both drivers build their problem through `scaled_problem` above, so the comparison
    is controlled: same scaling, same bounds, same `jax.jacfwd` Jacobian, same sign
    convention translation. Only the solver differs.

    **Sign conventions.** cottax states inequalities as `g <= 0`; SLSQP's `ineq`
    constraints are `c(x) >= 0`, so they are negated -- the same flip `VmconDriver`
    documents for VMCON's `i >= 0`, arrived at independently for a different library.
    The objective is condition 0, equalities follow, inequalities last: the positional
    contract `ConditionMap` cannot carry (see `VmconDriver.n_equality`).

    Not a replacement for `VmconDriver`: PROCESS's own solver stays the reference for
    any claim about reproducing PROCESS. This is a second opinion, which this project's
    own history says is worth having -- five confident diagnoses were overturned by
    measurement in one session.
    """

    drives = Optimise
    requires = (Start,)

    n_equality: int = 0
    n_inequality: int = 0
    bounds: tuple = ()
    scaled: bool = True
    condition_scale: tuple = ()
    max_iter: int = 100
    tolerance: float = 1e-8
    callback: object = None
    """`f(iteration, x_unscaled) -> None`, or `None`. Plain callable, leaf-free, same
    treatment as `VmconDriver.callback` -- and, like it, now called from inside a
    `jax.pure_callback`."""

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)`, the same three `VmconDriver` reports, so the
        two SQPs can be compared on their verdicts and not only on their answers.

        `Steps` is `OptimizeResult.nit`, `Converged` is `.success`, `Status` is `.status`
        -- **scipy's alphabet, not `pyvmcon`'s**, which is exactly why `Status` is a
        port-local kind whose meaning belongs to the driver that writes it. `0` is
        *"Optimization terminated successfully"*, `9` *"Iteration limit reached"*, `4`
        *"Inequality constraints incompatible"*, and so on; `scipy` owns the list.

        **What is lost against the deleted `Outcome` sink**: `message`, `nfev` and
        `fun`. A message is a string and cannot survive a trace (the whole reason
        `Status` is an integer); `nfev` and `fun` were never read by anything in this
        tree, and either could be added as a further `DriverOut` kind the day something
        wants one.
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, then `steps`, `converged` and `status`.

        Wrapped in `jax.pure_callback` exactly as `VmconDriver` is, and for the same
        reason -- `scipy` is host code, and a `Drive` answered by it would otherwise be
        a hole in any traced schedule. Every caveat `VmconDriver.__call__` and
        `_sqp_callback` state applies here unchanged, the missing JVP included.

        Raises
        ------
        ValueError
            If no `Start` data is supplied, or if the declared equality/inequality
            counts do not account for every condition the block reads.
        """
        from scipy.optimize import minimize

        start = start_from(data, "SlsqpDriver", conditions)
        expected = 1 + self.n_equality + self.n_inequality
        if expected != len(conditions.conditions):
            raise ValueError(
                f"SlsqpDriver was told {self.n_equality} equalities and "
                f"{self.n_inequality} inequalities, i.e. {expected} conditions with the "
                f"objective, but the block declares {len(conditions.conditions)} "
                f"({', '.join(written(conditions.conditions))})"
            )

        _flat, unravel = ravel_pytree(start)
        meq = self.n_equality
        driver, user_callback = self, self.callback
        max_iter, tolerance = self.max_iter, self.tolerance

        def host(live, flat_start):
            # `_both` discarded on purpose. `scipy`'s SLSQP takes separate `fun` and
            # `jac` callables and its line search calls `fun` alone at trial points, so
            # the fused program would pay a whole Jacobian for each of those. The `at`
            # cache below already removes the *duplicate* evaluation SLSQP would
            # otherwise make at one point; what it cannot remove is a derivative scipy
            # never asked for.
            evaluate, jacobian, _both, _unravel, scale, _, (lower, upper) = (
                scaled_problem(driver, live, flat_start, unravel)
            )
            x0 = flat_start * scale

            # One evaluation per point, reused by objective and every constraint: SLSQP
            # calls `fun`, `jac` and each constraint separately at the same `x`, and an
            # evaluation here converges a whole block.
            cache: dict = {}

            def at(x):
                key = x.tobytes()
                if key not in cache:
                    cache.clear()  # only the current point is ever wanted
                    cache[key] = (evaluate(x), jacobian(x))
                return cache[key]

            iteration = [0]

            def objective(x):
                return float(at(np.asarray(x))[0][0])

            def objective_gradient(x):
                return at(np.asarray(x))[1][0]

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda x: at(np.asarray(x))[0][1 : 1 + meq],
                    "jac": lambda x: at(np.asarray(x))[1][1 : 1 + meq],
                },
                {
                    # cottax `g <= 0` -> SLSQP `c(x) >= 0`.
                    "type": "ineq",
                    "fun": lambda x: -at(np.asarray(x))[0][1 + meq :],
                    "jac": lambda x: -at(np.asarray(x))[1][1 + meq :],
                },
            ]
            constraints = [c for c in constraints if len(np.atleast_1d(c["fun"](x0)))]

            def on_step(xk):
                iteration[0] += 1
                if user_callback is not None:
                    user_callback(iteration[0], np.asarray(xk) / scale)

            result = minimize(
                objective,
                x0,
                jac=objective_gradient,
                bounds=list(zip(lower, upper, strict=True)),
                constraints=constraints,
                method="SLSQP",
                options={"maxiter": max_iter, "ftol": tolerance},
                callback=on_step,
            )
            # No `self.last_result = ...`: an `eqx.Module` is frozen, and a driver that
            # mutated itself would not survive being reused across blocks anyway. What
            # the solver said is `reports`' job now, and the mutable `Outcome` sink that
            # used to carry it is deleted.
            if user_callback is not None:
                user_callback(-1, np.asarray(result.x, dtype=float) / scale)
            return (
                np.asarray(result.x, dtype=float) / scale,
                int(result.nit),
                bool(result.success),
                int(result.status),
            )

        return _sqp_callback(conditions, start, host)


class SeededNewtonDriver(AbstractDriver):
    """`cottax.drivers.NewtonDriver`, plus a fallback starting guess derived from the
    block's own **context** when the one supplied in `env` is unusable.

    Exists because a `RootFind`'s starting guess is seeded from the converged
    `DataStructure` (`mda_harness.KNOWN_MINT_VALUES`), and some of those seeds are
    fields PROCESS only has *after* a run. Cold, they are `0.0`. That is fatal rather
    than merely inaccurate for the coil island: measured at a cold design point, the
    `Intersect` residual is **exactly flat** (`-8329.4857`) everywhere below
    `x ~ 0.1`, so at `0.0` the derivative is zero and Newton cannot move at all --
    `optimistix` aborts and takes the whole schedule with it.

    **The coil island no longer uses this fallback, and the reason is the interesting
    part.** `intersect`'s guess *is* PROCESS's own,
    `(r_coil_minor / (20 if i_tf_sc_mat == 6 else 10)) ** 2` -- and the port used to
    discard it, on the argument that a starting guess is a property of the algorithm and
    not an edge of the model. `ROOT_FIND_SEEDS` therefore re-derived it from
    `.stellarator.r_coil_minor` read out of `ConditionMap.context`, which carries what
    the block closed over -- and `r_coil_minor` was only in that context because a
    switch kwarg made the pre-`intersect` node declare `.tfcoil.j_tf_wp` on every
    material, an edge only Bi-2212 has. With `i_tf_sc_mat` split into occupants
    (`_audit/next_steps.md` §14.5) the fake edge is gone and so is the context; the
    occupant owns the guess and `cottax.rewrites.Supply` points the `Start` port at it
    (`mda.supply_starts`). A guess PROCESS computes is an edge of the model after all --
    what is not is the *algorithm* that consumes it.

    Measured cold, before the change: Newton from `0.0` fails, Newton from the guess
    (`0.1786`) converges. The supplied port delivers that same number without a
    fallback, so the measurement still describes why the port exists.

    `seed` is only consulted when the supplied `start` is missing or unusable, so a
    warm run -- every harness here -- keeps its existing starting values and its
    existing answers bit for bit.
    """

    drives = RootFind
    requires = (Start,)

    rtol: float = 1e-4
    atol: float = 1e-4
    seed: object = None
    """`f(ConditionMap) -> tuple` giving one starting value per unknown, or `None`.

    Not `eqx.field(static=True)`: a plain callable is already a leaf-free static field,
    the same treatment `VmconDriver.callback` documents.
    """

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        start = data.get(Start)
        if self.seed is not None and not _usable(start):
            start = self.seed(conditions)
        if start is None:
            raise ValueError(
                f"SeededNewtonDriver needs a starting value for every unknown "
                f"({', '.join(v.path_str() for v in conditions.unknowns)}) -- supply "
                f"one in env at its `^guess.*` port, or give this driver a `seed`"
            )
        flat_guess, unravel = ravel_pytree(start)

        def residual(flat, args=None):
            out, _ = ravel_pytree(conditions(*unravel(flat)))
            return out

        solution = optx.root_find(
            residual, optx.Newton(rtol=self.rtol, atol=self.atol), flat_guess
        )
        return unravel(solution.value)


def _usable(start) -> bool:
    """Whether `start` is a starting guess at all.

    Exactly zero counts as unusable, not as a value. That is a judgement, and it is the
    right one *here*: `0.0` is what this port's seeding writes when a field has no
    converged value to copy, so it means "absent", and a root find started at a
    structural placeholder is not a solve. A block whose true root is genuinely `0.0`
    would be re-seeded to the same neighbourhood by its own `seed` anyway.
    """
    if start is None:
        return False
    flat, _ = ravel_pytree(start)
    if isinstance(flat, jax.core.Tracer):
        # Under a trace there is no value to inspect, and `np.asarray` on a tracer
        # raises. Treat it as usable: a tracer is by construction not the concrete
        # `0.0` placeholder this guard exists to catch, and refusing here would make
        # every `Schedule` containing a seeded Newton unjittable and
        # undifferentiable -- which it was, until `mdf.py` hit exactly that.
        return True
    flat = np.asarray(flat)
    return (
        bool(flat.size)
        and bool(np.all(np.isfinite(flat)))
        # Exact comparison is deliberate, as in `VmconDriver`'s own scaling: it is
        # exactly zero -- the placeholder this port's seeding writes -- that means
        # "absent", not a neighbourhood of zero.
        and bool(np.any(flat != 0.0))  # noqa: RUF069
    )


class PicardDriver(CottaxPicardDriver):
    """`cottax.drivers.PicardDriver` at this port's tolerances -- `optx.fixed_point`, and
    therefore an **implicit adjoint**.

    **A subclass, because the hand-written loop this replaced was a defect.** Until
    2026-09-02 this class carried its own `jax.lax.while_loop`, so jax differentiated
    *through the iterations actually taken* -- and the convergence test makes that trip
    count a function of `x`, so the Jacobian jumped whenever it changed. On
    `stellarator_helias` it does: the four `PicardDriver`s exit after `[2, 2, 3, 4]`
    passes at the start point and `[2, 2, 2, 4]` at iterate 30 of the trajectory, an
    iterate where `objf` blows out to 1.378. `SeededNewtonDriver` above has always gone
    through `optx.root_find` and always had an adjoint; the two were never symmetric.

    An implicit adjoint takes the derivative at the converged point through the implicit
    function theorem, so it **cannot depend on how many passes were taken**. Measured,
    `stellarator_helias` MDF at `1e-6`, perturbing `rmajor`
    (`_audit/optimise_design.md` §31.27):

    | perturbation | 0 | ±1e-13 | ±1e-12 | ±1e-11 | ±1e-10 |
    |---|---|---|---|---|---|
    | the old hand-rolled loop | 66 | 129, 84 | 101, 227 | 399, 206 | **31 stopped, 83 stopped** |
    | this | **45** | 45, 45 | 45, 45 | 45, 48 | 42, 56 |

    **PROCESS takes 46.** Every implicit solve converged, `max|eq| <= 2.36e-07`, where the
    hand-rolled loop failed two of nine and spanned 31--399 iterations.

    **Why PROCESS never had the problem**, and it is not luck: `Caller.call_models` is
    warm-started from the previous evaluation's `DataStructure`, so its pass count is
    locally constant -- measured at 2 across perturbations from `0` to `±1e-8`. This port
    freezes the inner guess with `mdf.prime` **deliberately**, to make the map a pure
    function of `x`; that decision also made the trip count a function of `x`, and without
    an adjoint the trip count reached the derivative.

    **It does not make compilation cheaper** -- the opposite. Forward mode through a
    `while_loop` does not unroll (the tangent rides in the loop carry, and the program is
    byte-identical in size at `max_steps` 20 and 100), so differentiating through was the
    *compact* option: 15 893 StableHLO lines against this class's **17 800** (+12 %), XLA
    5.91 s against 6.12 s. The adjoint's linear solve is added work. It is bought for
    correctness and stability, not speed (§31.28).

    **Tolerances are this port's, the budget is cottax's.** `rtol`/`atol` stay at
    `1e-6`/`1e-8` -- the values the hand-rolled loop used, so the fixed points are held to
    the same standard as before. `max_steps` becomes cottax's **256** rather than the old
    `max_iter = 20`: under `throw=False` a fixed point that runs out of budget returns a
    *non-converged* iterate, and an implicit derivative taken at a point that is not a
    fixed point is not meaningful, so a budget that was merely generous for a
    differentiate-through loop is load-bearing here.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_steps: int = 256

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """`cottax.drivers.PicardDriver.__call__`, behind this port's refusal message.

        The **only** thing this override adds is `start_from`: cottax indexes `data[Start]`
        directly and a missing start therefore surfaces as a bare `KeyError`, where this
        port has always named the driver and told the caller what is missing. The solve
        itself is entirely the superclass's -- no loop, no tolerance handling and no
        adjoint is re-implemented here, which is the whole point of subclassing.
        """
        start_from(data, "PicardDriver", conditions)
        return super().__call__(conditions, data)


class VmconDriver(AbstractDriver):
    """PROCESS's own SQP (`pyvmcon`) answering `Optimise`, fed `jax.jacfwd` instead of
    finite differences.

    **This class is the thesis of the rewrite, expressed as one substitution.**
    `pyvmcon.AbstractProblem.__call__(x) -> Result(f, df, eq, deq, ie, die)` demands
    that the caller supply every derivative at the iterate. PROCESS fills them from
    `Evaluators.fcnvmc2`, which re-runs its entire Gauss-Seidel pipeline `2n` times per
    SQP iteration at a **1 % relative** perturbation
    (`process/core/solver/evaluators.py:118-141`, `data.numerics.epsfcn`). This driver
    fills them from **one `jax.jacfwd` of the block's `ConditionMap`** -- the whole
    Jacobian, exactly, from a single trace. Same algorithm, same problem, same
    convergence test; only the derivative changes, which is what makes any difference
    between this and PROCESS's answer attributable to the model and the derivatives
    rather than to the optimiser.

    Choosing `pyvmcon` rather than `scipy`'s `SLSQP` is deliberate for that reason: it
    is *the same solver* PROCESS calls (`process/core/solver/solver.py:192,246-260`),
    already installed, already a PROCESS dependency. A different SQP would confound "the
    port's model differs" with "the port's optimiser differs".

    What this costs, stated rather than buried
    ------------------------------------------
    `pyvmcon` is NumPy, so the solve runs on the host. Since 2026-09-01 it does so
    inside **one `jax.pure_callback`** (`_sqp_callback`), which is what makes a `Drive`
    answered by this driver *traceable*: the whole SAND schedule is now one XLA program
    with one host round trip in the middle of it, where before the `Drive` refused to
    trace at all (`TracerArrayConversionError` on the unknowns vector) and
    `sand_harness.run_schedule` fell back to a step-by-step walk.

    **"One compile" here means the optimiser is *outside* the compiler, not compiled by
    it.** The SQP loop, the `cvxpy` QP, the line search and this driver's own
    `jax.jit(flat_conditions)`/`jax.jit(jacfwd(...))` all still run as host code inside
    the callback, at their own cost and with their own compiles. What the wrap buys is
    that everything *around* the solve fuses into one program and that the block stops
    being a hole in the graph -- not that VMCON got faster.

    **There is no JVP.** `jax.pure_callback` defines no derivative rule, and
    differentiating through a converged constrained solve needs the implicit KKT
    system, which this wrap does not provide. Measured rather than assumed: `jax.jvp`
    or `jax.grad` through a `pure_callback` raises *"Pure callbacks do not support JVP.
    Please use `jax.custom_jvp` to use callbacks while taking gradients."* -- so a
    `VmconDriver` nested inside another differentiated block fails **loudly, naming the
    fix**, and not as a silent zero. That is a smaller loss than it first looks, and it
    is still a loss: the refusal used to come at *trace* time and now comes only when
    something asks for a derivative. Nothing here nests one. `pyvmcon.solve` returns the
    equality and inequality Lagrange multipliers that system needs, so writing the
    `custom_jvp` is possible -- see `_audit/optimise_design.md` §22 for what it would
    take. Until it exists, treat this driver as an outermost block only.

    Sign convention -- checked, not assumed
    ---------------------------------------
    `cottax.problem.Optimise` declares `inequalities` as `g(x) <= 0`
    (`~/jaxgraph/src/cottax/problem.py:80`); `pyvmcon` declares them feasible when
    `i(x) >= 0` (`pyvmcon.problem.AbstractProblem`'s own docstring). So this driver
    passes `-g`. That matches PROCESS exactly: `constraint_eqns` appends
    `-normalised_residual` with the comment *"Reverse the sign so it works as an
    inequality constraint (cc(i) > 0)"* (`process/core/solver/constraints.py:2007`), and
    `functional_process.sand` puts the **normalised residual** (index 1 of
    `leq`/`geq`/`eq`'s 4-tuple) in each `^cond.*`. Equalities are passed through
    unnegated: `h = 0` and `-h = 0` are the same constraint set, `pyvmcon`'s convergence
    test takes `abs(lambda_eq @ eq)`, and its QP linearisation is sign-symmetric for an
    equality -- only the Lagrange multiplier's sign flips, which nothing reads.

    Splitting the conditions -- and why the driver has to be told
    -------------------------------------------------------------
    `ConditionMap` carries `body`, `unknowns`, `conditions`, `context` and nothing else
    (`~/jaxgraph/src/cottax/evaluate.py:135-178`) -- a flat tuple of condition names with
    no type information and no reference back to the problem. So a driver whose
    `drives = Optimise` **cannot ask** which condition is the objective and which are
    equalities. `Drive` knows (`Drive.problem`), and passes only the `ConditionMap`
    (`evaluate.py:288`). The ordering is nonetheless reliable and derivable:
    `Drive.conditions` is the problem node's `reads` (`evaluate.py:246-249`) and
    `Optimise.inputs` is `(objective, *equalities, *inequalities)`
    (`problem.py:82-84`), so counts recover the split. `n_equality`/`n_inequality` are
    therefore fields, set by whoever assembles the `Drive` from the `Optimise` node
    itself (`functional_process.mda.default_drivers` does exactly that -- it has the
    problem definition in hand and never has to count). `__call__` still re-checks the
    total against `len(conditions.conditions)`, so a stale pair fails loudly rather than
    silently mislabelling a constraint. The upstream fix -- give `ConditionMap` the
    problem definition -- is recorded in `_audit/optimise_design.md` §8.

    Bounds and scaling live here, not on the problem
    -----------------------------------------------
    `Optimise` has `objective`/`design`/`equalities`/`inequalities` and no place for a
    box bound, while PROCESS carries one per iteration variable and hands VMCON `lbs`/
    `ubs` as distinct arguments -- VMCON treats a bound as a bound, and re-expressing it
    as two extra inequality constraints would change the QP subproblem and therefore the
    iterates. `bounds` is keyed by `VarPath`, not positional: cottax's own rule is that
    every query takes exact names, and a positional bound vector is a contract that can
    only be checked at assignment time. An unknown with no entry is unbounded on both
    sides.

    `scaled=True` reproduces PROCESS's own conditioning: `load_iteration_variables` sets
    `scale[i] = 1/x_i` from the **starting** value and hands VMCON `x * scale`, so every
    design variable enters at exactly `1.0`
    (`process/core/solver/iteration_variables.py:348-352`). On the reference run the
    unscaled design vector spans twenty orders of magnitude
    (`nd_plasma_electrons_vol_avg` ~ 1.7e20 against `hfact` ~ 1.06), so this is not
    cosmetic. The scale is derived from `start`, exactly as PROCESS derives it, never
    from a table; an entry that starts at exactly `0.0` keeps a scale of `1.0` (PROCESS
    refuses such a variable outright, `check_iteration_variable`; this driver's unknowns
    include SAND coupling variables PROCESS never sees, so it degrades instead).

    Failure is a returned point, not an exception
    ---------------------------------------------
    `pyvmcon` signals non-convergence by raising (`VMCONConvergenceException` and its
    subclasses). This follows `process/core/solver/solver.py:262-272` and returns `e.x`
    -- the best point reached -- rather than propagating, so a `Schedule` run always
    produces an env. **Which of the four ways it ended is now data**: `reports` names
    `Steps`, `Converged` and `Status`, so the verdict comes back through the
    `^driver_out.*` ports `Assign` mints and lands in the env like every other value.
    `callback` remains, and remains useful -- it is a per-iteration *effect*, which is
    what a callback is legitimately for -- but it is no longer the only way a caller
    learns whether the solve converged, and the mutable `Outcome` sink that used to
    carry `SlsqpDriver`'s verdict is gone entirely.
    """

    drives = Optimise
    requires = (Start,)

    n_equality: int
    n_inequality: int
    bounds: tuple[tuple[VarPath, float, float], ...] = ()
    """`(unknown, lower, upper)`, in any order. Unknowns absent here are unbounded."""
    condition_scale: tuple[tuple[VarPath, float], ...] = ()
    """`(condition, positive factor)`. Each named condition -- and its whole Jacobian row
    -- is multiplied by its factor before being handed to VMCON. Conditions absent here
    are passed through untouched.

    Multiplying a constraint by a positive constant leaves the feasible set and the
    optimum exactly unchanged (only its Lagrange multiplier rescales), so this is
    conditioning, not a change of problem. It exists because **SAND's residual equalities
    arrive in physical units while PROCESS's own constraints arrive already normalised**:
    `constraints.leq`/`geq` return `value/bound - 1`, an O(1) number by construction,
    whereas `Residualise`'s `g(u) - u` is in whatever units `u` has. On the reference run
    that is a spread from `1e-3` to `1e5` across one equality block, and VMCON -- which
    weights every constraint equally in its merit function and its QP -- takes steps that
    satisfy the small ones and destroy the large ones. Measured: without this, C2 from
    PROCESS's own converged point runs 100 iterations with `max|eq|` stuck at `2e5`.

    Deliberately **explicit and per-condition rather than automatic**: PROCESS's own
    fourteen constraints must keep scale `1.0` or the iterates stop being comparable with
    PROCESS's. `functional_process.sand.residual_condition_scales` supplies factors for
    exactly the residual conditions and nothing else.

    Nothing here bounds the factors, and that is the caller's problem to get right: a
    single row weighted far above the rest wrecks the QP for every *other* row too. Once
    measured, in exactly this driver -- one residual whose unknown was identically zero
    was handed `1e12` by a clamped `1/max(|u|, floor)`, which took the condition number
    of the Jacobian this driver hands VMCON (rows by `condition_scale`, columns by
    `scaled`) from `2.1e4` to `6.7e12`, and Stage C2 from 62 SQP iterations to 73 -- and,
    on the tree state where it was first seen, to `max_iter` without converging.
    `functional_process.sand.residual_condition_scales`' docstring records the rule that
    replaced it and the caveats on that iteration count.

    **Small factors are not the safe direction either.** Down-weighting a row buys
    conditioning by telling VMCON that constraint matters less. Measured on the
    coil-island (`Intersect`) residual, which is the largest row left and the only one
    whose units are genuinely *not* its unknown's: equilibrating it by its own row norm
    takes the condition number `2.1e4` -> `85`, and C2 from converging in 62 iterations
    to `max_iter` without converging. Condition number is a diagnostic here, not an
    objective to minimise.

    This used to end by citing "a fifth to a third of QP subproblems solving
    inaccurately by `cvxpy`'s own warning" as the unexplained residue. **That does not
    reproduce, and the sentence is withdrawn.** Counting `cvxpy`'s own `Problem.status`
    over the whole SAND stellarator matrix -- both starts, both QP solvers, both
    tolerances, 1854 subproblems -- gives `optimal_inaccurate` **zero times**
    (`_audit/optimise_design.md` §15). The original observation was made while this
    driver was silently running OSQP, whose ADMM reports `user_limit` rather than
    `optimal_inaccurate` when it gives up on a subproblem; `qsp_solver` above is the
    finding that replaced it."""
    scaled: bool = True
    """Whether to solve in PROCESS's `x * (1/x_start)` scaled coordinates."""
    qsp_solver: str = "CLARABEL"
    """Which `cvxpy` solver `pyvmcon` hands each QP subproblem to, by name.

    **This is PROCESS's choice, and it has to be stated here because `pyvmcon`'s is
    different.** `pyvmcon.solve_qsp` calls `qsp.solve(**{"solver": cp.OSQP, **options})`
    -- OSQP unless the caller says otherwise -- and PROCESS says otherwise, every run:
    `qsp_options={"solver": cvxpy.CLARABEL}` (`process/core/solver/solver.py:253`). This
    driver used to pass no `qsp_options` at all, so it solved PROCESS's own QP
    subproblems with a *different solver from PROCESS*, which is exactly the confound
    the class docstring says the choice of `pyvmcon` exists to avoid.

    The difference is not cosmetic. OSQP is first-order ADMM with `eps_abs = eps_rel =
    1e-5` by default; CLARABEL is an interior-point method that returns the QP solution
    to ~1e-9. VMCON's search direction *is* the QP solution, and its Hessian update
    `calculate_new_B` is driven by the multipliers the QP returns, so a QP answered to
    1e-5 gives both an inaccurate step and a corrupted quasi-Newton update -- which
    shows up not as a wrong answer but as many more iterations to reach the same one.
    Measured on the MDF stellarator: see `_audit/optimise_design.md` §15."""
    fused: bool = False
    """Take the value and the Jacobian from **one** compiled program rather than two.

    `jax.jacfwd` computes the primal internally and discards it, so the default path
    compiles the block's primal twice -- once as `host_cache.bind`'s `values`, once
    inside its `jacobian`. `pyvmcon.AbstractProblem.__call__` demands both at every
    point it evaluates, so there is no value-only call on this path to lose, and
    `bind`'s `values_and_jacobian` (`jax.jacfwd(..., has_aux=True)`, whose aux is the
    primal the JVP already had) removes the duplicate program outright.

    **Off by default, because it is not bitwise, and this problem is sensitive at the
    last bit.** Measured 2026-09-03 on `stellarator_helias`'s cold SAND block
    (`_audit/optimise_design.md` §31.30), fused against split at the same point:

    | | |
    |---|---|
    | condition values | **bitwise identical**, 0 of 21 cells |
    | Jacobian | **10 of 294 cells differ**, max `4.44e-16` |

    and the cause is measured rather than guessed: the *jaxpr* is the same, because the
    same `has_aux` program with its primal output **dropped again** reproduces the split
    Jacobian bit for bit. What moves the ten cells is the extra **live output** -- XLA
    fuses and schedules the tangent computation differently once the primal must also be
    materialised. `jax.lax.optimization_barrier` on the primal, on both outputs, and a
    hand-written `vmap(jvp)` spelling all give the identical ten cells, so there is no
    spelling of "fused" that is bitwise. [measured]

    **What those ten cells did, end to end** -- the reason this is a field and not the
    default. `--native` cold matrix, `stellarator_helias` and `large_tokamak_nof`:

    - stellarator MDF: converged, 66 it, `objf 1.21775739` -- **unchanged**
    - tokamak MDF: converged, 7 it, `objf 1.6` -- **unchanged**
    - tokamak SAND: converged, 10 it, `objf 1.6` -- **unchanged**
    - stellarator SAND: **converged, 169 it, `objf 1.21775743`, largest equality
      residual `1.04e-07`** becomes **stopped, 134 it, `objf 1.22762089`, largest
      equality residual `1.96e-02`**

    Three rows of four are untouched and one flips out of convergence. That is exactly
    the behaviour §19, §20 and §21.2 record -- an SQP trajectory turning on a 2-ulp
    input -- and it is why `reference_cold_matrix.txt` is not re-baselined for it: the
    fused answer is not *wrong*, it is a different walk of a chaotic trajectory, and
    trading a converged row for a stopped one is not a compile-time saving worth taking
    silently. [measured]

    **What it is worth, so the trade is stated and not implied.** In *emitted MLIR
    lines*, which is a property of the program and so cannot be moved by machine load --
    whole-row wall clock could not be used, the unchanged tokamak MDF arm having read
    98.3 s and 62.8 s across two passes [measured]:

    | | stellarator SAND | tokamak SAND |
    |---|---|---|
    | `values` + `jacobian` | 5 160 + 11 428 | 15 160 + 39 646 |
    | fused | **11 544** | **39 884** |
    | removed | **5 044** | **14 922** |

    i.e. the fused program is the Jacobian program plus **116 lines** of primal outputs
    (238 on the tokamak) where `values` costs 5 160 to emit on its own. At the survey's
    0.41-0.52 ms per line cold that is ~2.1-2.6 s and ~6.1-7.8 s per block, on each of
    the two blocks a configuration solves. [measured]

    Per call it also helps, which is not a compile-time effect at all: 0.667 ms (values)
    + 0.786 ms (jacobian) = 1.45 ms against **0.763 ms** fused, medians over 20 calls --
    the redundant primal was being *computed* every iteration too. [measured] An
    independent A/B put the whole-row saving at 3.44 s of a 46 s stellarator row and
    ~12.9 s of a 119 s tokamak row.

    Turn it on to reproduce any of the above, or on a configuration whose trajectory is
    measured to be insensitive. Do not turn it on globally without re-running the cold
    matrix and reading every row."""
    epsfcn: float | None = None
    """When set, replace `jax.jacfwd` with **PROCESS's own finite difference** at this
    relative perturbation, so that the derivative stops being a difference between the
    port and PROCESS.

    `Evaluators.fcnvmc2` perturbs each *scaled* coordinate by `x_i * (1 +/- epsfcn)` and
    divides by the realised `xfor - xbac` (`evaluators.py:118-141`); because
    `load_iteration_variables` puts every scaled coordinate at exactly `1.0`, that is a
    relative perturbation of `epsfcn` about the start. This field reproduces that
    formula in the driver's own scaled coordinates, condition scaling and all, at a cost
    of `2n` condition evaluations per SQP iteration instead of one `jacfwd`.

    `None` -- the default -- keeps the exact Jacobian, which is the whole point of the
    rewrite. This exists to *measure* what PROCESS's smoothing buys: `epsfcn = 0.01` on
    the reference run is a 1 % perturbation, far outside the regime where a difference
    quotient approximates a derivative, so PROCESS's SQP is being handed the gradient of
    a **smoothed** function. Whether that smoothing helps or hurts the iteration count
    is a measurement, not a matter of opinion, and this is the switch that takes it."""
    initial_b: float | None = None
    """`pyvmcon`'s `initial_B`, as a multiple of the identity. `None` means `I`.

    PROCESS's own knob (`_Solver.set_b`, `solver_handler.py:101`), where it is part of a
    recovery ladder: on `SolverOutputCondition.NO_SOLUTION`, *and only if VMCON never
    iterated*, PROCESS retries the whole solve with `2I`. That guard means PROCESS's
    ladder cannot fire on a failure at iteration 45 or 207, so this field is here to be
    *measured* rather than to copy the ladder.

    What it is for: VMCON starts every solve with `B = I` in scaled coordinates, i.e.
    with no curvature information at all. Far from the optimum that costs little,
    because the gradient dominates and almost any descent direction makes progress. At a
    nearly-stationary start it is the whole problem -- the gradient is small, so the step
    is decided by the constraint linearisation, and the line search then rejects, shrinks
    `alpha`, and hands `calculate_new_B` a tiny `ksi` that makes a poor update. That is
    the mechanism proposed for why the *warm* start is harder than the cold one
    (`_audit/optimise_design.md` §15.4), and this field is how to test it."""
    max_iter: int = 100
    tolerance: float = 1.0e-8
    """`pyvmcon`'s `epsilon`. PROCESS's own is `data.numerics.epsvmc`."""
    callback: object = None
    """`f(iteration, result, x, convergence_parameter) -> None`, in the driver's own
    *unscaled* coordinates, or `None`. `eqx.field(static=True)` is deliberately not used
    -- a plain callable is already a static leaf-free field for `eqx.filter_jit`.

    **A per-iteration effect, and it now runs inside a `pure_callback`.** That is
    legitimate where the effect is the point (`run_cold_matrix._recorder` printing or
    recording a trace while a long solve runs) and is not the channel any *reported*
    number travels on any more -- `reports` is. jax is free to elide a `pure_callback`
    whose outputs are unused; these are used, so it runs, but a caller must not read the
    trace as proof the solve happened."""

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)` -- what `__call__` returns after the unknowns.

        `Steps` is how many times `pyvmcon.solve` called its per-iteration callback,
        which is `len(trace)` for the harnesses that record one and is the number both
        ladder harnesses print. `Converged` is whether `solve` returned rather than
        raising. `Status` is `VMCON_CONVERGED`/`VMCON_STATUS`, i.e. *which* `pyvmcon`
        exception fired -- three genuinely different outcomes that a caller comparing
        this driver against `SlsqpDriver` has to be able to tell apart.

        A property and not a `ClassVar` because that is what `AbstractDriver.reports`
        is; this driver always says all three, and a driver whose report costs something
        makes it conditional on its own fields, which a class attribute could not
        express.
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, then `steps`, `converged` and `status` --
        `AbstractDriver`'s own contract, see its abstract `__call__` docstring.

        The solve happens on the host inside one `jax.pure_callback` (`_sqp_callback`),
        so this is traceable: everything that reads a *concrete* start -- `design_scale`,
        the scaled bounds, the callback's unscaling -- lives inside `host` and sees
        NumPy, while everything derivable from the driver's own fields is built out here
        and refuses out here, as an ordinary Python error rather than one surfacing from
        inside a callback.

        Raises
        ------
        ValueError
            If no `Start` data is supplied (there is no shape to guess a pytree from),
            or if `n_equality + n_inequality + 1` does not equal the number of
            conditions the block declares, or if `condition_scale` names something the
            block does not read.
        """
        start = start_from(data, "VmconDriver", conditions)
        expected = 1 + self.n_equality + self.n_inequality
        if expected != len(conditions.conditions):
            raise ValueError(
                f"VmconDriver was told {self.n_equality} equalities and "
                f"{self.n_inequality} inequalities, i.e. {expected} conditions with the "
                f"objective, but the block declares {len(conditions.conditions)} "
                f"({', '.join(written(conditions.conditions))}) -- `ConditionMap` "
                f"carries no type information, so this split is the caller's to get "
                f"right (see this class's docstring)"
            )

        from pyvmcon import Result, VMCONConvergenceException, solve
        from pyvmcon.problem import AbstractProblem

        _flat, unravel = ravel_pytree(start)
        meq = self.n_equality
        # Asked **out here**, before the callback, even though `scaled_problem` asks it
        # again inside: a name that is not a condition of this block is a statement
        # about the driver's own fields, and this class's contract is that such a
        # refusal is an ordinary Python error rather than one surfacing from inside a
        # `jax.pure_callback`. The duplicate check costs a set difference.
        stray = {var for var, _factor in self.condition_scale} - set(
            conditions.conditions
        )
        if stray:
            raise ValueError(
                f"condition_scale names {written(tuple(stray))}, which this block does "
                f"not read as a condition (it reads {written(conditions.conditions)})"
            )
        epsfcn = self.epsfcn
        callback = self.callback
        # Every field read out here rather than through `self` inside `host`, so that
        # what the callback closes over is a handful of plain values and it is obvious
        # by inspection that nothing live crosses the boundary except `dynamic` and the
        # start. `driver` is the exception, and it is the same exception `SlsqpDriver`
        # already makes: `scaled_problem` takes a driver, and an `eqx.Module` of floats,
        # tuples and a plain callable is leaf-free either way.
        driver = self
        fused = self.fused
        n_inequality = self.n_inequality
        max_iter, tolerance = self.max_iter, self.tolerance
        qsp_solver, initial_b = self.qsp_solver, self.initial_b

        def host(live, flat_start):
            """One VMCON solve, on the host, on concrete NumPy.

            Everything that needs a *concrete* start is here and nowhere else: the
            scaling, the scaled bounds, the callback's unscaling. `live` is the
            recombined `ConditionMap`, so the model inside is a jax function again and
            `jax.jacfwd` applies to it exactly as it did before the wrap -- but the
            compilation of it is **not** this driver's business and is not done here.
            `host_cache.bind` is module-level and memoises on the condition map's own
            structure, so a fresh `eqx.combine` per solve is a cache hit rather than the
            two re-compiles §22.2 measured on every call.

            **The problem itself is `scaled_problem`'s**, which is where the scaling,
            the chain rule and the scaled bounds live for both SQP drivers. This body
            used to carry its own copy of all three -- see that function's docstring for
            why one is better than two identical ones, and §31.30 for the bitwise check
            that the unification moved nothing.
            """
            # **Compiled, and deliberately unlike `cottax.drivers.SLSQPDriver`**, which
            # leaves its inner model eager. An SQP iteration here converges a whole
            # PROCESS block; running it op by op costs far more than the one trace it
            # replaces, and the `pure_callback` boundary is per *solve*, not per
            # iteration, so there is nothing about the wrap that makes a compiled inner
            # model wrong. `_audit/optimise_design.md` §22 has what dropping it would
            # cost.
            #
            # **Bound once per solve** (`host_cache.bind`, through `scaled_problem`):
            # the condition map is partitioned and flattened here rather than on every
            # iteration, so a call flattens 313 pytree leaves instead of 5 462. That
            # flatten was 8.37 ms of a 10.58 ms call and was the single largest
            # per-iteration cost this driver had -- 10.58 -> 0.73 ms, bitwise identical
            # (§31.14). `bind` memoises across solves, so §24.1's "a second solve is a
            # cache hit" is kept, not given back.
            scaled_values, split_jac, both, _unravel, scale, _cond, scaled_box = (
                scaled_problem(driver, live, flat_start, unravel)
            )
            scaled_lower, scaled_upper = scaled_box

            def finite_difference(x_scaled):
                """`Evaluators.fcnvmc2`'s own quotient, in this driver's coordinates."""
                columns = []
                for i in range(len(x_scaled)):
                    forward = np.array(x_scaled, dtype=float)
                    backward = np.array(x_scaled, dtype=float)
                    forward[i] = x_scaled[i] * (1.0 + epsfcn)
                    backward[i] = x_scaled[i] * (1.0 - epsfcn)
                    step = forward[i] - backward[i]
                    columns.append(
                        (scaled_values(forward) - scaled_values(backward)) / step
                    )
                return np.stack(columns, axis=1)

            # Flipped by the first `_Problem.__call__`. `_refuse_inert_objective` is a
            # statement about the problem, not about an iterate, and its docstring says
            # why running it per-iteration would fail working configurations.
            started = [True]

            class _Problem(AbstractProblem):
                def __call__(_self, x_scaled):  # noqa: N805 -- pyvmcon's own signature
                    # **Two programs by default, one under `fused`.** `pyvmcon` asks for
                    # the value *and* every derivative at every point it evaluates,
                    # line-search trials included, so nothing on this path ever wants
                    # the value alone (§31.23 counted 552 of each on one row) and a
                    # fused program would be free. It is not bitwise, which is why it is
                    # a field and not the default -- see `VmconDriver.fused`.
                    #
                    # `epsfcn` could not use it anyway: its quotient is `2n` value-only
                    # evaluations, and asking a fused program for them would compute
                    # `2n` exact Jacobians to throw away.
                    if epsfcn is not None:
                        values = scaled_values(x_scaled)
                        full = finite_difference(x_scaled)
                    elif fused:
                        values, full = both(x_scaled)
                    else:
                        values, full = scaled_values(x_scaled), split_jac(x_scaled)
                    _refuse_non_finite(values, full, conditions)
                    if started[0]:
                        started[0] = False
                        _refuse_inert_objective(full, conditions)
                    return Result(
                        f=values[0],
                        df=full[0],
                        eq=values[1 : 1 + meq],
                        deq=full[1 : 1 + meq],
                        # cottax says `g <= 0`, VMCON wants `i >= 0` -- see the
                        # docstring.
                        ie=-values[1 + meq :],
                        die=-full[1 + meq :],
                    )

                @property
                def num_equality(_self):  # noqa: N805
                    return meq

                @property
                def num_inequality(_self):  # noqa: N805
                    return n_inequality

            # `scaled_lower`/`scaled_upper` come from `scaled_problem` above -- bounds
            # are on the design variables, so they scale with them, and a negative scale
            # (a variable starting below zero) swaps which bound is which.

            # **Always installed, and provably inert.** `pyvmcon.solve` substitutes its
            # own `lambda _i, _result, _x, _con: None` when handed `None`, so counting
            # here calls a callback where the library would have called one anyway and
            # cannot change an iterate. It is how `Steps` is measured: `pyvmcon` returns
            # no iteration count, and this is the same number `len(trace)` gives the two
            # ladder harnesses.
            steps = [0]

            def wrapped(i, result, x_scaled, convergence):
                steps[0] += 1
                if callback is not None:
                    callback(
                        i,
                        result,
                        np.asarray(x_scaled, dtype=float) / scale,
                        convergence,
                    )

            status = VMCON_CONVERGED
            # `sqp` is the optimiser's own cost -- `cvxpy` canonicalisation, CLARABEL, the
            # line search -- because `phase` is exclusive and every graph evaluation
            # underneath scopes itself as `model` (`host_cache.flat_conditions`). The two
            # therefore separate "the model is expensive" from "the optimiser is
            # expensive", which no single wall-clock number can.
            from functional_process.phase_timing import phase  # noqa: PLC0415

            try:
                with phase("sqp"):
                    x_scaled, _lambda_eq, _lambda_ie, _result = solve(
                        _Problem(),
                        flat_start * scale,
                        scaled_lower,
                        scaled_upper,
                        max_iter=max_iter,
                        epsilon=tolerance,
                        qsp_options={"solver": qsp_solver},
                        initial_B=(
                            None
                            if initial_b is None
                            else np.identity(len(flat_start)) * initial_b
                        ),
                        callback=wrapped,
                    )
            except VMCONConvergenceException as e:
                # `solver.py:262-272`'s own pattern: keep the best point, report the
                # failure out of band rather than propagating out of a `Schedule` run.
                # "Out of band" now means a reported port, not a caller's mutable sink.
                x_scaled = e.x
                status = VMCON_STATUS.get(
                    type(e).__name__, VMCON_STATUS["VMCONConvergenceException"]
                )
            return (
                np.asarray(x_scaled, dtype=float) / scale,
                steps[0],
                status == VMCON_CONVERGED,
                status,
            )

        return _sqp_callback(conditions, start, host)
