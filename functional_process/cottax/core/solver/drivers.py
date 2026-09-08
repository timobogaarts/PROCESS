"""Generic `AbstractDriver`s for `cottax.problem.FixedPoint` and
`cottax.problem.Optimise`, local to this port.
"""

import warnings

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

from functional_process.cottax.core.solver.host_cache import bind


UNSCALABLE_BELOW = 1e-12
"""Magnitude below which a start value cannot condition its own coordinate."""


def design_scale(flat_start):
    """`1 / x_start` per coordinate -- PROCESS's conditioning -- with a floor."""
    scale = np.ones_like(flat_start)
    np.divide(  # noqa: RUF069
        1.0, flat_start, out=scale, where=np.abs(flat_start) > UNSCALABLE_BELOW
    )
    return scale


def scaled_problem(driver, conditions: ConditionMap, flat_start, unravel):
    """The pieces every SQP driver here needs, built once from a block's `ConditionMap`.
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
        """`(values, jacobian)` from one program -- `evaluate` and `jacobian` fused."""
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


_SUMMARY_HEADS = 3
"""How many names of each kind `summary` quotes before `...`."""


def _first_few(names) -> str:
    """`(a, b, c, ...)`, or `(none)` -- `summary`'s bracket."""
    if not names:
        return "(none)"
    shown = ", ".join(names[:_SUMMARY_HEADS])
    return f"({shown}{', ...' if len(names) > _SUMMARY_HEADS else ''})"


class NonFiniteProblemError(ValueError):
    """What `_refuse_non_finite` raises -- a `ValueError` with a name to catch it by."""

    summary: str = ""
    """The three lists in one line, for a caller with one cell to put them in."""

    bad_values: tuple = ()
    """Conditions whose **value** is not finite, by `path_str()`."""
    bad_rows: tuple = ()
    """Conditions whose **derivative row** holds a non-finite cell, by `path_str()`."""
    zero_columns: tuple = ()
    """Unknowns whose Jacobian column is identically zero, by `path_str()`."""
    n_conditions: int = 0
    """How many conditions the block declares, so a caller can say "3 of 30"."""


def _refuse_non_finite(values, jacobian, conditions: ConditionMap) -> None:
    """Raise if any condition value or derivative is not finite, naming which."""
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
        "`sqrt`, at exactly `0.0` -- reached because an unknown was started there."
    )
    # The lists as fields, so a caller reporting this need parse nothing. Nothing
    # extracts them from the *message* any more -- see `VMCON_NON_FINITE` for why the
    # compiled path stopped depending on the message surviving at all.
    refusal.summary = summary
    refusal.bad_values = tuple(bad_values)
    refusal.bad_rows = tuple(bad_rows)
    refusal.zero_columns = tuple(zeroed)
    refusal.n_conditions = len(names)
    raise refusal


def non_finite_summary(conditions: ConditionMap, unravel, flat_start) -> str | None:
    """`_refuse_non_finite`'s one-line verdict at `flat_start`, or `None` if it is
    clean.
    """
    values, jacobian, _fused = bind(conditions, unravel)
    flat = jnp.asarray(np.asarray(flat_start, dtype=float))
    try:
        _refuse_non_finite(
            np.asarray(values(flat), dtype=float),
            np.asarray(jacobian(flat), dtype=float),
            conditions,
        )
    except NonFiniteProblemError as refusal:
        return refusal.summary
    return None


def _refuse_inert_objective(jacobian, conditions: ConditionMap) -> None:
    """Raise if the objective's gradient row is identically zero at the **start**."""
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
        "`$PY -m functional_process.cottax.boundary --inert --input <IN.DAT>` names the path "
        "without running anything; see `_audit/optimise_design.md` §26."
    )


def _name_singular_equalities(jacobian, conditions: ConditionMap, meq: int) -> None:
    """Warn naming the equality rows behind scipy's *"Singular matrix C"*, if any."""
    jacobian = np.asarray(jacobian, dtype=float)
    block = jacobian[1 : 1 + meq]
    if block.size == 0:
        return
    names = [c.path_str() for c in conditions.conditions][1 : 1 + meq]
    # **Inert relative to the block, not literally `!= 0.0`.** An exact test was tried
    # and is a false negative on the case this function exists for: `helias_5b` with
    # `ixc = 3` added leaves `c11`'s row at `[-1.6e-16, -0.0, -0.0, -0.0]`, sixteen
    # orders below its siblings and every bit as inert, and the exact test reported
    # "no zero row" while the block was still rank-deficient on that same row
    # (`_audit/optimise_design.md` §48). Scaled against the largest row so the
    # comparison is unit-free, and `1e-10` is far below any row a design variable
    # genuinely moves -- the live rows in these blocks run 1e-1 to 1e+1.
    scale = np.max(np.abs(block)) or 1.0
    inert = np.max(np.abs(block), axis=1) <= 1e-10 * scale
    zero = [n for n, flat in zip(names, inert, strict=True) if flat]
    rank = int(np.linalg.matrix_rank(block))
    if not zero and rank == min(block.shape):
        return  # scipy said singular, this point does not show why -- say nothing
    detail = (
        f"row(s) inert to every design variable "
        f"(max |row| under 1e-10 of the block's largest): {zero}"
        if zero
        else f"no inert row, but rank {rank} of {min(block.shape)}"
    )
    warnings.warn(
        f"SLSQP reported a singular LSQ subproblem, and the equality block "
        f"({block.shape[0]}x{block.shape[1]}) is degenerate at this point -- {detail}. "
        f"An equality whose row is inert is one the design variables "
        f"{[u.path_str() for u in conditions.unknowns]} cannot move: it is satisfied "
        f"or not by the boundary values alone. `pyvmcon` tolerates such a row and "
        f"scipy does not, so this is a statement about the problem rather than about "
        f"the solver; see `_audit/optimise_design.md` §46.",
        RuntimeWarning,
        stacklevel=2,
    )


class Status(DriverOut):
    """The integer code the driver's own solver library stopped with."""

    label = "status"


VMCON_CONVERGED = 0
"""`Status` for a `VmconDriver` solve `pyvmcon.solve` returned from normally."""

VMCON_STATUS: dict[str, int] = {
    "VMCONConvergenceException": 1,
    "QSPSolverException": 2,
    "LineSearchConvergenceException": 3,
}
"""`Status` per `pyvmcon` failure, keyed by **exception class name**."""

VMCON_NON_FINITE = 4
"""`Status` for a solve `_refuse_non_finite` stopped before VMCON could take a step."""


try:  # pragma: no cover -- the fallback is exercised only on a jax that moved this
    from jax._src.core import trace_state_clean as _trace_state_clean
except ImportError:  # pragma: no cover

    def _trace_state_clean() -> bool:
        """`True` when jax's own answer is unavailable -- see `_nothing_is_tracing`."""
        return True


def _nothing_is_tracing(values) -> bool:
    """Is this call outside every jax trace, so a host call needs no callback at all?"""
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values)
    ):
        return False
    return _trace_state_clean()


def _sqp_callback(conditions: ConditionMap, start, host):
    """`jax.pure_callback` around one host-side SQP solve, plus its verdict."""
    flat_guess, unravel = ravel_pytree(start)

    def solved(live, flat):
        """One solve, with the four host answers at the dtypes the wrap declares."""
        x, steps, converged, status = host(live, np.asarray(flat, dtype=np.float64))
        return (
            np.asarray(x, dtype=np.asarray(flat).dtype),
            np.int32(steps),
            np.bool_(converged),
            np.int32(status),
        )

    if _nothing_is_tracing((conditions, flat_guess)):
        # **Eager: call the host directly, and do not partition at all.**
        #
        # Two costs go with the callback, and both are measured (2026-09-03,
        # `_audit/optimise_design.md` §32.3):
        #
        # 1. `wrapped` below is a fresh closure every solve, and `jax.pure_callback`
        #    puts it in the primitive's parameters as a `_FlatCallback` that hashes on
        #    the *identity* of the function it wraps (`jax._src.callback`). So an eager
        #    `pure_callback` misses jax's cache every time and compiles a fresh
        #    seven-line `jit_pure_callback` program per solve -- **24 ms** on
        #    `stellarator_helias` MDF, **40 ms** on `large_tokamak_nof` MDF, and the
        #    *only* compile a steady-state solve still paid.
        # 2. The partition exists solely so `jax.pure_callback` carries arrays and the
        #    `fn`s ride in a closure; `wrapped` recombines them at the other end. With
        #    no boundary to cross, `eqx.combine(*eqx.partition(c, is_array))` is `c`,
        #    and the round trip is **38 ms** on the stellarator and **58-62 ms** on the
        #    tokamak, the latter ~14 % of that steady solve.
        #
        # Outside a trace the callback is buying nothing: there is no program for the
        # host round trip to sit inside. So the host runs here on the condition map it
        # was handed, and the outputs are put back on device at exactly the dtypes the
        # `ShapeDtypeStruct`s declare, which is what keeps this a shortcut rather than a
        # second code path.
        #
        # **The arguments are handed over as they are, NOT converted to NumPy**, and
        # that too is measured rather than stylistic: `jax.pure_callback`'s own eager
        # impl does `device_put(args, cpu)` and passes jax arrays, so converting here
        # would be a *different* boundary from the one this is shortcutting -- and an
        # expensive one, since `bind` then closes the 312 leaves over a `jax.jit` that
        # is called ~550 times a solve and re-transfers each NumPy leaf on every call.
        # Measured on `stellarator_helias` MDF: converting cost **+22 %** on the solve,
        # which is how the conversion was found at all.
        answer, steps, converged, status = (
            jnp.asarray(value) for value in solved(conditions, flat_guess)
        )
    else:
        dynamic, static = eqx.partition(conditions, eqx.is_array)

        def wrapped(dyn, flat):
            return solved(eqx.combine(dyn, static), flat)

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
    """The `Start` tuple out of a driver's `data` mapping, or a clear refusal."""
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
    """`f(iteration, x_unscaled) -> None`, or `None`."""

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)`, the same three `VmconDriver` reports, so the
        two SQPs can be compared on their verdicts and not only on their answers.
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, then `steps`, `converged` and `status`."""
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
            # `_both` discarded on purpose. `scipy`'s SLSQP takes separate `fun`
            # and `jac` callables and its line search calls `fun` alone at trial points,
            # so a fused program would pay a whole Jacobian for each of those.
            #
            # **That argument is only true because the cache below is lazy, and for a
            # while it was not** (`_audit/optimise_design.md` §42, correction 3). `at`
            # used to derive at every distinct point regardless, so the saving this
            # sentence claims was never taken: the capped `stellarator_helias` SAND arm
            # computed `nfev 3518` Jacobians against the `njev 501` scipy asked for. An
            # argument about what a caller *asks* for is worth nothing until the code
            # only computes what is asked.
            evaluate, jacobian, _both, _unravel, scale, _, (lower, upper) = (
                scaled_problem(driver, live, flat_start, unravel)
            )
            x0 = flat_start * scale

            # One evaluation per point, reused by objective and every constraint:
            # SLSQP calls `fun`, `jac` and each constraint separately at the same `x`,
            # and an evaluation here converges a whole block.
            #
            # **Lazily, in both halves separately** -- the value and the Jacobian are
            # cached independently, so a point `scipy` only ever asks a *value* for
            # costs a value. That is the line search, and it is most of what SLSQP
            # does: before this split, `at` derived at every distinct point and the
            # capped `stellarator_helias` SAND arm computed `nfev 3518` Jacobians
            # against the `njev 501` scipy asked for (`_audit/optimise_design.md` §42).
            #
            # The two-slot cache is not an optimisation either. One slot is enough for
            # the value, because scipy asks `fun` and every constraint's `fun` at one
            # point before moving on -- but the *callback* runs after the line search
            # has already walked past the accepted iterate, so a single slot would
            # evict the point the callback is about to ask about and re-derive it every
            # iteration. Two slots make that a hit, which is why this is a small
            # `dict` walked in insertion order rather than one entry replaced.
            cache: dict = {}
            KEPT = 2

            def _slot(x):
                key = x.tobytes()
                if key not in cache:
                    while len(cache) >= KEPT:
                        del cache[next(iter(cache))]
                    cache[key] = [None, None]
                return cache[key]

            def values_at(x):
                slot = _slot(x)
                if slot[0] is None:
                    slot[0] = evaluate(x)
                return slot[0]

            def jacobian_at(x):
                slot = _slot(x)
                if slot[1] is None:
                    slot[1] = jacobian(x)
                return slot[1]

            iteration = [0]

            def objective(x):
                return float(values_at(np.asarray(x))[0])

            def objective_gradient(x):
                return jacobian_at(np.asarray(x))[0]

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda x: values_at(np.asarray(x))[1 : 1 + meq],
                    "jac": lambda x: jacobian_at(np.asarray(x))[1 : 1 + meq],
                },
                {
                    # cottax `g <= 0` -> SLSQP `c(x) >= 0`.
                    "type": "ineq",
                    "fun": lambda x: -values_at(np.asarray(x))[1 + meq :],
                    "jac": lambda x: -jacobian_at(np.asarray(x))[1 + meq :],
                },
            ]
            constraints = [c for c in constraints if len(np.atleast_1d(c["fun"](x0)))]

            class _Iterate:
                """What one accepted iterate looks like to a `callback`."""

                def __init__(self, x):
                    values = values_at(x)
                    self._x = x
                    self.f = values[0]
                    self.eq = values[1 : 1 + meq]
                    self.ie = -values[1 + meq :]

                @property
                def df(self):
                    return jacobian_at(self._x)[0]

                @property
                def deq(self):
                    return jacobian_at(self._x)[1 : 1 + meq]

                @property
                def die(self):
                    return -jacobian_at(self._x)[1 + meq :]

            def on_step(xk):
                iteration[0] += 1
                if user_callback is not None:
                    x = np.asarray(xk)
                    # `inf`, because `scipy` publishes no per-iterate convergence
                    # measure and has not declared convergence at any of these points.
                    # See `_verdict` for what the *final* call carries and why the two
                    # differ.
                    user_callback(iteration[0], _Iterate(x), x / scale, float("inf"))

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
            if int(result.status) == 6:  # "Singular matrix C in LSQ subproblem"
                _name_singular_equalities(
                    jacobian_at(np.asarray(result.x, dtype=float)), live, meq
                )
            # No `self.last_result = ...`: an `eqx.Module` is frozen, and a driver that
            # mutated itself would not survive being reused across blocks anyway. What
            # the solver said is `reports`' job now, and the mutable `Outcome` sink that
            # used to carry it is deleted.
            if user_callback is not None:
                final = np.asarray(result.x, dtype=float)
                # **The fourth argument is `VmconDriver`'s convergence parameter, and
                # SLSQP forms nothing equivalent -- so what goes here is scipy's own
                # verdict, encoded so that the one thing which reads it can read it.**
                # `run_cold_matrix._status` is that reader, and all it does is compare
                # this number to a tolerance; `0.0` therefore means *"scipy said
                # `success`"* and `inf` *"it did not"*. Writing `nan` instead was tried
                # and is wrong in the direction that matters: `nan <= tol` is `False`,
                # so every SLSQP row read `stopped` including the ones where scipy had
                # said "Optimization terminated successfully" and the residuals were
                # five orders better than VMCON's (`_audit/optimise_design.md` §42).
                # A column that turns a success into a failure is not the cautious
                # choice, it is the wrong answer.
                converged = 0.0 if result.success else float("inf")
                user_callback(-1, _Iterate(final), final / scale, converged)
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
    """

    drives = RootFind
    requires = (Start,)

    rtol: float = 1e-4
    atol: float = 1e-4
    seed: object = None
    """`f(ConditionMap) -> tuple` giving one starting value per unknown, or `None`."""

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
    """Whether `start` is a starting guess at all."""
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
    """`cottax.drivers.PicardDriver` at this port's tolerances -- `optx.fixed_point`,
    and therefore an **implicit adjoint**.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_steps: int = 256

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """`cottax.drivers.PicardDriver.__call__`, behind this port's refusal message.
        """
        start_from(data, "PicardDriver", conditions)
        return super().__call__(conditions, data)


class VmconDriver(AbstractDriver):
    """PROCESS's own SQP (`pyvmcon`) answering `Optimise`, fed `jax.jacfwd` instead of
    finite differences.
    """

    drives = Optimise
    requires = (Start,)

    n_equality: int
    n_inequality: int
    bounds: tuple[tuple[VarPath, float, float], ...] = ()
    """`(unknown, lower, upper)`, in any order. Unknowns absent here are unbounded."""
    condition_scale: tuple[tuple[VarPath, float], ...] = ()
    """`(condition, positive factor)`."""
    scaled: bool = True
    """Whether to solve in PROCESS's `x * (1/x_start)` scaled coordinates."""
    qsp_solver: str = "CLARABEL"
    """Which `cvxpy` solver `pyvmcon` hands each QP subproblem to, by name."""
    fused: bool = True
    """Take the value and the Jacobian from **one** compiled program rather than two."""
    epsfcn: float | None = None
    """When set, replace `jax.jacfwd` with **PROCESS's own finite difference** at this
    relative perturbation, so that the derivative stops being a difference between the
    port and PROCESS.
    """
    initial_b: float | None = None
    """`pyvmcon`'s `initial_B`, as a multiple of the identity."""
    max_iter: int = 100
    tolerance: float = 1.0e-8
    """`pyvmcon`'s `epsilon`. PROCESS's own is `data.numerics.epsvmc`."""
    callback: object = None
    """`f(iteration, result, x, convergence_parameter) -> None`, in the driver's own
    *unscaled* coordinates, or `None`.
    """

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)` -- what `__call__` returns after the unknowns.
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, then `steps`, `converged` and `status` --
        `AbstractDriver`'s own contract, see its abstract `__call__` docstring.
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
            """One VMCON solve, on the host, on concrete NumPy."""
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
                    # **One program by default, two under `fused=False`.** `pyvmcon`
                    # asks for the value *and* every derivative at every point it
                    # evaluates, line-search trials included, so nothing on this path
                    # ever wants the value alone (§31.23 counted 552 of each on one row)
                    # and the fused program is strictly cheaper here. It is still not
                    # bitwise, which is why the split path survives as a field rather
                    # than being deleted -- see `VmconDriver.fused` and §40.
                    #
                    # **`SlsqpDriver` must not do this**, and does not: scipy's
                    # SLSQP calls `fun` alone during its line search, and that driver's
                    # cache is lazy, so a fused program would pay a whole Jacobian per
                    # trial point -- 1.73 ms against 13.7 ms on `large_tokamak_nof` MDF
                    # (§41). `fused` is this class's field and not `scaled_problem`'s
                    # for exactly that reason: the answer differs per driver.
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
            from functional_process.cottax.phase_timing import phase  # noqa: PLC0415

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
            except NonFiniteProblemError:
                # **Caught here, inside the callback, on purpose** -- see
                # `VMCON_NON_FINITE`. `_refuse_non_finite` raises because raising is the
                # right failure for a *direct* call, but `jax.pure_callback` promises a
                # pure function of its inputs and an exception is a side effect: jax may
                # elide the callback under DCE, run it twice, or reorder it, and what a
                # compiled callback does with a Python exception is implementation
                # detail (it arrives as `JaxRuntimeError` carrying the traceback as
                # *text*). So the refusal stops being an exception at this boundary and
                # becomes a status code, which is what `Status` exists for.
                #
                # The start is returned untouched, exactly as a
                # `VMCONConvergenceException` returns `e.x`: there is no better point,
                # because the solve never took a step.
                x_scaled = flat_start * scale
                status = VMCON_NON_FINITE
            return (
                np.asarray(x_scaled, dtype=float) / scale,
                steps[0],
                status == VMCON_CONVERGED,
                status,
            )

        return _sqp_callback(conditions, start, host)
