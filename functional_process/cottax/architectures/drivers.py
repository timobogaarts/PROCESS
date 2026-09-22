"""Generic `AbstractDriver`s for `cottax.pytree.problem.FixedPoint` and
`cottax.pytree.problem.Optimise`, local to this port.
"""

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from cottax.execution.drivers import PicardDriver as CottaxPicardDriver
from cottax.execution.schedule import ConditionMap, Driver
from cottax.pytree.problem import (
    Converged,
    DriverReport,
    Start,
    Steps,
    is_fixed_point,
    is_optimise,
    is_root_find,
)
from cottax.pytree.problem.condition import Inequality, Objective
from cottax.pytree.spec import VarPath
from jax.flatten_util import ravel_pytree

from functional_process.cottax.architectures.host_cache import bind
from functional_process.cottax.paths import written

UNSCALABLE_BELOW = 1e-12
"""Magnitude below which a start value cannot condition its own coordinate."""


def design_scale(flat_start):
    """`1 / x_start` per coordinate -- PROCESS's conditioning -- with a floor."""
    scale = np.ones_like(flat_start)
    np.divide(  # noqa: RUF069
        1.0, flat_start, out=scale, where=np.abs(flat_start) > UNSCALABLE_BELOW
    )
    return scale


def condition_sizes(conditions: ConditionMap, start) -> tuple[int, ...]:
    """How many flat entries each condition contributes, in `conditions` order --
    `1` for a scalar, the size for an array. Traced (`jax.eval_shape`), never run.
    """
    shapes = jax.eval_shape(lambda s: conditions(*s), tuple(start))
    return tuple(int(np.prod(sh.shape, dtype=int)) for sh in shapes)


def entry_names(names, sizes) -> list[str]:
    """One spelling per flat entry: `x` for a scalar, `x[i]` for an array's entries."""
    out = []
    for name, size in zip(names, sizes, strict=True):
        out.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
    return out


def by_role(
    conditions: ConditionMap, name: str, n_equality: int, n_inequality: int
) -> ConditionMap:
    """`conditions` with its conditions in the order the SQP drivers here partition
    them by position: `(objective, *equalities, *inequalities)`, `roles` permuted
    alongside.

    Since cottax `bc1130a` a condition map's conditions arrive **as the statement wrote
    them** and `ConditionMap.roles` says what each is, parallel -- "a driver splits by
    role, never by position". Before that the seam itself promised this order. A SAND
    problem, `Combine`d from residualised fixed points and the optimiser, writes its
    conditions interleaved, so a count-based split there would quietly read an
    inequality as an equality. The counts the driver was told are checked against the
    roles, which is the check the old length test stood in for.
    """
    roles = conditions.roles
    if len(roles) != len(conditions.conditions):
        raise ValueError(
            f"{name}: `ConditionMap.roles` ({len(roles)}) is not parallel to its "
            f"conditions ({len(conditions.conditions)})"
        )
    rank = {Objective: 0, Inequality: 2}       # every other role is an equality
    order = sorted(range(len(roles)), key=lambda i: rank.get(roles[i], 1))
    counted = (
        sum(1 for r in roles if r is Objective),
        sum(1 for r in roles if rank.get(r, 1) == 1),
        sum(1 for r in roles if r is Inequality),
    )
    if counted != (1, n_equality, n_inequality):
        raise ValueError(
            f"{name} was told {n_equality} equalities and {n_inequality} inequalities "
            f"with one objective, but the block's roles count "
            f"{counted[0]} objective(s), {counted[1]} equalities and {counted[2]} "
            f"inequalities over {', '.join(written(conditions.conditions))}"
        )
    if order == list(range(len(roles))):
        return conditions
    return dataclasses.replace(
        conditions,
        conditions=tuple(conditions.conditions[i] for i in order),
        roles=tuple(roles[i] for i in order),
    )


def entry_count(sizes, first: int, count: int) -> int:
    """How many flat entries conditions `first .. first + count` occupy."""
    return int(sum(sizes[first : first + count]))


def scaled_problem(driver, conditions: ConditionMap, flat_start, unravel, sizes=None):
    """The pieces every SQP driver here needs, built once from a block's `ConditionMap`.

    Everything is at the level of **flat entries**: an array-valued unknown is as many
    design coordinates as it has elements (`ravel_pytree`'s order), and an array-valued
    condition as many rows. `condition_scale` names conditions and bounds name
    unknowns, and both are spread over their entries here.
    """
    flat_start = np.asarray(flat_start, dtype=float)
    start = unravel(jnp.asarray(flat_start))
    if sizes is None:
        sizes = condition_sizes(conditions, start)

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
    condition_scale = np.repeat(
        np.array([by_name.get(c, 1.0) for c in conditions.conditions], dtype=float),
        sizes,
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
    lower = np.concatenate([
        np.full(int(np.size(value)), limits.get(v, (-np.inf, np.inf))[0], dtype=float)
        for v, value in zip(conditions.unknowns, start, strict=True)
    ])
    upper = np.concatenate([
        np.full(int(np.size(value)), limits.get(v, (-np.inf, np.inf))[1], dtype=float)
        for v, value in zip(conditions.unknowns, start, strict=True)
    ])
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


def finite_difference_jacobian(evaluate, x_scaled, epsfcn):
    """`Evaluators.fcnvmc2`'s own quotient, in a driver's scaled coordinates: each
    coordinate perturbed to `x * (1 +/- epsfcn)`, `2n` value-only calls of `evaluate`,
    the columns stacked. What `VmconDriver(epsfcn=...)` and `SlsqpDriver(epsfcn=...)`
    hand their SQP instead of `jax.jacfwd`, so that the derivative stops being a
    difference between the port and PROCESS.
    """
    columns = []
    for i in range(len(x_scaled)):
        forward = np.array(x_scaled, dtype=float)
        backward = np.array(x_scaled, dtype=float)
        forward[i] = x_scaled[i] * (1.0 + epsfcn)
        backward[i] = x_scaled[i] * (1.0 - epsfcn)
        step = forward[i] - backward[i]
        columns.append((evaluate(forward) - evaluate(backward)) / step)
    return np.stack(columns, axis=1)


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
    """Conditions whose **value** is not finite, by `spelling`."""
    bad_rows: tuple = ()
    """Conditions whose **derivative row** holds a non-finite cell, by `spelling`."""
    zero_columns: tuple = ()
    """Unknowns whose Jacobian column is identically zero, by `spelling`."""
    n_conditions: int = 0
    """How many conditions the block declares, so a caller can say "3 of 30"."""


def _refuse_non_finite(values, jacobian, conditions: ConditionMap, sizes=None) -> None:
    """Raise if any condition value or derivative is not finite, naming which."""
    names = _names_for(conditions, values, sizes)
    bad_values = [n for n, v in zip(names, values, strict=True) if not np.isfinite(v)]
    bad_rows = [
        n for n, row in zip(names, jacobian, strict=True) if not np.all(np.isfinite(row))
    ]
    if not bad_values and not bad_rows:
        return
    unknowns = _unknown_names_for(conditions, jacobian.shape[1])
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


def _names_for(conditions: ConditionMap, values, sizes=None) -> list[str]:
    """A spelling per flat entry of `values` -- the conditions' own names when every
    condition is a scalar, else spread over each one's entries by `sizes`."""
    names = [c.spelling for c in conditions.conditions]
    if sizes is None:
        if len(values) == len(names):
            return names
        raise ValueError(
            f"{len(values)} condition entries for {len(names)} conditions and no sizes"
        )
    return entry_names(names, sizes)


def _unknown_names_for(conditions: ConditionMap, n_entries: int) -> list[str]:
    """A spelling per design coordinate -- the unknowns' names when every unknown is a
    scalar, else positional (`x[i]`), since this diagnostic has no start to size by."""
    names = [u.spelling for u in conditions.unknowns]
    if n_entries == len(names):
        return names
    return [f"x[{i}]" for i in range(n_entries)]


def non_finite_summary(conditions: ConditionMap, unravel, flat_start, sizes=None) -> str | None:
    """`_refuse_non_finite`'s one-line verdict at `flat_start`, or `None` if it is
    clean.
    """
    values, jacobian, _fused = bind(conditions, unravel)
    flat = jnp.asarray(np.asarray(flat_start, dtype=float))
    try:
        if sizes is None:
            sizes = condition_sizes(conditions, unravel(flat))
        _refuse_non_finite(
            np.asarray(values(flat), dtype=float),
            np.asarray(jacobian(flat), dtype=float),
            conditions,
            sizes,
        )
    except NonFiniteProblemError as refusal:
        return refusal.summary
    return None


def _refuse_inert_objective(jacobian, conditions: ConditionMap, sizes=None) -> None:
    """Raise if the objective's gradient row is identically zero at the **start**."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.size == 0 or np.any(jacobian[0] != 0.0):
        return
    names = _names_for(conditions, jacobian, sizes)
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
        f"  design variables: {[u.spelling for u in conditions.unknowns]}\n"
        f"  other conditions with an all-zero row: {others or 'none'}\n"
        "The usual cause is a MISSING PRODUCER -- the objective reads a path this "
        "configuration's graph does not own, so it is a boundary input frozen at its "
        "seed while PROCESS computes a live value. "
        "`$PY -m functional_process.cottax.boundary --inert --input <IN.DAT>` names the path "
        "without running anything; see `_audit/optimise_design.md` §26."
    )


def _name_singular_equalities(jacobian, conditions: ConditionMap, meq: int, sizes=None) -> None:
    """Warn naming the equality rows behind scipy's *"Singular matrix C"*, if any.
    `meq` counts flat entries."""
    jacobian = np.asarray(jacobian, dtype=float)
    block = jacobian[1 : 1 + meq]
    if block.size == 0:
        return
    names = _names_for(conditions, jacobian, sizes)[1 : 1 + meq]
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
        f"{[u.spelling for u in conditions.unknowns]} cannot move: it is satisfied "
        f"or not by the boundary values alone. `pyvmcon` tolerates such a row and "
        f"scipy does not, so this is a statement about the problem rather than about "
        f"the solver; see `_audit/optimise_design.md` §46.",
        RuntimeWarning,
        stacklevel=2,
    )


class Status(DriverReport):
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
            f"({', '.join(v.spelling for v in conditions.unknowns)}) -- supply one "
            f"in env at its `^guess.*` port, or give this driver a `seed`"
        )
    return start


class SlsqpDriver(Driver):
    """`scipy.optimize.minimize(method="SLSQP")` answering `Optimise`, on exactly the
    problem `VmconDriver` receives.
    """

    accepts = staticmethod(is_optimise)
    requires = (Start,)

    n_equality: int = 0
    n_inequality: int = 0
    bounds: tuple = ()
    scaled: bool = True
    condition_scale: tuple = ()
    max_iter: int = 100
    tolerance: float = 1e-8
    epsfcn: float | None = None
    """When set, `finite_difference_jacobian` at this relative step replaces
    `jax.jacfwd` -- `VmconDriver.epsfcn`, for this SQP.
    """
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

        conditions = by_role(
            conditions, "SlsqpDriver", self.n_equality, self.n_inequality
        )
        start = start_from(data, "SlsqpDriver", conditions)

        _flat, unravel = ravel_pytree(start)
        # In flat entries: an array-valued condition is as many rows as it has elements.
        sizes = condition_sizes(conditions, start)
        meq = entry_count(sizes, 1, self.n_equality)
        driver, user_callback = self, self.callback
        max_iter, tolerance, epsfcn = self.max_iter, self.tolerance, self.epsfcn

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
                scaled_problem(driver, live, flat_start, unravel, sizes)
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
                    slot[1] = (
                        jacobian(x)
                        if epsfcn is None
                        else finite_difference_jacobian(evaluate, x, epsfcn)
                    )
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
                    jacobian_at(np.asarray(result.x, dtype=float)), live, meq, sizes
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
                # `session._status` is that reader, and all it does is compare
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


BOXED_CONVERGED = 0
"""`Status` for a `BoxedSlsqpDriver` run that converged: SLSQP succeeded strictly
inside its move box, or a re-centred call moved by less than `tol`."""

BOXED_STATUS: dict[str, int] = {
    "iteration limit": 1,
    "outer limit": 2,
    "box shrunk": 3,
}
"""`Status` per way a `BoxedSlsqpDriver` run stopped short: the major-iteration
budget spent, the outer-call budget spent, or the move box halved below `delta_min`
without a feasible success."""


class BoxedSlsqpDriver(Driver):
    """`SlsqpDriver`'s problem, solved under **move limits**: SLSQP has no trust
    region, and on a problem whose conditions are sample statistics (a CVaR, a median:
    piecewise smooth, with kinks where samples change order) its first unboxed QP step
    lands far out where the model stops converging. So every SLSQP call is boxed to
    `x (1 -+ delta)` around its start in the driver's scaled coordinates, and the run
    is a loop of such calls:

    - a call whose answer sits on a face of the box (and not on the problem's own
      bound) is **re-centred** there and the next call starts from it;
    - a call that ends **infeasible** (an inequality above `gtol`, an equality beyond
      it) or that SLSQP gave up on (a status other than success, "positive directional
      derivative" or the iteration limit) **halves** the box and restarts from the
      incumbent -- the cheapest feasible point seen, or, until there is one, the least
      infeasible. An infeasible call that is *less* infeasible than any point before,
      with no feasible point yet, is progress and is re-centred with the box kept, so a
      run started outside the feasible region walks in box by box (`paper_tests/
      ouu.smoke` started at the deterministic optimum and had no need of this);
    - the run has **converged** when SLSQP reports success strictly inside its box, a
      re-centred call moves by less than `tol` (relative, scaled), or two re-centred
      calls in a row improve the objective by less than `ftol_outer`.

    Two hooks a body computing its own statistics wants:

    - `jacobian`: `f(flat_x) -> [n_conditions, n_x]` in the block's own condition
      order and the unknowns' flat order, unscaled. When given, values are taken
      **eagerly** through the condition map (so a body memoising per point is hit) and
      the rows from this; when `None`, `jax.jacfwd` through the condition map, as
      `SlsqpDriver`.
    - `warm_start`: `f(flat_x) -> None`, called with each iterate SLSQP accepts and
      with the incumbent on a restart -- the body then starts its next
      evaluation from that point's answer, never from a line-search trial's, which is
      what keeps a far trial from handing bad starts to a nearer one.

    A per-call memo (three points: SLSQP asks the objective and the constraints at one
    point, and its callback at the accepted one) keeps a value and a Jacobian per
    distinct point. Reports `Steps` (major iterations over every call), `Converged`
    and `Status` (`BOXED_CONVERGED`, or a `BOXED_STATUS` code).
    """

    accepts = staticmethod(is_optimise)
    requires = (Start,)

    n_equality: int = 0
    n_inequality: int = 0
    bounds: tuple = ()
    scaled: bool = True
    condition_scale: tuple = ()
    max_iter: int = 100
    """Major iterations, summed over every SLSQP call."""
    max_outer: int = 60
    """How many SLSQP calls the loop allows itself."""
    tolerance: float = 1e-6
    """SLSQP's own `ftol`, per call."""
    delta: float = 0.25
    """The move box: each call may move each coordinate by this fraction of itself."""
    delta_min: float = 1e-3
    """The box below which a restart is given up on."""
    tol: float = 1e-4
    """A re-centred call that moves less than this (scaled, relative) has converged."""
    ftol_outer: float = 1e-5
    """Two re-centred calls improving `f` by less than this (relative) have converged."""
    gtol: float = 1e-4
    """An inequality above this, or an equality beyond it, makes a call infeasible."""
    jacobian: object = None
    """`f(flat_x) -> matrix` in the block's condition order, or `None` for `jacfwd`."""
    warm_start: object = None
    """`f(flat_x) -> None`, called with each accepted iterate, or `None`."""
    callback: object = None
    """`f(entry: dict) -> None` after every SLSQP call, with what it did, or `None`."""

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)`, as the other SQP drivers here."""
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, then `steps`, `converged` and `status`."""
        from scipy.optimize import minimize  # noqa: PLC0415

        as_written = conditions.conditions
        conditions = by_role(
            conditions, "BoxedSlsqpDriver", self.n_equality, self.n_inequality
        )
        order = [as_written.index(c) for c in conditions.conditions]
        start = start_from(data, "BoxedSlsqpDriver", conditions)
        _flat, unravel = ravel_pytree(start)
        sizes = condition_sizes(conditions, start)
        meq = entry_count(sizes, 1, self.n_equality)
        driver = self
        own_jacobian, warm_start, user_callback = (
            self.jacobian,
            self.warm_start,
            self.callback,
        )
        max_iter, max_outer, ftol = self.max_iter, self.max_outer, self.tolerance
        delta_min, tol, ftol_outer, gtol = (
            self.delta_min,
            self.tol,
            self.ftol_outer,
            self.gtol,
        )

        def host(live, flat_start):
            evaluate, jacobian, _both, _unravel, scale, cond_scale, (lower, upper) = (
                scaled_problem(driver, live, flat_start, unravel, sizes)
            )
            if own_jacobian is not None:
                # Eagerly through the condition map: the body is called outside any
                # trace, so a body that memoises per point (the batched program) is
                # hit, and its own rows are what the Jacobian is.
                def evaluate(x_scaled):
                    flat = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
                    raw, _ = ravel_pytree(live(*_unravel(flat)))
                    return np.asarray(raw, dtype=float) * cond_scale

                def jacobian(x_scaled):
                    raw = np.asarray(
                        own_jacobian(np.asarray(x_scaled, dtype=float) / scale),
                        dtype=float,
                    )[order]
                    return raw * cond_scale[:, None] / scale[None, :]

            cache: dict = {}
            KEPT = 3

            def _slot(x):
                key = x.tobytes()
                if key not in cache:
                    while len(cache) >= KEPT:
                        del cache[next(iter(cache))]
                    cache[key] = [None, None]
                return cache[key]

            def values_at(x):
                slot = _slot(np.asarray(x, dtype=float))
                if slot[0] is None:
                    slot[0] = evaluate(np.asarray(x, dtype=float))
                return slot[0]

            def jacobian_at(x):
                slot = _slot(np.asarray(x, dtype=float))
                if slot[1] is None:
                    slot[1] = jacobian(np.asarray(x, dtype=float))
                return slot[1]

            def objective(x):
                return float(values_at(x)[0])

            def objective_gradient(x):
                return jacobian_at(x)[0]

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda x: values_at(x)[1 : 1 + meq],
                    "jac": lambda x: jacobian_at(x)[1 : 1 + meq],
                },
                {
                    "type": "ineq",
                    "fun": lambda x: -values_at(x)[1 + meq :],
                    "jac": lambda x: -jacobian_at(x)[1 + meq :],
                },
            ]
            x0 = flat_start * scale
            constraints = [c for c in constraints if len(np.atleast_1d(c["fun"](x0)))]

            def violation_at(v) -> float:
                """The largest violation: `|eq|` and `ie` above zero, `0` if none."""
                worst = 0.0
                if meq:
                    worst = max(worst, float(np.max(np.abs(v[1 : 1 + meq]))))
                if len(v) > 1 + meq:
                    worst = max(worst, float(np.max(v[1 + meq :])))
                return worst

            def feasible_at(v) -> bool:
                return violation_at(v) <= gtol

            def adopt(x_scaled):
                if warm_start is not None:
                    warm_start(np.asarray(x_scaled, dtype=float) / scale)

            x = np.asarray(x0, dtype=float)
            delta = driver.delta
            best = None
            stalls, total_nit, n_outer = 0, 0, 0
            converged, status, reason = False, BOXED_STATUS["iteration limit"], ""
            last_status = None
            while total_nit < max_iter and n_outer < max_outer:
                half = delta * np.where(np.abs(x) > 0.0, np.abs(x), 1.0)
                lo = np.maximum(lower, x - half)
                hi = np.minimum(upper, x + half)
                result = minimize(
                    objective,
                    x,
                    jac=objective_gradient,
                    bounds=list(zip(lo, hi, strict=True)),
                    constraints=constraints,
                    method="SLSQP",
                    options={"maxiter": max_iter - total_nit, "ftol": ftol},
                    callback=adopt,
                )
                n_outer += 1
                last_status = int(result.status)
                total_nit += max(int(result.nit), 1)
                x_new = np.asarray(result.x, dtype=float)
                # SLSQP leaves its answer ~1e-6 inside a bound, so a face is read to
                # 1e-5 of the box's half-width; a face that is the problem's own
                # bound is not one.
                face = 1e-5 * half
                on_face = bool(
                    np.any((np.abs(x_new - lo) < face) & (lo > lower + 1e-12))
                    | np.any((np.abs(x_new - hi) < face) & (hi < upper - 1e-12))
                )
                v = values_at(x_new)
                feasible = feasible_at(v)
                moved = float(np.max(np.abs(x_new - x)))
                entry = {
                    "outer": n_outer,
                    "nit": int(result.nit),
                    "nfev": int(result.nfev),
                    "njev": int(result.njev),
                    "status": last_status,
                    "message": str(result.message),
                    "on_move_limit": on_face,
                    "move_limit": delta,
                    "f": float(v[0]),
                    "max_eq": float(np.max(np.abs(v[1 : 1 + meq]))) if meq else 0.0,
                    "max_ie": float(np.max(v[1 + meq :])) if len(v) > 1 + meq else 0.0,
                    "feasible": feasible,
                    "moved": moved,
                    "x": (x_new / scale).tolist(),
                    "action": "",
                }
                previous_best = (
                    best["f"] if best is not None and best["feasible"] else None
                )
                # The incumbent: the cheapest feasible point seen, or, until one has
                # been, the least infeasible -- so a run started outside the feasible
                # region can walk in, box by box, instead of shrinking where it stands.
                violation = violation_at(v)
                improved = (
                    best is None
                    or (feasible and (not best["feasible"] or v[0] < best["f"]))
                    or (
                        not feasible
                        and not best["feasible"]
                        and violation < best["violation"]
                    )
                )
                if improved:
                    best = {
                        "feasible": feasible,
                        "f": float(v[0]),
                        "violation": violation,
                        "x": x_new.copy(),
                    }
                if not feasible or last_status not in {0, 8, 9}:
                    # Ended infeasible, or SLSQP gave up (incompatible constraints).
                    # A status-8 line search at a feasible point is not that: the
                    # statistics are piecewise smooth, so the linear model is wrong at
                    # a kink, and the call is re-centred like any other.
                    if improved and not feasible and last_status in {0, 8, 9}:
                        # Less infeasible than anything before, with no feasible point
                        # yet: progress, so re-centre with the box as it is.
                        x = x_new
                        adopt(x)
                        stalls = 0
                        entry["action"] = "re-centred (infeasible, but less so)"
                        if user_callback is not None:
                            user_callback(entry)
                        continue
                    # Otherwise a smaller box from the incumbent.
                    delta *= 0.5
                    if best is not None:
                        x = best["x"].copy()
                        adopt(x)
                    entry["action"] = f"restart from the incumbent, box +-{delta:.3g}"
                    if user_callback is not None:
                        user_callback(entry)
                    if delta < delta_min:
                        status = BOXED_STATUS["box shrunk"]
                        reason = f"move limit shrunk below {delta_min:g}"
                        break
                    continue
                x = x_new
                adopt(x)
                if last_status == 0 and not on_face:
                    converged, reason = True, "SLSQP success strictly inside the box"
                elif moved < tol:
                    converged, reason = True, f"re-centred call moved {moved:.2e} < tol"
                elif previous_best is not None and (
                    v[0] >= previous_best - ftol_outer * abs(previous_best)
                ):
                    stalls += 1
                    if stalls >= 2:
                        converged = True
                        reason = f"two re-centred calls improved f by < {ftol_outer:g}"
                else:
                    stalls = 0
                entry["action"] = reason if converged else "re-centred"
                if user_callback is not None:
                    user_callback(entry)
                if converged:
                    status = BOXED_CONVERGED
                    break
                if last_status == 9:
                    status = BOXED_STATUS["iteration limit"]
                    break
            else:
                status = (
                    BOXED_STATUS["iteration limit"]
                    if total_nit >= max_iter
                    else BOXED_STATUS["outer limit"]
                )
            # The answer: the last point where it is feasible and no worse than the
            # best, else the incumbent, else the last point.
            v_final = values_at(x)
            if feasible_at(v_final) and (
                best is None or not best["feasible"] or v_final[0] <= best["f"] + 1e-12
            ):
                answer = x
            elif best is not None:
                answer = best["x"]
            else:
                answer = x
            adopt(answer)
            return (
                np.asarray(answer, dtype=float) / scale,
                int(total_nit),
                bool(converged),
                int(status),
            )

        return _sqp_callback(conditions, start, host)


class SeededNewtonDriver(Driver):
    """`cottax.execution.drivers.NewtonDriver`, plus a fallback starting guess derived from the
    block's own **context** when the one supplied in `env` is unusable.
    """

    accepts = staticmethod(is_root_find)
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
                f"({', '.join(v.spelling for v in conditions.unknowns)}) -- supply "
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


def condition_scale(conditions: ConditionMap, start):
    """Per-condition scale for a Newton's residual norm and its tolerance test.

    A residualised fixed point's gap `^cond.X = g(^hat.X, ...) - ^hat.X` is in the unit
    of the unknown `^hat.X` (a density here: 1e20), so it is scaled by that unknown's
    start; a condition with no like-named unknown -- a normalised constraint, scale 1 --
    by 1. Without this a combined problem's `max|r|` is the density gap, and a tolerance
    of 1e-10 on it is below float64's resolution at 1e20.
    """
    tails = {
        u.spelling.split(".", 1)[1]: jnp.asarray(x).reshape(-1)
        for u, x in zip(conditions.unknowns, start, strict=True)
    }
    scales = []
    for c in conditions.conditions:
        tail = c.spelling.split(".", 1)[1] if "." in c.spelling else c.spelling
        x = tails.get(tail)
        scales.append(
            jnp.where(x == 0.0, 1.0, jnp.abs(x))  # noqa: RUF069 -- exactly zero: no scale
            if x is not None
            else jnp.ones(1)
        )
    return jnp.concatenate(scales)


class SafeguardedNewtonDriver(SeededNewtonDriver):
    """Newton on a `RootFind`, reporting its verdict (`Steps`, `Converged`, `Status`)
    instead of raising it, with the two safeguards an outer line search needs: a
    relative step cap and backtracking on the residual norm, in coordinates scaled by
    the starting guess.

    An undamped Newton on a residual with **no root** runs away: an outer optimiser's
    trial point can put the power balance where it tends to a non-zero constant as
    `hfact -> inf`, `optimistix.Newton` then reaches `hfact = 5e211`, a derivative
    downstream comes back non-finite and the outer driver refuses the whole solve.
    optimistix's damped least-squares solvers raise from inside `lineax` on such a
    block -- a trial step of theirs sends a nested Picard non-finite and `error_if`
    turns that into an exception rather than a `nan` -- so the loop is written out:
    `jax.lax.custom_root` for the implicit derivative, a `while_loop` of capped,
    backtracked steps inside. Where a root exists it converges like Newton; where none
    exists it stops at a finite point with a non-zero residual (`Converged` false), so
    the outer line search sees a finite, infeasible merit and backs off.

    What one step costs: the residual and its directional derivative come from one
    `jax.jvp` at the trial point, and the accepted trial's derivative is the next step's
    Jacobian, so a step is one evaluation of the cycle, not two. With
    `jacobian="broyden"` (the default) the full Jacobian is built once, at the start,
    by `n` forward tangents, and every later step updates it by Broyden's rank-1 secant
    formula from the primal residuals it evaluates anyway -- one primal evaluation and
    no tangent per step, superlinear, about one step more than Newton at the tail.
    `"newton"` keeps the exact Jacobian per step. The start Jacobian's columns are
    taken by one `jvp` each rather than `vmap` over the basis: XLA fuses the `vmap`ped
    tangents with the primal into one kernel that is slower than separate ones, and
    the stacked tangent intermediate is the largest buffer of the call.

    The residual is scaled per condition (`condition_scale`), and the step cap
    shortens the whole Newton direction by one factor rather than clipping each
    component, which keeps the direction a Newton direction.
    """

    rtol: float = 1e-10
    atol: float = 1e-10
    max_steps: int = 40
    cap: float = 0.5
    """Largest step, relative to the current iterate: the Newton direction scaled by
    one factor so that no component moves by more than this fraction."""
    halvings: int = 12
    """Backtracking budget per Newton step. **Only an active point may spend it**
    (`worse` below is masked by `norm(r) > tol`): under `jax.vmap` a `while_loop` runs
    its body for every point until the last one's predicate is false, with `select` on
    the carry -- so a point the outer loop has already finished still evaluates a
    Newton step, and at a root that step is noise, `norm(r_new) >= norm(r)` half the
    time, and its halving loop would run to the cap, each halving one evaluation of
    the whole cycle for the whole batch. For a single point the mask changes nothing:
    the outer `go_on` implies it."""
    jacobian: str = "broyden"
    """`"broyden"` or `"newton"` -- see the class docstring."""

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)` -- what `__call__` returns after the unknowns.
        `Status` is 0 converged, 1 out of steps, 2 stalled (no step lowered the
        residual within the backtracking budget).
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """The root of `conditions` from `data[Start]`, then the verdict.

        Raises
        ------
        ValueError
            If there is no starting value and no `seed`.
        """
        from jax import lax  # noqa: PLC0415

        start = data.get(Start)
        if self.seed is not None and not _usable(start):
            start = self.seed(conditions)
        if start is None:
            raise ValueError(
                f"SafeguardedNewtonDriver needs a starting value for every unknown "
                f"({', '.join(v.spelling for v in conditions.unknowns)}) -- supply "
                f"one in env at its `^guess.*` port, or give this driver a `seed`"
            )
        flat_guess, unravel = ravel_pytree(start)
        scale = jnp.where(flat_guess == 0.0, 1.0, flat_guess)  # noqa: RUF069
        rscale = condition_scale(conditions, start)
        tol, cap, halvings, max_steps = (
            self.rtol,
            self.cap,
            self.halvings,
            self.max_steps,
        )
        n = flat_guess.size
        broyden = self.jacobian == "broyden"

        def residual(u):
            out, _ = ravel_pytree(conditions(*unravel(u * scale)))
            return out

        def norm(r):
            return jnp.max(jnp.abs(r / rscale))

        def value_and_jacobian(f, u):
            """`f(u)` and its Jacobian from one primal pass: one `jvp` per column."""
            eye = jnp.eye(n, dtype=u.dtype)
            r, first = jax.jvp(f, (u,), (eye[0],))
            columns = [first] + [jax.jvp(f, (u,), (eye[i],))[1] for i in range(1, n)]
            return r, jnp.stack(columns, axis=1)

        def solve(f, u0):
            def step(state):
                u, r, jac, k, _stalled = state
                du = -jnp.linalg.solve(jac, r)
                du *= jnp.minimum(
                    1.0, cap / jnp.max(jnp.abs(du) / jnp.maximum(jnp.abs(u), 1e-3))
                )
                active = norm(r) > tol  # false where the outer loop is done (vmap)

                if broyden:

                    def worse(bs):
                        _t, r_new, j = bs
                        bad = ~jnp.all(jnp.isfinite(r_new)) | (norm(r_new) >= norm(r))
                        return bad & (j < halvings) & active

                    def halve(bs):
                        t, _r_new, j = bs
                        t *= 0.5
                        return t, f(u + t * du), j + 1

                    t, r_new, _j = lax.while_loop(worse, halve, (1.0, f(u + du), 0))
                    s_, y_ = t * du, r_new - r
                    jac_new = jac + jnp.outer(y_ - jac @ s_, s_) / jnp.maximum(
                        s_ @ s_, 1e-300
                    )
                else:

                    def worse(bs):
                        _t, r_new, _jac_new, j = bs
                        bad = ~jnp.all(jnp.isfinite(r_new)) | (norm(r_new) >= norm(r))
                        return bad & (j < halvings) & active

                    def halve(bs):
                        t, _r_new, _jac_new, j = bs
                        t *= 0.5
                        return t, *value_and_jacobian(f, u + t * du), j + 1

                    t, r_new, jac_new, _j = lax.while_loop(
                        worse, halve, (1.0, *value_and_jacobian(f, u + du), 0)
                    )
                accepted = jnp.all(jnp.isfinite(r_new)) & (norm(r_new) < norm(r))
                u_new = jnp.where(accepted, u + t * du, u)
                r_new = jnp.where(accepted, r_new, r)
                jac_new = jnp.where(accepted, jac_new, jac)
                return u_new, r_new, jac_new, k + 1, ~accepted

            def go_on(state):
                _u, r, _jac, k, stalled = state
                return (norm(r) > tol) & (k < max_steps) & ~stalled

            r0, jac0 = value_and_jacobian(f, u0)
            u, r, _jac, k, stalled = lax.while_loop(
                go_on, step, (u0, r0, jac0, 0, False)
            )
            # As floats: `custom_root`'s aux must carry a float tangent.
            return u, jnp.stack([k, norm(r) <= tol, stalled]).astype(float)

        def tangent_solve(g, y):
            return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)

        u, aux = lax.custom_root(
            residual, jnp.ones_like(flat_guess), solve, tangent_solve, has_aux=True
        )
        aux = jax.lax.stop_gradient(aux)
        steps, converged, stalled = aux[0].astype(int), aux[1] > 0.5, aux[2] > 0.5
        status = jnp.where(converged, 0, jnp.where(stalled, 2, 1))
        return (*unravel(u * scale), steps, converged, status)


def is_single_unknown_root_find(node) -> bool:
    """`is_root_find`, and exactly one unknown -- what `BracketedRootDriver` answers.
    Whether that unknown is a scalar is a fact about its value, not the node, and is
    checked at the call.
    """
    return is_root_find(node) and len(node.unknowns) == 1


BRACKET_CONVERGED = 0
BRACKET_OUT_OF_STEPS = 1
BRACKET_COLLAPSED = 2
BRACKET_NOT_FOUND = 3
BRACKET_NON_FINITE = 4
"""`Status` of a `BracketedRootDriver` solve: converged; `max_steps` spent with the
bracket still open; the bracket collapsed to the working precision of the unknown
without the residual reaching `rtol` (a residual that is not continuous there); no
sign change found within `max_expansions` of the first bracket; a residual that came
back non-finite inside the bracket (a nested solve that blew up), at which point the
bracket can no longer be maintained."""


def _count(value):
    """An `int32` counter, so a `lax.cond`'s two branches agree on the dtype."""
    return jnp.asarray(value, dtype=jnp.int32)


class BracketedRootDriver(Driver):
    """A **single-unknown** `RootFind` answered by a bracketed method, so that it
    converges from any start: find two points where the residual changes sign, then
    Newton inside the bracket with bisection as the fallback (`rtsafe`), then the
    implicit derivative through `jax.lax.custom_root`.

    Why not the safeguarded Newton: from a start far from the root a Newton, capped
    and backtracked or not, stalls where the residual flattens, and the closing root
    find of `architectures.closing` is exactly that from an outer optimiser's far trial
    design or a belief sample's far root -- 35-42 % of samples at the handoff's far
    design (`plans/handoff_2026-09-17.md`). A bracket is the one guarantee a 1-D root
    has that an n-D one does not: once the residual is known to change sign between
    two points, bisection cannot lose the root, and a Newton step is only ever taken
    when it lands inside the bracket and shrinks it at least as fast as bisection
    would (Numerical Recipes' `rtsafe` rule), so the tail is quadratic and the head is
    safe.

    **The bracket.** The first pair tried is the unknown's `bounds` when it has them
    (the file's `boundl`/`boundu`, `closing.close` fills them from the session), else
    `(x0 / growth, x0 * growth)`. Where the residual has one sign over that pair the
    pair is widened geometrically -- the lower end divided by `growth`, the upper
    multiplied -- up to `max_expansions` times, and the first widening that crosses
    zero becomes the bracket, tightened to the two adjacent points. Bounds are a first
    guess of where the root is and not a wall: an outer optimiser's trial design can
    put the density's root outside the file's `[3e19, 3e20]`, and the flattened Newton
    finds it there, so this driver must too. Geometric widening presumes the unknown
    is one-signed and its root on the start's side of zero, as every PROCESS iteration
    variable is; a start of exactly `0.0` cannot be widened about and is refused.

    **Warm start.** When `|r(x0)|` is already within tolerance, nothing is bracketed
    and no step is taken: `Steps` is 1 (the one evaluation). Otherwise a bracket is
    always found first -- two or more evaluations of the block's body -- and the
    Newton begins from `x0` when it lies inside the bracket, else from its midpoint.

    **Traceable and batchable.** All `lax`: `cond` on the warm start, a `while_loop`
    for the widening and one for the solve, no Python branching on a value, so it
    runs under `jax.vmap` (rows finished early keep evaluating -- a `while_loop`'s
    body runs until the last row's predicate is false, `select` on the carry) and
    under `jax.jit`. The derivative of the root with respect to everything the block
    reads comes from `lax.custom_root`'s implicit function theorem, one scalar
    division, so `jax.jacfwd` through it never sees the loops. What the residual
    itself nests -- a Picard on the cycle's cut copies (`closing.close(flatten=False)`)
    -- carries its own implicit adjoint (`optimistix`), so the residual's derivative
    per Newton step is one `jvp` through that.

    `Steps` reports the **number of residual evaluations**, bracketing included --
    the cost, in units of the block's body, which is what a comparison with a Newton's
    step count should be made in. `Status` is one of the `BRACKET_*` codes. Tolerance
    is `|r| <= atol + rtol * condition_scale`, the safeguarded Newton's test with an
    absolute floor.
    """

    accepts = staticmethod(is_single_unknown_root_find)
    requires = (Start,)

    rtol: float = 1e-10
    atol: float = 1e-10
    bounds: tuple = ()
    """`((VarPath, lower, upper), ...)` as the SQP drivers spell them; the entry for
    this problem's unknown, if any, is the first bracket tried.
    """
    growth: float = 2.0
    """Factor each widening of the bracket multiplies its upper end by and divides its
    lower end by.
    """
    max_expansions: int = 20
    """Widenings tried before `BRACKET_NOT_FOUND`: a factor `growth ** max_expansions`
    each way of the first pair, 2^20 = a million by default.
    """
    max_steps: int = 60
    """Newton-or-bisection steps after the bracket is found. A bisection halves the
    bracket, so 60 steps from any bracket of finite relative width reach float64's
    resolution even if every step were a bisection.
    """
    xtol: float = 1e-15
    """The relative width of the bracket at which the loop stops as `BRACKET_COLLAPSED`
    if the residual is still above tolerance.
    """

    @property
    def reports(self) -> tuple:
        """`(Steps, Converged, Status)`."""
        return (Steps, Converged, Status)

    def bracket_for(self, unknown: VarPath):
        """`(lower, upper)` of `bounds` for `unknown`, or `None`."""
        for var, lo, hi in self.bounds:
            if var == unknown:
                return float(lo), float(hi)
        return None

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """The root of `conditions` from `data[Start]`, then the verdict.

        Raises
        ------
        ValueError
            If there is no starting value, the unknown is not a scalar, or the
            start is concretely `0.0`.
        """
        from jax import lax  # noqa: PLC0415

        start = data.get(Start)
        if start is None or len(conditions.unknowns) != 1:
            raise ValueError(
                f"BracketedRootDriver answers one scalar unknown from a starting value "
                f"-- this block has {len(conditions.unknowns)} unknown(s) "
                f"({', '.join(v.spelling for v in conditions.unknowns)})"
                + (" and no start" if start is None else "")
            )
        flat_guess, unravel = ravel_pytree(start)
        if flat_guess.size != 1:
            raise ValueError(
                f"BracketedRootDriver answers one scalar unknown, and "
                f"{conditions.unknowns[0].spelling} has {flat_guess.size} entries"
            )
        x0 = flat_guess[0]
        if not isinstance(x0, jax.core.Tracer) and float(x0) == 0.0:  # noqa: RUF069
            raise ValueError(
                f"BracketedRootDriver cannot widen a bracket about a start of exactly "
                f"0.0 for {conditions.unknowns[0].spelling} -- supply a start of the "
                f"root's sign and magnitude at its `^guess.*` port"
            )
        # Coordinates scaled by the start, as `SafeguardedNewtonDriver` has them: the
        # start is `u = 1`, and the widening is geometric about it.
        scale = jnp.where(x0 == 0.0, 1.0, x0)  # noqa: RUF069
        rscale = condition_scale(conditions, start)[0]
        tol = self.atol + self.rtol * rscale
        growth = self.growth
        max_expansions, max_steps, xtol = self.max_expansions, self.max_steps, self.xtol
        pair = self.bracket_for(conditions.unknowns[0])

        def residual(u):
            out, _ = ravel_pytree(conditions(*unravel(jnp.reshape(u * scale, (1,)))))
            return out[0]

        def value_and_slope(f, u):
            return jax.jvp(f, (u,), (jnp.ones_like(u),))

        def first_pair(u0):
            if pair is None:
                return u0 / growth, u0 * growth
            lo, hi = jnp.asarray(pair[0]) / scale, jnp.asarray(pair[1]) / scale
            # A negative start flips the ends' order in `u`.
            return jnp.minimum(lo, hi), jnp.maximum(lo, hi)

        def crosses(ra, rb):
            return jnp.isfinite(ra) & jnp.isfinite(rb) & (jnp.sign(ra) != jnp.sign(rb))

        def find_bracket(f, u0):
            """`(a, ra, b, rb, evaluations, found)` -- a sign change, or the last pair
            tried.
            """
            a, b = first_pair(u0)
            ra, rb = f(a), f(b)

            def widen(state):
                a, ra, b, rb, k, _found = state
                a2, b2 = a / growth, b * growth
                ra2, rb2 = f(a2), f(b2)
                above = crosses(rb, rb2)
                below = crosses(ra2, ra)
                # Crossed above: the bracket is `(b, b2)`; below: `(a2, a)`; neither:
                # the widened pair, and widen again.
                new_a = jnp.where(above, b, a2)
                new_ra = jnp.where(above, rb, ra2)
                new_b = jnp.where(above, b2, jnp.where(below, a, b2))
                new_rb = jnp.where(above, rb2, jnp.where(below, ra, rb2))
                return new_a, new_ra, new_b, new_rb, k + 1, above | below

            def go_on(state):
                _a, _ra, _b, _rb, k, found = state
                return ~found & (k < max_expansions)

            a, ra, b, rb, k, found = lax.while_loop(
                go_on, widen, (a, ra, b, rb, _count(0), crosses(ra, rb))
            )
            return a, ra, b, rb, _count(2) + 2 * k, found

        def solve(f, u_init):
            r0 = f(u_init)
            warm = jnp.abs(r0) <= tol

            def already(_):
                # The start is a root: nothing to bracket, nothing to step.
                return (
                    u_init,
                    r0,
                    _count(1),
                    jnp.asarray(True),
                    _count(BRACKET_CONVERGED),
                )

            def from_cold(_):
                a, ra, b, rb, n_eval, found = find_bracket(f, u_init)
                inside = (a < u_init) & (u_init < b)
                u = jnp.where(inside, u_init, 0.5 * (a + b))
                r, dr = value_and_slope(f, u)
                # The bracket is not oriented (`ra < 0 < rb` is not required): the
                # update below compares a new residual's sign against `ra`'s.

                def step(state):
                    u, r, dr, a, ra, b, rb, dx_old, k, n_eval, ok = state
                    newton = u - r / dr
                    inside = (a < newton) & (newton < b)
                    fast = jnp.abs(2.0 * r) < jnp.abs(dx_old * dr)
                    take = inside & fast & jnp.isfinite(newton)
                    u_new = jnp.where(take, newton, 0.5 * (a + b))
                    dx = jnp.abs(u_new - u)
                    r_new, dr_new = value_and_slope(f, u_new)
                    finite = jnp.isfinite(r_new)
                    same_as_a = jnp.sign(r_new) == jnp.sign(ra)
                    a_new = jnp.where(finite & same_as_a, u_new, a)
                    ra_new = jnp.where(finite & same_as_a, r_new, ra)
                    b_new = jnp.where(finite & ~same_as_a, u_new, b)
                    rb_new = jnp.where(finite & ~same_as_a, r_new, rb)
                    return (
                        jnp.where(finite, u_new, u),
                        jnp.where(finite, r_new, r),
                        jnp.where(finite, dr_new, dr),
                        a_new,
                        ra_new,
                        b_new,
                        rb_new,
                        dx,
                        k + 1,
                        n_eval + 1,
                        ok & finite,
                    )

                def go_on(state):
                    u, r, _dr, a, _ra, b, _rb, _dx, k, _n, ok = state
                    open_ = (b - a) > xtol * jnp.maximum(jnp.abs(u), 1e-300)
                    return (jnp.abs(r) > tol) & (k < max_steps) & open_ & ok

                init = (u, r, dr, a, ra, b, rb, b - a, _count(0), n_eval + 1, found)
                u, r, _dr, a, _ra, b, _rb, _dx, k, n_eval, ok = lax.while_loop(
                    go_on, step, init
                )
                converged = jnp.abs(r) <= tol
                status = jnp.where(
                    converged,
                    BRACKET_CONVERGED,
                    jnp.where(
                        ~found,
                        BRACKET_NOT_FOUND,
                        jnp.where(
                            ~ok,
                            BRACKET_NON_FINITE,
                            jnp.where(
                                k >= max_steps, BRACKET_OUT_OF_STEPS, BRACKET_COLLAPSED
                            ),
                        ),
                    ),
                )
                return u, r, n_eval, converged, _count(status)

            u, r, n_eval, converged, status = lax.cond(warm, already, from_cold, None)
            # As floats: `custom_root`'s aux must carry a float tangent.
            return u, jnp.stack([n_eval, converged, status, r]).astype(float)

        def tangent_solve(g, y):
            return y / g(jnp.ones_like(y))

        u, aux = lax.custom_root(
            residual, jnp.ones_like(x0), solve, tangent_solve, has_aux=True
        )
        aux = jax.lax.stop_gradient(aux)
        steps, converged, status = (
            aux[0].astype(int),
            aux[1] > 0.5,
            aux[2].astype(int),
        )
        return (*unravel(jnp.reshape(u * scale, (1,))), steps, converged, status)


class PicardDriver(CottaxPicardDriver):
    """`cottax.execution.drivers.PicardDriver` at this port's tolerances -- `optx.fixed_point`,
    and therefore an **implicit adjoint**.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_steps: int = 256
    report_steps: bool = False
    """Report `Steps` -- how many iterates the contraction took -- as a graph output.
    Off by default so the graphs every reference was measured on carry no new names."""

    @property
    def reports(self) -> tuple:
        return (Steps,) if self.report_steps else ()

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """`cottax.execution.drivers.PicardDriver.__call__`, behind this port's refusal
        message, and with `throw=False` on the `optx.fixed_point` call -- **not**
        `super().__call__()`, which cottax's own `PicardDriver` (`~/jaxgraph`) leaves
        at optimistix's default `throw=True`.

        `throw=True` raises out of `equinox`'s `EnumerationItem.error_if` when a step
        budget is exhausted, deliberately (`cottax.execution.drivers.optimistix
        .PicardDriver`'s own docstring: "a budget is only honest if running out of it
        is loud"). Under `jax.jacfwd` composed with a batched (`vmap`) call, that
        `error_if` reproducibly fails to *lower* at all -- `jaxlib.mlir...MLIRError:
        "jit(branched_error_if_impl)": operand type mismatch: expected
        'tensor<1xf64>', got 'tensor<Nx1xf64>'` -- for at least one driven sub-problem
        of this graph (`^mda.fwbs.f_ster_div_single`, found 2026-09-22 while pruning
        `configurations.kinds.BELIEFS`: dropping every belief that reaches that node
        left it a function of first-stage constants alone, and *that* is what
        triggered it -- not vmap size, not which belief, the mix of a first-stage
        input into an otherwise per-sample call). `throw=False` reports the same
        verdict as data instead of as a raised exception -- which this port already
        reads (`Steps`/`Converged` when `report_steps=True`; the caller has no way to
        see the raised one anyway inside a batched trace) -- and the crash is gone
        with it, verified by rerunning the pruned belief table (no `tdiv`, no other
        workaround) through the exact `flexibility.py` machinery
        (`hoist=False`, `jacfwd` through the batched program).
        """
        start = start_from(data, "PicardDriver", conditions)
        flat_guess, unravel = ravel_pytree(start)

        def iterate(flat, args):
            nxt, _ = ravel_pytree(conditions(*unravel(flat)))
            return nxt

        solver = optx.FixedPointIteration(rtol=self.rtol, atol=self.atol)
        solution = optx.fixed_point(
            iterate, solver, flat_guess, max_steps=self.max_steps, throw=False
        )
        if not self.report_steps:
            return unravel(solution.value)
        return (*unravel(solution.value), solution.stats["num_steps"])


class SweepDriver(Driver):
    """One application of a `FixedPoint`'s map, `u <- g(u)`, and no test of convergence.

    Not a solver: what one pass of PROCESS's own pipeline does to its coupling
    variables, as a driver so a schedule can run it. `evaluate.cold_state` uses it
    to compute the state a recipe's cut copies start from -- every quantity as the
    models before it in call order just left it, which is where PROCESS starts too.
    """

    accepts = staticmethod(is_fixed_point)
    requires = (Start,)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        return tuple(conditions(*data[Start]))


class VmconDriver(Driver):
    """PROCESS's own SQP (`pyvmcon`) answering `Optimise`, fed `jax.jacfwd` instead of
    finite differences.
    """

    accepts = staticmethod(is_optimise)
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
        # In `(objective, *equalities, *inequalities)` order, by role: everything
        # below partitions by position, and the counts are checked against the roles.
        conditions = by_role(
            conditions, "VmconDriver", self.n_equality, self.n_inequality
        )
        start = start_from(data, "VmconDriver", conditions)

        from pyvmcon import Result, VMCONConvergenceException, solve
        from pyvmcon.problem import AbstractProblem

        _flat, unravel = ravel_pytree(start)
        # In flat entries: an array-valued condition is as many rows as it has elements.
        sizes = condition_sizes(conditions, start)
        meq = entry_count(sizes, 1, self.n_equality)
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
        n_inequality = entry_count(sizes, 1 + self.n_equality, self.n_inequality)
        max_iter, tolerance = self.max_iter, self.tolerance
        qsp_solver, initial_b = self.qsp_solver, self.initial_b

        def host(live, flat_start):
            """One VMCON solve, on the host, on concrete NumPy."""
            # **Compiled, and deliberately unlike `cottax.execution.drivers.SLSQPDriver`**, which
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
                scaled_problem(driver, live, flat_start, unravel, sizes)
            )
            scaled_lower, scaled_upper = scaled_box

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
                        full = finite_difference_jacobian(
                            scaled_values, x_scaled, epsfcn
                        )
                    elif fused:
                        values, full = both(x_scaled)
                    else:
                        values, full = scaled_values(x_scaled), split_jac(x_scaled)
                    _refuse_non_finite(values, full, conditions, sizes)
                    if started[0]:
                        started[0] = False
                        _refuse_inert_objective(full, conditions, sizes)
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
            try:
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
