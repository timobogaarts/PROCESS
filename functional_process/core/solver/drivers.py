"""Generic `AbstractDriver`s for `cottax.problem.FixedPoint` and
`cottax.problem.Optimise`, local to this port.

`cottax.evaluate.AbstractDriver`'s own docstring names the intended pairing directly:
*"a Newton drives `RootFind`, a Picard `FixedPoint`, an optimiser `Optimise`."*
`cottax.drivers.NewtonDriver` (`~/jaxgraph/src/cottax/drivers/optimistix.py`) implements
the first; nothing in `cottax` implements the other two. Written here, in
`functional_process`, not in `~/jaxgraph` -- unlike the earlier `Feasibility`/`to_graph`
gaps (genuine core-library holes, fixed upstream), a driver is exactly the kind of
generic, swappable *solver choice* `AbstractDriver` exists to make pluggable, not core
graph machinery; this port is free to bring its own without asking `cottax` to grow one.
The argument is stronger for `VmconDriver` than for `PicardDriver`: an SQP is a much
larger algorithm choice, and the backing solver (`pyvmcon`) is a **PROCESS** dependency,
not a `cottax` one.

Every `FixedPointFunction`/`FixedPointCut` block the factory assembles
(`DeltaEtaStep`, `CryoQNucStep`, the two cut cross-node cycles once cut, ...) is a
candidate for `PicardDriver` -- it answers any `FixedPoint`, generically, the same way
`NewtonDriver` answers any `RootFind`. `VmconDriver` answers the single `Optimise`
`functional_process.sand` assembles.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from cottax.evaluate import AbstractDriver, ConditionMap
from cottax.problem import FixedPoint, Optimise, RootFind, Start
from cottax.spec import VarPath
from cottax.tools.path import written
from jax.flatten_util import ravel_pytree


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


def scaled_problem(driver, conditions: ConditionMap, start: tuple):
    """The pieces every SQP driver here needs, built once from a block's `ConditionMap`.

    Returns `(evaluate, jacobian, unravel, scale, condition_scale, bounds)`, where
    `evaluate`/`jacobian` take a **scaled** flat design vector and return
    already-scaled values and Jacobian, `unravel` puts a flat vector back into the
    block's unknown pytree, and `bounds` is `(lower, upper)` in scaled coordinates.

    Extracted so `VmconDriver` and `SlsqpDriver` differ **only** in which solver they
    hand the same problem to. That is the whole point of having two: if one takes a
    step from a point where the other cannot, the difference is the solver's QP
    handling, and if neither can, the difference is the problem. A shared builder is
    what makes that a controlled comparison rather than two implementations that might
    be scaling differently.

    The scaling rules are `VmconDriver`'s, unchanged and deliberately not re-derived
    here -- design variables by `1 / x_start` (PROCESS's own conditioning, from
    `load_iteration_variables`), conditions by `driver.condition_scale`, and bounds by
    the same design factor with the swap a negative scale forces.
    """
    flat_start, unravel = ravel_pytree(start)
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

    def flat_conditions(flat_x):
        return jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat_x))])

    _evaluate = jax.jit(flat_conditions)
    _jacobian = jax.jit(jax.jacfwd(flat_conditions))

    def evaluate(x_scaled):
        flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
        return np.asarray(_evaluate(flat_x), dtype=float) * condition_scale

    def jacobian(x_scaled):
        flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
        # d/dx_scaled = (d/dx) / scale -- one chain-rule factor per column, one
        # `condition_scale` factor per row.
        return (
            np.asarray(_jacobian(flat_x), dtype=float)
            * condition_scale[:, None]
            / scale[None, :]
        )

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
        unravel,
        scale,
        condition_scale,
        (scaled_lower, scaled_upper),
    )


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
    raise ValueError(
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
    treatment as `VmconDriver.callback`."""

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, positionally.

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

        evaluate, jacobian, unravel, scale, _, (lower, upper) = scaled_problem(
            self, conditions, start
        )
        meq = self.n_equality
        x0 = np.asarray(ravel_pytree(start)[0], dtype=float) * scale

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
            if self.callback is not None:
                self.callback(iteration[0], np.asarray(xk) / scale)

        result = minimize(
            objective,
            x0,
            jac=objective_gradient,
            bounds=list(zip(lower, upper, strict=True)),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tolerance},
            callback=on_step,
        )
        # No `self.last_result = ...`: an `eqx.Module` is frozen, and a driver that
        # mutated itself would not survive being reused across blocks anyway. What the
        # solver said is reported through `callback`, like `VmconDriver`'s.
        if self.callback is not None:
            self.callback(-1, np.asarray(result.x, dtype=float) / scale)
        return unravel(jnp.asarray(np.asarray(result.x, dtype=float) / scale))


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


class PicardDriver(AbstractDriver):
    """Fixed-point (Gauss-Seidel/Picard) iteration answering `FixedPoint`.

    Repeatedly evaluates the block's conditions and feeds the result back in as the
    next guess, until the iterate stops changing (by `rtol`/`atol`, `jnp.allclose`'s own
    convention) or `max_iter` is reached -- exactly the shape PROCESS's own
    `Caller.call_models` already uses (re-running the whole pipeline up to 10 times and
    checking `check_agreement`'s idempotence), not a new algorithm invented for this
    port. Bounded and `jax.lax.while_loop`-based, so it stays traceable/jittable like
    `NewtonDriver`.

    Needs a `start` for every unknown, same requirement and same reasoning as
    `NewtonDriver`: there is no shape to guess a pytree from. For the block-by-block
    verification harness this driver exists for, the natural `start` is the reference
    PROCESS run's own converged value at that `VarPath` -- not `ITERATION_VARIABLES`'
    generic defaults, which matter for a real from-scratch optimisation with no known
    answer yet, not for checking whether this graph reproduces an answer PROCESS
    already found.
    """

    drives = FixedPoint
    requires = (Start,)

    rtol: float = 1e-6
    atol: float = 1e-8
    max_iter: int = 20

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, positionally -- `AbstractDriver`'s own
        contract, see its abstract `__call__` docstring.

        Raises
        ------
        ValueError
            If no `Start` data is supplied -- there is no shape to guess a pytree from.
        """
        start = start_from(data, "PicardDriver", conditions)
        flat_start, unravel = ravel_pytree(start)

        def cond_fn(carry):
            it, _current, converged = carry
            return jnp.logical_and(it < self.max_iter, jnp.logical_not(converged))

        def body_fn(carry):
            it, current, _converged = carry
            next_flat, _ = ravel_pytree(conditions(*unravel(current)))
            tol = self.atol + self.rtol * jnp.abs(next_flat)
            converged = jnp.all(jnp.abs(next_flat - current) <= tol)
            return (it + 1, next_flat, converged)

        _it, final_flat, _converged = jax.lax.while_loop(
            cond_fn, body_fn, (0, flat_start, jnp.asarray(False))
        )
        return unravel(final_flat)


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
    `pyvmcon` is NumPy, so **a `Drive` answered by this driver is not one traced
    program** the way `PicardDriver`'s `lax.while_loop` and `NewtonDriver`'s
    `optimistix` are. The trace boundary is *per SQP iteration*: each iteration is one
    jitted condition evaluation plus one jitted `jacfwd`, with the QP subproblem solved
    on the host in between. That is acceptable at the outermost block (nothing
    differentiates through it) and would be **wrong nested inside anything** -- a
    `VmconDriver` inside another driven block cannot be differentiated through.

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
    produces an env. Whether it converged is reported out of band through `callback`
    (same shape as PROCESS's own `_solver_callback`), which is the only way a caller
    learns it, and is exactly what `functional_process.sand.solve` records.
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
    max_iter: int = 100
    tolerance: float = 1.0e-8
    """`pyvmcon`'s `epsilon`. PROCESS's own is `data.numerics.epsvmc`."""
    callback: object = None
    """`f(iteration, result, x, convergence_parameter) -> None`, in the driver's own
    *unscaled* coordinates, or `None`. `eqx.field(static=True)` is deliberately not used
    -- a plain callable is already a static leaf-free field for `eqx.filter_jit`, and
    nothing jits this driver."""

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """Values for the block's unknowns, positionally -- `AbstractDriver`'s own
        contract, see its abstract `__call__` docstring.

        Raises
        ------
        ValueError
            If no `Start` data is supplied (there is no shape to guess a pytree from),
            or if `n_equality + n_inequality + 1` does not equal the number of
            conditions the block declares.
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

        flat_start, unravel = ravel_pytree(start)
        flat_start = np.asarray(flat_start, dtype=float)
        # PROCESS's own conditioning, from the starting point, exactly as
        # `load_iteration_variables` derives it -- and through `design_scale`, so this
        # path and `scaled_problem`'s (SLSQP) share one rule rather than two spellings
        # of it. The floor matters: this used to test `flat_start != 0.0`, which a
        # coordinate that is *numerically* zero passes, and `design_scale`'s docstring
        # carries the case that found it.
        scale = design_scale(flat_start) if self.scaled else np.ones_like(flat_start)

        def flat_conditions(flat_x):
            values = conditions(*unravel(flat_x))
            return jnp.stack([jnp.asarray(v) for v in values])

        evaluate = jax.jit(flat_conditions)
        jacobian = jax.jit(jax.jacfwd(flat_conditions))

        meq = self.n_equality
        by_name = {var: float(factor) for var, factor in self.condition_scale}
        unknown_names = set(conditions.unknowns)
        stray = set(by_name) - set(conditions.conditions)
        if stray:
            raise ValueError(
                f"condition_scale names {written(tuple(stray))}, which this block does "
                f"not read as a condition (it reads {written(conditions.conditions)})"
            )
        del unknown_names
        condition_scale = np.array(
            [by_name.get(c, 1.0) for c in conditions.conditions], dtype=float
        )

        epsfcn = self.epsfcn

        def scaled_values(x_scaled):
            flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
            return np.asarray(evaluate(flat_x), dtype=float) * condition_scale

        def finite_difference(x_scaled):
            """`Evaluators.fcnvmc2`'s own quotient, in this driver's coordinates."""
            columns = []
            for i in range(len(x_scaled)):
                forward = np.array(x_scaled, dtype=float)
                backward = np.array(x_scaled, dtype=float)
                forward[i] = x_scaled[i] * (1.0 + epsfcn)
                backward[i] = x_scaled[i] * (1.0 - epsfcn)
                step = forward[i] - backward[i]
                columns.append((scaled_values(forward) - scaled_values(backward)) / step)
            return np.stack(columns, axis=1)

        class _Problem(AbstractProblem):
            def __call__(_self, x_scaled):  # noqa: N805 -- pyvmcon's own signature
                flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
                values = np.asarray(evaluate(flat_x), dtype=float) * condition_scale
                # d/dx_scaled = (d/dx) / scale -- one chain-rule factor per column; and
                # one `condition_scale` factor per row, see that field.
                full = (
                    finite_difference(x_scaled)
                    if epsfcn is not None
                    else np.asarray(jacobian(flat_x), dtype=float)
                    * condition_scale[:, None]
                    / scale[None, :]
                )
                _refuse_non_finite(values, full, conditions)
                return Result(
                    f=values[0],
                    df=full[0],
                    eq=values[1 : 1 + meq],
                    deq=full[1 : 1 + meq],
                    # cottax says `g <= 0`, VMCON wants `i >= 0` -- see the docstring.
                    ie=-values[1 + meq :],
                    die=-full[1 + meq :],
                )

            @property
            def num_equality(_self):  # noqa: N805
                return meq

            @property
            def num_inequality(_self):  # noqa: N805
                return self.n_inequality

        bounds = {var: (lo, hi) for var, lo, hi in self.bounds}
        lower = np.array(
            [bounds.get(v, (-np.inf, np.inf))[0] for v in conditions.unknowns],
            dtype=float,
        )
        upper = np.array(
            [bounds.get(v, (-np.inf, np.inf))[1] for v in conditions.unknowns],
            dtype=float,
        )
        # Bounds are on the design variables, so they scale with them; a negative scale
        # (a variable starting below zero) swaps which bound is which.
        scaled_lower = np.where(scale > 0, lower * scale, upper * scale)
        scaled_upper = np.where(scale > 0, upper * scale, lower * scale)

        callback = self.callback
        if callback is None:
            wrapped = None
        else:

            def wrapped(i, result, x_scaled, convergence):
                callback(
                    i, result, np.asarray(x_scaled, dtype=float) / scale, convergence
                )

        try:
            x_scaled, _lambda_eq, _lambda_ie, _result = solve(
                _Problem(),
                flat_start * scale,
                scaled_lower,
                scaled_upper,
                max_iter=self.max_iter,
                epsilon=self.tolerance,
                qsp_options={"solver": self.qsp_solver},
                callback=wrapped,
            )
        except VMCONConvergenceException as e:
            # `solver.py:262-272`'s own pattern: keep the best point, report the failure
            # out of band rather than propagating out of a `Schedule` run.
            x_scaled = e.x

        return unravel(jnp.asarray(np.asarray(x_scaled, dtype=float) / scale))
