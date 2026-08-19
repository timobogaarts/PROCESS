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

Every `FixedPointFunction`/`FixedPointCut` block registered in `total_process.py`
(`EtathLiqStep`, `DeltaEtaStep`, the two cut cross-node cycles once cut, ...) is a
candidate for `PicardDriver` -- it answers any `FixedPoint`, generically, the same way
`NewtonDriver` answers any `RootFind`. `VmconDriver` answers the single `Optimise`
`functional_process.sand` assembles.
"""

import jax
import jax.numpy as jnp
import numpy as np
from cottax.evaluate import AbstractDriver, ConditionMap
from cottax.problem import FixedPoint, Optimise
from cottax.spec import VarPath
from cottax.tools.path import written
from jax.flatten_util import ravel_pytree


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

    rtol: float = 1e-6
    atol: float = 1e-8
    max_iter: int = 20

    def __call__(self, conditions: ConditionMap, start: tuple | None) -> tuple:
        """Values for the block's unknowns, positionally -- `AbstractDriver`'s own
        contract, see its abstract `__call__` docstring.

        Raises
        ------
        ValueError
            If `start` is `None` -- there is no shape to guess a pytree from.
        """
        if start is None:
            raise ValueError(
                f"PicardDriver needs a starting value for every unknown "
                f"({', '.join(v.path_str() for v in conditions.unknowns)}) -- supply "
                f"one in env, same as any other unowned input"
            )
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
    exactly the residual conditions and nothing else."""
    scaled: bool = True
    """Whether to solve in PROCESS's `x * (1/x_start)` scaled coordinates."""
    max_iter: int = 100
    tolerance: float = 1.0e-8
    """`pyvmcon`'s `epsilon`. PROCESS's own is `data.numerics.epsvmc`."""
    callback: object = None
    """`f(iteration, result, x, convergence_parameter) -> None`, in the driver's own
    *unscaled* coordinates, or `None`. `eqx.field(static=True)` is deliberately not used
    -- a plain callable is already a static leaf-free field for `eqx.filter_jit`, and
    nothing jits this driver."""

    def __call__(self, conditions: ConditionMap, start: tuple | None) -> tuple:
        """Values for the block's unknowns, positionally -- `AbstractDriver`'s own
        contract, see its abstract `__call__` docstring.

        Raises
        ------
        ValueError
            If `start` is `None` (there is no shape to guess a pytree from), or if
            `n_equality + n_inequality + 1` does not equal the number of conditions the
            block declares.
        """
        if start is None:
            raise ValueError(
                f"VmconDriver needs a starting value for every unknown "
                f"({', '.join(v.path_str() for v in conditions.unknowns)}) -- supply "
                f"one in env, same as any other unowned input"
            )
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
        # `load_iteration_variables` derives it.
        # `np.divide(..., where=)` rather than `np.where(cond, 1/x, 1)`: the latter
        # evaluates `1/x` for every entry first, so a zero start warns (and, at -0.0,
        # produces -inf) before the select discards it.
        # Exact comparison is deliberate -- it is exactly zero, not a neighbourhood of
        # zero, that has no scale.
        scale = np.ones_like(flat_start)
        if self.scaled:
            np.divide(1.0, flat_start, out=scale, where=flat_start != 0.0)  # noqa: RUF069

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

        class _Problem(AbstractProblem):
            def __call__(_self, x_scaled):  # noqa: N805 -- pyvmcon's own signature
                flat_x = jnp.asarray(np.asarray(x_scaled, dtype=float) / scale)
                values = np.asarray(evaluate(flat_x), dtype=float) * condition_scale
                # d/dx_scaled = (d/dx) / scale -- one chain-rule factor per column; and
                # one `condition_scale` factor per row, see that field.
                full = (
                    np.asarray(jacobian(flat_x), dtype=float)
                    * condition_scale[:, None]
                    / scale[None, :]
                )
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
                callback=wrapped,
            )
        except VMCONConvergenceException as e:
            # `solver.py:262-272`'s own pattern: keep the best point, report the failure
            # out of band rather than propagating out of a `Schedule` run.
            x_scaled = e.x

        return unravel(jnp.asarray(np.asarray(x_scaled, dtype=float) / scale))
