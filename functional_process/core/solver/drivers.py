"""A generic `AbstractDriver` for `cottax.problem.FixedPoint`, local to this port.

`cottax.evaluate.AbstractDriver`'s own docstring names the intended pairing directly:
*"a Newton drives `RootFind`, a Picard `FixedPoint`, an optimiser `Optimise`."*
`cottax.drivers.NewtonDriver` (`~/jaxgraph/src/cottax/drivers/optimistix.py`) implements
the first; nothing in `cottax` implements the second yet. Written here, in
`functional_process`, not in `~/jaxgraph` -- unlike the earlier `Feasibility`/`to_graph`
gaps (genuine core-library holes, fixed upstream), a Picard driver is exactly the kind of
generic, swappable *solver choice* `AbstractDriver` exists to make pluggable, not core
graph machinery; this port is free to bring its own without asking `cottax` to grow one.

Every `FixedPointFunction`/`FixedPointCut` block registered in `total_process.py`
(`EtathLiqStep`, `DeltaEtaStep`, the two cut cross-node cycles once cut, ...) is a
candidate for this driver -- it answers any `FixedPoint`, generically, the same way
`NewtonDriver` answers any `RootFind`.
"""

import jax
import jax.numpy as jnp
from cottax.evaluate import AbstractDriver, ConditionMap
from cottax.problem import FixedPoint
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
