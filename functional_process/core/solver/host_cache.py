"""Compiled, cached calls into a `ConditionMap` from a **host-side** solver loop.

Why this module exists at all, in one sentence: `pyvmcon` and `scipy.optimize` iterate
in Python, on the host, so there is no enclosing jit for the per-iteration model call to
be hoisted into, and something has to own the compilation of it.

That is a forced inelegance and it is kept here, next to the drivers that caused it,
rather than pushed anywhere more central:

- **not on `ConditionMap`.** `cottax.evaluate.ConditionMap` is a plain callable and
  should stay one. Handing out pre-compiled callables would put a notion of "compiled"
  into the graph's own vocabulary, and only a host-loop driver ever needs it.
- **not inside a driver.** `jax.jit` caches on the identity of the function it wraps, so
  a jit built inside a solve is a fresh object every solve and jax's cache misses --
  measured at 2 XLA compiles on *every* call of an already-compiled SAND schedule
  (`_audit/optimise_design.md` §22.2; §24.1 is the removal). Hoisting into the driver's
  `__call__` fixes that only for the life of one trace: a driver invoked eagerly twice
  rebuilds it. A module-level function has one identity for the life of the process, so
  the cache is keyed on its arguments, which is where a cache key belongs.

A jax-native constrained optimiser would import none of this. It would call
`jax.jacfwd(conditions)` inside its own trace and never meet a compilation boundary at
all, which is the measure of how much of this module is essential (none) and how much is
the price of a Python solver loop (all of it).

**What makes the cache hit**, both facts measured rather than assumed (§24.1):

- `eqx.filter_jit` traces the array leaves of its arguments and keys the cache on
  everything else, wrapped so that it hashes **by value**. A `ConditionMap` rebuilt by
  `eqx.combine(dyn, static)` on every solve -- which is what `_sqp_callback` must do,
  since `jax.pure_callback` carries only arrays and the `fn`s have to ride in the
  closure -- is a *different object* with an *equal* static half, so it is one key.
- `unravel` rides as an ordinary (non-array) argument, i.e. statically.
  `jax.flatten_util.ravel_pytree` returns a `jax._src.util.HashablePartial`, which also
  hashes by value, so two `unravel`s built from the same pytree structure are one key
  and not two.

**The cost, stated once for both functions.** The condition map's live arrays are now
*arguments* to a compiled program rather than constants folded into one. That is exactly
the mechanism that makes the cache reusable, and it is also the one thing that can move
an answer: XLA may constant-fold a closed-over array and may not fold an argument. This
problem is measurably sensitive at the last bit (§19, §20, §21.2), so §24.2 is a bitwise
cold-matrix check and not an assertion that nothing moved.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.evaluate import ConditionMap


@eqx.filter_jit
def flat_conditions(conditions: ConditionMap, flat_x, unravel):
    """The block's conditions, stacked, at one flat design vector.

    `flat_x`/`unravel` rather than the unknowns themselves because that is the shape a
    host-side SQP works in: `pyvmcon` and `scipy` both hand out and take back one flat
    vector, and `unravel` is the `ravel_pytree` companion of the block's `Start`.
    """
    return jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat_x))])


@eqx.filter_jit
def flat_condition_jacobian(conditions: ConditionMap, flat_x, unravel):
    """`d(conditions)/d(flat_x)` by forward-mode AD -- `flat_conditions`' Jacobian.

    **A second jitted function, not `jax.jacfwd(flat_conditions)`.** `jax.jacfwd` returns
    an ordinary Python function that traces its argument under JVP; it is not compiled,
    and calling one eagerly runs the whole block op by op. So the `jacfwd` goes *inside*
    and the jit outside, and this -- one forward pass per design variable -- is the
    expensive call of the two.
    """
    return jax.jacfwd(
        lambda flat: jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat))])
    )(flat_x)
