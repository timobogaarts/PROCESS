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

_BOUND: list = []
"""`bind`'s memo: `[((treedef, static), (values, jacobian, values_and_jacobian))]`, one
entry per block.

**A list scanned with `==`, not a dict**, and that is not laziness. `equinox.Module`'s
`__eq__` and `__hash__` disagree here: for the static half of
`eqx.partition((conditions, unravel), eqx.is_array)`, two partitions of the *same* block
compare **equal** while their **hashes differ** (measured 2026-09-02,
`_audit/optimise_design.md` §31.14). Anything keyed on the hash -- a `dict`, or
`jax.jit(..., static_argnums=...)` -- therefore misses every time and retraces the whole
block on every solve, which is exactly the defect §24.1 removed. `eqx.filter_jit` avoids
it by wrapping the static half in a by-value-hashing wrapper, which is *why* today's
spelling caches at all.

The scan is `O(blocks)` at ~25 ms per comparison and runs **once per solve**, against a
compile of ~15 s. A process holds a handful of blocks (MDF's and SAND's, per
configuration), so the list never grows to a size where this matters.
"""


def bind(conditions: ConditionMap, unravel):
    """`(values, jacobian, values_and_jacobian)` for this block, each taking only
    `flat_x`.

    **What this is for.** `flat_conditions` below passes the whole `ConditionMap` as an
    argument, and `eqx.filter_jit` re-partitions, re-flattens and re-hashes it on *every*
    call. That pytree is **5 462 leaves** on `stellarator_helias` MDF, and flattening it
    is 8.37 ms of a 10.58 ms call -- the dominant cost of a host-side SQP iteration, and
    the reason `_audit/optimise_design.md` §18.3 (0.5 ms) and §24.4 (4.66 ms) disagreed
    for a month: §18.3 measured a program with the map folded in as a *constant*.

    Binding does that work **once per solve** instead. The static half is closed over and
    only the **312 array leaves** are passed positionally to a plain `jax.jit`, so a call
    flattens 313 leaves rather than 5 462. Measured interleaved in one process, medians
    over 40 rounds (§31.14):

    | | ms/call |
    |---|---|
    | `flat_conditions` (today) | 10.58 |
    | **bound** | **0.73** |
    | bare `jax.jit` dispatch of one scalar | 0.26 |

    **14.5x, and 0.26 ms of what remains is jax's own dispatch floor** -- so this is
    within about 2x of what any host-side loop can reach. Output is bitwise identical to
    `flat_conditions`.

    **The third callable, and what it is for: compile time.** `jax.jacfwd` computes the
    primal internally -- `vmap(jvp(...))` produces `(y, jac)` and the `has_aux=False`
    spelling throws `y` away -- so `values` and `jacobian` above are two programs over
    the *same* body, and the primal half of the block is traced, lowered and compiled
    **twice**. `values_and_jacobian` is `jax.jacfwd(..., has_aux=True)` over a body
    returning `(stacked, stacked)`: `jvp_subtrace_aux` yields the aux as
    `JVPTracer.primal`, i.e. literally the primal jax already had, so one program
    returns both. A caller that needs both at every point pays one trace instead of two.

    **Which caller could use it, and which could not.**
    `pyvmcon.AbstractProblem.__call__(x)` returns `Result(f, df, eq, deq, ie, die)` --
    value *and* derivative at every point, line-search trial points included -- so
    `VmconDriver` has no value-only call at all (§31.23 counted 552 `values` against 552
    `jacobian` on one row). `SlsqpDriver` hands `scipy` separate `fun` and `jac`
    callables and scipy's line search *does* call `fun` alone, so fusing there would pay
    a whole Jacobian per trial point. `epsfcn`'s finite difference is the other
    value-only caller. Hence three callables and not a replacement: this module hands out
    what is available and the *driver* chooses.

    **It is not bitwise, so nothing chooses it by default.** `VmconDriver.fused` is off,
    and its docstring carries the measurement: the values agree bit for bit, ten of 294
    Jacobian cells move by `4.44e-16`, and one cold-matrix row flips from `converged` to
    `stopped` on that. The *jaxpr* is not the difference -- the same `has_aux` program
    with its primal output dropped again reproduces the split Jacobian exactly -- the
    extra **live output** is, because XLA schedules the tangent computation differently
    once the primal must be materialised too. `_audit/optimise_design.md` §31.30.

    **When something does choose it, the saving arrives by not calling, not by not
    building.** `jax.jit` is lazy: constructing all three wrappers costs nothing, and
    the trace/lower/compile happen on a wrapper's *first call*. So `bind` builds three
    and a solve compiles only the ones it calls -- with `fused` on, the value-only
    program is never traced, and with it off `values_and_jacobian` is never traced. Both
    directions are free, which is what makes the switch cost nothing to carry. [measured
    -- §31.30 counts the emitted programs per row rather than reasoning about laziness.]

    Not free at bind time: `eqx.partition` alone is ~63 ms. That is why the result is
    memoised in `_BOUND` -- see its docstring for why the memo is a list and not a dict.
    """
    dynamic, static = eqx.partition((conditions, unravel), eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(dynamic)

    for (cached_treedef, cached_static), bound in _BOUND:
        if cached_treedef == treedef and cached_static == static:
            return tuple(_timed(fn, leaves) for fn in bound)

    def rebuild(array_leaves):
        return eqx.combine(jax.tree_util.tree_unflatten(treedef, array_leaves), static)

    @jax.jit
    def values(array_leaves, flat_x):
        block, unflatten = rebuild(array_leaves)
        return jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat_x))])

    @jax.jit
    def jacobian(array_leaves, flat_x):
        block, unflatten = rebuild(array_leaves)
        return jax.jacfwd(
            lambda flat: jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat))])
        )(flat_x)

    @jax.jit
    def values_and_jacobian(array_leaves, flat_x):
        block, unflatten = rebuild(array_leaves)

        def stacked_twice(flat):
            # Evaluated **once** and returned twice, not called twice: the second slot
            # is `has_aux`'s, and `jvp_subtrace_aux` takes `.primal` off the tracer it
            # is handed. Calling the body a second time would trace the block twice and
            # give the whole change back.
            out = jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat))])
            return out, out

        derivative, primal = jax.jacfwd(stacked_twice, has_aux=True)(flat_x)
        return primal, derivative

    bound = (values, jacobian, values_and_jacobian)
    _BOUND.append(((treedef, static), bound))
    return tuple(_timed(fn, leaves) for fn in bound)


def _timed(fn, leaves):
    """`fn` with `leaves` bound, inside `phase("model")` -- see `flat_conditions`."""

    def call(flat_x):
        from functional_process.phase_timing import phase  # noqa: PLC0415

        with phase("model"):
            return fn(leaves, flat_x)

    return call


def flat_conditions(conditions: ConditionMap, flat_x, unravel):
    """Timed wrapper around `_flat_conditions`; see `phase_timing`.

    The `model` phase is what a host-side SQP spends *evaluating the graph*, as opposed to
    what it spends in `cvxpy` and its own line search. Splitting them is the only way to
    tell "the model is expensive" from "the optimiser is expensive", and
    `_audit/optimise_design.md` §24 measured those at roughly two-thirds and one-third on
    one configuration -- a number worth having on every arm rather than once.
    """
    from functional_process.phase_timing import phase  # noqa: PLC0415

    with phase("model"):
        return _flat_conditions(conditions, flat_x, unravel)


@eqx.filter_jit
def _flat_conditions(conditions: ConditionMap, flat_x, unravel):
    """The block's conditions, stacked, at one flat design vector.

    `flat_x`/`unravel` rather than the unknowns themselves because that is the shape a
    host-side SQP works in: `pyvmcon` and `scipy` both hand out and take back one flat
    vector, and `unravel` is the `ravel_pytree` companion of the block's `Start`.
    """
    return jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat_x))])


def flat_condition_jacobian(conditions: ConditionMap, flat_x, unravel):
    """Timed wrapper around `_flat_condition_jacobian`; see `flat_conditions`."""
    from functional_process.phase_timing import phase  # noqa: PLC0415

    with phase("model"):
        return _flat_condition_jacobian(conditions, flat_x, unravel)


@eqx.filter_jit
def _flat_condition_jacobian(conditions: ConditionMap, flat_x, unravel):
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
