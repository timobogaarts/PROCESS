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

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.evaluate import ConditionMap

# **There is no memo here any more** (`_audit/optimise_design.md` §37). `_BOUND` was a
# list of compiled blocks scanned with `==`, and it existed because `bind` built its
# `jax.jit` wrappers *inside* the call: a fresh function object every solve, so jax's own
# cache was keyed on something that changed. It is gone, and so is `_Bound`,
# `_BOUND_LIMIT` and the eviction that came with them. The three programs below are
# module level, so they have one identity for the life of the process, and the block's
# structure rides as a `static_argnums` argument that compares **by value** -- which is
# what jax's cache wants and what the port could not offer until `sand._bind` stopped
# building `functools.partial`s (§35) and `cottax` stopped tolerating arrays in a graph
# (§34).


class _Structure:
    """`_flat_key`'s token, with its hash computed **once**.

    A `static_argnums` argument is hashed on *every call*, and this one's `frozen` half
    is a tuple of ~5 000 leaves. Measured on `stellarator_helias` MDF, medians over 60
    calls in one process: a **fresh** token each call is **9.23 ms**, the same token
    reused is **0.89 ms**. So the memoised hash is worth a factor of ten and is the one
    thing deleting `_BOUND` could have quietly cost
    (`_audit/optimise_design.md` §37).

    Memoising the hash removes it, and the precedent is `cottax`'s own `Graph.__hash__`,
    which memoises in `__dict__` for exactly this reason: the object is frozen, so
    nothing goes stale, and a warm one and a cold one stay one jit cache key. `__eq__`
    short-circuits on identity, which is the common case -- `bind` hands the same
    instance to all three programs for the life of a solve -- and falls back to the deep
    comparison that makes a *re-assembled* block a cache hit, which is the whole point.
    """

    __slots__ = ("_hash", "key")

    def __init__(self, key):
        self.key = key
        self._hash = hash(key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other or (type(other) is _Structure and self.key == other.key)


def _flat_key(tree):
    """`(key, array_leaves)` for `tree` -- **one** flatten, no partition.

    The whole point of this function is what it does *not* do. `eqx.partition` over the
    5 462-leaf `(ConditionMap, unravel)` pytree is **44.5 ms** and the `static ==`
    comparison another **15.8 ms** per entry (measured 2026-09-03 on
    `stellarator_helias` MDF, §32.1), and `bind` used to pay both *before* it could tell
    whether it already held the block -- so a cache **hit** cost 60-120 ms, which is
    12-47 % of a steady-state tokamak solve. This is 7.4 ms and answers the same
    question.

    `key` is `(treedef, mask, frozen)` where `treedef` is the flatten of the *whole*
    tree, `mask` says which leaf positions hold arrays, and `frozen` is every non-array
    leaf in order. Those three **determine the partition**: `treedef` plus `mask` is
    exactly the dynamic half's treedef, and `treedef` plus `mask` plus `frozen` is
    exactly the static half's value. So two trees with equal keys have partitions whose
    compiled programs are interchangeable, which is the property `bind` needs.

    The `mask` is in the key and not implied by the other two. Without it,
    `[array, 1.0]` and `[1.0, array]` share a `treedef` and a `frozen`, and their
    dynamic halves differ; it costs a `bytes` comparison to rule out.

    **Array leaves come out in partition order, checked rather than assumed**:
    `jax.tree_util.tree_flatten(eqx.partition(t, eqx.is_array)[0])[0]` and this
    function's `arrays` are the *same objects in the same order* on both reference
    configurations (§32.1) and on a synthetic block
    (`tests/functional_process/core/solver/test_host_cache.py`
    `::test_the_cheap_key_agrees_with_the_partition`).

    The `==` this key is compared with inherits one exposure from the `static ==` it
    replaces, unchanged and worth naming: two frozen leaves that compare equal but are
    not the same value (`0.0` and `-0.0`, `True` and `1`) would be a hit. Equinox's own
    `Module.__eq__` compares its leaves with `==` too, so this is the same set of hits
    the fallback would have allowed, not a new one.

    **The removal condition §32.5 stated has been met, in a better form, and what it
    removed is the memo rather than this function** (`_audit/optimise_design.md` §37).
    The condition was *"when `cottax`'s `Graph` carries a precomputed `(static_key,
    array_leaves)` pair, delete this function and key `bind` on what the graph already
    has"*. `Graph` does not carry a pair; it is **static outright** and holds no array at
    all (§34), which is the same guarantee arrived at from the other side -- so there is
    nothing to key on that this does not already produce more cheaply than the graph
    could hand over.

    So `key` stopped being a *cache probe* and became the **structure token**: it is
    passed to the three module-level programs below as a `static_argnums` argument, and
    jax's own cache does what `_BOUND` was doing. That is the "do not keep both" the note
    asked for -- the second cache is the one that went.

    `key` must therefore be **hashable**, which it is: a `treedef`, a `bytes` mask, and a
    tuple of the non-array leaves. That last part is the clause the array ban bought --
    before §34 a frozen leaf could be a `jax.Array` sitting on a declaration, and a
    `static_argnums` argument holding one is a `TypeError`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    mask = bytes(eqx.is_array(leaf) for leaf in leaves)
    arrays = [leaf for leaf, live in zip(leaves, mask, strict=True) if live]
    frozen = tuple(leaf for leaf, live in zip(leaves, mask, strict=True) if not live)
    return (treedef, mask, frozen), arrays


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

    Binding costs one flatten, and nothing is memoised
    --------------------------------------------------
    Binding is once per *solve*, which is free against a first solve's ~15 s of
    compilation and is not free at all against the ~1 s of a **repeated** solve in one
    process -- the regime `functional_process.session` exists for. It is `_flat_key`'s
    single flatten and no more: no `eqx.partition` (98.7 ms measured on
    `stellarator_helias` MDF), no `static ==` scan (18.2 ms a deep comparison), and no
    memo to scan them against.

    **`_BOUND` is gone** (§37). It existed because these three programs were built
    *inside* this function -- fresh `jax.jit` objects every solve, so jax's own cache was
    keyed on something that changed and the memo had to hold the wrappers. They are
    module level now and `key` rides as a `static_argnums` argument, so jax's cache is
    keyed on the block's structure **by value**: a re-assembled block hits it, which the
    memo never did (§34.8a measured that miss, and §35 removed its last cause).
    """
    key, leaves = _flat_key((conditions, unravel))
    structure = _Structure(key)
    return tuple(
        _timed(fn, structure, leaves)
        for fn in (_values, _jacobian, _values_and_jacobian)
    )


def _rebuild(structure, array_leaves):
    """`(block, unflatten)` from the structure token and this call's array leaves.

    The inverse of `_flat_key`: `mask` says which leaf positions were arrays, so the two
    sequences interleave back into the original flatten in order. A module-level function
    and not a closure, which is the whole reason the programs below can be module level
    too -- see `_audit/optimise_design.md` §37 and
    `~/jaxgraph/plans/closures_and_undeclared_inputs.md`, whose rule this satisfies by
    construction: the structure arrives as an argument and nothing is captured.

    Runs at **trace** time, once per compiled program, never per call.
    """
    treedef, mask, frozen = structure.key
    arrays, frozens = iter(array_leaves), iter(frozen)
    leaves = [next(arrays) if live else next(frozens) for live in mask]
    return jax.tree_util.tree_unflatten(treedef, leaves)


@functools.partial(jax.jit, static_argnums=0)
def _values(structure, array_leaves, flat_x):
    """The block's conditions, stacked, at one flat design vector."""
    block, unflatten = _rebuild(structure, array_leaves)
    return jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat_x))])


@functools.partial(jax.jit, static_argnums=0)
def _jacobian(structure, array_leaves, flat_x):
    """`d(conditions)/d(flat_x)`, forward mode."""
    block, unflatten = _rebuild(structure, array_leaves)
    return jax.jacfwd(
        lambda flat: jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat))])
    )(flat_x)


@functools.partial(jax.jit, static_argnums=0)
def _values_and_jacobian(structure, array_leaves, flat_x):
    """Both, from one trace of the block -- see this module's `bind` docstring."""
    block, unflatten = _rebuild(structure, array_leaves)

    def stacked_twice(flat):
        # Evaluated **once** and returned twice, not called twice: the second slot is
        # `has_aux`'s, and `jvp_subtrace_aux` takes `.primal` off the tracer it is
        # handed. Calling the body a second time would trace the block twice and give
        # the whole change back.
        out = jnp.stack([jnp.asarray(v) for v in block(*unflatten(flat))])
        return out, out

    derivative, primal = jax.jacfwd(stacked_twice, has_aux=True)(flat_x)
    return primal, derivative


def _timed(fn, structure, leaves):
    """`fn` with `structure` and `leaves` bound, inside `phase("model")`.

    The one closure left in this module, and it closes over **structure and arrays that
    are already arguments of the compiled program** -- not over anything the graph cannot
    see. It exists so a driver can hold three callables of `flat_x` alone, which is the
    shape `pyvmcon` and `scipy` ask for.
    """

    def call(flat_x):
        from functional_process.phase_timing import phase  # noqa: PLC0415

        with phase("model"):
            return fn(structure, leaves, flat_x)

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
