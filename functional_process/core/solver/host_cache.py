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

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.evaluate import ConditionMap


@dataclasses.dataclass
class _Bound:
    """One memoised block: how to recognise it, and the two compiled callables."""

    key: tuple
    """`_flat_key`'s cheap key -- what a *hit* is decided on. Mutable, because a hit on
    the `==` fallback below records the cheap key it did not match, so the same block is
    recognised cheaply next time."""
    treedef: object
    static: object
    """`eqx.partition((conditions, unravel), eqx.is_array)`'s halves -- the `==`
    fallback's key, kept as a net under `key`. See `bind`."""
    values: object
    jacobian: object


_BOUND: list[_Bound] = []
"""`bind`'s memo, one entry per block, **scanned with `==` and never hashed**.

**Not a dict, and that is not laziness.** `equinox.Module`'s `__eq__` and `__hash__`
disagree here: for the static half of `eqx.partition((conditions, unravel),
eqx.is_array)`, two partitions of the *same* block compare **equal** while their
**hashes differ** (measured 2026-09-02, `_audit/optimise_design.md` §31.14; re-measured
2026-09-03, §32.1 -- still true). Anything keyed on that hash -- a `dict`, or
`jax.jit(..., static_argnums=...)` -- therefore misses every time and retraces the whole
block on every solve, which is exactly the defect §24.1 removed. `eqx.filter_jit` avoids
it by wrapping the static half in a by-value-hashing wrapper, which is *why*
`flat_conditions` caches at all.

**This docstring used to end "the list never grows to a size where this matters", and
that was falsified by measurement** (§32.2): a loop that re-*assembles* the problem per
solve -- which is what calling `run_cold_matrix.run_one` repeatedly does -- appends **two
entries per solve, without bound**, because the freshly built block matches nothing.
`_BOUND_LIMIT` below is the bound that was missing; `functional_process.session` is the
entry point that makes the re-assembly unnecessary in the first place.
"""

_BOUND_LIMIT = 16
"""How many blocks the memo keeps, oldest evicted first.

A process legitimately holds a handful: MDF's block and SAND's, per configuration, and
`run_cold_matrix` clears the memo between configurations anyway. Sixteen is therefore
generous for every intended use and still a *bound*, which the unbounded list was not.

**Eviction is not free and must not be silent to a reader of this file**: dropping an
entry drops the two `jax.jit` wrappers that own the block's compiled programs, so the
next solve of that block re-traces and re-compiles it (~15 s on `stellarator_helias`).
Reaching the limit at all means something is rebuilding blocks in a loop -- the trap
`functional_process.session` exists to route around -- and the right fix is to stop
rebuilding them, not to raise this number.
"""


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

    **This is a stopgap with a stated removal condition** (§32.5). A parallel session is
    designing the same fix one layer up, in `cottax`: a `Graph` that carries both halves
    -- the structural key *and* the precomputed array-leaf list -- computed once at
    construction, where the structure is actually known. **When `cottax`'s `Graph`
    carries a precomputed `(static_key, array_leaves)` pair, delete this function and
    key `bind` on what the graph already has.** Do not keep both: a second cache that
    outlives its reason, because nobody dared remove it, is the failure mode this note
    exists to prevent. It is landing here first because the 60-120 ms a hit costs today
    is worth removing today, and because a number measured in a 180-line module is what
    justifies putting the pair in `Graph` at all.

    The `==` this key is compared with inherits one exposure from the `static ==` it
    replaces, unchanged and worth naming: two frozen leaves that compare equal but are
    not the same value (`0.0` and `-0.0`, `True` and `1`) would be a hit. Equinox's own
    `Module.__eq__` compares its leaves with `==` too, so this is the same set of hits
    the fallback would have allowed, not a new one.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    mask = bytes(eqx.is_array(leaf) for leaf in leaves)
    arrays = [leaf for leaf, live in zip(leaves, mask, strict=True) if live]
    frozen = tuple(leaf for leaf, live in zip(leaves, mask, strict=True) if not live)
    return (treedef, mask, frozen), arrays


def bind(conditions: ConditionMap, unravel):
    """`(values, jacobian)` for this block, each taking only `flat_x`.

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

    A hit is cheap, and it used not to be
    -------------------------------------
    Binding is once per *solve*, which is free against a first solve's ~15 s of
    compilation and is not free at all against the ~1 s of a **repeated** solve in one
    process -- the regime `functional_process.session` exists for. Until 2026-09-03 the
    memo could only be consulted after `eqx.partition` (44.5 ms) and was then scanned
    with `static ==` (15.8 ms an entry), so a *hit* paid the expensive half of a *miss*:
    60 ms on `stellarator_helias` MDF and 100-120 ms on `large_tokamak_nof`, the latter
    17 % of that solve. `_flat_key` above answers the same question in 7.4 ms from a
    single flatten, and it is tried **first**.

    The `(treedef, static)` scan is kept behind it as a net rather than replaced. The
    cheap key is the stricter of the two -- equal frozen leaves and an equal treedef is
    what `Module.__eq__` compares anyway -- so the fallback is expected never to fire;
    if it does, the entry it hits records the cheap key it did not match, so the block
    is recognised cheaply from then on. A *miss* on the cheap key is only ever slow,
    never wrong.

    Not free at bind time on a genuine miss: `eqx.partition` is ~44 ms, against the
    compile it is about to pay for.
    """
    tree = (conditions, unravel)
    key, leaves = _flat_key(tree)

    for entry in _BOUND:
        if entry.key == key:
            return _timed(entry.values, leaves), _timed(entry.jacobian, leaves)

    dynamic, static = eqx.partition(tree, eqx.is_array)
    dyn_leaves, treedef = jax.tree_util.tree_flatten(dynamic)

    for entry in _BOUND:
        if entry.treedef == treedef and entry.static == static:
            entry.key = key
            return _timed(entry.values, dyn_leaves), _timed(entry.jacobian, dyn_leaves)

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

    _BOUND.append(_Bound(key, treedef, static, values, jacobian))
    del _BOUND[:-_BOUND_LIMIT]
    return _timed(values, dyn_leaves), _timed(jacobian, dyn_leaves)


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
