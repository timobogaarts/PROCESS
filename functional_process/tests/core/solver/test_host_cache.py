"""Tests for `host_cache.bind`'s memo and the cheap key it is looked up by.

The whole of `_flat_key`'s claim is that **one flatten answers the question two
traversals used to** -- so these tests pin the two properties that claim rests on
(the array leaves are the partition's, in order; equal blocks produce equal keys) and
the two properties the memo is *for* (a second bind of an equal block is a hit that
does not partition, and the memo is bounded).

A four-leaf `eqx.Module` rather than a real `MdfConditionMap`: `_flat_key` and `bind`
care about a pytree's *shape*, not its size, and a real one costs an MDA prime. The size
claims (5 462 leaves, 44.5 ms to partition) are measurements and live in
`_audit/optimise_design.md` §32, not in a test.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from functional_process.cottax.core.solver import host_cache
from functional_process.cottax.core.solver.drivers import _nothing_is_tracing


class _Block(eqx.Module):
    """A `ConditionMap`-shaped callable: array leaves, non-array leaves, and a `__call__`
    that takes the unknowns positionally and returns one value per condition.
    """

    weight: jax.Array
    offset: float
    label: str

    def __call__(self, u):
        return (self.weight * u + self.offset, u - 1.0)


def _block(weight=2.0, offset=3.0, label="block"):
    return _Block(weight=jnp.asarray(weight), offset=offset, label=label)


def _unravel():
    """The `ravel_pytree` companion of a one-unknown start, as `bind` receives it."""
    return ravel_pytree((jnp.asarray(1.0),))[1]


# **There is no `_empty_memo` fixture any more** (`_audit/optimise_design.md` §37).
# It existed because `_BOUND` was module state shared with every other test in the
# process, so a memo left populated made a later test's `len(_BOUND)` assertion depend on
# collection order. `bind` holds no state now -- jax's own cache does -- and the tests
# below assert compile counts and program identity instead of memo lengths, neither of
# which any other test can perturb.


def test_the_cheap_key_agrees_with_the_partition():
    """`_flat_key`'s array leaves are `eqx.partition`'s dynamic leaves, in order.

    This is the identity the fast path stands on: `bind` hands these leaves to a
    `jax.jit` whose closure was built from `tree_flatten(dynamic)`'s treedef, so if the
    two orders could differ the cached program would be called with its arguments
    permuted -- silently, and with a plausible wrong answer rather than a crash.
    """
    tree = (_block(), _unravel())
    _key, arrays = host_cache._flat_key(tree)
    dynamic, _static = eqx.partition(tree, eqx.is_array)
    expected, _treedef = jax.tree_util.tree_flatten(dynamic)
    assert len(arrays) == len(expected)
    assert all(got is want for got, want in zip(arrays, expected, strict=True))


def test_two_independently_built_equal_blocks_have_equal_keys():
    """The key is by *value*, which is the only reason anything hits at all.

    Two `eqx.partition`s of the same block compare equal and hash differently (§31.14),
    so a by-value key is forced -- and it has to recognise a block rebuilt from scratch,
    because that is what `_sqp_callback`/`mdf.condition_map` hand `bind` on every solve.
    This used to be the memo's key; since §37 it is jax's, which needs the same property
    and hashes as well as compares -- so `_Structure` wraps it and memoises the hash.
    """
    first, _ = host_cache._flat_key((_block(), _unravel()))
    second, _ = host_cache._flat_key((_block(), _unravel()))
    assert first == second


def test_a_non_array_leaf_that_differs_is_a_different_key():
    """A frozen leaf is part of the traced program, so it must not be cached over.

    `offset` rides in the closure as a constant; a key that ignored it would hand a
    second block the first block's compiled program.
    """
    first, _ = host_cache._flat_key((_block(offset=3.0), _unravel()))
    second, _ = host_cache._flat_key((_block(offset=4.0), _unravel()))
    assert first != second


def test_an_array_leaf_that_differs_is_the_same_key():
    """An array leaf is an *argument*, not a constant, so two blocks differing only in
    their array values share one program -- which is the whole point of binding.
    """
    first, _ = host_cache._flat_key((_block(weight=2.0), _unravel()))
    second, _ = host_cache._flat_key((_block(weight=9.0), _unravel()))
    assert first == second


def _compiles(fn):
    """`(result, XLA compilations)` -- §22.2's instrument, scoped to one call."""
    from jax._src import compiler

    seen = []
    real = compiler.backend_compile_and_load
    compiler.backend_compile_and_load = lambda *a, **k: (
        seen.append(1),
        real(*a, **k),
    )[1]
    try:
        return fn(), len(seen)
    finally:
        compiler.backend_compile_and_load = real


def test_a_second_bind_of_an_equal_block_compiles_nothing(monkeypatch):
    """The point of §37: a re-assembled block is a **jax cache hit**, with no memo.

    `_BOUND` could never do this. It compared `(treedef, static)` and a re-assembled
    block's `static` never matched, because the bodies held `functools.partial`s that
    compare by identity (§34.8a measured six such leaves of 4902; §35 removed them). Now
    the structure token is a `static_argnums` argument that compares **by value**, so two
    independently built equal blocks are one entry in jax's own cache.

    Counted rather than timed, because a wall clock on a loaded machine cannot tell a
    trace from a cache hit reliably and the compile count is the claim worth pinning.
    """
    values, _jac, _both = host_cache.bind(_block(), _unravel())
    values(jnp.asarray([1.0]))  # pay for the first compile

    again, compiles = _compiles(
        lambda: host_cache.bind(_block(), _unravel())[0](jnp.asarray([1.0]))
    )
    assert compiles == 0

    # And no `eqx.partition`: binding is `_flat_key`'s single flatten and nothing else.
    partitions = []
    original = eqx.partition
    monkeypatch.setattr(
        host_cache.eqx,
        "partition",
        lambda *args, **kwargs: (partitions.append(1), original(*args, **kwargs))[1],
    )
    host_cache.bind(_block(), _unravel())
    assert not partitions


def test_a_structurally_different_block_is_its_own_program():
    """Different constants, different program -- see `test_a_non_array_leaf_...`.

    A frozen leaf is part of the structure token, so it is part of jax's cache key: this
    compiles, where `test_a_second_bind_...` does not.
    """
    host_cache.bind(_block(offset=3.0), _unravel())[0](jnp.asarray([1.0]))
    _out, compiles = _compiles(
        lambda: host_cache.bind(_block(offset=4.0), _unravel())[0](jnp.asarray([1.0]))
    )
    assert compiles >= 1


def test_binding_holds_no_state():
    """There is nothing left to grow, which is the other half of deleting the memo.

    `_BOUND` had to be bounded (`_BOUND_LIMIT = 16`, with eviction) because a loop that
    re-assembled appended two entries a solve without limit (§32.2). Nothing here
    accumulates: `bind` returns three closures over an argument and this module holds no
    container at all.
    """
    assert not hasattr(host_cache, "_BOUND")
    assert not hasattr(host_cache, "_BOUND_LIMIT")
    assert not hasattr(host_cache, "_Bound")


def test_the_bound_callables_compute_the_block_and_its_derivative():
    """A cache is worth nothing if it caches the wrong thing: all three bound callables
    must be the block's own values and their Jacobian at the same point.
    """
    block = _block(weight=2.0, offset=3.0)
    values, jacobian, fused = host_cache.bind(block, _unravel())
    at = jnp.asarray([4.0])
    assert values(at).tolist() == pytest.approx([2.0 * 4.0 + 3.0, 4.0 - 1.0])
    assert jacobian(at).ravel().tolist() == pytest.approx([2.0, 1.0])
    fused_values, fused_jacobian = fused(at)
    assert fused_values.tolist() == pytest.approx([2.0 * 4.0 + 3.0, 4.0 - 1.0])
    assert fused_jacobian.ravel().tolist() == pytest.approx([2.0, 1.0])


def test_nothing_is_tracing_is_true_on_concrete_values():
    """The eager branch of `_sqp_callback` is what a host-side solve normally takes."""
    assert _nothing_is_tracing((jnp.asarray(1.0), 2.0, "label"))


def test_nothing_is_tracing_is_false_under_a_trace():
    """Under a trace the `pure_callback` is load-bearing: without it a `Drive` answered
    by `pyvmcon` is a hole in the program (`_sqp_callback`'s own docstring).
    """
    seen = []

    @jax.jit
    def probe(x):
        seen.append(_nothing_is_tracing((x, 2.0)))
        return x

    probe(jnp.asarray(1.0))
    assert seen == [False]
