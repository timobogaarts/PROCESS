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

from functional_process.core.solver import host_cache
from functional_process.core.solver.drivers import _nothing_is_tracing


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


@pytest.fixture(autouse=True)
def _empty_memo():
    """Every test starts and leaves with an empty memo.

    `_BOUND` is module state shared with every other test in the process, and a memo
    left populated would make a later test's `len(_BOUND)` assertion depend on the order
    pytest happened to choose.
    """
    host_cache._BOUND.clear()
    yield
    host_cache._BOUND.clear()


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
    """The key is by *value*, which is the only reason the memo hits at all.

    Two `eqx.partition`s of the same block compare equal and hash differently
    (`_BOUND`'s docstring, §31.14), so a by-value key is forced -- and it has to
    recognise a block rebuilt from scratch, because that is what
    `_sqp_callback`/`mdf.condition_map` hand `bind` on every solve.
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


def test_a_second_bind_of_an_equal_block_is_a_hit_and_does_not_partition(monkeypatch):
    """The measured point of the change: a hit costs one flatten and no partition.

    Counted rather than timed, because a wall clock on a loaded machine cannot tell
    60 ms from 7 ms reliably and the structural claim is the one worth pinning.
    """
    values, jacobian = host_cache.bind(_block(), _unravel())
    assert len(host_cache._BOUND) == 1

    partitions = []
    original = eqx.partition
    monkeypatch.setattr(
        host_cache.eqx,
        "partition",
        lambda *args, **kwargs: (partitions.append(1), original(*args, **kwargs))[1],
    )
    again_values, again_jacobian = host_cache.bind(_block(), _unravel())

    assert not partitions
    assert len(host_cache._BOUND) == 1
    # The *same* jitted callables, not merely equal ones: a fresh pair would own a
    # fresh jax cache entry and re-trace on its first call.
    assert (
        again_values.__closure__[0].cell_contents is values.__closure__[0].cell_contents
    )
    assert (
        again_jacobian.__closure__[0].cell_contents
        is jacobian.__closure__[0].cell_contents
    )


def test_a_structurally_different_block_gets_its_own_entry():
    """Different constants, different program -- see `test_a_non_array_leaf_...`."""
    host_cache.bind(_block(offset=3.0), _unravel())
    host_cache.bind(_block(offset=4.0), _unravel())
    assert len(host_cache._BOUND) == 2


def test_the_memo_is_bounded():
    """`_BOUND` used to be documented as never growing; the naive repeated-solve loop
    falsified that at two entries a solve (§32.2). The bound is the correction.
    """
    for i in range(host_cache._BOUND_LIMIT + 4):
        host_cache.bind(_block(offset=float(i)), _unravel())
    assert len(host_cache._BOUND) == host_cache._BOUND_LIMIT
    # Oldest evicted, newest kept: the block most recently solved is the one most
    # likely to be solved again.
    newest, _ = host_cache._flat_key((
        _block(offset=float(host_cache._BOUND_LIMIT + 3)),
        _unravel(),
    ))
    assert host_cache._BOUND[-1].key == newest


def test_the_bound_callables_compute_the_block_and_its_derivative():
    """A cache is worth nothing if it caches the wrong thing: the bound pair must be the
    block's own values and their Jacobian at the same point.
    """
    block = _block(weight=2.0, offset=3.0)
    values, jacobian = host_cache.bind(block, _unravel())
    at = jnp.asarray([4.0])
    assert values(at).tolist() == pytest.approx([2.0 * 4.0 + 3.0, 4.0 - 1.0])
    assert jacobian(at).ravel().tolist() == pytest.approx([2.0, 1.0])


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
