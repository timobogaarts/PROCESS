"""Compiled, cached calls into a `ConditionMap` from a **host-side** solver loop."""

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
    """`_flat_key`'s token, with its hash computed **once**."""

    __slots__ = ("_hash", "key")

    def __init__(self, key):
        self.key = key
        self._hash = hash(key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other or (type(other) is _Structure and self.key == other.key)


def _flat_key(tree):
    """`(key, array_leaves)` for `tree` -- **one** flatten, no partition."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    mask = bytes(eqx.is_array(leaf) for leaf in leaves)
    arrays = [leaf for leaf, live in zip(leaves, mask, strict=True) if live]
    frozen = tuple(leaf for leaf, live in zip(leaves, mask, strict=True) if not live)
    return (treedef, mask, frozen), arrays


def bind(conditions: ConditionMap, unravel):
    """`(values, jacobian, values_and_jacobian)` for this block, each taking only
    `flat_x`.
    """
    key, leaves = _flat_key((conditions, unravel))
    structure = _Structure(key)
    return tuple(
        _timed(fn, structure, leaves)
        for fn in (_values, _jacobian, _values_and_jacobian)
    )


def _rebuild(structure, array_leaves):
    """`(block, unflatten)` from the structure token and this call's array leaves."""
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
    """`fn` with `structure` and `leaves` bound, inside `phase("model")`."""

    def call(flat_x):
        from functional_process.cottax.phase_timing import phase  # noqa: PLC0415

        with phase("model"):
            return fn(structure, leaves, flat_x)

    return call


def flat_conditions(conditions: ConditionMap, flat_x, unravel):
    """Timed wrapper around `_flat_conditions`; see `phase_timing`."""
    from functional_process.cottax.phase_timing import phase  # noqa: PLC0415

    with phase("model"):
        return _flat_conditions(conditions, flat_x, unravel)


@eqx.filter_jit
def _flat_conditions(conditions: ConditionMap, flat_x, unravel):
    """The block's conditions, stacked, at one flat design vector."""
    return jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat_x))])


def flat_condition_jacobian(conditions: ConditionMap, flat_x, unravel):
    """Timed wrapper around `_flat_condition_jacobian`; see `flat_conditions`."""
    from functional_process.cottax.phase_timing import phase  # noqa: PLC0415

    with phase("model"):
        return _flat_condition_jacobian(conditions, flat_x, unravel)


@eqx.filter_jit
def _flat_condition_jacobian(conditions: ConditionMap, flat_x, unravel):
    """`d(conditions)/d(flat_x)` by forward-mode AD -- `flat_conditions`' Jacobian."""
    return jax.jacfwd(
        lambda flat: jnp.stack([jnp.asarray(v) for v in conditions(*unravel(flat))])
    )(flat_x)
