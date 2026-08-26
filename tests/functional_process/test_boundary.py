"""
The reference machine's boundary, pinned.

What is tested is the *policy*, not the list: that a read with no producer is refused
rather than silently served from PROCESS's `DataStructure`, that the two kinds of
boundary entry are counted apart, and that the reference machine's own boundary is
exactly what the audit says it is. The list itself lives in
`functional_process/reference_boundary.txt` and is generated, never typed.
"""

import pytest

from cottax.graph import Graph
from cottax.spec import CallableNode, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey, unminted
from cottax.tools.path import path_map
from functional_process.boundary import (GUESSED, INPUT, boundary, category,
                                         check_boundary, counts, read_pin, readers_of)
from functional_process.indat import GRAPH
from functional_process.mda import driven_graph


def V(*keys) -> VarPath:
    from jax.tree_util import GetAttrKey
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def G(*keys) -> VarPath:
    from jax.tree_util import GetAttrKey
    return VarPath((MintKey("guess"), *(GetAttrKey(k) for k in keys)))


def N(*keys) -> NodePath:
    from jax.tree_util import DictKey
    return NodePath(tuple(DictKey(k) for k in keys))


def call(reads, owns):
    return CallableNode(inputs=tuple(In(r) for r in reads),
                        outputs=tuple(Out(o) for o in owns), fn=lambda *a: None)


@pytest.fixture
def small():
    """`.a` is produced; `.b` and a start are not."""
    return Graph(path_map({
        N("g", "x"): call([V("b"), G("y")], [V("a")]),
        N("g", "z"): call([V("a")], [V("c")]),
    }))


# ============================================================== the two categories
def test_a_start_port_is_not_counted_as_an_input():
    """The split the whole measure rests on: landing a producer and declaring a problem
    move the total in opposite directions, so a single number can sit still while both
    halves move.
    """
    assert category(V("physics", "rmajor")) == INPUT
    assert category(G("physics", "rmajor")) == GUESSED


def test_boundary_is_categorised_and_stably_ordered(small):
    assert boundary(small) == ((GUESSED, G("y")), (INPUT, V("b")))
    assert counts(boundary(small)) == {INPUT: 1, GUESSED: 1}


# ============================================================== the check
def test_an_unallowed_read_is_refused_and_its_readers_named(small):
    with pytest.raises(ValueError, match=r"\.b") as caught:
        check_boundary(small, [G("y")])
    assert N("g", "x").path_str() in str(caught.value)   # the node left holding it
    assert "silently" in str(caught.value)


def test_the_declared_boundary_passes(small):
    check_boundary(small, [V("b"), G("y")])


def test_a_shrunken_boundary_does_not_fail_the_check(small):
    """One-directional on purpose: a producer landing must not break a build. The pin
    test below is what notices a shrink, as a pin to regenerate.
    """
    check_boundary(small, [V("b"), G("y"), V("never-read")])


def test_readers_of_names_every_consumer(small):
    assert readers_of(small, V("a")) == (N("g", "z"),)


# ============================================================== the reference machine
def test_the_reference_machine_s_boundary_is_the_pin():
    """Equality, not containment: a boundary that grew is a lost producer and a boundary
    that shrank is a producer landed, and both want the pin regenerated --
    `$PY -m functional_process.boundary --write`.
    """
    driven = driven_graph(GRAPH)
    assert [(kind, var.path_str()) for kind, var in boundary(driven)] == list(read_pin())


def test_the_split_is_316_inputs_and_one_guess_per_driven_unknown():
    """The guess half is mechanical -- `Initialise` mints one `Start` per driven unknown
    -- so it is derived, not audited, and pinning it separately is what keeps the
    audited half honest when a problem is added.
    """
    driven = driven_graph(GRAPH)
    have = counts(boundary(driven))
    assert have[INPUT] == len(GRAPH.unowned_inputs) == 316
    assert have[INPUT] + have[GUESSED] == len(driven.unowned_inputs) == 333

    # Every guess pairs with an unknown, and an unknown is owned *inside* the driven
    # graph -- which is what makes the guess half derived rather than audited. Asked of
    # ownership rather than of a `Schedule`, so it does not depend on the driver layer.
    owned = set(driven.owners)
    guesses = [var for kind, var in boundary(driven) if kind == GUESSED]
    assert len(guesses) == have[GUESSED] == 17
    assert all(unminted(var) in owned for var in guesses)
