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
from functional_process.boundary import (
    GUESSED,
    INPUT,
    TOKAMAK_INPUT_FILE,
    TOKAMAK_PIN,
    boundary,
    category,
    check_boundary,
    counts,
    read_pin,
    readers_of,
)
from functional_process.indat import GRAPH, graph_for, machine_from_indat
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
    return CallableNode(
        inputs=tuple(In(r) for r in reads),
        outputs=tuple(Out(o) for o in owns),
        fn=lambda *a: None,
    )


@pytest.fixture
def small():
    """`.a` is produced; `.b` and a start are not."""
    return Graph(
        path_map({
            N("g", "x"): call([V("b"), G("y")], [V("a")]),
            N("g", "z"): call([V("a")], [V("c")]),
        })
    )


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
    assert N("g", "x").path_str() in str(caught.value)  # the node left holding it
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


def test_the_split_is_311_inputs_and_one_guess_per_unsupplied_driven_unknown():
    """The guess half is mechanical -- `Assign` mints one `Start` per driven unknown --
    so it is derived, not audited, and pinning it separately is what keeps the audited
    half honest when a problem is added.

    **Not quite one per unknown any more**: a `Start` that `mda.supply_starts` points at
    a node (`cottax.rewrites.Supply`) has a producer, so it is not a boundary input at
    all. That is the direction this number should move in -- a guess PROCESS itself
    computes is an edge of the model, not something the caller must hand in -- and it is
    why the two halves are counted apart. 316 -> 311 input and 17 -> 16 guess when
    `i_tf_sc_mat` became a slot (`_audit/next_steps.md` §14.5): five material fields that
    only unselected branches read, plus `^guess.stellarator.wp_width_r_min`, which
    `.stellarator.wp_width_r_min_guess` now supplies.
    """
    driven = driven_graph(GRAPH)
    have = counts(boundary(driven))
    assert have[INPUT] == len(GRAPH.unowned_inputs) == 311
    assert have[INPUT] + have[GUESSED] == len(driven.unowned_inputs) == 327

    # Every guess pairs with an unknown, and an unknown is owned *inside* the driven
    # graph -- which is what makes the guess half derived rather than audited. Asked of
    # ownership rather than of a `Schedule`, so it does not depend on the driver layer.
    owned = set(driven.owners)
    guesses = [var for kind, var in boundary(driven) if kind == GUESSED]
    assert len(guesses) == have[GUESSED] == 16
    assert all(unminted(var) in owned for var in guesses)


# ============================================================== the tokamak machine
def test_the_tokamak_s_boundary_is_its_own_pin():
    """The second device, pinned in its own file, by the same rule as the first.

    Equality again, and regenerated the same way --
    `$PY -m functional_process.boundary --machine --write`. What makes this worth a
    second pin rather than a second column is that a boundary is a property of **one
    assembled graph**: these two machines share five subsystems and a physics core and
    differ in everything else, so the two lists are two measurements, not two views of
    one.
    """
    driven = driven_graph(graph_for(machine_from_indat(TOKAMAK_INPUT_FILE)))
    assert [(kind, var.path_str()) for kind, var in boundary(driven)] == list(
        read_pin(TOKAMAK_PIN)
    )


def test_the_tokamak_reads_more_than_the_stellarator_and_guesses_less():
    """339 inputs and 15 guesses, against the stellarator's 311 and 16.

    Both halves are the expected shape and neither is obviously good news, which is why
    they are pinned as numbers rather than described:

    * **More inputs, from more nodes.** The tokamak graph is 170 nodes to the
      stellarator's 159, and it still reads 28 more variables it does not produce --
      because eleven of `Tokamak`'s twenty-five slots are still empty, and
      `_audit/tokamak_boundary.md` is the enumeration of what that costs. A device with
      *more* ported nodes and *more* boundary inputs is exactly what a half-ported device
      looks like.
    * **One fewer guess**, and it is not a saving: a guess is a `Start` port minted per
      driven unknown, so the count tracks how many problems the graph declares. The
      tokamak has no counterpart to `.stellarator.coils.intersect`'s.

    The numbers move whenever a producer lands, which is what makes them worth pinning:
    growth in the input half is a **lost** producer, and that is the failure this whole
    module exists to catch.
    """
    stell = counts(boundary(driven_graph(GRAPH)))
    tok = counts(
        boundary(driven_graph(graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))))
    )
    assert (tok[INPUT], tok[GUESSED]) == (339, 15)
    assert (stell[INPUT], stell[GUESSED]) == (311, 16)
