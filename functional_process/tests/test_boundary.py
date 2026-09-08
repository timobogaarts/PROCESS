"""
The reference machine's boundary, pinned.

What is tested is the *policy*, not the list: that a read with no producer is refused
rather than silently served from PROCESS's `DataStructure`, that the two kinds of
boundary entry are counted apart, and that the reference machine's own boundary is
exactly what the audit says it is. The list itself lives in
`functional_process/reference_boundary.txt` and is generated, never typed.
"""

from pathlib import Path

import pytest
from cottax.graph import Graph
from cottax.spec import ImplementedFunction, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey, unminted
from cottax.tools.path import path_map

from functional_process.cottax.boundary import (
    GUESSED,
    INPUT,
    MISSING_PRODUCERS_INPUT_FILE,
    MISSING_PRODUCERS_PIN,
    STATED,
    TOKAMAK_INPUT_FILE,
    TOKAMAK_PIN,
    boundary,
    category,
    check_boundary,
    computed_by_process,
    counts,
    frozen_cone,
    frozen_reads,
    inert_conditions,
    owned_elsewhere,
    problem_graph,
    read_pin,
    readers_of,
    refuse_inert_conditions,
    unproduced_but_computed,
)
from functional_process.cottax.indat import (
    GRAPH,
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    graph_for,
    machine_from_indat,
)
from functional_process.cottax.mda import driven_graph
from functional_process.cottax.run_cold_matrix import CONFIGURATIONS
from functional_process.cottax.sand import iteration_variable_path
from functional_process.cottax.sand_harness import reference_run


def V(*keys) -> VarPath:
    from jax.tree_util import GetAttrKey

    return VarPath(tuple(GetAttrKey(k) for k in keys))


def G(*keys) -> VarPath:
    from jax.tree_util import GetAttrKey

    return VarPath((MintKey("guess"), *(GetAttrKey(k) for k in keys)))


def C(*keys) -> VarPath:
    """A condition's minted path -- `sand.constraint_nodes`' own `^cond.<place>`."""
    from jax.tree_util import GetAttrKey

    return VarPath((MintKey("cond"), *(GetAttrKey(k) for k in keys)))


def N(*keys) -> NodePath:
    from jax.tree_util import DictKey

    return NodePath(tuple(DictKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(
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
    assert counts(boundary(small)) == {INPUT: 1, GUESSED: 1, STATED: 0}


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
    `$PY -m functional_process.cottax.boundary --write`.
    """
    driven = driven_graph(GRAPH)
    assert [(kind, var.path_str()) for kind, var in boundary(driven)] == list(read_pin())


def test_the_split_is_289_inputs_and_one_guess_per_unsupplied_driven_unknown():
    """The two halves of the boundary are counted apart, and the guess half is derived.

    An `input` is a read no node produces: audited, one row at a time, and the number
    that has to come down. A `guess` is a `Start` port `Assign` mints for a driven
    unknown: mechanical, one per unknown, and it goes *up* when a problem is declared.
    Adding them would let one number sit still while both halves moved, which is the
    whole reason the measure is split.

    So the second half is asserted as a RULE rather than as a count -- every guess pairs
    with a variable the driven graph owns -- and only the audited half is a pinned
    number. Asked of ownership rather than of a `Schedule`, so it does not depend on the
    driver layer.
    """
    driven = driven_graph(GRAPH)
    have = counts(boundary(driven))
    assert have[INPUT] == 289
    assert have[STATED] == 16
    assert have[INPUT] + have[STATED] == len(GRAPH.unowned_inputs) == 305
    assert (
        have[INPUT] + have[GUESSED] + have[STATED] == len(driven.unowned_inputs) == 311
    )

    # Every guess pairs with an unknown, and an unknown is owned *inside* the driven
    # graph -- which is what makes the guess half derived rather than audited. Asked of
    # ownership rather than of a `Schedule`, so it does not depend on the driver layer.
    owned = set(driven.owners)
    guesses = [var for kind, var in boundary(driven) if kind == GUESSED]
    assert len(guesses) == have[GUESSED] == 6
    assert all(unminted(var) in owned for var in guesses)


# ============================================================== the tokamak machine
def test_the_tokamak_s_boundary_is_its_own_pin():
    """The second device, pinned in its own file, by the same rule as the first.

    Equality again, and regenerated the same way --
    `$PY -m functional_process.cottax.boundary --machine --write`. What makes this
    worth a second pin rather than a second column is that a boundary is a property of
    **one assembled graph**: these two machines share five subsystems and a physics core
    and differ in everything else, so the two lists are two measurements, not two views
    of one.
    """
    driven = driven_graph(graph_for(machine_from_indat(TOKAMAK_INPUT_FILE)))
    assert [(kind, var.path_str()) for kind, var in boundary(driven)] == list(
        read_pin(TOKAMAK_PIN)
    )


# ================================================ boundary entries PROCESS computes
def test_no_new_boundary_input_is_something_process_computes():
    """Every missing producer, and nothing else, is on the pin.

    A boundary input PROCESS's own models compute is a **missing producer**: the port
    reads a value it should have derived. `unproduced_but_computed` finds them and the
    pin lists them, so this is equality in both directions -- a new one is an unported
    write, and one that disappears is a producer landed and wants
    `$PY -m functional_process.cottax.boundary --missing --write`.

    The pin is down to one row. That number is in the pin file, not here, so this test
    does not need editing when it moves.
    """
    computed = computed_by_process(MISSING_PRODUCERS_INPUT_FILE)
    reference = reference_run(MISSING_PRODUCERS_INPUT_FILE)
    graph = driven_graph(graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE)))
    design = {iteration_variable_path(i) for i in reference.ixc}
    found = [v.path_str() for v in unproduced_but_computed(graph, computed, design)]
    pinned = [
        line.strip()
        for line in Path(MISSING_PRODUCERS_PIN).read_text().splitlines()
        if line.strip()
    ]
    assert found == pinned


def test_the_stellarator_has_no_reactor_structure_cost():
    """`.costs.reactor_structure_cost` is a tokamak slot and `None` on a stellarator.

    The one slot in this tree whose occupant is decided by the *device*, so it is worth
    a test that says so from both sides rather than only from the tokamak's. `st_strc`
    sets `.structure.fncmass`/`.gsmass` to a literal `0.0`, so an occupant here would
    compute an exact zero out of a subsystem the device does not have -- the
    `EcrhDensityLimit` bug class, which this port has now named in three places and
    should not re-create in a fourth. `.costs.c2214` stays the `0.0` boundary input it
    always was on that machine.
    """
    assert machine_from_indat(MISSING_PRODUCERS_INPUT_FILE).costs.reactor_structure_cost
    assert REFERENCE_MACHINE.costs.reactor_structure_cost is None
    assert V("costs", "c2214") not in GRAPH.owners


def test_the_pf_magnet_cost_landed_without_moving_its_hole():
    """`.costs.c2222` has a producer, `.pf_coil.j_crit_str_pf` has one too, and neither
    is on the pin -- which is the whole claim, because closing the first *without* the
    second would have been a hole moved rather than filled.

    The stellarator half is `reactor_structure_cost`'s argument exactly:
    `caller.py:272-275` returns before `pfcoil.run()`, so every `.pf_coil.*` field the
    account reads keeps its dataclass default and the node would compute an exact zero
    out of a subsystem the device does not have. `.costs.c2222` stays the `0.0` boundary
    input it always was there.
    """
    tokamak = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    owners = {var.path_str(): node.path_str() for var, node in tokamak.owners.items()}
    assert owners[".costs.c2222"] == ".costs.pf_magnet_cost"
    assert owners[".pf_coil.j_crit_str_pf"] == ".tokamak.pf_coil.strand_critical_current"

    pin = Path(MISSING_PRODUCERS_PIN).read_text()
    assert ".costs.c2222" not in pin
    assert ".pf_coil.j_crit_str_pf" not in pin

    # The live arm is `PER_KG`, and the point of the split is that it declares neither
    # of the two critical-current strand fields. `.pf_coil.j_crit_str_cs` has an owner
    # of its own, so only the PF one would show as a boundary read here; both are
    # asserted absent from the account's own inputs, which is the edge count the
    # refusal was about.
    account = next(
        node
        for name, node in tokamak.definitions.items()
        if name.path_str() == ".costs.pf_magnet_cost"
    )
    reads = {port.var.path_str() for port in account.inputs}
    assert not (
        reads
        & {
            ".costs.sc_mat_cost_0",
            ".tfcoil.j_crit_str_0",
            ".pf_coil.j_crit_str_pf",
            ".pf_coil.j_crit_str_cs",
        }
    )

    assert machine_from_indat(MISSING_PRODUCERS_INPUT_FILE).costs.pf_magnet_cost
    assert REFERENCE_MACHINE.costs.pf_magnet_cost is None
    assert V("costs", "c2222") not in GRAPH.owners
    assert V("pf_coil", "j_crit_str_pf") not in GRAPH.owners


# ======================================================= inert conditions (§26)
#
# The guard for the defect class `_audit/optimise_design.md` §26 is about: a condition
# no design variable reaches, whose Jacobian row is therefore identically zero and which
# the optimiser cannot steer. `st_regression` is the case that motivated it -- an
# objective reading a path only the *stellarator* graph owns, frozen at its cold `0.0`,
# so VMCON solved a feasibility problem and reported `converged`.


@pytest.fixture
def steerable():
    """`.d` is a design variable; `^cond.ok` moves with it, `^cond.dead` does not."""
    return Graph(
        path_map({
            N("model"): call([V("d"), V("frozen")], [V("mid")]),
            N("Ok"): call([V("mid")], [C("ok")]),
            N("Dead"): call([V("frozen"), V("other")], [C("dead")]),
        })
    )


def test_a_condition_the_design_reaches_is_not_inert(steerable):
    assert inert_conditions(steerable, [V("d")], [C("ok")]) == ()


def test_a_condition_the_design_cannot_reach_is_named_with_its_frozen_operands(
    steerable,
):
    (row,) = inert_conditions(steerable, [V("d")], [C("ok"), C("dead")])
    assert row.condition == C("dead")
    assert row.node == N("Dead")
    assert row.frozen == (V("frozen"), V("other"))
    assert (row.operands, row.cone) == (2, 2)


def test_frozen_is_the_node_s_own_operands_and_cone_is_the_whole_ancestry(steerable):
    """The correction §26 records: the cone is never empty, so it cannot discriminate.
    `.Ok` reads one owned variable and rests on one boundary input behind it.
    """
    assert frozen_reads(steerable, N("Ok"), [V("d")]) == ()
    assert frozen_cone(steerable, N("Ok"), [V("d")]) == 1


def test_a_design_variable_is_not_frozen():
    """`mdf.mdf_graph` inserts the conditions and not the `Optimise`, so every active
    `ixc` entry is an unowned input of the graph the check runs on. Without the
    subtraction `.Ok`'s cone counts `.d`, the one variable the optimiser is steering.
    """
    graph = Graph(
        path_map({
            N("model"): call([V("d"), V("frozen")], [V("mid")]),
            N("Ok"): call([V("mid")], [C("ok")]),
        })
    )
    assert frozen_cone(graph, N("Ok")) == 2
    assert frozen_cone(graph, N("Ok"), [V("d")]) == 1


def test_a_design_variable_the_graph_does_not_carry_is_dropped_not_raised(steerable):
    """An `ixc` the assembled graph has no variable for is a different defect with its
    own report; this check must not be the thing that fails on it.
    """
    assert inert_conditions(steerable, [V("d"), V("absent")], [C("ok")]) == ()


def test_the_refusal_names_the_row_its_operands_and_the_cause(steerable):
    with pytest.raises(ValueError, match=r"not reachable from any design variable"):
        refuse_inert_conditions(steerable, [V("d")], [C("ok"), C("dead")])
    with pytest.raises(ValueError, match=r"operand\(s\) frozen") as caught:
        refuse_inert_conditions(steerable, [V("d")], [C("dead")])
    message = str(caught.value)
    assert N("Dead").path_str() in message
    assert V("frozen").path_str() in message
    assert "missing producer" in message


def test_a_steerable_problem_is_not_refused(steerable):
    refuse_inert_conditions(steerable, [V("d")], [C("ok")])


def test_st_regression_s_objective_is_inert_and_the_other_six_files_are_clean():
    """**The measurement, and the whole point of the check.** Assembly only -- no
    PROCESS run, no seed, no solve -- so the seven-configuration census is seven graph
    builds.
    """
    expected = {
        "stellarator_helias": set(),
        "helias_5b": set(),
        "large_tokamak_nof": set(),
        "large_tokamak_eval": set(),
        "low_aspect_ratio_DEMO": set(),
        "spherical_tokamak_eval": set(),
        "st_regression": set(),
    }
    found = {}
    for input_file in CONFIGURATIONS:
        stem = Path(input_file).name.removesuffix(".IN.DAT")
        graph, design, driven, _reported = problem_graph(input_file)
        rows = inert_conditions(graph, design, driven)
        found[stem] = {row.node.path_str() for row in rows}
        if stem == "st_regression":
            # The half the census cannot show: the objective is not merely absent from
            # the inert list, it reads a path this graph *owns*. A row that vanished
            # because the condition stopped being assembled would look identical above.
            assert V("current_drive", "big_q_plasma") in graph.owners
            assert V("current_drive", "big_q_plasma") not in set(graph.unowned_inputs)
    assert found == expected


def test_an_evaluation_file_s_inequalities_are_reported_and_not_driven():
    """PROCESS root-finds the equalities alone on `i_process_run_mode = -2` and never
    examines the inequalities, so an inert one there is not a defect -- eight of
    `large_tokamak_eval`'s 23 are inert by design. They come back as `reported` rather
    than being dropped because an inert *reported* row can still mislead a reader, and
    `spherical_tokamak_eval` used to carry the proof: its `.Constraint56` read a frozen
    `0.0` against a bound of `40` where PROCESS at its own answer reads `40.28`, i.e. a
    violated constraint the port printed as satisfied.
    """
    root = Path(TOKAMAK_INPUT_FILE).parent
    graph, design, driven, reported = problem_graph(
        str(root / "spherical_tokamak_eval.IN.DAT")
    )
    assert inert_conditions(graph, design, driven) == ()
    loose = {row.node.path_str() for row in inert_conditions(graph, design, reported)}
    assert loose == set()

    graph, design, driven, reported = problem_graph(TOKAMAK_INPUT_FILE)
    assert inert_conditions(graph, design, driven) == ()
    assert len(inert_conditions(graph, design, reported)) == 8


def test_owned_elsewhere_finds_big_q_plasma_and_is_a_lead_not_a_verdict():
    """The cheap cross-configuration discriminator: a path this graph reads, does not
    own, and another configuration's graph *does* own.

    `.physics.aspect` is what keeps the instrument honest and stays asserted -- a
    stellarator output and a genuine tokamak *input*, so `owned_elsewhere` ranks work
    rather than deciding it. A discriminator with no false positive in it would be a
    verdict, and this one is not.
    """
    root = Path(TOKAMAK_INPUT_FILE).parent
    graph, _design, _driven, _reported = problem_graph(
        str(root / "st_regression.IN.DAT")
    )
    rows = dict(owned_elsewhere(graph, {"reference": GRAPH}))
    assert V("current_drive", "big_q_plasma") not in rows
    assert V("current_drive", "big_q_plasma") in graph.owners
    assert V("physics", "aspect") in rows

    # **The model graph alone still does not see it**, which is §26's finding in one
    # assertion and is the part the registration does *not* change: nothing among the
    # models reads `big_q_plasma`, only the objective node does, so every measurement
    # taken on `driven_graph(graph_for(...))` -- the pins, `provider.answers_for`,
    # `unproduced_but_computed` -- was blind to it by construction and would be blind to
    # the next path of this shape. `inert_conditions` is the instrument that is not.
    models = graph_for(machine_from_indat(str(root / "st_regression.IN.DAT")))
    assert models.readers.get(V("current_drive", "big_q_plasma"), ()) == ()


# Every producer whose landing closed a missing-producer row, as `path -> owning node`.
#
# **One table instead of six tests.** Each of those tests took one wave of producers,
# asserted the same three things about each, and spent most of its length recounting
# which wave it was and what the count moved from and to. The assertions are kept
# verbatim here; the narration is in the commits that landed them.
#
# What this pins that `test_the_tokamak_s_boundary_is_its_own_pin` does not: **which
# node owns each path**. Unwire one and the pin catches it, because the path becomes a
# boundary input; re-home it under a different node and only this table notices.
LANDED_PRODUCERS = {
    ".physics.beta_poloidal_vol_avg": ".tokamak.plasma_beta.poloidal",
    ".tfcoil.sig_tf_case": ".tokamak.cicc_superconducting_tf_coil.tf_stress",
    ".tfcoil.sig_tf_wp": ".tokamak.cicc_superconducting_tf_coil.tf_stress",
    ".tfcoil.str_wp": ".tokamak.cicc_superconducting_tf_coil.tf_stress",
    ".tfcoil.vforce": ".tokamak.cicc_superconducting_tf_coil.tf_field_and_force",
    ".build.z_tf_top": ".tokamak.build.tf_top_height",
    ".build.dz_tf_upper_lower_midplane": ".tokamak.build.tf_top_height",
    ".build.dz_blkt_upper": ".tokamak.build.blkt_upper_thickness",
    ".build.dr_tf_inner_bore": ".tokamak.build.tf_inner_bore",
    ".pf_power.srcktpm": ".power.pf_coil_power",
    ".pf_power.ensxpfm": ".power.pf_coil_power",
    ".heat_transport.peakmva": ".power.pf_coil_power",
    ".pf_coil.p_pf_electric_supplies_mw": ".power.pf_coil_power",
    ".pf_coil.temp_cs_superconductor_margin": ".tokamak.cs_coil.temperature_margin",
    ".fwbs.dewmkg": ".tokamak.cryostat",
    ".buildings.dz_tf_cryostat": ".tokamak.cryostat",
    ".costs.c2214": ".costs.reactor_structure_cost",
    ".cs_fatigue.n_cycle": ".tokamak.cs_fatigue",
    ".cs_fatigue.dz_cs_turn_conduit": ".tokamak.cs_coil.turn_geometry",
}

PF_COIL_ONLY = (
    ".pf_power.srcktpm",
    ".pf_power.ensxpfm",
    ".heat_transport.peakmva",
    ".pf_coil.p_pf_electric_supplies_mw",
    ".pf_coil.temp_cs_superconductor_margin",
)
"""The subset a stellarator must NOT own: it has no PF coils and never calls
`Power.run`, so a stellarator owning a PF-coil power supply or a central-solenoid field
is a mis-wiring the tokamak's own pin cannot see."""


@pytest.mark.parametrize(("path", "owner"), sorted(LANDED_PRODUCERS.items()))
def test_a_landed_producer_is_owned_by_its_node_and_off_the_pin(path, owner):
    """Owned, owned by the node named, and absent from the missing-producer pin."""
    graph = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    owners = {var.path_str(): node.path_str() for var, node in graph.owners.items()}
    assert path in owners, f"{path} lost its producer"
    assert owners[path] == owner
    assert path not in Path(MISSING_PRODUCERS_PIN).read_text()


def test_a_stellarator_owns_none_of_the_pf_coil_producers():
    """The device-dependent half, which no single machine's pin can state."""
    stellarator = graph_for(machine_from_indat(REFERENCE_INPUT_FILE))
    owned = {var.path_str() for var in stellarator.owners}
    assert not (owned & set(PF_COIL_ONLY)), (
        "a stellarator has no PF coils and never calls `Power.run`; nothing on it may "
        "own a PF-coil power-supply or central-solenoid field"
    )
