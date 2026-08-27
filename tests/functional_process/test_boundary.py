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


def test_the_split_is_297_inputs_and_one_guess_per_unsupplied_driven_unknown():
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

    **311 -> 297 input and 16 -> 6 guess** with §14.2's switch conversion, and the two
    halves moved for two different reasons:

    * The **ten guesses** are ten driven unknowns that stopped being driven, because
      splitting a switch showed the fixed point was the switch's artefact and not the
      model's. `^guess.costs.cplife` and `^guess.heat_transport.eta_turbine` were the
      two `FixedPoint`s `switch_kwarg_survey.md` §4.7 measured as **the identity map**;
      the other eight (`etath_liq`, `temp_turbine_coolant_in`,
      `p_fw_div_heat_deposited_mw`, `p_fw_blkt_coolant_pump_mw`, and the four `q*`)
      became ordinary `ExplicitFunction`s once the pass-through arm was spelled as an
      empty slot instead of a self-read.
    * The **input half is +2 -16**. The two additions are the honest consequence of the
      first bullet: `.costs.cplife` at `itart = 0` and `.heat_transport.eta_turbine` at
      `i_thermal_electric_conversion = USER_INPUT` are fields PROCESS itself takes as
      inputs, and the tree says so now instead of driving a residual that determines
      nothing. The sixteen removals are dead reads that left with their arms -- see
      `_audit/next_steps.md` §14.11 for the per-variable attribution.
    """
    driven = driven_graph(GRAPH)
    have = counts(boundary(driven))
    assert have[INPUT] == len(GRAPH.unowned_inputs) == 297
    assert have[INPUT] + have[GUESSED] == len(driven.unowned_inputs) == 303

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


def test_the_tokamak_reads_more_than_the_stellarator_and_guesses_more():
    """347 inputs and 11 guesses, against the stellarator's 297 and 6.

    The cold-boundary wave (2026-08-27) moved the input half from 349 to 347 by
    landing `cold_boundary.md`'s four missing producers: nine rows closed
    (`dr_fw_inboard`/`dr_fw_outboard`, `r_tf_inboard_in`/`_mid`/`_out`, `dr_cs_bore`,
    `res_plasma`, `p_plasma_ohmic_mw`, `vs_cs_pf_total_burn`) against seven genuine
    new reads the four nodes declare (`dr_bore`, `dr_cs_tf_gap`, `fseppc`, `fcspc`,
    `sigallpc`, `dr_fw_wall`, `plasma_res_factor`). The guess count is unchanged at
    11: registering `pfcoil.vsec` merged the volt-second/burn-time and PF coil cycles
    into one nine-node SCC, but its `FixedPoint` owns the same three unknowns the two
    halves owned (`mda.CUTS`, measured in `test_mda.py`).

    Both halves are the expected shape and neither is obviously good news, which is why
    they are pinned as numbers rather than described:

    * **More inputs, from more nodes.** The tokamak graph is larger than the
      stellarator's and still reads ~50 more variables it does not produce. Waves 2/3's
      consolidation moved this from 328 to 349 while *closing* ten rows
      (`.physics.plasma_current`, `.physics.alphaj`, `.physics.f_c_plasma_auxiliary`,
      `.fwbs.vol_shld_total`, `.times.t_plant_pulse_burn` and five `.pf_coil.*`
      extents/masses): the eleven newly registered slots' nodes declare thirty-one
      genuine inputs of their own, each named in advance by its record's "boundary
      inputs this slot then needs" list. Growth from a landed producer's own declared
      reads is the boundary doing its job; growth from a *lost* producer is the defect,
      and the pin-equality test above is what tells the two apart, row by row.
    * **More guesses, and they are not a cost.** A guess is a `Start` port minted per
      driven *unknown*, so the count tracks how much of the graph is genuinely coupled
      rather than how much is missing. Measured: the two machines share five
      (`fusden_alpha_total`, `proton_rate_density`, `temp_plasma_ion_vol_avg_kev`,
      `power.delta_eta`, `vacuum.d_duct`); the stellarator has
      `^guess.fwbs.f_ster_div_single` alone; and the tokamak has six of its own --
      `^guess.tfcoil.dx_tf_wp_primary_toroidal` and `^guess.tfcoil.dr_tf_plasma_case`
      closing the build/winding-pack cycles,
      `^guess.physics.f_temp_plasma_electron_density_vol_avg` on the pedestal density
      cycle, `^guess.times.t_plant_pulse_burn` on the volt-second/burn-time cycle, and
      `^guess.pf_coil.ind_pf_cs_plasma_mutual` + `^guess.pf_coil.n_pf_coil_turns` --
      PROCESS's own `first_call` seeds -- on the PF coil cycle (`mda.CUTS`). So
      5 + 1 = 6 and 5 + 6 = 11.

    The numbers move whenever a producer lands, which is what makes them worth pinning:
    growth in the input half is a **lost** producer, and that is the failure this whole
    module exists to catch.
    """
    stell = counts(boundary(driven_graph(GRAPH)))
    tok = counts(
        boundary(driven_graph(graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))))
    )
    assert (tok[INPUT], tok[GUESSED]) == (347, 11)
    assert (stell[INPUT], stell[GUESSED]) == (297, 6)
