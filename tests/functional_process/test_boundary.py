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

from functional_process.boundary import (
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
from functional_process.indat import (
    GRAPH,
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    graph_for,
    machine_from_indat,
)
from functional_process.mda import driven_graph
from functional_process.run_cold_matrix import CONFIGURATIONS
from functional_process.sand import iteration_variable_path
from functional_process.sand_harness import reference_run


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
    `$PY -m functional_process.boundary --write`.
    """
    driven = driven_graph(GRAPH)
    assert [(kind, var.path_str()) for kind, var in boundary(driven)] == list(read_pin())


def test_the_split_is_289_inputs_and_one_guess_per_unsupplied_driven_unknown():
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

    **297 -> 289 input, guesses unchanged**, with `models/initialisation` (the port of
    `init.py` and `st_init`'s `off` writes, `_audit/init_audit.md` §5b). All eight are
    removals and every one is a producer landing: `.tfcoil.eff_tf_cryo` and
    `.buildings.esbldgm3` from `init.py`, and `st_init`'s `.build.dr_cs`,
    `.build.dr_cs_tf_gap` and its four pulse phase durations. They are exactly the
    stellarator pins' eight `off` rows -- the paths where believing the input file and
    the dataclass defaults gave a machine with a central solenoid and a 1000 s burn.

    **289 and 6 unchanged, and 16 `stated` added** (`_audit/optimise_design.md` §34).
    Nine are the *same* producer landings this docstring's last paragraph counts --
    `.tfcoil.eff_tf_cryo`, `.buildings.esbldgm3`, `.costs.c2253`, the solenoid pair and
    the four pulse durations -- still owned by a node, now reading their statement from
    `^stated.<the place>` instead of holding it as a field; the other seven are
    `LModeProfileReset`'s, converted in the same pass and for the same reason. The
    `input` count is the number that must not grow and it did not; had `stated` been
    folded into it the row would read 305 and the distinction would be gone.
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
    """378 inputs and 11 guesses, against the stellarator's 289 and 6.

    **377 -> 378 on 2026-09-01** (missing-producer wave 2, `optimise_design.md` §29):
    three producers landed and exactly **one** new declared read came with them,
    `.constraints.f_fw_rad_max` -- the radiation-wall-load peaking factor, a plain
    `IN.DAT` constant (`core/input.py`, `constraint_variables.py:57`) that PROCESS
    computes nowhere, which is the discriminator the test below applies and not this
    one. None of the three producers' *outputs* was on this file's boundary
    (`.build.r_cp_top` is read only by the picture-frame TF shape, which
    `large_tokamak_eval` does not take; `.physics.p_plasma_separatrix_rmajor_mw` and
    `.constraints.pflux_fw_rad_max_mw` are read only by constraints 56 and 67, which
    this file does not state), so the input half goes **up** by one here while the two
    spherical tokamaks -- where those outputs *are* read -- lose a `computed` row each.
    The stellarator half does not move at all: none of the three nodes is on that
    device.

    **378 -> 377 with the `.tfcoil.dcond` antichain fix.** One node read that array
    *whole* (`.costs.pf_magnet_cost`'s `PER_KG` arms) while three read it *by element*,
    and a path named both whole and indexed is not an antichain -- which `cottax`'s
    pytree machinery refuses, so `mda_env` could not cross a structure boundary and ran
    eagerly. The `PER_KG` occupants now declare `FromExactly(tfcoil.dcond[k])` per
    material, following `radiation_power.py`'s precedent, and the whole-array read is
    gone. The stellarator never read it, which is why only the tokamak half moves.

    **384 -> 378 and 297 -> 289** with `models/initialisation`: six of `init.py`'s
    writes land on this tokamak (`eff_tf_cryo`, `eyoung_ins`, `eyoung_cond_axial`,
    `eyoung_cond_trans`, `rho_pf_coil`, `f_nd_beam_electron`) against two on the
    stellarator, and `st_init`'s six are the stellarator's alone -- so the gap narrows
    from 87 to 89 in the input half while both halves shrink. The device-dependence is
    the point: the same seed file writes different fields on the two machines.

    **369 -> 378 with the TF stress chain** (`models/tfcoil/stress.py`, registry row
    55), and this is the clearest instance in the file of growth being the good kind.
    Ten arrived and one left. The ten are `.tfcoil.eyoung_steel`, `.eyoung_ins`,
    `.eyoung_copper`, `.eyoung_cond_axial`, `.eyoung_cond_trans`, their four matching
    `poisson_*`, and `.tfcoil.f_vforce_inboard` -- every one a material constant or a
    load-split fraction PROCESS itself reads from the input file and computes nowhere,
    which `--missing` confirms rather than this test asserting it. The one that left is
    `.tfcoil.str_wp`, which now has a producer. So the input count went **up** while the
    port got strictly closer to being a function of its inputs, which is exactly why the
    two halves of `test_no_new_boundary_input_is_something_process_computes` and this
    one are separate measurements: a total alone cannot tell a declared read from a lost
    producer.

    **Also 2026-08-30**, in the same day's third wave, five producers landing against one new declared read,
    and the guess half unmoved. The five are the second half of the same wave as the row
    below: `.blanket.deg_blkt_inboard_poloidal_plasma`
    (`.tokamak.ccfe_hcpb.inboard_poloidal_angle`), `.buildings.dz_tf_cryostat`
    (`.tokamak.cryostat`, extended), `.fwbs.p_div_rad_total_mw`
    (`.tokamak.divertor.heat_flux_split`, extended), `.physics.dlamie`
    (`.tokamak.physics.coulomb_logarithm`) and
    `.physics.pflux_plasma_surface_neutron_avg_mw`
    (`.tokamak.physics.plasma_surface_neutron_flux`). The one addition,
    `.build.f_z_cryostat`, is a genuine PROCESS input (`core/input.py:443`) that the
    cryostat's new vertical chain declares -- growth from a landed producer's own reads,
    the good kind.

    That the guess count did **not** move is the second half of the claim: none of the
    five closed a loop, in particular not the blanket angle, whose outboard sibling
    would have (see `BlanketInboardPoloidalAngle`'s docstring for why that one stays
    out).

    **361 -> 369 on 2026-08-30**, in five waves, and the set of them is
    the whole argument for why this number is pinned rather than bounded.

    **361 -> 360.** `.physics.beta_poloidal_vol_avg` left, because
    `.tokamak.plasma_beta.poloidal` landed. It had sat there as an unproduced read since
    `batch5.md` recorded it, feeding `0.0` into `calculate_equilibrium_currents`
    (`optimise_design.md` §16).

    **360 -> 356**, the cleanest run this measure has had: four producers landed and
    **not one new read came with them**, because every field the four nodes declare was
    already on the boundary or already owned. `.tokamak.build.tf_top_height` took
    `.build.z_tf_top` and `.build.dz_tf_upper_lower_midplane`,
    `.tokamak.build.blkt_upper_thickness` took `.build.dz_blkt_upper` (the first one's
    own missing dependency) and `.tokamak.build.tf_inner_bore` took
    `.build.dr_tf_inner_bore`. `z_tf_top` is the one that was doing damage:
    `models/tfcoil/base.py::TfCoilShapeDShapeSingleNull` places the coil's arcs from it
    and `models/pfcoil/geometry.py` places the divertor PF coils from it, so at the cold
    `0.0` the graph drew a TF coil whose top sat on the midplane.

    **356 -> 357**, and this one moves the count the *wrong* way while being the same
    kind of good news. `.power.pf_coil_power` (`Power.pfpwr`) landed: four rows leave
    (`.pf_power.srcktpm`, `.pf_power.ensxpfm`, `.heat_transport.peakmva`,
    `.pf_coil.p_pf_electric_supplies_mw`) and five arrive, which are that node's own
    genuine `IN.DAT` reads -- `.pf_coil.etapsu`, `.pf_coil.rho_pf_coil`,
    `.pf_coil.rhopfbus`, `.pf_power.f_p_pf_energy_store_loss`,
    `.pf_power.f_p_pf_psu_loss`. None of the five is a field PROCESS computes. **So the
    input count going up is not evidence of anything on its own**: what discriminates is
    which side of `computed_by_process` each row falls on, which is the test below and
    not this one.

    The two halves of the missing-producer wave (2026-08-27) moved the input half from
    347 to **361** together: the TF half +9, the CS/physics half −2 +7. Each half's own
    account follows; the merged pin is their union, regenerated on the merged tree.

    The TF half is the clean version of this measure's own good case: **nine additions,
    zero removals, and the guess half unmoved.** Four producers landed --
    `cicc_superconductor_properties` (`.tfcoil.j_tf_wp_critical`, constraint 33),
    `tf_superconductor_temperature_margin` (constraint 36),
    `tf_coil_quench_heat_current_density` (constraint 35) and `vv_stress_on_quench`
    (constraint 65) -- and *nothing left the boundary*, because all four outputs were
    read only by the constraint surface and never by another node. So the whole move is
    the four nodes' own declared reads:

    | new input | declared by |
    |---|---|
    | `.build.r_vv_inboard_out`, `.build.dr_vv_shells`, `.tfcoil.theta1_coil`, `.tfcoil.theta1_vv` | the Itoh VV-quench surrogate |
    | `.tfcoil.tftmp`, `.tfcoil.str_wp` | the critical surface and the temperature margin, both |
    | `.tfcoil.rrr_tf_cu`, `.tfcoil.t_tf_quench_detection`, `.constraints.flu_tf_neutron_fast_max` | the hotspot criterion |

    Eight of the nine are genuine PROCESS inputs. The ninth, **`.tfcoil.str_wp`, is a
    read whose producer is `stresscl` and therefore unported** -- so this row is the
    boundary saying out loud that constraints 33 and 36 depend on the TF stress chain,
    which the graph could not express while their producers were absent.

    The guess count is unchanged at 11, which is the same claim the SCC check makes from
    the other side: none of the four closed a loop.

    The CS half of the same wave added the last four: `.pf_coil.fcuohsu`,
    `.pf_coil.temp_cs_superconductor_operating`, `.tfcoil.poisson_steel` and
    `.tfcoil.str_cs_con_res` -- run inputs that no ported node had declared until the
    CS's critical-current and stress chains landed, closing constraints 26, 27 and half
    of 72. Nothing left the list there at all, for the same reason as below: those
    fields had only constraint readers, and constraints are outside the graph.

    The physics half of the missing-producer wave (2026-08-27) moved the input half
    from 347 to 348, and the direction is the interesting part: **six §11.5 rows closed
    and the count went up**. Four of the six (`beta_thermal_vol_avg`,
    `beta_toroidal_vol_avg`, `beta_vol_avg_max`, `p_div_bt_q_aspect_rmajor_mw`) were
    never in this list at all, because only a *constraint* read them and constraints are
    outside the graph -- so their producers landed as pure additions of declared reads
    (`.physics.beta_beam`) with nothing to cancel against. The other two
    (`nd_plasma_pedestal_electron`, `nd_plasma_separatrix_electron`) did leave, and
    their producer declared the two Greenwald fractions in their place: -2, +3.

    That is the pin earning its keep in the *opposite* direction from the paragraph
    below: the input count alone would have read as a regression here, and only the
    row-by-row equality test above says which rows moved and why.

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
    assert (tok[INPUT], tok[GUESSED]) == (378, 11)
    assert (stell[INPUT], stell[GUESSED]) == (289, 6)


# ================================================ boundary entries PROCESS computes
def test_no_new_boundary_input_is_something_process_computes():
    """The one missing producer left on the MDA graph, pinned so it can only go down.

    **Twenty-two -> one across six waves on 2026-08-30.** One of those waves landed
    five rows at once, and they are
    worth naming because they are five different shapes of the same hole:

    | row | producer | what the port was using instead |
    |---|---|---|
    | `.physics.dlamie` | `.tokamak.physics.coulomb_logarithm` | `0.0` -- and the only writer in `process/` is `Physics.run`, which a stellarator never reaches, so the same field is a genuine input on one machine and a missing producer on the other |
    | `.physics.pflux_plasma_surface_neutron_avg_mw` | `.tokamak.physics.plasma_surface_neutron_flux` | `0.0`, and the whole first-wall neutron flux is `ffwal` times it |
    | `.fwbs.p_div_rad_total_mw` | `.tokamak.divertor.heat_flux_split` | `0.0`, read by four nodes; the producer was one already-pure `@staticmethod` wave 1 chose not to port |
    | `.blanket.deg_blkt_inboard_poloidal_plasma` | `.tokamak.ccfe_hcpb.inboard_poloidal_angle` | `0.0`, which made the divertor subtend 90 degrees instead of 26 |
    | `.buildings.dz_tf_cryostat` | `.tokamak.cryostat` | **`2.5`, not `0.0`** -- a PROCESS `InputVariable` that `external_cryo_geometry` overwrites with `5.573` before the one live reader runs |

    The last of those is the reason this check is asked of PROCESS's *write set* rather
    than of the seeds: twenty of the original twenty-two rows sat at exactly `0.0` and
    could have been found by looking for zeros. That one could not.

    **This is the check that was missing, and it is the reason it was missing.** A
    boundary `input` entry is either one of the ~109 genuine `IN.DAT` inputs or a read
    whose producer is not ported -- this module's docstring says exactly that and counts
    them together, because nothing could tell them apart. A field PROCESS *writes* every
    pipeline pass is definitionally the second kind: nothing in the graph owns it, so it
    stays frozen at whatever the seed supplied while PROCESS recomputes it.

    Twenty-two such rows were found on `large_tokamak_nof` on 2026-08-30, and the
    damage was not subtle: `.physics.beta_poloidal_vol_avg` sat at `0.0` against
    PROCESS's `1.0874` inside `calculate_equilibrium_currents`' O(1) bracket, and the
    error propagated to the burn time (55x) and `stress_shear_cs_peak` (708x), which is
    constraint 72, which is *active* at PROCESS's optimum. Every cold tokamak solve in
    the matrix failed on it.

    **None of the harness's existing stages could see this.** Stage A and C2 seed
    boundary inputs from PROCESS's *converged* `DataStructure` (`sand_harness.
    ground_truth`), which hands every missing producer exactly the right value -- so the
    port reproduced PROCESS to 1e-9 at the one point the bug is structurally invisible.
    Only a cold start exposes it, and only if something asks this question.

    **Measured on `driven_graph`, like every other pin in this file** -- one row,
    down from twenty-two across six waves on 2026-08-30: `.build.z_tf_top`, `.build.dz_tf_upper_lower_midplane`,
    `.build.dz_blkt_upper` and `.build.dr_tf_inner_bore` left when
    `models/namespace.py::Build` gained `tf_top_height`, `blkt_upper_thickness` and
    `tf_inner_bore`, and `.pf_power.srcktpm`, `.pf_power.ensxpfm`,
    `.heat_transport.peakmva` and `.pf_coil.p_pf_electric_supplies_mw` left when
    `.power.pf_coil_power` landed `Power.pfpwr`; and `.physics.dlamie`,
    `.physics.pflux_plasma_surface_neutron_avg_mw`, `.fwbs.p_div_rad_total_mw`,
    `.blanket.deg_blkt_inboard_poloidal_plasma` and `.buildings.dz_tf_cryostat` left
    with the physics/divertor/cryostat wave; and `.tfcoil.str_wp` left with the TF
    stress chain (`models/tfcoil/stress.py`, registry row 55), the sixth wave.

    The MDF-assembled graph used to show **more than this one does** --
    `.tfcoil.sig_tf_case`, `.tfcoil.sig_tf_wp`, and
    `.pf_coil.temp_cs_superconductor_margin` before that -- because the constraint
    surface declares reads the MDA graph never makes. That is still the reason two
    graphs are not pinned in one list, but as of the stress wave all three of those rows
    are owned and the difference between the two graphs is no longer a difference in
    what is missing.

    `.pf_coil.temp_cs_superconductor_margin` is the demonstration that this is a real
    gap and not a bookkeeping one: constraint 60 was comparing a frozen `0.0` against a
    real bound for as long as it sat there, and nothing in *this* list could have said
    so. `.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` were the same and worse: `leq`'s
    normalised residual at a frozen `0.0` operand is exactly `-1.0`, *constant in every
    design variable*, so constraints 31 and 32 -- both active on `large_tokamak_nof` --
    were not merely wrong but contributing zero Jacobian rows. They now read
    `-2.236e-05` (nearly binding) and `-1.347e-01`, agreeing with PROCESS to 6e-16 and
    2e-15. See `_audit/units/models/tfcoil/stress.md`.

    Equality against the pin, not `<=`: a row leaving is a producer landing and should
    update the pin deliberately (`--missing --write`, implemented 2026-08-30 -- the flag
    this line named before anything answered it), and a row arriving is the defect.
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


def test_the_landed_poloidal_beta_producer_is_not_on_the_boundary():
    """`.physics.beta_poloidal_vol_avg` is owned, and the pin does not list it.

    The narrow regression guard for the one producer landed on 2026-08-30. Worth its own
    test rather than trusting the list above: this is the row that broke every cold
    tokamak solve, and a refactor that quietly unwired `.tokamak.plasma_beta.poloidal`
    would put it straight back without changing any value test -- for the same reason
    nothing caught it the first time.
    """
    graph = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    beta_poloidal = V("physics", "beta_poloidal_vol_avg")
    assert beta_poloidal in graph.owners
    assert graph.owners[beta_poloidal].path_str() == ".tokamak.plasma_beta.poloidal"
    assert (
        ".physics.beta_poloidal_vol_avg" not in Path(MISSING_PRODUCERS_PIN).read_text()
    )


def test_the_tf_stresses_constraints_31_and_32_read_have_a_producer():
    """`.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` are owned, and so is `.tfcoil.str_wp`.

    The narrow guard for the TF stress chain (`models/tfcoil/stress.py`, registry row
    55), and the one in this file whose absence was worst. These three were **not** on
    the pin above, because no MDA node reads them -- only the constraint surface does,
    and that surface is added later, on a different graph. So the check that would have
    caught them is not "is this on the missing-producers list" but "does the thing
    constraint 31 compares against come from anywhere", which is what this asserts.

    With no producer, `leq`'s normalised residual at a frozen `0.0` operand is exactly
    `-1.0` -- constant in every design variable, so constraints 31 and 32 were
    contributing zero Jacobian rows on `large_tokamak_nof`, which activates both
    (`IN.DAT:146-147`). At PROCESS's converged point they now read `-2.236e-05` (nearly
    binding against `sig_tf_case_max`) and `-1.347e-01`.

    `.tfcoil.str_wp` is here too because it is the same wave and the same node, and
    because it is the one of the three that *was* on the pin -- it reaches the MDA graph
    through the Nb3Sn critical surface (constraint 33) and the temperature margin (36),
    both of which read it at `i_str_wp == 1`.
    """
    graph = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    pinned = Path(MISSING_PRODUCERS_PIN).read_text()
    owner = ".tokamak.cicc_superconducting_tf_coil.tf_stress"
    for field in ("sig_tf_case", "sig_tf_wp", "str_wp"):
        var = V("tfcoil", field)
        assert var in graph.owners, var.path_str()
        assert graph.owners[var].path_str() == owner
        assert var.path_str() not in pinned

    # `.tfcoil.vforce` is the dependency that made the stresses portable at all: it is
    # `stresscl`'s only load input and nothing in the port produced it either, so
    # landing the stresses without it would have swapped one frozen `0.0` for another.
    vforce = V("tfcoil", "vforce")
    assert vforce in graph.owners
    assert (
        graph.owners[vforce].path_str()
        == ".tokamak.cicc_superconducting_tf_coil.tf_field_and_force"
    )


def test_the_landed_vertical_and_bore_producers_are_not_on_the_boundary():
    """The four producers landed later on 2026-08-30, each named with its owner.

    Same reasoning as the test above: these are exactly the wirings a refactor could
    quietly undo without failing a single value test, because the harness's Stage A and
    C2 seed every boundary input from PROCESS's *converged* `DataStructure`. Named
    rather than left to the pin because two of the four are read by nodes that draw the
    machine -- `.build.z_tf_top` places the TF coil's arcs
    (`TfCoilShapeDShapeSingleNull`) and the divertor PF coils (`pfcoil/geometry.py`) --
    and `.build.dz_blkt_upper` is the first one's own dependency, so an unwiring there
    would silently un-land the other.
    """
    graph = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    pinned = Path(MISSING_PRODUCERS_PIN).read_text()
    for area, field, owner in (
        ("build", "z_tf_top", ".tokamak.build.tf_top_height"),
        ("build", "dz_tf_upper_lower_midplane", ".tokamak.build.tf_top_height"),
        ("build", "dz_blkt_upper", ".tokamak.build.blkt_upper_thickness"),
        ("build", "dr_tf_inner_bore", ".tokamak.build.tf_inner_bore"),
    ):
        var = V(area, field)
        assert var in graph.owners, var.path_str()
        assert graph.owners[var].path_str() == owner
        assert var.path_str() not in pinned


def test_the_pf_power_producers_landed_and_the_stellarator_still_has_none():
    """`Power.pfpwr`'s four fields plus the CS margin are owned; the pin lists none.

    The same narrow guard as the one above, for the five producers landed on 2026-08-30.
    Four come from one node -- `.power.pf_coil_power`, the port of `Power.pfpwr`, which
    `cost_boundary_inputs.md` §7 had recorded as "not ported anywhere in
    `functional_process/` ... there is nothing to register" -- and the fifth,
    `.pf_coil.temp_cs_superconductor_margin`, from `.tokamak.cs_coil.temperature_margin`,
    the deferral `models/pfcoil/superconductor.py` recorded as owed to a shared root
    find with the TF coil's margin.

    **The stellarator half is the assertion that matters most here**, and it is not
    symmetry for its own sake: `Power` is one of the five namespaces `models/tokamak/
    namespace.py` calls device-agnostic, and `pf_coil_power` is the first slot in it
    that is not. A stellarator has no PF coils and `stellarator.py` never calls
    `Power.run`, so the slot is `None` there -- absence spelled as absence. A regression
    that filled it would compute a PF power supply for a machine that has none, which is
    exactly the `EcrhDensityLimit` bug class, and no value test anywhere would notice.
    """
    tokamak = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    expected = {
        V("pf_power", "srcktpm"): ".power.pf_coil_power",
        V("pf_power", "ensxpfm"): ".power.pf_coil_power",
        V("heat_transport", "peakmva"): ".power.pf_coil_power",
        V("pf_coil", "p_pf_electric_supplies_mw"): ".power.pf_coil_power",
        V("pf_coil", "temp_cs_superconductor_margin"): (
            ".tokamak.cs_coil.temperature_margin"
        ),
    }
    pin = Path(MISSING_PRODUCERS_PIN).read_text()
    for var, owner in expected.items():
        assert var in tokamak.owners, f"{var.path_str()} lost its producer"
        assert tokamak.owners[var].path_str() == owner
        assert var.path_str() not in pin

    stellarator = graph_for(machine_from_indat(REFERENCE_INPUT_FILE))
    assert not (set(expected) & set(stellarator.owners)), (
        "a stellarator has no PF coils and never calls `Power.run`; nothing on it may "
        "own a PF-coil power-supply or central-solenoid field"
    )


def test_the_five_landed_producers_own_what_process_computes():
    """The four rows that left the missing-producer pin on 2026-08-30, by owner.

    The same narrow guard the poloidal-beta test above is, for the same reason and one
    wave later: every one of these was a field PROCESS recomputes every pipeline pass
    while the port read `0.0`, and an unwiring that put any of them back -- a slot
    reverted to `None`, a `Costs` occupant dropped on the wrong device, an import
    removed -- would change no value test, because Stage A and C2 seed boundary inputs
    from PROCESS's converged answer and hand a missing producer exactly the right
    number.

    `.cs_fatigue.n_cycle` is on this list without ever having been on the pin, and that
    is the sharpest case here: nothing in the *graph* read it, so `boundary` never saw
    it. Constraint 90 did, from outside, and read `0.0` -- `1 - 0 / n_cycle_min` is
    exactly `+1.000000` with a zero gradient row, which is what stopped every
    `low_aspect_ratio_DEMO` solve at zero iterations.
    """
    graph = graph_for(machine_from_indat(MISSING_PRODUCERS_INPUT_FILE))
    owners = {var.path_str(): node.path_str() for var, node in graph.owners.items()}
    # Two agents ported `.build.dz_blkt_upper` independently on 2026-08-30 and named
    # the slot differently; the build wave's `blkt_upper_thickness` is the one that
    # landed. The duplication is recorded rather than hidden: it is what parallel
    # work on one boundary list costs, and the pin is what made it visible.
    assert owners[".build.dz_blkt_upper"] == ".tokamak.build.blkt_upper_thickness"
    assert owners[".fwbs.dewmkg"] == ".tokamak.cryostat"
    assert owners[".buildings.dz_tf_cryostat"] == ".tokamak.cryostat"
    assert owners[".costs.c2214"] == ".costs.reactor_structure_cost"
    assert owners[".cs_fatigue.n_cycle"] == ".tokamak.cs_fatigue"
    assert owners[".cs_fatigue.dz_cs_turn_conduit"] == ".tokamak.cs_coil.turn_geometry"

    pinned = Path(MISSING_PRODUCERS_PIN).read_text()
    for landed in (
        ".build.dz_blkt_upper",
        ".fwbs.dewmkg",
        ".buildings.dz_tf_cryostat",
        ".costs.c2214",
    ):
        assert landed not in pinned


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

    Account 222.2 was ported and tested long before it was registered, and the refusal
    (`_audit/cost_boundary_inputs.md` §13.2) was about the node's shape and not about a
    missing producer: one class carrying `.costs.supercond_cost_model` as a static kwarg,
    over two arms whose strand-cost reads are disjoint. Registering it as one node would
    have declared `.costs.sc_mat_cost_0`, `.tfcoil.j_crit_str_0`,
    `.pf_coil.j_crit_str_cs` and `.pf_coil.j_crit_str_pf` on a run whose `PER_KG` arm
    reads none of them -- and the last of those four is a field `superconpf` computes
    (`1.1017899e9` A/m^2 on this machine) that nothing owned, so `.costs.c2222` would
    have come off the pin and `.pf_coil.j_crit_str_pf` gone on. The split makes the live
    arm honest; `PFStrandCriticalCurrentDensity` makes the other arm's read producible.

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

    **The name is kept and the expectation has moved, on purpose.** `st_regression`'s
    `.Objective` was the row this check was written for: `objective_metric_5` reads
    `.current_drive.big_q_plasma`, which for a whole session was owned only by
    `models/stellarator/heating.py` and so was a frozen boundary input on this tokamak
    (`optimise_design.md` §26, §27.4). It has a tokamak producer since 2026-09-02 --
    `models/physics/current_drive.py::FusionGain`, `.tokamak.current_drive.fusion_gain`,
    a port of the source's own last line (`current_drive.py:2301-2308`) -- so the census
    is now **clean on six of seven**, and this assertion is what would say so if the
    node were ever unregistered again.

    `helias_5b`'s `.Constraint11` is the *other* kind and is deliberately still
    reported: nothing is missing there, the file's three iteration variables simply do
    not move a stellarator's radial build.

    **`.Constraint56` and `.Constraint67` left this list on 2026-09-01**
    (`optimise_design.md` §29). They were the same defect on constraints -- both
    operands frozen, i.e. a `leq` between two constants -- and both now have producers
    (`.tokamak.physics.psep_over_r_metric`, `.tokamak.radiated_wall_load`).
    **This assertion is the check's own regression test in both directions**: it caught
    the defect when the two constraints and the objective were inert, and it now pins
    that none of the three is.
    """
    expected = {
        "stellarator_helias": set(),
        "helias_5b": {".Constraint11"},
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

    **That file's reported-only list is empty as of 2026-09-01** -- the first
    configuration's to be (`optimise_design.md` §29.6) -- so the *separation* is
    asserted on `large_tokamak_eval`, which still has eight, and
    `spherical_tokamak_eval` now pins the fix. Both halves are kept: dropping the
    spherical assertion would lose the regression test for exactly the defect the wave
    closed.
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

    **It found `big_q_plasma`, and the lead was right** -- the node existed in
    `models/stellarator/heating.py`, every one of its four operands was already on the
    tokamak graph, and the fix was a registration plus a two-line port of the tokamak's
    own line (`current_drive.py:2301-2308` → `FusionGain`, 2026-09-02). So the row this
    test was named for is *gone*, and its absence is now the assertion: `big_q_plasma`
    is owned here.

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
