"""
Sizing a second device against the tree, without building it.

What is pinned here is the *classification* -- which integers of an input file are
topology decisions, which the factory already dispatches on, which the tree has pinned as
a static kwarg, and which it has never read -- plus the headline counts for
`large_tokamak_eval`, because those are the numbers a plan gets made from and a plan made
against a stale number is the failure mode `next_steps.md` §13.11 records twice.
"""

import pytest

from functional_process import indat
from functional_process.machine_survey import (
    NOT_TOPOLOGY,
    ROUTED_AWAY,
    SHAPE,
    Row,
    assembly_verdict,
    factory_fields,
    pinned_switches,
    report,
    slot_registries,
    survey,
    unoccupied_registries,
)

TOKAMAK = "tests/regression/input_files/large_tokamak_eval.IN.DAT"
HELIAS = "tests/regression/input_files/stellarator_helias.IN.DAT"
SPHERICAL = "tests/regression/input_files/spherical_tokamak_eval.IN.DAT"


def test_the_factory_s_own_fields_are_read_from_its_source():
    """Derived, not a second list to maintain: a registry added without a row here
    would otherwise show up as "the port has never read it".
    """
    fields = factory_fields()
    assert {"istell", "ipowerflow", "i_cost_model"} <= fields
    assert len(fields) >= 8


def test_the_tree_s_pinned_switches_are_introspected_not_parsed():
    from functional_process.indat import GRAPH

    pinned = pinned_switches(GRAPH)
    assert "i_p_coolant_pumping" in pinned
    # Three switches left this set by becoming slots: `i_confinement_time`/`i_rad_loss`
    # (the confinement split) and `i_tf_sc_mat` (`_audit/next_steps.md` §14.5). Their
    # absence is the assertion now -- a switch the factory dispatches on cannot drift
    # from the file, so there is nothing left for this walker to police.
    assert "i_confinement_time" not in pinned
    assert "i_rad_loss" not in pinned
    assert "i_tf_sc_mat" not in pinned


def test_no_pinned_switch_contradicts_the_tokamak_any_more():
    """**The contradiction list is empty**, and that is the tokamak wave's result here.

    `i_p_coolant_pumping` was the last of four. It was `pinned` and DISAGREEing --
    hardcoded `1` on five nodes, against the file's `3` -- and it did not become
    `factory` by anyone deciding to tidy it: assembling a tokamak with the real value
    made `power`'s `p_fw_blkt_coolant_pump_mw_step` and `.tokamak.ccfe_hcpb.pumping_power`
    both claim `.primary_pumping.p_fw_blkt_coolant_pump_mw`, and cottax refused the graph
    by name. A pinned switch that disagrees with the file is a latent dual-ownership bug,
    and this is the first time one of the four was cashed in as such.

    The other three closed earlier: `i_confinement_time` with the confinement split,
    `inuclear` and `i_pulsed_plant` with their own slots.

    Asserted as an empty list rather than deleted, because the check is still live in
    both directions: a new static kwarg that disagrees with this file fails here, and
    `test_the_reference_run_contradicts_nothing` asks the same of the stellarator.
    """
    (row,) = [r for r in survey(TOKAMAK) if r.name == "i_p_coolant_pumping"]
    assert row.verdict == "factory" and "DISAGREES" not in row.detail
    assert [r.name for r in survey(TOKAMAK) if "DISAGREES" in r.detail] == []


def test_a_switch_the_factory_dispatches_on_cannot_contradict_anything():
    """`i_confinement_time` was the fourth contradiction until the confinement split.

    It is `factory` now, not `pinned`, which is what closing one of them looks like from
    here: the tokamak asks for 34, the registry has an occupant for 34, and no value is
    transcribed anywhere that could disagree with the file. The `factory`/`pinned` split
    moves as switches are converted; `unknown` is the tokamak's actual model debt and
    does not move until a model is written.
    """
    (row,) = [r for r in survey(TOKAMAK) if r.name == "i_confinement_time"]
    assert row.verdict == "factory" and "DISAGREES" not in row.detail


def test_the_reference_run_contradicts_nothing():
    """The tree was built from this file, so every pin must agree with it. A failure
    here means a static kwarg drifted from the run it models -- `switch_audit`'s
    defect class, asked of the input file instead of the converged state.
    """
    assert [r for r in survey(HELIAS) if "DISAGREES" in r.detail] == []


def test_run_control_and_array_artefacts_are_excluded_with_a_reason():
    verdicts = {r.name: r.verdict for r in survey(TOKAMAK)}
    for name in ("icc", "ixc", "i_process_run_mode", "p_fusion_total_max_mw"):
        assert verdicts[name] == "not-topology", name
        assert NOT_TOPOLOGY[name]


def test_a_count_can_be_new_work_and_a_factory_key_at_once():
    """`n_tf_coils` is in `SHAPE` -- it sizes arrays rather than selecting a model -- and
    the factory nonetheless dispatches on it.

    Both halves are true and neither cancels the other. `superconducting.py`'s ripple fit
    has four arms keyed on `round(n_tf_coils)`, selecting different fit coefficients
    *and* different numbers of owned outputs, so it is a switch at that one site; it is
    still a count everywhere else, and `n_pf_coil_groups` beside it is a count with no
    site at all. The row keeps its `SHAPE` detail text for that reason.

    `superconducting.md`'s open question 2 is what makes this worth pinning: a build-time
    branch on a count is sound only while the count is not an optimiser unknown.
    `n_tf_coils` is not one today (`iteration_variables.py` does not list it).
    """
    (row,) = [r for r in survey(TOKAMAK) if r.name == "n_tf_coils"]
    assert row.verdict == "factory" and row.name in SHAPE
    # `n_pf_coil_groups` was the count with no site; it has one now --
    # `_pf_coil_system_arm` refuses any group topology other than the ported
    # `(4, (2,2,3,3), (1,1,2,2))` -- so it is the second count-and-key at once.
    (groups,) = [r for r in survey(TOKAMAK) if r.name == "n_pf_coil_groups"]
    assert groups.verdict == "factory" and groups.name in SHAPE


def test_the_large_tokamak_is_three_new_decisions():
    """`next_steps.md` §13.9 estimated "~16 genuinely new topology decisions for a
    conventional large tokamak" by hand, and the first measurement agreed at 17.
    **Three are left**, and the fourteen that closed are what the two tokamak porting
    waves and their consolidations bought.

    This is the number the plan gets made from, so it is pinned rather than described:
    `unknown` is the tokamak's actual model debt and moves only when a model is written.
    """
    rows = survey(TOKAMAK)
    kinds = {
        kind: sum(1 for r in rows if r.verdict == kind)
        for kind in ("factory", "pinned", "unknown", "not-topology")
    }
    # 7 -> 15 -> 17 -> **24** `factory`, 3 -> 2 -> **0** `pinned`, 17 -> 10 -> **3**
    # `unknown`, over three waves. The first wave moved `i_div_heat_load`,
    # `i_hcd_primary`, `i_plasma_current`, `i_plasma_geometry`, `i_single_null`,
    # `pulsetimings` and `n_tf_coils` out of `unknown`; waves 2/3's consolidation
    # moved `i_alphaj`, `i_bootstrap_current`, `i_cs_superconductor`,
    # `i_density_limit`, `i_ind_plasma_internal_norm`, `i_pf_superconductor` and
    # `n_pf_coil_groups`; `i_p_coolant_pumping` and `i_plasma_pedestal` left `pinned`
    # in the first wave.
    #
    # **`pinned` is empty**, and that is `_audit/next_steps.md` §14.2's whole point:
    # a switch this file sets is now either read by the factory or genuinely not a
    # topology decision. The last two, `i_blkt_dual_coolant` and
    # `i_thermal_electric_conversion`, are still `eqx.field(static=True)` on
    # `ComponentThermalPowers`/`DeltaEtaStep` -- but the factory reads them and threads
    # them, so no slot can contradict the file. `machine_survey` classifies on what the
    # factory reads, not on what a node carries, which is why they count as `factory`.
    #
    # What is left in `unknown` is the genuinely undecided remainder:
    # `i_beta_component` (the beta-limit component split, `physics.py` decision 7),
    # `i_plant_availability` (read in `availability.py`'s dispatch, whose non-default
    # arms are unported) and `i_shld_primary_heat` (read in `power.py`).
    assert kinds == {"factory": 24, "pinned": 0, "unknown": 3, "not-topology": 6}
    assert sum(kinds.values()) == len(rows) == 33


def test_an_unknown_row_says_which_of_three_reasons_it_is():
    """ "`unknown`" means "no slot dispatches on it", and that is three different
    situations. The report used to call all three "the port has never read it", which
    is false for **every** `unknown` row this file produces.

    Two of the three are read, just not by the machine tree: the constraint/objective
    layer binds `i_beta_component` and `i_plant_availability` as static kwargs, and a
    node declares `.heat_transport.i_shld_primary_heat` as an ordinary `In` -- the
    latter being `switch_kwarg_survey.md` §0's "declared port carrying a switch integer"
    seen from a second direction, i.e. work rather than absence. The verdict is
    deliberately left as `unknown` for all three (the counts above are cited from
    `next_steps.md`); only the sentence changes.
    """
    detail = {r.name: r.detail for r in survey(TOKAMAK) if r.verdict == "unknown"}
    assert set(detail) == {
        "i_beta_component",
        "i_plant_availability",
        "i_shld_primary_heat",
    }
    assert "static kwarg" in detail["i_beta_component"]
    assert "static kwarg" in detail["i_plant_availability"]
    assert ".heat_transport.i_shld_primary_heat" in detail["i_shld_primary_heat"]
    assert not any("never read" in d for d in detail.values())


def test_the_coolprop_flag_marks_a_neighbourhood_not_a_branch():
    """It says "some module reading this switch also reaches CoolProp", which is a
    scheduling hint and not a verdict on the switch's own branch. Asserted as the
    behaviour rather than as the docstring: a test that reads prose passes when the
    prose is wrong.
    """
    from functional_process.machine_survey import COOLPROP_MODULES

    for row in survey(TOKAMAK):
        touching = [r for r in row.readers if r in COOLPROP_MODULES]
        assert row.coolprop == bool(touching), row.name
        if row.coolprop:
            # ... and the same switch is read by plenty that has nothing to do with it.
            assert len(row.readers) > len(touching)


def test_the_report_no_longer_names_a_first_deliverable():
    """The "band (b) slots are the first tokamak deliverable" paragraph is **gone**,
    because it is printed only when some pinned switch contradicts the file and none
    does. Asserted as its absence together with the count line that explains the absence,
    so this cannot pass by the paragraph merely being reworded.
    """
    text = report(TOKAMAK)
    assert "band (b)" not in text
    assert "0 of which this file contradicts" in text
    assert "i_confinement_time" in text


def test_a_value_with_no_occupant_and_no_recorded_reason_is_not_reported_as_dispatched():
    """The blind spot the ST closing wave found, pinned.

    `_slot_occupant` has **two** failure modes and this module used to check one. A value
    in `UNPORTED` is a refusal with a reason; a value in neither the registry nor
    `UNPORTED` is a `ValueError`, and until 2026-08-29 it reported as *"the factory
    dispatches on it"* -- true of the field, false of the value. That is how
    `i_beta_norm_max = 0` survived a re-survey whose entire purpose was to enumerate what
    both spherical tokamaks still needed.

    Asserted on a value that is deliberately absent from a registry today rather than on
    `i_beta_norm_max` itself, which the same wave gave an occupant: pinning the fixed case
    would test nothing.
    """
    assert unoccupied_registries("i_beta_norm_max", 1) == ()
    assert unoccupied_registries("i_beta_norm_max", 0) == ()
    # `2` (Menard) is in `UNPORTED` and out of the registry -- both signals fire, and the
    # recorded reason is the one that should be shown.
    assert unoccupied_registries("i_beta_norm_max", 2) == ("BETA_NORM_MAX",)
    row = next(r for r in survey(SPHERICAL) if r.name == "i_beta_norm_max")
    assert row.verdict == "factory"
    assert row.detail == "the factory dispatches on it"


def test_every_routed_away_entry_is_earned():
    """A `ROUTED_AWAY` exemption must be true in both directions.

    Same discipline as `_harness/boundary.py`'s register: the value must genuinely be
    absent from the registry it is exempted from **and** present in the one it is routed
    to, so an exemption cannot outlive the routing it describes.
    """
    for (field, value), (absent_from, routed_to) in ROUTED_AWAY.items():
        assert absent_from in slot_registries()[field], (
            f"{absent_from} is not a registry `_slot_occupant` reads for {field}"
        )
        assert value not in getattr(indat, absent_from), (
            f"{field} == {value} now has an entry in {absent_from}; the exemption is "
            "stale"
        )
        assert getattr(indat, routed_to), f"{routed_to} is empty"


def test_the_report_ends_with_what_the_factory_actually_does():
    """The switch table cannot see a slot dispatched on a *derived* arm index, so the
    report ends with one real assembly attempt.

    Both tracked spherical tokamaks were surveyed as "exactly four switch values away"
    while also being refused by `pf_coil_system_arm`, whose name appears in no `IN.DAT`.
    A column can never show that; an assembly attempt always can.
    """
    assert assembly_verdict(TOKAMAK) == "ASSEMBLES."
    assert report(TOKAMAK).rstrip().endswith("ASSEMBLES.")
    refusal = assembly_verdict(SPHERICAL)
    assert refusal.startswith("ASSEMBLY REFUSED")
