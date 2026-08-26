"""
Sizing a second device against the tree, without building it.

What is pinned here is the *classification* -- which integers of an input file are
topology decisions, which the factory already dispatches on, which the tree has pinned as
a static kwarg, and which it has never read -- plus the headline counts for
`large_tokamak_eval`, because those are the numbers a plan gets made from and a plan made
against a stale number is the failure mode `next_steps.md` §13.11 records twice.
"""

import pytest

from functional_process.machine_survey import (
    NOT_TOPOLOGY,
    SHAPE,
    Row,
    factory_fields,
    pinned_switches,
    report,
    survey,
)

TOKAMAK = "tests/regression/input_files/large_tokamak_eval.IN.DAT"
HELIAS = "tests/regression/input_files/stellarator_helias.IN.DAT"


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
    (groups,) = [r for r in survey(TOKAMAK) if r.name == "n_pf_coil_groups"]
    assert groups.verdict == "unknown" and groups.name in SHAPE


def test_the_large_tokamak_is_ten_new_decisions():
    """`next_steps.md` §13.9 estimated "~16 genuinely new topology decisions for a
    conventional large tokamak" by hand, and the first measurement agreed at 17.
    **Ten are left**, and the seven that closed are what the first tokamak porting wave
    bought.

    This is the number the plan gets made from, so it is pinned rather than described:
    `unknown` is the tokamak's actual model debt and moves only when a model is written.
    """
    rows = survey(TOKAMAK)
    kinds = {
        kind: sum(1 for r in rows if r.verdict == kind)
        for kind in ("factory", "pinned", "unknown", "not-topology")
    }
    # 7 -> 15 `factory`, 3 -> 2 `pinned`, 17 -> 10 `unknown`, over one wave. The seven
    # that moved out of `unknown` are `i_div_heat_load`, `i_hcd_primary`,
    # `i_plasma_current`, `i_plasma_geometry`, `i_single_null`, `pulsetimings` and
    # `n_tf_coils`; `i_p_coolant_pumping` and `i_plasma_pedestal` moved out of `pinned`.
    # What is left is the unported half of the device: bootstrap current, the density
    # limit, plasma inductance, beta components, the PF coil system and the shield.
    assert kinds == {"factory": 15, "pinned": 2, "unknown": 10, "not-topology": 6}
    assert sum(kinds.values()) == len(rows) == 33


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
