"""
Sizing a second device against the tree, without building it.

What is pinned here is the *classification* -- which integers of an input file are
topology decisions, which the factory already dispatches on, which the tree has pinned as
a static kwarg, and which it has never read -- plus the headline counts for
`large_tokamak_eval`, because those are the numbers a plan gets made from and a plan made
against a stale number is the failure mode `next_steps.md` §13.11 records twice.
"""

import pytest

from functional_process.machine_survey import (NOT_TOPOLOGY, SHAPE, Row, factory_fields,
                                               pinned_switches, report, survey)

TOKAMAK = "tests/regression/input_files/large_tokamak_eval.IN.DAT"
HELIAS = "tests/regression/input_files/stellarator_helias.IN.DAT"


def test_the_factory_s_own_fields_are_read_from_its_source():
    '''Derived, not a second list to maintain: a registry added without a row here
    would otherwise show up as "the port has never read it".
    '''
    fields = factory_fields()
    assert {"istell", "ipowerflow", "i_cost_model"} <= fields
    assert len(fields) >= 8


def test_the_tree_s_pinned_switches_are_introspected_not_parsed():
    from functional_process.indat import GRAPH

    pinned = pinned_switches(GRAPH)
    assert "i_tf_sc_mat" in pinned
    # `i_confinement_time` was `{38}` here until the confinement node became slots. Its
    # absence is the assertion now: a switch the factory dispatches on cannot drift from
    # the file, so there is nothing left for this walker to police.
    assert "i_confinement_time" not in pinned
    assert "i_rad_loss" not in pinned


def test_a_file_that_contradicts_a_pinned_switch_says_so():
    (row,) = [r for r in survey(TOKAMAK) if r.name == "i_p_coolant_pumping"]
    assert row.verdict == "pinned" and "DISAGREES" in row.detail and "1" in row.detail
    # It is the last one. `i_confinement_time`, `inuclear` and `i_pulsed_plant` were the
    # other three and are all `factory` now; this is the only contradiction left.
    assert [r.name for r in survey(TOKAMAK) if "DISAGREES" in r.detail] == [
        "i_p_coolant_pumping"
    ]


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


def test_a_count_is_new_work_but_not_a_model_choice():
    (row,) = [r for r in survey(TOKAMAK) if r.name == "n_tf_coils"]
    assert row.verdict == "unknown" and row.name in SHAPE


def test_the_large_tokamak_is_seventeen_new_decisions():
    '''`next_steps.md` §13.9 estimated "~16 genuinely new topology decisions for a
    conventional large tokamak" by hand. This is the same number measured.
    '''
    rows = survey(TOKAMAK)
    kinds = {kind: sum(1 for r in rows if r.verdict == kind)
             for kind in ("factory", "pinned", "unknown", "not-topology")}
    assert kinds == {"factory": 6, "pinned": 4, "unknown": 17, "not-topology": 6}
    assert sum(kinds.values()) == len(rows) == 33


def test_the_coolprop_flag_marks_a_neighbourhood_not_a_branch():
    '''It says "some module reading this switch also reaches CoolProp", which is a
    scheduling hint and not a verdict on the switch's own branch. Asserted as the
    behaviour rather than as the docstring: a test that reads prose passes when the
    prose is wrong.
    '''
    from functional_process.machine_survey import COOLPROP_MODULES

    for row in survey(TOKAMAK):
        touching = [r for r in row.readers if r in COOLPROP_MODULES]
        assert row.coolprop == bool(touching), row.name
        if row.coolprop:
            # ... and the same switch is read by plenty that has nothing to do with it.
            assert len(row.readers) > len(touching)


def test_the_report_names_the_first_deliverable():
    text = report(TOKAMAK)
    assert "band (b)" in text and "i_confinement_time" in text
