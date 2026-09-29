"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CROCOSuperconductingTFCoil`, the cross-conductor (CroCo) REBCO-tape TF coil.
"""

from cottax.interfaces import V

from functional_process.cottax.paths import superconducting_tfcoil, tfcoil
from functional_process.cottax.stated import states
from functional_process.models.tfcoil.croco import (
    calculate_durham_rebco_croco_temperature_margin,
    calculate_hazelton_zhai_rebco_croco_temperature_margin,
    croco_averaged_turn_geometry_from_current_per_turn,
    croco_cable_geometry,
    croco_cable_space_properties,
    croco_inboard_areas_and_fractions,
    croco_superconductor_properties_gl_rebco,
    croco_superconductor_properties_hijc_rebco,
    croco_turn_cable_space_cooling_fraction,
    croco_turn_cable_space_extra_void,  # noqa: F401 -- re-exported for indat.py / tests
    temperature_margin_hijc_rebco,  # noqa: F401 -- re-exported for tests/.../test_croco.py
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


# The CrocoTurnGeometry family:
# The family that owns the CroCo winding-pack turn geometry.


# Both turn-dimension input flags `False`.
#
# PROCESS's default and both ST files'.
CrocoAveragedTurnGeometryFromCurrentPerTurn = tfcoil.function(
    croco_averaged_turn_geometry_from_current_per_turn,
    "a_tf_turn_insulation",
    "n_tf_coil_turns",
    "dx_tf_turn_general",
    V.superconducting_tfcoil.dr_tf_turn,
    V.superconducting_tfcoil.dx_tf_turn,
    "dx_tf_turn_conduit_full_average",
    V.superconducting_tfcoil.dx_tf_turn_cable_space_average,
    reads={
        "a_tf_wp_no_insulation": superconducting_tfcoil,
    },
)


# cottax node: `tf_turn_croco_cable_space_properties`.
CrocoCableSpaceProperties = superconducting_tfcoil.function(
    croco_cable_space_properties,
    "dia_tf_turn_croco_cable",
    V.tfcoil.a_tf_turn_cable_space_no_void,
    "a_tf_turn_cable_space_effective",
    V.tfcoil.a_tf_turn_steel,
    reads={
        "dx_tf_turn_conduit_full_average": tfcoil,
        "dx_tf_turn_steel": tfcoil,
    },
)


# cottax node: `superconductors.calculate_croco_cable_geometry`.
CrocoCableGeometry = superconducting_tfcoil.function(
    croco_cable_geometry,
    "dia_tf_croco_strand_tape_region",
    "n_tf_croco_strand_hts_tapes",
    "a_tf_croco_strand_copper_total",
    "a_tf_croco_strand_hastelloy",
    "a_tf_croco_strand_solder",
    "a_tf_croco_strand_rebco",
    "a_tf_croco_strand",
    "dr_tf_hts_tape",
    "dx_tf_hts_tape_total",
    "dx_tf_croco_strand_tape_stack",
    reads={
        "dia_croco_strand": V.superconducting_tfcoil.dia_tf_turn_croco_cable,
        "dx_croco_strand_copper": V.superconducting_tfcoil.dx_tf_croco_strand_copper,
        "dx_hts_tape_rebco": V.superconducting_tfcoil.dx_tf_hts_tape_rebco,
        "dx_hts_tape_copper": V.superconducting_tfcoil.dx_tf_hts_tape_copper,
        "dx_hts_tape_hastelloy": V.superconducting_tfcoil.dx_tf_hts_tape_hastelloy,
    },
)


# cottax node: `run`'s literal `f_a_tf_turn_cable_space_extra_void = 0.0`
# (`superconducting.py:3894`).
#
# The ported literal, *stated* at
# `^stated.tfcoil.f_a_tf_turn_cable_space_extra_void` rather than produced inside the
# body -- a value built during the trace is a constant exactly as the literal was, and
# one held on the declaration is an array the graph may not carry (`models/stated.py`,
# `_audit/optimise_design.md` §28, §34).
CrocoTurnCableSpaceExtraVoid = states(tfcoil, "f_a_tf_turn_cable_space_extra_void")


# cottax node: `tf_croco_inboard_areas_and_fractions`. No switch.
CrocoInboardAreasAndFractions = tfcoil.function(
    croco_inboard_areas_and_fractions,
    "a_tf_wp_coolant_channels",
    "a_tf_wp_conductor",
    "a_tf_wp_extra_void",
    "a_tf_coil_wp_turn_insulation",
    "a_tf_wp_steel",
    V.superconducting_tfcoil.a_tf_coil_inboard_steel,
    V.superconducting_tfcoil.f_a_tf_coil_inboard_steel,
    V.superconducting_tfcoil.a_tf_coil_inboard_insulation,
    V.superconducting_tfcoil.f_a_tf_coil_inboard_insulation,
    reads={
        "a_tf_wp_ground_insulation": superconducting_tfcoil,
        "a_tf_croco_strand": superconducting_tfcoil,
    },
)


# cottax node: the one live line of `run`'s inline copper block
# (`superconducting.py:3947-3955`).
CrocoTurnCableSpaceCoolingFraction = superconducting_tfcoil.function(
    croco_turn_cable_space_cooling_fraction,
    "f_a_tf_turn_cable_space_cooling",
    reads={
        "a_tf_turn_cable_space_no_void": tfcoil,
    },
)


# The CrocoSuperconductorProperties family:
# The family that owns the CroCo critical-current chain -- constraint 33's read.


# `i_tf_sc_mat == 9` *(live on both tracked ST files)*.
HazeltonZhaiRebcoCrocoSuperconductorProperties = tfcoil.function(
    croco_superconductor_properties_hijc_rebco,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.cur_tf_turn_croco_strand_critical,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "cur_tf_turn": V.tfcoil.c_tf_turn,
        "temp_tf_peak": V.tfcoil.tftmp,
        "dr_tf_hts_tape": superconducting_tfcoil,
        "dx_tf_hts_tape_rebco": superconducting_tfcoil,
        "dx_tf_hts_tape_total": superconducting_tfcoil,
        "a_tf_croco_strand": superconducting_tfcoil,
    },
)


# `i_tf_sc_mat == 8` -- Durham GL REBCO, the one tape arm that reads a strain.
DurhamRebcoCrocoSuperconductorProperties = tfcoil.function(
    croco_superconductor_properties_gl_rebco,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.cur_tf_turn_croco_strand_critical,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "cur_tf_turn": V.tfcoil.c_tf_turn,
        "temp_tf_peak": V.tfcoil.tftmp,
        "strain": V.tfcoil.str_wp,
        "a_tf_croco_strand": superconducting_tfcoil,
    },
)


# `i_tf_sc_mat == 8` -- its temperature margin, on the unclipped strain.
DurhamRebcoCrocoTemperatureMargin = tfcoil.function(
    calculate_durham_rebco_croco_temperature_margin,
    "temp_tf_superconductor_margin",
    "temp_margin",
    reads={
        "j_tf_superconductor": superconducting_tfcoil,
        "b_tf_superconductor_critical_zero_temp_strain": superconducting_tfcoil,
        "temp_tf_superconductor_critical_zero_field_strain": superconducting_tfcoil,
    },
)


# `i_tf_sc_mat == 9` -- constraint 36's read on a CroCo machine.
HazeltonZhaiRebcoCrocoTemperatureMargin = tfcoil.function(
    calculate_hazelton_zhai_rebco_croco_temperature_margin,
    "temp_tf_superconductor_margin",
    "temp_margin",
    reads={
        "j_tf_superconductor": superconducting_tfcoil,
        "b_tf_superconductor_critical_zero_temp_strain": superconducting_tfcoil,
        "temp_tf_superconductor_critical_zero_field_strain": superconducting_tfcoil,
        "dr_tf_hts_tape": superconducting_tfcoil,
        "dx_tf_hts_tape_rebco": superconducting_tfcoil,
        "dx_tf_hts_tape_total": superconducting_tfcoil,
    },
)
