"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CROCOSuperconductingTFCoil`, the cross-conductor (CroCo) REBCO-tape TF coil.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.stated import StatesValues
from functional_process.cottax.tfcoil.superconducting import (
    TfSuperconductorTemperatureMargin,
)
from functional_process.cottax.paths import superconducting_tfcoil, tfcoil
from functional_process.models.tfcoil.croco import (
    calculate_hazelton_zhai_rebco_croco_temperature_margin,
    croco_averaged_turn_geometry_from_current_per_turn,
    croco_cable_geometry,
    croco_cable_space_properties,
    croco_inboard_areas_and_fractions,
    croco_superconductor_properties_hijc_rebco,
    croco_turn_cable_space_cooling_fraction,
    croco_turn_cable_space_extra_void,  # noqa: F401 -- re-exported for indat.py / tests
    temperature_margin_hijc_rebco,  # noqa: F401 -- re-exported for tests/.../test_croco.py
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class CrocoTurnGeometry(ExplicitFunction):
    """The family that owns the CroCo winding-pack turn geometry."""


class CrocoAveragedTurnGeometryFromCurrentPerTurn(CrocoTurnGeometry):
    """Both turn-dimension input flags `False` -- PROCESS's default and both ST files'.
    """

    a_tf_turn_insulation = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        j_tf_wp=From(tfcoil),
        c_tf_turn=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        layer_ins=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
    ):
        return croco_averaged_turn_geometry_from_current_per_turn(
            j_tf_wp=j_tf_wp,
            c_tf_turn=c_tf_turn,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            layer_ins=layer_ins,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        )


class CrocoCableSpaceProperties(ExplicitFunction):
    """cottax node: `tf_turn_croco_cable_space_properties`."""

    dia_tf_turn_croco_cable = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_turn_conduit_full_average=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
    ):
        return croco_cable_space_properties(
            dx_tf_turn_conduit_full_average=dx_tf_turn_conduit_full_average,
            dx_tf_turn_steel=dx_tf_turn_steel,
        )


class CrocoCableGeometry(ExplicitFunction):
    """cottax node: `superconductors.calculate_croco_cable_geometry`."""

    dia_tf_croco_strand_tape_region = OutputInto(superconducting_tfcoil)
    n_tf_croco_strand_hts_tapes = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_copper_total = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_hastelloy = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_solder = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_rebco = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand = OutputInto(superconducting_tfcoil)
    dr_tf_hts_tape = OutputInto(superconducting_tfcoil)
    dx_tf_hts_tape_total = OutputInto(superconducting_tfcoil)
    dx_tf_croco_strand_tape_stack = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        dia_tf_turn_croco_cable=From(superconducting_tfcoil),
        dx_tf_croco_strand_copper=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_copper=From(superconducting_tfcoil),
        dx_tf_hts_tape_hastelloy=From(superconducting_tfcoil),
    ):
        return croco_cable_geometry(
            dia_croco_strand=dia_tf_turn_croco_cable,
            dx_croco_strand_copper=dx_tf_croco_strand_copper,
            dx_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_hts_tape_copper=dx_tf_hts_tape_copper,
            dx_hts_tape_hastelloy=dx_tf_hts_tape_hastelloy,
        )


class CrocoTurnCableSpaceExtraVoid(StatesValues):
    """cottax node: `run`'s literal `f_a_tf_turn_cable_space_extra_void = 0.0`
    (`superconducting.py:3894`).
    """

    f_a_tf_turn_cable_space_extra_void = OutputInto(tfcoil)
    """The ported literal, *stated* at
    `^stated.tfcoil.f_a_tf_turn_cable_space_extra_void` rather than produced inside the
    body -- a value built during the trace is a constant exactly as the literal was, and
    one held on the declaration is an array the graph may not carry (`models/stated.py`,
    `_audit/optimise_design.md` §28, §34).
    """


class CrocoInboardAreasAndFractions(ExplicitFunction):
    """cottax node: `tf_croco_inboard_areas_and_fractions`. No switch."""

    a_tf_wp_coolant_channels = OutputInto(tfcoil)
    a_tf_wp_conductor = OutputInto(tfcoil)
    a_tf_wp_extra_void = OutputInto(tfcoil)
    a_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    a_tf_wp_steel = OutputInto(tfcoil)
    a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        a_tf_turn_insulation=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
        a_tf_wp_ground_insulation=From(superconducting_tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_inboard_areas_and_fractions(
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            n_tf_coil_turns=n_tf_coil_turns,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_insulation=a_tf_turn_insulation,
            a_tf_turn_steel=a_tf_turn_steel,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            n_tf_coils=n_tf_coils,
            a_tf_inboard_total=a_tf_inboard_total,
            a_tf_wp_ground_insulation=a_tf_wp_ground_insulation,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class CrocoTurnCableSpaceCoolingFraction(ExplicitFunction):
    """cottax node: the one live line of `run`'s inline copper block
    (`superconducting.py:3947-3955`).
    """

    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_turn_cable_space_cooling_fraction(
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class CrocoSuperconductorProperties(ExplicitFunction):
    """The family that owns the CroCo critical-current chain -- constraint 33's read."""

    j_tf_wp_critical = OutputInto(tfcoil)
    j_crit_str_tf = OutputInto(tfcoil)
    f_c_tf_turn_operating_critical = OutputInto(superconducting_tfcoil)
    j_tf_coil_turn = OutputInto(superconducting_tfcoil)
    j_tf_superconductor = OutputInto(superconducting_tfcoil)
    cur_tf_turn_croco_strand_critical = OutputInto(superconducting_tfcoil)
    c_tf_turn_cables_critical = OutputInto(superconducting_tfcoil)
    j_tf_superconductor_critical = OutputInto(superconducting_tfcoil)
    b_tf_superconductor_critical_zero_temp_strain = OutputInto(superconducting_tfcoil)
    temp_tf_superconductor_critical_zero_field_strain = OutputInto(
        superconducting_tfcoil
    )


class HazeltonZhaiRebcoCrocoSuperconductorProperties(CrocoSuperconductorProperties):
    """`i_tf_sc_mat == 9` *(live on both tracked ST files)*."""

    def __call__(
        self,
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        c_tf_turn=From(tfcoil),
        tftmp=From(tfcoil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_superconductor_properties_hijc_rebco(
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            cur_tf_turn=c_tf_turn,
            temp_tf_peak=tftmp,
            dr_tf_hts_tape=dr_tf_hts_tape,
            dx_tf_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total=dx_tf_hts_tape_total,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class HazeltonZhaiRebcoCrocoTemperatureMargin(TfSuperconductorTemperatureMargin):
    """`i_tf_sc_mat == 9` -- constraint 36's read on a CroCo machine."""

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        return calculate_hazelton_zhai_rebco_croco_temperature_margin(
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            dr_tf_hts_tape,
            dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total,
            tftmp,
        )
