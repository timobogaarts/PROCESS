"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit
#9).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.models.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,  # noqa: F401
    calculate_centering_force_max_mn,  # noqa: F401
    calculate_centering_force_min_mn,  # noqa: F401
    calculate_max_force_density,  # noqa: F401
    calculate_max_force_density_mnm,  # noqa: F401
    calculate_max_lateral_force_density,  # noqa: F401
    calculate_max_radial_force_density,  # noqa: F401
    calculate_maximum_stress,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.mass import (
    calculate_coils_mass,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.quench import (
    calculate_quench_protection,  # noqa: F401
)
from functional_process.cottax.paths import (
    build,
    constraints,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.models.physics.superconductors import (
    bi2212,  # noqa: F401
    gl_nbti,  # noqa: F401
    gl_rebco,  # noqa: F401
    itersc,  # noqa: F401
    jcrit_nbti,  # noqa: F401
    jcrit_rebco,  # noqa: F401
    western_superconducting_nb3sn,  # noqa: F401
)
from functional_process.models.stellarator.coils.calculate import (
    calculate_bi2212_winding_pack_intersect_inputs,
    calculate_casing,
    calculate_coil_coil_toroidal_gap,  # noqa: F401
    calculate_coil_cross_sectional_area,
    calculate_coil_half_widths,
    calculate_coil_radial_thickness,
    calculate_coil_toroidal_thickness,
    calculate_coils_summary_variables,
    calculate_current,
    calculate_durham_nbti_winding_pack_intersect_inputs,
    calculate_horizontal_ports,
    calculate_inductance,  # noqa: F401
    calculate_len_tf_coil,
    calculate_plasma_facing_coil_area,
    calculate_stored_magnetic_energy,
    calculate_tfcryoarea,
    calculate_user_defined_nb3sn_winding_pack_intersect_inputs,
    calculate_vertical_ports,
    calculate_winding_pack_geometry,
    calculate_z_tf_inside_half,
    jcrit_bi2212,  # noqa: F401
    jcrit_croco_rebco,
    jcrit_durham_nbti,  # noqa: F401
    jcrit_durham_rebco,
    jcrit_iter_nb3sn,
    jcrit_old_lubell_nbti,
    jcrit_user_defined_nb3sn,  # noqa: F401
    jcrit_wst_nb3sn,
    select_coil_coil_toroidal_gap,
    st_coil,  # noqa: F401
    winding_pack_curves,  # noqa: F401
    winding_pack_intersect_inputs_curves,
    winding_pack_post_intersect,
    winding_pack_pre_intersect,  # noqa: F401
    winding_pack_total_size,  # noqa: F401
)
from functional_process.vocabulary import (
    SuperconductorModel,  # noqa: F401
)

# CoilToroidalThickness()
CoilToroidalThickness = tfcoil.function(
    calculate_coil_toroidal_thickness, "dx_tf_inboard_out_toroidal"
)


# CoilRadialThickness()
CoilRadialThickness = build.function(
    calculate_coil_radial_thickness,
    "dr_tf_inboard",
)


# CoilCrossSectionalArea()
CoilCrossSectionalArea = tfcoil.function(
    calculate_coil_cross_sectional_area,
    "a_tf_leg_outboard",
    "a_tf_coil_inboard_case",
)


# CoilHalfWidths()
CoilHalfWidths = tfcoil.function(calculate_coil_half_widths, "tfocrn", "tficrn")


# PlasmaFacingCoilArea()
PlasmaFacingCoilArea = tfcoil.function(
    calculate_plasma_facing_coil_area, "tfsai", "tfsao"
)


# CoilCoilToroidalGap()
CoilCoilToroidalGap = tfcoil.function(
    select_coil_coil_toroidal_gap,
    "toroidalgap",
)


# CoilsSummaryVariables()
CoilsSummaryVariables = tfcoil.function(
    calculate_coils_summary_variables,
    "a_tf_inboard_total",
    "c_tf_total",
    "j_tf_coil_full_area",
    "r_b_tf_inboard_peak_symmetric",
    reads={
        "coilcurrent": stellarator.place("coilcurrent"),
        "awp_rad": V.tfcoil.dr_tf_wp_with_insulation,
    },
)


# StoredMagneticEnergy()
StoredMagneticEnergy = tfcoil.function(
    calculate_stored_magnetic_energy,
    "e_tf_magnetic_stored_total_gj",
)


# WindingPackGeometry()
WindingPackGeometry = tfcoil.function(
    calculate_winding_pack_geometry, "a_tf_turn_cable_space_no_void", "a_tf_turn_steel"
)


# `coilcurrent` has no PROCESS storage location -- it is a local in `st_coil`,
# threaded manually into `winding_pack_total_size` and
# `calculate_coils_summary_variables`.
CoilCurrent = stellarator.function(
    calculate_current,
    "coilcurrent",
    "f_st_i_total",
)


# The WindingPackIntersectInputs family:
# The family that owns the *pre*-`intersect` half of `winding_pack_total_size`: the
# sampled `(wp_width_r, lhs, rhs)` curves `coils.py`'s `Intersect`
# (`ImplicitFunction`/`RootFind`) needs as its own `From`s, and the starting guess it
# is driven from.


# `(sample_lower_divisor, guess_divisor)`: `_MATERIAL_SAMPLING`'s row for a material --
# the ordinary pair, and the one PROCESS treats differently (CroCo REBCO).
_ORDINARY = (40.0, 10.0)
_CROCO_REBCO = (150.0, 20.0)

# Every occupant reads these, at these places; the Bi2212, user-defined Nb3Sn and
# Durham NbTi occupants read two more, from `.tfcoil`.
_INTERSECT_READS = {
    "r_coil_major": stellarator.place("r_coil_major"),
    "r_coil_minor": stellarator.place("r_coil_minor"),
    "coilcurrent": stellarator.place("coilcurrent"),
    "n_tf_coils": tfcoil.place("n_tf_coils"),
    "stella_config_a1": stellarator_config.place("stella_config_a1"),
    "stella_config_a2": stellarator_config.place("stella_config_a2"),
    "stella_config_wp_ratio": stellarator_config.place("stella_config_wp_ratio"),
    "tftmp": tfcoil.place("tftmp"),
    "tmargmin": tfcoil.place("tmargmin"),
    "f_a_tf_turn_cable_copper": tfcoil.place("f_a_tf_turn_cable_copper"),
    "f_a_tf_turn_cable_space_extra_void": tfcoil.place(
        "f_a_tf_turn_cable_space_extra_void"
    ),
    "f_j_tf_wp_critical_max": constraints.place("f_j_tf_wp_critical_max"),
    "a_tf_turn_cable_space_no_void": tfcoil.place("a_tf_turn_cable_space_no_void"),
    "dx_tf_turn_general": tfcoil.place("dx_tf_turn_general"),
}
_INTERSECT_OUTPUTS = ("wp_width_r", "lhs", "rhs", "wp_width_r_min_guess")


def _iter_nb3sn_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    return winding_pack_intersect_inputs_curves(
        jcrit_iter_nb3sn,
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )


# `i_tf_sc_mat == ITER_NB3SN` (1) -- PROCESS's own default and this run's value.
IterNb3snWindingPackIntersectInputs = stellarator.function(
    _iter_nb3sn_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _bi2212_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    fhts,
    j_tf_wp,
):
    return calculate_bi2212_winding_pack_intersect_inputs(
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        fhts,
        j_tf_wp,
    )


# `i_tf_sc_mat == BI2212` (2).
Bi2212WindingPackIntersectInputs = stellarator.function(
    _bi2212_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _old_lubell_nbti_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    return winding_pack_intersect_inputs_curves(
        jcrit_old_lubell_nbti,
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )


# `i_tf_sc_mat == OLD_LUBELL_NBTI` (3). Literals only, no material read.
OldLubellNbtiWindingPackIntersectInputs = stellarator.function(
    _old_lubell_nbti_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _user_defined_nb3sn_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    bcritsc,
    tcritsc,
):
    return calculate_user_defined_nb3sn_winding_pack_intersect_inputs(
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        bcritsc,
        tcritsc,
    )


# `i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- the only occupant reading
# `.tfcoil.bcritsc`/`.tfcoil.tcritsc`, which are exactly what "user-defined" means
# here.
UserDefinedNb3snWindingPackIntersectInputs = stellarator.function(
    _user_defined_nb3sn_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _wst_nb3sn_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    return winding_pack_intersect_inputs_curves(
        jcrit_wst_nb3sn,
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )


# `i_tf_sc_mat == WST_NB3SN` (5). Literals only, no material read.
WstNb3snWindingPackIntersectInputs = stellarator.function(
    _wst_nb3sn_winding_pack_intersect_inputs, *_INTERSECT_OUTPUTS, reads=_INTERSECT_READS
)


def _croco_rebco_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    return winding_pack_intersect_inputs_curves(
        jcrit_croco_rebco,
        *_CROCO_REBCO,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )


# `i_tf_sc_mat == CROCO_REBCO` (6) -- the one occupant with different sampling.
CrocoRebcoWindingPackIntersectInputs = stellarator.function(
    _croco_rebco_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _durham_nbti_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    b_crit_upper_nbti,
    t_crit_nbti,
):
    return calculate_durham_nbti_winding_pack_intersect_inputs(
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        b_crit_upper_nbti,
        t_crit_nbti,
    )


# `i_tf_sc_mat == DURHAM_NBTI` (7) -- the only occupant reading
# `.tfcoil.b_crit_upper_nbti`/`.tfcoil.t_crit_nbti`.
DurhamNbtiWindingPackIntersectInputs = stellarator.function(
    _durham_nbti_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


def _durham_rebco_winding_pack_intersect_inputs(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    return winding_pack_intersect_inputs_curves(
        jcrit_durham_rebco,
        *_ORDINARY,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )


# `i_tf_sc_mat == DURHAM_REBCO` (8).
DurhamRebcoWindingPackIntersectInputs = stellarator.function(
    _durham_rebco_winding_pack_intersect_inputs,
    *_INTERSECT_OUTPUTS,
    reads=_INTERSECT_READS,
)


# cottax node: the *post*-`intersect` half of `winding_pack_total_size` --
# everything downstream of the resolved crossing point.
#
# `winding_pack_post_intersect`'s return tuple is already in this exact order (see its
# own docstring), so the declaration below delegates directly.
WindingPackTotalSizePost = tfcoil.function(
    winding_pack_post_intersect,
    "b_tf_inboard_peak_symmetric",
    "dx_tf_wp_primary_toroidal",
    "dx_tf_wp_secondary_toroidal",
    "dr_tf_wp_with_insulation",
    "j_tf_wp",
    "n_tf_coil_turns",
    "c_tf_turn",
    "a_tf_wp_conductor",
    "a_tf_wp_extra_void",
    "a_tf_coil_wp_turn_insulation",
    "a_tf_wp_steel",
    "a_tf_wp_no_insulation",
    "a_tf_wp_with_insulation",
    reads={
        "wp_width_r_min": stellarator.place("wp_width_r_min"),
        "coilcurrent": stellarator.place("coilcurrent"),
    },
)


# CoilCasing()
CoilCasing = tfcoil.function(
    calculate_casing, "dr_tf_plasma_case", "dx_tf_side_case_min"
)


# VerticalPorts()
VerticalPorts = stellarator.function(
    calculate_vertical_ports,
    "vporttmax",
    "vportpmax",
    "vportamax",
)


# HorizontalPorts()
HorizontalPorts = stellarator.function(
    calculate_horizontal_ports,
    "hporttmax",
    "hportpmax",
    "hportamax",
)


# cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`.
ZTfInsideHalf = build.function(
    calculate_z_tf_inside_half,
    "z_tf_inside_half",
)


# cottax node: `calculate_len_tf_coil`, owning `.tfcoil.len_tf_coil`.
LenTfCoil = tfcoil.function(
    calculate_len_tf_coil,
    "len_tf_coil",
)


# cottax node: `calculate_tfcryoarea`, owning `.tfcoil.tfcryoarea`.
TfCryoArea = tfcoil.function(
    calculate_tfcryoarea,
    "tfcryoarea",
)
