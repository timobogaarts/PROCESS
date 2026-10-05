"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CICCSuperconductingTFCoil` and the `SuperconductingTFCoil` layer above it.
"""

from cottax.relational_pytree import V

from functional_process.cottax.paths import (
    superconducting_tfcoil,
    tfcoil,
)
from functional_process.models.tfcoil.superconducting import (
    calculate_a_tf_turn,
    calculate_old_lubell_nbti_temperature_margin,
    calculate_temperature_margin_with_strain,
    calculate_vv_stress_on_quench,
    cicc_averaged_turn_geometry_from_current_per_turn,
    cicc_integer_turn_geometry,
    cicc_superconductor_properties_durham_nbti,
    cicc_superconductor_properties_itersc,
    cicc_superconductor_properties_lubell_nbti,
    cicc_superconductor_properties_wst_nb3sn,
    dx_tf_side_case_double_rectangular,
    dx_tf_side_case_rectangular,
    dx_tf_side_case_trapezoidal,
    peak_b_tf_inboard_with_ripple_flat,
    peak_b_tf_inboard_with_ripple_kovari,
    solve_current_sharing_temperature,  # noqa: F401 -- re-exported for pfcoil/superconductor.py
    superconducting_tf_coil_areas_and_masses_conventional,
    superconducting_tf_coil_areas_and_masses_spherical_tokamak,
    superconducting_tf_wp_geometry_double_rectangular,
    superconducting_tf_wp_geometry_rectangular,
    superconducting_tf_wp_geometry_trapezoidal,
    temperature_margin_itersc,
    temperature_margin_lubell_nbti,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
    temperature_margin_wst_nb3sn,
    tf_case_areas_circular_front,
    tf_case_areas_straight_front,
    tf_cicc_inboard_areas_and_fractions,
    tf_wp_currents,
    vv_stress_on_quench,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
    vv_stress_quench_from_build,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
)

_RIPPLE_FIT_COEFFICIENTS = {
    16: (0.28101, 1.8481, -0.88159, 0.93834),
    18: (0.29153, 1.81600, -0.84178, 0.90426),
    20: (0.29853, 1.82130, -0.85031, 0.89808),
}
"""M."""


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------

_WP_GEOMETRY_OUTPUTS = (
    "r_tf_wp_inboard_inner",
    "r_tf_wp_inboard_outer",
    "r_tf_wp_inboard_centre",
    "dx_tf_wp_toroidal_min",
    "dr_tf_wp_no_insulation",
    "dx_tf_wp_primary_toroidal",
    "dx_tf_wp_secondary_toroidal",
    "dx_tf_wp_toroidal_average",
    "a_tf_wp_with_insulation",
    "a_tf_wp_no_insulation",
    "a_tf_wp_ground_insulation",
)
"""Documentation only -- the declaration order every `SuperconductingTfWpGeometry`
occupant repeats.
"""


# The SuperconductingTfWpGeometry family:
# The family that owns the inboard winding-pack geometry.


# `i_tf_wp_geom == 0` (rectangular).
SuperconductingTfWpGeometryRectangular = superconducting_tfcoil.function(
    superconducting_tf_wp_geometry_rectangular,
    "r_tf_wp_inboard_inner",
    "r_tf_wp_inboard_outer",
    "r_tf_wp_inboard_centre",
    "dx_tf_wp_toroidal_min",
    "dr_tf_wp_no_insulation",
    V.tfcoil.dx_tf_wp_primary_toroidal,
    V.tfcoil.dx_tf_wp_secondary_toroidal,
    "dx_tf_wp_toroidal_average",
    "a_tf_wp_with_insulation",
    "a_tf_wp_no_insulation",
    "a_tf_wp_ground_insulation",
)


# `i_tf_wp_geom == 1` (double rectangular) -- `large_tokamak_eval`'s arm.
SuperconductingTfWpGeometryDoubleRectangular = superconducting_tfcoil.function(
    superconducting_tf_wp_geometry_double_rectangular,
    "r_tf_wp_inboard_inner",
    "r_tf_wp_inboard_outer",
    "r_tf_wp_inboard_centre",
    "dx_tf_wp_toroidal_min",
    "dr_tf_wp_no_insulation",
    V.tfcoil.dx_tf_wp_primary_toroidal,
    V.tfcoil.dx_tf_wp_secondary_toroidal,
    "dx_tf_wp_toroidal_average",
    "a_tf_wp_with_insulation",
    "a_tf_wp_no_insulation",
    "a_tf_wp_ground_insulation",
)


# `i_tf_wp_geom == 2` (trapezoidal).
SuperconductingTfWpGeometryTrapezoidal = superconducting_tfcoil.function(
    superconducting_tf_wp_geometry_trapezoidal,
    "r_tf_wp_inboard_inner",
    "r_tf_wp_inboard_outer",
    "r_tf_wp_inboard_centre",
    "dx_tf_wp_toroidal_min",
    "dr_tf_wp_no_insulation",
    V.tfcoil.dx_tf_wp_primary_toroidal,
    V.tfcoil.dx_tf_wp_secondary_toroidal,
    "dx_tf_wp_toroidal_average",
    "a_tf_wp_with_insulation",
    "a_tf_wp_no_insulation",
    "a_tf_wp_ground_insulation",
)


# The TfCaseAreas family:
# The family that owns the four TF case areas. `i_tf_case_geom` decides it.


# `i_tf_case_geom == 0` (circular front case) -- the reference arm.
TfCaseAreasCircularFront = tfcoil.function(
    tf_case_areas_circular_front,
    "a_tf_coil_inboard_case",
    "a_tf_coil_outboard_case",
    V.superconducting_tfcoil.a_tf_plasma_case,
    V.superconducting_tfcoil.a_tf_coil_nose_case,
    reads={
        "a_tf_wp_with_insulation": superconducting_tfcoil.place(
            "a_tf_wp_with_insulation"
        )
    },
)


# `i_tf_case_geom == 1` (straight front case). Reads `dr_tf_plasma_case`.
TfCaseAreasStraightFront = tfcoil.function(
    tf_case_areas_straight_front,
    "a_tf_coil_inboard_case",
    "a_tf_coil_outboard_case",
    V.superconducting_tfcoil.a_tf_plasma_case,
    V.superconducting_tfcoil.a_tf_coil_nose_case,
    reads={
        "a_tf_wp_with_insulation": superconducting_tfcoil.place(
            "a_tf_wp_with_insulation"
        )
    },
)


# The DxTfSideCase family:
# The family that owns the sidewall case thicknesses. `i_tf_wp_geom` decides it.


# `i_tf_wp_geom == 0`.
DxTfSideCaseRectangular = superconducting_tfcoil.function(
    dx_tf_side_case_rectangular,
    "dx_tf_side_case_average",
    V.tfcoil.dx_tf_side_case_peak,
)


# `i_tf_wp_geom == 1` -- the reference arm.
DxTfSideCaseDoubleRectangular = superconducting_tfcoil.function(
    dx_tf_side_case_double_rectangular,
    "dx_tf_side_case_average",
    V.tfcoil.dx_tf_side_case_peak,
)


# `i_tf_wp_geom == 2`: constant thickness, one read.
DxTfSideCaseTrapezoidal = superconducting_tfcoil.function(
    dx_tf_side_case_trapezoidal,
    "dx_tf_side_case_average",
    V.tfcoil.dx_tf_side_case_peak,
)


# cottax node: `tf_wp_currents`.
TfWpCurrents = tfcoil.function(
    tf_wp_currents,
    "j_tf_wp",
    reads={
        "a_tf_wp_no_insulation": superconducting_tfcoil.place("a_tf_wp_no_insulation"),
    },
)


# The PeakBTfInboardWithRipple family:
# The family that owns `.tfcoil.b_tf_inboard_peak_with_ripple`.


# The fitted coil counts share one body; the fit's `coefficients` differ.
def _peak_b_tf_inboard_with_ripple_kovari(
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
    coefficients,
):
    return peak_b_tf_inboard_with_ripple_kovari(
        n_tf_coils=n_tf_coils,
        dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
        dr_tf_wp_no_insulation=dr_tf_wp_no_insulation,
        r_tf_wp_inboard_centre=r_tf_wp_inboard_centre,
        b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
        coefficients=coefficients,
    )


def _peak_b_tf_inboard_with_ripple_16_coils(
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
):
    return _peak_b_tf_inboard_with_ripple_kovari(
        n_tf_coils,
        dx_tf_wp_primary_toroidal,
        dr_tf_wp_no_insulation,
        r_tf_wp_inboard_centre,
        b_tf_inboard_peak_symmetric,
        _RIPPLE_FIT_COEFFICIENTS[16],
    )


def _peak_b_tf_inboard_with_ripple_18_coils(
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
):
    return _peak_b_tf_inboard_with_ripple_kovari(
        n_tf_coils,
        dx_tf_wp_primary_toroidal,
        dr_tf_wp_no_insulation,
        r_tf_wp_inboard_centre,
        b_tf_inboard_peak_symmetric,
        _RIPPLE_FIT_COEFFICIENTS[18],
    )


def _peak_b_tf_inboard_with_ripple_20_coils(
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
):
    return _peak_b_tf_inboard_with_ripple_kovari(
        n_tf_coils,
        dx_tf_wp_primary_toroidal,
        dr_tf_wp_no_insulation,
        r_tf_wp_inboard_centre,
        b_tf_inboard_peak_symmetric,
        _RIPPLE_FIT_COEFFICIENTS[20],
    )


_PEAK_B_TF_INBOARD_WITH_RIPPLE_OUTPUTS = (
    "tf_fit_t",
    "tf_fit_z",
    "f_b_tf_inboard_peak_ripple_symmetric",
    V.tfcoil.b_tf_inboard_peak_with_ripple,
)

# `round(n_tf_coils) == 16` -- `large_tokamak_eval.IN.DAT:377` sets exactly 16.
PeakBTfInboardWithRipple16Coils = superconducting_tfcoil.function(
    _peak_b_tf_inboard_with_ripple_16_coils,
    *_PEAK_B_TF_INBOARD_WITH_RIPPLE_OUTPUTS,
)


# `round(n_tf_coils) == 18`.
PeakBTfInboardWithRipple18Coils = superconducting_tfcoil.function(
    _peak_b_tf_inboard_with_ripple_18_coils,
    *_PEAK_B_TF_INBOARD_WITH_RIPPLE_OUTPUTS,
)


# `round(n_tf_coils) == 20`.
PeakBTfInboardWithRipple20Coils = superconducting_tfcoil.function(
    _peak_b_tf_inboard_with_ripple_20_coils,
    *_PEAK_B_TF_INBOARD_WITH_RIPPLE_OUTPUTS,
)


# Any other coil count: `1.09 * b_tf_inboard_peak_symmetric`, one read, one output.
PeakBTfInboardWithRippleFlatAllowance = tfcoil.function(
    peak_b_tf_inboard_with_ripple_flat, "b_tf_inboard_peak_with_ripple"
)


# The CiccTurnGeometry family:
# The family that owns the CICC winding-pack turn geometry.


# The CiccAveragedTurnGeometry family:
# The averaged (`i_tf_turns_integer == 0`) sub-family.


# Both input flags `False` -- PROCESS's default and `large_tokamak_eval`'s arm.
CiccAveragedTurnGeometryFromCurrentPerTurn = tfcoil.function(
    cicc_averaged_turn_geometry_from_current_per_turn,
    "a_tf_turn_cable_space_no_void",
    "a_tf_turn_steel",
    "a_tf_turn_insulation",
    "n_tf_coil_turns",
    "dx_tf_turn_general",
    V.superconducting_tfcoil.dr_tf_turn,
    V.superconducting_tfcoil.dx_tf_turn,
    "dx_tf_turn_conduit_full_average",
    V.superconducting_tfcoil.radius_tf_turn_cable_space_corners,
    V.superconducting_tfcoil.dx_tf_turn_cable_space_average,
    V.superconducting_tfcoil.a_tf_turn_cable_space_effective,
    V.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling,
    reads={
        "a_tf_wp_no_insulation": superconducting_tfcoil.place("a_tf_wp_no_insulation"),
    },
)


# `i_tf_turns_integer == 1`.
#
# Rectangular turns on a fixed layers x pancakes grid.
CiccIntegerTurnGeometry = superconducting_tfcoil.function(
    cicc_integer_turn_geometry,
    "radius_tf_turn_cable_space_corners",
    "dr_tf_turn",
    "dx_tf_turn",
    V.tfcoil.a_tf_turn_cable_space_no_void,
    V.tfcoil.a_tf_turn_steel,
    V.tfcoil.a_tf_turn_insulation,
    V.tfcoil.c_tf_turn,
    V.tfcoil.n_tf_coil_turns,
    "dr_tf_turn_conduit_full",
    "dx_tf_turn_conduit_full_toroidal",
    V.tfcoil.dx_tf_turn_conduit_full_average,
    "dr_tf_turn_cable_space",
    "dx_tf_turn_cable_space",
    "dx_tf_turn_cable_space_average",
    "a_tf_turn_cable_space_effective",
    "f_a_tf_turn_cable_space_cooling",
    V.tfcoil.dx_tf_turn_general,
)


# cottax node: `tf_cicc_inboard_areas_and_fractions`.
CiccInboardAreasAndFractions = tfcoil.function(
    tf_cicc_inboard_areas_and_fractions,
    "a_tf_wp_coolant_channels",
    "a_tf_wp_conductor",
    "a_tf_wp_extra_void",
    "a_tf_coil_wp_turn_insulation",
    "a_tf_wp_steel",
    V.superconducting_tfcoil.a_tf_coil_inboard_steel,
    V.superconducting_tfcoil.f_a_tf_coil_inboard_steel,
    V.superconducting_tfcoil.a_tf_coil_inboard_insulation,
    V.superconducting_tfcoil.f_a_tf_coil_inboard_insulation,
)


# cottax node: `run`'s inline `.tfcoil.a_tf_turn` (`superconducting.py:2700`).
TfTurnArea = tfcoil.function(calculate_a_tf_turn, "a_tf_turn")


# The family that owns the superconducting TF coil masses: one occupant per `itart`
# arm. The superconductor's density is `.tfcoil.den_tf_sc_material`, the entry of
# `.tfcoil.dcond` at `.tfcoil.i_tf_sc_mat`, derived from a configuration's values
# (`cottax.input.native`), so the material index reaches no node.


def _superconducting_tf_coil_areas_and_masses_conventional(
    len_tf_coil,
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    den_tf_wp_turn_insulation,
    z_tf_inside_half,
    dr_tf_inboard,
    den_tf_coil_case,
    a_tf_coil_inboard_case,
    a_tf_coil_outboard_case,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    return superconducting_tf_coil_areas_and_masses_conventional(
        len_tf_coil=len_tf_coil,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
        z_tf_inside_half=z_tf_inside_half,
        dr_tf_inboard=dr_tf_inboard,
        den_tf_coil_case=den_tf_coil_case,
        a_tf_coil_inboard_case=a_tf_coil_inboard_case,
        a_tf_coil_outboard_case=a_tf_coil_outboard_case,
        n_tf_coil_turns=n_tf_coil_turns,
        a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels=a_tf_wp_coolant_channels,
        den_tf_sc_material=den_tf_sc_material,
        a_tf_turn_steel=a_tf_turn_steel,
        den_steel=den_steel,
        a_tf_coil_wp_turn_insulation=a_tf_coil_wp_turn_insulation,
        n_tf_coils=n_tf_coils,
    )


def _superconducting_tf_coil_areas_and_masses_spherical_tokamak(
    len_tf_coil,
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    den_tf_wp_turn_insulation,
    z_tf_inside_half,
    dr_tf_inboard,
    den_tf_coil_case,
    a_tf_coil_inboard_case,
    a_tf_coil_outboard_case,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    return superconducting_tf_coil_areas_and_masses_spherical_tokamak(
        len_tf_coil=len_tf_coil,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
        z_tf_inside_half=z_tf_inside_half,
        dr_tf_inboard=dr_tf_inboard,
        den_tf_coil_case=den_tf_coil_case,
        a_tf_coil_inboard_case=a_tf_coil_inboard_case,
        a_tf_coil_outboard_case=a_tf_coil_outboard_case,
        n_tf_coil_turns=n_tf_coil_turns,
        a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels=a_tf_wp_coolant_channels,
        den_tf_sc_material=den_tf_sc_material,
        a_tf_turn_steel=a_tf_turn_steel,
        den_steel=den_steel,
        a_tf_coil_wp_turn_insulation=a_tf_coil_wp_turn_insulation,
        n_tf_coils=n_tf_coils,
    )


_TF_COIL_MASSES = {
    "Conventional": (
        _superconducting_tf_coil_areas_and_masses_conventional,
        "m_tf_coil_wp_insulation",
        "cplen",
        "m_tf_coil_case",
        "m_tf_coil_superconductor",
        "m_tf_coil_copper",
        "m_tf_wp_steel_conduit",
        "m_tf_coil_wp_turn_insulation",
        "m_tf_coil_conductor",
        "m_tf_coil",
        "m_tf_coils_total",
    ),
    "SphericalTokamak": (
        _superconducting_tf_coil_areas_and_masses_spherical_tokamak,
        "m_tf_coil_wp_insulation",
        "cplen",
        "m_tf_coil_case",
        "m_tf_coil_superconductor",
        "m_tf_coil_copper",
        "m_tf_wp_steel_conduit",
        "m_tf_coil_wp_turn_insulation",
        "m_tf_coil_conductor",
        "m_tf_coil",
        "m_tf_coils_total",
        "whtcp",
        "whttflgs",
    ),
}
"""Per `itart` arm: the body, then its outputs."""


def _tf_coil_masses(arm):
    """The occupant for this arm."""
    body, *outputs = _TF_COIL_MASSES[arm]
    return tfcoil.function(
        body,
        *outputs,
        reads={
            "a_tf_wp_with_insulation": superconducting_tfcoil.place(
                "a_tf_wp_with_insulation"
            ),
            "a_tf_wp_no_insulation": superconducting_tfcoil.place(
                "a_tf_wp_no_insulation"
            ),
        },
    )


# `itart == 0`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO]`.
SuperconductingTfCoilAreasAndMassesConventional = _tf_coil_masses("Conventional")


# `itart == 1`: `SC_TF_MASSES[SPHERICAL_TOKAMAK]`.
SuperconductingTfCoilAreasAndMassesSphericalTokamak = _tf_coil_masses("SphericalTokamak")


# cottax node: `.superconducting_tfcoil.vv_stress_quench`, constraint 65's read.
VvStressOnQuench = superconducting_tfcoil.function(
    calculate_vv_stress_on_quench,
    "vv_stress_quench",
)


# The CiccSuperconductorProperties family:
# The family that owns the CICC critical-current chain -- constraint 33's read.


# `i_tf_sc_mat == 1` -- `large_tokamak_eval.IN.DAT:374`'s own arm.
def _iter_nb3sn_cicc_superconductor_properties(
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak_with_ripple,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    str_wp,
    tftmp,
):
    return cicc_superconductor_properties_itersc(
        a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
        a_tf_turn=a_tf_turn,
        b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        c_tf_turn=c_tf_turn,
        strain=str_wp,
        temp_tf_coolant_peak_field=tftmp,
        b_c20max=32.97,
        temp_c0max=16.06,
    )


IterNb3snCiccSuperconductorProperties = tfcoil.function(
    _iter_nb3sn_cicc_superconductor_properties,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
)


# `i_tf_sc_mat == 4` -- the ITER fit with `(bcritsc, tcritsc)` read from input.
def _user_defined_nb3sn_cicc_superconductor_properties(
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    strain,
    temp_tf_coolant_peak_field,
    b_c20max,
    temp_c0max,
):
    return cicc_superconductor_properties_itersc(
        a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
        a_tf_turn=a_tf_turn,
        b_tf_inboard_peak=b_tf_inboard_peak,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        c_tf_turn=c_tf_turn,
        strain=strain,
        temp_tf_coolant_peak_field=temp_tf_coolant_peak_field,
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )


UserDefinedNb3snCiccSuperconductorProperties = tfcoil.function(
    _user_defined_nb3sn_cicc_superconductor_properties,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "strain": V.tfcoil.str_wp,
        "temp_tf_coolant_peak_field": V.tfcoil.tftmp,
        "b_c20max": V.tfcoil.bcritsc,
        "temp_c0max": V.tfcoil.tcritsc,
    },
)


# `i_tf_sc_mat == 5` -- `low_aspect_ratio_DEMO.IN.DAT:910`'s arm.
WstNb3snCiccSuperconductorProperties = tfcoil.function(
    cicc_superconductor_properties_wst_nb3sn,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "strain": V.tfcoil.str_wp,
        "temp_tf_coolant_peak_field": V.tfcoil.tftmp,
    },
)


# `i_tf_sc_mat == 3` -- and the arm that reads **no strain**.
def _old_lubell_nbti_cicc_superconductor_properties(
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    temp_tf_coolant_peak_field,
):
    return cicc_superconductor_properties_lubell_nbti(
        a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
        a_tf_turn=a_tf_turn,
        b_tf_inboard_peak=b_tf_inboard_peak,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        c_tf_turn=c_tf_turn,
        temp_tf_coolant_peak_field=temp_tf_coolant_peak_field,
    )


OldLubellNbtiCiccSuperconductorProperties = tfcoil.function(
    _old_lubell_nbti_cicc_superconductor_properties,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "temp_tf_coolant_peak_field": V.tfcoil.tftmp,
    },
)


# `i_tf_sc_mat == 7` -- Durham Ginzburg-Landau NbTi.
def _durham_nbti_cicc_superconductor_properties(
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    strain,
    temp_tf_coolant_peak_field,
    b_crit_upper_nbti,
    t_crit_nbti,
):
    return cicc_superconductor_properties_durham_nbti(
        a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
        a_tf_turn=a_tf_turn,
        b_tf_inboard_peak=b_tf_inboard_peak,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        c_tf_turn=c_tf_turn,
        strain=strain,
        temp_tf_coolant_peak_field=temp_tf_coolant_peak_field,
        b_crit_upper_nbti=b_crit_upper_nbti,
        t_crit_nbti=t_crit_nbti,
    )


DurhamNbtiCiccSuperconductorProperties = tfcoil.function(
    _durham_nbti_cicc_superconductor_properties,
    "j_tf_wp_critical",
    "j_crit_str_tf",
    V.superconducting_tfcoil.f_c_tf_turn_operating_critical,
    V.superconducting_tfcoil.j_tf_coil_turn,
    V.superconducting_tfcoil.j_tf_superconductor,
    V.superconducting_tfcoil.c_tf_turn_cables_critical,
    V.superconducting_tfcoil.j_tf_superconductor_critical,
    V.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain,
    V.superconducting_tfcoil.temp_tf_superconductor_critical_zero_field_strain,
    reads={
        "b_tf_inboard_peak": V.tfcoil.b_tf_inboard_peak_with_ripple,
        "strain": V.tfcoil.str_wp,
        "temp_tf_coolant_peak_field": V.tfcoil.tftmp,
    },
)


# The TfSuperconductorTemperatureMargin family:
# The family that owns the TF temperature margin -- constraint 36's read.


# The strained arms share `calculate_temperature_margin_with_strain`; the fit differs.
def _temperature_margin_with_strain_itersc(
    j_tf_superconductor,
    b_tf_inboard_peak_with_ripple,
    str_wp,
    b_tf_superconductor_critical_zero_temp_strain,
    temp_tf_superconductor_critical_zero_field_strain,
    tftmp,
):
    return calculate_temperature_margin_with_strain(
        temperature_margin_itersc,
        j_tf_superconductor,
        b_tf_inboard_peak_with_ripple,
        str_wp,
        b_tf_superconductor_critical_zero_temp_strain,
        temp_tf_superconductor_critical_zero_field_strain,
        tftmp,
    )


def _temperature_margin_with_strain_wst_nb3sn(
    j_tf_superconductor,
    b_tf_inboard_peak_with_ripple,
    str_wp,
    b_tf_superconductor_critical_zero_temp_strain,
    temp_tf_superconductor_critical_zero_field_strain,
    tftmp,
):
    return calculate_temperature_margin_with_strain(
        temperature_margin_wst_nb3sn,
        j_tf_superconductor,
        b_tf_inboard_peak_with_ripple,
        str_wp,
        b_tf_superconductor_critical_zero_temp_strain,
        temp_tf_superconductor_critical_zero_field_strain,
        tftmp,
    )


_TEMPERATURE_MARGIN_WITH_STRAIN_READS = {
    "b_tf_superconductor_critical_zero_temp_strain": superconducting_tfcoil.place(
        "b_tf_superconductor_critical_zero_temp_strain"
    ),
    "temp_tf_superconductor_critical_zero_field_strain": superconducting_tfcoil.place(
        "temp_tf_superconductor_critical_zero_field_strain"
    ),
}


# `i_tf_sc_mat == 1` *(live)*. `process/models/superconductors.py:1259`.
IterNb3snTfSuperconductorTemperatureMargin = tfcoil.function(
    _temperature_margin_with_strain_itersc,
    "temp_tf_superconductor_margin",
    "temp_margin",
    reads=_TEMPERATURE_MARGIN_WITH_STRAIN_READS,
)


# `i_tf_sc_mat == 4`.
UserDefinedNb3snTfSuperconductorTemperatureMargin = tfcoil.function(
    _temperature_margin_with_strain_itersc,
    "temp_tf_superconductor_margin",
    "temp_margin",
    reads=_TEMPERATURE_MARGIN_WITH_STRAIN_READS,
)


# `i_tf_sc_mat == 5`. `process/models/superconductors.py:1263-1265`.
WstNb3snTfSuperconductorTemperatureMargin = tfcoil.function(
    _temperature_margin_with_strain_wst_nb3sn,
    "temp_tf_superconductor_margin",
    "temp_margin",
    reads=_TEMPERATURE_MARGIN_WITH_STRAIN_READS,
)


# `i_tf_sc_mat == 3` -- one read fewer, and one literal more.
OldLubellNbtiTfSuperconductorTemperatureMargin = tfcoil.function(
    calculate_old_lubell_nbti_temperature_margin,
    "temp_tf_superconductor_margin",
    "temp_margin",
)
