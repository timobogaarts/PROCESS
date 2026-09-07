"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CICCSuperconductingTFCoil` and the `SuperconductingTFCoil` layer above it.
"""

from abc import abstractmethod

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    divertor,
    fwbs,
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


class SuperconductingTfWpGeometry(ExplicitFunction):
    """The family that owns the inboard winding-pack geometry."""


class SuperconductingTfWpGeometryRectangular(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 0` (rectangular)."""

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_rectangular(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class SuperconductingTfWpGeometryDoubleRectangular(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 1` (double rectangular) -- `large_tokamak_eval`'s arm."""

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_double_rectangular(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class SuperconductingTfWpGeometryTrapezoidal(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 2` (trapezoidal)."""

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_trapezoidal(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class TfCaseAreas(ExplicitFunction):
    """The family that owns the four TF case areas. `i_tf_case_geom` decides it."""


class TfCaseAreasCircularFront(TfCaseAreas):
    """`i_tf_case_geom == 0` (circular front case) -- the reference arm."""

    a_tf_coil_inboard_case = OutputInto(tfcoil)
    a_tf_coil_outboard_case = OutputInto(tfcoil)
    a_tf_plasma_case = OutputInto(superconducting_tfcoil)
    a_tf_coil_nose_case = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_inboard_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_leg_outboard=From(tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        r_tf_inboard_out=From(build),
        tan_theta_coil=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_inboard_in=From(build),
    ):
        return tf_case_areas_circular_front(
            a_tf_inboard_total=a_tf_inboard_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_leg_outboard=a_tf_leg_outboard,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            r_tf_inboard_out=r_tf_inboard_out,
            tan_theta_coil=tan_theta_coil,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_inboard_in=r_tf_inboard_in,
        )


class TfCaseAreasStraightFront(TfCaseAreas):
    """`i_tf_case_geom == 1` (straight front case). Reads `dr_tf_plasma_case`."""

    a_tf_coil_inboard_case = OutputInto(tfcoil)
    a_tf_coil_outboard_case = OutputInto(tfcoil)
    a_tf_plasma_case = OutputInto(superconducting_tfcoil)
    a_tf_coil_nose_case = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_inboard_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_leg_outboard=From(tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_inboard_in=From(build),
    ):
        return tf_case_areas_straight_front(
            a_tf_inboard_total=a_tf_inboard_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_leg_outboard=a_tf_leg_outboard,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            tan_theta_coil=tan_theta_coil,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            dr_tf_plasma_case=dr_tf_plasma_case,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_inboard_in=r_tf_inboard_in,
        )


class DxTfSideCase(ExplicitFunction):
    """The family that owns the sidewall case thicknesses. `i_tf_wp_geom` decides it."""


class DxTfSideCaseRectangular(DxTfSideCase):
    """`i_tf_wp_geom == 0`."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_side_case_min=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return dx_tf_side_case_rectangular(
            dx_tf_side_case_min=dx_tf_side_case_min,
            tan_theta_coil=tan_theta_coil,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        )


class DxTfSideCaseDoubleRectangular(DxTfSideCase):
    """`i_tf_wp_geom == 1` -- the reference arm."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_side_case_min=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return dx_tf_side_case_double_rectangular(
            dx_tf_side_case_min=dx_tf_side_case_min,
            tan_theta_coil=tan_theta_coil,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        )


class DxTfSideCaseTrapezoidal(DxTfSideCase):
    """`i_tf_wp_geom == 2`: constant thickness, one read."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(self, dx_tf_side_case_min=From(tfcoil)):
        return dx_tf_side_case_trapezoidal(dx_tf_side_case_min=dx_tf_side_case_min)


class TfWpCurrents(ExplicitFunction):
    """cottax node: `tf_wp_currents`."""

    j_tf_wp = OutputInto(tfcoil)

    def __call__(
        self,
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
    ):
        return tf_wp_currents(
            c_tf_total=c_tf_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        )


class PeakBTfInboardWithRipple(ExplicitFunction):
    """The family that owns `.tfcoil.b_tf_inboard_peak_with_ripple`."""


class _PeakBTfInboardWithRippleKovari(PeakBTfInboardWithRipple):
    """Shared declaration for the three fitted coil counts; `coefficients` differs."""

    coefficients = ()

    tf_fit_t = OutputInto(superconducting_tfcoil)
    tf_fit_z = OutputInto(superconducting_tfcoil)
    f_b_tf_inboard_peak_ripple_symmetric = OutputInto(superconducting_tfcoil)
    b_tf_inboard_peak_with_ripple = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        dx_tf_wp_primary_toroidal=From(tfcoil),
        dr_tf_wp_no_insulation=From(superconducting_tfcoil),
        r_tf_wp_inboard_centre=From(superconducting_tfcoil),
        b_tf_inboard_peak_symmetric=From(tfcoil),
    ):
        return peak_b_tf_inboard_with_ripple_kovari(
            n_tf_coils=n_tf_coils,
            dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
            dr_tf_wp_no_insulation=dr_tf_wp_no_insulation,
            r_tf_wp_inboard_centre=r_tf_wp_inboard_centre,
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
            coefficients=self.coefficients,
        )


class PeakBTfInboardWithRipple16Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 16` -- `large_tokamak_eval.IN.DAT:377` sets exactly 16."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[16]


class PeakBTfInboardWithRipple18Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 18`."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[18]


class PeakBTfInboardWithRipple20Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 20`."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[20]


class PeakBTfInboardWithRippleFlatAllowance(PeakBTfInboardWithRipple):
    """Any other coil count: `1.09 * b_tf_inboard_peak_symmetric`, one read, one output.
    """

    b_tf_inboard_peak_with_ripple = OutputInto(tfcoil)

    def __call__(self, b_tf_inboard_peak_symmetric=From(tfcoil)):
        return peak_b_tf_inboard_with_ripple_flat(
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric
        )


class CiccTurnGeometry(ExplicitFunction):
    """The family that owns the CICC winding-pack turn geometry."""


class CiccAveragedTurnGeometry(CiccTurnGeometry):
    """The averaged (`i_tf_turns_integer == 0`) sub-family."""


class CiccAveragedTurnGeometryFromCurrentPerTurn(CiccAveragedTurnGeometry):
    """Both input flags `False` -- PROCESS's default and `large_tokamak_eval`'s arm."""

    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)
    a_tf_turn_insulation = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    radius_tf_turn_cable_space_corners = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        j_tf_wp=From(tfcoil),
        c_tf_turn=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        layer_ins=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
    ):
        return cicc_averaged_turn_geometry_from_current_per_turn(
            j_tf_wp=j_tf_wp,
            c_tf_turn=c_tf_turn,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            layer_ins=layer_ins,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        )


class CiccIntegerTurnGeometry(CiccTurnGeometry):
    """`i_tf_turns_integer == 1` -- rectangular turns on a fixed layers x pancakes grid.
    """

    radius_tf_turn_cable_space_corners = OutputInto(superconducting_tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)
    a_tf_turn_insulation = OutputInto(tfcoil)
    c_tf_turn = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dr_tf_turn_conduit_full = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_toroidal = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    dr_tf_turn_cable_space = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_wp_with_insulation=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        n_tf_wp_layers=From(tfcoil),
        dx_tf_wp_toroidal_min=From(superconducting_tfcoil),
        n_tf_wp_pancakes=From(tfcoil),
        c_tf_coil=From(superconducting_tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
    ):
        return cicc_integer_turn_geometry(
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            n_tf_wp_layers=n_tf_wp_layers,
            dx_tf_wp_toroidal_min=dx_tf_wp_toroidal_min,
            n_tf_wp_pancakes=n_tf_wp_pancakes,
            c_tf_coil=c_tf_coil,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        )


class CiccInboardAreasAndFractions(ExplicitFunction):
    """cottax node: `tf_cicc_inboard_areas_and_fractions`."""

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
        n_tf_coil_turns=From(tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        a_tf_turn_insulation=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_wp_ground_insulation=From(superconducting_tfcoil),
    ):
        return tf_cicc_inboard_areas_and_fractions(
            n_tf_coil_turns=n_tf_coil_turns,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_insulation=a_tf_turn_insulation,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_coils=n_tf_coils,
            a_tf_inboard_total=a_tf_inboard_total,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_wp_ground_insulation=a_tf_wp_ground_insulation,
        )


class TfTurnArea(ExplicitFunction):
    """cottax node: `run`'s inline `.tfcoil.a_tf_turn` (`superconducting.py:2700`)."""

    a_tf_turn = OutputInto(tfcoil)

    def __call__(
        self,
        c_tf_total=From(tfcoil),
        j_tf_wp=From(tfcoil),
        n_tf_coils=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
    ):
        return calculate_a_tf_turn(
            c_tf_total=c_tf_total,
            j_tf_wp=j_tf_wp,
            n_tf_coils=n_tf_coils,
            n_tf_coil_turns=n_tf_coil_turns,
        )


class SuperconductingTfCoilAreasAndMasses(ExplicitFunction):
    """The family that owns the superconducting TF coil masses."""

    @abstractmethod
    def _masses(
        self,
        *,
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
        """Run this `itart` arm, given the density its material occupant read."""
        raise NotImplementedError(
            f"{type(self).__name__} names an `i_tf_sc_mat` material but no `itart` arm; "
            "a usable occupant pairs one of each -- see `indat.SC_TF_MASSES`."
        )


class SuperconductingTfCoilAreasAndMassesConventional(
    SuperconductingTfCoilAreasAndMasses
):
    """The `itart == 0` (conventional aspect ratio) **arm** -- one of the two axes."""

    m_tf_coil_wp_insulation = OutputInto(tfcoil)
    cplen = OutputInto(tfcoil)
    m_tf_coil_case = OutputInto(tfcoil)
    m_tf_coil_superconductor = OutputInto(tfcoil)
    m_tf_coil_copper = OutputInto(tfcoil)
    m_tf_wp_steel_conduit = OutputInto(tfcoil)
    m_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    m_tf_coil_conductor = OutputInto(tfcoil)
    m_tf_coil = OutputInto(tfcoil)
    m_tf_coils_total = OutputInto(tfcoil)

    def _masses(
        self,
        *,
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
        """`superconducting_tf_coil_areas_and_masses_conventional`, the arm this class
        is.
        """
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


class SuperconductingTfCoilAreasAndMassesSphericalTokamak(
    SuperconductingTfCoilAreasAndMasses
):
    """The `itart == 1` (spherical tokamak) **arm** -- one of the two axes."""

    m_tf_coil_wp_insulation = OutputInto(tfcoil)
    cplen = OutputInto(tfcoil)
    m_tf_coil_case = OutputInto(tfcoil)
    m_tf_coil_superconductor = OutputInto(tfcoil)
    m_tf_coil_copper = OutputInto(tfcoil)
    m_tf_wp_steel_conduit = OutputInto(tfcoil)
    m_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    m_tf_coil_conductor = OutputInto(tfcoil)
    m_tf_coil = OutputInto(tfcoil)
    m_tf_coils_total = OutputInto(tfcoil)
    whtcp = OutputInto(tfcoil)
    whttflgs = OutputInto(tfcoil)

    def _masses(
        self,
        *,
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
        """`superconducting_tf_coil_areas_and_masses_spherical_tokamak`, the arm this
        class is.
        """
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


class IterNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- ITER Nb3Sn."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[0]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class Bi2212TfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == BI2212` (2) -- Bi-2212."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[1]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class OldLubellNbtiTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == OLD_LUBELL_NBTI` (3) -- old Lubell NbTi."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[2]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class UserDefinedNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- user-defined Nb3Sn."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[3]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class WstNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == WST_NB3SN` (5) -- WST Nb3Sn."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[4]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class CrocoRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- CroCo REBCO."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[5]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class DurhamNbtiTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == DURHAM_NBTI` (7) -- Durham Ginzburg-Landau NbTi."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[6]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class DurhamRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == DURHAM_REBCO` (8) -- Durham Ginzburg-Landau REBCO."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[7]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class HazeltonZhaiRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == HAZELTON_ZHAI_REBCO` (9) -- Hazelton-Zhai REBCO."""

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[8]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class IterNb3snSuperconductingTfCoilAreasAndMassesConventional(
    IterNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 1)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    ITER_NB3SN]`.
    """


class IterNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    IterNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 1)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK, ITER_NB3SN]`.
    """


class Bi2212SuperconductingTfCoilAreasAndMassesConventional(
    Bi2212TfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 2)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    BI2212]`.
    """


class Bi2212SuperconductingTfCoilAreasAndMassesSphericalTokamak(
    Bi2212TfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 2)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK, BI2212]`."""


class OldLubellNbtiSuperconductingTfCoilAreasAndMassesConventional(
    OldLubellNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 3)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    OLD_LUBELL_NBTI]`.
    """


class OldLubellNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    OldLubellNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 3)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK,
    OLD_LUBELL_NBTI]`.
    """


class UserDefinedNb3snSuperconductingTfCoilAreasAndMassesConventional(
    UserDefinedNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 4)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    USER_DEFINED_NB3SN]`.
    """


class UserDefinedNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    UserDefinedNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 4)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK,
    USER_DEFINED_NB3SN]`.
    """


class WstNb3snSuperconductingTfCoilAreasAndMassesConventional(
    WstNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 5)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    WST_NB3SN]`.
    """


class WstNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    WstNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 5)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK, WST_NB3SN]`.
    """


class CrocoRebcoSuperconductingTfCoilAreasAndMassesConventional(
    CrocoRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 6)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    CROCO_REBCO]`.
    """


class CrocoRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    CrocoRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 6)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK, CROCO_REBCO]`.
    """


class DurhamNbtiSuperconductingTfCoilAreasAndMassesConventional(
    DurhamNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 7)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    DURHAM_NBTI]`.
    """


class DurhamNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    DurhamNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 7)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK, DURHAM_NBTI]`.
    """


class DurhamRebcoSuperconductingTfCoilAreasAndMassesConventional(
    DurhamRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 8)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    DURHAM_REBCO]`.
    """


class DurhamRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    DurhamRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 8)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK,
    DURHAM_REBCO]`.
    """


class HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesConventional(
    HazeltonZhaiRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 9)`: `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO,
    HAZELTON_ZHAI_REBCO]`.
    """


class HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    HazeltonZhaiRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 9)`: `SC_TF_MASSES[SPHERICAL_TOKAMAK,
    HAZELTON_ZHAI_REBCO]`.
    """


class VvStressOnQuench(ExplicitFunction):
    """cottax node: `.superconducting_tfcoil.vv_stress_quench`, constraint 65's read."""

    vv_stress_quench = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        r_tf_inboard_mid=From(build),
        r_tf_outboard_mid=From(build),
        r_tf_inboard_out=From(build),
        tfa=From(tfcoil),
        z_plasma_xpoint_upper=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_shld_upper=From(build),
        dz_vv_upper=From(build),
        r_vv_inboard_out=From(build),
        dr_vv_outboard=From(build),
        dr_tf_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_shld_vv_gap_outboard=From(build),
        len_tf_coil=From(tfcoil),
        theta1_coil=From(tfcoil),
        theta1_vv=From(tfcoil),
        n_tf_coils=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        c_tf_coil=From(superconducting_tfcoil),
        dr_vv_shells=From(build),
    ):
        return calculate_vv_stress_on_quench(
            z_tf_inside_half,
            dr_tf_inboard,
            r_tf_inboard_mid,
            r_tf_outboard_mid,
            r_tf_inboard_out,
            tfa,
            z_plasma_xpoint_upper,
            dz_xpoint_divertor,
            dz_divertor,
            dz_shld_upper,
            dz_vv_upper,
            r_vv_inboard_out,
            dr_vv_outboard,
            dr_tf_outboard,
            dr_tf_shld_gap,
            dr_shld_thermal_outboard,
            dr_shld_vv_gap_outboard,
            len_tf_coil,
            theta1_coil,
            theta1_vv,
            n_tf_coils,
            n_tf_coil_turns,
            a_tf_coil_inboard_steel,
            a_tf_plasma_case,
            a_tf_coil_nose_case,
            dx_tf_side_case_average,
            t_tf_superconductor_quench,
            c_tf_coil,
            dr_vv_shells,
        )


class CiccSuperconductorProperties(ExplicitFunction):
    """The family that owns the CICC critical-current chain -- constraint 33's read."""

    j_tf_wp_critical = OutputInto(tfcoil)
    j_crit_str_tf = OutputInto(tfcoil)
    f_c_tf_turn_operating_critical = OutputInto(superconducting_tfcoil)
    j_tf_coil_turn = OutputInto(superconducting_tfcoil)
    j_tf_superconductor = OutputInto(superconducting_tfcoil)
    c_tf_turn_cables_critical = OutputInto(superconducting_tfcoil)
    j_tf_superconductor_critical = OutputInto(superconducting_tfcoil)
    b_tf_superconductor_critical_zero_temp_strain = OutputInto(superconducting_tfcoil)
    temp_tf_superconductor_critical_zero_field_strain = OutputInto(
        superconducting_tfcoil
    )


class IterNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 1` -- `large_tokamak_eval.IN.DAT:374`'s own arm."""

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
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


class UserDefinedNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 4` -- the ITER fit with `(bcritsc, tcritsc)` read from input."""

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
        bcritsc=From(tfcoil),
        tcritsc=From(tfcoil),
    ):
        return cicc_superconductor_properties_itersc(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
            b_c20max=bcritsc,
            temp_c0max=tcritsc,
        )


class WstNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 5` -- `low_aspect_ratio_DEMO.IN.DAT:910`'s arm."""

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
    ):
        return cicc_superconductor_properties_wst_nb3sn(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
        )


class OldLubellNbtiCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 3` -- and the arm that reads **no strain**."""

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        tftmp=From(tfcoil),
    ):
        return cicc_superconductor_properties_lubell_nbti(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            temp_tf_coolant_peak_field=tftmp,
        )


class DurhamNbtiCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 7` -- Durham Ginzburg-Landau NbTi."""

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
        b_crit_upper_nbti=From(tfcoil),
        t_crit_nbti=From(tfcoil),
    ):
        return cicc_superconductor_properties_durham_nbti(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
            b_crit_upper_nbti=b_crit_upper_nbti,
            t_crit_nbti=t_crit_nbti,
        )


class TfSuperconductorTemperatureMargin(ExplicitFunction):
    """The family that owns the TF temperature margin -- constraint 36's read."""

    temp_tf_superconductor_margin = OutputInto(tfcoil)
    temp_margin = OutputInto(tfcoil)


class _TemperatureMarginWithStrain(TfSuperconductorTemperatureMargin):
    """The strained arms' shared declaration; the `fit` attribute picks the fit."""

    fit = staticmethod(temperature_margin_itersc)

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        str_wp=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        return calculate_temperature_margin_with_strain(
            self.fit,
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            str_wp,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            tftmp,
        )


class IterNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 1` *(live)*. `process/models/superconductors.py:1259`."""

    fit = staticmethod(temperature_margin_itersc)


class UserDefinedNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 4`."""

    fit = staticmethod(temperature_margin_itersc)


class WstNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 5`. `process/models/superconductors.py:1263-1265`."""

    fit = staticmethod(temperature_margin_wst_nb3sn)


class OldLubellNbtiTfSuperconductorTemperatureMargin(TfSuperconductorTemperatureMargin):
    """`i_tf_sc_mat == 3` -- one read fewer, and one literal more."""

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        return calculate_old_lubell_nbti_temperature_margin(
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            tftmp,
        )
