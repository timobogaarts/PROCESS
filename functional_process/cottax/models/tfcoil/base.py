"""Pure-functional port of `process/models/tfcoil/base.py` -- the device-agnostic TF
coil layer that `SuperconductingTFCoil.run_base_superconducting_tf` reaches by
inheritance.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    physics,
    superconducting_tfcoil,
    tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.tfcoil.base import (
    calculate_r_b_tf_inboard_peak,
    calculate_tf_global_geometry_circular_case,
    calculate_tf_global_geometry_straight_case,
    circumference,  # noqa: F401 -- re-exported for tests/.../test_base.py
    dr_tf_plasma_case_from_fraction,
    dr_tf_plasma_case_from_input,
    dx_tf_side_case_min_from_fraction,
    generic_tf_coil_area_and_masses,
    tf_coil_self_inductance_d_shape,
    tf_coil_self_inductance_picture_frame,
    tf_coil_shape_inner_d_shape_double_null,
    tf_coil_shape_inner_d_shape_single_null,
    tf_coil_shape_inner_picture_frame_tart,
    tf_current,
    tf_stored_magnetic_energy,
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class TfGlobalGeometry(ExplicitFunction):
    """The family that owns `tf_global_geometry`'s nine unswitched outputs."""


class TfGlobalGeometryCircularCase(TfGlobalGeometry, WrapsFunction):
    """`i_tf_case_geom == TFPlasmaCaseType.CIRCULAR` (0).

    `large_tokamak_eval`'s arm.
    """

    fn = calculate_tf_global_geometry_circular_case

    n_tf_coils = From(tfcoil)
    r_tf_inboard_out = From(build)
    r_tf_inboard_in = From(build)
    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)


class TfGlobalGeometryStraightCase(TfGlobalGeometry, WrapsFunction):
    """`i_tf_case_geom == TFPlasmaCaseType.STRAIGHT` (1)."""

    fn = calculate_tf_global_geometry_straight_case

    n_tf_coils = From(tfcoil)
    r_tf_inboard_out = From(build)
    r_tf_inboard_in = From(build)
    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)


class DrTfPlasmaCaseFromInput(FixedPointFunction):
    """`i_f_dr_tf_plasma_case == False`.

    `large_tokamak_eval`'s arm, and a self-loop.
    """

    dr_tf_plasma_case = OutputInto(tfcoil)

    def step(
        self,
        dr_tf_plasma_case=From(tfcoil),
        r_tf_inboard_in=From(build),
        dr_tf_inboard=From(build),
        n_tf_coils=From(tfcoil),
    ):
        return dr_tf_plasma_case_from_input(
            dr_tf_plasma_case=dr_tf_plasma_case,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_inboard=dr_tf_inboard,
            n_tf_coils=n_tf_coils,
        )


class DrTfPlasmaCaseFromFraction(WrapsFunction):
    """`i_f_dr_tf_plasma_case == True`: a plain node, no loop."""

    fn = dr_tf_plasma_case_from_fraction

    f_dr_tf_plasma_case = From(tfcoil)
    dr_tf_inboard = From(build)
    r_tf_inboard_in = From(build)
    n_tf_coils = From(tfcoil)

    dr_tf_plasma_case = OutputInto(tfcoil)


class DxTfSideCaseMinFromFraction(WrapsFunction):
    """`tfc_sidewall_is_fraction == True`."""

    fn = dx_tf_side_case_min_from_fraction

    casths_fraction = From(tfcoil)
    r_tf_inboard_in = From(build)
    dr_tf_nose_case = From(tfcoil)
    n_tf_coils = From(tfcoil)

    dx_tf_side_case_min = OutputInto(tfcoil)


class RBTfInboardPeak(WrapsFunction):
    """cottax node: `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak`."""

    fn = calculate_r_b_tf_inboard_peak

    r_tf_inboard_out = From(build)
    dr_tf_plasma_case = From(tfcoil)
    dx_tf_wp_insulation = From(tfcoil)
    dx_tf_wp_insertion_gap = From(tfcoil)

    r_b_tf_inboard_peak = OutputInto(tfcoil)


class TfCurrent(WrapsFunction):
    """cottax node: `tf_current`, ports declared. No switch, so no family."""

    fn = tf_current

    n_tf_coils = From(tfcoil)
    b_plasma_toroidal_on_axis = From(physics)
    rmajor = From(physics)
    r_b_tf_inboard_peak = From(tfcoil)
    a_tf_inboard_total = From(tfcoil)

    b_tf_inboard_peak_symmetric = OutputInto(tfcoil)
    c_tf_total = OutputInto(tfcoil)
    c_tf_coil = OutputInto(superconducting_tfcoil)
    j_tf_coil_full_area = OutputInto(tfcoil)


class TfCoilShape(ExplicitFunction):
    """The family that owns `.tfcoil.len_tf_coil` and the arc arrays."""


class TfCoilShapeDShapeSingleNull(TfCoilShape, WrapsFunction):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 1` -- the reference arm."""

    fn = tf_coil_shape_inner_d_shape_single_null

    r_tf_inboard_out = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    r_tf_outboard_in = From(superconducting_tfcoil)
    z_tf_inside_half = From(build)
    z_tf_top = From(build)
    dr_tf_inboard = From(build)

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)


class TfCoilShapeDShapeDoubleNull(TfCoilShape, WrapsFunction):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 0`."""

    fn = tf_coil_shape_inner_d_shape_double_null

    r_tf_inboard_out = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    r_tf_outboard_in = From(superconducting_tfcoil)
    z_tf_inside_half = From(build)
    dr_tf_inboard = From(build)

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)


class TfCoilShapePictureFrameTart(TfCoilShape, WrapsFunction):
    """`i_tf_shape == 2`, `itart == 1` -- both ST regression files' arm."""

    fn = tf_coil_shape_inner_picture_frame_tart

    r_cp_top = From(build)
    r_tf_outboard_in = From(superconducting_tfcoil)
    z_tf_inside_half = From(build)
    z_tf_top = From(build)
    dr_tf_inboard = From(build)
    r_tf_outboard_mid = From(build)

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)


class TfCoilSelfInductance(ExplicitFunction):
    """The family that owns `.tfcoil.ind_tf_coil`. `(itart, i_tf_shape)` decides it."""


class TfCoilSelfInductanceDShape(TfCoilSelfInductance, WrapsFunction):
    """`itart == 0` and `i_tf_shape == 1` -- the reference arm."""

    fn = tf_coil_self_inductance_d_shape

    dr_tf_inboard = From(build)
    r_tf_arc = From(tfcoil)
    z_tf_arc = From(tfcoil)

    ind_tf_coil = OutputInto(tfcoil)


class TfCoilSelfInductancePictureFrame(TfCoilSelfInductance, WrapsFunction):
    """Everything else (`i_tf_shape == 2`, or `itart == 1`): the closed form."""

    fn = tf_coil_self_inductance_picture_frame

    z_tf_inside_half = From(build)
    dr_tf_outboard = From(build)
    r_tf_outboard_mid = From(build)
    r_tf_inboard_mid = From(build)

    ind_tf_coil = OutputInto(tfcoil)


class TfStoredMagneticEnergy(WrapsFunction):
    """cottax node: `tf_stored_magnetic_energy`. Owns one of the slot's ten reads."""

    fn = tf_stored_magnetic_energy

    ind_tf_coil = From(tfcoil)
    c_tf_total = From(tfcoil)
    n_tf_coils = From(tfcoil)

    e_tf_magnetic_stored_total = OutputInto(tfcoil)
    e_tf_magnetic_stored_total_gj = OutputInto(tfcoil)
    e_tf_coil_magnetic_stored = OutputInto(tfcoil)


class GenericTfCoilAreaAndMasses(WrapsFunction):
    """cottax node: `generic_tf_coil_area_and_masses`. Owns `.tfcoil.tfcryoarea`."""

    fn = generic_tf_coil_area_and_masses

    r_tf_inboard_out = From(build)
    r_tf_inboard_in = From(build)
    rad_tf_coil_inboard_toroidal_half = From(superconducting_tfcoil)
    tan_theta_coil = From(superconducting_tfcoil)
    len_tf_coil = From(tfcoil)
    r_tf_inboard_mid = From(build)
    r_tf_outboard_mid = From(build)

    tfocrn = OutputInto(tfcoil)
    tficrn = OutputInto(tfcoil)
    tfcryoarea = OutputInto(tfcoil)
