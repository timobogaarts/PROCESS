"""Pure-functional port of `process/models/tfcoil/base.py` -- the device-agnostic TF
coil layer that `SuperconductingTFCoil.run_base_superconducting_tf` reaches by
inheritance.
"""

from cottax.assemblies.jax import V

from functional_process.cottax.paths import (
    superconducting_tfcoil,
    tfcoil,
)
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


# The TfGlobalGeometry family:
# The family that owns `tf_global_geometry`'s nine unswitched outputs.


# `i_tf_case_geom == TFPlasmaCaseType.CIRCULAR` (0).
#
# `large_tokamak_eval`'s arm.
TfGlobalGeometryCircularCase = superconducting_tfcoil.function(
    calculate_tf_global_geometry_circular_case,
    "rad_tf_coil_inboard_toroidal_half",
    "tan_theta_coil",
    V.tfcoil.a_tf_inboard_total,
    "r_tf_outboard_in",
    "r_tf_outboard_out",
    V.tfcoil.dx_tf_inboard_out_toroidal,
    V.tfcoil.a_tf_leg_outboard,
    V.tfcoil.dr_tf_full_midplane,
    V.tfcoil.dr_tf_internal_midplane,
)


# `i_tf_case_geom == TFPlasmaCaseType.STRAIGHT` (1).
TfGlobalGeometryStraightCase = superconducting_tfcoil.function(
    calculate_tf_global_geometry_straight_case,
    "rad_tf_coil_inboard_toroidal_half",
    "tan_theta_coil",
    V.tfcoil.a_tf_inboard_total,
    "r_tf_outboard_in",
    "r_tf_outboard_out",
    V.tfcoil.dx_tf_inboard_out_toroidal,
    V.tfcoil.a_tf_leg_outboard,
    V.tfcoil.dr_tf_full_midplane,
    V.tfcoil.dr_tf_internal_midplane,
)


# `i_f_dr_tf_plasma_case == False`.
#
# `large_tokamak_eval`'s arm. **A plain node, not a self-loop**: the value clamped
# here is the IN.DAT's `dr_tf_plasma_case`, read as the free place
# `.tfcoil.dr_tf_plasma_case_in` (`evaluate.KNOWN_MINT_VALUES` resolves it back to
# the same PROCESS field), and never this node's own earlier output. The clamp is
# idempotent -- `max(max(u, c), c) == max(u, c)` -- so the "fixed point" it used to
# declare was reached by the first evaluation from any starting guess. Its sibling
# `DrTfPlasmaCaseFromFraction` never read the field at all, which is the same
# quantity written without the loop.
def _dr_tf_plasma_case_from_input(
    dr_tf_plasma_case_in, r_tf_inboard_in, dr_tf_inboard, n_tf_coils
):
    return dr_tf_plasma_case_from_input(
        dr_tf_plasma_case=dr_tf_plasma_case_in,
        r_tf_inboard_in=r_tf_inboard_in,
        dr_tf_inboard=dr_tf_inboard,
        n_tf_coils=n_tf_coils,
    )


DrTfPlasmaCaseFromInput = tfcoil.function(
    _dr_tf_plasma_case_from_input,
    "dr_tf_plasma_case",
)


# `i_f_dr_tf_plasma_case == True`: a plain node, no loop.
DrTfPlasmaCaseFromFraction = tfcoil.function(
    dr_tf_plasma_case_from_fraction,
    "dr_tf_plasma_case",
)


# `tfc_sidewall_is_fraction == True`.
DxTfSideCaseMinFromFraction = tfcoil.function(
    dx_tf_side_case_min_from_fraction,
    "dx_tf_side_case_min",
)


# cottax node: `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak`.
RBTfInboardPeak = tfcoil.function(
    calculate_r_b_tf_inboard_peak,
    "r_b_tf_inboard_peak",
)


# cottax node: `tf_current`, ports declared. No switch, so no family.
TfCurrent = tfcoil.function(
    tf_current,
    "b_tf_inboard_peak_symmetric",
    "c_tf_total",
    V.superconducting_tfcoil.c_tf_coil,
    "j_tf_coil_full_area",
)


# The TfCoilShape family:
# The family that owns `.tfcoil.len_tf_coil` and the arc arrays.


# `i_tf_shape == 1`, `itart == 0`, `i_single_null == 1` -- the reference arm.
TfCoilShapeDShapeSingleNull = tfcoil.function(
    tf_coil_shape_inner_d_shape_single_null,
    "len_tf_coil",
    "tfa",
    "tfb",
    "r_tf_arc",
    "z_tf_arc",
)


# `i_tf_shape == 1`, `itart == 0`, `i_single_null == 0`.
TfCoilShapeDShapeDoubleNull = tfcoil.function(
    tf_coil_shape_inner_d_shape_double_null,
    "len_tf_coil",
    "tfa",
    "tfb",
    "r_tf_arc",
    "z_tf_arc",
)


# `i_tf_shape == 2`, `itart == 1` -- both ST regression files' arm.
TfCoilShapePictureFrameTart = tfcoil.function(
    tf_coil_shape_inner_picture_frame_tart,
    "len_tf_coil",
    "tfa",
    "tfb",
    "r_tf_arc",
    "z_tf_arc",
)


# The TfCoilSelfInductance family:
# The family that owns `.tfcoil.ind_tf_coil`. `(itart, i_tf_shape)` decides it.


# `itart == 0` and `i_tf_shape == 1` -- the reference arm.
TfCoilSelfInductanceDShape = tfcoil.function(
    tf_coil_self_inductance_d_shape,
    "ind_tf_coil",
)


# Everything else (`i_tf_shape == 2`, or `itart == 1`): the closed form.
TfCoilSelfInductancePictureFrame = tfcoil.function(
    tf_coil_self_inductance_picture_frame,
    "ind_tf_coil",
)


# cottax node: `tf_stored_magnetic_energy`. Owns one of the slot's ten reads.
TfStoredMagneticEnergy = tfcoil.function(
    tf_stored_magnetic_energy,
    "e_tf_magnetic_stored_total",
    "e_tf_magnetic_stored_total_gj",
    "e_tf_coil_magnetic_stored",
)


# cottax node: `generic_tf_coil_area_and_masses`. Owns `.tfcoil.tfcryoarea`.
GenericTfCoilAreaAndMasses = tfcoil.function(
    generic_tf_coil_area_and_masses,
    "tfocrn",
    "tficrn",
    "tfcryoarea",
)
