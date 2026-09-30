"""Pure-functional port of `process/models/build.py`'s `Build` -- the tokamak radial and
vertical build.
"""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import build, tfcoil
from functional_process.models.build import (
    calculate_divertor_geometry_conventional,
    calculate_divertor_geometry_spherical_tokamak,
    calculate_dr_shld_vv_gap_outboard,
    calculate_dr_tf_inboard,
    calculate_dr_tf_inner_bore,
    calculate_dr_tf_outboard_superconducting,
    calculate_dr_tf_wp_with_insulation,
    calculate_dx_tf_wp_conductor_max_superconducting,
    calculate_dz_blkt_upper,
    calculate_r_cp_top_from_tf_inboard_out,
    calculate_r_shld_inboard_inner,
    calculate_r_shld_outboard_outer,
    calculate_r_tf_inboard_radii_no_cs_precomp,
    calculate_r_tf_inboard_radii_tf_outside_cs,
    calculate_r_tf_outboard_mid,  # noqa: F401 -- re-exported for tests
    calculate_r_tf_outboard_mid_dshape,
    calculate_r_tf_outboard_mid_picture_frame,
    calculate_r_tf_outboard_mid_unrippled,
    calculate_rbld,
    calculate_ripple_b_tf_plasma_edge_fitted,
    calculate_ripple_b_tf_plasma_edge_picture_frame,
    calculate_tf_top_height_double_null,
    calculate_tf_top_height_single_null,
    calculate_vacuum_vessel_and_shield_radii,
    calculate_z_plasma_xpoint,
    calculate_z_tf_inside_half,
    plasma_outboard_edge_toroidal_ripple_fitted,  # noqa: F401 -- re-exported for tests
    plasma_outboard_edge_toroidal_ripple_picture_frame,  # noqa: F401 -- re-exported for tests
)
from functional_process.models.safe_math import safe_sqrt  # noqa: F401

# cottax node: `calculate_z_plasma_xpoint`. No switch.
PlasmaXpointHeights = build.function(
    calculate_z_plasma_xpoint,
    "z_plasma_xpoint_upper",
    "z_plasma_xpoint_lower",
)


# cottax node: `calculate_dz_blkt_upper`, owning `.build.dz_blkt_upper`.
DzBlktUpper = build.function(calculate_dz_blkt_upper, "dz_blkt_upper")


# cottax node: `calculate_divertor_geometry_conventional`.
DivertorGeometryConventional = build.function(
    calculate_divertor_geometry_conventional,
    "dz_xpoint_divertor",
    "rspo",
)


# cottax node: `calculate_divertor_geometry_spherical_tokamak`.
DivertorGeometrySphericalTokamak = build.function(
    calculate_divertor_geometry_spherical_tokamak,
    "dz_xpoint_divertor",
)


# cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`.
ZTfInsideHalf = build.function(
    calculate_z_tf_inside_half,
    "z_tf_inside_half",
)


# The TfTopHeight family:
# The family that owns `.build.z_tf_top` and `.build.dz_tf_upper_lower_midplane`.


# cottax node: `calculate_tf_top_height_single_null`.
TfTopHeightSingleNull = build.function(
    calculate_tf_top_height_single_null, "z_tf_top", "dz_tf_upper_lower_midplane"
)


# cottax node: `calculate_tf_top_height_double_null`.
TfTopHeightDoubleNull = build.function(
    calculate_tf_top_height_double_null, "z_tf_top", "dz_tf_upper_lower_midplane"
)


# cottax node: `calculate_dz_blkt_upper`.
BlktUpperThickness = build.function(calculate_dz_blkt_upper, "dz_blkt_upper")


# cottax node: `calculate_dr_tf_inboard`.
DrTfInboardFromWindingPack = build.function(
    calculate_dr_tf_inboard,
    "dr_tf_inboard",
)


# cottax node: `calculate_dr_tf_wp_with_insulation`.
DrTfWpWithInsulationFromInboardBuild = tfcoil.function(
    calculate_dr_tf_wp_with_insulation,
    "dr_tf_wp_with_insulation",
)


# cottax node: `calculate_r_tf_inboard_radii_tf_outside_cs`.
TfInboardRadiiTfOutsideCs = build.function(
    calculate_r_tf_inboard_radii_tf_outside_cs,
    "dr_cs_bore",
    "dr_cs_precomp",
    "r_tf_inboard_in",
    "r_tf_inboard_mid",
    "r_tf_inboard_out",
)


# cottax node: `calculate_r_tf_inboard_radii_no_cs_precomp`.
TfInboardRadiiNoCsPrecomp = build.function(
    calculate_r_tf_inboard_radii_no_cs_precomp,
    "dr_cs_bore",
    "dr_cs_precomp",
    "r_tf_inboard_in",
    "r_tf_inboard_mid",
    "r_tf_inboard_out",
)


# cottax node: `calculate_r_shld_inboard_inner`. No switch.
ShldInboardInnerRadius = build.function(
    calculate_r_shld_inboard_inner,
    "r_shld_inboard_inner",
)


# cottax node: `calculate_r_shld_outboard_outer`. No switch.
ShldOutboardOuterRadius = build.function(
    calculate_r_shld_outboard_outer,
    "r_shld_outboard_outer",
)


# cottax node: `calculate_dr_tf_outboard_superconducting`.
DrTfOutboardSuperconducting = build.function(
    calculate_dr_tf_outboard_superconducting, "dr_tf_outboard"
)


# cottax node: `calculate_dx_tf_wp_conductor_max_superconducting`.
WpConductorMaxWidthSuperconducting = tfcoil.function(
    calculate_dx_tf_wp_conductor_max_superconducting, "dx_tf_wp_conductor_max"
)


# cottax node: `calculate_r_tf_outboard_mid_unrippled`.
TfOutboardMidUnrippled = build.function(
    calculate_r_tf_outboard_mid_unrippled, "r_tf_outboard_mid_unrippled"
)


# cottax node: `calculate_r_tf_outboard_mid_dshape`, the ripple constraint on the
# outboard TF leg.
TfOutboardMidDShape = build.function(
    calculate_r_tf_outboard_mid_dshape,
    "r_tf_outboard_mid",
    reads={"dx_tf_wp_conductor_max": tfcoil.place("dx_tf_wp_conductor_max")},
)


# cottax node: `calculate_ripple_b_tf_plasma_edge_fitted` --
# `plasma_outboard_edge_toroidal_ripple_fitted`, evaluated at the final leg radius.
TfOutboardEdgeRipple = tfcoil.function(
    calculate_ripple_b_tf_plasma_edge_fitted,
    "ripple_b_tf_plasma_edge",
)


# cottax node: `calculate_r_tf_outboard_mid_picture_frame`, the ripple constraint
# on the outboard TF leg.
TfOutboardMidPictureFrame = build.function(
    calculate_r_tf_outboard_mid_picture_frame,
    "r_tf_outboard_mid",
)


# cottax node: `calculate_ripple_b_tf_plasma_edge_picture_frame` --
# `plasma_outboard_edge_toroidal_ripple_picture_frame`, evaluated at the final leg
# radius.
TfOutboardEdgeRipplePictureFrame = tfcoil.function(
    calculate_ripple_b_tf_plasma_edge_picture_frame,
    "ripple_b_tf_plasma_edge",
)


# cottax node: `calculate_dr_shld_vv_gap_outboard`.
ShldVvGapOutboard = build.function(
    calculate_dr_shld_vv_gap_outboard, "dr_shld_vv_gap_outboard"
)


# cottax node: `calculate_dr_tf_inner_bore`.
TfInnerBore = build.function(calculate_dr_tf_inner_bore, "dr_tf_inner_bore")


# cottax node: `calculate_vacuum_vessel_and_shield_radii`.
VacuumVesselAndShieldRadiiTfOutsideCs = build.function(
    calculate_vacuum_vessel_and_shield_radii,
    "r_vv_inboard_out",
    "r_sh_inboard_in",
    "r_sh_inboard_out",
)


# cottax node: `calculate_rbld`.
RadialBuildToPlasmaCentre = build.function(
    calculate_rbld,
    "rbld",
)


# cottax node: `calculate_r_cp_top_from_tf_inboard_out`, ports declared.
RCpTopFromTfInboardOut = build.function(
    calculate_r_cp_top_from_tf_inboard_out, "r_cp_top"
)
