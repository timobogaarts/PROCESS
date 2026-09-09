"""Pure-functional port of `process/models/build.py`'s `Build` -- the tokamak radial and
vertical build.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import build, divertor, physics, tfcoil
from functional_process.cottax.wraps import WrapsFunction
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
    calculate_r_tf_outboard_mid,
    calculate_r_tf_outboard_mid_unrippled,
    calculate_rbld,
    calculate_tf_top_height_double_null,
    calculate_tf_top_height_single_null,
    calculate_vacuum_vessel_and_shield_radii,
    calculate_z_plasma_xpoint,
    calculate_z_tf_inside_half,
    plasma_outboard_edge_toroidal_ripple_fitted,
    plasma_outboard_edge_toroidal_ripple_picture_frame,
)
from functional_process.models.safe_math import safe_sqrt  # noqa: F401


class PlasmaXpointHeights(WrapsFunction):
    """cottax node: `calculate_z_plasma_xpoint`. No switch."""

    fn = calculate_z_plasma_xpoint

    rminor = From(physics)
    kappa = From(physics)

    z_plasma_xpoint_upper = OutputInto(build)
    z_plasma_xpoint_lower = OutputInto(build)


class DzBlktUpper(WrapsFunction):
    """cottax node: `calculate_dz_blkt_upper`, owning `.build.dz_blkt_upper`."""

    fn = calculate_dz_blkt_upper

    dr_blkt_inboard = From(build)
    dr_blkt_outboard = From(build)

    dz_blkt_upper = OutputInto(build)


class DivertorGeometryConventional(WrapsFunction):
    """cottax node: `calculate_divertor_geometry_conventional`."""

    fn = calculate_divertor_geometry_conventional

    rmajor = From(physics)
    rminor = From(physics)
    kappa = From(physics)
    triang = From(physics)
    plsepi = From(build)
    plsepo = From(build)
    plleni = From(build)
    plleno = From(build)
    betai = From(divertor)
    betao = From(divertor)

    dz_xpoint_divertor = OutputInto(build)
    rspo = OutputInto(build)


class DivertorGeometrySphericalTokamak(WrapsFunction):
    """cottax node: `calculate_divertor_geometry_spherical_tokamak`."""

    fn = calculate_divertor_geometry_spherical_tokamak

    rminor = From(physics)

    dz_xpoint_divertor = OutputInto(build)


class ZTfInsideHalf(WrapsFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`."""

    fn = calculate_z_tf_inside_half

    z_plasma_xpoint_upper = From(build)
    dz_xpoint_divertor = From(build)
    dz_divertor = From(divertor)
    dz_shld_lower = From(build)
    dz_vv_lower = From(build)
    dz_shld_vv_gap = From(build)
    dz_shld_thermal = From(build)
    dr_tf_shld_gap = From(build)

    z_tf_inside_half = OutputInto(build)


class TfTopHeight(ExplicitFunction):
    """The family that owns `.build.z_tf_top` and `.build.dz_tf_upper_lower_midplane`."""


class TfTopHeightSingleNull(TfTopHeight, WrapsFunction):
    """cottax node: `calculate_tf_top_height_single_null`."""

    fn = calculate_tf_top_height_single_null

    z_tf_inside_half = From(build)
    dr_tf_inboard = From(build)
    dr_tf_shld_gap = From(build)
    dz_shld_thermal = From(build)
    dz_shld_vv_gap = From(build)
    dz_vv_upper = From(build)
    dz_shld_upper = From(build)
    dr_shld_blkt_gap = From(build)
    dz_blkt_upper = From(build)
    dr_fw_inboard = From(build)
    dr_fw_outboard = From(build)
    dz_fw_plasma_gap = From(build)
    z_plasma_xpoint_upper = From(build)

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)


class TfTopHeightDoubleNull(TfTopHeight, WrapsFunction):
    """cottax node: `calculate_tf_top_height_double_null`."""

    fn = calculate_tf_top_height_double_null

    z_tf_inside_half = From(build)
    dr_tf_inboard = From(build)

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)


class BlktUpperThickness(WrapsFunction):
    """cottax node: `calculate_dz_blkt_upper`."""

    fn = calculate_dz_blkt_upper

    dr_blkt_inboard = From(build)
    dr_blkt_outboard = From(build)

    dz_blkt_upper = OutputInto(build)


class DrTfInboardFromWindingPack(WrapsFunction):
    """cottax node: `calculate_dr_tf_inboard`."""

    fn = calculate_dr_tf_inboard

    dr_tf_wp_with_insulation = From(tfcoil)
    dr_tf_plasma_case = From(tfcoil)
    dr_tf_nose_case = From(tfcoil)

    dr_tf_inboard = OutputInto(build)


class DrTfWpWithInsulationFromInboardBuild(WrapsFunction):
    """cottax node: `calculate_dr_tf_wp_with_insulation`."""

    fn = calculate_dr_tf_wp_with_insulation

    dr_tf_inboard = From(build)
    dr_tf_plasma_case = From(tfcoil)
    dr_tf_nose_case = From(tfcoil)

    dr_tf_wp_with_insulation = OutputInto(tfcoil)


class TfInboardRadiiTfOutsideCs(WrapsFunction):
    """cottax node: `calculate_r_tf_inboard_radii_tf_outside_cs`."""

    fn = calculate_r_tf_inboard_radii_tf_outside_cs

    dr_bore = From(build)
    dr_cs = From(build)
    fseppc = From(build)
    fcspc = From(build)
    sigallpc = From(build)
    dr_cs_tf_gap = From(build)
    dr_tf_inboard = From(build)

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)


class TfInboardRadiiNoCsPrecomp(WrapsFunction):
    """cottax node: `calculate_r_tf_inboard_radii_no_cs_precomp`."""

    fn = calculate_r_tf_inboard_radii_no_cs_precomp

    dr_bore = From(build)
    dr_cs = From(build)
    dr_cs_tf_gap = From(build)
    dr_tf_inboard = From(build)

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)


class ShldInboardInnerRadius(WrapsFunction):
    """cottax node: `calculate_r_shld_inboard_inner`. No switch."""

    fn = calculate_r_shld_inboard_inner

    rmajor = From(physics)
    rminor = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_inboard = From(build)
    dr_blkt_inboard = From(build)
    dr_shld_inboard = From(build)

    r_shld_inboard_inner = OutputInto(build)


class ShldOutboardOuterRadius(WrapsFunction):
    """cottax node: `calculate_r_shld_outboard_outer`. No switch."""

    fn = calculate_r_shld_outboard_outer

    rmajor = From(physics)
    rminor = From(physics)
    dr_fw_plasma_gap_outboard = From(build)
    dr_fw_outboard = From(build)
    dr_blkt_outboard = From(build)
    dr_shld_outboard = From(build)

    r_shld_outboard_outer = OutputInto(build)


class DrTfOutboardSuperconducting(WrapsFunction):
    """cottax node: `calculate_dr_tf_outboard_superconducting`."""

    fn = calculate_dr_tf_outboard_superconducting

    dr_tf_inboard = From(build)

    dr_tf_outboard = OutputInto(build)


class WpConductorMaxWidthSuperconducting(WrapsFunction):
    """cottax node: `calculate_dx_tf_wp_conductor_max_superconducting`."""

    fn = calculate_dx_tf_wp_conductor_max_superconducting

    dx_tf_wp_primary_toroidal = From(tfcoil)
    dx_tf_wp_insulation = From(tfcoil)
    dx_tf_wp_insertion_gap = From(tfcoil)

    dx_tf_wp_conductor_max = OutputInto(tfcoil)


class TfOutboardMidUnrippled(WrapsFunction):
    """cottax node: `calculate_r_tf_outboard_mid_unrippled`."""

    fn = calculate_r_tf_outboard_mid_unrippled

    r_shld_outboard_outer = From(build)
    dr_shld_blkt_gap = From(build)
    dr_vv_outboard = From(build)
    gapomin = From(build)
    dr_shld_thermal_outboard = From(build)
    dr_tf_shld_gap = From(build)
    dr_tf_outboard = From(build)

    r_tf_outboard_mid_unrippled = OutputInto(build)


class TfOutboardMidDShape(ExplicitFunction):
    """cottax node: the ripple constraint on the outboard TF leg."""

    r_tf_outboard_mid = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid_unrippled=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
        dx_tf_wp_conductor_max=From(tfcoil),
    ):
        _, r_tf_outboard_midmin = plasma_outboard_edge_toroidal_ripple_fitted(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid_unrippled,
            n_tf_coils,
            rmajor,
            rminor,
            dx_tf_wp_conductor_max,
        )
        return calculate_r_tf_outboard_mid(
            r_tf_outboard_mid_unrippled, r_tf_outboard_midmin
        )


class TfOutboardEdgeRipple(ExplicitFunction):
    """cottax node: `plasma_outboard_edge_toroidal_ripple_fitted`, evaluated at the
    final leg radius.
    """

    ripple_b_tf_plasma_edge = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
        dx_tf_wp_conductor_max=From(tfcoil),
    ):
        ripple_b_tf_plasma_edge, _ = plasma_outboard_edge_toroidal_ripple_fitted(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid,
            n_tf_coils,
            rmajor,
            rminor,
            dx_tf_wp_conductor_max,
        )
        return ripple_b_tf_plasma_edge


class TfOutboardMidPictureFrame(ExplicitFunction):
    """cottax node: the ripple constraint on the outboard TF leg."""

    r_tf_outboard_mid = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid_unrippled=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        _, r_tf_outboard_midmin = plasma_outboard_edge_toroidal_ripple_picture_frame(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid_unrippled,
            n_tf_coils,
            rmajor,
            rminor,
        )
        return calculate_r_tf_outboard_mid(
            r_tf_outboard_mid_unrippled, r_tf_outboard_midmin
        )


class TfOutboardEdgeRipplePictureFrame(ExplicitFunction):
    """cottax node: `plasma_outboard_edge_toroidal_ripple_picture_frame`, evaluated at
    the final leg radius.
    """

    ripple_b_tf_plasma_edge = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        ripple_b_tf_plasma_edge, _ = plasma_outboard_edge_toroidal_ripple_picture_frame(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid,
            n_tf_coils,
            rmajor,
            rminor,
        )
        return ripple_b_tf_plasma_edge


class ShldVvGapOutboard(WrapsFunction):
    """cottax node: `calculate_dr_shld_vv_gap_outboard`."""

    fn = calculate_dr_shld_vv_gap_outboard

    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)
    dr_vv_outboard = From(build)
    r_shld_outboard_outer = From(build)
    dr_shld_thermal_outboard = From(build)
    dr_tf_shld_gap = From(build)
    dr_shld_blkt_gap = From(build)

    dr_shld_vv_gap_outboard = OutputInto(build)


class TfInnerBore(WrapsFunction):
    """cottax node: `calculate_dr_tf_inner_bore`."""

    fn = calculate_dr_tf_inner_bore

    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)
    r_tf_inboard_mid = From(build)
    dr_tf_inboard = From(build)

    dr_tf_inner_bore = OutputInto(build)


class VacuumVesselAndShieldRadiiTfOutsideCs(WrapsFunction):
    """cottax node: `calculate_vacuum_vessel_and_shield_radii`."""

    fn = calculate_vacuum_vessel_and_shield_radii

    r_tf_inboard_out = From(build)
    dr_tf_shld_gap = From(build)
    dr_shld_thermal_inboard = From(build)
    dr_shld_vv_gap_inboard = From(build)
    dr_vv_inboard = From(build)
    dr_shld_inboard = From(build)

    r_vv_inboard_out = OutputInto(build)
    r_sh_inboard_in = OutputInto(build)
    r_sh_inboard_out = OutputInto(build)


class RadialBuildToPlasmaCentre(WrapsFunction):
    """cottax node: `calculate_rbld`."""

    fn = calculate_rbld

    r_sh_inboard_out = From(build)
    dr_shld_blkt_gap = From(build)
    dr_blkt_inboard = From(build)
    dr_fw_inboard = From(build)
    dr_fw_plasma_gap_inboard = From(build)
    rminor = From(physics)

    rbld = OutputInto(build)


class RCpTopFromTfInboardOut(WrapsFunction):
    """cottax node: `calculate_r_cp_top_from_tf_inboard_out`, ports declared."""

    fn = calculate_r_cp_top_from_tf_inboard_out

    r_tf_inboard_out = From(build)

    r_cp_top = OutputInto(build)
