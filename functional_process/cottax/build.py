"""Pure-functional port of `process/models/build.py`'s `Build` -- the tokamak radial and
vertical build.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

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
from functional_process.cottax.paths import build, divertor, physics, tfcoil


class PlasmaXpointHeights(ExplicitFunction):
    """cottax node: `calculate_z_plasma_xpoint`. No switch."""

    z_plasma_xpoint_upper = OutputInto(build)
    z_plasma_xpoint_lower = OutputInto(build)

    def __call__(self, rminor=From(physics), kappa=From(physics)):
        return calculate_z_plasma_xpoint(rminor, kappa)


class DzBlktUpper(ExplicitFunction):
    """cottax node: `calculate_dz_blkt_upper`, owning `.build.dz_blkt_upper`."""

    dz_blkt_upper = OutputInto(build)

    def __call__(self, dr_blkt_inboard=From(build), dr_blkt_outboard=From(build)):
        return calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard)


class DivertorGeometryConventional(ExplicitFunction):
    """cottax node: `calculate_divertor_geometry_conventional`."""

    dz_xpoint_divertor = OutputInto(build)
    rspo = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        triang=From(physics),
        plsepi=From(build),
        plsepo=From(build),
        plleni=From(build),
        plleno=From(build),
        betai=From(divertor),
        betao=From(divertor),
    ):
        return calculate_divertor_geometry_conventional(
            rmajor,
            rminor,
            kappa,
            triang,
            plsepi,
            plsepo,
            plleni,
            plleno,
            betai,
            betao,
        )


class DivertorGeometrySphericalTokamak(ExplicitFunction):
    """cottax node: `calculate_divertor_geometry_spherical_tokamak`."""

    dz_xpoint_divertor = OutputInto(build)

    def __call__(self, rminor=From(physics)):
        return calculate_divertor_geometry_spherical_tokamak(rminor)


class ZTfInsideHalf(ExplicitFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`."""

    z_tf_inside_half = OutputInto(build)

    def __call__(
        self,
        z_plasma_xpoint_upper=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_shld_lower=From(build),
        dz_vv_lower=From(build),
        dz_shld_vv_gap=From(build),
        dz_shld_thermal=From(build),
        dr_tf_shld_gap=From(build),
    ):
        return calculate_z_tf_inside_half(
            z_plasma_xpoint_upper,
            dz_xpoint_divertor,
            dz_divertor,
            dz_shld_lower,
            dz_vv_lower,
            dz_shld_vv_gap,
            dz_shld_thermal,
            dr_tf_shld_gap,
        )


class TfTopHeight(ExplicitFunction):
    """The family that owns `.build.z_tf_top` and `.build.dz_tf_upper_lower_midplane`.
    """


class TfTopHeightSingleNull(TfTopHeight):
    """cottax node: `calculate_tf_top_height_single_null`."""

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        dr_tf_shld_gap=From(build),
        dz_shld_thermal=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_upper=From(build),
        dz_shld_upper=From(build),
        dr_shld_blkt_gap=From(build),
        dz_blkt_upper=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        dz_fw_plasma_gap=From(build),
        z_plasma_xpoint_upper=From(build),
    ):
        return calculate_tf_top_height_single_null(
            z_tf_inside_half,
            dr_tf_inboard,
            dr_tf_shld_gap,
            dz_shld_thermal,
            dz_shld_vv_gap,
            dz_vv_upper,
            dz_shld_upper,
            dr_shld_blkt_gap,
            dz_blkt_upper,
            dr_fw_inboard,
            dr_fw_outboard,
            dz_fw_plasma_gap,
            z_plasma_xpoint_upper,
        )


class TfTopHeightDoubleNull(TfTopHeight):
    """cottax node: `calculate_tf_top_height_double_null`."""

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_tf_top_height_double_null(z_tf_inside_half, dr_tf_inboard)


class BlktUpperThickness(ExplicitFunction):
    """cottax node: `calculate_dz_blkt_upper`."""

    dz_blkt_upper = OutputInto(build)

    def __call__(
        self,
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
    ):
        return calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard)


class DrTfInboardFromWindingPack(ExplicitFunction):
    """cottax node: `calculate_dr_tf_inboard`."""

    dr_tf_inboard = OutputInto(build)

    def __call__(
        self,
        dr_tf_wp_with_insulation=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        dr_tf_nose_case=From(tfcoil),
    ):
        return calculate_dr_tf_inboard(
            dr_tf_wp_with_insulation, dr_tf_plasma_case, dr_tf_nose_case
        )


class DrTfWpWithInsulationFromInboardBuild(ExplicitFunction):
    """cottax node: `calculate_dr_tf_wp_with_insulation`."""

    dr_tf_wp_with_insulation = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_inboard=From(build),
        dr_tf_plasma_case=From(tfcoil),
        dr_tf_nose_case=From(tfcoil),
    ):
        return calculate_dr_tf_wp_with_insulation(
            dr_tf_inboard, dr_tf_plasma_case, dr_tf_nose_case
        )


class TfInboardRadiiTfOutsideCs(ExplicitFunction):
    """cottax node: `calculate_r_tf_inboard_radii_tf_outside_cs`."""

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)

    def __call__(
        self,
        dr_bore=From(build),
        dr_cs=From(build),
        fseppc=From(build),
        fcspc=From(build),
        sigallpc=From(build),
        dr_cs_tf_gap=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_r_tf_inboard_radii_tf_outside_cs(
            dr_bore,
            dr_cs,
            fseppc,
            fcspc,
            sigallpc,
            dr_cs_tf_gap,
            dr_tf_inboard,
        )


class TfInboardRadiiNoCsPrecomp(ExplicitFunction):
    """cottax node: `calculate_r_tf_inboard_radii_no_cs_precomp`."""

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)

    def __call__(
        self,
        dr_bore=From(build),
        dr_cs=From(build),
        dr_cs_tf_gap=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_r_tf_inboard_radii_no_cs_precomp(
            dr_bore,
            dr_cs,
            dr_cs_tf_gap,
            dr_tf_inboard,
        )


class ShldInboardInnerRadius(ExplicitFunction):
    """cottax node: `calculate_r_shld_inboard_inner`. No switch."""

    r_shld_inboard_inner = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_inboard=From(build),
    ):
        return calculate_r_shld_inboard_inner(
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            dr_fw_inboard,
            dr_blkt_inboard,
            dr_shld_inboard,
        )


class ShldOutboardOuterRadius(ExplicitFunction):
    """cottax node: `calculate_r_shld_outboard_outer`. No switch."""

    r_shld_outboard_outer = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
    ):
        return calculate_r_shld_outboard_outer(
            rmajor,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dr_blkt_outboard,
            dr_shld_outboard,
        )


class DrTfOutboardSuperconducting(ExplicitFunction):
    """cottax node: `calculate_dr_tf_outboard_superconducting`."""

    dr_tf_outboard = OutputInto(build)

    def __call__(self, dr_tf_inboard=From(build)):
        return calculate_dr_tf_outboard_superconducting(dr_tf_inboard)


class WpConductorMaxWidthSuperconducting(ExplicitFunction):
    """cottax node: `calculate_dx_tf_wp_conductor_max_superconducting`."""

    dx_tf_wp_conductor_max = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_wp_primary_toroidal=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return calculate_dx_tf_wp_conductor_max_superconducting(
            dx_tf_wp_primary_toroidal, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
        )


class TfOutboardMidUnrippled(ExplicitFunction):
    """cottax node: `calculate_r_tf_outboard_mid_unrippled`."""

    r_tf_outboard_mid_unrippled = OutputInto(build)

    def __call__(
        self,
        r_shld_outboard_outer=From(build),
        dr_shld_blkt_gap=From(build),
        dr_vv_outboard=From(build),
        gapomin=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_r_tf_outboard_mid_unrippled(
            r_shld_outboard_outer,
            dr_shld_blkt_gap,
            dr_vv_outboard,
            gapomin,
            dr_shld_thermal_outboard,
            dr_tf_shld_gap,
            dr_tf_outboard,
        )


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


class ShldVvGapOutboard(ExplicitFunction):
    """cottax node: `calculate_dr_shld_vv_gap_outboard`."""

    dr_shld_vv_gap_outboard = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        dr_vv_outboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_blkt_gap=From(build),
    ):
        return calculate_dr_shld_vv_gap_outboard(
            r_tf_outboard_mid,
            dr_tf_outboard,
            dr_vv_outboard,
            r_shld_outboard_outer,
            dr_shld_thermal_outboard,
            dr_tf_shld_gap,
            dr_shld_blkt_gap,
        )


class TfInnerBore(ExplicitFunction):
    """cottax node: `calculate_dr_tf_inner_bore`."""

    dr_tf_inner_bore = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        r_tf_inboard_mid=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_dr_tf_inner_bore(
            r_tf_outboard_mid, dr_tf_outboard, r_tf_inboard_mid, dr_tf_inboard
        )


class VacuumVesselAndShieldRadiiTfOutsideCs(ExplicitFunction):
    """cottax node: `calculate_vacuum_vessel_and_shield_radii`."""

    r_vv_inboard_out = OutputInto(build)
    r_sh_inboard_in = OutputInto(build)
    r_sh_inboard_out = OutputInto(build)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_thermal_inboard=From(build),
        dr_shld_vv_gap_inboard=From(build),
        dr_vv_inboard=From(build),
        dr_shld_inboard=From(build),
    ):
        return calculate_vacuum_vessel_and_shield_radii(
            r_tf_inboard_out,
            dr_tf_shld_gap,
            dr_shld_thermal_inboard,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            dr_shld_inboard,
        )


class RadialBuildToPlasmaCentre(ExplicitFunction):
    """cottax node: `calculate_rbld`."""

    rbld = OutputInto(build)

    def __call__(
        self,
        r_sh_inboard_out=From(build),
        dr_shld_blkt_gap=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
    ):
        return calculate_rbld(
            r_sh_inboard_out,
            dr_shld_blkt_gap,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
        )


class RCpTopFromTfInboardOut(ExplicitFunction):
    """cottax node: `calculate_r_cp_top_from_tf_inboard_out`, ports declared."""

    r_cp_top = OutputInto(build)

    def __call__(self, r_tf_inboard_out=From(build)):
        return calculate_r_cp_top_from_tf_inboard_out(r_tf_inboard_out)
