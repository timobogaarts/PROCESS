"""The namespaces of the model modules that are files rather than packages."""

import dataclasses

import equinox as eqx
from cottax.relational_pytree import Explicit

from functional_process.cottax.models.build import (
    BlktUpperThickness,
    PlasmaXpointHeights,
    RadialBuildToPlasmaCentre,
    ShldInboardInnerRadius,
    ShldOutboardOuterRadius,
    ShldVvGapOutboard,
    TfInnerBore,
    TfOutboardMidUnrippled,
    ZTfInsideHalf,
)
from functional_process.cottax.models.divertor import DivertorHeatFluxSplit


class Build(eqx.Module):
    """The tokamak's radial and vertical build -- nineteen slots, twenty-five classes."""

    plasma_xpoint_heights: Explicit = PlasmaXpointHeights
    """`.build.z_plasma_xpoint_upper`/`_lower`. Unswitched."""

    divertor_geometry: Explicit | None = dataclasses.field(kw_only=True)
    """`.physics.itart`, **and** the input `.build.dz_xpoint_divertor < 1e-5`."""

    z_tf_inside_half: Explicit = ZTfInsideHalf
    """`.build.z_tf_inside_half`, from the vertical stack at `build.py:807`."""

    tf_top_height: Explicit = dataclasses.field(kw_only=True)
    """`.physics.i_single_null` -- `.build.z_tf_top` and
    `.build.dz_tf_upper_lower_midplane`, both arms written (`build.py:820-841`).
    """

    blkt_upper_thickness: Explicit = BlktUpperThickness
    """`.build.dz_blkt_upper`, the mean of the two radial blanket thicknesses
    (`build.py:1664-1667`).
    """

    dr_tf_inboard_winding_pack: Explicit = dataclasses.field(kw_only=True)
    """Whether iteration variable 140 is active -- and the two arms own **different
    fields**, being exact inverses of one relation.
    """

    tf_inboard_radii: Explicit = dataclasses.field(kw_only=True)
    """`(.build.i_tf_inside_cs, .build.i_cs_precomp)` -- `(0, 1)` (both defaults, both
    live) is written; `TF_INSIDE_CS` and the no-precompression arm are UNPORTED
    (`indat._tf_inboard_radii_arm`).
    """

    vacuum_vessel_and_shield_radii: Explicit = dataclasses.field(kw_only=True)
    """`.build.i_tf_inside_cs` -- `TF_OUTSIDE_CS` is written, `TF_INSIDE_CS` UNPORTED
    (`indat.VACUUM_VESSEL_AND_SHIELD_RADII`).
    """

    radial_build_to_plasma_centre: Explicit = RadialBuildToPlasmaCentre
    """`.build.rbld`, PROCESS's own "should be equal to `rmajor`" accumulation, which is
    what constraint 11 says -- active on three of the four tracked tokamak files.
    """

    shld_inboard_inner_radius: Explicit = ShldInboardInnerRadius
    shld_outboard_outer_radius: Explicit = ShldOutboardOuterRadius
    """The two shield radii, built inwards and outwards from the plasma."""

    r_cp_top: Explicit = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_tf_sup)` -- every cell but the resistive spherical
    tokamak, which is `indat._r_cp_top_arm`'s UNPORTED arm `-1`.
    """

    dr_tf_outboard: Explicit = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_sup`."""

    wp_conductor_max_width: Explicit = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_sup`, and the owner of the mint `.tfcoil.dx_tf_wp_conductor_max`.
    """

    tf_outboard_mid_unrippled: Explicit = TfOutboardMidUnrippled
    """The mint `.build.r_tf_outboard_mid_unrippled`: the value PROCESS assigns to
    `.build.r_tf_outboard_mid` at `:1901` and then overwrites in place at `:1939`.
    """

    tf_outboard_mid: Explicit = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape`."""

    tf_outboard_edge_ripple: Explicit = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape`, and PROCESS's **second** call to the same ripple fit -- the
    one whose answer survives into `.tfcoil.ripple_b_tf_plasma_edge`.
    """

    shld_vv_gap_outboard: Explicit = ShldVvGapOutboard
    """`.build.dr_shld_vv_gap_outboard`."""

    tf_inner_bore: Explicit = TfInnerBore
    """`.build.dr_tf_inner_bore`, the midplane bore between the two TF legs
    (`build.py:1911-1913`, rewritten verbatim at `:1949-1955`).
    """


class Divertor(eqx.Module):
    """The tokamak divertor -- two nodes, one of them switched."""

    heat_flux_split: Explicit = DivertorHeatFluxSplit
    """`.fwbs.f_ster_div_single`, `.fwbs.p_div_nuclear_heat_total_mw`,
    `.fwbs.p_div_rad_total_mw` and `.divertor.deg_div_poloidal_plasma`.
    """

    heat_load: Explicit = dataclasses.field(kw_only=True)
    """`.divertor.i_div_heat_load` -- `2` (Wade) on `large_tokamak_eval.IN.DAT:139`."""
