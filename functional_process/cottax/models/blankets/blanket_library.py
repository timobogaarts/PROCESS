"""Pure-functional port of `process/models/blankets/blanket_library.py`'s tokamak
`component_volumes` chain.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.blankets.blanket_library import (
    apply_coverage_factors_double_null,
    apply_coverage_factors_single_null,
    calculate_blkt_half_height_double_null,
    calculate_blkt_half_height_single_null,
    calculate_blkt_inboard_poloidal_plasma_angle,
    calculate_dshaped_blkt_areas,
    calculate_dshaped_blkt_volumes,
    calculate_elliptical_blkt_areas,
    calculate_elliptical_blkt_volumes,
)
from functional_process.models.engineering.ivc_functions import dshellarea, dshellvol
from functional_process.cottax.paths import blanket, build, divertor, fwbs, physics

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `dshellarea`/
# `dshellvol`/`jnp`, which are unused now that their real uses left with the functions
# (see `power/electric_production.py`'s commit for why a partial list is the wrong move).
__all__ = [
    "BlanketAreas",
    "BlanketCoverageFactors",
    "BlanketCoverageFactorsDoubleNull",
    "BlanketCoverageFactorsSingleNull",
    "BlanketHalfHeight",
    "BlanketHalfHeightDoubleNull",
    "BlanketHalfHeightSingleNull",
    "BlanketInboardPoloidalAngle",
    "BlanketVolumes",
    "DShapedBlanketAreas",
    "DShapedBlanketVolumes",
    "EllipticalBlanketAreas",
    "EllipticalBlanketVolumes",
    "ExplicitFunction",
    "From",
    "OutputInto",
    "apply_coverage_factors_double_null",
    "apply_coverage_factors_single_null",
    "blanket",
    "build",
    "calculate_blkt_half_height_double_null",
    "calculate_blkt_half_height_single_null",
    "calculate_blkt_inboard_poloidal_plasma_angle",
    "calculate_dshaped_blkt_areas",
    "calculate_dshaped_blkt_volumes",
    "calculate_elliptical_blkt_areas",
    "calculate_elliptical_blkt_volumes",
    "divertor",
    "dshellarea",
    "dshellvol",
    "fwbs",
    "jnp",
    "physics",
]


class BlanketHalfHeight(ExplicitFunction):
    """The family that owns `.blanket.dz_blkt_half`: one occupant per `n_divertors` arm
    of `BlanketLibrary.calculate_blkt_half_height`.
    """


class BlanketHalfHeightSingleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_single_null`. `n_divertors == 1`."""

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
    ):
        return calculate_blkt_half_height_single_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
            z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            dr_fw_inboard,
            dr_fw_outboard,
        )


class BlanketHalfHeightDoubleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_double_null`."""

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
    ):
        return calculate_blkt_half_height_double_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
        )


class BlanketAreas(ExplicitFunction):
    """The family that owns the three `.build.a_blkt_*_full_coverage` fields: one
    occupant per arm of `component_volumes`' shape decision (`itart == 1 or
    i_fw_blkt_vv_shape == D_SHAPED`, `blanket_library.py:90-93`).
    """


class BlanketVolumes(ExplicitFunction):
    """The family that owns the three `.fwbs.vol_blkt_*_full_coverage` fields, on the
    same shape predicate as `BlanketAreas` and with the same two arms.
    """


class EllipticalBlanketAreas(BlanketAreas):
    """cottax node: `calculate_elliptical_blkt_areas`."""

    a_blkt_inboard_surface_full_coverage = OutputInto(build)
    a_blkt_outboard_surface_full_coverage = OutputInto(build)
    a_blkt_total_surface_full_coverage = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
    ):
        return calculate_elliptical_blkt_areas(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
        )


class DShapedBlanketAreas(BlanketAreas):
    """cottax node: `calculate_dshaped_blkt_areas`."""

    a_blkt_inboard_surface_full_coverage = OutputInto(build)
    a_blkt_outboard_surface_full_coverage = OutputInto(build)
    a_blkt_total_surface_full_coverage = OutputInto(build)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dz_blkt_half=From(blanket),
    ):
        return calculate_dshaped_blkt_areas(
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dz_blkt_half,
        )


class EllipticalBlanketVolumes(BlanketVolumes):
    """cottax node: `calculate_elliptical_blkt_volumes`. Same arm as the areas above."""

    vol_blkt_inboard_full_coverage = OutputInto(fwbs)
    vol_blkt_outboard_full_coverage = OutputInto(fwbs)
    vol_blkt_total_full_coverage = OutputInto(fwbs)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
        dz_blkt_upper=From(build),
    ):
        return calculate_elliptical_blkt_volumes(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
            dz_blkt_upper,
        )


class DShapedBlanketVolumes(BlanketVolumes):
    """cottax node: `calculate_dshaped_blkt_volumes`."""

    vol_blkt_inboard_full_coverage = OutputInto(fwbs)
    vol_blkt_outboard_full_coverage = OutputInto(fwbs)
    vol_blkt_total_full_coverage = OutputInto(fwbs)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dz_blkt_half=From(blanket),
        dr_blkt_outboard=From(build),
        dz_blkt_upper=From(build),
    ):
        return calculate_dshaped_blkt_volumes(
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dz_blkt_half,
            dr_blkt_outboard,
            dz_blkt_upper,
        )


class BlanketCoverageFactors(ExplicitFunction):
    """The family that owns `.fwbs.vol_blkt_total` and the five fields written beside
    it: one occupant per `n_divertors` arm of `BlanketLibrary.apply_coverage_factors`.
    """


class BlanketCoverageFactorsSingleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_single_null`. `n_divertors == 1`."""

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_single_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )


class BlanketCoverageFactorsDoubleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_double_null`."""

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_double_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )


class BlanketInboardPoloidalAngle(ExplicitFunction):
    """cottax node: `calculate_blkt_inboard_poloidal_plasma_angle`."""

    deg_blkt_inboard_poloidal_plasma = OutputInto(blanket)

    def __call__(
        self,
        rminor=From(physics),
        dz_blkt_half=From(blanket),
        dr_fw_plasma_gap_inboard=From(build),
    ):
        return calculate_blkt_inboard_poloidal_plasma_angle(
            rminor, dz_blkt_half, dr_fw_plasma_gap_inboard
        )
