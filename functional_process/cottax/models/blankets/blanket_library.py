"""Pure-functional port of `process/models/blankets/blanket_library.py`'s tokamak
`component_volumes` chain.
"""

import jax.numpy as jnp
from cottax.interfaces import V

from functional_process.cottax.paths import blanket, build, divertor, fwbs, physics
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


# The BlanketHalfHeight family:
# The family that owns `.blanket.dz_blkt_half`: one occupant per `n_divertors` arm
# of `BlanketLibrary.calculate_blkt_half_height`.


# cottax node: `calculate_blkt_half_height_single_null`. `n_divertors == 1`.
BlanketHalfHeightSingleNull = blanket.function(
    calculate_blkt_half_height_single_null,
    "dz_blkt_half",
    reads={
        "z_plasma_xpoint_lower": build,
        "dz_xpoint_divertor": build,
        "dz_divertor": divertor,
        "dz_blkt_upper": build,
        "z_plasma_xpoint_upper": build,
        "dr_fw_plasma_gap_inboard": build,
        "dr_fw_plasma_gap_outboard": build,
        "dr_fw_inboard": build,
        "dr_fw_outboard": build,
    },
)


# cottax node: `calculate_blkt_half_height_double_null`.
BlanketHalfHeightDoubleNull = blanket.function(
    calculate_blkt_half_height_double_null,
    "dz_blkt_half",
    reads={
        "z_plasma_xpoint_lower": build,
        "dz_xpoint_divertor": build,
        "dz_divertor": divertor,
        "dz_blkt_upper": build,
    },
)


# The BlanketAreas family:
# The family that owns the three `.build.a_blkt_*_full_coverage` fields: one
# occupant per arm of `component_volumes`' shape decision (`itart == 1 or
# i_fw_blkt_vv_shape == D_SHAPED`, `blanket_library.py:90-93`).


# The BlanketVolumes family:
# The family that owns the three `.fwbs.vol_blkt_*_full_coverage` fields, on the
# same shape predicate as `BlanketAreas` and with the same two arms.


# cottax node: `calculate_elliptical_blkt_areas`.
EllipticalBlanketAreas = build.function(
    calculate_elliptical_blkt_areas,
    "a_blkt_inboard_surface_full_coverage",
    "a_blkt_outboard_surface_full_coverage",
    "a_blkt_total_surface_full_coverage",
    reads={
        "rmajor": physics,
        "rminor": physics,
        "triang": physics,
        "dz_blkt_half": blanket,
    },
)


# cottax node: `calculate_dshaped_blkt_areas`.
DShapedBlanketAreas = build.function(
    calculate_dshaped_blkt_areas,
    "a_blkt_inboard_surface_full_coverage",
    "a_blkt_outboard_surface_full_coverage",
    "a_blkt_total_surface_full_coverage",
    reads={
        "rminor": physics,
        "dz_blkt_half": blanket,
    },
)


# cottax node: `calculate_elliptical_blkt_volumes`. Same arm as the areas above.
EllipticalBlanketVolumes = fwbs.function(
    calculate_elliptical_blkt_volumes,
    "vol_blkt_inboard_full_coverage",
    "vol_blkt_outboard_full_coverage",
    "vol_blkt_total_full_coverage",
    reads={
        "rmajor": physics,
        "rminor": physics,
        "triang": physics,
        "r_shld_inboard_inner": build,
        "dr_shld_inboard": build,
        "dr_blkt_inboard": build,
        "r_shld_outboard_outer": build,
        "dr_shld_outboard": build,
        "dr_blkt_outboard": build,
        "dz_blkt_half": blanket,
        "dz_blkt_upper": build,
    },
)


# cottax node: `calculate_dshaped_blkt_volumes`.
DShapedBlanketVolumes = fwbs.function(
    calculate_dshaped_blkt_volumes,
    "vol_blkt_inboard_full_coverage",
    "vol_blkt_outboard_full_coverage",
    "vol_blkt_total_full_coverage",
    reads={
        "r_shld_inboard_inner": build,
        "dr_shld_inboard": build,
        "dr_blkt_inboard": build,
        "dr_fw_inboard": build,
        "dr_fw_plasma_gap_inboard": build,
        "rminor": physics,
        "dr_fw_plasma_gap_outboard": build,
        "dr_fw_outboard": build,
        "dz_blkt_half": blanket,
        "dr_blkt_outboard": build,
        "dz_blkt_upper": build,
    },
)


# The BlanketCoverageFactors family:
# The family that owns `.fwbs.vol_blkt_total` and the five fields written beside
# it: one occupant per `n_divertors` arm of `BlanketLibrary.apply_coverage_factors`.


# cottax node: `apply_coverage_factors_single_null`. `n_divertors == 1`.
BlanketCoverageFactorsSingleNull = build.function(
    apply_coverage_factors_single_null,
    "a_blkt_outboard_surface",
    "a_blkt_total_surface",
    V.fwbs.vol_blkt_outboard,
    V.fwbs.vol_blkt_inboard,
    "a_blkt_inboard_surface",
    V.fwbs.vol_blkt_total,
    reads={
        "f_ster_div_single": fwbs,
        "f_a_fw_outboard_hcd": fwbs,
        "vol_blkt_total_full_coverage": fwbs,
        "vol_blkt_inboard_full_coverage": fwbs,
    },
)


# cottax node: `apply_coverage_factors_double_null`.
BlanketCoverageFactorsDoubleNull = build.function(
    apply_coverage_factors_double_null,
    "a_blkt_outboard_surface",
    "a_blkt_total_surface",
    V.fwbs.vol_blkt_outboard,
    V.fwbs.vol_blkt_inboard,
    "a_blkt_inboard_surface",
    V.fwbs.vol_blkt_total,
    reads={
        "f_ster_div_single": fwbs,
        "f_a_fw_outboard_hcd": fwbs,
        "vol_blkt_total_full_coverage": fwbs,
        "vol_blkt_inboard_full_coverage": fwbs,
    },
)


# cottax node: `calculate_blkt_inboard_poloidal_plasma_angle`.
BlanketInboardPoloidalAngle = blanket.function(
    calculate_blkt_inboard_poloidal_plasma_angle,
    "deg_blkt_inboard_poloidal_plasma",
    reads={
        "rminor": physics,
        "dr_fw_plasma_gap_inboard": build,
    },
)
