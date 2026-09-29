"""Pure-functional port of `Stellarator.st_phys`'s genuinely-new sub-computations (chunk
1B of unit #1).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    physics,
)
from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.models.stellarator.plasma_physics import (
    calculate_clipped_radiation_powers,
    calculate_fusion_power_totals_mw,
    calculate_fusion_totals_no_beam,
    calculate_heating_and_radiation_power,  # noqa: F401
    calculate_heating_and_radiation_power_ignited,
    calculate_heating_and_radiation_power_non_ignited,
    calculate_neutron_wall_load,  # noqa: F401
    calculate_neutron_wall_load_first_wall_area_comprehensive_2014,
    calculate_neutron_wall_load_first_wall_area_pre_2014,
    calculate_neutron_wall_load_scaled_plasma_surface,
    calculate_poloidal_field_from_rotational_transform,
    calculate_radiated_wall_load_and_fraction,  # noqa: F401
    calculate_radiated_wall_load_first_wall_area_comprehensive_2014,
    calculate_radiated_wall_load_first_wall_area_pre_2014,
    calculate_radiated_wall_load_scaled_plasma_surface,
    calculate_stellarator_beta_and_rho_star,
    calculate_thermal_energy_totals,
    calculate_total_field,
    select_stellarator_beta_and_stored_energy,
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)

# cottax node: `calculate_total_field`, ports declared.
TotalField = physics.function(calculate_total_field, "b_plasma_total")


# cottax node: `calculate_poloidal_field_from_rotational_transform`, ports
# declared.
PoloidalFieldFromRotationalTransform = physics.function(
    calculate_poloidal_field_from_rotational_transform,
    "b_plasma_surface_poloidal_average",
)


# cottax node: `calculate_stellarator_beta_and_rho_star`, ports declared.
StellaratorBetaAndRhoStar = physics.function(
    calculate_stellarator_beta_and_rho_star,
    "beta_total_vol_avg",
    "e_plasma_beta",
    "rho_star",
)


# cottax node: `calculate_fusion_power_totals_mw`, ports declared.
FusionPowerTotalsMw = physics.function(
    calculate_fusion_power_totals_mw,
    "p_plasma_dt_mw",
    "p_dhe3_total_mw",
    "p_dd_total_mw",
)


# cottax node: `calculate_fusion_totals_no_beam`, ports declared.
FusionTotalsNoBeam = physics.function(
    calculate_fusion_totals_no_beam,
    "fusden_total",
    "fusden_alpha_total",
    "p_dt_total_mw",
)


# cottax node: `calculate_clipped_radiation_powers`, ports declared.
ClippedRadiationPowers = physics.function(
    calculate_clipped_radiation_powers,
    "pden_plasma_core_rad_mw",
    "pden_plasma_outer_rad_mw",
    "p_plasma_inner_rad_mw",
    "p_plasma_outer_rad_mw",
)


# The NeutronWallLoad family:
# The `calculate_neutron_wall_load` family -- one occupant per arm of the
# `.physics.i_pflux_fw_neutron` x `.heat_transport.ipowerflow` dispatch.


# `i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- PROCESS's own default
# (`physics_variables.py:1006`) and the reference run's.
NeutronWallLoadScaledPlasmaSurface = physics.function(
    calculate_neutron_wall_load_scaled_plasma_surface, "pflux_fw_neutron_mw"
)


# `i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0).
NeutronWallLoadFirstWallAreaPre2014 = physics.function(
    calculate_neutron_wall_load_first_wall_area_pre_2014,
    "pflux_fw_neutron_mw",
)


# `i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow ==
# COMPREHENSIVE_2014` (1) -- PROCESS's own `ipowerflow` default.
NeutronWallLoadFirstWallAreaComprehensive2014 = physics.function(
    calculate_neutron_wall_load_first_wall_area_comprehensive_2014,
    "pflux_fw_neutron_mw",
)


# The HeatingAndRadiationPower family:
# The `calculate_heating_and_radiation_power` family -- one occupant per
# `.physics.i_plasma_ignited` value.


# `i_plasma_ignited == IGNITED` (1) -- the reference run's.
HeatingAndRadiationPowerIgnited = physics.function(
    calculate_heating_and_radiation_power_ignited,
    "p_plasma_rad_mw",
    "psolradmw",
    "p_plasma_separatrix_mw",
    "p_fw_alpha_mw",
)


# `i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
# (`physics_variables.py:881`).
HeatingAndRadiationPowerNonIgnited = physics.function(
    calculate_heating_and_radiation_power_non_ignited,
    "p_plasma_rad_mw",
    "psolradmw",
    "p_plasma_separatrix_mw",
    "p_fw_alpha_mw",
)


# The RadiatedWallLoadAndFraction family:
# The `calculate_radiated_wall_load_and_fraction` family -- the same three arms as
# `NeutronWallLoad`, applied to `.physics.p_plasma_rad_mw`.


# `i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- the reference run's.
RadiatedWallLoadScaledPlasmaSurface = physics.function(
    calculate_radiated_wall_load_scaled_plasma_surface,
    "pflux_fw_rad_mw",
    V.constraints.pflux_fw_rad_max_mw,
    "rad_fraction_total",
)


# `i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0).
RadiatedWallLoadFirstWallAreaPre2014 = physics.function(
    calculate_radiated_wall_load_first_wall_area_pre_2014,
    "pflux_fw_rad_mw",
    V.constraints.pflux_fw_rad_max_mw,
    "rad_fraction_total",
)


# `i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow ==
# COMPREHENSIVE_2014` (1).
RadiatedWallLoadFirstWallAreaComprehensive2014 = physics.function(
    calculate_radiated_wall_load_first_wall_area_comprehensive_2014,
    "pflux_fw_rad_mw",
    V.constraints.pflux_fw_rad_max_mw,
    "rad_fraction_total",
)


# cottax node: `calculate_thermal_energy_totals`, ports declared.
ThermalEnergyTotals = physics.function(
    calculate_thermal_energy_totals,
    "eden_plasma_thermal_vol_avg",
    "e_plasma_thermal_total",
)


def select_stellarator_beta_and_stored_energy(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    vol_plasma,
    m_ions_total_amu,
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    eps,
    rmajor,
):
    """`(beta_total_vol_avg, e_plasma_beta)` half of
    `calculate_stellarator_beta_and_rho_star` -- `rho_star` is `DimensionlessPlasma
    Parameters`' output (see `StellaratorBetaAndStoredEnergy`'s own docstring for why
    this class must not also produce it), so it is discarded here exactly as the
    declaration already discarded it.
    """
    beta_total_vol_avg, e_plasma_beta, _rho_star = (
        calculate_stellarator_beta_and_rho_star(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )
    )
    return beta_total_vol_avg, e_plasma_beta


# cottax node: `calculate_stellarator_beta_and_rho_star` minus its `rho_star`
# output -- the registerable form of `StellaratorBetaAndRhoStar` above.
StellaratorBetaAndStoredEnergy = physics.function(
    select_stellarator_beta_and_stored_energy, "beta_total_vol_avg", "e_plasma_beta"
)
