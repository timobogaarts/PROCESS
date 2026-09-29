"""Pure-functional port of `process/models/blankets/hcpb.py`'s `CCFE_HCPB`."""

import jax.numpy as jnp
from cottax.interfaces import V

from functional_process.cottax.paths import (
    build,
    ccfe_hcpb,
    current_drive,
    divertor,
    first_wall,
    fwbs,
    heat_transport,
    physics,
    primary_pumping,
    tfcoil,
)
from functional_process.cottax.stated import states
from functional_process.models.blankets.hcpb import (
    calculate_centrepost_angle_fraction,
    calculate_centrepost_fast_neutron_flux_superconducting,
    calculate_centrepost_neutronics_absent,
    calculate_centrepost_neutronics_spherical_tokamak_superconducting,
    calculate_centrepost_nuclear_heating_superconducting,
    calculate_component_masses,
    calculate_divertor_surface_and_plate_mass_double_null,
    calculate_divertor_surface_and_plate_mass_single_null,
    calculate_first_wall_radiation_powers,
    calculate_fw_coolant_void_fractions,
    calculate_nuclear_heating_magnets_conventional,
    calculate_nuclear_heating_magnets_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_double_null_conventional,
    calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_single_null_conventional,
    calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak,
    calculate_pumping_power_mechanical_with_pressure_drop,
    nuclear_heating_blanket,
    nuclear_heating_fw,
    nuclear_heating_shield_conventional,
    nuclear_heating_shield_spherical_tokamak,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.vocabulary import constants

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `jnp`/`constants`/
# `safe_pow`/`safe_sqrt` and the four functions no surviving declaration calls directly
# (`calculate_centrepost_neutronics_absent`, `calculate_centrepost_angle_fraction`,
# `calculate_centrepost_fast_neutron_flux_superconducting`,
# `calculate_centrepost_nuclear_heating_superconducting` -- the last three are called
# only from inside `calculate_centrepost_neutronics_spherical_tokamak_superconducting`,
# not from any node), which are the only names actually unused here (see
# `power/electric_production.py`'s commit for why a partial list is the wrong move).
__all__ = [
    "CentrepostNeutronicsAbsent",
    "CentrepostNeutronicsSphericalTokamakSuperconducting",
    "ComponentMasses",
    "DivertorSurfaceAndPlateMassDoubleNull",
    "DivertorSurfaceAndPlateMassSingleNull",
    "FirstWallCoolantVoidFractions",
    "FirstWallRadiationPowers",
    "NuclearHeatingBlanket",
    "NuclearHeatingFw",
    "NuclearHeatingMagnetsConventional",
    "NuclearHeatingMagnetsSphericalTokamak",
    "NuclearHeatingRenormalisationDoubleNullConventional",
    "NuclearHeatingRenormalisationDoubleNullSphericalTokamak",
    "NuclearHeatingRenormalisationSingleNullConventional",
    "NuclearHeatingRenormalisationSingleNullSphericalTokamak",
    "NuclearHeatingShieldConventional",
    "NuclearHeatingShieldSphericalTokamak",
    "PumpingPowerMechanicalWithPressureDrop",
    "build",
    "calculate_centrepost_angle_fraction",
    "calculate_centrepost_fast_neutron_flux_superconducting",
    "calculate_centrepost_neutronics_absent",
    "calculate_centrepost_neutronics_spherical_tokamak_superconducting",
    "calculate_centrepost_nuclear_heating_superconducting",
    "calculate_component_masses",
    "calculate_divertor_surface_and_plate_mass_double_null",
    "calculate_divertor_surface_and_plate_mass_single_null",
    "calculate_first_wall_radiation_powers",
    "calculate_fw_coolant_void_fractions",
    "calculate_nuclear_heating_magnets_conventional",
    "calculate_nuclear_heating_magnets_spherical_tokamak",
    "calculate_nuclear_heating_renormalisation_double_null_conventional",
    "calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak",
    "calculate_nuclear_heating_renormalisation_single_null_conventional",
    "calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak",
    "calculate_pumping_power_mechanical_with_pressure_drop",
    "ccfe_hcpb",
    "constants",
    "current_drive",
    "divertor",
    "first_wall",
    "fwbs",
    "heat_transport",
    "jnp",
    "nuclear_heating_blanket",
    "nuclear_heating_fw",
    "nuclear_heating_shield_conventional",
    "nuclear_heating_shield_spherical_tokamak",
    "physics",
    "primary_pumping",
    "safe_pow",
    "safe_sqrt",
    "tfcoil",
]


# cottax node: `calculate_fw_coolant_void_fractions`.
FirstWallCoolantVoidFractions = fwbs.function(
    calculate_fw_coolant_void_fractions,
    "f_a_fw_coolant_inboard",
    "f_a_fw_coolant_outboard",
)


# The DivertorSurfaceAndPlateMass family:
# The family that owns `.divertor.a_div_surface_total` and `.divertor.m_div_plate`:
# one occupant per `n_divertors` arm of `component_masses`' `hcpb.py:353-367`.


# cottax node: `calculate_divertor_surface_and_plate_mass_single_null`.
DivertorSurfaceAndPlateMassSingleNull = divertor.function(
    calculate_divertor_surface_and_plate_mass_single_null,
    "a_div_surface_total",
    "m_div_plate",
)


# cottax node: `calculate_divertor_surface_and_plate_mass_double_null`.
DivertorSurfaceAndPlateMassDoubleNull = divertor.function(
    calculate_divertor_surface_and_plate_mass_double_null,
    "a_div_surface_total",
    "m_div_plate",
)


# cottax node: `calculate_component_masses`.
ComponentMasses = fwbs.function(
    calculate_component_masses,
    "m_fw_blkt_div_coolant_total",
    "fwclfr",
    "whtshld",
    "wpenshld",
    "vol_fw_total",
    "m_fw_total",
    "fw_armour_vol",
    "fw_armour_mass",
    "f_vol_blkt_li4sio4",
    "f_vol_blkt_tibe12",
    "m_blkt_tibe12",
    "m_blkt_li4sio4",
    "m_blkt_beryllium",
    "m_blkt_li2o",
    "f_vol_blkt_steel",
    "m_blkt_steel_total",
    "m_blkt_total",
    "armour_fw_bl_mass",
)


# The NuclearHeatingMagnets family:
# The family that owns the nine `nuclear_heating_magnets` outputs: one occupant per
# value of `.physics.itart` (`hcpb.py:495-575`).


# cottax node: `calculate_nuclear_heating_magnets_conventional`. `itart == 0`.
NuclearHeatingMagnetsConventional = ccfe_hcpb.function(
    calculate_nuclear_heating_magnets_conventional,
    "armour_density",
    "fw_density",
    "blanket_density",
    "shield_density",
    "vv_density",
    "x_blanket",
    "x_shield",
    "tfc_nuc_heating",
    "p_tf_nuclear_heat_mw_unnormalised",
)


# cottax node: `calculate_nuclear_heating_magnets_spherical_tokamak`.
NuclearHeatingMagnetsSphericalTokamak = ccfe_hcpb.function(
    calculate_nuclear_heating_magnets_spherical_tokamak,
    "armour_density",
    "fw_density",
    "blanket_density",
    "shield_density",
    "vv_density",
    "x_blanket",
    "x_shield",
    "tfc_nuc_heating",
    "p_tf_nuclear_heat_mw_unnormalised",
)


# cottax node: `nuclear_heating_fw`, unchanged.
NuclearHeatingFw = ccfe_hcpb.function(
    nuclear_heating_fw,
    "p_fw_nuclear_heat_total_mw_unnormalised",
)


# cottax node: `nuclear_heating_blanket`, unchanged.
NuclearHeatingBlanket = ccfe_hcpb.function(
    nuclear_heating_blanket,
    "p_blkt_nuclear_heat_total_mw_unnormalised",
    "exp_blanket",
)


# The NuclearHeatingShield family:
# The family that owns the four `nuclear_heating_shield` outputs: one occupant per
# value of `.physics.itart` (`hcpb.py:748-769`).


# cottax node: `nuclear_heating_shield_conventional`.
NuclearHeatingShieldConventional = ccfe_hcpb.function(
    nuclear_heating_shield_conventional,
    "p_shld_nuclear_heat_mw_unnormalised",
    "exp_shield1",
    "exp_shield2",
    "shld_u_nuc_heating",
)


# cottax node: `nuclear_heating_shield_spherical_tokamak`.
NuclearHeatingShieldSphericalTokamak = ccfe_hcpb.function(
    nuclear_heating_shield_spherical_tokamak,
    "p_shld_nuclear_heat_mw_unnormalised",
    "exp_shield1",
    "exp_shield2",
    "shld_u_nuc_heating",
)


# The CentrepostNeutronics family:
# The family that owns `run()`'s centrepost block (`hcpb.py:103-148`): one occupant
# per cell of the joint `(itart, i_tf_sup)` arm PROCESS's three `st_*` routines cut
# between them.


# cottax node: `calculate_centrepost_neutronics_absent`.
CentrepostNeutronicsAbsent = states(
    fwbs, "pnuc_cp_tf", "p_cp_shield_nuclear_heat_mw", "pnuc_cp", "neut_flux_cp"
)


# cottax node: `calculate_centrepost_neutronics_spherical_tokamak_superconducting`.
CentrepostNeutronicsSphericalTokamakSuperconducting = ccfe_hcpb.function(
    calculate_centrepost_neutronics_spherical_tokamak_superconducting,
    "f_geom_cp",
    V.fwbs.neut_flux_cp,
    V.fwbs.pnuc_cp_tf,
    "p_cp_shield_nuclear_heat_mw_fit",
    V.fwbs.pnuc_cp,
)


# The NuclearHeatingRenormalisation family:
# The family that owns the four renormalised nuclear-heating powers: one occupant
# per cell of the `(n_divertors, itart)` pair `hcpb.py:195-276` branches on.


# cottax node:
# `calculate_nuclear_heating_renormalisation_single_null_conventional`.
NuclearHeatingRenormalisationSingleNullConventional = ccfe_hcpb.function(
    calculate_nuclear_heating_renormalisation_single_null_conventional,
    "pnuc_tot_blk_sector",
    V.fwbs.p_fw_nuclear_heat_total_mw,
    V.fwbs.p_blkt_nuclear_heat_total_mw,
    V.fwbs.p_shld_nuclear_heat_mw,
    V.fwbs.p_tf_nuclear_heat_mw,
    V.fwbs.p_blkt_multiplication_mw,
)


# cottax node:
# `calculate_nuclear_heating_renormalisation_double_null_conventional`.
NuclearHeatingRenormalisationDoubleNullConventional = ccfe_hcpb.function(
    calculate_nuclear_heating_renormalisation_double_null_conventional,
    "pnuc_tot_blk_sector",
    V.fwbs.p_fw_nuclear_heat_total_mw,
    V.fwbs.p_blkt_nuclear_heat_total_mw,
    V.fwbs.p_shld_nuclear_heat_mw,
    V.fwbs.p_tf_nuclear_heat_mw,
    V.fwbs.p_blkt_multiplication_mw,
)


# cottax node:
# `calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak`.
NuclearHeatingRenormalisationSingleNullSphericalTokamak = ccfe_hcpb.function(
    calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak,
    "pnuc_tot_blk_sector",
    V.fwbs.p_fw_nuclear_heat_total_mw,
    V.fwbs.p_blkt_nuclear_heat_total_mw,
    V.fwbs.p_shld_nuclear_heat_mw,
    V.fwbs.p_tf_nuclear_heat_mw,
    V.fwbs.p_cp_shield_nuclear_heat_mw,
    V.fwbs.p_blkt_multiplication_mw,
)


# cottax node:
# `calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak`.
NuclearHeatingRenormalisationDoubleNullSphericalTokamak = ccfe_hcpb.function(
    calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak,
    "pnuc_tot_blk_sector",
    V.fwbs.p_fw_nuclear_heat_total_mw,
    V.fwbs.p_blkt_nuclear_heat_total_mw,
    V.fwbs.p_shld_nuclear_heat_mw,
    V.fwbs.p_tf_nuclear_heat_mw,
    V.fwbs.p_cp_shield_nuclear_heat_mw,
    V.fwbs.p_blkt_multiplication_mw,
)


# cottax node: `calculate_first_wall_radiation_powers`.
FirstWallRadiationPowers = fwbs.function(
    calculate_first_wall_radiation_powers,
    "p_fw_hcd_rad_total_mw",
    "p_fw_rad_total_mw",
    "psurffwo",
    "psurffwi",
)


# cottax node: `calculate_pumping_power_mechanical_with_pressure_drop`.
PumpingPowerMechanicalWithPressureDrop = primary_pumping.function(
    calculate_pumping_power_mechanical_with_pressure_drop,
    "p_fw_blkt_coolant_pump_mw",
    V.heat_transport.p_shld_coolant_pump_mw,
    V.heat_transport.p_div_coolant_pump_mw,
)
