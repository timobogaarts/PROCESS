"""Pure-functional port of the **tokamak arm** of `process/models/physics/physics.py`."""

from cottax.assemblies.jax import V

from functional_process.cottax.paths import physics, times
from functional_process.models.physics.physics import (
    calculate_beta_limit_from_norm,
    calculate_beta_norm_max_wesson,
    calculate_continuous_plant_ramp_times,
    calculate_coulomb_logarithm_ion_electron,
    calculate_pflux_plasma_surface_neutron_avg_mw,
    calculate_plasma_energy_from_beta,
    calculate_poloidal_beta,
    calculate_pulsed_plant_ramp_times,
    calculate_separatrix_power,
    calculate_surface_averaged_poloidal_field_amperes,
    calculate_thermal_beta,
    calculate_toroidal_beta,
    calculate_total_radiation_power,
    calculate_unclipped_radiation_powers,
    force_positive_separatrix_power,
    plasma_ohmic_heating,
)

# The SurfaceAveragedPoloidalField family:
# The family that owns `.physics.b_plasma_surface_poloidal_average`.


# `i_plasma_current != PENG_DIVERTOR_SCALING`: Ampere's law over the perimeter.
SurfaceAveragedPoloidalFieldAmperes = physics.function(
    calculate_surface_averaged_poloidal_field_amperes,
    "b_plasma_surface_poloidal_average",
    reads={
        "cur_plasma": V.physics.plasma_current,
    },
)


# cottax node: `calculate_unclipped_radiation_powers`, ports declared.
UnclippedRadiationPowers = physics.function(
    calculate_unclipped_radiation_powers,
    "pden_plasma_core_rad_mw",
    "pden_plasma_outer_rad_mw",
    "p_plasma_inner_rad_mw",
    "p_plasma_outer_rad_mw",
)


# cottax node: `calculate_total_radiation_power`, ports declared.
TotalRadiationPower = physics.function(
    calculate_total_radiation_power, "p_plasma_rad_mw"
)


# The SeparatrixPower family:
# The family that owns `.physics.p_plasma_separatrix_mw_raw`.


# `i_plasma_ignited == NON_IGNITED`: injected heating crosses the separatrix.
SeparatrixPowerNonIgnited = physics.function(
    calculate_separatrix_power,
    "p_plasma_separatrix_mw_raw",
)


# cottax node: `force_positive_separatrix_power`, ports declared.
PositiveSeparatrixPower = physics.function(
    force_positive_separatrix_power, "p_plasma_separatrix_mw"
)


# The PulseRampTimes family:
# The family that owns the plasma-current ramp times, `physics.py:463-498`.


# `i_pulsed_plant == 1` and `pulsetimings == 0` -- `large_tokamak_eval`'s arm.
PulseRampTimesPulsedDefault = times.function(
    calculate_pulsed_plant_ramp_times,
    "t_plant_pulse_plasma_current_ramp_up",
    "t_plant_pulse_plasma_current_ramp_down",
)


# `i_pulsed_plant != 1` and `i_t_current_ramp_up == 0`.
#
# The spherical tokamaks'.
PulseRampTimesContinuousDefault = times.function(
    calculate_continuous_plant_ramp_times,
    "t_plant_pulse_plasma_current_ramp_up",
    "t_plant_pulse_coil_precharge",
    "t_plant_pulse_plasma_current_ramp_down",
)


# cottax node: `calculate_plasma_energy_from_beta`, total-beta binding.
PlasmaEnergyFromBeta = physics.function(
    calculate_plasma_energy_from_beta,
    "e_plasma_beta",
    reads={
        "beta": V.physics.beta_total_vol_avg,
        "b_field": V.physics.b_plasma_total,
    },
)


# cottax node: `plasma_ohmic_heating`.
PlasmaOhmicHeating = physics.function(
    plasma_ohmic_heating,
    "pden_plasma_ohmic_mw",
    "p_plasma_ohmic_mw",
    "f_res_plasma_neo",
    "res_plasma",
    reads={
        "zeff": V.physics.n_charge_plasma_effective_vol_avg,
    },
)


# cottax node: `calculate_coulomb_logarithm_ion_electron`.
CoulombLogarithmIonElectron = physics.function(
    calculate_coulomb_logarithm_ion_electron, "dlamie"
)


# cottax node: `calculate_pflux_plasma_surface_neutron_avg_mw`.
PlasmaSurfaceNeutronFlux = physics.function(
    calculate_pflux_plasma_surface_neutron_avg_mw, "pflux_plasma_surface_neutron_avg_mw"
)


# cottax node: `calculate_beta_norm_max_wesson`.
BetaNormMaxWesson = physics.function(calculate_beta_norm_max_wesson, "beta_norm_max")


# cottax node: `calculate_beta_limit_from_norm`. Unswitched.
BetaLimitFromNorm = physics.function(calculate_beta_limit_from_norm, "beta_vol_avg_max")


# cottax node: `calculate_toroidal_beta`. Unswitched.
ToroidalBeta = physics.function(calculate_toroidal_beta, "beta_toroidal_vol_avg")


# cottax node: `calculate_poloidal_beta`.
PoloidalBeta = physics.function(
    calculate_poloidal_beta,
    "beta_poloidal_vol_avg",
    reads={
        "b_plasma_poloidal_average": V.physics.b_plasma_surface_poloidal_average,
        "beta": V.physics.beta_total_vol_avg,
    },
)


# cottax node: `calculate_thermal_beta`.
ThermalBeta = physics.function(calculate_thermal_beta, "beta_thermal_vol_avg")
