"""Pure-functional port of the **tokamak arm** of `process/models/physics/physics.py`."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import current_drive, physics, times
from functional_process.cottax.wraps import WrapsFunction
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


class SurfaceAveragedPoloidalField(ExplicitFunction):
    """The family that owns `.physics.b_plasma_surface_poloidal_average`."""


class SurfaceAveragedPoloidalFieldAmperes(SurfaceAveragedPoloidalField, WrapsFunction):
    """`i_plasma_current != PENG_DIVERTOR_SCALING`: Ampere's law over the perimeter."""

    fn = calculate_surface_averaged_poloidal_field_amperes

    cur_plasma = FromExactly(physics.plasma_current)
    len_plasma_poloidal = From(physics)

    b_plasma_surface_poloidal_average = OutputInto(physics)


class UnclippedRadiationPowers(WrapsFunction):
    """cottax node: `calculate_unclipped_radiation_powers`, ports declared."""

    fn = calculate_unclipped_radiation_powers

    pden_plasma_core_rad_mw_unclipped = From(physics)
    pden_plasma_outer_rad_mw_unclipped = From(physics)
    vol_plasma = From(physics)

    pden_plasma_core_rad_mw = OutputInto(physics)
    pden_plasma_outer_rad_mw = OutputInto(physics)
    p_plasma_inner_rad_mw = OutputInto(physics)
    p_plasma_outer_rad_mw = OutputInto(physics)


class TotalRadiationPower(WrapsFunction):
    """cottax node: `calculate_total_radiation_power`, ports declared."""

    fn = calculate_total_radiation_power

    pden_plasma_rad_mw = From(physics)
    vol_plasma = From(physics)

    p_plasma_rad_mw = OutputInto(physics)


class SeparatrixPower(ExplicitFunction):
    """The family that owns `.physics.p_plasma_separatrix_mw_raw`."""


class SeparatrixPowerNonIgnited(SeparatrixPower, WrapsFunction):
    """`i_plasma_ignited == NON_IGNITED`: injected heating crosses the separatrix."""

    fn = calculate_separatrix_power

    f_p_alpha_plasma_deposited = From(physics)
    p_alpha_total_mw = From(physics)
    p_non_alpha_charged_mw = From(physics)
    p_hcd_injected_total_mw = From(current_drive)
    p_plasma_ohmic_mw = From(physics)
    p_plasma_rad_mw = From(physics)

    p_plasma_separatrix_mw_raw = OutputInto(physics)


class PositiveSeparatrixPower(WrapsFunction):
    """cottax node: `force_positive_separatrix_power`, ports declared."""

    fn = force_positive_separatrix_power

    p_plasma_separatrix_mw_raw = From(physics)

    p_plasma_separatrix_mw = OutputInto(physics)


class PulseRampTimes(ExplicitFunction):
    """The family that owns the plasma-current ramp times, `physics.py:463-498`."""


class PulseRampTimesPulsedDefault(PulseRampTimes, WrapsFunction):
    """`i_pulsed_plant == 1` and `pulsetimings == 0` -- `large_tokamak_eval`'s arm."""

    fn = calculate_pulsed_plant_ramp_times

    plasma_current = From(physics)

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)


class PulseRampTimesContinuousDefault(PulseRampTimes, WrapsFunction):
    """`i_pulsed_plant != 1` and `i_t_current_ramp_up == 0`.

    The spherical tokamaks'.
    """

    fn = calculate_continuous_plant_ramp_times

    plasma_current = From(physics)

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_coil_precharge = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)


class PlasmaEnergyFromBeta(WrapsFunction):
    """cottax node: `calculate_plasma_energy_from_beta`, total-beta binding."""

    fn = calculate_plasma_energy_from_beta

    beta = FromExactly(physics.beta_total_vol_avg)
    b_field = FromExactly(physics.b_plasma_total)
    vol_plasma = From(physics)

    e_plasma_beta = OutputInto(physics)


class PlasmaOhmicHeating(WrapsFunction):
    """cottax node: `plasma_ohmic_heating`."""

    fn = plasma_ohmic_heating

    f_c_plasma_inductive = From(physics)
    kappa95 = From(physics)
    plasma_current = From(physics)
    rmajor = From(physics)
    rminor = From(physics)
    temp_plasma_electron_density_weighted_kev = From(physics)
    vol_plasma = From(physics)
    zeff = FromExactly(physics.n_charge_plasma_effective_vol_avg)
    plasma_res_factor = From(physics)

    pden_plasma_ohmic_mw = OutputInto(physics)
    p_plasma_ohmic_mw = OutputInto(physics)
    f_res_plasma_neo = OutputInto(physics)
    res_plasma = OutputInto(physics)


class CoulombLogarithmIonElectron(WrapsFunction):
    """cottax node: `calculate_coulomb_logarithm_ion_electron`."""

    fn = calculate_coulomb_logarithm_ion_electron

    nd_plasma_electrons_vol_avg = From(physics)
    temp_plasma_electron_vol_avg_kev = From(physics)

    dlamie = OutputInto(physics)


class PlasmaSurfaceNeutronFlux(WrapsFunction):
    """cottax node: `calculate_pflux_plasma_surface_neutron_avg_mw`."""

    fn = calculate_pflux_plasma_surface_neutron_avg_mw

    p_neutron_total_mw = From(physics)
    a_plasma_surface = From(physics)

    pflux_plasma_surface_neutron_avg_mw = OutputInto(physics)


class BetaNormMaxWesson(WrapsFunction):
    """cottax node: `calculate_beta_norm_max_wesson`."""

    fn = calculate_beta_norm_max_wesson

    ind_plasma_internal_norm = From(physics)

    beta_norm_max = OutputInto(physics)


class BetaLimitFromNorm(WrapsFunction):
    """cottax node: `calculate_beta_limit_from_norm`. Unswitched."""

    fn = calculate_beta_limit_from_norm

    b_plasma_toroidal_on_axis = From(physics)
    beta_norm_max = From(physics)
    plasma_current = From(physics)
    rminor = From(physics)

    beta_vol_avg_max = OutputInto(physics)


class ToroidalBeta(WrapsFunction):
    """cottax node: `calculate_toroidal_beta`. Unswitched."""

    fn = calculate_toroidal_beta

    beta_total_vol_avg = From(physics)
    b_plasma_total = From(physics)
    b_plasma_toroidal_on_axis = From(physics)

    beta_toroidal_vol_avg = OutputInto(physics)


class PoloidalBeta(WrapsFunction):
    """cottax node: `calculate_poloidal_beta`."""

    fn = calculate_poloidal_beta

    b_plasma_total = From(physics)
    b_plasma_poloidal_average = FromExactly(physics.b_plasma_surface_poloidal_average)
    beta = FromExactly(physics.beta_total_vol_avg)

    beta_poloidal_vol_avg = OutputInto(physics)


class ThermalBeta(WrapsFunction):
    """cottax node: `calculate_thermal_beta`."""

    fn = calculate_thermal_beta

    beta_total_vol_avg = From(physics)
    beta_fast_alpha = From(physics)
    beta_beam = From(physics)

    beta_thermal_vol_avg = OutputInto(physics)
