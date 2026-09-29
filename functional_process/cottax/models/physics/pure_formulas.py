"""Pure-functional port of five already-pure formulas from
`process/models/physics/physics.py`.
"""

from functional_process.cottax.paths import current_drive, physics
from functional_process.models.physics.pure_formulas import (
    calaculate_stored_thermal_energy,
    calculate_total_plasma_heating_power,
    fast_alpha_beta,
    fast_alpha_beta_iter_physics_rules,
    fast_alpha_beta_ward,
    phyaux,
    rether,
)

__all__ = [
    "fast_alpha_beta",
]


# cottax node: `rether`, unchanged, ports declared.
def _ion_electron_equilibration(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    dlamie,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
    n_charge_plasma_effective_mass_weighted_vol_avg,
):
    return rether(
        alphan,
        alphat,
        nd_plasma_electrons_vol_avg,
        dlamie,
        temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev,
        n_charge_plasma_effective_mass_weighted_vol_avg,
    )


IonElectronEquilibration = physics.function(
    _ion_electron_equilibration, "pden_ion_electron_equilibration_mw"
)


# cottax node: `phyaux`, ports declared.
def _auxiliary_physics_quantities(
    aspect,
    nd_plasma_fuel_ions_vol_avg,
    fusden_total,
    fusden_alpha_total,
    plasma_current,
    nd_plasma_alphas_thermal_vol_avg,
    t_energy_confinement,
    vol_plasma,
    burnup_in,
    tauratio,
):
    return phyaux(
        aspect,
        nd_plasma_fuel_ions_vol_avg,
        fusden_total,
        fusden_alpha_total,
        plasma_current,
        1.0,  # `sbar`: fixed at 1.0 in PROCESS, and never varied here
        nd_plasma_alphas_thermal_vol_avg,
        t_energy_confinement,
        vol_plasma,
        burnup_in,
        tauratio,
    )


AuxiliaryPhysicsQuantities = physics.function(
    _auxiliary_physics_quantities,
    "burnup",
    "figmer",
    "fusrat",
    "molflow_plasma_fuelling_required",
    "rndfuel",
    "t_alpha_confinement",
    "f_t_alpha_energy_confinement",
)


# cottax node: `calculate_total_plasma_heating_power`, unchanged, ports declared.
def _total_plasma_heating_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    return calculate_total_plasma_heating_power(
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        p_hcd_injected_total_mw,
    )


TotalPlasmaHeatingPower = physics.function(
    _total_plasma_heating_power,
    "p_plasma_heating_total_mw",
    reads={
        "p_hcd_injected_total_mw": current_drive,
    },
)


# cottax node: `calaculate_stored_thermal_energy`, electron binding.
def _electron_thermal_energy(
    vol_plasma, nd_plasma_electrons_vol_avg, temp_plasma_electron_density_weighted_kev
):
    return calaculate_stored_thermal_energy(
        vol_plasma,
        nd_plasma_electrons_vol_avg,
        temp_plasma_electron_density_weighted_kev,
    )


ElectronThermalEnergy = physics.function(
    _electron_thermal_energy,
    "eden_plasma_electrons_thermal_vol_avg",
    "e_plasma_electrons_thermal",
)


# cottax node: `calaculate_stored_thermal_energy`, ion binding.
def _ion_thermal_energy(
    vol_plasma, nd_plasma_ions_total_vol_avg, temp_plasma_ion_density_weighted_kev
):
    return calaculate_stored_thermal_energy(
        vol_plasma,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_ion_density_weighted_kev,
    )


IonThermalEnergy = physics.function(
    _ion_thermal_energy, "eden_plasma_ions_thermal_vol_avg", "e_plasma_ions_thermal"
)


# The FastAlphaBeta family:
# The `fast_alpha_beta` family -- one occupant per `.physics.i_beta_fast_alpha`
# value.


# `i_beta_fast_alpha == ITER_PHYSICS_RULES` (0) -- the ITER Physics Design
# Guidelines fraction (`physics.py:2043`).
def _fast_alpha_beta_iter_physics_rules(
    b_plasma_surface_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    f_plasma_fuel_deuterium,
):
    return fast_alpha_beta_iter_physics_rules(
        b_plasma_surface_poloidal_average,
        b_plasma_toroidal_on_axis,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        pden_alpha_total_mw,
        pden_plasma_alpha_mw,
        f_plasma_fuel_deuterium,
    )


FastAlphaBetaIterPhysicsRules = physics.function(
    _fast_alpha_beta_iter_physics_rules, "beta_fast_alpha"
)


# `i_beta_fast_alpha == WARD` (1) -- PROCESS's own default
# (`physics_variables.py:238`) and the reference run's.
def _fast_alpha_beta_ward(
    b_plasma_surface_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    f_plasma_fuel_deuterium,
):
    return fast_alpha_beta_ward(
        b_plasma_surface_poloidal_average,
        b_plasma_toroidal_on_axis,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        pden_alpha_total_mw,
        pden_plasma_alpha_mw,
        f_plasma_fuel_deuterium,
    )


FastAlphaBetaWard = physics.function(_fast_alpha_beta_ward, "beta_fast_alpha")
