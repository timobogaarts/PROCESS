"""Pure-functional port of five already-pure formulas from
`process/models/physics/physics.py`.
"""

from functional_process.cottax.paths import physics
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
IonElectronEquilibration = physics.function(
    rether,
    "pden_ion_electron_equilibration_mw",
    reads={"te": "temp_plasma_electron_vol_avg_kev"},
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
TotalPlasmaHeatingPower = physics.function(
    calculate_total_plasma_heating_power,
    "p_plasma_heating_total_mw",
)


# cottax node: `calaculate_stored_thermal_energy`, electron binding.
ElectronThermalEnergy = physics.function(
    calaculate_stored_thermal_energy,
    "eden_plasma_electrons_thermal_vol_avg",
    "e_plasma_electrons_thermal",
    reads={
        "nd_plasma_vol_avg": "nd_plasma_electrons_vol_avg",
        "temp_plasma_density_weighted_vol_avg_kev": "temp_plasma_electron_density_weighted_kev",
    },
)


# cottax node: `calaculate_stored_thermal_energy`, ion binding.
IonThermalEnergy = physics.function(
    calaculate_stored_thermal_energy,
    "eden_plasma_ions_thermal_vol_avg",
    "e_plasma_ions_thermal",
    reads={
        "nd_plasma_vol_avg": "nd_plasma_ions_total_vol_avg",
        "temp_plasma_density_weighted_vol_avg_kev": "temp_plasma_ion_density_weighted_kev",
    },
)


# The FastAlphaBeta family:
# The `fast_alpha_beta` family -- one occupant per `.physics.i_beta_fast_alpha`
# value.


# `i_beta_fast_alpha == ITER_PHYSICS_RULES` (0) -- the ITER Physics Design
# Guidelines fraction (`physics.py:2043`).
FastAlphaBetaIterPhysicsRules = physics.function(
    fast_alpha_beta_iter_physics_rules,
    "beta_fast_alpha",
    reads={"b_plasma_poloidal_average": "b_plasma_surface_poloidal_average"},
)


# `i_beta_fast_alpha == WARD` (1) -- PROCESS's own default
# (`physics_variables.py:238`) and the reference run's.
FastAlphaBetaWard = physics.function(
    fast_alpha_beta_ward,
    "beta_fast_alpha",
    reads={"b_plasma_poloidal_average": "b_plasma_surface_poloidal_average"},
)
