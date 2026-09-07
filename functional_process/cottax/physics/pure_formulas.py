"""Pure-functional port of five already-pure formulas from
`process/models/physics/physics.py`.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

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


class IonElectronEquilibration(ExplicitFunction):
    """cottax node: `rether`, unchanged, ports declared."""

    pden_ion_electron_equilibration_mw = OutputInto(physics)

    def __call__(
        self,
        alphan=From(physics),
        alphat=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        dlamie=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        n_charge_plasma_effective_mass_weighted_vol_avg=From(physics),
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


class AuxiliaryPhysicsQuantities(ExplicitFunction):
    """cottax node: `phyaux`, ports declared."""

    sbar: float = 1.0

    burnup = OutputInto(physics)
    figmer = OutputInto(physics)
    fusrat = OutputInto(physics)
    molflow_plasma_fuelling_required = OutputInto(physics)
    rndfuel = OutputInto(physics)
    t_alpha_confinement = OutputInto(physics)
    f_t_alpha_energy_confinement = OutputInto(physics)

    def __call__(
        self,
        aspect=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        fusden_total=From(physics),
        fusden_alpha_total=From(physics),
        plasma_current=From(physics),
        nd_plasma_alphas_thermal_vol_avg=From(physics),
        t_energy_confinement=From(physics),
        vol_plasma=From(physics),
        burnup_in=From(physics),
        tauratio=From(physics),
    ):
        return phyaux(
            aspect,
            nd_plasma_fuel_ions_vol_avg,
            fusden_total,
            fusden_alpha_total,
            plasma_current,
            self.sbar,
            nd_plasma_alphas_thermal_vol_avg,
            t_energy_confinement,
            vol_plasma,
            burnup_in,
            tauratio,
        )


class TotalPlasmaHeatingPower(ExplicitFunction):
    """cottax node: `calculate_total_plasma_heating_power`, unchanged, ports declared."""

    p_plasma_heating_total_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_total_plasma_heating_power(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class ElectronThermalEnergy(ExplicitFunction):
    """cottax node: `calaculate_stored_thermal_energy`, electron binding."""

    eden_plasma_electrons_thermal_vol_avg = OutputInto(physics)
    e_plasma_electrons_thermal = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
    ):
        return calaculate_stored_thermal_energy(
            vol_plasma,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
        )


class IonThermalEnergy(ExplicitFunction):
    """cottax node: `calaculate_stored_thermal_energy`, ion binding."""

    eden_plasma_ions_thermal_vol_avg = OutputInto(physics)
    e_plasma_ions_thermal = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
    ):
        return calaculate_stored_thermal_energy(
            vol_plasma,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
        )


class FastAlphaBeta(ExplicitFunction):
    """The `fast_alpha_beta` family -- one occupant per `.physics.i_beta_fast_alpha`
    value.
    """

    beta_fast_alpha = OutputInto(physics)


class FastAlphaBetaIterPhysicsRules(FastAlphaBeta):
    """`i_beta_fast_alpha == ITER_PHYSICS_RULES` (0) -- the ITER Physics Design
    Guidelines fraction (`physics.py:2043`).
    """

    def __call__(
        self,
        b_plasma_surface_poloidal_average=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        pden_alpha_total_mw=From(physics),
        pden_plasma_alpha_mw=From(physics),
        f_plasma_fuel_deuterium=From(physics),
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


class FastAlphaBetaWard(FastAlphaBeta):
    """`i_beta_fast_alpha == WARD` (1) -- PROCESS's own default
    (`physics_variables.py:238`) and the reference run's.
    """

    def __call__(
        self,
        b_plasma_surface_poloidal_average=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        pden_alpha_total_mw=From(physics),
        pden_plasma_alpha_mw=From(physics),
        f_plasma_fuel_deuterium=From(physics),
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
