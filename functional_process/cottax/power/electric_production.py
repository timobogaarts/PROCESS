"""Pure-functional port of the AC/electric-production sub-unit of
`process/models/power.py` (registry unit #14, chunk C).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CostOfElectricityModel,
    SphericalTokamakModel,
)
from functional_process.cottax.paths import (
    buildings,
    fwbs,
    heat_transport,
    pf_coil,
    pf_power,
    physics,
    power,
    tfcoil,
    times,
)
from functional_process.models.power.electric_production import (
    calculate_acpow,
    calculate_acpow_line,
    calculate_acpow_motor_generator_flywheel,
    calculate_plant_electric_production,
    calculate_plant_electric_production_reactor,
    calculate_plant_electric_production_resistive_centrepost_liquid_breeder,
    centrepost_coolant_pump_power_absent,
    centrepost_coolant_pump_power_resistive,
    gross_electric_power_liquid_breeder,
    gross_electric_power_single_coolant,
    power_profiles_over_time,
)
from functional_process.vocabulary import PumpingPowerModelTypes, TFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just the few that turn
# out to be unused here (see this file's own first-pass commit for why a partial list
# is the wrong move: it silently drops the D101/D102 checks on the classes below).
__all__ = [
    "Acpow",
    "AcpowLine",
    "AcpowMotorGeneratorFlywheel",
    "BlanketDualCoolantModel",
    "CostOfElectricityModel",
    "ExplicitFunction",
    "From",
    "OutputInto",
    "PlantElectricProductionLiquidBreeder",
    "PlantElectricProductionReactor",
    "PlantElectricProductionResistiveCentrepostLiquidBreeder",
    "PlantElectricProductionResistiveCentrepostSingleCoolant",
    "PlantElectricProductionSingleCoolant",
    "PowerProfilesOverTime",
    "PumpingPowerModelTypes",
    "SphericalTokamakModel",
    "TFConductorModel",
    "buildings",
    "calculate_acpow",
    "calculate_acpow_line",
    "calculate_acpow_motor_generator_flywheel",
    "calculate_plant_electric_production",
    "calculate_plant_electric_production_reactor",
    "calculate_plant_electric_production_resistive_centrepost_liquid_breeder",
    "centrepost_coolant_pump_power_absent",
    "centrepost_coolant_pump_power_resistive",
    "eqx",
    "fwbs",
    "gross_electric_power_liquid_breeder",
    "gross_electric_power_single_coolant",
    "heat_transport",
    "jnp",
    "pf_coil",
    "pf_power",
    "physics",
    "power",
    "power_profiles_over_time",
    "tfcoil",
    "times",
]


class Acpow(ExplicitFunction):
    """The `calculate_acpow` family -- one occupant per
    `.pf_power.i_pf_energy_storage_source` value.
    """

    pacpmw = OutputInto(heat_transport)
    tlvpmw = OutputInto(heat_transport)


class AcpowLine(Acpow):
    """`i_pf_energy_storage_source == LINE` (2) -- the reference run's."""

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        peakmva=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
    ):
        return calculate_acpow_line(
            p_tf_electric_supplies_mw,
            srcktpm,
            peakmva,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
        )


class AcpowMotorGeneratorFlywheel(Acpow):
    """`i_pf_energy_storage_source == MGF` (1) -- all power from motor-generator
    flywheel units, PROCESS's own default (`pf_power_variables.py:18`).
    """

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
        fmgdmw=From(heat_transport),
    ):
        return calculate_acpow_motor_generator_flywheel(
            p_tf_electric_supplies_mw,
            srcktpm,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
            fmgdmw,
        )


class PowerProfilesOverTime(ExplicitFunction):
    """cottax node: `power_profiles_over_time`."""

    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def __call__(
        self,
        p_plant_electric_base_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        p_plant_electric_gross_mw=From(heat_transport),
        p_plant_electric_net_mw=From(heat_transport),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return power_profiles_over_time(
            p_plant_electric_base_total_mw,
            p_cryo_plant_electric_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_tf_electric_supplies_mw,
            p_pf_electric_supplies_mw,
            p_coolant_pump_elec_total_mw,
            p_hcd_electric_total_mw,
            p_fusion_total_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_net_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionReactor(ExplicitFunction):
    """The `calculate_plant_electric_production` family at `ireactor == 1` -- one
    occupant per `(itart, i_tf_sup)` x `(i_blkt_dual_coolant, i_p_coolant_pumping)` arm
    pair.
    """

    p_cp_coolant_pump_elec_mw = OutputInto(power)
    p_plant_electric_base_total_mw = OutputInto(heat_transport)
    fachtmw = OutputInto(heat_transport)
    p_plant_core_systems_elec_mw = OutputInto(power)
    p_plant_secondary_heat_mw = OutputInto(heat_transport)
    p_plant_electric_gross_mw = OutputInto(heat_transport)
    p_turbine_loss_mw = OutputInto(power)
    p_plant_electric_recirc_mw = OutputInto(heat_transport)
    p_plant_electric_net_mw = OutputInto(heat_transport)
    f_p_plant_electric_recirc = OutputInto(heat_transport)
    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def _production(
        self,
        p_cp_coolant_pump_elec_mw,
        p_plant_electric_gross_mw,
        p_plant_electric_base,
        a_plant_floor_effective,
        pflux_plant_floor_electric,
        p_cryo_plant_electric_mw,
        p_tf_electric_supplies_mw,
        p_tritium_plant_electric_mw,
        vachtmw,
        p_pf_electric_supplies_mw,
        p_hcd_electric_loss_mw,
        p_coolant_pump_loss_total_mw,
        p_div_secondary_heat_mw,
        p_shld_secondary_heat_mw,
        p_hcd_secondary_heat_mw,
        p_tf_nuclear_heat_mw,
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    ):
        """The twenty-three outputs, given the two quantities the four arms disagree
        about.
        """
        return calculate_plant_electric_production_reactor(
            p_cp_coolant_pump_elec_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionSingleCoolant(PlantElectricProductionReactor):
    """No resistive centrepost, one coolant -- the reference run's and the conventional
    tokamak's (`itart = 0`, `i_blkt_dual_coolant = 0`).
    """

    def __call__(
        self,
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionLiquidBreeder(PlantElectricProductionReactor):
    """No resistive centrepost; liquid breeder with its own turbine efficiency
    (`i_blkt_dual_coolant > 0` and `i_p_coolant_pumping == MECHANICAL`).
    """

    def __call__(
        self,
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
            gross_electric_power_liquid_breeder(
                p_plant_primary_heat_mw,
                eta_turbine,
                p_blkt_liquid_breeder_heat_deposited_mw,
                etath_liq,
            ),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionResistiveCentrepostSingleCoolant(
    PlantElectricProductionReactor
):
    """Resistive centrepost (`itart == 1` and `i_tf_sup == 0`), one coolant."""

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_resistive(p_cp_coolant_pump_elec),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionResistiveCentrepostLiquidBreeder(
    PlantElectricProductionReactor
):
    """Resistive centrepost and liquid breeder -- both extra reads at once."""

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return calculate_plant_electric_production_resistive_centrepost_liquid_breeder(
            p_cp_coolant_pump_elec,
            p_blkt_liquid_breeder_heat_deposited_mw,
            etath_liq,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )
