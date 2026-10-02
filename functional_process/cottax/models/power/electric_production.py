"""Pure-functional port of the AC/electric-production sub-unit of
`process/models/power.py` (registry unit #14, chunk C).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.cx import V

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
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CostOfElectricityModel,
    SphericalTokamakModel,
)
from functional_process.vocabulary import PumpingPowerModelTypes, TFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just the few that turn
# out to be unused here (see this file's own first-pass commit for why a partial list
# is the wrong move: it silently drops the D101/D102 checks on the classes below).
__all__ = [
    "AcpowLine",
    "AcpowMotorGeneratorFlywheel",
    "BlanketDualCoolantModel",
    "CostOfElectricityModel",
    "PlantElectricProductionLiquidBreeder",
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


# The Acpow family:
# The `calculate_acpow` family -- one occupant per
# `.pf_power.i_pf_energy_storage_source` value.


# `i_pf_energy_storage_source == LINE` (2) -- the reference run's.
AcpowLine = heat_transport.function(
    calculate_acpow_line,
    "pacpmw",
    "tlvpmw",
)


# `i_pf_energy_storage_source == MGF` (1) -- all power from motor-generator
# flywheel units, PROCESS's own default (`pf_power_variables.py:18`).
AcpowMotorGeneratorFlywheel = heat_transport.function(
    calculate_acpow_motor_generator_flywheel,
    "pacpmw",
    "tlvpmw",
)


# cottax node: `power_profiles_over_time`.
PowerProfilesOverTime = power.function(
    power_profiles_over_time,
    "e_plant_net_electric_pulse_kwh",
    "e_plant_net_electric_pulse_mj",
    "p_plant_electric_base_total_profile_mw",
    "p_plant_electric_gross_profile_mw",
    "p_plant_electric_net_profile_mw",
    "p_hcd_electric_total_profile_mw",
    "p_coolant_pump_elec_total_profile_mw",
    "p_tf_electric_supplies_profile_mw",
    "p_pf_electric_supplies_profile_mw",
    "vachtmw_profile_mw",
    "p_tritium_plant_electric_profile_mw",
    "p_cryo_plant_electric_profile_mw",
    "p_fusion_total_profile_mw",
)


# The PlantElectricProductionReactor family:
# The `calculate_plant_electric_production` family at `ireactor == 1` -- one
# occupant per `(itart, i_tf_sup)` x `(i_blkt_dual_coolant, i_p_coolant_pumping)` arm
# pair.


def _production_single_coolant(
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
    return calculate_plant_electric_production_reactor(
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


# No resistive centrepost, one coolant -- the reference run's and the conventional
# tokamak's (`itart = 0`, `i_blkt_dual_coolant = 0`).
PlantElectricProductionSingleCoolant = power.function(
    _production_single_coolant,
    "p_cp_coolant_pump_elec_mw",
    V.heat_transport.p_plant_electric_base_total_mw,
    V.heat_transport.fachtmw,
    "p_plant_core_systems_elec_mw",
    V.heat_transport.p_plant_secondary_heat_mw,
    V.heat_transport.p_plant_electric_gross_mw,
    "p_turbine_loss_mw",
    V.heat_transport.p_plant_electric_recirc_mw,
    V.heat_transport.p_plant_electric_net_mw,
    V.heat_transport.f_p_plant_electric_recirc,
    "e_plant_net_electric_pulse_kwh",
    "e_plant_net_electric_pulse_mj",
    "p_plant_electric_base_total_profile_mw",
    "p_plant_electric_gross_profile_mw",
    "p_plant_electric_net_profile_mw",
    "p_hcd_electric_total_profile_mw",
    "p_coolant_pump_elec_total_profile_mw",
    "p_tf_electric_supplies_profile_mw",
    "p_pf_electric_supplies_profile_mw",
    "vachtmw_profile_mw",
    "p_tritium_plant_electric_profile_mw",
    "p_cryo_plant_electric_profile_mw",
    "p_fusion_total_profile_mw",
)


def _production_liquid_breeder(
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
):
    return calculate_plant_electric_production_reactor(
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


# No resistive centrepost; liquid breeder with its own turbine efficiency
# (`i_blkt_dual_coolant > 0` and `i_p_coolant_pumping == MECHANICAL`).
PlantElectricProductionLiquidBreeder = power.function(
    _production_liquid_breeder,
    "p_cp_coolant_pump_elec_mw",
    V.heat_transport.p_plant_electric_base_total_mw,
    V.heat_transport.fachtmw,
    "p_plant_core_systems_elec_mw",
    V.heat_transport.p_plant_secondary_heat_mw,
    V.heat_transport.p_plant_electric_gross_mw,
    "p_turbine_loss_mw",
    V.heat_transport.p_plant_electric_recirc_mw,
    V.heat_transport.p_plant_electric_net_mw,
    V.heat_transport.f_p_plant_electric_recirc,
    "e_plant_net_electric_pulse_kwh",
    "e_plant_net_electric_pulse_mj",
    "p_plant_electric_base_total_profile_mw",
    "p_plant_electric_gross_profile_mw",
    "p_plant_electric_net_profile_mw",
    "p_hcd_electric_total_profile_mw",
    "p_coolant_pump_elec_total_profile_mw",
    "p_tf_electric_supplies_profile_mw",
    "p_pf_electric_supplies_profile_mw",
    "vachtmw_profile_mw",
    "p_tritium_plant_electric_profile_mw",
    "p_cryo_plant_electric_profile_mw",
    "p_fusion_total_profile_mw",
)


def _production_resistive_centrepost_single_coolant(
    p_cp_coolant_pump_elec,
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
    return calculate_plant_electric_production_reactor(
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


# Resistive centrepost (`itart == 1` and `i_tf_sup == 0`), one coolant.
PlantElectricProductionResistiveCentrepostSingleCoolant = power.function(
    _production_resistive_centrepost_single_coolant,
    "p_cp_coolant_pump_elec_mw",
    V.heat_transport.p_plant_electric_base_total_mw,
    V.heat_transport.fachtmw,
    "p_plant_core_systems_elec_mw",
    V.heat_transport.p_plant_secondary_heat_mw,
    V.heat_transport.p_plant_electric_gross_mw,
    "p_turbine_loss_mw",
    V.heat_transport.p_plant_electric_recirc_mw,
    V.heat_transport.p_plant_electric_net_mw,
    V.heat_transport.f_p_plant_electric_recirc,
    "e_plant_net_electric_pulse_kwh",
    "e_plant_net_electric_pulse_mj",
    "p_plant_electric_base_total_profile_mw",
    "p_plant_electric_gross_profile_mw",
    "p_plant_electric_net_profile_mw",
    "p_hcd_electric_total_profile_mw",
    "p_coolant_pump_elec_total_profile_mw",
    "p_tf_electric_supplies_profile_mw",
    "p_pf_electric_supplies_profile_mw",
    "vachtmw_profile_mw",
    "p_tritium_plant_electric_profile_mw",
    "p_cryo_plant_electric_profile_mw",
    "p_fusion_total_profile_mw",
)


# Resistive centrepost and liquid breeder -- both extra reads at once.
PlantElectricProductionResistiveCentrepostLiquidBreeder = power.function(
    calculate_plant_electric_production_resistive_centrepost_liquid_breeder,
    "p_cp_coolant_pump_elec_mw",
    V.heat_transport.p_plant_electric_base_total_mw,
    V.heat_transport.fachtmw,
    "p_plant_core_systems_elec_mw",
    V.heat_transport.p_plant_secondary_heat_mw,
    V.heat_transport.p_plant_electric_gross_mw,
    "p_turbine_loss_mw",
    V.heat_transport.p_plant_electric_recirc_mw,
    V.heat_transport.p_plant_electric_net_mw,
    V.heat_transport.f_p_plant_electric_recirc,
    "e_plant_net_electric_pulse_kwh",
    "e_plant_net_electric_pulse_mj",
    "p_plant_electric_base_total_profile_mw",
    "p_plant_electric_gross_profile_mw",
    "p_plant_electric_net_profile_mw",
    "p_hcd_electric_total_profile_mw",
    "p_coolant_pump_elec_total_profile_mw",
    "p_tf_electric_supplies_profile_mw",
    "p_pf_electric_supplies_profile_mw",
    "vachtmw_profile_mw",
    "p_tritium_plant_electric_profile_mw",
    "p_cryo_plant_electric_profile_mw",
    "p_fusion_total_profile_mw",
)
