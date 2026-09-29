"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces import V

from functional_process.cottax.paths import (
    current_drive,
    fwbs,
    heat_transport,
    pf_power,
    physics,
    power,
    primary_pumping,
    structure,
    tfcoil,
    times,
)
from functional_process.models.power.thermal_cryo import (
    calculate_component_thermal_powers,
    calculate_component_thermal_powers_owned,
    calculate_component_thermal_powers_owned_mech_dual_ccfe,
    calculate_component_thermal_powers_owned_mech_dual_other,
    calculate_component_thermal_powers_owned_mech_liquid_ccfe,
    calculate_component_thermal_powers_owned_mech_liquid_other,
    calculate_component_thermal_powers_owned_mech_solid_ccfe,
    calculate_component_thermal_powers_owned_mech_solid_other,
    calculate_component_thermal_powers_owned_summed_dual_ccfe,
    calculate_component_thermal_powers_owned_summed_dual_other,
    calculate_component_thermal_powers_owned_summed_liquid_ccfe,
    calculate_component_thermal_powers_owned_summed_liquid_other,
    calculate_component_thermal_powers_owned_summed_solid_ccfe,
    calculate_component_thermal_powers_owned_summed_solid_other,
    calculate_cryo,
    calculate_cryo_loads,
    calculate_cryo_non_superconducting,
    calculate_cryo_plant_loads,
    calculate_cryo_plant_loads_active,
    calculate_cryo_plant_loads_inactive,
    calculate_cryo_q_loads,
    calculate_cryo_q_loads_resistive_tf,
    calculate_cryo_q_loads_superconducting_tf,
    calculate_cryo_qnuc,
    calculate_cryo_qnuc_when_computed,
    calculate_cryo_superconducting_computed,
    calculate_cryo_superconducting_user_input,
    calculate_delta_eta,
    calculate_delta_eta_next,
    calculate_delta_eta_next_mech_liquid_ccfe,
    calculate_delta_eta_next_mech_liquid_other,
    calculate_delta_eta_next_mech_solid_ccfe,
    calculate_delta_eta_next_mech_solid_other,
    calculate_delta_eta_next_summed_liquid_ccfe,
    calculate_delta_eta_next_summed_liquid_other,
    calculate_delta_eta_next_summed_solid_ccfe,
    calculate_delta_eta_next_summed_solid_other,
    calculate_helpow,
    calculate_p_div_heat_deposited_mw,
    calculate_p_fw_blkt_coolant_pump_mw,
    calculate_p_fw_blkt_coolant_pump_mw_summed,
    calculate_p_fw_blkt_heat_deposited_mw,
    calculate_p_fw_div_heat_deposited_mw,
    calculate_p_fw_div_heat_deposited_mw_summed,
    calculate_p_fw_heat_deposited_mw,
    calculate_p_shld_heat_deposited_mw,
    calculate_plant_thermal_efficiency,
    calculate_plant_thermal_efficiency_2,
    cryo_is_active,
    eta_turbine_ccfe_hcpb_value,
    eta_turbine_ccfe_hcpb_value_with_divertor,
    eta_turbine_steam_rankine_cycle,
    eta_turbine_supercritical_co2,
    etath_liq_supercritical_co2,
    temp_turbine_coolant_in_from_blanket_coolant,
    temp_turbine_coolant_in_from_liquid_breeder,
)
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CoilNuclearHeatingModel,
)
from functional_process.vocabulary import (
    BlktModelTypes,
    ElectricConversionModelTypes,
    PFConductorModel,
    ProcessValueError,
    PumpingPowerModelTypes,
    TFConductorModel,
    constants,
)

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just the few that turn
# out to be unused here (see `power/electric_production.py`'s commit for why a partial
# list is the wrong move: it silently drops the D101/D102 checks on the rest).
__all__ = [
    "BlanketDualCoolantModel",
    "BlktModelTypes",
    "CoilNuclearHeatingModel",
    "ComponentThermalPowersMechDualCcfe",
    "ComponentThermalPowersMechDualOther",
    "ComponentThermalPowersMechLiquidCcfe",
    "ComponentThermalPowersMechLiquidOther",
    "ComponentThermalPowersMechSolidCcfe",
    "ComponentThermalPowersMechSolidOther",
    "ComponentThermalPowersSummedDualCcfe",
    "ComponentThermalPowersSummedDualOther",
    "ComponentThermalPowersSummedLiquidCcfe",
    "ComponentThermalPowersSummedLiquidOther",
    "ComponentThermalPowersSummedSolidCcfe",
    "ComponentThermalPowersSummedSolidOther",
    "CryoLoadsActive",
    "CryoLoadsInactive",
    "CryoQLoadsResistiveTf",
    "CryoQLoadsSuperconductingTf",
    "CryoQNuc",
    "DeltaEtaStepMechLiquidCcfe",
    "DeltaEtaStepMechLiquidOther",
    "DeltaEtaStepMechSolidCcfe",
    "DeltaEtaStepMechSolidOther",
    "DeltaEtaStepSummedLiquidCcfe",
    "DeltaEtaStepSummedLiquidOther",
    "DeltaEtaStepSummedSolidCcfe",
    "DeltaEtaStepSummedSolidOther",
    "ElectricConversionModelTypes",
    "EtaTurbineCcfeHcpbValue",
    "EtaTurbineCcfeHcpbValueWithDivertor",
    "EtaTurbineSteamRankineCycle",
    "EtaTurbineSupercriticalCo2",
    "EtathLiqSupercriticalCo2",
    "PFConductorModel",
    "PFwBlktCoolantPumpMw",
    "PFwDivHeatDepositedMwSummed",
    "ProcessValueError",
    "PumpingPowerModelTypes",
    "TFConductorModel",
    "TempTurbineCoolantInFromBlanketCoolant",
    "TempTurbineCoolantInFromLiquidBreeder",
    "calculate_component_thermal_powers",
    "calculate_component_thermal_powers_owned",
    "calculate_component_thermal_powers_owned_mech_dual_ccfe",
    "calculate_component_thermal_powers_owned_mech_dual_other",
    "calculate_component_thermal_powers_owned_mech_liquid_ccfe",
    "calculate_component_thermal_powers_owned_mech_liquid_other",
    "calculate_component_thermal_powers_owned_mech_solid_ccfe",
    "calculate_component_thermal_powers_owned_mech_solid_other",
    "calculate_component_thermal_powers_owned_summed_dual_ccfe",
    "calculate_component_thermal_powers_owned_summed_dual_other",
    "calculate_component_thermal_powers_owned_summed_liquid_ccfe",
    "calculate_component_thermal_powers_owned_summed_liquid_other",
    "calculate_component_thermal_powers_owned_summed_solid_ccfe",
    "calculate_component_thermal_powers_owned_summed_solid_other",
    "calculate_cryo",
    "calculate_cryo_loads",
    "calculate_cryo_non_superconducting",
    "calculate_cryo_plant_loads",
    "calculate_cryo_plant_loads_active",
    "calculate_cryo_plant_loads_inactive",
    "calculate_cryo_q_loads",
    "calculate_cryo_q_loads_resistive_tf",
    "calculate_cryo_q_loads_superconducting_tf",
    "calculate_cryo_qnuc",
    "calculate_cryo_qnuc_when_computed",
    "calculate_cryo_superconducting_computed",
    "calculate_cryo_superconducting_user_input",
    "calculate_delta_eta",
    "calculate_delta_eta_next",
    "calculate_delta_eta_next_mech_liquid_ccfe",
    "calculate_delta_eta_next_mech_liquid_other",
    "calculate_delta_eta_next_mech_solid_ccfe",
    "calculate_delta_eta_next_mech_solid_other",
    "calculate_delta_eta_next_summed_liquid_ccfe",
    "calculate_delta_eta_next_summed_liquid_other",
    "calculate_delta_eta_next_summed_solid_ccfe",
    "calculate_delta_eta_next_summed_solid_other",
    "calculate_helpow",
    "calculate_p_div_heat_deposited_mw",
    "calculate_p_fw_blkt_coolant_pump_mw",
    "calculate_p_fw_blkt_coolant_pump_mw_summed",
    "calculate_p_fw_blkt_heat_deposited_mw",
    "calculate_p_fw_div_heat_deposited_mw",
    "calculate_p_fw_div_heat_deposited_mw_summed",
    "calculate_p_fw_heat_deposited_mw",
    "calculate_p_shld_heat_deposited_mw",
    "calculate_plant_thermal_efficiency",
    "calculate_plant_thermal_efficiency_2",
    "constants",
    "cryo_is_active",
    "current_drive",
    "eqx",
    "eta_turbine_ccfe_hcpb_value",
    "eta_turbine_ccfe_hcpb_value_with_divertor",
    "eta_turbine_steam_rankine_cycle",
    "eta_turbine_supercritical_co2",
    "etath_liq_supercritical_co2",
    "fwbs",
    "heat_transport",
    "jnp",
    "pf_power",
    "physics",
    "power",
    "primary_pumping",
    "structure",
    "temp_turbine_coolant_in_from_blanket_coolant",
    "temp_turbine_coolant_in_from_liquid_breeder",
    "tfcoil",
    "times",
]


# The ComponentThermalPowers family:
# cottax node: `calculate_component_thermal_powers`'s outputs **other than**
# the six self-referencing fields split into their own occupant families/
# `FixedPointFunction`s elsewhere in this module (`DeltaEtaStep`, `EtaTurbine`,
# `EtathLiq`, `TempTurbineCoolantIn`, `PFwDivHeatDepositedMw`,
# `PFwBlktCoolantPumpMw`). Bodiless family base: twelve occupants below, the full
# `PumpingPowerModelTypes` (binary here) x `BlanketDualCoolantModel` (all three
# values) x `ElectricConversionModelTypes` (binary here) product --
# `_audit/switch_kwarg_survey.md`'s exemption for this node ("too costly to
# split") is withdrawn, see `naming_convention.md` § "Switches are not ports".
# Every arm shares this base's reads and outputs unchanged and adds only `fn`
# (`cottax/wraps.py`), so none of the twelve restates the 24-read signature.


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_SOLID_BREEDER`, `i_thermal_electric_conversion ==
# CCFE_HCPB_VALUE`.
ComponentThermalPowersSummedSolidCcfe = power.function(
    calculate_component_thermal_powers_owned_summed_solid_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_SOLID_BREEDER`, `i_thermal_electric_conversion` anything else.
ComponentThermalPowersSummedSolidOther = power.function(
    calculate_component_thermal_powers_owned_summed_solid_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_LIQUID_BREEDER`, `i_thermal_electric_conversion ==
# CCFE_HCPB_VALUE`.
ComponentThermalPowersSummedLiquidCcfe = power.function(
    calculate_component_thermal_powers_owned_summed_liquid_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_LIQUID_BREEDER`, `i_thermal_electric_conversion` anything
# else.
ComponentThermalPowersSummedLiquidOther = power.function(
    calculate_component_thermal_powers_owned_summed_liquid_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == DUAL_COOLANT`, `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
ComponentThermalPowersSummedDualCcfe = power.function(
    calculate_component_thermal_powers_owned_summed_dual_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == DUAL_COOLANT`, `i_thermal_electric_conversion` anything else.
ComponentThermalPowersSummedDualOther = power.function(
    calculate_component_thermal_powers_owned_summed_dual_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER`,
# `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
ComponentThermalPowersMechSolidCcfe = power.function(
    calculate_component_thermal_powers_owned_mech_solid_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER`,
# `i_thermal_electric_conversion` anything else.
ComponentThermalPowersMechSolidOther = power.function(
    calculate_component_thermal_powers_owned_mech_solid_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_LIQUID_BREEDER`,
# `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
ComponentThermalPowersMechLiquidCcfe = power.function(
    calculate_component_thermal_powers_owned_mech_liquid_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_LIQUID_BREEDER`,
# `i_thermal_electric_conversion` anything else.
ComponentThermalPowersMechLiquidOther = power.function(
    calculate_component_thermal_powers_owned_mech_liquid_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == DUAL_COOLANT`, `i_thermal_electric_conversion ==
# CCFE_HCPB_VALUE`.
ComponentThermalPowersMechDualCcfe = power.function(
    calculate_component_thermal_powers_owned_mech_dual_ccfe,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == DUAL_COOLANT`, `i_thermal_electric_conversion` anything
# else.
ComponentThermalPowersMechDualOther = power.function(
    calculate_component_thermal_powers_owned_mech_dual_other,
    "p_fw_blkt_coolant_pump_elec_mw",
    "p_shld_coolant_pump_elec_mw",
    "p_div_coolant_pump_elec_mw",
    "p_blkt_breeder_pump_elec_mw",
    "p_coolant_pump_total_mw",
    V.heat_transport.p_coolant_pump_elec_total_mw,
    V.heat_transport.p_coolant_pump_loss_total_mw,
    V.heat_transport.p_hcd_electric_loss_mw,
    "p_blkt_liquid_breeder_heat_deposited_mw",
    "p_fw_blkt_heat_deposited_mw",
    "p_fw_heat_deposited_mw",
    "p_blkt_heat_deposited_mw",
    "p_shld_heat_deposited_mw",
    "p_div_heat_deposited_mw",
    V.heat_transport.p_plant_primary_heat_mw,
    V.heat_transport.p_div_secondary_heat_mw,
    "i_div_primary_heat",
    "f_p_div_primary_heat",
    V.heat_transport.p_shld_secondary_heat_mw,
    V.heat_transport.p_hcd_secondary_heat_mw,
    V.heat_transport.n_primary_heat_exchangers,
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "eta_coolant_pump_electric": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_div_coolant_pump_mw": heat_transport,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_hcd_electric_total_mw": heat_transport,
        "p_hcd_injected_total_mw": current_drive,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "f_nuc_pow_bz_liq": fwbs,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_fw_hcd_nuclear_heat_mw": fwbs,
        "p_fw_hcd_rad_total_mw": fwbs,
        "i_shld_primary_heat": heat_transport,
    },
)


# The DeltaEtaStep family:
# `.power.delta_eta`, computed from the heat flows. Bodiless family base: eight
# occupants below cover `PumpingPowerModelTypes` (binary here: pump powers
# summed or passed through) x `BlanketDualCoolantModel` (binary here --
# coarser than `ComponentThermalPowers`'s three-way use of the same switch,
# since only `calculate_p_fw_blkt_heat_deposited_mw`'s `in (1, 2)` guard reads
# it here) x `ElectricConversionModelTypes` (binary here, same split
# `calculate_delta_eta` uses). `_audit/switch_kwarg_survey.md`'s exemption for
# this node is withdrawn, the same as `ComponentThermalPowers`'s.
#
# **No longer a `FixedPointFunction`.** The entering `.power.delta_eta` is PROCESS's
# incoming field value, not a previous iterate of this node, so it is read as the
# free place `.power.delta_eta_in` (`evaluate.KNOWN_MINT_VALUES` resolves it back to
# the same field) and the node owns `.power.delta_eta` outright. The self-loop, its
# minted `^cond.power.delta_eta`, its cut and its driver are all gone. The read
# itself stays, as it always did, because PROCESS's routine really does read the
# field -- and it remains numerically inert
# (`test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta`), which is what made
# the loop a formality rather than an iteration: `g` does not depend on `u`, so the
# first evaluation was always the answer.


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
# `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
DeltaEtaStepSummedSolidCcfe = power.function(
    calculate_delta_eta_next_summed_solid_ccfe,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
# == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
# `i_thermal_electric_conversion` anything else.
DeltaEtaStepSummedSolidOther = power.function(
    calculate_delta_eta_next_summed_solid_other,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant`
# in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}` (breeder-pump term added),
# `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
DeltaEtaStepSummedLiquidCcfe = power.function(
    calculate_delta_eta_next_summed_liquid_ccfe,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant`
# in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}` (breeder-pump term added),
# `i_thermal_electric_conversion` anything else.
DeltaEtaStepSummedLiquidOther = power.function(
    calculate_delta_eta_next_summed_liquid_other,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
# `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
DeltaEtaStepMechSolidCcfe = power.function(
    calculate_delta_eta_next_mech_solid_ccfe,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
# `i_thermal_electric_conversion` anything else.
DeltaEtaStepMechSolidOther = power.function(
    calculate_delta_eta_next_mech_solid_other,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant` in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}`
# (breeder-pump term added), `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
DeltaEtaStepMechLiquidCcfe = power.function(
    calculate_delta_eta_next_mech_liquid_ccfe,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# `i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
# `i_blkt_dual_coolant` in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}`
# (breeder-pump term added), `i_thermal_electric_conversion` anything else.
DeltaEtaStepMechLiquidOther = power.function(
    calculate_delta_eta_next_mech_liquid_other,
    "delta_eta",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
        "p_fw_blkt_coolant_pump_mw": primary_pumping,
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_blkt_breeder_pump_mw": heat_transport,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_cp_shield_nuclear_heat_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "p_shld_coolant_pump_mw": heat_transport,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
        "p_div_coolant_pump_mw": heat_transport,
        "i_shld_primary_heat": heat_transport,
        "delta_eta": V.power.delta_eta_in,
    },
)


# The EtaTurbine family:
# The `.heat_transport.eta_turbine` family -- one occupant per arm of
# `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type`.


# `i_thermal_electric_conversion == CCFE_HCPB_VALUE` (0) with `i_blanket_type ==
# CCFE_HCPB`.
def _eta_turbine_ccfe_hcpb_value():
    return eta_turbine_ccfe_hcpb_value()


EtaTurbineCcfeHcpbValue = heat_transport.function(
    _eta_turbine_ccfe_hcpb_value, "eta_turbine"
)


# `i_thermal_electric_conversion == CCFE_HCPB_VALUE_WITH_DIVERTOR` (1) with
# `i_blanket_type == CCFE_HCPB`.
EtaTurbineCcfeHcpbValueWithDivertor = heat_transport.function(
    eta_turbine_ccfe_hcpb_value_with_divertor,
    "eta_turbine",
    reads={
        "delta_eta": power,
    },
)


# `i_thermal_electric_conversion == STEAM_RANKINE_CYCLE` (3) with `i_blanket_type
# == CCFE_HCPB`.
EtaTurbineSteamRankineCycle = heat_transport.function(
    eta_turbine_steam_rankine_cycle,
    "eta_turbine",
    reads={
        "temp_blkt_coolant_out": fwbs,
        "delta_eta": power,
    },
)


# `i_thermal_electric_conversion == SUPERCRITICAL_CO2_BRAYTON_CYCLE` (4), at any
# blanket type.
EtaTurbineSupercriticalCo2 = heat_transport.function(
    eta_turbine_supercritical_co2,
    "eta_turbine",
    reads={
        "temp_blkt_coolant_out": fwbs,
    },
)


# The EtathLiq family:
# The `.heat_transport.etath_liq` family -- one occupant per
# `.fwbs.secondary_cycle_liq` value.


# `secondary_cycle_liq == 4` -- the reference run's.
EtathLiqSupercriticalCo2 = heat_transport.function(
    etath_liq_supercritical_co2,
    "etath_liq",
    reads={
        "outlet_temp_liq": fwbs,
    },
)


# The TempTurbineCoolantIn family:
# The `.heat_transport.temp_turbine_coolant_in` family -- one occupant per arm of
# `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type` x
# `.fwbs.secondary_cycle_liq`.


# `secondary_cycle_liq == 4` -- the reference run's.
TempTurbineCoolantInFromLiquidBreeder = heat_transport.function(
    temp_turbine_coolant_in_from_liquid_breeder,
    "temp_turbine_coolant_in",
    reads={
        "outlet_temp_liq": fwbs,
    },
)


# `secondary_cycle_liq == 2` with an `i_thermal_electric_conversion` arm that
# writes the turbine inlet temperature -- `STEAM_RANKINE_CYCLE` with a CCFE HCPB
# blanket, or `SUPERCRITICAL_CO2_BRAYTON_CYCLE` at any blanket.
TempTurbineCoolantInFromBlanketCoolant = heat_transport.function(
    temp_turbine_coolant_in_from_blanket_coolant,
    "temp_turbine_coolant_in",
    reads={
        "temp_blkt_coolant_out": fwbs,
    },
)


# The PFwDivHeatDepositedMw family:
# The `.heat_transport.p_fw_div_heat_deposited_mw` family -- one occupant per arm
# of `.fwbs.i_p_coolant_pumping`.


# `i_p_coolant_pumping != MECHANICAL_WITH_PRESSURE_DROP` -- the reference run's.
PFwDivHeatDepositedMwSummed = heat_transport.function(
    calculate_p_fw_div_heat_deposited_mw_summed,
    "p_fw_div_heat_deposited_mw",
    reads={
        "p_fw_nuclear_heat_total_mw": fwbs,
        "p_fw_rad_total_mw": fwbs,
        "p_beam_orbit_loss_mw": current_drive,
        "p_fw_alpha_mw": physics,
        "p_beam_shine_through_mw": current_drive,
        "p_plasma_separatrix_mw": physics,
        "p_div_nuclear_heat_total_mw": fwbs,
        "p_div_rad_total_mw": fwbs,
    },
)


# `.primary_pumping.p_fw_blkt_coolant_pump_mw` at the `.fwbs.i_p_coolant_pumping`
# values where `power` owns it: `USER_INPUT` and `FRACTION_OF_HEAT`.
PFwBlktCoolantPumpMw = primary_pumping.function(
    calculate_p_fw_blkt_coolant_pump_mw_summed,
    "p_fw_blkt_coolant_pump_mw",
    reads={
        "p_fw_coolant_pump_mw": heat_transport,
        "p_blkt_coolant_pump_mw": heat_transport,
    },
)


# `calculate_cryo` (and its three arms `calculate_cryo_superconducting_computed`,
# `calculate_cryo_superconducting_user_input`, `calculate_cryo_non_superconducting`)
# is not declared as a node: it both reads and owns `.fwbs.qnuc`, which a node may
# not. The replacement for `Power.cryo`/`.fwbs.qnuc` is the three-way split below and
# in `models/power/namespace.py` (`CryoQNuc`, `CryoQLoads`, `CryoLoads`) -- see
# `CryoQNuc`'s own comment for the degeneracy argument that split rests on.


# `.fwbs.qnuc` when PROCESS computes it: `inuclear == 0` and `i_tf_sup == 1`.
CryoQNuc = fwbs.function(calculate_cryo_qnuc_when_computed, "qnuc")


# The CryoQLoads family:
# The `.power.qss`/`qac`/`qcl`/`qmisc` family -- one occupant per arm of
# `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.


# `i_tf_sup == SUPERCONDUCTING` (1) -- the reference run's.
CryoQLoadsSuperconductingTf = power.function(
    calculate_cryo_q_loads_superconducting_tf,
    "qss",
    "qac",
    "qcl",
    "qmisc",
    reads={
        "tfcryoarea": tfcoil,
        "coldmass": structure,
        "ensxpfm": pf_power,
        "t_plant_pulse_plasma_present": times,
        "c_tf_turn": tfcoil,
        "n_tf_coils": tfcoil,
        "qnuc": fwbs,
    },
)


# `i_tf_sup != SUPERCONDUCTING` with `i_pf_conductor == SUPERCONDUCTING` -- the PF
# coils are what needs cooling.
CryoQLoadsResistiveTf = power.function(
    calculate_cryo_q_loads_resistive_tf,
    "qss",
    "qac",
    "qcl",
    "qmisc",
    reads={
        "coldmass": structure,
        "ensxpfm": pf_power,
        "t_plant_pulse_plasma_present": times,
        "qnuc": fwbs,
    },
)


# The CryoLoads family:
# The unconditionally-owned part of `Power.calculate_cryo_loads` -- one occupant
# per arm of `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.


# `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` -- the reference run's.
CryoLoadsActive = heat_transport.function(
    calculate_cryo_plant_loads_active,
    "helpow",
    "p_cryo_plant_electric_mw",
    "helpow_cryal",
    V.tfcoil.cryo_cool_req,
    reads={
        "eff_tf_cryo": tfcoil,
        "temp_tf_cryo": tfcoil,
        "temp_cp_coolant_inlet": tfcoil,
        "qss": power,
        "qac": power,
        "qcl": power,
        "qmisc": power,
        "qnuc": fwbs,
    },
)


# `i_tf_sup != 1` with resistive PF coils: PROCESS never calls `Power.cryo`, so
# `helpow` and `p_cryo_plant_electric_mw` are literal zeros.
CryoLoadsInactive = heat_transport.function(
    calculate_cryo_plant_loads_inactive,
    "helpow",
    "p_cryo_plant_electric_mw",
    "helpow_cryal",
    V.tfcoil.cryo_cool_req,
    reads={
        "temp_tf_cryo": tfcoil,
        "temp_cp_coolant_inlet": tfcoil,
    },
)
