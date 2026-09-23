"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

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
from functional_process.cottax.wraps import WrapsFunction
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
    "ComponentThermalPowers",
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
    "Cryo",
    "CryoLoads",
    "CryoLoadsActive",
    "CryoLoadsInactive",
    "CryoNonSuperconducting",
    "CryoQLoads",
    "CryoQLoadsResistiveTf",
    "CryoQLoadsSuperconductingTf",
    "CryoQNuc",
    "CryoSuperconductingComputed",
    "CryoSuperconductingUserInput",
    "DeltaEtaStep",
    "DeltaEtaStepMechLiquidCcfe",
    "DeltaEtaStepMechLiquidOther",
    "DeltaEtaStepMechSolidCcfe",
    "DeltaEtaStepMechSolidOther",
    "DeltaEtaStepSummedLiquidCcfe",
    "DeltaEtaStepSummedLiquidOther",
    "DeltaEtaStepSummedSolidCcfe",
    "DeltaEtaStepSummedSolidOther",
    "ElectricConversionModelTypes",
    "EtaTurbine",
    "EtaTurbineCcfeHcpbValue",
    "EtaTurbineCcfeHcpbValueWithDivertor",
    "EtaTurbineSteamRankineCycle",
    "EtaTurbineSupercriticalCo2",
    "EtathLiq",
    "EtathLiqSupercriticalCo2",
    "ExplicitFunction",
    "FixedPointFunction",
    "From",
    "OutputInto",
    "PFConductorModel",
    "PFwBlktCoolantPumpMw",
    "PFwDivHeatDepositedMw",
    "PFwDivHeatDepositedMwSummed",
    "ProcessValueError",
    "PumpingPowerModelTypes",
    "TFConductorModel",
    "TempTurbineCoolantIn",
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


class ComponentThermalPowers(WrapsFunction):
    """cottax node: `calculate_component_thermal_powers`'s outputs **other than**
    the six self-referencing fields split into their own occupant families/
    `FixedPointFunction`s elsewhere in this module (`DeltaEtaStep`, `EtaTurbine`,
    `EtathLiq`, `TempTurbineCoolantIn`, `PFwDivHeatDepositedMw`,
    `PFwBlktCoolantPumpMw`). Bodiless family base: twelve occupants below, the full
    `PumpingPowerModelTypes` (binary here) x `BlanketDualCoolantModel` (all three
    values) x `ElectricConversionModelTypes` (binary here) product --
    `_audit/switch_kwarg_survey.md`'s exemption for this node ("too costly to
    split") is withdrawn, see `naming_convention.md` § "Switches are not ports".
    Every arm shares this base's reads and outputs unchanged and adds only `fn`
    (`cottax/wraps.py`), so none of the twelve restates the 24-read signature.
    """

    # .primary_pumping.p_fw_blkt_coolant_pump_mw is NOT declared here --
    # `PFwBlktCoolantPumpMw` owns it. Still read below, as a plain FromExactly: the
    # pump totals sum it.
    p_fw_blkt_coolant_pump_elec_mw = OutputInto(power)
    p_shld_coolant_pump_elec_mw = OutputInto(power)
    p_div_coolant_pump_elec_mw = OutputInto(power)
    p_blkt_breeder_pump_elec_mw = OutputInto(power)
    p_coolant_pump_total_mw = OutputInto(power)
    p_coolant_pump_elec_total_mw = OutputInto(heat_transport)
    p_coolant_pump_loss_total_mw = OutputInto(heat_transport)
    p_hcd_electric_loss_mw = OutputInto(heat_transport)
    p_blkt_liquid_breeder_heat_deposited_mw = OutputInto(power)
    p_fw_blkt_heat_deposited_mw = OutputInto(power)
    p_fw_heat_deposited_mw = OutputInto(power)
    p_blkt_heat_deposited_mw = OutputInto(power)
    p_shld_heat_deposited_mw = OutputInto(power)
    p_div_heat_deposited_mw = OutputInto(power)
    # .heat_transport.p_fw_div_heat_deposited_mw / .eta_turbine / .etath_liq /
    # .temp_turbine_coolant_in are NOT declared here -- each is owned by its own
    # family below -- and, since `_audit/next_steps.md` §14.2, **not read here
    # either**: this node recomputed all four and discarded the results.
    p_plant_primary_heat_mw = OutputInto(heat_transport)
    p_div_secondary_heat_mw = OutputInto(heat_transport)
    i_div_primary_heat = OutputInto(power)
    f_p_div_primary_heat = OutputInto(power)
    # .power.delta_eta is NOT declared here -- DeltaEtaStep's FixedPoint problem
    # node owns it (see "The delta_eta self-loop" in thermal_cryo.md) -- and is no
    # longer read here either, for the same reason as the four above.
    p_shld_secondary_heat_mw = OutputInto(heat_transport)
    p_hcd_secondary_heat_mw = OutputInto(heat_transport)
    n_primary_heat_exchangers = OutputInto(heat_transport)

    p_fw_coolant_pump_mw = From(heat_transport)
    p_blkt_coolant_pump_mw = From(heat_transport)
    p_fw_blkt_coolant_pump_mw = From(primary_pumping)
    eta_coolant_pump_electric = From(fwbs)
    p_shld_coolant_pump_mw = From(heat_transport)
    p_div_coolant_pump_mw = From(heat_transport)
    p_blkt_breeder_pump_mw = From(heat_transport)
    p_hcd_electric_total_mw = From(heat_transport)
    p_hcd_injected_total_mw = From(current_drive)
    p_blkt_nuclear_heat_total_mw = From(fwbs)
    f_nuc_pow_bz_liq = From(fwbs)
    p_fw_nuclear_heat_total_mw = From(fwbs)
    p_fw_rad_total_mw = From(fwbs)
    p_beam_orbit_loss_mw = From(current_drive)
    p_fw_alpha_mw = From(physics)
    p_beam_shine_through_mw = From(current_drive)
    p_cp_shield_nuclear_heat_mw = From(fwbs)
    p_shld_nuclear_heat_mw = From(fwbs)
    p_plasma_separatrix_mw = From(physics)
    p_div_nuclear_heat_total_mw = From(fwbs)
    p_div_rad_total_mw = From(fwbs)
    p_fw_hcd_nuclear_heat_mw = From(fwbs)
    p_fw_hcd_rad_total_mw = From(fwbs)
    i_shld_primary_heat = From(heat_transport)


class ComponentThermalPowersSummedSolidCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_SOLID_BREEDER`, `i_thermal_electric_conversion ==
    CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_summed_solid_ccfe


class ComponentThermalPowersSummedSolidOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_SOLID_BREEDER`, `i_thermal_electric_conversion` anything else.
    """

    fn = calculate_component_thermal_powers_owned_summed_solid_other


class ComponentThermalPowersSummedLiquidCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_LIQUID_BREEDER`, `i_thermal_electric_conversion ==
    CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_summed_liquid_ccfe


class ComponentThermalPowersSummedLiquidOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_LIQUID_BREEDER`, `i_thermal_electric_conversion` anything
    else.
    """

    fn = calculate_component_thermal_powers_owned_summed_liquid_other


class ComponentThermalPowersSummedDualCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == DUAL_COOLANT`, `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_summed_dual_ccfe


class ComponentThermalPowersSummedDualOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == DUAL_COOLANT`, `i_thermal_electric_conversion` anything else.
    """

    fn = calculate_component_thermal_powers_owned_summed_dual_other


class ComponentThermalPowersMechSolidCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER`,
    `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_mech_solid_ccfe


class ComponentThermalPowersMechSolidOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER`,
    `i_thermal_electric_conversion` anything else.
    """

    fn = calculate_component_thermal_powers_owned_mech_solid_other


class ComponentThermalPowersMechLiquidCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_LIQUID_BREEDER`,
    `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_mech_liquid_ccfe


class ComponentThermalPowersMechLiquidOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_LIQUID_BREEDER`,
    `i_thermal_electric_conversion` anything else.
    """

    fn = calculate_component_thermal_powers_owned_mech_liquid_other


class ComponentThermalPowersMechDualCcfe(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == DUAL_COOLANT`, `i_thermal_electric_conversion ==
    CCFE_HCPB_VALUE`.
    """

    fn = calculate_component_thermal_powers_owned_mech_dual_ccfe


class ComponentThermalPowersMechDualOther(ComponentThermalPowers):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == DUAL_COOLANT`, `i_thermal_electric_conversion` anything
    else.
    """

    fn = calculate_component_thermal_powers_owned_mech_dual_other


class DeltaEtaStep(ExplicitFunction):
    """`.power.delta_eta`, computed from the heat flows. Bodiless family base: eight
    occupants below cover `PumpingPowerModelTypes` (binary here: pump powers
    summed or passed through) x `BlanketDualCoolantModel` (binary here --
    coarser than `ComponentThermalPowers`'s three-way use of the same switch,
    since only `calculate_p_fw_blkt_heat_deposited_mw`'s `in (1, 2)` guard reads
    it here) x `ElectricConversionModelTypes` (binary here, same split
    `calculate_delta_eta` uses). `_audit/switch_kwarg_survey.md`'s exemption for
    this node is withdrawn, the same as `ComponentThermalPowers`'s.

    `__call__` is written once here and dispatches to `self._compute` -- each arm
    below overrides only that (`WrapsFunction` cannot help, since it synthesises a
    `__call__` of its own; `AvailSt`/`_compute` in
    `models/availability/availability.py` is the precedent for this shape).

    **No longer a `FixedPointFunction`.** The entering `.power.delta_eta` is PROCESS's
    incoming field value, not a previous iterate of this node, so it is read as the
    free place `.power.delta_eta_in` (`evaluate.KNOWN_MINT_VALUES` resolves it back to
    the same field) and the node owns `.power.delta_eta` outright. The self-loop, its
    minted `^cond.power.delta_eta`, its cut and its driver are all gone. The read
    itself stays, as it always did, because PROCESS's routine really does read the
    field -- and it remains numerically inert
    (`test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta`), which is what made
    the loop a formality rather than an iteration: `g` does not depend on `u`, so the
    first evaluation was always the answer.
    """

    delta_eta = OutputInto(power)

    def __call__(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
        p_fw_blkt_coolant_pump_mw=From(primary_pumping),
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        p_blkt_breeder_pump_mw=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_cp_shield_nuclear_heat_mw=From(fwbs),
        p_shld_nuclear_heat_mw=From(fwbs),
        p_shld_coolant_pump_mw=From(heat_transport),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_div_coolant_pump_mw=From(heat_transport),
        i_shld_primary_heat=From(heat_transport),
        delta_eta_in=From(power),
    ):
        return self._compute(
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_blkt_nuclear_heat_total_mw,
            p_blkt_breeder_pump_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
            p_cp_shield_nuclear_heat_mw,
            p_shld_nuclear_heat_mw,
            p_shld_coolant_pump_mw,
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_div_coolant_pump_mw,
            i_shld_primary_heat,
            delta_eta_in,
        )


class DeltaEtaStepSummedSolidCcfe(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
    `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_summed_solid_ccfe(*args)


class DeltaEtaStepSummedSolidOther(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant
    == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
    `i_thermal_electric_conversion` anything else.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_summed_solid_other(*args)


class DeltaEtaStepSummedLiquidCcfe(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant`
    in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}` (breeder-pump term added),
    `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_summed_liquid_ccfe(*args)


class DeltaEtaStepSummedLiquidOther(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{USER_INPUT, FRACTION_OF_HEAT}`, `i_blkt_dual_coolant`
    in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}` (breeder-pump term added),
    `i_thermal_electric_conversion` anything else.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_summed_liquid_other(*args)


class DeltaEtaStepMechSolidCcfe(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
    `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_mech_solid_ccfe(*args)


class DeltaEtaStepMechSolidOther(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant == SINGLE_COOLANT_SOLID_BREEDER` (no breeder-pump term),
    `i_thermal_electric_conversion` anything else.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_mech_solid_other(*args)


class DeltaEtaStepMechLiquidCcfe(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant` in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}`
    (breeder-pump term added), `i_thermal_electric_conversion == CCFE_HCPB_VALUE`.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_mech_liquid_ccfe(*args)


class DeltaEtaStepMechLiquidOther(DeltaEtaStep):
    """`i_p_coolant_pumping` in `{MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`,
    `i_blkt_dual_coolant` in `{SINGLE_COOLANT_LIQUID_BREEDER, DUAL_COOLANT}`
    (breeder-pump term added), `i_thermal_electric_conversion` anything else.
    """

    @staticmethod
    def _compute(*args):
        return calculate_delta_eta_next_mech_liquid_other(*args)


class EtaTurbine(ExplicitFunction):
    """The `.heat_transport.eta_turbine` family -- one occupant per arm of
    `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type`.
    """

    eta_turbine = OutputInto(heat_transport)


class EtaTurbineCcfeHcpbValue(EtaTurbine):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE` (0) with `i_blanket_type ==
    CCFE_HCPB`.
    """

    def __call__(self):
        return eta_turbine_ccfe_hcpb_value()


class EtaTurbineCcfeHcpbValueWithDivertor(EtaTurbine, WrapsFunction):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE_WITH_DIVERTOR` (1) with
    `i_blanket_type == CCFE_HCPB`.
    """

    fn = eta_turbine_ccfe_hcpb_value_with_divertor

    delta_eta = From(power)


class EtaTurbineSteamRankineCycle(EtaTurbine, WrapsFunction):
    """`i_thermal_electric_conversion == STEAM_RANKINE_CYCLE` (3) with `i_blanket_type
    == CCFE_HCPB`.
    """

    fn = eta_turbine_steam_rankine_cycle

    temp_blkt_coolant_out = From(fwbs)
    delta_eta = From(power)


class EtaTurbineSupercriticalCo2(EtaTurbine, WrapsFunction):
    """`i_thermal_electric_conversion == SUPERCRITICAL_CO2_BRAYTON_CYCLE` (4), at any
    blanket type.
    """

    fn = eta_turbine_supercritical_co2

    temp_blkt_coolant_out = From(fwbs)


class EtathLiq(ExplicitFunction):
    """The `.heat_transport.etath_liq` family -- one occupant per
    `.fwbs.secondary_cycle_liq` value.
    """

    etath_liq = OutputInto(heat_transport)


class EtathLiqSupercriticalCo2(EtathLiq, WrapsFunction):
    """`secondary_cycle_liq == 4` -- the reference run's."""

    fn = etath_liq_supercritical_co2

    outlet_temp_liq = From(fwbs)


class TempTurbineCoolantIn(ExplicitFunction):
    """The `.heat_transport.temp_turbine_coolant_in` family -- one occupant per arm of
    `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type` x
    `.fwbs.secondary_cycle_liq`.
    """

    temp_turbine_coolant_in = OutputInto(heat_transport)


class TempTurbineCoolantInFromLiquidBreeder(TempTurbineCoolantIn, WrapsFunction):
    """`secondary_cycle_liq == 4` -- the reference run's."""

    fn = temp_turbine_coolant_in_from_liquid_breeder

    outlet_temp_liq = From(fwbs)


class TempTurbineCoolantInFromBlanketCoolant(TempTurbineCoolantIn, WrapsFunction):
    """`secondary_cycle_liq == 2` with an `i_thermal_electric_conversion` arm that
    writes the turbine inlet temperature -- `STEAM_RANKINE_CYCLE` with a CCFE HCPB
    blanket, or `SUPERCRITICAL_CO2_BRAYTON_CYCLE` at any blanket.
    """

    fn = temp_turbine_coolant_in_from_blanket_coolant

    temp_blkt_coolant_out = From(fwbs)


class PFwDivHeatDepositedMw(ExplicitFunction):
    """The `.heat_transport.p_fw_div_heat_deposited_mw` family -- one occupant per arm
    of `.fwbs.i_p_coolant_pumping`.
    """

    p_fw_div_heat_deposited_mw = OutputInto(heat_transport)


class PFwDivHeatDepositedMwSummed(PFwDivHeatDepositedMw, WrapsFunction):
    """`i_p_coolant_pumping != MECHANICAL_WITH_PRESSURE_DROP` -- the reference run's."""

    fn = calculate_p_fw_div_heat_deposited_mw_summed

    p_fw_nuclear_heat_total_mw = From(fwbs)
    p_fw_rad_total_mw = From(fwbs)
    p_fw_coolant_pump_mw = From(heat_transport)
    p_beam_orbit_loss_mw = From(current_drive)
    p_fw_alpha_mw = From(physics)
    p_beam_shine_through_mw = From(current_drive)
    p_plasma_separatrix_mw = From(physics)
    p_div_nuclear_heat_total_mw = From(fwbs)
    p_div_rad_total_mw = From(fwbs)
    p_div_coolant_pump_mw = From(heat_transport)


class PFwBlktCoolantPumpMw(WrapsFunction):
    """`.primary_pumping.p_fw_blkt_coolant_pump_mw` at the `.fwbs.i_p_coolant_pumping`
    values where `power` owns it: `USER_INPUT` and `FRACTION_OF_HEAT`.
    """

    fn = calculate_p_fw_blkt_coolant_pump_mw_summed

    p_fw_coolant_pump_mw = From(heat_transport)
    p_blkt_coolant_pump_mw = From(heat_transport)

    p_fw_blkt_coolant_pump_mw = OutputInto(primary_pumping)


class Cryo(ExplicitFunction):
    """cottax node: `calculate_cryo` -- unregistered scaffolding, kept only as a
    negative property: `to_graph` refuses it, because it both reads and owns
    `.fwbs.qnuc` (`test_cryo_cannot_be_a_plain_node`). The real replacement for
    `Power.cryo`/`.fwbs.qnuc` is the three-way split registered below and in
    `models/power/namespace.py` (`CryoQNuc`, `CryoQLoads`, `CryoLoads`) -- see
    `CryoQNuc`'s own docstring for the degeneracy argument that split rests on.

    Bodiless-in-spirit family base: three arms below, one per distinct behaviour of
    `.tfcoil.i_tf_sup` x `.fwbs.inuclear` (`i_tf_sup == SUPERCONDUCTING` with
    `inuclear == FRANCES_FOX` computes `.fwbs.qnuc`; the other two pass it through
    unchanged, differing only in which `calculate_cryo_q_loads` arm they take). Not a
    `WrapsFunction`: that mechanism declares reads as class attributes, which cannot
    coexist with an `OutputInto` of the same name on the same class -- exactly the
    self-loop this node is kept to demonstrate. `__call__` is written once here,
    unchanged from before de-staticizing, and dispatches to `self._compute`; each arm
    overrides only that (`DeltaEtaStep`/`AvailSt` are the precedent for this shape).
    De-staticizing `i_tf_sup`/`inuclear` into arms does not remove the self-loop and
    must not try to: `qnuc` stays declared as both a read and an output on every arm,
    deliberately, exactly as the un-split node did.
    """

    helpow = OutputInto(heat_transport)
    qss = OutputInto(power)
    qac = OutputInto(power)
    qcl = OutputInto(power)
    qmisc = OutputInto(power)
    qnuc = OutputInto(fwbs)

    def __call__(
        self,
        tfcryoarea=From(tfcoil),
        coldmass=From(structure),
        p_tf_nuclear_heat_mw=From(fwbs),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
        c_tf_turn=From(tfcoil),
        n_tf_coils=From(tfcoil),
        qnuc=From(fwbs),
    ):
        return self._compute(
            tfcryoarea,
            coldmass,
            p_tf_nuclear_heat_mw,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoSuperconductingComputed(Cryo):
    """`i_tf_sup == SUPERCONDUCTING`, `inuclear == FRANCES_FOX` -- `.fwbs.qnuc` is
    computed, not read, but the read stays declared regardless (see `Cryo`'s
    docstring).
    """

    @staticmethod
    def _compute(*args):
        return calculate_cryo_superconducting_computed(*args)


class CryoSuperconductingUserInput(Cryo):
    """`i_tf_sup == SUPERCONDUCTING`, `inuclear == USER_INPUT` -- `.fwbs.qnuc` passed
    through unchanged.
    """

    @staticmethod
    def _compute(*args):
        return calculate_cryo_superconducting_user_input(*args)


class CryoNonSuperconducting(Cryo):
    """`i_tf_sup != SUPERCONDUCTING` -- resistive `calculate_cryo_q_loads` arm,
    `.fwbs.qnuc` passed through unchanged regardless of `inuclear`.
    """

    @staticmethod
    def _compute(*args):
        return calculate_cryo_non_superconducting(*args)


class CryoQNuc(WrapsFunction):
    """`.fwbs.qnuc` when PROCESS computes it: `inuclear == 0` and `i_tf_sup == 1`."""

    fn = calculate_cryo_qnuc_when_computed

    p_tf_nuclear_heat_mw = From(fwbs)

    qnuc = OutputInto(fwbs)


class CryoQLoads(ExplicitFunction):
    """The `.power.qss`/`qac`/`qcl`/`qmisc` family -- one occupant per arm of
    `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.
    """

    qss = OutputInto(power)
    qac = OutputInto(power)
    qcl = OutputInto(power)
    qmisc = OutputInto(power)


class CryoQLoadsSuperconductingTf(CryoQLoads, WrapsFunction):
    """`i_tf_sup == SUPERCONDUCTING` (1) -- the reference run's."""

    fn = calculate_cryo_q_loads_superconducting_tf

    tfcryoarea = From(tfcoil)
    coldmass = From(structure)
    ensxpfm = From(pf_power)
    t_plant_pulse_plasma_present = From(times)
    c_tf_turn = From(tfcoil)
    n_tf_coils = From(tfcoil)
    qnuc = From(fwbs)


class CryoQLoadsResistiveTf(CryoQLoads, WrapsFunction):
    """`i_tf_sup != SUPERCONDUCTING` with `i_pf_conductor == SUPERCONDUCTING` -- the PF
    coils are what needs cooling.
    """

    fn = calculate_cryo_q_loads_resistive_tf

    coldmass = From(structure)
    ensxpfm = From(pf_power)
    t_plant_pulse_plasma_present = From(times)
    qnuc = From(fwbs)


class CryoLoads(ExplicitFunction):
    """The unconditionally-owned part of `Power.calculate_cryo_loads` -- one occupant
    per arm of `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.
    """

    helpow = OutputInto(heat_transport)
    p_cryo_plant_electric_mw = OutputInto(heat_transport)
    helpow_cryal = OutputInto(heat_transport)
    cryo_cool_req = OutputInto(tfcoil)


class CryoLoadsActive(CryoLoads, WrapsFunction):
    """`i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` -- the reference run's."""

    fn = calculate_cryo_plant_loads_active

    eff_tf_cryo = From(tfcoil)
    temp_tf_cryo = From(tfcoil)
    temp_cp_coolant_inlet = From(tfcoil)
    qss = From(power)
    qac = From(power)
    qcl = From(power)
    qmisc = From(power)
    qnuc = From(fwbs)


class CryoLoadsInactive(CryoLoads, WrapsFunction):
    """`i_tf_sup != 1` with resistive PF coils: PROCESS never calls `Power.cryo`, so
    `helpow` and `p_cryo_plant_electric_mw` are literal zeros.
    """

    fn = calculate_cryo_plant_loads_inactive

    temp_tf_cryo = From(tfcoil)
    temp_cp_coolant_inlet = From(tfcoil)
