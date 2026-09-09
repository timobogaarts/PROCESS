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

from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CoilNuclearHeatingModel,
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
from functional_process.models.power.thermal_cryo import (
    calculate_component_thermal_powers,
    calculate_component_thermal_powers_owned,
    calculate_cryo,
    calculate_cryo_loads,
    calculate_cryo_plant_loads,
    calculate_cryo_plant_loads_active,
    calculate_cryo_plant_loads_inactive,
    calculate_cryo_q_loads,
    calculate_cryo_q_loads_resistive_tf,
    calculate_cryo_q_loads_superconducting_tf,
    calculate_cryo_qnuc,
    calculate_cryo_qnuc_when_computed,
    calculate_delta_eta,
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
    "Cryo",
    "CryoLoads",
    "CryoLoadsActive",
    "CryoLoadsInactive",
    "CryoQLoads",
    "CryoQLoadsResistiveTf",
    "CryoQLoadsSuperconductingTf",
    "CryoQNuc",
    "CryoQNucStep",
    "DeltaEtaStep",
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
    "calculate_cryo",
    "calculate_cryo_loads",
    "calculate_cryo_plant_loads",
    "calculate_cryo_plant_loads_active",
    "calculate_cryo_plant_loads_inactive",
    "calculate_cryo_q_loads",
    "calculate_cryo_q_loads_resistive_tf",
    "calculate_cryo_q_loads_superconducting_tf",
    "calculate_cryo_qnuc",
    "calculate_cryo_qnuc_when_computed",
    "calculate_delta_eta",
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


class ComponentThermalPowers(ExplicitFunction):
    """cottax node: `calculate_component_thermal_powers`'s outputs **other than** the
    six self-referencing fields split into their own `FixedPointFunction`s below
    (`DeltaEtaStep`, `EtaTurbineStep`, `EtathLiqStep`, `TempTurbineCoolantInStep`,
    `PFwDivHeatDepositedMwStep`, `PFwBlktCoolantPumpMwStep`).
    """

    i_p_coolant_pumping: PumpingPowerModelTypes = eqx.field(static=True)
    i_blkt_dual_coolant: BlanketDualCoolantModel = eqx.field(static=True)
    i_thermal_electric_conversion: ElectricConversionModelTypes = eqx.field(static=True)

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
    # .temp_turbine_coolant_in are NOT declared here -- each is owned by its own family
    # below -- and, since `_audit/next_steps.md` §14.2, **not read here either**: this
    # node recomputed all four and discarded the results.
    p_plant_primary_heat_mw = OutputInto(heat_transport)
    p_div_secondary_heat_mw = OutputInto(heat_transport)
    i_div_primary_heat = OutputInto(power)
    f_p_div_primary_heat = OutputInto(power)
    # .power.delta_eta is NOT declared here -- DeltaEtaStep's FixedPoint problem node
    # owns it (see "The delta_eta self-loop" in thermal_cryo.md) -- and is no longer
    # read here either, for the same reason as the four above.
    p_shld_secondary_heat_mw = OutputInto(heat_transport)
    p_hcd_secondary_heat_mw = OutputInto(heat_transport)
    n_primary_heat_exchangers = OutputInto(heat_transport)

    def __call__(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
        p_fw_blkt_coolant_pump_mw=From(primary_pumping),
        eta_coolant_pump_electric=From(fwbs),
        p_shld_coolant_pump_mw=From(heat_transport),
        p_div_coolant_pump_mw=From(heat_transport),
        p_blkt_breeder_pump_mw=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_hcd_injected_total_mw=From(current_drive),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        f_nuc_pow_bz_liq=From(fwbs),
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_cp_shield_nuclear_heat_mw=From(fwbs),
        p_shld_nuclear_heat_mw=From(fwbs),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_fw_hcd_nuclear_heat_mw=From(fwbs),
        p_fw_hcd_rad_total_mw=From(fwbs),
        i_shld_primary_heat=From(heat_transport),
    ):
        return calculate_component_thermal_powers_owned(
            self.i_p_coolant_pumping,
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
            eta_coolant_pump_electric,
            p_shld_coolant_pump_mw,
            p_div_coolant_pump_mw,
            p_blkt_breeder_pump_mw,
            p_hcd_electric_total_mw,
            p_hcd_injected_total_mw,
            self.i_blkt_dual_coolant,
            p_blkt_nuclear_heat_total_mw,
            f_nuc_pow_bz_liq,
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
            p_cp_shield_nuclear_heat_mw,
            p_shld_nuclear_heat_mw,
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_fw_hcd_nuclear_heat_mw,
            p_fw_hcd_rad_total_mw,
            i_shld_primary_heat,
            self.i_thermal_electric_conversion,
        )


class DeltaEtaStep(FixedPointFunction):
    """The `.power.delta_eta` self-loop, cut."""

    i_p_coolant_pumping: PumpingPowerModelTypes = eqx.field(static=True)
    i_blkt_dual_coolant: BlanketDualCoolantModel = eqx.field(static=True)
    i_thermal_electric_conversion: ElectricConversionModelTypes = eqx.field(static=True)

    delta_eta = OutputInto(power)

    def step(
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
        delta_eta=From(power),
    ):
        del delta_eta  # see class docstring -- verified numerically inert here

        p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(
            self.i_p_coolant_pumping,
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
        )
        p_fw_blkt_heat_deposited_mw = calculate_p_fw_blkt_heat_deposited_mw(
            self.i_blkt_dual_coolant,
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_blkt_nuclear_heat_total_mw,
            p_blkt_breeder_pump_mw,
            p_fw_blkt_coolant_pump_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
        )
        p_shld_heat_deposited_mw = calculate_p_shld_heat_deposited_mw(
            p_cp_shield_nuclear_heat_mw, p_shld_nuclear_heat_mw, p_shld_coolant_pump_mw
        )
        p_div_heat_deposited_mw = calculate_p_div_heat_deposited_mw(
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_div_coolant_pump_mw,
        )
        _, _, _, _, delta_eta_next = calculate_delta_eta(
            p_fw_blkt_heat_deposited_mw,
            i_shld_primary_heat,
            p_shld_heat_deposited_mw,
            p_div_heat_deposited_mw,
            self.i_thermal_electric_conversion,
        )
        return delta_eta_next


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


class EtaTurbineCcfeHcpbValueWithDivertor(EtaTurbine):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE_WITH_DIVERTOR` (1) with
    `i_blanket_type == CCFE_HCPB`.
    """

    def __call__(self, delta_eta=From(power)):
        return eta_turbine_ccfe_hcpb_value_with_divertor(delta_eta)


class EtaTurbineSteamRankineCycle(EtaTurbine):
    """`i_thermal_electric_conversion == STEAM_RANKINE_CYCLE` (3) with `i_blanket_type
    == CCFE_HCPB`.
    """

    def __call__(
        self,
        temp_blkt_coolant_out=From(fwbs),
        delta_eta=From(power),
    ):
        return eta_turbine_steam_rankine_cycle(temp_blkt_coolant_out, delta_eta)


class EtaTurbineSupercriticalCo2(EtaTurbine):
    """`i_thermal_electric_conversion == SUPERCRITICAL_CO2_BRAYTON_CYCLE` (4), at any
    blanket type.
    """

    def __call__(self, temp_blkt_coolant_out=From(fwbs)):
        return eta_turbine_supercritical_co2(temp_blkt_coolant_out)


class EtathLiq(ExplicitFunction):
    """The `.heat_transport.etath_liq` family -- one occupant per
    `.fwbs.secondary_cycle_liq` value.
    """

    etath_liq = OutputInto(heat_transport)


class EtathLiqSupercriticalCo2(EtathLiq):
    """`secondary_cycle_liq == 4` -- the reference run's."""

    def __call__(self, outlet_temp_liq=From(fwbs)):
        return etath_liq_supercritical_co2(outlet_temp_liq)


class TempTurbineCoolantIn(ExplicitFunction):
    """The `.heat_transport.temp_turbine_coolant_in` family -- one occupant per arm of
    `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type` x
    `.fwbs.secondary_cycle_liq`.
    """

    temp_turbine_coolant_in = OutputInto(heat_transport)


class TempTurbineCoolantInFromLiquidBreeder(TempTurbineCoolantIn):
    """`secondary_cycle_liq == 4` -- the reference run's."""

    def __call__(self, outlet_temp_liq=From(fwbs)):
        return temp_turbine_coolant_in_from_liquid_breeder(outlet_temp_liq)


class TempTurbineCoolantInFromBlanketCoolant(TempTurbineCoolantIn):
    """`secondary_cycle_liq == 2` with an `i_thermal_electric_conversion` arm that
    writes the turbine inlet temperature -- `STEAM_RANKINE_CYCLE` with a CCFE HCPB
    blanket, or `SUPERCRITICAL_CO2_BRAYTON_CYCLE` at any blanket.
    """

    def __call__(self, temp_blkt_coolant_out=From(fwbs)):
        return temp_turbine_coolant_in_from_blanket_coolant(temp_blkt_coolant_out)


class PFwDivHeatDepositedMw(ExplicitFunction):
    """The `.heat_transport.p_fw_div_heat_deposited_mw` family -- one occupant per arm
    of `.fwbs.i_p_coolant_pumping`.
    """

    p_fw_div_heat_deposited_mw = OutputInto(heat_transport)


class PFwDivHeatDepositedMwSummed(PFwDivHeatDepositedMw):
    """`i_p_coolant_pumping != MECHANICAL_WITH_PRESSURE_DROP` -- the reference run's."""

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_fw_coolant_pump_mw=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_div_coolant_pump_mw=From(heat_transport),
    ):
        return calculate_p_fw_div_heat_deposited_mw_summed(
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_fw_coolant_pump_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_div_coolant_pump_mw,
        )


class PFwBlktCoolantPumpMw(ExplicitFunction):
    """`.primary_pumping.p_fw_blkt_coolant_pump_mw` at the `.fwbs.i_p_coolant_pumping`
    values where `power` owns it: `USER_INPUT` and `FRACTION_OF_HEAT`.
    """

    p_fw_blkt_coolant_pump_mw = OutputInto(primary_pumping)

    def __call__(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
    ):
        return calculate_p_fw_blkt_coolant_pump_mw_summed(
            p_fw_coolant_pump_mw, p_blkt_coolant_pump_mw
        )


class Cryo(ExplicitFunction):
    """cottax node: `calculate_cryo`."""

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    inuclear: CoilNuclearHeatingModel = eqx.field(static=True)

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
        return calculate_cryo(
            self.i_tf_sup,
            self.inuclear,
            tfcryoarea,
            coldmass,
            p_tf_nuclear_heat_mw,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoQNuc(ExplicitFunction):
    """`.fwbs.qnuc` when PROCESS computes it: `inuclear == 0` and `i_tf_sup == 1`."""

    qnuc = OutputInto(fwbs)

    def __call__(self, p_tf_nuclear_heat_mw=From(fwbs)):
        return calculate_cryo_qnuc_when_computed(p_tf_nuclear_heat_mw)


class CryoQNucStep(FixedPointFunction):
    """The `.fwbs.qnuc` self-loop, cut."""

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    inuclear: CoilNuclearHeatingModel = eqx.field(static=True)

    qnuc = OutputInto(fwbs)

    def step(
        self,
        qnuc=From(fwbs),
        p_tf_nuclear_heat_mw=From(fwbs),
    ):
        return calculate_cryo_qnuc(
            self.i_tf_sup, self.inuclear, p_tf_nuclear_heat_mw, qnuc
        )


class CryoQLoads(ExplicitFunction):
    """The `.power.qss`/`qac`/`qcl`/`qmisc` family -- one occupant per arm of
    `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.
    """

    qss = OutputInto(power)
    qac = OutputInto(power)
    qcl = OutputInto(power)
    qmisc = OutputInto(power)


class CryoQLoadsSuperconductingTf(CryoQLoads):
    """`i_tf_sup == SUPERCONDUCTING` (1) -- the reference run's."""

    def __call__(
        self,
        qnuc=From(fwbs),
        tfcryoarea=From(tfcoil),
        coldmass=From(structure),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
        c_tf_turn=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return calculate_cryo_q_loads_superconducting_tf(
            tfcryoarea,
            coldmass,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoQLoadsResistiveTf(CryoQLoads):
    """`i_tf_sup != SUPERCONDUCTING` with `i_pf_conductor == SUPERCONDUCTING` -- the PF
    coils are what needs cooling.
    """

    def __call__(
        self,
        qnuc=From(fwbs),
        coldmass=From(structure),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
    ):
        return calculate_cryo_q_loads_resistive_tf(
            coldmass, ensxpfm, t_plant_pulse_plasma_present, qnuc
        )


class CryoLoads(ExplicitFunction):
    """The unconditionally-owned part of `Power.calculate_cryo_loads` -- one occupant
    per arm of `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.
    """

    helpow = OutputInto(heat_transport)
    p_cryo_plant_electric_mw = OutputInto(heat_transport)
    helpow_cryal = OutputInto(heat_transport)
    cryo_cool_req = OutputInto(tfcoil)


class CryoLoadsActive(CryoLoads):
    """`i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` -- the reference run's."""

    def __call__(
        self,
        eff_tf_cryo=From(tfcoil),
        temp_tf_cryo=From(tfcoil),
        temp_cp_coolant_inlet=From(tfcoil),
        qss=From(power),
        qac=From(power),
        qcl=From(power),
        qmisc=From(power),
        qnuc=From(fwbs),
    ):
        return calculate_cryo_plant_loads_active(
            eff_tf_cryo,
            temp_tf_cryo,
            temp_cp_coolant_inlet,
            qss,
            qac,
            qcl,
            qmisc,
            qnuc,
        )


class CryoLoadsInactive(CryoLoads):
    """`i_tf_sup != 1` with resistive PF coils: PROCESS never calls `Power.cryo`, so
    `helpow` and `p_cryo_plant_electric_mw` are literal zeros.
    """

    def __call__(
        self,
        temp_tf_cryo=From(tfcoil),
        temp_cp_coolant_inlet=From(tfcoil),
    ):
        return calculate_cryo_plant_loads_inactive(temp_tf_cryo, temp_cp_coolant_inlet)
