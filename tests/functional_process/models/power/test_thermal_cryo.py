"""Harness cases for `functional_process/cottax/power/thermal_cryo.py`.

Audit record: `functional_process/_audit/units/models/power/thermal_cryo.md`. No
legacy points exist in `tests/unit/models/test_power.py` for
`component_thermal_powers`/`plant_thermal_efficiency`/`plant_thermal_efficiency_2`
(fuzz-only, same situation as chunk A); `cryo` does have legacy points there, reused
below.

The node-level tests at the bottom of this file (`DeltaEtaStep`/`ComponentThermalPowers`
split) are new -- see `thermal_cryo.md`'s "The `delta_eta` self-loop" section
and `DeltaEtaStep`'s own docstring for the full reasoning.
"""

import inspect

import jax
import pytest
from cottax.interfaces.pytree_namespace_module import to_graph

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.indat import (
    CRYO_LOADS,
    CRYO_Q_LOADS,
    ETA_TURBINE,
    ETATH_LIQ,
    P_FW_BLKT_COOLANT_PUMP,
    P_FW_DIV_HEAT_DEPOSITED,
    TEMP_TURBINE_COOLANT_IN,
    _cryo_loads_arm,
    _cryo_q_loads_arm,
    _eta_turbine_arm,
    _p_fw_blkt_coolant_pump_arm,
    _p_fw_div_heat_deposited_arm,
    _temp_turbine_coolant_in_arm,
)
from functional_process.cottax.power.thermal_cryo import (
    ComponentThermalPowers,
    Cryo,
    CryoQNucStep,
    DeltaEtaStep,
    calculate_component_thermal_powers,
    calculate_cryo,
    calculate_cryo_loads,
    calculate_plant_thermal_efficiency,
    calculate_plant_thermal_efficiency_2,
)
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CoilNuclearHeatingModel,
)
from process.core.model import DataStructure
from process.data_structure.blanket_variables import BlktModelTypes
from process.data_structure.pfcoil_variables import PFConductorModel
from process.models.power import (
    ElectricConversionModelTypes,
    Power,
    PumpingPowerModelTypes,
)
from process.models.tfcoil.base import TFConductorModel

# ---------------------------------------------------------------------------
# plant_thermal_efficiency
# ---------------------------------------------------------------------------


def _reference_plant_thermal_efficiency(
    eta_turbine,
    delta_eta,
    temp_blkt_coolant_out,
    temp_turbine_coolant_in,
    i_thermal_electric_conversion,
    i_blanket_type,
):
    data = DataStructure()
    data.fwbs.i_thermal_electric_conversion = i_thermal_electric_conversion
    data.fwbs.i_blanket_type = i_blanket_type
    data.power.delta_eta = delta_eta
    data.fwbs.temp_blkt_coolant_out = temp_blkt_coolant_out
    data.heat_transport.temp_turbine_coolant_in = temp_turbine_coolant_in

    p = Power()
    p.data = data
    eta_turbine_out = p.plant_thermal_efficiency(eta_turbine)
    return eta_turbine_out, data.heat_transport.temp_turbine_coolant_in


def _plant_thermal_efficiency_samples():
    """One fuzz batch per non-crashing `i_thermal_electric_conversion` value.

    `i_thermal_electric_conversion == 4` (`SUPERCRITICAL_CO2_BRAYTON_CYCLE`) is
    deliberately excluded -- PROCESS's own reference crashes with `AttributeError`
    before computing anything there, see `thermal_cryo.md`'s real finding
    and `calculate_plant_thermal_efficiency`'s docstring.
    """
    bounds = {
        "eta_turbine": (0.2, 0.6),
        "delta_eta": (0.0, 0.2),
        "temp_blkt_coolant_out": (500.0, 900.0),
        "temp_turbine_coolant_in": (400.0, 900.0),
    }
    samples = []
    for value in (
        ElectricConversionModelTypes.CCFE_HCPB_VALUE,
        ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR,
        ElectricConversionModelTypes.USER_INPUT,
        ElectricConversionModelTypes.STEAM_RANKINE_CYCLE,
    ):
        fixed = {
            "i_thermal_electric_conversion": int(value),
            "i_blanket_type": int(BlktModelTypes.CCFE_HCPB),
        }
        samples.extend(
            fuzz_samples(bounds, count=15, seed=20260818 + int(value), fixed=fixed)
        )
    # NOTE: `i_blanket_type != CCFE_HCPB` inside these same three branches is *not*
    # exercised here -- PROCESS's own `logger.log(msg)` call on that path
    # (`power.py:2033`/2989/2989-ish) passes a single positional argument to
    # `logging.Logger.log`, which requires `(level, msg, *args)`, so PROCESS itself
    # raises `TypeError: Logger.log() missing 1 required positional argument: 'msg'`
    # for *any* input reaching that branch -- confirmed by fuzzing this exact case
    # during this port's construction (see `thermal_cryo.md`'s real
    # findings). The port's own "else: pass eta_turbine through unchanged" behaviour
    # is unambiguous from the branch's structure and cannot be diffed against a
    # PROCESS reference that never returns.
    return samples


class TestPlantThermalEfficiency(Tier1Contract):
    audit_record = "models/power/thermal_cryo.md"
    reference = _reference_plant_thermal_efficiency
    ported = calculate_plant_thermal_efficiency
    static_argnames = ("i_thermal_electric_conversion", "i_blanket_type")
    samples = _plant_thermal_efficiency_samples()


# ---------------------------------------------------------------------------
# plant_thermal_efficiency_2
# ---------------------------------------------------------------------------


def _reference_plant_thermal_efficiency_2(
    etath_liq, outlet_temp_liq, temp_turbine_coolant_in, secondary_cycle_liq
):
    data = DataStructure()
    data.fwbs.secondary_cycle_liq = secondary_cycle_liq
    data.fwbs.outlet_temp_liq = outlet_temp_liq
    data.heat_transport.temp_turbine_coolant_in = temp_turbine_coolant_in

    p = Power()
    p.data = data
    etath_liq_out = p.plant_thermal_efficiency_2(etath_liq)
    return etath_liq_out, data.heat_transport.temp_turbine_coolant_in


def _plant_thermal_efficiency_2_samples():
    bounds = {
        "etath_liq": (0.2, 0.6),
        "outlet_temp_liq": (500.0, 900.0),
        "temp_turbine_coolant_in": (400.0, 900.0),
    }
    samples = []
    for value in (2, 4):
        samples.extend(
            fuzz_samples(
                bounds,
                count=20,
                seed=30260818 + value,
                fixed={"secondary_cycle_liq": value},
            )
        )
    return samples


class TestPlantThermalEfficiency2(Tier1Contract):
    audit_record = "models/power/thermal_cryo.md"
    reference = _reference_plant_thermal_efficiency_2
    ported = calculate_plant_thermal_efficiency_2
    static_argnames = ("secondary_cycle_liq",)
    samples = _plant_thermal_efficiency_2_samples()


# ---------------------------------------------------------------------------
# component_thermal_powers
# ---------------------------------------------------------------------------

_CTP_STATIC_ARGNAMES = (
    "i_p_coolant_pumping",
    "i_blkt_dual_coolant",
    "i_thermal_electric_conversion",
    "i_blanket_type",
    "secondary_cycle_liq",
)


def _reference_component_thermal_powers(**kwargs):
    data = DataStructure()
    data.fwbs.i_p_coolant_pumping = kwargs["i_p_coolant_pumping"]
    data.heat_transport.p_fw_coolant_pump_mw = kwargs["p_fw_coolant_pump_mw"]
    data.heat_transport.p_blkt_coolant_pump_mw = kwargs["p_blkt_coolant_pump_mw"]
    data.primary_pumping.p_fw_blkt_coolant_pump_mw = kwargs["p_fw_blkt_coolant_pump_mw"]
    data.fwbs.eta_coolant_pump_electric = kwargs["eta_coolant_pump_electric"]
    data.heat_transport.p_shld_coolant_pump_mw = kwargs["p_shld_coolant_pump_mw"]
    data.heat_transport.p_div_coolant_pump_mw = kwargs["p_div_coolant_pump_mw"]
    data.heat_transport.p_blkt_breeder_pump_mw = kwargs["p_blkt_breeder_pump_mw"]
    data.heat_transport.p_hcd_electric_total_mw = kwargs["p_hcd_electric_total_mw"]
    data.current_drive.p_hcd_injected_total_mw = kwargs["p_hcd_injected_total_mw"]
    data.fwbs.i_blkt_dual_coolant = kwargs["i_blkt_dual_coolant"]
    data.fwbs.p_blkt_nuclear_heat_total_mw = kwargs["p_blkt_nuclear_heat_total_mw"]
    data.fwbs.f_nuc_pow_bz_liq = kwargs["f_nuc_pow_bz_liq"]
    data.fwbs.p_fw_nuclear_heat_total_mw = kwargs["p_fw_nuclear_heat_total_mw"]
    data.fwbs.p_fw_rad_total_mw = kwargs["p_fw_rad_total_mw"]
    data.current_drive.p_beam_orbit_loss_mw = kwargs["p_beam_orbit_loss_mw"]
    data.physics.p_fw_alpha_mw = kwargs["p_fw_alpha_mw"]
    data.current_drive.p_beam_shine_through_mw = kwargs["p_beam_shine_through_mw"]
    data.fwbs.p_cp_shield_nuclear_heat_mw = kwargs["p_cp_shield_nuclear_heat_mw"]
    data.fwbs.p_shld_nuclear_heat_mw = kwargs["p_shld_nuclear_heat_mw"]
    data.physics.p_plasma_separatrix_mw = kwargs["p_plasma_separatrix_mw"]
    data.fwbs.p_div_nuclear_heat_total_mw = kwargs["p_div_nuclear_heat_total_mw"]
    data.fwbs.p_div_rad_total_mw = kwargs["p_div_rad_total_mw"]
    data.heat_transport.p_fw_div_heat_deposited_mw = kwargs["p_fw_div_heat_deposited_mw"]
    data.fwbs.p_fw_hcd_nuclear_heat_mw = kwargs["p_fw_hcd_nuclear_heat_mw"]
    data.fwbs.p_fw_hcd_rad_total_mw = kwargs["p_fw_hcd_rad_total_mw"]
    data.heat_transport.i_shld_primary_heat = kwargs["i_shld_primary_heat"]
    data.fwbs.i_thermal_electric_conversion = kwargs["i_thermal_electric_conversion"]
    data.fwbs.i_blanket_type = kwargs["i_blanket_type"]
    data.heat_transport.eta_turbine = kwargs["eta_turbine"]
    data.heat_transport.etath_liq = kwargs["etath_liq"]
    data.power.delta_eta = kwargs["delta_eta"]
    data.fwbs.temp_blkt_coolant_out = kwargs["temp_blkt_coolant_out"]
    data.fwbs.outlet_temp_liq = kwargs["outlet_temp_liq"]
    data.heat_transport.temp_turbine_coolant_in = kwargs["temp_turbine_coolant_in"]
    data.fwbs.secondary_cycle_liq = kwargs["secondary_cycle_liq"]
    # ireactor left at its DataStructure default (not read by this method).

    p = Power()
    p.data = data
    p.component_thermal_powers()

    return (
        data.primary_pumping.p_fw_blkt_coolant_pump_mw,
        data.power.p_fw_blkt_coolant_pump_elec_mw,
        data.power.p_shld_coolant_pump_elec_mw,
        data.power.p_div_coolant_pump_elec_mw,
        data.power.p_blkt_breeder_pump_elec_mw,
        data.power.p_coolant_pump_total_mw,
        data.heat_transport.p_coolant_pump_elec_total_mw,
        data.heat_transport.p_coolant_pump_loss_total_mw,
        data.heat_transport.p_hcd_electric_loss_mw,
        data.power.p_blkt_liquid_breeder_heat_deposited_mw,
        data.power.p_fw_blkt_heat_deposited_mw,
        data.power.p_fw_heat_deposited_mw,
        data.power.p_blkt_heat_deposited_mw,
        data.power.p_shld_heat_deposited_mw,
        data.power.p_div_heat_deposited_mw,
        data.heat_transport.p_fw_div_heat_deposited_mw,
        data.heat_transport.eta_turbine,
        data.heat_transport.etath_liq,
        data.heat_transport.temp_turbine_coolant_in,
        data.heat_transport.p_plant_primary_heat_mw,
        data.heat_transport.p_div_secondary_heat_mw,
        data.power.i_div_primary_heat,
        data.power.f_p_div_primary_heat,
        data.power.delta_eta,
        data.heat_transport.p_shld_secondary_heat_mw,
        data.heat_transport.p_hcd_secondary_heat_mw,
        data.heat_transport.n_primary_heat_exchangers,
    )


def _component_thermal_powers_samples():
    bounds = {
        "p_fw_coolant_pump_mw": (0.0, 50.0),
        "p_blkt_coolant_pump_mw": (0.0, 100.0),
        "p_fw_blkt_coolant_pump_mw": (0.0, 150.0),
        "eta_coolant_pump_electric": (0.7, 0.95),
        "p_shld_coolant_pump_mw": (0.0, 20.0),
        "p_div_coolant_pump_mw": (0.0, 20.0),
        "p_blkt_breeder_pump_mw": (0.0, 10.0),
        "p_hcd_electric_total_mw": (100.0, 300.0),
        "p_hcd_injected_total_mw": (50.0, 100.0),
        "p_blkt_nuclear_heat_total_mw": (100.0, 1500.0),
        "f_nuc_pow_bz_liq": (0.0, 1.0),
        "p_fw_nuclear_heat_total_mw": (10.0, 200.0),
        "p_fw_rad_total_mw": (10.0, 400.0),
        "p_beam_orbit_loss_mw": (0.0, 10.0),
        "p_fw_alpha_mw": (0.0, 50.0),
        "p_beam_shine_through_mw": (0.0, 5.0),
        "p_cp_shield_nuclear_heat_mw": (0.0, 50.0),
        "p_shld_nuclear_heat_mw": (0.0, 100.0),
        "p_plasma_separatrix_mw": (50.0, 300.0),
        "p_div_nuclear_heat_total_mw": (0.0, 50.0),
        "p_div_rad_total_mw": (0.0, 50.0),
        "p_fw_div_heat_deposited_mw": (50.0, 800.0),
        "p_fw_hcd_nuclear_heat_mw": (0.0, 20.0),
        "p_fw_hcd_rad_total_mw": (0.0, 20.0),
        "i_shld_primary_heat": (0.0, 1.0),
        "eta_turbine": (0.2, 0.6),
        "etath_liq": (0.2, 0.6),
        "delta_eta": (0.0, 0.2),
        "temp_blkt_coolant_out": (500.0, 900.0),
        "outlet_temp_liq": (500.0, 900.0),
        "temp_turbine_coolant_in": (400.0, 900.0),
    }
    samples = []
    combos = [
        (
            PumpingPowerModelTypes.FRACTION_OF_HEAT,
            0,
            ElectricConversionModelTypes.CCFE_HCPB_VALUE,
            BlktModelTypes.CCFE_HCPB,
            2,
        ),
        (
            PumpingPowerModelTypes.USER_INPUT,
            1,
            ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR,
            BlktModelTypes.CCFE_HCPB,
            4,
        ),
        (
            PumpingPowerModelTypes.MECHANICAL,
            2,
            ElectricConversionModelTypes.STEAM_RANKINE_CYCLE,
            BlktModelTypes.CCFE_HCPB,
            2,
        ),
        (
            PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP,
            0,
            ElectricConversionModelTypes.USER_INPUT,
            BlktModelTypes.DCLL,
            4,
        ),
    ]
    for i_p_coolant_pumping, i_blkt_dual_coolant, i_tec, i_bt, scl in combos:
        fixed = {
            "i_p_coolant_pumping": int(i_p_coolant_pumping),
            "i_blkt_dual_coolant": i_blkt_dual_coolant,
            "i_thermal_electric_conversion": int(i_tec),
            "i_blanket_type": int(i_bt),
            "secondary_cycle_liq": scl,
        }
        samples.extend(
            fuzz_samples(
                bounds,
                count=15,
                seed=40260818 + int(i_p_coolant_pumping) + int(i_tec),
                fixed=fixed,
            )
        )
    return samples


class TestComponentThermalPowers(Tier1Contract):
    audit_record = "models/power/thermal_cryo.md"
    reference = _reference_component_thermal_powers
    ported = calculate_component_thermal_powers
    static_argnames = _CTP_STATIC_ARGNAMES
    samples = _component_thermal_powers_samples()


# ---------------------------------------------------------------------------
# cryo
# ---------------------------------------------------------------------------


def _reference_cryo(
    i_tf_sup,
    inuclear,
    tfcryoarea,
    coldmass,
    p_tf_nuclear_heat_mw,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
):
    data = DataStructure()
    data.fwbs.qnuc = qnuc
    data.fwbs.inuclear = inuclear

    p = Power()
    p.data = data
    helpow = p.cryo(
        i_tf_sup=i_tf_sup,
        coldmass=coldmass,
        c_tf_turn=c_tf_turn,
        ensxpfm=ensxpfm,
        p_tf_nuclear_heat_mw=p_tf_nuclear_heat_mw,
        n_tf_coils=n_tf_coils,
        tfcryoarea=tfcryoarea,
        t_plant_pulse_plasma_present=t_plant_pulse_plasma_present,
    )
    return (
        helpow,
        data.power.qss,
        data.power.qac,
        data.power.qcl,
        data.power.qmisc,
        data.fwbs.qnuc,
    )


class TestCryo(Tier1Contract):
    audit_record = "models/power/thermal_cryo.md"
    reference = _reference_cryo
    ported = calculate_cryo
    static_argnames = ("i_tf_sup", "inuclear")

    # tests/unit/models/test_power.py::test_cryo, both parametrised legacy points.
    samples = [
        legacy_sample(
            "baseline-2018-point-1",
            i_tf_sup=1,
            inuclear=1,
            coldmass=47352637.039762333,
            c_tf_turn=74026.751437500003,
            ensxpfm=37429.525515086898,
            p_tf_nuclear_heat_mw=0.044178296011112193,
            n_tf_coils=16,
            tfcryoarea=0.0,
            t_plant_pulse_plasma_present=10364.426139387357,
            qnuc=12920.0,
        ),
        legacy_sample(
            "baseline-2018-point-2",
            i_tf_sup=1,
            inuclear=1,
            coldmass=47308985.527808741,
            c_tf_turn=74026.751437500003,
            ensxpfm=37427.228965055205,
            p_tf_nuclear_heat_mw=0.045535131445547841,
            n_tf_coils=16,
            tfcryoarea=0.0,
            t_plant_pulse_plasma_present=364.42613938735633,
            qnuc=12920.0,
        ),
        *fuzz_samples(
            {
                "coldmass": (1.0e6, 6.0e7),
                "c_tf_turn": (1.0e3, 1.0e5),
                "ensxpfm": (1.0e3, 6.0e4),
                "p_tf_nuclear_heat_mw": (0.0, 1.0),
                "n_tf_coils": (10.0, 24.0),
                "tfcryoarea": (0.0, 5000.0),
                "t_plant_pulse_plasma_present": (300.0, 12000.0),
                "qnuc": (0.0, 20000.0),
            },
            count=20,
            seed=50260818,
            fixed={"i_tf_sup": 1, "inuclear": 0},
        ),
        *fuzz_samples(
            {
                "coldmass": (1.0e6, 6.0e7),
                "c_tf_turn": (1.0e3, 1.0e5),
                "ensxpfm": (1.0e3, 6.0e4),
                "p_tf_nuclear_heat_mw": (0.0, 1.0),
                "n_tf_coils": (10.0, 24.0),
                "tfcryoarea": (0.0, 5000.0),
                "t_plant_pulse_plasma_present": (300.0, 12000.0),
                "qnuc": (0.0, 20000.0),
            },
            count=15,
            seed=60260818,
            fixed={"i_tf_sup": 0, "inuclear": 1},
        ),
    ]


# ---------------------------------------------------------------------------
# calculate_cryo_loads
# ---------------------------------------------------------------------------


def _reference_cryo_loads(
    i_tf_sup,
    i_pf_conductor,
    inuclear,
    tfcryoarea,
    coldmass,
    p_tf_nuclear_heat_mw,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
    eff_tf_cryo,
    temp_tf_cryo,
    p_cp_resistive,
    p_tf_leg_resistive,
    p_tf_joints_resistive,
    pnuc_cp_tf,
    temp_cp_coolant_inlet,
    qss,
    qac,
    qcl,
    qmisc,
):
    data = DataStructure()
    data.tfcoil.i_tf_sup = i_tf_sup
    data.pf_coil.i_pf_conductor = i_pf_conductor
    data.fwbs.inuclear = inuclear
    data.tfcoil.tfcryoarea = tfcryoarea
    data.structure.coldmass = coldmass
    data.fwbs.p_tf_nuclear_heat_mw = p_tf_nuclear_heat_mw
    data.pf_power.ensxpfm = ensxpfm
    data.times.t_plant_pulse_plasma_present = t_plant_pulse_plasma_present
    data.tfcoil.c_tf_turn = c_tf_turn
    data.tfcoil.n_tf_coils = n_tf_coils
    data.fwbs.qnuc = qnuc
    data.tfcoil.eff_tf_cryo = eff_tf_cryo
    data.tfcoil.temp_tf_cryo = temp_tf_cryo
    data.tfcoil.p_cp_resistive = p_cp_resistive
    data.tfcoil.p_tf_leg_resistive = p_tf_leg_resistive
    data.tfcoil.p_tf_joints_resistive = p_tf_joints_resistive
    data.fwbs.pnuc_cp_tf = pnuc_cp_tf
    data.tfcoil.temp_cp_coolant_inlet = temp_cp_coolant_inlet
    data.power.qss = qss
    data.power.qac = qac
    data.power.qcl = qcl
    data.power.qmisc = qmisc

    p = Power()
    p.data = data
    p.calculate_cryo_loads()

    return (
        data.heat_transport.helpow,
        data.heat_transport.p_cryo_plant_electric_mw,
        data.heat_transport.helpow_cryal,
        data.tfcoil.cryo_cool_req,
        data.power.qss,
        data.power.qac,
        data.power.qcl,
        data.power.qmisc,
        data.fwbs.qnuc,
    )


def _cryo_loads_samples():
    bounds = {
        "tfcryoarea": (0.0, 5000.0),
        "coldmass": (1.0e6, 6.0e7),
        "p_tf_nuclear_heat_mw": (0.0, 1.0),
        "ensxpfm": (1.0e3, 6.0e4),
        "t_plant_pulse_plasma_present": (300.0, 12000.0),
        "c_tf_turn": (1.0e3, 1.0e5),
        "n_tf_coils": (10.0, 24.0),
        "qnuc": (0.0, 20000.0),
        "eff_tf_cryo": (0.1, 0.3),
        "temp_tf_cryo": (4.0, 20.0),
        "p_cp_resistive": (0.0, 1.0e7),
        "p_tf_leg_resistive": (0.0, 1.0e7),
        "p_tf_joints_resistive": (0.0, 1.0e6),
        "pnuc_cp_tf": (0.0, 5.0),
        "temp_cp_coolant_inlet": (300.0, 350.0),
        "qss": (0.0, 30000.0),
        "qac": (0.0, 30000.0),
        "qcl": (0.0, 30000.0),
        "qmisc": (0.0, 30000.0),
    }
    samples = []
    for i_tf_sup, i_pf_conductor, inuclear in (
        (1, PFConductorModel.RESISTIVE, 1),
        (1, PFConductorModel.RESISTIVE, 0),
        (0, PFConductorModel.SUPERCONDUCTING, 1),
        (2, PFConductorModel.RESISTIVE, 1),
        (0, PFConductorModel.RESISTIVE, 1),
    ):
        fixed = {
            "i_tf_sup": i_tf_sup,
            "i_pf_conductor": int(i_pf_conductor),
            "inuclear": inuclear,
        }
        samples.extend(
            fuzz_samples(bounds, count=12, seed=70260818 + i_tf_sup, fixed=fixed)
        )
    return samples


class TestCryoLoads(Tier1Contract):
    audit_record = "models/power/thermal_cryo.md"
    reference = _reference_cryo_loads
    ported = calculate_cryo_loads
    static_argnames = ("i_tf_sup", "i_pf_conductor", "inuclear")
    samples = _cryo_loads_samples()


# ---------------------------------------------------------------------------
# DeltaEtaStep -- the `.power.delta_eta` self-loop, cut into a FixedPointFunction
# node, and ComponentThermalPowers's corresponding drop of that Output.
# ---------------------------------------------------------------------------

_DELTA_ETA_SWITCH_COMBOS = [
    pytest.param(
        PumpingPowerModelTypes.USER_INPUT,
        0,
        ElectricConversionModelTypes.CCFE_HCPB_VALUE,
        id="user_input-0-ccfe_hcpb",
    ),
    pytest.param(
        PumpingPowerModelTypes.FRACTION_OF_HEAT,
        1,
        ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR,
        id="fraction_of_heat-1-ccfe_hcpb_with_divertor",
    ),
    pytest.param(
        PumpingPowerModelTypes.MECHANICAL,
        2,
        ElectricConversionModelTypes.STEAM_RANKINE_CYCLE,
        id="mechanical-2-steam_rankine",
    ),
    pytest.param(
        PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP,
        0,
        ElectricConversionModelTypes.USER_INPUT,
        id="mechanical_with_pressure_drop-0-user_input",
    ),
]


@pytest.mark.parametrize(
    ("i_p_coolant_pumping", "i_blkt_dual_coolant", "i_thermal_electric_conversion"),
    _DELTA_ETA_SWITCH_COMBOS,
)
def test_delta_eta_step_to_graph_builds(
    i_p_coolant_pumping, i_blkt_dual_coolant, i_thermal_electric_conversion
):
    """`to_graph(DeltaEtaStep(...))` succeeds -- the actual point of this split.

    Before this split, `to_graph(ComponentThermalPowers(...))` raised `ValueError:
    reads ['.power.delta_eta', ...], which it also owns` for every configuration
    (`cottax` refuses to build a node that both reads and owns one `VarPath`).
    `DeltaEtaStep`'s built-in `FixedPointFunction` cut mints a `^cond.power.delta_eta`
    copy for the body to write and the real `.power.delta_eta` for the paired
    `FixedPoint` problem node to own, so neither piece reads and owns the same path.
    """
    node = DeltaEtaStep(
        i_p_coolant_pumping=PumpingPowerModelTypes(int(i_p_coolant_pumping)),
        i_blkt_dual_coolant=BlanketDualCoolantModel(int(i_blkt_dual_coolant)),
        i_thermal_electric_conversion=ElectricConversionModelTypes(
            int(i_thermal_electric_conversion)
        ),
    )
    graph = to_graph(node)
    names = {n.path_str() for n in graph.nodes}
    assert names == {"['DeltaEtaStep']", "^problem['DeltaEtaStep']"}


_SIX_SELF_LOOP_VARPATHS = (
    ".power.delta_eta",
    ".heat_transport.eta_turbine",
    ".heat_transport.etath_liq",
    ".heat_transport.temp_turbine_coolant_in",
    ".heat_transport.p_fw_div_heat_deposited_mw",
    ".primary_pumping.p_fw_blkt_coolant_pump_mw",
)


def _component_thermal_powers():
    """The node as `machine_from_indat` builds it -- three static switches, not five.

    `i_blanket_type` and `secondary_cycle_liq` left with the reads they fed: both
    reached only `calculate_plant_thermal_efficiency`/`_2`, whose results this node
    discards (`_audit/next_steps.md` §14.2).
    """
    return ComponentThermalPowers(
        i_p_coolant_pumping=PumpingPowerModelTypes.USER_INPUT,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_thermal_electric_conversion=ElectricConversionModelTypes.CCFE_HCPB_VALUE,
    )


def test_component_thermal_powers_neither_owns_nor_reads_five_of_the_six():
    """**This test asserted the opposite of its second half until this pass, and the
    change is the finding.**

    All six `VarPath`s below are owned by their own occupant family. This node used to
    *read* all six as well -- and `switch_kwarg_survey.md` §6 measured what that cost:
    the node recomputed five of them internally and **discarded** the results, so it
    declared itself a consumer of four fixed points it does not consume. Only
    `.primary_pumping.p_fw_blkt_coolant_pump_mw` is genuinely used (the pump totals
    sum it), and it is the one still read.
    """
    node = _component_thermal_powers()
    owned = {o.var.path_str() for o in node.outputs}
    read = {i.var.path_str() for i in node.inputs}
    for path in _SIX_SELF_LOOP_VARPATHS:
        assert path not in owned
    assert ".primary_pumping.p_fw_blkt_coolant_pump_mw" in read
    for path in _SIX_SELF_LOOP_VARPATHS:
        if path != ".primary_pumping.p_fw_blkt_coolant_pump_mw":
            assert path not in read
    # The two fields that fed only the discarded efficiency calls go with them.
    assert ".fwbs.temp_blkt_coolant_out" not in read
    assert ".fwbs.outlet_temp_liq" not in read


def test_component_thermal_powers_to_graph_builds_cleanly():
    """`to_graph(ComponentThermalPowers(...))` no longer raises at all.

    Before the six-way split it raised on six self-referencing fields; with all six
    owned elsewhere -- and five of them no longer even read -- it assembles as an
    ordinary single-node graph.
    """
    node = _component_thermal_powers()
    graph = to_graph(node)
    assert {n.path_str() for n in graph.nodes} == {"['ComponentThermalPowers']"}


def _delta_eta_step_kwargs(**overrides):
    kwargs = {
        "p_fw_coolant_pump_mw": 12.0,
        "p_blkt_coolant_pump_mw": 30.0,
        "p_fw_blkt_coolant_pump_mw": 45.0,
        "p_fw_nuclear_heat_total_mw": 80.0,
        "p_fw_rad_total_mw": 120.0,
        "p_blkt_nuclear_heat_total_mw": 600.0,
        "p_blkt_breeder_pump_mw": 3.0,
        "p_beam_orbit_loss_mw": 2.0,
        "p_fw_alpha_mw": 15.0,
        "p_beam_shine_through_mw": 1.0,
        "p_cp_shield_nuclear_heat_mw": 5.0,
        "p_shld_nuclear_heat_mw": 20.0,
        "p_shld_coolant_pump_mw": 8.0,
        "p_plasma_separatrix_mw": 120.0,
        "p_div_nuclear_heat_total_mw": 10.0,
        "p_div_rad_total_mw": 15.0,
        "p_div_coolant_pump_mw": 6.0,
        "i_shld_primary_heat": 1.0,
        "delta_eta": 0.05,
    }
    kwargs.update(overrides)
    return kwargs


def _component_thermal_powers_call_kwargs(step_kwargs):
    """The subset of `calculate_component_thermal_powers`'s ~30 parameters that
    `DeltaEtaStep.step`'s inputs correspond to, plus fixed/irrelevant values for the
    rest (the ones only `eta_turbine`/`etath_liq`/`temp_turbine_coolant_in` depend
    on) -- used only to cross-check `DeltaEtaStep` against the underlying pure
    function it shares helpers with, not as a harness sample.
    """
    return {
        "p_fw_coolant_pump_mw": step_kwargs["p_fw_coolant_pump_mw"],
        "p_blkt_coolant_pump_mw": step_kwargs["p_blkt_coolant_pump_mw"],
        "p_fw_blkt_coolant_pump_mw": step_kwargs["p_fw_blkt_coolant_pump_mw"],
        "eta_coolant_pump_electric": 0.85,
        "p_shld_coolant_pump_mw": step_kwargs["p_shld_coolant_pump_mw"],
        "p_div_coolant_pump_mw": step_kwargs["p_div_coolant_pump_mw"],
        "p_blkt_breeder_pump_mw": step_kwargs["p_blkt_breeder_pump_mw"],
        "p_hcd_electric_total_mw": 200.0,
        "p_hcd_injected_total_mw": 70.0,
        "p_blkt_nuclear_heat_total_mw": step_kwargs["p_blkt_nuclear_heat_total_mw"],
        "f_nuc_pow_bz_liq": 0.5,
        "p_fw_nuclear_heat_total_mw": step_kwargs["p_fw_nuclear_heat_total_mw"],
        "p_fw_rad_total_mw": step_kwargs["p_fw_rad_total_mw"],
        "p_beam_orbit_loss_mw": step_kwargs["p_beam_orbit_loss_mw"],
        "p_fw_alpha_mw": step_kwargs["p_fw_alpha_mw"],
        "p_beam_shine_through_mw": step_kwargs["p_beam_shine_through_mw"],
        "p_cp_shield_nuclear_heat_mw": step_kwargs["p_cp_shield_nuclear_heat_mw"],
        "p_shld_nuclear_heat_mw": step_kwargs["p_shld_nuclear_heat_mw"],
        "p_plasma_separatrix_mw": step_kwargs["p_plasma_separatrix_mw"],
        "p_div_nuclear_heat_total_mw": step_kwargs["p_div_nuclear_heat_total_mw"],
        "p_div_rad_total_mw": step_kwargs["p_div_rad_total_mw"],
        "p_fw_div_heat_deposited_mw": 500.0,
        "p_fw_hcd_nuclear_heat_mw": 1.0,
        "p_fw_hcd_rad_total_mw": 1.0,
        "i_shld_primary_heat": step_kwargs["i_shld_primary_heat"],
        "eta_turbine": 0.4,
        "etath_liq": 0.4,
        "delta_eta": step_kwargs["delta_eta"],
        "temp_blkt_coolant_out": 700.0,
        "outlet_temp_liq": 700.0,
        "temp_turbine_coolant_in": 600.0,
    }


@pytest.mark.parametrize(
    ("i_p_coolant_pumping", "i_blkt_dual_coolant", "i_thermal_electric_conversion"),
    _DELTA_ETA_SWITCH_COMBOS,
)
def test_delta_eta_step_matches_calculate_component_thermal_powers(
    i_p_coolant_pumping, i_blkt_dual_coolant, i_thermal_electric_conversion
):
    """`DeltaEtaStep.step` computes exactly the `delta_eta` element
    `calculate_component_thermal_powers` would, for the same inputs -- both call the
    same extracted helpers (`calculate_p_fw_blkt_coolant_pump_mw`/
    `calculate_p_fw_blkt_heat_deposited_mw`/`calculate_p_shld_heat_deposited_mw`/
    `calculate_p_div_heat_deposited_mw`/`calculate_delta_eta`), so this pins that the
    node-level split introduced no discrepancy versus the (unmodified, still
    separately tested) pure function.
    """
    step_kwargs = _delta_eta_step_kwargs()
    node = DeltaEtaStep(
        i_p_coolant_pumping=PumpingPowerModelTypes(int(i_p_coolant_pumping)),
        i_blkt_dual_coolant=BlanketDualCoolantModel(int(i_blkt_dual_coolant)),
        i_thermal_electric_conversion=ElectricConversionModelTypes(
            int(i_thermal_electric_conversion)
        ),
    )
    delta_eta_from_step = node.step(**step_kwargs)

    full_kwargs = _component_thermal_powers_call_kwargs(step_kwargs)
    full_result = calculate_component_thermal_powers(
        int(i_p_coolant_pumping),
        full_kwargs["p_fw_coolant_pump_mw"],
        full_kwargs["p_blkt_coolant_pump_mw"],
        full_kwargs["p_fw_blkt_coolant_pump_mw"],
        full_kwargs["eta_coolant_pump_electric"],
        full_kwargs["p_shld_coolant_pump_mw"],
        full_kwargs["p_div_coolant_pump_mw"],
        full_kwargs["p_blkt_breeder_pump_mw"],
        full_kwargs["p_hcd_electric_total_mw"],
        full_kwargs["p_hcd_injected_total_mw"],
        i_blkt_dual_coolant,
        full_kwargs["p_blkt_nuclear_heat_total_mw"],
        full_kwargs["f_nuc_pow_bz_liq"],
        full_kwargs["p_fw_nuclear_heat_total_mw"],
        full_kwargs["p_fw_rad_total_mw"],
        full_kwargs["p_beam_orbit_loss_mw"],
        full_kwargs["p_fw_alpha_mw"],
        full_kwargs["p_beam_shine_through_mw"],
        full_kwargs["p_cp_shield_nuclear_heat_mw"],
        full_kwargs["p_shld_nuclear_heat_mw"],
        full_kwargs["p_plasma_separatrix_mw"],
        full_kwargs["p_div_nuclear_heat_total_mw"],
        full_kwargs["p_div_rad_total_mw"],
        full_kwargs["p_fw_div_heat_deposited_mw"],
        full_kwargs["p_fw_hcd_nuclear_heat_mw"],
        full_kwargs["p_fw_hcd_rad_total_mw"],
        full_kwargs["i_shld_primary_heat"],
        int(i_thermal_electric_conversion),
        int(BlktModelTypes.CCFE_HCPB),
        full_kwargs["eta_turbine"],
        full_kwargs["etath_liq"],
        full_kwargs["delta_eta"],
        full_kwargs["temp_blkt_coolant_out"],
        full_kwargs["outlet_temp_liq"],
        full_kwargs["temp_turbine_coolant_in"],
        2,
    )
    delta_eta_from_full = full_result[23]

    assert delta_eta_from_step == pytest.approx(delta_eta_from_full, rel=1e-12)


@pytest.mark.parametrize(
    ("i_p_coolant_pumping", "i_blkt_dual_coolant", "i_thermal_electric_conversion"),
    _DELTA_ETA_SWITCH_COMBOS,
)
def test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta(
    i_p_coolant_pumping, i_blkt_dual_coolant, i_thermal_electric_conversion
):
    """The entering `.power.delta_eta` value has **zero** effect on the value
    `DeltaEtaStep.step` produces -- not merely "small," exactly zero, confirmed by
    `jax.grad`.

    This is the surprising finding this split turned up (see `calculate_delta_eta`'s
    and `DeltaEtaStep`'s docstrings): the two `calculate_plant_thermal_efficiency`
    branches that read `delta_eta` write `eta_turbine`, and nothing downstream of
    `eta_turbine` feeds back into computing the next `delta_eta` within this call. So
    although `.power.delta_eta` is a genuine structural self-reference (same `VarPath`
    read and owned -- `to_graph` rejects it as a plain node, see
    `test_delta_eta_step_to_graph_builds`), it is a numerically inert one: were a
    driver ever assigned to the resulting `FixedPoint` problem, it would converge in
    exactly one iteration from any starting point. Same lesson `_audit/next_steps.md`
    § 5's `Divertor` case taught this project -- verify a cycle is real, don't assume
    it from the shape alone.
    """
    node = DeltaEtaStep(
        i_p_coolant_pumping=PumpingPowerModelTypes(int(i_p_coolant_pumping)),
        i_blkt_dual_coolant=BlanketDualCoolantModel(int(i_blkt_dual_coolant)),
        i_thermal_electric_conversion=ElectricConversionModelTypes(
            int(i_thermal_electric_conversion)
        ),
    )
    base_kwargs = _delta_eta_step_kwargs()

    def delta_eta_next(delta_eta):
        out = node.step(**{**base_kwargs, "delta_eta": delta_eta})
        return out

    grad = jax.grad(delta_eta_next)(base_kwargs["delta_eta"])
    assert grad == 0.0


# ---------------------------------------------------------------------------
# EtaTurbineStep / EtathLiqStep / TempTurbineCoolantInStep /
# PFwDivHeatDepositedMwStep / PFwBlktCoolantPumpMwStep -- the five remaining
# ComponentThermalPowers self-loops, cut the same way DeltaEtaStep's was.
# ---------------------------------------------------------------------------

_CTP_FULL_KWARGS = {
    "p_fw_coolant_pump_mw": 12.0,
    "p_blkt_coolant_pump_mw": 30.0,
    "p_fw_blkt_coolant_pump_mw": 45.0,
    "eta_coolant_pump_electric": 0.85,
    "p_shld_coolant_pump_mw": 8.0,
    "p_div_coolant_pump_mw": 6.0,
    "p_blkt_breeder_pump_mw": 3.0,
    "p_hcd_electric_total_mw": 200.0,
    "p_hcd_injected_total_mw": 70.0,
    "p_blkt_nuclear_heat_total_mw": 600.0,
    "f_nuc_pow_bz_liq": 0.5,
    "p_fw_nuclear_heat_total_mw": 80.0,
    "p_fw_rad_total_mw": 120.0,
    "p_beam_orbit_loss_mw": 2.0,
    "p_fw_alpha_mw": 15.0,
    "p_beam_shine_through_mw": 1.0,
    "p_cp_shield_nuclear_heat_mw": 5.0,
    "p_shld_nuclear_heat_mw": 20.0,
    "p_plasma_separatrix_mw": 120.0,
    "p_div_nuclear_heat_total_mw": 10.0,
    "p_div_rad_total_mw": 15.0,
    "p_fw_div_heat_deposited_mw": 500.0,
    "p_fw_hcd_nuclear_heat_mw": 1.0,
    "p_fw_hcd_rad_total_mw": 1.0,
    "i_shld_primary_heat": 1.0,
    "eta_turbine": 0.4,
    "etath_liq": 0.4,
    "delta_eta": 0.05,
    "temp_blkt_coolant_out": 700.0,
    "outlet_temp_liq": 700.0,
    "temp_turbine_coolant_in": 600.0,
}


def _call_full_ctp(
    i_p_coolant_pumping,
    i_blkt_dual_coolant,
    i_thermal_electric_conversion,
    i_blanket_type,
    secondary_cycle_liq,
    **overrides,
):
    """`calculate_component_thermal_powers`, called with `_CTP_FULL_KWARGS` plus
    `overrides`, for cross-checking the five node-level splits below against the
    unmodified pure function -- same role `_component_thermal_powers_call_kwargs`
    plays for `DeltaEtaStep`'s own test above."""
    kw = {**_CTP_FULL_KWARGS, **overrides}
    return calculate_component_thermal_powers(
        int(i_p_coolant_pumping),
        kw["p_fw_coolant_pump_mw"],
        kw["p_blkt_coolant_pump_mw"],
        kw["p_fw_blkt_coolant_pump_mw"],
        kw["eta_coolant_pump_electric"],
        kw["p_shld_coolant_pump_mw"],
        kw["p_div_coolant_pump_mw"],
        kw["p_blkt_breeder_pump_mw"],
        kw["p_hcd_electric_total_mw"],
        kw["p_hcd_injected_total_mw"],
        i_blkt_dual_coolant,
        kw["p_blkt_nuclear_heat_total_mw"],
        kw["f_nuc_pow_bz_liq"],
        kw["p_fw_nuclear_heat_total_mw"],
        kw["p_fw_rad_total_mw"],
        kw["p_beam_orbit_loss_mw"],
        kw["p_fw_alpha_mw"],
        kw["p_beam_shine_through_mw"],
        kw["p_cp_shield_nuclear_heat_mw"],
        kw["p_shld_nuclear_heat_mw"],
        kw["p_plasma_separatrix_mw"],
        kw["p_div_nuclear_heat_total_mw"],
        kw["p_div_rad_total_mw"],
        kw["p_fw_div_heat_deposited_mw"],
        kw["p_fw_hcd_nuclear_heat_mw"],
        kw["p_fw_hcd_rad_total_mw"],
        kw["i_shld_primary_heat"],
        int(i_thermal_electric_conversion),
        int(i_blanket_type),
        kw["eta_turbine"],
        kw["etath_liq"],
        kw["delta_eta"],
        kw["temp_blkt_coolant_out"],
        kw["outlet_temp_liq"],
        kw["temp_turbine_coolant_in"],
        secondary_cycle_liq,
    )


# index of each field in calculate_component_thermal_powers's return tuple -- see
# that function's own docstring's Returns section.
_CTP_RETURN_INDEX = {
    "p_fw_blkt_coolant_pump_mw": 0,
    "p_fw_div_heat_deposited_mw": 15,
    "eta_turbine": 16,
    "etath_liq": 17,
    "temp_turbine_coolant_in": 18,
}


# ---------------------------------------------------------------------------
# The five occupant families that used to be `*Step` `FixedPointFunction`s.
#
# **Every one of these tests changed shape in the same way** (`_audit/next_steps.md`
# §14.2): what used to be "one node per switch combination, and here is what its
# self-gradient is" is now "one occupant per arm, and it does not read what it owns at
# all". The self-gradient parametrisations are gone because the self-read is gone; what
# replaces them is an assertion that the occupant `indat` selects for a given
# combination computes exactly the element `calculate_component_thermal_powers` would,
# and that the combinations where PROCESS's body is a pass-through select **no
# occupant** rather than an identity map.
# ---------------------------------------------------------------------------

_PASS_THROUGH_ETA_TURBINE = [
    (ElectricConversionModelTypes.USER_INPUT, BlktModelTypes.CCFE_HCPB),
    (ElectricConversionModelTypes.CCFE_HCPB_VALUE, BlktModelTypes.DCLL),
    (ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR, BlktModelTypes.DCLL),
    (ElectricConversionModelTypes.STEAM_RANKINE_CYCLE, BlktModelTypes.DCLL),
]
"""The four `(i_thermal_electric_conversion, i_blanket_type)` combinations whose
`calculate_plant_thermal_efficiency` arm is `return eta_turbine`.

These are exactly the combinations whose self-gradient this module used to pin at
**1.0** -- the identity map. An identity map is not a fixed point; it is the statement
that the field is an input, so each of them now selects `None`."""

_COMPUTING_ETA_TURBINE = [
    (ElectricConversionModelTypes.CCFE_HCPB_VALUE, BlktModelTypes.CCFE_HCPB),
    (
        ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR,
        BlktModelTypes.CCFE_HCPB,
    ),
    (ElectricConversionModelTypes.STEAM_RANKINE_CYCLE, BlktModelTypes.CCFE_HCPB),
    (
        ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE,
        BlktModelTypes.CCFE_HCPB,
    ),
    (ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE, BlktModelTypes.DCLL),
]
"""The combinations that compute `eta_turbine` -- the ones this module used to pin at
self-gradient **0.0**, i.e. the ones that never read the entering value anyway."""


def _occupant(registry, arm):
    """The occupant class a registry holds at one arm, or `None`."""
    return registry[arm]


@pytest.mark.parametrize(
    ("i_thermal_electric_conversion", "i_blanket_type"),
    _COMPUTING_ETA_TURBINE,
    ids=[f"{c.name}-{b.name}" for c, b in _COMPUTING_ETA_TURBINE],
)
def test_eta_turbine_occupant_matches_calculate_component_thermal_powers(
    i_thermal_electric_conversion, i_blanket_type
):
    """The occupant `_eta_turbine_arm` selects computes exactly the `eta_turbine`
    element `calculate_component_thermal_powers` would, for the same inputs -- and does
    so **without reading `.heat_transport.eta_turbine`**, which is what stopped it being
    a `FixedPointFunction`.
    """
    occupant = _occupant(
        ETA_TURBINE, _eta_turbine_arm(i_thermal_electric_conversion, i_blanket_type)
    )
    node = occupant()
    declared = inspect.signature(type(node).__call__).parameters
    got = node(**{
        k: _CTP_FULL_KWARGS[k]
        for k in ("delta_eta", "temp_blkt_coolant_out")
        if k in declared
    })
    full = _call_full_ctp(
        i_p_coolant_pumping=PumpingPowerModelTypes.USER_INPUT,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=i_thermal_electric_conversion,
        i_blanket_type=i_blanket_type,
        secondary_cycle_liq=2,
    )
    assert "eta_turbine" not in declared
    assert got == pytest.approx(full[_CTP_RETURN_INDEX["eta_turbine"]], rel=1e-12)


@pytest.mark.parametrize(
    ("i_thermal_electric_conversion", "i_blanket_type"),
    _PASS_THROUGH_ETA_TURBINE,
    ids=[f"{c.name}-{b.name}" for c, b in _PASS_THROUGH_ETA_TURBINE],
)
def test_eta_turbine_pass_through_arms_have_no_occupant(
    i_thermal_electric_conversion, i_blanket_type
):
    """`.heat_transport.eta_turbine` is an **input** wherever PROCESS's body is
    `return eta_turbine`, and the tree says so with an empty slot.

    This is the reference run's own case (`USER_INPUT`), and it is what
    `switch_kwarg_survey.md` §4.7 asked to be made visible: the `FixedPoint` that was
    driven here determined nothing, on a variable `.costs.coe` depends on.
    """
    arm = _eta_turbine_arm(i_thermal_electric_conversion, i_blanket_type)
    assert ETA_TURBINE[arm] is None


def test_etath_liq_occupant_matches_calculate_component_thermal_powers():
    """`secondary_cycle_liq == 4` computes `etath_liq` from `.fwbs.outlet_temp_liq`
    alone; `== 2` is a pass-through and has no occupant."""
    node = ETATH_LIQ[ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE]()
    got = node(outlet_temp_liq=_CTP_FULL_KWARGS["outlet_temp_liq"])
    full = _call_full_ctp(
        i_p_coolant_pumping=PumpingPowerModelTypes.USER_INPUT,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=4,
    )
    assert got == pytest.approx(full[_CTP_RETURN_INDEX["etath_liq"]], rel=1e-12)
    assert ETATH_LIQ[ElectricConversionModelTypes.USER_INPUT] is None


@pytest.mark.parametrize(
    ("i_thermal_electric_conversion", "i_blanket_type", "secondary_cycle_liq"),
    [
        (
            ElectricConversionModelTypes.USER_INPUT,
            BlktModelTypes.CCFE_HCPB,
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE,
        ),
        (
            ElectricConversionModelTypes.STEAM_RANKINE_CYCLE,
            BlktModelTypes.CCFE_HCPB,
            ElectricConversionModelTypes.USER_INPUT,
        ),
        (
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE,
            BlktModelTypes.DCLL,
            ElectricConversionModelTypes.USER_INPUT,
        ),
    ],
    ids=["stage-two-writes", "stage-one-rankine", "stage-one-co2"],
)
def test_temp_turbine_coolant_in_occupant_matches_calculate_component_thermal_powers(
    i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq
):
    """Whichever stage writes `temp_turbine_coolant_in`, the selected occupant computes
    the same value the whole composite does -- reading only that stage's own source
    field."""
    arm = _temp_turbine_coolant_in_arm(
        i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq
    )
    node = TEMP_TURBINE_COOLANT_IN[arm]()
    declared = inspect.signature(type(node).__call__).parameters
    got = node(**{
        k: _CTP_FULL_KWARGS[k]
        for k in ("outlet_temp_liq", "temp_blkt_coolant_out")
        if k in declared
    })
    full = _call_full_ctp(
        i_p_coolant_pumping=PumpingPowerModelTypes.USER_INPUT,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=i_thermal_electric_conversion,
        i_blanket_type=i_blanket_type,
        secondary_cycle_liq=int(secondary_cycle_liq),
    )
    assert "temp_turbine_coolant_in" not in declared
    assert got == pytest.approx(
        full[_CTP_RETURN_INDEX["temp_turbine_coolant_in"]], rel=1e-12
    )


def test_temp_turbine_coolant_in_pass_through_arm_has_no_occupant():
    """Both stages passing the entering value through is the "it is an input" case."""
    arm = _temp_turbine_coolant_in_arm(
        ElectricConversionModelTypes.USER_INPUT,
        BlktModelTypes.CCFE_HCPB,
        ElectricConversionModelTypes.USER_INPUT,
    )
    assert TEMP_TURBINE_COOLANT_IN[arm] is None


@pytest.mark.parametrize(
    "i_p_coolant_pumping",
    [
        PumpingPowerModelTypes.USER_INPUT,
        PumpingPowerModelTypes.FRACTION_OF_HEAT,
        PumpingPowerModelTypes.MECHANICAL,
    ],
    ids=lambda v: v.name,
)
def test_p_fw_div_heat_deposited_occupant_matches_calculate_component_thermal_powers(
    i_p_coolant_pumping,
):
    """Every `i_p_coolant_pumping` value except `MECHANICAL_WITH_PRESSURE_DROP`
    recomputes the field, and the occupant reproduces the composite's element."""
    node = P_FW_DIV_HEAT_DEPOSITED[_p_fw_div_heat_deposited_arm(i_p_coolant_pumping)]()
    declared = inspect.signature(type(node).__call__).parameters
    got = node(**{k: v for k, v in _CTP_FULL_KWARGS.items() if k in declared})
    full = _call_full_ctp(
        i_p_coolant_pumping=i_p_coolant_pumping,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=2,
    )
    assert "p_fw_div_heat_deposited_mw" not in declared
    assert got == pytest.approx(
        full[_CTP_RETURN_INDEX["p_fw_div_heat_deposited_mw"]], rel=1e-12
    )


def test_p_fw_div_heat_deposited_pass_through_arm_has_no_occupant():
    """`MECHANICAL_WITH_PRESSURE_DROP` passes the entering value through; the only
    other producer in `process/` is `models/ife.py`, out of scope, so the field is a
    boundary input there."""
    arm = _p_fw_div_heat_deposited_arm(
        PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP
    )
    assert P_FW_DIV_HEAT_DEPOSITED[arm] is None


@pytest.mark.parametrize(
    "i_p_coolant_pumping",
    [PumpingPowerModelTypes.USER_INPUT, PumpingPowerModelTypes.FRACTION_OF_HEAT],
    ids=lambda v: v.name,
)
def test_p_fw_blkt_coolant_pump_occupant_matches_calculate_component_thermal_powers(
    i_p_coolant_pumping,
):
    """On the two values where `power` owns
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`, the occupant is the plain sum the
    composite computes -- and reads neither the field it owns nor the switch."""
    node = P_FW_BLKT_COOLANT_PUMP[_p_fw_blkt_coolant_pump_arm(i_p_coolant_pumping)]()
    declared = inspect.signature(type(node).__call__).parameters
    got = node(**{k: v for k, v in _CTP_FULL_KWARGS.items() if k in declared})
    full = _call_full_ctp(
        i_p_coolant_pumping=i_p_coolant_pumping,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=2,
    )
    assert "p_fw_blkt_coolant_pump_mw" not in declared
    assert got == pytest.approx(
        full[_CTP_RETURN_INDEX["p_fw_blkt_coolant_pump_mw"]], rel=1e-12
    )


# ---------------------------------------------------------------------------
# CryoQNucStep / CryoQLoadsStep / CryoLoads -- `Power.calculate_cryo_loads`'s five
# conditionally-owned `q*` fields cut into two `FixedPointFunction`s, and the four
# unconditionally-owned outputs left as an ordinary `ExplicitFunction`.
# ---------------------------------------------------------------------------

_CRYO_KWARGS = {
    "tfcryoarea": 1200.0,
    "coldmass": 4.7e7,
    "p_tf_nuclear_heat_mw": 0.045,
    "ensxpfm": 37429.5,
    "t_plant_pulse_plasma_present": 10364.4,
    "c_tf_turn": 74026.75,
    "n_tf_coils": 16.0,
    "qnuc": 12920.0,
    "eff_tf_cryo": 0.13,
    "temp_tf_cryo": 4.5,
    "p_cp_resistive": 1.0e6,
    "p_tf_leg_resistive": 2.0e6,
    "p_tf_joints_resistive": 1.0e5,
    "pnuc_cp_tf": 1.5,
    "temp_cp_coolant_inlet": 313.15,
    "qss": 21000.0,
    "qac": 3600.0,
    "qcl": 16100.0,
    "qmisc": 18700.0,
}

# (i_tf_sup, i_pf_conductor, inuclear) -- the same five switch triples
# `_cryo_loads_samples` fuzzes, so the node split is exercised on every arm the
# Tier-1 contract covers, not only the reference configuration's.
_CRYO_SWITCH_COMBOS = [
    pytest.param(1, PFConductorModel.RESISTIVE, 1, id="sc_tf-resistive_pf-inuclear1"),
    pytest.param(1, PFConductorModel.RESISTIVE, 0, id="sc_tf-resistive_pf-inuclear0"),
    pytest.param(0, PFConductorModel.SUPERCONDUCTING, 1, id="res_tf-sc_pf-inuclear1"),
    pytest.param(2, PFConductorModel.RESISTIVE, 1, id="al_tf-resistive_pf-inuclear1"),
    pytest.param(0, PFConductorModel.RESISTIVE, 1, id="res_tf-resistive_pf-inuclear1"),
]


def test_cryo_cannot_be_a_plain_node():
    """`Cryo` stays unregistered because `cottax` will not build it -- the same
    position `PlantThermalEfficiency` is in, and the reason the split below exists.

    `Power.cryo` reads `.fwbs.qnuc` (the incumbent, kept when `inuclear == 1`) and
    writes it (when `inuclear == 0 and i_tf_sup == 1`), which is `_audit/
    next_steps.md` §5's Shape B: *"a node may not read what it owns"*, a hard
    construction error rather than a style preference. Asserted by construction here,
    not merely stated in a comment.
    """
    with pytest.raises(ValueError, match="which it also owns"):
        to_graph(
            Cryo(
                i_tf_sup=TFConductorModel.SUPERCONDUCTING,
                inuclear=CoilNuclearHeatingModel.FRANCES_FOX,
            )
        )


@pytest.mark.parametrize(("i_tf_sup", "i_pf_conductor", "inuclear"), _CRYO_SWITCH_COMBOS)
def test_cryo_split_nodes_all_assemble(i_tf_sup, i_pf_conductor, inuclear):
    """Each of the three replacement nodes builds a graph on every switch arm.

    `CryoQNucStep`/`CryoQLoadsStep` mint their own `^cond` copies (so the body writes
    the copy and the paired `FixedPoint` owns the real `VarPath`); `CryoLoads` reads
    all five `q*` as plain `FromExactly`s and owns none of them, so it is an ordinary
    single-node graph.
    """
    qnuc_node = CryoQNucStep(
        i_tf_sup=TFConductorModel(int(i_tf_sup)),
        inuclear=CoilNuclearHeatingModel(int(inuclear)),
    )
    assert {n.path_str() for n in to_graph(qnuc_node).nodes} == {
        "['CryoQNucStep']",
        "^problem['CryoQNucStep']",
    }

    # `CryoQLoads` and `CryoLoads` are families now, not `FixedPointFunction`s: the
    # switches select an occupant (or, for the `q*` fields outside PROCESS's guard,
    # *no* occupant) instead of being carried into a body that reads what it owns.
    q_occupant = CRYO_Q_LOADS[_cryo_q_loads_arm(i_tf_sup, int(i_pf_conductor))]
    if q_occupant is not None:
        q_node = q_occupant()
        assert {n.path_str() for n in to_graph(q_node).nodes} == {
            f"['{type(q_node).__name__}']"
        }

    loads = CRYO_LOADS[_cryo_loads_arm(i_tf_sup, int(i_pf_conductor))]()
    assert {n.path_str() for n in to_graph(loads).nodes} == {
        f"['{type(loads).__name__}']"
    }


def test_cryo_split_ownership_is_a_partition():
    """The three nodes own exactly the nine `VarPath`s `Power.calculate_cryo_loads`
    writes, with no overlap.

    Overlap would be an ownership collision `Graph` rejects; a gap would leave a field
    PROCESS computes on this run's path as a boundary input, which is the defect
    `_audit/boundary_inputs_audit.md` §4c (b9)/(b10) records.
    """
    qnuc_node = CryoQNucStep(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        inuclear=CoilNuclearHeatingModel.FRANCES_FOX,
    )
    q_node = CRYO_Q_LOADS[
        _cryo_q_loads_arm(
            TFConductorModel.SUPERCONDUCTING, PFConductorModel.SUPERCONDUCTING
        )
    ]()
    loads = CRYO_LOADS[
        _cryo_loads_arm(
            TFConductorModel.SUPERCONDUCTING, PFConductorModel.SUPERCONDUCTING
        )
    ]()

    owned = [{o.var.path_str() for o in n.outputs} for n in (qnuc_node, q_node, loads)]
    assert owned[0] == {".fwbs.qnuc"}
    assert owned[1] == {".power.qss", ".power.qac", ".power.qcl", ".power.qmisc"}
    assert owned[2] == {
        ".heat_transport.helpow",
        ".heat_transport.p_cryo_plant_electric_mw",
        ".heat_transport.helpow_cryal",
        ".tfcoil.cryo_cool_req",
    }
    assert owned[0] & owned[1] == set()
    assert (owned[0] | owned[1]) & owned[2] == set()

    # `CryoLoads` must still *read* every `q*` -- it builds `helpow` from them.
    read = {i.var.path_str() for i in loads.inputs}
    assert owned[0] | owned[1] <= read


@pytest.mark.parametrize(("i_tf_sup", "i_pf_conductor", "inuclear"), _CRYO_SWITCH_COMBOS)
def test_cryo_split_reproduces_calculate_cryo_loads(i_tf_sup, i_pf_conductor, inuclear):
    """Running the three nodes in schedule order reproduces `calculate_cryo_loads`
    exactly, on every switch arm.

    This is what makes the split safe to register: `calculate_cryo_loads` is the
    function the Tier-1 contract validates against real PROCESS
    (`TestCryoLoads`), so pinning the node-level composition to it transfers that
    validation to the nodes without a second reference run.
    """
    qnuc = CryoQNucStep(
        i_tf_sup=TFConductorModel(int(i_tf_sup)),
        inuclear=CoilNuclearHeatingModel(int(inuclear)),
    ).step(
        qnuc=_CRYO_KWARGS["qnuc"],
        p_tf_nuclear_heat_mw=_CRYO_KWARGS["p_tf_nuclear_heat_mw"],
    )
    if i_tf_sup == 2:
        pytest.skip(
            "aluminium TF has no occupant -- `('i_tf_sup', 2)` is UNPORTED at the "
            "`power.tf_power` slot, so no machine reaches these nodes on that value. "
            "`calculate_cryo_loads` still covers it and `TestCryoLoads` still diffs "
            "that against PROCESS."
        )
    q_occupant = CRYO_Q_LOADS[_cryo_q_loads_arm(i_tf_sup, int(i_pf_conductor))]
    if q_occupant is None:
        qss, qac, qcl, qmisc = (
            _CRYO_KWARGS["qss"],
            _CRYO_KWARGS["qac"],
            _CRYO_KWARGS["qcl"],
            _CRYO_KWARGS["qmisc"],
        )
    else:
        q_node = q_occupant()
        declared = inspect.signature(type(q_node).__call__).parameters
        qss, qac, qcl, qmisc = q_node(**{
            k: (qnuc if k == "qnuc" else _CRYO_KWARGS[k])
            for k in declared
            if k != "self"
        })
    loads = CRYO_LOADS[_cryo_loads_arm(i_tf_sup, int(i_pf_conductor))]()
    declared = inspect.signature(type(loads).__call__).parameters
    supplied = {
        **_CRYO_KWARGS,
        "qss": qss,
        "qac": qac,
        "qcl": qcl,
        "qmisc": qmisc,
        "qnuc": qnuc,
    }
    helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req = loads(**{
        k: supplied[k] for k in declared if k != "self"
    })

    expected = calculate_cryo_loads(
        i_tf_sup,
        int(i_pf_conductor),
        inuclear,
        _CRYO_KWARGS["tfcryoarea"],
        _CRYO_KWARGS["coldmass"],
        _CRYO_KWARGS["p_tf_nuclear_heat_mw"],
        _CRYO_KWARGS["ensxpfm"],
        _CRYO_KWARGS["t_plant_pulse_plasma_present"],
        _CRYO_KWARGS["c_tf_turn"],
        _CRYO_KWARGS["n_tf_coils"],
        _CRYO_KWARGS["qnuc"],
        _CRYO_KWARGS["eff_tf_cryo"],
        _CRYO_KWARGS["temp_tf_cryo"],
        _CRYO_KWARGS["p_cp_resistive"],
        _CRYO_KWARGS["p_tf_leg_resistive"],
        _CRYO_KWARGS["p_tf_joints_resistive"],
        _CRYO_KWARGS["pnuc_cp_tf"],
        _CRYO_KWARGS["temp_cp_coolant_inlet"],
        _CRYO_KWARGS["qss"],
        _CRYO_KWARGS["qac"],
        _CRYO_KWARGS["qcl"],
        _CRYO_KWARGS["qmisc"],
    )
    got = (
        helpow,
        p_cryo_plant_electric_mw,
        helpow_cryal,
        cryo_cool_req,
        qss,
        qac,
        qcl,
        qmisc,
        qnuc,
    )
    for g, e in zip(got, expected, strict=True):
        assert g == pytest.approx(e, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ("i_tf_sup", "inuclear", "expected_grad"),
    [
        pytest.param(1, 0, 0.0, id="owned-recomputed"),
        pytest.param(1, 1, 1.0, id="inuclear1-identity"),
        pytest.param(0, 0, 1.0, id="resistive_tf-identity"),
        pytest.param(2, 0, 1.0, id="aluminium_tf-identity"),
    ],
)
def test_cryo_q_nuc_step_gradient(i_tf_sup, inuclear, expected_grad):
    """`d(qnuc_next)/d(qnuc)` is exactly `0` where PROCESS recomputes `.fwbs.qnuc`
    and exactly `1` everywhere else -- the two arms of the fixed point, confirmed by
    `jax.grad` rather than asserted from the source's shape.

    `1` is the degenerate case: the residual `g(u) - u` is then structurally zero and
    `functional_process.sand.degenerate_fixed_points` drops the problem, reverting
    `.fwbs.qnuc` to a boundary input -- which is exactly PROCESS's *"if inuclear = 1:
    qnuc is input"* (`process/models/power.py:1825`), recovered from structure.
    """
    node = CryoQNucStep(
        i_tf_sup=TFConductorModel(int(i_tf_sup)),
        inuclear=CoilNuclearHeatingModel(int(inuclear)),
    )

    def qnuc_next(qnuc):
        return node.step(
            qnuc=qnuc, p_tf_nuclear_heat_mw=_CRYO_KWARGS["p_tf_nuclear_heat_mw"]
        )

    assert jax.grad(qnuc_next)(_CRYO_KWARGS["qnuc"]) == expected_grad


def test_cryo_q_loads_has_no_self_read_on_either_computing_arm():
    """**The replacement for `test_cryo_q_loads_step_gradient`.**

    That test pinned `d(qss_next)/d(qss)` -- and every other diagonal entry -- at
    exactly `0` where `Power.cryo` runs and exactly `1` where it does not: per switch
    combination, the node either ignored the four fields it owned or was the identity in
    all four. Both halves are structural facts about the switch, not numerical ones, and
    the split states them directly (`_audit/next_steps.md` §14.2): the two computing
    occupants declare none of the four as reads, and the non-computing arm has no
    occupant at all, so `.power.qss`/`qac`/`qcl`/`qmisc` are boundary inputs there --
    which is what "the identity map" meant.
    """
    owned = {".power.qss", ".power.qac", ".power.qcl", ".power.qmisc"}
    for arm in (0, 1):
        node = CRYO_Q_LOADS[arm]()
        assert {o.var.path_str() for o in node.outputs} == owned
        assert owned & {i.var.path_str() for i in node.inputs} == set()
    assert (
        CRYO_Q_LOADS[
            _cryo_q_loads_arm(
                TFConductorModel.WATER_COOLED_COPPER, PFConductorModel.RESISTIVE
            )
        ]
        is None
    )
