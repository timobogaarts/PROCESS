"""Harness cases for `functional_process/models/power_B_thermal_cryo.py`.

Audit record: `functional_process/models/power_B_thermal_cryo.md`. No legacy points
exist in `tests/unit/models/test_power.py` for
`component_thermal_powers`/`plant_thermal_efficiency`/`plant_thermal_efficiency_2`
(fuzz-only, same situation as chunk A); `cryo` does have legacy points there, reused
below.
"""

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.models.power_B_thermal_cryo import (
    calculate_component_thermal_powers,
    calculate_cryo,
    calculate_cryo_loads,
    calculate_plant_thermal_efficiency,
    calculate_plant_thermal_efficiency_2,
)
from process.core.model import DataStructure
from process.data_structure.blanket_variables import BlktModelTypes
from process.data_structure.pfcoil_variables import PFConductorModel
from process.models.power import ElectricConversionModelTypes, Power, PumpingPowerModelTypes

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
    before computing anything there, see `power_B_thermal_cryo.md`'s real finding
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
    # during this port's construction (see `power_B_thermal_cryo.md`'s real
    # findings). The port's own "else: pass eta_turbine through unchanged" behaviour
    # is unambiguous from the branch's structure and cannot be diffed against a
    # PROCESS reference that never returns.
    return samples


class TestPlantThermalEfficiency(Tier1Contract):
    audit_record = "models/power_B_thermal_cryo.md"
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
    audit_record = "models/power_B_thermal_cryo.md"
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
    data.primary_pumping.p_fw_blkt_coolant_pump_mw = kwargs[
        "p_fw_blkt_coolant_pump_mw"
    ]
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
    data.heat_transport.p_fw_div_heat_deposited_mw = kwargs[
        "p_fw_div_heat_deposited_mw"
    ]
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
    audit_record = "models/power_B_thermal_cryo.md"
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
    return helpow, data.power.qss, data.power.qac, data.power.qcl, data.power.qmisc, data.fwbs.qnuc


class TestCryo(Tier1Contract):
    audit_record = "models/power_B_thermal_cryo.md"
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
    audit_record = "models/power_B_thermal_cryo.md"
    reference = _reference_cryo_loads
    ported = calculate_cryo_loads
    static_argnames = ("i_tf_sup", "i_pf_conductor", "inuclear")
    samples = _cryo_loads_samples()
