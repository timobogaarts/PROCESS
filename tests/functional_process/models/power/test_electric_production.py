"""Harness cases for `functional_process/cottax/power/electric_production.py`.

Audit record: `functional_process/_audit/units/models/power/electric_production.md`.
Legacy points exist in `tests/unit/models/test_power.py` for `acpow`
(`test_acpow`, 2 points) and `plant_electric_production`
(`test_plant_electric_production`, 2 points) -- reused below.
`power_profiles_over_time` has no dedicated PROCESS unit test (it is exercised
indirectly through `test_plant_electric_production`, whose legacy points also
exercise this port's own composed `calculate_plant_electric_production`) --
fuzz-only, verified by hand against `Power.power_profiles_over_time` while building
this port (see `electric_production.md`).
"""

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.cottax.power.electric_production import (
    calculate_acpow,
    calculate_plant_electric_production,
    power_profiles_over_time,
)
from process.core.model import DataStructure
from process.models.power import Power
from process.models.pulse import PulseTimings

# ---------------------------------------------------------------------------
# acpow
# ---------------------------------------------------------------------------


def _reference_acpow(
    p_tf_electric_supplies_mw,
    srcktpm,
    peakmva,
    i_pf_energy_storage_source,
    p_hcd_electric_total_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_coolant_pump_elec_total_mw,
    p_tritium_plant_electric_mw,
    p_plant_electric_base_total_mw,
    fmgdmw,
):
    data = DataStructure()
    data.heat_transport.p_tf_electric_supplies_mw = p_tf_electric_supplies_mw
    data.pf_power.srcktpm = srcktpm
    data.heat_transport.peakmva = peakmva
    data.pf_power.i_pf_energy_storage_source = i_pf_energy_storage_source
    data.heat_transport.p_hcd_electric_total_mw = p_hcd_electric_total_mw
    data.heat_transport.p_cryo_plant_electric_mw = p_cryo_plant_electric_mw
    data.heat_transport.vachtmw = vachtmw
    data.heat_transport.p_coolant_pump_elec_total_mw = p_coolant_pump_elec_total_mw
    data.heat_transport.p_tritium_plant_electric_mw = p_tritium_plant_electric_mw
    data.heat_transport.p_plant_electric_base_total_mw = p_plant_electric_base_total_mw
    data.heat_transport.fmgdmw = fmgdmw

    p = Power()
    p.data = data
    p.acpow(output=False)
    return data.heat_transport.pacpmw, data.heat_transport.tlvpmw


def _acpow_samples():
    bounds = {
        "p_tf_electric_supplies_mw": (1.0, 20.0),
        "srcktpm": (100.0, 2000.0),
        "peakmva": (10.0, 800.0),
        "p_hcd_electric_total_mw": (50.0, 300.0),
        "p_cryo_plant_electric_mw": (10.0, 200.0),
        "vachtmw": (0.1, 5.0),
        "p_coolant_pump_elec_total_mw": (10.0, 400.0),
        "p_tritium_plant_electric_mw": (1.0, 30.0),
        "p_plant_electric_base_total_mw": (0.0, 100.0),
        "fmgdmw": (0.0, 20.0),
    }
    samples = [
        # tests/unit/models/test_power.py::test_acpow, both parametrised points.
        legacy_sample(
            "baseline-2018-point-1",
            p_tf_electric_supplies_mw=9.1507079104675704,
            srcktpm=1071.1112934857531,
            peakmva=736.39062584245937,
            i_pf_energy_storage_source=2,
            p_hcd_electric_total_mw=129.94611930107126,
            p_cryo_plant_electric_mw=37.900388528497025,
            vachtmw=0.5,
            p_coolant_pump_elec_total_mw=234.28554165620102,
            p_tritium_plant_electric_mw=15,
            p_plant_electric_base_total_mw=0,
            fmgdmw=0,
        ),
        legacy_sample(
            "baseline-2018-point-2",
            p_tf_electric_supplies_mw=9.1507079104675704,
            srcktpm=1069.8879533693198,
            peakmva=90.673341440806112,
            i_pf_energy_storage_source=2,
            p_hcd_electric_total_mw=129.94611930107126,
            p_cryo_plant_electric_mw=108.74512702403499,
            vachtmw=0.5,
            p_coolant_pump_elec_total_mw=234.2162627659944,
            p_tritium_plant_electric_mw=15,
            p_plant_electric_base_total_mw=62.23714391536082,
            fmgdmw=0,
        ),
    ]
    for value in (0, 1, 2):
        samples.extend(
            fuzz_samples(
                bounds,
                count=15,
                seed=80260818 + value,
                fixed={"i_pf_energy_storage_source": value},
            )
        )
    return samples


class TestAcpow(Tier1Contract):
    audit_record = "models/power/electric_production.md"
    reference = _reference_acpow
    ported = calculate_acpow
    static_argnames = ("i_pf_energy_storage_source",)
    samples = _acpow_samples()


# ---------------------------------------------------------------------------
# power_profiles_over_time
# ---------------------------------------------------------------------------


def _reference_power_profiles_over_time(
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
):
    pulse_timings = PulseTimings(
        t_plant_pulse_coil_precharge=t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up=t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp=t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn=t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down=t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell=t_plant_pulse_dwell,
    )
    return Power.power_profiles_over_time(
        p_plant_electric_base_total_mw=p_plant_electric_base_total_mw,
        p_cryo_plant_electric_mw=p_cryo_plant_electric_mw,
        p_tritium_plant_electric_mw=p_tritium_plant_electric_mw,
        vachtmw=vachtmw,
        p_tf_electric_supplies_mw=p_tf_electric_supplies_mw,
        p_pf_electric_supplies_mw=p_pf_electric_supplies_mw,
        p_coolant_pump_elec_total_mw=p_coolant_pump_elec_total_mw,
        p_hcd_electric_total_mw=p_hcd_electric_total_mw,
        p_fusion_total_mw=p_fusion_total_mw,
        p_plant_electric_gross_mw=p_plant_electric_gross_mw,
        p_plant_electric_net_mw=p_plant_electric_net_mw,
        pulse_timings=pulse_timings,
    )


class TestPowerProfilesOverTime(Tier1Contract):
    audit_record = "models/power/electric_production.md"
    reference = _reference_power_profiles_over_time
    ported = power_profiles_over_time
    samples = fuzz_samples(
        {
            "p_plant_electric_base_total_mw": (0.0, 100.0),
            "p_cryo_plant_electric_mw": (10.0, 200.0),
            "p_tritium_plant_electric_mw": (1.0, 30.0),
            "vachtmw": (0.1, 5.0),
            "p_tf_electric_supplies_mw": (1.0, 20.0),
            "p_pf_electric_supplies_mw": (0.01, 5.0),
            "p_coolant_pump_elec_total_mw": (10.0, 400.0),
            "p_hcd_electric_total_mw": (50.0, 300.0),
            "p_fusion_total_mw": (500.0, 3000.0),
            "p_plant_electric_gross_mw": (0.0, 1500.0),
            "p_plant_electric_net_mw": (0.0, 1000.0),
            "t_plant_pulse_coil_precharge": (1.0, 50.0),
            "t_plant_pulse_plasma_current_ramp_up": (10.0, 500.0),
            "t_plant_pulse_fusion_ramp": (1.0, 50.0),
            "t_plant_pulse_burn": (100.0, 10000.0),
            "t_plant_pulse_plasma_current_ramp_down": (10.0, 100.0),
            "t_plant_pulse_dwell": (10.0, 2000.0),
        },
        count=40,
        seed=90260818,
    )


# ---------------------------------------------------------------------------
# plant_electric_production
# ---------------------------------------------------------------------------

_PEP_STATIC_ARGNAMES = (
    "itart",
    "i_tf_sup",
    "ireactor",
    "i_blkt_dual_coolant",
    "i_p_coolant_pumping",
)


def _reference_plant_electric_production(**kwargs):
    data = DataStructure()
    data.physics.itart = kwargs["itart"]
    data.tfcoil.i_tf_sup = kwargs["i_tf_sup"]
    data.tfcoil.p_cp_coolant_pump_elec = kwargs["p_cp_coolant_pump_elec"]
    data.heat_transport.p_plant_electric_base = kwargs["p_plant_electric_base"]
    data.buildings.a_plant_floor_effective = kwargs["a_plant_floor_effective"]
    data.heat_transport.pflux_plant_floor_electric = kwargs["pflux_plant_floor_electric"]
    data.heat_transport.p_cryo_plant_electric_mw = kwargs["p_cryo_plant_electric_mw"]
    data.heat_transport.p_tf_electric_supplies_mw = kwargs["p_tf_electric_supplies_mw"]
    data.heat_transport.p_tritium_plant_electric_mw = kwargs[
        "p_tritium_plant_electric_mw"
    ]
    data.heat_transport.vachtmw = kwargs["vachtmw"]
    data.pf_coil.p_pf_electric_supplies_mw = kwargs["p_pf_electric_supplies_mw"]
    data.heat_transport.p_hcd_electric_loss_mw = kwargs["p_hcd_electric_loss_mw"]
    data.heat_transport.p_coolant_pump_loss_total_mw = kwargs[
        "p_coolant_pump_loss_total_mw"
    ]
    data.heat_transport.p_div_secondary_heat_mw = kwargs["p_div_secondary_heat_mw"]
    data.heat_transport.p_shld_secondary_heat_mw = kwargs["p_shld_secondary_heat_mw"]
    data.heat_transport.p_hcd_secondary_heat_mw = kwargs["p_hcd_secondary_heat_mw"]
    data.fwbs.p_tf_nuclear_heat_mw = kwargs["p_tf_nuclear_heat_mw"]
    data.costs.ireactor = kwargs["ireactor"]
    data.fwbs.i_blkt_dual_coolant = kwargs["i_blkt_dual_coolant"]
    data.fwbs.i_p_coolant_pumping = kwargs["i_p_coolant_pumping"]
    data.heat_transport.p_plant_primary_heat_mw = kwargs["p_plant_primary_heat_mw"]
    data.power.p_blkt_liquid_breeder_heat_deposited_mw = kwargs[
        "p_blkt_liquid_breeder_heat_deposited_mw"
    ]
    data.heat_transport.eta_turbine = kwargs["eta_turbine"]
    data.heat_transport.etath_liq = kwargs["etath_liq"]
    data.heat_transport.p_hcd_electric_total_mw = kwargs["p_hcd_electric_total_mw"]
    data.heat_transport.p_coolant_pump_elec_total_mw = kwargs[
        "p_coolant_pump_elec_total_mw"
    ]
    data.heat_transport.p_plant_electric_gross_mw = kwargs["p_plant_electric_gross_mw"]
    data.power.p_turbine_loss_mw = kwargs["p_turbine_loss_mw"]
    data.heat_transport.p_plant_electric_recirc_mw = kwargs["p_plant_electric_recirc_mw"]
    data.heat_transport.p_plant_electric_net_mw = kwargs["p_plant_electric_net_mw"]
    data.heat_transport.f_p_plant_electric_recirc = kwargs["f_p_plant_electric_recirc"]
    data.physics.p_fusion_total_mw = kwargs["p_fusion_total_mw"]
    data.times.t_plant_pulse_coil_precharge = kwargs["t_plant_pulse_coil_precharge"]
    data.times.t_plant_pulse_plasma_current_ramp_up = kwargs[
        "t_plant_pulse_plasma_current_ramp_up"
    ]
    data.times.t_plant_pulse_fusion_ramp = kwargs["t_plant_pulse_fusion_ramp"]
    data.times.t_plant_pulse_burn = kwargs["t_plant_pulse_burn"]
    data.times.t_plant_pulse_plasma_current_ramp_down = kwargs[
        "t_plant_pulse_plasma_current_ramp_down"
    ]
    data.times.t_plant_pulse_dwell = kwargs["t_plant_pulse_dwell"]

    p = Power()
    p.data = data
    p.plant_electric_production()

    return (
        data.power.p_cp_coolant_pump_elec_mw,
        data.heat_transport.p_plant_electric_base_total_mw,
        data.heat_transport.fachtmw,
        data.power.p_plant_core_systems_elec_mw,
        data.heat_transport.p_plant_secondary_heat_mw,
        data.heat_transport.p_plant_electric_gross_mw,
        data.power.p_turbine_loss_mw,
        data.heat_transport.p_plant_electric_recirc_mw,
        data.heat_transport.p_plant_electric_net_mw,
        data.heat_transport.f_p_plant_electric_recirc,
        data.power.e_plant_net_electric_pulse_kwh,
        data.power.e_plant_net_electric_pulse_mj,
        data.power.p_plant_electric_base_total_profile_mw,
        data.power.p_plant_electric_gross_profile_mw,
        data.power.p_plant_electric_net_profile_mw,
        data.power.p_hcd_electric_total_profile_mw,
        data.power.p_coolant_pump_elec_total_profile_mw,
        data.power.p_tf_electric_supplies_profile_mw,
        data.power.p_pf_electric_supplies_profile_mw,
        data.power.vachtmw_profile_mw,
        data.power.p_tritium_plant_electric_profile_mw,
        data.power.p_cryo_plant_electric_profile_mw,
        data.power.p_fusion_total_profile_mw,
    )


def _plant_electric_production_legacy(label, **overrides):
    """A `tests/unit/models/test_power.py::test_plant_electric_production` point.

    That legacy test only sets a subset of fields (the rest come from the `power`
    fixture's freshly-constructed `DataStructure`, i.e. every dataclass default) --
    the remaining fields below are those defaults, made explicit so this port's
    reference adapter (which starts from its own fresh `DataStructure`) sees exactly
    what the legacy test saw.
    """
    kwargs = {
        "itart": 0,
        "i_tf_sup": overrides.pop("i_tf_sup"),
        "p_cp_coolant_pump_elec": 0.0,
        "p_plant_electric_base": 0.0,
        "a_plant_floor_effective": 0.0,
        "pflux_plant_floor_electric": 0.0,
        "p_pf_electric_supplies_mw": overrides.pop("p_pf_electric_supplies_mw"),
        "p_hcd_electric_loss_mw": overrides.pop("p_hcd_electric_loss_mw"),
        "p_coolant_pump_loss_total_mw": overrides.pop("p_coolant_pump_loss_total_mw"),
        "p_div_secondary_heat_mw": overrides.pop("p_div_secondary_heat_mw"),
        "p_shld_secondary_heat_mw": overrides.pop("p_shld_secondary_heat_mw"),
        "p_hcd_secondary_heat_mw": 0.0,
        "p_tf_nuclear_heat_mw": overrides.pop("p_tf_nuclear_heat_mw"),
        "ireactor": overrides.pop("ireactor"),
        "i_blkt_dual_coolant": 0,
        "i_p_coolant_pumping": overrides.pop("i_p_coolant_pumping"),
        "p_blkt_liquid_breeder_heat_deposited_mw": 0.0,
        "etath_liq": 0.375,
        "p_turbine_loss_mw": 0.0,
        "p_plant_electric_recirc_mw": 0.0,
        "f_p_plant_electric_recirc": 0.0,
        "t_plant_pulse_coil_precharge": 30.0,
        "t_plant_pulse_plasma_current_ramp_up": 30.0,
        "t_plant_pulse_fusion_ramp": 10.0,
        "t_plant_pulse_burn": 1000.0,
        "t_plant_pulse_plasma_current_ramp_down": 30.0,
        "t_plant_pulse_dwell": 100.0,
        **overrides,
    }
    return legacy_sample(label, **kwargs)


def _plant_electric_production_samples():
    samples = [
        # tests/unit/models/test_power.py::test_plant_electric_production.
        _plant_electric_production_legacy(
            "baseline-2018-point-1",
            i_p_coolant_pumping=3,
            p_tf_nuclear_heat_mw=0.044178296011112193,
            p_shld_secondary_heat_mw=0,
            vachtmw=0.5,
            p_plant_primary_heat_mw=2620.2218111502593,
            p_hcd_electric_total_mw=129.94611930107126,
            p_tritium_plant_electric_mw=15,
            p_tf_electric_supplies_mw=9.1507079104675704,
            p_coolant_pump_elec_total_mw=234.28554165620102,
            eta_turbine=0.37500000000000006,
            p_cryo_plant_electric_mw=37.900388528497025,
            p_div_secondary_heat_mw=0,
            p_hcd_electric_loss_mw=77.967671580642758,
            p_coolant_pump_loss_total_mw=30.457120415306122,
            p_pf_electric_supplies_mw=0.89998039031509891,
            p_fusion_total_mw=1985.785106643267,
            ireactor=1,
            i_tf_sup=1,
            p_plant_electric_gross_mw=0.0,
            p_plant_electric_net_mw=0.0,
        ),
        _plant_electric_production_legacy(
            "baseline-2018-point-2",
            i_p_coolant_pumping=3,
            p_tf_nuclear_heat_mw=0.045535131445547841,
            p_shld_secondary_heat_mw=0,
            vachtmw=0.5,
            p_plant_primary_heat_mw=2619.4223856129224,
            p_hcd_electric_total_mw=129.94611930107126,
            p_tritium_plant_electric_mw=15,
            p_tf_electric_supplies_mw=9.1507079104675704,
            p_coolant_pump_elec_total_mw=234.2162627659944,
            eta_turbine=0.37500000000000006,
            p_cryo_plant_electric_mw=108.74512702403499,
            p_div_secondary_heat_mw=0,
            p_hcd_electric_loss_mw=77.967671580642758,
            p_coolant_pump_loss_total_mw=30.448114159579291,
            p_pf_electric_supplies_mw=0.068213156646500808,
            p_fusion_total_mw=1985.1653095257811,
            ireactor=1,
            i_tf_sup=1,
            p_plant_electric_gross_mw=0.0,
            p_plant_electric_net_mw=0.0,
        ),
    ]

    bounds = {
        "p_cp_coolant_pump_elec": (0.0, 1.0e6),
        "p_plant_electric_base": (0.0, 1.0e7),
        "a_plant_floor_effective": (1000.0, 50000.0),
        "pflux_plant_floor_electric": (10.0, 200.0),
        "p_cryo_plant_electric_mw": (10.0, 200.0),
        "p_tf_electric_supplies_mw": (1.0, 20.0),
        "p_tritium_plant_electric_mw": (1.0, 30.0),
        "vachtmw": (0.1, 5.0),
        "p_pf_electric_supplies_mw": (0.01, 5.0),
        "p_hcd_electric_loss_mw": (10.0, 100.0),
        "p_coolant_pump_loss_total_mw": (5.0, 60.0),
        "p_div_secondary_heat_mw": (0.0, 50.0),
        "p_shld_secondary_heat_mw": (0.0, 50.0),
        "p_hcd_secondary_heat_mw": (0.0, 20.0),
        "p_tf_nuclear_heat_mw": (0.0, 1.0),
        "p_plant_primary_heat_mw": (1000.0, 3000.0),
        "p_blkt_liquid_breeder_heat_deposited_mw": (0.0, 500.0),
        "eta_turbine": (0.2, 0.6),
        "etath_liq": (0.2, 0.6),
        "p_hcd_electric_total_mw": (50.0, 300.0),
        "p_coolant_pump_elec_total_mw": (10.0, 400.0),
        "p_plant_electric_gross_mw": (0.0, 1500.0),
        "p_turbine_loss_mw": (0.0, 1500.0),
        "p_plant_electric_recirc_mw": (0.0, 800.0),
        "p_plant_electric_net_mw": (0.0, 1000.0),
        "f_p_plant_electric_recirc": (0.0, 0.9),
        "p_fusion_total_mw": (500.0, 3000.0),
        "t_plant_pulse_coil_precharge": (1.0, 50.0),
        "t_plant_pulse_plasma_current_ramp_up": (10.0, 500.0),
        "t_plant_pulse_fusion_ramp": (1.0, 50.0),
        "t_plant_pulse_burn": (100.0, 10000.0),
        "t_plant_pulse_plasma_current_ramp_down": (10.0, 100.0),
        "t_plant_pulse_dwell": (10.0, 2000.0),
    }
    combos = [
        (0, 0, 1, 0, 3),  # ireactor off: gross/net/recirc are pass-throughs
        (0, 1, 1, 0, 3),
        (0, 1, 1, 1, 2),  # dual-coolant + MECHANICAL: the alternate gross-power arm
        (
            1,
            0,
            0,
            0,
            1,
        ),  # tight-aspect-ratio, resistive centrepost: owns p_cp_coolant_pump_elec_mw
    ]
    for itart, i_tf_sup, ireactor, i_blkt_dual_coolant, i_p_coolant_pumping in combos:
        fixed = {
            "itart": itart,
            "i_tf_sup": i_tf_sup,
            "ireactor": ireactor,
            "i_blkt_dual_coolant": i_blkt_dual_coolant,
            "i_p_coolant_pumping": i_p_coolant_pumping,
        }
        samples.extend(
            fuzz_samples(
                bounds, count=15, seed=100260818 + ireactor + itart, fixed=fixed
            )
        )
    return samples


class TestPlantElectricProduction(Tier1Contract):
    audit_record = "models/power/electric_production.md"
    reference = _reference_plant_electric_production
    ported = calculate_plant_electric_production
    static_argnames = _PEP_STATIC_ARGNAMES
    samples = _plant_electric_production_samples()
