"""Harness cases for the ported subset of `costs/costs.py` (registry unit #18).

All 23 ported functions are tier-1 -- see `costs.md`. No PROCESS unit test exists with
convenient, individually-liftable literal parameter sets for these methods (`tests/unit/
models/test_costs_1990.py` drives them from JSON-backed `monkeypatch` fixtures spanning
many fields at once, not a small literal kwargs dict per case) -- fuzz-only, the same
provenance `build.py`/`forces.py`/`mass.py`/`quench.py` already established as acceptable
for this project when no convenient legacy literal exists (`unit_registry.md`'s row 2).
A handful of `legacy_sample` points are still given per function to pin each branch
deterministically (`ireactor`/`ife`/`ifueltyp`/`itart`/`i_tf_sup`/`i_vacuum_pump_type`
values), since fuzzing alone would only hit every branch by chance.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.costs.costs import (
    calculate_atmospheric_recovery_cost,
    calculate_auxiliary_component_cooling_cost,
    calculate_auxiliary_facility_power_cost,
    calculate_blanket_cost,
    calculate_constructed_cost,
    calculate_cost_of_electricity,
    calculate_cryogenic_system_cost,
    calculate_diesel_generators_cost,
    calculate_divertor_cost,
    calculate_electric_plant_equipment_cost,
    calculate_energy_storage_cost,
    calculate_first_wall_cost,
    calculate_fuel_handling_cost,
    calculate_fuel_processing_cost,
    calculate_fuelling_system_cost,
    calculate_fusion_power_island_cost,
    calculate_heat_rejection_cost,
    calculate_heat_transport_system_cost,
    calculate_indirect_costs,
    calculate_instrumentation_and_control_cost,
    calculate_low_voltage_cost,
    calculate_magnets_cost,
    calculate_maintenance_equipment_cost,
    calculate_misc_plant_equipment_cost,
    calculate_nuclear_building_ventilation_cost,
    calculate_pf_coil_power_conditioning_cost,
    calculate_pf_magnet_cost,
    calculate_power_conditioning_cost,
    calculate_power_injection_cost,
    calculate_reactor_cooling_system_cost,
    calculate_reactor_cost,
    calculate_reactor_structure_cost,
    calculate_shield_cost,
    calculate_structures_cost,
    calculate_switchyard_cost,
    calculate_tf_coil_power_conditioning_cost,
    calculate_tf_magnet_cost_resistive,
    calculate_tf_magnet_cost_superconducting,
    calculate_total_plant_direct_cost,
    calculate_transformers_cost,
    calculate_turbine_plant_equipment_cost,
    calculate_vacuum_system_cost,
    calculate_vacuum_vessel_assembly_cost,
    convert_fpy_to_calendar,
)
from process.core.model import DataStructure
from process.models.costs.costs import Costs


def _make_costs():
    costs = Costs()
    costs.data = DataStructure()
    return costs


def _reference_convert_fpy_to_calendar(
    life_blkt_fpy, life_plant, f_t_plant_available, life_div_fpy, itart, cplife
):
    costs = _make_costs()
    costs.data.fwbs.life_blkt_fpy = life_blkt_fpy
    costs.data.costs.life_plant = life_plant
    costs.data.costs.f_t_plant_available = f_t_plant_available
    costs.data.costs.life_div_fpy = life_div_fpy
    costs.data.physics.itart = itart
    costs.data.costs.cplife = cplife
    costs.convert_fpy_to_calendar()
    return (
        costs.data.fwbs.life_blkt,
        costs.data.costs.cdrlife_cal,
        costs.data.costs.life_div,
        costs.data.costs.cplife_cal,
    )


def _reference_structures_cost(
    csi,
    lsa,
    cland,
    ucrb,
    rbvol,
    UCMB,
    rmbvol,
    UCWS,
    wsvol,
    UCTR,
    triv,
    UCEL,
    elevol,
    UCAD,
    admvol,
    UCCO,
    convol,
    UCSH,
    shovol,
    UCCR,
    cryvol,
    ireactor,
    cturbb,
):
    costs = _make_costs()
    costs.data.costs.csi = csi
    costs.data.costs.lsa = lsa
    costs.data.costs.cland = cland
    costs.data.costs.ucrb = ucrb
    costs.data.buildings.rbvol = rbvol
    costs.data.costs.UCMB = UCMB
    costs.data.buildings.rmbvol = rmbvol
    costs.data.costs.UCWS = UCWS
    costs.data.buildings.wsvol = wsvol
    costs.data.costs.UCTR = UCTR
    costs.data.buildings.triv = triv
    costs.data.costs.UCEL = UCEL
    costs.data.buildings.elevol = elevol
    costs.data.costs.UCAD = UCAD
    costs.data.buildings.admvol = admvol
    costs.data.costs.UCCO = UCCO
    costs.data.buildings.convol = convol
    costs.data.costs.UCSH = UCSH
    costs.data.buildings.shovol = shovol
    costs.data.costs.UCCR = UCCR
    costs.data.buildings.cryvol = cryvol
    costs.data.costs.ireactor = ireactor
    costs.data.costs.cturbb = cturbb
    costs.acc21()
    c = costs.data.costs
    return (
        c.c211,
        c.c212,
        c.c213,
        c.c2141,
        c.c2142,
        c.c214,
        c.c215,
        c.c216,
        c.c2171,
        c.c2172,
        c.c2173,
        c.c2174,
        c.c217,
        c.c21,
    )


def _reference_indirect_costs(cfind, lsa, cdirt, cowner, fcontng):
    costs = _make_costs()
    costs.data.costs.cfind = cfind
    costs.data.costs.lsa = lsa
    costs.data.costs.cdirt = cdirt
    costs.data.costs.cowner = cowner
    costs.data.costs.fcontng = fcontng
    costs.acc9()
    return costs.data.costs.cindrt, costs.data.costs.ccont


def _reference_reactor_structure_cost(gsmass, UCGSS, lsa, fkind):
    costs = _make_costs()
    costs.data.structure.gsmass = gsmass
    costs.data.costs.UCGSS = UCGSS
    costs.data.costs.lsa = lsa
    costs.data.costs.fkind = fkind
    costs.acc2214()
    return costs.data.costs.c2214


def _reference_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind):
    costs = _make_costs()
    costs.data.fwbs.m_vv = m_vv
    costs.data.costs.uccryo = uccryo
    costs.data.costs.lsa = lsa
    costs.data.costs.fkind = fkind
    costs.acc2223()
    return costs.data.costs.c2223


def _reference_divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.divertor.a_div_surface_total = a_div_surface_total
    costs.data.costs.ucdiv = ucdiv
    costs.data.costs.fkind = fkind
    costs.data.costs.ifueltyp = ifueltyp
    costs.acc2215()
    return costs.data.costs.c2215, costs.data.costs.divcst


def _reference_vacuum_system_cost(
    i_vacuum_pump_type,
    n_vac_pumps_high,
    UCCPMP,
    UCTPMP,
    n_vv_vacuum_ducts,
    UCBPMP,
    dlscal,
    UCDUCT,
    dia_vv_vacuum_ducts,
    UCVALV,
    m_vv_vacuum_duct_shield,
    UCVDSH,
    UCVIAC,
    fkind,
):
    costs = _make_costs()
    costs.data.vacuum.i_vacuum_pump_type = i_vacuum_pump_type
    costs.data.vacuum.n_vac_pumps_high = n_vac_pumps_high
    costs.data.costs.UCCPMP = UCCPMP
    costs.data.costs.UCTPMP = UCTPMP
    costs.data.vacuum.n_vv_vacuum_ducts = n_vv_vacuum_ducts
    costs.data.costs.UCBPMP = UCBPMP
    costs.data.vacuum.dlscal = dlscal
    costs.data.costs.UCDUCT = UCDUCT
    costs.data.vacuum.dia_vv_vacuum_ducts = dia_vv_vacuum_ducts
    costs.data.costs.UCVALV = UCVALV
    costs.data.vacuum.m_vv_vacuum_duct_shield = m_vv_vacuum_duct_shield
    costs.data.costs.UCVDSH = UCVDSH
    costs.data.costs.UCVIAC = UCVIAC
    costs.data.costs.fkind = fkind
    costs.acc224()
    c = costs.data.costs
    return c.c2241, c.c2242, c.c2243, c.c2244, c.c2245, c.c2246, c.c224


def _reference_tf_coil_power_conditioning_cost(
    uctfps,
    tfckw,
    tfcmw,
    i_tf_sup,
    uctfbr,
    n_tf_coils,
    c_tf_turn,
    v_tf_coil_dump_quench_kv,
    uctfsw,
    UCTFDR,
    e_tf_magnetic_stored_total_gj,
    UCTFGR,
    UCTFIC,
    uctfbus,
    m_tf_bus,
    ucbus,
    len_tf_bus,
    fkind,
):
    costs = _make_costs()
    costs.data.costs.uctfps = uctfps
    costs.data.tfcoil.tfckw = tfckw
    costs.data.tfcoil.tfcmw = tfcmw
    costs.data.tfcoil.i_tf_sup = i_tf_sup
    costs.data.costs.uctfbr = uctfbr
    costs.data.tfcoil.n_tf_coils = n_tf_coils
    costs.data.tfcoil.c_tf_turn = c_tf_turn
    costs.data.tfcoil.v_tf_coil_dump_quench_kv = v_tf_coil_dump_quench_kv
    costs.data.costs.uctfsw = uctfsw
    costs.data.costs.UCTFDR = UCTFDR
    costs.data.tfcoil.e_tf_magnetic_stored_total_gj = e_tf_magnetic_stored_total_gj
    costs.data.costs.UCTFGR = UCTFGR
    costs.data.costs.UCTFIC = UCTFIC
    costs.data.costs.uctfbus = uctfbus
    costs.data.tfcoil.m_tf_bus = m_tf_bus
    costs.data.costs.ucbus = ucbus
    costs.data.tfcoil.len_tf_bus = len_tf_bus
    costs.data.costs.fkind = fkind
    costs.acc2251()
    c = costs.data.costs
    return c.c22511, c.c22512, c.c22513, c.c22514, c.c22515, c.c2251


def _reference_pf_coil_power_conditioning_cost(
    ucpfps,
    peakmva,
    ucpfic,
    pfckts,
    ucpfb,
    spfbusl,
    acptmax,
    ucpfbs,
    srcktpm,
    ucpfbk,
    vpfskv,
    ucpfdr1,
    ensxpfm,
    ucpfcb,
    fkind,
):
    costs = _make_costs()
    costs.data.costs.ucpfps = ucpfps
    costs.data.heat_transport.peakmva = peakmva
    costs.data.costs.ucpfic = ucpfic
    costs.data.pf_power.pfckts = pfckts
    costs.data.costs.ucpfb = ucpfb
    costs.data.pf_power.spfbusl = spfbusl
    costs.data.pf_power.acptmax = acptmax
    costs.data.costs.ucpfbs = ucpfbs
    costs.data.pf_power.srcktpm = srcktpm
    costs.data.costs.ucpfbk = ucpfbk
    costs.data.pf_power.vpfskv = vpfskv
    costs.data.costs.ucpfdr1 = ucpfdr1
    costs.data.pf_power.ensxpfm = ensxpfm
    costs.data.costs.ucpfcb = ucpfcb
    costs.data.costs.fkind = fkind
    costs.acc2252()
    c = costs.data.costs
    return c.c22521, c.c22522, c.c22523, c.c22524, c.c22525, c.c22526, c.c22527, c.c2252


def _reference_reactor_cooling_system_cost(
    uchts,
    i_blkt_coolant_type,
    p_fw_div_heat_deposited_mw,
    p_blkt_nuclear_heat_total_mw,
    p_shld_nuclear_heat_mw,
    lsa,
    fkind,
    UCPHX,
    n_primary_heat_exchangers,
    p_plant_primary_heat_mw,
):
    costs = _make_costs()
    costs.data.costs.uchts = uchts
    costs.data.fwbs.i_blkt_coolant_type = i_blkt_coolant_type
    costs.data.heat_transport.p_fw_div_heat_deposited_mw = p_fw_div_heat_deposited_mw
    costs.data.fwbs.p_blkt_nuclear_heat_total_mw = p_blkt_nuclear_heat_total_mw
    costs.data.fwbs.p_shld_nuclear_heat_mw = p_shld_nuclear_heat_mw
    costs.data.costs.lsa = lsa
    costs.data.costs.fkind = fkind
    costs.data.costs.UCPHX = UCPHX
    costs.data.heat_transport.n_primary_heat_exchangers = n_primary_heat_exchangers
    costs.data.heat_transport.p_plant_primary_heat_mw = p_plant_primary_heat_mw
    costs.acc2261()
    c = costs.data.costs
    return c.cpp, c.chx, c.c2261


def _reference_fuelling_system_cost(ucf1, fkind):
    costs = _make_costs()
    costs.data.costs.ucf1 = ucf1
    costs.data.costs.fkind = fkind
    costs.acc2271()
    return costs.data.costs.c2271


def _reference_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind):
    costs = _make_costs()
    costs.data.costs.UCNBV = UCNBV
    costs.data.buildings.volrci = volrci
    costs.data.buildings.wsvol = wsvol
    costs.data.costs.fkind = fkind
    costs.acc2274()
    return costs.data.costs.c2274


def _reference_instrumentation_and_control_cost(uciac, fkind):
    costs = _make_costs()
    costs.data.costs.uciac = uciac
    costs.data.costs.fkind = fkind
    costs.acc228()
    return costs.data.costs.c228


def _reference_maintenance_equipment_cost(ucme, fkind):
    costs = _make_costs()
    costs.data.costs.ucme = ucme
    costs.data.costs.fkind = fkind
    costs.acc229()
    return costs.data.costs.c229


def _reference_turbine_plant_equipment_cost(
    ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
):
    costs = _make_costs()
    costs.data.costs.ireactor = ireactor
    costs.data.costs.ucturb = ucturb
    costs.data.fwbs.i_blkt_coolant_type = i_blkt_coolant_type
    costs.data.heat_transport.p_plant_electric_gross_mw = p_plant_electric_gross_mw
    costs.acc23()
    return costs.data.costs.c23


def _reference_switchyard_cost(UCSWYD, lsa):
    costs = _make_costs()
    costs.data.costs.UCSWYD = UCSWYD
    costs.data.costs.lsa = lsa
    costs.acc241()
    return costs.data.costs.c241


def _reference_transformers_cost(
    UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa
):
    costs = _make_costs()
    costs.data.costs.UCPP = UCPP
    costs.data.heat_transport.pacpmw = pacpmw
    costs.data.costs.UCAP = UCAP
    costs.data.heat_transport.p_plant_electric_base_total_mw = (
        p_plant_electric_base_total_mw
    )
    costs.data.costs.lsa = lsa
    costs.acc242()
    return costs.data.costs.c242


def _reference_low_voltage_cost(UCLV, tlvpmw, lsa):
    costs = _make_costs()
    costs.data.costs.UCLV = UCLV
    costs.data.heat_transport.tlvpmw = tlvpmw
    costs.data.costs.lsa = lsa
    costs.acc243()
    return costs.data.costs.c243


def _reference_diesel_generators_cost(UCDGEN, lsa):
    costs = _make_costs()
    costs.data.costs.UCDGEN = UCDGEN
    costs.data.costs.lsa = lsa
    costs.acc244()
    return costs.data.costs.c244


def _reference_auxiliary_facility_power_cost(UCAF, lsa):
    costs = _make_costs()
    costs.data.costs.UCAF = UCAF
    costs.data.costs.lsa = lsa
    costs.acc245()
    return costs.data.costs.c245


def _reference_electric_plant_equipment_cost(c241, c242, c243, c244, c245):
    costs = _make_costs()
    costs.data.costs.c241 = c241
    costs.data.costs.c242 = c242
    costs.data.costs.c243 = c243
    costs.data.costs.c244 = c244
    costs.data.costs.c245 = c245
    costs.acc24()
    return costs.data.costs.c24


def _reference_misc_plant_equipment_cost(ucmisc, lsa):
    costs = _make_costs()
    costs.data.costs.ucmisc = ucmisc
    costs.data.costs.lsa = lsa
    costs.acc25()
    return costs.data.costs.c25


def _reference_heat_rejection_cost(
    ireactor,
    p_fusion_total_mw,
    p_hcd_electric_total_mw,
    tfcmw,
    p_plant_primary_heat_mw,
    p_plant_electric_gross_mw,
    uchrs,
    lsa,
):
    costs = _make_costs()
    costs.data.costs.ireactor = ireactor
    costs.data.physics.p_fusion_total_mw = p_fusion_total_mw
    costs.data.heat_transport.p_hcd_electric_total_mw = p_hcd_electric_total_mw
    costs.data.tfcoil.tfcmw = tfcmw
    costs.data.heat_transport.p_plant_primary_heat_mw = p_plant_primary_heat_mw
    costs.data.heat_transport.p_plant_electric_gross_mw = p_plant_electric_gross_mw
    costs.data.costs.uchrs = uchrs
    costs.data.costs.lsa = lsa
    costs.acc26()
    return costs.data.costs.c26


class TestConvertFpyToCalendar(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_convert_fpy_to_calendar
    ported = convert_fpy_to_calendar
    static_argnames = ("itart",)

    samples = [
        legacy_sample(
            "fast-branches",
            life_blkt_fpy=5.0,
            life_plant=30.0,
            f_t_plant_available=0.8,
            life_div_fpy=3.0,
            itart=0,
            cplife=10.0,
        ),
        legacy_sample(
            "slow-branches",
            life_blkt_fpy=40.0,
            life_plant=30.0,
            f_t_plant_available=0.8,
            life_div_fpy=40.0,
            itart=1,
            cplife=40.0,
        ),
        legacy_sample(
            "itart-fast",
            life_blkt_fpy=5.0,
            life_plant=30.0,
            f_t_plant_available=0.8,
            life_div_fpy=3.0,
            itart=1,
            cplife=10.0,
        ),
    ]
    fuzz_bounds = {
        "life_blkt_fpy": (1.0, 50.0),
        "life_plant": (10.0, 40.0),
        "f_t_plant_available": (0.5, 1.0),
        "life_div_fpy": (1.0, 50.0),
        "cplife": (1.0, 50.0),
    }
    fuzz_fixed = {"itart": 0}


class TestStructuresCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_structures_cost
    ported = calculate_structures_cost
    static_argnames = ("lsa", "ireactor")

    samples = [
        legacy_sample(
            "reactor",
            csi=16.0,
            lsa=4,
            cland=19.2,
            ucrb=400.0,
            rbvol=1.0e5,
            UCMB=260.0,
            rmbvol=5.0e4,
            UCWS=460.0,
            wsvol=1.0e4,
            UCTR=370.0,
            triv=4.0e4,
            UCEL=380.0,
            elevol=8.0e3,
            UCAD=180.0,
            admvol=1.0e4,
            UCCO=350.0,
            convol=6.0e3,
            UCSH=115.0,
            shovol=5.0e3,
            UCCR=460.0,
            cryvol=2.0e4,
            ireactor=1,
            cturbb=100.0,
        ),
        legacy_sample(
            "non-reactor-lsa1",
            csi=16.0,
            lsa=1,
            cland=19.2,
            ucrb=400.0,
            rbvol=1.0e5,
            UCMB=260.0,
            rmbvol=5.0e4,
            UCWS=460.0,
            wsvol=1.0e4,
            UCTR=370.0,
            triv=4.0e4,
            UCEL=380.0,
            elevol=8.0e3,
            UCAD=180.0,
            admvol=1.0e4,
            UCCO=350.0,
            convol=6.0e3,
            UCSH=115.0,
            shovol=5.0e3,
            UCCR=460.0,
            cryvol=2.0e4,
            ireactor=0,
            cturbb=100.0,
        ),
    ]
    fuzz_bounds = {
        "csi": (1.0, 50.0),
        "cland": (1.0, 50.0),
        "ucrb": (100.0, 800.0),
        "rbvol": (1.0e4, 1.0e6),
        "UCMB": (100.0, 500.0),
        "rmbvol": (1.0e4, 1.0e5),
        "UCWS": (100.0, 800.0),
        "wsvol": (1.0e3, 5.0e4),
        "UCTR": (100.0, 800.0),
        "triv": (1.0e3, 1.0e5),
        "UCEL": (100.0, 800.0),
        "elevol": (1.0e3, 5.0e4),
        "UCAD": (50.0, 400.0),
        "admvol": (1.0e3, 5.0e4),
        "UCCO": (100.0, 800.0),
        "convol": (1.0e3, 3.0e4),
        "UCSH": (50.0, 400.0),
        "shovol": (1.0e3, 3.0e4),
        "UCCR": (100.0, 800.0),
        "cryvol": (1.0e3, 5.0e4),
        "cturbb": (10.0, 500.0),
    }
    fuzz_fixed = {"lsa": 4, "ireactor": 1}


class TestIndirectCosts(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_indirect_costs
    ported = calculate_indirect_costs
    static_argnames = ("lsa",)

    samples = [
        legacy_sample(
            "lsa4",
            cfind=[0.244, 0.244, 0.244, 0.29],
            lsa=4,
            cdirt=1000.0,
            cowner=0.15,
            fcontng=0.195,
        ),
        legacy_sample(
            "lsa1",
            cfind=[0.244, 0.244, 0.244, 0.29],
            lsa=1,
            cdirt=1000.0,
            cowner=0.15,
            fcontng=0.195,
        ),
    ]
    fuzz_bounds = {"cdirt": (100.0, 1.0e4), "cowner": (0.0, 0.5), "fcontng": (0.0, 0.5)}
    fuzz_fixed = {"cfind": [0.244, 0.244, 0.244, 0.29], "lsa": 4}


class TestReactorStructureCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_reactor_structure_cost
    ported = calculate_reactor_structure_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", gsmass=5.0e5, UCGSS=35.0, lsa=4, fkind=1.0)]
    fuzz_bounds = {"gsmass": (1.0e4, 1.0e6), "UCGSS": (10.0, 100.0), "fkind": (0.5, 1.0)}
    fuzz_fixed = {"lsa": 4}


class TestVacuumVesselAssemblyCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_vacuum_vessel_assembly_cost
    ported = calculate_vacuum_vessel_assembly_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", m_vv=9.0e6, uccryo=32.0, lsa=4, fkind=1.0)]
    fuzz_bounds = {"m_vv": (1.0e5, 2.0e7), "uccryo": (10.0, 100.0), "fkind": (0.5, 1.0)}
    fuzz_fixed = {"lsa": 4}


class TestDivertorCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_divertor_cost
    ported = calculate_divertor_cost
    static_argnames = ("ife", "ifueltyp")

    samples = [
        legacy_sample(
            "not-ife-capital",
            ife=0,
            a_div_surface_total=100.0,
            ucdiv=2.8e5,
            fkind=1.0,
            ifueltyp=0,
        ),
        legacy_sample(
            "not-ife-fuel-cost",
            ife=0,
            a_div_surface_total=100.0,
            ucdiv=2.8e5,
            fkind=1.0,
            ifueltyp=1,
        ),
        legacy_sample(
            "not-ife-both",
            ife=0,
            a_div_surface_total=100.0,
            ucdiv=2.8e5,
            fkind=1.0,
            ifueltyp=2,
        ),
        legacy_sample(
            "ife",
            ife=1,
            a_div_surface_total=100.0,
            ucdiv=2.8e5,
            fkind=1.0,
            ifueltyp=0,
        ),
    ]
    fuzz_bounds = {
        "a_div_surface_total": (10.0, 1000.0),
        "ucdiv": (1.0e4, 1.0e6),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "ifueltyp": 0}


class TestVacuumSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_vacuum_system_cost
    ported = calculate_vacuum_system_cost
    static_argnames = ("i_vacuum_pump_type",)

    samples = [
        legacy_sample(
            "turbomolecular",
            i_vacuum_pump_type=0,
            n_vac_pumps_high=8.0,
            UCCPMP=2.5e5,
            UCTPMP=2.5e5,
            n_vv_vacuum_ducts=8.0,
            UCBPMP=2.5e4,
            dlscal=1.0,
            UCDUCT=3.0e4,
            dia_vv_vacuum_ducts=1.0,
            UCVALV=1.5e5,
            m_vv_vacuum_duct_shield=1.0e4,
            UCVDSH=90.0,
            UCVIAC=1.5e7,
            fkind=1.0,
        ),
        legacy_sample(
            "compound-cryopump",
            i_vacuum_pump_type=1,
            n_vac_pumps_high=8.0,
            UCCPMP=2.5e5,
            UCTPMP=2.5e5,
            n_vv_vacuum_ducts=8.0,
            UCBPMP=2.5e4,
            dlscal=1.0,
            UCDUCT=3.0e4,
            dia_vv_vacuum_ducts=1.0,
            UCVALV=1.5e5,
            m_vv_vacuum_duct_shield=1.0e4,
            UCVDSH=90.0,
            UCVIAC=1.5e7,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "n_vac_pumps_high": (1.0, 20.0),
        "UCCPMP": (1.0e4, 1.0e6),
        "UCTPMP": (1.0e4, 1.0e6),
        "n_vv_vacuum_ducts": (1.0, 20.0),
        "UCBPMP": (1.0e3, 1.0e5),
        "dlscal": (0.1, 10.0),
        "UCDUCT": (1.0e3, 1.0e5),
        "dia_vv_vacuum_ducts": (0.1, 5.0),
        "UCVALV": (1.0e4, 1.0e6),
        "m_vv_vacuum_duct_shield": (1.0e2, 5.0e4),
        "UCVDSH": (10.0, 500.0),
        "UCVIAC": (1.0e6, 5.0e7),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"i_vacuum_pump_type": 0}


class TestTfCoilPowerConditioningCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_tf_coil_power_conditioning_cost
    ported = calculate_tf_coil_power_conditioning_cost
    static_argnames = ("i_tf_sup",)

    samples = [
        legacy_sample(
            "resistive",
            uctfps=2.4e4,
            tfckw=1.0e3,
            tfcmw=50.0,
            i_tf_sup=0,
            uctfbr=1.22e6,
            n_tf_coils=16.0,
            c_tf_turn=6.0e4,
            v_tf_coil_dump_quench_kv=20.0,
            uctfsw=1.0e5,
            UCTFDR=1.75e-3,
            e_tf_magnetic_stored_total_gj=40.0,
            UCTFGR=1.0e5,
            UCTFIC=1.0e4,
            uctfbus=1.0,
            m_tf_bus=1.0e4,
            ucbus=460.0,
            len_tf_bus=300.0,
            fkind=1.0,
        ),
        legacy_sample(
            "superconducting",
            uctfps=2.4e4,
            tfckw=1.0e3,
            tfcmw=50.0,
            i_tf_sup=1,
            uctfbr=1.22e6,
            n_tf_coils=16.0,
            c_tf_turn=6.0e4,
            v_tf_coil_dump_quench_kv=20.0,
            uctfsw=1.0e5,
            UCTFDR=1.75e-3,
            e_tf_magnetic_stored_total_gj=40.0,
            UCTFGR=1.0e5,
            UCTFIC=1.0e4,
            uctfbus=1.0,
            m_tf_bus=1.0e4,
            ucbus=460.0,
            len_tf_bus=300.0,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "uctfps": (1.0e3, 1.0e5),
        "tfckw": (1.0e2, 1.0e4),
        "tfcmw": (1.0, 200.0),
        "uctfbr": (1.0e5, 5.0e6),
        "n_tf_coils": (10.0, 24.0),
        "c_tf_turn": (1.0e3, 1.0e5),
        "v_tf_coil_dump_quench_kv": (1.0, 50.0),
        "uctfsw": (1.0e3, 1.0e6),
        "UCTFDR": (1.0e-4, 1.0e-2),
        "e_tf_magnetic_stored_total_gj": (1.0, 200.0),
        "UCTFGR": (1.0e3, 1.0e6),
        "UCTFIC": (1.0e2, 1.0e5),
        "uctfbus": (0.1, 10.0),
        "m_tf_bus": (1.0e2, 5.0e4),
        "ucbus": (100.0, 1000.0),
        "len_tf_bus": (10.0, 1000.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"i_tf_sup": 1}


class TestPfCoilPowerConditioningCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_pf_coil_power_conditioning_cost
    ported = calculate_pf_coil_power_conditioning_cost

    samples = [
        legacy_sample(
            "nominal",
            ucpfps=3.5e4,
            peakmva=300.0,
            ucpfic=1.0e4,
            pfckts=12.0,
            ucpfb=1.0e4,
            spfbusl=300.0,
            acptmax=3.0e4,
            ucpfbs=1.0e5,
            srcktpm=3.0e4,
            ucpfbk=3.0e4,
            vpfskv=20.0,
            ucpfdr1=1.5e4,
            ensxpfm=1000.0,
            ucpfcb=7.5e4,
            fkind=1.0,
        ),
        legacy_sample(
            "zero-circuits",
            ucpfps=3.5e4,
            peakmva=300.0,
            ucpfic=1.0e4,
            pfckts=0.0,
            ucpfb=1.0e4,
            spfbusl=300.0,
            acptmax=3.0e4,
            ucpfbs=1.0e5,
            srcktpm=3.0e4,
            ucpfbk=3.0e4,
            vpfskv=20.0,
            ucpfdr1=1.5e4,
            ensxpfm=1000.0,
            ucpfcb=7.5e4,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "ucpfps": (1.0e3, 1.0e5),
        "peakmva": (10.0, 1000.0),
        "ucpfic": (1.0e3, 1.0e5),
        "pfckts": (1.0, 30.0),
        "ucpfb": (1.0e3, 1.0e5),
        "spfbusl": (10.0, 1000.0),
        "acptmax": (1.0e3, 1.0e5),
        "ucpfbs": (1.0e4, 1.0e6),
        "srcktpm": (1.0e3, 1.0e5),
        "ucpfbk": (1.0e3, 1.0e5),
        "vpfskv": (1.0, 50.0),
        "ucpfdr1": (1.0e3, 1.0e5),
        "ensxpfm": (10.0, 5.0e3),
        "ucpfcb": (1.0e4, 1.0e6),
        "fkind": (0.5, 1.0),
    }


class TestReactorCoolingSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_reactor_cooling_system_cost
    ported = calculate_reactor_cooling_system_cost
    static_argnames = ("lsa", "i_blkt_coolant_type")

    samples = [
        legacy_sample(
            "water",
            uchts=[15.3, 19.1],
            i_blkt_coolant_type=1,
            p_fw_div_heat_deposited_mw=200.0,
            p_blkt_nuclear_heat_total_mw=300.0,
            p_shld_nuclear_heat_mw=20.0,
            lsa=4,
            fkind=1.0,
            UCPHX=15.0,
            n_primary_heat_exchangers=3,
            p_plant_primary_heat_mw=1500.0,
        ),
        legacy_sample(
            "helium",
            uchts=[15.3, 19.1],
            i_blkt_coolant_type=2,
            p_fw_div_heat_deposited_mw=200.0,
            p_blkt_nuclear_heat_total_mw=300.0,
            p_shld_nuclear_heat_mw=20.0,
            lsa=4,
            fkind=1.0,
            UCPHX=15.0,
            n_primary_heat_exchangers=3,
            p_plant_primary_heat_mw=1500.0,
        ),
    ]
    fuzz_bounds = {
        "p_fw_div_heat_deposited_mw": (10.0, 1000.0),
        "p_blkt_nuclear_heat_total_mw": (10.0, 1000.0),
        "p_shld_nuclear_heat_mw": (1.0, 200.0),
        "fkind": (0.5, 1.0),
        "UCPHX": (1.0, 100.0),
        "n_primary_heat_exchangers": (1.0, 6.0),
        "p_plant_primary_heat_mw": (100.0, 3000.0),
    }
    fuzz_fixed = {"uchts": [15.3, 19.1], "i_blkt_coolant_type": 1, "lsa": 4}


class TestFuellingSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fuelling_system_cost
    ported = calculate_fuelling_system_cost

    samples = [legacy_sample("nominal", ucf1=2.23e7, fkind=1.0)]
    fuzz_bounds = {"ucf1": (1.0e6, 5.0e7), "fkind": (0.5, 1.0)}


class TestNuclearBuildingVentilationCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_nuclear_building_ventilation_cost
    ported = calculate_nuclear_building_ventilation_cost

    samples = [
        legacy_sample("nominal", UCNBV=1.0e6, volrci=1.0e5, wsvol=1.0e4, fkind=1.0)
    ]
    fuzz_bounds = {
        "UCNBV": (1.0e5, 5.0e6),
        "volrci": (1.0e4, 5.0e5),
        "wsvol": (1.0e3, 5.0e4),
        "fkind": (0.5, 1.0),
    }


class TestInstrumentationAndControlCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_instrumentation_and_control_cost
    ported = calculate_instrumentation_and_control_cost

    samples = [legacy_sample("nominal", uciac=4.0e7, fkind=1.0)]
    fuzz_bounds = {"uciac": (1.0e6, 1.0e8), "fkind": (0.5, 1.0)}


class TestMaintenanceEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_maintenance_equipment_cost
    ported = calculate_maintenance_equipment_cost

    samples = [legacy_sample("nominal", ucme=1.25e8, fkind=1.0)]
    fuzz_bounds = {"ucme": (1.0e7, 5.0e8), "fkind": (0.5, 1.0)}


class TestTurbinePlantEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_turbine_plant_equipment_cost
    ported = calculate_turbine_plant_equipment_cost
    static_argnames = ("ireactor", "i_blkt_coolant_type")

    samples = [
        legacy_sample(
            "reactor",
            ireactor=1,
            ucturb=[230.0e6, 245.0e6],
            i_blkt_coolant_type=1,
            p_plant_electric_gross_mw=1200.0,
        ),
        legacy_sample(
            "non-reactor",
            ireactor=0,
            ucturb=[230.0e6, 245.0e6],
            i_blkt_coolant_type=1,
            p_plant_electric_gross_mw=1200.0,
        ),
    ]
    fuzz_bounds = {"p_plant_electric_gross_mw": (100.0, 3000.0)}
    fuzz_fixed = {"ireactor": 1, "ucturb": [230.0e6, 245.0e6], "i_blkt_coolant_type": 1}


class TestSwitchyardCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_switchyard_cost
    ported = calculate_switchyard_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCSWYD=1.9e7, lsa=4)]
    fuzz_bounds = {"UCSWYD": (1.0e6, 5.0e7)}
    fuzz_fixed = {"lsa": 4}


class TestTransformersCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_transformers_cost
    ported = calculate_transformers_cost
    static_argnames = ("lsa",)

    samples = [
        legacy_sample(
            "nominal",
            UCPP=48.0,
            pacpmw=50.0,
            UCAP=15.0,
            p_plant_electric_base_total_mw=10.0,
            lsa=4,
        )
    ]
    fuzz_bounds = {
        "UCPP": (10.0, 200.0),
        "pacpmw": (1.0, 500.0),
        "UCAP": (1.0, 100.0),
        "p_plant_electric_base_total_mw": (1.0, 100.0),
    }
    fuzz_fixed = {"lsa": 4}


class TestLowVoltageCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_low_voltage_cost
    ported = calculate_low_voltage_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCLV=265.0, tlvpmw=10.0, lsa=4)]
    fuzz_bounds = {"UCLV": (50.0, 500.0), "tlvpmw": (0.1, 100.0)}
    fuzz_fixed = {"lsa": 4}


class TestDieselGeneratorsCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_diesel_generators_cost
    ported = calculate_diesel_generators_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCDGEN=1.7e6, lsa=4)]
    fuzz_bounds = {"UCDGEN": (1.0e5, 5.0e6)}
    fuzz_fixed = {"lsa": 4}


class TestAuxiliaryFacilityPowerCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_auxiliary_facility_power_cost
    ported = calculate_auxiliary_facility_power_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCAF=1.5e6, lsa=4)]
    fuzz_bounds = {"UCAF": (1.0e5, 5.0e6)}
    fuzz_fixed = {"lsa": 4}


class TestElectricPlantEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_electric_plant_equipment_cost
    ported = calculate_electric_plant_equipment_cost

    samples = [
        legacy_sample("nominal", c241=10.0, c242=15.0, c243=8.0, c244=6.0, c245=2.0)
    ]
    fuzz_bounds = {
        "c241": (0.0, 100.0),
        "c242": (0.0, 100.0),
        "c243": (0.0, 100.0),
        "c244": (0.0, 100.0),
        "c245": (0.0, 100.0),
    }


class TestMiscPlantEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_misc_plant_equipment_cost
    ported = calculate_misc_plant_equipment_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", ucmisc=2.5e7, lsa=4)]
    fuzz_bounds = {"ucmisc": (1.0e6, 1.0e8)}
    fuzz_fixed = {"lsa": 4}


class TestHeatRejectionCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_heat_rejection_cost
    ported = calculate_heat_rejection_cost
    static_argnames = ("ireactor", "lsa")

    samples = [
        legacy_sample(
            "reactor",
            ireactor=1,
            p_fusion_total_mw=2000.0,
            p_hcd_electric_total_mw=100.0,
            tfcmw=50.0,
            p_plant_primary_heat_mw=2200.0,
            p_plant_electric_gross_mw=1200.0,
            uchrs=1.0e7,
            lsa=4,
        ),
        legacy_sample(
            "non-reactor",
            ireactor=0,
            p_fusion_total_mw=2000.0,
            p_hcd_electric_total_mw=100.0,
            tfcmw=50.0,
            p_plant_primary_heat_mw=2200.0,
            p_plant_electric_gross_mw=1200.0,
            uchrs=1.0e7,
            lsa=4,
        ),
    ]
    fuzz_bounds = {
        "p_fusion_total_mw": (100.0, 5000.0),
        "p_hcd_electric_total_mw": (1.0, 500.0),
        "tfcmw": (1.0, 200.0),
        "p_plant_primary_heat_mw": (100.0, 3000.0),
        "p_plant_electric_gross_mw": (100.0, 2000.0),
        "uchrs": (1.0e6, 5.0e7),
    }
    fuzz_fixed = {"ireactor": 1, "lsa": 4}


# --------------------------------------------------------------------------------------
# Second porting wave: the `.costs.coe` chain (18 further `Costs` methods plus the two
# accumulations `Costs.run()` performs inline). See `costs.md`'s coverage map.
#
# Several of these methods are *accumulators* that call their own sub-accounts before
# summing (`acc221`, `acc222`, `acc225`, `acc22`) or live inline in `Costs.run()`
# (`cdirt`, `concost`). Their reference wrappers neutralise those sub-calls with
# instance-level no-ops so the check exercises exactly the accumulation the node ports
# and nothing else -- the sub-accounts have their own contracts right here.
# --------------------------------------------------------------------------------------

_NO_ARG_METHODS_RUN_CALLS = (
    "convert_fpy_to_calendar",
    "acc21",
    "acc22",
    "acc23",
    "acc241",
    "acc242",
    "acc243",
    "acc244",
    "acc245",
    "acc24",
    "acc25",
    "acc26",
    "acc9",
    "coelc",
)


def _neutralise(costs, *names):
    """Replace named `Costs` methods with instance-level no-ops."""
    for name in names:
        setattr(costs, name, lambda: None)


def _reference_first_wall_cost(
    ife, lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp
):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.lsa = lsa
    costs.data.costs.UCFWA = UCFWA
    costs.data.costs.UCFWS = UCFWS
    costs.data.first_wall.a_fw_total = a_fw_total
    costs.data.costs.UCFWPS = UCFWPS
    costs.data.costs.fkind = fkind
    costs.data.costs.ifueltyp = ifueltyp
    costs.acc2211()
    return costs.data.costs.c2211, costs.data.costs.fwallcst


def _reference_blanket_cost(
    ife,
    lsa,
    m_blkt_beryllium,
    ucblbe,
    m_blkt_li2o,
    ucblli2o,
    m_blkt_steel_total,
    ucblss,
    m_blkt_vanadium,
    ucblvd,
    fkind,
    ifueltyp,
):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.lsa = lsa
    costs.data.fwbs.m_blkt_beryllium = m_blkt_beryllium
    costs.data.costs.ucblbe = ucblbe
    costs.data.fwbs.m_blkt_li2o = m_blkt_li2o
    costs.data.costs.ucblli2o = ucblli2o
    costs.data.fwbs.m_blkt_steel_total = m_blkt_steel_total
    costs.data.costs.ucblss = ucblss
    costs.data.fwbs.m_blkt_vanadium = m_blkt_vanadium
    costs.data.costs.ucblvd = ucblvd
    costs.data.costs.fkind = fkind
    costs.data.costs.ifueltyp = ifueltyp
    costs.acc2212()
    c = costs.data.costs
    return (
        c.c22121,
        c.c22122,
        c.c22123,
        c.c22124,
        c.c22125,
        c.c22126,
        c.c22127,
        c.c2212,
        c.blkcst,
    )


def _reference_shield_cost(ife, lsa, whtshld, ucshld, wpenshld, ucpens, fkind):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.lsa = lsa
    costs.data.fwbs.whtshld = whtshld
    costs.data.costs.ucshld = ucshld
    costs.data.fwbs.wpenshld = wpenshld
    costs.data.costs.ucpens = ucpens
    costs.data.costs.fkind = fkind
    costs.acc2213()
    c = costs.data.costs
    return c.c22131, c.c22132, c.c2213


def _reference_reactor_cost(c2211, c2212, c2213, c2214, c2215):
    costs = _make_costs()
    costs.data.costs.c2211 = c2211
    costs.data.costs.c2212 = c2212
    costs.data.costs.c2213 = c2213
    costs.data.costs.c2214 = c2214
    costs.data.costs.c2215 = c2215
    _neutralise(costs, "acc2211", "acc2212", "acc2213", "acc2214", "acc2215")
    costs.acc221()
    return costs.data.costs.c221


def _reference_tf_magnet_cost_superconducting(
    supercond_cost_model,
    lsa,
    ucsc,
    i_tf_sc_mat,
    m_tf_coil_superconductor,
    len_tf_coil,
    n_tf_coil_turns,
    sc_mat_cost_0,
    j_crit_str_0,
    j_crit_str_tf,
    uccu,
    m_tf_coil_copper,
    cconshtf,
    cconfix,
    n_tf_coils,
    ucwindtf,
    m_tf_coil_case,
    uccase,
    aintmass,
    UCINT,
    clgsmass,
    UCGSS,
    fkind,
):
    costs = _make_costs()
    costs.data.tfcoil.i_tf_sup = 1  # TFConductorModel.SUPERCONDUCTING
    costs.data.costs.supercond_cost_model = supercond_cost_model
    costs.data.costs.lsa = lsa
    costs.data.costs.ucsc = ucsc
    costs.data.tfcoil.i_tf_sc_mat = i_tf_sc_mat
    costs.data.tfcoil.m_tf_coil_superconductor = m_tf_coil_superconductor
    costs.data.tfcoil.len_tf_coil = len_tf_coil
    costs.data.tfcoil.n_tf_coil_turns = n_tf_coil_turns
    costs.data.costs.sc_mat_cost_0 = sc_mat_cost_0
    costs.data.tfcoil.j_crit_str_0 = j_crit_str_0
    costs.data.tfcoil.j_crit_str_tf = j_crit_str_tf
    costs.data.costs.uccu = uccu
    costs.data.tfcoil.m_tf_coil_copper = m_tf_coil_copper
    costs.data.costs.cconshtf = cconshtf
    costs.data.costs.cconfix = cconfix
    costs.data.tfcoil.n_tf_coils = n_tf_coils
    costs.data.costs.ucwindtf = ucwindtf
    costs.data.tfcoil.m_tf_coil_case = m_tf_coil_case
    costs.data.costs.uccase = uccase
    costs.data.structure.aintmass = aintmass
    costs.data.costs.UCINT = UCINT
    costs.data.structure.clgsmass = clgsmass
    costs.data.costs.UCGSS = UCGSS
    costs.data.costs.fkind = fkind
    costs.acc2221()
    c = costs.data.costs
    return c.c22211, c.c22212, c.c22213, c.c22214, c.c22215, c.c2221


def _reference_tf_magnet_cost_resistive(
    lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
):
    costs = _make_costs()
    costs.data.tfcoil.i_tf_sup = 0  # resistive copper
    costs.data.costs.lsa = lsa
    costs.data.tfcoil.whtcp = whtcp
    costs.data.costs.uccpcl1 = uccpcl1
    costs.data.tfcoil.whttflgs = whttflgs
    costs.data.costs.uccpclb = uccpclb
    costs.data.physics.itart = itart
    costs.data.costs.ifueltyp = ifueltyp
    costs.data.costs.fkind = fkind
    costs.acc2221()
    c = costs.data.costs
    return c.c22211, c.c22212, c.c2221, c.cpstcst


def _reference_pf_magnet_cost(
    n_cs_pf_coils,
    iohcl,
    i_pf_conductor,
    supercond_cost_model,
    lsa,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    ucsc,
    i_pf_superconductor,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    dcond,
    sc_mat_cost_0,
    j_crit_str_0,
    j_crit_str_pf,
    uccu,
    cconfix,
    i_cs_superconductor,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    j_crit_str_cs,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
):
    costs = _make_costs()
    costs.data.pf_coil.n_cs_pf_coils = n_cs_pf_coils
    costs.data.build.iohcl = iohcl
    costs.data.pf_coil.i_pf_conductor = i_pf_conductor
    costs.data.costs.supercond_cost_model = supercond_cost_model
    costs.data.costs.lsa = lsa
    costs.data.pf_coil.r_pf_coil_middle = np.asarray(r_pf_coil_middle, dtype=float)
    costs.data.pf_coil.n_pf_coil_turns = np.asarray(n_pf_coil_turns, dtype=float)
    costs.data.costs.cconshpf = cconshpf
    costs.data.costs.ucsc = ucsc
    costs.data.pf_coil.i_pf_superconductor = i_pf_superconductor
    costs.data.pf_coil.fcupfsu = fcupfsu
    costs.data.pf_coil.f_a_pf_coil_void = np.asarray(f_a_pf_coil_void, dtype=float)
    costs.data.pf_coil.c_pf_cs_coils_peak_ma = np.asarray(
        c_pf_cs_coils_peak_ma, dtype=float
    )
    costs.data.pf_coil.j_pf_coil_wp_peak = np.asarray(j_pf_coil_wp_peak, dtype=float)
    costs.data.tfcoil.dcond = dcond
    costs.data.costs.sc_mat_cost_0 = sc_mat_cost_0
    costs.data.tfcoil.j_crit_str_0 = j_crit_str_0
    costs.data.pf_coil.j_crit_str_pf = j_crit_str_pf
    costs.data.costs.uccu = uccu
    costs.data.costs.cconfix = cconfix
    costs.data.pf_coil.i_cs_superconductor = i_cs_superconductor
    costs.data.pf_coil.a_cs_cable_space = a_cs_cable_space
    costs.data.pf_coil.f_a_cs_void = f_a_cs_void
    costs.data.pf_coil.fcuohsu = fcuohsu
    costs.data.pf_coil.j_crit_str_cs = j_crit_str_cs
    costs.data.costs.ucwindpf = ucwindpf
    costs.data.costs.uccase = uccase
    costs.data.pf_coil.m_pf_coil_structure_total = m_pf_coil_structure_total
    costs.data.costs.ucfnc = ucfnc
    costs.data.structure.fncmass = fncmass
    costs.data.costs.fkind = fkind
    costs.acc2222()
    c = costs.data.costs
    return c.c22221, c.c22222, c.c22223, c.c22224, c.c2222


def _reference_magnets_cost(ife, c2221, c2222, c2223):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.c2221 = c2221
    costs.data.costs.c2222 = c2222
    costs.data.costs.c2223 = c2223
    _neutralise(costs, "acc2221", "acc2222", "acc2223")
    costs.acc222()
    return costs.data.costs.c222


def _reference_power_injection_cost(
    ife,
    ucech,
    p_hcd_ecrh_injected_total_mw,
    i_hcd_primary,
    uclh,
    ucich,
    p_hcd_lowhyb_injected_total_mw,
    ucnbi,
    p_beam_injected_mw,
    ifueltyp,
    fcdfuel,
    fkind,
):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.ucech = ucech
    costs.data.current_drive.p_hcd_ecrh_injected_total_mw = p_hcd_ecrh_injected_total_mw
    costs.data.current_drive.i_hcd_primary = i_hcd_primary
    costs.data.costs.uclh = uclh
    costs.data.costs.ucich = ucich
    costs.data.current_drive.p_hcd_lowhyb_injected_total_mw = (
        p_hcd_lowhyb_injected_total_mw
    )
    costs.data.costs.ucnbi = ucnbi
    costs.data.current_drive.p_beam_injected_mw = p_beam_injected_mw
    costs.data.costs.ifueltyp = ifueltyp
    costs.data.costs.fcdfuel = fcdfuel
    costs.data.costs.fkind = fkind
    costs.acc223()
    c = costs.data.costs
    return c.c2231, c.c2232, c.c2233, c.c223, c.cdcost


def _reference_energy_storage_cost(
    i_pulsed_plant, istore, p_plant_electric_net_mw, fkind
):
    costs = _make_costs()
    costs.data.pulse.i_pulsed_plant = i_pulsed_plant
    costs.data.pulse.istore = istore
    costs.data.heat_transport.p_plant_electric_net_mw = p_plant_electric_net_mw
    costs.data.costs.fkind = fkind
    costs.acc2253()
    return costs.data.costs.c2253


def _reference_power_conditioning_cost(ife, c2251, c2252, c2253):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.c2251 = c2251
    costs.data.costs.c2252 = c2252
    costs.data.costs.c2253 = c2253
    _neutralise(costs, "acc2251", "acc2252", "acc2253")
    costs.acc225()
    return costs.data.costs.c225


def _reference_auxiliary_component_cooling_cost(
    ife,
    lsa,
    UCAHTS,
    p_hcd_electric_loss_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_tritium_plant_electric_mw,
    fachtmw,
    fkind,
):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.lsa = lsa
    costs.data.costs.UCAHTS = UCAHTS
    costs.data.heat_transport.p_hcd_electric_loss_mw = p_hcd_electric_loss_mw
    costs.data.heat_transport.p_cryo_plant_electric_mw = p_cryo_plant_electric_mw
    costs.data.heat_transport.vachtmw = vachtmw
    costs.data.heat_transport.p_tritium_plant_electric_mw = p_tritium_plant_electric_mw
    costs.data.heat_transport.fachtmw = fachtmw
    costs.data.costs.fkind = fkind
    costs.acc2262()
    return costs.data.costs.cppa, costs.data.costs.c2262


def _reference_cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind):
    costs = _make_costs()
    costs.data.costs.lsa = lsa
    costs.data.costs.uccry = uccry
    costs.data.tfcoil.temp_tf_cryo = temp_tf_cryo
    costs.data.heat_transport.helpow = helpow
    costs.data.costs.fkind = fkind
    costs.acc2263()
    return costs.data.costs.c2263


def _reference_heat_transport_system_cost(c2261, c2262, c2263):
    costs = _make_costs()
    costs.data.costs.c2261 = c2261
    costs.data.costs.c2262 = c2262
    costs.data.costs.c2263 = c2263
    costs.acc226()
    return costs.data.costs.c226


def _reference_fuel_processing_cost(ife, rndfuel, m_fuel_amu, UCFPR, fkind):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.physics.rndfuel = rndfuel
    costs.data.physics.m_fuel_amu = m_fuel_amu
    costs.data.costs.UCFPR = UCFPR
    costs.data.costs.fkind = fkind
    costs.acc2272()
    return costs.data.physics.wtgpd, costs.data.costs.c2272


def _reference_atmospheric_recovery_cost(
    f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind
):
    costs = _make_costs()
    costs.data.physics.f_plasma_fuel_tritium = f_plasma_fuel_tritium
    costs.data.costs.UCDTC = UCDTC
    costs.data.buildings.volrci = volrci
    costs.data.buildings.wsvol = wsvol
    costs.data.costs.fkind = fkind
    costs.acc2273()
    return costs.data.costs.c2273


def _reference_fuel_handling_cost(c2271, c2272, c2273, c2274):
    costs = _make_costs()
    costs.data.costs.c2271 = c2271
    costs.data.costs.c2272 = c2272
    costs.data.costs.c2273 = c2273
    costs.data.costs.c2274 = c2274
    costs.acc227()
    return costs.data.costs.c227


def _reference_fusion_power_island_cost(
    c221, c222, c223, c224, c225, c226, c227, c228, c229
):
    costs = _make_costs()
    for name, value in zip(
        ("c221", "c222", "c223", "c224", "c225", "c226", "c227", "c228", "c229"),
        (c221, c222, c223, c224, c225, c226, c227, c228, c229),
        strict=True,
    ):
        setattr(costs.data.costs, name, value)
    _neutralise(
        costs,
        "acc221",
        "acc222",
        "acc223",
        "acc224",
        "acc225",
        "acc2261",
        "acc2262",
        "acc2263",
        "acc226",
        "acc2271",
        "acc2272",
        "acc2273",
        "acc2274",
        "acc227",
        "acc228",
        "acc229",
    )
    costs.acc22()
    return costs.data.costs.crctcore, costs.data.costs.c22


def _reference_total_plant_direct_cost(c21, c22, c23, c24, c25, c26):
    costs = _make_costs()
    for name, value in zip(
        ("c21", "c22", "c23", "c24", "c25", "c26"),
        (c21, c22, c23, c24, c25, c26),
        strict=True,
    ):
        setattr(costs.data.costs, name, value)
    _neutralise(costs, *_NO_ARG_METHODS_RUN_CALLS)
    costs.run()
    return costs.data.costs.cdirt


def _reference_constructed_cost(cdirt, cindrt, ccont):
    """`concost = cdirt + cindrt + ccont` (`costs.py:77-79`).

    `Costs.run()` recomputes `cdirt` from `c21`..`c26` immediately before this line, so
    the requested `cdirt` is fed in through `c21` with the other five accounts zeroed --
    the accumulation itself is checked by `_reference_total_plant_direct_cost`.
    """
    costs = _make_costs()
    costs.data.costs.c21 = cdirt
    for name in ("c22", "c23", "c24", "c25", "c26"):
        setattr(costs.data.costs, name, 0.0)
    costs.data.costs.cindrt = cindrt
    costs.data.costs.ccont = ccont
    _neutralise(costs, *_NO_ARG_METHODS_RUN_CALLS)
    costs.run()
    return costs.data.costs.concost


def _reference_cost_of_electricity(
    ife,
    itart,
    p_plant_electric_net_mw,
    f_t_plant_available,
    t_plant_pulse_burn,
    t_plant_pulse_total,
    concost,
    fcap0,
    fcr0,
    discount_rate,
    life_blkt,
    fwallcst,
    blkcst,
    cfind,
    lsa,
    fcap0cp,
    ifueltyp,
    life_blkt_fpy,
    life_plant,
    life_div,
    divcst,
    life_div_fpy,
    cplife_cal,
    cpstcst,
    cplife,
    cdrlife_cal,
    cdcost,
    fcdfuel,
    ucoam,
    ucfuel,
    f_plasma_fuel_helium3,
    wtgpd,
    uche3,
    ucwst,
    decomf,
    dintrt,
    dtlife,
):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.physics.itart = itart
    costs.data.heat_transport.p_plant_electric_net_mw = p_plant_electric_net_mw
    costs.data.costs.f_t_plant_available = f_t_plant_available
    costs.data.times.t_plant_pulse_burn = t_plant_pulse_burn
    costs.data.times.t_plant_pulse_total = t_plant_pulse_total
    costs.data.costs.concost = concost
    costs.data.costs.fcap0 = fcap0
    costs.data.costs.fcr0 = fcr0
    costs.data.costs.discount_rate = discount_rate
    costs.data.fwbs.life_blkt = life_blkt
    costs.data.costs.fwallcst = fwallcst
    costs.data.costs.blkcst = blkcst
    costs.data.costs.cfind = cfind
    costs.data.costs.lsa = lsa
    costs.data.costs.fcap0cp = fcap0cp
    costs.data.costs.ifueltyp = ifueltyp
    costs.data.fwbs.life_blkt_fpy = life_blkt_fpy
    costs.data.costs.life_plant = life_plant
    costs.data.costs.life_div = life_div
    costs.data.costs.divcst = divcst
    costs.data.costs.life_div_fpy = life_div_fpy
    costs.data.costs.cplife_cal = cplife_cal
    costs.data.costs.cpstcst = cpstcst
    costs.data.costs.cplife = cplife
    costs.data.costs.cdrlife_cal = cdrlife_cal
    costs.data.costs.cdcost = cdcost
    costs.data.costs.fcdfuel = fcdfuel
    costs.data.costs.ucoam = ucoam
    costs.data.costs.ucfuel = ucfuel
    costs.data.physics.f_plasma_fuel_helium3 = f_plasma_fuel_helium3
    costs.data.physics.wtgpd = wtgpd
    costs.data.costs.uche3 = uche3
    costs.data.costs.ucwst = ucwst
    costs.data.costs.decomf = decomf
    costs.data.costs.dintrt = dintrt
    costs.data.costs.dtlife = dtlife
    costs.coelc()
    c = costs.data.costs
    return c.moneyint, c.capcost, c.coecap, c.coeoam, c.coefuelt, c.coe


# `.costs.cfind`/`ucoam`/`ucwst` defaults, `cost_variables.py`; `ucsc`/`sc_mat_cost_0`/
# `dcond`/`j_crit_str_0` are the length-9 material tables, values as on the reference
# converged run (`costs.md` § coverage map).
_CFIND = [0.244, 0.244, 0.244, 0.29]
_UCOAM = [68.8, 68.8, 68.8, 74.4]
_UCWST = [0.0, 3.94, 5.91, 7.88]
_UCSC = [600.0, 600.0, 300.0, 600.0, 600.0, 600.0, 300.0, 1200.0, 1200.0]
_SC_MAT_COST_0 = [4.8, 2.0, 1.0, 4.8, 4.8, 47.4, 1.0, 47.4, 47.4]
_DCOND = [6080.0] * 9
_J_CRIT_STR_0 = [
    5.96905476e8,
    1.92550153e9,
    7.24544683e8,
    5.49858624e8,
    6.69284510e8,
    1.0e8,
    8.98964415e8,
    1.15875300e9,
    8.65652123e8,
]


class TestFirstWallCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_first_wall_cost
    ported = calculate_first_wall_cost
    static_argnames = ("ife", "lsa", "ifueltyp")

    samples = [
        legacy_sample(
            "capital",
            ife=0,
            lsa=2,
            UCFWA=6.0e4,
            UCFWS=5.3e4,
            a_fw_total=3182.3,
            UCFWPS=1.0e7,
            fkind=1.0,
            ifueltyp=0,
        ),
        legacy_sample(
            "fuel",
            ife=0,
            lsa=2,
            UCFWA=6.0e4,
            UCFWS=5.3e4,
            a_fw_total=3182.3,
            UCFWPS=1.0e7,
            fkind=1.0,
            ifueltyp=1,
        ),
        legacy_sample(
            "capital-plus-replacement",
            ife=0,
            lsa=4,
            UCFWA=6.0e4,
            UCFWS=5.3e4,
            a_fw_total=3182.3,
            UCFWPS=1.0e7,
            fkind=1.0,
            ifueltyp=2,
        ),
    ]
    fuzz_bounds = {
        "UCFWA": (1.0e4, 1.0e5),
        "UCFWS": (1.0e4, 1.0e5),
        "a_fw_total": (100.0, 5000.0),
        "UCFWPS": (1.0e6, 1.0e8),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "lsa": 2, "ifueltyp": 0}


class TestBlanketCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_blanket_cost
    ported = calculate_blanket_cost
    static_argnames = ("ife", "lsa", "ifueltyp")

    samples = [
        legacy_sample(
            "capital",
            ife=0,
            lsa=2,
            m_blkt_beryllium=1.13e6,
            ucblbe=260.0,
            m_blkt_li2o=5.0e5,
            ucblli2o=600.0,
            m_blkt_steel_total=1.28e6,
            ucblss=90.0,
            m_blkt_vanadium=0.0,
            ucblvd=280.0,
            fkind=1.0,
            ifueltyp=0,
        ),
        legacy_sample(
            "fuel",
            ife=0,
            lsa=2,
            m_blkt_beryllium=1.13e6,
            ucblbe=260.0,
            m_blkt_li2o=5.0e5,
            ucblli2o=600.0,
            m_blkt_steel_total=1.28e6,
            ucblss=90.0,
            m_blkt_vanadium=1.0e5,
            ucblvd=280.0,
            fkind=1.0,
            ifueltyp=1,
        ),
        legacy_sample(
            "capital-plus-replacement",
            ife=0,
            lsa=4,
            m_blkt_beryllium=1.13e6,
            ucblbe=260.0,
            m_blkt_li2o=5.0e5,
            ucblli2o=600.0,
            m_blkt_steel_total=1.28e6,
            ucblss=90.0,
            m_blkt_vanadium=1.0e5,
            ucblvd=280.0,
            fkind=1.0,
            ifueltyp=2,
        ),
    ]
    fuzz_bounds = {
        "m_blkt_beryllium": (0.0, 5.0e6),
        "ucblbe": (100.0, 500.0),
        "m_blkt_li2o": (0.0, 5.0e6),
        "ucblli2o": (100.0, 1000.0),
        "m_blkt_steel_total": (0.0, 5.0e6),
        "ucblss": (50.0, 200.0),
        "m_blkt_vanadium": (0.0, 5.0e5),
        "ucblvd": (100.0, 500.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "lsa": 2, "ifueltyp": 0}


class TestShieldCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_shield_cost
    ported = calculate_shield_cost
    static_argnames = ("ife", "lsa")

    samples = [
        legacy_sample(
            "nominal",
            ife=0,
            lsa=2,
            whtshld=4.53e6,
            ucshld=32.0,
            wpenshld=4.53e6,
            ucpens=32.0,
            fkind=1.0,
        )
    ]
    fuzz_bounds = {
        "whtshld": (1.0e5, 1.0e7),
        "ucshld": (10.0, 100.0),
        "wpenshld": (1.0e5, 1.0e7),
        "ucpens": (10.0, 100.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "lsa": 2}


class TestReactorCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_reactor_cost
    ported = calculate_reactor_cost

    samples = [
        legacy_sample(
            "nominal", c2211=277.2, c2212=531.5, c2213=217.3, c2214=0.0, c2215=22.6
        )
    ]
    fuzz_bounds = {
        "c2211": (0.0, 1000.0),
        "c2212": (0.0, 1000.0),
        "c2213": (0.0, 1000.0),
        "c2214": (0.0, 1000.0),
        "c2215": (0.0, 1000.0),
    }


class TestTfMagnetCostSuperconducting(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_tf_magnet_cost_superconducting
    ported = calculate_tf_magnet_cost_superconducting
    static_argnames = ("supercond_cost_model", "lsa", "i_tf_sc_mat")

    samples = [
        legacy_sample(
            "legacy-cost-model",
            supercond_cost_model=0,
            lsa=2,
            ucsc=_UCSC,
            i_tf_sc_mat=1,
            m_tf_coil_superconductor=1.0e5,
            len_tf_coil=50.0,
            n_tf_coil_turns=200.0,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_tf=6.0e8,
            uccu=75.0,
            m_tf_coil_copper=2.0e5,
            cconshtf=75.0,
            cconfix=80.0,
            n_tf_coils=50.0,
            ucwindtf=480.0,
            m_tf_coil_case=5.0e5,
            uccase=50.0,
            aintmass=5.3e6,
            UCINT=30.0,
            clgsmass=1.06e6,
            UCGSS=35.0,
            fkind=1.0,
        ),
        legacy_sample(
            "strand-cost-model",
            supercond_cost_model=1,
            lsa=4,
            ucsc=_UCSC,
            i_tf_sc_mat=1,
            m_tf_coil_superconductor=1.0e5,
            len_tf_coil=50.0,
            n_tf_coil_turns=200.0,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_tf=6.0e8,
            uccu=75.0,
            m_tf_coil_copper=2.0e5,
            cconshtf=75.0,
            cconfix=80.0,
            n_tf_coils=50.0,
            ucwindtf=480.0,
            m_tf_coil_case=5.0e5,
            uccase=50.0,
            aintmass=5.3e6,
            UCINT=30.0,
            clgsmass=1.06e6,
            UCGSS=35.0,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "m_tf_coil_superconductor": (1.0e4, 1.0e6),
        "len_tf_coil": (10.0, 200.0),
        "n_tf_coil_turns": (10.0, 500.0),
        "j_crit_str_tf": (1.0e8, 1.0e9),
        "uccu": (10.0, 200.0),
        "m_tf_coil_copper": (1.0e4, 1.0e6),
        "cconshtf": (10.0, 200.0),
        "cconfix": (10.0, 200.0),
        "n_tf_coils": (10.0, 60.0),
        "ucwindtf": (100.0, 1000.0),
        "m_tf_coil_case": (1.0e4, 1.0e6),
        "uccase": (10.0, 200.0),
        "aintmass": (1.0e5, 1.0e7),
        "UCINT": (10.0, 100.0),
        "clgsmass": (1.0e5, 1.0e7),
        "UCGSS": (10.0, 100.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {
        "supercond_cost_model": 0,
        "lsa": 2,
        "ucsc": _UCSC,
        "i_tf_sc_mat": 1,
        "sc_mat_cost_0": _SC_MAT_COST_0,
        "j_crit_str_0": _J_CRIT_STR_0,
    }


class TestTfMagnetCostResistive(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_tf_magnet_cost_resistive
    ported = calculate_tf_magnet_cost_resistive
    static_argnames = ("lsa", "itart", "ifueltyp")

    samples = [
        legacy_sample(
            "conventional",
            lsa=2,
            whtcp=1.0e6,
            uccpcl1=250.0,
            whttflgs=2.0e6,
            uccpclb=150.0,
            itart=0,
            ifueltyp=0,
            fkind=1.0,
        ),
        legacy_sample(
            "tart-fuel",
            lsa=2,
            whtcp=1.0e6,
            uccpcl1=250.0,
            whttflgs=2.0e6,
            uccpclb=150.0,
            itart=1,
            ifueltyp=1,
            fkind=1.0,
        ),
        legacy_sample(
            "tart-capital-plus-replacement",
            lsa=4,
            whtcp=1.0e6,
            uccpcl1=250.0,
            whttflgs=2.0e6,
            uccpclb=150.0,
            itart=1,
            ifueltyp=2,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "whtcp": (1.0e4, 1.0e7),
        "uccpcl1": (50.0, 500.0),
        "whttflgs": (1.0e4, 1.0e7),
        "uccpclb": (50.0, 500.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"lsa": 2, "itart": 0, "ifueltyp": 0}


_PF_R = [5.0, 6.0, 7.0, 8.0]
_PF_TURNS = [100.0, 120.0, 140.0, 160.0]
_PF_VOID = [0.3, 0.3, 0.3, 0.3]
_PF_CURRENT = [10.0, -12.0, 14.0, 16.0]
_PF_J = [1.0e7, 1.1e7, 1.2e7, 1.3e7]


class TestPfMagnetCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_pf_magnet_cost
    ported = calculate_pf_magnet_cost
    static_argnames = (
        "n_cs_pf_coils",
        "iohcl",
        "i_pf_conductor",
        "supercond_cost_model",
        "lsa",
        "i_pf_superconductor",
        "i_cs_superconductor",
    )

    samples = [
        legacy_sample(
            "stellarator-no-pf-coils",
            n_cs_pf_coils=0,
            iohcl=0,
            i_pf_conductor=0,
            supercond_cost_model=0,
            lsa=2,
            r_pf_coil_middle=_PF_R,
            n_pf_coil_turns=_PF_TURNS,
            cconshpf=70.0,
            ucsc=_UCSC,
            i_pf_superconductor=1,
            fcupfsu=0.69,
            f_a_pf_coil_void=_PF_VOID,
            c_pf_cs_coils_peak_ma=_PF_CURRENT,
            j_pf_coil_wp_peak=_PF_J,
            dcond=_DCOND,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_pf=6.0e8,
            uccu=75.0,
            cconfix=80.0,
            i_cs_superconductor=1,
            a_cs_cable_space=0.1,
            f_a_cs_void=0.3,
            fcuohsu=0.7,
            j_crit_str_cs=6.0e8,
            ucwindpf=465.0,
            uccase=50.0,
            m_pf_coil_structure_total=0.0,
            ucfnc=35.0,
            fncmass=0.0,
            fkind=1.0,
        ),
        legacy_sample(
            "superconducting-with-cs",
            n_cs_pf_coils=4,
            iohcl=1,
            i_pf_conductor=0,
            supercond_cost_model=0,
            lsa=2,
            r_pf_coil_middle=_PF_R,
            n_pf_coil_turns=_PF_TURNS,
            cconshpf=70.0,
            ucsc=_UCSC,
            i_pf_superconductor=1,
            fcupfsu=0.69,
            f_a_pf_coil_void=_PF_VOID,
            c_pf_cs_coils_peak_ma=_PF_CURRENT,
            j_pf_coil_wp_peak=_PF_J,
            dcond=_DCOND,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_pf=6.0e8,
            uccu=75.0,
            cconfix=80.0,
            i_cs_superconductor=1,
            a_cs_cable_space=0.1,
            f_a_cs_void=0.3,
            fcuohsu=0.7,
            j_crit_str_cs=6.0e8,
            ucwindpf=465.0,
            uccase=50.0,
            m_pf_coil_structure_total=1.0e6,
            ucfnc=35.0,
            fncmass=5.0e5,
            fkind=1.0,
        ),
        legacy_sample(
            "superconducting-strand-cost-model",
            n_cs_pf_coils=4,
            iohcl=1,
            i_pf_conductor=0,
            supercond_cost_model=1,
            lsa=4,
            r_pf_coil_middle=_PF_R,
            n_pf_coil_turns=_PF_TURNS,
            cconshpf=70.0,
            ucsc=_UCSC,
            i_pf_superconductor=1,
            fcupfsu=0.69,
            f_a_pf_coil_void=_PF_VOID,
            c_pf_cs_coils_peak_ma=_PF_CURRENT,
            j_pf_coil_wp_peak=_PF_J,
            dcond=_DCOND,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_pf=6.0e8,
            uccu=75.0,
            cconfix=80.0,
            i_cs_superconductor=1,
            a_cs_cable_space=0.1,
            f_a_cs_void=0.3,
            fcuohsu=0.7,
            j_crit_str_cs=6.0e8,
            ucwindpf=465.0,
            uccase=50.0,
            m_pf_coil_structure_total=1.0e6,
            ucfnc=35.0,
            fncmass=5.0e5,
            fkind=1.0,
        ),
        legacy_sample(
            "resistive-no-cs",
            n_cs_pf_coils=3,
            iohcl=0,
            i_pf_conductor=1,
            supercond_cost_model=0,
            lsa=2,
            r_pf_coil_middle=_PF_R,
            n_pf_coil_turns=_PF_TURNS,
            cconshpf=70.0,
            ucsc=_UCSC,
            i_pf_superconductor=1,
            fcupfsu=0.69,
            f_a_pf_coil_void=_PF_VOID,
            c_pf_cs_coils_peak_ma=_PF_CURRENT,
            j_pf_coil_wp_peak=_PF_J,
            dcond=_DCOND,
            sc_mat_cost_0=_SC_MAT_COST_0,
            j_crit_str_0=_J_CRIT_STR_0,
            j_crit_str_pf=6.0e8,
            uccu=75.0,
            cconfix=80.0,
            i_cs_superconductor=1,
            a_cs_cable_space=0.1,
            f_a_cs_void=0.3,
            fcuohsu=0.7,
            j_crit_str_cs=6.0e8,
            ucwindpf=465.0,
            uccase=50.0,
            m_pf_coil_structure_total=1.0e6,
            ucfnc=35.0,
            fncmass=5.0e5,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "cconshpf": (10.0, 200.0),
        "fcupfsu": (0.1, 0.9),
        "j_crit_str_pf": (1.0e8, 1.0e9),
        "uccu": (10.0, 200.0),
        "cconfix": (10.0, 200.0),
        "a_cs_cable_space": (0.01, 1.0),
        "f_a_cs_void": (0.1, 0.5),
        "fcuohsu": (0.1, 0.9),
        "j_crit_str_cs": (1.0e8, 1.0e9),
        "ucwindpf": (100.0, 1000.0),
        "uccase": (10.0, 200.0),
        "m_pf_coil_structure_total": (0.0, 5.0e6),
        "ucfnc": (10.0, 100.0),
        "fncmass": (0.0, 5.0e6),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {
        "n_cs_pf_coils": 4,
        "iohcl": 1,
        "i_pf_conductor": 0,
        "supercond_cost_model": 0,
        "lsa": 2,
        "r_pf_coil_middle": _PF_R,
        "n_pf_coil_turns": _PF_TURNS,
        "ucsc": _UCSC,
        "i_pf_superconductor": 1,
        "f_a_pf_coil_void": _PF_VOID,
        "c_pf_cs_coils_peak_ma": _PF_CURRENT,
        "j_pf_coil_wp_peak": _PF_J,
        "dcond": _DCOND,
        "sc_mat_cost_0": _SC_MAT_COST_0,
        "j_crit_str_0": _J_CRIT_STR_0,
        "i_cs_superconductor": 1,
    }


class TestMagnetsCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_magnets_cost
    ported = calculate_magnets_cost
    static_argnames = ("ife",)

    samples = [
        legacy_sample("magnetic", ife=0, c2221=989.5, c2222=0.0, c2223=952.2),
        legacy_sample("ife", ife=1, c2221=989.5, c2222=0.0, c2223=952.2),
    ]
    fuzz_bounds = {
        "c2221": (0.0, 5000.0),
        "c2222": (0.0, 5000.0),
        "c2223": (0.0, 5000.0),
    }
    fuzz_fixed = {"ife": 0}


class TestPowerInjectionCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_power_injection_cost
    ported = calculate_power_injection_cost
    static_argnames = ("ife", "i_hcd_primary", "ifueltyp")

    samples = [
        legacy_sample(
            "capital-lower-hybrid",
            ife=0,
            ucech=3.0,
            p_hcd_ecrh_injected_total_mw=50.0,
            i_hcd_primary=5,
            uclh=3.3,
            ucich=3.0,
            p_hcd_lowhyb_injected_total_mw=30.0,
            ucnbi=3.3,
            p_beam_injected_mw=20.0,
            ifueltyp=0,
            fcdfuel=0.1,
            fkind=1.0,
        ),
        legacy_sample(
            "fuel-ich",
            ife=0,
            ucech=3.0,
            p_hcd_ecrh_injected_total_mw=50.0,
            i_hcd_primary=2,
            uclh=3.3,
            ucich=3.0,
            p_hcd_lowhyb_injected_total_mw=30.0,
            ucnbi=3.3,
            p_beam_injected_mw=20.0,
            ifueltyp=1,
            fcdfuel=0.1,
            fkind=1.0,
        ),
        legacy_sample(
            "capital-ifueltyp-2-leaves-c2233-unwritten",
            ife=0,
            ucech=3.0,
            p_hcd_ecrh_injected_total_mw=50.0,
            i_hcd_primary=5,
            uclh=3.3,
            ucich=3.0,
            p_hcd_lowhyb_injected_total_mw=30.0,
            ucnbi=3.3,
            p_beam_injected_mw=20.0,
            ifueltyp=2,
            fcdfuel=0.1,
            fkind=0.8,
        ),
    ]
    fuzz_bounds = {
        "ucech": (0.5, 10.0),
        "p_hcd_ecrh_injected_total_mw": (0.0, 200.0),
        "uclh": (0.5, 10.0),
        "ucich": (0.5, 10.0),
        "p_hcd_lowhyb_injected_total_mw": (0.0, 200.0),
        "ucnbi": (0.5, 10.0),
        "p_beam_injected_mw": (0.0, 200.0),
        "fcdfuel": (0.01, 0.5),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "i_hcd_primary": 5, "ifueltyp": 0}


class TestEnergyStorageCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_energy_storage_cost
    ported = calculate_energy_storage_cost
    static_argnames = ("i_pulsed_plant", "istore")

    samples = [
        legacy_sample(
            "steady-state",
            i_pulsed_plant=0,
            istore=1,
            p_plant_electric_net_mw=1000.0,
            fkind=1.0,
        ),
        legacy_sample(
            "pulsed-option-1",
            i_pulsed_plant=1,
            istore=1,
            p_plant_electric_net_mw=1000.0,
            fkind=1.0,
        ),
        legacy_sample(
            "pulsed-option-2",
            i_pulsed_plant=1,
            istore=2,
            p_plant_electric_net_mw=1000.0,
            fkind=0.8,
        ),
    ]
    fuzz_bounds = {
        "p_plant_electric_net_mw": (100.0, 2000.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"i_pulsed_plant": 0, "istore": 1}


class TestPowerConditioningCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_power_conditioning_cost
    ported = calculate_power_conditioning_cost
    static_argnames = ("ife",)

    samples = [
        legacy_sample("magnetic", ife=0, c2251=330.8, c2252=0.0, c2253=0.0),
        legacy_sample("ife", ife=1, c2251=330.8, c2252=0.0, c2253=0.0),
    ]
    fuzz_bounds = {
        "c2251": (0.0, 1000.0),
        "c2252": (0.0, 1000.0),
        "c2253": (0.0, 1000.0),
    }
    fuzz_fixed = {"ife": 0}


class TestAuxiliaryComponentCoolingCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_auxiliary_component_cooling_cost
    ported = calculate_auxiliary_component_cooling_cost
    static_argnames = ("ife", "lsa")

    samples = [
        legacy_sample(
            "nominal",
            ife=0,
            lsa=2,
            UCAHTS=31.0,
            p_hcd_electric_loss_mw=10.0,
            p_cryo_plant_electric_mw=20.0,
            vachtmw=0.5,
            p_tritium_plant_electric_mw=15.0,
            fachtmw=60.0,
            fkind=1.0,
        )
    ]
    fuzz_bounds = {
        "UCAHTS": (1.0, 100.0),
        "p_hcd_electric_loss_mw": (0.0, 200.0),
        "p_cryo_plant_electric_mw": (0.0, 200.0),
        "vachtmw": (0.0, 20.0),
        "p_tritium_plant_electric_mw": (0.0, 100.0),
        "fachtmw": (0.0, 200.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0, "lsa": 2}


class TestCryogenicSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_cryogenic_system_cost
    ported = calculate_cryogenic_system_cost
    static_argnames = ("lsa",)

    samples = [
        legacy_sample(
            "nominal",
            lsa=2,
            uccry=93000.0,
            temp_tf_cryo=4.5,
            helpow=271064.2,
            fkind=1.0,
        )
    ]
    fuzz_bounds = {
        "uccry": (1.0e4, 5.0e5),
        "temp_tf_cryo": (1.0, 30.0),
        "helpow": (1.0e3, 1.0e6),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"lsa": 2}


class TestHeatTransportSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_heat_transport_system_cost
    ported = calculate_heat_transport_system_cost

    samples = [legacy_sample("nominal", c2261=145.8, c2262=21.1, c2263=284.3)]
    fuzz_bounds = {
        "c2261": (0.0, 1000.0),
        "c2262": (0.0, 1000.0),
        "c2263": (0.0, 1000.0),
    }


class TestFuelProcessingCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fuel_processing_cost
    ported = calculate_fuel_processing_cost
    static_argnames = ("ife",)

    samples = [
        legacy_sample(
            "nominal",
            ife=0,
            rndfuel=1.06e21,
            m_fuel_amu=2.5145,
            UCFPR=1.5e8,
            fkind=1.0,
        )
    ]
    fuzz_bounds = {
        "rndfuel": (1.0e19, 1.0e22),
        "m_fuel_amu": (2.0, 3.0),
        "UCFPR": (1.0e7, 1.0e9),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"ife": 0}


class TestAtmosphericRecoveryCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_atmospheric_recovery_cost
    ported = calculate_atmospheric_recovery_cost

    samples = [
        legacy_sample(
            "with-tritium",
            f_plasma_fuel_tritium=0.5,
            UCDTC=0.0,
            volrci=1.0e5,
            wsvol=1.0e4,
            fkind=1.0,
        ),
        legacy_sample(
            "d-he3-only",
            f_plasma_fuel_tritium=0.0,
            UCDTC=400.0,
            volrci=1.0e5,
            wsvol=1.0e4,
            fkind=1.0,
        ),
        legacy_sample(
            "nominal",
            f_plasma_fuel_tritium=0.5,
            UCDTC=400.0,
            volrci=1.0e5,
            wsvol=1.0e4,
            fkind=0.9,
        ),
    ]
    fuzz_bounds = {
        "f_plasma_fuel_tritium": (0.0, 1.0),
        "UCDTC": (1.0, 1000.0),
        "volrci": (1.0e4, 5.0e5),
        "wsvol": (1.0e3, 5.0e4),
        "fkind": (0.5, 1.0),
    }


class TestFuelHandlingCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fuel_handling_cost
    ported = calculate_fuel_handling_cost

    samples = [
        legacy_sample("nominal", c2271=22.3, c2272=143.0, c2273=162.0, c2274=157.2)
    ]
    fuzz_bounds = {
        "c2271": (0.0, 500.0),
        "c2272": (0.0, 500.0),
        "c2273": (0.0, 500.0),
        "c2274": (0.0, 500.0),
    }


class TestFusionPowerIslandCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fusion_power_island_cost
    ported = calculate_fusion_power_island_cost

    samples = [
        legacy_sample(
            "nominal",
            c221=1048.6,
            c222=1941.7,
            c223=0.0,
            c224=102.6,
            c225=330.8,
            c226=451.2,
            c227=484.5,
            c228=150.0,
            c229=300.0,
        )
    ]
    fuzz_bounds = dict.fromkeys(
        ("c221", "c222", "c223", "c224", "c225", "c226", "c227", "c228", "c229"),
        (0.0, 3000.0),
    )


class TestTotalPlantDirectCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_total_plant_direct_cost
    ported = calculate_total_plant_direct_cost

    samples = [
        legacy_sample(
            "nominal", c21=1363.5, c22=4809.5, c23=263.8, c24=30.3, c25=22.1, c26=81.2
        )
    ]
    fuzz_bounds = dict.fromkeys(
        ("c21", "c22", "c23", "c24", "c25", "c26"), (0.0, 6000.0)
    )


class TestConstructedCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_constructed_cost
    ported = calculate_constructed_cost

    samples = [legacy_sample("nominal", cdirt=6570.4, cindrt=1843.7, ccont=1262.1)]
    fuzz_bounds = {
        "cdirt": (0.0, 20000.0),
        "cindrt": (0.0, 5000.0),
        "ccont": (0.0, 5000.0),
    }


class TestCostOfElectricity(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_cost_of_electricity
    ported = calculate_cost_of_electricity
    static_argnames = ("ife", "itart", "lsa", "ifueltyp")

    samples = [
        legacy_sample(
            "reference-run",
            ife=0,
            itart=0,
            p_plant_electric_net_mw=1000.0,
            f_t_plant_available=0.75,
            t_plant_pulse_burn=31557600.0,
            t_plant_pulse_total=31559410.0,
            concost=9676.16,
            fcap0=1.15,
            fcr0=0.065,
            discount_rate=0.06,
            life_blkt=19.48,
            fwallcst=0.0,
            blkcst=0.0,
            cfind=_CFIND,
            lsa=2,
            fcap0cp=1.06,
            ifueltyp=0,
            life_blkt_fpy=25.98,
            life_plant=40.0,
            life_div=7.86,
            divcst=0.0,
            life_div_fpy=10.48,
            cplife_cal=0.0,
            cpstcst=0.0,
            cplife=0.0,
            cdrlife_cal=19.48,
            cdcost=0.0,
            fcdfuel=0.1,
            ucoam=_UCOAM,
            ucfuel=3.45,
            f_plasma_fuel_helium3=0.0,
            wtgpd=764.38,
            uche3=1.0e6,
            ucwst=_UCWST,
            decomf=0.1,
            dintrt=0.0,
            dtlife=0.0,
        ),
        legacy_sample(
            "fuel-costs-active",
            ife=0,
            itart=0,
            p_plant_electric_net_mw=1200.0,
            f_t_plant_available=0.75,
            t_plant_pulse_burn=31557600.0,
            t_plant_pulse_total=31559410.0,
            concost=9676.16,
            fcap0=1.15,
            fcr0=0.065,
            discount_rate=0.06,
            life_blkt=19.48,
            fwallcst=280.0,
            blkcst=530.0,
            cfind=_CFIND,
            lsa=2,
            fcap0cp=1.06,
            ifueltyp=1,
            life_blkt_fpy=25.98,
            life_plant=40.0,
            life_div=7.86,
            divcst=22.6,
            life_div_fpy=10.48,
            cplife_cal=0.0,
            cpstcst=0.0,
            cplife=0.0,
            cdrlife_cal=19.48,
            cdcost=100.0,
            fcdfuel=0.1,
            ucoam=_UCOAM,
            ucfuel=3.45,
            f_plasma_fuel_helium3=0.01,
            wtgpd=764.38,
            uche3=1.0e6,
            ucwst=_UCWST,
            decomf=0.1,
            dintrt=0.0,
            dtlife=0.0,
        ),
        legacy_sample(
            "prorated-replacements-and-centrepost",
            ife=0,
            itart=1,
            p_plant_electric_net_mw=1200.0,
            f_t_plant_available=0.75,
            t_plant_pulse_burn=31557600.0,
            t_plant_pulse_total=31559410.0,
            concost=9676.16,
            fcap0=1.15,
            fcr0=0.065,
            discount_rate=0.06,
            life_blkt=19.48,
            fwallcst=280.0,
            blkcst=530.0,
            cfind=_CFIND,
            lsa=4,
            fcap0cp=1.06,
            ifueltyp=2,
            life_blkt_fpy=25.98,
            life_plant=40.0,
            life_div=7.86,
            divcst=22.6,
            life_div_fpy=10.48,
            cplife_cal=3.0,
            cpstcst=150.0,
            cplife=4.0,
            cdrlife_cal=19.48,
            cdcost=100.0,
            fcdfuel=0.1,
            ucoam=_UCOAM,
            ucfuel=3.45,
            f_plasma_fuel_helium3=0.01,
            wtgpd=764.38,
            uche3=1.0e6,
            ucwst=_UCWST,
            decomf=0.1,
            dintrt=0.01,
            dtlife=2.0,
        ),
        legacy_sample(
            "negative-net-electric-power-clamps-to-zero",
            ife=0,
            itart=0,
            p_plant_electric_net_mw=-50.0,
            f_t_plant_available=0.75,
            t_plant_pulse_burn=31557600.0,
            t_plant_pulse_total=31559410.0,
            concost=9676.16,
            fcap0=1.15,
            fcr0=0.065,
            discount_rate=0.06,
            life_blkt=19.48,
            fwallcst=0.0,
            blkcst=0.0,
            cfind=_CFIND,
            lsa=2,
            fcap0cp=1.06,
            ifueltyp=0,
            life_blkt_fpy=25.98,
            life_plant=40.0,
            life_div=7.86,
            divcst=0.0,
            life_div_fpy=10.48,
            cplife_cal=0.0,
            cpstcst=0.0,
            cplife=0.0,
            cdrlife_cal=19.48,
            cdcost=0.0,
            fcdfuel=0.1,
            ucoam=_UCOAM,
            ucfuel=3.45,
            f_plasma_fuel_helium3=0.0,
            wtgpd=764.38,
            uche3=1.0e6,
            ucwst=_UCWST,
            decomf=0.1,
            dintrt=0.0,
            dtlife=0.0,
        ),
    ]
    fuzz_bounds = {
        "p_plant_electric_net_mw": (100.0, 2000.0),
        "f_t_plant_available": (0.3, 0.95),
        "t_plant_pulse_burn": (1.0e6, 3.2e7),
        "t_plant_pulse_total": (3.2e7, 3.3e7),
        "concost": (1000.0, 20000.0),
        "fcap0": (1.0, 1.5),
        "fcr0": (0.02, 0.15),
        "discount_rate": (0.02, 0.12),
        "life_blkt": (2.0, 30.0),
        "fwallcst": (0.0, 500.0),
        "blkcst": (0.0, 1000.0),
        "fcap0cp": (1.0, 1.5),
        "life_blkt_fpy": (2.0, 30.0),
        "life_plant": (20.0, 50.0),
        "life_div": (2.0, 30.0),
        "divcst": (0.0, 200.0),
        "life_div_fpy": (2.0, 30.0),
        "cplife_cal": (1.0, 20.0),
        "cpstcst": (0.0, 500.0),
        "cplife": (1.0, 20.0),
        "cdrlife_cal": (2.0, 30.0),
        "cdcost": (0.0, 500.0),
        "fcdfuel": (0.01, 0.5),
        "ucfuel": (1.0, 10.0),
        "f_plasma_fuel_helium3": (0.0, 0.2),
        "wtgpd": (10.0, 5000.0),
        "uche3": (1.0e5, 1.0e7),
        "decomf": (0.01, 0.3),
        "dintrt": (0.0, 0.02),
        "dtlife": (0.0, 5.0),
    }
    fuzz_fixed = {
        "ife": 0,
        "itart": 0,
        "lsa": 2,
        "ifueltyp": 0,
        "cfind": _CFIND,
        "ucoam": _UCOAM,
        "ucwst": _UCWST,
    }
