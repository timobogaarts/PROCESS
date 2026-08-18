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

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.costs.costs import (
    calculate_auxiliary_facility_power_cost,
    calculate_diesel_generators_cost,
    calculate_divertor_cost,
    calculate_electric_plant_equipment_cost,
    calculate_fuelling_system_cost,
    calculate_heat_rejection_cost,
    calculate_indirect_costs,
    calculate_instrumentation_and_control_cost,
    calculate_low_voltage_cost,
    calculate_maintenance_equipment_cost,
    calculate_misc_plant_equipment_cost,
    calculate_nuclear_building_ventilation_cost,
    calculate_pf_coil_power_conditioning_cost,
    calculate_reactor_cooling_system_cost,
    calculate_reactor_structure_cost,
    calculate_structures_cost,
    calculate_switchyard_cost,
    calculate_tf_coil_power_conditioning_cost,
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
    csi, lsa, cland, ucrb, rbvol, UCMB, rmbvol, UCWS, wsvol, UCTR, triv, UCEL, elevol,
    UCAD, admvol, UCCO, convol, UCSH, shovol, UCCR, cryvol, ireactor, cturbb,
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
        c.c211, c.c212, c.c213, c.c2141, c.c2142, c.c214, c.c215, c.c216, c.c2171,
        c.c2172, c.c2173, c.c2174, c.c217, c.c21,
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
    i_vacuum_pump_type, n_vac_pumps_high, UCCPMP, UCTPMP, n_vv_vacuum_ducts, UCBPMP,
    dlscal, UCDUCT, dia_vv_vacuum_ducts, UCVALV, m_vv_vacuum_duct_shield, UCVDSH, UCVIAC,
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
    uctfps, tfckw, tfcmw, i_tf_sup, uctfbr, n_tf_coils, c_tf_turn,
    v_tf_coil_dump_quench_kv, uctfsw, UCTFDR, e_tf_magnetic_stored_total_gj, UCTFGR,
    UCTFIC, uctfbus, m_tf_bus, ucbus, len_tf_bus, fkind,
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
    ucpfps, peakmva, ucpfic, pfckts, ucpfb, spfbusl, acptmax, ucpfbs, srcktpm, ucpfbk,
    vpfskv, ucpfdr1, ensxpfm, ucpfcb, fkind,
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
    uchts, i_blkt_coolant_type, p_fw_div_heat_deposited_mw, p_blkt_nuclear_heat_total_mw,
    p_shld_nuclear_heat_mw, lsa, fkind, UCPHX, n_primary_heat_exchangers,
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


def _reference_transformers_cost(UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa):
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
    ireactor, p_fusion_total_mw, p_hcd_electric_total_mw, tfcmw, p_plant_primary_heat_mw,
    p_plant_electric_gross_mw, uchrs, lsa,
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
            "fast-branches", life_blkt_fpy=5.0, life_plant=30.0,
            f_t_plant_available=0.8, life_div_fpy=3.0, itart=0, cplife=10.0,
        ),
        legacy_sample(
            "slow-branches", life_blkt_fpy=40.0, life_plant=30.0,
            f_t_plant_available=0.8, life_div_fpy=40.0, itart=1, cplife=40.0,
        ),
        legacy_sample(
            "itart-fast", life_blkt_fpy=5.0, life_plant=30.0, f_t_plant_available=0.8,
            life_div_fpy=3.0, itart=1, cplife=10.0,
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
            "reactor", csi=16.0, lsa=4, cland=19.2, ucrb=400.0, rbvol=1.0e5,
            UCMB=260.0, rmbvol=5.0e4, UCWS=460.0, wsvol=1.0e4, UCTR=370.0, triv=4.0e4,
            UCEL=380.0, elevol=8.0e3, UCAD=180.0, admvol=1.0e4, UCCO=350.0, convol=6.0e3,
            UCSH=115.0, shovol=5.0e3, UCCR=460.0, cryvol=2.0e4, ireactor=1, cturbb=100.0,
        ),
        legacy_sample(
            "non-reactor-lsa1", csi=16.0, lsa=1, cland=19.2, ucrb=400.0, rbvol=1.0e5,
            UCMB=260.0, rmbvol=5.0e4, UCWS=460.0, wsvol=1.0e4, UCTR=370.0, triv=4.0e4,
            UCEL=380.0, elevol=8.0e3, UCAD=180.0, admvol=1.0e4, UCCO=350.0, convol=6.0e3,
            UCSH=115.0, shovol=5.0e3, UCCR=460.0, cryvol=2.0e4, ireactor=0, cturbb=100.0,
        ),
    ]
    fuzz_bounds = {
        "csi": (1.0, 50.0), "cland": (1.0, 50.0), "ucrb": (100.0, 800.0),
        "rbvol": (1.0e4, 1.0e6), "UCMB": (100.0, 500.0), "rmbvol": (1.0e4, 1.0e5),
        "UCWS": (100.0, 800.0), "wsvol": (1.0e3, 5.0e4), "UCTR": (100.0, 800.0),
        "triv": (1.0e3, 1.0e5), "UCEL": (100.0, 800.0), "elevol": (1.0e3, 5.0e4),
        "UCAD": (50.0, 400.0), "admvol": (1.0e3, 5.0e4), "UCCO": (100.0, 800.0),
        "convol": (1.0e3, 3.0e4), "UCSH": (50.0, 400.0), "shovol": (1.0e3, 3.0e4),
        "UCCR": (100.0, 800.0), "cryvol": (1.0e3, 5.0e4), "cturbb": (10.0, 500.0),
    }
    fuzz_fixed = {"lsa": 4, "ireactor": 1}


class TestIndirectCosts(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_indirect_costs
    ported = calculate_indirect_costs
    static_argnames = ("lsa",)

    samples = [
        legacy_sample(
            "lsa4", cfind=[0.244, 0.244, 0.244, 0.29], lsa=4, cdirt=1000.0, cowner=0.15,
            fcontng=0.195,
        ),
        legacy_sample(
            "lsa1", cfind=[0.244, 0.244, 0.244, 0.29], lsa=1, cdirt=1000.0, cowner=0.15,
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
            "not-ife-capital", ife=0, a_div_surface_total=100.0, ucdiv=2.8e5,
            fkind=1.0, ifueltyp=0,
        ),
        legacy_sample(
            "not-ife-fuel-cost", ife=0, a_div_surface_total=100.0, ucdiv=2.8e5,
            fkind=1.0, ifueltyp=1,
        ),
        legacy_sample(
            "not-ife-both", ife=0, a_div_surface_total=100.0, ucdiv=2.8e5, fkind=1.0,
            ifueltyp=2,
        ),
        legacy_sample(
            "ife", ife=1, a_div_surface_total=100.0, ucdiv=2.8e5, fkind=1.0, ifueltyp=0,
        ),
    ]
    fuzz_bounds = {"a_div_surface_total": (10.0, 1000.0), "ucdiv": (1.0e4, 1.0e6), "fkind": (0.5, 1.0)}
    fuzz_fixed = {"ife": 0, "ifueltyp": 0}


class TestVacuumSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_vacuum_system_cost
    ported = calculate_vacuum_system_cost
    static_argnames = ("i_vacuum_pump_type",)

    samples = [
        legacy_sample(
            "turbomolecular", i_vacuum_pump_type=0, n_vac_pumps_high=8.0,
            UCCPMP=2.5e5, UCTPMP=2.5e5, n_vv_vacuum_ducts=8.0, UCBPMP=2.5e4,
            dlscal=1.0, UCDUCT=3.0e4, dia_vv_vacuum_ducts=1.0, UCVALV=1.5e5,
            m_vv_vacuum_duct_shield=1.0e4, UCVDSH=90.0, UCVIAC=1.5e7, fkind=1.0,
        ),
        legacy_sample(
            "compound-cryopump", i_vacuum_pump_type=1, n_vac_pumps_high=8.0,
            UCCPMP=2.5e5, UCTPMP=2.5e5, n_vv_vacuum_ducts=8.0, UCBPMP=2.5e4,
            dlscal=1.0, UCDUCT=3.0e4, dia_vv_vacuum_ducts=1.0, UCVALV=1.5e5,
            m_vv_vacuum_duct_shield=1.0e4, UCVDSH=90.0, UCVIAC=1.5e7, fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "n_vac_pumps_high": (1.0, 20.0), "UCCPMP": (1.0e4, 1.0e6),
        "UCTPMP": (1.0e4, 1.0e6), "n_vv_vacuum_ducts": (1.0, 20.0),
        "UCBPMP": (1.0e3, 1.0e5), "dlscal": (0.1, 10.0), "UCDUCT": (1.0e3, 1.0e5),
        "dia_vv_vacuum_ducts": (0.1, 5.0), "UCVALV": (1.0e4, 1.0e6),
        "m_vv_vacuum_duct_shield": (1.0e2, 5.0e4), "UCVDSH": (10.0, 500.0),
        "UCVIAC": (1.0e6, 5.0e7), "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"i_vacuum_pump_type": 0}


class TestTfCoilPowerConditioningCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_tf_coil_power_conditioning_cost
    ported = calculate_tf_coil_power_conditioning_cost
    static_argnames = ("i_tf_sup",)

    samples = [
        legacy_sample(
            "resistive", uctfps=2.4e4, tfckw=1.0e3, tfcmw=50.0, i_tf_sup=0,
            uctfbr=1.22e6, n_tf_coils=16.0, c_tf_turn=6.0e4,
            v_tf_coil_dump_quench_kv=20.0, uctfsw=1.0e5, UCTFDR=1.75e-3,
            e_tf_magnetic_stored_total_gj=40.0, UCTFGR=1.0e5, UCTFIC=1.0e4,
            uctfbus=1.0, m_tf_bus=1.0e4, ucbus=460.0, len_tf_bus=300.0, fkind=1.0,
        ),
        legacy_sample(
            "superconducting", uctfps=2.4e4, tfckw=1.0e3, tfcmw=50.0, i_tf_sup=1,
            uctfbr=1.22e6, n_tf_coils=16.0, c_tf_turn=6.0e4,
            v_tf_coil_dump_quench_kv=20.0, uctfsw=1.0e5, UCTFDR=1.75e-3,
            e_tf_magnetic_stored_total_gj=40.0, UCTFGR=1.0e5, UCTFIC=1.0e4,
            uctfbus=1.0, m_tf_bus=1.0e4, ucbus=460.0, len_tf_bus=300.0, fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "uctfps": (1.0e3, 1.0e5), "tfckw": (1.0e2, 1.0e4), "tfcmw": (1.0, 200.0),
        "uctfbr": (1.0e5, 5.0e6), "n_tf_coils": (10.0, 24.0), "c_tf_turn": (1.0e3, 1.0e5),
        "v_tf_coil_dump_quench_kv": (1.0, 50.0), "uctfsw": (1.0e3, 1.0e6),
        "UCTFDR": (1.0e-4, 1.0e-2), "e_tf_magnetic_stored_total_gj": (1.0, 200.0),
        "UCTFGR": (1.0e3, 1.0e6), "UCTFIC": (1.0e2, 1.0e5), "uctfbus": (0.1, 10.0),
        "m_tf_bus": (1.0e2, 5.0e4), "ucbus": (100.0, 1000.0), "len_tf_bus": (10.0, 1000.0),
        "fkind": (0.5, 1.0),
    }
    fuzz_fixed = {"i_tf_sup": 1}


class TestPfCoilPowerConditioningCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_pf_coil_power_conditioning_cost
    ported = calculate_pf_coil_power_conditioning_cost

    samples = [
        legacy_sample(
            "nominal", ucpfps=3.5e4, peakmva=300.0, ucpfic=1.0e4, pfckts=12.0,
            ucpfb=1.0e4, spfbusl=300.0, acptmax=3.0e4, ucpfbs=1.0e5, srcktpm=3.0e4,
            ucpfbk=3.0e4, vpfskv=20.0, ucpfdr1=1.5e4, ensxpfm=1000.0, ucpfcb=7.5e4,
            fkind=1.0,
        ),
        legacy_sample(
            "zero-circuits", ucpfps=3.5e4, peakmva=300.0, ucpfic=1.0e4, pfckts=0.0,
            ucpfb=1.0e4, spfbusl=300.0, acptmax=3.0e4, ucpfbs=1.0e5, srcktpm=3.0e4,
            ucpfbk=3.0e4, vpfskv=20.0, ucpfdr1=1.5e4, ensxpfm=1000.0, ucpfcb=7.5e4,
            fkind=1.0,
        ),
    ]
    fuzz_bounds = {
        "ucpfps": (1.0e3, 1.0e5), "peakmva": (10.0, 1000.0), "ucpfic": (1.0e3, 1.0e5),
        "pfckts": (1.0, 30.0), "ucpfb": (1.0e3, 1.0e5), "spfbusl": (10.0, 1000.0),
        "acptmax": (1.0e3, 1.0e5), "ucpfbs": (1.0e4, 1.0e6), "srcktpm": (1.0e3, 1.0e5),
        "ucpfbk": (1.0e3, 1.0e5), "vpfskv": (1.0, 50.0), "ucpfdr1": (1.0e3, 1.0e5),
        "ensxpfm": (10.0, 5.0e3), "ucpfcb": (1.0e4, 1.0e6), "fkind": (0.5, 1.0),
    }


class TestReactorCoolingSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_reactor_cooling_system_cost
    ported = calculate_reactor_cooling_system_cost
    static_argnames = ("lsa", "i_blkt_coolant_type")

    samples = [
        legacy_sample(
            "water", uchts=[15.3, 19.1], i_blkt_coolant_type=1,
            p_fw_div_heat_deposited_mw=200.0, p_blkt_nuclear_heat_total_mw=300.0,
            p_shld_nuclear_heat_mw=20.0, lsa=4, fkind=1.0, UCPHX=15.0,
            n_primary_heat_exchangers=3, p_plant_primary_heat_mw=1500.0,
        ),
        legacy_sample(
            "helium", uchts=[15.3, 19.1], i_blkt_coolant_type=2,
            p_fw_div_heat_deposited_mw=200.0, p_blkt_nuclear_heat_total_mw=300.0,
            p_shld_nuclear_heat_mw=20.0, lsa=4, fkind=1.0, UCPHX=15.0,
            n_primary_heat_exchangers=3, p_plant_primary_heat_mw=1500.0,
        ),
    ]
    fuzz_bounds = {
        "p_fw_div_heat_deposited_mw": (10.0, 1000.0),
        "p_blkt_nuclear_heat_total_mw": (10.0, 1000.0),
        "p_shld_nuclear_heat_mw": (1.0, 200.0), "fkind": (0.5, 1.0),
        "UCPHX": (1.0, 100.0), "n_primary_heat_exchangers": (1.0, 6.0),
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

    samples = [legacy_sample("nominal", UCNBV=1.0e6, volrci=1.0e5, wsvol=1.0e4, fkind=1.0)]
    fuzz_bounds = {
        "UCNBV": (1.0e5, 5.0e6), "volrci": (1.0e4, 5.0e5), "wsvol": (1.0e3, 5.0e4),
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
            "reactor", ireactor=1, ucturb=[230.0e6, 245.0e6], i_blkt_coolant_type=1,
            p_plant_electric_gross_mw=1200.0,
        ),
        legacy_sample(
            "non-reactor", ireactor=0, ucturb=[230.0e6, 245.0e6], i_blkt_coolant_type=1,
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
            "nominal", UCPP=48.0, pacpmw=50.0, UCAP=15.0,
            p_plant_electric_base_total_mw=10.0, lsa=4,
        )
    ]
    fuzz_bounds = {
        "UCPP": (10.0, 200.0), "pacpmw": (1.0, 500.0), "UCAP": (1.0, 100.0),
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
        "c241": (0.0, 100.0), "c242": (0.0, 100.0), "c243": (0.0, 100.0),
        "c244": (0.0, 100.0), "c245": (0.0, 100.0),
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
            "reactor", ireactor=1, p_fusion_total_mw=2000.0,
            p_hcd_electric_total_mw=100.0, tfcmw=50.0, p_plant_primary_heat_mw=2200.0,
            p_plant_electric_gross_mw=1200.0, uchrs=1.0e7, lsa=4,
        ),
        legacy_sample(
            "non-reactor", ireactor=0, p_fusion_total_mw=2000.0,
            p_hcd_electric_total_mw=100.0, tfcmw=50.0, p_plant_primary_heat_mw=2200.0,
            p_plant_electric_gross_mw=1200.0, uchrs=1.0e7, lsa=4,
        ),
    ]
    fuzz_bounds = {
        "p_fusion_total_mw": (100.0, 5000.0), "p_hcd_electric_total_mw": (1.0, 500.0),
        "tfcmw": (1.0, 200.0), "p_plant_primary_heat_mw": (100.0, 3000.0),
        "p_plant_electric_gross_mw": (100.0, 2000.0), "uchrs": (1.0e6, 5.0e7),
    }
    fuzz_fixed = {"ireactor": 1, "lsa": 4}
