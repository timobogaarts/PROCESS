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

from functional_process.cottax._harness.process_reference import process_reference
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


_reference_convert_fpy_to_calendar = process_reference(
    _make_costs, "convert_fpy_to_calendar", (
        "fwbs.life_blkt",
        "costs.cdrlife_cal",
        "costs.life_div",
        "costs.cplife_cal",
    )
)


_reference_structures_cost = process_reference(
    _make_costs, "acc21", (
        "costs.c211",
        "costs.c212",
        "costs.c213",
        "costs.c2141",
        "costs.c2142",
        "costs.c214",
        "costs.c215",
        "costs.c216",
        "costs.c2171",
        "costs.c2172",
        "costs.c2173",
        "costs.c2174",
        "costs.c217",
        "costs.c21",
    )
)


_reference_indirect_costs = process_reference(
    _make_costs, "acc9", (
        "costs.cindrt",
        "costs.ccont",
    )
)


_reference_reactor_structure_cost = process_reference(
    _make_costs, "acc2214", "costs.c2214"
)


_reference_vacuum_vessel_assembly_cost = process_reference(
    _make_costs, "acc2223", "costs.c2223"
)


_reference_divertor_cost = process_reference(
    _make_costs, "acc2215", (
        "costs.c2215",
        "costs.divcst",
    )
)


_reference_vacuum_system_cost = process_reference(
    _make_costs, "acc224", (
        "costs.c2241",
        "costs.c2242",
        "costs.c2243",
        "costs.c2244",
        "costs.c2245",
        "costs.c2246",
        "costs.c224",
    )
)


_reference_tf_coil_power_conditioning_cost = process_reference(
    _make_costs, "acc2251", (
        "costs.c22511",
        "costs.c22512",
        "costs.c22513",
        "costs.c22514",
        "costs.c22515",
        "costs.c2251",
    )
)


_reference_pf_coil_power_conditioning_cost = process_reference(
    _make_costs, "acc2252", (
        "costs.c22521",
        "costs.c22522",
        "costs.c22523",
        "costs.c22524",
        "costs.c22525",
        "costs.c22526",
        "costs.c22527",
        "costs.c2252",
    )
)


_reference_reactor_cooling_system_cost = process_reference(
    _make_costs, "acc2261", (
        "costs.cpp",
        "costs.chx",
        "costs.c2261",
    )
)


_reference_fuelling_system_cost = process_reference(
    _make_costs, "acc2271", "costs.c2271"
)


_reference_nuclear_building_ventilation_cost = process_reference(
    _make_costs, "acc2274", "costs.c2274"
)


_reference_instrumentation_and_control_cost = process_reference(
    _make_costs, "acc228", "costs.c228"
)


_reference_maintenance_equipment_cost = process_reference(
    _make_costs, "acc229", "costs.c229"
)


_reference_turbine_plant_equipment_cost = process_reference(
    _make_costs, "acc23", "costs.c23"
)


_reference_switchyard_cost = process_reference(
    _make_costs, "acc241", "costs.c241"
)


_reference_transformers_cost = process_reference(
    _make_costs, "acc242", "costs.c242"
)


_reference_low_voltage_cost = process_reference(
    _make_costs, "acc243", "costs.c243"
)


_reference_diesel_generators_cost = process_reference(
    _make_costs, "acc244", "costs.c244"
)


_reference_auxiliary_facility_power_cost = process_reference(
    _make_costs, "acc245", "costs.c245"
)


_reference_electric_plant_equipment_cost = process_reference(
    _make_costs, "acc24", "costs.c24"
)


_reference_misc_plant_equipment_cost = process_reference(
    _make_costs, "acc25", "costs.c25"
)


_reference_heat_rejection_cost = process_reference(
    _make_costs, "acc26", "costs.c26"
)


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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
    fuzz_fixed = {"cfind": [0.244, 0.244, 0.244, 0.29], "lsa": 4}


class TestReactorStructureCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_reactor_structure_cost
    ported = calculate_reactor_structure_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", gsmass=5.0e5, UCGSS=35.0, lsa=4, fkind=1.0)]
    fuzz = True
    fuzz_fixed = {"lsa": 4}


class TestVacuumVesselAssemblyCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_vacuum_vessel_assembly_cost
    ported = calculate_vacuum_vessel_assembly_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", m_vv=9.0e6, uccryo=32.0, lsa=4, fkind=1.0)]
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True


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
    fuzz = True
    fuzz_fixed = {"uchts": [15.3, 19.1], "i_blkt_coolant_type": 1, "lsa": 4}


class TestFuellingSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fuelling_system_cost
    ported = calculate_fuelling_system_cost

    samples = [legacy_sample("nominal", ucf1=2.23e7, fkind=1.0)]
    fuzz = True


class TestNuclearBuildingVentilationCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_nuclear_building_ventilation_cost
    ported = calculate_nuclear_building_ventilation_cost

    samples = [
        legacy_sample("nominal", UCNBV=1.0e6, volrci=1.0e5, wsvol=1.0e4, fkind=1.0)
    ]
    fuzz = True


class TestInstrumentationAndControlCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_instrumentation_and_control_cost
    ported = calculate_instrumentation_and_control_cost

    samples = [legacy_sample("nominal", uciac=4.0e7, fkind=1.0)]
    fuzz = True


class TestMaintenanceEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_maintenance_equipment_cost
    ported = calculate_maintenance_equipment_cost

    samples = [legacy_sample("nominal", ucme=1.25e8, fkind=1.0)]
    fuzz = True


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
    fuzz = True
    fuzz_fixed = {"ireactor": 1, "ucturb": [230.0e6, 245.0e6], "i_blkt_coolant_type": 1}


class TestSwitchyardCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_switchyard_cost
    ported = calculate_switchyard_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCSWYD=1.9e7, lsa=4)]
    fuzz = True
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
    fuzz = True
    fuzz_fixed = {"lsa": 4}


class TestLowVoltageCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_low_voltage_cost
    ported = calculate_low_voltage_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCLV=265.0, tlvpmw=10.0, lsa=4)]
    fuzz = True
    fuzz_fixed = {"lsa": 4}


class TestDieselGeneratorsCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_diesel_generators_cost
    ported = calculate_diesel_generators_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCDGEN=1.7e6, lsa=4)]
    fuzz = True
    fuzz_fixed = {"lsa": 4}


class TestAuxiliaryFacilityPowerCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_auxiliary_facility_power_cost
    ported = calculate_auxiliary_facility_power_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", UCAF=1.5e6, lsa=4)]
    fuzz = True
    fuzz_fixed = {"lsa": 4}


class TestElectricPlantEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_electric_plant_equipment_cost
    ported = calculate_electric_plant_equipment_cost

    samples = [
        legacy_sample("nominal", c241=10.0, c242=15.0, c243=8.0, c244=6.0, c245=2.0)
    ]
    fuzz = True


class TestMiscPlantEquipmentCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_misc_plant_equipment_cost
    ported = calculate_misc_plant_equipment_cost
    static_argnames = ("lsa",)

    samples = [legacy_sample("nominal", ucmisc=2.5e7, lsa=4)]
    fuzz = True
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
    fuzz = True
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


_reference_first_wall_cost = process_reference(
    _make_costs, "acc2211", (
        "costs.c2211",
        "costs.fwallcst",
    )
)


_reference_blanket_cost = process_reference(
    _make_costs, "acc2212", (
        "costs.c22121",
        "costs.c22122",
        "costs.c22123",
        "costs.c22124",
        "costs.c22125",
        "costs.c22126",
        "costs.c22127",
        "costs.c2212",
        "costs.blkcst",
    )
)


_reference_shield_cost = process_reference(
    _make_costs, "acc2213", (
        "costs.c22131",
        "costs.c22132",
        "costs.c2213",
    )
)


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


_reference_power_injection_cost = process_reference(
    _make_costs, "acc223", (
        "costs.c2231",
        "costs.c2232",
        "costs.c2233",
        "costs.c223",
        "costs.cdcost",
    )
)


_reference_energy_storage_cost = process_reference(
    _make_costs, "acc2253", "costs.c2253"
)


def _reference_power_conditioning_cost(ife, c2251, c2252, c2253):
    costs = _make_costs()
    costs.data.ife.ife = ife
    costs.data.costs.c2251 = c2251
    costs.data.costs.c2252 = c2252
    costs.data.costs.c2253 = c2253
    _neutralise(costs, "acc2251", "acc2252", "acc2253")
    costs.acc225()
    return costs.data.costs.c225


_reference_auxiliary_component_cooling_cost = process_reference(
    _make_costs, "acc2262", (
        "costs.cppa",
        "costs.c2262",
    )
)


_reference_cryogenic_system_cost = process_reference(
    _make_costs, "acc2263", "costs.c2263"
)


_reference_heat_transport_system_cost = process_reference(
    _make_costs, "acc226", "costs.c226"
)


_reference_fuel_processing_cost = process_reference(
    _make_costs, "acc2272", (
        "physics.wtgpd",
        "costs.c2272",
    )
)


_reference_atmospheric_recovery_cost = process_reference(
    _make_costs, "acc2273", "costs.c2273"
)


_reference_fuel_handling_cost = process_reference(
    _make_costs, "acc227", "costs.c227"
)


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


_reference_cost_of_electricity = process_reference(
    _make_costs, "coelc", (
        "costs.moneyint",
        "costs.capcost",
        "costs.coecap",
        "costs.coeoam",
        "costs.coefuelt",
        "costs.coe",
    )
)


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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True


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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
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
    fuzz = True
    fuzz_fixed = {"lsa": 2}


class TestHeatTransportSystemCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_heat_transport_system_cost
    ported = calculate_heat_transport_system_cost

    samples = [legacy_sample("nominal", c2261=145.8, c2262=21.1, c2263=284.3)]
    fuzz = True


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
    fuzz = True
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
    fuzz = True


class TestFuelHandlingCost(Tier1Contract):
    audit_record = "models/costs/costs.md"
    reference = _reference_fuel_handling_cost
    ported = calculate_fuel_handling_cost

    samples = [
        legacy_sample("nominal", c2271=22.3, c2272=143.0, c2273=162.0, c2274=157.2)
    ]
    fuzz = True


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
    fuzz = True


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
    fuzz = True
    fuzz_fixed = {
        "ife": 0,
        "itart": 0,
        "lsa": 2,
        "ifueltyp": 0,
        "cfind": _CFIND,
        "ucoam": _UCOAM,
        "ucwst": _UCWST,
    }
