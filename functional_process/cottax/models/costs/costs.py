"""Pure-functional port of a self-contained subset of `process/models/costs/costs.py`
(the 1990-style cost model, registry unit #18).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces import V

from functional_process.cottax.models.pfcoil.masses import (
    I_CS_SUPERCONDUCTOR,
    I_CS_SUPERCONDUCTOR_WST_NB3SN,
    I_PF_SUPERCONDUCTOR,
    I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO,
)
from functional_process.cottax.paths import (
    buildings,
    costs,
    current_drive,
    divertor,
    first_wall,
    fwbs,
    heat_transport,
    ife,
    pf_coil,
    pf_power,
    physics,
    structure,
    tfcoil,
    times,
    vacuum,
)
from functional_process.cottax.stated import states
from functional_process.models.costs.costs import (
    calculate_atmospheric_recovery_cost,
    calculate_auxiliary_component_cooling_cost,
    calculate_auxiliary_component_cooling_cost_magnetic_confinement,
    calculate_auxiliary_facility_power_cost,
    calculate_blanket_cost,
    calculate_blanket_cost_magnetic_confinement,
    calculate_constructed_cost,
    calculate_cost_of_electricity,
    calculate_cost_of_electricity_conventional_aspect_ratio,
    calculate_cost_of_electricity_spherical_tokamak,
    calculate_cryogenic_system_cost,
    calculate_diesel_generators_cost,
    calculate_divertor_cost,
    calculate_electric_plant_equipment_cost,
    calculate_energy_storage_cost,
    calculate_energy_storage_cost_electrowatt_option_1,
    calculate_energy_storage_cost_electrowatt_option_2,
    calculate_first_wall_cost,
    calculate_first_wall_cost_magnetic_confinement,
    calculate_fuel_handling_cost,
    calculate_fuel_processing_cost,
    calculate_fuel_processing_cost_magnetic_confinement,
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
    calculate_pf_magnet_cost_per_kam,
    calculate_pf_magnet_cost_per_kam_no_central_solenoid,
    calculate_pf_magnet_cost_per_kg,
    calculate_pf_magnet_cost_per_kg_no_central_solenoid,
    calculate_power_conditioning_cost,
    calculate_power_injection_cost,
    calculate_power_injection_cost_magnetic_confinement,
    calculate_reactor_cooling_system_cost,
    calculate_reactor_cost,
    calculate_reactor_structure_cost,
    calculate_shield_cost,
    calculate_shield_cost_magnetic_confinement,
    calculate_structures_cost,
    calculate_switchyard_cost,
    calculate_tf_coil_power_conditioning_cost,
    calculate_tf_magnet_cost_resistive,
    calculate_tf_magnet_cost_superconducting,
    calculate_tf_magnet_cost_superconducting_per_kam,
    calculate_tf_magnet_cost_superconducting_per_kg,
    calculate_total_plant_direct_cost,
    calculate_transformers_cost,
    calculate_turbine_plant_equipment_cost,
    calculate_vacuum_system_cost,
    calculate_vacuum_vessel_assembly_cost,
    convert_fpy_to_calendar,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.vocabulary import PFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public"
# once one is present, so this lists every public name this module resolved before
# step 2 of `_audit/formulas_split.md` moved the pure functions out -- not just the
# handful actually unused here (see `power/electric_production.py`'s commit for why
# a partial list is the wrong move).

__all__ = [
    "I_CS_SUPERCONDUCTOR",
    "I_CS_SUPERCONDUCTOR_WST_NB3SN",
    "I_PF_SUPERCONDUCTOR",
    "I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO",
    "AtmosphericRecoveryCost",
    "AuxiliaryComponentCoolingCost",
    "AuxiliaryFacilityPowerCost",
    "BlanketCost",
    "ConstructedCost",
    "ConvertFpyToCalendar",
    "CostOfElectricityConventionalAspectRatio",
    "CostOfElectricitySphericalTokamak",
    "CryogenicSystemCost",
    "DieselGeneratorsCost",
    "DivertorCost",
    "ElectricPlantEquipmentCost",
    "EnergyStorageCostPulsedElectrowattOption1",
    "EnergyStorageCostPulsedElectrowattOption2",
    "EnergyStorageCostUnpulsed",
    "FirstWallCost",
    "FuelHandlingCost",
    "FuelProcessingCost",
    "FuellingSystemCost",
    "FusionPowerIslandCost",
    "HeatRejectionCost",
    "HeatTransportSystemCost",
    "IndirectCosts",
    "InstrumentationAndControlCost",
    "LowVoltageCost",
    "MagnetsCost",
    "MaintenanceEquipmentCost",
    "MiscPlantEquipmentCost",
    "NuclearBuildingVentilationCost",
    "PFConductorModel",
    "PfCoilPowerConditioningCost",
    "PfMagnetCostPerKam",
    "PfMagnetCostPerKamNoCentralSolenoid",
    "PfMagnetCostPerKg",
    "PfMagnetCostPerKgCsWstNb3Sn",
    "PfMagnetCostPerKgNoCentralSolenoid",
    "PowerConditioningCost",
    "PowerInjectionCost",
    "ReactorCoolingSystemCost",
    "ReactorCost",
    "ReactorStructureCost",
    "ShieldCost",
    "StructuresCost",
    "SwitchyardCost",
    "TfCoilPowerConditioningCost",
    "TfMagnetCostResistive",
    "TfMagnetCostSuperconductingPerKam",
    "TfMagnetCostSuperconductingPerKg",
    "TotalPlantDirectCost",
    "TransformersCost",
    "TurbinePlantEquipmentCost",
    "VacuumSystemCost",
    "VacuumVesselAssemblyCost",
    "buildings",
    "calculate_atmospheric_recovery_cost",
    "calculate_auxiliary_component_cooling_cost",
    "calculate_auxiliary_component_cooling_cost_magnetic_confinement",
    "calculate_auxiliary_facility_power_cost",
    "calculate_blanket_cost",
    "calculate_blanket_cost_magnetic_confinement",
    "calculate_constructed_cost",
    "calculate_cost_of_electricity",
    "calculate_cost_of_electricity_conventional_aspect_ratio",
    "calculate_cost_of_electricity_spherical_tokamak",
    "calculate_cryogenic_system_cost",
    "calculate_diesel_generators_cost",
    "calculate_divertor_cost",
    "calculate_electric_plant_equipment_cost",
    "calculate_energy_storage_cost",
    "calculate_energy_storage_cost_electrowatt_option_1",
    "calculate_energy_storage_cost_electrowatt_option_2",
    "calculate_first_wall_cost",
    "calculate_first_wall_cost_magnetic_confinement",
    "calculate_fuel_handling_cost",
    "calculate_fuel_processing_cost",
    "calculate_fuel_processing_cost_magnetic_confinement",
    "calculate_fuelling_system_cost",
    "calculate_fusion_power_island_cost",
    "calculate_heat_rejection_cost",
    "calculate_heat_transport_system_cost",
    "calculate_indirect_costs",
    "calculate_instrumentation_and_control_cost",
    "calculate_low_voltage_cost",
    "calculate_magnets_cost",
    "calculate_maintenance_equipment_cost",
    "calculate_misc_plant_equipment_cost",
    "calculate_nuclear_building_ventilation_cost",
    "calculate_pf_coil_power_conditioning_cost",
    "calculate_pf_magnet_cost",
    "calculate_pf_magnet_cost_per_kam",
    "calculate_pf_magnet_cost_per_kam_no_central_solenoid",
    "calculate_pf_magnet_cost_per_kg",
    "calculate_pf_magnet_cost_per_kg_no_central_solenoid",
    "calculate_power_conditioning_cost",
    "calculate_power_injection_cost",
    "calculate_power_injection_cost_magnetic_confinement",
    "calculate_reactor_cooling_system_cost",
    "calculate_reactor_cost",
    "calculate_reactor_structure_cost",
    "calculate_shield_cost",
    "calculate_shield_cost_magnetic_confinement",
    "calculate_structures_cost",
    "calculate_switchyard_cost",
    "calculate_tf_coil_power_conditioning_cost",
    "calculate_tf_magnet_cost_resistive",
    "calculate_tf_magnet_cost_superconducting",
    "calculate_tf_magnet_cost_superconducting_per_kam",
    "calculate_tf_magnet_cost_superconducting_per_kg",
    "calculate_total_plant_direct_cost",
    "calculate_transformers_cost",
    "calculate_turbine_plant_equipment_cost",
    "calculate_vacuum_system_cost",
    "calculate_vacuum_vessel_assembly_cost",
    "convert_fpy_to_calendar",
    "costs",
    "current_drive",
    "divertor",
    "eqx",
    "first_wall",
    "fwbs",
    "heat_transport",
    "ife",
    "jnp",
    "pf_coil",
    "pf_power",
    "physics",
    "safe_pow",
    "safe_sqrt",
    "structure",
    "tfcoil",
    "times",
    "vacuum",
]


# cottax node: `convert_fpy_to_calendar`.
def _convert_fpy_to_calendar(
    life_blkt_fpy, life_plant, f_t_plant_available, life_div_fpy, itart, cplife
):
    return convert_fpy_to_calendar(
        life_blkt_fpy, life_plant, f_t_plant_available, life_div_fpy, itart, cplife
    )


ConvertFpyToCalendar = fwbs.function(
    _convert_fpy_to_calendar,
    "life_blkt",
    V.costs.cdrlife_cal,
    V.costs.life_div,
    V.costs.cplife_cal,
    reads={
        "life_plant": costs,
        "f_t_plant_available": costs,
        "life_div_fpy": costs,
        "itart": physics,
        "cplife": costs,
    },
)


# cottax node: `calculate_structures_cost` (Account 21).
def _structures_cost(
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
    return calculate_structures_cost(
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
    )


StructuresCost = costs.function(
    _structures_cost,
    "c211",
    "c212",
    "c213",
    "c2141",
    "c2142",
    "c214",
    "c215",
    "c216",
    "c2171",
    "c2172",
    "c2173",
    "c2174",
    "c217",
    "c21",
    reads={
        "rbvol": buildings,
        "rmbvol": buildings,
        "wsvol": buildings,
        "triv": buildings,
        "elevol": buildings,
        "admvol": buildings,
        "convol": buildings,
        "shovol": buildings,
        "cryvol": buildings,
    },
)


# cottax node: `calculate_indirect_costs` (Account 9).
def _indirect_costs(cfind, lsa, cdirt, cowner, fcontng):
    return calculate_indirect_costs(cfind, lsa, cdirt, cowner, fcontng)


IndirectCosts = costs.function(_indirect_costs, "cindrt", "ccont")


# cottax node: `calculate_reactor_structure_cost` (Account 221.4).
def _reactor_structure_cost(gsmass, UCGSS, lsa, fkind):
    return calculate_reactor_structure_cost(gsmass, UCGSS, lsa, fkind)


ReactorStructureCost = costs.function(
    _reactor_structure_cost,
    "c2214",
    reads={
        "gsmass": structure,
    },
)


# cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3).
def _vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind):
    return calculate_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind)


VacuumVesselAssemblyCost = costs.function(
    _vacuum_vessel_assembly_cost,
    "c2223",
    reads={
        "m_vv": fwbs,
    },
)


# cottax node: `calculate_divertor_cost` (Account 221.5).
def _divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp):
    return calculate_divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp)


DivertorCost = costs.function(
    _divertor_cost,
    "c2215",
    "divcst",
    reads={
        "ife": ife,
        "a_div_surface_total": divertor,
    },
)


# cottax node: `calculate_vacuum_system_cost` (Account 224).
def _vacuum_system_cost(
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
    return calculate_vacuum_system_cost(
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
    )


VacuumSystemCost = costs.function(
    _vacuum_system_cost,
    "c2241",
    "c2242",
    "c2243",
    "c2244",
    "c2245",
    "c2246",
    "c224",
    reads={
        "i_vacuum_pump_type": vacuum,
        "n_vac_pumps_high": vacuum,
        "n_vv_vacuum_ducts": vacuum,
        "dlscal": vacuum,
        "dia_vv_vacuum_ducts": vacuum,
        "m_vv_vacuum_duct_shield": vacuum,
    },
)


# cottax node: `calculate_tf_coil_power_conditioning_cost` (Account 225.1).
def _tf_coil_power_conditioning_cost(
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
    return calculate_tf_coil_power_conditioning_cost(
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
    )


TfCoilPowerConditioningCost = costs.function(
    _tf_coil_power_conditioning_cost,
    "c22511",
    "c22512",
    "c22513",
    "c22514",
    "c22515",
    "c2251",
    reads={
        "tfckw": tfcoil,
        "tfcmw": tfcoil,
        "i_tf_sup": tfcoil,
        "n_tf_coils": tfcoil,
        "c_tf_turn": tfcoil,
        "v_tf_coil_dump_quench_kv": tfcoil,
        "e_tf_magnetic_stored_total_gj": tfcoil,
        "m_tf_bus": tfcoil,
        "len_tf_bus": tfcoil,
    },
)


# cottax node: `calculate_pf_coil_power_conditioning_cost` (Account 225.2).
def _pf_coil_power_conditioning_cost(
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
    return calculate_pf_coil_power_conditioning_cost(
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
    )


PfCoilPowerConditioningCost = costs.function(
    _pf_coil_power_conditioning_cost,
    "c22521",
    "c22522",
    "c22523",
    "c22524",
    "c22525",
    "c22526",
    "c22527",
    "c2252",
    reads={
        "peakmva": heat_transport,
        "pfckts": pf_power,
        "spfbusl": pf_power,
        "acptmax": pf_power,
        "srcktpm": pf_power,
        "vpfskv": pf_power,
        "ensxpfm": pf_power,
    },
)


# cottax node: `calculate_reactor_cooling_system_cost` (Account 2261).
def _reactor_cooling_system_cost(
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
    return calculate_reactor_cooling_system_cost(
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
    )


ReactorCoolingSystemCost = costs.function(
    _reactor_cooling_system_cost,
    "cpp",
    "chx",
    "c2261",
    reads={
        "i_blkt_coolant_type": fwbs,
        "p_fw_div_heat_deposited_mw": heat_transport,
        "p_blkt_nuclear_heat_total_mw": fwbs,
        "p_shld_nuclear_heat_mw": fwbs,
        "n_primary_heat_exchangers": heat_transport,
        "p_plant_primary_heat_mw": heat_transport,
    },
)


# cottax node: `calculate_fuelling_system_cost` (Account 2271).
def _fuelling_system_cost(ucf1, fkind):
    return calculate_fuelling_system_cost(ucf1, fkind)


FuellingSystemCost = costs.function(_fuelling_system_cost, "c2271")


# cottax node: `calculate_nuclear_building_ventilation_cost` (Account 2274).
def _nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind):
    return calculate_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind)


NuclearBuildingVentilationCost = costs.function(
    _nuclear_building_ventilation_cost,
    "c2274",
    reads={
        "volrci": buildings,
        "wsvol": buildings,
    },
)


# cottax node: `calculate_instrumentation_and_control_cost` (Account 228).
def _instrumentation_and_control_cost(uciac, fkind):
    return calculate_instrumentation_and_control_cost(uciac, fkind)


InstrumentationAndControlCost = costs.function(_instrumentation_and_control_cost, "c228")


# cottax node: `calculate_maintenance_equipment_cost` (Account 229).
def _maintenance_equipment_cost(ucme, fkind):
    return calculate_maintenance_equipment_cost(ucme, fkind)


MaintenanceEquipmentCost = costs.function(_maintenance_equipment_cost, "c229")


# cottax node: `calculate_turbine_plant_equipment_cost` (Account 23).
def _turbine_plant_equipment_cost(
    ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
):
    return calculate_turbine_plant_equipment_cost(
        ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
    )


TurbinePlantEquipmentCost = costs.function(
    _turbine_plant_equipment_cost,
    "c23",
    reads={
        "i_blkt_coolant_type": fwbs,
        "p_plant_electric_gross_mw": heat_transport,
    },
)


# cottax node: `calculate_switchyard_cost` (Account 241).
def _switchyard_cost(UCSWYD, lsa):
    return calculate_switchyard_cost(UCSWYD, lsa)


SwitchyardCost = costs.function(_switchyard_cost, "c241")


# cottax node: `calculate_transformers_cost` (Account 242).
def _transformers_cost(UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa):
    return calculate_transformers_cost(
        UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa
    )


TransformersCost = costs.function(
    _transformers_cost,
    "c242",
    reads={
        "pacpmw": heat_transport,
        "p_plant_electric_base_total_mw": heat_transport,
    },
)


# cottax node: `calculate_low_voltage_cost` (Account 243).
def _low_voltage_cost(UCLV, tlvpmw, lsa):
    return calculate_low_voltage_cost(UCLV, tlvpmw, lsa)


LowVoltageCost = costs.function(
    _low_voltage_cost,
    "c243",
    reads={
        "tlvpmw": heat_transport,
    },
)


# cottax node: `calculate_diesel_generators_cost` (Account 244).
def _diesel_generators_cost(UCDGEN, lsa):
    return calculate_diesel_generators_cost(UCDGEN, lsa)


DieselGeneratorsCost = costs.function(_diesel_generators_cost, "c244")


# cottax node: `calculate_auxiliary_facility_power_cost` (Account 245).
def _auxiliary_facility_power_cost(UCAF, lsa):
    return calculate_auxiliary_facility_power_cost(UCAF, lsa)


AuxiliaryFacilityPowerCost = costs.function(_auxiliary_facility_power_cost, "c245")


# cottax node: `calculate_electric_plant_equipment_cost` (Account 24, total).
def _electric_plant_equipment_cost(c241, c242, c243, c244, c245):
    return calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245)


ElectricPlantEquipmentCost = costs.function(_electric_plant_equipment_cost, "c24")


# cottax node: `calculate_misc_plant_equipment_cost` (Account 25).
def _misc_plant_equipment_cost(ucmisc, lsa):
    return calculate_misc_plant_equipment_cost(ucmisc, lsa)


MiscPlantEquipmentCost = costs.function(_misc_plant_equipment_cost, "c25")


# cottax node: `calculate_heat_rejection_cost` (Account 26).
def _heat_rejection_cost(
    ireactor,
    p_fusion_total_mw,
    p_hcd_electric_total_mw,
    tfcmw,
    p_plant_primary_heat_mw,
    p_plant_electric_gross_mw,
    uchrs,
    lsa,
):
    return calculate_heat_rejection_cost(
        ireactor,
        p_fusion_total_mw,
        p_hcd_electric_total_mw,
        tfcmw,
        p_plant_primary_heat_mw,
        p_plant_electric_gross_mw,
        uchrs,
        lsa,
    )


HeatRejectionCost = costs.function(
    _heat_rejection_cost,
    "c26",
    reads={
        "p_fusion_total_mw": physics,
        "p_hcd_electric_total_mw": heat_transport,
        "tfcmw": tfcoil,
        "p_plant_primary_heat_mw": heat_transport,
        "p_plant_electric_gross_mw": heat_transport,
    },
)


# --------------------------------------------------------------------------------------
# Second porting wave's nodes: the `.costs.coe` chain. Registered in `total_process.py`
# under `.costs.i_cost_model == 0`; see that switch's own comment block.
# --------------------------------------------------------------------------------------


# cottax node: `calculate_first_wall_cost` (Account 221.1).
def _first_wall_cost(lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp):
    return calculate_first_wall_cost_magnetic_confinement(
        lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp
    )


FirstWallCost = costs.function(
    _first_wall_cost,
    "c2211",
    "fwallcst",
    reads={
        "a_fw_total": first_wall,
    },
)


# cottax node: `calculate_blanket_cost` (Account 221.2).
def _blanket_cost(
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
    return calculate_blanket_cost_magnetic_confinement(
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
    )


BlanketCost = costs.function(
    _blanket_cost,
    "c22121",
    "c22122",
    "c22123",
    "c22124",
    "c22125",
    "c22126",
    "c22127",
    "c2212",
    "blkcst",
    reads={
        "m_blkt_beryllium": fwbs,
        "m_blkt_li2o": fwbs,
        "m_blkt_steel_total": fwbs,
        "m_blkt_vanadium": fwbs,
    },
)


# cottax node: `calculate_shield_cost` (Account 221.3).
def _shield_cost(lsa, whtshld, ucshld, wpenshld, ucpens, fkind):
    return calculate_shield_cost_magnetic_confinement(
        lsa, whtshld, ucshld, wpenshld, ucpens, fkind
    )


ShieldCost = costs.function(
    _shield_cost,
    "c22131",
    "c22132",
    "c2213",
    reads={
        "whtshld": fwbs,
        "wpenshld": fwbs,
    },
)


# cottax node: `calculate_reactor_cost` (Account 221 total).
def _reactor_cost(c2211, c2212, c2213, c2214, c2215):
    return calculate_reactor_cost(c2211, c2212, c2213, c2214, c2215)


ReactorCost = costs.function(_reactor_cost, "c221")


# The TfMagnetCostSuperconducting family:
# The Account 222.1 superconducting-TF family (`.tfcoil.i_tf_sup == 1`).


# `.costs.supercond_cost_model == PER_KG` (0) -- PROCESS's own default
# (`cost_variables.py:552`) and the reference run's.
def _tf_magnet_cost_superconducting_per_kg(
    lsa,
    ucsc,
    i_tf_sc_mat,
    m_tf_coil_superconductor,
    len_tf_coil,
    n_tf_coil_turns,
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
    return calculate_tf_magnet_cost_superconducting_per_kg(
        lsa,
        ucsc,
        i_tf_sc_mat,
        m_tf_coil_superconductor,
        len_tf_coil,
        n_tf_coil_turns,
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
    )


TfMagnetCostSuperconductingPerKg = costs.function(
    _tf_magnet_cost_superconducting_per_kg,
    "c22211",
    "c22212",
    "c22213",
    "c22214",
    "c22215",
    "c2221",
    reads={
        "i_tf_sc_mat": tfcoil,
        "m_tf_coil_superconductor": tfcoil,
        "len_tf_coil": tfcoil,
        "n_tf_coil_turns": tfcoil,
        "m_tf_coil_copper": tfcoil,
        "n_tf_coils": tfcoil,
        "m_tf_coil_case": tfcoil,
        "aintmass": structure,
        "clgsmass": structure,
    },
)


# `.costs.supercond_cost_model == PER_KAM` (1) -- strand cost scaled by critical
# current density.
def _tf_magnet_cost_superconducting_per_kam(
    lsa,
    sc_mat_cost_0,
    i_tf_sc_mat,
    j_crit_str_0,
    j_crit_str_tf,
    len_tf_coil,
    n_tf_coil_turns,
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
    return calculate_tf_magnet_cost_superconducting_per_kam(
        lsa,
        sc_mat_cost_0,
        i_tf_sc_mat,
        j_crit_str_0,
        j_crit_str_tf,
        len_tf_coil,
        n_tf_coil_turns,
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
    )


TfMagnetCostSuperconductingPerKam = costs.function(
    _tf_magnet_cost_superconducting_per_kam,
    "c22211",
    "c22212",
    "c22213",
    "c22214",
    "c22215",
    "c2221",
    reads={
        "i_tf_sc_mat": tfcoil,
        "j_crit_str_0": tfcoil,
        "j_crit_str_tf": tfcoil,
        "len_tf_coil": tfcoil,
        "n_tf_coil_turns": tfcoil,
        "m_tf_coil_copper": tfcoil,
        "n_tf_coils": tfcoil,
        "m_tf_coil_case": tfcoil,
        "aintmass": structure,
        "clgsmass": structure,
    },
)


# cottax node: `calculate_tf_magnet_cost_resistive` (Account 222.1,
# `.tfcoil.i_tf_sup != 1`).
def _tf_magnet_cost_resistive(
    lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
):
    return calculate_tf_magnet_cost_resistive(
        lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
    )


TfMagnetCostResistive = costs.function(
    _tf_magnet_cost_resistive,
    "c22211",
    "c22212",
    "c2221",
    "cpstcst",
    reads={
        "whtcp": tfcoil,
        "whttflgs": tfcoil,
        "itart": physics,
    },
)


# The PfMagnetCost family:
# The Account 222.2 PF-magnet family.
#
# `n_cs_pf_coils` (the topology's coil count) and `i_pf_conductor` are reads, last in
# every occupant: `.pf_coil.n_cs_pf_coils` and `.pf_coil.i_pf_conductor`.


# `supercond_cost_model == PER_KG` (0) with a central solenoid -- the whole
# calculation, shared by two occupants: the two conductor densities are the only thing
# they differ in, and each reads its own.
def _pf_magnet_cost_per_kg(
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
    den_pf_conductor,
    den_cs_conductor,
    uccu,
    cconfix,
    i_cs_superconductor,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
    n_cs_pf_coils,
    i_pf_conductor,
):
    return calculate_pf_magnet_cost_per_kg(
        n_cs_pf_coils,
        i_pf_conductor,
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
        den_pf_conductor,
        den_cs_conductor,
        uccu,
        cconfix,
        i_cs_superconductor,
        a_cs_cable_space,
        f_a_cs_void,
        fcuohsu,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    )


# `supercond_cost_model == PER_KG` (0) with a central solenoid, on the
# `(i_pf_superconductor, i_cs_superconductor) = (3, 1)` pair -- NbTi PF coils, ITER
# Nb3Sn CS.
PfMagnetCostPerKg = costs.function(
    _pf_magnet_cost_per_kg,
    "c22221",
    "c22222",
    "c22223",
    "c22224",
    "c2222",
    reads={
        "r_pf_coil_middle": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "i_pf_superconductor": pf_coil,
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "den_pf_conductor": V.tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1],
        "den_cs_conductor": V.tfcoil.dcond[I_CS_SUPERCONDUCTOR - 1],
        "i_cs_superconductor": pf_coil,
        "a_cs_cable_space": pf_coil,
        "f_a_cs_void": pf_coil,
        "fcuohsu": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
        "i_pf_conductor": pf_coil,
    },
)


# The same arm on the `(3, 5)` pair -- NbTi PF coils, WST Nb3Sn CS,
# `low_aspect_ratio_DEMO.IN.DAT`'s materials, `indat._pf_coil_system_arm` arm `1`.
PfMagnetCostPerKgCsWstNb3Sn = costs.function(
    _pf_magnet_cost_per_kg,
    "c22221",
    "c22222",
    "c22223",
    "c22224",
    "c2222",
    reads={
        "r_pf_coil_middle": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "i_pf_superconductor": pf_coil,
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "den_pf_conductor": V.tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1],
        "den_cs_conductor": V.tfcoil.dcond[I_CS_SUPERCONDUCTOR_WST_NB3SN - 1],
        "i_cs_superconductor": pf_coil,
        "a_cs_cable_space": pf_coil,
        "f_a_cs_void": pf_coil,
        "fcuohsu": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
        "i_pf_conductor": pf_coil,
    },
)


# `supercond_cost_model == PER_KG` (0) on a machine with `.build.iohcl == 0` --
# both tracked spherical tokamaks.
def _pf_magnet_cost_per_kg_no_central_solenoid(
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
    uccu,
    cconfix,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
    n_cs_pf_coils,
    i_pf_conductor,
):
    return calculate_pf_magnet_cost_per_kg_no_central_solenoid(
        n_cs_pf_coils,
        i_pf_conductor,
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
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    )


PfMagnetCostPerKgNoCentralSolenoid = costs.function(
    _pf_magnet_cost_per_kg_no_central_solenoid,
    "c22221",
    "c22222",
    "c22223",
    "c22224",
    "c2222",
    reads={
        "r_pf_coil_middle": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "i_pf_superconductor": pf_coil,
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
        "i_pf_conductor": pf_coil,
    },
)


# `supercond_cost_model == PER_KAM` (1) with a central solenoid -- strand cost
# scaled by critical current density.
def _pf_magnet_cost_per_kam(
    lsa,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    i_pf_superconductor,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
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
    n_cs_pf_coils,
    i_pf_conductor,
):
    return calculate_pf_magnet_cost_per_kam(
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        i_pf_superconductor,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
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
    )


PfMagnetCostPerKam = costs.function(
    _pf_magnet_cost_per_kam,
    "c22221",
    "c22222",
    "c22223",
    "c22224",
    "c2222",
    reads={
        "r_pf_coil_middle": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "i_pf_superconductor": pf_coil,
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "j_crit_str_0": tfcoil,
        "j_crit_str_pf": pf_coil,
        "i_cs_superconductor": pf_coil,
        "a_cs_cable_space": pf_coil,
        "f_a_cs_void": pf_coil,
        "fcuohsu": pf_coil,
        "j_crit_str_cs": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
        "i_pf_conductor": pf_coil,
    },
)


# `supercond_cost_model == PER_KAM` (1) on a machine with `.build.iohcl == 0`.
def _pf_magnet_cost_per_kam_no_central_solenoid(
    lsa,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    i_pf_superconductor,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    sc_mat_cost_0,
    j_crit_str_0,
    j_crit_str_pf,
    uccu,
    cconfix,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
    n_cs_pf_coils,
    i_pf_conductor,
):
    return calculate_pf_magnet_cost_per_kam_no_central_solenoid(
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        i_pf_superconductor,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        sc_mat_cost_0,
        j_crit_str_0,
        j_crit_str_pf,
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    )


PfMagnetCostPerKamNoCentralSolenoid = costs.function(
    _pf_magnet_cost_per_kam_no_central_solenoid,
    "c22221",
    "c22222",
    "c22223",
    "c22224",
    "c2222",
    reads={
        "r_pf_coil_middle": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "i_pf_superconductor": pf_coil,
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "j_crit_str_0": tfcoil,
        "j_crit_str_pf": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
        "i_pf_conductor": pf_coil,
    },
)


# cottax node: `calculate_magnets_cost` (Account 222 total).
def _magnets_cost(ife, c2221, c2222, c2223):
    return calculate_magnets_cost(ife, c2221, c2222, c2223)


MagnetsCost = costs.function(
    _magnets_cost,
    "c222",
    reads={
        "ife": ife,
    },
)


# cottax node: `calculate_power_injection_cost` (Account 223).
def _power_injection_cost(
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
    return calculate_power_injection_cost_magnetic_confinement(
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
    )


PowerInjectionCost = costs.function(
    _power_injection_cost,
    "c2231",
    "c2232",
    "c2233",
    "c223",
    "cdcost",
    reads={
        "p_hcd_ecrh_injected_total_mw": current_drive,
        "i_hcd_primary": current_drive,
        "p_hcd_lowhyb_injected_total_mw": current_drive,
        "p_beam_injected_mw": current_drive,
    },
)


# The EnergyStorageCost family:
# The family owning `.costs.c2253` (Account 225.3): one occupant per arm.


# `i_pulsed_plant == 0`: no storage, so `.costs.c2253` is zero and nothing is read.
EnergyStorageCostUnpulsed = states(costs, "c2253")


# The EnergyStorageCostPulsed family:
# `i_pulsed_plant == 1`: an ELECTROWATT thermal-storage design, scaled by net
# electric power.


# `.pulse.istore == ELECTROWATT_OPTION_1` (1) -- PROCESS's own default
# (`pulse_variables.py:16`).
def _energy_storage_cost_pulsed_electrowatt_option1(p_plant_electric_net_mw, fkind):
    return calculate_energy_storage_cost_electrowatt_option_1(
        p_plant_electric_net_mw, fkind
    )


EnergyStorageCostPulsedElectrowattOption1 = costs.function(
    _energy_storage_cost_pulsed_electrowatt_option1,
    "c2253",
    reads={
        "p_plant_electric_net_mw": heat_transport,
    },
)


# `.pulse.istore == ELECTROWATT_OPTION_2` (2).
def _energy_storage_cost_pulsed_electrowatt_option2(p_plant_electric_net_mw, fkind):
    return calculate_energy_storage_cost_electrowatt_option_2(
        p_plant_electric_net_mw, fkind
    )


EnergyStorageCostPulsedElectrowattOption2 = costs.function(
    _energy_storage_cost_pulsed_electrowatt_option2,
    "c2253",
    reads={
        "p_plant_electric_net_mw": heat_transport,
    },
)


# cottax node: `calculate_power_conditioning_cost` (Account 225 total).
def _power_conditioning_cost(ife, c2251, c2252, c2253):
    return calculate_power_conditioning_cost(ife, c2251, c2252, c2253)


PowerConditioningCost = costs.function(
    _power_conditioning_cost,
    "c225",
    reads={
        "ife": ife,
    },
)


# cottax node: `calculate_auxiliary_component_cooling_cost` (Account 2262).
def _auxiliary_component_cooling_cost(
    lsa,
    UCAHTS,
    p_hcd_electric_loss_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_tritium_plant_electric_mw,
    fachtmw,
    fkind,
):
    return calculate_auxiliary_component_cooling_cost_magnetic_confinement(
        lsa,
        UCAHTS,
        p_hcd_electric_loss_mw,
        p_cryo_plant_electric_mw,
        vachtmw,
        p_tritium_plant_electric_mw,
        fachtmw,
        fkind,
    )


AuxiliaryComponentCoolingCost = costs.function(
    _auxiliary_component_cooling_cost,
    "cppa",
    "c2262",
    reads={
        "p_hcd_electric_loss_mw": heat_transport,
        "p_cryo_plant_electric_mw": heat_transport,
        "vachtmw": heat_transport,
        "p_tritium_plant_electric_mw": heat_transport,
        "fachtmw": heat_transport,
    },
)


# cottax node: `calculate_cryogenic_system_cost` (Account 2263).
def _cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind):
    return calculate_cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind)


CryogenicSystemCost = costs.function(
    _cryogenic_system_cost,
    "c2263",
    reads={
        "temp_tf_cryo": tfcoil,
        "helpow": heat_transport,
    },
)


# cottax node: `calculate_heat_transport_system_cost` (Account 226 total).
def _heat_transport_system_cost(c2261, c2262, c2263):
    return calculate_heat_transport_system_cost(c2261, c2262, c2263)


HeatTransportSystemCost = costs.function(_heat_transport_system_cost, "c226")


# cottax node: `calculate_fuel_processing_cost` (Account 2272).
def _fuel_processing_cost(rndfuel, m_fuel_amu, UCFPR, fkind):
    return calculate_fuel_processing_cost_magnetic_confinement(
        rndfuel, m_fuel_amu, UCFPR, fkind
    )


FuelProcessingCost = physics.function(
    _fuel_processing_cost,
    "wtgpd",
    V.costs.c2272,
    reads={
        "UCFPR": costs,
        "fkind": costs,
    },
)


# cottax node: `calculate_atmospheric_recovery_cost` (Account 2273).
def _atmospheric_recovery_cost(f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind):
    return calculate_atmospheric_recovery_cost(
        f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind
    )


AtmosphericRecoveryCost = costs.function(
    _atmospheric_recovery_cost,
    "c2273",
    reads={
        "f_plasma_fuel_tritium": physics,
        "volrci": buildings,
        "wsvol": buildings,
    },
)


# cottax node: `calculate_fuel_handling_cost` (Account 227 total).
def _fuel_handling_cost(c2271, c2272, c2273, c2274):
    return calculate_fuel_handling_cost(c2271, c2272, c2273, c2274)


FuelHandlingCost = costs.function(_fuel_handling_cost, "c227")


# cottax node: `calculate_fusion_power_island_cost` (Account 22 total).
def _fusion_power_island_cost(c221, c222, c223, c224, c225, c226, c227, c228, c229):
    return calculate_fusion_power_island_cost(
        c221, c222, c223, c224, c225, c226, c227, c228, c229
    )


FusionPowerIslandCost = costs.function(_fusion_power_island_cost, "crctcore", "c22")


# cottax node: `calculate_total_plant_direct_cost`.
def _total_plant_direct_cost(c21, c22, c23, c24, c25, c26):
    return calculate_total_plant_direct_cost(c21, c22, c23, c24, c25, c26)


TotalPlantDirectCost = costs.function(_total_plant_direct_cost, "cdirt")


# cottax node: `calculate_constructed_cost`.
def _constructed_cost(cdirt, cindrt, ccont):
    return calculate_constructed_cost(cdirt, cindrt, ccont)


ConstructedCost = costs.function(_constructed_cost, "concost")


# The CostOfElectricity family:
# The `Costs.coelc` family -- sole producer of `.costs.coe`, the `i_figure_merit ==
# 6` objective.


# `.physics.itart == 0` -- the reference run's, and PROCESS's own default
# (`physics_variables.py:994`).
def _cost_of_electricity_conventional_aspect_ratio(
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
    return calculate_cost_of_electricity_conventional_aspect_ratio(
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
    )


CostOfElectricityConventionalAspectRatio = costs.function(
    _cost_of_electricity_conventional_aspect_ratio,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads={
        "p_plant_electric_net_mw": heat_transport,
        "t_plant_pulse_burn": times,
        "t_plant_pulse_total": times,
        "life_blkt": fwbs,
        "life_blkt_fpy": fwbs,
        "f_plasma_fuel_helium3": physics,
        "wtgpd": physics,
    },
)


# `.physics.itart == 1` -- the spherical tokamak, which additionally pays to
# replace its centrepost (`costs.py:2769-2783`).
def _cost_of_electricity_spherical_tokamak(
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
    return calculate_cost_of_electricity_spherical_tokamak(
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
    )


CostOfElectricitySphericalTokamak = costs.function(
    _cost_of_electricity_spherical_tokamak,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads={
        "p_plant_electric_net_mw": heat_transport,
        "t_plant_pulse_burn": times,
        "t_plant_pulse_total": times,
        "life_blkt": fwbs,
        "life_blkt_fpy": fwbs,
        "f_plasma_fuel_helium3": physics,
        "wtgpd": physics,
    },
)
