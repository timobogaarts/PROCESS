"""Pure-functional port of a self-contained subset of `process/models/costs/costs.py`
(the 1990-style cost model, registry unit #18).
"""

import inspect

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces import V

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
    calculate_auxiliary_facility_power_cost_formula,
    calculate_blanket_cost,
    calculate_blanket_cost_capital,
    calculate_blanket_cost_capital_then_fuel,
    calculate_blanket_cost_fuel,
    calculate_blanket_cost_magnetic_confinement,
    calculate_constructed_cost,
    calculate_cost_of_electricity,
    calculate_cost_of_electricity_conventional_aspect_ratio,
    calculate_cost_of_electricity_conventional_aspect_ratio_capital,
    calculate_cost_of_electricity_conventional_aspect_ratio_capital_then_fuel,
    calculate_cost_of_electricity_conventional_aspect_ratio_fuel,
    calculate_cost_of_electricity_spherical_tokamak,
    calculate_cost_of_electricity_spherical_tokamak_capital,
    calculate_cost_of_electricity_spherical_tokamak_capital_then_fuel,
    calculate_cost_of_electricity_spherical_tokamak_fuel,
    calculate_cryogenic_system_cost,
    calculate_cryogenic_system_cost_formula,
    calculate_diesel_generators_cost,
    calculate_diesel_generators_cost_formula,
    calculate_divertor_cost,
    calculate_divertor_cost_capital,
    calculate_divertor_cost_capital_then_fuel,
    calculate_divertor_cost_fuel,
    calculate_electric_plant_equipment_cost,
    calculate_energy_storage_cost,
    calculate_energy_storage_cost_electrowatt_option_1,
    calculate_energy_storage_cost_electrowatt_option_2,
    calculate_first_wall_cost,
    calculate_first_wall_cost_capital,
    calculate_first_wall_cost_capital_then_fuel,
    calculate_first_wall_cost_fuel,
    calculate_first_wall_cost_magnetic_confinement,
    calculate_fuel_handling_cost,
    calculate_fuel_processing_cost,
    calculate_fuel_processing_cost_magnetic_confinement,
    calculate_fuelling_system_cost,
    calculate_fusion_power_island_cost,
    calculate_heat_rejection_cost,
    calculate_heat_rejection_cost_non_reactor,
    calculate_heat_rejection_cost_reactor,
    calculate_heat_transport_system_cost,
    calculate_indirect_costs,
    calculate_indirect_costs_formula,
    calculate_instrumentation_and_control_cost,
    calculate_low_voltage_cost,
    calculate_low_voltage_cost_formula,
    calculate_magnets_cost,
    calculate_magnets_cost_magnetic_confinement,
    calculate_maintenance_equipment_cost,
    calculate_misc_plant_equipment_cost,
    calculate_misc_plant_equipment_cost_formula,
    calculate_nuclear_building_ventilation_cost,
    calculate_pf_coil_power_conditioning_cost,
    calculate_pf_magnet_cost,
    calculate_pf_magnet_cost_per_kam,
    calculate_pf_magnet_cost_per_kam_no_central_solenoid,
    calculate_pf_magnet_cost_per_kam_no_central_solenoid_superconducting,
    calculate_pf_magnet_cost_per_kam_superconducting,
    calculate_pf_magnet_cost_per_kg,
    calculate_pf_magnet_cost_per_kg_no_central_solenoid,
    calculate_pf_magnet_cost_per_kg_no_central_solenoid_superconducting,
    calculate_pf_magnet_cost_per_kg_superconducting,
    calculate_power_conditioning_cost,
    calculate_power_conditioning_cost_magnetic_confinement,
    calculate_power_injection_cost,
    calculate_power_injection_cost_capital,
    calculate_power_injection_cost_fuel,
    calculate_power_injection_cost_magnetic_confinement,
    calculate_reactor_cooling_system_cost,
    calculate_reactor_cooling_system_cost_formula,
    calculate_reactor_cost,
    calculate_reactor_structure_cost,
    calculate_reactor_structure_cost_formula,
    calculate_shield_cost,
    calculate_shield_cost_magnetic_confinement,
    calculate_structures_cost,
    calculate_structures_cost_non_reactor,
    calculate_structures_cost_reactor,
    calculate_switchyard_cost,
    calculate_switchyard_cost_formula,
    calculate_tf_coil_power_conditioning_cost,
    calculate_tf_coil_power_conditioning_cost_resistive,
    calculate_tf_coil_power_conditioning_cost_superconducting,
    calculate_tf_magnet_cost_resistive,
    calculate_tf_magnet_cost_resistive_formula,
    calculate_tf_magnet_cost_superconducting,
    calculate_tf_magnet_cost_superconducting_per_kam,
    calculate_tf_magnet_cost_superconducting_per_kg,
    calculate_total_plant_direct_cost,
    calculate_transformers_cost,
    calculate_transformers_cost_formula,
    calculate_turbine_plant_equipment_cost,
    calculate_turbine_plant_equipment_cost_non_reactor,
    calculate_turbine_plant_equipment_cost_reactor,
    calculate_vacuum_system_cost,
    calculate_vacuum_system_cost_for_pump,
    calculate_vacuum_vessel_assembly_cost,
    calculate_vacuum_vessel_assembly_cost_formula,
    convert_fpy_to_calendar,
    convert_fpy_to_calendar_conventional_aspect_ratio,
    convert_fpy_to_calendar_spherical_tokamak,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.vocabulary import PFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public"
# once one is present, so this lists every public name this module resolved before
# step 2 of `_audit/formulas_split.md` moved the pure functions out -- not just the
# handful actually unused here (see `power/electric_production.py`'s commit for why
# a partial list is the wrong move).

__all__ = [
    "AtmosphericRecoveryCost",
    "AuxiliaryComponentCoolingCost",
    "AuxiliaryFacilityPowerCost",
    "BlanketCostCapital",
    "BlanketCostCapitalThenFuel",
    "BlanketCostFuel",
    "ConstructedCost",
    "ConvertFpyToCalendarConventionalAspectRatio",
    "ConvertFpyToCalendarSphericalTokamak",
    "CostOfElectricityConventionalAspectRatioCapital",
    "CostOfElectricityConventionalAspectRatioCapitalThenFuel",
    "CostOfElectricityConventionalAspectRatioFuel",
    "CostOfElectricitySphericalTokamakCapital",
    "CostOfElectricitySphericalTokamakCapitalThenFuel",
    "CostOfElectricitySphericalTokamakFuel",
    "CryogenicSystemCost",
    "DieselGeneratorsCost",
    "DivertorCostCapital",
    "DivertorCostCapitalThenFuel",
    "DivertorCostFuel",
    "ElectricPlantEquipmentCost",
    "EnergyStorageCostPulsedElectrowattOption1",
    "EnergyStorageCostPulsedElectrowattOption2",
    "EnergyStorageCostUnpulsed",
    "FirstWallCostCapital",
    "FirstWallCostCapitalThenFuel",
    "FirstWallCostFuel",
    "FuelHandlingCost",
    "FuelProcessingCost",
    "FuellingSystemCost",
    "FusionPowerIslandCost",
    "HeatRejectionCostNonReactor",
    "HeatRejectionCostReactor",
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
    "PfMagnetCostPerKgNoCentralSolenoid",
    "PowerConditioningCost",
    "PowerInjectionCostCapitalIonCyclotron",
    "PowerInjectionCostCapitalLowerHybrid",
    "PowerInjectionCostFuelIonCyclotron",
    "PowerInjectionCostFuelLowerHybrid",
    "ReactorCoolingSystemCost",
    "ReactorCost",
    "ReactorStructureCost",
    "ShieldCost",
    "StructuresCostNonReactor",
    "StructuresCostReactor",
    "SwitchyardCost",
    "TfCoilPowerConditioningCostResistive",
    "TfCoilPowerConditioningCostSuperconducting",
    "TfMagnetCostResistive",
    "TfMagnetCostSuperconductingPerKam",
    "TfMagnetCostSuperconductingPerKg",
    "TotalPlantDirectCost",
    "TransformersCost",
    "TurbinePlantEquipmentCostNonReactor",
    "TurbinePlantEquipmentCostReactor",
    "VacuumSystemCostCryopump",
    "VacuumSystemCostTurbomolecular",
    "VacuumVesselAssemblyCost",
    "buildings",
    "calculate_atmospheric_recovery_cost",
    "calculate_auxiliary_component_cooling_cost",
    "calculate_auxiliary_component_cooling_cost_magnetic_confinement",
    "calculate_auxiliary_facility_power_cost",
    "calculate_blanket_cost",
    "calculate_blanket_cost_capital",
    "calculate_blanket_cost_capital_then_fuel",
    "calculate_blanket_cost_fuel",
    "calculate_blanket_cost_magnetic_confinement",
    "calculate_constructed_cost",
    "calculate_cost_of_electricity",
    "calculate_cost_of_electricity_conventional_aspect_ratio",
    "calculate_cost_of_electricity_conventional_aspect_ratio_capital",
    "calculate_cost_of_electricity_conventional_aspect_ratio_capital_then_fuel",
    "calculate_cost_of_electricity_conventional_aspect_ratio_fuel",
    "calculate_cost_of_electricity_spherical_tokamak",
    "calculate_cost_of_electricity_spherical_tokamak_capital",
    "calculate_cost_of_electricity_spherical_tokamak_capital_then_fuel",
    "calculate_cost_of_electricity_spherical_tokamak_fuel",
    "calculate_cryogenic_system_cost",
    "calculate_diesel_generators_cost",
    "calculate_divertor_cost",
    "calculate_divertor_cost_capital",
    "calculate_divertor_cost_capital_then_fuel",
    "calculate_divertor_cost_fuel",
    "calculate_electric_plant_equipment_cost",
    "calculate_energy_storage_cost",
    "calculate_energy_storage_cost_electrowatt_option_1",
    "calculate_energy_storage_cost_electrowatt_option_2",
    "calculate_first_wall_cost",
    "calculate_first_wall_cost_capital",
    "calculate_first_wall_cost_capital_then_fuel",
    "calculate_first_wall_cost_fuel",
    "calculate_first_wall_cost_magnetic_confinement",
    "calculate_fuel_handling_cost",
    "calculate_fuel_processing_cost",
    "calculate_fuel_processing_cost_magnetic_confinement",
    "calculate_fuelling_system_cost",
    "calculate_fusion_power_island_cost",
    "calculate_heat_rejection_cost",
    "calculate_heat_rejection_cost_non_reactor",
    "calculate_heat_rejection_cost_reactor",
    "calculate_heat_transport_system_cost",
    "calculate_indirect_costs",
    "calculate_instrumentation_and_control_cost",
    "calculate_low_voltage_cost",
    "calculate_magnets_cost",
    "calculate_magnets_cost_magnetic_confinement",
    "calculate_maintenance_equipment_cost",
    "calculate_misc_plant_equipment_cost",
    "calculate_nuclear_building_ventilation_cost",
    "calculate_pf_coil_power_conditioning_cost",
    "calculate_pf_magnet_cost",
    "calculate_pf_magnet_cost_per_kam",
    "calculate_pf_magnet_cost_per_kam_no_central_solenoid",
    "calculate_pf_magnet_cost_per_kam_no_central_solenoid_superconducting",
    "calculate_pf_magnet_cost_per_kam_superconducting",
    "calculate_pf_magnet_cost_per_kg",
    "calculate_pf_magnet_cost_per_kg_no_central_solenoid",
    "calculate_pf_magnet_cost_per_kg_no_central_solenoid_superconducting",
    "calculate_pf_magnet_cost_per_kg_superconducting",
    "calculate_power_conditioning_cost",
    "calculate_power_conditioning_cost_magnetic_confinement",
    "calculate_power_injection_cost",
    "calculate_power_injection_cost_capital",
    "calculate_power_injection_cost_fuel",
    "calculate_power_injection_cost_magnetic_confinement",
    "calculate_reactor_cooling_system_cost",
    "calculate_reactor_cost",
    "calculate_reactor_structure_cost",
    "calculate_shield_cost",
    "calculate_shield_cost_magnetic_confinement",
    "calculate_structures_cost",
    "calculate_structures_cost_non_reactor",
    "calculate_structures_cost_reactor",
    "calculate_switchyard_cost",
    "calculate_tf_coil_power_conditioning_cost",
    "calculate_tf_coil_power_conditioning_cost_resistive",
    "calculate_tf_coil_power_conditioning_cost_superconducting",
    "calculate_tf_magnet_cost_resistive",
    "calculate_tf_magnet_cost_superconducting",
    "calculate_tf_magnet_cost_superconducting_per_kam",
    "calculate_tf_magnet_cost_superconducting_per_kg",
    "calculate_total_plant_direct_cost",
    "calculate_transformers_cost",
    "calculate_turbine_plant_equipment_cost",
    "calculate_turbine_plant_equipment_cost_non_reactor",
    "calculate_turbine_plant_equipment_cost_reactor",
    "calculate_vacuum_system_cost",
    "calculate_vacuum_system_cost_for_pump",
    "calculate_vacuum_vessel_assembly_cost",
    "convert_fpy_to_calendar",
    "convert_fpy_to_calendar_conventional_aspect_ratio",
    "convert_fpy_to_calendar_spherical_tokamak",
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


def _restricted(fn, reads):
    """`reads`, less the parameters `fn` does not take: one table of places shared by
    the variants of one account, each variant reading only what its branch reads."""
    taken = inspect.signature(fn).parameters
    return {name: place for name, place in reads.items() if name in taken}


# cottax node: `convert_fpy_to_calendar`.
_CONVERT_FPY_TO_CALENDAR_READS = {
    "life_plant": costs,
    "f_t_plant_available": costs,
    "life_div_fpy": costs,
    "cplife": costs,
}

# `.physics.itart != 1`: no centrepost, `cplife_cal` is zero.
ConvertFpyToCalendarConventionalAspectRatio = fwbs.function(
    convert_fpy_to_calendar_conventional_aspect_ratio,
    "life_blkt",
    V.costs.cdrlife_cal,
    V.costs.life_div,
    V.costs.cplife_cal,
    reads=_restricted(
        convert_fpy_to_calendar_conventional_aspect_ratio,
        _CONVERT_FPY_TO_CALENDAR_READS,
    ),
)

# `.physics.itart == 1`: the centrepost's calendar lifetime too.
ConvertFpyToCalendarSphericalTokamak = fwbs.function(
    convert_fpy_to_calendar_spherical_tokamak,
    "life_blkt",
    V.costs.cdrlife_cal,
    V.costs.life_div,
    V.costs.cplife_cal,
    reads=_restricted(
        convert_fpy_to_calendar_spherical_tokamak, _CONVERT_FPY_TO_CALENDAR_READS
    ),
)


# cottax node: `calculate_structures_cost` (Account 21).
_STRUCTURES_COST_READS = {
    "rbvol": buildings,
    "rmbvol": buildings,
    "wsvol": buildings,
    "triv": buildings,
    "elevol": buildings,
    "admvol": buildings,
    "convol": buildings,
    "shovol": buildings,
    "cryvol": buildings,
}

# `.costs.ireactor == 1`: with a turbine building.
StructuresCostReactor = costs.function(
    calculate_structures_cost_reactor,
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
    reads=_STRUCTURES_COST_READS,
)

# `.costs.ireactor == 0`: no turbine building, `c213` zero.
StructuresCostNonReactor = costs.function(
    calculate_structures_cost_non_reactor,
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
    reads=_STRUCTURES_COST_READS,
)


# cottax node: `calculate_indirect_costs` (Account 9).
def _indirect_costs(cfind_lsa, cdirt, cowner, fcontng):
    return calculate_indirect_costs_formula(cfind_lsa, cdirt, cowner, fcontng)


IndirectCosts = costs.function(_indirect_costs, "cindrt", "ccont")


# cottax node: `calculate_reactor_structure_cost` (Account 221.4).
def _reactor_structure_cost(gsmass, UCGSS, cmlsa_reactor_structure, fkind):
    return calculate_reactor_structure_cost_formula(
        gsmass, UCGSS, cmlsa_reactor_structure, fkind
    )


ReactorStructureCost = costs.function(
    _reactor_structure_cost,
    "c2214",
    reads={
        "gsmass": structure,
    },
)


# cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3).
def _vacuum_vessel_assembly_cost(m_vv, uccryo, cmlsa_magnets, fkind):
    return calculate_vacuum_vessel_assembly_cost_formula(
        m_vv, uccryo, cmlsa_magnets, fkind
    )


VacuumVesselAssemblyCost = costs.function(
    _vacuum_vessel_assembly_cost,
    "c2223",
    reads={
        "m_vv": fwbs,
    },
)


# cottax node: `calculate_divertor_cost` (Account 221.5).
# One occupant per `.costs.ifueltyp` (the magnetic-confinement arm; `ife == 1` is
# refused by the converter): whether the divertor is a capital or a fuel cost.
DivertorCostCapital = costs.function(
    calculate_divertor_cost_capital,
    "c2215",
    "divcst",
    reads={"a_div_surface_total": divertor},
)
DivertorCostFuel = costs.function(
    calculate_divertor_cost_fuel,
    "c2215",
    "divcst",
    reads={"a_div_surface_total": divertor},
)
DivertorCostCapitalThenFuel = costs.function(
    calculate_divertor_cost_capital_then_fuel,
    "c2215",
    "divcst",
    reads={"a_div_surface_total": divertor},
)


# cottax node: `calculate_vacuum_system_cost` (Account 224).
# `.vacuum.i_vacuum_pump_type == 0`: turbomolecular pumps, `.costs.UCTPMP`.
VacuumSystemCostTurbomolecular = costs.function(
    calculate_vacuum_system_cost_for_pump,
    "c2241",
    "c2242",
    "c2243",
    "c2244",
    "c2245",
    "c2246",
    "c224",
    reads={
        "n_vac_pumps_high": vacuum,
        "uc_vac_pump": V.costs.UCTPMP,
        "n_vv_vacuum_ducts": vacuum,
        "dlscal": vacuum,
        "dia_vv_vacuum_ducts": vacuum,
        "m_vv_vacuum_duct_shield": vacuum,
    },
)

# `.vacuum.i_vacuum_pump_type == 1`: compound cryopumps, `.costs.UCCPMP`.
VacuumSystemCostCryopump = costs.function(
    calculate_vacuum_system_cost_for_pump,
    "c2241",
    "c2242",
    "c2243",
    "c2244",
    "c2245",
    "c2246",
    "c224",
    reads={
        "n_vac_pumps_high": vacuum,
        "uc_vac_pump": V.costs.UCCPMP,
        "n_vv_vacuum_ducts": vacuum,
        "dlscal": vacuum,
        "dia_vv_vacuum_ducts": vacuum,
        "m_vv_vacuum_duct_shield": vacuum,
    },
)


# cottax node: `calculate_tf_coil_power_conditioning_cost` (Account 225.1).
_TF_COIL_POWER_CONDITIONING_COST_READS = {
    "tfckw": tfcoil,
    "tfcmw": tfcoil,
    "n_tf_coils": tfcoil,
    "c_tf_turn": tfcoil,
    "v_tf_coil_dump_quench_kv": tfcoil,
    "e_tf_magnetic_stored_total_gj": tfcoil,
    "m_tf_bus": tfcoil,
    "len_tf_bus": tfcoil,
}

# `.tfcoil.i_tf_sup == 1`: superconducting, with breakers and a superconducting bus.
TfCoilPowerConditioningCostSuperconducting = costs.function(
    calculate_tf_coil_power_conditioning_cost_superconducting,
    "c22511",
    "c22512",
    "c22513",
    "c22514",
    "c22515",
    "c2251",
    reads=_restricted(
        calculate_tf_coil_power_conditioning_cost_superconducting,
        _TF_COIL_POWER_CONDITIONING_COST_READS,
    ),
)

# `.tfcoil.i_tf_sup == 0`: resistive, bussing costed by mass.
TfCoilPowerConditioningCostResistive = costs.function(
    calculate_tf_coil_power_conditioning_cost_resistive,
    "c22511",
    "c22512",
    "c22513",
    "c22514",
    "c22515",
    "c2251",
    reads=_restricted(
        calculate_tf_coil_power_conditioning_cost_resistive,
        _TF_COIL_POWER_CONDITIONING_COST_READS,
    ),
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
    uchts_blkt_coolant,
    p_fw_div_heat_deposited_mw,
    p_blkt_nuclear_heat_total_mw,
    p_shld_nuclear_heat_mw,
    cmlsa_heat_transport,
    fkind,
    UCPHX,
    n_primary_heat_exchangers,
    p_plant_primary_heat_mw,
):
    return calculate_reactor_cooling_system_cost_formula(
        uchts_blkt_coolant,
        p_fw_div_heat_deposited_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        cmlsa_heat_transport,
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
# `.costs.ireactor == 1`.
TurbinePlantEquipmentCostReactor = costs.function(
    calculate_turbine_plant_equipment_cost_reactor,
    "c23",
    reads={
        "p_plant_electric_gross_mw": heat_transport,
    },
)

# `.costs.ireactor == 0`: no turbine plant, `c23` zero and nothing read.
TurbinePlantEquipmentCostNonReactor = costs.function(
    calculate_turbine_plant_equipment_cost_non_reactor, "c23"
)


# cottax node: `calculate_switchyard_cost` (Account 241).
def _switchyard_cost(UCSWYD, cmlsa_electrical_plant):
    return calculate_switchyard_cost_formula(UCSWYD, cmlsa_electrical_plant)


SwitchyardCost = costs.function(_switchyard_cost, "c241")


# cottax node: `calculate_transformers_cost` (Account 242).
def _transformers_cost(
    UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, cmlsa_electrical_plant
):
    return calculate_transformers_cost_formula(
        UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, cmlsa_electrical_plant
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
def _low_voltage_cost(UCLV, tlvpmw, cmlsa_electrical_plant):
    return calculate_low_voltage_cost_formula(UCLV, tlvpmw, cmlsa_electrical_plant)


LowVoltageCost = costs.function(
    _low_voltage_cost,
    "c243",
    reads={
        "tlvpmw": heat_transport,
    },
)


# cottax node: `calculate_diesel_generators_cost` (Account 244).
def _diesel_generators_cost(UCDGEN, cmlsa_electrical_plant):
    return calculate_diesel_generators_cost_formula(UCDGEN, cmlsa_electrical_plant)


DieselGeneratorsCost = costs.function(_diesel_generators_cost, "c244")


# cottax node: `calculate_auxiliary_facility_power_cost` (Account 245).
def _auxiliary_facility_power_cost(UCAF, cmlsa_electrical_plant):
    return calculate_auxiliary_facility_power_cost_formula(UCAF, cmlsa_electrical_plant)


AuxiliaryFacilityPowerCost = costs.function(_auxiliary_facility_power_cost, "c245")


# cottax node: `calculate_electric_plant_equipment_cost` (Account 24, total).
def _electric_plant_equipment_cost(c241, c242, c243, c244, c245):
    return calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245)


ElectricPlantEquipmentCost = costs.function(_electric_plant_equipment_cost, "c24")


# cottax node: `calculate_misc_plant_equipment_cost` (Account 25).
def _misc_plant_equipment_cost(ucmisc, cmlsa_misc_plant):
    return calculate_misc_plant_equipment_cost_formula(ucmisc, cmlsa_misc_plant)


MiscPlantEquipmentCost = costs.function(_misc_plant_equipment_cost, "c25")


# cottax node: `calculate_heat_rejection_cost` (Account 26).
# `.costs.ireactor == 1`: the primary heat not turned into electricity.
HeatRejectionCostReactor = costs.function(
    calculate_heat_rejection_cost_reactor,
    "c26",
    reads={
        "p_plant_primary_heat_mw": heat_transport,
        "p_plant_electric_gross_mw": heat_transport,
    },
)

# `.costs.ireactor == 0`: fusion, heating and TF power, all rejected.
HeatRejectionCostNonReactor = costs.function(
    calculate_heat_rejection_cost_non_reactor,
    "c26",
    reads={
        "p_fusion_total_mw": physics,
        "p_hcd_electric_total_mw": heat_transport,
        "tfcmw": tfcoil,
    },
)


# --------------------------------------------------------------------------------------
# Second porting wave's nodes: the `.costs.coe` chain. Registered in `total_process.py`
# under `.costs.i_cost_model == 0`; see that switch's own comment block.
# --------------------------------------------------------------------------------------


# cottax node: `calculate_first_wall_cost` (Account 221.1).
# One occupant per `.costs.ifueltyp`, as `DivertorCost*`.
FirstWallCostCapital = costs.function(
    calculate_first_wall_cost_capital,
    "c2211",
    "fwallcst",
    reads={"a_fw_total": first_wall},
)
FirstWallCostFuel = costs.function(
    calculate_first_wall_cost_fuel,
    "c2211",
    "fwallcst",
    reads={"a_fw_total": first_wall},
)
FirstWallCostCapitalThenFuel = costs.function(
    calculate_first_wall_cost_capital_then_fuel,
    "c2211",
    "fwallcst",
    reads={"a_fw_total": first_wall},
)


# cottax node: `calculate_blanket_cost` (Account 221.2).
_BLANKET_COST_READS = {
    "m_blkt_beryllium": fwbs,
    "m_blkt_li2o": fwbs,
    "m_blkt_steel_total": fwbs,
    "m_blkt_vanadium": fwbs,
}

# One occupant per `.costs.ifueltyp`, as `DivertorCost*`.
BlanketCostCapital = costs.function(
    calculate_blanket_cost_capital,
    "c22121",
    "c22122",
    "c22123",
    "c22124",
    "c22125",
    "c22126",
    "c22127",
    "c2212",
    "blkcst",
    reads=_BLANKET_COST_READS,
)
BlanketCostFuel = costs.function(
    calculate_blanket_cost_fuel,
    "c22121",
    "c22122",
    "c22123",
    "c22124",
    "c22125",
    "c22126",
    "c22127",
    "c2212",
    "blkcst",
    reads=_BLANKET_COST_READS,
)
BlanketCostCapitalThenFuel = costs.function(
    calculate_blanket_cost_capital_then_fuel,
    "c22121",
    "c22122",
    "c22123",
    "c22124",
    "c22125",
    "c22126",
    "c22127",
    "c2212",
    "blkcst",
    reads=_BLANKET_COST_READS,
)


# cottax node: `calculate_shield_cost` (Account 221.3).
def _shield_cost(
    cmlsa_first_wall_blanket_shield, whtshld, ucshld, wpenshld, ucpens, fkind
):
    return calculate_shield_cost_magnetic_confinement(
        cmlsa_first_wall_blanket_shield, whtshld, ucshld, wpenshld, ucpens, fkind
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
    cmlsa_magnets,
    ucsc_tf,
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
        cmlsa_magnets,
        ucsc_tf,
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
    cmlsa_magnets,
    sc_mat_cost_0_tf,
    j_crit_str_0_tf,
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
        cmlsa_magnets,
        sc_mat_cost_0_tf,
        j_crit_str_0_tf,
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
        "j_crit_str_0_tf": tfcoil,
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
    cmlsa_magnets, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
):
    return calculate_tf_magnet_cost_resistive_formula(
        cmlsa_magnets, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
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
# `n_cs_pf_coils` (the topology's coil count) is a read, last in every occupant. Every
# occupant is the superconducting arm: the converter refuses resistive PF coils.


# `supercond_cost_model == PER_KG` (0) with a central solenoid. The PF and CS
# superconductors' unit costs and densities are the entries at their material indices,
# derived from a configuration's values (`.costs.ucsc_pf`, `.costs.ucsc_cs`,
# `.pf_coil.den_pf_conductor`, `.pf_coil.den_cs_conductor`), so one occupant serves every
# material pair.
def _pf_magnet_cost_per_kg(
    cmlsa_magnets,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    ucsc_pf,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    den_pf_conductor,
    den_cs_conductor,
    uccu,
    cconfix,
    ucsc_cs,
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
):
    return calculate_pf_magnet_cost_per_kg_superconducting(
        n_cs_pf_coils,
        cmlsa_magnets,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        ucsc_pf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        den_pf_conductor,
        den_cs_conductor,
        uccu,
        cconfix,
        ucsc_cs,
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
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "den_pf_conductor": pf_coil,
        "den_cs_conductor": pf_coil,
        "a_cs_cable_space": pf_coil,
        "f_a_cs_void": pf_coil,
        "fcuohsu": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
    },
)


# `supercond_cost_model == PER_KG` (0) on a machine with `.build.iohcl == 0` --
# both tracked spherical tokamaks.
def _pf_magnet_cost_per_kg_no_central_solenoid(
    cmlsa_magnets,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    ucsc_pf,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    den_pf_conductor,
    uccu,
    cconfix,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
    n_cs_pf_coils,
):
    return calculate_pf_magnet_cost_per_kg_no_central_solenoid_superconducting(
        n_cs_pf_coils,
        cmlsa_magnets,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        ucsc_pf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        den_pf_conductor,
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
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "den_pf_conductor": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
    },
)


# `supercond_cost_model == PER_KAM` (1) with a central solenoid -- strand cost
# scaled by critical current density.
def _pf_magnet_cost_per_kam(
    cmlsa_magnets,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    sc_mat_cost_0_pf,
    j_crit_str_0_pf,
    j_crit_str_pf,
    uccu,
    cconfix,
    sc_mat_cost_0_cs,
    j_crit_str_0_cs,
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
):
    return calculate_pf_magnet_cost_per_kam_superconducting(
        n_cs_pf_coils,
        cmlsa_magnets,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        sc_mat_cost_0_pf,
        j_crit_str_0_pf,
        j_crit_str_pf,
        uccu,
        cconfix,
        sc_mat_cost_0_cs,
        j_crit_str_0_cs,
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
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "j_crit_str_0_pf": pf_coil,
        "j_crit_str_pf": pf_coil,
        "j_crit_str_0_cs": pf_coil,
        "a_cs_cable_space": pf_coil,
        "f_a_cs_void": pf_coil,
        "fcuohsu": pf_coil,
        "j_crit_str_cs": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
    },
)


# `supercond_cost_model == PER_KAM` (1) on a machine with `.build.iohcl == 0`.
def _pf_magnet_cost_per_kam_no_central_solenoid(
    cmlsa_magnets,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    sc_mat_cost_0_pf,
    j_crit_str_0_pf,
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
):
    return calculate_pf_magnet_cost_per_kam_no_central_solenoid_superconducting(
        n_cs_pf_coils,
        cmlsa_magnets,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        sc_mat_cost_0_pf,
        j_crit_str_0_pf,
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
        "fcupfsu": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "j_crit_str_0_pf": pf_coil,
        "j_crit_str_pf": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "fncmass": structure,
        "n_cs_pf_coils": pf_coil,
    },
)


# cottax node: `calculate_magnets_cost` (Account 222 total).
MagnetsCost = costs.function(calculate_magnets_cost_magnetic_confinement, "c222")


# cottax node: `calculate_power_injection_cost` (Account 223).
# `.costs.ifueltyp != 1`, `i_hcd_primary != 2` (Account 223.2 at `.costs.uclh`).
PowerInjectionCostCapitalLowerHybrid = costs.function(
    calculate_power_injection_cost_capital,
    "c2231",
    "c2232",
    "c2233",
    "c223",
    "cdcost",
    reads=_restricted(
        calculate_power_injection_cost_capital,
        {
            "p_hcd_ecrh_injected_total_mw": current_drive,
            "uc_lower_hybrid": V.costs.uclh,
            "p_hcd_lowhyb_injected_total_mw": current_drive,
            "p_beam_injected_mw": current_drive,
        },
    ),
)

# `.costs.ifueltyp != 1`, `i_hcd_primary == 2` (Account 223.2 at `.costs.ucich`).
PowerInjectionCostCapitalIonCyclotron = costs.function(
    calculate_power_injection_cost_capital,
    "c2231",
    "c2232",
    "c2233",
    "c223",
    "cdcost",
    reads=_restricted(
        calculate_power_injection_cost_capital,
        {
            "p_hcd_ecrh_injected_total_mw": current_drive,
            "uc_lower_hybrid": V.costs.ucich,
            "p_hcd_lowhyb_injected_total_mw": current_drive,
            "p_beam_injected_mw": current_drive,
        },
    ),
)

# `.costs.ifueltyp == 1`, `i_hcd_primary != 2` (Account 223.2 at `.costs.uclh`).
PowerInjectionCostFuelLowerHybrid = costs.function(
    calculate_power_injection_cost_fuel,
    "c2231",
    "c2232",
    "c2233",
    "c223",
    "cdcost",
    reads=_restricted(
        calculate_power_injection_cost_fuel,
        {
            "p_hcd_ecrh_injected_total_mw": current_drive,
            "uc_lower_hybrid": V.costs.uclh,
            "p_hcd_lowhyb_injected_total_mw": current_drive,
            "p_beam_injected_mw": current_drive,
        },
    ),
)

# `.costs.ifueltyp == 1`, `i_hcd_primary == 2` (Account 223.2 at `.costs.ucich`).
PowerInjectionCostFuelIonCyclotron = costs.function(
    calculate_power_injection_cost_fuel,
    "c2231",
    "c2232",
    "c2233",
    "c223",
    "cdcost",
    reads=_restricted(
        calculate_power_injection_cost_fuel,
        {
            "p_hcd_ecrh_injected_total_mw": current_drive,
            "uc_lower_hybrid": V.costs.ucich,
            "p_hcd_lowhyb_injected_total_mw": current_drive,
            "p_beam_injected_mw": current_drive,
        },
    ),
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
PowerConditioningCost = costs.function(
    calculate_power_conditioning_cost_magnetic_confinement, "c225"
)


# cottax node: `calculate_auxiliary_component_cooling_cost` (Account 2262).
def _auxiliary_component_cooling_cost(
    cmlsa_heat_transport,
    UCAHTS,
    p_hcd_electric_loss_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_tritium_plant_electric_mw,
    fachtmw,
    fkind,
):
    return calculate_auxiliary_component_cooling_cost_magnetic_confinement(
        cmlsa_heat_transport,
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
def _cryogenic_system_cost(cmlsa_heat_transport, uccry, temp_tf_cryo, helpow, fkind):
    return calculate_cryogenic_system_cost_formula(
        cmlsa_heat_transport, uccry, temp_tf_cryo, helpow, fkind
    )


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
_COST_OF_ELECTRICITY_READS = {
    "p_plant_electric_net_mw": heat_transport,
    "t_plant_pulse_burn": times,
    "t_plant_pulse_total": times,
    "life_blkt": fwbs,
    "life_blkt_fpy": fwbs,
    "f_plasma_fuel_helium3": physics,
    "wtgpd": physics,
}

# `.physics.itart == 0`; `.costs.ifueltyp == 0`.
CostOfElectricityConventionalAspectRatioCapital = costs.function(
    calculate_cost_of_electricity_conventional_aspect_ratio_capital,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_conventional_aspect_ratio_capital,
        _COST_OF_ELECTRICITY_READS,
    ),
)

# `.physics.itart == 0`; `ifueltyp == 1`.
CostOfElectricityConventionalAspectRatioFuel = costs.function(
    calculate_cost_of_electricity_conventional_aspect_ratio_fuel,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_conventional_aspect_ratio_fuel,
        _COST_OF_ELECTRICITY_READS,
    ),
)

# `.physics.itart == 0`; `ifueltyp == 2`.
CostOfElectricityConventionalAspectRatioCapitalThenFuel = costs.function(
    calculate_cost_of_electricity_conventional_aspect_ratio_capital_then_fuel,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_conventional_aspect_ratio_capital_then_fuel,
        _COST_OF_ELECTRICITY_READS,
    ),
)

# `.physics.itart == 1`, the centrepost replaced too; `.costs.ifueltyp == 0`.
CostOfElectricitySphericalTokamakCapital = costs.function(
    calculate_cost_of_electricity_spherical_tokamak_capital,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_spherical_tokamak_capital,
        _COST_OF_ELECTRICITY_READS,
    ),
)

# `.physics.itart == 1`, the centrepost replaced too; `ifueltyp == 1`.
CostOfElectricitySphericalTokamakFuel = costs.function(
    calculate_cost_of_electricity_spherical_tokamak_fuel,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_spherical_tokamak_fuel, _COST_OF_ELECTRICITY_READS
    ),
)

# `.physics.itart == 1`, the centrepost replaced too; `ifueltyp == 2`.
CostOfElectricitySphericalTokamakCapitalThenFuel = costs.function(
    calculate_cost_of_electricity_spherical_tokamak_capital_then_fuel,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
    reads=_restricted(
        calculate_cost_of_electricity_spherical_tokamak_capital_then_fuel,
        _COST_OF_ELECTRICITY_READS,
    ),
)


# `.physics.itart == 1` -- the spherical tokamak, which additionally pays to
# replace its centrepost (`costs.py:2769-2783`).
