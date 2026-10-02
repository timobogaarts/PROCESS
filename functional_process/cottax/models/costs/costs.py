"""Pure-functional port of a self-contained subset of `process/models/costs/costs.py`
(the 1990-style cost model, registry unit #18).
"""

import inspect

import equinox as eqx
import jax.numpy as jnp
from cottax.cx import V

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


# `.physics.itart != 1`: no centrepost, `cplife_cal` is zero.
ConvertFpyToCalendarConventionalAspectRatio = fwbs.function(
    convert_fpy_to_calendar_conventional_aspect_ratio,
    "life_blkt",
    V.costs.cdrlife_cal,
    V.costs.life_div,
    V.costs.cplife_cal,
)

# `.physics.itart == 1`: the centrepost's calendar lifetime too.
ConvertFpyToCalendarSphericalTokamak = fwbs.function(
    convert_fpy_to_calendar_spherical_tokamak,
    "life_blkt",
    V.costs.cdrlife_cal,
    V.costs.life_div,
    V.costs.cplife_cal,
)


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
)


# cottax node: `calculate_indirect_costs` (Account 9).
IndirectCosts = costs.function(calculate_indirect_costs_formula, "cindrt", "ccont")


# cottax node: `calculate_reactor_structure_cost` (Account 221.4).
ReactorStructureCost = costs.function(
    calculate_reactor_structure_cost_formula,
    "c2214",
)


# cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3).
VacuumVesselAssemblyCost = costs.function(
    calculate_vacuum_vessel_assembly_cost_formula,
    "c2223",
)


# cottax node: `calculate_divertor_cost` (Account 221.5).
# One occupant per `.costs.ifueltyp` (the magnetic-confinement arm; `ife == 1` is
# refused by the converter): whether the divertor is a capital or a fuel cost.
DivertorCostCapital = costs.function(
    calculate_divertor_cost_capital,
    "c2215",
    "divcst",
)
DivertorCostFuel = costs.function(
    calculate_divertor_cost_fuel,
    "c2215",
    "divcst",
)
DivertorCostCapitalThenFuel = costs.function(
    calculate_divertor_cost_capital_then_fuel,
    "c2215",
    "divcst",
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
    reads={"uc_vac_pump": V.costs.UCTPMP},
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
    reads={"uc_vac_pump": V.costs.UCCPMP},
)


# `.tfcoil.i_tf_sup == 1`: superconducting, with breakers and a superconducting bus.
TfCoilPowerConditioningCostSuperconducting = costs.function(
    calculate_tf_coil_power_conditioning_cost_superconducting,
    "c22511",
    "c22512",
    "c22513",
    "c22514",
    "c22515",
    "c2251",
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
)


# cottax node: `calculate_pf_coil_power_conditioning_cost` (Account 225.2).
PfCoilPowerConditioningCost = costs.function(
    calculate_pf_coil_power_conditioning_cost,
    "c22521",
    "c22522",
    "c22523",
    "c22524",
    "c22525",
    "c22526",
    "c22527",
    "c2252",
)


# cottax node: `calculate_reactor_cooling_system_cost` (Account 2261).
ReactorCoolingSystemCost = costs.function(
    calculate_reactor_cooling_system_cost_formula,
    "cpp",
    "chx",
    "c2261",
)


# cottax node: `calculate_fuelling_system_cost` (Account 2271).
FuellingSystemCost = costs.function(calculate_fuelling_system_cost, "c2271")


# cottax node: `calculate_nuclear_building_ventilation_cost` (Account 2274).
NuclearBuildingVentilationCost = costs.function(
    calculate_nuclear_building_ventilation_cost,
    "c2274",
)


# cottax node: `calculate_instrumentation_and_control_cost` (Account 228).
InstrumentationAndControlCost = costs.function(
    calculate_instrumentation_and_control_cost, "c228"
)


# cottax node: `calculate_maintenance_equipment_cost` (Account 229).
MaintenanceEquipmentCost = costs.function(calculate_maintenance_equipment_cost, "c229")


# cottax node: `calculate_turbine_plant_equipment_cost` (Account 23).
# `.costs.ireactor == 1`.
TurbinePlantEquipmentCostReactor = costs.function(
    calculate_turbine_plant_equipment_cost_reactor,
    "c23",
)

# `.costs.ireactor == 0`: no turbine plant, `c23` zero and nothing read.
TurbinePlantEquipmentCostNonReactor = costs.function(
    calculate_turbine_plant_equipment_cost_non_reactor, "c23"
)


# cottax node: `calculate_switchyard_cost` (Account 241).
SwitchyardCost = costs.function(calculate_switchyard_cost_formula, "c241")


# cottax node: `calculate_transformers_cost` (Account 242).
TransformersCost = costs.function(
    calculate_transformers_cost_formula,
    "c242",
)


# cottax node: `calculate_low_voltage_cost` (Account 243).
LowVoltageCost = costs.function(
    calculate_low_voltage_cost_formula,
    "c243",
)


# cottax node: `calculate_diesel_generators_cost` (Account 244).
DieselGeneratorsCost = costs.function(calculate_diesel_generators_cost_formula, "c244")


# cottax node: `calculate_auxiliary_facility_power_cost` (Account 245).
AuxiliaryFacilityPowerCost = costs.function(
    calculate_auxiliary_facility_power_cost_formula, "c245"
)


# cottax node: `calculate_electric_plant_equipment_cost` (Account 24, total).
ElectricPlantEquipmentCost = costs.function(
    calculate_electric_plant_equipment_cost, "c24"
)


# cottax node: `calculate_misc_plant_equipment_cost` (Account 25).
MiscPlantEquipmentCost = costs.function(
    calculate_misc_plant_equipment_cost_formula, "c25"
)


# cottax node: `calculate_heat_rejection_cost` (Account 26).
# `.costs.ireactor == 1`: the primary heat not turned into electricity.
HeatRejectionCostReactor = costs.function(
    calculate_heat_rejection_cost_reactor,
    "c26",
)

# `.costs.ireactor == 0`: fusion, heating and TF power, all rejected.
HeatRejectionCostNonReactor = costs.function(
    calculate_heat_rejection_cost_non_reactor,
    "c26",
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
)
FirstWallCostFuel = costs.function(
    calculate_first_wall_cost_fuel,
    "c2211",
    "fwallcst",
)
FirstWallCostCapitalThenFuel = costs.function(
    calculate_first_wall_cost_capital_then_fuel,
    "c2211",
    "fwallcst",
)


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
)


# cottax node: `calculate_shield_cost` (Account 221.3).
ShieldCost = costs.function(
    calculate_shield_cost_magnetic_confinement,
    "c22131",
    "c22132",
    "c2213",
)


# cottax node: `calculate_reactor_cost` (Account 221 total).
ReactorCost = costs.function(calculate_reactor_cost, "c221")


# The TfMagnetCostSuperconducting family:
# The Account 222.1 superconducting-TF family (`.tfcoil.i_tf_sup == 1`).


# `.costs.supercond_cost_model == PER_KG` (0) -- PROCESS's own default
# (`cost_variables.py:552`) and the reference run's.
TfMagnetCostSuperconductingPerKg = costs.function(
    calculate_tf_magnet_cost_superconducting_per_kg,
    "c22211",
    "c22212",
    "c22213",
    "c22214",
    "c22215",
    "c2221",
)


# `.costs.supercond_cost_model == PER_KAM` (1) -- strand cost scaled by critical
# current density.
TfMagnetCostSuperconductingPerKam = costs.function(
    calculate_tf_magnet_cost_superconducting_per_kam,
    "c22211",
    "c22212",
    "c22213",
    "c22214",
    "c22215",
    "c2221",
    reads={"j_crit_str_0_tf": tfcoil.place("j_crit_str_0_tf")},
)


# cottax node: `calculate_tf_magnet_cost_resistive` (Account 222.1,
# `.tfcoil.i_tf_sup != 1`).
TfMagnetCostResistive = costs.function(
    calculate_tf_magnet_cost_resistive_formula,
    "c22211",
    "c22212",
    "c2221",
    "cpstcst",
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
        "den_pf_conductor": pf_coil.place("den_pf_conductor"),
        "den_cs_conductor": pf_coil.place("den_cs_conductor"),
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
    reads={"den_pf_conductor": pf_coil.place("den_pf_conductor")},
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
        "j_crit_str_0_pf": pf_coil.place("j_crit_str_0_pf"),
        "j_crit_str_0_cs": pf_coil.place("j_crit_str_0_cs"),
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
    reads={"j_crit_str_0_pf": pf_coil.place("j_crit_str_0_pf")},
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
            "p_hcd_ecrh_injected_total_mw": current_drive.place(
                "p_hcd_ecrh_injected_total_mw"
            ),
            "uc_lower_hybrid": V.costs.uclh,
            "p_hcd_lowhyb_injected_total_mw": current_drive.place(
                "p_hcd_lowhyb_injected_total_mw"
            ),
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
            "p_hcd_ecrh_injected_total_mw": current_drive.place(
                "p_hcd_ecrh_injected_total_mw"
            ),
            "uc_lower_hybrid": V.costs.ucich,
            "p_hcd_lowhyb_injected_total_mw": current_drive.place(
                "p_hcd_lowhyb_injected_total_mw"
            ),
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
            "p_hcd_ecrh_injected_total_mw": current_drive.place(
                "p_hcd_ecrh_injected_total_mw"
            ),
            "uc_lower_hybrid": V.costs.uclh,
            "p_hcd_lowhyb_injected_total_mw": current_drive.place(
                "p_hcd_lowhyb_injected_total_mw"
            ),
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
            "p_hcd_ecrh_injected_total_mw": current_drive.place(
                "p_hcd_ecrh_injected_total_mw"
            ),
            "uc_lower_hybrid": V.costs.ucich,
            "p_hcd_lowhyb_injected_total_mw": current_drive.place(
                "p_hcd_lowhyb_injected_total_mw"
            ),
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
EnergyStorageCostPulsedElectrowattOption1 = costs.function(
    calculate_energy_storage_cost_electrowatt_option_1,
    "c2253",
)


# `.pulse.istore == ELECTROWATT_OPTION_2` (2).
EnergyStorageCostPulsedElectrowattOption2 = costs.function(
    calculate_energy_storage_cost_electrowatt_option_2,
    "c2253",
)


# cottax node: `calculate_power_conditioning_cost` (Account 225 total).
PowerConditioningCost = costs.function(
    calculate_power_conditioning_cost_magnetic_confinement, "c225"
)


# cottax node: `calculate_auxiliary_component_cooling_cost` (Account 2262).
AuxiliaryComponentCoolingCost = costs.function(
    calculate_auxiliary_component_cooling_cost_magnetic_confinement,
    "cppa",
    "c2262",
)


# cottax node: `calculate_cryogenic_system_cost` (Account 2263).
CryogenicSystemCost = costs.function(
    calculate_cryogenic_system_cost_formula,
    "c2263",
)


# cottax node: `calculate_heat_transport_system_cost` (Account 226 total).
HeatTransportSystemCost = costs.function(calculate_heat_transport_system_cost, "c226")


# cottax node: `calculate_fuel_processing_cost` (Account 2272).
FuelProcessingCost = physics.function(
    calculate_fuel_processing_cost_magnetic_confinement,
    "wtgpd",
    V.costs.c2272,
)


# cottax node: `calculate_atmospheric_recovery_cost` (Account 2273).
AtmosphericRecoveryCost = costs.function(
    calculate_atmospheric_recovery_cost,
    "c2273",
)


# cottax node: `calculate_fuel_handling_cost` (Account 227 total).
FuelHandlingCost = costs.function(calculate_fuel_handling_cost, "c227")


# cottax node: `calculate_fusion_power_island_cost` (Account 22 total).
FusionPowerIslandCost = costs.function(
    calculate_fusion_power_island_cost, "crctcore", "c22"
)


# cottax node: `calculate_total_plant_direct_cost`.
TotalPlantDirectCost = costs.function(calculate_total_plant_direct_cost, "cdirt")


# cottax node: `calculate_constructed_cost`.
ConstructedCost = costs.function(calculate_constructed_cost, "concost")


# The CostOfElectricity family:
# The `Costs.coelc` family -- sole producer of `.costs.coe`, the `i_figure_merit ==
# 6` objective.


# `.physics.itart == 0` -- the reference run's, and PROCESS's own default
# (`physics_variables.py:994`).
# `.physics.itart == 0`; `.costs.ifueltyp == 0`.
CostOfElectricityConventionalAspectRatioCapital = costs.function(
    calculate_cost_of_electricity_conventional_aspect_ratio_capital,
    "moneyint",
    "capcost",
    "coecap",
    "coeoam",
    "coefuelt",
    "coe",
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
)


# `.physics.itart == 1` -- the spherical tokamak, which additionally pays to
# replace its centrepost (`costs.py:2769-2783`).
