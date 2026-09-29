"""`helias_5b`, stated.

Converted from `tests/regression/input_files/helias_5b.IN.DAT` by
`input.indat.configuration_from_indat`; regenerate rather than hand-edit --
`tests/test_configurations.py` checks the two agree.
"""

import numpy as np

from cottax.interfaces import V, place
from functional_process.configurations import Configuration, Optimisation
from functional_process.cottax.models.availability.availability import (
    AvailNeutronFluence,
)
from functional_process.cottax.models.availability.namespace import Availability
from functional_process.cottax.models.buildings.buildings import Bldgs, TfCoilEnvelope
from functional_process.cottax.models.buildings.namespace import Buildings
from functional_process.cottax.models.constraints import (
    CapitalCostReactor,
    Constraint16,
    Constraint24TotalBeta,
    Constraint2IgnitedCoreRadiation,
    Constraint84,
)
from functional_process.cottax.models.costs.costs import (
    AtmosphericRecoveryCost,
    AuxiliaryComponentCoolingCost,
    AuxiliaryFacilityPowerCost,
    BlanketCostCapital,
    ConstructedCost,
    ConvertFpyToCalendarConventionalAspectRatio,
    CostOfElectricityConventionalAspectRatioCapital,
    CryogenicSystemCost,
    DieselGeneratorsCost,
    DivertorCostCapital,
    ElectricPlantEquipmentCost,
    EnergyStorageCostUnpulsed,
    FirstWallCostCapital,
    FuelHandlingCost,
    FuelProcessingCost,
    FuellingSystemCost,
    FusionPowerIslandCost,
    HeatRejectionCostReactor,
    HeatTransportSystemCost,
    IndirectCosts,
    InstrumentationAndControlCost,
    LowVoltageCost,
    MagnetsCost,
    MaintenanceEquipmentCost,
    MiscPlantEquipmentCost,
    NuclearBuildingVentilationCost,
    PowerConditioningCost,
    PowerInjectionCostCapitalLowerHybrid,
    ReactorCoolingSystemCost,
    ReactorCost,
    ShieldCost,
    StructuresCostReactor,
    SwitchyardCost,
    TfCoilPowerConditioningCostSuperconducting,
    TfMagnetCostSuperconductingPerKg,
    TotalPlantDirectCost,
    TransformersCost,
    TurbinePlantEquipmentCostReactor,
    VacuumSystemCostCryopump,
    VacuumVesselAssemblyCost,
)
from functional_process.cottax.models.costs.namespace import Costs
from functional_process.cottax.models.initialisation import (
    EnergyStorageBuildingVolume,
    Initialisation,
    StellaratorPulseTimes,
    StellaratorSolenoidAbsent,
    TfCryoplantEfficiency,
)
from functional_process.cottax.models.physics.composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaCompositionIgnited,
)
from functional_process.cottax.models.physics.confinement_time import (
    ConfinementScalingInputs,
    ConfinementTailCoreRadiation,
    DoubleAndTripleProduct,
    Iss04ConfinementTime,
    IterPhysicsBasisElongation,
    PlasmaPowerLossIgnitedCoreRadiation,
)
from functional_process.cottax.models.physics.dimensionless_parameters import (
    DimensionlessPlasmaParameters,
)
from functional_process.cottax.models.physics.exhaust import RadiationFraction
from functional_process.cottax.models.physics.fusion_reactions import (
    FusionRates,
    SetFusionPowers,
)
from functional_process.cottax.models.physics.namespace import (
    Physics,
    PhysicsConfinementTime,
    PhysicsProfiles,
    ProfileParameterisationParabolic,
)
from functional_process.cottax.models.physics.plasma_profiles import (
    IonVolAvgTemperature,
    LModeProfileReset,
    ParabolicGradientLengths,
    ParabolicProfileValues,
    ProfileFactors,
)
from functional_process.cottax.models.physics.profiles import (
    DensityProfile,
    NeProfileIntegral,
    ParabolicOnAxisDensities,
    ParabolicOnAxisTemperatures,
    ParabolicTemperatureProfile,
    ProfileGrid,
    TeProfileIntegral,
)
from functional_process.cottax.models.physics.pure_formulas import (
    AuxiliaryPhysicsQuantities,
    ElectronThermalEnergy,
    FastAlphaBetaWard,
    IonElectronEquilibration,
    IonThermalEnergy,
    TotalPlasmaHeatingPower,
)
from functional_process.cottax.models.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
)
from functional_process.cottax.models.power.electric_production import (
    AcpowLine,
    PlantElectricProductionSingleCoolant,
)
from functional_process.cottax.models.power.namespace import Power
from functional_process.cottax.models.power.tf_coil_power import TfPowerSuperconducting
from functional_process.cottax.models.power.thermal_cryo import (
    ComponentThermalPowersSummedSolidOtherShieldPrimary,
    CryoLoadsActive,
    CryoQLoadsSuperconductingTf,
    CryoQNuc,
    DeltaEtaStepSummedSolidOtherShieldPrimary,
    EtathLiqSupercriticalCo2,
    PFwBlktCoolantPumpMw,
    PFwDivHeatDepositedMwSummed,
    TempTurbineCoolantInFromLiquidBreeder,
)
from functional_process.cottax.models.stellarator.build import (
    AFwTotalWithPowerflow,
    Build,
)
from functional_process.cottax.models.stellarator.coils.calculate import (
    CoilCasing,
    CoilCoilToroidalGap,
    CoilCrossSectionalArea,
    CoilCurrent,
    CoilHalfWidths,
    CoilRadialThickness,
    CoilToroidalThickness,
    CoilsSummaryVariables,
    HorizontalPorts,
    IterNb3snWindingPackIntersectInputs,
    LenTfCoil,
    PlasmaFacingCoilArea,
    StoredMagneticEnergy,
    TfCryoArea,
    VerticalPorts,
    WindingPackGeometry,
    WindingPackTotalSizePost,
    ZTfInsideHalf,
)
from functional_process.cottax.models.stellarator.coils.coils import Intersect
from functional_process.cottax.models.stellarator.coils.forces import (
    MaxForceDensity,
    MaximumStress,
)
from functional_process.cottax.models.stellarator.coils.mass import IterNb3snCoilsMass
from functional_process.cottax.models.stellarator.coils.quench import QuenchProtection
from functional_process.cottax.models.stellarator.density_limits import (
    EcrhDensityLimit,
    SudoDensityLimit,
)
from functional_process.cottax.models.stellarator.divertor import Divertor
from functional_process.cottax.models.stellarator.geometry import (
    DefaultAspectRatio,
    StellaratorPlasmaGeometry,
    StellaratorScalingFactors,
)
from functional_process.cottax.models.stellarator.heating import (
    BeamCurrent,
    EcrhHeating,
    FusionGain,
    InjectedPowerTotal,
)
from functional_process.cottax.models.stellarator.initialization import PulseDurations
from functional_process.cottax.models.stellarator.namespace import (
    Stellarator,
    StellaratorCoils,
    StellaratorFwbs,
)
from functional_process.cottax.models.stellarator.neoclassics import (
    EffectiveThermalDiffusivity,
    ProfileValues,
)
from functional_process.cottax.models.stellarator.plasma_physics import (
    ClippedRadiationPowers,
    FusionPowerTotalsMw,
    FusionTotalsNoBeam,
    HeatingAndRadiationPowerIgnited,
    NeutronWallLoadScaledPlasmaSurface,
    PoloidalFieldFromRotationalTransform,
    RadiatedWallLoadScaledPlasmaSurface,
    StellaratorBetaAndStoredEnergy,
    ThermalEnergyTotals,
    TotalField,
)
from functional_process.cottax.models.stellarator.preset_config import (
    StellaratorMachineConfig,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s1_s5 import (
    CryostatAndVvGeometry,
    FwBlanketShieldGeometryWithPowerflow,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPowerUserInputPumping,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s3 import (
    DivertorPlateMass,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
    ShieldMass,
)
from functional_process.cottax.models.stellarator.structure import StructureMasses
from functional_process.cottax.models.total_process import StellaratorProcess
from functional_process.cottax.models.vacuum.namespace import Vacuum
from functional_process.cottax.models.vacuum.vacuum import VacuumOldCryopumpDwell
from functional_process.models.stellarator.preset_config import machine_config_for_istell

machine = StellaratorProcess(
    initialisation=Initialisation(
        tf_cryoplant_efficiency=TfCryoplantEfficiency,
        tf_insulation_youngs_modulus=None,
        tf_conductor_youngs_modulus=None,
        pf_coil_resistivity=None,
        beam_electron_density_fraction=None,
        energy_storage_building_volume=EnergyStorageBuildingVolume,
        double_null_upper_build=None,
        stellarator_solenoid_absent=StellaratorSolenoidAbsent,
        stellarator_pulse_times=StellaratorPulseTimes,
    ),
    costs=Costs(
        convert_fpy_to_calendar=ConvertFpyToCalendarConventionalAspectRatio,
        structures_cost=StructuresCostReactor,
        first_wall_cost=FirstWallCostCapital,
        blanket_cost=BlanketCostCapital,
        shield_cost=ShieldCost,
        reactor_structure_cost=None,
        divertor_cost=DivertorCostCapital,
        reactor_cost=ReactorCost,
        tf_magnet_cost_superconducting=TfMagnetCostSuperconductingPerKg,
        pf_magnet_cost=None,
        vacuum_vessel_assembly_cost=VacuumVesselAssemblyCost,
        magnets_cost=MagnetsCost,
        power_injection_cost=PowerInjectionCostCapitalLowerHybrid,
        vacuum_system_cost=VacuumSystemCostCryopump,
        tf_coil_power_conditioning_cost=TfCoilPowerConditioningCostSuperconducting,
        pf_coil_power_conditioning_cost=None,
        energy_storage_cost=EnergyStorageCostUnpulsed,
        power_conditioning_cost=PowerConditioningCost,
        reactor_cooling_system_cost=ReactorCoolingSystemCost,
        auxiliary_component_cooling_cost=AuxiliaryComponentCoolingCost,
        cryogenic_system_cost=CryogenicSystemCost,
        heat_transport_system_cost=HeatTransportSystemCost,
        fuelling_system_cost=FuellingSystemCost,
        fuel_processing_cost=FuelProcessingCost,
        atmospheric_recovery_cost=AtmosphericRecoveryCost,
        nuclear_building_ventilation_cost=NuclearBuildingVentilationCost,
        fuel_handling_cost=FuelHandlingCost,
        instrumentation_and_control_cost=InstrumentationAndControlCost,
        maintenance_equipment_cost=MaintenanceEquipmentCost,
        fusion_power_island_cost=FusionPowerIslandCost,
        turbine_plant_equipment_cost=TurbinePlantEquipmentCostReactor,
        switchyard_cost=SwitchyardCost,
        transformers_cost=TransformersCost,
        low_voltage_cost=LowVoltageCost,
        diesel_generators_cost=DieselGeneratorsCost,
        auxiliary_facility_power_cost=AuxiliaryFacilityPowerCost,
        electric_plant_equipment_cost=ElectricPlantEquipmentCost,
        misc_plant_equipment_cost=MiscPlantEquipmentCost,
        heat_rejection_cost=HeatRejectionCostReactor,
        total_plant_direct_cost=TotalPlantDirectCost,
        indirect_costs=IndirectCosts,
        constructed_cost=ConstructedCost,
        cost_of_electricity=CostOfElectricityConventionalAspectRatioCapital,
    ),
    stellarator=Stellarator(
        machine_config=StellaratorMachineConfig,
        heating=EcrhHeating,
        fw_area=AFwTotalWithPowerflow,
        coils=StellaratorCoils(
            coil_toroidal_thickness=CoilToroidalThickness,
            coil_radial_thickness=CoilRadialThickness,
            coil_cross_sectional_area=CoilCrossSectionalArea,
            coil_half_widths=CoilHalfWidths,
            plasma_facing_coil_area=PlasmaFacingCoilArea,
            coil_coil_toroidal_gap=CoilCoilToroidalGap,
            coils_summary_variables=CoilsSummaryVariables,
            stored_magnetic_energy=StoredMagneticEnergy,
            winding_pack_geometry=WindingPackGeometry,
            coil_current=CoilCurrent,
            coil_casing=CoilCasing,
            vertical_ports=VerticalPorts,
            horizontal_ports=HorizontalPorts,
            z_tf_inside_half=ZTfInsideHalf,
            tf_cryo_area=TfCryoArea,
            len_tf_coil=LenTfCoil,
            coils_mass=IterNb3snCoilsMass,
            max_force_density=MaxForceDensity,
            maximum_stress=MaximumStress,
            quench_protection=QuenchProtection,
            winding_pack_intersect_inputs=IterNb3snWindingPackIntersectInputs,
            intersect=Intersect,
            winding_pack_total_size_post=WindingPackTotalSizePost,
        ),
        fwbs=StellaratorFwbs(
            blanket_shield_power=DetailedPowerflowBlanketShieldPowerUserInputPumping,
            blanket_masses=BlanketComponentMasses,
            fw_blanket_shield_geometry=FwBlanketShieldGeometryWithPowerflow,
            cryostat_and_vv_geometry=CryostatAndVvGeometry,
            divertor_plate_mass=DivertorPlateMass,
            shield_mass=ShieldMass,
        ),
        sudo_density_limit=SudoDensityLimit,
        structure_masses=StructureMasses,
        build=Build,
        divertor=Divertor,
        injected_power_total=InjectedPowerTotal,
        beam_current=BeamCurrent,
        fusion_gain=FusionGain,
        pulse_durations=PulseDurations,
        profile_values=ProfileValues,
        effective_thermal_diffusivity=EffectiveThermalDiffusivity,
        stellarator_beta_and_stored_energy=StellaratorBetaAndStoredEnergy,
        poloidal_field_from_rotational_transform=PoloidalFieldFromRotationalTransform,
        total_field=TotalField,
        clipped_radiation_powers=ClippedRadiationPowers,
        neutron_wall_load=NeutronWallLoadScaledPlasmaSurface,
        heating_and_radiation_power=HeatingAndRadiationPowerIgnited,
        radiated_wall_load_and_fraction=RadiatedWallLoadScaledPlasmaSurface,
        thermal_energy_totals=ThermalEnergyTotals,
        default_aspect_ratio=DefaultAspectRatio,
        stellarator_scaling_factors=StellaratorScalingFactors,
        stellarator_plasma_geometry=StellaratorPlasmaGeometry,
    ),
    physics=Physics(
        fusion_power_totals_mw=FusionPowerTotalsMw,
        fusion_totals_no_beam=FusionTotalsNoBeam,
        profiles=PhysicsProfiles(
            parameterisation=ProfileParameterisationParabolic(
                ecrh_density_limit=EcrhDensityLimit,
                parabolic_temperature_profile=ParabolicTemperatureProfile,
                parabolic_on_axis_densities=ParabolicOnAxisDensities,
                parabolic_on_axis_temperatures=ParabolicOnAxisTemperatures,
                parabolic_gradient_lengths=ParabolicGradientLengths,
                parabolic_profile_values=ParabolicProfileValues,
                l_mode_profile_reset=LModeProfileReset,
            ),
            profile_factors=ProfileFactors,
            profile_grid=ProfileGrid,
            ne_profile_integral=NeProfileIntegral,
            te_profile_integral=TeProfileIntegral,
            density_profile=DensityProfile,
            ion_vol_avg_temperature=IonVolAvgTemperature,
        ),
        confinement_time=PhysicsConfinementTime(
            inputs=ConfinementScalingInputs,
            elongation=IterPhysicsBasisElongation,
            power_loss=PlasmaPowerLossIgnitedCoreRadiation,
            scaling=Iss04ConfinementTime,
            double_and_triple_product=DoubleAndTripleProduct,
            tail=ConfinementTailCoreRadiation,
        ),
        fusion_rates=FusionRates,
        set_fusion_powers=SetFusionPowers,
        synchrotron_radiation_power=SynchrotronRadiationPower,
        impurity_radiation_totals=ImpurityRadiationTotals,
        plasma_radiation_powers=PlasmaRadiationPowers,
        ion_electron_equilibration=IonElectronEquilibration,
        auxiliary_physics_quantities=AuxiliaryPhysicsQuantities,
        total_plasma_heating_power=TotalPlasmaHeatingPower,
        electron_thermal_energy=ElectronThermalEnergy,
        ion_thermal_energy=IonThermalEnergy,
        fast_alpha_beta=FastAlphaBetaWard,
        plasma_composition=PlasmaCompositionIgnited,
        calculate_effective_charge_ionisation_profiles=CalculateEffectiveChargeIonisationProfiles,
        dimensionless_plasma_parameters=DimensionlessPlasmaParameters,
        radiation_fraction=RadiationFraction,
    ),
    power=Power(
        pf_coil_power=None,
        tf_power=TfPowerSuperconducting,
        component_thermal_powers=ComponentThermalPowersSummedSolidOtherShieldPrimary,
        delta_eta_step=DeltaEtaStepSummedSolidOtherShieldPrimary,
        eta_turbine=None,
        etath_liq=EtathLiqSupercriticalCo2,
        temp_turbine_coolant_in=TempTurbineCoolantInFromLiquidBreeder,
        p_fw_div_heat_deposited_mw=PFwDivHeatDepositedMwSummed,
        p_fw_blkt_coolant_pump_mw=PFwBlktCoolantPumpMw,
        cryo_q_nuc=CryoQNuc,
        cryo_q_loads=CryoQLoadsSuperconductingTf,
        cryo_loads=CryoLoadsActive,
        acpow=AcpowLine,
    ),
    buildings=Buildings(
        sizing=Bldgs,
        tf_coil_envelope=TfCoilEnvelope,
    ),
    vacuum=Vacuum(
        vacuum_old=VacuumOldCryopumpDwell,
    ),
    availability=Availability(
        electric_production=PlantElectricProductionSingleCoolant,
        avail=AvailNeutronFluence,
        cplife_avail=None,
    ),
)

values = {
    "physics": {
        "nd_plasma_electrons_vol_avg": 2.0914e20,
        "hfact": 1.2187,
        "alphan": 0.35,
        "alphat": 1.2,
        "aspect": 12.22,
        "b_plasma_toroidal_on_axis": 5.5,
        "i_plasma_ignited": 1,
        "i_plasma_pedestal": 0,
        "i_rad_loss": 1,
        "i_confinement_time": 38,
        "rmajor": 22.0,
        "f_sync_reflect": 0.6,
        "temp_plasma_electron_vol_avg_kev": 7.0,
        "f_temp_plasma_ion_electron": 0.95,
        "beta_vol_avg_max": 0.05,
        "beta_vol_avg_min": 0.01,
    },
    "stellarator": {
        "istell": 1,
        "bmn": 0.001,
        "f_asym": 1.1,
        "f_rad": 0.85,
        "f_w": 0.6,
        "flpitch": 0.001,
        "iotabar": 0.9,
        "isthtr": 1,
        "shear": 0.5,
    },
    "build": {
        "dr_blkt_inboard": 0.7,
        "dr_blkt_outboard": 0.8,
        "dr_cryostat": 0.15,
        "dr_vv_inboard": 0.35,
        "dr_vv_outboard": 0.35,
        "dz_vv_upper": 0.35,
        "dz_vv_lower": 0.35,
        "dr_shld_vv_gap_inboard": 0.025,
        "gapomin": 0.025,
        "dr_fw_plasma_gap_inboard": 0.15,
        "dr_fw_plasma_gap_outboard": 0.3,
        "dr_shld_inboard": 0.4,
        "dr_shld_outboard": 0.7,
        "dz_shld_upper": 0.7,
        "dz_xpoint_divertor": 0.0,
    },
    "constraints": {
        "p_plant_electric_net_required_mw": 1000.0,
    },
    "current_drive": {
        "eta_ecrh_injector_wall_plug": 0.7,
        "p_hcd_primary_extra_heat_mw": 0.0,
    },
    "divertor": {
        "anginc": 0.035,
        "tdiv": 3.0,
        "xpertin": 1.5,
    },
    "fwbs": {
        "den_steel": 7800.0,
        "f_p_blkt_multiplication": 1.18,
        "eta_coolant_pump_electric": 1.0,
        "fblbe": 0.47,
        "fblli2o": 0.07,
        "fbllipb": 0.0,
        "fblss": 0.13,
        "fblvd": 0.0,
        "fhole": 0.0,
        "fwclfr": 0.1,
        "i_p_coolant_pumping": 0,
        "i_thermal_electric_conversion": 2,
        "f_a_blkt_cooling_channels": 0.1,
        "vfshld": 0.6,
        "radius_fw_channel": 0.008,
    },
    "heat_transport": {
        "p_blkt_coolant_pump_mw": 120.0,
        "p_fw_coolant_pump_mw": 56.0,
        "p_div_coolant_pump_mw": 24.0,
        "eta_turbine": 0.4,
    },
    "impurity_radiation": {
        "radius_plasma_core_norm": 0.6,
        "f_nd_impurity_electrons": np.array([
            1.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1e-05,
        ]),
    },
    "globals": {
        "maxcal": 100,
        "runtitle": "HELIAS-5B",
    },
    "tfcoil": {
        "f_a_tf_turn_cable_copper": 0.69,
        "i_tf_sc_mat": 1,
        "tftmp": 4.2,
        "temp_tf_cryo": 4.2,
        "dx_tf_turn_general": 0.056,
        "dx_tf_turn_insulation": 0.002,
        "dx_tf_turn_steel": 0.0012,
        "f_a_tf_turn_cable_space_extra_void": 0.3,
        "dr_tf_nose_case": 0.06,
    },
    "costs": {
        "i_cost_model": 0,
        "abktflnc": 15.0,
        "adivflnc": 25.0,
        "f_t_plant_available": 0.75,
        "dintrt": 0.0,
        "fcap0": 1.15,
        "fcap0cp": 1.06,
        "fcontng": 0.15,
        "fcr0": 0.065,
        "fkind": 1.0,
        "i_plant_availability": 0,
        "ifueltyp": 0,
        "ireactor": 1,
        "lsa": 2,
        "discount_rate": 0.06,
        "life_plant": 40.0,
        "ucblvd": 280.0,
        "ucdiv": 500000.0,
        "ucme": 300000000.0,
    },
    "stellarator_config": {
        "machine_config": machine_config_for_istell(1),
    },
}

problem = Optimisation(
    design={
        place(V.physics.temp_plasma_electron_vol_avg_kev): (4.0, 25.0),
        place(V.physics.nd_plasma_electrons_vol_avg): (3.005e19, 5.005e20),
        place(V.physics.hfact): (0.1, 2.0),
    },
    equalities=(Constraint2IgnitedCoreRadiation, Constraint16),
    inequalities=(Constraint84, Constraint24TotalBeta),
    minimise=CapitalCostReactor,
)

CONFIGURATION = Configuration(
    name="helias_5b",
    machine=machine,
    values=values,
    problem=problem,
)
