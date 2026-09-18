"""`spherical_tokamak_eval`, stated.

Converted from `tests/regression/input_files/spherical_tokamak_eval.IN.DAT` by
`input.indat.configuration_from_indat`; regenerate rather than hand-edit --
`tests/test_configurations.py` checks the two agree.
"""

import numpy as np

from functional_process.configurations import Configuration, Problem
from functional_process.cottax.models.availability.availability import AvailNeutronFluence, CplifeAvailSuperconducting
from functional_process.cottax.models.availability.namespace import Availability
from functional_process.cottax.models.blankets.blanket_library import BlanketCoverageFactorsDoubleNull, BlanketHalfHeightDoubleNull, BlanketInboardPoloidalAngle, DShapedBlanketAreas, DShapedBlanketVolumes
from functional_process.cottax.models.blankets.hcpb import CentrepostNeutronicsSphericalTokamakSuperconducting, ComponentMasses, DivertorSurfaceAndPlateMassDoubleNull, FirstWallCoolantVoidFractions, FirstWallRadiationPowers, NuclearHeatingBlanket, NuclearHeatingFw, NuclearHeatingMagnetsSphericalTokamak, NuclearHeatingRenormalisationDoubleNullSphericalTokamak, NuclearHeatingShieldSphericalTokamak, PumpingPowerMechanicalWithPressureDrop
from functional_process.cottax.models.blankets.namespace import CcfeHcpb
from functional_process.cottax.models.build import BlktUpperThickness, DrTfOutboardSuperconducting, DrTfWpWithInsulationFromInboardBuild, PlasmaXpointHeights, RCpTopFromTfInboardOut, RadialBuildToPlasmaCentre, ShldInboardInnerRadius, ShldOutboardOuterRadius, ShldVvGapOutboard, TfInboardRadiiNoCsPrecomp, TfInnerBore, TfOutboardEdgeRipplePictureFrame, TfOutboardMidPictureFrame, TfOutboardMidUnrippled, TfTopHeightDoubleNull, VacuumVesselAndShieldRadiiTfOutsideCs, WpConductorMaxWidthSuperconducting, ZTfInsideHalf
from functional_process.cottax.models.buildings.buildings import Bldgs, TfCoilEnvelope
from functional_process.cottax.models.buildings.namespace import Buildings
from functional_process.cottax.models.costs.costs import AtmosphericRecoveryCost, AuxiliaryComponentCoolingCost, AuxiliaryFacilityPowerCost, BlanketCost, ConstructedCost, ConvertFpyToCalendar, CostOfElectricitySphericalTokamak, CryogenicSystemCost, DieselGeneratorsCost, DivertorCost, ElectricPlantEquipmentCost, EnergyStorageCostUnpulsed, FirstWallCost, FuelHandlingCost, FuelProcessingCost, FuellingSystemCost, FusionPowerIslandCost, HeatRejectionCost, HeatTransportSystemCost, IndirectCosts, InstrumentationAndControlCost, LowVoltageCost, MagnetsCost, MaintenanceEquipmentCost, MiscPlantEquipmentCost, NuclearBuildingVentilationCost, PfCoilPowerConditioningCost, PfMagnetCostPerKgNoCentralSolenoid, PowerConditioningCost, PowerInjectionCost, ReactorCoolingSystemCost, ReactorCost, ReactorStructureCost, ShieldCost, StructuresCost, SwitchyardCost, TfCoilPowerConditioningCost, TfMagnetCostSuperconductingPerKg, TotalPlantDirectCost, TransformersCost, TurbinePlantEquipmentCost, VacuumSystemCost, VacuumVesselAssemblyCost
from functional_process.cottax.models.costs.namespace import Costs
from functional_process.cottax.models.cryostat import Cryostat
from functional_process.cottax.models.cs_fatigue import CsFatigue
from functional_process.cottax.models.divertor import DivertorHeatFluxSplit, DivertorHeatLoadWadeDoubleNull
from functional_process.cottax.models.fw import FirstWallDShapedDoubleNull, FirstWallGeometry, RadiatedWallLoad
from functional_process.cottax.models.initialisation import BeamElectronDensityFraction, DoubleNullUpperBuild, EnergyStorageBuildingVolume, Initialisation, PfCoilResistivity, TfConductorYoungsModulus, TfCryoplantEfficiency, TfInsulationYoungsModulus
from functional_process.cottax.models.namespace import Build, Divertor
from functional_process.cottax.models.pfcoil.currents import PFCoilEquilibriumCurrentsNoCentralSolenoid, PFCoilInitiationCurrentsNoCentralSolenoid, PFCoilTimePointCurrentsNoCentralSolenoid
from functional_process.cottax.models.pfcoil.fields import PFCoilCurrentWaveformNoCentralSolenoid, PFCoilPeakFieldNoCentralSolenoid
from functional_process.cottax.models.pfcoil.geometry import PFCoilPlacementSphericalTokamak, PFCoilPositionsNoCentralSolenoid
from functional_process.cottax.models.pfcoil.inductance import PFCoilInductanceNoCentralSolenoid
from functional_process.cottax.models.pfcoil.masses import PFCoilMassesNoCentralSolenoid, PFCoilSizesNoCentralSolenoid
from functional_process.cottax.models.pfcoil.namespace import PFCoilSphericalTokamak
from functional_process.cottax.models.pfcoil.superconductor import PFStrandCriticalCurrentDensityHazeltonZhaiRebco
from functional_process.cottax.models.pfcoil.volt_seconds import PFCoilTurnCurrentsNoCentralSolenoid, PFCoilVoltSecondsNoCentralSolenoid
from functional_process.cottax.models.physics.bootstrap_current import PlasmaCurrentFractions, SauterBootstrapCurrentFraction, SceneDiamagneticCurrent, ScenePfirschSchluterCurrent
from functional_process.cottax.models.physics.composition import CalculateEffectiveChargeIonisationProfiles, PlasmaCompositionNonIgnited
from functional_process.cottax.models.physics.confinement_time import ConfinementScalingInputs, ConfinementTailCoreRadiation, DoubleAndTripleProduct, IterIpb98y2ConfinementTime, IterPhysicsBasisElongation, PlasmaPowerLossNonIgnitedCoreRadiation
from functional_process.cottax.models.physics.current_drive import FusionGain, HcdElectricTotalNonIgnited, HcdInjectedPowerTotal, HcdPrimaryEfficiencyFreethyEcrhOMode, HcdPrimaryInjectedPower, HcdPrimaryPowersElectronCyclotronNoSecondary, HcdSecondaryDrivenCurrent, HcdSecondaryHeatingNone
from functional_process.cottax.models.physics.density_limit import EnforcedDensityLimitGreenwald, GreenwaldDensityLimit, GreenwaldFraction, TokamakDensityLimit
from functional_process.cottax.models.physics.dimensionless_parameters import DimensionlessPlasmaParameters
from functional_process.cottax.models.physics.exhaust import EuDemoReAttachmentMetric, PsepOverRMetric, RadiationFraction
from functional_process.cottax.models.physics.fusion_reactions import FusionRates, SetFusionPowers
from functional_process.cottax.models.physics.l_h_transition import Martin08AspectNominalLHThresholdPower
from functional_process.cottax.models.physics.namespace import Physics, PhysicsConfinementTime, PhysicsProfiles, ProfileParameterisationPedestal
from functional_process.cottax.models.physics.physics import BetaLimitFromNorm, CoulombLogarithmIonElectron, PlasmaEnergyFromBeta, PlasmaOhmicHeating, PlasmaSurfaceNeutronFlux, PoloidalBeta, PositiveSeparatrixPower, PulseRampTimesContinuousDefault, SeparatrixPowerNonIgnited, SurfaceAveragedPoloidalFieldAmperes, ThermalBeta, ToroidalBeta, TotalRadiationPower, UnclippedRadiationPowers
from functional_process.cottax.models.physics.plasma_current import FiestaStPlasmaCurrent, PlasmaCylindricalSafetyFactor, TokamakPlasmaCurrent
from functional_process.cottax.models.physics.plasma_fields import PlasmaFields, PlasmaInboardToroidalField, PlasmaOutboardToroidalField, TotalMagneticField, TotalMagneticFieldInboard, TotalMagneticFieldOutboard
from functional_process.cottax.models.physics.plasma_geometry import DoubleArcPlasmaGeometry, Ipdg89XPointPlasmaShape, PlasmaMinorRadius
from functional_process.cottax.models.physics.plasma_inductance import PlasmaInternalInductanceScalings, PlasmaVoltSecondRequirements, TokamakPlasmaInductance
from functional_process.cottax.models.physics.plasma_profiles import IonVolAvgTemperature, PedestalProfileValues, ProfileFactors
from functional_process.cottax.models.physics.profiles import DensityProfile, NeProfileIntegral, PedestalOnAxisDensities, PedestalOnAxisTemperatures, PedestalSeparatrixDensities, PedestalTemperatureProfile, ProfileGrid, TeProfileIntegral
from functional_process.cottax.models.physics.pure_formulas import AuxiliaryPhysicsQuantities, ElectronThermalEnergy, FastAlphaBetaWard, IonElectronEquilibration, IonThermalEnergy, TotalPlasmaHeatingPower
from functional_process.cottax.models.physics.radiation_power import ImpurityRadiationTotals, PlasmaRadiationPowers, SynchrotronRadiationPower
from functional_process.cottax.models.physics.scrape_off_layer import Eich2013SOLPowerDecayLength, Mast2014SOLPowerDecayLength1, Mast2014SOLPowerDecayLength2, OutboardSOLEich13ParallelPowerFlux, OutboardSOLParallelPowerFlux, OutboardSOLPowerDecayLengthEich2013, TokamakScrapeOffLayer, UpstreamSOLOutboardEich13ParallelArea, UpstreamSOLOutboardParallelArea
from functional_process.cottax.models.physics.tokamak_namespace import TokamakCurrentDrive, TokamakPhysics, TokamakPlasmaBeta, TokamakPlasmaGeom, TokamakPulse
from functional_process.cottax.models.power.electric_production import AcpowLine, PlantElectricProductionSingleCoolant
from functional_process.cottax.models.power.namespace import Power
from functional_process.cottax.models.power.pf_coil_power import PfCoilPowerSuppliesNoCentralSolenoid
from functional_process.cottax.models.power.tf_coil_power import TfPowerSuperconducting
from functional_process.cottax.models.power.thermal_cryo import ComponentThermalPowersMechSolidOther, CryoLoadsActive, CryoQLoadsSuperconductingTf, CryoQNuc, DeltaEtaStepMechSolidOther, EtathLiqSupercriticalCo2, TempTurbineCoolantInFromLiquidBreeder
from functional_process.cottax.models.shield import DShapedShieldVolumes, DoubleNullShieldHalfHeight, TokamakShield
from functional_process.cottax.models.stellarator.initialization import PulseDurations
from functional_process.cottax.models.stellarator.plasma_physics import FusionPowerTotalsMw, FusionTotalsNoBeam
from functional_process.cottax.models.structure import Structure
from functional_process.cottax.models.tfcoil.base import DrTfPlasmaCaseFromFraction, DxTfSideCaseMinFromFraction, GenericTfCoilAreaAndMasses, RBTfInboardPeak, TfCoilSelfInductancePictureFrame, TfCoilShapePictureFrameTart, TfCurrent, TfGlobalGeometryCircularCase, TfStoredMagneticEnergy
from functional_process.cottax.models.tfcoil.croco import CrocoAveragedTurnGeometryFromCurrentPerTurn, CrocoCableGeometry, CrocoCableSpaceProperties, CrocoInboardAreasAndFractions, CrocoTurnCableSpaceCoolingFraction, CrocoTurnCableSpaceExtraVoid, HazeltonZhaiRebcoCrocoSuperconductorProperties, HazeltonZhaiRebcoCrocoTemperatureMargin
from functional_process.cottax.models.tfcoil.namespace import CrocoSuperconductingTfCoil
from functional_process.cottax.models.tfcoil.quench import TfCoilDumpQuenchVoltage, TfCoilQuenchHeatCurrentDensity
from functional_process.cottax.models.tfcoil.stress import TfFieldAndForceClampedJoints, TfStressExtendedPlaneStrainBuckedCaseAveragedTurn
from functional_process.cottax.models.tfcoil.superconducting import DxTfSideCaseTrapezoidal, HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak, PeakBTfInboardWithRippleFlatAllowance, SuperconductingTfWpGeometryTrapezoidal, TfCaseAreasCircularFront, TfTurnArea, TfWpCurrents, VvStressOnQuench
from functional_process.cottax.models.tokamak.namespace import Tokamak
from functional_process.cottax.models.total_process import TokamakProcess
from functional_process.cottax.models.vacuum.namespace import Vacuum
from functional_process.cottax.models.vacuum.vacuum import DuctDiameterRootFind, VacuumOld, VacuumVesselDShapedDoubleNull
from functional_process.vocabulary.enums import PFConductorModel

machine = TokamakProcess(
    initialisation=Initialisation(
        tf_cryoplant_efficiency=TfCryoplantEfficiency(),
        tf_insulation_youngs_modulus=TfInsulationYoungsModulus(),
        tf_conductor_youngs_modulus=TfConductorYoungsModulus(),
        pf_coil_resistivity=PfCoilResistivity(),
        beam_electron_density_fraction=BeamElectronDensityFraction(),
        energy_storage_building_volume=EnergyStorageBuildingVolume(),
        double_null_upper_build=DoubleNullUpperBuild(),
        stellarator_solenoid_absent=None,
        stellarator_pulse_times=None,
    ),
    costs=Costs(
        convert_fpy_to_calendar=ConvertFpyToCalendar(),
        structures_cost=StructuresCost(),
        first_wall_cost=FirstWallCost(),
        blanket_cost=BlanketCost(),
        shield_cost=ShieldCost(),
        reactor_structure_cost=ReactorStructureCost(),
        divertor_cost=DivertorCost(),
        reactor_cost=ReactorCost(),
        tf_magnet_cost_superconducting=TfMagnetCostSuperconductingPerKg(),
        pf_magnet_cost=PfMagnetCostPerKgNoCentralSolenoid(
            n_cs_pf_coils=8,
            i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
        ),
        vacuum_vessel_assembly_cost=VacuumVesselAssemblyCost(),
        magnets_cost=MagnetsCost(),
        power_injection_cost=PowerInjectionCost(),
        vacuum_system_cost=VacuumSystemCost(),
        tf_coil_power_conditioning_cost=TfCoilPowerConditioningCost(),
        pf_coil_power_conditioning_cost=PfCoilPowerConditioningCost(),
        energy_storage_cost=EnergyStorageCostUnpulsed(),
        power_conditioning_cost=PowerConditioningCost(),
        reactor_cooling_system_cost=ReactorCoolingSystemCost(),
        auxiliary_component_cooling_cost=AuxiliaryComponentCoolingCost(),
        cryogenic_system_cost=CryogenicSystemCost(),
        heat_transport_system_cost=HeatTransportSystemCost(),
        fuelling_system_cost=FuellingSystemCost(),
        fuel_processing_cost=FuelProcessingCost(),
        atmospheric_recovery_cost=AtmosphericRecoveryCost(),
        nuclear_building_ventilation_cost=NuclearBuildingVentilationCost(),
        fuel_handling_cost=FuelHandlingCost(),
        instrumentation_and_control_cost=InstrumentationAndControlCost(),
        maintenance_equipment_cost=MaintenanceEquipmentCost(),
        fusion_power_island_cost=FusionPowerIslandCost(),
        turbine_plant_equipment_cost=TurbinePlantEquipmentCost(),
        switchyard_cost=SwitchyardCost(),
        transformers_cost=TransformersCost(),
        low_voltage_cost=LowVoltageCost(),
        diesel_generators_cost=DieselGeneratorsCost(),
        auxiliary_facility_power_cost=AuxiliaryFacilityPowerCost(),
        electric_plant_equipment_cost=ElectricPlantEquipmentCost(),
        misc_plant_equipment_cost=MiscPlantEquipmentCost(),
        heat_rejection_cost=HeatRejectionCost(),
        total_plant_direct_cost=TotalPlantDirectCost(),
        indirect_costs=IndirectCosts(),
        constructed_cost=ConstructedCost(),
        cost_of_electricity=CostOfElectricitySphericalTokamak(),
    ),
    tokamak=Tokamak(
        plasma_geom=TokamakPlasmaGeom(
            minor_radius=PlasmaMinorRadius(),
            shape=Ipdg89XPointPlasmaShape(),
            geometry=DoubleArcPlasmaGeometry(),
        ),
        physics=TokamakPhysics(
            unclipped_radiation_powers=UnclippedRadiationPowers(),
            total_radiation_power=TotalRadiationPower(),
            separatrix_power=SeparatrixPowerNonIgnited(),
            ohmic_heating=PlasmaOhmicHeating(),
            positive_separatrix_power=PositiveSeparatrixPower(),
            re_attachment_metric=EuDemoReAttachmentMetric(),
            psep_over_r_metric=PsepOverRMetric(),
            coulomb_logarithm=CoulombLogarithmIonElectron(),
            plasma_surface_neutron_flux=PlasmaSurfaceNeutronFlux(),
        ),
        plasma_inductance=TokamakPlasmaInductance(
            scalings=PlasmaInternalInductanceScalings(),
            internal_inductance_norm=None,
            volt_seconds=PlasmaVoltSecondRequirements(),
        ),
        plasma_beta=TokamakPlasmaBeta(
            energy_from_beta=PlasmaEnergyFromBeta(),
            norm_max=None,
            limit=BetaLimitFromNorm(),
            toroidal=ToroidalBeta(),
            poloidal=PoloidalBeta(),
            thermal=ThermalBeta(),
        ),
        plasma_current=TokamakPlasmaCurrent(
            plasma_current=FiestaStPlasmaCurrent(),
            cylindrical_safety_factor=PlasmaCylindricalSafetyFactor(),
            current_profile_index=None,
        ),
        bootstrap_current=SauterBootstrapCurrentFraction(
            n_plasma_profile_elements=201,
        ),
        diamagnetic_current=SceneDiamagneticCurrent(),
        pfirsch_schluter_current=ScenePfirschSchluterCurrent(),
        current_fractions=PlasmaCurrentFractions(),
        l_h_transition=Martin08AspectNominalLHThresholdPower(),
        scrape_off_layer=TokamakScrapeOffLayer(
            eich2013_sol_power_decay_length=Eich2013SOLPowerDecayLength(),
            mast2014_sol_power_decay_length_1=Mast2014SOLPowerDecayLength1(),
            mast2014_sol_power_decay_length_2=Mast2014SOLPowerDecayLength2(),
            outboard_power_decay_length=OutboardSOLPowerDecayLengthEich2013(),
            upstream_sol_outboard_parallel_area=UpstreamSOLOutboardParallelArea(),
            upstream_sol_outboard_eich13_parallel_area=UpstreamSOLOutboardEich13ParallelArea(),
            outboard_sol_parallel_power_flux=OutboardSOLParallelPowerFlux(),
            outboard_sol_eich13_parallel_power_flux=OutboardSOLEich13ParallelPowerFlux(),
        ),
        density_limit=TokamakDensityLimit(
            greenwald_density_limit=GreenwaldDensityLimit(),
            enforced_density_limit=EnforcedDensityLimitGreenwald(),
            greenwald_fraction=GreenwaldFraction(),
        ),
        current_drive=TokamakCurrentDrive(
            primary_efficiency=HcdPrimaryEfficiencyFreethyEcrhOMode(),
            secondary_heating=HcdSecondaryHeatingNone(),
            secondary_driven_current=HcdSecondaryDrivenCurrent(),
            primary_injected_power=HcdPrimaryInjectedPower(),
            primary_powers=HcdPrimaryPowersElectronCyclotronNoSecondary(),
            injected_power_total=HcdInjectedPowerTotal(),
            electric_total=HcdElectricTotalNonIgnited(),
            fusion_gain=FusionGain(),
        ),
        plasma_fields=PlasmaFields(
            surface_averaged_poloidal_field=SurfaceAveragedPoloidalFieldAmperes(),
            total_magnetic_field=TotalMagneticField(),
            plasma_inboard_toroidal_field=PlasmaInboardToroidalField(),
            plasma_outboard_toroidal_field=PlasmaOutboardToroidalField(),
            total_magnetic_field_inboard=TotalMagneticFieldInboard(),
            total_magnetic_field_outboard=TotalMagneticFieldOutboard(),
        ),
        build=Build(
            plasma_xpoint_heights=PlasmaXpointHeights(),
            divertor_geometry=None,
            z_tf_inside_half=ZTfInsideHalf(),
            tf_top_height=TfTopHeightDoubleNull(),
            blkt_upper_thickness=BlktUpperThickness(),
            dr_tf_inboard_winding_pack=DrTfWpWithInsulationFromInboardBuild(),
            tf_inboard_radii=TfInboardRadiiNoCsPrecomp(),
            vacuum_vessel_and_shield_radii=VacuumVesselAndShieldRadiiTfOutsideCs(),
            radial_build_to_plasma_centre=RadialBuildToPlasmaCentre(),
            shld_inboard_inner_radius=ShldInboardInnerRadius(),
            shld_outboard_outer_radius=ShldOutboardOuterRadius(),
            r_cp_top=RCpTopFromTfInboardOut(),
            dr_tf_outboard=DrTfOutboardSuperconducting(),
            wp_conductor_max_width=WpConductorMaxWidthSuperconducting(),
            tf_outboard_mid_unrippled=TfOutboardMidUnrippled(),
            tf_outboard_mid=TfOutboardMidPictureFrame(),
            tf_outboard_edge_ripple=TfOutboardEdgeRipplePictureFrame(),
            shld_vv_gap_outboard=ShldVvGapOutboard(),
            tf_inner_bore=TfInnerBore(),
        ),
        cicc_superconducting_tf_coil=CrocoSuperconductingTfCoil(
            tf_global_geometry=TfGlobalGeometryCircularCase(),
            dr_tf_plasma_case=DrTfPlasmaCaseFromFraction(),
            dx_tf_side_case_min=DxTfSideCaseMinFromFraction(),
            r_b_tf_inboard_peak=RBTfInboardPeak(),
            tf_current=TfCurrent(),
            tf_coil_shape=TfCoilShapePictureFrameTart(),
            tf_coil_self_inductance=TfCoilSelfInductancePictureFrame(),
            tf_stored_magnetic_energy=TfStoredMagneticEnergy(),
            generic_tf_coil_area_and_masses=GenericTfCoilAreaAndMasses(),
            superconducting_tf_wp_geometry=SuperconductingTfWpGeometryTrapezoidal(),
            tf_case_areas=TfCaseAreasCircularFront(),
            dx_tf_side_case=DxTfSideCaseTrapezoidal(),
            tf_wp_currents=TfWpCurrents(),
            peak_b_tf_inboard_with_ripple=PeakBTfInboardWithRippleFlatAllowance(),
            tf_turn_area=TfTurnArea(),
            superconducting_tf_coil_areas_and_masses=HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(),
            tf_field_and_force=TfFieldAndForceClampedJoints(),
            tf_stress=TfStressExtendedPlaneStrainBuckedCaseAveragedTurn(),
            tf_superconductor_temperature_margin=HazeltonZhaiRebcoCrocoTemperatureMargin(),
            vv_stress_on_quench=VvStressOnQuench(),
            tf_coil_dump_quench_voltage=TfCoilDumpQuenchVoltage(),
            tf_coil_quench_heat_current_density=TfCoilQuenchHeatCurrentDensity.at(tftmp=20.0, temp_tf_conductor_quench_max=150.0),
            croco_turn_geometry=CrocoAveragedTurnGeometryFromCurrentPerTurn(),
            croco_cable_space_properties=CrocoCableSpaceProperties(),
            croco_cable_geometry=CrocoCableGeometry(),
            croco_turn_cable_space_extra_void=CrocoTurnCableSpaceExtraVoid(),
            croco_inboard_areas_and_fractions=CrocoInboardAreasAndFractions(),
            croco_turn_cable_space_cooling_fraction=CrocoTurnCableSpaceCoolingFraction(),
            croco_superconductor_properties=HazeltonZhaiRebcoCrocoSuperconductorProperties(),
        ),
        pf_coil=PFCoilSphericalTokamak(
            placement=PFCoilPlacementSphericalTokamak(),
            positions=PFCoilPositionsNoCentralSolenoid(),
            initiation_currents=PFCoilInitiationCurrentsNoCentralSolenoid(),
            equilibrium_currents=PFCoilEquilibriumCurrentsNoCentralSolenoid(),
            time_point_currents=PFCoilTimePointCurrentsNoCentralSolenoid(),
            waveform=PFCoilCurrentWaveformNoCentralSolenoid(),
            peak_field=PFCoilPeakFieldNoCentralSolenoid(),
            sizes=PFCoilSizesNoCentralSolenoid(),
            masses=PFCoilMassesNoCentralSolenoid(),
            strand_critical_current=PFStrandCriticalCurrentDensityHazeltonZhaiRebco(),
            inductance=PFCoilInductanceNoCentralSolenoid(),
            turn_currents=PFCoilTurnCurrentsNoCentralSolenoid(),
            volt_seconds=PFCoilVoltSecondsNoCentralSolenoid(),
        ),
        cs_coil=None,
        cs_fatigue=CsFatigue(),
        pulse=TokamakPulse(
            ramp_times=PulseRampTimesContinuousDefault(),
            durations=PulseDurations(),
            burn_time=None,
        ),
        divertor=Divertor(
            heat_flux_split=DivertorHeatFluxSplit(),
            heat_load=DivertorHeatLoadWadeDoubleNull(),
        ),
        first_wall=FirstWallDShapedDoubleNull(),
        first_wall_geometry=FirstWallGeometry(),
        radiated_wall_load=RadiatedWallLoad(),
        shield=TokamakShield(
            half_height=DoubleNullShieldHalfHeight(),
            volumes=DShapedShieldVolumes(),
        ),
        vacuum_vessel=VacuumVesselDShapedDoubleNull(),
        ccfe_hcpb=CcfeHcpb(
            blanket_half_height=BlanketHalfHeightDoubleNull(),
            blanket_areas=DShapedBlanketAreas(),
            blanket_volumes=DShapedBlanketVolumes(),
            blanket_coverage_factors=BlanketCoverageFactorsDoubleNull(),
            inboard_poloidal_angle=BlanketInboardPoloidalAngle(),
            first_wall_coolant_void_fractions=FirstWallCoolantVoidFractions(),
            divertor_surface_and_plate_mass=DivertorSurfaceAndPlateMassDoubleNull(),
            component_masses=ComponentMasses(),
            nuclear_heating_magnets=NuclearHeatingMagnetsSphericalTokamak(),
            nuclear_heating_fw=NuclearHeatingFw(),
            nuclear_heating_blanket=NuclearHeatingBlanket(),
            nuclear_heating_shield=NuclearHeatingShieldSphericalTokamak(),
            centrepost_neutronics=CentrepostNeutronicsSphericalTokamakSuperconducting(),
            nuclear_heating_renormalisation=NuclearHeatingRenormalisationDoubleNullSphericalTokamak(),
            first_wall_radiation_powers=FirstWallRadiationPowers(),
            pumping_power=PumpingPowerMechanicalWithPressureDrop(),
        ),
        cryostat=Cryostat(),
        structure=Structure(),
    ),
    physics=Physics(
        fusion_power_totals_mw=FusionPowerTotalsMw(),
        fusion_totals_no_beam=FusionTotalsNoBeam(),
        profiles=PhysicsProfiles(
            parameterisation=ProfileParameterisationPedestal(
                pedestal_separatrix=PedestalSeparatrixDensities(),
                pedestal_temperature_profile=PedestalTemperatureProfile(),
                pedestal_on_axis_densities=PedestalOnAxisDensities(),
                pedestal_on_axis_temperatures=PedestalOnAxisTemperatures(),
                pedestal_profile_values=PedestalProfileValues(),
            ),
            profile_factors=ProfileFactors(),
            profile_grid=ProfileGrid(
                n_plasma_profile_elements=201,
            ),
            ne_profile_integral=NeProfileIntegral(),
            te_profile_integral=TeProfileIntegral(),
            density_profile=DensityProfile(),
            ion_vol_avg_temperature=IonVolAvgTemperature(),
        ),
        confinement_time=PhysicsConfinementTime(
            inputs=ConfinementScalingInputs(),
            elongation=IterPhysicsBasisElongation(),
            power_loss=PlasmaPowerLossNonIgnitedCoreRadiation(),
            scaling=IterIpb98y2ConfinementTime(),
            double_and_triple_product=DoubleAndTripleProduct(),
            tail=ConfinementTailCoreRadiation(),
        ),
        fusion_rates=FusionRates(),
        set_fusion_powers=SetFusionPowers(),
        synchrotron_radiation_power=SynchrotronRadiationPower(),
        impurity_radiation_totals=ImpurityRadiationTotals(
            imp_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        ),
        plasma_radiation_powers=PlasmaRadiationPowers(),
        ion_electron_equilibration=IonElectronEquilibration(),
        auxiliary_physics_quantities=AuxiliaryPhysicsQuantities(),
        total_plasma_heating_power=TotalPlasmaHeatingPower(),
        electron_thermal_energy=ElectronThermalEnergy(),
        ion_thermal_energy=IonThermalEnergy(),
        fast_alpha_beta=FastAlphaBetaWard(),
        plasma_composition=PlasmaCompositionNonIgnited(),
        calculate_effective_charge_ionisation_profiles=CalculateEffectiveChargeIonisationProfiles(),
        dimensionless_plasma_parameters=DimensionlessPlasmaParameters(),
        radiation_fraction=RadiationFraction(),
    ),
    power=Power(
        pf_coil_power=PfCoilPowerSuppliesNoCentralSolenoid(),
        tf_power=TfPowerSuperconducting(),
        component_thermal_powers=ComponentThermalPowersMechSolidOther(),
        delta_eta_step=DeltaEtaStepMechSolidOther(),
        eta_turbine=None,
        etath_liq=EtathLiqSupercriticalCo2(),
        temp_turbine_coolant_in=TempTurbineCoolantInFromLiquidBreeder(),
        p_fw_div_heat_deposited_mw=None,
        p_fw_blkt_coolant_pump_mw=None,
        cryo_q_nuc=CryoQNuc(),
        cryo_q_loads=CryoQLoadsSuperconductingTf(),
        cryo_loads=CryoLoadsActive(),
        acpow=AcpowLine(),
    ),
    buildings=Buildings(
        sizing=Bldgs(),
        tf_coil_envelope=TfCoilEnvelope(),
    ),
    vacuum=Vacuum(
        vacuum_old=VacuumOld(),
        duct_diameter_root_find=DuctDiameterRootFind(),
    ),
    availability=Availability(
        electric_production=PlantElectricProductionSingleCoolant(),
        avail=AvailNeutronFluence(),
        cplife_avail=CplifeAvailSuperconducting(),
    ),
)

values = {
    "build": {
        "dr_blkt_inboard": 0.0,
        "dr_blkt_outboard": 1.0,
        "dr_bore": 0.23375250334739459,
        "dr_vv_inboard": 0.2,
        "dr_vv_outboard": 0.3,
        "dz_vv_upper": 0.3,
        "dz_vv_lower": 0.3,
        "dr_shld_vv_gap_inboard": 0.01,
        "dr_cs_tf_gap": 0.0,
        "gapomin": 0.0,
        "iohcl": 0,
        "i_cs_precomp": 0,
        "i_tf_inside_cs": 0,
        "dr_cs": 0.20016400484967947,
        "i_r_cp_top": 2,
        "dr_fw_plasma_gap_inboard": 0.1,
        "dr_fw_plasma_gap_outboard": 0.1,
        "dr_shld_inboard": 0.39314459807893426,
        "dz_shld_lower": 0.6,
        "dr_shld_outboard": 0.3,
        "dz_shld_upper": 0.3,
        "f_dr_tf_outboard_inboard": 1.0,
        "dr_tf_shld_gap": 0.01,
        "dr_shld_thermal_inboard": 0.05,
        "dr_shld_thermal_outboard": 0.07,
        "dz_shld_thermal": 0.075,
        "dz_shld_vv_gap": 2.0,
        "dz_xpoint_divertor": 0.75,
        "dr_shld_blkt_gap": 0.01,
        "i_blkt_inboard": 0,
        "dr_tf_inboard": 0.9,
    },
    "superconducting_tfcoil": {
        "i_tf_turn_type": 2,
        "dx_tf_hts_tape_rebco": 1e-06,
        "dx_tf_croco_strand_copper": 0.002,
        "dx_tf_hts_tape_copper": 0.0002,
        "dx_tf_hts_tape_hastelloy": 1e-05,
    },
    "buildings": {
        "i_bldgs_size": 0,
    },
    "constraints": {
        "pflux_fw_rad_max": 1.2,
        "f_fw_rad_max": 1.0,
        "p_fusion_total_max_mw": 2500.0,
        "p_plasma_separatrix_rmajor_max_mw": 40.0,
        "f_t_alpha_energy_confinement_min": 5.0,
        "p_plant_electric_net_required_mw": 100.0,
    },
    "costs": {
        "i_cost_model": 0,
        "i_plant_availability": 0,
        "ifueltyp": 0,
        "lsa": 2,
        "output_costs": 1,
    },
    "current_drive": {
        "f_c_plasma_bootstrap_max": 0.9,
        "n_ecrh_harmonic": 2.0,
        "i_ecrh_wave_mode": 0,
        "eta_ecrh_injector_wall_plug": 0.45,
        "feffcd": 1.0,
        "i_hcd_primary": 13,
        "i_hcd_calculations": 1,
        "p_hcd_injected_max": 150.0,
    },
    "divertor": {
        "dz_divertor": 1.0,
    },
    "fwbs": {
        "i_fw_blkt_vv_shape": 1,
        "fw_armour_thickness": 0.003,
        "i_blanket_type": 1,
        "inuclear": 0,
        "i_p_coolant_pumping": 3,
        "i_thermal_electric_conversion": 2,
        "i_fw_coolant_type": 1,
        "etaiso": 0.9,
        "eta_coolant_pump_electric": 0.87,
        "i_fw_blkt_shared_coolant": 0,
        "outlet_temp_liq": 873.0,
    },
    "globals": {
        "runtitle": "st regression",
        "maxcal": 2000,
    },
    "heat_transport": {
        "eta_turbine": 0.4,
        "ipowerflow": 0,
        "i_shld_primary_heat": 1,
    },
    "impurity_radiation": {
        "radius_plasma_core_norm": 0.75,
        "f_p_plasma_core_rad_reduction": 0.7,
        "f_nd_impurity_electrons": np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0010071399747886383, 5e-05]),
    },
    "pf_coil": {
        "i_pf_conductor": 0,
        "i_pf_superconductor": 9,
        "i_r_pf_outside_tf_placement": 1,
        "n_pf_coil_groups": 4,
        "i_pf_location": np.array([2.0, 3.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "n_pf_coils_in_group": np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "rref": np.array([7.0, 7.0, 7.0, 2.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
        "zref": np.array([3.6, 1.2, 2.5, 5.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    "physics": {
        "alphaj": 0.1,
        "i_alphaj": 0,
        "alphan": 0.9,
        "alphat": 1.4,
        "aspect": 1.8,
        "beta_total_vol_avg": 0.13134204235647895,
        "b_plasma_toroidal_on_axis": 3.0,
        "nd_plasma_electrons_vol_avg": 9.69888313737236e+19,
        "beta_norm_max": 5.0,
        "i_beta_norm_max": 0,
        "f_p_div_lower": 0.5,
        "f_nd_plasma_pedestal_greenwald": 0.1,
        "f_nd_plasma_separatrix_greenwald": 0.1,
        "hfact": 1.2,
        "i_bootstrap_current": 4,
        "i_beta_component": 3,
        "i_plasma_current": 9,
        "i_diamagnetic_current": 2,
        "i_density_limit": 7,
        "i_plasma_pedestal": 1,
        "i_pfirsch_schluter_current": 1,
        "radius_plasma_pedestal_density_norm": 0.95,
        "radius_plasma_pedestal_temp_norm": 0.925,
        "tbeta": 2.0,
        "temp_plasma_pedestal_kev": 4.5,
        "temp_plasma_separatrix_kev": 0.125,
        "i_plasma_geometry": 0,
        "itart": 1,
        "itartpf": 1,
        "kappa": 2.8,
        "q95": 5.835830999686161,
        "q0": 2.0,
        "f_nd_alpha_thermal_electron": 0.08870796537675113,
        "ind_plasma_internal_norm": 0.3,
        "i_ind_plasma_internal_norm": 0,
        "rmajor": 4.5,
        "i_single_null": 0,
        "f_sync_reflect": 0.6,
        "temp_plasma_electron_vol_avg_kev": 11.814206849688595,
        "f_temp_plasma_ion_electron": 1.0,
        "triang": 0.5,
    },
    "pulse": {
        "i_pulsed_plant": 0,
    },
    "tfcoil": {
        "sig_tf_case_max": 850000000.0,
        "sig_tf_wp_max": 700000000.0,
        "f_dr_tf_plasma_case": 1e-12,
        "casths_fraction": 0.65,
        "i_tf_stress_model": 0,
        "i_tf_tresca": 0,
        "i_tf_wp_geom": 2,
        "i_tf_case_geom": 0,
        "i_tf_turns_integer": 0,
        "i_tf_sc_mat": 9,
        "i_tf_sup": 1,
        "i_tf_shape": 2,
        "i_tf_bucking": 1,
        "eyoung_res_tf_buck": 205000000000.0,
        "ripple_b_tf_plasma_edge_max": 1.0,
        "i_cp_joints": 0,
        "n_tf_coils": 12.0,
        "tftmp": 20.0,
        "dr_tf_nose_case": 0.1293140904093427,
        "dr_tf_wp_with_insulation": 0.6044340543574178,
        "temp_tf_cryo": 20.0,
        "f_vforce_inboard": 0.5,
    },
    "times": {
        "t_plant_pulse_burn": 1000.0,
        "t_plant_pulse_dwell": 100.0,
    },
}

problem = Problem(
    ixc=(4, 6, 29),
    icc=(1, 2, 11, 9, 5, 24, 15, 62, 81, 17, 56, 33, 31, 32, 67, 30, 46, 16),
    n_equality=3,
    i_figure_merit=7,
    bounds={
        4: (5.0, 25.0),
        6: (5e+19, 5e+20),
        29: (0.1, 0.8),
    },
    switches={
        "bkt_life_csf": 0,
        "i_beta_component": 3,
        "i_cp_lifetime": 0,
        "i_density_limit": 7,
        "i_plant_availability": 0,
        "i_plasma_ignited": 0,
        "i_q95_fixed": 0,
        "i_rad_loss": 1,
        "i_tf_bucking": 1,
        "i_tf_inside_cs": 0,
        "i_tf_sup": 1,
        "ibkt_life": 0,
        "ireactor": 1,
        "istell": 0,
        "itart": 1,
    },
    root_find=True,
)

CONFIGURATION = Configuration(
    name="spherical_tokamak_eval",
    machine=machine,
    values=values,
    problem=problem,
)
