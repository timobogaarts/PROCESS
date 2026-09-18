"""`low_aspect_ratio_DEMO`, stated.

Converted from `tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT` by
`input.indat.configuration_from_indat`; regenerate rather than hand-edit --
`tests/test_configurations.py` checks the two agree.
"""

import numpy as np

from functional_process.configurations import Configuration, Problem
from functional_process.cottax.models.availability.availability import AvailDisplacementsPerAtom
from functional_process.cottax.models.availability.namespace import Availability
from functional_process.cottax.models.blankets.blanket_library import BlanketCoverageFactorsSingleNull, BlanketHalfHeightSingleNull, BlanketInboardPoloidalAngle, EllipticalBlanketAreas, EllipticalBlanketVolumes
from functional_process.cottax.models.blankets.hcpb import CentrepostNeutronicsAbsent, ComponentMasses, DivertorSurfaceAndPlateMassSingleNull, FirstWallCoolantVoidFractions, FirstWallRadiationPowers, NuclearHeatingBlanket, NuclearHeatingFw, NuclearHeatingMagnetsConventional, NuclearHeatingRenormalisationSingleNullConventional, NuclearHeatingShieldConventional, PumpingPowerMechanicalWithPressureDrop
from functional_process.cottax.models.blankets.namespace import CcfeHcpb
from functional_process.cottax.models.build import BlktUpperThickness, DivertorGeometryConventional, DrTfOutboardSuperconducting, DrTfWpWithInsulationFromInboardBuild, PlasmaXpointHeights, RCpTopFromTfInboardOut, RadialBuildToPlasmaCentre, ShldInboardInnerRadius, ShldOutboardOuterRadius, ShldVvGapOutboard, TfInboardRadiiTfOutsideCs, TfInnerBore, TfOutboardEdgeRipple, TfOutboardMidDShape, TfOutboardMidUnrippled, TfTopHeightSingleNull, VacuumVesselAndShieldRadiiTfOutsideCs, WpConductorMaxWidthSuperconducting, ZTfInsideHalf
from functional_process.cottax.models.buildings.buildings import Bldgs, TfCoilEnvelope
from functional_process.cottax.models.buildings.namespace import Buildings
from functional_process.cottax.models.costs.costs import AtmosphericRecoveryCost, AuxiliaryComponentCoolingCost, AuxiliaryFacilityPowerCost, BlanketCost, ConstructedCost, ConvertFpyToCalendar, CostOfElectricityConventionalAspectRatio, CryogenicSystemCost, DieselGeneratorsCost, DivertorCost, ElectricPlantEquipmentCost, EnergyStorageCostPulsedElectrowattOption1, FirstWallCost, FuelHandlingCost, FuelProcessingCost, FuellingSystemCost, FusionPowerIslandCost, HeatRejectionCost, HeatTransportSystemCost, IndirectCosts, InstrumentationAndControlCost, LowVoltageCost, MagnetsCost, MaintenanceEquipmentCost, MiscPlantEquipmentCost, NuclearBuildingVentilationCost, PfCoilPowerConditioningCost, PfMagnetCostPerKgCsWstNb3Sn, PowerConditioningCost, PowerInjectionCost, ReactorCoolingSystemCost, ReactorCost, ReactorStructureCost, ShieldCost, StructuresCost, SwitchyardCost, TfCoilPowerConditioningCost, TfMagnetCostSuperconductingPerKg, TotalPlantDirectCost, TransformersCost, TurbinePlantEquipmentCost, VacuumSystemCost, VacuumVesselAssemblyCost
from functional_process.cottax.models.costs.namespace import Costs
from functional_process.cottax.models.cryostat import Cryostat
from functional_process.cottax.models.cs_fatigue import CsFatigue
from functional_process.cottax.models.divertor import DivertorHeatFluxSplit, DivertorHeatLoadWadeSingleNull
from functional_process.cottax.models.fw import FirstWallGeometry, FirstWallSingleNull, RadiatedWallLoad
from functional_process.cottax.models.initialisation import BeamElectronDensityFraction, Initialisation, PfCoilResistivity, TfConductorYoungsModulus, TfCryoplantEfficiency, TfInsulationYoungsModulus
from functional_process.cottax.models.namespace import Build, Divertor
from functional_process.cottax.models.pfcoil.currents import CSCurrentDensityPulseStart, CSFluxSwing, PFCoilEquilibriumCurrents, PFCoilInitiationCurrents, PFCoilTimePointCurrents
from functional_process.cottax.models.pfcoil.fields import CSCoilPeakField, PFCoilCurrentWaveform, PFCoilPeakField
from functional_process.cottax.models.pfcoil.geometry import CSCoilGeometry, CSCoilTurnGeometry, PFCoilPlacement, PFCoilPositions
from functional_process.cottax.models.pfcoil.inductance import PFCoilInductance
from functional_process.cottax.models.pfcoil.masses import PFCoilMassesCsWstNb3Sn, PFCoilSizes
from functional_process.cottax.models.pfcoil.namespace import CSCoil, PFCoilCsWstNb3Sn
from functional_process.cottax.models.pfcoil.stresses import CSCoilStresses
from functional_process.cottax.models.pfcoil.superconductor import CSCriticalCurrentDensitiesWstNb3Sn, CSTemperatureMarginWstNb3Sn, PFStrandCriticalCurrentDensity
from functional_process.cottax.models.pfcoil.volt_seconds import PFCoilTurnCurrents, PFCoilVoltSeconds
from functional_process.cottax.models.physics.bootstrap_current import NoDiamagneticCurrent, NoPfirschSchluterCurrent, PlasmaCurrentFractions, SauterBootstrapCurrentFraction
from functional_process.cottax.models.physics.composition import CalculateEffectiveChargeIonisationProfiles, PlasmaCompositionNonIgnited
from functional_process.cottax.models.physics.confinement_time import ConfinementScalingInputs, ConfinementTailCoreRadiation, DoubleAndTripleProduct, IterIpb98y2ConfinementTime, IterPhysicsBasisElongation, PlasmaPowerLossNonIgnitedCoreRadiation
from functional_process.cottax.models.physics.current_drive import FusionGain, HcdElectricTotalNonIgnited, HcdInjectedPowerTotal, HcdPrimaryEfficiencyUserInputEcrh, HcdPrimaryInjectedPower, HcdPrimaryPowersElectronCyclotronNoSecondary, HcdSecondaryDrivenCurrent, HcdSecondaryHeatingNone
from functional_process.cottax.models.physics.density_limit import EnforcedDensityLimitGreenwald, GreenwaldDensityLimit, GreenwaldFraction, TokamakDensityLimit
from functional_process.cottax.models.physics.dimensionless_parameters import DimensionlessPlasmaParameters
from functional_process.cottax.models.physics.exhaust import EuDemoReAttachmentMetric, PsepOverRMetric, RadiationFraction
from functional_process.cottax.models.physics.fusion_reactions import FusionRates, SetFusionPowers
from functional_process.cottax.models.physics.l_h_transition import Martin08AspectNominalLHThresholdPower
from functional_process.cottax.models.physics.namespace import Physics, PhysicsConfinementTime, PhysicsProfiles, ProfileParameterisationPedestal
from functional_process.cottax.models.physics.physics import BetaLimitFromNorm, BetaNormMaxWesson, CoulombLogarithmIonElectron, PlasmaEnergyFromBeta, PlasmaOhmicHeating, PlasmaSurfaceNeutronFlux, PoloidalBeta, PositiveSeparatrixPower, PulseRampTimesPulsedDefault, SeparatrixPowerNonIgnited, SurfaceAveragedPoloidalFieldAmperes, ThermalBeta, ToroidalBeta, TotalRadiationPower, UnclippedRadiationPowers
from functional_process.cottax.models.physics.plasma_current import Ipdg89PlasmaCurrent, PlasmaCylindricalSafetyFactor, TokamakPlasmaCurrent
from functional_process.cottax.models.physics.plasma_fields import PlasmaFields, PlasmaInboardToroidalField, PlasmaOutboardToroidalField, TotalMagneticField, TotalMagneticFieldInboard, TotalMagneticFieldOutboard
from functional_process.cottax.models.physics.plasma_geometry import CreateDataEuDemoXPointPlasmaShape, DoubleArcPlasmaGeometry, PlasmaMinorRadius
from functional_process.cottax.models.physics.plasma_inductance import PlasmaInternalInductanceScalings, PlasmaVoltSecondRequirements, TokamakPlasmaInductance
from functional_process.cottax.models.physics.plasma_profiles import IonVolAvgTemperature, PedestalProfileValues, ProfileFactors
from functional_process.cottax.models.physics.profiles import DensityProfile, NeProfileIntegral, PedestalOnAxisDensities, PedestalOnAxisTemperatures, PedestalSeparatrixDensities, PedestalTemperatureProfile, ProfileGrid, TeProfileIntegral
from functional_process.cottax.models.physics.pure_formulas import AuxiliaryPhysicsQuantities, ElectronThermalEnergy, FastAlphaBetaWard, IonElectronEquilibration, IonThermalEnergy, TotalPlasmaHeatingPower
from functional_process.cottax.models.physics.radiation_power import ImpurityRadiationTotals, PlasmaRadiationPowers, SynchrotronRadiationPower
from functional_process.cottax.models.physics.scrape_off_layer import Eich2013SOLPowerDecayLength, Mast2014SOLPowerDecayLength1, Mast2014SOLPowerDecayLength2, OutboardSOLEich13ParallelPowerFlux, OutboardSOLParallelPowerFlux, OutboardSOLPowerDecayLengthEich2013, TokamakScrapeOffLayer, UpstreamSOLOutboardEich13ParallelArea, UpstreamSOLOutboardParallelArea
from functional_process.cottax.models.physics.tokamak_namespace import TokamakCurrentDrive, TokamakPhysics, TokamakPlasmaBeta, TokamakPlasmaGeom, TokamakPulse
from functional_process.cottax.models.power.electric_production import AcpowLine, PlantElectricProductionSingleCoolant
from functional_process.cottax.models.power.namespace import Power
from functional_process.cottax.models.power.pf_coil_power import PfCoilPowerSuppliesReference
from functional_process.cottax.models.power.tf_coil_power import TfPowerSuperconducting
from functional_process.cottax.models.power.thermal_cryo import ComponentThermalPowersMechSolidOther, CryoLoadsActive, CryoQLoadsSuperconductingTf, DeltaEtaStepMechSolidOther, EtathLiqSupercriticalCo2, TempTurbineCoolantInFromLiquidBreeder
from functional_process.cottax.models.pulse import PulseBurnTime
from functional_process.cottax.models.shield import EllipticalShieldVolumes, SingleNullShieldHalfHeight, TokamakShield
from functional_process.cottax.models.stellarator.initialization import PulseDurations
from functional_process.cottax.models.stellarator.plasma_physics import FusionPowerTotalsMw, FusionTotalsNoBeam
from functional_process.cottax.models.structure import Structure
from functional_process.cottax.models.tfcoil.base import DrTfPlasmaCaseFromInput, GenericTfCoilAreaAndMasses, RBTfInboardPeak, TfCoilSelfInductanceDShape, TfCoilShapeDShapeSingleNull, TfCurrent, TfGlobalGeometryCircularCase, TfStoredMagneticEnergy
from functional_process.cottax.models.tfcoil.namespace import CiccSuperconductingTfCoil
from functional_process.cottax.models.tfcoil.quench import TfCoilDumpQuenchVoltage, TfCoilQuenchHeatCurrentDensity
from functional_process.cottax.models.tfcoil.stress import TfFieldAndForceClampedJoints, TfStressPlaneStressBuckedCaseIntegerTurn
from functional_process.cottax.models.tfcoil.superconducting import CiccInboardAreasAndFractions, CiccIntegerTurnGeometry, DxTfSideCaseRectangular, PeakBTfInboardWithRipple16Coils, SuperconductingTfWpGeometryRectangular, TfCaseAreasCircularFront, TfTurnArea, TfWpCurrents, VvStressOnQuench, WstNb3snCiccSuperconductorProperties, WstNb3snSuperconductingTfCoilAreasAndMassesConventional, WstNb3snTfSuperconductorTemperatureMargin
from functional_process.cottax.models.tokamak.namespace import Tokamak
from functional_process.cottax.models.total_process import TokamakProcess
from functional_process.cottax.models.vacuum.namespace import Vacuum
from functional_process.cottax.models.vacuum.vacuum import DuctDiameterRootFind, VacuumOld, VacuumVesselEllipticalSingleNull
from functional_process.vocabulary.enums import PFConductorModel

machine = TokamakProcess(
    initialisation=Initialisation(
        tf_cryoplant_efficiency=TfCryoplantEfficiency(),
        tf_insulation_youngs_modulus=TfInsulationYoungsModulus(),
        tf_conductor_youngs_modulus=TfConductorYoungsModulus(),
        pf_coil_resistivity=PfCoilResistivity(),
        beam_electron_density_fraction=BeamElectronDensityFraction(),
        energy_storage_building_volume=None,
        double_null_upper_build=None,
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
        pf_magnet_cost=PfMagnetCostPerKgCsWstNb3Sn(
            n_cs_pf_coils=7,
            i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
        ),
        vacuum_vessel_assembly_cost=VacuumVesselAssemblyCost(),
        magnets_cost=MagnetsCost(),
        power_injection_cost=PowerInjectionCost(),
        vacuum_system_cost=VacuumSystemCost(),
        tf_coil_power_conditioning_cost=TfCoilPowerConditioningCost(),
        pf_coil_power_conditioning_cost=PfCoilPowerConditioningCost(),
        energy_storage_cost=EnergyStorageCostPulsedElectrowattOption1(),
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
        cost_of_electricity=CostOfElectricityConventionalAspectRatio(),
    ),
    tokamak=Tokamak(
        plasma_geom=TokamakPlasmaGeom(
            minor_radius=PlasmaMinorRadius(),
            shape=CreateDataEuDemoXPointPlasmaShape(),
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
            norm_max=BetaNormMaxWesson(),
            limit=BetaLimitFromNorm(),
            toroidal=ToroidalBeta(),
            poloidal=PoloidalBeta(),
            thermal=ThermalBeta(),
        ),
        plasma_current=TokamakPlasmaCurrent(
            plasma_current=Ipdg89PlasmaCurrent(),
            cylindrical_safety_factor=PlasmaCylindricalSafetyFactor(),
            current_profile_index=None,
        ),
        bootstrap_current=SauterBootstrapCurrentFraction(
            n_plasma_profile_elements=201,
        ),
        diamagnetic_current=NoDiamagneticCurrent(),
        pfirsch_schluter_current=NoPfirschSchluterCurrent(),
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
            primary_efficiency=HcdPrimaryEfficiencyUserInputEcrh(),
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
            divertor_geometry=DivertorGeometryConventional(),
            z_tf_inside_half=ZTfInsideHalf(),
            tf_top_height=TfTopHeightSingleNull(),
            blkt_upper_thickness=BlktUpperThickness(),
            dr_tf_inboard_winding_pack=DrTfWpWithInsulationFromInboardBuild(),
            tf_inboard_radii=TfInboardRadiiTfOutsideCs(),
            vacuum_vessel_and_shield_radii=VacuumVesselAndShieldRadiiTfOutsideCs(),
            radial_build_to_plasma_centre=RadialBuildToPlasmaCentre(),
            shld_inboard_inner_radius=ShldInboardInnerRadius(),
            shld_outboard_outer_radius=ShldOutboardOuterRadius(),
            r_cp_top=RCpTopFromTfInboardOut(),
            dr_tf_outboard=DrTfOutboardSuperconducting(),
            wp_conductor_max_width=WpConductorMaxWidthSuperconducting(),
            tf_outboard_mid_unrippled=TfOutboardMidUnrippled(),
            tf_outboard_mid=TfOutboardMidDShape(),
            tf_outboard_edge_ripple=TfOutboardEdgeRipple(),
            shld_vv_gap_outboard=ShldVvGapOutboard(),
            tf_inner_bore=TfInnerBore(),
        ),
        cicc_superconducting_tf_coil=CiccSuperconductingTfCoil(
            tf_global_geometry=TfGlobalGeometryCircularCase(),
            dr_tf_plasma_case=DrTfPlasmaCaseFromInput(),
            dx_tf_side_case_min=None,
            r_b_tf_inboard_peak=RBTfInboardPeak(),
            tf_current=TfCurrent(),
            tf_coil_shape=TfCoilShapeDShapeSingleNull(),
            tf_coil_self_inductance=TfCoilSelfInductanceDShape(),
            tf_stored_magnetic_energy=TfStoredMagneticEnergy(),
            generic_tf_coil_area_and_masses=GenericTfCoilAreaAndMasses(),
            superconducting_tf_wp_geometry=SuperconductingTfWpGeometryRectangular(),
            tf_case_areas=TfCaseAreasCircularFront(),
            dx_tf_side_case=DxTfSideCaseRectangular(),
            tf_wp_currents=TfWpCurrents(),
            peak_b_tf_inboard_with_ripple=PeakBTfInboardWithRipple16Coils(),
            tf_turn_area=TfTurnArea(),
            superconducting_tf_coil_areas_and_masses=WstNb3snSuperconductingTfCoilAreasAndMassesConventional(),
            tf_field_and_force=TfFieldAndForceClampedJoints(),
            tf_stress=TfStressPlaneStressBuckedCaseIntegerTurn(),
            tf_superconductor_temperature_margin=WstNb3snTfSuperconductorTemperatureMargin(),
            vv_stress_on_quench=VvStressOnQuench(),
            tf_coil_dump_quench_voltage=TfCoilDumpQuenchVoltage(),
            tf_coil_quench_heat_current_density=TfCoilQuenchHeatCurrentDensity.at(tftmp=4.75, temp_tf_conductor_quench_max=150.0),
            cicc_turn_geometry=CiccIntegerTurnGeometry(),
            cicc_inboard_areas_and_fractions=CiccInboardAreasAndFractions(),
            cicc_superconductor_properties=WstNb3snCiccSuperconductorProperties(),
        ),
        pf_coil=PFCoilCsWstNb3Sn(
            placement=PFCoilPlacement(),
            positions=PFCoilPositions(),
            initiation_currents=PFCoilInitiationCurrents(),
            equilibrium_currents=PFCoilEquilibriumCurrents(),
            time_point_currents=PFCoilTimePointCurrents(),
            waveform=PFCoilCurrentWaveform(),
            peak_field=PFCoilPeakField(),
            sizes=PFCoilSizes(),
            masses=PFCoilMassesCsWstNb3Sn(),
            strand_critical_current=PFStrandCriticalCurrentDensity(),
            inductance=PFCoilInductance(),
            turn_currents=PFCoilTurnCurrents(),
            volt_seconds=PFCoilVoltSeconds(),
        ),
        cs_coil=CSCoil(
            geometry=CSCoilGeometry(),
            turn_geometry=CSCoilTurnGeometry(),
            current_density_pulse_start=CSCurrentDensityPulseStart(),
            flux_swing=CSFluxSwing(),
            peak_field=CSCoilPeakField(),
            critical_current=CSCriticalCurrentDensitiesWstNb3Sn(),
            temperature_margin=CSTemperatureMarginWstNb3Sn(),
            stresses=CSCoilStresses(),
        ),
        cs_fatigue=CsFatigue(),
        pulse=TokamakPulse(
            ramp_times=PulseRampTimesPulsedDefault(),
            durations=PulseDurations(),
            burn_time=PulseBurnTime(),
        ),
        divertor=Divertor(
            heat_flux_split=DivertorHeatFluxSplit(),
            heat_load=DivertorHeatLoadWadeSingleNull(),
        ),
        first_wall=FirstWallSingleNull(),
        first_wall_geometry=FirstWallGeometry(),
        radiated_wall_load=RadiatedWallLoad(),
        shield=TokamakShield(
            half_height=SingleNullShieldHalfHeight(),
            volumes=EllipticalShieldVolumes(),
        ),
        vacuum_vessel=VacuumVesselEllipticalSingleNull(),
        ccfe_hcpb=CcfeHcpb(
            blanket_half_height=BlanketHalfHeightSingleNull(),
            blanket_areas=EllipticalBlanketAreas(),
            blanket_volumes=EllipticalBlanketVolumes(),
            blanket_coverage_factors=BlanketCoverageFactorsSingleNull(),
            inboard_poloidal_angle=BlanketInboardPoloidalAngle(),
            first_wall_coolant_void_fractions=FirstWallCoolantVoidFractions(),
            divertor_surface_and_plate_mass=DivertorSurfaceAndPlateMassSingleNull(),
            component_masses=ComponentMasses(),
            nuclear_heating_magnets=NuclearHeatingMagnetsConventional(),
            nuclear_heating_fw=NuclearHeatingFw(),
            nuclear_heating_blanket=NuclearHeatingBlanket(),
            nuclear_heating_shield=NuclearHeatingShieldConventional(),
            centrepost_neutronics=CentrepostNeutronicsAbsent(),
            nuclear_heating_renormalisation=NuclearHeatingRenormalisationSingleNullConventional(),
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
        pf_coil_power=PfCoilPowerSuppliesReference(),
        tf_power=TfPowerSuperconducting(),
        component_thermal_powers=ComponentThermalPowersMechSolidOther(),
        delta_eta_step=DeltaEtaStepMechSolidOther(),
        eta_turbine=None,
        etath_liq=EtathLiqSupercriticalCo2(),
        temp_turbine_coolant_in=TempTurbineCoolantInFromLiquidBreeder(),
        p_fw_div_heat_deposited_mw=None,
        p_fw_blkt_coolant_pump_mw=None,
        cryo_q_nuc=None,
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
        avail=AvailDisplacementsPerAtom(),
        cplife_avail=None,
    ),
)

values = {
    "globals": {
        "maxcal": 100,
        "runtitle": "EU-DEMO LAR 2023 baseline",
    },
    "physics": {
        "b_plasma_toroidal_on_axis": 4.3968920403531495,
        "rmajor": 8.6,
        "temp_plasma_electron_vol_avg_kev": 11.002209848943476,
        "beta_total_vol_avg": 0.037196562691344616,
        "nd_plasma_electrons_vol_avg": 6.883600416583643e+19,
        "q95": 3.6,
        "f_c_plasma_non_inductive": 0.4926778753808451,
        "f_nd_alpha_thermal_electron": 0.06149554941478568,
        "alphan": 1.0,
        "alphat": 1.45,
        "aspect": 2.8,
        "beta_norm_max": 3.0,
        "fkzohm": 1.0245,
        "ejima_coeff": 0.2,
        "hfact": 1.1,
        "i_bootstrap_current": 4,
        "i_beta_component": 1,
        "i_plasma_current": 4,
        "i_density_limit": 7,
        "i_beta_fast_alpha": 1,
        "i_confinement_time": 34,
        "i_plasma_geometry": 10,
        "m_s_limit": 0.2,
        "kappa": 1.848,
        "triang": 0.5,
        "q0": 1.0,
        "i_single_null": 1,
        "f_sync_reflect": 0.6,
        "plasma_res_factor": 0.66,
        "i_ind_plasma_internal_norm": 0,
        "ind_plasma_internal_norm": 1.21,
        "i_plasma_pedestal": 1,
        "f_nd_plasma_pedestal_greenwald": 0.85,
        "nd_plasma_pedestal_electron": 6.78e+19,
        "nd_plasma_separatrix_electron": 2e+19,
        "radius_plasma_pedestal_density_norm": 0.94,
        "radius_plasma_pedestal_temp_norm": 0.94,
        "tbeta": 2.0,
        "temp_plasma_pedestal_kev": 4.199,
        "temp_plasma_separatrix_kev": 0.1,
    },
    "constraints": {
        "f_nd_plasma_electron_limit_max": 1.2,
        "f_j_tf_wp_critical_max": 1.0,
        "f_h_mode_margin": 1.1,
        "p_plant_electric_net_required_mw": 350.0,
        "t_burn_min": 5000.0,
        "pflux_fw_neutron_max_mw": 8.0,
        "p_div_bt_q_aspect_rmajor_max_mw": 6.0,
        "b_tf_inboard_max": 11.2,
        "i_q95_fixed": 1,
        "q95_fixed": 3.3,
        "flu_tf_neutron_fast_max": 0.0,
        "p_hcd_injected_min_mw": 40.0,
    },
    "current_drive": {
        "p_hcd_primary_extra_heat_mw": 10.0,
        "f_c_plasma_bootstrap_max": 0.99,
        "i_hcd_primary": 10,
        "eta_cd_norm_ecrh": 0.3,
        "eta_ecrh_injector_wall_plug": 0.4,
        "p_hcd_injected_max": 50.0,
    },
    "build": {
        "dr_tf_inboard": 0.9998067431742009,
        "dr_cs": 0.6482773936989767,
        "dr_bore": 2.0266655751565996,
        "dr_cs_tf_gap": 0.05,
        "dr_shld_vv_gap_inboard": 0.02,
        "dr_tf_shld_gap": 0.05,
        "dr_vv_inboard": 0.3,
        "dr_shld_inboard": 0.3,
        "dr_shld_blkt_gap": 0.02,
        "dr_blkt_inboard": 0.755,
        "dr_fw_plasma_gap_inboard": 0.225,
        "dr_fw_plasma_gap_outboard": 0.225,
        "dr_blkt_outboard": 0.982,
        "dr_shld_outboard": 0.8,
        "dr_vv_outboard": 0.3,
        "dr_cryostat": 0.15,
        "gapomin": 0.2,
        "dz_vv_upper": 0.3,
        "dz_shld_vv_gap": 0.05,
        "dz_shld_upper": 0.3,
        "dz_vv_lower": 0.3,
        "iohcl": 1,
    },
    "pf_coil": {
        "j_cs_flat_top_end": 16348974.510353195,
        "f_j_cs_start_pulse_end_flat_top": 0.8513539690626537,
        "f_a_cs_turn_steel": 0.7597743217586591,
        "stress_cs_steel_max": 660000000.0,
        "i_pf_superconductor": 3,
        "n_pf_coil_groups": 4,
        "f_z_cs_tf_internal": 0.9,
        "rpf2": -1.825,
        "fcuohsu": 0.7,
        "i_cs_superconductor": 5,
        "c_pf_coil_turn_peak_input": np.array([42200.0, 42200.0, 42200.0, 42200.0, 43000.0, 43000.0, 43000.0, 43000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "i_pf_location": np.array([2.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "n_pf_coils_in_group": np.array([1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "j_pf_coil_wp_peak": np.array([11000000.0, 11000000.0, 6000000.0, 6000000.0, 8000000.0, 8000000.0, 8000000.0, 8000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "zref": np.array([3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    "tfcoil": {
        "t_tf_superconductor_quench": 23.19188632487349,
        "dr_tf_nose_case": 0.25906136679742703,
        "dx_tf_turn_steel": 0.008,
        "f_a_tf_turn_cable_copper": 0.9048073494888599,
        "c_tf_turn": 65000.0,
        "sig_tf_case_max": 580000000.0,
        "sig_tf_wp_max": 580000000.0,
        "n_tf_coils": 16.0,
        "i_tf_sc_mat": 5,
        "dr_tf_plasma_case": 0.06,
        "dx_tf_side_case_min": 0.05,
        "ripple_b_tf_plasma_edge_max": 0.6,
        "dia_tf_turn_coolant_channel": 0.01,
        "tftmp": 4.75,
        "dx_tf_turn_insulation": 0.002,
        "dx_tf_wp_insulation": 0.008,
        "tmargmin": 1.5,
        "f_a_tf_turn_cable_space_extra_void": 0.3,
        "i_tf_turns_integer": 1,
        "n_tf_wp_pancakes": 20,
        "n_tf_wp_layers": 10,
        "v_tf_coil_dump_quench_max_kv": 10.0,
    },
    "divertor": {
        "dz_divertor": 0.621,
        "pflux_div_heat_load_max_mw": 10.0,
        "prn1": 0.4,
    },
    "costs": {
        "output_costs": 0,
        "i_cost_model": 0,
        "abktflnc": 15.0,
        "adivflnc": 20.0,
        "f_t_plant_available": 0.75,
        "dintrt": 0.0,
        "fcap0": 1.15,
        "fcap0cp": 1.06,
        "fcontng": 0.15,
        "fcr0": 0.065,
        "fkind": 1.0,
        "i_plant_availability": 0,
        "ibkt_life": 1,
        "life_dpa": 70.0,
        "ifueltyp": 1,
        "lsa": 2,
        "discount_rate": 0.06,
        "life_plant": 40.0,
        "ucblvd": 280.0,
        "ucdiv": 500000.0,
        "ucme": 300000000.0,
    },
    "cs_fatigue": {
        "bkt_life_csf": 1.0,
        "t_crack_vertical": 0.00065,
        "sf_vertical_crack": 1.0,
        "sf_radial_crack": 1.0,
        "sf_fast_fracture": 1.0,
        "residual_sig_hoop": 150000000.0,
        "paris_coefficient": 3.86e-11,
        "paris_power_law": 2.394,
        "walker_coefficient": 0.5,
        "fracture_toughness": 150.0,
    },
    "fwbs": {
        "vfshld": 0.6,
        "f_p_blkt_multiplication": 1.2,
        "i_p_coolant_pumping": 3,
        "eta_coolant_pump_electric": 0.9,
        "etaiso": 0.85,
        "i_thermal_electric_conversion": 2,
        "i_blanket_type": 1,
        "inuclear": 1,
        "qnuc": 12920.0,
    },
    "primary_pumping": {
        "dp_he": 268000.0,
        "f_p_fw_blkt_pump": 0.26666667,
        "t_out_bb": 793.15,
    },
    "heat_transport": {
        "ipowerflow": 0,
        "i_shld_primary_heat": 1,
        "eta_turbine": 0.316,
    },
    "impurity_radiation": {
        "radius_plasma_core_norm": 0.75,
        "f_p_plasma_core_rad_reduction": 0.6,
        "f_nd_impurity_electrons": np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00015, 5e-05]),
    },
    "pulse": {
        "i_pulsed_plant": 1,
    },
    "times": {
        "t_plant_pulse_dwell": 0.0,
        "pulsetimings": 0,
        "t_plant_pulse_coil_precharge": 500.0,
    },
}

problem = Problem(
    ixc=(2, 3, 4, 5, 6, 13, 16, 18, 29, 37, 41, 44, 56, 57, 58, 59, 109, 122, 135),
    icc=(1, 2, 11, 30, 5, 8, 13, 15, 16, 24, 26, 27, 31, 32, 33, 34, 35, 36, 60, 62, 65, 68, 72, 81, 90),
    n_equality=4,
    i_figure_merit=-14,
    bounds={
        2: (0.01, 20.0),
        3: (8.0, 8.6),
        4: (5.0, 150.0),
        5: (0.001, 1.0),
        6: (2e+19, 1e+21),
        13: (0.3, 5.0),
        16: (0.1, 10.0),
        18: (3.6, 50.0),
        29: (0.1, 10.0),
        37: (100000.0, 100000000.0),
        41: (0.001, 1.0),
        44: (0.001, 1.0),
        56: (0.1, 100.0),
        57: (0.05, 1.0),
        58: (0.008, 0.1),
        59: (0.5, 0.94),
        109: (0.05, 0.15),
        122: (0.001, 0.95),
        135: (1e-08, 0.01),
    },
    switches={
        "bkt_life_csf": 1,
        "i_beta_component": 1,
        "i_cp_lifetime": 0,
        "i_density_limit": 7,
        "i_plant_availability": 0,
        "i_plasma_ignited": 0,
        "i_q95_fixed": 1,
        "i_rad_loss": 1,
        "i_tf_bucking": 1,
        "i_tf_inside_cs": 0,
        "i_tf_sup": 1,
        "ibkt_life": 1,
        "ireactor": 1,
        "istell": 0,
        "itart": 0,
    },
)

CONFIGURATION = Configuration(
    name="low_aspect_ratio_DEMO",
    machine=machine,
    values=values,
    problem=problem,
)
