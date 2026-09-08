"""PROCESS's input encoding, and the one place this port reads it."""

import functools
import re
from pathlib import Path

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import to_graph

from functional_process.cottax.availability.availability import (
    AvailDisplacementsPerAtom,
    AvailNeutronFluence,
    CplifeAvailResistive,
    CplifeAvailSuperconducting,
)
from functional_process.cottax.availability.namespace import Availability
from functional_process.cottax.blankets.blanket_library import (
    BlanketCoverageFactorsDoubleNull,
    BlanketCoverageFactorsSingleNull,
    BlanketHalfHeightDoubleNull,
    BlanketHalfHeightSingleNull,
    DShapedBlanketAreas,
    DShapedBlanketVolumes,
    EllipticalBlanketAreas,
    EllipticalBlanketVolumes,
)
from functional_process.cottax.blankets.hcpb import (
    CentrepostNeutronicsAbsent,
    calculate_centrepost_neutronics_absent,
    CentrepostNeutronicsSphericalTokamakSuperconducting,
    DivertorSurfaceAndPlateMassDoubleNull,
    DivertorSurfaceAndPlateMassSingleNull,
    NuclearHeatingMagnetsConventional,
    NuclearHeatingMagnetsSphericalTokamak,
    NuclearHeatingRenormalisationDoubleNullConventional,
    NuclearHeatingRenormalisationDoubleNullSphericalTokamak,
    NuclearHeatingRenormalisationSingleNullConventional,
    NuclearHeatingRenormalisationSingleNullSphericalTokamak,
    NuclearHeatingShieldConventional,
    NuclearHeatingShieldSphericalTokamak,
    PumpingPowerMechanicalWithPressureDrop,
)
from functional_process.cottax.blankets.namespace import CcfeHcpb
from functional_process.cottax.build import (
    DivertorGeometryConventional,
    DivertorGeometrySphericalTokamak,
    DrTfInboardFromWindingPack,
    DrTfOutboardSuperconducting,
    DrTfWpWithInsulationFromInboardBuild,
    RCpTopFromTfInboardOut,
    TfInboardRadiiNoCsPrecomp,
    TfInboardRadiiTfOutsideCs,
    TfOutboardEdgeRipple,
    TfOutboardEdgeRipplePictureFrame,
    TfOutboardMidDShape,
    TfOutboardMidPictureFrame,
    TfTopHeightDoubleNull,
    TfTopHeightSingleNull,
    VacuumVesselAndShieldRadiiTfOutsideCs,
    WpConductorMaxWidthSuperconducting,
)
from functional_process.cottax.buildings.buildings import (
    Bldgs,
    BldgsSizes,
)
from functional_process.cottax.buildings.namespace import Buildings
from functional_process.cottax.costs.costs import (
    CostOfElectricityConventionalAspectRatio,
    CostOfElectricitySphericalTokamak,
    EnergyStorageCostPulsedElectrowattOption1,
    EnergyStorageCostPulsedElectrowattOption2,
    EnergyStorageCostUnpulsed,
    PfCoilPowerConditioningCost,
    PfMagnetCostPerKam,
    PfMagnetCostPerKamNoCentralSolenoid,
    PfMagnetCostPerKg,
    PfMagnetCostPerKgCsWstNb3Sn,
    PfMagnetCostPerKgNoCentralSolenoid,
    ReactorStructureCost,
    TfMagnetCostSuperconductingPerKam,
    TfMagnetCostSuperconductingPerKg,
)
from functional_process.cottax.costs.namespace import Costs
from functional_process.cottax.divertor import (
    DivertorHeatLoadWadeDoubleNull,
    DivertorHeatLoadWadeSingleNull,
)
from functional_process.cottax.fw import (
    FirstWallDoubleNull,
    FirstWallDShapedDoubleNull,
    FirstWallSingleNull,
)
from functional_process.cottax.initialisation import (
    BeamElectronDensityFraction,
    DoubleNullUpperBuild,
    EnergyStorageBuildingVolume,
    Initialisation,
    StellaratorPulseTimes,
    StellaratorSolenoidAbsent,
    PfCoilResistivity,
    TfConductorYoungsModulus,
    TfCryoplantEfficiency,
    TfInsulationYoungsModulus,
)
from functional_process.cottax.namespace import Build, Divertor
from functional_process.cottax.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.cottax.pfcoil.namespace import (
    CSCoil,
    PFCoil,
    PFCoilCsWstNb3Sn,
    PFCoilSphericalTokamak,
)
from functional_process.cottax.pfcoil.superconductor import (
    CSCriticalCurrentDensitiesIterNb3Sn,
    CSCriticalCurrentDensitiesWstNb3Sn,
    CSTemperatureMarginIterNb3Sn,
    CSTemperatureMarginWstNb3Sn,
)
from functional_process.cottax.physics.bootstrap_current import (
    NoDiamagneticCurrent,
    NoPfirschSchluterCurrent,
    SauterBootstrapCurrentFraction,
    SceneDiamagneticCurrent,
    ScenePfirschSchluterCurrent,
)
from functional_process.cottax.physics.composition import (
    PlasmaCompositionIgnited,
    PlasmaCompositionNonIgnited,
)
from functional_process.cottax.physics.confinement_time import (
    ConfinementTailCoreRadiation,
    Iss04ConfinementTime,
    IterIpb98y2ConfinementTime,
    PlasmaPowerLossIgnitedCoreRadiation,
    PlasmaPowerLossNonIgnitedCoreRadiation,
)
from functional_process.cottax.physics.current_drive import (
    HcdElectricTotalIgnited,
    HcdElectricTotalNonIgnited,
    HcdPrimaryEfficiencyFreethyEcrhOMode,
    HcdPrimaryEfficiencyUserInputEcrh,
    HcdPrimaryPowersElectronCyclotronNoSecondary,
    HcdSecondaryHeatingNone,
)
from functional_process.cottax.physics.density_limit import (
    EnforcedDensityLimitGreenwald,
    TokamakDensityLimit,
)
from functional_process.cottax.physics.l_h_transition import (
    Martin08AspectLowerLHThresholdPower,
    Martin08AspectNominalLHThresholdPower,
    Martin08AspectUpperLHThresholdPower,
    Martin08LowerLHThresholdPower,
    Martin08NominalLHThresholdPower,
    Martin08UpperLHThresholdPower,
)
from functional_process.cottax.physics.namespace import (
    Physics,
    PhysicsConfinementTime,
    PhysicsProfiles,
    ProfileParameterisationParabolic,
    ProfileParameterisationPedestal,
)
from functional_process.cottax.physics.physics import (
    BetaNormMaxWesson,
    PulseRampTimesContinuousDefault,
    PulseRampTimesPulsedDefault,
    SeparatrixPowerNonIgnited,
    SurfaceAveragedPoloidalFieldAmperes,
)
from functional_process.cottax.physics.plasma_current import (
    FiestaStPlasmaCurrent,
    Ipdg89PlasmaCurrent,
    TokamakPlasmaCurrent,
    WessonCurrentProfileIndex,
)
from functional_process.cottax.physics.plasma_fields import PlasmaFields
from functional_process.cottax.physics.plasma_geometry import (
    CreateDataEuDemoXPointPlasmaShape,
    DoubleArcPlasmaGeometry,
    Ipdg89XPointPlasmaShape,
)
from functional_process.cottax.physics.plasma_inductance import (
    PlasmaInternalInductanceNormWesson,
    TokamakPlasmaInductance,
)
from functional_process.cottax.physics.profiles import (
    GreenwaldDensityFractions,
    PedestalSeparatrixDensities,
)
from functional_process.cottax.physics.pure_formulas import (
    FastAlphaBetaIterPhysicsRules,
    FastAlphaBetaWard,
)
from functional_process.cottax.physics.scrape_off_layer import (
    OutboardSOLPowerDecayLengthEich2013,
    TokamakScrapeOffLayer,
)
from functional_process.cottax.physics.tokamak_namespace import (
    TokamakCurrentDrive,
    TokamakPhysics,
    TokamakPlasmaBeta,
    TokamakPlasmaGeom,
    TokamakPulse,
)
from functional_process.cottax.power.electric_production import (
    AcpowLine,
    AcpowMotorGeneratorFlywheel,
    PlantElectricProductionLiquidBreeder,
    PlantElectricProductionResistiveCentrepostLiquidBreeder,
    PlantElectricProductionResistiveCentrepostSingleCoolant,
    PlantElectricProductionSingleCoolant,
    PowerProfilesOverTime,
)
from functional_process.cottax.power.namespace import Power
from functional_process.cottax.physics.tokamak_namespace import PulseBurnTime
from functional_process.cottax.power.pf_coil_power import PfCoilPowerSupplies
from functional_process.cottax.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.cottax.power.thermal_cryo import (
    ComponentThermalPowers,
    CryoLoadsActive,
    CryoLoadsInactive,
    CryoQLoadsResistiveTf,
    CryoQLoadsSuperconductingTf,
    CryoQNuc,
    DeltaEtaStep,
    EtathLiqSupercriticalCo2,
    EtaTurbineCcfeHcpbValue,
    EtaTurbineCcfeHcpbValueWithDivertor,
    EtaTurbineSteamRankineCycle,
    EtaTurbineSupercriticalCo2,
    PFwBlktCoolantPumpMw,
    PFwDivHeatDepositedMwSummed,
    TempTurbineCoolantInFromBlanketCoolant,
    TempTurbineCoolantInFromLiquidBreeder,
)
from functional_process.cottax.shield import (
    DoubleNullShieldHalfHeight,
    DShapedShieldVolumes,
    EllipticalShieldVolumes,
    SingleNullShieldHalfHeight,
    TokamakShield,
)
from functional_process.cottax.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
)
from functional_process.cottax.stellarator.coils.calculate import (
    Bi2212WindingPackIntersectInputs,
    CrocoRebcoWindingPackIntersectInputs,
    DurhamNbtiWindingPackIntersectInputs,
    DurhamRebcoWindingPackIntersectInputs,
    IterNb3snWindingPackIntersectInputs,
    OldLubellNbtiWindingPackIntersectInputs,
    UserDefinedNb3snWindingPackIntersectInputs,
    WstNb3snWindingPackIntersectInputs,
)
from functional_process.cottax.stellarator.coils.mass import (
    Bi2212CoilsMass,
    CrocoRebcoCoilsMass,
    DurhamNbtiCoilsMass,
    DurhamRebcoCoilsMass,
    IterNb3snCoilsMass,
    OldLubellNbtiCoilsMass,
    UserDefinedNb3snCoilsMass,
    WstNb3snCoilsMass,
)
from functional_process.cottax.stellarator.density_limits import EcrhDensityLimit
from functional_process.cottax.stellarator.heating import (
    EcrhHeating,
    LowhybHeating,
)
from functional_process.cottax.stellarator.namespace import (
    BlanketShieldPowerExponential,
    Stellarator,
    StellaratorCoils,
    StellaratorFwbs,
)
from functional_process.cottax.stellarator.plasma_physics import (
    HeatingAndRadiationPowerIgnited,
    HeatingAndRadiationPowerNonIgnited,
    NeutronWallLoadFirstWallAreaComprehensive2014,
    NeutronWallLoadFirstWallAreaPre2014,
    NeutronWallLoadScaledPlasmaSurface,
    RadiatedWallLoadFirstWallAreaComprehensive2014,
    RadiatedWallLoadFirstWallAreaPre2014,
    RadiatedWallLoadScaledPlasmaSurface,
)
from functional_process.cottax.stellarator.preset_config import (
    StellaratorMachineConfig,
    machine_config_for_istell,
)
from functional_process.cottax.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPower,
    DetailedPowerflowBlanketShieldPowerUserInputPumping,
)
from functional_process.cottax.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
)
from functional_process.cottax.structure import Structure
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    BlanketLifetimeModel,
    CoilNuclearHeatingModel,
    FastAlphaPressureModel,
    IFEModel,
    NeutronWallLoadModel,
    PFEnergyStorageSource,
    PlantOperationModel,
    PowerFlowModel,
    SphericalTokamakModel,
    SuperconductorCostModel,
    ThermalStorageModel,
)
from functional_process.cottax.tfcoil.base import (
    DrTfPlasmaCaseFromFraction,
    DrTfPlasmaCaseFromInput,
    DxTfSideCaseMinFromFraction,
    TfCoilSelfInductanceDShape,
    TfCoilSelfInductancePictureFrame,
    TfCoilShapeDShapeDoubleNull,
    TfCoilShapeDShapeSingleNull,
    TfCoilShapePictureFrameTart,
    TfGlobalGeometryCircularCase,
    TfGlobalGeometryStraightCase,
)
from functional_process.cottax.physics.plasma_profiles import lmode_profile_reset
from functional_process.cottax.tfcoil.croco import (
    CrocoAveragedTurnGeometryFromCurrentPerTurn,
    croco_turn_cable_space_extra_void,
    HazeltonZhaiRebcoCrocoSuperconductorProperties,
    HazeltonZhaiRebcoCrocoTemperatureMargin,
)
from functional_process.cottax.tfcoil.namespace import (
    CiccSuperconductingTfCoil,
    CrocoSuperconductingTfCoil,
)
from functional_process.cottax.tfcoil.quench import (
    TfCoilQuenchHeatCurrentDensity,
    helium_properties_at_quench_nodes,
)
from functional_process.cottax.tfcoil.stress import (
    TfFieldAndForceClampedJoints,
    TfStressExtendedPlaneStrainBuckedCaseAveragedTurn,
    TfStressPlaneStressBuckedCaseAveragedTurn,
    TfStressPlaneStressBuckedCaseIntegerTurn,
)
from functional_process.cottax.tfcoil.superconducting import (
    Bi2212SuperconductingTfCoilAreasAndMassesConventional,
    Bi2212SuperconductingTfCoilAreasAndMassesSphericalTokamak,
    CiccAveragedTurnGeometryFromCurrentPerTurn,
    CiccIntegerTurnGeometry,
    CrocoRebcoSuperconductingTfCoilAreasAndMassesConventional,
    CrocoRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    DurhamNbtiCiccSuperconductorProperties,
    DurhamNbtiSuperconductingTfCoilAreasAndMassesConventional,
    DurhamNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    DurhamRebcoSuperconductingTfCoilAreasAndMassesConventional,
    DurhamRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    DxTfSideCaseDoubleRectangular,
    DxTfSideCaseRectangular,
    DxTfSideCaseTrapezoidal,
    HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesConventional,
    HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    IterNb3snCiccSuperconductorProperties,
    IterNb3snSuperconductingTfCoilAreasAndMassesConventional,
    IterNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    IterNb3snTfSuperconductorTemperatureMargin,
    OldLubellNbtiCiccSuperconductorProperties,
    OldLubellNbtiSuperconductingTfCoilAreasAndMassesConventional,
    OldLubellNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    OldLubellNbtiTfSuperconductorTemperatureMargin,
    PeakBTfInboardWithRipple16Coils,
    PeakBTfInboardWithRipple18Coils,
    PeakBTfInboardWithRipple20Coils,
    PeakBTfInboardWithRippleFlatAllowance,
    SuperconductingTfWpGeometryDoubleRectangular,
    SuperconductingTfWpGeometryRectangular,
    SuperconductingTfWpGeometryTrapezoidal,
    TfCaseAreasCircularFront,
    TfCaseAreasStraightFront,
    UserDefinedNb3snCiccSuperconductorProperties,
    UserDefinedNb3snSuperconductingTfCoilAreasAndMassesConventional,
    UserDefinedNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    UserDefinedNb3snTfSuperconductorTemperatureMargin,
    WstNb3snCiccSuperconductorProperties,
    WstNb3snSuperconductingTfCoilAreasAndMassesConventional,
    WstNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
    WstNb3snTfSuperconductorTemperatureMargin,
)
from functional_process.cottax.tokamak.namespace import Tokamak
from functional_process.cottax.vacuum.vacuum import (
    VacuumVesselDShapedDoubleNull,
    VacuumVesselEllipticalDoubleNull,
    VacuumVesselEllipticalSingleNull,
)
from functional_process.cottax.importer import Imported, read_indat
from functional_process.cottax.total_process import StellaratorProcess, TokamakProcess
from functional_process.vocabulary import ITERATION_VARIABLES
from functional_process.vocabulary import FiguresOfMerit
from functional_process.vocabulary.input_variables import INPUT_VARIABLES
from functional_process.vocabulary import BlktModelTypes
from functional_process.vocabulary import TFCSRadialConfiguration
from functional_process.vocabulary import DivertorHeatLoadModel
from functional_process.vocabulary import PFConductorModel
from functional_process.vocabulary import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    CurrentProfileIndexModel,
    DivertorNumberModels,
    OutbordSOLPowerDecayLengthModel,
    PlasmaIgnitionModel,
)
from functional_process.vocabulary import TFWPIntegerTurnType
from functional_process.vocabulary import FwBlktVVShape
from functional_process.vocabulary import BootstrapCurrentFractionModel
from functional_process.vocabulary import (
    CurrentDriveMethodType,
    CurrentDriveModel,
)
from functional_process.vocabulary import DensityLimitModel
from functional_process.vocabulary import PlasmaConfinementTransitionModel
from functional_process.vocabulary import IndInternalNormModel
from functional_process.vocabulary import (
    PlasmaCurrentModel,
    PlasmaDiamagneticCurrentModel,
)
from functional_process.vocabulary import (
    PlasmaGeometryModelType,
    PlasmaShapeModelType,
)
from functional_process.vocabulary import (
    ElectricConversionModelTypes,
    PumpingPowerModelTypes,
)
from functional_process.vocabulary import (
    SuperconductorMaterial,
    SuperconductorModel,
    SuperconductorShape,
)
from functional_process.vocabulary import (
    TFCoilShapeModel,
    TFConductorModel,
    TFPlasmaCaseType,
)
from functional_process.vocabulary import (
    SuperconductingTFTurnType,
    SuperconductingTFWPShapeType,
)

REFERENCE_STELLA_CONF = (
    Path(__file__).resolve().parent.parent.parent
    / "tests/regression/input_files/stellarator_helias.stella_conf.json"
)
"""`REFERENCE_INPUT_FILE`'s `istell == 6` machine-config companion."""


REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The run this whole port is validated against -- `mda_harness.py`, `mda_constraint_
harness.py` and every number in `_audit/next_steps.md` § 8 use it.
"""

_I_PLASMA_GEOMETRY_REASON = (
    "eleven of `i_plasma_geometry`'s thirteen values are unwritten (0 and 10 are "
    "written). Each reads a genuinely different set of fields (the dispatch table is "
    "`plasma_geometry.md` \u00a7 'the `i_plasma_geometry` dispatch'), and none is live "
    "on any tracked regression input. Under this wave's binding policy each needs its "
    "own occupant class rather than a family grouped by reads-identical sets, which is "
    "what supersedes that record's open question 1"
)
"""Shared by the eleven refused `i_plasma_geometry` values -- one reason, eleven values.
"""

_I_HCD_SECONDARY_REASON = (
    "every non-zero `i_hcd_secondary` needs its own efficiency model *and* its own "
    "wall-plug block (`current_drive.py:1885-2063`), *and* changes which technology "
    "accumulator the primary block's `+=` starts from -- see `current_drive.md` "
    "\u00a7 'the accumulators'. Three consequences per value, none of them written"
)
"""Shared by every refused `i_hcd_secondary` value -- one reason, twelve values."""

REFERENCE_MACHINE_SWITCHES = {
    "istell": 6,  # `stellarator_helias.IN.DAT:137`
    "isthtr": 1,  # `:139` -- equals PROCESS's own default, listed anyway
    "i_plasma_pedestal": 0,  # `:118`
    "i_cost_model": 0,  # `:248`
    "ireactor": 1,  # `:245` -- equals PROCESS's own default, listed anyway
    # The three confinement switches. They were `eqx.field(static=True)` kwargs
    # transcribed into the tree until the confinement node was split into slots; the
    # factory reads them now, so they belong here like any other switch the file sets.
    "i_confinement_time": 38,  # `:121` -- ISS04
    "i_rad_loss": 1,  # `:122` -- CORE_ONLY
    "i_plasma_ignited": 1,  # `:126` -- IGNITED
    # Read by the factory since `winding_pack_intersect_inputs` became a slot
    # (`_audit/next_steps.md` §14.5); a static kwarg on that node until then. Equals
    # PROCESS's own default (`tfcoil_variables.py:246`), listed anyway.
    "i_tf_sc_mat": 1,  # `:235` -- ITER Nb3Sn
}
"""The switch values `REFERENCE_INPUT_FILE` actually sets, as a faithful transcription.
"""

_I_STR_WP_ZERO_REASON = (
    "`i_str_wp == 0` feeds the critical-surface fits `.tfcoil.str_tf_con_res` where "
    "`== 1` feeds `.tfcoil.str_wp` (process/models/tfcoil/superconducting.py:2897-2900 "
    "and :2744-2747). That is a **read**, and a `From` default is fixed when the class "
    "body executes, so the arm is a class axis and not a kwarg -- five more `__call__` "
    "bodies differing in one parameter name. `1` is PROCESS's own default "
    "(tfcoil_variables.py:508) and no tracked input file sets the switch at all, so the "
    "arm is unreachable; it is refused here rather than baked so that a file which does "
    "set it stops loudly instead of silently reading the other strain"
)

_BI2212_UNBOUND_REASON = (
    "PROCESS's own Bi-2212 branch cannot return. "
    "`tf_cable_in_conduit_superconductor_properties` assigns `bc20m`/`tc0m` on every "
    "arm except this one (process/models/tfcoil/superconducting.py:2941-2984) and then "
    "returns `TFSuperconductorLimits(..., bc20m=bc20m, tc0m=tc0m)` at :3160, so "
    "`i_tf_sc_mat == 2` raises `UnboundLocalError` before any value exists to agree "
    "with. There is nothing to port -- this is a PROCESS defect, recorded in "
    "`_audit/units/models/tfcoil/superconducting.md` as D5"
)

_BI2212_MARGIN_REASON = (
    "`calculate_superconductor_temperature_margin` short-circuits Bi-2212 to "
    "`temp_tf_superconductor_margin = 0.0` and, unlike every other arm, never writes "
    "`.tfcoil.temp_margin` (process/models/tfcoil/superconducting.py:1231-1233) -- "
    "conditional ownership, so a genuinely different occupant. Not written, because the "
    "arm is unreachable anyway: its sibling `tf_cable_in_conduit_superconductor_"
    "properties` raises `UnboundLocalError` at the same switch value, so no machine can "
    "reach this node with a real `j_superconductor`"
)

_SC_TAPE_REASON = (
    "`i_tf_sc_mat` 6, 8 and 9 are `SuperconductorShape.TAPE` "
    "(process/models/superconductors.py:101-124), and "
    "`tf_cable_in_conduit_superconductor_properties` refuses a non-CABLE shape in its "
    "first four lines (process/models/tfcoil/superconducting.py:2882-2889) before any "
    "arithmetic. A tape machine takes `CROCOSuperconductingTFCoil` instead -- which is "
    "what both tracked ST files do, `i_tf_turn_type = 2` "
    "(spherical_tokamak_eval.IN.DAT:72, st_regression.IN.DAT:800) -- so this slot is "
    "never reached at those values and there is no PROCESS behaviour to port. Its "
    "mirror image is `_SC_CABLE_REASON`, which refuses the six cable values on the two "
    "CroCo slots"
)

_SC_CABLE_REASON = (
    "`i_tf_sc_mat` 1, 2, 3, 4, 5 and 7 are `SuperconductorShape.CABLE` "
    "(process/models/superconductors.py:71-124), and "
    "`tf_croco_superconductor_properties` refuses a non-TAPE shape in its first four "
    "lines (process/models/tfcoil/superconducting.py:4435-4441) before any arithmetic. "
    "A cable machine takes `CICCSuperconductingTFCoil` instead, i.e. "
    "`i_tf_turn_type = 1`, so this slot is never reached at those values and there is "
    "no PROCESS behaviour to port. The exact mirror of `_SC_TAPE_REASON`: between them "
    "the two slots' registries partition the nine materials, which is what makes the "
    "pair of slots a partition of one switch rather than two overlapping guesses"
)

_CROCO_REBCO_MARGIN_REASON = (
    "`i_tf_sc_mat == 6` (CROCO_REBCO) has a CroCo properties arm that runs -- "
    "`jcrit_rebco` at (process/models/tfcoil/superconducting.py:4450-4455) -- and then "
    "dies one call later. `calculate_superconductor_temperature_margin` dispatches on "
    "`{1, 3, 4, 5, 7, 8, 9}` (:1235) and 6 is not in it, so it falls through to "
    "`raise ProcessValueError('Unknown superconductor type...')` (:1290-1292); `run` "
    "calls the margin unconditionally (:4006-4017), so no CroCo machine can complete a "
    "pass at this value. Refused in **both** CroCo slots, so the properties one cannot "
    "assemble alone into a machine PROCESS itself cannot run -- the same reason "
    "`_BI2212_UNBOUND_REASON` is keyed on both cable-in-conduit slots"
)

_DURHAM_REBCO_CROCO_REASON = (
    "`i_tf_sc_mat == 8` (DURHAM_REBCO) is a real and complete CroCo arm -- `gl_rebco` "
    "for the properties (process/models/tfcoil/superconducting.py:4459-4480) and "
    "branch 8 of `superconductor_current_density_margin` "
    "(process/models/superconductors.py:1268-1270) for the margin -- and the only arm "
    "of either that actually *uses* a strain, which is why `i_str_wp` stays a key of "
    "both registries. `gl_rebco` is already ported "
    "(functional_process/models/physics/superconductors.py), so this is a small "
    "addition rather than a missing model. **No tracked input file selects it**: both "
    "CroCo machines set `i_tf_sc_mat = 9` (spherical_tokamak_eval.IN.DAT:355, "
    "st_regression.IN.DAT:827). Refused until something reaches it, rather than written "
    "against no measurement"
)

_CROCO_INTEGER_TURN_REASON = (
    "**PROCESS refuses it, not this port.** `CROCOSuperconductingTFCoil.run` raises "
    "`ProcessValueError('Integer turn geometry not implemented for CroCo conductor.')` "
    "at `i_tf_turns_integer == 1` (process/models/tfcoil/superconducting.py:3834-3840). "
    "There is no arm to port. Both tracked CroCo files set `i_tf_turns_integer = 0` "
    "(spherical_tokamak_eval.IN.DAT:354, st_regression.IN.DAT:1042)"
)

_CROCO_TURN_DIMENSION_INPUT_REASON = (
    "`tf_croco_averaged_turn_geometry`'s first two branches -- the turn side given "
    "directly (`i_dx_tf_turn_general_input`) or via the cable space "
    "(`i_dx_tf_turn_cable_space_general_input`), "
    "process/models/tfcoil/superconducting.py:4321-4339 -- **own** "
    "`.tfcoil.c_tf_turn` and `.tfcoil.dx_tf_turn_general` where the written third arm "
    "reads them, so they are different nodes and not a kwarg, exactly as on the "
    "cable-in-conduit side (`_cicc_turn_geometry_arm`). Neither is written; both "
    "PROCESS defaults are `False` (tfcoil_variables.py:108,127) and both tracked CroCo "
    "files leave them there"
)

_TF_STRESS_MODEL_REASON = (
    "`i_tf_stress_model == 2` reaches `extended_plane_strain` through the same `elif` "
    "as `0` (process/models/tfcoil/base.py:3005), which **is** ported "
    "(`tf_stress_extended_plane_strain_bucked_case`). What `2` additionally does is "
    "switch off two zero-bore guards: `stresscl` raises `ProcessValueError` 245 when "
    "`r_tf_inboard_in` is zero at `0` or `1` (`base.py:2524-2527`) and patches "
    "`radtf[0]` to `1e-9` (`:2963-2965`), and `2` does neither, because the plane-"
    "strain solver's inner boundary condition is zero *displacement* rather than zero "
    "stress there. So `2` is `0` plus a reachable `rad[0] == 0`, on which the solver "
    "returns `nan` in the first element of six of its eight arrays -- PROCESS's own "
    "unit test records that as `nan_init`. Not aliased onto `0`: 'the same code path "
    "on the values a tracked file happens to take' is a measurement, and no tracked "
    "file sets `2`, so nobody has made it"
)

_EXTENDED_PLANE_STRAIN_INTEGER_TURN_REASON = (
    "`i_tf_stress_model == 0` with `i_tf_turns_integer == 1`. The solver is ported and "
    "so is the `i_tf_bucking == 1` layer stack; what is not written is the *integer-"
    "turn* read of the cable-space width the transverse smearing is built on "
    "(`base.py:2745-2749`, `.superconducting_tfcoil.dr_tf_turn_cable_space` rather than "
    "`.dx_tf_turn_cable_space_average`). A different read is a different node, exactly "
    "as on the plane-stress side, where both arms *are* written. Unreachable on both "
    "tracked spherical tokamaks -- `spherical_tokamak_eval.IN.DAT:354` and "
    "`st_regression.IN.DAT:1042` both set `i_tf_turns_integer = 0` -- so it is refused "
    "rather than written blind; writing it is a copy of the sibling arm the day a file "
    "needs it"
)

_TF_BUCKING_REASON = (
    "`i_tf_bucking != 1` rebuilds the layer stack. At `>= 2` (bucked and wedged) the "
    "central solenoid becomes the innermost stress layer and its smeared properties are "
    "reconstructed from scratch inside `stresscl` out of nine `.pf_coil` fields it "
    "otherwise never reads -- `j_cs_flat_top_end`, `j_cs_pulse_start`, "
    "`c_pf_coil_turn_peak_input`, `n_pf_coils_in_group`, `f_dr_dz_cs_turn`, "
    "`radius_cs_turn_corners`, `f_a_cs_turn_steel`, `a_cs_poloidal`, `i_pf_conductor` "
    "(process/models/tfcoil/base.py:2531-2650) -- plus two stress-distribution "
    "corrections that only run there (`:3048-3066`) and a fourth output, "
    "`.tfcoil.sig_tf_cs_bucked`. At `3` a Kapton interlayer is added on top of that. At "
    "`0` there is no bucking cylinder at all, which PROCESS only defaults to for water-"
    "cooled copper (`init.py:891-895`) -- not a conductor this namespace holds. A "
    "different reads-set and a different owns-set in every case; none is written"
)

_DURHAM_NBTI_COMPLEX_REASON = (
    "PROCESS's own residual leaves the real numbers on this arm. "
    "`superconductor_current_density_margin` branch 7 calls `gl_nbti`, which raises a "
    "negative base to a fractional power while `scipy.optimize.newton`'s secant search "
    "probes above `t_c0`; Python returns a `complex`. Measured at "
    "`b_tf_inboard_peak = 8.0`: `optimize.newton` converges and PROCESS returns "
    "`0.4561454861673191+1.2475645615451133e-12j`, a complex temperature margin. "
    "Measured at `b_tf_inboard_peak = 12.5`: the same call dies with `TypeError: '<=' "
    "not supported between instances of 'complex' and 'float'`. There is no real-valued "
    "PROCESS answer to agree with. **Scoped to this slot** -- the same material's "
    "`CICC_SUPERCONDUCTOR_PROPERTIES` occupant is written and agrees exactly, because "
    "that function evaluates the fit once at `tftmp` instead of searching upward"
)

UNPORTED = {
    # **`istell` has no rows here at all any more.** The five machine presets
    # (`("istell", 1..5)`, `_ISTELL_PRESET_REASON`) were the last, and they left on
    # 2026-08-30 when `machine_config_for_istell` wired arms 1-5 to the same
    # `StellaratorMachineConfig` node arm 6 already had. Nothing about the refusal's
    # recorded reason was wrong -- the reason said the preset *data* was unwired, and it
    # was -- but the reason had also stopped being about anything unported: the copy
    # mechanism it named was written, tested on all five presets, and only unreachable.
    # That is the second entry (after `("istell", 0)`) to leave this table by the
    # configuration it stood for becoming buildable rather than by being reclassified.
    ("i_plasma_ignited_i_rad_loss", -1): (
        "the head of `calculate_confinement_time` is written for the two **core-only** "
        "radiation arms -- ignited (the Helias run's) and non-ignited (the conventional "
        "tokamak's, added by the first tokamak wave). The remaining four combinations "
        "are real PROCESS branches reading genuinely different variables (total radiated "
        "power under FULL_RADIATION; no radiation term at all under NO_RADIATION) and "
        "none is written yet. Refused rather than approximated: an unwritten arm "
        "assembled from a written one's reads is the invented-edge defect this split "
        "exists to remove"
    ),
    ("i_rad_loss", 0): (
        "the FULL_RADIATION tail reads `.physics.pden_plasma_rad_mw` where the "
        "CORE_ONLY tail reads synchrotron and inner radiation -- a different reads-set, "
        "so a different occupant, and it is not written yet"
    ),
    ("i_pulsed_plant_istore", -1): (
        "Account 225.3's `istore == 3` arm (a stainless-steel thermal storage block) "
        "reads `.heat_transport.p_plant_primary_heat_mw`, `.times.t_plant_pulse_no_burn` "
        "and `.pulse.dtstor`, which options 1 and 2 do not -- a third reads-set, and no "
        "occupant is written for it"
    ),
    ("i_rad_loss", 2): (
        "the NO_RADIATION tail leaves `hstar` as `hfact` and reads no radiation term at "
        "all; not written yet"
    ),
    ("isthtr", 3): (
        "the NBI branch of `st_heat` calls `current_drive.culnbi()`, a model that is "
        "not audited yet (registry unit #5)"
    ),
    **{
        ("i_beta_norm_max", value): (
            f"`get_beta_norm_max_value` selects `.physics.beta_norm_max_{name}` "
            f"(`physics.py:3723-3743`), computed at `physics.py:3766-3800`. Only the "
            "Wesson arm is transcribed; each of the others is its own formula with its "
            "own reads (the original and Menard scalings read `.physics.eps`, Tholerus "
            "reads `.physics.c_beta` and the two thermal pressures, Stambaugh reads "
            "`.current_drive.f_c_plasma_bootstrap`, `.physics.kappa` and `.aspect`), so "
            "each needs its own occupant rather than a shared node declaring the union"
        )
        for value, name in (
            (2, "original_scaling"),
            (3, "menard"),
            (4, "tholerus"),
            (5, "stambaugh"),
        )
    },
    ("blktmodel_ipowerflow_i_p_coolant_pumping", 4): (
        "`i_p_coolant_pumping` 2 (`MECHANICAL`) or 3 (`MECHANICAL_WITH_PRESSURE_DROP`) "
        "with `ipowerflow == 1`: `stellarator.py:924-928` raises "
        '`ProcessValueError("i_p_coolant_pumping = 0 or 1 only for stellarator")`. '
        "There is no arm to port -- PROCESS refuses the configuration, so this port "
        "refuses it too rather than assembling a graph that answers a question PROCESS "
        "will not. This is also the default `i_p_coolant_pumping` "
        "(`fwbs_variables.py:249` is `2`), so a stellarator input file that never "
        "states the switch is refused here, which is the correct answer and not an "
        "over-strict one."
    ),
    ("blktmodel_ipowerflow_i_p_coolant_pumping", 0): (
        "S2's blktmodel == 1 arm is `blanket_neutronics()`, which calls "
        "`self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with zero "
        "arguments against 2-/7-keyword-argument @staticmethods -- a live PROCESS bug "
        "that would TypeError the moment this arm actually executes (unit_registry.md "
        "row 13, next_steps.md §3). hcpb.py's own 3 ported nodes "
        "(NuclearHeatingBlanket/Shield/Magnets) exist but are not usable here until "
        "that call site has a resolution."
    ),
    ("blktmodel_blkttype", 0): (
        "the blktmodel != 0 blanket-mass arm (stellarator.py:1093-1181) computes "
        "m_blkt_steel_total/m_blkt_beryllium from six .build.bl{u,m,p}{i,o}th "
        "sub-assembly thicknesses, additionally writes .fwbs.whtblbreed and "
        ".fwbs.f_a_blkt_cooling_channels, and writes neither .fwbs.m_blkt_li2o nor "
        ".fwbs.m_blkt_vanadium at all -- a different node with a different port set, "
        "not written yet. Refused rather than assembled empty: BlanketCost reads all "
        "four masses unconditionally, so an empty arm would silently hand it boundary "
        "values for fields PROCESS does compute on that arm."
    ),
    ("blktmodel_blkttype", 1): (
        "the liquid-breeder sub-arm (blkttype in {1, 2}, WCLL/HCLL, "
        "stellarator.py:1058-1066) writes .fwbs.wtbllipb and .fwbs.m_blkt_lithium in "
        "place of .fwbs.m_blkt_li2o/.m_blkt_beryllium -- different fields, not a "
        "different formula for the same ones. Not ported: neither replacement field "
        "has a reader in this graph, and stellarator_helias.IN.DAT leaves blkttype at "
        "its default of 3. Values 1 and 2 select the identical formula, so this one "
        "entry covers both; there is no separate value=2 entry because there is no "
        "separate behaviour to name."
    ),
    ("i_tf_sup", 2): (
        "aluminium TF (i_tf_sup == 2) runs the identical calculate_tf_power_resistive "
        "branch as i_tf_sup == 0 -- `Power.tfpwr` dispatches on `i_tf_sup != 1` only, "
        "one formula for both. Request `.tfcoil.i_tf_sup == 0` instead; it fills the "
        "slot with the same occupant. Kept as a refused value rather than a second "
        "registry entry pointing at TfPowerResistive so the claim stays visible."
    ),
    ("i_tf_sc_mat", 9): (
        "HAZELTON_ZHAI_REBCO is a `SuperconductorModel` member with no branch in "
        "`jcrit_from_material` at all (process/models/stellarator/coils/coils.py:52-160 "
        "handles 1..8 and then raises `Illegal value for i_pf_superconductor`), so "
        "there is no PROCESS arm to port and no reads-set to declare. A ninth occupant "
        "would have to invent the model, not port it. **This refusal is scoped to the "
        "critical-surface slots** -- `WINDING_PACK_MATERIAL` and `COILS_MASS_MATERIAL`, "
        "which key on the bare field name. `SC_TF_MASSES` keys on "
        "`itart_i_tf_sc_mat_sc_tf_masses` and has all nine materials, because the "
        "tokamak TF mass path uses `i_tf_sc_mat` only to index `.tfcoil.dcond`, never "
        "to dispatch: `dcond[8] == 8500.0` exists, so value 9 is portable there. The "
        "two `i_str_wp_i_tf_sc_mat_*` slots key on their own composite names for the "
        "same reason and carry their own, differently-scoped refusals below"
    ),
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 1)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 3)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 4)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 5)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 7)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (1, 2)): _BI2212_UNBOUND_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 2)): _BI2212_UNBOUND_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (1, 6)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 6)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (1, 8)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 8)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (1, 9)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", (0, 9)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 1)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 3)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 4)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 5)): _I_STR_WP_ZERO_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (1, 2)): _BI2212_MARGIN_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 2)): _BI2212_MARGIN_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (1, 6)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 6)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (1, 7)): _DURHAM_NBTI_COMPLEX_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 7)): _DURHAM_NBTI_COMPLEX_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (1, 8)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 8)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (1, 9)): _SC_TAPE_REASON,
    ("i_str_wp_i_tf_sc_mat_temp_margin", (0, 9)): _SC_TAPE_REASON,
    # `("i_tf_turn_type", CROSS_CONDUCTOR)` was here from 2026-08-29 to 2026-08-30 and
    # is **gone rather than reworded**: the CroCo turn is ported
    # (`models/tfcoil/croco.py`), so `machine_from_indat` threads the switch to
    # `_tokamak_device` and it selects a namespace instead of raising. It is the second
    # entry ever to leave this table, after `("istell", 0)`, and for the same reason --
    # the arm was written. What is left of the refusal is finer-grained and lives in the
    # comprehension at the foot of this dict: the three CroCo arms `croco.py` does not
    # write, and the six cable materials its properties function refuses outright.
    ("ife", IFEModel.INERTIAL_CONFINEMENT): (
        "inertial confinement is a different device, and PROCESS spells it as an `if` "
        "inside seven Account-22x cost methods rather than as a device class. Each of "
        "the seven arms reads a genuinely different set of `.ife.*` fields -- "
        "`.ife.fwmatm`/`.blmatm`/`.shmatm` (2-D material-mass arrays) for Accounts "
        "221.1/221.2/221.3, `.ife.ifedrv` and `.cdriv0..3` for 223, `.ife.tdspmw`/"
        "`.tfacmw` for 2262, `.ife.gain`/`.edrive`/`.fburn` for 2272, and "
        "`.ife.uctarg`/`.reprat` for `coelc` -- and **the whole `.ife.*` subsystem is "
        "unported** (`unit_registry.md` has no `ife` unit). Refused here rather than "
        "seven times inside seven node bodies: the reason strings moved verbatim from "
        "those bodies' `NotImplementedError`s, which the composite functions still "
        "raise for the harness's benefit"
    ),
    ("i_pf_energy_storage_source", PFEnergyStorageSource.MGF_PF_LINE_HEATING): (
        "`i_pf_energy_storage_source == 3` (PF power from MGF, heating from line) runs "
        "the identical `acpow` arithmetic as `== 1`: `power.py:1401`/`:1417` both test "
        "`!= 2` only, and `pf_power_variables.py:18-24` says so outright -- *'options 1 "
        "and 3 are not treated differently'*. Request `.pf_power."
        "i_pf_energy_storage_source == 1` instead; it fills the slot with the same "
        "occupant. Kept as a refused value rather than a second registry entry pointing "
        "at `AcpowMotorGeneratorFlywheel` so the claim stays visible -- the same "
        "discipline `('i_tf_sup', 2)` follows"
    ),
    ("i_cost_model", 1): (
        "KOVARI_2014 (i_cost_model == 1) is PROCESS's own default cost model and is "
        "unported: costs_2015.py has no cottax nodes, so on that arm this port computes "
        "no cost of electricity at all and .costs.coe/.costs.concost would surface as "
        "unowned boundary inputs. Filling the slot with the 1990 model instead would "
        "compute *a different number for the same field* -- worse than the "
        "EcrhDensityLimit bug class, which merely computed a value the configuration "
        "never asks for. This used to be spelled as a slot holding None; it is a "
        "refusal now, because a tree with no optional slots cannot say 'absent'"
    ),
    # ---- the tokamak's refusals -------------------------------------------------
    #
    # Every entry below was written by the porting agent that measured it, and the
    # reason is that record's own words. Two of them are refusals *PROCESS shares* --
    # `i_hcd_primary` 6 and 7 cannot execute in PROCESS at all -- which is a new
    # category for this table: not "this port has not written the arm" but "there is no
    # arm to write until someone fixes the reference implementation".
    # Value 10 (`CREATE_DATA_EU_DEMO_X_POINT`) is written -- it is live on
    # `low_aspect_ratio_DEMO.IN.DAT` -- so it is lifted out of the refused range.
    **dict.fromkeys(
        (
            ("i_plasma_geometry", PlasmaGeometryModelType(v))
            for v in (*range(1, 10), 11, 12)
        ),
        _I_PLASMA_GEOMETRY_REASON,
    ),
    ("plasma_geometry_arm", 1): (
        "the Sauter arm of `i_plasma_current == 8 or i_plasma_shape == SAUTER` "
        "(`plasma_geometry.py:467-470`). `sauter_geometry`/`calculate_geometry_sauter` "
        "are ported as pure functions -- they are cheap and touch no `self.data` -- but "
        "deliberately not wired to an occupant: the arm is live on no tracked input, so "
        "it has no regression oracle at all. Porting a formula and binding it are "
        "different acts, and this is a case where only the first is justified"
    ),
    ("surface_poloidal_field_arm", 1): (
        "`i_plasma_current == 2` (PENG_DIVERTOR_SCALING) computes the surface-averaged "
        "poloidal field from `q95`, `aspect`, `b_plasma_toroidal_on_axis`, `kappa` and "
        "`triang` via `PlasmaCurrent.plascar_bpol` -- a disjoint reads-set and a call "
        "into `plasma_current.py`, which is unported (`.tokamak.plasma_current` is an "
        "empty slot). Every other `i_plasma_current` value takes the Ampere arm, which "
        "is written: PROCESS's own test at `plasma_fields.py:83` is `!= 2`"
    ),
    ("i_plasma_ignited_separatrix", PlasmaIgnitionModel.IGNITED): (
        "`physics.py:793-798`: on an ignited plasma the power crossing the separatrix "
        "omits the injected-heating term -- PROCESS passes the literal `0.0` where the "
        "non-ignited arm passes `.current_drive.p_hcd_injected_total_mw`. One read of "
        "difference, and it is a cross-area edge, so it is a separate occupant rather "
        "than a kwarg; the occupant is not written. `large_tokamak_eval.IN.DAT` leaves "
        "`i_plasma_ignited` at PROCESS's own default `0`, so the written arm is the one "
        "a conventional tokamak takes. What reaches this refusal is a **stellarator** "
        "input file re-read as a tokamak (`istell = 0` over `stellarator_helias.IN.DAT`, "
        "which sets `i_plasma_ignited = 1`), and that is worth keeping as a refusal "
        "rather than filling in: it is the sharpest demonstration in the port that two "
        "devices' physics arms are genuinely different and not merely differently filed"
    ),
    ("pulse_ramp_times_arm", 1): (
        "`i_pulsed_plant != 1` with `i_t_current_ramp_up != 0` writes nothing at all -- "
        "the three ramp times are inputs. That is absence rather than a refusal in "
        "principle, and it is filed here rather than as a `None` occupant because the "
        "slot's other three arms all produce something: a slot that is sometimes absent "
        "and sometimes not needs its absence declared per arm, and no occupant in this "
        "port does that yet. Flagged rather than improvised"
    ),
    ("pulse_ramp_times_arm", 3): (
        "`i_pulsed_plant == 1` with `pulsetimings != 0` (`physics.py:485-498`) reads "
        "`.times.t_plant_pulse_coil_precharge` and writes it back -- `max(precharge, "
        "ramp-up)`, a ratchet -- so its occupant would read what it owns, which cottax "
        "refuses. It needs either a `FixedPointFunction` or a producer split, and the "
        "honest observation is that the 'real producer' here is the input file with "
        "PROCESS's outer loop turning it into a ratchet. Left unwritten pending that "
        "decision rather than approximated (`physics.md` open question 2)"
    ),
    ("i_hcd_primary", 0): (
        "`NO_CURRENT_DRIVE` raises `ProcessValueError` at `current_drive.py:1800` -- a "
        "primary heating system is mandatory. No arm to port"
    ),
    ("i_hcd_primary", 1): (
        "needs `LowerHybrid.lower_hybrid_fenstermacher` and `.feffcd`; not written"
    ),
    ("i_hcd_primary", 2): ("needs `IonCyclotron.ion_cyclotron_ipdg89`; not written"),
    ("i_hcd_primary", 3): (
        "needs `ElectronCyclotron.electron_cyclotron_fenstermacher` and "
        "`.physics.dlamee`; not written"
    ),
    ("i_hcd_primary", 4): ("needs `LowerHybrid.lower_hybrid_ehst`; not written"),
    ("i_hcd_primary", 5): (
        "needs `NeutralBeam.iternb` and the whole beam wall-plug block "
        "(`current_drive.py:2191-2260`); not written. This is PROCESS's own default "
        "(`current_drive_variables.py:190`), so a file that never sets `i_hcd_primary` "
        "is refused here rather than assembled -- deliberately: the wall-plug block it "
        "needs is a different reads-set, not a different constant"
    ),
    ("i_hcd_primary", 6): (
        "**PROCESS cannot execute this arm.** `cullhy` -> `lhrad` -> `lheval` reaches "
        "`calculate_profile_y`, which returns `None`, and raises `TypeError` at "
        "`current_drive.py:1498`. A live defect in the reference implementation, found "
        "by this port and recorded in `current_drive.md` § 'A live PROCESS bug in "
        "two sibling arms'. There is no behaviour to port until it is fixed, and "
        "guessing what it should have been would be inventing physics"
    ),
    ("i_hcd_primary", 7): (
        "**PROCESS cannot execute this arm.** `culecd` reaches the same "
        "`calculate_profile_y` and raises `TypeError` at `current_drive.py:815`. The "
        "sibling of value 6 and the same defect; both are recorded in "
        "`current_drive.md`, and closing `_audit/next_steps.md` §2's "
        "`calculate_profile_y` flag is what finding them did"
    ),
    ("i_hcd_primary", 8): (
        "needs `NeutralBeam.culnbi` and its `sigbeam`/`cfnbi`/`xlmbdabi` chain, plus the "
        "beam wall-plug block; not written"
    ),
    ("i_hcd_primary", 12): (
        "needs `ElectronBernstein.electron_bernstein_freethy` and the EBW block "
        "(`current_drive.py:2162-2187`); not written"
    ),
    # `("i_hcd_primary", 13)` was a refusal until 2026-08-27; the value dispatches to
    # the nested `i_ecrh_wave_mode` registry now (`_hcd_primary_efficiency`), so its
    # refusals are keyed on the inner switch:
    ("i_ecrh_wave_mode", 1): (
        "the X-mode arm of `ElectronCyclotron.electron_cyclotron_freethy` "
        "(`current_drive.py:1076-1077`, the right-hand cut-off). Live on no tracked "
        "input: both files that select `i_hcd_primary = 13` set `i_ecrh_wave_mode = 0` "
        "(`spherical_tokamak_eval.IN.DAT:130`, `st_regression.IN.DAT:2665`), which is "
        "also PROCESS's default (`current_drive_variables.py:116`). The branch itself "
        "is one line inside the shared pure function "
        "(`freethy_electron_cyclotron_efficiency`), transcribed and value-checked "
        "against the reference -- the two wave modes read identical variable sets, "
        "`traceability_policy.md`'s static-kwarg exception -- but binding it is a "
        "separate act from porting it (the `plasma_geometry_arm` 1 precedent above): "
        "no configuration asks for it, so no occupant pins it"
    ),
    **dict.fromkeys(
        (
            ("i_hcd_secondary", v)
            for v in CurrentDriveModel
            if v is not CurrentDriveModel.NO_CURRENT_DRIVE
        ),
        _I_HCD_SECONDARY_REASON,
    ),
    ("i_hcd_calculations", 0): (
        "the whole heating-and-current-drive body is skipped, so "
        "`.heat_transport.p_hcd_primary_electric_mw` keeps its `None` default and any "
        "consumer of it fails. This is topology rather than an occupant -- `1` means "
        "these nodes exist and `0` means none of them does -- and the honest spelling "
        "for `0` would be an empty `.tokamak.current_drive` slot, which is not written "
        "because the `None`-defaulted field makes 'nothing is computed' and 'something "
        "downstream will crash' the same configuration in PROCESS"
    ),
    ("hcd_primary_powers_arm", -1): (
        "the primary/secondary technology pair is the one genuinely combinatorial "
        "dispatch in this port: the primary block's `+=` (`current_drive.py:2147`) "
        "starts from whatever the secondary block left in the same technology's field "
        "(`:1955`), so five primary methods times six secondary methods are in "
        "principle distinct arms and one cell is written -- electron cyclotron with no "
        "secondary. `current_drive.md` names the fix and declines to make it: a "
        "per-technology 'secondary contribution' field would turn the product back into "
        "two slots, but it needs a name PROCESS does not have and "
        "`naming_convention.md` forbids minting one quietly"
    ),
    ("divertor_geometry_arm", -2): (
        "`.physics.itart == 0` with the input `.build.dz_xpoint_divertor` not "
        "effectively zero: `process/models/build.py:800-801` keeps the user's value and "
        "`divgeom` runs for `.build.rspo` alone. That `rspo`-only occupant owns one "
        "field where the conventional arm's owns two -- conditional ownership by run "
        "configuration -- and is not written. (`itart == 1` with the input set is arm "
        "`-3`, `None`, not this: the early return at `:863` writes nothing, so there "
        "is no `rspo`-only remainder to refuse)"
    ),
    ("i_tf_sup_build", 0): (
        "copper TF (`i_tf_sup == 0`) changes both build nodes it touches: the outboard "
        "leg scales by `.build.f_dr_tf_outboard_inboard`, and the ripple fit's conductor "
        "width comes from `.superconducting_tfcoil.r_tf_wp_inboard_outer` and "
        "`.tfcoil.n_tf_coils` instead of the three `dx_tf_wp_*` fields. Two disjoint "
        "reads-sets, neither written. Keyed on a joint name rather than on `i_tf_sup` "
        "itself because `.power.tf_power` already answers that switch for a different "
        "slot, with a different disposition"
    ),
    ("i_tf_sup_build", 2): (
        "aluminium TF takes the same non-superconducting build arms as `0`; see that "
        "entry. Kept as a refused value rather than a second registry entry pointing at "
        "the copper occupant, the same discipline `('i_tf_sup', 2)` follows for "
        "`.power.tf_power`"
    ),
    ("tf_inboard_radii_arm", -1): (
        "TF coil inside the CS (`i_tf_inside_cs == 1`): `r_tf_inboard_in = dr_bore` "
        "alone (`process/models/build.py:1692`) and `dr_cs_bore` gains a "
        "`dr_tf_inboard` term (`:1694-1698`) -- a different reads-set for the inner "
        "radius, so a different occupant. Not written"
    ),
    ("i_tf_inside_cs_vacuum_shield", TFCSRadialConfiguration.TF_INSIDE_CS): (
        "TF coil inside the CS (`i_tf_inside_cs == 1`): the inboard vacuum-vessel "
        "radius accumulates `dr_cs`, `dr_cs_tf_gap` and `dr_cs_precomp` on top of the "
        "TF leg (`process/models/build.py:1836-1845`) -- three reads the written arm "
        "never takes, so a different occupant and not a kwarg. The same value refuses "
        "`tf_inboard_radii_arm` (-1) for the same reason, one block earlier -- which "
        "is why the key is a per-slot name (`i_tf_sup_build`'s convention): one integer "
        "decides two slots, and a file setting it is refused at the earlier one, with "
        "that slot's message. Not written"
    ),
    ("tf_coil_shape_arm", -1): (
        "`i_tf_shape == D_SHAPE` with `.physics.itart == 1`: the centrepost D-shape "
        "(`tfcoil/base.py:528-549`) reads `.build.r_cp_top` and "
        "`.build.dr_tf_outboard`, sums only arcs 1-2 of the four, and starts "
        "`len_tf_coil` from "
        "`2 * (r_tf_arc[1] - r_tf_arc[0])` rather than a z-span. Not written -- and not "
        "the arm a spherical tokamak with `i_tf_shape = 2` takes; that is arm `2`"
    ),
    ("tf_coil_shape_arm", -2): (
        "`i_tf_shape == PICTURE_FRAME` with `.physics.itart == 0`: the picture frame "
        "closed on a full inboard leg, reading `.build.r_tf_inboard_out` and "
        "`.build.r_tf_inboard_mid` where its `itart == 1` sibling (arm `2`, written) "
        "reads `.build.r_cp_top` alone (`tfcoil/base.py:553-573`). Not written"
    ),
    ("cicc_turn_geometry_arm", -1): (
        "`i_dx_tf_turn_general_input == True` **owns** `.tfcoil.c_tf_turn` where the "
        "written arm reads it: the turn width is given and the current per turn follows, "
        "rather than the other way round. A `VarPath` moving from a node's inputs to its "
        "outputs is something no static kwarg can express, which is why this is an "
        "occupant. Not written"
    ),
    ("cicc_turn_geometry_arm", -2): (
        "`i_dx_tf_turn_cable_space_general_input == True`: same shape as the previous "
        "entry with the cable space given instead of the turn width, and the same "
        "ownership inversion. Not written"
    ),
    ("centrepost_neutronics_arm", -1): (
        "a **water-cooled copper** centrepost (`itart == 1`, `i_tf_sup == 0`). Its "
        "nuclear heating is the written arm's -- `hcpb.py:1192` branches on aluminium "
        "alone, so copper and superconducting share one MCNP fit -- but its fast "
        "neutron flux is not: `:1114` fires only for `SUPERCONDUCTING`, so a copper "
        "centrepost's `.fwbs.neut_flux_cp` is the literal `0` of `:1112`. That is a "
        "different occupant, not the written one with a zeroed input, and writing it "
        "means writing a node whose fourth output is a constant. No input file in this "
        "repository asks for it. Not written"
    ),
    ("centrepost_neutronics_arm", -2): (
        "a **helium-cooled aluminium** centrepost (`itart == 1`, `i_tf_sup == 2`). Both "
        "halves differ from the written arm: the flux is zero as for copper, and the "
        "nuclear heating takes `hcpb.py:1192-1197`'s two-line aluminium fit, whose own "
        "source comment says of its shield term `WARINING, this is an extraoilation "
        "from TF heat ... DO NOT TRUST THIS VALUE !!`. Porting a number PROCESS itself "
        "disowns is work to schedule deliberately, not to pick up in passing. Not "
        "written"
    ),
    ("i_p_coolant_pumping", 0): (
        "`USER_INPUT` has no arm at all in `powerflow_calc` (`hcpb.py:816-...` is an "
        "`if`/`elif` chain over the other three values) -- the pumping powers are "
        "inputs. Absence rather than a refusal in principle. **This is the tokamak "
        "slot only.** The stellarator's own `USER_INPUT` arm was the identical shape "
        "and is ported (2026-08-31): `st_fwbs`'s "
        "`DetailedPowerflowBlanketShieldPowerUserInputPumping`, arm 3 of "
        "`blktmodel_ipowerflow_i_p_coolant_pumping`, where the four "
        "`.heat_transport.p_*_coolant_pump_mw` fields are simply not owned. The same "
        "answer here is a one-line change -- `PumpingPowerModelTypes.USER_INPUT: None` "
        "in `PUMPING_POWER`, `build=` made `None`-tolerant, and `| None` added to "
        "`models/blankets/namespace.py`'s `pumping_power` annotation -- and is left "
        "undone only because no tokamak input file in this repository sets it (all "
        "five set `3`), so nothing would measure it"
    ),
    ("i_p_coolant_pumping", 1): (
        "`FRACTION_OF_HEAT` (`hcpb.py:817-838`) owns a **different set**: "
        "`.heat_transport.p_fw_coolant_pump_mw`, `p_blkt_coolant_pump_mw`, "
        "`p_shld_coolant_pump_mw` and `p_div_coolant_pump_mw`, where the written arm "
        "owns the last two plus `.primary_pumping.p_fw_blkt_coolant_pump_mw`. A partial "
        "overlap by construction, which is `next_steps.md` §12.2's 'alternatives are "
        "keyed on output -- nearly'. It also needs "
        "`engineering/ivc_functions.py::pumping_powers_as_fractions`, which is not "
        "ported. Not written"
    ),
    ("i_p_coolant_pumping", 2): (
        "`MECHANICAL` (`hcpb.py:840-862`) reaches `primary_coolant_properties`/"
        "`thermo_hydraulic_model` and hence **CoolProp**, and so does `fw.py`'s "
        "`FirstWall.fw_temp` on the same value. That is `_audit/next_steps.md` §5's "
        "unresolved wrapping policy, not an unwritten formula: the arm is dormant rather "
        "than absent, and a second tokamak input file wakes it"
    ),
    ("i_blanket_type", 5): (
        "DCLL routes to `process/models/blankets/dcll.py` at `caller.py:347-349` -- a "
        "different occupant of `.tokamak.ccfe_hcpb` entirely, with its own liquid-metal "
        "breeder model. Nothing of it is ported"
    ),
    ("r_cp_top_arm", -1): (
        "the **resistive** spherical tokamak's centrepost top radius "
        "(`build.py:1750-1810`, `itart == 1 and i_tf_sup != 1`). Three sub-arms behind "
        "`.build.i_r_cp_top`, and all three own `.build.f_r_cp` as well as "
        "`.build.r_cp_top`, so the write set differs from the ported arm's and they "
        "cannot be one occupant. All three also apply the same clamp -- "
        "`r_cp_top = max(r_cp_top, 1.01 * r_tf_inboard_out)`, spelled in PROCESS as an "
        "`if` plus a `logger.error` -- which is a genuine `jnp.where` and not a domain "
        "error. Unwritten because no input file in this repository selects it: both "
        "spherical tokamaks are `i_tf_sup = 1`, which fails the outer guard before "
        "`i_r_cp_top` is read at all, and everything else is `itart = 0`. Note that "
        "both ST files *do* set `i_r_cp_top = 2`, and on both it is inert"
    ),
    ("first_wall_arm", -2): (
        "the D-shaped first wall **at a single divertor** -- one cell of "
        "`_first_wall_arm`'s 2x2 shape x divertor-count grid, the only one unwritten. "
        "Every ingredient exists: `calculate_dshaped_first_wall_areas`, "
        "`calculate_first_wall_half_height` and `apply_first_wall_coverage_factors` are "
        "all ported and harness-tested; what is missing is the composite that chains "
        "them and its occupant class, and neither would introduce any arithmetic. It is "
        "unwritten because no input file in this repository selects it -- this wave's "
        "reachability-first discipline -- not because the formula is unknown. The two "
        "spherical-tokamak files that motivated the D-shaped arm are double-null "
        "(`i_single_null = 0`), so they take arm `2`. Until 2026-08-27 this entry meant "
        "the D-shaped first wall at *any* divertor count"
    ),
    ("first_wall_arm", -3): (
        "`.physics.i_pflux_fw_neutron != 1` normalises the neutron wall load by "
        "`.first_wall.a_fw_total` instead of scaling `ffwal` by the plasma surface flux "
        "(`fw.py:121-135`) -- and `.first_wall.a_fw_total` is a field **this same "
        "occupant owns**, so that arm is a node reading its own output and would need "
        "the `FixedPointFunction` treatment. Not written, and flagged for whoever writes "
        "it that the shape is the obstacle rather than the formula. (`fw.md` writes this "
        "value as `0`; `physics_variables.py:1006-1010` declares the domain as `1` or "
        "`2` and PROCESS's own test is `== 1` versus everything else, so the refusal is "
        "keyed on the arm rather than on either spelling of the other value.)"
    ),
    ("vacuum_vessel_arm", -2): (
        "the D-shaped vacuum vessel **at a single divertor** -- the same unwritten cell "
        "of the same 2x2 grid as `('first_wall_arm', -2)`, and unwritten for the same "
        "reason: no input file selects it. `calculate_dshaped_vessel_volumes` and "
        "`calculate_vessel_half_height` are both ported and harness-tested; only the "
        "composite chaining them and its occupant are missing. Until 2026-08-27 this "
        "entry meant the D-shaped vessel at any divertor count, and said `dshellvol` "
        "still had to be added to "
        "`functional_process/models/engineering/ivc_functions.py`; it has been"
    ),
    ("structure_arm", -1): (
        "`(i_tf_sup != 1, i_pf_conductor superconducting)`: `.structure.coldmass` is "
        "`pfmass` alone, because `structure.py:165-166`'s `+= tfmass + aintmass + "
        "dewmass` is skipped. One fewer term, three fewer reads. Not written"
    ),
    ("structure_arm", -2): (
        "`(i_tf_sup == 1, i_pf_conductor resistive)`: `.structure.coldmass` is "
        "`tfmass + aintmass + dewmass`, because `structure.py:167-168`'s `+= pfmass` is "
        "skipped. Not written"
    ),
    ("structure_arm", -3): (
        "`(i_tf_sup != 1, i_pf_conductor resistive)`: `.structure.coldmass` is exactly "
        "`0.0`, both additive terms skipped. Not written -- and worth not folding into "
        "the live occupant with a `jnp.where`, because a node that owns a field whose "
        "value is structurally zero is a different node from one that sums two masses"
    ),
    ("divertor_heat_load_arm", -1): (
        "`i_div_heat_load == 0` (`USER_INPUT`) reads nothing and prints the existing "
        "value -- absence, and the same per-arm-absence gap as `('i_p_coolant_pumping', "
        "0)`. Not written"
    ),
    ("divertor_heat_load_arm", -2): (
        "`i_div_heat_load == 1` (`PENG_CHAMBER`, `divtart`) reads `triang`, "
        "`dz_xpoint_divertor`, `dr_fw_plasma_gap_inboard`, `i_single_null`, "
        "`dz_divertor` and `.tfcoil.drtop` -- none of which `divwade` reads. A "
        "tight-aspect-ratio model, disjoint from the written one. Not written"
    ),
    ("i_cost_model", 2): (
        "i_cost_model == 2 injects a user-supplied Model instance at runtime "
        "(process/main.py's `costs` setter, lines 766-768) -- there is no PROCESS-side "
        "subgraph to port at all, so no occupant can exist here. Refused rather than "
        "left absent: unlike KOVARI_2014, a caller asking for this has a model in mind "
        "that this graph has never seen."
    ),
    # ---- waves 2/3's refusals (consolidation round 2) ---------------------------
    #
    # Each reason is its unit's audit record's own words, distilled: `plasma_current.md`,
    # `bootstrap_current.md`, `l_h_transition.md`, `density_limit.md`,
    # `scrape_off_layer.md`, `plasma_inductance.md`, `shield.md` and the five
    # `pfcoil/*.md` records.
    ("i_plasma_current", PlasmaCurrentModel.PENG_ANALYTIC_FIT): (
        "Peng analytic fit; not live on any tracked input. "
        "`calculate_current_coefficient_peng` is a 5-line pure staticmethod when needed"
    ),
    ("i_plasma_current", PlasmaCurrentModel.PENG_DIVERTOR_SCALING): (
        "Peng divertor (TART/STAR); not live, and structurally unlike every other arm "
        "-- bypasses the cylindrical current and needs `plascar_bpol`'s two-branch "
        "`arctan`/`log`. Also the arm that changes "
        "`.physics.b_plasma_surface_poloidal_average` (`plasma_fields.py:83`; see "
        "`('surface_poloidal_field_arm', 1)`)"
    ),
    ("i_plasma_current", PlasmaCurrentModel.ITER_SCALING): (
        "simple ITER cylindrical (`fq = 1`); not live"
    ),
    ("i_plasma_current", PlasmaCurrentModel.TODD_EMPIRICAL_SCALING_I): (
        "Todd I; not live. Identical reads to Todd II, differing by one literal -- "
        "**two** occupant classes when ported, per §14.2 and `plasma_current.md` open "
        "question 4's ruling, not one with a static kwarg"
    ),
    ("i_plasma_current", PlasmaCurrentModel.TODD_EMPIRICAL_SCALING_II): (
        "Todd II; see Todd I"
    ),
    ("i_plasma_current", PlasmaCurrentModel.CONNOR_HASTIE_MODEL): (
        "Connor-Hastie; not live, and the only arm that makes the current chain a "
        "genuine SCC (it reads `.physics.alphaj`, which the chain's own "
        "`current_profile_index` occupant owns) -- needs a declared driven block, not "
        "just a transcription (`plasma_current.md` § 'the cycle that is not live here')"
    ),
    ("i_plasma_current", PlasmaCurrentModel.SAUTER_SCALING): (
        "Sauter; not live. Must be wired together with `plasma_geometry.py`'s "
        "`PlasmaGeometryArm` Sauter occupant -- `_plasma_geometry_arm` owns the "
        "disjunction, one input value, two slots"
    ),
    ("i_ind_plasma_internal_norm", IndInternalNormModel.MENARD): (
        "Menard ST scaling -- an ordinary sibling occupant, one line to add: owns "
        "`.physics.ind_plasma_internal_norm`, reads "
        "`.physics.ind_plasma_internal_norm_menard`. Not written because it is not "
        "this run's value (`plasma_inductance.md`)"
    ),
    **dict.fromkeys(
        (
            ("i_bootstrap_current", v)
            for v in BootstrapCurrentFractionModel
            if v
            not in {
                BootstrapCurrentFractionModel.USER_INPUT,
                BootstrapCurrentFractionModel.SAUTER,
            }
        ),
        "a closed-form scaling in volume-averaged scalars, sharing none of the Sauter "
        "arm's profile reads; each needs its own occupant and its own harness contract "
        "to be worth anything (`bootstrap_current.md` § 'not ported in this pass'). "
        "The family PROCESS computes and discards is deliberately not computed",
    ),
    ("i_diamagnetic_current", PlasmaDiamagneticCurrentModel.HENDER_ST_FIT): (
        "`diamagnetic_fraction_hender` (`plasma_current.py:1138-1153`); not live "
        "(PROCESS's own default is 0 and the reference file leaves it)"
    ),
    **dict.fromkeys(
        (
            ("i_l_h_threshold", v)
            for v in PlasmaConfinementTransitionModel
            if v
            not in {
                PlasmaConfinementTransitionModel.MARTIN08_NOMINAL,
                PlasmaConfinementTransitionModel.MARTIN08_UPPER,
                PlasmaConfinementTransitionModel.MARTIN08_LOWER,
                PlasmaConfinementTransitionModel.MARTIN08_ASPECT_NOMINAL,
                PlasmaConfinementTransitionModel.MARTIN08_ASPECT_UPPER,
                PlasmaConfinementTransitionModel.MARTIN08_ASPECT_LOWER,
            }
        ),
        "the formula is ported and Tier-1-tested (`l_h_transition.md`'s full-closure "
        "table) but no occupant node is wired: not live on any tracked input, so "
        "wiring one later is a small, mechanical addition (declare the reads, write "
        "the `OutputInto`), not a re-derivation",
    ),
    **dict.fromkeys(
        (
            ("i_density_limit", v)
            for v in DensityLimitModel
            if v is not DensityLimitModel.GREENWALD
        ),
        "the formula is ported and Tier-1-tested against PROCESS's own staticmethod "
        "(`density_limit.md` '## UNPORTED') but no occupant node is wired -- dead work "
        "at this switch value on the reference arm; only a node class and a "
        "registration are needed the day an input file selects it",
    ),
    ("i_len_sol_outboard_power_decay", OutbordSOLPowerDecayLengthModel.MAST_2014_1): (
        "a one-line passthrough occupant selecting the MAST-1 length the graph already "
        "computes unconditionally; not live (`scrape_off_layer.md` § switches touched)"
    ),
    ("i_len_sol_outboard_power_decay", OutbordSOLPowerDecayLengthModel.MAST_2014_2): (
        "same shape as MAST-1, selecting the MAST-2 length; not live"
    ),
    ("pf_coil_system_arm", -2): (
        "an `i_pf_location`/group topology no occupant set matches. Two are written: "
        "`n_pf_coil_groups = 4` with `i_pf_location = (2, 2, 3, 3)` and "
        "`n_pf_coils_in_group = (1, 1, 2, 2)` (a machine with a central solenoid), and "
        "`(2, 3, 3, 4)` with `(2, 2, 2, 2)` (a machine without one). The pattern fixes "
        "every array index in the package (`pfcoil/__init__.py`'s `PFCoilTopology`), "
        "so a third pattern is a third `PFCoilTopology` and a third set of node "
        "instances. Not written"
    ),
    ("pf_coil_system_arm", -3): (
        "`.physics.itart == 1` **and** `.physics.itartpf == 0`: PROCESS's Peng and "
        "Strickler ST arm. It places an `i_pf_location = 2` group at "
        "`z_tf_inside_half - zref[g]` (`pfcoil.py:1250-1253`), computes `ccls` from "
        "`aspect**1.6` and never calls `efc` (`:411-454`), and `init.py:639-643` "
        "overwrites `i_pf_location[:3]` before any of that -- genuinely different read "
        "sets in placement and currents. Not written. **`itart = 1` alone does not "
        "reach it**: both tracked spherical tokamaks set `itartpf = 1`, and those two "
        "sites are the only ones in `process/` that read `itartpf` at all"
    ),
    ("pf_coil_system_arm", -4): (
        "`.pf_coil.i_pf_current == 0`: inverts which of `ccl0`/`ccl0_ma` is input and "
        "which is output (`pfcoil.py:678-685`) -- a dual-role `VarPath` across "
        "occupants that no second class can simply bind the other way "
        "(`currents.md` open questions). Not written"
    ),
    ("pf_coil_system_arm", -5): (
        "`.pf_coil.i_pf_conductor == 1` (RESISTIVE): four separate mass/power bodies "
        "with different read sets (`pfcoil.py:917-1002`). Not written"
    ),
    ("pf_coil_system_arm", -6): (
        "a superconductor choice with no occupant. With a central solenoid the ported "
        "pairs are (`i_pf_superconductor == 3` NbTi, `i_cs_superconductor == 1` ITER "
        "Nb3Sn), arm 0, and (3 NbTi, 5 WST Nb3Sn), arm 1 "
        "(`low_aspect_ratio_DEMO.IN.DAT`); without one, `i_pf_superconductor == 9` "
        "(Hazelton/Zhai REBCO tape), arm 2, and `i_cs_superconductor` selects nothing "
        "because there is no CS conductor to weigh or to take a critical surface of. "
        "The switch's effect in the ported closure is which element of `.tfcoil.dcond` "
        "is read and which critical surface `superconpf` takes, and per the binding "
        "policy that is a different occupant per value, not a parameter (`masses.md` "
        "§ switches touched). Anything else: not written"
    ),
    ("pf_coil_system_arm", -7): (
        "an outside-TF placement with no occupant. Two are written: `i_tf_shape == "
        "D_SHAPE` with `i_r_pf_outside_tf_placement == 0`, where the coil's radius "
        "follows the TF curve as `sqrt(r^2 - z^2)` with its `isinf` kludge "
        "(`pfcoil.py:1327-1339`), and `i_tf_shape == PICTURE_FRAME` **or** "
        "`i_r_pf_outside_tf_placement == 1`, where it is stacked flat at "
        "`r_pf_outside_tf_midplane` (`:1322-1326`). The two switches enter only "
        "through that disjunction, so the pair `(D_SHAPE, 1)` is the second occupant "
        "and not a third. Refused only where the topology's own arm disagrees -- see "
        "`_pf_coil_system_deviations`"
    ),
    ("tf_field_and_force_arm", True): (
        "`itart == 1` **and** `i_cp_joints == 1`: a spherical tokamak whose centrepost "
        "is joined to the outboard legs by sliding joints, so the two carry separate "
        "vertical tensions. `tf_field_and_force` then computes the centrepost's from a "
        "different closed form (`base.py:1774-1800`), takes the outboard leg's as the "
        "remainder, and **owns** `.tfcoil.f_vforce_inboard`, which the clamped arm "
        "reads and returns unchanged -- so it is a different node, not a kwarg. It is "
        "also unreachable on any superconducting machine that does not ask for it "
        "explicitly: `init.py:752-756` resolves the `i_cp_joints == -1` default to `0` "
        "whenever `i_tf_sup == 1`, and this slot exists only on a superconducting coil. "
        "Not written"
    ),
    **{
        ("tf_stress_arm", (_model, _bucking, _integer)): (
            _TF_STRESS_MODEL_REASON
            if _model == 2
            else _TF_BUCKING_REASON
            if _bucking != 1
            else _EXTENDED_PLANE_STRAIN_INTEGER_TURN_REASON
        )
        for _model in (0, 1, 2)
        for _bucking in (0, 1, 2, 3)
        for _integer in (0, 1)
        if (_model, _bucking, _integer) not in {(1, 1, 0), (1, 1, 1), (0, 1, 0)}
    },
    # ---- the CroCo namespace's three registries, 2026-08-30 -----------------------
    #
    # Written as comprehensions for the same reason `SC_TF_MASSES` is: the refusal is a
    # *rule* over the switch's values (cable shapes are refused by PROCESS's own guard,
    # `i_str_wp == 0` by this port), and thirty-eight hand-written rows would be
    # thirty-eight chances to key one of them wrong.
    ("croco_turn_geometry_arm", 1): _CROCO_INTEGER_TURN_REASON,
    ("croco_turn_geometry_arm", -1): _CROCO_TURN_DIMENSION_INPUT_REASON,
    ("croco_turn_geometry_arm", -2): _CROCO_TURN_DIMENSION_INPUT_REASON,
    **{
        (_slot, (_str_wp, _mat)): (
            _SC_CABLE_REASON
            if _mat.sc_shape is not SuperconductorShape.TAPE
            else _CROCO_REBCO_MARGIN_REASON
            if _mat is SuperconductorModel.CROCO_REBCO
            else _DURHAM_REBCO_CROCO_REASON
            if _mat is SuperconductorModel.DURHAM_REBCO
            else _I_STR_WP_ZERO_REASON
        )
        for _slot in (
            "i_str_wp_i_tf_sc_mat_croco_sc_properties",
            "i_str_wp_i_tf_sc_mat_croco_temp_margin",
        )
        for _str_wp in (0, 1)
        for _mat in SuperconductorModel
        if (_str_wp, _mat) != (1, SuperconductorModel.HAZELTON_ZHAI_REBCO)
    },
}
"""Why a known PROCESS value has no occupant, verbatim from the `Alternative(unported=)`
declarations this replaced.
"""


def _slot_occupant(field, value, registry, *, build=None):
    """One registry lookup, with both failure modes spelled out."""
    if value in registry:
        occupant = registry[value]
        return build(occupant) if build is not None else occupant()
    if (field, value) in UNPORTED:
        raise NotImplementedError(
            f"{field} == {value} is a real PROCESS branch but is not ported: "
            f"{UNPORTED[field, value]}"
        )
    raise ValueError(
        f"{field} == {value} is not a known value; this port has occupants for "
        f"{sorted(registry)} and records why it has none for "
        f"{sorted(v for f, v in UNPORTED if f == field)}"
    )


def _refuse_unported_switch(field, value):
    """Refuse a switch value this port has no occupant for, where the switch decides no
    slot of its own.
    """
    raise NotImplementedError(
        f"{field} == {value} is a real PROCESS branch but is not ported: "
        f"{UNPORTED[field, value]}"
    )


def _wall_load_arm(i_pflux_fw_neutron: int, ipowerflow: int) -> int:
    """`(i_pflux_fw_neutron, ipowerflow)` -> the wall-load arm, for **both** wall-load
    slots.
    """
    if NeutronWallLoadModel(int(i_pflux_fw_neutron)) is (
        NeutronWallLoadModel.SCALED_PLASMA_SURFACE_AREA
    ):
        return 0
    return 1 if PowerFlowModel(int(ipowerflow)) is PowerFlowModel.PRE_2014 else 2


NEUTRON_WALL_LOAD = {
    0: NeutronWallLoadScaledPlasmaSurface,
    1: NeutronWallLoadFirstWallAreaPre2014,
    2: NeutronWallLoadFirstWallAreaComprehensive2014,
}
"""`_wall_load_arm(...)` -> the neutron wall-load occupant. Keyed by arm index."""

RADIATED_WALL_LOAD = {
    0: RadiatedWallLoadScaledPlasmaSurface,
    1: RadiatedWallLoadFirstWallAreaPre2014,
    2: RadiatedWallLoadFirstWallAreaComprehensive2014,
}
"""`_wall_load_arm(...)` -> the radiated wall-load occupant, same arm index."""

HEATING_AND_RADIATION_POWER = {
    PlasmaIgnitionModel.IGNITED: HeatingAndRadiationPowerIgnited,
    PlasmaIgnitionModel.NON_IGNITED: HeatingAndRadiationPowerNonIgnited,
}
"""`.physics.i_plasma_ignited` -> the stellarator heating/radiation occupant."""


FAST_ALPHA_BETA = {
    FastAlphaPressureModel.ITER_PHYSICS_RULES: FastAlphaBetaIterPhysicsRules,
    FastAlphaPressureModel.WARD: FastAlphaBetaWard,
}
"""`.physics.i_beta_fast_alpha` -> the fast-alpha-pressure occupant."""

PLASMA_COMPOSITION = {
    PlasmaIgnitionModel.IGNITED: PlasmaCompositionIgnited,
    PlasmaIgnitionModel.NON_IGNITED: PlasmaCompositionNonIgnited,
}
"""`.physics.i_plasma_ignited` -> the plasma-composition occupant."""


CONFINEMENT_SCALING = {
    ConfinementTimeModel.ISS04_STELLARATOR: Iss04ConfinementTime,
    ConfinementTimeModel.ITER_IPB98Y2: IterIpb98y2ConfinementTime,
}
"""`i_confinement_time` -> the scaling-law occupant."""

COILS_MASS_MATERIAL = {
    SuperconductorModel.ITER_NB3SN: IterNb3snCoilsMass,
    SuperconductorModel.BI2212: Bi2212CoilsMass,
    SuperconductorModel.OLD_LUBELL_NBTI: OldLubellNbtiCoilsMass,
    SuperconductorModel.USER_DEFINED_NB3SN: UserDefinedNb3snCoilsMass,
    SuperconductorModel.WST_NB3SN: WstNb3snCoilsMass,
    SuperconductorModel.CROCO_REBCO: CrocoRebcoCoilsMass,
    SuperconductorModel.DURHAM_NBTI: DurhamNbtiCoilsMass,
    SuperconductorModel.DURHAM_REBCO: DurhamRebcoCoilsMass,
}
"""`i_tf_sc_mat` -> the occupant of `stellarator.coils.coils_mass`."""

WINDING_PACK_MATERIAL = {
    SuperconductorModel.ITER_NB3SN: IterNb3snWindingPackIntersectInputs,
    SuperconductorModel.BI2212: Bi2212WindingPackIntersectInputs,
    SuperconductorModel.OLD_LUBELL_NBTI: OldLubellNbtiWindingPackIntersectInputs,
    SuperconductorModel.USER_DEFINED_NB3SN: UserDefinedNb3snWindingPackIntersectInputs,
    SuperconductorModel.WST_NB3SN: WstNb3snWindingPackIntersectInputs,
    SuperconductorModel.CROCO_REBCO: CrocoRebcoWindingPackIntersectInputs,
    SuperconductorModel.DURHAM_NBTI: DurhamNbtiWindingPackIntersectInputs,
    SuperconductorModel.DURHAM_REBCO: DurhamRebcoWindingPackIntersectInputs,
}
"""`i_tf_sc_mat` -> the occupant of `stellarator.coils.winding_pack_intersect_inputs`.
"""

CONFINEMENT_TAIL = {
    ConfinementRadiationLossModel.CORE_ONLY: ConfinementTailCoreRadiation
}
"""`i_rad_loss` -> the occupant owning everything downstream of the law."""


def _plasma_power_loss_arm(i_plasma_ignited: int, i_rad_loss: int) -> int:
    """`(i_plasma_ignited, i_rad_loss)` -> the head's arm."""
    ignited = PlasmaIgnitionModel(int(i_plasma_ignited))
    radiation = ConfinementRadiationLossModel(int(i_rad_loss))
    if radiation is ConfinementRadiationLossModel.CORE_ONLY:
        return 0 if ignited is PlasmaIgnitionModel.IGNITED else 1
    return -1


def _cryo_q_nuc_arm(inuclear: int, i_tf_sup: int) -> int:
    """`(inuclear, i_tf_sup)` -> whether anything owns `.fwbs.qnuc`."""
    computed = (
        CoilNuclearHeatingModel(int(inuclear)) is CoilNuclearHeatingModel.FRANCES_FOX
        and TFConductorModel(int(i_tf_sup)) is TFConductorModel.SUPERCONDUCTING
    )
    return 0 if computed else 1


CRYO_Q_NUC = {0: CryoQNuc, 1: None}
"""The `.fwbs.qnuc` arm -> its occupant, or `None` for "nothing owns it"."""


def _eta_turbine_arm(i_thermal_electric_conversion, i_blanket_type) -> int:
    """`(i_thermal_electric_conversion, i_blanket_type)` -> who owns
    `.heat_transport.eta_turbine`, if anyone.
    """
    conversion = ElectricConversionModelTypes(int(i_thermal_electric_conversion))
    blanket = BlktModelTypes(int(i_blanket_type))
    if conversion is ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE:
        return 3
    if blanket is not BlktModelTypes.CCFE_HCPB:
        return 4
    return {
        ElectricConversionModelTypes.CCFE_HCPB_VALUE: 0,
        ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR: 1,
        ElectricConversionModelTypes.STEAM_RANKINE_CYCLE: 2,
    }.get(conversion, 4)


ETA_TURBINE = {
    0: EtaTurbineCcfeHcpbValue,
    1: EtaTurbineCcfeHcpbValueWithDivertor,
    2: EtaTurbineSteamRankineCycle,
    3: EtaTurbineSupercriticalCo2,
    4: None,
}
"""`_eta_turbine_arm(...)` -> the `.heat_transport.eta_turbine` occupant, or `None`."""

ETATH_LIQ = {
    ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE: (
        EtathLiqSupercriticalCo2
    ),
    ElectricConversionModelTypes.USER_INPUT: None,
}
"""`.fwbs.secondary_cycle_liq` -> the `.heat_transport.etath_liq` occupant, or `None`.
"""


def _temp_turbine_coolant_in_arm(
    i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq
) -> int:
    """`(i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq)` -> who
    owns `.heat_transport.temp_turbine_coolant_in`, if anyone.
    """
    if (
        ElectricConversionModelTypes(int(secondary_cycle_liq))
        is ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
    ):
        return 0
    stage_one_writes = _eta_turbine_arm(
        i_thermal_electric_conversion, i_blanket_type
    ) in (2, 3)
    return 1 if stage_one_writes else 2


TEMP_TURBINE_COOLANT_IN = {
    0: TempTurbineCoolantInFromLiquidBreeder,
    1: TempTurbineCoolantInFromBlanketCoolant,
    2: None,
}
"""`_temp_turbine_coolant_in_arm(...)` -> the occupant, or `None`."""


def _p_fw_div_heat_deposited_arm(i_p_coolant_pumping) -> int:
    """`.fwbs.i_p_coolant_pumping` -> who owns
    `.heat_transport.p_fw_div_heat_deposited_mw`.
    """
    return (
        1
        if PumpingPowerModelTypes(int(i_p_coolant_pumping))
        is PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP
        else 0
    )


P_FW_DIV_HEAT_DEPOSITED = {0: PFwDivHeatDepositedMwSummed, 1: None}
"""The `.heat_transport.p_fw_div_heat_deposited_mw` ownership arm -> its occupant, or
`None`.
"""


def _p_fw_blkt_coolant_pump_arm(i_p_coolant_pumping) -> int:
    """`.fwbs.i_p_coolant_pumping` -> who owns
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`.
    """
    return (
        1
        if PumpingPowerModelTypes(int(i_p_coolant_pumping))
        in (
            PumpingPowerModelTypes.MECHANICAL,
            PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP,
        )
        else 0
    )


P_FW_BLKT_COOLANT_PUMP = {0: PFwBlktCoolantPumpMw, 1: None}
"""The `.primary_pumping.p_fw_blkt_coolant_pump_mw` ownership arm -> its occupant, or
`None` for "the blanket owns it".
"""


def _energy_storage_arm(i_pulsed_plant: int, istore: int) -> int:
    """`(i_pulsed_plant, istore)` -> Account 225.3's arm."""
    if PlantOperationModel(int(i_pulsed_plant)) is PlantOperationModel.CONTINUOUS:
        return 0
    return {
        ThermalStorageModel.ELECTROWATT_OPTION_1: 1,
        ThermalStorageModel.ELECTROWATT_OPTION_2: 2,
    }.get(ThermalStorageModel(int(istore)), -1)


ENERGY_STORAGE = {
    0: EnergyStorageCostUnpulsed,
    1: EnergyStorageCostPulsedElectrowattOption1,
    2: EnergyStorageCostPulsedElectrowattOption2,
}
"""Account 225.3's arm -> its occupant."""
"""The `.fwbs.qnuc` arm -> its occupant, or `None` for "nothing owns it"."""


PLASMA_POWER_LOSS = {
    0: PlasmaPowerLossIgnitedCoreRadiation,
    1: PlasmaPowerLossNonIgnitedCoreRadiation,
}
"""The head's arm index -> its occupant."""

HEATING = {1: EcrhHeating, 2: LowhybHeating}
""".stellarator.isthtr` -> the auxiliary-heating occupant."""

FW_AREA = {0: AFwTotalNoPowerflow, 1: AFwTotalWithPowerflow}
"""`.heat_transport.ipowerflow` -> the first-wall-area occupant."""

BETA_NORM_MAX = {0: None, 1: BetaNormMaxWesson}
"""`.physics.i_beta_norm_max` -> `.tokamak.plasma_beta.norm_max`'s occupant."""

PROFILE_PARAMETERISATION = {
    0: ProfileParameterisationParabolic,
    1: ProfileParameterisationPedestal,
}
"""`.physics.i_plasma_pedestal` -> the profile-shape occupant."""


PEDESTAL_SEPARATRIX = {
    0: GreenwaldDensityFractions,
    1: PedestalSeparatrixDensities,
}
"""`.physics.i_nd_plasma_pedestal_separatrix` -> the pedestal/separatrix-density
occupant, **nested under `i_plasma_pedestal == 1`**.
"""


def _profile_parameterisation(
    i_plasma_pedestal, i_nd_plasma_pedestal_separatrix, *, is_stellarator
):
    """The profile-shape occupant, with each arm's own nested slot filled."""
    return _slot_occupant(
        "i_plasma_pedestal",
        i_plasma_pedestal,
        PROFILE_PARAMETERISATION,
        build=lambda cls: (
            cls(
                pedestal_separatrix=_slot_occupant(
                    "i_nd_plasma_pedestal_separatrix",
                    i_nd_plasma_pedestal_separatrix,
                    PEDESTAL_SEPARATRIX,
                )
            )
            if cls is ProfileParameterisationPedestal
            else cls(ecrh_density_limit=(EcrhDensityLimit() if is_stellarator else None))
        ),
    )


ST_INIT_I_PLASMA_PEDESTAL = 0
"""What `.physics.i_plasma_pedestal` is on a stellarator run, whatever the IN.DAT says.
"""

BUILDING_SIZING = {
    0: Bldgs,
    1: functools.partial(BldgsSizes, i_hcd_primary=CurrentDriveModel.ITER_NEUTRAL_BEAM),
}
"""`.buildings.i_bldgs_size` -> the building-size occupant."""

AVAIL = {
    BlanketLifetimeModel.NEUTRON_FLUENCE: AvailNeutronFluence,
    BlanketLifetimeModel.FUSION_POWER: AvailDisplacementsPerAtom,
}
"""`.costs.ibkt_life` -> the component-lifetime occupant."""


def _cplife_arm(itart: int, i_tf_sup: int) -> int:
    """`(itart, i_tf_sup)` -> who owns `.costs.cplife`, if anyone."""
    if SphericalTokamakModel(int(itart)) is not SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return 0
    return (
        1 if TFConductorModel(int(i_tf_sup)) is TFConductorModel.SUPERCONDUCTING else 2
    )


CPLIFE = {0: None, 1: CplifeAvailSuperconducting, 2: CplifeAvailResistive}
"""The `.costs.cplife` arm -> its occupant, or `None` for "nothing owns it"."""


def _cryo_q_loads_arm(i_tf_sup, i_pf_conductor) -> int:
    """`(i_tf_sup, i_pf_conductor)` -> who owns `.power.qss`/`qac`/`qcl`/`qmisc`."""
    if TFConductorModel(int(i_tf_sup)) is TFConductorModel.SUPERCONDUCTING:
        return 0
    return (
        1
        if PFConductorModel(int(i_pf_conductor)) is PFConductorModel.SUPERCONDUCTING
        else 2
    )


CRYO_Q_LOADS = {
    0: CryoQLoadsSuperconductingTf,
    1: CryoQLoadsResistiveTf,
    2: None,
}
"""`_cryo_q_loads_arm(...)` -> the occupant, or `None` for "nothing owns them"."""


def _cryo_loads_arm(i_tf_sup, i_pf_conductor) -> int:
    """`(i_tf_sup, i_pf_conductor)` -> the cryoplant-load occupant."""
    return 0 if _cryo_q_loads_arm(i_tf_sup, i_pf_conductor) != 2 else 1


CRYO_LOADS = {0: CryoLoadsActive, 1: CryoLoadsInactive}
"""`_cryo_loads_arm(...)` -> the cryoplant-load occupant."""


ACPOW = {
    PFEnergyStorageSource.LINE: AcpowLine,
    PFEnergyStorageSource.MGF: AcpowMotorGeneratorFlywheel,
}
"""`.pf_power.i_pf_energy_storage_source` -> the plant AC power occupant."""


TF_POWER = {0: TfPowerResistive, 1: TfPowerSuperconducting}
"""`.tfcoil.i_tf_sup` -> the TF-power occupant."""


def _electric_production_arm(
    ireactor: int, itart: int, i_tf_sup: int, i_blkt_dual_coolant, i_p_coolant_pumping
) -> int:
    """`(ireactor, itart, i_tf_sup, i_blkt_dual_coolant, i_p_coolant_pumping)` -> the
    electric-production arm.
    """
    if ireactor != 1:
        return 0
    centrepost = (
        SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        and TFConductorModel(int(i_tf_sup)) is TFConductorModel.WATER_COOLED_COPPER
    )
    liquid = (
        BlanketDualCoolantModel(int(i_blkt_dual_coolant))
        is not BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER
        and PumpingPowerModelTypes(int(i_p_coolant_pumping))
        is PumpingPowerModelTypes.MECHANICAL
    )
    return 1 + 2 * int(centrepost) + int(liquid)


ELECTRIC_PRODUCTION = {
    0: PowerProfilesOverTime,
    1: PlantElectricProductionSingleCoolant,
    2: PlantElectricProductionLiquidBreeder,
    3: PlantElectricProductionResistiveCentrepostSingleCoolant,
    4: PlantElectricProductionResistiveCentrepostLiquidBreeder,
}
"""`_electric_production_arm(...)` -> the electric-production occupant."""


def _no_cost_of_electricity():
    """The absent occupant of `costs.cost_of_electricity`: `ireactor != 1 or ipnet !=
    0`.
    """
    return None  # noqa: RET501 -- the returned `None` is the occupant, not a fall-off


TF_MAGNET_COST_SUPERCONDUCTING = {
    SuperconductorCostModel.PER_KG: TfMagnetCostSuperconductingPerKg,
    SuperconductorCostModel.PER_KAM: TfMagnetCostSuperconductingPerKam,
}
"""`.costs.supercond_cost_model` -> the Account 222.1 occupant."""

PF_MAGNET_COST = {
    0: PfMagnetCostPerKg,
    1: PfMagnetCostPerKam,
    2: PfMagnetCostPerKgNoCentralSolenoid,
    3: PfMagnetCostPerKamNoCentralSolenoid,
    4: PfMagnetCostPerKgCsWstNb3Sn,
}
"""`_pf_magnet_cost_arm(supercond_cost_model, iohcl, pf_coil_arm)` -> the Account 222.2
occupant.
"""


def _pf_magnet_cost_arm(supercond_cost_model, iohcl, pf_coil_arm) -> int:
    """`.costs.supercond_cost_model` x `.build.iohcl` x the PF coil system's own arm ->
    `PF_MAGNET_COST`'s arm index.
    """
    if (
        SuperconductorCostModel(int(supercond_cost_model))
        is SuperconductorCostModel.PER_KG
    ):
        if int(iohcl) == 0:
            return 2
        return 4 if int(pf_coil_arm) == 1 else 0
    return 1 if int(iohcl) != 0 else 3


COST_OF_ELECTRICITY = {
    0: _no_cost_of_electricity,
    1: CostOfElectricityConventionalAspectRatio,
    2: CostOfElectricitySphericalTokamak,
}
"""`_cost_of_electricity_arm(ireactor, ipnet, itart)` -> the cost-of-electricity
occupant, or `None`.
"""


def _cost_of_electricity_arm(ireactor: int, ipnet: int, itart: int) -> int:
    """Which arm of `Costs.run()`'s cost-of-electricity dispatch three switches select.
    """
    if ireactor != 1 or ipnet != 0:
        return 0
    return (
        2
        if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        else 1
    )


BLANKET_SHIELD_POWER = {
    1: BlanketShieldPowerExponential,
    2: DetailedPowerflowBlanketShieldPower,
    3: DetailedPowerflowBlanketShieldPowerUserInputPumping,
}
"""`_blanket_shield_power_arm(blktmodel, ipowerflow, i_p_coolant_pumping)` -> the
blanket/shield-power occupant.
"""

BLANKET_MASSES = {2: BlanketComponentMasses}
"""`_blanket_mass_arm(blktmodel, blkttype)` -> the blanket-mass occupant, same kind of
key.
"""


def _blanket_shield_power_arm(
    blktmodel: int, ipowerflow: int, i_p_coolant_pumping: int
) -> int:
    """Which arm of `st_fwbs`'s blanket/shield-power dispatch three switches select."""
    if blktmodel == 1:
        return 0
    if ipowerflow != 1:
        return 1
    pumping = PumpingPowerModelTypes(int(i_p_coolant_pumping))
    if pumping is PumpingPowerModelTypes.FRACTION_OF_HEAT:
        return 2
    if pumping is PumpingPowerModelTypes.USER_INPUT:
        return 3
    return 4


def _blanket_mass_arm(blktmodel: int, blkttype: int) -> int:
    """Which arm of `st_fwbs`'s blanket-mass dispatch a pair of switches selects."""
    if blktmodel != 0:
        return 0
    return 1 if blkttype in {1, 2} else 2


COST_MODEL = {0: Costs}
"""`.costs.i_cost_model` -> the cost-model occupant."""

DEVICE = {
    0: TokamakProcess,
    1: StellaratorProcess,
    2: StellaratorProcess,
    3: StellaratorProcess,
    4: StellaratorProcess,
    5: StellaratorProcess,
    6: StellaratorProcess,
}
"""`.stellarator.istell` -> the **device class**, and the first thing the factory reads.
"""

# ---------------------------------------------------------------------------
# The tokamak's own slots.
#
# Everything from here to `_INDAT_INTEGER` answers a switch that only a
# `TokamakProcess` asks. The shapes are the ones the stellarator registries above
# already use -- a dict keyed on an enum where one switch decides a slot, an
# `_*_arm` function turning a tuple of legal switch values into an **arm index**
# where several do -- and the discipline is the same: no switch value is ever a
# registry key for a joint dispatch, and no switch has a default outside its own
# declared domain.
#
# Two things here are new, and both are switch-shaped without being `i_*` integers:
# whether an **iteration variable** is active (`140 in ixc`, which decides which of
# two inverse assignments `build.py` makes) and whether an **input** is effectively
# zero (`.build.dz_xpoint_divertor < 1e-5`, which decides whether `divgeom` owns
# that field or leaves it an input). Both belong here for exactly the reason
# `machine_from_indat`'s docstring gives for every other switch: neither can change
# between two evaluations of one assembled graph, because `ixc` is fixed for a solve
# and an input is an input.
# ---------------------------------------------------------------------------


def _n_divertors(i_single_null: int) -> int:
    """`.physics.i_single_null` -> `.divertor.n_divertors`, as `init.py:606-617` does.
    """
    return (
        2
        if DivertorNumberModels(int(i_single_null)) is (DivertorNumberModels.DOUBLE_NULL)
        else 1
    )


def _fw_blkt_vv_shape_arm(itart: int, i_fw_blkt_vv_shape: int) -> int:
    """`(itart, i_fw_blkt_vv_shape)` -> the first-wall/blanket/vessel shape arm."""
    d_shaped = (
        SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        or FwBlktVVShape(int(i_fw_blkt_vv_shape)) is FwBlktVVShape.D_SHAPED
    )
    return 0 if d_shaped else 1


def _plasma_geometry_arm(i_plasma_current: int, i_plasma_shape: int) -> int:
    """`(i_plasma_current, i_plasma_shape)` -> the plasma-geometry arm."""
    sauter = (
        PlasmaCurrentModel(int(i_plasma_current)) is PlasmaCurrentModel.SAUTER_SCALING
        or PlasmaShapeModelType(int(i_plasma_shape)) is PlasmaShapeModelType.SAUTER
    )
    return 1 if sauter else 0


def _tf_shape(i_tf_shape: int, itart: int) -> TFCoilShapeModel:
    """`.tfcoil.i_tf_shape`, with `0` resolved the way `init.py` resolves it."""
    shape = TFCoilShapeModel(int(i_tf_shape))
    if shape is not TFCoilShapeModel.DEFAULT:
        return shape
    return (
        TFCoilShapeModel.PICTURE_FRAME
        if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        else TFCoilShapeModel.D_SHAPE
    )


def _tf_wp_geom(
    i_tf_wp_geom: int, i_tf_turns_integer: int
) -> SuperconductingTFWPShapeType:
    """`.tfcoil.i_tf_wp_geom`, with `UNSET` resolved the way `init.py:977-989` does."""
    geom = SuperconductingTFWPShapeType(int(i_tf_wp_geom))
    if geom is not SuperconductingTFWPShapeType.UNSET:
        return geom
    return (
        SuperconductingTFWPShapeType.RECTANGULAR
        if TFWPIntegerTurnType(int(i_tf_turns_integer)) is TFWPIntegerTurnType.INTEGER
        else SuperconductingTFWPShapeType.DOUBLE_RECTANGULAR
    )


# ---- `.tokamak.plasma_geom` -------------------------------------------------------

PLASMA_SHAPE = {
    PlasmaGeometryModelType.IPDG89_X_POINT: Ipdg89XPointPlasmaShape,
    PlasmaGeometryModelType.CREATE_DATA_EU_DEMO_X_POINT: (
        CreateDataEuDemoXPointPlasmaShape
    ),
}
"""`.physics.i_plasma_geometry` -> the kappa95/triang95 occupant."""

PLASMA_GEOMETRY = {0: DoubleArcPlasmaGeometry}
"""`_plasma_geometry_arm(i_plasma_current, i_plasma_shape)` -> the geometry occupant."""

# ---- `.tokamak.plasma_fields` and `.tokamak.physics` ------------------------------


def _surface_poloidal_field_arm(i_plasma_current: int) -> int:
    """`i_plasma_current` -> the poloidal-field arm."""
    return (
        1
        if PlasmaCurrentModel(int(i_plasma_current))
        is PlasmaCurrentModel.PENG_DIVERTOR_SCALING
        else 0
    )


SURFACE_POLOIDAL_FIELD = {0: SurfaceAveragedPoloidalFieldAmperes}
"""The poloidal-field arm -> its occupant. Ampere's law over the plasma perimeter."""

SEPARATRIX_POWER = {PlasmaIgnitionModel.NON_IGNITED: SeparatrixPowerNonIgnited}
"""`.physics.i_plasma_ignited` -> the separatrix-power occupant."""


def _pulse_ramp_times_arm(
    i_pulsed_plant: int, pulsetimings: int, i_t_current_ramp_up: int
) -> int:
    """`(i_pulsed_plant, pulsetimings, i_t_current_ramp_up)` -> the ramp-time arm."""
    if PlantOperationModel(int(i_pulsed_plant)) is PlantOperationModel.CONTINUOUS:
        return 0 if int(i_t_current_ramp_up) == 0 else 1
    return 2 if int(pulsetimings) == 0 else 3


PULSE_RAMP_TIMES = {
    0: PulseRampTimesContinuousDefault,
    2: PulseRampTimesPulsedDefault,
}
"""The ramp-time arm -> its occupant."""

# ---- `.tokamak.current_drive` -----------------------------------------------------

HCD_PRIMARY_EFFICIENCY = {
    CurrentDriveModel.USER_INPUT_ELECTRON_CYCLOTRON: HcdPrimaryEfficiencyUserInputEcrh
}
"""`.current_drive.i_hcd_primary` -> the primary current-drive efficiency occupant."""

HCD_PRIMARY_EFFICIENCY_FREETHY = {0: HcdPrimaryEfficiencyFreethyEcrhOMode}
"""`.current_drive.i_ecrh_wave_mode` -> the Freethy ECCD occupant, given `i_hcd_primary
== 13`.
"""


def _hcd_primary_efficiency(i_hcd_primary: int, i_ecrh_wave_mode: int):
    """The primary-efficiency occupant, resolving the one *nested* switch this slot has.
    """
    model = CurrentDriveModel(int(i_hcd_primary))
    if model is CurrentDriveModel.FREETHY_ELECTRON_CYCLOTRON:
        return _slot_occupant(
            "i_ecrh_wave_mode", int(i_ecrh_wave_mode), HCD_PRIMARY_EFFICIENCY_FREETHY
        )
    return _slot_occupant("i_hcd_primary", model, HCD_PRIMARY_EFFICIENCY)


HCD_SECONDARY_HEATING = {CurrentDriveModel.NO_CURRENT_DRIVE: HcdSecondaryHeatingNone}
"""`.current_drive.i_hcd_secondary` -> the secondary-heating occupant."""


def _hcd_primary_powers_arm(i_hcd_primary: int, i_hcd_secondary: int) -> int:
    """`(i_hcd_primary, i_hcd_secondary)` -> the primary-powers arm."""
    primary = CurrentDriveModel(int(i_hcd_primary)).method
    secondary = CurrentDriveModel(int(i_hcd_secondary))
    if (primary, secondary) == (
        CurrentDriveMethodType.ELECTRON_CYCLOTRON,
        CurrentDriveModel.NO_CURRENT_DRIVE,
    ):
        return 0
    return -1


HCD_PRIMARY_POWERS = {0: HcdPrimaryPowersElectronCyclotronNoSecondary}
"""The primary-powers arm -> its occupant. See `_hcd_primary_powers_arm`."""

HCD_CALCULATIONS = {1: TokamakCurrentDrive}
"""`.current_drive.i_hcd_calculations` -> the `.tokamak.current_drive` namespace itself.
"""

HCD_ELECTRIC_TOTAL = {
    PlasmaIgnitionModel.NON_IGNITED: HcdElectricTotalNonIgnited,
    PlasmaIgnitionModel.IGNITED: HcdElectricTotalIgnited,
}
"""`.physics.i_plasma_ignited` -> the wall-plug-power occupant."""

# ---- `.tokamak.build` -------------------------------------------------------------


def _divertor_geometry_arm(itart: int, dz_xpoint_divertor: float) -> int:
    """`(itart, input dz_xpoint_divertor)` -> `divgeom`'s arm."""
    if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return -1 if float(dz_xpoint_divertor) < 1e-5 else -3
    return 0 if float(dz_xpoint_divertor) < 1e-5 else -2


DIVERTOR_GEOMETRY = {
    0: DivertorGeometryConventional,
    -1: DivertorGeometrySphericalTokamak,
    -3: None,
}
"""`divgeom`'s arm -> its occupant, **or `None`**."""

TF_TOP_HEIGHT = {
    DivertorNumberModels.SINGLE_NULL: TfTopHeightSingleNull,
    DivertorNumberModels.DOUBLE_NULL: TfTopHeightDoubleNull,
}
"""`.physics.i_single_null` -> the occupant of `.tokamak.build.tf_top_height`."""

DR_TF_INBOARD_WINDING_PACK = {
    0: DrTfInboardFromWindingPack,
    1: DrTfWpWithInsulationFromInboardBuild,
}
"""`140 in ixc` -> which of two **inverse** assignments `build.py` makes."""


def _r_cp_top_arm(itart: int, i_tf_sup: int) -> int:
    """`(itart, i_tf_sup)` -> the centrepost-top-radius slot's arm."""
    return -1 if int(itart) == 1 and int(i_tf_sup) != 1 else 0


R_CP_TOP = {
    0: RCpTopFromTfInboardOut,
}
"""`_r_cp_top_arm(...)` -> `.build.r_cp_top`'s occupant."""


def _tf_inboard_radii_arm(i_tf_inside_cs: int, i_cs_precomp: int) -> int:
    """`(i_tf_inside_cs, i_cs_precomp)` -> the CS-to-TF radial slice's arm."""
    if (
        TFCSRadialConfiguration(int(i_tf_inside_cs))
        is TFCSRadialConfiguration.TF_INSIDE_CS
    ):
        return -1
    return 0 if int(i_cs_precomp) != 0 else -2


TF_INBOARD_RADII = {
    0: TfInboardRadiiTfOutsideCs,
    -2: TfInboardRadiiNoCsPrecomp,
}
"""`_tf_inboard_radii_arm(...)` -> the CS-to-TF radial-slice occupant
(`cold_boundary.md` producer 2, added 2026-08-27; arm -2 by the same day's ST frontier
wave).
"""

VACUUM_SHIELD_RADII = {
    TFCSRadialConfiguration.TF_OUTSIDE_CS: VacuumVesselAndShieldRadiiTfOutsideCs,
}
"""`.build.i_tf_inside_cs` -> the inboard vacuum-vessel/shield radial slice
(`build.py:1833-1860`), added 2026-08-29.
"""

DR_TF_OUTBOARD = {TFConductorModel.SUPERCONDUCTING: DrTfOutboardSuperconducting}
WP_CONDUCTOR_MAX_WIDTH = {
    TFConductorModel.SUPERCONDUCTING: WpConductorMaxWidthSuperconducting
}
"""`.tfcoil.i_tf_sup` -> the two build nodes that differ by conductor."""

TF_OUTBOARD_MID = {
    TFCoilShapeModel.D_SHAPE: TfOutboardMidDShape,
    TFCoilShapeModel.PICTURE_FRAME: TfOutboardMidPictureFrame,
}
TF_OUTBOARD_EDGE_RIPPLE = {
    TFCoilShapeModel.D_SHAPE: TfOutboardEdgeRipple,
    TFCoilShapeModel.PICTURE_FRAME: TfOutboardEdgeRipplePictureFrame,
}
"""`.tfcoil.i_tf_shape` (resolved by `_tf_shape`) -> the two ripple calls, per shape."""

# ---- `.tokamak.cicc_superconducting_tf_coil` --------------------------------------

TF_GLOBAL_GEOMETRY = {
    TFPlasmaCaseType.CIRCULAR: TfGlobalGeometryCircularCase,
    TFPlasmaCaseType.STRAIGHT: TfGlobalGeometryStraightCase,
}
TF_CASE_AREAS = {
    TFPlasmaCaseType.CIRCULAR: TfCaseAreasCircularFront,
    TFPlasmaCaseType.STRAIGHT: TfCaseAreasStraightFront,
}
"""`.tfcoil.i_tf_case_geom` -> two slots, both arms written for each."""

DR_TF_PLASMA_CASE = {False: DrTfPlasmaCaseFromInput, True: DrTfPlasmaCaseFromFraction}
"""`.tfcoil.i_f_dr_tf_plasma_case` -> the plasma-case thickness occupant, and the one
slot in this port whose two arms are **different kinds of node**.
"""

DX_TF_SIDE_CASE_MIN = {True: DxTfSideCaseMinFromFraction, False: None}
"""`.tfcoil.tfc_sidewall_is_fraction` -> the sidewall-thickness occupant, **or `None`**.
"""


def _tf_coil_shape_arm(
    i_tf_shape: TFCoilShapeModel, itart: int, i_single_null: int
) -> int:
    """`(i_tf_shape, itart, i_single_null)` -> the TF coil shape arm."""
    tart = SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
    if i_tf_shape is not TFCoilShapeModel.D_SHAPE:
        return 2 if tart else -2
    if tart:
        return -1
    return (
        0
        if DivertorNumberModels(int(i_single_null)) is DivertorNumberModels.SINGLE_NULL
        else 1
    )


TF_COIL_SHAPE = {
    0: TfCoilShapeDShapeSingleNull,
    1: TfCoilShapeDShapeDoubleNull,
    2: TfCoilShapePictureFrameTart,
}
"""The TF-coil-shape arm -> its occupant."""


def _tf_self_inductance_arm(i_tf_shape: TFCoilShapeModel, itart: int) -> int:
    """`(itart, i_tf_shape)` -> the self-inductance arm."""
    if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return 1
    return 0 if i_tf_shape is TFCoilShapeModel.D_SHAPE else 1


TF_COIL_SELF_INDUCTANCE = {
    0: TfCoilSelfInductanceDShape,
    1: TfCoilSelfInductancePictureFrame,
}
"""The self-inductance arm -> its occupant."""

SC_TF_WP_GEOMETRY = {
    SuperconductingTFWPShapeType.RECTANGULAR: SuperconductingTfWpGeometryRectangular,
    SuperconductingTFWPShapeType.DOUBLE_RECTANGULAR: (
        SuperconductingTfWpGeometryDoubleRectangular
    ),
    SuperconductingTFWPShapeType.TRAPEZOIDAL: SuperconductingTfWpGeometryTrapezoidal,
}
DX_TF_SIDE_CASE = {
    SuperconductingTFWPShapeType.RECTANGULAR: DxTfSideCaseRectangular,
    SuperconductingTFWPShapeType.DOUBLE_RECTANGULAR: DxTfSideCaseDoubleRectangular,
    SuperconductingTFWPShapeType.TRAPEZOIDAL: DxTfSideCaseTrapezoidal,
}
"""`.tfcoil.i_tf_wp_geom` (resolved by `_tf_wp_geom`) -> two slots, all three arms
written for each.
"""


def _peak_b_ripple_arm(n_tf_coils: float) -> int:
    """`round(n_tf_coils)` -> the ripple-fit arm; `-1` is the flat-allowance fallback.
    """
    count = round(float(n_tf_coils))
    return count if count in {16, 18, 20} else -1


PEAK_B_TF_RIPPLE = {
    16: PeakBTfInboardWithRipple16Coils,
    18: PeakBTfInboardWithRipple18Coils,
    20: PeakBTfInboardWithRipple20Coils,
    -1: PeakBTfInboardWithRippleFlatAllowance,
}
"""The ripple-fit arm -> its occupant. See `_peak_b_ripple_arm`."""


def _cicc_turn_geometry_arm(
    i_tf_turns_integer: int,
    i_dx_tf_turn_general_input: int,
    i_dx_tf_turn_cable_space_general_input: int,
) -> int:
    """`(i_tf_turns_integer, i_dx_tf_turn_general_input,
    i_dx_tf_turn_cable_space_general_input)` -> the turn-geometry arm.
    """
    if int(i_tf_turns_integer):
        return 1
    if int(i_dx_tf_turn_general_input):
        return -1
    return -2 if int(i_dx_tf_turn_cable_space_general_input) else 0


CICC_TURN_GEOMETRY = {
    0: CiccAveragedTurnGeometryFromCurrentPerTurn,
    1: CiccIntegerTurnGeometry,
}
"""The turn-geometry arm -> its occupant. See `_cicc_turn_geometry_arm`."""

SC_TF_MASSES = {
    (_itart, _mat): _occupant
    for _mat, (_conventional, _spherical) in {
        SuperconductorModel.ITER_NB3SN: (
            IterNb3snSuperconductingTfCoilAreasAndMassesConventional,
            IterNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.BI2212: (
            Bi2212SuperconductingTfCoilAreasAndMassesConventional,
            Bi2212SuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.OLD_LUBELL_NBTI: (
            OldLubellNbtiSuperconductingTfCoilAreasAndMassesConventional,
            OldLubellNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.USER_DEFINED_NB3SN: (
            UserDefinedNb3snSuperconductingTfCoilAreasAndMassesConventional,
            UserDefinedNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.WST_NB3SN: (
            WstNb3snSuperconductingTfCoilAreasAndMassesConventional,
            WstNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.CROCO_REBCO: (
            CrocoRebcoSuperconductingTfCoilAreasAndMassesConventional,
            CrocoRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.DURHAM_NBTI: (
            DurhamNbtiSuperconductingTfCoilAreasAndMassesConventional,
            DurhamNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.DURHAM_REBCO: (
            DurhamRebcoSuperconductingTfCoilAreasAndMassesConventional,
            DurhamRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
        SuperconductorModel.HAZELTON_ZHAI_REBCO: (
            HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesConventional,
            HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak,
        ),
    }.items()
    for _itart, _occupant in (
        (SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO, _conventional),
        (SphericalTokamakModel.SPHERICAL_TOKAMAK, _spherical),
    )
}
"""`(.physics.itart, .tfcoil.i_tf_sc_mat)` -> the superconducting TF mass occupant."""


def _tf_field_and_force_arm(itart: int, i_cp_joints: int) -> bool:
    """`(itart, i_cp_joints)` -> whether the centrepost joints slide."""
    if int(i_cp_joints) == -1:
        i_cp_joints = 0
    return bool(int(itart) == 1 and int(i_cp_joints) == 1)


TF_FIELD_AND_FORCE = {False: TfFieldAndForceClampedJoints}
"""`itart == 1 and i_cp_joints == 1` -> the vertical-tension occupant."""


def _tf_stress_arm(
    i_tf_stress_model: int, i_tf_bucking: int, i_tf_turns_integer: int
) -> tuple[int, int, int]:
    """`(i_tf_stress_model, i_tf_bucking, i_tf_turns_integer)` -> the stress arm."""
    return (int(i_tf_stress_model), int(i_tf_bucking), 1 if i_tf_turns_integer else 0)


TF_STRESS = {
    (1, 1, 0): TfStressPlaneStressBuckedCaseAveragedTurn,
    (1, 1, 1): TfStressPlaneStressBuckedCaseIntegerTurn,
    (0, 1, 0): TfStressExtendedPlaneStrainBuckedCaseAveragedTurn,
}
"""The stress arm -> its occupant. See `_tf_stress_arm` and `stress.py`'s docstring."""

CICC_SUPERCONDUCTOR_PROPERTIES = {
    (1, SuperconductorModel.ITER_NB3SN): IterNb3snCiccSuperconductorProperties,
    (1, SuperconductorModel.OLD_LUBELL_NBTI): OldLubellNbtiCiccSuperconductorProperties,
    (
        1,
        SuperconductorModel.USER_DEFINED_NB3SN,
    ): UserDefinedNb3snCiccSuperconductorProperties,
    (1, SuperconductorModel.WST_NB3SN): WstNb3snCiccSuperconductorProperties,
    (1, SuperconductorModel.DURHAM_NBTI): DurhamNbtiCiccSuperconductorProperties,
}
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the critical-current occupant."""

TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN = {
    (1, SuperconductorModel.ITER_NB3SN): IterNb3snTfSuperconductorTemperatureMargin,
    (
        1,
        SuperconductorModel.OLD_LUBELL_NBTI,
    ): OldLubellNbtiTfSuperconductorTemperatureMargin,
    (
        1,
        SuperconductorModel.USER_DEFINED_NB3SN,
    ): UserDefinedNb3snTfSuperconductorTemperatureMargin,
    (1, SuperconductorModel.WST_NB3SN): WstNb3snTfSuperconductorTemperatureMargin,
}
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the temperature-margin occupant."""


def _croco_turn_geometry_arm(
    i_tf_turns_integer: int,
    i_dx_tf_turn_general_input: int,
    i_dx_tf_turn_cable_space_general_input: int,
) -> int:
    """`(i_tf_turns_integer, i_dx_tf_turn_general_input,
    i_dx_tf_turn_cable_space_general_input)` -> the CroCo turn-geometry arm.
    """
    return _cicc_turn_geometry_arm(
        i_tf_turns_integer,
        i_dx_tf_turn_general_input,
        i_dx_tf_turn_cable_space_general_input,
    )


CROCO_TURN_GEOMETRY = {0: CrocoAveragedTurnGeometryFromCurrentPerTurn}
"""The CroCo turn-geometry arm -> its occupant."""

CROCO_SUPERCONDUCTOR_PROPERTIES = {
    (
        1,
        SuperconductorModel.HAZELTON_ZHAI_REBCO,
    ): HazeltonZhaiRebcoCrocoSuperconductorProperties,
}
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the CroCo critical-current occupant."""

CROCO_TEMPERATURE_MARGIN = {
    (
        1,
        SuperconductorModel.HAZELTON_ZHAI_REBCO,
    ): HazeltonZhaiRebcoCrocoTemperatureMargin,
}
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the CroCo temperature-margin occupant.
"""

# ---- `.tokamak.ccfe_hcpb` ---------------------------------------------------------

BLANKET_HALF_HEIGHT = {
    1: BlanketHalfHeightSingleNull,
    2: BlanketHalfHeightDoubleNull,
}
BLANKET_COVERAGE_FACTORS = {
    1: BlanketCoverageFactorsSingleNull,
    2: BlanketCoverageFactorsDoubleNull,
}
DIVERTOR_SURFACE_MASS = {
    1: DivertorSurfaceAndPlateMassSingleNull,
    2: DivertorSurfaceAndPlateMassDoubleNull,
}
"""`.divertor.n_divertors` (derived by `_n_divertors`) -> three slots, each total."""

BLANKET_AREAS = {0: DShapedBlanketAreas, 1: EllipticalBlanketAreas}
BLANKET_VOLUMES = {0: DShapedBlanketVolumes, 1: EllipticalBlanketVolumes}
"""`_fw_blkt_vv_shape_arm(itart, i_fw_blkt_vv_shape)` -> two slots, both **total** since
2026-08-27 (the D-shaped wave, for `spherical_tokamak_eval.IN.DAT` and
`st_regression.IN.DAT`, which set `i_fw_blkt_vv_shape = 1` *and* `itart = 1`).
"""

NUCLEAR_HEATING_MAGNETS = {
    SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO: NuclearHeatingMagnetsConventional,
    SphericalTokamakModel.SPHERICAL_TOKAMAK: NuclearHeatingMagnetsSphericalTokamak,
}
NUCLEAR_HEATING_SHIELD = {
    SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO: NuclearHeatingShieldConventional,
    SphericalTokamakModel.SPHERICAL_TOKAMAK: NuclearHeatingShieldSphericalTokamak,
}
"""`.physics.itart` -> two **total** slots, since 2026-08-27 (the centrepost wave)."""


def _centrepost_neutronics_arm(itart: int, i_tf_sup: int) -> int:
    """`(itart, i_tf_sup)` -> the centrepost-neutronics arm."""
    conventional = (
        SphericalTokamakModel(int(itart))
        is SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO
    )
    if conventional:
        return 0
    conductor = TFConductorModel(int(i_tf_sup))
    if conductor is TFConductorModel.SUPERCONDUCTING:
        return 1
    return -1 if conductor is TFConductorModel.WATER_COOLED_COPPER else -2


CENTREPOST_NEUTRONICS = {
    0: CentrepostNeutronicsAbsent,
    1: CentrepostNeutronicsSphericalTokamakSuperconducting,
}
"""The centrepost-neutronics arm -> its occupant."""


def _nuclear_heating_renormalisation_arm(n_divertors: int, itart: int) -> int:
    """`(n_divertors, itart)` -> the renormalisation arm."""
    conventional = (
        SphericalTokamakModel(int(itart))
        is SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO
    )
    single_null = int(n_divertors) == 1
    if conventional:
        return 0 if single_null else 1
    return 2 if single_null else 3


NUCLEAR_HEATING_RENORMALISATION = {
    0: NuclearHeatingRenormalisationSingleNullConventional,
    1: NuclearHeatingRenormalisationDoubleNullConventional,
    2: NuclearHeatingRenormalisationSingleNullSphericalTokamak,
    3: NuclearHeatingRenormalisationDoubleNullSphericalTokamak,
}
"""The renormalisation arm -> its occupant."""

PUMPING_POWER = {
    PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP: (
        PumpingPowerMechanicalWithPressureDrop
    )
}
"""`.fwbs.i_p_coolant_pumping` -> the pumping-power occupant, and the clearest case in
this port of arms that **do not own the same set**.
"""

BLANKET_MODEL = {BlktModelTypes.CCFE_HCPB: CcfeHcpb}
"""`.fwbs.i_blanket_type` -> the occupant of `.tokamak.ccfe_hcpb`."""

# ---- the four single-node tokamak slots -------------------------------------------


def _first_wall_arm(n_divertors: int, shape_arm: int, i_pflux_fw_neutron: int) -> int:
    """`(n_divertors, shape arm, i_pflux_fw_neutron)` -> `FirstWall`'s arm."""
    if int(i_pflux_fw_neutron) != 1:
        return -3
    if shape_arm == 0 and int(n_divertors) == 1:
        return -2
    return {(1, 1): 0, (1, 2): 1, (0, 2): 2}[shape_arm, int(n_divertors)]


FIRST_WALL = {
    0: FirstWallSingleNull,
    1: FirstWallDoubleNull,
    2: FirstWallDShapedDoubleNull,
}
"""`_first_wall_arm(...)` -> `.tokamak.first_wall`'s occupant, three cells of a 2 x 2.
"""


def _vacuum_vessel_arm(n_divertors: int, shape_arm: int) -> int:
    """`(n_divertors, shape arm)` -> the vacuum vessel's arm."""
    if shape_arm == 0 and int(n_divertors) == 1:
        return -2
    return {(1, 1): 0, (1, 2): 1, (0, 2): 2}[shape_arm, int(n_divertors)]


VACUUM_VESSEL = {
    0: VacuumVesselEllipticalSingleNull,
    1: VacuumVesselEllipticalDoubleNull,
    2: VacuumVesselDShapedDoubleNull,
}
"""`_vacuum_vessel_arm(...)` -> `.tokamak.vacuum_vessel`'s occupant, three cells of a 2
x 2.
"""


def _structure_arm(i_tf_sup: int, i_pf_conductor: int) -> int:
    """`(i_tf_sup, i_pf_conductor)` -> `Structure`'s arm: one cell of a 2x2."""
    sc_tf = TFConductorModel(int(i_tf_sup)) is TFConductorModel.SUPERCONDUCTING
    sc_pf = PFConductorModel(int(i_pf_conductor)) is not PFConductorModel.RESISTIVE
    if sc_tf and sc_pf:
        return 0
    if sc_pf:
        return -1
    return -2 if sc_tf else -3


STRUCTURE = {0: Structure}
"""`_structure_arm(i_tf_sup, i_pf_conductor)` -> `.tokamak.structure`'s occupant."""


def _divertor_heat_load_arm(i_div_heat_load: int, n_divertors: int) -> int:
    """`(i_div_heat_load, n_divertors)` -> the divertor heat-load arm."""
    model = DivertorHeatLoadModel(int(i_div_heat_load))
    if model is DivertorHeatLoadModel.USER_INPUT:
        return -1
    if model is DivertorHeatLoadModel.PENG_CHAMBER:
        return -2
    return 0 if int(n_divertors) == 1 else 1


DIVERTOR_HEAT_LOAD = {
    0: DivertorHeatLoadWadeSingleNull,
    1: DivertorHeatLoadWadeDoubleNull,
}
"""The divertor heat-load arm -> its occupant. See `_divertor_heat_load_arm`."""


# ---------------------------------------------------------------------------
# Waves 2/3 (consolidation round 2): the plasma-current chain, the current
# fractions, the L-H threshold, the density limit, the scrape-off layer, the
# plasma inductance, the shield and the PF coil system.
# ---------------------------------------------------------------------------

PLASMA_CURRENT_SCALING = {
    PlasmaCurrentModel.IPDG89_SCALING: Ipdg89PlasmaCurrent,
    PlasmaCurrentModel.FIESTA_ST_SCALING: FiestaStPlasmaCurrent,
}
"""`.physics.i_plasma_current` -> `.tokamak.plasma_current.plasma_current`'s occupant.
"""

CURRENT_PROFILE_INDEX = {
    CurrentProfileIndexModel.USER_INPUT: None,
    CurrentProfileIndexModel.WESSON: WessonCurrentProfileIndex,
}
"""`.physics.i_alphaj` -> the current-profile-index occupant."""

IND_PLASMA_INTERNAL_NORM = {
    IndInternalNormModel.USER_INPUT: None,
    IndInternalNormModel.WESSON: PlasmaInternalInductanceNormWesson,
}
"""`.physics.i_ind_plasma_internal_norm` -> the normalised-internal-inductance occupant,
in `.tokamak.plasma_inductance`.
"""

BOOTSTRAP_CURRENT = {
    BootstrapCurrentFractionModel.USER_INPUT: None,
    BootstrapCurrentFractionModel.SAUTER: SauterBootstrapCurrentFraction,
}
"""`.physics.i_bootstrap_current` -> `.tokamak.bootstrap_current`'s occupant."""

DIAMAGNETIC_CURRENT = {
    PlasmaDiamagneticCurrentModel.NONE: NoDiamagneticCurrent,
    PlasmaDiamagneticCurrentModel.SCENE_FIT: SceneDiamagneticCurrent,
}
"""`.physics.i_diamagnetic_current` -> `.tokamak.diamagnetic_current`'s occupant."""

PFIRSCH_SCHLUTER_CURRENT = {0: NoPfirschSchluterCurrent, 1: ScenePfirschSchluterCurrent}
"""`.physics.i_pfirsch_schluter_current` -> `.tokamak.pfirsch_schluter_current`'s
occupant.
"""

L_H_THRESHOLD = {
    PlasmaConfinementTransitionModel.MARTIN08_NOMINAL: Martin08NominalLHThresholdPower,
    PlasmaConfinementTransitionModel.MARTIN08_UPPER: Martin08UpperLHThresholdPower,
    PlasmaConfinementTransitionModel.MARTIN08_LOWER: Martin08LowerLHThresholdPower,
    PlasmaConfinementTransitionModel.MARTIN08_ASPECT_NOMINAL: (
        Martin08AspectNominalLHThresholdPower
    ),
    PlasmaConfinementTransitionModel.MARTIN08_ASPECT_UPPER: (
        Martin08AspectUpperLHThresholdPower
    ),
    PlasmaConfinementTransitionModel.MARTIN08_ASPECT_LOWER: (
        Martin08AspectLowerLHThresholdPower
    ),
}
"""`.physics.i_l_h_threshold` -> `.tokamak.l_h_transition`'s occupant."""

DENSITY_LIMIT_ENFORCED = {DensityLimitModel.GREENWALD: EnforcedDensityLimitGreenwald}
"""`.physics.i_density_limit` -> `.tokamak.density_limit.enforced_density_limit`'s
occupant.
"""

SOL_OUTBOARD_POWER_DECAY = {
    OutbordSOLPowerDecayLengthModel.USER_INPUT: None,
    OutbordSOLPowerDecayLengthModel.EICH_2013: OutboardSOLPowerDecayLengthEich2013,
}
"""`.physics.i_len_sol_outboard_power_decay` -> the selector occupant in
`.tokamak.scrape_off_layer`.
"""

SHIELD_HALF_HEIGHT = {1: SingleNullShieldHalfHeight, 2: DoubleNullShieldHalfHeight}
"""`_n_divertors(i_single_null)` -> `.tokamak.shield.half_height`'s occupant."""

SHIELD_VOLUMES = {0: DShapedShieldVolumes, 1: EllipticalShieldVolumes}
"""`_fw_blkt_vv_shape_arm(itart, i_fw_blkt_vv_shape)` -> `.tokamak.shield.volumes`'s
occupant -- the fifth slot keyed on that existing joint predicate, per `shield.md`'s
'join that key at consolidation, not mint an independent one'.
"""


def _pf_coil_topology(iohcl):
    """`.build.iohcl` -> the `PFCoilTopology` the ported occupant set was written for.
    """
    return REFERENCE_TOPOLOGY if int(iohcl) != 0 else SPHERICAL_TOKAMAK_TOPOLOGY


def _pf_coil_system_arm(
    iohcl,
    n_pf_coil_groups,
    i_pf_location,
    n_pf_coils_in_group,
    itart,
    itartpf,
    i_pf_current,
    i_pf_conductor,
    i_pf_superconductor,
    i_cs_superconductor,
    i_tf_shape,
    i_r_pf_outside_tf_placement,
) -> int:
    """Every switch the PF coil system's thirteen nodes branch on, resolved to one arm.
    """
    deviations = _pf_coil_system_deviations(
        iohcl=iohcl,
        n_pf_coil_groups=n_pf_coil_groups,
        i_pf_location=i_pf_location,
        n_pf_coils_in_group=n_pf_coils_in_group,
        itart=itart,
        itartpf=itartpf,
        i_pf_current=i_pf_current,
        i_pf_conductor=i_pf_conductor,
        i_pf_superconductor=i_pf_superconductor,
        i_cs_superconductor=i_cs_superconductor,
        i_tf_shape=i_tf_shape,
        i_r_pf_outside_tf_placement=i_r_pf_outside_tf_placement,
    )
    if deviations:
        return deviations[0]
    return _pf_coil_material_arm(iohcl, i_cs_superconductor)


def _pf_coil_material_arm(iohcl, i_cs_superconductor) -> int:
    """`_pf_coil_system_arm`'s **positive** arms alone: `.build.iohcl` x the CS
    superconductor material, with no deviation check in front of it.
    """
    if int(iohcl) == 0:
        return 2
    return (
        0
        if SuperconductorModel(int(i_cs_superconductor))
        is SuperconductorModel.ITER_NB3SN
        else 1
    )


def _pf_coil_system_deviations(
    *,
    iohcl,
    n_pf_coil_groups,
    i_pf_location,
    n_pf_coils_in_group,
    itart,
    itartpf,
    i_pf_current,
    i_pf_conductor,
    i_pf_superconductor,
    i_cs_superconductor,
    i_tf_shape,
    i_r_pf_outside_tf_placement,
) -> tuple[int, ...]:
    """**Every** dimension of the joint PF configuration with no occupant, not just the
    first -- in `_pf_coil_system_arm`'s own order, so `deviations[0]` is exactly the arm
    that function used to return by short-circuit.
    """
    has_cs = int(iohcl) != 0
    topology = _pf_coil_topology(iohcl)

    # `place_pf_outside_tf` reads its two switches only through this disjunction
    # (`pfcoil.py:1322-1326`), and each written occupant bakes one answer to it:
    # `PFCoilPlacement` the curve, `PFCoilPlacementSphericalTokamak` the stack.
    stacked_outside_tf = (
        i_tf_shape is TFCoilShapeModel.PICTURE_FRAME
        or int(i_r_pf_outside_tf_placement) == 1
    )
    occupant_stacks_outside_tf = not has_cs

    deviations = []
    if (
        int(n_pf_coil_groups) != topology.n_pf_coil_groups
        or tuple(int(v) for v in i_pf_location[: topology.n_pf_coil_groups])
        != tuple(int(v) for v in topology.i_pf_location)
        or tuple(int(v) for v in n_pf_coils_in_group[: topology.n_pf_coil_groups])
        != topology.n_pf_coils_in_group
    ):
        deviations.append(-2)
    if (
        SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        and int(itartpf) == 0
    ):
        deviations.append(-3)
    if int(i_pf_current) == 0:
        deviations.append(-4)
    if PFConductorModel(int(i_pf_conductor)) is not PFConductorModel.SUPERCONDUCTING:
        deviations.append(-5)
    if has_cs:
        supported_superconductors = (
            SuperconductorModel(int(i_pf_superconductor)),
            SuperconductorModel(int(i_cs_superconductor)),
        ) in {
            (SuperconductorModel.OLD_LUBELL_NBTI, SuperconductorModel.ITER_NB3SN),
            (SuperconductorModel.OLD_LUBELL_NBTI, SuperconductorModel.WST_NB3SN),
        }
    else:
        supported_superconductors = (
            SuperconductorModel(int(i_pf_superconductor))
            is SuperconductorModel.HAZELTON_ZHAI_REBCO
        )
    if not supported_superconductors:
        deviations.append(-6)
    if stacked_outside_tf is not occupant_stacks_outside_tf:
        deviations.append(-7)
    return tuple(deviations)


def _refuse_pf_coil_system(deviations):
    """`_slot_occupant`'s `NotImplementedError`, but for every deviating dimension."""
    field = "pf_coil_system_arm"
    first, *rest = deviations
    opening = (
        f"{field} == {first} is a real PROCESS branch but is not ported: "
        f"{UNPORTED[field, first]}"
    )
    closing = (
        f"{len(deviations)} of the six dimensions of the PF coil system's joint "
        f"configuration deviate at once; every one of them needs an occupant set "
        f"before this file assembles."
    )
    message = [
        opening,
        *(f"AND {field} == {arm}: {UNPORTED[field, arm]}" for arm in rest),
        closing,
    ]
    raise NotImplementedError("\n\n".join(message))


CS_COIL = {0: CSCoil, 1: CSCoil, 2: None}
"""`_pf_coil_system_arm` -> `.tokamak.cs_coil`'s occupant namespace."""

CS_SUPERCONDUCTOR = {
    SuperconductorModel.ITER_NB3SN: CSCriticalCurrentDensitiesIterNb3Sn,
    SuperconductorModel.WST_NB3SN: CSCriticalCurrentDensitiesWstNb3Sn,
}
"""`.pf_coil.i_cs_superconductor` -> `.tokamak.cs_coil.critical_current`'s occupant."""

CS_TEMPERATURE_MARGIN = {
    SuperconductorModel.ITER_NB3SN: CSTemperatureMarginIterNb3Sn,
    SuperconductorModel.WST_NB3SN: CSTemperatureMarginWstNb3Sn,
}
"""`.pf_coil.i_cs_superconductor` -> `.tokamak.cs_coil.temperature_margin`'s occupant.
"""

PF_COIL = {0: PFCoil, 1: PFCoilCsWstNb3Sn, 2: PFCoilSphericalTokamak}
"""`_pf_coil_system_arm` -> `.tokamak.pf_coil`'s occupant namespace."""


def _as_imported(source):
    """`Imported` for a path, or the one already read."""
    return source if isinstance(source, Imported) else read_indat(source)


_INTEGER_TEXT = re.compile(r"-?\d+")


def switches_from_indat(input_file):
    """Every `name = <integer>` this input file sets, as a plain dict."""
    found = {}
    for assignment in _as_imported(input_file).assignments:
        if assignment.index is None and _INTEGER_TEXT.fullmatch(assignment.text):
            found[assignment.name] = int(assignment.text)
    return found


def numbers_from_indat(input_file):
    """Every `name = <number>` this input file sets, as a plain dict of floats."""
    found = {}
    for assignment in _as_imported(input_file).assignments:
        if assignment.index is not None:
            continue
        try:
            found[assignment.name] = float(assignment.text.lower().replace("d", "e"))
        except ValueError:
            continue
    return found


def int_lists_from_indat(input_file):
    """Every `name = <int>, <int>, ...` this input file sets, as a dict of tuples."""
    found = {}
    for assignment in _as_imported(input_file).assignments:
        if assignment.index is not None or "," not in assignment.text:
            continue
        items = [v.strip() for v in assignment.text.split(",") if v.strip()]
        if len(items) > 1 and all(_INTEGER_TEXT.fullmatch(v) for v in items):
            found[assignment.name] = tuple(int(v) for v in items)
    return found


def iteration_variables_from_indat(input_file):
    """The `ixc` this input file declares, as a frozenset of iteration-variable IDs."""
    return frozenset(_as_imported(input_file).problem.ixc)


def problem_from_indat(input_file):
    """The problem statement this input file declares -- `importer.Problem`, in order.
    """
    return _as_imported(input_file).problem


def objective_selection(i_figure_merit):
    """Which figure of merit the run states, and in which direction --
    `sand.ObjectiveSelection`, resolved here and nowhere else.
    """
    from functional_process.cottax.core.solver import objectives  # noqa: PLC0415
    from functional_process.cottax.sand import ObjectiveSelection  # noqa: PLC0415

    merit = FiguresOfMerit(abs(int(i_figure_merit)))
    return ObjectiveSelection(
        metric=objectives.OBJECTIVE_METRICS[merit],
        maximise=int(i_figure_merit) < 0,
    )


# ------------------------------------------------------------- sentinel resolution
#
# `_audit/init_audit.md` §2a: eight `init.py` writes are a **sentinel** resolved to a
# value, not a default. §24.2 item 2 says why they cannot be nodes as stated -- each
# reads and writes one path -- and what the shape is instead: the importer emits a `raw`
# namespace and a resolution maps raw -> resolved. These are that step, done in the
# factory because that is where the answer is needed, and each is *one* function so no
# consumer can carry a second copy of the rule.
#
# Only the sentinels the factory or the switch values actually need are here.
# `i_tf_wp_geom` and `i_tf_shape` are resolved elsewhere in this module against the same
# `init.py` lines; `eff_tf_cryo`, `eyoung_ins`, `eyoung_cond_axial` and `i_cp_joints` are
# *values*, so they belong to the boundary and not to assembly, and
# `n_equality_constraints` belongs to the problem statement (`problem_from_indat`).


def resolve_i_tf_bucking(i_tf_bucking, i_tf_sup):
    """`init.py:891-895`: the `-1` sentinel -> `0` for copper, `1` for anything else."""
    if int(i_tf_bucking) != -1:
        return int(i_tf_bucking)
    return 0 if int(i_tf_sup) == TFConductorModel.WATER_COOLED_COPPER else 1


EFF_TF_CRYO_UNSET = -1.0
"""`tfcoil_variables.py`'s `eff_tf_cryo` default: a **sentinel**, not an efficiency."""


def resolve_eff_tf_cryo(eff_tf_cryo, i_tf_sup):
    """`init.py:933-940`: the `-1.0` sentinel -> the cryoplant efficiency for a magnet.
    """
    conductor = TFConductorModel(int(i_tf_sup))
    if abs(float(eff_tf_cryo) + 1.0) >= 1e-6:
        return float(eff_tf_cryo)
    if conductor is TFConductorModel.SUPERCONDUCTING:
        return 0.13
    if conductor is TFConductorModel.HELIUM_COOLED_ALUMINIUM:
        return 0.40
    return float(eff_tf_cryo)


EYOUNG_INS_UNSET = 1.0e8
"""`tfcoil_variables.py:383`'s `eyoung_ins` default -- **a sentinel that looks like an
answer**.
"""

I_TF_COND_EYOUNG_AXIAL_DEFAULT = 0
"""`tfcoil_variables.py:275`."""

I_TF_COND_EYOUNG_TRANS_DEFAULT = 1
"""`tfcoil_variables.py:287`. Read only when `i_tf_cond_eyoung_axial == 2`."""


def resolve_eyoung_ins(eyoung_ins, i_tf_sup):
    """`init.py:961-975`: the insulation Young's modulus, by magnet technology."""
    conductor = TFConductorModel(int(i_tf_sup))
    if float(eyoung_ins) > EYOUNG_INS_UNSET:
        return float(eyoung_ins)
    if conductor in {
        TFConductorModel.WATER_COOLED_COPPER,
        TFConductorModel.SUPERCONDUCTING,
    }:
        return 20.0e9
    if conductor is TFConductorModel.HELIUM_COOLED_ALUMINIUM:
        return 2.5e9
    return float(eyoung_ins)


EYOUNG_COND_AXIAL_LITERATURE = {
    # Nyilas, A et al., Superconductor Science and Technology 16, no. 9 (2003): 1036-42.
    # https://doi.org/10.1088/0953-2048/16/9/313.
    SuperconductorMaterial.NB3SN: 32e9,
    # Brown, M. et al., IOP Conference Series: Materials Science and Engineering 279
    # (2017): 012022. https://doi.org/10.1088/1757-899X/279/1/012022.
    SuperconductorMaterial.BI2212: 80e9,
    # Vedrine, P. et al., IEEE Transactions on Applied Superconductivity 9, no. 2
    # (1999): 236-39. https://doi.org/10.1109/77.783280.
    SuperconductorMaterial.NBTI: 6.8e9,
    # Fujishiro, H. et al., Physica C: Superconductivity, 426-431 (2005): 699-704.
    # https://doi.org/10.1016/j.physc.2005.01.045.
    SuperconductorMaterial.REBCO: 145e9,
}
"""`init.py:1002-1027`'s literature table: conductor axial Young's modulus (Pa) keyed on
the superconductor material, each with the DOI `init.py` carries beside it.
"""


def resolve_eyoung_cond(
    eyoung_cond_axial,
    eyoung_cond_trans,
    i_tf_cond_eyoung_axial,
    i_tf_cond_eyoung_trans,
    i_tf_sc_mat,
):
    """`init.py:992-1034`: `(eyoung_cond_axial, eyoung_cond_trans)`, both at once."""
    axial, trans = float(eyoung_cond_axial), float(eyoung_cond_trans)
    if int(i_tf_cond_eyoung_axial) == 0:
        # Conductor stiffness is not considered.
        return 0.0, 0.0
    if int(i_tf_cond_eyoung_axial) != 2:
        # `== 1`: both are the user's, and `init.py` writes neither.
        return axial, trans
    axial = EYOUNG_COND_AXIAL_LITERATURE.get(
        SuperconductorModel(int(i_tf_sc_mat)).material, axial
    )
    return axial, (0.0 if int(i_tf_cond_eyoung_trans) == 0 else axial)


I_PF_CONDUCTOR_DEFAULT = 0
"""`pfcoil_variables.py:230` -- `0` is superconducting, which is the arm `init.py:1140`
zeroes the resistivity on.
"""

I_HCD_CALCULATIONS_DEFAULT = 1
"""`current_drive_variables.py:223`."""

I_HCD_PRIMARY_DEFAULT = 5
"""`current_drive_variables.py:190` -- one of the two NBI values, so a silent file has a
beam.
"""

NBI_PRIMARY_HEATING = frozenset({5, 8})
"""The `i_hcd_primary` values that are neutral beam injection, as `init.py:1146` spells
them: a bare set of two integers, with no enum in PROCESS behind it.
"""


def resolve_rho_pf_coil(rho_pf_coil, i_pf_conductor):
    """`init.py:1140`: a superconducting PF coil has zero resistivity."""
    if PFConductorModel(int(i_pf_conductor)) is PFConductorModel.SUPERCONDUCTING:
        return 0.0
    return float(rho_pf_coil)


def resolve_f_nd_beam_electron(f_nd_beam_electron, i_hcd_calculations, i_hcd_primary):
    """`init.py:1145-1147`: no NBI means no hot beam density."""
    if int(i_hcd_calculations) == 1 and int(i_hcd_primary) in NBI_PRIMARY_HEATING:
        return float(f_nd_beam_electron)
    return 0.0


I_PULSED_PLANT_DEFAULT = 0
"""`pulse_variables.py:30`."""

I_SINGLE_NULL_DEFAULT = 1
"""`physics_variables.py`'s own default: a single-null plasma, the arm under which
`init.py` writes none of the three upper-build identities.
"""


def resolve_esbldgm3(esbldgm3, i_pulsed_plant):
    """`init.py:827`: a steady-state plant needs no energy storage building."""
    if int(i_pulsed_plant) == 1:
        return float(esbldgm3)
    return 0.0


# ------------------------------------------------- the stated values (§34)


def _stated_get(data, area, field, default):
    """`data.<area>.<field>`, or `default` where the state does not hold it."""
    try:
        return getattr(getattr(data, area), field)
    except (AttributeError, KeyError):
        return default


def _stated_eyoung_cond(data, which):
    """One of `resolve_eyoung_cond`'s pair -- `0` axial, `1` transverse."""
    return resolve_eyoung_cond(
        _stated_get(data, "tfcoil", "eyoung_cond_axial", 6.6e8),
        _stated_get(data, "tfcoil", "eyoung_cond_trans", 0.0),
        _stated_get(
            data, "tfcoil", "i_tf_cond_eyoung_axial", I_TF_COND_EYOUNG_AXIAL_DEFAULT
        ),
        _stated_get(
            data, "tfcoil", "i_tf_cond_eyoung_trans", I_TF_COND_EYOUNG_TRANS_DEFAULT
        ),
        _stated_get(data, "tfcoil", "i_tf_sc_mat", 1),
    )[which]


STATED_VALUES = {
    # `models/initialisation.py` -- `init.py`'s and `st_init`'s writes. The six
    # resolutions read their raw value and their switch back off the state and re-apply
    # `resolve_*`; every one of those is idempotent on its own answer, so the entry gives
    # the same number whether the state is PROCESS's (already resolved) or native's
    # (still raw). That is the property `test_stated.py` pins.
    "^stated.tfcoil.eff_tf_cryo": lambda d: resolve_eff_tf_cryo(
        _stated_get(d, "tfcoil", "eff_tf_cryo", EFF_TF_CRYO_UNSET),
        _stated_get(d, "tfcoil", "i_tf_sup", 1),
    ),
    "^stated.tfcoil.eyoung_ins": lambda d: resolve_eyoung_ins(
        _stated_get(d, "tfcoil", "eyoung_ins", EYOUNG_INS_UNSET),
        _stated_get(d, "tfcoil", "i_tf_sup", 1),
    ),
    "^stated.tfcoil.eyoung_cond_axial": lambda d: _stated_eyoung_cond(d, 0),
    "^stated.tfcoil.eyoung_cond_trans": lambda d: _stated_eyoung_cond(d, 1),
    "^stated.pf_coil.rho_pf_coil": lambda d: resolve_rho_pf_coil(
        _stated_get(d, "pf_coil", "rho_pf_coil", 2.5e-8),
        _stated_get(d, "pf_coil", "i_pf_conductor", I_PF_CONDUCTOR_DEFAULT),
    ),
    "^stated.physics.f_nd_beam_electron": lambda d: resolve_f_nd_beam_electron(
        _stated_get(d, "physics", "f_nd_beam_electron", 0.005),
        _stated_get(
            d, "current_drive", "i_hcd_calculations", I_HCD_CALCULATIONS_DEFAULT
        ),
        _stated_get(d, "current_drive", "i_hcd_primary", I_HCD_PRIMARY_DEFAULT),
    ),
    "^stated.buildings.esbldgm3": lambda d: resolve_esbldgm3(
        _stated_get(d, "buildings", "esbldgm3", 1.0e3),
        _stated_get(d, "pulse", "i_pulsed_plant", 0),
    ),
    # `st_init`'s literals. No state is consulted: the value is the *port's* claim about
    # what PROCESS's own source says, and reading it back off a `DataStructure` would
    # make a cold seed (`.build.dr_cs` at `build_variables.py`'s `0.811 m`) answer for a
    # machine that has no solenoid -- §28.7's measurement, one level out.
    "^stated.build.dr_cs": lambda d: 0.0,
    "^stated.build.dr_cs_tf_gap": lambda d: 0.0,
    "^stated.times.t_plant_pulse_coil_precharge": lambda d: 0.0,
    "^stated.times.t_plant_pulse_plasma_current_ramp_up": lambda d: 0.0,
    "^stated.times.t_plant_pulse_burn": lambda d: 3.15576e7,
    "^stated.times.t_plant_pulse_plasma_current_ramp_down": lambda d: 0.0,
    # The remaining literal carriers, each the arm on which PROCESS's own source is a
    # literal assignment (or nothing at all, the `DataStructure` default standing).
    "^stated.costs.c2253": lambda d: 0.0,
    "^stated.current_drive.f_c_plasma_diamagnetic": lambda d: 0.0,
    "^stated.current_drive.f_c_plasma_pfirsch_schluter": lambda d: 0.0,
    "^stated.current_drive.eta_cd_hcd_secondary": lambda d: 0.0,
    "^stated.current_drive.p_hcd_secondary_extra_heat_mw": lambda d: 0.0,
    "^stated.heat_transport.p_hcd_secondary_electric_mw": lambda d: 0.0,
    # `models/physics/plasma_profiles.LModeProfileReset` -- `init.py`'s neighbour on
    # the parabolic arm, seven fields PROCESS coerces inside the pipeline. Not in
    # §28.1's fourteen (it holds no field, so the array ban does not reach it) and the
    # same defect: seven literals from an input-less body are seven compile-time
    # constants. Ordered as the declaration's outputs are.
    "^stated.physics.radius_plasma_pedestal_temp_norm": (
        lambda d: lmode_profile_reset()[0]
    ),
    "^stated.physics.radius_plasma_pedestal_density_norm": (
        lambda d: lmode_profile_reset()[1]
    ),
    "^stated.physics.temp_plasma_pedestal_kev": lambda d: lmode_profile_reset()[2],
    "^stated.physics.temp_plasma_separatrix_kev": lambda d: lmode_profile_reset()[3],
    "^stated.physics.nd_plasma_pedestal_electron": lambda d: lmode_profile_reset()[4],
    "^stated.physics.nd_plasma_separatrix_electron": lambda d: lmode_profile_reset()[5],
    "^stated.physics.tbeta": lambda d: lmode_profile_reset()[6],
    # The two whose literal is a *ported unit*, called rather than restated -- the unit
    # is still the source of the number, as it was when it filled a `carried()` field's
    # `default_factory`.
    "^stated.tfcoil.f_a_tf_turn_cable_space_extra_void": (
        lambda d: croco_turn_cable_space_extra_void()
    ),
    "^stated.fwbs.pnuc_cp_tf": (lambda d: calculate_centrepost_neutronics_absent()[0]),
    "^stated.fwbs.p_cp_shield_nuclear_heat_mw": (
        lambda d: calculate_centrepost_neutronics_absent()[1]
    ),
    "^stated.fwbs.pnuc_cp": (lambda d: calculate_centrepost_neutronics_absent()[2]),
    "^stated.fwbs.neut_flux_cp": (lambda d: calculate_centrepost_neutronics_absent()[3]),
}
"""What each `models/stated.StatesValues` output is stated to be, by place."""


SEED_OWNED_FIELDS = (
    "eff_tf_cryo",
    "eyoung_ins",
    "eyoung_cond_axial",
    "eyoung_cond_trans",
    "rho_pf_coil",
    "f_nd_beam_electron",
    "esbldgm3",
    "dz_shld_upper",
    "dz_vv_upper",
    "dr_cs",
    "dr_cs_tf_gap",
    "t_plant_pulse_coil_precharge",
    "t_plant_pulse_plasma_current_ramp_up",
    "t_plant_pulse_burn",
    "t_plant_pulse_plasma_current_ramp_down",
)
"""Every field `models/initialisation` may own, as an `IN.DAT`/`ITERATION_VARIABLES`
name.
"""


def _refuse_seed_owned_unknowns(ixc, owned):
    """Refuse a machine whose `ixc` names a field one of `_initialisation`'s nodes owns.
    """
    frozen = {
        name
        for identifier in ixc
        if (variable := ITERATION_VARIABLES.get(int(identifier))) is not None
        for name in ((variable.target_name or variable.name),)
        if name in owned
    }
    if frozen:
        raise NotImplementedError(
            f"{sorted(frozen)} is an iteration variable on this run and is also "
            f"written by `process/core/init.py` or `st_init`, which "
            f"`models/initialisation` ports as a constant node resolved at assembly. A "
            f"constant that owns an unknown overwrites the optimiser's value on every "
            f"evaluation, so this machine is refused rather than assembled with one. "
            f"Resolving it means giving the field a real producer, not relaxing this "
            f"check"
        )


def _initialisation(imported, device, i_tf_sup, i_tf_sc_mat, ixc):
    """The seed's own writes, as occupied slots: `models/initialisation.Initialisation`.
    """
    tokamak = device is TokamakProcess
    i_pulsed_plant = imported.get("pulse", "i_pulsed_plant", I_PULSED_PLANT_DEFAULT)
    i_single_null = imported.get("physics", "i_single_null", I_SINGLE_NULL_DEFAULT)
    owned = {"eff_tf_cryo"}
    if int(i_pulsed_plant) != 1:
        owned.add("esbldgm3")
    if tokamak and int(i_single_null) == 0:
        owned.add("dz_shld_upper")
    if not tokamak:
        owned |= {
            "dr_cs",
            "dr_cs_tf_gap",
            "t_plant_pulse_coil_precharge",
            "t_plant_pulse_plasma_current_ramp_up",
            "t_plant_pulse_burn",
            "t_plant_pulse_plasma_current_ramp_down",
        }
    if tokamak:
        owned |= {
            "eyoung_ins",
            "eyoung_cond_axial",
            "eyoung_cond_trans",
            "rho_pf_coil",
            "f_nd_beam_electron",
        }
    _refuse_seed_owned_unknowns(ixc, owned & set(SEED_OWNED_FIELDS))
    # **No occupant takes a value any more.** Which slots are filled is still decided
    # here, from the file's switches; *what* each one writes is `STATED_VALUES`', read
    # from the env at `^stated.<the place>`. A resolved number held on a declaration is
    # an array `cottax` refuses in a graph, and a Python one is a constant XLA folds and
    # `filter_jit` keys on -- `models/stated.py`, `_audit/optimise_design.md` §34.
    return Initialisation(
        tf_cryoplant_efficiency=TfCryoplantEfficiency(),
        tf_insulation_youngs_modulus=TfInsulationYoungsModulus() if tokamak else None,
        tf_conductor_youngs_modulus=TfConductorYoungsModulus() if tokamak else None,
        pf_coil_resistivity=PfCoilResistivity() if tokamak else None,
        beam_electron_density_fraction=BeamElectronDensityFraction()
        if tokamak
        else None,
        energy_storage_building_volume=EnergyStorageBuildingVolume()
        if int(i_pulsed_plant) != 1
        else None,
        double_null_upper_build=DoubleNullUpperBuild()
        if tokamak and int(i_single_null) == 0
        else None,
        # `st_init` returns at its first line on `istell == 0`, so both of its slots are
        # the stellarator arm of one dispatch and neither is a kwarg on a shared node.
        stellarator_solenoid_absent=None if tokamak else StellaratorSolenoidAbsent(),
        stellarator_pulse_times=None if tokamak else StellaratorPulseTimes(),
    )


# ------------------------------------------------------------------ presence (§24.2)


def presence_flags_from_indat(input_file):
    """`init.py:925-930`'s two presence flags, as `{name: bool}`."""
    imported = _as_imported(input_file)
    return {
        # `tfcoil_variables.py:86` -- `dx_tf_side_case_min` defaults to `0.0`.
        "tfc_sidewall_is_fraction": not imported.named("dx_tf_side_case_min"),
        # `tfcoil_variables.py:77` -- `dr_tf_plasma_case` defaults to `0.0`.
        "i_f_dr_tf_plasma_case": not imported.named("dr_tf_plasma_case"),
    }


# ------------------------------------------------------------------ switch values


SWITCH_VALUE_DEFAULTS = {
    "bkt_life_csf": 0.0,  # `cs_fatigue_variables.py:31` -- a float, read as an int
    "i_beta_component": 0,  # `physics_variables.py:835`
    "i_cp_lifetime": 0,  # `cost_variables.py:334`
    "i_density_limit": 8,  # `physics_variables.py:863`
    "i_plant_availability": 2,  # `cost_variables.py:408`
    "i_plasma_ignited": 0,  # `physics_variables.py:881`
    "i_q95_fixed": 0,  # `constraint_variables.py:52`
    "i_rad_loss": 1,  # `physics_variables.py:954`
    "i_tf_bucking": -1,  # `tfcoil_variables.py:308` -- a **sentinel**, resolved below
    "i_tf_inside_cs": 0,  # `build_variables.py:189`
    "i_tf_sup": 1,  # `tfcoil_variables.py:261`
    "ibkt_life": 0,  # `cost_variables.py:416`
    "ireactor": 1,  # `cost_variables.py:521`
    "istell": 0,  # `stellarator_variables.py:46`
    "itart": 0,  # `physics_variables.py:994`
}
"""`sand.SWITCH_PARAMETER_NAMES` -> PROCESS's own `DataStructure` default."""


def switch_values_from_indat(input_file):
    """`sand`'s static switch arguments for one run, read from the **file**."""
    imported = _as_imported(input_file)
    values = {
        name: int(imported.get(INPUT_VARIABLES[name].module, name, default))
        for name, default in SWITCH_VALUE_DEFAULTS.items()
    }
    values["i_tf_bucking"] = resolve_i_tf_bucking(
        values["i_tf_bucking"], values["i_tf_sup"]
    )
    return values


_QUENCH_GRID_FIELDS = ("tftmp", "temp_tf_conductor_quench_max")
"""The two `.tfcoil` inputs the quench quadrature grid -- and therefore the helium
property table -- is a function of.
"""


def _quench_helium_table(numbers, ixc):
    """`(temp_he_peak, temp_quench_max, den_helium, cp_helium)` for this machine."""
    frozen = {
        name
        for identifier in ixc
        if (variable := ITERATION_VARIABLES.get(int(identifier))) is not None
        for name in ((variable.target_name or variable.name),)
        if name in _QUENCH_GRID_FIELDS
    }
    if frozen:
        raise NotImplementedError(
            f"{sorted(frozen)} is an iteration variable on this run, and "
            f"`TfCoilQuenchHeatCurrentDensity` carries the helium property table as a "
            f"static field evaluated at the quadrature grid those two temperatures "
            f"define. An unknown there moves the states CoolProp was asked about while "
            f"the table stays put -- so this machine is refused rather than assembled "
            f"with a stale table. Resolving it means giving the helium properties a "
            f"producer (`quench.md` OQ1's option (b) or (c)), not relaxing this check"
        )
    temp_he_peak = float(numbers.get("tftmp", 4.75))
    temp_quench_max = float(numbers.get("temp_tf_conductor_quench_max", 150.0))
    den_helium, cp_helium = helium_properties_at_quench_nodes(
        temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
    )
    return temp_he_peak, temp_quench_max, den_helium, cp_helium


def _tokamak_device(
    switches,
    numbers,
    ixc,
    int_lists,
    presence,
    i_tf_sup,
    i_plasma_ignited,
    itart,
    i_tf_sc_mat,
    i_tf_turn_type,
):
    """The `Tokamak` an IN.DAT describes -- twenty-six slots of the twenty-eight filled.
    """
    i_single_null = switches.get("i_single_null", 1)  # `physics_variables.py:1366`
    n_divertors = _n_divertors(i_single_null)
    # One predicate, four slots -- blanket areas, blanket volumes, first wall and vacuum
    # vessel -- resolved once. Three audit records reached it independently
    # (`blanket_library.md`, `fw.md`, `vacuum.md`) and agreed, which is what makes
    # writing it once safe rather than merely tidy.
    shape_arm = _fw_blkt_vv_shape_arm(itart, switches.get("i_fw_blkt_vv_shape", 2))
    # `i_tf_shape` and `i_tf_wp_geom` are *resolved*, not read: PROCESS's own `init.py`
    # replaces the auto-select value of each before any model runs, so the raw file
    # value names no arm. See `_tf_shape` / `_tf_wp_geom`.
    i_tf_shape = _tf_shape(switches.get("i_tf_shape", 0), itart)
    # `i_tf_turns_integer` is answered once, here, because it decides **two** slots:
    # the WP geometry resolution below and the turn-geometry occupant. The
    # `low_aspect_ratio_DEMO` mis-assembly (2026-08-27) happened precisely because it
    # reached only the first -- the survey reported "the factory dispatches on it"
    # while `cicc_turn_geometry` silently kept the averaged occupant.
    i_tf_turns_integer = switches.get("i_tf_turns_integer", 0)  # `tfcoil_variables.py`
    i_tf_wp_geom = _tf_wp_geom(switches.get("i_tf_wp_geom", -1), i_tf_turns_integer)
    i_tf_case_geom = TFPlasmaCaseType(switches.get("i_tf_case_geom", 0))
    # `i_str_wp` decides two slots -- the critical-current surface and the temperature
    # margin -- and both read the strain from the field it names, so it is answered once
    # here for the same reason `i_tf_turns_integer` is.
    i_str_wp = switches.get("i_str_wp", 1)  # `tfcoil_variables.py:508`
    (
        quench_temp_he_peak,
        quench_temp_max,
        quench_den_helium,
        quench_cp_helium,
    ) = _quench_helium_table(numbers, ixc)
    i_plasma_current = switches.get("i_plasma_current", 4)  # `physics_variables.py:843`
    i_hcd_primary = switches.get("i_hcd_primary", 5)  # `current_drive_variables.py:190`
    i_hcd_secondary = switches.get("i_hcd_secondary", 0)  # `:206`
    i_pf_conductor = switches.get("i_pf_conductor", 0)  # `pfcoil_variables.py:230`

    def pick(field, registry, default, **kw):
        return _slot_occupant(field, switches.get(field, default), registry, **kw)

    plasma_geom = TokamakPlasmaGeom(
        shape=_slot_occupant(
            "i_plasma_geometry",
            PlasmaGeometryModelType(switches.get("i_plasma_geometry", 0)),
            PLASMA_SHAPE,
        ),
        geometry=_slot_occupant(
            "plasma_geometry_arm",
            _plasma_geometry_arm(i_plasma_current, switches.get("i_plasma_shape", 0)),
            PLASMA_GEOMETRY,
        ),
    )
    plasma_fields = PlasmaFields(
        surface_averaged_poloidal_field=_slot_occupant(
            "surface_poloidal_field_arm",
            _surface_poloidal_field_arm(i_plasma_current),
            SURFACE_POLOIDAL_FIELD,
        )
    )
    physics = TokamakPhysics(
        separatrix_power=_slot_occupant(
            "i_plasma_ignited_separatrix",
            PlasmaIgnitionModel(int(i_plasma_ignited)),
            SEPARATRIX_POWER,
        )
    )
    # `Pulse.run` gates the burn-time calculation on `i_pulsed_plant == 1`
    # (`pulse.py:154-162`), so on a steady-state machine PROCESS never computes it and
    # `.times.t_plant_pulse_burn` stays the file's own input. An empty slot is how that
    # is said here -- see `TokamakPulse.burn_time` for the 13 rows it was costing on the
    # two `i_pulsed_plant = 0` files.
    is_pulsed = (
        PlantOperationModel(int(switches.get("i_pulsed_plant", 0)))
        is not PlantOperationModel.CONTINUOUS
    )
    pulse = TokamakPulse(
        ramp_times=_slot_occupant(
            "pulse_ramp_times_arm",
            _pulse_ramp_times_arm(
                switches.get("i_pulsed_plant", 0),  # `pulse_variables.py:30`
                switches.get("pulsetimings", 1),  # `times_variables.py:12`
                switches.get("i_t_current_ramp_up", 0),  # `:44`
            ),
            PULSE_RAMP_TIMES,
        ),
        burn_time=PulseBurnTime() if is_pulsed else None,
    )
    current_drive = _slot_occupant(
        "i_hcd_calculations",
        switches.get("i_hcd_calculations", 1),  # `current_drive_variables.py:223`
        HCD_CALCULATIONS,
        build=lambda cls: cls(
            primary_efficiency=_hcd_primary_efficiency(
                i_hcd_primary,
                switches.get("i_ecrh_wave_mode", 0),  # `current_drive_variables.py:116`
            ),
            secondary_heating=_slot_occupant(
                "i_hcd_secondary",
                CurrentDriveModel(i_hcd_secondary),
                HCD_SECONDARY_HEATING,
            ),
            primary_powers=_slot_occupant(
                "hcd_primary_powers_arm",
                _hcd_primary_powers_arm(i_hcd_primary, i_hcd_secondary),
                HCD_PRIMARY_POWERS,
            ),
            electric_total=_slot_occupant(
                "i_plasma_ignited",
                PlasmaIgnitionModel(int(i_plasma_ignited)),
                HCD_ELECTRIC_TOTAL,
            ),
        ),
    )
    build = Build(
        divertor_geometry=_slot_occupant(
            "divertor_geometry_arm",
            _divertor_geometry_arm(
                itart,
                # `build_variables.py:326` -- and read as a *number*, because what this
                # arm turns on is the entering value and not a switch.
                numbers.get("dz_xpoint_divertor", 0.0),
            ),
            DIVERTOR_GEOMETRY,
            # `None` is an occupant on arm -3, not a refusal: a spherical tokamak whose
            # input sets `dz_xpoint_divertor` discards `divgeom`'s early return at
            # `build.py:800` and owns nothing.
            build=lambda cls: None if cls is None else cls(),
        ),
        tf_top_height=_slot_occupant(
            "i_single_null",
            DivertorNumberModels(int(i_single_null)),
            TF_TOP_HEIGHT,
        ),
        dr_tf_inboard_winding_pack=_slot_occupant(
            "dr_tf_inboard_winding_pack",
            0 if 140 in ixc else 1,
            DR_TF_INBOARD_WINDING_PACK,
        ),
        r_cp_top=_slot_occupant(
            "r_cp_top_arm",
            _r_cp_top_arm(
                itart,
                i_tf_sup,
            ),
            R_CP_TOP,
        ),
        tf_inboard_radii=_slot_occupant(
            "tf_inboard_radii_arm",
            _tf_inboard_radii_arm(
                switches.get("i_tf_inside_cs", 0),  # `build_variables.py:189`
                switches.get("i_cs_precomp", 1),  # `build_variables.py:183`
            ),
            TF_INBOARD_RADII,
        ),
        vacuum_vessel_and_shield_radii=_slot_occupant(
            "i_tf_inside_cs_vacuum_shield",
            TFCSRadialConfiguration(
                int(switches.get("i_tf_inside_cs", 0))  # `build_variables.py:189`
            ),
            VACUUM_SHIELD_RADII,
        ),
        dr_tf_outboard=_slot_occupant("i_tf_sup_build", i_tf_sup, DR_TF_OUTBOARD),
        wp_conductor_max_width=_slot_occupant(
            "i_tf_sup_build", i_tf_sup, WP_CONDUCTOR_MAX_WIDTH
        ),
        tf_outboard_mid=_slot_occupant("i_tf_shape_build", i_tf_shape, TF_OUTBOARD_MID),
        tf_outboard_edge_ripple=_slot_occupant(
            "i_tf_shape_build", i_tf_shape, TF_OUTBOARD_EDGE_RIPPLE
        ),
    )
    # ---- the TF coil, one namespace or the other ----------------------------------
    #
    # `i_tf_turn_type` selects a whole PROCESS `Model` class at `caller.py:298-313`, and
    # the two classes share `run_base_superconducting_tf` and everything after the turn.
    # So the twenty-two slots they share are built **once**, here, and handed to whichever
    # subclass the switch names -- which is what `models/tfcoil/namespace.py`'s
    # `SuperconductingTfCoil` base exists to make possible. Building them twice, once in
    # each branch, would be exactly the transcription `model_tree_design.md` §8 step 4d
    # removed from the tree, thirteen slots at a time.
    shared_tf_coil_slots = dict(
        tf_global_geometry=_slot_occupant(
            "i_tf_case_geom", i_tf_case_geom, TF_GLOBAL_GEOMETRY
        ),
        # Neither flag is a declared PROCESS input, so neither is in `switches` and
        # neither ever could be: `init.py:925-930` sets each from whether the file
        # *named* the partner field. `presence_flags_from_indat` asks `Imported.named`,
        # which is the only question that can answer it (§24.2 item 1).
        dr_tf_plasma_case=_slot_occupant(
            "i_f_dr_tf_plasma_case",
            presence["i_f_dr_tf_plasma_case"],
            DR_TF_PLASMA_CASE,
        ),
        # `None` is an occupant here, not a refusal: at `False` -- the file named
        # `dx_tf_side_case_min` -- PROCESS computes no `.tfcoil.dx_tf_side_case_min` at
        # all and the field is an input.
        dx_tf_side_case_min=_slot_occupant(
            "tfc_sidewall_is_fraction",
            presence["tfc_sidewall_is_fraction"],
            DX_TF_SIDE_CASE_MIN,
            build=lambda cls: None if cls is None else cls(),
        ),
        tf_coil_shape=_slot_occupant(
            "tf_coil_shape_arm",
            _tf_coil_shape_arm(i_tf_shape, itart, i_single_null),
            TF_COIL_SHAPE,
        ),
        tf_coil_self_inductance=_slot_occupant(
            "tf_self_inductance_arm",
            _tf_self_inductance_arm(i_tf_shape, itart),
            TF_COIL_SELF_INDUCTANCE,
        ),
        superconducting_tf_wp_geometry=_slot_occupant(
            "i_tf_wp_geom", i_tf_wp_geom, SC_TF_WP_GEOMETRY
        ),
        tf_case_areas=_slot_occupant("i_tf_case_geom", i_tf_case_geom, TF_CASE_AREAS),
        dx_tf_side_case=_slot_occupant("i_tf_wp_geom", i_tf_wp_geom, DX_TF_SIDE_CASE),
        peak_b_tf_inboard_with_ripple=_slot_occupant(
            "peak_b_ripple_arm",
            # `tfcoil_variables.py:625` -- a float field, and the fit is selected by its
            # rounded value, which is why this is `numbers` and not `switches`.
            _peak_b_ripple_arm(numbers.get("n_tf_coils", 16.0)),
            PEAK_B_TF_RIPPLE,
        ),
        # Two switches, one slot -- see `SC_TF_MASSES`. `itart` and `i_tf_sc_mat` are
        # both *threaded*, resolved once in `machine_from_indat`; `i_tf_sc_mat` is the
        # same local the stellarator branch gives `WINDING_PACK_MATERIAL` and
        # `COILS_MASS_MATERIAL`, so no two consumers of that switch can disagree.
        superconducting_tf_coil_areas_and_masses=_slot_occupant(
            "itart_i_tf_sc_mat_sc_tf_masses",
            (SphericalTokamakModel(int(itart)), i_tf_sc_mat),
            SC_TF_MASSES,
        ),
        tf_field_and_force=_slot_occupant(
            "tf_field_and_force_arm",
            _tf_field_and_force_arm(
                itart,
                switches.get("i_cp_joints", -1),  # `tfcoil_variables.py:589`
            ),
            TF_FIELD_AND_FORCE,
        ),
        tf_stress=_slot_occupant(
            "tf_stress_arm",
            _tf_stress_arm(
                switches.get("i_tf_stress_model", 1),  # `tfcoil_variables.py:211`
                # The `-1` sentinel is `init.py:891-895`'s to resolve, and it needs the
                # conductor to do it -- see `resolve_i_tf_bucking`.
                resolve_i_tf_bucking(
                    switches.get("i_tf_bucking", -1),  # `tfcoil_variables.py:308`
                    i_tf_sup,
                ),
                i_tf_turns_integer,  # resolved above, beside `i_tf_wp_geom`
            ),
            TF_STRESS,
        ),
        tf_coil_quench_heat_current_density=TfCoilQuenchHeatCurrentDensity(
            tftmp=quench_temp_he_peak,
            temp_tf_conductor_quench_max=quench_temp_max,
            den_helium_at_nodes=quench_den_helium,
            cp_helium_at_nodes=quench_cp_helium,
        ),
    )
    # The turn-dimension flags decide the turn geometry on **both** turn types, from the
    # same three-way branch (`_croco_turn_geometry_arm` delegates), so they are read once
    # here rather than inside either branch below.
    turn_geometry_arm_inputs = (
        i_tf_turns_integer,  # resolved above, beside `i_tf_wp_geom`
        switches.get("i_dx_tf_turn_general_input", 0),  # `tfcoil_variables.py:108`
        switches.get("i_dx_tf_turn_cable_space_general_input", 0),  # `:127`
    )
    # The critical-current and temperature-margin slots both key on
    # `(i_str_wp, i_tf_sc_mat)`. `i_str_wp` is `tfcoil_variables.py:508`'s default; the
    # pair is built once so no two of the four registries below can disagree about
    # either switch, which is the same cross-slot coherence `i_tf_sc_mat` already gets
    # from being resolved above the device branch.
    superconductor_arm = (i_str_wp, i_tf_sc_mat)
    if i_tf_turn_type is SuperconductingTFTurnType.CROSS_CONDUCTOR:
        tf_coil = CrocoSuperconductingTfCoil(
            **shared_tf_coil_slots,
            croco_turn_geometry=_slot_occupant(
                "croco_turn_geometry_arm",
                _croco_turn_geometry_arm(*turn_geometry_arm_inputs),
                CROCO_TURN_GEOMETRY,
            ),
            croco_superconductor_properties=_slot_occupant(
                "i_str_wp_i_tf_sc_mat_croco_sc_properties",
                superconductor_arm,
                CROCO_SUPERCONDUCTOR_PROPERTIES,
            ),
            tf_superconductor_temperature_margin=_slot_occupant(
                "i_str_wp_i_tf_sc_mat_croco_temp_margin",
                superconductor_arm,
                CROCO_TEMPERATURE_MARGIN,
            ),
        )
    else:
        tf_coil = CiccSuperconductingTfCoil(
            **shared_tf_coil_slots,
            cicc_turn_geometry=_slot_occupant(
                "cicc_turn_geometry_arm",
                _cicc_turn_geometry_arm(*turn_geometry_arm_inputs),
                CICC_TURN_GEOMETRY,
            ),
            cicc_superconductor_properties=_slot_occupant(
                "i_str_wp_i_tf_sc_mat_cicc_sc_properties",
                superconductor_arm,
                CICC_SUPERCONDUCTOR_PROPERTIES,
            ),
            tf_superconductor_temperature_margin=_slot_occupant(
                "i_str_wp_i_tf_sc_mat_temp_margin",
                superconductor_arm,
                TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN,
            ),
        )
    ccfe_hcpb = _slot_occupant(
        "i_blanket_type",
        BlktModelTypes(switches.get("i_blanket_type", 1)),  # `fwbs_variables.py:70`
        BLANKET_MODEL,
        build=lambda cls: cls(
            blanket_half_height=_slot_occupant(
                "n_divertors", n_divertors, BLANKET_HALF_HEIGHT
            ),
            blanket_areas=_slot_occupant(
                "fw_blkt_vv_shape_arm", shape_arm, BLANKET_AREAS
            ),
            blanket_volumes=_slot_occupant(
                "fw_blkt_vv_shape_arm", shape_arm, BLANKET_VOLUMES
            ),
            blanket_coverage_factors=_slot_occupant(
                "n_divertors", n_divertors, BLANKET_COVERAGE_FACTORS
            ),
            divertor_surface_and_plate_mass=_slot_occupant(
                "n_divertors", n_divertors, DIVERTOR_SURFACE_MASS
            ),
            nuclear_heating_magnets=_slot_occupant(
                "itart_hcpb", SphericalTokamakModel(int(itart)), NUCLEAR_HEATING_MAGNETS
            ),
            nuclear_heating_shield=_slot_occupant(
                "itart_hcpb", SphericalTokamakModel(int(itart)), NUCLEAR_HEATING_SHIELD
            ),
            centrepost_neutronics=_slot_occupant(
                "centrepost_neutronics_arm",
                _centrepost_neutronics_arm(itart, i_tf_sup),
                CENTREPOST_NEUTRONICS,
            ),
            nuclear_heating_renormalisation=_slot_occupant(
                "nuclear_heating_renormalisation_arm",
                _nuclear_heating_renormalisation_arm(n_divertors, itart),
                NUCLEAR_HEATING_RENORMALISATION,
            ),
            pumping_power=pick(
                "i_p_coolant_pumping",
                PUMPING_POWER,
                2,  # `fwbs_variables.py:249`
                build=lambda occupant: occupant(),
            ),
        ),
    )

    def none_or_call(cls):
        # `None` is an occupant, not a refusal, in the four registries that carry it
        # (`CURRENT_PROFILE_INDEX` and friends): PROCESS's arm computes nothing and the
        # field is a run input. Same shape as `DX_TF_SIDE_CASE_MIN` above.
        return None if cls is None else cls()

    plasma_current = TokamakPlasmaCurrent(
        plasma_current=_slot_occupant(
            "i_plasma_current",
            PlasmaCurrentModel(int(i_plasma_current)),
            PLASMA_CURRENT_SCALING,
        ),
        current_profile_index=_slot_occupant(
            "i_alphaj",
            CurrentProfileIndexModel(switches.get("i_alphaj", 0)),  # `:951`
            CURRENT_PROFILE_INDEX,
            build=none_or_call,
        ),
    )
    plasma_inductance = TokamakPlasmaInductance(
        internal_inductance_norm=_slot_occupant(
            "i_ind_plasma_internal_norm",
            # `physics_variables.py:948`
            IndInternalNormModel(switches.get("i_ind_plasma_internal_norm", 0)),
            IND_PLASMA_INTERNAL_NORM,
            build=none_or_call,
        )
    )
    bootstrap_current = _slot_occupant(
        "i_bootstrap_current",
        BootstrapCurrentFractionModel(switches.get("i_bootstrap_current", 3)),  # `:818`
        BOOTSTRAP_CURRENT,
        # The profile grid's shape, the same `201` the `ProfileGrid` registration in
        # `models/physics/namespace.py` carries -- a resolution, not a switch, and
        # `switch_audit` value-checks it against `.physics.n_plasma_profile_elements`.
        build=lambda cls: None if cls is None else cls(n_plasma_profile_elements=201),
    )
    scrape_off_layer = TokamakScrapeOffLayer(
        outboard_power_decay_length=_slot_occupant(
            "i_len_sol_outboard_power_decay",
            OutbordSOLPowerDecayLengthModel(
                switches.get("i_len_sol_outboard_power_decay", 1)  # `:1718`
            ),
            SOL_OUTBOARD_POWER_DECAY,
            build=none_or_call,
        )
    )
    density_limit = TokamakDensityLimit(
        enforced_density_limit=_slot_occupant(
            "i_density_limit",
            DensityLimitModel(switches.get("i_density_limit", 8)),  # `:863`
            DENSITY_LIMIT_ENFORCED,
        )
    )
    # One predicate, thirteen slots, resolved once -- see `_pf_coil_system_arm`.
    pf_coil_switches = {
        "iohcl": switches.get("iohcl", 1),  # `build_variables.py:177`
        "n_pf_coil_groups": switches.get(
            "n_pf_coil_groups", 3
        ),  # `pfcoil_variables.py:320`
        "i_pf_location": int_lists.get("i_pf_location", (2, 2, 3, 0)),  # `:220`
        "n_pf_coils_in_group": int_lists.get(
            "n_pf_coils_in_group", (1, 1, 2, 0)
        ),  # `:310`
        "itart": itart,
        "itartpf": switches.get("itartpf", 0),  # `physics_variables.py:1000`
        "i_pf_current": switches.get("i_pf_current", 1),  # `pfcoil_variables.py:279`
        "i_pf_conductor": i_pf_conductor,
        "i_pf_superconductor": switches.get("i_pf_superconductor", 1),  # `:254`
        "i_cs_superconductor": switches.get("i_cs_superconductor", 1),  # `:239`
        "i_tf_shape": i_tf_shape,
        "i_r_pf_outside_tf_placement": switches.get(
            "i_r_pf_outside_tf_placement", 0
        ),  # `:287`
    }
    pf_coil_arm = _pf_coil_system_arm(**pf_coil_switches)
    # The refusal names **every** deviating dimension, not just the first -- see
    # `_pf_coil_system_deviations`. Raised here rather than left to `_slot_occupant`
    # below, which can only ever see one arm index and so can only ever say one thing.
    if pf_coil_arm < 0:
        _refuse_pf_coil_system(_pf_coil_system_deviations(**pf_coil_switches))
    shield = TokamakShield(
        half_height=_slot_occupant("n_divertors", n_divertors, SHIELD_HALF_HEIGHT),
        volumes=_slot_occupant("fw_blkt_vv_shape_arm", shape_arm, SHIELD_VOLUMES),
    )

    return Tokamak(
        plasma_geom=plasma_geom,
        physics=physics,
        plasma_beta=TokamakPlasmaBeta(
            norm_max=_slot_occupant(
                "i_beta_norm_max",
                switches.get("i_beta_norm_max", 1),  # `physics_variables.py`
                BETA_NORM_MAX,
                build=none_or_call,
            )
        ),
        plasma_inductance=plasma_inductance,
        plasma_current=plasma_current,
        bootstrap_current=bootstrap_current,
        diamagnetic_current=_slot_occupant(
            "i_diamagnetic_current",
            PlasmaDiamagneticCurrentModel(
                switches.get("i_diamagnetic_current", 0)  # `physics_variables.py:856`
            ),
            DIAMAGNETIC_CURRENT,
        ),
        pfirsch_schluter_current=_slot_occupant(
            "i_pfirsch_schluter_current",
            switches.get("i_pfirsch_schluter_current", 0),  # `:895`; no PROCESS enum
            PFIRSCH_SCHLUTER_CURRENT,
        ),
        l_h_transition=_slot_occupant(
            "i_l_h_threshold",
            PlasmaConfinementTransitionModel(
                switches.get("i_l_h_threshold", 19)  # `physics_variables.py:1234`
            ),
            L_H_THRESHOLD,
        ),
        scrape_off_layer=scrape_off_layer,
        density_limit=density_limit,
        plasma_fields=plasma_fields,
        current_drive=current_drive,
        pulse=pulse,
        build=build,
        cicc_superconducting_tf_coil=tf_coil,
        pf_coil=_slot_occupant("pf_coil_system_arm", pf_coil_arm, PF_COIL),
        cs_coil=_slot_occupant(
            "pf_coil_system_arm",
            pf_coil_arm,
            CS_COIL,
            # `None` on arm 2: this machine has no central solenoid, so the slot is
            # empty rather than filled with nodes computing zeros. See `CS_COIL`.
            build=lambda cls: (
                None
                if cls is None
                else cls(
                    critical_current=_slot_occupant(
                        "i_cs_superconductor",
                        SuperconductorModel(
                            int(
                                switches.get("i_cs_superconductor", 1)
                            )  # `pfcoil_vars:225`
                        ),
                        CS_SUPERCONDUCTOR,
                    ),
                    temperature_margin=_slot_occupant(
                        "i_cs_superconductor",
                        SuperconductorModel(
                            int(
                                switches.get("i_cs_superconductor", 1)
                            )  # `pfcoil_vars:225`
                        ),
                        CS_TEMPERATURE_MARGIN,
                    ),
                )
            ),
        ),
        shield=shield,
        divertor=Divertor(
            heat_load=_slot_occupant(
                "divertor_heat_load_arm",
                _divertor_heat_load_arm(
                    switches.get("i_div_heat_load", 2),  # `divertor_variables.py:63`
                    n_divertors,
                ),
                DIVERTOR_HEAT_LOAD,
            )
        ),
        first_wall=_slot_occupant(
            "first_wall_arm",
            _first_wall_arm(
                n_divertors,
                shape_arm,
                switches.get("i_pflux_fw_neutron", 1),  # `physics_variables.py:1006`
            ),
            FIRST_WALL,
        ),
        vacuum_vessel=_slot_occupant(
            "vacuum_vessel_arm",
            _vacuum_vessel_arm(n_divertors, shape_arm),
            VACUUM_VESSEL,
        ),
        ccfe_hcpb=ccfe_hcpb,
        structure=_slot_occupant(
            "structure_arm",
            _structure_arm(i_tf_sup, i_pf_conductor),  # threaded, answered once
            STRUCTURE,
        ),
    )


def machine_from_indat(input_file, stella_conf=None):
    """The `StellaratorProcess` an IN.DAT describes -- the only thing that builds one.
    """
    # One read of the file, four views of it: `importer.read_indat` is the parser now,
    # and the readers below take the `Imported` it returns rather than the path, so a
    # machine costs one parse instead of four scans.
    imported = read_indat(input_file)
    switches = switches_from_indat(imported)

    def pick(field, registry, default, **kw):
        return _slot_occupant(field, switches.get(field, default), registry, **kw)

    # The device is resolved first, on its own, because it decides which *class* is being
    # built and therefore which of the resolutions below are even asked. PROCESS's own
    # default is `istell = 0`, a tokamak, and that is now a device rather than a refusal:
    # a file that never mentions `istell` builds a `TokamakProcess`. A value PROCESS
    # has never had still fails here, first, naming `istell` rather than whichever slot
    # the constructor happened to evaluate first.
    istell = switches.get("istell", 0)
    device = _slot_occupant("istell", istell, DEVICE, build=lambda cls: cls)
    # Confinement is three slots now, not one, and every switch it used to carry as a
    # static kwarg is answered here instead -- see `PhysicsConfinementTime`. The values
    # are read from the file with PROCESS's own defaults as fallbacks, where the tree
    # previously hardcoded them; `i_plasma_ignited` in particular was a registration bug
    # once (`0` against the input file's `1`, the residual 1.2 % on
    # `t_energy_confinement`) and a value read from the file cannot drift that way.
    i_confinement_time = switches.get("i_confinement_time", 34)
    i_rad_loss = switches.get("i_rad_loss", 1)
    i_plasma_ignited = switches.get("i_plasma_ignited", 0)
    confinement_scaling = _slot_occupant(
        "i_confinement_time",
        ConfinementTimeModel(int(i_confinement_time)),
        CONFINEMENT_SCALING,
    )
    confinement_tail = _slot_occupant(
        "i_rad_loss",
        ConfinementRadiationLossModel(int(i_rad_loss)),
        CONFINEMENT_TAIL,
    )
    plasma_power_loss = _slot_occupant(
        "i_plasma_ignited_i_rad_loss",
        _plasma_power_loss_arm(i_plasma_ignited, i_rad_loss),
        PLASMA_POWER_LOSS,
    )
    # `i_tf_sup` and `ipowerflow` each decide a slot *and* were each transcribed onto
    # nodes that branch on them internally -- five sites for the first, two for the
    # second -- so a machine could be resistive at `power.tf_power` and superconducting
    # at the five, or pre-2014 at `fw_area` and comprehensive-2014 at the two.
    # Resolved into a local once, here, and threaded below; the nodes lost their
    # constructor kwarg (`switch_kwarg_survey.md` §4.1/§4.3, band (a) items 1 and 3).
    #
    # The slot is resolved *before* the value is threaded, deliberately: `i_tf_sup == 2`
    # is an `UNPORTED` refusal, so no unported value ever reaches an occupant's field.
    i_tf_sup = switches.get("i_tf_sup", 1)
    tf_power = _slot_occupant("i_tf_sup", i_tf_sup, TF_POWER)
    i_tf_sup = TFConductorModel(i_tf_sup)
    # `ife` decides no slot of its own: it is a *device*, and the seven Account-22x
    # cost nodes that branch on it have no inertial-confinement arm at all. Answered
    # once, here, before anything is built -- `_audit/next_steps.md` §14.2. Its default
    # is `ife_variables.py:253`.
    ife = IFEModel(int(switches.get("ife", 0)))
    if ife is not IFEModel.MAGNETIC_CONFINEMENT:
        _refuse_unported_switch("ife", ife)
    # `i_tf_turn_type` decides no slot either -- it selects the whole TF `Model` class
    # (`CICCSuperconductingTFCoil` vs `CROCOSuperconductingTFCoil`) at
    # `core/caller.py:298-313`, above every model, exactly as `ife` does one level up.
    # **Both are ported since 2026-08-30**, so it is resolved and *threaded* rather than
    # refused: `_tokamak_device` hands the shared slots to whichever
    # `SuperconductingTfCoil` subclass it names. Read only on the superconducting arm,
    # because that is the only branch of `caller.py` that reads it -- a copper or
    # aluminium machine's `i_tf_turn_type` decides nothing, in PROCESS or here, and the
    # cable-in-conduit default keeps that arm exactly where it was.
    # `superconducting_tf_coil_variables.py:194` is the default.
    i_tf_turn_type = SuperconductingTFTurnType.CABLE_IN_CONDUIT
    if i_tf_sup is TFConductorModel.SUPERCONDUCTING:
        i_tf_turn_type = SuperconductingTFTurnType(
            int(switches.get("i_tf_turn_type", 1))
        )
    # `itart` decides four slots that used to hardcode it and, on a tokamak, ten more
    # inside `_tokamak_device`. Read here, above the device branch, and threaded --
    # `physics_variables.py:994` is the default.
    itart = SphericalTokamakModel(int(switches.get("itart", 0)))
    # The superconductor. Read **above** the device branch since 2026-08-27, because it
    # now decides slots on both arms: `winding_pack_intersect_inputs` and `coils_mass` on
    # a stellarator, `superconducting_tf_coil_areas_and_masses` on a tokamak. One local,
    # threaded to all three, is the only thing that makes them agree by construction --
    # and the tokamak slot got here by *not* agreeing: it answered this switch with a
    # module constant `dcond[0]` for every value (`_audit/units/models/tfcoil/
    # superconducting.md`, 2026-08-27, and `_audit/next_steps.md` §14.11 for the same
    # failure one file over). `tfcoil_variables.py:246` is the default.
    #
    # No slot is resolved here: the two stellarator slots refuse value 9 and the tokamak
    # slot accepts it, so each resolves against its own registry below.
    i_tf_sc_mat = SuperconductorModel(int(switches.get("i_tf_sc_mat", 1)))
    # `ireactor` decides two slots, not one: which electric-production occupant runs,
    # and -- jointly with `ipnet` and `itart` -- whether `costs.cost_of_electricity`
    # exists at all and which centrepost treatment it uses. `cost_variables.py:521`/
    # `:515` for the first two defaults.
    ireactor = switches.get("ireactor", 1)
    # `supercond_cost_model` decides two slots since 2026-08-30 -- Account 222.1's and
    # Account 222.2's -- so it is read once here and threaded, rather than read twice
    # inside the constructor. `cost_variables.py:552` is the default.
    supercond_cost_model = SuperconductorCostModel(
        int(switches.get("supercond_cost_model", 0))
    )
    # `pfcoil_variables.py:230` -- the PF conductor decides, jointly with `i_tf_sup`,
    # whether the cryoplant runs at all, and it is `PfMagnetCost`'s one remaining static
    # branch. Read above `costs` rather than below it since Account 222.2 came back.
    i_pf_conductor = PFConductorModel(int(switches.get("i_pf_conductor", 0)))
    # `build_variables.py:177`, the same read and the same default `_tokamak_device`'s
    # `pf_coil_switches` makes -- Account 222.2 is the one slot outside that function
    # that needs the PF coil topology, and the two live in different scopes. Kept as a
    # second read of one dict entry rather than a second *answer*: both go through
    # `_pf_coil_topology`, so a machine cannot hold two topologies again.
    pf_magnet_cost_iohcl = switches.get("iohcl", 1)
    # The PF coil system's own positive arm, asked of the same two switches
    # `_pf_coil_system_arm` asks -- see `_pf_coil_material_arm` for why it is one
    # function and not two copies of one `if`.
    pf_magnet_cost_material_arm = _pf_coil_material_arm(
        pf_magnet_cost_iohcl,
        switches.get("i_cs_superconductor", 1),  # `pfcoil_variables.py:239`
    )
    cost_of_electricity = _slot_occupant(
        "ireactor_ipnet_itart",
        _cost_of_electricity_arm(ireactor, switches.get("ipnet", 0), itart),
        COST_OF_ELECTRICITY,
    )
    # ---- the five subsystems both devices have, built once ------------------------
    #
    # Identical arguments on either arm, so they are resolved above the branch rather
    # than transcribed into two constructor calls. That is not tidying: a second
    # transcription of `cost_of_electricity`/`i_tf_sup`/`ireactor` is exactly the shape
    # step 4d removed from the tree ("a switch is answered once"), and writing the
    # tokamak's copy by hand would have re-created it five times over.
    costs = pick(
        "i_cost_model",
        COST_MODEL,
        1,
        build=lambda cls: cls(
            cost_of_electricity=cost_of_electricity,
            # **The one slot in this tree decided by the device rather than by a
            # switch**, and the exception is the point of it: Account 221.4 costs a
            # reactor structure, and a stellarator has none to cost. `st_strc` sets
            # both of its masses to a literal `0.0`, so the node would compute an exact
            # zero from a subsystem the device does not have -- the `EcrhDensityLimit`
            # bug class -- while a tokamak's `.tokamak.structure` slot owns both and
            # PROCESS computes a real number. `None` is absence, not refusal, by
            # `UNPORTED`'s own rule; `models/costs/namespace.py` carries the argument.
            reactor_structure_cost=(
                ReactorStructureCost() if device is TokamakProcess else None
            ),
            # The second, landed 2026-08-30 once `Power.pfpwr` gave it its seven reads.
            # Same argument as 221.4: `stellarator.py` never calls `Power.run`, so there
            # is no PF coil power supply to condition.
            pf_coil_power_conditioning_cost=(
                PfCoilPowerConditioningCost() if device is TokamakProcess else None
            ),
            energy_storage_cost=_slot_occupant(
                "i_pulsed_plant_istore",
                _energy_storage_arm(
                    switches.get("i_pulsed_plant", 0), switches.get("istore", 1)
                ),
                ENERGY_STORAGE,
            ),
            # `cost_variables.py:552` -- the two strand-cost formulas read disjoint
            # fields, so this is a slot rather than the static kwarg it was.
            tf_magnet_cost_superconducting=_slot_occupant(
                "supercond_cost_model",
                supercond_cost_model,
                TF_MAGNET_COST_SUPERCONDUCTING,
            ),
            # The third device-decided slot, and the only one that is *also* switched:
            # `None` on a stellarator (no PF coil system to cost, §12.2's argument),
            # and on a tokamak the `supercond_cost_model` x `iohcl` occupant one account
            # later.
            #
            # **`iohcl` used to be pinned here to `PRESENT` and it was wrong on two
            # tracked files** (`_audit/switch_consultation_audit.md` §2). The comment
            # that stood here justified the pin by a `_pf_coil_system_deviations`
            # refusal -- "`-1` refuses `iohcl != 1`" -- that commit `253c426a` retired
            # when the no-central-solenoid arm landed. The refusal stopped firing and
            # the hardcoded answer behind it was never revisited, so both spherical
            # tokamaks costed six PF coils plus a solenoid they do not have while the
            # PF coil *system* eleven hundred lines above read the same switch
            # correctly. The switch is now a second occupant (§14.2: no switch is a
            # static kwarg) and the count comes from the same `_pf_coil_topology` the
            # PF coil system is measured against, so the two cannot disagree again.
            #
            # `n_cs_pf_coils` and `i_pf_conductor` stay static kwargs and are not
            # switches: the first is the topology's coil count and the second branches
            # inside every arm rather than between them. `_pf_coil_system_deviations`
            # `-5` still fixes the conductor before any tokamak finishes assembling.
            pf_magnet_cost=(
                _slot_occupant(
                    "supercond_cost_model_iohcl",
                    _pf_magnet_cost_arm(
                        supercond_cost_model,
                        pf_magnet_cost_iohcl,
                        pf_magnet_cost_material_arm,
                    ),
                    PF_MAGNET_COST,
                    build=lambda cls: cls(
                        n_cs_pf_coils=_pf_coil_topology(
                            pf_magnet_cost_iohcl
                        ).n_cs_pf_coils,
                        i_pf_conductor=i_pf_conductor,
                    ),
                )
                if device is TokamakProcess
                else None
            ),
        ),
    )
    # `i_p_coolant_pumping` decides five things in `power` and one in `.tokamak.
    # ccfe_hcpb`, and until this pass all six carried a hardcoded copy of the Helias
    # run's answer. Resolved once, here, and threaded -- `fwbs_variables.py:249` is the
    # default. The slot resolution comes *before* the value is threaded, the same
    # discipline `i_tf_sup` follows, so no unported value ever reaches an occupant's
    # static field.
    # (`i_pf_conductor` is read above `costs`, which needs it too.)
    i_p_coolant_pumping = PumpingPowerModelTypes(switches.get("i_p_coolant_pumping", 2))
    # The thermal-efficiency family. Three switches, four slots, and every one of them
    # used to carry the answer as a static kwarg -- `fwbs_variables.py:264` for
    # `i_thermal_electric_conversion`, `:70` for `i_blanket_type`, `:273` for
    # `secondary_cycle_liq`. Read once here and turned into arm indices below.
    i_thermal_electric_conversion = ElectricConversionModelTypes(
        int(switches.get("i_thermal_electric_conversion", 0))
    )
    i_blanket_type = BlktModelTypes(int(switches.get("i_blanket_type", 1)))
    i_blkt_dual_coolant = BlanketDualCoolantModel(
        int(switches.get("i_blkt_dual_coolant", 0))  # `fwbs_variables.py:526`
    )
    secondary_cycle_liq = ElectricConversionModelTypes(
        int(switches.get("secondary_cycle_liq", 4))
    )
    power = Power(
        # `Power.pfpwr`, the PF-coil power supply -- present on a tokamak and genuinely
        # absent on a stellarator, which has no PF coils and whose `stellarator.py`
        # never calls `Power.run` at all. The only slot in this namespace whose
        # occupancy is decided by the device rather than by a switch.
        # **The topology is threaded, not defaulted.** `PfCoilPowerSupplies.topology`
        # is a static field the class's own docstring says "one occupant serves both"
        # through -- and this call site never set it, so every machine got
        # `REFERENCE_TOPOLOGY`. On a solenoid-less machine that is 7 CS+PF coils where
        # there are 8, and `pfpwr`'s `pfckts = (n_pf_cs_plasma_circuits - 2) + 6` comes
        # out **12 against PROCESS's 13** -- a 1/13 that propagates into `spfbusl`,
        # `acptmax`, `srcktpm` and six Account 22.5.2 power-conditioning rows, at
        # 2e-02 to 2.5e-01 relative. Found by running `cold_start` on
        # `spherical_tokamak_eval` and `st_regression` for the first time.
        #
        # `_pf_coil_topology`'s own docstring already named this shape: the cost side
        # once held `N_CS_PF_COILS` (7, with a solenoid) while the PF coil system
        # correctly held 8 and none, and the answer is "written once and read twice
        # rather than transcribed". This is the third reader, and it was missed.
        pf_coil_power=(
            PfCoilPowerSupplies(topology=_pf_coil_topology(pf_magnet_cost_iohcl))
            if device is TokamakProcess
            else None
        ),
        tf_power=tf_power,
        # `pf_power_variables.py:18` -- the two arms read complementary fields.
        acpow=_slot_occupant(
            "i_pf_energy_storage_source",
            PFEnergyStorageSource(int(switches.get("i_pf_energy_storage_source", 2))),
            ACPOW,
        ),
        # **The last two slots in the tree that still carry a switch as a static
        # kwarg** (`_audit/next_steps.md` §14.2). `i_blanket_type` and
        # `secondary_cycle_liq` left `ComponentThermalPowers` with the seven dead reads
        # they fed; the three below are real branches on both nodes, and splitting them
        # is a 2 x 3 x 2 product of occupants over a 26-read signature -- written up in
        # §14.11 rather than improvised here. Every value is threaded from the file, so
        # neither can contradict the slots the same switches decide.
        component_thermal_powers=ComponentThermalPowers(
            i_p_coolant_pumping=i_p_coolant_pumping,
            i_blkt_dual_coolant=i_blkt_dual_coolant,
            i_thermal_electric_conversion=i_thermal_electric_conversion,
        ),
        delta_eta_step=DeltaEtaStep(
            i_p_coolant_pumping=i_p_coolant_pumping,
            i_blkt_dual_coolant=i_blkt_dual_coolant,
            i_thermal_electric_conversion=i_thermal_electric_conversion,
        ),
        eta_turbine=_slot_occupant(
            "eta_turbine_arm",
            _eta_turbine_arm(i_thermal_electric_conversion, i_blanket_type),
            ETA_TURBINE,
            build=lambda cls: None if cls is None else cls(),
        ),
        etath_liq=_slot_occupant(
            "secondary_cycle_liq",
            secondary_cycle_liq,
            ETATH_LIQ,
            build=lambda cls: None if cls is None else cls(),
        ),
        temp_turbine_coolant_in=_slot_occupant(
            "temp_turbine_coolant_in_arm",
            _temp_turbine_coolant_in_arm(
                i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq
            ),
            TEMP_TURBINE_COOLANT_IN,
            build=lambda cls: None if cls is None else cls(),
        ),
        p_fw_div_heat_deposited_mw=_slot_occupant(
            "p_fw_div_heat_deposited_arm",
            _p_fw_div_heat_deposited_arm(i_p_coolant_pumping),
            P_FW_DIV_HEAT_DEPOSITED,
            build=lambda cls: None if cls is None else cls(),
        ),
        p_fw_blkt_coolant_pump_mw=_slot_occupant(
            "p_fw_blkt_coolant_pump_arm",
            _p_fw_blkt_coolant_pump_arm(i_p_coolant_pumping),
            P_FW_BLKT_COOLANT_PUMP,
            build=lambda cls: None if cls is None else cls(),
        ),
        cryo_q_nuc=_slot_occupant(
            "inuclear_i_tf_sup",
            _cryo_q_nuc_arm(switches.get("inuclear", 0), i_tf_sup),
            CRYO_Q_NUC,
            build=lambda cls: None if cls is None else cls(),
        ),
        cryo_q_loads=_slot_occupant(
            "cryo_q_loads_arm",
            _cryo_q_loads_arm(i_tf_sup, i_pf_conductor),
            CRYO_Q_LOADS,
            build=lambda cls: None if cls is None else cls(),
        ),
        cryo_loads=_slot_occupant(
            "cryo_loads_arm",
            _cryo_loads_arm(i_tf_sup, i_pf_conductor),
            CRYO_LOADS,
        ),
    )
    buildings = Buildings(sizing=pick("i_bldgs_size", BUILDING_SIZING, 0))
    availability = Availability(
        electric_production=_slot_occupant(
            "electric_production_arm",
            _electric_production_arm(
                ireactor,
                itart,
                i_tf_sup,
                i_blkt_dual_coolant,
                i_p_coolant_pumping,
            ),
            ELECTRIC_PRODUCTION,
        ),
        avail=_slot_occupant(
            # `cost_variables.py:416`
            "ibkt_life",
            BlanketLifetimeModel(int(switches.get("ibkt_life", 0))),
            AVAIL,
        ),
        cplife_avail=_slot_occupant(
            "cplife_arm",
            _cplife_arm(itart, i_tf_sup),
            CPLIFE,
            build=lambda cls: None if cls is None else cls(),
        ),
    )
    # `physics_variables.py:875` -- both arms ported, identical reads. A slot rather
    # than a static kwarg under `_audit/next_steps.md` §14.2.
    fast_alpha_beta = _slot_occupant(
        "i_beta_fast_alpha",
        FastAlphaPressureModel(int(switches.get("i_beta_fast_alpha", 1))),
        FAST_ALPHA_BETA,
    )
    # `i_plasma_ignited` is already resolved above for the confinement head; this is its
    # fourth consumer and it is threaded, not re-read.
    plasma_composition = _slot_occupant(
        "i_plasma_ignited",
        PlasmaIgnitionModel(int(i_plasma_ignited)),
        PLASMA_COMPOSITION,
    )
    confinement_time = PhysicsConfinementTime(
        power_loss=plasma_power_loss,
        scaling=confinement_scaling,
        tail=confinement_tail,
    )

    # `init.py` runs on both devices, so its resolutions are built once, above the
    # branch, exactly like every other shared slot here.
    initialisation = _initialisation(
        imported, device, i_tf_sup, i_tf_sc_mat, iteration_variables_from_indat(imported)
    )

    if device is TokamakProcess:
        return TokamakProcess(
            initialisation=initialisation,
            # Everything device-specific, and no longer `Tokamak()`: twenty-six of
            # its twenty-eight slots have occupants, most of them switched. `i_tf_sup`,
            # `i_plasma_ignited`, `itart` and `i_tf_sc_mat` are *threaded* rather than
            # re-read, because a switch is answered once.
            tokamak=_tokamak_device(
                switches,
                numbers_from_indat(imported),
                iteration_variables_from_indat(imported),
                int_lists_from_indat(imported),
                presence_flags_from_indat(imported),
                i_tf_sup,
                i_plasma_ignited,
                itart,
                i_tf_sc_mat,
                i_tf_turn_type,
            ),
            costs=costs,
            physics=Physics(
                profiles=PhysicsProfiles(
                    # **The file decides it here, and on a stellarator it does not.**
                    # `ST_INIT_I_PLASMA_PEDESTAL` exists because `st_init` overwrites
                    # this field on every `istell != 0` run; `st_init` does not run on a
                    # tokamak, so on this arm the file's value is live and reading it is
                    # what reproduces PROCESS. `physics_variables.py:889`'s default is
                    # `1`, and `large_tokamak_eval.IN.DAT:291` sets `1` explicitly.
                    #
                    # The pedestal occupant has no `ecrh_density_limit` slot at all,
                    # which is how the one stellarator-only physics node stays out of a
                    # tokamak by construction rather than by an exception: PROCESS
                    # computes no ECRH density limit outside `i_plasma_pedestal == 0`.
                    parameterisation=_profile_parameterisation(
                        switches.get("i_plasma_pedestal", 1),
                        # `physics_variables.py`'s default is `1`
                        # (`GREENWALD_FRACTION`); `large_tokamak_eval.IN.DAT` never
                        # mentions the switch, so the default is what runs.
                        switches.get("i_nd_plasma_pedestal_separatrix", 1),
                        is_stellarator=False,
                    ),
                ),
                confinement_time=confinement_time,
                fast_alpha_beta=fast_alpha_beta,
                plasma_composition=plasma_composition,
            ),
            power=power,
            buildings=buildings,
            availability=availability,
        )

    # ---- `istell` in 1..6: everything only a stellarator asks ---------------------
    #
    # Below the branch because a tokamak has none of it: no machine config, no
    # `isthtr`, and neither joint blanket dispatch. Reading them anyway would make a
    # tokamak refusable for a stellarator's reason -- `blktmodel = 1` refuses at
    # `blanket_neutronics()`, which a tokamak never reaches.
    #
    # `machine_config_for_istell` is `load_stellarator_config`'s `match istell` and
    # nothing else: arms 1-5 return a preset table, arm 6 opens the companion JSON. The
    # default only matters on arm 6 -- a preset machine reads no file, so a `helias_5b`
    # run needs no `stella_conf` companion and is not given a stellarator's one by
    # accident.
    machine_config = StellaratorMachineConfig(
        machine_config=machine_config_for_istell(
            istell,
            config_file=(REFERENCE_STELLA_CONF if stella_conf is None else stella_conf),
        )
    )
    # The two joint dispatches, resolved into named locals before the constructor call
    # for the same reason `istell` is: so the *first* thing a refused combination
    # reports is the one the caller asked for, not whichever slot Python evaluated
    # first. Every default here is PROCESS's own -- `fwbs_variables.py:479` for
    # `blktmodel`, `:494` for `blkttype`, `heat_transport_variables.py:94` for
    # `ipowerflow` -- and the switch *values* are turned into **arm indices** by the two
    # named functions above, which is the only thing the registries are keyed on.
    #
    # This used to read `blktmodel = switches.get("blktmodel", 2)` and pass that value
    # through where an arm index was wanted. `2` is not a legal `blktmodel` at all: it
    # was a sentinel meaning "not set", picked so the reference run happened to land on
    # arm 2. The consequence was an inverted mapping -- stating PROCESS's own default
    # `blktmodel = 0` was refused, while `blktmodel = 1` (KIT HCPB neutronics) silently
    # assembled `BlanketShieldPowerExponential`, a node written for a different switch's
    # arm. That is the `ScTfCoilNuclearHeating` bug class, reintroduced by a key
    # derivation instead of by a registration.
    blktmodel = switches.get("blktmodel", 0)
    ipowerflow = switches.get("ipowerflow", 1)
    blanket_shield_power = _slot_occupant(
        "blktmodel_ipowerflow_i_p_coolant_pumping",
        _blanket_shield_power_arm(
            blktmodel,
            ipowerflow,
            # `fwbs_variables.py:249`. A stellarator that leaves it there lands on
            # arm 4, which refuses -- correctly: PROCESS raises on that value too.
            switches.get("i_p_coolant_pumping", 2),
        ),
        BLANKET_SHIELD_POWER,
    )
    blanket_masses = _slot_occupant(
        "blktmodel_blkttype",
        _blanket_mass_arm(blktmodel, switches.get("blkttype", 3)),
        BLANKET_MASSES,
    )
    fw_area = _slot_occupant("ipowerflow", ipowerflow, FW_AREA)
    wall_load_arm = _wall_load_arm(
        switches.get("i_pflux_fw_neutron", 1),  # `physics_variables.py:1006`
        ipowerflow,
    )
    ipowerflow = PowerFlowModel(ipowerflow)
    # The superconductor, and with it whether the coils block is a cycle: only the
    # Bi-2212 occupant reads `.tfcoil.j_tf_wp`, which `winding_pack_total_size_post`
    # owns. `i_tf_sc_mat` itself is resolved above the device branch now -- the tokamak's
    # TF mass slot asks the same switch, and threading one local is what stops the two
    # devices' answers from drifting apart.
    winding_pack_intersect_inputs = _slot_occupant(
        "i_tf_sc_mat", i_tf_sc_mat, WINDING_PACK_MATERIAL
    )
    # The second consumer of the same switch, and until `_audit/next_steps.md` §14.2 it
    # answered the question itself, with a module constant no instrument could see.
    coils_mass = _slot_occupant("i_tf_sc_mat", i_tf_sc_mat, COILS_MASS_MATERIAL)
    return StellaratorProcess(
        initialisation=initialisation,
        costs=costs,
        stellarator=Stellarator(
            coils=StellaratorCoils(
                winding_pack_intersect_inputs=winding_pack_intersect_inputs,
                coils_mass=coils_mass,
            ),
            machine_config=machine_config,
            heating=pick("isthtr", HEATING, 1),
            fw_area=fw_area,
            fwbs=StellaratorFwbs(
                blanket_shield_power=blanket_shield_power,
                blanket_masses=blanket_masses,
            ),
            # One arm index, two slots: `_wall_load_arm` is the whole dispatch and
            # both registries are keyed on it. `physics_variables.py:1006` is
            # `i_pflux_fw_neutron`'s default.
            neutron_wall_load=_slot_occupant(
                "i_pflux_fw_neutron_ipowerflow", wall_load_arm, NEUTRON_WALL_LOAD
            ),
            radiated_wall_load_and_fraction=_slot_occupant(
                "i_pflux_fw_neutron_ipowerflow", wall_load_arm, RADIATED_WALL_LOAD
            ),
            heating_and_radiation_power=_slot_occupant(
                "i_plasma_ignited",
                PlasmaIgnitionModel(int(i_plasma_ignited)),
                HEATING_AND_RADIATION_POWER,
            ),
        ),
        physics=Physics(
            profiles=PhysicsProfiles(
                # Not `switches.get("i_plasma_pedestal", 1)`: `st_init` overwrites the
                # file's value on every stellarator run, so the file cannot decide this
                # slot and this port must not pretend it does. See
                # `ST_INIT_I_PLASMA_PEDESTAL`.
                parameterisation=_profile_parameterisation(
                    ST_INIT_I_PLASMA_PEDESTAL,
                    # Never read: `ST_INIT_I_PLASMA_PEDESTAL` is the parabolic arm and
                    # `_profile_parameterisation` asks `PEDESTAL_SEPARATRIX` only on the
                    # pedestal one. Passed rather than made optional so that the two
                    # call sites answer the same three questions and a reader can see
                    # that this device declines the third.
                    switches.get("i_nd_plasma_pedestal_separatrix", 1),
                    is_stellarator=True,
                ),
            ),
            confinement_time=confinement_time,
            fast_alpha_beta=fast_alpha_beta,
            plasma_composition=plasma_composition,
        ),
        power=power,
        buildings=buildings,
        availability=availability,
    )


REFERENCE_MACHINE = machine_from_indat(REFERENCE_INPUT_FILE)
"""The machine `stellarator_helias.IN.DAT` describes -- the run this port is validated
against (`istell = 6`, `i_plasma_pedestal = 0`, `i_cost_model = 0`; every other switch
at PROCESS's own default).
"""


def graph_for(machine=None):
    """The assembled graph for one machine; `REFERENCE_MACHINE` if unstated."""
    return to_graph(REFERENCE_MACHINE if machine is None else machine)


GRAPH = graph_for()
"""`REFERENCE_MACHINE`'s graph -- the `stellarator_helias.IN.DAT` run this port is
validated against.
"""

if __name__ == "__main__":
    n_vars = sum(
        len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values()
    )
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")

    print("\ncycles, per machine:")
    for label, machine in (
        ("the reference machine", REFERENCE_MACHINE),
        (
            # Both slots `ipowerflow` decides, not just `fw_area`: it also picks arm 1
            # of the blanket/shield-power dispatch. Swapping one and not the other used
            # to be the only spelling available, because arm 1 was unreachable through
            # `machine_from_indat` at all -- the joint key was derived from an illegal
            # `blktmodel` sentinel. It is reachable now, and this what-if says
            # `ipowerflow = 0` coherently.
            "ipowerflow = 0",
            eqx.tree_at(
                lambda m: (
                    m.stellarator.fw_area,
                    m.stellarator.fwbs.blanket_shield_power,
                ),
                REFERENCE_MACHINE,
                (AFwTotalNoPowerflow(), BlanketShieldPowerExponential()),
            ),
        ),
    ):
        cycles = graph_for(machine).cycles
        print(f"  {label}: {[[n.path_str() for n in c] for c in cycles] or 'acyclic'}")
