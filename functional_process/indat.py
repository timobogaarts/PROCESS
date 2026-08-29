"""PROCESS's input encoding, and the one place this port reads it.

`total_process.py` is the tree -- typed slots and the models that fill them, and not one
`i_*` integer anywhere in it. This module is the adapter between that tree and the
legacy IN.DAT format: `switches_from_indat` reads the integers, the registries below map
each one to the occupant it selects, `UNPORTED` records why a real PROCESS value has
none, and `machine_from_indat` assembles the `StellaratorProcess` an input file
describes. Everything switch-shaped is PROCESS's input encoding, not the machine, which
is why it is all here and none of it is beside a subsystem.

`machine_from_indat`'s own docstring is where the argument lives for why assembly time
is the only correct place to resolve a switch (short version: no switch in PROCESS is
ever an iteration variable or a scan variable, so no switch can change between two
evaluations of one assembled graph).

`GRAPH` and `graph_for` live here too, and that is the honest filing rather than a
convenience: `graph_for()` with no argument *is* `REFERENCE_MACHINE`, the graph of one
particular legacy input file, and every caller in this package calls it that way. The
alternative -- keeping them in `total_process.py` -- is not merely churn-minimising, it
is a genuine import cycle, since this module must import `StellaratorProcess` from
there. Run this module directly for a smoke check (builds the graph, prints its
node/port counts and each machine's cycles); `render_xdsm.py`, `mda.py`, `mdf.py`,
`sand_harness.py` and `run_mda_harness.py` import `GRAPH`/`graph_for` from here.
"""

import functools
import re
from pathlib import Path

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import to_graph

from functional_process.models.availability.availability import (
    AvailDisplacementsPerAtom,
    AvailNeutronFluence,
    CplifeAvailResistive,
    CplifeAvailSuperconducting,
)
from functional_process.models.availability.namespace import Availability
from functional_process.models.blankets.blanket_library import (
    BlanketCoverageFactorsDoubleNull,
    BlanketCoverageFactorsSingleNull,
    BlanketHalfHeightDoubleNull,
    BlanketHalfHeightSingleNull,
    DShapedBlanketAreas,
    DShapedBlanketVolumes,
    EllipticalBlanketAreas,
    EllipticalBlanketVolumes,
)
from functional_process.models.blankets.hcpb import (
    CentrepostNeutronicsAbsent,
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
from functional_process.models.blankets.namespace import CcfeHcpb
from functional_process.models.build import (
    DivertorGeometryConventional,
    DivertorGeometrySphericalTokamak,
    DrTfInboardFromWindingPack,
    DrTfOutboardSuperconducting,
    DrTfWpWithInsulationFromInboardBuild,
    TfInboardRadiiNoCsPrecomp,
    TfInboardRadiiTfOutsideCs,
    VacuumVesselAndShieldRadiiTfOutsideCs,
    TfOutboardEdgeRipple,
    TfOutboardEdgeRipplePictureFrame,
    TfOutboardMidDShape,
    TfOutboardMidPictureFrame,
    WpConductorMaxWidthSuperconducting,
)
from functional_process.models.buildings.buildings import (
    Bldgs,
    BldgsSizes,
)
from functional_process.models.buildings.namespace import Buildings
from functional_process.models.costs.costs import (
    CostOfElectricityConventionalAspectRatio,
    CostOfElectricitySphericalTokamak,
    EnergyStorageCostPulsedElectrowattOption1,
    EnergyStorageCostPulsedElectrowattOption2,
    EnergyStorageCostUnpulsed,
    TfMagnetCostSuperconductingPerKam,
    TfMagnetCostSuperconductingPerKg,
)
from functional_process.models.costs.namespace import Costs
from functional_process.models.divertor import (
    DivertorHeatLoadWadeDoubleNull,
    DivertorHeatLoadWadeSingleNull,
)
from functional_process.models.fw import (
    FirstWallDoubleNull,
    FirstWallDShapedDoubleNull,
    FirstWallSingleNull,
)
from functional_process.models.namespace import Build, Divertor
from functional_process.models.pfcoil.namespace import (
    CSCoil,
    PFCoil,
    PFCoilCsWstNb3Sn,
)
from functional_process.models.pfcoil.superconductor import (
    CSCriticalCurrentDensitiesIterNb3Sn,
    CSCriticalCurrentDensitiesWstNb3Sn,
)
from functional_process.models.physics.bootstrap_current import (
    NoDiamagneticCurrent,
    NoPfirschSchluterCurrent,
    SauterBootstrapCurrentFraction,
    SceneDiamagneticCurrent,
    ScenePfirschSchluterCurrent,
)
from functional_process.models.physics.composition import (
    PlasmaCompositionIgnited,
    PlasmaCompositionNonIgnited,
)
from functional_process.models.physics.confinement_time import (
    ConfinementTailCoreRadiation,
    Iss04ConfinementTime,
    IterIpb98y2ConfinementTime,
    PlasmaPowerLossIgnitedCoreRadiation,
    PlasmaPowerLossNonIgnitedCoreRadiation,
)
from functional_process.models.physics.current_drive import (
    HcdElectricTotalIgnited,
    HcdElectricTotalNonIgnited,
    HcdPrimaryEfficiencyFreethyEcrhOMode,
    HcdPrimaryEfficiencyUserInputEcrh,
    HcdPrimaryPowersElectronCyclotronNoSecondary,
    HcdSecondaryHeatingNone,
)
from functional_process.models.physics.density_limit import (
    EnforcedDensityLimitGreenwald,
    TokamakDensityLimit,
)
from functional_process.models.physics.l_h_transition import (
    Martin08AspectLowerLHThresholdPower,
    Martin08AspectNominalLHThresholdPower,
    Martin08AspectUpperLHThresholdPower,
    Martin08LowerLHThresholdPower,
    Martin08NominalLHThresholdPower,
    Martin08UpperLHThresholdPower,
)
from functional_process.models.physics.namespace import (
    Physics,
    PhysicsConfinementTime,
    PhysicsProfiles,
    ProfileParameterisationParabolic,
    ProfileParameterisationPedestal,
)
from functional_process.models.physics.physics import (
    BetaNormMaxWesson,
    PulseRampTimesContinuousDefault,
    PulseRampTimesPulsedDefault,
    SeparatrixPowerNonIgnited,
    SurfaceAveragedPoloidalFieldAmperes,
)
from functional_process.models.physics.plasma_current import (
    FiestaStPlasmaCurrent,
    Ipdg89PlasmaCurrent,
    TokamakPlasmaCurrent,
    WessonCurrentProfileIndex,
)
from functional_process.models.physics.plasma_fields import PlasmaFields
from functional_process.models.physics.plasma_geometry import (
    CreateDataEuDemoXPointPlasmaShape,
    DoubleArcPlasmaGeometry,
    Ipdg89XPointPlasmaShape,
)
from functional_process.models.physics.plasma_inductance import (
    PlasmaInternalInductanceNormWesson,
    TokamakPlasmaInductance,
)
from functional_process.models.physics.profiles import (
    GreenwaldDensityFractions,
    PedestalSeparatrixDensities,
)
from functional_process.models.physics.pure_formulas import (
    FastAlphaBetaIterPhysicsRules,
    FastAlphaBetaWard,
)
from functional_process.models.physics.scrape_off_layer import (
    OutboardSOLPowerDecayLengthEich2013,
    TokamakScrapeOffLayer,
)
from functional_process.models.physics.tokamak_namespace import (
    TokamakCurrentDrive,
    TokamakPhysics,
    TokamakPlasmaBeta,
    TokamakPlasmaGeom,
    TokamakPulse,
)
from functional_process.models.power.electric_production import (
    AcpowLine,
    AcpowMotorGeneratorFlywheel,
    PlantElectricProductionLiquidBreeder,
    PlantElectricProductionResistiveCentrepostLiquidBreeder,
    PlantElectricProductionResistiveCentrepostSingleCoolant,
    PlantElectricProductionSingleCoolant,
    PowerProfilesOverTime,
)
from functional_process.models.power.namespace import Power
from functional_process.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power.thermal_cryo import (
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
from functional_process.models.shield import (
    DoubleNullShieldHalfHeight,
    DShapedShieldVolumes,
    EllipticalShieldVolumes,
    SingleNullShieldHalfHeight,
    TokamakShield,
)
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
)
from functional_process.models.stellarator.coils.calculate import (
    Bi2212WindingPackIntersectInputs,
    CrocoRebcoWindingPackIntersectInputs,
    DurhamNbtiWindingPackIntersectInputs,
    DurhamRebcoWindingPackIntersectInputs,
    IterNb3snWindingPackIntersectInputs,
    OldLubellNbtiWindingPackIntersectInputs,
    UserDefinedNb3snWindingPackIntersectInputs,
    WstNb3snWindingPackIntersectInputs,
)
from functional_process.models.stellarator.coils.mass import (
    Bi2212CoilsMass,
    CrocoRebcoCoilsMass,
    DurhamNbtiCoilsMass,
    DurhamRebcoCoilsMass,
    IterNb3snCoilsMass,
    OldLubellNbtiCoilsMass,
    UserDefinedNb3snCoilsMass,
    WstNb3snCoilsMass,
)
from functional_process.models.stellarator.density_limits import EcrhDensityLimit
from functional_process.models.stellarator.heating import (
    EcrhHeating,
    LowhybHeating,
)
from functional_process.models.stellarator.namespace import (
    BlanketShieldPowerExponential,
    Stellarator,
    StellaratorCoils,
    StellaratorFwbs,
)
from functional_process.models.stellarator.plasma_physics import (
    HeatingAndRadiationPowerIgnited,
    HeatingAndRadiationPowerNonIgnited,
    NeutronWallLoadFirstWallAreaComprehensive2014,
    NeutronWallLoadFirstWallAreaPre2014,
    NeutronWallLoadScaledPlasmaSurface,
    RadiatedWallLoadFirstWallAreaComprehensive2014,
    RadiatedWallLoadFirstWallAreaPre2014,
    RadiatedWallLoadScaledPlasmaSurface,
)
from functional_process.models.stellarator.preset_config import (
    StellaratorMachineConfig,
    read_stellarator_config_file,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPower,
)
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
)
from functional_process.models.structure import Structure
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
from functional_process.models.tfcoil.base import (
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
from functional_process.models.tfcoil.namespace import CiccSuperconductingTfCoil
from functional_process.models.tfcoil.quench import (
    TfCoilQuenchHeatCurrentDensity,
    helium_properties_at_quench_nodes,
)
from functional_process.models.tfcoil.superconducting import (
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
from functional_process.models.tokamak.namespace import Tokamak
from functional_process.models.vacuum.vacuum import (
    VacuumVesselDShapedDoubleNull,
    VacuumVesselEllipticalDoubleNull,
    VacuumVesselEllipticalSingleNull,
)
from functional_process.total_process import StellaratorProcess, TokamakProcess
from process.core.solver.iteration_variables import ITERATION_VARIABLES
from process.data_structure.blanket_variables import BlktModelTypes
from process.data_structure.build_variables import TFCSRadialConfiguration
from process.data_structure.divertor_variables import DivertorHeatLoadModel
from process.data_structure.pfcoil_variables import PFConductorModel
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    CurrentProfileIndexModel,
    DivertorNumberModels,
    OutbordSOLPowerDecayLengthModel,
    PlasmaIgnitionModel,
)
from process.data_structure.superconducting_tf_coil_variables import TFWPIntegerTurnType
from process.models.build import FwBlktVVShape
from process.models.physics.bootstrap_current import BootstrapCurrentFractionModel
from process.models.physics.current_drive import (
    CurrentDriveMethodType,
    CurrentDriveModel,
)
from process.models.physics.density_limit import DensityLimitModel
from process.models.physics.l_h_transition import PlasmaConfinementTransitionModel
from process.models.physics.physics import IndInternalNormModel
from process.models.physics.plasma_current import (
    PlasmaCurrentModel,
    PlasmaDiamagneticCurrentModel,
)
from process.models.physics.plasma_geometry import (
    PlasmaGeometryModelType,
    PlasmaShapeModelType,
)
from process.models.power import (
    ElectricConversionModelTypes,
    PumpingPowerModelTypes,
)
from process.models.superconductors import SuperconductorModel
from process.models.tfcoil.base import (
    TFCoilShapeModel,
    TFConductorModel,
    TFPlasmaCaseType,
)
from process.models.tfcoil.superconducting import (
    SuperconductingTFTurnType,
    SuperconductingTFWPShapeType,
)

REFERENCE_STELLA_CONF = (
    Path(__file__).resolve().parent.parent
    / "tests/regression/input_files/stellarator_helias.stella_conf.json"
)
"""`REFERENCE_INPUT_FILE`'s `istell == 6` machine-config companion.

`Stellarator.st_new_config()` opens `f"{data.globals.output_prefix}stella_conf.json"`
before anything else runs, so for the reference run this file *is* the machine being
designed. Read here, at assembly time, and handed to `StellaratorMachineConfig` as static
data -- the whole point of unit #8's shape decision (`preset_config.md`): `istell == 6`'s
file I/O is a `non-traceable-external-call` that never has to enter a traced body,
because which machine is being designed cannot change during a solve.

Named beside `REFERENCE_INPUT_FILE` below rather than next to the tree, because the
`.stellarator.istell` switch needs it; the two must stay companions (same
stem, same directory), which is what PROCESS's own `output_prefix` convention enforces
for a real run."""


REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The run this whole port is validated against -- `mda_harness.py`, `mda_constraint_
harness.py` and every number in `_audit/next_steps.md` \u00a7 8 use it. Named here so
`REFERENCE_CONFIGURATION` can be checked against it mechanically instead of by eye."""

_ISTELL_PRESET_REASON = (
    "`istell` in 1..5 selects one of five hardcoded machine presets (Helias 5/4/3, "
    "W7-X 30/50) copied onto `StellaratorConfigData` by `preset_config.py`'s reflective "
    "`hasattr`/`setattr` loop; only `istell == 6` (config read from file) is in scope. "
    "See `core/solver/switches.md` § `data.stellarator.istell` -- second role, whose "
    "disposition is still open in `_audit/next_steps.md` § 2. The confinement-time "
    "binding itself would be identical to the `istell == 6` occupant; it is the "
    "surrounding preset data that is unported"
)
"""Shared by all five refused `istell` presets -- one reason, five values."""

_I_PLASMA_GEOMETRY_REASON = (
    "eleven of `i_plasma_geometry`'s thirteen values are unwritten (0 and 10 are "
    "written). Each reads a genuinely different set of fields (the dispatch table is "
    "`plasma_geometry.md` \u00a7 'the `i_plasma_geometry` dispatch'), and none is live "
    "on any tracked regression input. Under this wave's binding policy each needs its "
    "own occupant class rather than a family grouped by reads-identical sets, which is "
    "what supersedes that record's open question 1"
)
"""Shared by the eleven refused `i_plasma_geometry` values -- one reason, eleven values.

Enumerated rather than written as a single sentinel key, because `_slot_occupant` looks
`UNPORTED` up by the value it was actually handed: a sentinel would turn every one of
these into *"not a known value"*, which is the message reserved for a typo."""

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

**Every switch the file sets explicitly is listed, including ones whose value happens to
equal PROCESS's own default** (`isthtr`, `ireactor`). Listing them regardless makes this
a transcription of the file rather than a diff against PROCESS's defaults, and means a
future change to a default cannot silently move the reference run.
`test_machine.py` parses the file and checks this dict against it, both ways, so
the two cannot drift.

This exists as data, not as behaviour: `machine_from_indat` reads the file itself. It is
here so the check has something to compare against.
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
    "never reached at those values and there is no PROCESS behaviour to port"
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
    ("istell", 1): _ISTELL_PRESET_REASON,
    ("istell", 2): _ISTELL_PRESET_REASON,
    ("istell", 3): _ISTELL_PRESET_REASON,
    ("istell", 4): _ISTELL_PRESET_REASON,
    ("istell", 5): _ISTELL_PRESET_REASON,
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
    ("blktmodel_ipowerflow", 0): (
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
    ("i_tf_turn_type", SuperconductingTFTurnType.CROSS_CONDUCTOR): (
        "the CroCo (cross-conductor) turn selects a **different PROCESS `Model` class**, "
        "`CROCOSuperconductingTFCoil` (`process/models/tfcoil/superconducting.py:3773`), "
        "not a different arm inside the cable-in-conduit one: `core/caller.py:298-313` "
        "dispatches on `i_tf_turn_type` above every model and runs "
        "`models.croco_sctfcoil` instead of `models.cicc_sctfcoil`. Its `run` computes a "
        "REBCO tape stack -- `tf_croco_averaged_turn_geometry`, "
        "`tf_turn_croco_cable_space_properties`, `calculate_croco_cable_geometry`, "
        "`tf_croco_inboard_areas_and_fractions` and the `.superconducting_tfcoil."
        "*croco*`/`*hts_tape*` fields they own -- none of which exists in this port, and "
        "it refuses integer turn geometry outright (`:3838`). That is a whole unit, not "
        "an occupant. **Both tracked spherical tokamaks need it** "
        "(`spherical_tokamak_eval.IN.DAT:72`, `st_regression.IN.DAT:800`), which is the "
        "measured reason neither assembles yet. Refused here, above the device branch, "
        "because until 2026-08-29 nothing asked: a machine with this value assembled "
        "**silently as cable-in-conduit**, and only the tape-superconductor refusal "
        "inside `CICC_SUPERCONDUCTOR_PROPERTIES` was accidentally catching the two ST "
        "files"
    ),
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
        "`USER_INPUT` has no arm at all in `powerflow_calc` -- the pumping powers are "
        "inputs. Absence rather than a refusal in principle, and filed here because the "
        "slot's other arms all produce something and no occupant in this port declares "
        "per-arm absence yet"
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
    ("pf_coil_system_arm", -1): (
        "`.build.iohcl == 0`: no central solenoid at all -- no CS filaments, "
        "`c_cs_flat_top_end = 0`, the flux swing's `:626-661` arm, `ohcalc` skipped "
        "entirely and index 6 of every coil array left at zero. A different occupant "
        "set for every node in the package. Not written (`pfcoil/geometry.md`, "
        "`currents.md`, `fields.md`, `masses.md`, `inductance.md`)"
    ),
    ("pf_coil_system_arm", -2): (
        "an `i_pf_location`/group topology other than `n_pf_coil_groups = 4`, "
        "`i_pf_location = (2, 2, 3, 3)`, `n_pf_coils_in_group = (1, 1, 2, 2)`: the "
        "pattern fixes every array index in the package (`pfcoil/__init__.py`'s module "
        "constants), so a different pattern is a different occupant per node. Not "
        "written"
    ),
    ("pf_coil_system_arm", -3): (
        "`.physics.itart == 1` (or `itartpf != 0`): the ST arm places coils from "
        "`z_tf_inside_half - zref[g]`, computes `ccls` from `aspect**1.6` and never "
        "calls `efc` -- genuinely different read sets in placement and currents. Not "
        "written"
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
        "a PF/CS superconductor pair with no occupant: the ported pairs are "
        "(`i_pf_superconductor == 3` NbTi, `i_cs_superconductor == 1` ITER Nb3Sn), "
        "arm 0, and (3 NbTi, 5 WST Nb3Sn), arm 1 (`low_aspect_ratio_DEMO.IN.DAT`). "
        "The switch's only effect in the ported closure is which element of "
        "`.tfcoil.dcond` is read, and per the binding policy that is still a "
        "different occupant per pair, not a parameter (`masses.md` § switches "
        "touched). Any other pair: not written"
    ),
    ("pf_coil_system_arm", -7): (
        "`i_tf_shape == PICTURE_FRAME` or `i_r_pf_outside_tf_placement == 1`: the "
        "outside-TF coil is placed flat at `r_pf_outside_tf_midplane`, dropping the "
        "`sqrt(r^2 - z^2)` and its `isinf` kludge (`pfcoil.py:1323-1339`). Not written"
    ),
}
"""Why a known PROCESS value has no occupant, verbatim from the `Alternative(unported=)`
declarations this replaced.

**Refusal, and nothing else.** A value in here raises `NotImplementedError` naming the
reason. Its quieter sibling -- a slot holding `None`, meaning *"this configuration's
graph does not compute these values"* -- lives in `COST_OF_ELECTRICITY`, in
`CRYO_Q_NUC`, in `models/tokamak/namespace.py`'s two still-empty slots, and in the
empty-at-this-value slots of the wave-2/3 registries (`CURRENT_PROFILE_INDEX`,
`IND_PLASMA_INTERNAL_NORM`, `BOOTSTRAP_CURRENT`, `SOL_OUTBOARD_POWER_DECAY` -- each
`USER_INPUT` arm is PROCESS computing nothing, the field a run input).

**`("istell", 0)` left this table**, and it is the only entry ever to have done so by
being *built* rather than by being found unreachable. Its reason was that assembling a
tokamak *"would give stellarator geometry, stellarator coils and stellarator FWBS driven
by a tokamak confinement scaling -- a graph that looks complete and is wrong"*. That was
true of `StellaratorProcess`, which is the only thing this factory could build at the
time. It is not true of `TokamakProcess`: the device-specific slot is `Tokamak`, whose
slots were all empty when the entry left (twenty-six of twenty-eight are occupied now),
so a tokamak machine assembles the shared subsystems and
**nothing** stellarator-specific. The two arms of that old reason are now both answered
structurally rather than by refusal -- the wrong-geometry half by a different device
class, the missing-physics half by empty slots that surface as boundary inputs and are
enumerated by name in `_audit/tokamak_boundary.md`.

The `| None`s that used to be in this table all left, two because they were unreachable
(every joint key outside `BLANKET_MASSES`/`BLANKET_SHIELD_POWER` already raised) and two
because the configurations they stood for, `i_cost_model == 1` and `istell == 0`, were
ones this port could not honestly assemble; the distinction between the two kinds
survives in the reasons: `i_cost_model == 1` would hand you a graph that computes no cost
of electricity, `== 2` would hand you one that looks complete and is wrong.

**When a value belongs here and when it belongs in a registry as `None`.** Refuse where
*this port* has not written the arm, or has written something that would be wrong on it.
Assemble absence where **PROCESS itself computes nothing** -- `ireactor != 1 or
ipnet != 0` is the only such case in the tree, and there the six `.costs.coe`-chain
fields keep their entering values in PROCESS exactly as they surface as boundary inputs
here. Refusing that one instead would have made `PowerProfilesOverTime`, a ported and
registered occupant, unreachable through this factory.

Keyed by `(field, value)`. For the two dispatches that read two integers at once the
`field` is the joint name `blktmodel_ipowerflow` / `blktmodel_blkttype` and the `value`
is an **arm index**, not a switch value -- see `_blanket_shield_power_arm` /
`_blanket_mass_arm`, whose docstrings are the mapping.

One of those arms, `("blktmodel_blkttype", 0)`, is unreachable through
`machine_from_indat` and kept anyway: `blktmodel == 1` selects arm 0 of *both*
dispatches, and the shield-power slot is resolved first, so the reason that surfaces is
the `blanket_neutronics()` one. The mass-arm reason is still the correct record of what
`stellarator.py:1093-1181` does, and it is what a future occupant of that arm has to
answer; it is not deleted merely because a sibling refusal masks it.
"""


def _slot_occupant(field, value, registry, *, build=None):
    """One registry lookup, with both failure modes spelled out.

    A miss on the registry *and* on `UNPORTED` means a value PROCESS has never had, or a
    typo -- reported with the values that do exist, which is the "a typo'd value fails
    loudly" property the old `Switch.choose` had and is worth keeping.

    Raises
    ------
    NotImplementedError
        The value is a real PROCESS branch this port has not written an occupant for;
        the recorded reason is in the message.
    ValueError
        The value is not one PROCESS has, or is a typo.
    """
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

    Two of them. `ife == 1` is a whole **device** -- inertial rather than
    magnetic confinement -- and PROCESS spells it not as a device class but as an `if`
    inside seven separate Account-22x cost methods, each reading a different set of
    `.ife.*` fields, none of them ported. So there is nothing for a registry to hold:
    the whole answer is "no", and the six nodes that carried `ife` as an
    `eqx.field(static=True)` are unconditionally the magnetic-confinement occupants now.
    Answering it here, once, at assembly, is `_audit/next_steps.md` §14.2's shape; the
    alternative it withdrew was seven bodies each holding the integer and each raising
    at trace time.

    `i_tf_turn_type == 2` is the second, added by the ST closing wave (2026-08-29), and
    it is the same shape one level down: PROCESS resolves it **above every model**, in
    `core/caller.py:298-313`, and runs `CROCOSuperconductingTFCoil` instead of
    `CICCSuperconductingTFCoil` -- a different `Model` class, not a different slot inside
    one. `models/tfcoil/namespace.py`'s own docstring said exactly this and nothing
    checked it: before this refusal existed, an input setting `i_tf_turn_type = 2`
    **assembled silently as a cable-in-conduit machine**, measured on a copy of
    `large_tokamak_eval.IN.DAT` with the one line added. That is
    `low_aspect_ratio_DEMO`'s integer-turn mis-assembly (`next_steps.md` §15) a second
    time, and it is why the refusal is here rather than left to the tape-superconductor
    refusal that was catching the two spherical tokamaks by accident.

    Asked only on the superconducting arm, because that is the only branch of
    `caller.py` that reads it; a copper or aluminium machine's `i_tf_turn_type` decides
    nothing, in PROCESS or here.

    The value must already be an `IntEnum` member, so a value PROCESS has never had has
    failed at the enum call before reaching here -- `_slot_occupant`'s `ValueError`
    branch has no counterpart to write.
    """
    raise NotImplementedError(
        f"{field} == {value} is a real PROCESS branch but is not ported: "
        f"{UNPORTED[field, value]}"
    )


def _wall_load_arm(i_pflux_fw_neutron: int, ipowerflow: int) -> int:
    """`(i_pflux_fw_neutron, ipowerflow)` -> the wall-load arm, for **both** wall-load
    slots.

    `stellarator.py:2095-2117` and `:2223-2257`, transcribed:

    ```
    if i_pflux_fw_neutron == 1:  -> arm 0   scaled by the plasma surface
    elif ipowerflow == 0:        -> arm 1   first-wall area, pre-2014
    else:                        -> arm 2   first-wall area, comprehensive 2014
    ```

    One dispatch, two registries: `NEUTRON_WALL_LOAD` and `RADIATED_WALL_LOAD` are the
    same three arms applied to the neutron and the photon power, which is why
    `switch_kwarg_survey.md` band (b2) says "the same switch, so one family serves
    both". `ipowerflow` is not consulted at all on arm 0 -- which is why this is a joint
    arm index and not two nested slots.
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
"""`.physics.i_plasma_ignited` -> the stellarator heating/radiation occupant.

The ignited occupant does not read `.current_drive.p_hcd_injected_total_mw`."""


FAST_ALPHA_BETA = {
    FastAlphaPressureModel.ITER_PHYSICS_RULES: FastAlphaBetaIterPhysicsRules,
    FastAlphaPressureModel.WARD: FastAlphaBetaWard,
}
"""`.physics.i_beta_fast_alpha` -> the fast-alpha-pressure occupant.

Both values are ported, so this registry is total. The two occupants read **identical**
fields -- `switch_kwarg_survey.md` band (c), zero invented edges -- and are split anyway
under `_audit/next_steps.md` §14.2. The argument is `model_tree_design.md` §4's: an enum
family has no cheap escape when a third published formula needs a read the family cannot
express."""

PLASMA_COMPOSITION = {
    PlasmaIgnitionModel.IGNITED: PlasmaCompositionIgnited,
    PlasmaIgnitionModel.NON_IGNITED: PlasmaCompositionNonIgnited,
}
"""`.physics.i_plasma_ignited` -> the plasma-composition occupant.

Both values are ported. The ignited occupant does **not** read
`.physics.f_nd_beam_electron`: an ignited plasma has no beam ions, and that single read
is the edge one node carrying the switch invented."""


CONFINEMENT_SCALING = {
    ConfinementTimeModel.ISS04_STELLARATOR: Iss04ConfinementTime,
    ConfinementTimeModel.ITER_IPB98Y2: IterIpb98y2ConfinementTime,
}
"""`i_confinement_time` -> the scaling-law occupant.

**Keyed on the law, not on the device**, which is the change: `CONFINEMENT_TIME` was
`{6: StellaratorConfinementTime}` keyed on `istell`, and answered a question it was not
really asking. `StellaratorConfinementTime` differed from its base in exactly one read
binding -- PROCESS hands ISS04 the rotational transform through a parameter its own
source calls `q95` -- so with an occupant per law the binding follows from the law
(`iss04_stellarator_confinement_time`'s parameter *is* `iotabar`) and the device drops
out of the question entirely.

Two entries for ~40 reachable values, by `switch_kwarg_survey.md` band (d)'s rule: an
occupant per value **this port supports**. 38 is the Helias run's, 34 the conventional
tokamak's, and 34 is one of the four values `_audit/tokamak_scope.md` found the tree
contradicting.
"""

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
"""`i_tf_sc_mat` -> the occupant of `stellarator.coils.coils_mass`.

**The same key as `WINDING_PACK_MATERIAL`, deliberately**: the two nodes answer one
switch and must not disagree. Until `_audit/next_steps.md` §14.2 this one did not
answer it at all -- `mass.py` carried a module constant `I_TF_SC_MAT_ITER_NB3SN = 1`
baked into a `FromExactly(tfcoil.dcond[0])` default, which `switch_audit` cannot see
because it walks `eqx.field(static=True)` and a module constant is not one. The eight
occupants differ in exactly one read, `.tfcoil.dcond[k]`."""

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

Eight of nine, one per branch `jcrit_from_material` implements; `HAZELTON_ZHAI_REBCO`
(9) is in `UNPORTED` because PROCESS's own dispatch has no branch for it either.

**The registry that deletes an invented cycle.** This was an
`i_tf_sc_mat=SuperconductorModel.ITER_NB3SN` static kwarg on one node that branched
internally and therefore declared all eight branches' reads at once -- six of them dead
at this run's value, and one of the six, `.tfcoil.j_tf_wp`, machine-checked to be the
sole back-edge closing the four-node coils SCC (`_audit/switch_kwarg_survey.md` §4.6).
Only `Bi2212WindingPackIntersectInputs` declares it. See `_audit/next_steps.md` §14.5.
"""

CONFINEMENT_TAIL = {
    ConfinementRadiationLossModel.CORE_ONLY: ConfinementTailCoreRadiation
}
"""`i_rad_loss` -> the occupant owning everything downstream of the law.

One entry: the other two arms read different variables (`FULL_RADIATION` reads total
radiated power where this reads synchrotron plus inner) and neither is written yet.
"""


def _plasma_power_loss_arm(i_plasma_ignited: int, i_rad_loss: int) -> int:
    """`(i_plasma_ignited, i_rad_loss)` -> the head's arm.

    A joint dispatch, in the same shape as `_blanket_shield_power_arm`: the head adds
    injected heating when the plasma is not ignited and subtracts one of two radiation
    terms, so neither switch decides it alone. Only the combination both reference runs
    use is written; anything else falls to `UNPORTED` through `_slot_occupant`.
    """
    ignited = PlasmaIgnitionModel(int(i_plasma_ignited))
    radiation = ConfinementRadiationLossModel(int(i_rad_loss))
    if radiation is ConfinementRadiationLossModel.CORE_ONLY:
        return 0 if ignited is PlasmaIgnitionModel.IGNITED else 1
    return -1


def _cryo_q_nuc_arm(inuclear: int, i_tf_sup: int) -> int:
    """`(inuclear, i_tf_sup)` -> whether anything owns `.fwbs.qnuc`.

    PROCESS computes it only when both hold; otherwise its own comment applies --
    *"Issue #511: if inuclear = 1: qnuc is input"* -- and an input is what an **empty
    slot** means here. Arm `1` is therefore `None`, not a refusal: an unowned read is a
    correct answer for this field, and saying so structurally is what replaced a
    `FixedPoint` whose residual `sand.degenerate_fixed_points` had to differentiate at
    runtime to discover was the identity.
    """
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

    `power.py:1985-2046`, transcribed. Five `i_thermal_electric_conversion` values,
    three of which nest an `i_blanket_type` test, and **four of the eight resulting
    arms are `return eta_turbine`** -- the efficiency is a user input there:

    ```
    (CCFE_HCPB_VALUE, CCFE_HCPB)               -> arm 0   the literal 0.411
    (CCFE_HCPB_VALUE_WITH_DIVERTOR, CCFE_HCPB) -> arm 1   0.411 - delta_eta
    (STEAM_RANKINE_CYCLE, CCFE_HCPB)           -> arm 2   the Rankine log fit
    (SUPERCRITICAL_CO2_BRAYTON_CYCLE, any)     -> arm 3   the CO2 log fit
    everything else                            -> arm 4   nothing owns it
    ```

    Arm `4` is `None`, not a refusal, for the same reason `_cryo_q_nuc_arm`'s arm 1 is:
    PROCESS's own body there is `return eta_turbine`, and "the value it already had" is
    what an empty slot means. It is the reference run's arm (`USER_INPUT`), which is why
    splitting this switch is what `switch_kwarg_survey.md` §4.7 predicted would remove
    a `FixedPoint` that determines nothing.
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
"""`_eta_turbine_arm(...)` -> the `.heat_transport.eta_turbine` occupant, or `None`.

Arm `0` is a node with **no inputs at all**; the other three read one or two fields
each, where the single switch-carrying node declared all three."""

ETATH_LIQ = {
    ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE: (
        EtathLiqSupercriticalCo2
    ),
    ElectricConversionModelTypes.USER_INPUT: None,
}
"""`.fwbs.secondary_cycle_liq` -> the `.heat_transport.etath_liq` occupant, or `None`.

Keyed on the switch itself rather than an arm index, because PROCESS accepts only two
values here (`power.py:2112-2115` raises otherwise) and both are named. `USER_INPUT`
(2) is the "the efficiency is an input" arm and is `None`."""


def _temp_turbine_coolant_in_arm(
    i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq
) -> int:
    """`(i_thermal_electric_conversion, i_blanket_type, secondary_cycle_liq)` -> who
    owns `.heat_transport.temp_turbine_coolant_in`, if anyone.

    Two stages write it in order (`power.py:1985-2046` then `:2073-2116`), and the
    second **overwrites** the first:

    ```
    secondary_cycle_liq == 4                  -> arm 0   outlet_temp_liq - 20
    else, stage one writes it                 -> arm 1   temp_blkt_coolant_out - 20
    else                                      -> arm 2   nothing owns it
    ```

    "Stage one writes it" is `(STEAM_RANKINE_CYCLE, CCFE_HCPB)` or
    `SUPERCRITICAL_CO2_BRAYTON_CYCLE` -- i.e. arms 2 and 3 of `_eta_turbine_arm`, which
    is why this reads that function rather than restating its condition.
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

    `power.py:955-961` recomputes it on every value except
    `MECHANICAL_WITH_PRESSURE_DROP`, where it passes the entering value through -- and
    the field's only other producer anywhere in `process/` is `models/ife.py`, which is
    out of scope. So arm `1` is `None`: on that value the field is a boundary input, and
    saying so is what stops this node being a `FixedPointFunction`.
    """
    return (
        1
        if PumpingPowerModelTypes(int(i_p_coolant_pumping))
        is PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP
        else 0
    )


P_FW_DIV_HEAT_DEPOSITED = {0: PFwDivHeatDepositedMwSummed, 1: None}
"""The `.heat_transport.p_fw_div_heat_deposited_mw` ownership arm -> its occupant, or
`None`."""


def _p_fw_blkt_coolant_pump_arm(i_p_coolant_pumping) -> int:
    """`.fwbs.i_p_coolant_pumping` -> who owns
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`.

    `process/models/power.py:815-820` writes it only on `USER_INPUT` and
    `FRACTION_OF_HEAT`; on `MECHANICAL` and `MECHANICAL_WITH_PRESSURE_DROP` the field
    arrives from `process/models/blankets/hcpb.py` instead. So arm `1` is `None` -- not a
    refusal, because *something* does own the field on those arms, just not this node.

    **This is the first dual-ownership conflict in the port that two subsystems actually
    collided over**, rather than one being noted as a correspondence: `power`'s node and
    `.tokamak.ccfe_hcpb.pumping_power` both declared the `VarPath`, and cottax refused
    the graph by name. The stellarator never saw it because `stellarator_helias.IN.DAT`
    sets `i_p_coolant_pumping = 1`, the arm on which `power` genuinely owns it.
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
`None` for "the blanket owns it". See `_p_fw_blkt_coolant_pump_arm`."""


def _energy_storage_arm(i_pulsed_plant: int, istore: int) -> int:
    """`(i_pulsed_plant, istore)` -> Account 225.3's arm.

    A continuous plant never enters the `istore` dispatch at all, so `istore` is only a
    question once the plant is pulsed -- which is why this is a joint arm and not two
    slots: asking `istore` of a steady-state plant has no answer to be wrong about.
    """
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
"""Account 225.3's arm -> its occupant.

**Three arms now, not two.** `istore`'s two ported values read the same variables and
differ only in a literal, which `switch_kwarg_survey.md` band (c) argued made this the
one case where a static kwarg was right. `_audit/next_steps.md` §14.2 withdrew that: a
switch value selects an occupant whatever its reads. The two occupants are
indistinguishable by ports and `test_occupants_of_one_slot_differ` no longer asks them
to be -- it asks for distinct classes, and says in its own docstring that nothing now
catches a family whose members differ only in a literal."""
"""The `.fwbs.qnuc` arm -> its occupant, or `None` for "nothing owns it"."""


PLASMA_POWER_LOSS = {
    0: PlasmaPowerLossIgnitedCoreRadiation,
    1: PlasmaPowerLossNonIgnitedCoreRadiation,
}
"""The head's arm index -> its occupant. See `_plasma_power_loss_arm`.

**Two arms now, not one, and the second is what a tokamak needs.**
`large_tokamak_eval.IN.DAT` sets neither `i_plasma_ignited` nor `i_rad_loss`, so both
take PROCESS's own defaults -- `NON_IGNITED` (`physics_variables.py:881`) and
`CORE_ONLY` (`i_rad_loss = 1`) -- and the ignited arm the Helias run uses is the wrong
one for it by exactly one read, `.current_drive.p_hcd_injected_total_mw`. That single
term is the `.current_drive -> .physics` edge the confinement split exists to keep out
of the arm that does not make it, so assembling the ignited occupant for a
non-ignited machine would have been the invented-edge defect *and* a wrong number.

The four combinations involving `FULL_RADIATION` and `NO_RADIATION` are still refused;
the reason in `UNPORTED` was written when all five non-ignited-core combinations were,
and now covers four."""

HEATING = {1: EcrhHeating, 2: LowhybHeating}
""".stellarator.isthtr` -> the auxiliary-heating occupant."""

FW_AREA = {0: AFwTotalNoPowerflow, 1: AFwTotalWithPowerflow}
"""`.heat_transport.ipowerflow` -> the first-wall-area occupant."""

BETA_NORM_MAX = {0: None, 1: BetaNormMaxWesson}
"""`.physics.i_beta_norm_max` -> `.tokamak.plasma_beta.norm_max`'s occupant.

`1` (`WESSON`, `physics_variables.py`'s own default -- `large_tokamak_eval.IN.DAT` sets
`i_beta_component` and never mentions this switch) is a node. `0` (`USER_INPUT`) is
**`None`, an occupant and not a refusal**: `get_beta_norm_max_value`'s `model_map`
returns `physics_data.beta_norm_max` itself, so the arm's honest occupant is *no node*
and `.physics.beta_norm_max` staying a boundary input, exactly as PROCESS leaves it. The
other four values are formulas nobody has transcribed and live in `UNPORTED`.

**The `None` arm landed in the ST closing wave (2026-08-29), and the frontier probe is
why.** Until then this docstring recorded the gap and asserted "nothing in the tokamak or
stellarator scope selects `0`" -- which was false for both tracked spherical tokamaks
(`spherical_tokamak_eval.IN.DAT:265`, `st_regression.IN.DAT:323`), and
`machine_survey.report` did not say so: the survey checked `UNPORTED` only, so a value
that is in neither the registry nor `UNPORTED` reported as "the factory dispatches on
it". That blind spot is fixed in `machine_survey.slot_registries`; this entry is the
value it was hiding."""

PROFILE_PARAMETERISATION = {
    0: ProfileParameterisationParabolic,
    1: ProfileParameterisationPedestal,
}
"""`.physics.i_plasma_pedestal` -> the profile-shape occupant.

Both arms are real occupants and both assemble. On a **stellarator** only the parabolic
one is reachable through `machine_from_indat`, and that is `ST_INIT_I_PLASMA_PEDESTAL`'s
doing, not this registry's; on a **tokamak** the file decides, and
`large_tokamak_eval.IN.DAT:291` picks the pedestal arm. So both arms are now reached by a
real input file rather than only by an `eqx.tree_at` what-if.
"""


PEDESTAL_SEPARATRIX = {
    0: GreenwaldDensityFractions,
    1: PedestalSeparatrixDensities,
}
"""`.physics.i_nd_plasma_pedestal_separatrix` -> the pedestal/separatrix-density
occupant, **nested under `i_plasma_pedestal == 1`**.

Both values of the binary switch are written, and both arms are real occupants that
assemble -- but they are *inverses*, not competitors: `1` (`GREENWALD_FRACTION`,
PROCESS's default and both reference files') reads the two Greenwald fractions and owns
the two densities, `0` (`USER_INPUT`) reads the densities and owns the fractions. See
`ProfileParameterisationPedestal.pedestal_separatrix` for why that makes a slot default
wrong rather than merely unnecessary.

`_profile_parameterisation` below reaches this registry only on the pedestal arm, which
is how the port spells "this switch only exists when that one has this value" --
`profiles.md`'s open question 2, answered by the slot mechanism rather than by an
addition to `configuration.TOPOLOGY_SWITCHES`."""


def _profile_parameterisation(
    i_plasma_pedestal, i_nd_plasma_pedestal_separatrix, *, is_stellarator
):
    """The profile-shape occupant, with each arm's own nested slot filled.

    Three questions, one slot, and the first two are genuinely independent:
    `i_plasma_pedestal` decides which *arm* runs, and the device decides whether that
    arm's `ecrh_density_limit` exists. `st_d_limit_ecrh` lives in
    `models/stellarator/density_limits.py` and is reached only from `st_phys`, so a
    parabolic **tokamak** computes no ECRH density limit any more than a pedestal one
    does -- `None`, and `.stellarator.dlimit_ecrh`/`bt_max_ecrh` surface as boundary
    inputs, which is what PROCESS leaves them as.

    The third is *not* independent, and that is the point of answering it here:
    `i_nd_plasma_pedestal_separatrix` decides a slot that only the **pedestal** arm has
    (`physics.py:363-368` reads it inside `if i_plasma_pedestal == PEDESTAL_PROFILE`).
    Asking it on the parabolic arm would be asking a question PROCESS never asks, so
    the parabolic branch below never touches `PEDESTAL_SEPARATRIX` -- and a stellarator,
    pinned to the parabolic arm by `ST_INIT_I_PLASMA_PEDESTAL`, therefore cannot reach
    it at all.

    The static `i_plasma_pedestal=PARABOLIC_PROFILE` is written here, once, immediately
    beside the arm that selects it -- it used to be a slot default in
    `models/physics/namespace.py`, which could not express the device half of the
    question.
    """
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

`process/models/stellarator/initialization.py:31` -- `st_init`, which runs on every
`istell != 0` run -- assigns `data.physics.i_plasma_pedestal = 0` unconditionally, in
the same block that zeroes the central solenoid (`data.build.iohcl = 0`, `:24`). So the
file's value is **dead** on this device: an IN.DAT saying `istell = 6,
i_plasma_pedestal = 1` runs parabolic profiles in PROCESS, and the factory used to read
that `1` and assemble `ProfileParameterisationPedestal` for it -- a configuration
PROCESS cannot produce.

**Read from the forcing rather than from the file, and not refused.** The two honest
options were to pin the arm or to reject a file whose value `st_init` will overwrite;
pinning is what reproduces PROCESS. Refusing would make this port decline an input file
PROCESS runs happily, and the factory's job is to model the run, not to police the file.
That the file's value is ignored is said here, in the docstring of the constant that
ignores it, and pinned by
`test_switch_coverage.test_a_process_forced_switch_cannot_move_the_machine`.

`switch_kwarg_survey.md` §7 records the same shape for `iohcl`, which no test that
compares against the input file can see at all, because neither the file nor the factory
ever mentions it.
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
"""`.costs.ibkt_life` -> the component-lifetime occupant.

Both values ported. The neutron-fluence occupant reads neither `.costs.life_dpa` nor
`.physics.p_fusion_total_mw`; the displacement-damage one reads neither
`.costs.abktflnc` nor `.physics.pflux_fw_neutron_mw`."""


def _cplife_arm(itart: int, i_tf_sup: int) -> int:
    """`(itart, i_tf_sup)` -> who owns `.costs.cplife`, if anyone.

    `availability.py`'s `calculate_cplife_next`, transcribed:

    ```
    itart != 1               -> arm 0   nothing owns it; the field is an input
    itart == 1, i_tf_sup == 1 -> arm 1   the superconducting centrepost lifetime
    itart == 1, i_tf_sup != 1 -> arm 2   the resistive one
    ```

    Arm `0` is `None`, not a refusal, for the same reason `_cryo_q_nuc_arm`'s is:
    PROCESS's own body on that arm is `return cplife`, and "the value it already had" is
    what an **empty slot** means. Splitting this slot is what removed the `FixedPoint`
    -- the self-read existed only on the arm that is a pass-through.
    """
    if SphericalTokamakModel(int(itart)) is not SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return 0
    return (
        1 if TFConductorModel(int(i_tf_sup)) is TFConductorModel.SUPERCONDUCTING else 2
    )


CPLIFE = {0: None, 1: CplifeAvailSuperconducting, 2: CplifeAvailResistive}
"""The `.costs.cplife` arm -> its occupant, or `None` for "nothing owns it"."""


def _cryo_q_loads_arm(i_tf_sup, i_pf_conductor) -> int:
    """`(i_tf_sup, i_pf_conductor)` -> who owns `.power.qss`/`qac`/`qcl`/`qmisc`.

    `power.py:1054-1057` calls `Power.cryo` only when
    `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING`; outside that guard the four
    fields keep the values they entered with, which is an **empty slot**.

    ```
    i_tf_sup == 1                                 -> arm 0   TF terms present
    i_pf_conductor == SUPERCONDUCTING (otherwise)  -> arm 1   PF coils only
    neither                                        -> arm 2   nothing owns them
    ```
    """
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
    """`(i_tf_sup, i_pf_conductor)` -> the cryoplant-load occupant.

    The same guard as `_cryo_q_loads_arm`, but with a different consequence: these four
    fields are written on **every** path (`power.py:1049-1050` zeroes two of them before
    the guard), so there is no absent arm -- arm `1` computes literal zeros rather than
    nothing.

    Aluminium TF (`i_tf_sup == 2`) is a third arm in PROCESS and has no occupant here;
    it is refused earlier, at the `power.tf_power` slot, which is why this function is
    written as a two-way question.
    """
    return 0 if _cryo_q_loads_arm(i_tf_sup, i_pf_conductor) != 2 else 1


CRYO_LOADS = {0: CryoLoadsActive, 1: CryoLoadsInactive}
"""`_cryo_loads_arm(...)` -> the cryoplant-load occupant."""


ACPOW = {
    PFEnergyStorageSource.LINE: AcpowLine,
    PFEnergyStorageSource.MGF: AcpowMotorGeneratorFlywheel,
}
"""`.pf_power.i_pf_energy_storage_source` -> the plant AC power occupant.

Two of three: `MGF_PF_LINE_HEATING` (3) is in `UNPORTED`, for the reason
`('i_tf_sup', 2)` is -- PROCESS runs the byte-identical branch to `MGF`, and registering
a second entry pointing at the same class would state a distinction the arithmetic does
not have."""


TF_POWER = {0: TfPowerResistive, 1: TfPowerSuperconducting}
"""`.tfcoil.i_tf_sup` -> the TF-power occupant."""


def _electric_production_arm(
    ireactor: int, itart: int, i_tf_sup: int, i_blkt_dual_coolant, i_p_coolant_pumping
) -> int:
    """`(ireactor, itart, i_tf_sup, i_blkt_dual_coolant, i_p_coolant_pumping)` -> the
    electric-production arm.

    `power.py:1631-1772`, transcribed. Three questions, and the second and third are
    nested inside the first:

    ```
    ireactor != 1                                    -> arm 0   profiles only
    itart == 1 and i_tf_sup == 0                     -> +2      centrepost pump power
    i_blkt_dual_coolant > 0 and pumping == MECHANICAL -> +1      liquid-breeder turbine
    ```

    so arms 1..4 are `1 + 2 * centrepost + liquid`. Five switches, **two** conditions:
    neither is decided by one switch alone, which is why they are arm indices and not
    nested slots -- `switch_kwarg_survey.md` §3 reports both as "(joint)" for the same
    reason. `ireactor == 0` asks neither, because `PowerProfilesOverTime` computes
    neither the centrepost pump power nor the gross electric power.
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
"""`_electric_production_arm(...)` -> the electric-production occupant. Keyed by the
**arm index** that function documents, never by a switch value.

**Five arms now, not two.** All four of `PlantElectricProductionReactor`'s static
kwargs are gone (`_audit/next_steps.md` §14.2); the reference machine's arm is `1`,
which declares neither `.tfcoil.p_cp_coolant_pump_elec` nor `.heat_transport.etath_liq`
nor `.power.p_blkt_liquid_breeder_heat_deposited_mw` -- three edges no such machine
makes."""


def _no_cost_of_electricity():
    """The absent occupant of `costs.cost_of_electricity`: `ireactor != 1 or ipnet != 0`.

    `None`, and nothing else. PROCESS does not call `coelc()` on this arm at all, so
    `.costs.coe` and its five companions keep their entering values and surface as
    boundary inputs -- see that slot's own docstring for why absence is the honest
    occupant here and a refusal is not.
    """
    return None  # noqa: RET501 -- the returned `None` is the occupant, not a fall-off


TF_MAGNET_COST_SUPERCONDUCTING = {
    SuperconductorCostModel.PER_KG: TfMagnetCostSuperconductingPerKg,
    SuperconductorCostModel.PER_KAM: TfMagnetCostSuperconductingPerKam,
}
"""`.costs.supercond_cost_model` -> the Account 222.1 occupant.

Both values are ported, so this registry is total and `UNPORTED` has no entry for the
switch. It was an `eqx.field(static=True)` until `_audit/next_steps.md` §14.2: the two
arms are two one-line strand-cost formulas over **disjoint** fields, so the single node
declared `.costs.sc_mat_cost_0`, `.tfcoil.j_crit_str_0` and `.tfcoil.j_crit_str_tf` --
three edges the reference run does not make."""


COST_OF_ELECTRICITY = {
    0: _no_cost_of_electricity,
    1: CostOfElectricityConventionalAspectRatio,
    2: CostOfElectricitySphericalTokamak,
}
"""`_cost_of_electricity_arm(ireactor, ipnet, itart)` -> the cost-of-electricity
occupant, or `None`. Keyed by the **arm index** that function documents, never by a
switch value -- the same discipline the two blanket dispatches follow.

**Three arms now, not two.** `itart` was a static kwarg on the single occupant, along
with `ireactor`, `ipnet` and `ife`; under `_audit/next_steps.md` §14.2 none of the four
may be. Three of them were answering questions this slot had already answered, but
`itart` was a real branch: `costs.py:2769-2783`'s centrepost replacement cost exists
only on a spherical tokamak, so the one-occupant slot read `.costs.cplife_cal`,
`.costs.cpstcst` and `.costs.cplife` on a machine that reads none of the three."""


def _cost_of_electricity_arm(ireactor: int, ipnet: int, itart: int) -> int:
    """Which arm of `Costs.run()`'s cost-of-electricity dispatch three switches select.

    `process/models/costs/costs.py:82-83` and `:2769-2783`, transcribed:

    ```
    if ireactor != 1 or ipnet != 0:   -> arm 0   nothing is computed
    elif itart == 1:                  -> arm 2   CostOfElectricitySphericalTokamak
    else:                             -> arm 1   CostOfElectricityConventionalAspectRatio
    ```

    `ireactor`/`ipnet` are one condition, so one arm index rather than two keys:
    `ireactor == 0` ("do not calculate MW(electric) or c-o-e",
    `cost_variables.py:521-525`) and `ipnet == 1` ("let go < 0 (no c-o-e)", `:515-519`)
    are two ways of saying the same thing to the same `if`, and neither PROCESS nor this
    port distinguishes them downstream. `itart` is a second, *nested* question -- there
    is no centrepost to replace on a run that computes no cost of electricity -- which is
    why it joins this arm index rather than opening a sub-slot. Arm 1 is PROCESS's own
    defaults (`ireactor = 1`, `ipnet = 0`, `itart = 0`) and the reference run.
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
}
"""`_blanket_shield_power_arm(blktmodel, ipowerflow)` -> the blanket/shield-power
occupant. Keyed by the **arm index** that function documents, never by a switch value."""

BLANKET_MASSES = {2: BlanketComponentMasses}
"""`_blanket_mass_arm(blktmodel, blkttype)` -> the blanket-mass occupant, same kind of
key."""


def _blanket_shield_power_arm(blktmodel: int, ipowerflow: int) -> int:
    """Which arm of `st_fwbs`'s blanket/shield-power dispatch a pair of switches selects.

    `stellarator.py:608-...`, transcribed:

    ```
    if blktmodel == 1:              -> arm 0   blanket_neutronics(); UNPORTED
    else:                           # blktmodel == 0
        if ipowerflow == 0:         -> arm 1   BlanketShieldPowerExponential
        else:                       -> arm 2   DetailedPowerflowBlanketShieldPower
    ```

    So `blktmodel` is the **outer** test and `ipowerflow` only distinguishes the two
    arms *inside* `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`,
    `ipowerflow = 1`) and the reference run.
    """
    if blktmodel == 1:
        return 0
    return 2 if ipowerflow == 1 else 1


def _blanket_mass_arm(blktmodel: int, blkttype: int) -> int:
    """Which arm of `st_fwbs`'s blanket-mass dispatch a pair of switches selects.

    `stellarator.py:1056-1091`, transcribed:

    ```
    if blktmodel == 0:
        if blkttype in {1, 2}:      -> arm 1   liquid breeder (WCLL/HCLL); UNPORTED
        else:                       -> arm 2   BlanketComponentMasses (solid breeder)
    else:                           # blktmodel == 1
                                    -> arm 0   sub-assembly thicknesses; UNPORTED
    ```

    Again `blktmodel` is the outer test; `blkttype` is consulted only inside
    `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`, `blkttype = 3`)
    and the reference run. `blkttype`'s values 1 and 2 select the identical formula, so
    they share arm 1.
    """
    if blktmodel != 0:
        return 0
    return 1 if blkttype in {1, 2} else 2


COST_MODEL = {0: Costs}
"""`.costs.i_cost_model` -> the cost-model occupant. `1` (KOVARI_2014, PROCESS's own
default) and `2` are both refused, with their reasons in `UNPORTED`; the slot used to
default to `None` for the first of them and no longer can."""

DEVICE = {0: TokamakProcess, 6: StellaratorProcess}
"""`.stellarator.istell` -> the **device class**, and the first thing the factory reads.

The one registry whose values are classes rather than occupants, because what `istell`
selects is not a slot's occupant but which tree has slots at all. `_slot_occupant` is
still what looks it up -- with `build=lambda cls: cls`, since a device is constructed at
the end of the factory and not here -- so `istell in 1..5` keeps raising with
`_ISTELL_PRESET_REASON` and a value PROCESS has never had keeps failing loudly.

`0` is here rather than in `UNPORTED` as of the pass that built `TokamakProcess`; see
`UNPORTED`'s own docstring for why the recorded reason no longer describes what a tokamak
machine assembles.
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

    **Derived, not read.** `.divertor.n_divertors` is a `DataStructure` field with a
    default of `2` (`divertor_variables.py:94`), and that default is *never* what a run
    sees: `process/core/init.py` overwrites it from `.physics.i_single_null` before any
    model runs. A factory that read the field's own default would pick the double-null
    arm for a single-null machine -- the `ST_INIT_I_PLASMA_PEDESTAL` shape again, a
    field whose entering value is dead because PROCESS's own initialisation assigns it.

    Eight slots in this port are keyed on the result -- two in `blanket_library.py`, two
    in `hcpb.py`, one each in `fw.py`, `vacuum.py`, `divertor.py` and `shield.py` -- and
    every one of their audit records independently traced the derivation back to these
    same eleven lines. Since 2026-08-27 **all eight are total**: both values have an
    occupant everywhere, so no refusal keys on `n_divertors` any more.

    **`n_divertors` is read two ways in one wave, and both are correct.**
    `DivertorHeatFluxSplit` reads it as a plain multiplier and takes it as an ordinary
    input port; `divwade`, `hcpb.py:360` and four others *branch* on it and it selects
    their occupant. That is the policy `blanket_library.md` and `hcpb.md` both asked for
    in one line: **a switch read arithmetically is an ordinary input; a switch read to
    branch selects an occupant.** Nothing more is needed -- the two uses do not conflict,
    because a port and a slot key are different things.
    """
    return (
        2
        if DivertorNumberModels(int(i_single_null)) is (DivertorNumberModels.DOUBLE_NULL)
        else 1
    )


def _fw_blkt_vv_shape_arm(itart: int, i_fw_blkt_vv_shape: int) -> int:
    """`(itart, i_fw_blkt_vv_shape)` -> the first-wall/blanket/vessel shape arm.

    `process/models/blankets/blanket_library.py:90-93`, and the identical predicate at
    `fw.py:58-86` and `vacuum.py:758-791`:

    ```
    if itart == 1 or i_fw_blkt_vv_shape == D_SHAPED:  -> arm 0   D-shaped
    else:                                             -> arm 1   elliptical
    ```

    A joint arm rather than two keys, for the reason `blanket_library.md` gives and
    `switch_kwarg_survey.md` §4.3 prescribes: one arm is selected by two switches, so
    the pair becomes an index and neither integer is ever used as a key. Three separate
    audit records reached this predicate independently and agreed on it, which is why it
    is written once here and read by **five** slots -- `BLANKET_AREAS`,
    `BLANKET_VOLUMES` and `SHIELD_VOLUMES` directly, `FIRST_WALL` and `VACUUM_VESSEL`
    through `_first_wall_arm`/`_vacuum_vessel_arm`, which cross it with the divertor
    count.

    **Both arms are written since 2026-08-27** (the D-shaped wave, for
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`, which set
    `i_fw_blkt_vv_shape = 1` *and* `itart = 1` and so earn arm `0` twice over). Arm `0`
    used to refuse at all five slots at once and no longer refuses at any of them; the
    two slots that still have an unwritten cell refuse on the *product* with the divertor
    count, not on the shape (`('first_wall_arm', -2)`, `('vacuum_vessel_arm', -2)`).
    """
    d_shaped = (
        SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK
        or FwBlktVVShape(int(i_fw_blkt_vv_shape)) is FwBlktVVShape.D_SHAPED
    )
    return 0 if d_shaped else 1


def _plasma_geometry_arm(i_plasma_current: int, i_plasma_shape: int) -> int:
    """`(i_plasma_current, i_plasma_shape)` -> the plasma-geometry arm.

    `process/models/physics/plasma_geometry.py:467-470`:

    ```
    if i_plasma_current == 8 or i_plasma_shape == SAUTER:  -> arm 1   Sauter; UNPORTED
    else:                                                  -> arm 0   double arc
    ```

    **This function is the single owner of that disjunction**, and that is a coordination
    requirement rather than tidiness. `plasma_geometry.md` OQ2 says so explicitly: the
    pass that ports `plasma_current.py`'s own `i_plasma_current` topology split shares
    this predicate, and two independent derivations of one boolean is how the two halves
    drift apart. Call this; do not re-derive it.

    It is also the cleanest result in that record: *"a compound switch does not have to
    become a compound node, it becomes one predicate evaluated once by the assembler."*
    """
    sauter = (
        PlasmaCurrentModel(int(i_plasma_current)) is PlasmaCurrentModel.SAUTER_SCALING
        or PlasmaShapeModelType(int(i_plasma_shape)) is PlasmaShapeModelType.SAUTER
    )
    return 1 if sauter else 0


def _tf_shape(i_tf_shape: int, itart: int) -> TFCoilShapeModel:
    """`.tfcoil.i_tf_shape`, with `0` resolved the way `init.py` resolves it.

    `process/core/init.py:728-729` and `:775-776` replace the `DEFAULT` (`0`,
    "auto-select") value **before any model runs**: picture frame on a spherical tokamak,
    D-shape otherwise. So `0` is not a third arm, it is a request to be told which of the
    two real arms this machine takes.

    **Auto-select meta-values resolve in the factory, and get no occupant of their own.**
    That is a policy decision this pass makes, and it is worth stating once because
    `i_tf_wp_geom`'s `UNSET` below is the same shape: a switch value that PROCESS's own
    initialisation *replaces* names no arm, so there is nothing for an occupant to be
    written for. The alternative -- an occupant per meta-value -- would duplicate whichever
    real arm it resolves to under a second name, which is exactly what `build.md`'s open
    question 1 declined to do for `i_tf_shape == 0` and what this answers.
    """
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
    """`.tfcoil.i_tf_wp_geom`, with `UNSET` resolved the way `init.py:977-989` does.

    The same shape as `_tf_shape` above: `-1` is not an arm, it is PROCESS asking
    `i_tf_turns_integer` instead -- rectangular for integer turns, double-rectangular
    otherwise. `large_tokamak_eval.IN.DAT` sets neither, so `UNSET` plus `NON_INTEGER`
    resolves to `DOUBLE_RECTANGULAR`, and a factory that took the raw `-1` would have no
    occupant to offer at all.
    """
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
"""`.physics.i_plasma_geometry` -> the kappa95/triang95 occupant.

Two of thirteen. `plasma_geometry.md`'s open question "eight occupants or thirteen?" is
**superseded rather than answered** by this wave's binding policy: one class per value
ever supported, no grouping by reads-identical sets, so the other eleven are eleven
future classes and not a family to be split later. The second entry
(`CREATE_DATA_EU_DEMO_X_POINT`, 10, `low_aspect_ratio_DEMO.IN.DAT:372`) is the first
occupant to exercise the record's "conditional-ownership-by-run-config" finding through
this registry: it owns `.physics.kappa` where the IPDG89 occupant reads it."""

PLASMA_GEOMETRY = {0: DoubleArcPlasmaGeometry}
"""`_plasma_geometry_arm(i_plasma_current, i_plasma_shape)` -> the geometry occupant."""

# ---- `.tokamak.plasma_fields` and `.tokamak.physics` ------------------------------


def _surface_poloidal_field_arm(i_plasma_current: int) -> int:
    """`i_plasma_current` -> the poloidal-field arm. `plasma_fields.py:83` tests `!= 2`.

    Two arms and not nine. PROCESS's own test is binary, so nine occupants would invent
    eight distinctions the source does not make -- the mirror image of the usual
    complaint, and worth naming: the binding policy says one occupant per *value the
    port supports*, and what a value is depends on what the source branches on.
    """
    return (
        1
        if PlasmaCurrentModel(int(i_plasma_current))
        is PlasmaCurrentModel.PENG_DIVERTOR_SCALING
        else 0
    )


SURFACE_POLOIDAL_FIELD = {0: SurfaceAveragedPoloidalFieldAmperes}
"""The poloidal-field arm -> its occupant. Ampere's law over the plasma perimeter."""

SEPARATRIX_POWER = {PlasmaIgnitionModel.NON_IGNITED: SeparatrixPowerNonIgnited}
"""`.physics.i_plasma_ignited` -> the separatrix-power occupant.

The **`NON_IGNITED`** arm, which is the opposite of the arm `PLASMA_POWER_LOSS` answers
for the same switch on the stellarator runs. Both are correct; "the live arm" is a
property of a machine, not of a switch."""


def _pulse_ramp_times_arm(
    i_pulsed_plant: int, pulsetimings: int, i_t_current_ramp_up: int
) -> int:
    """`(i_pulsed_plant, pulsetimings, i_t_current_ramp_up)` -> the ramp-time arm.

    `process/models/physics/physics.py:463-498`, transcribed:

    ```
    if i_pulsed_plant != 1:
        if i_t_current_ramp_up == 0:  -> arm 0   ramp times from plasma_current / 5e5,
                                                 plus t_plant_pulse_coil_precharge
        else:                         -> arm 1   nothing is computed; all three are inputs
    else:
        if pulsetimings == 0:         -> arm 2   ramp-up = plasma_current / 1e5   (live)
        else:                         -> arm 3   precharge ratchets; UNPORTED, D3
    ```

    `pulsetimings` has its **only read in all of `process/models/**`** at `:476`, so this
    arm index is the whole of that topology decision. Arm 2 is the reference run's
    (`i_pulsed_plant = 1` at `large_tokamak_eval.IN.DAT:330`, `pulsetimings = 0` at
    `:392` -- and note the second is a *file* setting against PROCESS's own default of
    `1`, so this arm exists only because the file asks for it).
    """
    if PlantOperationModel(int(i_pulsed_plant)) is PlantOperationModel.CONTINUOUS:
        return 0 if int(i_t_current_ramp_up) == 0 else 1
    return 2 if int(pulsetimings) == 0 else 3


PULSE_RAMP_TIMES = {
    0: PulseRampTimesContinuousDefault,
    2: PulseRampTimesPulsedDefault,
}
"""The ramp-time arm -> its occupant. See `_pulse_ramp_times_arm`.

Arm 0 is the spherical tokamaks' (`i_pulsed_plant = 0` at
`spherical_tokamak_eval.IN.DAT:312` and `st_regression.IN.DAT:2979`,
`i_t_current_ramp_up` left at its default `0`); arm 2 is `large_tokamak_eval`'s."""

# ---- `.tokamak.current_drive` -----------------------------------------------------

HCD_PRIMARY_EFFICIENCY = {
    CurrentDriveModel.USER_INPUT_ELECTRON_CYCLOTRON: HcdPrimaryEfficiencyUserInputEcrh
}
"""`.current_drive.i_hcd_primary` -> the primary current-drive efficiency occupant.

One of thirteen values, and two of the eleven refusals are refusals PROCESS shares:
`CULHAM_LOWER_HYBRID` (6) and `CULHAM_ELECTRON_CYCLOTRON` (7) **cannot execute in
PROCESS at all** -- `calculate_profile_y` returns `None` and both arms raise
`TypeError`. Two live defects found by porting, recorded in `current_drive.md` and in
`UNPORTED` below.

`FREETHY_ELECTRON_CYCLOTRON` (13) is not in this registry although it is (partly)
ported: it is the one value with a switch nested *inside* it, so
`_hcd_primary_efficiency` routes it to `HCD_PRIMARY_EFFICIENCY_FREETHY` instead."""

HCD_PRIMARY_EFFICIENCY_FREETHY = {0: HcdPrimaryEfficiencyFreethyEcrhOMode}
"""`.current_drive.i_ecrh_wave_mode` -> the Freethy ECCD occupant, given
`i_hcd_primary == 13`. `0` is O-mode, the value both spherical tokamak files set
explicitly and PROCESS's default (`current_drive_variables.py:116`); X-mode (`1`) is an
`UNPORTED` refusal. Keys are plain ints because PROCESS has no enum for this switch --
`process/core/input.py:1096` declares it `int, choices=[0, 1]`."""


def _hcd_primary_efficiency(i_hcd_primary: int, i_ecrh_wave_mode: int):
    """The primary-efficiency occupant, resolving the one *nested* switch this slot has.

    `i_ecrh_wave_mode` exists only inside `i_hcd_primary == 13`: it is read at
    `current_drive.py:1767` by model 13's lambda and nowhere else in any model body
    (the only other appearance, `:2541-2542`, is the out-of-scope reporting shell), so
    the honest dispatch is a tree, not a product -- the outer registry stays keyed on
    `i_hcd_primary` (and its eleven refusals stay keyed on the switch a user would have
    to change), and only value 13 consults the inner registry. The alternative, a joint
    arm in the `_hcd_primary_powers_arm` / `i_plasma_ignited_i_rad_loss` style, is for
    dispatches where **both** switches shape every arm; here a joint key would have had
    to refuse `(1, O-mode)` and `(1, X-mode)` as distinct cells when PROCESS itself
    never reads the wave mode on model 1's arm -- two refusals for one branch, the
    invented-edge defect at the registry level. The nesting mirrors the source: the
    wave-mode `if` sits *inside* `electron_cyclotron_freethy`
    (`current_drive.py:1074-1079`), not beside `hcd_models`.

    Note what the inner switch selects is an *occupant that pins a static kwarg*, not a
    different reads-set: both wave modes read identical variables
    (`freethy_electron_cyclotron_efficiency`'s docstring carries the evidence, the unit's
    tests assert it). It still dispatches here rather than being threaded as a value
    because only O-mode has a written occupant -- the registry is where "X-mode is not
    written" can be said per `UNPORTED`'s contract.
    """
    model = CurrentDriveModel(int(i_hcd_primary))
    if model is CurrentDriveModel.FREETHY_ELECTRON_CYCLOTRON:
        return _slot_occupant(
            "i_ecrh_wave_mode", int(i_ecrh_wave_mode), HCD_PRIMARY_EFFICIENCY_FREETHY
        )
    return _slot_occupant("i_hcd_primary", model, HCD_PRIMARY_EFFICIENCY)


HCD_SECONDARY_HEATING = {CurrentDriveModel.NO_CURRENT_DRIVE: HcdSecondaryHeatingNone}
"""`.current_drive.i_hcd_secondary` -> the secondary-heating occupant. PROCESS's own
default (`current_drive_variables.py:206`), and a node that reads nothing."""


def _hcd_primary_powers_arm(i_hcd_primary: int, i_hcd_secondary: int) -> int:
    """`(i_hcd_primary, i_hcd_secondary)` -> the primary-powers arm.

    **The one genuinely combinatorial dispatch in this port**, and it is combinatorial
    because of an accumulator rather than a nested `if`: the primary block's `+=`
    (`current_drive.py:2147`) starts from whatever the *secondary* block left in the same
    technology's field (`:1955`, over the zero at `:1663`). So the arm is decided by the
    primary technology **and** the secondary technology together -- five methods by six,
    in principle, of which one cell is written.

    Keyed on `CurrentDriveModel.method` rather than on `i_hcd_primary` itself, because
    that is what the accumulator is indexed by: values `3`, `7`, `10` and `13` are all
    `ELECTRON_CYCLOTRON` and all land in the same field. That is the source's own
    grouping, not one invented here.

    `current_drive.md` names the fix and declines to make it: a per-technology
    "secondary contribution" field would turn this product back into two slots, but it
    needs a name PROCESS does not have.
    """
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

**Topology, not an occupant**, and the one registry in this file whose value is a
*namespace* rather than a node -- the same shape `DEVICE` has for a whole machine.
`1` means the heating-and-current-drive nodes exist; `0` means `physics.py:593` never
calls `CurrentDrive.run` and none of them does. It is read here rather than ignored
because a file setting `0` would otherwise assemble all seven nodes silently, which is
the `EcrhDensityLimit` bug class: a subgraph the configuration never asks for."""

HCD_ELECTRIC_TOTAL = {
    PlasmaIgnitionModel.NON_IGNITED: HcdElectricTotalNonIgnited,
    PlasmaIgnitionModel.IGNITED: HcdElectricTotalIgnited,
}
"""`.physics.i_plasma_ignited` -> the wall-plug-power occupant. **Both arms written** --
an ignited plasma draws no heating power, so its occupant reads nothing and returns
zero.

It owns `.heat_transport.p_hcd_electric_total_mw`, and that was checked against the
stellarator rather than assumed: the stellarator's heating slot owns
`.heat_transport.p_hcd_electric_loss_mw` and `.current_drive.p_hcd_injected_total_mw`
and leaves this field a boundary input, so the two devices do not collide. They could
not in any case -- ownership is a property of one assembled graph, and no graph has both
devices -- but a collision would have meant one of the two was wrong about its own
device."""

# ---- `.tokamak.build` -------------------------------------------------------------


def _divertor_geometry_arm(itart: int, dz_xpoint_divertor: float) -> int:
    """`(itart, input dz_xpoint_divertor)` -> `divgeom`'s arm.

    ```
    itart == 1, dz_xpoint_divertor <  1e-5  -> arm -1  DivertorGeometrySphericalTokamak
                                                       (1.75 * rminor; never writes
                                                       .build.rspo)
    itart == 1, dz_xpoint_divertor >= 1e-5  -> arm -3  None: the 1.75 * rminor is
                                                       computed and discarded at
                                                       build.py:800, nothing is owned
    itart == 0, dz_xpoint_divertor >= 1e-5  -> arm -2  divgeom runs for .build.rspo
                                                       alone and dz_xpoint_divertor
                                                       stays an input; UNPORTED
    otherwise                               -> arm  0  DivertorGeometryConventional
    ```

    The float condition is the only place in this factory a **float input** decides a
    slot, and it is a genuine one: `process/models/build.py:800-801` assigns
    `dz_xpoint_divertor = divht` only when the entering value is effectively zero, so
    whether a node owns that field is a run-configuration fact.
    `build.md` calls this `conditional-ownership-by-run-config` and uses the same shape to
    close `next_steps.md` §2's `dz_shld_upper` flag.

    The same latch is what splits `itart == 1` in two: `divgeom`'s early return at
    `:863` writes nothing itself, so when the run sets `dz_xpoint_divertor` -- both
    tracked spherical-tokamak inputs do, at `0.75` -- the arm owns *nothing* and the
    slot's occupant is `None`, absence rather than refusal, `DX_TF_SIDE_CASE_MIN`'s
    shape. Arm `-1` keeps the number `UNPORTED` refused it under now that it is
    written.
    """
    if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return -1 if float(dz_xpoint_divertor) < 1e-5 else -3
    return 0 if float(dz_xpoint_divertor) < 1e-5 else -2


DIVERTOR_GEOMETRY = {
    0: DivertorGeometryConventional,
    -1: DivertorGeometrySphericalTokamak,
    -3: None,
}
"""`divgeom`'s arm -> its occupant, **or `None`**. See `_divertor_geometry_arm`.

`-3` is the fourth slot in the tree spelled as absence, after
`costs.cost_of_electricity`, `power.cryo_q_nuc` and `DX_TF_SIDE_CASE_MIN`: on a
spherical tokamak whose input file sets `dz_xpoint_divertor`, PROCESS computes nothing
in `divgeom` that survives the `:800` latch, so there is no arm to refuse. Both tracked
spherical-tokamak regression inputs land here."""

DR_TF_INBOARD_WINDING_PACK = {
    0: DrTfInboardFromWindingPack,
    1: DrTfWpWithInsulationFromInboardBuild,
}
"""`140 in ixc` -> which of two **inverse** assignments `build.py` makes.

Arm 0 (`140 in ixc`) produces `.build.dr_tf_inboard` from the winding pack; arm 1
produces `.tfcoil.dr_tf_wp_with_insulation` from the inboard build. Different owned
fields, not different formulas for one field, which is why this cannot be a kwarg.

**The first slot in this port keyed on an iteration variable rather than a switch**, and
it belongs here for the same reason every switch does: `ixc` is fixed for a whole solve.
Its consequence is measured rather than assumed -- `large_tokamak_eval.IN.DAT` sets
`ixc = 4` and `ixc = 6` only, so `.build.dr_tf_inboard` stays a **boundary input** on
that run even though `tokamak_boundary.md` attributes it to this slot. That file's
attribution is an `ast` walk over `Assign` targets, which cannot see an `ixc` guard;
`build.md` records the contradiction rather than smoothing it."""


def _tf_inboard_radii_arm(i_tf_inside_cs: int, i_cs_precomp: int) -> int:
    """`(i_tf_inside_cs, i_cs_precomp)` -> the CS-to-TF radial slice's arm.

    ```
    i_tf_inside_cs == 1 (TF_INSIDE_CS)   -> arm -1  r_tf_inboard_in = dr_bore alone,
                                                    dr_cs_bore gains a TF term; UNPORTED
    i_cs_precomp == 0 (no structure)     -> arm -2  dr_cs_precomp = 0.0 literal,
                                                    fseppc/fcspc/sigallpc unread;
                                                    TfInboardRadiiNoCsPrecomp
    otherwise                            -> arm  0  TfInboardRadiiTfOutsideCs
    ```

    `cold_boundary.md` producer 2, added 2026-08-27; arm -2 ported the same day (ST
    frontier wave -- the live cell on both tracked spherical-tokamak files).
    """
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
wave). The remaining refused arm (`TF_INSIDE_CS`, -1) is a real PROCESS branch with a
different reads-set; see its `UNPORTED` entry."""

VACUUM_SHIELD_RADII = {
    TFCSRadialConfiguration.TF_OUTSIDE_CS: VacuumVesselAndShieldRadiiTfOutsideCs,
}
"""`.build.i_tf_inside_cs` -> the inboard vacuum-vessel/shield radial slice
(`build.py:1833-1860`), added 2026-08-29. `TF_INSIDE_CS` accumulates three further
central-solenoid thicknesses into the same radius and is UNPORTED.

**Keyed on `i_tf_inside_cs` alone, not on `_tf_inboard_radii_arm`'s joint
`(i_tf_inside_cs, i_cs_precomp)`.** The two slots ask the same switch for different
reasons and this block's arm does not depend on the precompression structure; sharing
the joint answer would say that it does.
"""

DR_TF_OUTBOARD = {TFConductorModel.SUPERCONDUCTING: DrTfOutboardSuperconducting}
WP_CONDUCTOR_MAX_WIDTH = {
    TFConductorModel.SUPERCONDUCTING: WpConductorMaxWidthSuperconducting
}
"""`.tfcoil.i_tf_sup` -> the two build nodes that differ by conductor.

Both non-superconducting arms read fields the superconducting arm never touches (the
outboard leg scales by `.build.f_dr_tf_outboard_inboard`; the ripple fit's conductor
width comes from `.superconducting_tfcoil.r_tf_wp_inboard_outer` and `.tfcoil.n_tf_coils`
instead of three `dx_tf_wp_*` fields), so declaring one arm's reads on the other would
be four invented edges. `build.md` § "the four reads that are not edges" measures
exactly that, and it is the third recorded instance in this port."""

TF_OUTBOARD_MID = {
    TFCoilShapeModel.D_SHAPE: TfOutboardMidDShape,
    TFCoilShapeModel.PICTURE_FRAME: TfOutboardMidPictureFrame,
}
TF_OUTBOARD_EDGE_RIPPLE = {
    TFCoilShapeModel.D_SHAPE: TfOutboardEdgeRipple,
    TFCoilShapeModel.PICTURE_FRAME: TfOutboardEdgeRipplePictureFrame,
}
"""`.tfcoil.i_tf_shape` (resolved by `_tf_shape`) -> the two ripple calls, per shape.

Two slots for PROCESS's two calls to one formula, and not one node owning both outputs:
the second call's answer is what lands in `.tfcoil.ripple_b_tf_plasma_edge`, and a
single node would have to read the radius it owns. `i_tf_shape == 0` has no entry in
either registry: it is an auto-select meta-value that `init.py:728`/`:775` replaces
before any model runs, so `_tf_shape` resolves it and it names no arm."""

# ---- `.tokamak.cicc_superconducting_tf_coil` --------------------------------------

TF_GLOBAL_GEOMETRY = {
    TFPlasmaCaseType.CIRCULAR: TfGlobalGeometryCircularCase,
    TFPlasmaCaseType.STRAIGHT: TfGlobalGeometryStraightCase,
}
TF_CASE_AREAS = {
    TFPlasmaCaseType.CIRCULAR: TfCaseAreasCircularFront,
    TFPlasmaCaseType.STRAIGHT: TfCaseAreasStraightFront,
}
"""`.tfcoil.i_tf_case_geom` -> two slots, both arms written for each.

`TF_GLOBAL_GEOMETRY`'s two occupants have **identical reads-sets** and are two classes
anyway, which is `next_steps.md` §14.2's rule applied where it costs something and buys
nothing locally: the value is that no reader ever has to check whether a given slot's
arms happen to agree."""

DR_TF_PLASMA_CASE = {False: DrTfPlasmaCaseFromInput, True: DrTfPlasmaCaseFromFraction}
"""`.tfcoil.i_f_dr_tf_plasma_case` -> the plasma-case thickness occupant, and the one
slot in this port whose two arms are **different kinds of node**.

`False` clamps the entering `.tfcoil.dr_tf_plasma_case` in place, which is a node reading
what it owns, so its occupant is a `FixedPointFunction`; `True` computes the thickness
from a fraction and never reads the entering value, so its occupant is an
`ExplicitFunction`. The loop is a property of the arm, not of the quantity -- as clean a
demonstration as this port has that a switch can decide graph *topology* and not merely
a formula."""

DX_TF_SIDE_CASE_MIN = {True: DxTfSideCaseMinFromFraction, False: None}
"""`.tfcoil.tfc_sidewall_is_fraction` -> the sidewall-thickness occupant, **or `None`**.

`False` is PROCESS's own default and the reference run's, and on it
`.tfcoil.dx_tf_side_case_min` is simply an input -- there is no arm at all. So this is
absence and not a refusal, by `UNPORTED`'s own rule: refuse where *this port* has not
written the arm, assemble absence where **PROCESS itself computes nothing**. It is the
third slot in the tree spelled that way, after `costs.cost_of_electricity` and
`power.cryo_q_nuc`."""


def _tf_coil_shape_arm(
    i_tf_shape: TFCoilShapeModel, itart: int, i_single_null: int
) -> int:
    """`(i_tf_shape, itart, i_single_null)` -> the TF coil shape arm.

    ```
    PICTURE_FRAME and itart == 1 -> arm  2  picture frame, TART       (both ST files)
    PICTURE_FRAME and itart == 0 -> arm -2  picture frame, conventional; UNPORTED
    D_SHAPE      and itart == 1  -> arm -1  centrepost D-shape;        UNPORTED
    D_SHAPE, itart == 0, i_single_null == 1 -> arm  0  D-shape, single null   (live)
    D_SHAPE, itart == 0, otherwise          -> arm  1  D-shape, double null
    ```

    **`itart` is not tested before `i_tf_shape`**, and the ordering is the whole content
    of this function. `tf_coil_shape_inner`'s dispatch (`process/models/tfcoil/base.py`
    `:498`, `:528`, `:551`) is `i_tf_shape` first: the `itart == 1` clause at `:528` is
    guarded by `i_tf_shape == D_SHAPE`, so a spherical tokamak with a picture-frame coil
    lands in the picture-frame branch and not in "the TART branch". An earlier version of
    this function returned `-1` for every `itart == 1`, which is why the two ST files
    were refused with a reason naming an arm they never reach.

    Three switches, and the arms read genuinely different variables -- `r_cp_top` on the
    two `itart == 1` arms, `z_tf_top` on all but the D-shape double-null one,
    `r_tf_outboard_mid`/`r_tf_inboard_mid` on the picture frame -- so nothing here could
    have been a kwarg.
    """
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
"""The TF-coil-shape arm -> its occupant. Owns `.tfcoil.len_tf_coil`, one of the two
`VarPath`s a tokamak and a stellarator both produce from entirely different formulas."""


def _tf_self_inductance_arm(i_tf_shape: TFCoilShapeModel, itart: int) -> int:
    """`(itart, i_tf_shape)` -> the self-inductance arm. `0` integrates the D-shape's
    arcs; `1` is the picture-frame closed form, which is also what a spherical tokamak
    takes. Both are written, so nothing here reaches `UNPORTED`.
    """
    if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK:
        return 1
    return 0 if i_tf_shape is TFCoilShapeModel.D_SHAPE else 1


TF_COIL_SELF_INDUCTANCE = {
    0: TfCoilSelfInductanceDShape,
    1: TfCoilSelfInductancePictureFrame,
}
"""The self-inductance arm -> its occupant. The D-shape arm reads three fields where
PROCESS's composite function takes nine; the other six belong to the sibling arm, and
that gap is the measurement the split exists to make."""

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
written for each. `UNSET` never appears as a key, because it is not an arm."""


def _peak_b_ripple_arm(n_tf_coils: float) -> int:
    """`round(n_tf_coils)` -> the ripple-fit arm; `-1` is the flat-allowance fallback.

    A **coil count** treated as a switch, which is legitimate here and would not be
    everywhere: the arms select different MAGINT fit coefficients *and* own different
    numbers of outputs -- the fallback returns before three of the four are assigned
    (`superconducting.py:1519`). `n_tf_coils` is not an iteration variable, which is what
    makes a build-time branch on it sound; `superconducting.md` OQ2 flags that this stops
    being true the day it becomes one.

    No value reaches `UNPORTED`: every coil count has an occupant, because PROCESS's own
    fallback is an arm rather than an error.
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

    `i_tf_turns_integer` is answered first because PROCESS's `run` does
    (`superconducting.py:2343-2439`): on the integer arm the two booleans are never
    consulted, so arm `1` wins regardless of them. The averaged sub-family's two
    booleans then name three arms differing in which of `.tfcoil.c_tf_turn` /
    `dx_tf_turn_general` / `dx_tf_turn_cable_space_general` each **reads** and which it
    **owns**. That ownership difference is why they cannot share one node even in
    principle -- a kwarg cannot move a `VarPath` from a node's inputs to its outputs.

    Arm 0 is `(0, False, False)`, PROCESS's default and the reference run's, and it is
    the arm on which `.tfcoil.c_tf_turn` has **no producer anywhere under
    `process/models/`**: it is iteration variable 60 and enters from the input file. That
    is why this slot produces nine of the ten variables `tokamak_boundary.md` lists
    against it, and why the tenth is an unknown rather than a gap. Arm 1 is the integer
    arm (`low_aspect_ratio_DEMO`'s), on which the same field **is** produced -- see
    `CiccIntegerTurnGeometry`.
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
"""`(.physics.itart, .tfcoil.i_tf_sc_mat)` -> the superconducting TF mass occupant.

**Two switches, one slot, and the eighteen entries are their full product** -- the only
registry in this file with a two-switch key rather than a computed `_*_arm` integer,
because neither axis reduces to the other:

* `itart` decides *ownership*: the spherical arm owns `whtcp` and `whttflgs` and the
  conventional arm does not (`superconducting.py:2085-2093`), so it cannot be a kwarg.
* `i_tf_sc_mat` decides *one read*, `.tfcoil.dcond[i_tf_sc_mat - 1]`, and a `FromExactly`
  default is fixed at class-definition time, so it cannot be a kwarg either.

Written as a comprehension over a material -> `(conventional, spherical)` table rather
than as eighteen flat lines, for the reason the family's own docstring gives: the defect
this registry closes was one switch answered twice, so the material is named once here
and paired with both arms mechanically.

**All nine materials, including `HAZELTON_ZHAI_REBCO` (9), which
`WINDING_PACK_MATERIAL` refuses.** That refusal (`UNPORTED["i_tf_sc_mat", 9]`) is about
`jcrit_from_material` having no branch 9. This slot never calls it: the material selects
a density from a nine-long table and nothing else, and `dcond[8] == 8500.0` is real. The
two ST files set exactly this value.

Before 2026-08-27 this registry was keyed on `itart` alone and **both** occupants baked
`dcond[0]` in a module constant -- `_audit/next_steps.md` §14.5's `CoilsMass` failure,
found a second time. `low_aspect_ratio_DEMO` (`i_tf_sc_mat = 5`) was assembling the
`dcond[0]` occupant and only escaped a wrong number because `dcond[4] == dcond[0]`."""

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
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the critical-current occupant.

The second two-switch key in this file, for the same reason `SC_TF_MASSES` has one and
with the same "neither axis reduces to the other" test: `i_tf_sc_mat` selects the fit
*and* changes the reads-set (arm 3 reads no strain, arms 4 and 7 read two constants each
that no other arm does), and `i_str_wp` selects **which field the strain is read from**
-- `.tfcoil.str_tf_con_res` at `0`, `.tfcoil.str_wp` at `1`
(`process/models/tfcoil/superconducting.py:2897-2900`). A `From` default is fixed when
the class body executes, so neither can be a kwarg.

Only the `i_str_wp == 1` row is written. `1` is PROCESS's default
(`tfcoil_variables.py:508`) and no tracked input file sets the switch at all, so arm `0`
is unreachable; it is in `UNPORTED` so a file that does set it is refused rather than
silently getting the other strain."""

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
"""`(.tfcoil.i_str_wp, .tfcoil.i_tf_sc_mat)` -> the temperature-margin occupant.

**One row shorter than `CICC_SUPERCONDUCTOR_PROPERTIES`, on purpose.**
`DURHAM_NBTI` (7) has a properties occupant and no margin occupant, because the two are
different PROCESS functions and only the second one is broken: `gl_nbti` returns a
`complex` while `scipy.optimize.newton`'s secant search probes above `t_c0`, so PROCESS
either converges on a complex margin or raises a `TypeError` comparing one to a float.
Measured both ways -- see `TfSuperconductorTemperatureMargin`'s docstring for the two
numbers. The refusal is keyed separately from the properties slot's so the two cannot be
confused, exactly as `WINDING_PACK_MATERIAL` and `SC_TF_MASSES` are kept apart on value
9."""

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
"""`.divertor.n_divertors` (derived by `_n_divertors`) -> three slots, each total.

All three gained their `2` occupant on 2026-08-27, with `SHIELD_HALF_HEIGHT` below and
the four arm-keyed slots further down: the double-null wave, run for
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`, which set `i_single_null = 0`
(`:292`, `:638`). `n_divertors` is now a switch this port answers everywhere it is read
to branch, so it no longer appears in `UNPORTED` at all."""

BLANKET_AREAS = {0: DShapedBlanketAreas, 1: EllipticalBlanketAreas}
BLANKET_VOLUMES = {0: DShapedBlanketVolumes, 1: EllipticalBlanketVolumes}
"""`_fw_blkt_vv_shape_arm(itart, i_fw_blkt_vv_shape)` -> two slots, both **total** since
2026-08-27 (the D-shaped wave, for `spherical_tokamak_eval.IN.DAT` and
`st_regression.IN.DAT`, which set `i_fw_blkt_vv_shape = 1` *and* `itart = 1`).

The D-shaped arm reads no `.physics.triang` and no outboard build radius where the
elliptical arm reads both, and reads four `.build` first-wall thicknesses the elliptical
arm does not -- unequal sets, so occupants rather than a parameter."""

NUCLEAR_HEATING_MAGNETS = {
    SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO: NuclearHeatingMagnetsConventional,
    SphericalTokamakModel.SPHERICAL_TOKAMAK: NuclearHeatingMagnetsSphericalTokamak,
}
NUCLEAR_HEATING_SHIELD = {
    SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO: NuclearHeatingShieldConventional,
    SphericalTokamakModel.SPHERICAL_TOKAMAK: NuclearHeatingShieldSphericalTokamak,
}
"""`.physics.itart` -> two **total** slots, since 2026-08-27 (the centrepost wave).

Both spherical occupants were written and harness-tested long before they could be
registered: a machine at `itart == 1` also needs `blanket_library`'s D-shaped geometry
and the centrepost neutronics chain, and filling these two slots without the rest would
have assembled a graph that looks complete and is wrong -- the `EcrhDensityLimit` bug
class. The D-shaped half arrived with `BLANKET_AREAS`/`BLANKET_VOLUMES` above and the
centrepost half with `CENTREPOST_NEUTRONICS` below, so `('itart_hcpb', 1)` is answered
rather than moved, and `hcpb.md`'s open question 3 is closed."""


def _centrepost_neutronics_arm(itart: int, i_tf_sup: int) -> int:
    """`(itart, i_tf_sup)` -> the centrepost-neutronics arm.

    ```
    itart == 0                 -> arm  0   CentrepostNeutronicsAbsent (four zeros)
    itart == 1, i_tf_sup == 1  -> arm  1   the superconducting centrepost   (both ST
                                           input files)
    itart == 1, i_tf_sup == 0  -> arm -1   water-cooled copper;   UNPORTED
    itart == 1, i_tf_sup == 2  -> arm -2   helium-cooled aluminium; UNPORTED
    ```

    **`itart` is asked first, and that is the answerable-condition-last ordering rather
    than a preference.** `itart == 0` is answered outright -- `hcpb.py:143-148` is four
    literal assignments and `i_tf_sup` is not read on that arm at all -- so a
    conventional machine must never be told anything about its conductor here. Only once
    `itart == 1` does `i_tf_sup` become a question, and then it is the one without a
    general answer.

    **Why a joint arm rather than two slots keyed on one switch each.** `run():103-141`
    is one straight-line block calling three routines, and the two that read `i_tf_sup`
    *partition it differently*: `st_tf_centrepost_fast_neut_flux` splits `{1}` from
    `{0, 2}` (`hcpb.py:1114`), while `st_centrepost_nuclear_heating` splits `{2}` from
    `{0, 1}` (`:1192`, and the comment above it says why -- the MCNP winding pack is
    large enough to be mostly copper, so one fit serves superconducting and copper
    alike). No single integer names the occupant of the block; the pair does.
    """
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
"""The centrepost-neutronics arm -> its occupant. See `_centrepost_neutronics_arm`.

Arm `1` joined on 2026-08-27 and is what makes `.physics.itart == 1` assemblable. It is
the **only** slot in this port whose two occupants own a different number of fields *and*
one of whose fields moves to a different node on the other arm:
`.fwbs.p_cp_shield_nuclear_heat_mw` belongs to arm `0` here and to the renormalisation
on arm `1`, because `hcpb.py` writes it twice and only on the spherical arm do the two
writes differ."""


def _nuclear_heating_renormalisation_arm(n_divertors: int, itart: int) -> int:
    """`(n_divertors, itart)` -> the renormalisation arm.

    ```
    itart == 0, n_divertors == 1 -> arm 0   single null, conventional
    itart == 0, n_divertors == 2 -> arm 1   double null, conventional
    itart == 1, n_divertors == 1 -> arm 2   single null, spherical
    itart == 1, n_divertors == 2 -> arm 3   double null, spherical  (both ST files)
    ```

    A joint arm because both switches gate the same block: `hcpb.py:215` reads
    `n_divertors` to pick `f_geom_blanket`, and `:103` reads `itart` to decide whether
    the centrepost terms at `:263` and `:268` contribute at all. On the conventional arms
    both of those terms are provably inert -- `f_geom_cp` and `.fwbs.pnuc_cp_tf` are the
    literal zeros of `:144-145` -- so those occupants do **not** declare them as reads,
    which is two invented edges avoided by knowing the arm. On the spherical arms both
    are live, and a third field, `.fwbs.p_cp_shield_nuclear_heat_mw`, becomes theirs.
    """
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
"""The renormalisation arm -> its occupant. See `_nuclear_heating_renormalisation_arm`.

Arm `1` (double-null, conventional) joined on 2026-08-27; arms `2` and `3` (the two
spherical cells) joined the same day with the centrepost chain, and the 2x2 is **total**.
The `itart` question is asked first out of habit rather than necessity now -- every cell
has an occupant, so no ordering of the two questions can name a precondition this port
does not meet."""

PUMPING_POWER = {
    PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP: (
        PumpingPowerMechanicalWithPressureDrop
    )
}
"""`.fwbs.i_p_coolant_pumping` -> the pumping-power occupant, and the clearest case in
this port of arms that **do not own the same set**.

Arm 1 owns four `.heat_transport.p_*_coolant_pump_mw` fields; arm 3 owns two of them plus
`.primary_pumping.p_fw_blkt_coolant_pump_mw`. Any `Switch` over this slot has a partial
overlap by construction, which is `next_steps.md` §12.2's "alternatives are keyed on
output -- nearly" with a name attached. Arm 2 additionally reaches CoolProp."""

BLANKET_MODEL = {BlktModelTypes.CCFE_HCPB: CcfeHcpb}
"""`.fwbs.i_blanket_type` -> the occupant of `.tokamak.ccfe_hcpb`.

Two live values in PROCESS (`1` CCFE HCPB, `5` DCLL; `2`-`4` are marked removed in
`fwbs_variables.py:70-78`), dispatched at `caller.py:343-349`. `1` is a default rather
than a file setting on `large_tokamak_eval.IN.DAT`, which is worth knowing: the slot is
switched even though this run never says so."""

# ---- the four single-node tokamak slots -------------------------------------------


def _first_wall_arm(n_divertors: int, shape_arm: int, i_pflux_fw_neutron: int) -> int:
    """`(n_divertors, shape arm, i_pflux_fw_neutron)` -> `FirstWall`'s arm.

    **The shape x divertor-count product.** `FirstWall.run()` branches on `n_divertors`
    twice with the shape branch between them (`process/models/fw.py:46-109`), and this
    port keeps the whole of `run()` as one node, so the occupant grid is 2 x 2 rather
    than two independent slots. Only `.tokamak.vacuum_vessel` shares that shape; the
    blanket and shield slots escape it because wave 1 split their `run()`s per branch.

    ```
    i_pflux_fw_neutron != 1                  -> arm -3   refused
    D-shaped and n_divertors == 1            -> arm -2   refused
    elliptical, n_divertors == 1             -> arm  0
    elliptical, n_divertors == 2             -> arm  1
    D-shaped,   n_divertors == 2             -> arm  2
    ```

    **The answerable conditions are asked last**, as they have been since 2026-08-27:
    `i_pflux_fw_neutron` can only ever be answered one way here, so it refuses first; the
    one unwritten cell of the grid refuses second; the three written cells fall out of
    the mapping at the end. A rejected file is told which of its preconditions broke, and
    is told it before anything this port *can* answer gets in the way.

    Arm `-2` was "the D-shaped first wall" until 2026-08-27; it now means the D-shaped
    first wall **at a single divertor** specifically, since the double-null cell is
    written.
    """
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

Arms `0` and `1` are the elliptical pair, differing by two reads
(`.build.z_plasma_xpoint_upper`, `.build.dz_fw_plasma_gap`); arm `2` drops
`.physics.triang` as well, because the D-shaped area formula does not use it. Sixteen
reads against arm `0`'s nineteen -- which is why these are occupants and not one node
with two switch parameters."""


def _vacuum_vessel_arm(n_divertors: int, shape_arm: int) -> int:
    """`(n_divertors, shape arm)` -> the vacuum vessel's arm. The same two conditions as
    `_first_wall_arm`'s, in the same shape (a 2 x 2 grid with one cell unwritten), and
    `vacuum.md` confirmed them independently rather than inheriting them from `fw.md` --
    which is worth recording, because three records reaching the same predicate
    separately is what makes it safe to write once.

    ```
    D-shaped and n_divertors == 1  -> arm -2   refused
    elliptical, n_divertors == 1   -> arm  0
    elliptical, n_divertors == 2   -> arm  1
    D-shaped,   n_divertors == 2   -> arm  2
    ```

    The one refusal is asked first and the three written cells fall out of the mapping
    last, the same ordering `_first_wall_arm` uses.
    """
    if shape_arm == 0 and int(n_divertors) == 1:
        return -2
    return {(1, 1): 0, (1, 2): 1, (0, 2): 2}[shape_arm, int(n_divertors)]


VACUUM_VESSEL = {
    0: VacuumVesselEllipticalSingleNull,
    1: VacuumVesselEllipticalDoubleNull,
    2: VacuumVesselDShapedDoubleNull,
}
"""`_vacuum_vessel_arm(...)` -> `.tokamak.vacuum_vessel`'s occupant, three cells of a
2 x 2.

**A confirmed registry prediction.** Unit #16 recorded `VacuumVessel` as *"confirmed
unreachable on the stellarator pipeline, no action needed"*; the tokamak trace reaches it
at `caller.py:331`, and this is the slot that follows.

Arm `2` is the sparsest reads-set in this wave: ten fields against arm `0`'s twenty, and
**no `.physics` port at all** (the D-shaped vessel anchors on the shield build, so
`rmajor`/`rminor`/`triang` are all gone on top of the double-null half-height's
seven)."""


def _structure_arm(i_tf_sup: int, i_pf_conductor: int) -> int:
    """`(i_tf_sup, i_pf_conductor)` -> `Structure`'s arm: one cell of a 2x2.

    Both switches gate independent, **additive** terms of one output (`.structure.
    coldmass`), and on the reference run both are true, so the occupant bakes in both
    terms and takes neither switch as a parameter. The other three cells are `UNPORTED`.

    `structure.md` flags this as a judgement call rather than a silent default, and it is
    the same "shared remainder" shape `traceability_policy.md` records as one of six
    deliberate deviations from strict per-value splitting -- two one-line terms inside a
    thirty-line body. This wave's stricter instruction is what settles it: no switch is a
    kwarg, so the live combination is one occupant with no switch parameter at all, and a
    resistive-TF or resistive-PF run needs its own class rather than an argument.
    """
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
    """`(i_div_heat_load, n_divertors)` -> the divertor heat-load arm.

    ```
    i_div_heat_load == 0 (USER_INPUT)     -> arm -1  reads nothing, prints; UNPORTED
    i_div_heat_load == 1 (PENG_CHAMBER)   -> arm -2  divtart, six other fields; UNPORTED
    n_divertors == 1                      -> arm  0  DivertorHeatLoadWadeSingleNull
    n_divertors == 2                      -> arm  1  DivertorHeatLoadWadeDoubleNull
    ```

    Joint, because `divwade`'s own double-null branch (`:377-382`) reads
    `.physics.f_p_div_lower` and takes a `max` the single-null arm does not -- so
    `n_divertors` is a second question asked only once `i_div_heat_load` has answered
    `WADE`, the same nesting `_energy_storage_arm` has for `istore`. Both of its answers
    are occupants since 2026-08-27; the two `i_div_heat_load` refusals are unchanged.
    """
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

The same integer also feeds `_plasma_geometry_arm` (the Sauter disjunction, which that
function owns -- `plasma_geometry.md` OQ2) and `_surface_poloidal_field_arm`; all three
consumers read the one threaded local, so the switch is answered once.

Two arms, and they differ in **read set**, not in constants: `IPDG89_SCALING` (4) reads
the 95%-flux-surface shaping pair, `FIESTA_ST_SCALING` (9) the separatrix pair. The
FIESTA arm is what both tracked spherical tokamaks select."""

CURRENT_PROFILE_INDEX = {
    CurrentProfileIndexModel.USER_INPUT: None,
    CurrentProfileIndexModel.WESSON: WessonCurrentProfileIndex,
}
"""`.physics.i_alphaj` -> the current-profile-index occupant. `None` is an occupant
here, not a refusal: PROCESS's `USER_INPUT` arm is `alphaj = alphaj`
(`physics.py:338`), so the field is a run input and the slot is empty."""

IND_PLASMA_INTERNAL_NORM = {
    IndInternalNormModel.USER_INPUT: None,
    IndInternalNormModel.WESSON: PlasmaInternalInductanceNormWesson,
}
"""`.physics.i_ind_plasma_internal_norm` -> the normalised-internal-inductance
occupant, in `.tokamak.plasma_inductance`. `USER_INPUT` selects the field from itself
(`physics.py:4760`) -- no node; `MENARD` is UNPORTED."""

BOOTSTRAP_CURRENT = {
    BootstrapCurrentFractionModel.USER_INPUT: None,
    BootstrapCurrentFractionModel.SAUTER: SauterBootstrapCurrentFraction,
}
"""`.physics.i_bootstrap_current` -> `.tokamak.bootstrap_current`'s occupant. At
`USER_INPUT` the fraction is an `IN.DAT` variable and the slot is empty; the Sauter
occupant carries the profile grid's shape (`n_plasma_profile_elements`) as its one
static kwarg -- a resolution, not a switch (`switch_elimination_design.md` §3(b))."""

DIAMAGNETIC_CURRENT = {
    PlasmaDiamagneticCurrentModel.NONE: NoDiamagneticCurrent,
    PlasmaDiamagneticCurrentModel.SCENE_FIT: SceneDiamagneticCurrent,
}
"""`.physics.i_diamagnetic_current` -> `.tokamak.diamagnetic_current`'s occupant. The
`NONE` arm is a real occupant (PROCESS assigns the literal zero), not an empty slot:
`PlasmaCurrentFractions` reads the fraction unconditionally. `SCENE_FIT` (2) is what
both tracked spherical tokamaks select; `HENDER_ST_FIT` (1) is UNPORTED."""

PFIRSCH_SCHLUTER_CURRENT = {0: NoPfirschSchluterCurrent, 1: ScenePfirschSchluterCurrent}
"""`.physics.i_pfirsch_schluter_current` -> `.tokamak.pfirsch_schluter_current`'s
occupant. Bare-integer keys: PROCESS declares no enum for this switch. Both values have
an occupant -- `1` (the SCENE fit) is what both tracked spherical tokamaks select."""

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
"""`.physics.i_l_h_threshold` -> `.tokamak.l_h_transition`'s occupant. Six of the
twenty-one values -- the Martin-2008 family, whose reads-sets `l_h_transition.md`
validated against the live arm (19) rather than assumed; the other fifteen formulas
are ported, tested and unwired."""

DENSITY_LIMIT_ENFORCED = {DensityLimitModel.GREENWALD: EnforcedDensityLimitGreenwald}
"""`.physics.i_density_limit` -> `.tokamak.density_limit.enforced_density_limit`'s
occupant. Only the *enforced* limit answers the switch; the Greenwald element and
fraction are computed unconditionally (`density_limit.md` § 'not actually
switch-gated')."""

SOL_OUTBOARD_POWER_DECAY = {
    OutbordSOLPowerDecayLengthModel.USER_INPUT: None,
    OutbordSOLPowerDecayLengthModel.EICH_2013: OutboardSOLPowerDecayLengthEich2013,
}
"""`.physics.i_len_sol_outboard_power_decay` -> the selector occupant in
`.tokamak.scrape_off_layer`. `USER_INPUT` has no `else` arm in PROCESS at all -- the
field keeps its entering value, so the slot is empty; the two MAST selectors are
UNPORTED one-liners."""

SHIELD_HALF_HEIGHT = {1: SingleNullShieldHalfHeight, 2: DoubleNullShieldHalfHeight}
"""`_n_divertors(i_single_null)` -> `.tokamak.shield.half_height`'s occupant. Both
values of the binary switch are written (`shield.md` 'ported' table) -- the one
registry in this wave that is total."""

SHIELD_VOLUMES = {0: DShapedShieldVolumes, 1: EllipticalShieldVolumes}
"""`_fw_blkt_vv_shape_arm(itart, i_fw_blkt_vv_shape)` -> `.tokamak.shield.volumes`'s
occupant -- the fifth slot keyed on that existing joint predicate, per `shield.md`'s
'join that key at consolidation, not mint an independent one'. **Total** since
2026-08-27: `calculate_dshaped_shield_volumes` had been a ported function without an
occupant since wave 1, precisely so that its occupant could hang on *this* key rather
than a freshly minted one, and the D-shaped wave supplied it."""


def _pf_coil_system_arm(  # noqa: PLR0911 -- one return per refused dimension, by design
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

    One predicate, thirteen slots -- the `_fw_blkt_vv_shape_arm` shape at package
    scale: the five `pfcoil/*.md` records name overlapping subsets of these switches
    and every ported occupant is the one for the single joint configuration below, so
    the factory resolves the conjunction once and both namespaces
    (`models/pfcoil/namespace.py`) are keyed on the result. Arm `0` is the supported
    configuration; each negative arm names which dimension deviated, in the order the
    records argue they differ most structurally (`UNPORTED` carries each reason).
    Positive arms are the ported superconductor pairs: arm `0` is the reference pair
    (`i_pf_superconductor = 3` NbTi, `i_cs_superconductor = 1` ITER Nb3Sn), arm `1` is
    `(3, 5)` -- NbTi PF, WST Nb3Sn CS, `low_aspect_ratio_DEMO.IN.DAT`'s pair. Every
    other switch is answered identically on both arms; the pair selects which
    `.tfcoil.dcond` element the masses occupant reads, and nothing else.

    `n_pf_coil_groups`/`i_pf_location`/`n_pf_coils_in_group` are the **coil-count
    topology** -- not switches in `naming_convention.md`'s sense, but they fix every
    array index in the package (`pfcoil/__init__.py`'s module constants), so a
    deviation refuses the same way a switch value without an occupant does.

    What this function deliberately cannot see: `noh = 30`, the CS pancake-segment
    count, a step function of the *converged* CS geometry rather than of any input
    (`inductance.md` § 'noh is a step function of the CS geometry'). It stays a module
    constant on `PFCoilInductance`.
    """
    if int(iohcl) == 0:
        return -1
    if (
        int(n_pf_coil_groups) != 4
        or tuple(int(v) for v in i_pf_location[:4]) != (2, 2, 3, 3)
        or tuple(int(v) for v in n_pf_coils_in_group[:4]) != (1, 1, 2, 2)
    ):
        return -2
    if SphericalTokamakModel(int(itart)) is SphericalTokamakModel.SPHERICAL_TOKAMAK or (
        int(itartpf) != 0
    ):
        return -3
    if int(i_pf_current) == 0:
        return -4
    if PFConductorModel(int(i_pf_conductor)) is not PFConductorModel.SUPERCONDUCTING:
        return -5
    sc_pair = (
        SuperconductorModel(int(i_pf_superconductor)),
        SuperconductorModel(int(i_cs_superconductor)),
    )
    if sc_pair == (
        SuperconductorModel.OLD_LUBELL_NBTI,
        SuperconductorModel.ITER_NB3SN,
    ):
        sc_arm = 0
    elif sc_pair == (
        SuperconductorModel.OLD_LUBELL_NBTI,
        SuperconductorModel.WST_NB3SN,
    ):
        sc_arm = 1
    else:
        return -6
    if (
        i_tf_shape is not TFCoilShapeModel.D_SHAPE
        or int(i_r_pf_outside_tf_placement) != 0
    ):
        return -7
    return sc_arm


CS_COIL = {0: CSCoil, 1: CSCoil}
"""`_pf_coil_system_arm` -> `.tokamak.cs_coil`'s occupant namespace.

The same namespace on both positive arms: nothing in `CSCoil` reads `.tfcoil.dcond` --
the CS conductor density is read by the masses node in `.tokamak.pf_coil`, which is
where the two arms differ.

Since 2026-08-27 the namespace has one factory-filled slot of its own
(`critical_current`), so the two arms are the same *class* and no longer the same
*instance* -- `_cs_coil` below builds it."""

CS_SUPERCONDUCTOR = {
    SuperconductorModel.ITER_NB3SN: CSCriticalCurrentDensitiesIterNb3Sn,
    SuperconductorModel.WST_NB3SN: CSCriticalCurrentDensitiesWstNb3Sn,
}
"""`.pf_coil.i_cs_superconductor` -> `.tokamak.cs_coil.critical_current`'s occupant.

**Total over the values that reach it, and therefore with no `UNPORTED` entries at
all** -- the second such registry, after `SHIELD_HALF_HEIGHT`. `superconpf` dispatches
on eight values, but `_pf_coil_system_arm` above has already refused six of them (arm
`-6`) before this slot is built: only `1` (ITER Nb3Sn) and `5` (WST Nb3Sn) survive its
`(i_pf_superconductor, i_cs_superconductor)` pair, and both are written.

**This switch is asked twice, on purpose.** `_pf_coil_system_arm` reads it as half of
that pair, which selects the *masses* occupant -- and its arm `1` covers both surviving
values, because which `.tfcoil.dcond` element a mass reads is a different question from
which critical-surface fit a current density comes from. Answering the second by reusing
the first's arm would silently give a WST Nb3Sn CS the ITER Nb3Sn critical surface, the
`EcrhDensityLimit` bug class `models/tokamak/namespace.py` names.

Widening `_pf_coil_system_arm`'s pair later would make this registry partial again, and
the six occupants it would then owe are enumerated in
`models/pfcoil/superconductor.py`'s module docstring -- the reasons live there rather
than in `UNPORTED`, because an `UNPORTED` entry nothing can reach is a refusal that
never fires."""

PF_COIL = {0: PFCoil, 1: PFCoilCsWstNb3Sn}
"""`_pf_coil_system_arm` -> `.tokamak.pf_coil`'s occupant namespace. Arm 1 differs in
exactly one slot occupant, `masses` (`.tfcoil.dcond[4]` as the CS conductor
density)."""


_INDAT_INTEGER = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$")


def switches_from_indat(input_file):
    """Every `name = <integer>` this input file sets, as a plain dict.

    Deliberately not a full IN.DAT parser: the only thing a machine is built from is
    integer switches, and PROCESS's own `SingleRun` is what reads everything else.
    A name the file never mentions is simply absent, which is what "falls through to the
    default" means.
    """
    text = Path(input_file).read_text()
    found = {}
    for line in text.splitlines():
        match = _INDAT_INTEGER.match(line)
        if match:
            found[match.group(1)] = int(match.group(2))
    return found


_INDAT_NUMBER = re.compile(
    r"\s*([A-Za-z_]\w*)\s*=\s*(-?[\d.]+(?:[eEdD][-+]?\d+)?)\s*(\*.*)?$"
)


def numbers_from_indat(input_file):
    """Every `name = <number>` this input file sets, as a plain dict of floats.

    `switches_from_indat`'s sibling, and needed for exactly one thing: the tokamak's
    `.build.dz_xpoint_divertor`, whose *input value* decides whether a node owns that
    field or it stays an input (`_divertor_geometry_arm`). A float is not a switch, but
    a float that decides which nodes exist is one for this factory's purposes, and the
    same argument applies -- an input cannot change between two evaluations of one
    assembled graph.

    Deliberately still not a full IN.DAT parser. A name the file never mentions is
    absent, which is what "falls through to the default" means.
    """
    text = Path(input_file).read_text()
    found = {}
    for line in text.splitlines():
        match = _INDAT_NUMBER.match(line)
        if match:
            found[match.group(1)] = float(
                match.group(2).replace("d", "e").replace("D", "e")
            )
    return found


_INDAT_INT_LIST = re.compile(
    r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+(?:\s*,\s*-?\d+)+)\s*,?\s*(\*.*)?$"
)


def int_lists_from_indat(input_file):
    """Every `name = <int>, <int>, ...` this input file sets, as a dict of tuples.

    The third sibling, and needed for exactly one consumer: the PF coil system's
    coil-count topology (`i_pf_location = 2,2,3,3`, `n_pf_coils_in_group = 1,1,2,2`),
    which `_pf_coil_system_arm` checks against the one supported pattern. A
    comma-separated list is not an integer switch, but a list that fixes every array
    index in a package decides which occupants exist, and the factory's standing test
    holds -- an input cannot change between two evaluations of one assembled graph.

    Deliberately still not a full IN.DAT parser; a name the file never mentions is
    absent, and the caller supplies PROCESS's own `DataStructure` default.
    """
    text = Path(input_file).read_text()
    found = {}
    for line in text.splitlines():
        match = _INDAT_INT_LIST.match(line)
        if match:
            found[match.group(1)] = tuple(
                int(v) for v in match.group(2).split(",") if v.strip()
            )
    return found


def iteration_variables_from_indat(input_file):
    """The `ixc` this input file declares, as a frozenset of iteration-variable IDs.

    `ixc` is the one name in an IN.DAT that legitimately repeats -- one line per active
    unknown -- so `switches_from_indat`'s last-wins dict cannot hold it. It is read here
    because **an iteration variable can decide graph topology**: `140 in ixc` picks
    which of two inverse assignments `process/models/build.py` makes, one producing
    `.build.dr_tf_inboard` and the other `.tfcoil.dr_tf_wp_with_insulation`.

    That is a genuinely new kind of key for this factory, and it satisfies the same test
    every switch does (`machine_from_indat`'s docstring): the active set is fixed for a
    whole solve -- `Scan` re-solves from scratch per point, and no PROCESS code adds to
    `ixc` mid-solve -- so it cannot change between two evaluations of one assembled
    graph. What an iteration variable's *value* does is a different question, and that
    one is the optimiser's.
    """
    text = Path(input_file).read_text()
    found = set()
    for line in text.splitlines():
        match = _INDAT_INTEGER.match(line)
        if match and match.group(1) == "ixc":
            found.add(int(match.group(2)))
    return frozenset(found)


_QUENCH_GRID_FIELDS = ("tftmp", "temp_tf_conductor_quench_max")
"""The two `.tfcoil` inputs the quench quadrature grid -- and therefore the helium
property table -- is a function of. Both must be run *inputs* for
`TfCoilQuenchHeatCurrentDensity`'s static table to be sound; `_quench_helium_table`
checks that and refuses otherwise."""


def _quench_helium_table(numbers, ixc):
    """`(temp_he_peak, temp_quench_max, den_helium, cp_helium)` for this machine.

    **The one place CoolProp is called in the whole port**, and it is called here --
    at machine-assembly time, once, outside every traced region -- rather than from a
    node body. `TfCoilQuenchHeatCurrentDensity`'s docstring carries the decision and the
    measurement behind it; this function carries the *guard* that makes the decision
    sound, and the guard is the load-bearing half.

    A static property table is correct exactly while the temperatures it was built at
    cannot move. Neither `.tfcoil.tftmp` nor `.tfcoil.temp_tf_conductor_quench_max` is
    written by any model, so the only way either could move during a solve is by being
    an iteration variable -- and that is checked, not assumed. A machine whose `ixc`
    names one is **refused**, because the alternative is a table silently evaluated at
    the wrong states while the optimiser walks away from them. Without this check the
    static field would be the same defect shape as the `dcond[0]` bake `SC_TF_MASSES`
    exists to have closed.

    The defaults are `tfcoil_variables.py`'s own (`tftmp = 4.75`,
    `temp_tf_conductor_quench_max = 150.0`).
    """
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
    switches, numbers, ixc, int_lists, i_tf_sup, i_plasma_ignited, itart, i_tf_sc_mat
):
    r"""The `Tokamak` an IN.DAT describes -- twenty-six slots of the twenty-eight filled.

    Split out of `machine_from_indat` rather than inlined, and the reason is length
    rather than principle: this is still the factory, and every `i_*` integer it reads is
    read here for the same reasons that function's docstring gives. It takes
    `switches`/`numbers`/`ixc` already parsed, plus the values `machine_from_indat`
    has already resolved and threaded -- `i_tf_sup`, `i_plasma_ignited`, `itart` and
    `i_tf_sc_mat` -- because **a switch is answered once**: re-reading any of them here
    would be the second transcription that `model_tree_design.md` §8 step 4d removed
    from the tree. `itart` joined that list when the four shared slots that hardcoded it
    became families (`_audit/next_steps.md` §14.2); `i_tf_sc_mat` joined it when
    `superconducting_tf_coil_areas_and_masses` stopped baking `.tfcoil.dcond[0]`, and it
    is threaded rather than read here for a stronger reason than tidiness -- it is the
    *same local* the stellarator branch hands `WINDING_PACK_MATERIAL` and
    `COILS_MASS_MATERIAL`, so the three consumers of that one switch cannot name three
    different materials.

    **The two slots this does not fill are not mentioned.** `cs_fatigue` and
    `water_use` keep `models/tokamak/namespace.py`'s `None`, whatever the file says:
    the first is DECIDED-DEFERRED (`cs_fatigue.md`'s `ncycle` decision), the second is
    a measured dead end (nothing in `process/` reads any `.water_use.*` output). A file
    that asks for a particular `i_bootstrap_current` *is* refused now -- waves 2/3
    filled the eleven slots the first wave left empty -- and
    `_audit/tokamak_boundary.md` is where the cost of the remaining absences is
    counted, variable by variable.

    Three switches are read here that no other part of this factory reads and that are
    not `i_*` integers at all:

    * `.physics.i_single_null`, which is not itself a slot key -- `_n_divertors` derives
      `.divertor.n_divertors` from it exactly as `process/core/init.py:606-617` does,
      because the `DataStructure` field's own default (`2`) is dead on every run.
    * `140 in ixc`, an **iteration variable**, which picks which of two inverse
      assignments `build.py` makes.
    * the input `.build.dz_xpoint_divertor`, a **float**, which decides whether
      `divgeom` owns that field or leaves it an input.

    All three are constants of a solve, which is the only property this factory ever
    asks of a key.
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
    pulse = TokamakPulse(
        ramp_times=_slot_occupant(
            "pulse_ramp_times_arm",
            _pulse_ramp_times_arm(
                switches.get("i_pulsed_plant", 0),  # `pulse_variables.py:30`
                switches.get("pulsetimings", 1),  # `times_variables.py:12`
                switches.get("i_t_current_ramp_up", 0),  # `:44`
            ),
            PULSE_RAMP_TIMES,
        )
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
        dr_tf_inboard_winding_pack=_slot_occupant(
            "dr_tf_inboard_winding_pack",
            0 if 140 in ixc else 1,
            DR_TF_INBOARD_WINDING_PACK,
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
    tf_coil = CiccSuperconductingTfCoil(
        tf_global_geometry=_slot_occupant(
            "i_tf_case_geom", i_tf_case_geom, TF_GLOBAL_GEOMETRY
        ),
        dr_tf_plasma_case=_slot_occupant(
            "i_f_dr_tf_plasma_case",
            bool(switches.get("i_f_dr_tf_plasma_case", 0)),  # `tfcoil_variables.py:83`
            DR_TF_PLASMA_CASE,
        ),
        # `None` is an occupant here, not a refusal: at the default `False` PROCESS
        # computes no `.tfcoil.dx_tf_side_case_min` at all and the field is an input.
        dx_tf_side_case_min=_slot_occupant(
            "tfc_sidewall_is_fraction",
            bool(switches.get("tfc_sidewall_is_fraction", 0)),  # `:95`
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
        cicc_turn_geometry=_slot_occupant(
            "cicc_turn_geometry_arm",
            _cicc_turn_geometry_arm(
                i_tf_turns_integer,  # resolved above, beside `i_tf_wp_geom`
                switches.get("i_dx_tf_turn_general_input", 0),  # `:108`
                switches.get("i_dx_tf_turn_cable_space_general_input", 0),  # `:127`
            ),
            CICC_TURN_GEOMETRY,
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
        # Both of these key on `(i_str_wp, i_tf_sc_mat)`. `i_str_wp` is
        # `tfcoil_variables.py:508`'s default; the pair is built once so the two slots
        # cannot disagree about either switch, which is the same cross-slot coherence
        # `i_tf_sc_mat` already gets from being resolved above the device branch.
        cicc_superconductor_properties=_slot_occupant(
            "i_str_wp_i_tf_sc_mat_cicc_sc_properties",
            (i_str_wp, i_tf_sc_mat),
            CICC_SUPERCONDUCTOR_PROPERTIES,
        ),
        tf_superconductor_temperature_margin=_slot_occupant(
            "i_str_wp_i_tf_sc_mat_temp_margin",
            (i_str_wp, i_tf_sc_mat),
            TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN,
        ),
        tf_coil_quench_heat_current_density=TfCoilQuenchHeatCurrentDensity(
            tftmp=quench_temp_he_peak,
            temp_tf_conductor_quench_max=quench_temp_max,
            den_helium_at_nodes=quench_den_helium,
            cp_helium_at_nodes=quench_cp_helium,
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
    pf_coil_arm = _pf_coil_system_arm(
        switches.get("iohcl", 1),  # `build_variables.py:177`
        switches.get("n_pf_coil_groups", 3),  # `pfcoil_variables.py:320`
        int_lists.get("i_pf_location", (2, 2, 3, 0)),  # `:220`
        int_lists.get("n_pf_coils_in_group", (1, 1, 2, 0)),  # `:310`
        itart,
        switches.get("itartpf", 0),  # `physics_variables.py:1000`
        switches.get("i_pf_current", 1),  # `pfcoil_variables.py:279`
        i_pf_conductor,
        switches.get("i_pf_superconductor", 1),  # `:254`
        switches.get("i_cs_superconductor", 1),  # `:239`
        i_tf_shape,
        switches.get("i_r_pf_outside_tf_placement", 0),  # `:287`
    )
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
            build=lambda cls: cls(
                critical_current=_slot_occupant(
                    "i_cs_superconductor",
                    SuperconductorModel(
                        int(switches.get("i_cs_superconductor", 1))  # `pfcoil_vars:225`
                    ),
                    CS_SUPERCONDUCTOR,
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
    r"""The `StellaratorProcess` an IN.DAT describes -- the only thing that builds one.

    Every slot below is passed explicitly because every slot below has no default: there
    is no `StellaratorProcess()` to apply deltas to any more. A switch the file does not
    mention still falls through to PROCESS's own default, but it falls through *here*,
    as the second argument to `switches.get`, where it is visible and cited.

    **The only place in this port an `i_*` integer is ever read.** Everything downstream
    sees a tree of model instances; nothing else has to know that `6` means Helias or
    that `blktmodel` and `ipowerflow` are consulted together.

    **Why assembly time is the only correct place for this, and not a preference.** Every
    switch PROCESS has is a constant for the whole solve:

        grep -n "\"i_\|'i_" process/core/solver/iteration_variables.py   -> no matches
        grep -n "\"i_\|'i_\|istell" process/core/scan.py                -> no matches

    No switch is an iteration variable and none is a scan variable, so no switch can
    change between two evaluations of one assembled graph, and `Scan` re-solves from
    scratch per point anyway. A switch therefore carries no derivative, participates in
    no edge, and has nothing to contribute to a `Graph` -- which is cottax's position
    that graph structure is decided by the caller once, not re-read per evaluation.

    The rejected alternative was one node owning the union of every variant's ports and
    branching internally. That would make a node read `eta_ecrh_injector_wall_plug` *and*
    `eta_lowhyb_injector_wall_plug` regardless of which is live, inventing graph edges
    that do not exist in the run being modelled, and would put a non-differentiable
    integer on a port. It also loses the result `Stellarator.fw_area` records: **a switch
    can decide whether the graph has a cycle**, which is not a fact any single fused node
    could express.

    Joint dispatch is ordinary code here rather than a mechanism: `blktmodel` is read
    together with `ipowerflow` for one slot and with `blkttype` for another, and
    `ireactor` together with `ipnet` for a third, by `_blanket_shield_power_arm` /
    `_blanket_mass_arm` / `_cost_of_electricity_arm`, each of which turns a tuple of
    **legal switch values** into the **arm index** its registry is keyed on. No switch
    value is ever used as a key, and no switch has a default outside its own declared
    domain. So is cross-slot coherence -- `istell == 6` sets both the machine config and
    the confinement binding, because they are two consequences of one choice, which is
    why the two are resolved together, into named locals, before anything else.
    `i_tf_sup`, `ipowerflow` and `ireactor` are read into locals for the same reason and
    a second one: each also has to reach a *static field* of some occupant that branches
    on it internally, and until step 4d each of those fields carried its own hardcoded
    copy of the answer. **A switch is resolved once, here, and threaded; it is never
    also written into a constructor kwarg** -- that is what
    `test_switch_coverage.test_no_slot_contradicts_a_factory_switch` now checks
    mechanically over the whole assembled tree, at every value each switch can take.

    One switch reaches the tree *without* being read from the file at all:
    `i_plasma_pedestal`, which `st_init` overwrites on every stellarator run. See
    `ST_INIT_I_PLASMA_PEDESTAL`.

    **The device is a branch here and nowhere else.** `istell` selects a *class* --
    `TokamakProcess` or `StellaratorProcess`, siblings, see `total_process.py` -- so it
    is read first and the function has two `return`s. Everything either device shares is
    resolved above the branch and passed to whichever constructor runs; everything only a
    stellarator asks (`stella_conf`, `isthtr`, both joint blanket dispatches,
    `ipowerflow`) is resolved *below* it, so a tokamak is never refused for a reason that
    belongs to a device it is not.

    `i_plasma_pedestal` is the one shared switch the two arms answer differently, and the
    asymmetry is PROCESS's: `st_init` overwrites it on every `istell != 0` run and does
    not run on a tokamak. See `ST_INIT_I_PLASMA_PEDESTAL`.

    Raises
    ------
    NotImplementedError
        The file asks for a real PROCESS branch this port has no occupant for. **A file
        that sets nothing at all no longer raises here**: PROCESS's own default is
        `istell = 0`, a tokamak, and there is a tokamak now -- a `TokamakProcess` whose
        device slot holds twenty-six occupied slots of twenty-eight and whose shared
        subsystems are the same ones a stellarator gets (it *does* still refuse
        further in, on `i_cost_model = 1` and its kin -- see
        `test_machine.TOKAMAK_BASELINE_INDAT`). `istell in 1..5` still raises, on the
        five hardcoded machine presets.
    """
    switches = switches_from_indat(input_file)

    def pick(field, registry, default, **kw):
        return _slot_occupant(field, switches.get(field, default), registry, **kw)

    # The device is resolved first, on its own, because it decides which *class* is being
    # built and therefore which of the resolutions below are even asked. PROCESS's own
    # default is `istell = 0`, a tokamak, and that is now a device rather than a refusal:
    # a file that never mentions `istell` builds a `TokamakProcess`. `istell in 1..5`
    # still raises here, first, naming `istell` rather than whichever slot the
    # constructor happened to evaluate first.
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
    # Asked only on the superconducting arm, because that is the only branch of
    # `caller.py` that reads it. `superconducting_tf_coil_variables.py:194` is the
    # default. Until this line existed, a CroCo machine assembled silently as
    # cable-in-conduit -- see `_refuse_unported_switch`'s docstring for the measurement.
    if i_tf_sup is TFConductorModel.SUPERCONDUCTING:
        i_tf_turn_type = SuperconductingTFTurnType(
            int(switches.get("i_tf_turn_type", 1))
        )
        if i_tf_turn_type is not SuperconductingTFTurnType.CABLE_IN_CONDUIT:
            _refuse_unported_switch("i_tf_turn_type", i_tf_turn_type)
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
                SuperconductorCostModel(int(switches.get("supercond_cost_model", 0))),
                TF_MAGNET_COST_SUPERCONDUCTING,
            ),
        ),
    )
    # `i_p_coolant_pumping` decides five things in `power` and one in `.tokamak.
    # ccfe_hcpb`, and until this pass all six carried a hardcoded copy of the Helias
    # run's answer. Resolved once, here, and threaded -- `fwbs_variables.py:249` is the
    # default. The slot resolution comes *before* the value is threaded, the same
    # discipline `i_tf_sup` follows, so no unported value ever reaches an occupant's
    # static field.
    # `pfcoil_variables.py:230` -- the PF conductor decides, jointly with `i_tf_sup`,
    # whether the cryoplant runs at all.
    i_pf_conductor = PFConductorModel(int(switches.get("i_pf_conductor", 0)))
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

    if device is TokamakProcess:
        return TokamakProcess(
            # Everything device-specific, and no longer `Tokamak()`: twenty-six of
            # its twenty-eight slots have occupants, most of them switched. `i_tf_sup`,
            # `i_plasma_ignited`, `itart` and `i_tf_sc_mat` are *threaded* rather than
            # re-read, because a switch is answered once.
            tokamak=_tokamak_device(
                switches,
                numbers_from_indat(input_file),
                iteration_variables_from_indat(input_file),
                int_lists_from_indat(input_file),
                i_tf_sup,
                i_plasma_ignited,
                itart,
                i_tf_sc_mat,
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

    # ---- `istell == 6`: everything only a stellarator asks ------------------------
    #
    # Below the branch because a tokamak has none of it: no machine-config file, no
    # `isthtr`, and neither joint blanket dispatch. Reading them anyway would make a
    # tokamak refusable for a stellarator's reason -- `blktmodel = 1` refuses at
    # `blanket_neutronics()`, which a tokamak never reaches.
    machine_config = StellaratorMachineConfig(
        machine_config=read_stellarator_config_file(
            REFERENCE_STELLA_CONF if stella_conf is None else stella_conf
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
        "blktmodel_ipowerflow",
        _blanket_shield_power_arm(blktmodel, ipowerflow),
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
against (`istell = 6`, `i_plasma_pedestal = 0`, `i_cost_model = 0`; every other switch at
PROCESS's own default)."""


def graph_for(machine=None):
    """The assembled graph for one machine; `REFERENCE_MACHINE` if unstated.

    **There is no bare form to fall back to any more, and that is the point.** Five
    registration bugs here shared one root cause: a value copied from PROCESS's bare
    `*_variables.py` defaults rather than from the run modelled (`i_confinement_time` 34
    vs 38, `i_thermal_electric_conversion` 0 vs 2, `i_p_coolant_pumping` 2 vs 1,
    `i_plasma_ignited` 0 vs 1, and `i_cost_model` 1 vs 0, which left `.costs.coe` with no
    producer and 43 nodes unregistered), each found by the MDA harness after the fact.
    The first fix was to stop *defaulting* to the silent-IN.DAT graph; this argument's
    default has been `REFERENCE_MACHINE` since. The second is that the silent-IN.DAT
    graph can no longer be built at all -- `StellaratorProcess()` raises, because every
    switched slot lost its default -- so a machine comes from an IN.DAT or from an
    explicit `eqx.tree_at` on one, and from nowhere else.
    """
    return to_graph(REFERENCE_MACHINE if machine is None else machine)


GRAPH = graph_for()
"""`REFERENCE_MACHINE`'s graph -- the `stellarator_helias.IN.DAT` run this port is
validated against."""

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
