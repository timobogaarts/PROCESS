"""PROCESS's *vocabulary*, vendored so the model layer imports with no `process` present.

§23.1 measured the model layer's dependency on PROCESS and found it is not physics: it is
physical constants, ~30 switch enums, two data tables and a list of area names -- numbers
and declarations, no behaviour. This package is those, copied.

§23.2 is the rule that makes the copy safe and it is not optional: **vendor for runtime,
assert equality in tests**. `tests/functional_process/test_vocabulary.py` asserts every
name here equals PROCESS's, and it lives in `tests/`, where `process` *is* importable.
Unit #8 chose to import the stellarator presets rather than re-type them because
"transcription buys a non-importing test and pays with a drift mode"; the equality test
is what pays that debt back, and a vendored value without one is exactly the defect that
record warned about.

Nothing here is re-derived, reformatted or tidied. Where PROCESS's source could be copied
verbatim it was (`constants.py`, `stellarator_presets.py`, the `ITERATION_VARIABLES`
literal); `enums.py` and `areas.py` were *generated* by introspecting PROCESS rather than
retyped, for the same reason.
"""

from functional_process.vocabulary import constants
from functional_process.vocabulary.areas import AREAS
from functional_process.vocabulary.enums import (
    AvailabilityModel,
    BetaComponentLimits,
    BlktModelTypes,
    BootstrapCurrentFractionModel,
    ConfinementMode,
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    CurrentDriveMethodType,
    CurrentDriveModel,
    CurrentProfileIndexModel,
    DensityLimitModel,
    DivertorHeatLoadModel,
    DivertorNumberModels,
    ElectricConversionModelTypes,
    FiguresOfMerit,
    FwBlktVVShape,
    IndInternalNormModel,
    OutbordSOLPowerDecayLengthModel,
    PFConductorModel,
    PlasmaConfinementTransitionModel,
    PlasmaCurrentModel,
    PlasmaDiamagneticCurrentModel,
    PlasmaGeometryModels,
    PlasmaGeometryModelType,
    PlasmaIgnitionModel,
    PlasmaShapeModelType,
    PumpingPowerModelTypes,
    SuperconductingTFTurnType,
    SuperconductingTFWPShapeType,
    TFCoilShapeModel,
    TFConductorModel,
    TFCSRadialConfiguration,
    TFPlasmaCaseType,
    TFWPIntegerTurnType,
)
from functional_process.vocabulary.exceptions import ProcessError, ProcessValueError
from functional_process.vocabulary.iteration_variables import (
    ITERATION_VARIABLES,
    IterationVariable,
)
from functional_process.vocabulary.stellarator_presets import (
    HELIAS3,
    HELIAS4,
    HELIAS5B,
    W7X30,
    W7X50,
)
from functional_process.vocabulary.superconductors import (
    SuperconductorMaterial,
    SuperconductorModel,
    SuperconductorShape,
    SuperconductorType,
)

__all__ = [
    "AREAS",
    "HELIAS3",
    "HELIAS4",
    "HELIAS5B",
    "ITERATION_VARIABLES",
    "W7X30",
    "W7X50",
    "AvailabilityModel",
    "BetaComponentLimits",
    "BlktModelTypes",
    "BootstrapCurrentFractionModel",
    "ConfinementMode",
    "ConfinementRadiationLossModel",
    "ConfinementTimeModel",
    "CurrentDriveMethodType",
    "CurrentDriveModel",
    "CurrentProfileIndexModel",
    "DensityLimitModel",
    "DivertorHeatLoadModel",
    "DivertorNumberModels",
    "ElectricConversionModelTypes",
    "FiguresOfMerit",
    "FwBlktVVShape",
    "IndInternalNormModel",
    "IterationVariable",
    "OutbordSOLPowerDecayLengthModel",
    "PFConductorModel",
    "PlasmaConfinementTransitionModel",
    "PlasmaCurrentModel",
    "PlasmaDiamagneticCurrentModel",
    "PlasmaGeometryModelType",
    "PlasmaGeometryModels",
    "PlasmaIgnitionModel",
    "PlasmaShapeModelType",
    "ProcessError",
    "ProcessValueError",
    "PumpingPowerModelTypes",
    "SuperconductingTFTurnType",
    "SuperconductingTFWPShapeType",
    "SuperconductorMaterial",
    "SuperconductorModel",
    "SuperconductorShape",
    "SuperconductorType",
    "TFCSRadialConfiguration",
    "TFCoilShapeModel",
    "TFConductorModel",
    "TFPlasmaCaseType",
    "TFWPIntegerTurnType",
    "constants",
]
