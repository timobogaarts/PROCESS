"""`functional_process.vocabulary` equals PROCESS's -- the price of vendoring it.

§23.1 measured the model layer's dependence on `process` and found it is vocabulary:
constants, switch enums, two data tables, a list of area names. §23.2 moved those into
`functional_process/vocabulary/` so the runtime carries no import, and this file is the
other half of that decision, without which it should not have been made.

Unit #8 declined to re-type `STELLARATOR_MACHINE_PRESETS` on the grounds that
"transcription buys a non-importing test and pays with a drift mode". The reasoning is
correct and the mitigation is exactly here: **`tests/` is where `process` is importable**
(`CLAUDE.md` § the environment -- co-importability in one interpreter is the harness's
whole design), so drift between the vendored copy and PROCESS fails a test instead of
silently changing an answer.

What is asserted, per kind:

- **constants** -- every public name in PROCESS's `constants` module, by value *and* by
  type, plus the reverse direction so a name added upstream is caught rather than quietly
  missing here.
- **enums** -- the full member `name -> value` mapping in both directions, so a renamed
  member, a changed integer, an added member and a dropped member each fail. Plus every
  `DynamicClassAttribute` PROCESS hangs off the member tuples -- most of these classes
  carry one (`full_name`, `description`, `abbreviation`) and two are branched on
  (`CurrentDriveModel.method`, `SuperconductorModel.sc_shape`). The attribute *list* is
  read off PROCESS, not typed here, so one added upstream is covered immediately.
- **`ITERATION_VARIABLES`** -- the whole table: the same ids, and per id every field of
  `IterationVariable` (`name`, `module`, bounds, `target_name`, `array_index`).
- **the stellarator presets** -- plain dict equality, all five.
- **`AREAS`** -- equal to `[f.name for f in dataclasses.fields(DataStructure)]`, order
  included, because `paths.py` used to compute exactly that expression.

The tests import PROCESS directly and deliberately. Nothing under `functional_process/`
may.
"""

import dataclasses
import importlib
from enum import Enum, IntEnum
from types import DynamicClassAttribute

import pytest

from functional_process import vocabulary

# --- the reference side: PROCESS itself ------------------------------------------------

import process.core.constants as process_constants  # isort: skip
from process.core.model import DataStructure  # isort: skip
from process.core.solver.iteration_variables import (  # isort: skip
    ITERATION_VARIABLES as PROCESS_ITERATION_VARIABLES,
)
from process.models.stellarator import preset_config as process_presets  # isort: skip


def _public(module):
    """`name -> value` for every public, non-module attribute of `module`."""
    return {
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("_")
        and not isinstance(getattr(module, name), type(module))
    }


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------


def test_constants_name_set_matches():
    """Neither side has a public name the other lacks.

    The reverse direction is the one that matters: a constant *added* to PROCESS would
    otherwise sit unnoticed until some ported model wanted it and read a stale copy.
    """
    assert set(_public(vocabulary.constants)) == set(_public(process_constants))


@pytest.mark.parametrize("name", sorted(_public(process_constants)))
def test_constant_value_and_type_match(name):
    """Each constant is identical in value *and* type.

    Type as well as value because these are float64 physics constants being handed to
    JAX: an `int` where PROCESS has a `float` is a weak-typing difference that survives
    every value test and shows up as promotion behaviour.
    """
    ours = getattr(vocabulary.constants, name)
    theirs = getattr(process_constants, name)
    assert type(ours) is type(theirs), name
    assert ours == theirs, name


# --------------------------------------------------------------------------------------
# switch enums
# --------------------------------------------------------------------------------------

ENUMS = {
    "BlktModelTypes": "process.data_structure.blanket_variables",
    "TFCSRadialConfiguration": "process.data_structure.build_variables",
    "DivertorHeatLoadModel": "process.data_structure.divertor_variables",
    "FiguresOfMerit": "process.data_structure.numerics",
    "PFConductorModel": "process.data_structure.pfcoil_variables",
    "ConfinementMode": "process.data_structure.physics_variables",
    "ConfinementRadiationLossModel": "process.data_structure.physics_variables",
    "ConfinementTimeModel": "process.data_structure.physics_variables",
    "CurrentProfileIndexModel": "process.data_structure.physics_variables",
    "DivertorNumberModels": "process.data_structure.physics_variables",
    "OutbordSOLPowerDecayLengthModel": "process.data_structure.physics_variables",
    "PlasmaIgnitionModel": "process.data_structure.physics_variables",
    "TFWPIntegerTurnType": "process.data_structure.superconducting_tf_coil_variables",
    "AvailabilityModel": "process.models.availability",
    "FwBlktVVShape": "process.models.build",
    "BootstrapCurrentFractionModel": "process.models.physics.bootstrap_current",
    "CurrentDriveMethodType": "process.models.physics.current_drive",
    "CurrentDriveModel": "process.models.physics.current_drive",
    "DensityLimitModel": "process.models.physics.density_limit",
    "PlasmaConfinementTransitionModel": "process.models.physics.l_h_transition",
    "BetaComponentLimits": "process.models.physics.physics",
    "IndInternalNormModel": "process.models.physics.physics",
    "PlasmaCurrentModel": "process.models.physics.plasma_current",
    "PlasmaDiamagneticCurrentModel": "process.models.physics.plasma_current",
    "PlasmaGeometryModelType": "process.models.physics.plasma_geometry",
    "PlasmaGeometryModels": "process.models.physics.plasma_geometry",
    "PlasmaShapeModelType": "process.models.physics.plasma_geometry",
    "ElectricConversionModelTypes": "process.models.power",
    "PumpingPowerModelTypes": "process.models.power",
    "SuperconductorMaterial": "process.models.superconductors",
    "SuperconductorModel": "process.models.superconductors",
    "SuperconductorShape": "process.models.superconductors",
    "SuperconductorType": "process.models.superconductors",
    "TFCoilShapeModel": "process.models.tfcoil.base",
    "TFConductorModel": "process.models.tfcoil.base",
    "TFPlasmaCaseType": "process.models.tfcoil.base",
    "SuperconductingTFTurnType": "process.models.tfcoil.superconducting",
    "SuperconductingTFWPShapeType": "process.models.tfcoil.superconducting",
}
"""Every enum `functional_process/vocabulary/enums.py` and `superconductors.py` vendor,
against the PROCESS module it was taken from. The list is also the coverage claim: a
vendored enum missing a row here is a vendored value with no equality test, which is the
one thing §23.2 forbids -- `test_every_vendored_enum_is_covered` enforces that."""


def _reference_enum(name):
    return getattr(importlib.import_module(ENUMS[name]), name)


def test_every_vendored_enum_is_covered():
    """`ENUMS` names every `IntEnum` the vocabulary exports, and nothing else.

    Without this the parametrised tests below silently stop covering an enum the moment
    someone vendors one and forgets the row -- exactly the drift mode this file exists to
    close.
    """
    exported = {
        name
        for name in vocabulary.__all__
        if isinstance(getattr(vocabulary, name), type)
        and issubclass(getattr(vocabulary, name), IntEnum)
    }
    assert exported == set(ENUMS)


@pytest.mark.parametrize("name", sorted(ENUMS))
def test_enum_members_match(name):
    """Same member names, same integer values, both directions."""
    ours = getattr(vocabulary, name)
    theirs = _reference_enum(name)
    assert {m.name: int(m.value) for m in ours} == {
        m.name: int(m.value) for m in theirs
    }, name


@pytest.mark.parametrize("name", sorted(ENUMS))
def test_enum_base_matches(name):
    """Same enum flavour.

    Every one of these is an `IntEnum` today; a switch read from an `IN.DAT` is compared
    against these as an integer, so a member that stopped being `int`-valued would break
    that comparison silently rather than loudly.
    """
    assert issubclass(getattr(vocabulary, name), IntEnum), name
    assert issubclass(_reference_enum(name), IntEnum), name


def _attached(cls):
    """The `DynamicClassAttribute` names PROCESS hangs off `cls`'s member tuples.

    Read off PROCESS's class rather than listed here, so an attribute added upstream is
    covered the moment it exists. These are the whole reason the enums are vendored as
    *source* rather than as generated `NAME = value` stubs -- most of these classes carry
    a table (`full_name`, `description`, `abbreviation`, `CurrentDriveModel.method`,
    `SuperconductorModel.sc_shape`, `PlasmaGeometryModelType.kappa_model`), and
    `indat.py` branches on two of them.
    """
    return sorted(
        name
        for name, value in vars(cls).items()
        if isinstance(value, DynamicClassAttribute)
    )


ATTACHED = {
    name: _attached(_reference_enum(name))
    for name in ENUMS
    if _attached(_reference_enum(name))
}


def test_some_enums_carry_attached_attributes():
    """`ATTACHED` is non-trivial.

    A refactor upstream that dropped `DynamicClassAttribute` for something else would
    empty this map and quietly turn every attribute test below into nothing. Naming the
    two that `indat.py` actually branches on pins the ones that matter.
    """
    assert "method" in ATTACHED["CurrentDriveModel"]
    assert "sc_shape" in ATTACHED["SuperconductorModel"]
    assert len(ATTACHED) > 15


@pytest.mark.parametrize(
    ("name", "attribute"),
    [(n, a) for n, attrs in sorted(ATTACHED.items()) for a in attrs],
)
def test_attached_attribute_matches(name, attribute):
    """Per member, the attached attribute equals PROCESS's.

    Compared by value rather than by identity because the vendored enums are genuinely
    different classes: `vocabulary.SuperconductorModel.CROCO_REBCO.sc_shape` is the
    *vendored* `SuperconductorShape.TAPE`, and `is` against PROCESS's would fail for a
    reason that says nothing about drift. An `Enum`-valued attribute is compared on
    `(name, value)`, which is exactly what makes two copies of a table the same table.
    """

    def comparable(v):
        return (type(v).__name__, v.name, v.value) if isinstance(v, Enum) else v

    ours = getattr(vocabulary, name)
    theirs = _reference_enum(name)
    for member in theirs:
        mine = getattr(getattr(ours, member.name), attribute)
        yours = getattr(member, attribute)
        assert comparable(mine) == comparable(yours), f"{name}.{member.name}.{attribute}"


# --------------------------------------------------------------------------------------
# ITERATION_VARIABLES
# --------------------------------------------------------------------------------------


def test_iteration_variable_ids_match():
    """The same set of ids, which is what an `IN.DAT`'s `ixc` list indexes."""
    assert set(vocabulary.ITERATION_VARIABLES) == set(PROCESS_ITERATION_VARIABLES)


@pytest.mark.parametrize("var_id", sorted(PROCESS_ITERATION_VARIABLES))
def test_iteration_variable_matches(var_id):
    """Every field of every entry, including the two that are usually `None`.

    `target_name` and `array_index` are the ones worth naming: they are how ids 125/126
    address one element of `f_nd_impurity_electron_array` under a different display name
    (`CLAUDE.md` § Difficulties), so a dropped `array_index` would silently retarget a
    whole array field.
    """
    ours = vocabulary.ITERATION_VARIABLES[var_id]
    theirs = PROCESS_ITERATION_VARIABLES[var_id]
    assert dataclasses.asdict(ours) == dataclasses.asdict(theirs), var_id


def test_iteration_variable_dataclass_fields_match():
    """The vendored `IterationVariable` has PROCESS's field list.

    `dataclasses.asdict` above compares whatever fields exist; this is what makes that
    comparison meaningful if PROCESS ever adds one.
    """
    assert [f.name for f in dataclasses.fields(vocabulary.IterationVariable)] == [
        f.name
        for f in dataclasses.fields(
            type(next(iter(PROCESS_ITERATION_VARIABLES.values())))
        )
    ]


# --------------------------------------------------------------------------------------
# stellarator presets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["HELIAS5B", "HELIAS4", "HELIAS3", "W7X30", "W7X50"])
def test_stellarator_preset_matches(name):
    """Dict equality against PROCESS's -- the test unit #8 said transcription would need.

    Whole-dict rather than key-by-key, so an added or dropped key fails as loudly as a
    changed number.
    """
    assert getattr(vocabulary, name) == getattr(process_presets, name)


# --------------------------------------------------------------------------------------
# DataStructure area names
# --------------------------------------------------------------------------------------


def test_areas_match():
    """`AREAS` is still `DataStructure`'s field list, in order.

    Order because `paths.py` exposes it as a tuple and `__all__` is built from it; a
    reordering would not change behaviour today, but an *added* area silently missing
    from the vendored tuple would make `data.<new_area>` raise at declaration time with a
    "did you mean" that named the wrong thing.
    """
    assert list(vocabulary.AREAS) == [f.name for f in dataclasses.fields(DataStructure)]
