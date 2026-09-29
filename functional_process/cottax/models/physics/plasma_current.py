"""Pure-functional port of the tokamak plasma-current chain."""

import dataclasses

import equinox as eqx
from cottax.interfaces import Explicit

from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_current import (
    calculate_current_coefficient_fiesta,
    calculate_current_coefficient_ipdg89,
    calculate_current_profile_index_wesson,
    calculate_cyclindrical_plasma_current,
    calculate_cylindrical_safety_factor,
    calculate_internal_inductance_wesson,
    calculate_plasma_current_fiesta,
    calculate_plasma_current_ipdg89,
)

__all__ = [
    "calculate_current_coefficient_fiesta",
    "calculate_current_coefficient_ipdg89",
    "calculate_cyclindrical_plasma_current",
]


# The PlasmaCurrentScaling family:
# The family that owns `.physics.plasma_current` under `i_plasma_current`.


# `i_plasma_current == IPDG89_SCALING` (4) -- the arm `large_tokamak_eval` takes.
Ipdg89PlasmaCurrent = physics.function(calculate_plasma_current_ipdg89, "plasma_current")


# `i_plasma_current == FIESTA_ST_SCALING` (9) -- the arm both tracked spherical
# tokamaks take (`spherical_tokamak_eval.IN.DAT:288`, `st_regression.IN.DAT`).
FiestaStPlasmaCurrent = physics.function(
    calculate_plasma_current_fiesta, "plasma_current"
)


# cottax node: `qstar`, ports declared.
PlasmaCylindricalSafetyFactor = physics.function(
    calculate_cylindrical_safety_factor, "qstar"
)


# The CurrentProfileIndexScaling family:
# The family that owns `.physics.alphaj` under `i_alphaj` (`physics.py:334-348`).


# `i_alphaj == WESSON` (1) -- `large_tokamak_eval.IN.DAT:275`.
WessonCurrentProfileIndex = physics.function(
    calculate_current_profile_index_wesson, "alphaj"
)


# The NormalisedInternalInductanceScaling family:
# The family that owns `.physics.ind_plasma_internal_norm` under
# `i_ind_plasma_internal_norm` (`physics.py:4738-4764`).


# `i_ind_plasma_internal_norm == WESSON` (1) -- `large_tokamak_eval.IN.DAT:311`.
WessonInternalInductance = physics.function(
    calculate_internal_inductance_wesson, "ind_plasma_internal_norm"
)


class TokamakPlasmaCurrent(eqx.Module):
    """`.tokamak.plasma_current` -- the plasma-current chain, three slots."""

    plasma_current: Explicit = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_current` -- nine values, one occupant."""

    cylindrical_safety_factor: Explicit = PlasmaCylindricalSafetyFactor
    """Unconditional -- PROCESS computes `qstar` outside every switch
    (`physics.py:303`).
    """

    current_profile_index: Explicit | None = dataclasses.field(kw_only=True)
    """`.physics.i_alphaj` -- `1` (Wesson) is written; `0` (USER_INPUT) is an **empty
    slot** (`physics.py:338` assigns the field to itself), under which `.physics.alphaj`
    is a boundary input with no producer.
    """
