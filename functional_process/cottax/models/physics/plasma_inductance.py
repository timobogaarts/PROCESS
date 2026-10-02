"""Pure-functional port of `PlasmaInductance`, `process/models/physics/physics.py`
(lines 4702-5150).
"""

import dataclasses

import equinox as eqx
from cottax.cx import Explicit

from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_inductance import (
    calculate_internal_inductance_menard,
    calculate_internal_inductance_wesson,
    calculate_normalised_internal_inductance_iter_3,
    calculate_volt_second_requirements,
    internal_inductance_norm_scalings,
    internal_inductance_norm_wesson,
)

__all__ = [
    "calculate_internal_inductance_menard",
    "calculate_internal_inductance_wesson",
    "calculate_normalised_internal_inductance_iter_3",
]


# cottax node: `.tokamak.plasma_inductance.scalings`.
PlasmaInternalInductanceScalings = physics.function(
    internal_inductance_norm_scalings,
    "ind_plasma_internal_norm_wesson",
    "ind_plasma_internal_norm_menard",
    "ind_plasma_internal_norm_iter_3",
)


# cottax node: `.tokamak.plasma_inductance.internal_inductance_norm`.
PlasmaInternalInductanceNormWesson = physics.function(
    internal_inductance_norm_wesson, "ind_plasma_internal_norm"
)


# cottax node: `.tokamak.plasma_inductance.volt_seconds`.
PlasmaVoltSecondRequirements = physics.function(
    calculate_volt_second_requirements,
    "vs_plasma_internal",
    "ind_plasma",
    "vs_plasma_burn_required",
    "vs_plasma_ramp_required",
    "vs_plasma_ind_ramp",
    "vs_plasma_res_ramp",
    "vs_plasma_total_required",
    "v_plasma_loop_burn",
)


class TokamakPlasmaInductance(eqx.Module):
    """`.tokamak.plasma_inductance` -- three slots, one of them switched."""

    scalings: Explicit = PlasmaInternalInductanceScalings
    """Unconditional -- PROCESS evaluates all three scalings before any switch is read
    (`physics.py:4721-4736`).
    """

    internal_inductance_norm: Explicit | None = dataclasses.field(kw_only=True)
    """`.physics.i_ind_plasma_internal_norm` -- `1` (Wesson) is written; `0`
    (USER_INPUT) is **no node at all** (`physics.py:4760` selects the field from itself,
    so the field is a run input and the slot is empty); `2` (Menard) is UNPORTED.
    """

    volt_seconds: Explicit = PlasmaVoltSecondRequirements
    """Unconditional -- `calculate_volt_second_requirements` has no branch at all."""
