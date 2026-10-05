"""Pure-functional port of `process/models/physics/density_limit.py`
(`PlasmaDensityLimit` -- the **tokamak** density limit).
"""

import dataclasses

import equinox as eqx
from cottax.relational_pytree import Explicit, V

from functional_process.cottax.paths import physics
from functional_process.models.physics.density_limit import (
    calculate_asdex_density_limit,
    calculate_asdex_new_density_limit,
    calculate_borrass_iter_i_density_limit,
    calculate_borrass_iter_ii_density_limit,
    calculate_greenwald_density_limit,
    calculate_greenwald_fraction,
    calculate_hugill_murakami_density_limit,
    calculate_jet_edge_radiation_density_limit,
    calculate_jet_simple_density_limit,
    select_enforced_density_limit_greenwald,
)

__all__ = [
    "calculate_asdex_density_limit",
    "calculate_asdex_new_density_limit",
    "calculate_borrass_iter_i_density_limit",
    "calculate_borrass_iter_ii_density_limit",
    "calculate_hugill_murakami_density_limit",
    "calculate_jet_edge_radiation_density_limit",
    "calculate_jet_simple_density_limit",
]


# Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`.
GreenwaldDensityLimit = physics.function(
    calculate_greenwald_density_limit,
    V.physics.nd_plasma_electron_max_array[6],
    reads={
        "c_plasma": V.physics.plasma_current,
    },
)


# The `i_density_limit == 7` (GREENWALD) occupant: selects the array element
# `GreenwaldDensityLimit` already produced.
EnforcedDensityLimitGreenwald = physics.function(
    select_enforced_density_limit_greenwald,
    "nd_plasma_electrons_max",
    reads={
        "nd_plasma_electron_max_array_7": V.physics.nd_plasma_electron_max_array[6],
    },
)


# Unconditional producer of `.physics.f_nd_plasma_greenwald`.
GreenwaldFraction = physics.function(
    calculate_greenwald_fraction,
    "f_nd_plasma_greenwald",
    reads={
        "nd_plasma_electron_max_array_7": V.physics.nd_plasma_electron_max_array[6],
    },
)


class TokamakDensityLimit(eqx.Module):
    """`.tokamak.density_limit` -- three slots, one switched."""

    greenwald_density_limit: Explicit = GreenwaldDensityLimit
    """Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`."""

    enforced_density_limit: Explicit = dataclasses.field(kw_only=True)
    """`.physics.i_density_limit` -- eight values, one occupant."""

    greenwald_fraction: Explicit = GreenwaldFraction
    """Unconditional producer of `.physics.f_nd_plasma_greenwald`."""
