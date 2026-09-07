"""Pure-functional port of `process/models/physics/density_limit.py`
(`PlasmaDensityLimit` -- the **tokamak** density limit).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    ModelNamespace,
    Output,
    OutputInto,
)

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


class GreenwaldDensityLimit(ExplicitFunction):
    """Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`."""

    nd_plasma_electron_max_array_7 = Output(physics.nd_plasma_electron_max_array[6])

    def __call__(self, plasma_current=From(physics), rminor=From(physics)):
        return calculate_greenwald_density_limit(c_plasma=plasma_current, rminor=rminor)


class EnforcedDensityLimitGreenwald(ExplicitFunction):
    """The `i_density_limit == 7` (GREENWALD) occupant: selects the array element
    `GreenwaldDensityLimit` already produced.
    """

    nd_plasma_electrons_max = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_max_array_7=FromExactly(
            physics.nd_plasma_electron_max_array[6]
        ),
    ):
        return select_enforced_density_limit_greenwald(nd_plasma_electron_max_array_7)


class GreenwaldFraction(ExplicitFunction):
    """Unconditional producer of `.physics.f_nd_plasma_greenwald`."""

    f_nd_plasma_greenwald = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        nd_plasma_electron_max_array_7=FromExactly(
            physics.nd_plasma_electron_max_array[6]
        ),
    ):
        return calculate_greenwald_fraction(
            nd_plasma_electron_line, nd_plasma_electron_max_array_7
        )


class TokamakDensityLimit(ModelNamespace):
    """`.tokamak.density_limit` -- three slots, one switched."""

    greenwald_density_limit: GreenwaldDensityLimit = GreenwaldDensityLimit()
    """Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`."""

    enforced_density_limit: EnforcedDensityLimitGreenwald = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_density_limit` -- eight values, one occupant."""

    greenwald_fraction: GreenwaldFraction = GreenwaldFraction()
    """Unconditional producer of `.physics.f_nd_plasma_greenwald`."""
