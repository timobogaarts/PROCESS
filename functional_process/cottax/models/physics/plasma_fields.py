"""The tokamak's magnetic-field closure: toroidal-field components plus every total
field `Physics.run` derives from them.
"""

import dataclasses

import equinox as eqx
from cottax.cx import Explicit, V

from functional_process.cottax.models.stellarator.plasma_physics import (
    calculate_total_field,
)
from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_fields import (
    calculate_plasma_inboard_toroidal_field,
    calculate_plasma_outboard_toroidal_field,
    calculate_toroidal_field_profile,
)

__all__ = [
    "calculate_toroidal_field_profile",
]


# cottax node: `physics.py:373-376`, the on-axis total field.
TotalMagneticField = physics.function(calculate_total_field, "b_plasma_total")


# `physics.py:394-401`, the inboard-midplane toroidal field component.
PlasmaInboardToroidalField = physics.function(
    calculate_plasma_inboard_toroidal_field, "b_plasma_inboard_toroidal"
)


# `physics.py:403-409`, the outboard-midplane toroidal field component.
PlasmaOutboardToroidalField = physics.function(
    calculate_plasma_outboard_toroidal_field, "b_plasma_outboard_toroidal"
)


# `physics.py:386-392`, the inboard total field.
TotalMagneticFieldInboard = physics.function(
    calculate_total_field,
    "b_plasma_inboard_total",
    reads={
        "b_plasma_toroidal_on_axis": V.physics.b_plasma_inboard_toroidal,
    },
)


# `physics.py:378-384`, the outboard total field.
TotalMagneticFieldOutboard = physics.function(
    calculate_total_field,
    "b_plasma_outboard_total",
    reads={
        "b_plasma_toroidal_on_axis": V.physics.b_plasma_outboard_toroidal,
    },
)


class PlasmaFields(eqx.Module):
    """`.tokamak.plasma_fields` -- the fields `PlasmaFields` produces for this graph."""

    surface_averaged_poloidal_field: Explicit = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_current` -- the Ampere arm is written, the Peng arm
    (`i_plasma_current == 2`) calls `PlasmaCurrent.plascar_bpol` and is UNPORTED.
    """

    total_magnetic_field: Explicit = TotalMagneticField
    """`.physics.b_plasma_total`. Unswitched."""

    plasma_inboard_toroidal_field: Explicit = PlasmaInboardToroidalField
    """`.physics.b_plasma_inboard_toroidal`. Unswitched."""

    plasma_outboard_toroidal_field: Explicit = PlasmaOutboardToroidalField
    """`.physics.b_plasma_outboard_toroidal`. Unswitched."""

    total_magnetic_field_inboard: Explicit = TotalMagneticFieldInboard
    """`.physics.b_plasma_inboard_total`."""

    total_magnetic_field_outboard: Explicit = TotalMagneticFieldOutboard
    """`.physics.b_plasma_outboard_total`."""
