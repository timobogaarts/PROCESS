"""The tokamak's magnetic-field closure: toroidal-field components plus every total
field `Physics.run` derives from them.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.physics.physics import SurfaceAveragedPoloidalField
from functional_process.cottax.stellarator.plasma_physics import calculate_total_field
from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_fields import (
    calculate_plasma_inboard_toroidal_field,
    calculate_plasma_outboard_toroidal_field,
    calculate_toroidal_field_profile,
)

__all__ = [
    "calculate_toroidal_field_profile",
]


class TotalMagneticField(ExplicitFunction):
    """cottax node: `physics.py:373-376`, the on-axis total field."""

    b_plasma_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PlasmaInboardToroidalField(ExplicitFunction):
    """`physics.py:394-401`, the inboard-midplane toroidal field component."""

    b_plasma_inboard_toroidal = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_plasma_inboard_toroidal_field(
            b_plasma_toroidal_on_axis, rmajor, rminor
        )


class PlasmaOutboardToroidalField(ExplicitFunction):
    """`physics.py:403-409`, the outboard-midplane toroidal field component."""

    b_plasma_outboard_toroidal = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_plasma_outboard_toroidal_field(
            b_plasma_toroidal_on_axis, rmajor, rminor
        )


class TotalMagneticFieldInboard(ExplicitFunction):
    """`physics.py:386-392`, the inboard total field."""

    b_plasma_inboard_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_inboard_toroidal=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_inboard_toroidal, b_plasma_surface_poloidal_average
        )


class TotalMagneticFieldOutboard(ExplicitFunction):
    """`physics.py:378-384`, the outboard total field."""

    b_plasma_outboard_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_outboard_toroidal=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_outboard_toroidal, b_plasma_surface_poloidal_average
        )


class PlasmaFields(ModelNamespace):
    """`.tokamak.plasma_fields` -- the fields `PlasmaFields` produces for this graph."""

    surface_averaged_poloidal_field: SurfaceAveragedPoloidalField = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_plasma_current` -- the Ampere arm is written, the Peng arm
    (`i_plasma_current == 2`) calls `PlasmaCurrent.plascar_bpol` and is UNPORTED.
    """

    total_magnetic_field: TotalMagneticField = TotalMagneticField()
    """`.physics.b_plasma_total`. Unswitched."""

    plasma_inboard_toroidal_field: PlasmaInboardToroidalField = (
        PlasmaInboardToroidalField()
    )
    """`.physics.b_plasma_inboard_toroidal`. Unswitched."""

    plasma_outboard_toroidal_field: PlasmaOutboardToroidalField = (
        PlasmaOutboardToroidalField()
    )
    """`.physics.b_plasma_outboard_toroidal`. Unswitched."""

    total_magnetic_field_inboard: TotalMagneticFieldInboard = TotalMagneticFieldInboard()
    """`.physics.b_plasma_inboard_total`."""

    total_magnetic_field_outboard: TotalMagneticFieldOutboard = (
        TotalMagneticFieldOutboard()
    )
    """`.physics.b_plasma_outboard_total`."""
