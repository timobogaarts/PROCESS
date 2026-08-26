"""The tokamak's total magnetic field -- one node, and no new pure function.

`process/models/physics/plasma_fields.py::PlasmaFields.calculate_total_magnetic_field`
is `sqrt(b_toroidal**2 + b_poloidal**2)`, and `Physics.run` calls it three times
(`physics.py:373-392`) for the on-axis, outboard and inboard totals. Only the first,
`.physics.b_plasma_total`, has a consumer in the assembled tokamak graph --
`physics.py::PlasmaEnergyFromBeta` reads it -- so only the first has a node here. The
other two are one line each whenever `.physics.b_plasma_outboard_total` or
`b_plasma_inboard_total` gains a reader.

**The formula is imported, not rewritten.** `models/stellarator/plasma_physics.py::
calculate_total_field` is the same expression, already ported and already
harness-tested (`stellarator.py:1916-1919` and `plasma_fields.py:104-119` are the same
Pythagorean sum with the same two arguments). Re-deriving it here would be a second
transcription of one formula, which is the defect class this port keeps finding; what a
tokamak needs is a **node of its own**, because a node is named by the slot that holds
it and `.stellarator.total_field` is not a place a `TokamakProcess` has.

`physics.md` open question 1 asked for exactly this and declined to write it, on the
ground that `plasma_fields.py` was not that agent's file: *"Either that node is
generalised, or `plasma_fields.py` gains a second occupant. One line either way."* This
is the second occupant, and the argument for it over generalising the stellarator's is
`base.md` §"Shared with the stellarator": one `VarPath`, two device-specific producers,
two slots, and any assembled machine binds exactly one.

No harness contract of its own: the pure function it calls already has one
(`tests/functional_process/models/stellarator/test_plasma_physics.py`, the
`calculate_total_field` contract), and a second contract over the same function would
diff PROCESS against itself. What this module adds is a binding, and bindings are what
the MDA harness checks.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.models.physics.physics import SurfaceAveragedPoloidalField
from functional_process.models.stellarator.plasma_physics import calculate_total_field
from functional_process.paths import physics


class TotalMagneticField(ExplicitFunction):
    """cottax node: `physics.py:373-376`, the on-axis total field.

    Reads the toroidal field on axis and the surface-averaged poloidal field --  the
    latter being `SurfaceAveragedPoloidalFieldAmperes`' own output, so this node sits
    immediately downstream of its slot-mate and the edge between them is the only one
    inside `.tokamak.plasma_fields`.

    Unswitched. `calculate_total_magnetic_field` has no branch at all; which *arguments*
    it is called with is what `Physics.run` varies across its three call sites, and that
    is a call-site fact rather than a switch.
    """

    b_plasma_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PlasmaFields(ModelNamespace):
    """`.tokamak.plasma_fields` -- the two fields `PlasmaFields` produces for this graph.

    `process/models/physics/plasma_fields.py::PlasmaFields` is injected into `Physics`
    (`physics.py:197`) rather than run from `caller.py`, which is why it is a slot of
    `Tokamak` in its own right: `tokamak_call_surface.md` §A recorded it entered, and a
    sub-model reached by injection is a slot where §A named it.

    The stellarator's counterpart is `.stellarator.poloidal_field_from_rotational_
    transform` plus `.stellarator.total_field` -- the same two quantities from a
    rotational transform instead of a plasma current.
    """

    surface_averaged_poloidal_field: SurfaceAveragedPoloidalField = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_plasma_current` -- the Ampere arm is written, the Peng arm
    (`i_plasma_current == 2`) calls `PlasmaCurrent.plascar_bpol` and is UNPORTED.
    Factory-filled; the annotation is the family base class."""

    total_magnetic_field: TotalMagneticField = TotalMagneticField()
    """`.physics.b_plasma_total`. Unswitched."""
