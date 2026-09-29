"""The devices: `StellaratorProcess` and `TokamakProcess`, one slot per subsystem."""

import dataclasses

import equinox as eqx

from functional_process.cottax.models.availability.namespace import Availability
from functional_process.cottax.models.buildings.namespace import Buildings
from functional_process.cottax.models.costs.namespace import Costs
from functional_process.cottax.models.initialisation import Initialisation
from functional_process.cottax.models.physics.namespace import Physics
from functional_process.cottax.models.power.namespace import Power
from functional_process.cottax.models.stellarator.namespace import Stellarator
from functional_process.cottax.models.tokamak.namespace import Tokamak
from functional_process.cottax.models.vacuum.namespace import Vacuum


class StellaratorProcess(eqx.Module):
    """One device, configured whole: every slot in this port, and what fills it."""

    initialisation: Initialisation = dataclasses.field(kw_only=True)
    """`process/core/init.py` and `st_init`'s own writes, as nodes."""

    costs: Costs = dataclasses.field(kw_only=True)
    """The cost model (`.costs.i_cost_model`), and a slot with exactly one occupant."""

    stellarator: Stellarator = dataclasses.field(kw_only=True)

    physics: Physics = dataclasses.field(kw_only=True)

    power: Power = dataclasses.field(kw_only=True)

    buildings: Buildings = dataclasses.field(kw_only=True)

    vacuum: Vacuum = dataclasses.field(kw_only=True)
    """The pumping model, its occupant chosen by the pump type and dwell switches."""

    availability: Availability = dataclasses.field(kw_only=True)


class TokamakProcess(eqx.Module):
    """A conventional tokamak: the shared subsystems, and one empty device slot."""

    initialisation: Initialisation = dataclasses.field(kw_only=True)
    """The seed's own writes, as nodes -- the same slot, and the same occupants, as on a
    stellarator.
    """

    costs: Costs = dataclasses.field(kw_only=True)
    """The cost model (`.costs.i_cost_model`), and the same one slot the stellarator
    has.
    """

    tokamak: Tokamak = dataclasses.field(kw_only=True)
    """Everything device-specific -- twenty-six of twenty-eight slots filled."""

    physics: Physics = dataclasses.field(kw_only=True)
    """The shared plasma-physics core -- 31 of its 33 nodes are device-agnostic."""

    power: Power = dataclasses.field(kw_only=True)
    """Thermal and electric power flows -- 21 nodes, no `.stellarator*` read among them.
    """

    buildings: Buildings = dataclasses.field(kw_only=True)

    vacuum: Vacuum = dataclasses.field(kw_only=True)

    availability: Availability = dataclasses.field(kw_only=True)
    """Plant availability -- 4 nodes, and the one place the two devices differ by a
    single stack frame.
    """
