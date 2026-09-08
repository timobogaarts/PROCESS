"""The devices: `StellaratorProcess` and `TokamakProcess`, one slot per subsystem."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.availability.namespace import Availability
from functional_process.cottax.buildings.namespace import Buildings
from functional_process.cottax.costs.namespace import Costs
from functional_process.cottax.initialisation import Initialisation
from functional_process.cottax.physics.namespace import Physics
from functional_process.cottax.power.namespace import Power
from functional_process.cottax.stellarator.namespace import Stellarator
from functional_process.cottax.tokamak.namespace import Tokamak
from functional_process.cottax.vacuum.namespace import Vacuum


class StellaratorProcess(ModelNamespace):
    """One device, configured whole: every slot in this port, and what fills it."""

    initialisation: Initialisation = dataclasses.field(kw_only=True)
    """`process/core/init.py` and `st_init`'s own writes, as nodes."""

    costs: Costs = dataclasses.field(kw_only=True)
    """The cost model (`.costs.i_cost_model`), and a slot with exactly one occupant."""

    stellarator: Stellarator = dataclasses.field(kw_only=True)

    physics: Physics = dataclasses.field(kw_only=True)

    power: Power = dataclasses.field(kw_only=True)

    buildings: Buildings = dataclasses.field(kw_only=True)

    vacuum: Vacuum = Vacuum()
    """The one sub-namespace that keeps a default: nothing inside it is switched, so
    there is nothing for `machine_from_indat` to decide and no configuration for a
    default to smuggle in.
    """

    availability: Availability = dataclasses.field(kw_only=True)


class TokamakProcess(ModelNamespace):
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

    vacuum: Vacuum = Vacuum()
    """Default for the same reason it is on a stellarator: nothing inside it is
    switched.
    """

    availability: Availability = dataclasses.field(kw_only=True)
    """Plant availability -- 4 nodes, and the one place the two devices differ by a
    single stack frame.
    """
