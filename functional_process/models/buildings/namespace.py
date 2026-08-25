"""The buildings subsystem's namespace.

Beside the nodes it names (`model_tree_design.md` §11).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.buildings.buildings import (
    Bldgs,
    BldgsSizes,
    TfCoilEnvelope,
)


class Buildings(ModelNamespace):
    """Plant buildings."""

    sizing: Bldgs | BldgsSizes = dataclasses.field(kw_only=True)
    """Which building-size model runs (`.buildings.i_bldgs_size`, default 0 = ITER 1992).

    The two occupants share `a_plant_floor_effective`/`volnucb`, which is what proved
    them mutually exclusive back when exclusivity had to be *detected* from colliding
    output ownership. It is now by construction: one slot, one occupant.
    """

    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    tf_coil_envelope: TfCoilEnvelope = TfCoilEnvelope()
