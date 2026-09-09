"""The buildings subsystem's namespace."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.models.buildings.buildings import (
    Bldgs,
    BldgsSizesBase,
    TfCoilEnvelope,
)


class Buildings(ModelNamespace):
    """Plant buildings."""

    sizing: Bldgs | BldgsSizesBase = dataclasses.field(kw_only=True)
    """Which building-size model runs (`.buildings.i_bldgs_size`, default 0 = ITER
    1992) -- `BldgsSizesBase`'s occupant further depends on
    `.current_drive.i_hcd_primary`'s method (`BldgsSizesNeutralBeam`/
    `BldgsSizesOtherHcd`).
    """

    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    tf_coil_envelope: TfCoilEnvelope = TfCoilEnvelope()
