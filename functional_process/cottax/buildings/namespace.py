"""The buildings subsystem's namespace."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.buildings.buildings import (
    Bldgs,
    BldgsSizes,
    TfCoilEnvelope,
)


class Buildings(ModelNamespace):
    """Plant buildings."""

    sizing: Bldgs | BldgsSizes = dataclasses.field(kw_only=True)
    """Which building-size model runs (`.buildings.i_bldgs_size`, default 0 = ITER
    1992).
    """

    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    tf_coil_envelope: TfCoilEnvelope = TfCoilEnvelope()
