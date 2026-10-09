"""The buildings subsystem's namespace."""

import dataclasses

import equinox as eqx
from cottax.assemblies.jax import Explicit

from functional_process.cottax.models.buildings.buildings import TfCoilEnvelope


class Buildings(eqx.Module):
    """Plant buildings."""

    sizing: Explicit = dataclasses.field(kw_only=True)
    """Which building-size model runs (`.buildings.i_bldgs_size`, default 0 = ITER
    1992) -- `BldgsSizesBase`'s occupant further depends on
    `.current_drive.i_hcd_primary`'s method (`BldgsSizesNeutralBeam`/
    `BldgsSizesOtherHcd`).
    """

    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    tf_coil_envelope: Explicit = TfCoilEnvelope
