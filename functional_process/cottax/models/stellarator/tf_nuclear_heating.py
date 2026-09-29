"""Pure-functional port of `Stellarator.sc_tf_coil_nuclear_heating_iter90` (chunk 1F)."""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import (
    fwbs,
)
from functional_process.models.stellarator.tf_nuclear_heating import (
    calculate_sc_tf_coil_nuclear_heating,
)

# cottax node: `calculate_sc_tf_coil_nuclear_heating`, unchanged, ports declared.
ScTfCoilNuclearHeating = fwbs.function(
    calculate_sc_tf_coil_nuclear_heating,
    "coilhtmx",
    "dpacop",
    "htheci",
    "flu_tf_neutron_fast_peak",
    "pheci",
    "pheco",
    "ptfiwp",
    "ptfowp",
    "raddose",
    "p_tf_nuclear_heat_mw",
)
