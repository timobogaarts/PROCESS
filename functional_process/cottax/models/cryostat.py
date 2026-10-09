"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial, the minimal closure for `.fwbs.r_cryostat_inboard`.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.assemblies.jax import V

from functional_process.cottax.paths import fwbs
from functional_process.models.cryostat import (
    calculate_external_cryo_geometry,
    calculate_r_cryostat_inboard,  # noqa: F401
)

# cottax node: `.tokamak.cryostat`, owning all seven fields
# `external_cryo_geometry` writes -- `.fwbs.r_cryostat_inboard` only until 2026-08-30.
Cryostat = fwbs.function(
    calculate_external_cryo_geometry,
    "r_cryostat_inboard",
    V.blanket.dz_pf_cryostat,
    "z_cryostat_half_inside",
    V.buildings.dz_tf_cryostat,
    "vol_cryostat_internal",
    "vol_cryostat",
    "dewmkg",
)
