"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial, the minimal closure for `.fwbs.r_cryostat_inboard`.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.cryostat import (
    calculate_external_cryo_geometry,
    calculate_r_cryostat_inboard,  # noqa: F401
)
from functional_process.cottax.paths import blanket, build, buildings, fwbs, pf_coil


class Cryostat(ExplicitFunction):
    """cottax node: `.tokamak.cryostat`, owning all seven fields
    `external_cryo_geometry` writes -- `.fwbs.r_cryostat_inboard` only until 2026-08-30.
    """

    r_cryostat_inboard = OutputInto(fwbs)
    dz_pf_cryostat = OutputInto(blanket)
    z_cryostat_half_inside = OutputInto(fwbs)
    dz_tf_cryostat = OutputInto(buildings)
    vol_cryostat_internal = OutputInto(fwbs)
    vol_cryostat = OutputInto(fwbs)
    dewmkg = OutputInto(fwbs)

    def __call__(
        self,
        r_pf_coil_outer=From(pf_coil),
        dr_pf_cryostat=From(fwbs),
        f_z_cryostat=From(build),
        z_pf_coil_upper=From(pf_coil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        dr_cryostat=From(build),
        vol_vv=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_external_cryo_geometry(
            r_pf_coil_outer,
            dr_pf_cryostat,
            f_z_cryostat,
            z_pf_coil_upper,
            z_tf_inside_half,
            dr_tf_inboard,
            dr_cryostat,
            vol_vv,
            den_steel,
        )
