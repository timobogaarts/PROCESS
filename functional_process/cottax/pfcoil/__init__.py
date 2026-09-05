"""PF-coil graph declarations.

The package's constants and shape types (`PFLocation`, `PFCoilTopology`, the coil
counts and indices) are physics, not graph, so they live in
`functional_process.models.pfcoil` now and are re-exported here -- every module that
reached them through this package still does.
"""

from functional_process.models.pfcoil import *  # noqa: F403
from functional_process.models.pfcoil import (  # noqa: F401
    CS_INDEX,
    LROW1,
    NFXF,
    NGC2,
    NPTS,
    N_COILS_IN_GROUP,
    N_CS_FILAMENTS,
    N_CS_PF_COILS,
    N_PF_COILS,
    N_PF_GROUPS,
    N_PF_GROUPS_MAX,
    PLASMA_INDEX,
    PFCoilTopology,
    PFLocation,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
