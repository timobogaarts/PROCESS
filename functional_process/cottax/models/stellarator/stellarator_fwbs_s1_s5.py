"""Pure-functional port of `st_fwbs`'s S1 and S5 sub-computations (see
`stellarator_E_fwbs_synthesis.md`).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.assemblies.jax import V

from functional_process.cottax.paths import (
    fwbs,
    stellarator_config,
)
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_cryostat_and_vv_geometry,
    calculate_fw_blanket_shield_geometry,  # noqa: F401 -- re-exported for its test
    calculate_fw_blanket_shield_geometry_no_powerflow,
    calculate_fw_blanket_shield_geometry_with_powerflow,
)

# cottax node: `calculate_fw_blanket_shield_geometry` (S1), one occupant per
# `.heat_transport.ipowerflow`.
_FW_BLANKET_SHIELD_GEOMETRY_READS = {
    "stella_config_neutron_peakfactor": stellarator_config.place(
        "stella_config_neutron_peakfactor"
    ),
}

FwBlanketShieldGeometryNoPowerflow = fwbs.function(
    calculate_fw_blanket_shield_geometry_no_powerflow,
    "life_fw_fpy",
    V.first_wall.a_fw_inboard,
    V.first_wall.a_fw_outboard,
    V.build.a_blkt_total_surface,
    V.build.a_blkt_inboard_surface,
    V.build.a_blkt_outboard_surface,
    "vol_blkt_inboard",
    "vol_blkt_outboard",
    "vol_blkt_total",
    V.build.a_shld_total_surface,
    V.build.a_shld_inboard_surface,
    V.build.a_shld_outboard_surface,
    "vol_shld_total",
    "pnucloss",
    "wallpf",
    reads=_FW_BLANKET_SHIELD_GEOMETRY_READS,
)

FwBlanketShieldGeometryWithPowerflow = fwbs.function(
    calculate_fw_blanket_shield_geometry_with_powerflow,
    "life_fw_fpy",
    V.first_wall.a_fw_inboard,
    V.first_wall.a_fw_outboard,
    V.build.a_blkt_total_surface,
    V.build.a_blkt_inboard_surface,
    V.build.a_blkt_outboard_surface,
    "vol_blkt_inboard",
    "vol_blkt_outboard",
    "vol_blkt_total",
    V.build.a_shld_total_surface,
    V.build.a_shld_inboard_surface,
    V.build.a_shld_outboard_surface,
    "vol_shld_total",
    "pnucloss",
    "wallpf",
    reads=_FW_BLANKET_SHIELD_GEOMETRY_READS,
)


# cottax node: `calculate_cryostat_and_vv_geometry` (S5), unchanged.
CryostatAndVvGeometry = fwbs.function(
    calculate_cryostat_and_vv_geometry,
    "r_cryostat_inboard",
    "vol_cryostat",
    "vol_vv",
    "m_vv",
    "dewmkg",
)
