"""Pure-functional port of `process/models/stellarator/build.py`'s `st_build`.

Unit #2.
"""

from cottax.interfaces import V

from functional_process.cottax.paths import (
    build,
    first_wall,
)
from functional_process.models.stellarator.build import (
    calculate_a_fw_total_no_powerflow,
    calculate_a_fw_total_with_powerflow,
    calculate_blktmodel_blanket_thickness,
    calculate_build,
)

# cottax node: `calculate_blktmodel_blanket_thickness`, ports declared.
BlktmodelBlanketThickness = build.function(
    calculate_blktmodel_blanket_thickness,
    "dr_blkt_inboard",
    "dr_blkt_outboard",
    "dz_shld_upper",
)


# cottax node: `calculate_build`, ports declared.
Build = build.function(
    calculate_build,
    "dz_blkt_upper",
    "dr_fw_inboard",
    "dr_fw_outboard",
    "dr_bore",
    "rbld",
    "required_radial_space",
    "available_radial_space",
    "r_shld_inboard_inner",
    "r_shld_outboard_outer",
    "dr_tf_outboard",
    "dr_shld_vv_gap_outboard",
    "r_tf_outboard_mid",
    "rspo",
    V.first_wall.a_fw_total_unadjusted,
)


# cottax node: `calculate_a_fw_total_no_powerflow`. Instantiate iff `ipowerflow == 0`.
AFwTotalNoPowerflow = first_wall.function(
    calculate_a_fw_total_no_powerflow,
    "a_fw_total",
)


# cottax node: `calculate_a_fw_total_with_powerflow`. Instantiate iff `ipowerflow != 0`.
AFwTotalWithPowerflow = first_wall.function(
    calculate_a_fw_total_with_powerflow,
    "a_fw_total",
)
