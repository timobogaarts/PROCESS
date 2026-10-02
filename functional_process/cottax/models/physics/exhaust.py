"""Pure-functional port of `process/models/physics/exhaust.py`."""

from cottax.cx import V

from functional_process.cottax.paths import physics
from functional_process.models.physics.exhaust import (
    calculate_eu_demo_re_attachment_metric,
    calculate_psep_over_r_metric,
    calculate_radiation_fraction,
)

# cottax node: `calculate_radiation_fraction`, ports declared.
RadiationFraction = physics.function(
    calculate_radiation_fraction,
    "f_p_plasma_separatrix_rad",
    reads={
        "p_plasma_heating_mw": V.physics.p_plasma_heating_total_mw,
    },
)


# cottax node: `calculate_eu_demo_re_attachment_metric`, ports declared.
EuDemoReAttachmentMetric = physics.function(
    calculate_eu_demo_re_attachment_metric,
    "p_div_bt_q_aspect_rmajor_mw",
    reads={
        "p_plasma_separatrix_mw": V.physics.p_plasma_separatrix_mw_raw,
    },
)


# cottax node: `calculate_psep_over_r_metric`, ports declared.
PsepOverRMetric = physics.function(
    calculate_psep_over_r_metric,
    "p_plasma_separatrix_rmajor_mw",
    reads={
        "p_plasma_separatrix_mw": V.physics.p_plasma_separatrix_mw_raw,
    },
)
