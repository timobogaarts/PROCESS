"""Pure-functional port of `Stellarator.sc_tf_coil_nuclear_heating_iter90` (chunk 1F)."""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import (
    build,
    costs,
    fwbs,
    physics,
    tfcoil,
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
    reads={
        "dr_shld_inboard": build,
        "dr_fw_inboard": build,
        "dr_blkt_inboard": build,
        "dr_shld_outboard": build,
        "dr_fw_outboard": build,
        "dr_blkt_outboard": build,
        "dr_tf_wp_with_insulation": tfcoil,
        "dx_tf_wp_insulation": tfcoil,
        "pflux_fw_neutron_mw": physics,
        "tfsai": tfcoil,
        "tfsao": tfcoil,
        "dr_tf_plasma_case": tfcoil,
        "f_t_plant_available": costs,
        "life_plant": costs,
    },
)
