"""Pure-functional port of S2 (`blanket_shield_tf_nuclear_power`), the `blktmodel` x
`ipowerflow` dispatch inside `Stellarator.st_fwbs` (registry unit #1's
`stellarator.py`).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    fwbs,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    calculate_detailed_powerflow_blanket_shield_power,
    calculate_detailed_powerflow_blanket_shield_power_user_input_pumping,
    calculate_exponential_attenuation_blanket_shield_power,
)

# cottax node: `calculate_exponential_attenuation_blanket_shield_power`.
ExponentialAttenuationBlanketShieldPower = fwbs.function(
    calculate_exponential_attenuation_blanket_shield_power,
    "p_blkt_multiplication_mw",
    "p_blkt_nuclear_heat_total_mw",
    "p_shld_nuclear_heat_mw",
)


# cottax node: `calculate_detailed_powerflow_blanket_shield_power`.
DetailedPowerflowBlanketShieldPower = fwbs.function(
    calculate_detailed_powerflow_blanket_shield_power,
    "p_div_nuclear_heat_total_mw",
    "p_fw_hcd_nuclear_heat_mw",
    "p_fw_hcd_rad_total_mw",
    "pradloss",
    "p_fw_rad_total_mw",
    "f_a_fw_coolant_inboard",
    "f_a_fw_coolant_outboard",
    "p_fw_nuclear_heat_total_mw",
    "p_blkt_multiplication_mw",
    "p_blkt_nuclear_heat_total_mw",
    V.heat_transport.p_fw_coolant_pump_mw,
    V.heat_transport.p_blkt_coolant_pump_mw,
    "p_shld_nuclear_heat_mw",
    V.heat_transport.p_shld_coolant_pump_mw,
    V.heat_transport.p_div_coolant_pump_mw,
    "p_tf_nuclear_heat_mw",
)


# cottax node:
# `calculate_detailed_powerflow_blanket_shield_power_user_input_pumping`.
DetailedPowerflowBlanketShieldPowerUserInputPumping = fwbs.function(
    calculate_detailed_powerflow_blanket_shield_power_user_input_pumping,
    "p_div_nuclear_heat_total_mw",
    "p_fw_hcd_nuclear_heat_mw",
    "p_fw_hcd_rad_total_mw",
    "pradloss",
    "p_fw_rad_total_mw",
    "f_a_fw_coolant_inboard",
    "f_a_fw_coolant_outboard",
    "p_fw_nuclear_heat_total_mw",
    "p_blkt_multiplication_mw",
    "p_blkt_nuclear_heat_total_mw",
    "p_shld_nuclear_heat_mw",
    "p_tf_nuclear_heat_mw",
)
