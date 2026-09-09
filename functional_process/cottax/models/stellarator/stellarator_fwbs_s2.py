"""Pure-functional port of S2 (`blanket_shield_tf_nuclear_power`), the `blktmodel` x
`ipowerflow` dispatch inside `Stellarator.st_fwbs` (registry unit #1's
`stellarator.py`).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    current_drive,
    first_wall,
    fwbs,
    heat_transport,
    physics,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    calculate_detailed_powerflow_blanket_shield_power,
    calculate_detailed_powerflow_blanket_shield_power_user_input_pumping,
    calculate_exponential_attenuation_blanket_shield_power,
)


class ExponentialAttenuationBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_exponential_attenuation_blanket_shield_power`."""

    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        pnucloss=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        f_a_blkt_cooling_channels=From(fwbs),
        fblli2o=From(fwbs),
        fblbe=From(fwbs),
        dr_blkt_outboard=From(build),
    ):
        return calculate_exponential_attenuation_blanket_shield_power(
            p_neutron_total_mw,
            pnucloss,
            f_p_blkt_multiplication,
            f_a_blkt_cooling_channels,
            fblli2o,
            fblbe,
            dr_blkt_outboard,
        )


class DetailedPowerflowBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_detailed_powerflow_blanket_shield_power`."""

    p_div_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_hcd_nuclear_heat_mw = OutputInto(fwbs)
    p_fw_hcd_rad_total_mw = OutputInto(fwbs)
    pradloss = OutputInto(fwbs)
    p_fw_rad_total_mw = OutputInto(fwbs)
    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_coolant_pump_mw = OutputInto(heat_transport)
    p_blkt_coolant_pump_mw = OutputInto(heat_transport)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_shld_coolant_pump_mw = OutputInto(heat_transport)
    p_div_coolant_pump_mw = OutputInto(heat_transport)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        pnucloss=From(fwbs),
        a_fw_inboard=From(first_wall),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        fhole=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        radius_fw_channel=From(fwbs),
        declfw=From(fwbs),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        declblkt=From(fwbs),
        f_p_fw_coolant_pump_total_heat=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        f_p_blkt_coolant_pump_total_heat=From(heat_transport),
        f_p_blkt_multiplication=From(fwbs),
        declshld=From(fwbs),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
        f_p_shld_coolant_pump_total_heat=From(heat_transport),
        p_plasma_separatrix_mw=From(physics),
        f_p_div_coolant_pump_total_heat=From(heat_transport),
    ):
        return calculate_detailed_powerflow_blanket_shield_power(
            p_neutron_total_mw,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            pnucloss,
            a_fw_inboard,
            a_fw_outboard,
            a_fw_total,
            p_plasma_rad_mw,
            fhole,
            dr_fw_inboard,
            dr_fw_outboard,
            radius_fw_channel,
            declfw,
            dr_blkt_inboard,
            dr_blkt_outboard,
            declblkt,
            f_p_fw_coolant_pump_total_heat,
            p_beam_orbit_loss_mw,
            f_p_blkt_coolant_pump_total_heat,
            f_p_blkt_multiplication,
            declshld,
            dr_shld_inboard,
            dr_shld_outboard,
            f_p_shld_coolant_pump_total_heat,
            p_plasma_separatrix_mw,
            f_p_div_coolant_pump_total_heat,
        )


class DetailedPowerflowBlanketShieldPowerUserInputPumping(ExplicitFunction):
    """cottax node:
    `calculate_detailed_powerflow_blanket_shield_power_user_input_pumping`.
    """

    p_div_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_hcd_nuclear_heat_mw = OutputInto(fwbs)
    p_fw_hcd_rad_total_mw = OutputInto(fwbs)
    pradloss = OutputInto(fwbs)
    p_fw_rad_total_mw = OutputInto(fwbs)
    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        pnucloss=From(fwbs),
        a_fw_inboard=From(first_wall),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        fhole=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        radius_fw_channel=From(fwbs),
        declfw=From(fwbs),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        declblkt=From(fwbs),
        f_p_blkt_coolant_pump_total_heat=From(heat_transport),
        f_p_blkt_multiplication=From(fwbs),
        declshld=From(fwbs),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
    ):
        return calculate_detailed_powerflow_blanket_shield_power_user_input_pumping(
            p_neutron_total_mw,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            pnucloss,
            a_fw_inboard,
            a_fw_outboard,
            a_fw_total,
            p_plasma_rad_mw,
            fhole,
            dr_fw_inboard,
            dr_fw_outboard,
            radius_fw_channel,
            declfw,
            dr_blkt_inboard,
            dr_blkt_outboard,
            declblkt,
            f_p_blkt_coolant_pump_total_heat,
            f_p_blkt_multiplication,
            declshld,
            dr_shld_inboard,
            dr_shld_outboard,
        )
