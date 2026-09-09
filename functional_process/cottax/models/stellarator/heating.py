"""Pure-functional port of `process/models/stellarator/heating.py` (registry unit #5)."""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    current_drive,
    heat_transport,
    physics,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.stellarator.heating import (
    calculate_beam_current,
    calculate_ecrh_heating,
    calculate_fusion_gain,
    calculate_injected_power_total,
    calculate_lowhyb_heating,
)


class EcrhHeating(WrapsFunction):
    """cottax node: `calculate_ecrh_heating`, unchanged, ports declared."""

    fn = calculate_ecrh_heating

    p_hcd_primary_extra_heat_mw = From(current_drive)
    eta_ecrh_injector_wall_plug = From(current_drive)

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)


class LowhybHeating(WrapsFunction):
    """cottax node: `calculate_lowhyb_heating`, unchanged, ports declared."""

    fn = calculate_lowhyb_heating

    p_hcd_primary_extra_heat_mw = From(current_drive)
    eta_lowhyb_injector_wall_plug = From(current_drive)

    p_hcd_lowhyb_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)


class InjectedPowerTotal(WrapsFunction):
    """cottax node: `calculate_injected_power_total`, unchanged, ports declared."""

    fn = calculate_injected_power_total

    p_hcd_injected_electrons_mw = From(current_drive)
    p_hcd_injected_ions_mw = From(current_drive)

    p_hcd_injected_total_mw = OutputInto(current_drive)


class BeamCurrent(WrapsFunction):
    """cottax node: `calculate_beam_current`, unchanged, ports declared."""

    fn = calculate_beam_current

    p_hcd_beam_injected_total_mw = From(current_drive)
    e_beam_kev = From(current_drive)

    c_beam_total = OutputInto(current_drive)


class FusionGain(WrapsFunction):
    """cottax node: `calculate_fusion_gain`, unchanged, ports declared."""

    fn = calculate_fusion_gain

    p_fusion_total_mw = From(physics)
    p_hcd_injected_total_mw = From(current_drive)
    p_beam_orbit_loss_mw = From(current_drive)
    p_plasma_ohmic_mw = From(physics)

    big_q_plasma = OutputInto(current_drive)
