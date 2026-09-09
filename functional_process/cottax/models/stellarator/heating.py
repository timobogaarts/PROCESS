"""Pure-functional port of `process/models/stellarator/heating.py` (registry unit #5).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    current_drive,
    heat_transport,
    physics,
)
from functional_process.models.stellarator.heating import (
    calculate_beam_current,
    calculate_ecrh_heating,
    calculate_fusion_gain,
    calculate_injected_power_total,
    calculate_lowhyb_heating,
)


class EcrhHeating(ExplicitFunction):
    """cottax node: `calculate_ecrh_heating`, unchanged, ports declared."""

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return calculate_ecrh_heating(
            p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug
        )


class LowhybHeating(ExplicitFunction):
    """cottax node: `calculate_lowhyb_heating`, unchanged, ports declared."""

    p_hcd_lowhyb_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_lowhyb_injector_wall_plug=From(current_drive),
    ):
        return calculate_lowhyb_heating(
            p_hcd_primary_extra_heat_mw, eta_lowhyb_injector_wall_plug
        )


class InjectedPowerTotal(ExplicitFunction):
    """cottax node: `calculate_injected_power_total`, unchanged, ports declared."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_injected_electrons_mw=From(current_drive),
        p_hcd_injected_ions_mw=From(current_drive),
    ):
        return calculate_injected_power_total(
            p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw
        )


class BeamCurrent(ExplicitFunction):
    """cottax node: `calculate_beam_current`, unchanged, ports declared."""

    c_beam_total = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_beam_injected_total_mw=From(current_drive),
        e_beam_kev=From(current_drive),
    ):
        return calculate_beam_current(p_hcd_beam_injected_total_mw, e_beam_kev)


class FusionGain(ExplicitFunction):
    """cottax node: `calculate_fusion_gain`, unchanged, ports declared."""

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return calculate_fusion_gain(
            p_fusion_total_mw,
            p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw,
        )
