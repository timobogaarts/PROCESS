"""Pure-functional port of `process/models/stellarator/heating.py` (registry unit #5)."""

import jax.numpy as jnp  # noqa: F401
from cottax.relational_pytree import V

from functional_process.cottax.paths import (
    current_drive,
)
from functional_process.models.stellarator.heating import (
    calculate_beam_current,
    calculate_ecrh_heating,
    calculate_fusion_gain,
    calculate_injected_power_total,
    calculate_lowhyb_heating,
)

# cottax node: `calculate_ecrh_heating`, unchanged, ports declared.
EcrhHeating = current_drive.function(
    calculate_ecrh_heating,
    "p_hcd_ecrh_injected_total_mw",
    "p_hcd_injected_ions_mw",
    "p_hcd_injected_electrons_mw",
    "eta_hcd_primary_injector_wall_plug",
    V.heat_transport.p_hcd_electric_total_mw,
)


# cottax node: `calculate_lowhyb_heating`, unchanged, ports declared.
LowhybHeating = current_drive.function(
    calculate_lowhyb_heating,
    "p_hcd_lowhyb_injected_total_mw",
    "p_hcd_injected_ions_mw",
    "p_hcd_injected_electrons_mw",
    "eta_hcd_primary_injector_wall_plug",
    V.heat_transport.p_hcd_electric_total_mw,
)


# cottax node: `calculate_injected_power_total`, unchanged, ports declared.
InjectedPowerTotal = current_drive.function(
    calculate_injected_power_total, "p_hcd_injected_total_mw"
)


# cottax node: `calculate_beam_current`, unchanged, ports declared.
BeamCurrent = current_drive.function(calculate_beam_current, "c_beam_total")


# cottax node: `calculate_fusion_gain`, unchanged, ports declared.
FusionGain = current_drive.function(
    calculate_fusion_gain,
    "big_q_plasma",
)
