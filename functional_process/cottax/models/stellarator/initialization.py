"""Pure-functional port of `st_init`'s only real computation (registry unit #6)."""

from cottax.interfaces.pytree_namespace_module import (
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    times,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.stellarator.initialization import (
    calculate_pulse_durations,
)


class PulseDurations(WrapsFunction):
    """cottax node: `calculate_pulse_durations`, unchanged, ports declared."""

    fn = calculate_pulse_durations

    t_plant_pulse_coil_precharge = From(times)
    t_plant_pulse_plasma_current_ramp_up = From(times)
    t_plant_pulse_burn = From(times)
    t_plant_pulse_plasma_current_ramp_down = From(times)
    t_plant_pulse_fusion_ramp = From(times)
    t_plant_pulse_dwell = From(times)

    t_plant_pulse_plasma_present = OutputInto(times)
    t_plant_pulse_no_burn = OutputInto(times)
    t_plant_pulse_total = OutputInto(times)
