"""Pure-functional port of `process/models/pulse.py` (`Pulse`, `.tokamak.pulse`)."""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import pf_coil, physics, times
from functional_process.models.pulse import calculate_burn_time


class PulseBurnTime(ExplicitFunction):
    """cottax node: `calculate_burn_time`, ports declared."""

    t_plant_pulse_burn = OutputInto(times)

    def __call__(
        self,
        vs_cs_pf_total_burn=From(pf_coil),
        v_plasma_loop_burn=From(physics),
        t_plant_pulse_fusion_ramp=From(times),
    ):
        return calculate_burn_time(
            vs_cs_pf_total_burn, v_plasma_loop_burn, t_plant_pulse_fusion_ramp
        )
