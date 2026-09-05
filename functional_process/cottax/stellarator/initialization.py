"""Pure-functional port of `st_init`'s only real computation (registry unit #6).

Audit record: `functional_process/_audit/units/models/stellarator/initialization.md`.
`st_init` is mostly a stellarator-mode device-preset table (16 unconditional literal
writes) plus one genuine, tiny pure function -- the pulse-duration sums -- ported here.
See the record's "proposed signature(s)" for why the 16 literals are not ported as a
node."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    times,
)
from functional_process.models.stellarator.initialization import (
    calculate_pulse_durations,
)


class PulseDurations(ExplicitFunction):
    """cottax node: `calculate_pulse_durations`, unchanged, ports declared."""

    t_plant_pulse_plasma_present = OutputInto(times)
    t_plant_pulse_no_burn = OutputInto(times)
    t_plant_pulse_total = OutputInto(times)

    def __call__(
        self,
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return calculate_pulse_durations(
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_dwell,
        )
