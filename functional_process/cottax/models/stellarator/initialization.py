"""Pure-functional port of `st_init`'s only real computation (registry unit #6)."""

from functional_process.cottax.paths import (
    times,
)
from functional_process.models.stellarator.initialization import (
    calculate_pulse_durations,
)

# cottax node: `calculate_pulse_durations`, unchanged, ports declared.
PulseDurations = times.function(
    calculate_pulse_durations,
    "t_plant_pulse_plasma_present",
    "t_plant_pulse_no_burn",
    "t_plant_pulse_total",
)
