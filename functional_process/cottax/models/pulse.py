"""Pure-functional port of `process/models/pulse.py` (`Pulse`, `.tokamak.pulse`)."""

from functional_process.cottax.paths import times
from functional_process.models.pulse import calculate_burn_time

# cottax node: `calculate_burn_time`, ports declared.
PulseBurnTime = times.function(
    calculate_burn_time,
    "t_plant_pulse_burn",
)
