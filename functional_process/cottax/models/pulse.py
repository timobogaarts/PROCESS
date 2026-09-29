"""Pure-functional port of `process/models/pulse.py` (`Pulse`, `.tokamak.pulse`)."""

from functional_process.cottax.paths import pf_coil, physics, times
from functional_process.models.pulse import calculate_burn_time

# cottax node: `calculate_burn_time`, ports declared.
PulseBurnTime = times.function(
    calculate_burn_time,
    "t_plant_pulse_burn",
    reads={
        "vs_cs_pf_total_burn": pf_coil,
        "v_plasma_loop_burn": physics,
    },
)
