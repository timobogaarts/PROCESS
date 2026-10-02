"""Pure-functional port of `process/models/stellarator/density_limits.py`."""

import jax.numpy as jnp  # noqa: F401
from cottax.cx import V

from functional_process.cottax.paths import (
    physics,
    stellarator,
)
from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.models.stellarator.density_limits import (
    calculate_ecrh_density_limit,  # noqa: F401
    calculate_ecrh_density_limit_parabolic,
    calculate_sudo_density_limit,
)

# cottax node: `calculate_sudo_density_limit`, unchanged, ports declared.
SudoDensityLimit = physics.function(
    calculate_sudo_density_limit, "nd_plasma_electrons_max"
)


# cottax node: `calculate_ecrh_density_limit_parabolic`, ports declared.
EcrhDensityLimit = stellarator.function(
    calculate_ecrh_density_limit_parabolic,
    "dlimit_ecrh",
    "bt_max_ecrh",
    reads={"gyro_frequency_max": V.stellarator.max_gyrotron_frequency},
)
