"""Pure-functional port of `process/models/stellarator/neoclassics.py` (registry unit
#7).
"""

import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401

from functional_process.cottax.paths import (
    neoclassics,
)
from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.models.stellarator.neoclassics import (
    KEV,  # noqa: F401
    NO_ROOTS,  # noqa: F401
    ROOTS,  # noqa: F401
    WEIGHTS,  # noqa: F401
    calculate_collision_frequency,  # noqa: F401
    calculate_drift_velocity,  # noqa: F401
    calculate_effective_thermal_diffusivity,
    calculate_gamma_flux,  # noqa: F401
    calculate_integrated_radial_transport_coefficient,  # noqa: F401
    calculate_kt,  # noqa: F401
    calculate_monoenergetic_transport_coefficient,  # noqa: F401
    calculate_normalized_collision_frequency,  # noqa: F401
    calculate_normalized_collision_frequency_from_temperature,  # noqa: F401
    calculate_plateau_transport_coefficient,  # noqa: F401
    calculate_profile_values,
    calculate_q_flux,  # noqa: F401
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)

# cottax node: `calculate_profile_values`, unchanged, ports declared.
# `.neoclassics.rho` is the normalised radius the neoclassical profiles are evaluated at
# -- PROCESS's own literal (0.6) at its one call site, read as a setting.
ProfileValues = neoclassics.function(
    calculate_profile_values,
    "densities",
    "temperatures",
    "dr_densities",
    "dr_temperatures",
)


# cottax node: `calculate_effective_thermal_diffusivity`, unchanged, ports
# declared.
EffectiveThermalDiffusivity = neoclassics.function(
    calculate_effective_thermal_diffusivity,
    "chi_process_e",
)
