"""Pure-functional port of `process/models/stellarator/neoclassics.py` (registry unit
#7).
"""

import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401

from functional_process.cottax.paths import (
    impurity_radiation,
    neoclassics,
    physics,
    stellarator,
    stellarator_config,
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
    reads={
        "temp_plasma_electron_on_axis_kev": physics,
        "temp_plasma_ion_on_axis_kev": physics,
        "alphat": physics,
        "nd_plasma_electron_on_axis": physics,
        "f_plasma_fuel_deuterium": physics,
        "nd_plasma_ions_on_axis": physics,
        "nd_plasma_alphas_thermal_vol_avg": physics,
        "alphan": physics,
        "rminor": physics,
    },
)


# cottax node: `calculate_effective_thermal_diffusivity`, unchanged, ports
# declared.
EffectiveThermalDiffusivity = neoclassics.function(
    calculate_effective_thermal_diffusivity,
    "chi_process_e",
    reads={
        "vol_plasma": physics,
        "f_st_rmajor": stellarator,
        "radius_plasma_core_norm": impurity_radiation,
        "rminor": physics,
        "stella_config_rminor_ref": stellarator_config,
        "a_plasma_surface": physics,
        "f_p_alpha_plasma_deposited": physics,
        "pden_alpha_total_mw": physics,
        "pden_plasma_core_rad_mw": physics,
        "nd_plasma_electron_on_axis": physics,
        "temp_plasma_electron_on_axis_kev": physics,
        "alphat": physics,
        "alphan": physics,
    },
)
