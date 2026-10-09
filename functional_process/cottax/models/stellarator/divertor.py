"""Pure-functional port of `st_div` (registry unit #4, `divertor.py`)."""

import jax.numpy as jnp  # noqa: F401
from cottax.assemblies.jax import V

from functional_process.cottax.paths import (
    divertor,
)
from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.models.stellarator.divertor import (
    calculate_divertor,
)

# cottax node: `calculate_divertor`, unchanged, ports declared.
Divertor = divertor.function(
    calculate_divertor,
    "pflux_div_heat_load_mw",
    "a_div_surface_total",
    V.fwbs.f_ster_div_single,
)
