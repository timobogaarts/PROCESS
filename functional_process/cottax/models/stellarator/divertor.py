"""Pure-functional port of `st_div` (registry unit #4, `divertor.py`)."""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    divertor,
    first_wall,
    physics,
    stellarator,
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
    reads={
        "flpitch": stellarator,
        "rmajor": physics,
        "p_plasma_separatrix_mw": physics,
        "m_fuel_amu": physics,
        "bmn": stellarator,
        "shear": stellarator,
        "n_res": stellarator,
        "f_w": stellarator,
        "m_res": stellarator,
        "fdivwet": stellarator,
        "f_asym": stellarator,
        "a_fw_total": first_wall,
    },
)
