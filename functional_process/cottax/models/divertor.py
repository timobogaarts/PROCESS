"""Pure-functional port of `process/models/divertor.py` (`Divertor`,
`.tokamak.divertor`) -- **not** `process/models/stellarator/divertor.py`, which is a
different model of a different device's divertor, already ported separately and half of
the stellarator graph's one non-`problem` cycle.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.cx import V

from functional_process.cottax.paths import divertor
from functional_process.models.divertor import (
    calculate_divertor_heat_flux_split,
    calculate_divertor_heat_load_wade,
    calculate_divertor_heat_load_wade_double_null,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401

# cottax node: `.tokamak.divertor`'s unconditional heat-flux split.
DivertorHeatFluxSplit = divertor.function(
    calculate_divertor_heat_flux_split,
    "deg_div_poloidal_plasma",
    V.fwbs.f_ster_div_single,
    V.fwbs.p_div_nuclear_heat_total_mw,
    V.fwbs.p_div_rad_total_mw,
)


# The DivertorHeatLoadWade family:
# The family that owns `.divertor.pflux_div_heat_load_mw` at `i_div_heat_load ==
# DivertorHeatLoadModel.WADE` (2): one occupant per `n_divertors` arm of
# `Divertor.divwade`'s own internal branch.


# cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 1`
# (single null) -- this run's derived value on `large_tokamak_eval.IN.DAT`, see module
# docstring.
DivertorHeatLoadWadeSingleNull = divertor.function(
    calculate_divertor_heat_load_wade,
    "pflux_div_heat_load_mw",
    reads={"b_plasma_poloidal_average": V.physics.b_plasma_surface_poloidal_average},
)


# cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 2`
# (double null) -- the value `spherical_tokamak_eval.IN.DAT` and
# `st_regression.IN.DAT` derive from `i_single_null = 0`.
DivertorHeatLoadWadeDoubleNull = divertor.function(
    calculate_divertor_heat_load_wade_double_null,
    "pflux_div_heat_load_mw",
    reads={"b_plasma_poloidal_average": V.physics.b_plasma_surface_poloidal_average},
)
