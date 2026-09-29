"""Pure-functional port of `process/models/cs_fatigue.py` (partial -- see "not ported")."""

import jax.numpy as jnp
from jax import lax  # noqa: F401

from functional_process.cottax.paths import cs_fatigue
from functional_process.models.cs_fatigue import (
    calculate_n_cycle,
    surface_stress_intensity_factor,  # noqa: F401
)
from functional_process.models.safe_math import safe_pow, safe_sqrt  # noqa: F401


def calculate_cs_fatigue_n_cycle_gated(
    f_c_plasma_inductive,
    stress_hoop_cs_inner,
    residual_sig_hoop,
    t_crack_vertical,
    dz_cs_turn_conduit,
    dr_cs_turn_conduit,
    paris_coefficient,
    paris_power_law,
    walker_coefficient,
    sf_vertical_crack,
    sf_radial_crack,
    fracture_toughness,
    sf_fast_fracture,
):
    """`CsFatigue`'s own `f_c_plasma_inductive > 0.0` gate around `calculate_n_cycle`
    (see that class's docstring for why the guard belongs to the binding), moved out of
    the declaration and into a named function -- `_audit/formulas_split.md` step 1.
    """
    return jnp.where(
        f_c_plasma_inductive > 0.0,
        calculate_n_cycle(
            stress_hoop_cs_inner,
            residual_sig_hoop,
            t_crack_vertical,
            dz_cs_turn_conduit,
            dr_cs_turn_conduit,
            paris_coefficient,
            paris_power_law,
            walker_coefficient,
            sf_vertical_crack,
            sf_radial_crack,
            fracture_toughness,
            sf_fast_fracture,
        ),
        0.0,
    )


# cottax node: `.tokamak.cs_fatigue`.
CsFatigue = cs_fatigue.function(
    calculate_cs_fatigue_n_cycle_gated,
    "n_cycle",
)
