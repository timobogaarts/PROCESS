"""Pure-functional port of `process/models/cs_fatigue.py` (partial -- see "not ported").
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto
from jax import lax  # noqa: F401

from functional_process.models.cs_fatigue import (
    calculate_n_cycle,
    surface_stress_intensity_factor,  # noqa: F401
)
from functional_process.models.safe_math import safe_pow, safe_sqrt  # noqa: F401
from functional_process.cottax.paths import cs_fatigue, pf_coil, physics


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


class CsFatigue(ExplicitFunction):
    """cottax node: `.tokamak.cs_fatigue`."""

    n_cycle = OutputInto(cs_fatigue)

    def __call__(
        self,
        stress_hoop_cs_inner=From(pf_coil),
        residual_sig_hoop=From(cs_fatigue),
        t_crack_vertical=From(cs_fatigue),
        dz_cs_turn_conduit=From(cs_fatigue),
        dr_cs_turn_conduit=From(cs_fatigue),
        paris_coefficient=From(cs_fatigue),
        paris_power_law=From(cs_fatigue),
        walker_coefficient=From(cs_fatigue),
        sf_vertical_crack=From(cs_fatigue),
        sf_radial_crack=From(cs_fatigue),
        fracture_toughness=From(cs_fatigue),
        sf_fast_fracture=From(cs_fatigue),
        f_c_plasma_inductive=From(physics),
    ):
        return calculate_cs_fatigue_n_cycle_gated(
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
        )
