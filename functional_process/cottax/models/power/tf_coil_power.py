"""Pure-functional port of the TF coil power conversion sub-unit of
`process/models/power.py` (registry unit #14, chunk A).
"""

import jax.numpy as jnp
import numpy as np
from cottax.interfaces import V

from functional_process.cottax.paths import buildings, heat_transport, physics, tfcoil
from functional_process.models.power.tf_coil_power import (
    calculate_tf_power_resistive,
    calculate_tf_power_superconducting,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.vocabulary import constants

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `jnp`/`np`/
# `safe_pow`/`safe_sqrt`/`constants`, which are unused now that their real uses left with
# the functions (see `power/electric_production.py`'s commit for why a partial list is
# the wrong move; this file's own first pass over-trimmed the import block and dropped
# these five without checking, a gap `_audit/formulas_split.md`'s name-preservation gate
# should have caught and did not, until a later cluster-wide re-check with clean caches).
__all__ = [
    "From",
    "OutputInto",
    "TfPowerResistive",
    "TfPowerSuperconducting",
    "buildings",
    "calculate_tf_power_resistive",
    "calculate_tf_power_superconducting",
    "constants",
    "heat_transport",
    "jnp",
    "np",
    "physics",
    "safe_pow",
    "safe_sqrt",
    "tfcoil",
]


# cottax node: `calculate_tf_power_resistive`.
TfPowerResistive = tfcoil.function(
    calculate_tf_power_resistive,
    "m_tf_bus",
    "vtfkv",
    "p_cp_resistive_mw",
    "p_tf_leg_resistive_mw",
    "p_tf_joints_resistive_mw",
    "tfcmw",
    V.heat_transport.p_tf_electric_supplies_mw,
    reads={
        "etatf": heat_transport,
    },
)


# cottax node: `calculate_tf_power_superconducting`.
TfPowerSuperconducting = tfcoil.function(
    calculate_tf_power_superconducting,
    "tfckw",
    "len_tf_bus",
    "drarea",
    V.buildings.tfcbv,
    V.heat_transport.p_tf_electric_supplies_mw,
    reads={
        "rmajor": physics,
        "etatf": heat_transport,
    },
)
