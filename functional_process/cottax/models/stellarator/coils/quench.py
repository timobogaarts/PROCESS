"""Pure-functional port of `process/models/stellarator/coils/quench.py`.

Registry #14.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.assemblies.jax import V

from functional_process.cottax.paths import (
    superconducting_tfcoil,
)
from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.models.stellarator.coils.quench import (
    calculate_quench_protection,
    calculate_quench_protection_current_density,  # noqa: F401
    calculate_vv_max_force_density_from_w7x_scaling,  # noqa: F401
    max_dump_voltage,  # noqa: F401
)

# cottax node: `calculate_quench_protection`, unchanged, ports declared.
QuenchProtection = superconducting_tfcoil.function(
    calculate_quench_protection,
    "f_vv_actual",
    "vv_stress_quench",
    V.tfcoil.j_tf_wp_quench_heat_max,
    V.rebco.coppera_m2,
    V.tfcoil.v_tf_coil_dump_quench_kv,
)


# `max_dump_voltage`, `calculate_quench_protection_current_density`, and
# `calculate_vv_max_force_density_from_w7x_scaling` get no separate node: in their one
# real call site (`calculate_quench_protection` above) several of their arguments are
# derived intermediates (`tf_energy_stored`, `f_cond`, `rad_vv`), not raw `data` fields,
# so a standalone node for any of them would need invented `VarPath`s for values that
# only exist inside this composed chain -- same treatment as `structure.md`
# gave `calculate_structure_masses`'s internal `intercoil_surface`.
