"""Pure-functional port of `process/models/stellarator/coils/quench.py`.

Registry #14.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    build,
    physics,
    superconducting_tfcoil,
    tfcoil,
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
    reads={
        "rmajor": physics,
        "rminor": physics,
        "dr_fw_plasma_gap_inboard": build,
        "dr_fw_inboard": build,
        "dr_blkt_inboard": build,
        "dr_shld_blkt_gap": build,
        "dr_shld_inboard": build,
        "dr_fw_plasma_gap_outboard": build,
        "dr_fw_outboard": build,
        "dr_blkt_outboard": build,
        "dr_shld_outboard": build,
        "b_plasma_toroidal_on_axis": physics,
        "c_tf_total": tfcoil,
        "t_tf_superconductor_quench": tfcoil,
        "dr_vv_inboard": build,
        "dr_vv_outboard": build,
        "t_tf_quench_detection": tfcoil,
        "f_a_tf_turn_cable_copper": tfcoil,
        "f_a_tf_turn_cable_space_extra_void": tfcoil,
        "tftmp": tfcoil,
        "a_tf_turn_cable_space_no_void": tfcoil,
        "dx_tf_turn_general": tfcoil,
        "a_tf_wp_conductor": tfcoil,
        "e_tf_magnetic_stored_total_gj": tfcoil,
        "n_tf_coils": tfcoil,
        "c_tf_turn": tfcoil,
    },
)


# `max_dump_voltage`, `calculate_quench_protection_current_density`, and
# `calculate_vv_max_force_density_from_w7x_scaling` get no separate node: in their one
# real call site (`calculate_quench_protection` above) several of their arguments are
# derived intermediates (`tf_energy_stored`, `f_cond`, `rad_vv`), not raw `data` fields,
# so a standalone node for any of them would need invented `VarPath`s for values that
# only exist inside this composed chain -- same treatment as `structure.md`
# gave `calculate_structure_masses`'s internal `intercoil_surface`.
