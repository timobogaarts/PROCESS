"""Pure-functional port of `process/models/stellarator/coils/quench.py` (registry #14).

Audit record: `functional_process/_audit/units/models/stellarator/coils/quench.md`. The
source's `calculate_quench_protection` orchestrates one sub-call
(`calculate_vv_max_force_density_from_W7X_scaling`) plus two already-pure functions
(`max_dump_voltage`, `calculate_quench_protection_current_density`), chained through
straight-line, unconditional `data` reads/writes -- `local-intermediate` throughout,
same treatment as `mass.py`'s 8-step chain."""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.paths import (
    build,
    physics,
    rebco,
    superconducting_tfcoil,
    tfcoil,
)
from functional_process.stellarator.coils.quench import (
    calculate_quench_protection,
    calculate_quench_protection_current_density,  # noqa: F401
    calculate_vv_max_force_density_from_w7x_scaling,  # noqa: F401
    max_dump_voltage,  # noqa: F401
)


class QuenchProtection(ExplicitFunction):
    """cottax node: `calculate_quench_protection`, unchanged, ports declared.

    `f_vv_actual` mints an invented `.superconducting_tfcoil.f_vv_actual` -- the source
    never stores it either, it is returned by `calculate_quench_protection` only to be
    forwarded straight into `coils/output.py`'s `write(...)` for the printout (confirmed
    by grep of `coils/calculate.py`). `vv_stress_quench` shares its area since the
    source assigns it there (`data.superconducting_tfcoil.vv_stress_quench`).
    """

    f_vv_actual = OutputInto(superconducting_tfcoil)
    vv_stress_quench = OutputInto(superconducting_tfcoil)
    j_tf_wp_quench_heat_max = OutputInto(tfcoil)
    coppera_m2 = OutputInto(rebco)
    v_tf_coil_dump_quench_kv = OutputInto(tfcoil)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_blkt_gap=From(build),
        dr_shld_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
        b_plasma_toroidal_on_axis=From(physics),
        c_tf_total=From(tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        t_tf_quench_detection=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        tftmp=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        e_tf_magnetic_stored_total_gj=From(tfcoil),
        n_tf_coils=From(tfcoil),
        c_tf_turn=From(tfcoil),
    ):
        return calculate_quench_protection(
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            dr_fw_inboard,
            dr_blkt_inboard,
            dr_shld_blkt_gap,
            dr_shld_inboard,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dr_blkt_outboard,
            dr_shld_outboard,
            b_plasma_toroidal_on_axis,
            c_tf_total,
            t_tf_superconductor_quench,
            dr_vv_inboard,
            dr_vv_outboard,
            t_tf_quench_detection,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            tftmp,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            a_tf_wp_conductor,
            e_tf_magnetic_stored_total_gj,
            n_tf_coils,
            c_tf_turn,
        )


# `max_dump_voltage`, `calculate_quench_protection_current_density`, and
# `calculate_vv_max_force_density_from_w7x_scaling` get no separate node: in their one
# real call site (`calculate_quench_protection` above) several of their arguments are
# derived intermediates (`tf_energy_stored`, `f_cond`, `rad_vv`), not raw `data` fields,
# so a standalone node for any of them would need invented `VarPath`s for values that
# only exist inside this composed chain -- same treatment as `structure.md`
# gave `calculate_structure_masses`'s internal `intercoil_surface`.
