"""Pure physics functions extracted from
`functional_process.models.stellarator.coils.quench`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_sqrt

_FORCE_DENSITY_REF_MN_PER_M3 = 2.54


_B_REF_T = 3.0


_I_TOTAL_REF_A = 1.3e6 * 50


_RMINOR_REF_M = 0.92


_TAU_REF_S = 3.0


_RMAJOR_REF_M = 5.2


_DR_VV_REF_M = 14e-3


_TEMP_K = (4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124)


_Q_CU_ARRAY_SA2M4 = (
    1.08514e17,
    1.12043e17,
    1.12406e17,
    1.05940e17,
    9.49741e16,
    8.43757e16,
    7.56346e16,
    6.85924e16,
    6.28575e16,
    5.81004e16,
    5.40838e16,
    5.06414e16,
    4.76531e16,
)


_Q_HE_ARRAY_SA2M4 = (
    3.44562e16,
    9.92398e15,
    4.90462e15,
    2.41524e15,
    1.26368e15,
    7.51617e14,
    5.01632e14,
    3.63641e14,
    2.79164e14,
    2.23193e14,
    1.83832e14,
    1.54863e14,
    1.32773e14,
)


def calculate_vv_max_force_density_from_w7x_scaling(
    rad_vv,
    b_plasma_toroidal_on_axis,
    c_tf_total,
    rminor,
    t_tf_superconductor_quench,
    dr_vv_inboard,
    dr_vv_outboard,
):
    """Actual vacuum-vessel force density from W7-X scaling (MN/m3)."""
    return (
        _FORCE_DENSITY_REF_MN_PER_M3
        * (
            _B_REF_T
            / b_plasma_toroidal_on_axis
            * _I_TOTAL_REF_A
            / c_tf_total
            * _RMINOR_REF_M**2
            / rminor**2
        )
        ** (-1)
        * (
            _TAU_REF_S
            / t_tf_superconductor_quench
            * _RMAJOR_REF_M
            / rad_vv
            * _DR_VV_REF_M
            / ((dr_vv_inboard + dr_vv_outboard) / 2)
        )
    )


def max_dump_voltage(tf_energy_stored, t_dump, current):
    """Max voltage during a fast TF coil discharge (V). Already pure in the source."""
    return 2 * (tf_energy_stored / t_dump) / current


def calculate_quench_protection_current_density(
    tau_quench, t_detect, f_cu, f_cond, temp, a_cable, a_turn
):
    """Current density limited by the quench-protection hotspot criterion (A/m2).

    Already pure in the source. `temp_k`/`q_cu_array_sa2m4`/`q_he_array_sa2m4` are
    fixed lookup tables (`np.interp` over 13 tabulated points), not iteration
    unknowns -- module-level constants here rather than arguments. `jnp.interp`
    (not `np.interp`) so this stays traceable.
    """
    q_he = jnp.interp(temp, jnp.asarray(_TEMP_K), jnp.asarray(_Q_HE_ARRAY_SA2M4))
    q_cu = jnp.interp(temp, jnp.asarray(_TEMP_K), jnp.asarray(_Q_CU_ARRAY_SA2M4))

    return (a_cable / a_turn) * safe_sqrt(
        1
        / (0.5 * tau_quench + t_detect)
        * (f_cu**2 * f_cond**2 * q_cu + f_cu * f_cond * (1 - f_cond) * q_he)
    )


def calculate_quench_protection(
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
):
    """Quench protection limits for the stellarator coils.

    Ports `calculate_quench_protection`'s full chain: vacuum-vessel stress from W7-X
    force-density scaling, quench-protection current density, copper current density,
    and max dump voltage.

    The source takes `coilcurrent` (total coils current, MA) as a separate explicit
    argument, but it is not an independent input: `coils/calculate.py`'s
    `winding_pack_total_size` sets `c_tf_total = n_tf_coils * coilcurrent * 1.0e6`
    (line 276) before `calculate_quench_protection` is ever called in `st_coil`'s call
    order (line 49 before line 118), and nothing else in that file writes `c_tf_total`.
    `coilcurrent` is therefore fully determined by two parameters already present here
    (`c_tf_total`, `n_tf_coils`) -- carrying it as a third, separately-sourced argument
    would be redundant, not a real second input. Derived internally instead, which also
    removes the only real-`data`-field gap this unit had (see the graph-node section
    in `functional_process/models/stellarator/coils/quench.py` and `quench.md`'s
    "calls into other models" note).

    Parameters
    ----------
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`. The source
        also approximates the vacuum-vessel major radius by `rmajor` directly
        (`rad_vv = data.physics.rmajor`, its own comment: "plasma r_major is just an
        approximation... exact calculations require 3D geometry") -- carried forward
        unchanged, not a decision made here.
    dr_fw_plasma_gap_inboard, dr_fw_inboard, dr_blkt_inboard, dr_shld_blkt_gap,
    dr_shld_inboard :
        Inboard build thicknesses feeding the vacuum-vessel inner radius (m).
        `.build.*`.
    dr_fw_plasma_gap_outboard, dr_fw_outboard, dr_blkt_outboard, dr_shld_outboard :
        Outboard build thicknesses feeding the vacuum-vessel outer radius (m).
        `.build.*` (`dr_shld_blkt_gap` is shared between the two sides in the source).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T). `.physics.b_plasma_toroidal_on_axis`.
    c_tf_total :
        Total TF coil current (A). `.tfcoil.c_tf_total`.
    t_tf_superconductor_quench :
        Quench time (s). `.tfcoil.t_tf_superconductor_quench`.
    dr_vv_inboard, dr_vv_outboard :
        Vacuum vessel inboard/outboard thickness (m). `.build.dr_vv_inboard` etc.
    t_tf_quench_detection :
        Quench detection time (s). `.tfcoil.t_tf_quench_detection`.
    f_a_tf_turn_cable_copper :
        Copper fraction of cable conductor. `.tfcoil.f_a_tf_turn_cable_copper`.
    f_a_tf_turn_cable_space_extra_void :
        Extra-void fraction of cable space. `.tfcoil.f_a_tf_turn_cable_space_extra_void`.
    tftmp :
        Peak helium coolant temperature in TF/PF coils (K). `.tfcoil.tftmp`.
    a_tf_turn_cable_space_no_void :
        Cable space area per turn, no void (m2). `.tfcoil.a_tf_turn_cable_space_no_void`.
    dx_tf_turn_general :
        TF coil turn side length (m). `.tfcoil.dx_tf_turn_general`.
    a_tf_wp_conductor :
        Winding pack conductor area (m2). `.tfcoil.a_tf_wp_conductor`.
    e_tf_magnetic_stored_total_gj :
        Total TF coil stored magnetic energy (GJ). `.tfcoil.e_tf_magnetic_stored_total_gj`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    c_tf_turn :
        TF coil current per turn (A). `.tfcoil.c_tf_turn`.

    Returns
    -------
    :
        `(f_vv_actual, vv_stress_quench, j_tf_wp_quench_heat_max, coppera_m2,
        v_tf_coil_dump_quench_kv)`.
    """
    rad_vv_in = (
        rmajor
        - rminor
        - dr_fw_plasma_gap_inboard
        - dr_fw_inboard
        - dr_blkt_inboard
        - dr_shld_blkt_gap
        - dr_shld_inboard
    )
    rad_vv_out = (
        rmajor
        + rminor
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
        + dr_blkt_outboard
        + dr_shld_blkt_gap
        + dr_shld_outboard
    )
    rad_vv = rmajor  # source approximation, see docstring
    coilcurrent = c_tf_total / (n_tf_coils * 1.0e6)  # see docstring

    f_vv_actual = calculate_vv_max_force_density_from_w7x_scaling(
        rad_vv,
        b_plasma_toroidal_on_axis,
        c_tf_total,
        rminor,
        t_tf_superconductor_quench,
        dr_vv_inboard,
        dr_vv_outboard,
    )

    a_vv = (rad_vv_out + rad_vv_in) / (rad_vv_out - rad_vv_in)
    zeta = 1 + ((a_vv - 1) * jnp.log((a_vv + 1) / (a_vv - 1)) / (2 * a_vv))
    vv_stress_quench = zeta * f_vv_actual * 1.0e6 * rad_vv_in

    j_tf_wp_quench_heat_max = calculate_quench_protection_current_density(
        tau_quench=t_tf_superconductor_quench,
        t_detect=t_tf_quench_detection,
        f_cu=f_a_tf_turn_cable_copper,
        f_cond=1 - f_a_tf_turn_cable_space_extra_void,
        temp=tftmp,
        a_cable=a_tf_turn_cable_space_no_void,
        a_turn=dx_tf_turn_general**2,
    )

    coppera_m2 = coilcurrent * 1.0e6 / (a_tf_wp_conductor * f_a_tf_turn_cable_copper)

    v_tf_coil_dump_quench_kv = (
        max_dump_voltage(
            tf_energy_stored=(e_tf_magnetic_stored_total_gj / n_tf_coils * 1.0e9),
            t_dump=t_tf_superconductor_quench,
            current=c_tf_turn,
        )
        / 1.0e3
    )

    return (
        f_vv_actual,
        vv_stress_quench,
        j_tf_wp_quench_heat_max,
        coppera_m2,
        v_tf_coil_dump_quench_kv,
    )
