"""Pure functions for the superconducting TF coil, extracted from
`functional_process/models/tfcoil/superconducting.py`.

That module still holds the graph declarations (`ExplicitFunction`/`FixedPointFunction`
occupants) that wire these functions to `VarPath`s; read its module docstring for scope,
the switch table, and the two findings that contradict `tokamak_boundary.md`. The audit
record is `functional_process/_audit/units/models/tfcoil/superconducting.md` and mirrors
these functions, not the declarations that call them.
"""

import jax
import jax.numpy as jnp

from functional_process.models.physics.superconductors import (
    gl_nbti,
    itersc,
    jcrit_nbti,
    western_superconducting_nb3sn,
)
from functional_process.models.safe_math import safe_sqrt
from functional_process.vocabulary import constants

_RIPPLE_FLAT_ALLOWANCE = 1.09
"""The `else` arm's ripple factor, `process/models/tfcoil/superconducting.py:1519`."""


def _wp_common_geometry(
    *,
    r_tf_inboard_in,
    dr_tf_nose_case,
    dr_tf_wp_with_insulation,
    tan_theta_coil,
    dx_tf_side_case_min,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """The five quantities computed before `i_tf_wp_geom` branches.

    `process/models/tfcoil/superconducting.py:1604-1622`.
    """
    r_tf_wp_inboard_inner = r_tf_inboard_in + dr_tf_nose_case
    r_tf_wp_inboard_outer = r_tf_wp_inboard_inner + dr_tf_wp_with_insulation
    r_tf_wp_inboard_centre = 0.5 * (r_tf_wp_inboard_inner + r_tf_wp_inboard_outer)

    dx_tf_wp_inner_toroidal = 2.0 * r_tf_wp_inboard_inner * tan_theta_coil
    dx_tf_wp_toroidal_min = dx_tf_wp_inner_toroidal - 2.0 * dx_tf_side_case_min

    dr_tf_wp_no_insulation = dr_tf_wp_with_insulation - 2.0 * (
        dx_tf_wp_insulation + dx_tf_wp_insertion_gap
    )
    return (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
    )


def superconducting_tf_wp_geometry_rectangular(
    *,
    r_tf_inboard_in,
    dr_tf_nose_case,
    dr_tf_wp_with_insulation,
    tan_theta_coil,
    dx_tf_side_case_min,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """`i_tf_wp_geom == RECTANGULAR (0)`. `superconducting.py:1628-1657`.

    Returns
    -------
    :
        `(r_tf_wp_inboard_inner, r_tf_wp_inboard_outer, r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min, dr_tf_wp_no_insulation, dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal, dx_tf_wp_toroidal_average,
        a_tf_wp_with_insulation, a_tf_wp_no_insulation, a_tf_wp_ground_insulation)`.
    """
    (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
    ) = _wp_common_geometry(
        r_tf_inboard_in=r_tf_inboard_in,
        dr_tf_nose_case=dr_tf_nose_case,
        dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        tan_theta_coil=tan_theta_coil,
        dx_tf_side_case_min=dx_tf_side_case_min,
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
    )

    dx_tf_wp_primary_toroidal = dx_tf_wp_toroidal_min
    dx_tf_wp_secondary_toroidal = dx_tf_wp_toroidal_min
    dx_tf_wp_toroidal_average = dx_tf_wp_toroidal_min

    a_tf_wp_with_insulation = dr_tf_wp_with_insulation * dx_tf_wp_primary_toroidal

    a_tf_wp_no_insulation = (
        dr_tf_wp_with_insulation - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
    ) * (
        dx_tf_wp_primary_toroidal - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
    )

    a_tf_wp_ground_insulation = (
        dr_tf_wp_with_insulation - 2.0 * dx_tf_wp_insertion_gap
    ) * (
        dx_tf_wp_primary_toroidal - 2.0 * dx_tf_wp_insertion_gap
    ) - a_tf_wp_no_insulation

    return (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dx_tf_wp_toroidal_average,
        a_tf_wp_with_insulation,
        a_tf_wp_no_insulation,
        a_tf_wp_ground_insulation,
    )


def superconducting_tf_wp_geometry_double_rectangular(
    *,
    r_tf_inboard_in,
    dr_tf_nose_case,
    dr_tf_wp_with_insulation,
    tan_theta_coil,
    dx_tf_side_case_min,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """`i_tf_wp_geom == DOUBLE_RECTANGULAR (1)` -- the reference arm.

    `superconducting.py:1661-1709`. Same return tuple as the rectangular arm.
    """
    (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
    ) = _wp_common_geometry(
        r_tf_inboard_in=r_tf_inboard_in,
        dr_tf_nose_case=dr_tf_nose_case,
        dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        tan_theta_coil=tan_theta_coil,
        dx_tf_side_case_min=dx_tf_side_case_min,
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
    )

    dx_tf_wp_primary_toroidal = 2.0 * (
        r_tf_wp_inboard_centre * tan_theta_coil - dx_tf_side_case_min
    )
    dx_tf_wp_secondary_toroidal = 2.0 * (
        r_tf_wp_inboard_inner * tan_theta_coil - dx_tf_side_case_min
    )
    dx_tf_wp_toroidal_average = 0.5 * (
        dx_tf_wp_primary_toroidal + dx_tf_wp_secondary_toroidal
    )

    a_tf_wp_with_insulation = dr_tf_wp_with_insulation * dx_tf_wp_toroidal_average

    a_tf_wp_no_insulation = (
        0.5
        * (
            dr_tf_wp_with_insulation
            - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
        )
        * (
            dx_tf_wp_primary_toroidal
            + dx_tf_wp_secondary_toroidal
            - 4.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
        )
    )

    a_tf_wp_ground_insulation = (
        0.5
        * (dr_tf_wp_with_insulation - 2.0 * dx_tf_wp_insertion_gap)
        * (
            dx_tf_wp_primary_toroidal
            + dx_tf_wp_secondary_toroidal
            - 4.0 * dx_tf_wp_insertion_gap
        )
        - a_tf_wp_no_insulation
    )

    return (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dx_tf_wp_toroidal_average,
        a_tf_wp_with_insulation,
        a_tf_wp_no_insulation,
        a_tf_wp_ground_insulation,
    )


def superconducting_tf_wp_geometry_trapezoidal(
    *,
    r_tf_inboard_in,
    dr_tf_nose_case,
    dr_tf_wp_with_insulation,
    tan_theta_coil,
    dx_tf_side_case_min,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """`i_tf_wp_geom == TRAPEZOIDAL (2)`. `superconducting.py:1713-1765`."""
    (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
    ) = _wp_common_geometry(
        r_tf_inboard_in=r_tf_inboard_in,
        dr_tf_nose_case=dr_tf_nose_case,
        dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        tan_theta_coil=tan_theta_coil,
        dx_tf_side_case_min=dx_tf_side_case_min,
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
    )

    dx_tf_wp_primary_toroidal = 2.0 * (
        r_tf_wp_inboard_outer * tan_theta_coil - dx_tf_side_case_min
    )
    dx_tf_wp_secondary_toroidal = 2.0 * (
        r_tf_wp_inboard_inner * tan_theta_coil - dx_tf_side_case_min
    )
    dx_tf_wp_toroidal_average = 0.5 * (
        dx_tf_wp_primary_toroidal + dx_tf_wp_secondary_toroidal
    )

    a_tf_wp_with_insulation = (
        dr_tf_wp_with_insulation
        * 0.5
        * (dx_tf_wp_primary_toroidal + dx_tf_wp_secondary_toroidal)
    )

    a_tf_wp_no_insulation = (
        (dr_tf_wp_with_insulation - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap))
        * (
            (
                dx_tf_wp_secondary_toroidal
                - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
            )
            + (
                dx_tf_wp_primary_toroidal
                - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
            )
        )
        / 2
    )

    a_tf_wp_ground_insulation = (
        dr_tf_wp_with_insulation - 2.0 * dx_tf_wp_insertion_gap
    ) * (
        (
            (dx_tf_wp_primary_toroidal - 2.0 * dx_tf_wp_insertion_gap)
            + (dx_tf_wp_secondary_toroidal - 2.0 * dx_tf_wp_insertion_gap)
        )
        / 2
    ) - a_tf_wp_no_insulation

    return (
        r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer,
        r_tf_wp_inboard_centre,
        dx_tf_wp_toroidal_min,
        dr_tf_wp_no_insulation,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dx_tf_wp_toroidal_average,
        a_tf_wp_with_insulation,
        a_tf_wp_no_insulation,
        a_tf_wp_ground_insulation,
    )


def _tf_case_areas(
    *,
    a_tf_inboard_total,
    n_tf_coils,
    a_tf_wp_with_insulation,
    a_tf_leg_outboard,
    rad_tf_coil_inboard_toroidal_half,
    tan_theta_coil,
    r_tf_wp_inboard_inner,
    r_tf_inboard_in,
    a_tf_plasma_case,
):
    """The three unswitched case areas plus the front area the caller supplies.

    `superconducting.py:1868-1892` less the `i_tf_case_geom` branch.
    """
    a_tf_coil_inboard_case = (a_tf_inboard_total / n_tf_coils) - a_tf_wp_with_insulation
    a_tf_coil_outboard_case = a_tf_leg_outboard - a_tf_wp_with_insulation
    a_tf_coil_nose_case = (
        tan_theta_coil * r_tf_wp_inboard_inner**2
        - rad_tf_coil_inboard_toroidal_half * r_tf_inboard_in**2
    )
    return (
        a_tf_coil_inboard_case,
        a_tf_coil_outboard_case,
        a_tf_plasma_case,
        a_tf_coil_nose_case,
    )


def tf_case_areas_circular_front(
    *,
    a_tf_inboard_total,
    n_tf_coils,
    a_tf_wp_with_insulation,
    a_tf_leg_outboard,
    rad_tf_coil_inboard_toroidal_half,
    r_tf_inboard_out,
    tan_theta_coil,
    r_tf_wp_inboard_outer,
    r_tf_wp_inboard_inner,
    r_tf_inboard_in,
):
    """`i_tf_case_geom == CIRCULAR (0)` -- the reference arm.

    `superconducting.py:1876-1880`. Note this arm does **not** read
    `.tfcoil.dr_tf_plasma_case`; the straight arm does. That is the invented edge the
    split removes.

    Returns
    -------
    :
        `(a_tf_coil_inboard_case, a_tf_coil_outboard_case, a_tf_plasma_case,
        a_tf_coil_nose_case)`, all m2.
    """
    a_tf_plasma_case = (rad_tf_coil_inboard_toroidal_half * r_tf_inboard_out**2) - (
        tan_theta_coil * r_tf_wp_inboard_outer**2
    )
    return _tf_case_areas(
        a_tf_inboard_total=a_tf_inboard_total,
        n_tf_coils=n_tf_coils,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_leg_outboard=a_tf_leg_outboard,
        rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
        tan_theta_coil=tan_theta_coil,
        r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
        r_tf_inboard_in=r_tf_inboard_in,
        a_tf_plasma_case=a_tf_plasma_case,
    )


def tf_case_areas_straight_front(
    *,
    a_tf_inboard_total,
    n_tf_coils,
    a_tf_wp_with_insulation,
    a_tf_leg_outboard,
    rad_tf_coil_inboard_toroidal_half,
    tan_theta_coil,
    r_tf_wp_inboard_outer,
    dr_tf_plasma_case,
    r_tf_wp_inboard_inner,
    r_tf_inboard_in,
):
    """`i_tf_case_geom == STRAIGHT (1)`. `superconducting.py:1881-1886`.

    Reads `dr_tf_plasma_case` and does **not** read `r_tf_inboard_out`.
    """
    a_tf_plasma_case = (
        (r_tf_wp_inboard_outer + dr_tf_plasma_case) ** 2 - r_tf_wp_inboard_outer**2
    ) * tan_theta_coil
    return _tf_case_areas(
        a_tf_inboard_total=a_tf_inboard_total,
        n_tf_coils=n_tf_coils,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_leg_outboard=a_tf_leg_outboard,
        rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
        tan_theta_coil=tan_theta_coil,
        r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
        r_tf_inboard_in=r_tf_inboard_in,
        a_tf_plasma_case=a_tf_plasma_case,
    )


def dx_tf_side_case_rectangular(
    *, dx_tf_side_case_min, tan_theta_coil, dr_tf_wp_with_insulation
):
    """`i_tf_wp_geom == RECTANGULAR (0)`. `superconducting.py:1906-1908,1931-1933`.

    Returns `(dx_tf_side_case_average, dx_tf_side_case_peak)`, m.
    """
    return (
        dx_tf_side_case_min + 0.5 * tan_theta_coil * dr_tf_wp_with_insulation,
        dx_tf_side_case_min + tan_theta_coil * dr_tf_wp_with_insulation,
    )


def dx_tf_side_case_double_rectangular(
    *, dx_tf_side_case_min, tan_theta_coil, dr_tf_wp_with_insulation
):
    """`i_tf_wp_geom == DOUBLE_RECTANGULAR (1)` -- the reference arm.

    `superconducting.py:1912-1914,1936-1938`.
    """
    return (
        dx_tf_side_case_min + 0.25 * tan_theta_coil * dr_tf_wp_with_insulation,
        dx_tf_side_case_min + 0.5 * tan_theta_coil * dr_tf_wp_with_insulation,
    )


def dx_tf_side_case_trapezoidal(*, dx_tf_side_case_min):
    """`i_tf_wp_geom == TRAPEZOIDAL (2)`: constant thickness, so peak == average.

    `superconducting.py:1918,1943`. Reads one variable where the other two arms read
    three.
    """
    return dx_tf_side_case_min, dx_tf_side_case_min


def tf_wp_currents(*, c_tf_total, n_tf_coils, a_tf_wp_no_insulation):
    """Winding-pack engineering current density (A/m2), floored at 1.

    Ports `tf_wp_currents`, `process/models/tfcoil/superconducting.py:1958-1970`.
    PROCESS takes the whole `DataStructure` and mutates it in place; the arithmetic is
    three reads and one write, promoted here to an ordinary signature.
    `max` -> `jnp.maximum`, since `c_tf_total` is differentiable.
    """
    return jnp.maximum(1.0, c_tf_total / (n_tf_coils * a_tf_wp_no_insulation))


def peak_b_tf_inboard_with_ripple_kovari(
    *,
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
    coefficients,
):
    """The MAGINT-fit arms of `peak_b_tf_inboard_with_ripple` (16, 18 or 20 coils).

    Ports `process/models/tfcoil/superconducting.py:1521-1556`. `coefficients` is the
    four-element fit tuple the coil count selects; the occupant classes below bind it,
    so no `a[]` lookup survives into the traced body.

    PROCESS's two out-of-fitted-range `logger.warning`s (`superconducting.py:1532-1545`)
    have no value effect and are dropped -- a pure function does not log, and the two
    ratios they test (`tf_fit_t`, `tf_fit_z`) are returned, so a caller that wants the
    check can make it.

    Returns
    -------
    :
        `(tf_fit_t, tf_fit_z, f_b_tf_inboard_peak_ripple_symmetric,
        b_tf_inboard_peak_with_ripple)` -- the two dimensionless winding-pack widths,
        the ripple factor, and the peak field (T). PROCESS returns only the last and
        stores the first three on `data.superconducting_tfcoil` as side effects.
    """
    a0, a1, a2, a3 = coefficients

    dx_tf_wp_toroidal_max = (
        2.0 * r_tf_wp_inboard_centre + dr_tf_wp_no_insulation
    ) * jnp.tan(jnp.pi / n_tf_coils)

    tf_fit_t = dx_tf_wp_primary_toroidal / dx_tf_wp_toroidal_max
    tf_fit_z = dr_tf_wp_no_insulation / dx_tf_wp_toroidal_max

    f_b_tf_inboard_peak_ripple_symmetric = (
        a0 + a1 * jnp.exp(-tf_fit_t) + a2 * tf_fit_z + a3 * tf_fit_z * tf_fit_t
    )
    return (
        tf_fit_t,
        tf_fit_z,
        f_b_tf_inboard_peak_ripple_symmetric,
        f_b_tf_inboard_peak_ripple_symmetric * b_tf_inboard_peak_symmetric,
    )


def peak_b_tf_inboard_with_ripple_flat(*, b_tf_inboard_peak_symmetric):
    """Every coil count outside {16, 18, 20}: a flat 9 % ripple allowance.

    Ports `process/models/tfcoil/superconducting.py:1518-1519`. Reads exactly one
    variable, and writes none of `tf_fit_t`/`tf_fit_z`/
    `f_b_tf_inboard_peak_ripple_symmetric` -- PROCESS returns before they are assigned.
    """
    return _RIPPLE_FLAT_ALLOWANCE * b_tf_inboard_peak_symmetric


def _cicc_averaged_turn_geometry_from_turn_area(
    *,
    a_tf_turn,
    dx_tf_turn_general,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    layer_ins,
    a_tf_wp_no_insulation,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_space_extra_void,
):
    """Everything `tf_cable_in_conduit_averaged_turn_geometry` does after its branch.

    `process/models/tfcoil/superconducting.py:3336-3404`, shared verbatim by all three
    arms -- they differ only in how `a_tf_turn`/`dx_tf_turn_general`/`c_tf_turn` are
    obtained (lines 3305-3334).

    **PROCESS's degenerate-cable fallback is reproduced exactly, ordering included.**
    Lines 3384-3399: when the cable-space area comes out non-positive *and* the conduit
    dimension is not itself negative, the rounded-corner radius is zeroed and the cable
    space recomputed as a plain square. That happens **after**
    `a_tf_turn_cable_space_effective` and `f_a_tf_turn_cable_space_cooling` have already
    been computed from the pre-fallback value, so those two keep the *un*-corrected
    number while `a_tf_turn_steel` uses the corrected one. Ported as written
    (`jnp.where`, since the condition is data-dependent), and flagged as defect **D2**
    in `superconducting.md` rather than tidied.
    """
    dr_tf_turn = dx_tf_turn_general
    dx_tf_turn = dx_tf_turn_general

    dx_tf_turn_conduit_full_average = (
        -layer_ins + safe_sqrt(layer_ins**2 + 4.0 * a_tf_turn)
    ) / 2 - 2.0 * dx_tf_turn_insulation

    n_tf_coil_turns = a_tf_wp_no_insulation / a_tf_turn
    a_tf_turn_insulation = a_tf_turn - dx_tf_turn_conduit_full_average**2

    radius_tf_turn_cable_space_corners = dx_tf_turn_steel * 0.75
    dx_tf_turn_cable_space_average = (
        dx_tf_turn_conduit_full_average - 2.0 * dx_tf_turn_steel
    )

    a_tf_turn_cable_space_no_void = (
        dx_tf_turn_cable_space_average**2
        - (4.0 - jnp.pi) * radius_tf_turn_cable_space_corners**2
    )

    a_tf_turn_cable_space_effective = (
        a_tf_turn_cable_space_no_void
        - ((jnp.pi / 4.0) * dia_tf_turn_coolant_channel * dia_tf_turn_coolant_channel)
        - (a_tf_turn_cable_space_no_void * f_a_tf_turn_cable_space_extra_void)
    )
    f_a_tf_turn_cable_space_cooling = 1 - (
        a_tf_turn_cable_space_effective / a_tf_turn_cable_space_no_void
    )

    degenerate = (a_tf_turn_cable_space_no_void <= 0.0) & (
        dx_tf_turn_conduit_full_average >= 0.0
    )
    radius_tf_turn_cable_space_corners = jnp.where(
        degenerate, 0.0, radius_tf_turn_cable_space_corners
    )
    a_tf_turn_cable_space_no_void = jnp.where(
        degenerate,
        dx_tf_turn_cable_space_average**2,
        a_tf_turn_cable_space_no_void,
    )

    a_tf_turn_steel = dx_tf_turn_conduit_full_average**2 - a_tf_turn_cable_space_no_void

    return (
        a_tf_turn_cable_space_no_void,
        a_tf_turn_steel,
        a_tf_turn_insulation,
        n_tf_coil_turns,
        dx_tf_turn_general,
        dr_tf_turn,
        dx_tf_turn,
        dx_tf_turn_conduit_full_average,
        radius_tf_turn_cable_space_corners,
        dx_tf_turn_cable_space_average,
        a_tf_turn_cable_space_effective,
        f_a_tf_turn_cable_space_cooling,
    )


def cicc_averaged_turn_geometry_from_current_per_turn(
    *,
    j_tf_wp,
    c_tf_turn,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    layer_ins,
    a_tf_wp_no_insulation,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_space_extra_void,
):
    """`i_dx_tf_turn_general_input == False` and
    `i_dx_tf_turn_cable_space_general_input == False` -- the reference arm.

    Ports `process/models/tfcoil/superconducting.py:3326-3334` plus the shared tail.
    `c_tf_turn` is an **input** on this arm and is not returned: PROCESS returns it
    unchanged (`superconducting.py:3411`) and `run` writes it back to the field it read
    it from, which is an identity, not a production. See the module docstring, finding 1.

    Returns
    -------
    :
        `(a_tf_turn_cable_space_no_void, a_tf_turn_steel, a_tf_turn_insulation,
        n_tf_coil_turns, dx_tf_turn_general, dr_tf_turn, dx_tf_turn,
        dx_tf_turn_conduit_full_average, radius_tf_turn_cable_space_corners,
        dx_tf_turn_cable_space_average, a_tf_turn_cable_space_effective,
        f_a_tf_turn_cable_space_cooling)`.
    """
    a_tf_turn = c_tf_turn / j_tf_wp
    dx_tf_turn_general = safe_sqrt(a_tf_turn)

    return _cicc_averaged_turn_geometry_from_turn_area(
        a_tf_turn=a_tf_turn,
        dx_tf_turn_general=dx_tf_turn_general,
        dx_tf_turn_steel=dx_tf_turn_steel,
        dx_tf_turn_insulation=dx_tf_turn_insulation,
        layer_ins=layer_ins,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
    )


def cicc_integer_turn_geometry(
    *,
    dr_tf_wp_with_insulation,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
    n_tf_wp_layers,
    dx_tf_wp_toroidal_min,
    n_tf_wp_pancakes,
    c_tf_coil,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_space_extra_void,
):
    """`tf_cable_in_conduit_integer_turn_geometry` -- the `i_tf_turns_integer == 1` arm.

    `process/models/tfcoil/superconducting.py:3423-3600`. The turn is **rectangular**,
    not square: the radial pitch is the winding pack's radial extent over
    `n_tf_wp_layers`, the toroidal pitch its minimum toroidal extent over
    `n_tf_wp_pancakes`, and nothing forces the two equal. That is the observable
    difference from the averaged arm, whose `dr_tf_turn == dx_tf_turn` by construction
    -- and the shape of the silent mis-assembly that motivated this occupant
    (`low_aspect_ratio_DEMO` computed a 0.0568 m square where PROCESS's converged turn
    is 0.0547 x 0.0591 m).

    Two `logger.error` guards on negative cable-space dimensions
    (`superconducting.py:3477-3495`) are logging only and dropped, same policy as
    `CiccInboardAreasAndFractions`. **The degenerate-cable fallback is reproduced
    exactly, ordering included** (`:3550-3567`): when the cable-space area comes out
    non-positive *and* neither cable-space dimension is itself negative, the
    rounded-corner radius is zeroed and the area recomputed as the plain rectangle --
    *after* `a_tf_turn_cable_space_effective` and `f_a_tf_turn_cable_space_cooling`
    were computed from the pre-fallback value, so those two keep the uncorrected
    number while `a_tf_turn_steel` uses the corrected one. Same defect shape as the
    averaged arm's **D2** (`superconducting.md`), ported as written via `jnp.where`.

    Unlike the averaged arm, `c_tf_turn` is **owned** here: `c_tf_coil` divided by the
    (fixed) turn count, `superconducting.py:3505`. The turn count itself is
    `n_tf_wp_layers * n_tf_wp_pancakes`, a product of two inputs -- constant in every
    derivative sense, but still a node output because downstream nodes read it.

    Returns
    -------
    :
        `(radius_tf_turn_cable_space_corners, dr_tf_turn, dx_tf_turn,
        a_tf_turn_cable_space_no_void, a_tf_turn_steel, a_tf_turn_insulation,
        c_tf_turn, n_tf_coil_turns, dr_tf_turn_conduit_full,
        dx_tf_turn_conduit_full_toroidal, dx_tf_turn_conduit_full_average,
        dr_tf_turn_cable_space, dx_tf_turn_cable_space,
        dx_tf_turn_cable_space_average, a_tf_turn_cable_space_effective,
        f_a_tf_turn_cable_space_cooling, dx_tf_turn_general)` -- `run`'s write order
        (`superconducting.py:2404-2439`).
    """
    radius_tf_turn_cable_space_corners = dx_tf_turn_steel * 0.75

    dr_tf_turn = (
        dr_tf_wp_with_insulation - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
    ) / n_tf_wp_layers
    dx_tf_turn = (
        dx_tf_wp_toroidal_min - 2.0 * (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)
    ) / n_tf_wp_pancakes

    dx_tf_turn_general = safe_sqrt(dr_tf_turn * dx_tf_turn)

    n_tf_coil_turns = n_tf_wp_layers * n_tf_wp_pancakes
    c_tf_turn = c_tf_coil / n_tf_coil_turns

    dr_tf_turn_conduit_full = dr_tf_turn - 2.0 * dx_tf_turn_insulation
    dx_tf_turn_conduit_full_toroidal = dx_tf_turn - 2.0 * dx_tf_turn_insulation
    dx_tf_turn_conduit_full_average = safe_sqrt(
        dr_tf_turn_conduit_full * dx_tf_turn_conduit_full_toroidal
    )

    dr_tf_turn_cable_space = dr_tf_turn_conduit_full - 2.0 * dx_tf_turn_steel
    dx_tf_turn_cable_space = dx_tf_turn_conduit_full_toroidal - 2.0 * dx_tf_turn_steel
    dx_tf_turn_cable_space_average = safe_sqrt(
        dr_tf_turn_cable_space * dx_tf_turn_cable_space
    )

    a_tf_turn_cable_space_no_void = (dr_tf_turn_cable_space * dx_tf_turn_cable_space) - (
        4.0 - jnp.pi
    ) * radius_tf_turn_cable_space_corners**2

    a_tf_turn_cable_space_effective = (
        a_tf_turn_cable_space_no_void
        - ((jnp.pi / 4.0) * dia_tf_turn_coolant_channel * dia_tf_turn_coolant_channel)
        - (a_tf_turn_cable_space_no_void * f_a_tf_turn_cable_space_extra_void)
    )
    f_a_tf_turn_cable_space_cooling = 1 - (
        a_tf_turn_cable_space_effective / a_tf_turn_cable_space_no_void
    )

    degenerate = (
        (a_tf_turn_cable_space_no_void <= 0.0)
        & (dr_tf_turn_cable_space >= 0.0)
        & (dx_tf_turn_cable_space >= 0.0)
    )
    radius_tf_turn_cable_space_corners = jnp.where(
        degenerate, 0.0, radius_tf_turn_cable_space_corners
    )
    a_tf_turn_cable_space_no_void = jnp.where(
        degenerate,
        dr_tf_turn_cable_space * dx_tf_turn_cable_space,
        a_tf_turn_cable_space_no_void,
    )

    a_tf_turn_steel = (
        dr_tf_turn_conduit_full * dx_tf_turn_conduit_full_toroidal
        - a_tf_turn_cable_space_no_void
    )
    a_tf_turn_insulation = (
        dr_tf_turn * dx_tf_turn - a_tf_turn_steel - a_tf_turn_cable_space_no_void
    )

    return (
        radius_tf_turn_cable_space_corners,
        dr_tf_turn,
        dx_tf_turn,
        a_tf_turn_cable_space_no_void,
        a_tf_turn_steel,
        a_tf_turn_insulation,
        c_tf_turn,
        n_tf_coil_turns,
        dr_tf_turn_conduit_full,
        dx_tf_turn_conduit_full_toroidal,
        dx_tf_turn_conduit_full_average,
        dr_tf_turn_cable_space,
        dx_tf_turn_cable_space,
        dx_tf_turn_cable_space_average,
        a_tf_turn_cable_space_effective,
        f_a_tf_turn_cable_space_cooling,
        dx_tf_turn_general,
    )


def tf_cicc_inboard_areas_and_fractions(
    *,
    n_tf_coil_turns,
    dia_tf_turn_coolant_channel,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    a_tf_turn_insulation,
    a_tf_turn_steel,
    n_tf_coils,
    a_tf_inboard_total,
    a_tf_coil_inboard_case,
    a_tf_wp_ground_insulation,
):
    """Inboard winding-pack areas and steel/insulation fractions. Unchanged.

    Ports `tf_cicc_inboard_areas_and_fractions`,
    `process/models/tfcoil/superconducting.py:3599-3670`. No switch.

    Returns
    -------
    :
        `(a_tf_wp_coolant_channels, a_tf_wp_conductor, a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation, a_tf_wp_steel, a_tf_coil_inboard_steel,
        f_a_tf_coil_inboard_steel, a_tf_coil_inboard_insulation,
        f_a_tf_coil_inboard_insulation)`.
    """
    a_tf_wp_coolant_channels = (
        0.25 * n_tf_coil_turns * jnp.pi * dia_tf_turn_coolant_channel**2
    )

    a_tf_wp_conductor = (
        a_tf_turn_cable_space_no_void
        * n_tf_coil_turns
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        - a_tf_wp_coolant_channels
    )

    a_tf_wp_extra_void = (
        a_tf_turn_cable_space_no_void
        * n_tf_coil_turns
        * f_a_tf_turn_cable_space_extra_void
    )

    a_tf_coil_wp_turn_insulation = n_tf_coil_turns * a_tf_turn_insulation
    a_tf_wp_steel = n_tf_coil_turns * a_tf_turn_steel

    a_tf_coil_inboard_steel = a_tf_coil_inboard_case + a_tf_wp_steel
    f_a_tf_coil_inboard_steel = n_tf_coils * a_tf_coil_inboard_steel / a_tf_inboard_total

    a_tf_coil_inboard_insulation = (
        a_tf_coil_wp_turn_insulation + a_tf_wp_ground_insulation
    )
    f_a_tf_coil_inboard_insulation = (
        n_tf_coils * a_tf_coil_inboard_insulation / a_tf_inboard_total
    )

    return (
        a_tf_wp_coolant_channels,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_coil_inboard_steel,
        f_a_tf_coil_inboard_steel,
        a_tf_coil_inboard_insulation,
        f_a_tf_coil_inboard_insulation,
    )


def calculate_a_tf_turn(*, c_tf_total, j_tf_wp, n_tf_coils, n_tf_coil_turns):
    """Cross-sectional area per turn (m2). `superconducting.py:2700-2704`, inline in
    `run`.
    """
    return c_tf_total / (j_tf_wp * n_tf_coils * n_tf_coil_turns)


def superconducting_tf_coil_areas_and_masses_conventional(
    *,
    len_tf_coil,
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    den_tf_wp_turn_insulation,
    z_tf_inside_half,
    dr_tf_inboard,
    den_tf_coil_case,
    a_tf_coil_inboard_case,
    a_tf_coil_outboard_case,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    """`itart == 0`: superconducting TF coil component masses.

    Ports `superconducting_tf_coil_areas_and_masses`,
    `process/models/tfcoil/superconducting.py:1972-2081`, taking the `else` at line
    2007. The `itart == 1` arm is
    `superconducting_tf_coil_areas_and_masses_spherical_tokamak` below; it reads
    **exactly the same fields** and differs in only two places -- the outboard length in
    the case-mass formula (`:1998-2006`) and the extra `whtcp`/`whttflgs` split
    (`:2086-2093`). The shared physics therefore lives in the private
    `_superconducting_tf_coil_masses` helper, the same treatment
    `hcpb.py::_nuclear_heating_shield` gives that `itart` pair.

    `.tfcoil.cplen` is written by the source (`line 1989`) and read back inside the same
    call (`2002`, `2012-2013`); it is `local-intermediate` in the sense
    `models/stellarator/coils/mass.py` uses, so it is a Python local here *and* a
    declared output, since it is a real field with readers elsewhere.

    Parameters
    ----------
    den_tf_sc_material :
        Superconductor density (kg/m3) -- `.tfcoil.dcond[i_tf_sc_mat - 1]`, already
        indexed by the material switch. Same treatment as
        `models/stellarator/coils/mass.py`: the index is chosen at assembly, by which
        material occupant fills the slot, so this pure function never sees the switch.

    Returns
    -------
    :
        `(m_tf_coil_wp_insulation, cplen, m_tf_coil_case, m_tf_coil_superconductor,
        m_tf_coil_copper, m_tf_wp_steel_conduit, m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor, m_tf_coil, m_tf_coils_total)` -- all kg except `cplen` (m).
    """
    cplen = calculate_cplen(
        z_tf_inside_half=z_tf_inside_half, dr_tf_inboard=dr_tf_inboard
    )
    return _superconducting_tf_coil_masses(
        len_tf_coil=len_tf_coil,
        cplen=cplen,
        # `:2013` -- the conventional arm's `len_tf_coil` *includes* the inboard leg, so
        # the outboard case length is the coil length less the centrepost.
        len_tf_coil_case_outboard=len_tf_coil - cplen,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
        den_tf_coil_case=den_tf_coil_case,
        a_tf_coil_inboard_case=a_tf_coil_inboard_case,
        a_tf_coil_outboard_case=a_tf_coil_outboard_case,
        n_tf_coil_turns=n_tf_coil_turns,
        a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels=a_tf_wp_coolant_channels,
        den_tf_sc_material=den_tf_sc_material,
        a_tf_turn_steel=a_tf_turn_steel,
        den_steel=den_steel,
        a_tf_coil_wp_turn_insulation=a_tf_coil_wp_turn_insulation,
        n_tf_coils=n_tf_coils,
    )


def superconducting_tf_coil_areas_and_masses_spherical_tokamak(
    *,
    len_tf_coil,
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    den_tf_wp_turn_insulation,
    z_tf_inside_half,
    dr_tf_inboard,
    den_tf_coil_case,
    a_tf_coil_inboard_case,
    a_tf_coil_outboard_case,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    """`itart == 1`: superconducting TF coil masses, split centrepost/outboard legs.

    Ports `superconducting_tf_coil_areas_and_masses`,
    `process/models/tfcoil/superconducting.py:1972-2093`, taking the `if` at line 1995
    and the `if` at line 2085. Same reads as the conventional arm, exactly -- the two
    differences are:

    1. **`:1996-1997`, stated in PROCESS's own comment**: at `itart == 1`,
       `.tfcoil.len_tf_coil` does *not* include the inboard leg (the centrepost), so the
       outboard case length in the 2.2-factor case mass is `len_tf_coil` rather than
       `len_tf_coil - cplen`. `cplen` itself is formed identically in both arms
       (`:1988-1991`, above the branch).
    2. **`:2085-2093`**: the total coil-set mass is apportioned between centrepost and
       outboard legs in the ratio of their lengths, writing `.tfcoil.whtcp` and
       `.tfcoil.whttflgs`. The conventional arm writes neither, which is why this is an
       occupant and not a kwarg -- **conditional ownership**.

    The apportioning denominator is `tfleng_sph = cplen + len_tf_coil` (`:2087`) and not
    `len_tf_coil`, which is consistent with (1): the two lengths are disjoint at
    `itart == 1`, so their sum is the whole coil. Ported as written.

    **`whtcp` is not a resistive-centrepost mass here.** PROCESS has a separate
    resistive-TART centrepost chain (`i_tf_sup = 0`), but on a superconducting TART
    (`i_tf_sup = 1`, which is what both ST regression files set) *this* line is the sole
    producer of `.tfcoil.whtcp` and `.tfcoil.whttflgs`, and it is a pure re-apportioning
    of `m_tf_coils_total`. It reads nothing a conventional run does not already produce.

    Parameters
    ----------
    den_tf_sc_material :
        Superconductor density (kg/m3) -- `.tfcoil.dcond[i_tf_sc_mat - 1]`, already
        indexed by the material switch, exactly as the conventional arm takes it.

    Returns
    -------
    :
        The conventional arm's ten values, then `(whtcp, whttflgs)` -- centrepost mass
        and outboard-leg mass (kg), both for the whole coil set.
    """
    cplen = calculate_cplen(
        z_tf_inside_half=z_tf_inside_half, dr_tf_inboard=dr_tf_inboard
    )
    masses = _superconducting_tf_coil_masses(
        len_tf_coil=len_tf_coil,
        cplen=cplen,
        # `:1996-2005` -- `len_tf_coil` excludes the centrepost at `itart == 1`.
        len_tf_coil_case_outboard=len_tf_coil,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
        den_tf_coil_case=den_tf_coil_case,
        a_tf_coil_inboard_case=a_tf_coil_inboard_case,
        a_tf_coil_outboard_case=a_tf_coil_outboard_case,
        n_tf_coil_turns=n_tf_coil_turns,
        a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels=a_tf_wp_coolant_channels,
        den_tf_sc_material=den_tf_sc_material,
        a_tf_turn_steel=a_tf_turn_steel,
        den_steel=den_steel,
        a_tf_coil_wp_turn_insulation=a_tf_coil_wp_turn_insulation,
        n_tf_coils=n_tf_coils,
    )

    m_tf_coils_total = masses[-1]

    # `:2086-2093` -- total TF mass apportioned by length between centrepost (`cplen`)
    # and outboard legs (`len_tf_coil`), which are disjoint at `itart == 1`.
    tfleng_sph = cplen + len_tf_coil
    whtcp = m_tf_coils_total * (cplen / tfleng_sph)
    whttflgs = m_tf_coils_total * (len_tf_coil / tfleng_sph)

    return (*masses, whtcp, whttflgs)


def calculate_cplen(*, z_tf_inside_half, dr_tf_inboard):
    """Length of the vertical (inboard) TF segment, m. `superconducting.py:1988-1991`.

    Formed above the `itart` branch, so both arms share it verbatim; a named function
    rather than a line copied into each, so the two cannot drift.
    """
    return (2.0 * z_tf_inside_half) + (2.0 * dr_tf_inboard)


def _superconducting_tf_coil_masses(
    *,
    len_tf_coil,
    cplen,
    len_tf_coil_case_outboard,
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    den_tf_wp_turn_insulation,
    den_tf_coil_case,
    a_tf_coil_inboard_case,
    a_tf_coil_outboard_case,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    """The part of `superconducting_tf_coil_areas_and_masses` both `itart` arms share.

    A private helper rather than a node: the two occupants differ only in
    `len_tf_coil_case_outboard`, and duplicating sixty lines of mass algebra to make that
    point would add a second place for it to drift. Same shape as
    `hcpb.py::_nuclear_heating_shield`.

    Parameters
    ----------
    cplen, len_tf_coil_case_outboard :
        Centrepost length (m) and the outboard length the 2.2-factor case mass uses,
        both formed by the calling arm.

    Returns
    -------
    :
        The public conventional arm's ten values.
    """
    m_tf_coil_wp_insulation = (
        len_tf_coil
        * (a_tf_wp_with_insulation - a_tf_wp_no_insulation)
        * den_tf_wp_turn_insulation
    )

    # The 2.2 factor fits the ITER-FDR 450 t value; CCFE note T&M/PKNIGHT/PROCESS/026.
    m_tf_coil_case = (
        2.2
        * den_tf_coil_case
        * (
            cplen * a_tf_coil_inboard_case
            + len_tf_coil_case_outboard * a_tf_coil_outboard_case
        )
    )

    m_tf_coil_superconductor = (
        len_tf_coil
        * n_tf_coil_turns
        * a_tf_turn_cable_space_no_void
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        * (1.0 - f_a_tf_turn_cable_copper)
        - len_tf_coil * a_tf_wp_coolant_channels
    ) * den_tf_sc_material

    m_tf_coil_copper = jnp.maximum(
        0.0,
        (
            len_tf_coil
            * n_tf_coil_turns
            * a_tf_turn_cable_space_no_void
            * (1.0 - f_a_tf_turn_cable_space_extra_void)
            * f_a_tf_turn_cable_copper
            - len_tf_coil * a_tf_wp_coolant_channels
        )
        * constants.DEN_COPPER,
    )

    m_tf_wp_steel_conduit = len_tf_coil * n_tf_coil_turns * a_tf_turn_steel * den_steel

    m_tf_coil_wp_turn_insulation = (
        len_tf_coil * a_tf_coil_wp_turn_insulation * den_tf_wp_turn_insulation
    )

    m_tf_coil_conductor = (
        m_tf_coil_superconductor
        + m_tf_coil_copper
        + m_tf_wp_steel_conduit
        + m_tf_coil_wp_turn_insulation
    )

    m_tf_coil = m_tf_coil_case + m_tf_coil_conductor + m_tf_coil_wp_insulation
    m_tf_coils_total = m_tf_coil * n_tf_coils

    return (
        m_tf_coil_wp_insulation,
        cplen,
        m_tf_coil_case,
        m_tf_coil_superconductor,
        m_tf_coil_copper,
        m_tf_wp_steel_conduit,
        m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor,
        m_tf_coil,
        m_tf_coils_total,
    )


def _safe_arcsin(x):
    """`jnp.arcsin(x)` with a finite derivative at `x == +-1`.

    `models/safe_math.py`'s idiom, applied to the other function in this port whose
    derivative is infinite at an argument the code reaches *exactly*: `arcsin'(x) =
    1 / sqrt(1 - x**2)`. The double `jnp.where` is load-bearing for the same reason it is
    there -- a single one still evaluates `arcsin` at the edge and still leaks the `inf`
    into the tangent.

    Value-identical: `arcsin(+-1)` is `+-pi/2` exactly, which is what the edge branch
    returns.

    **Why this is reached and not merely defensive.** `itoh_theta_factor_integral`
    evaluates `itoh_lambda_term` at a `tau` matrix two of whose six entries are the
    literals `1.0` and `-1.0` (`superconducting.py:4944-4947`). At `tau == 1` the
    argument `(1 + omega * tau) / (tau + omega)` is identically `1` for **every**
    `omega`, so the true derivative with respect to `omega` is zero and only the
    chain-rule formula is singular. Lives here rather than in `safe_math.py` only
    because this is the single site; it belongs there the next time a second one
    appears.
    """
    at_edge = jnp.abs(x) == 1.0
    return jnp.where(
        at_edge,
        jnp.sign(x) * (jnp.pi / 2.0),
        jnp.arcsin(jnp.where(at_edge, 0.0, x)),
    )


def itoh_lambda_term(*, tau, omega):
    """The Itoh appendix-A `lambda` integral. `superconducting.py:4869-4896`.

    PROCESS branches on `p = 1 - omega**2 < 0` and picks `arcsin` or `log`; both arms
    divide by `sqrt(|p|)`, which is why the branch is only about the numerator.

    Ported with `jnp.where` **and every argument substituted**, not merely selected: an
    unguarded `arcsin` of an out-of-range argument, or a `sqrt` of `p * (1 - tau**2)`
    when that product is negative, returns NaN on the *untaken* arm -- the same trap
    `quench.py`'s `copper_magneto_resistivity` documents for its `log10`.

    **Two further guards, and both are reached rather than defensive.**
    `itoh_theta_factor_integral` calls this at a `tau` matrix two of whose six entries
    are the literals `1.0` and `-1.0` (`superconducting.py:4944-4947`), and at
    `|tau| == 1` both arms sit exactly on a singularity of their own derivative formula
    while the value is perfectly ordinary:

    - `p * (1 - tau**2)` is **identically zero**, and `sqrt'(0)` is infinite -- so
      `safe_sqrt`, whose zero branch has derivative `0`, the convention
      `models/safe_math.py` documents.
    - `(1 + omega * tau) / (tau + omega)` is **identically `+-1`**, and `arcsin'(+-1)`
      is infinite -- so `_safe_arcsin` above.

    Measured before either: every one of the eighteen `d(vv_stress)/d(...)` came out
    `nan` against a finite PROCESS finite difference, while every value test passed.
    That is exactly the defect class `_audit/next_steps.md` §9 exists for, found in a
    fifth place.
    """
    p = 1.0 - omega**2.0
    negative = p < 0.0
    scale = 1.0 / jnp.sqrt(jnp.abs(p))

    arcsin_argument = jnp.where(negative, (1.0 + omega * tau) / (tau + omega), 0.0)
    radicand = jnp.where(negative, 0.0, p * (1.0 - tau**2.0))
    log_argument = jnp.where(
        negative,
        1.0,
        (2.0 * (1.0 + tau * omega - safe_sqrt(radicand))) / (tau + omega),
    )
    return scale * jnp.where(
        negative, _safe_arcsin(arcsin_argument), jnp.log(log_argument)
    )


def itoh_theta_factor_integral(*, ro_vv, ri_vv, rm_vv, h_vv, theta1_vv):
    """The theta factor of Itoh et al. Eq 4. `superconducting.py:4899-4960`.

    `theta1_vv` is in **radians** here -- `vv_stress_on_quench` converts before calling
    this and does *not* convert before calling `itoh_inductance_factor`, which divides
    by 90. Both are transcribed as written; the asymmetry is PROCESS's.

    PROCESS's `for k in range(len(omega))` over a `(2, 3)` `tau` and a `(3,)` `omega`
    becomes one vectorised `jnp.sum`.
    """
    theta2 = jnp.pi / 2.0 + theta1_vv
    a = (ro_vv - ri_vv) / 2.0
    rbar = (ro_vv + ri_vv) / 2.0
    delta = (rbar - rm_vv) / a
    kappa = h_vv / a
    iota = (1.0 + delta) / kappa

    denom = jnp.cos(theta1_vv) + jnp.sin(theta1_vv) - 1.0

    r1 = h_vv * ((jnp.cos(theta1_vv) + iota * (jnp.sin(theta1_vv) - 1.0)) / denom)
    r2 = h_vv * ((jnp.cos(theta1_vv) - 1.0 + iota * jnp.sin(theta1_vv)) / denom)
    r3 = h_vv * (1 - delta) / kappa

    rc1 = (h_vv / kappa) * ((rbar / a) + 1.0) - r1
    rc2 = rc1 + (r1 - r2) * jnp.cos(theta1_vv)
    rc3 = rc2
    zc2 = (r1 - r2) * jnp.sin(theta1_vv)
    zc3 = zc2 + r2 - r3

    tau_upper = jnp.stack([
        jnp.cos(theta1_vv),
        jnp.cos(theta1_vv + theta2),
        jnp.asarray(-1.0),
    ])
    tau_lower = jnp.stack([
        jnp.asarray(1.0),
        jnp.cos(theta1_vv),
        jnp.cos(theta1_vv + theta2),
    ])
    omega = jnp.stack([rc1 / r1, rc2 / r2, rc3 / r3])

    # PROCESS assumes up-down symmetry and sets `Zc6 = -Zc3` (`:5016`).
    chi1 = (zc3 + jnp.abs(-zc3)) / ri_vv
    chi2 = jnp.sum(
        jnp.abs(
            itoh_lambda_term(tau=tau_lower, omega=omega)
            - itoh_lambda_term(tau=tau_upper, omega=omega)
        )
    )
    return (chi1 + 2.0 * chi2) / (2.0 * jnp.pi)


def itoh_inductance_factor(*, h, ri, ro, rm, theta1):
    """Surrogate-2 inductance factor. `superconducting.py:5115-5153`.

    `theta1` is in **degrees** -- the body divides it by 90. See
    `itoh_theta_factor_integral` on the asymmetry.
    """
    major_radius = (ro + ri) / 2.0
    minor_radius = (ro - ri) / 2.0

    aspect_ratio = major_radius / minor_radius
    triangularity = (major_radius - rm) / minor_radius
    elongation = h / minor_radius

    return (
        4.933
        + 0.03728 * elongation
        + 0.06980 * triangularity
        - 3.551 * aspect_ratio
        + 0.7629 * aspect_ratio**2
        - 0.06298 * (theta1 / 90)
    )


def vv_stress_on_quench(
    *,
    h_coil,
    ri_coil,
    ro_coil,
    rm_coil,
    ccl_length_coil,
    theta1_coil,
    h_vv,
    ri_vv,
    ro_vv,
    rm_vv,
    theta1_vv,
    n_tf_coils,
    n_tf_coil_turns,
    s_rp,
    s_cc,
    taud,
    i_op,
    d_vv,
):
    """Tresca stress on the vacuum vessel during a TF quench, Pa.

    Ports `vv_stress_on_quench`, `process/models/tfcoil/superconducting.py:4963-5112`,
    line for line. The parameter names are PROCESS's own, lower-cased where it used
    `H_coil`/`H_vv` (`standards.md` has no capitals).
    """
    theta1_vv_rad = jnp.pi * (theta1_vv / 180.0)

    # Poloidal loop resistance (PLR) in ohms
    theta_vv = itoh_theta_factor_integral(
        ro_vv=ro_vv, ri_vv=ri_vv, rm_vv=rm_vv, h_vv=h_vv, theta1_vv=theta1_vv_rad
    )
    plr_coil = ((0.5 * ccl_length_coil) / (n_tf_coils * (s_cc + s_rp))) * 1e-6
    plr_vv = ((0.84 / d_vv) * theta_vv) * 1e-6

    # relevant self-inductances in henry (H)
    coil_structure_self_inductance = (
        (constants.RMU0 / jnp.pi)
        * h_coil
        * itoh_inductance_factor(
            h=h_coil, ri=ri_coil, ro=ro_coil, rm=rm_coil, theta1=theta1_coil
        )
    )
    vv_self_inductance = (
        (constants.RMU0 / jnp.pi)
        * h_vv
        * itoh_inductance_factor(h=h_vv, ri=ri_vv, ro=ro_vv, rm=rm_vv, theta1=theta1_vv)
    )

    lambda0 = 1 / taud
    lambda1 = plr_coil / coil_structure_self_inductance
    lambda2 = plr_vv / vv_self_inductance

    # approximate time at which the maximum force (and stress) occurs on the VV
    tmaxforce = jnp.log((lambda0 + lambda1) / (2 * lambda0)) / (lambda1 - lambda0)

    i0 = i_op * jnp.exp(-lambda0 * tmaxforce)
    i1 = (
        lambda0
        * n_tf_coils
        * n_tf_coil_turns
        * i_op
        * (
            (jnp.exp(-lambda1 * tmaxforce) - jnp.exp(-lambda0 * tmaxforce))
            / (lambda0 - lambda1)
        )
    )
    i2 = (lambda1 / lambda2) * i1

    a_vv = (ro_vv + ri_vv) / (ro_vv - ri_vv)
    b_vvi = (constants.RMU0 * (n_tf_coils * n_tf_coil_turns * i0 + i1 + (i2 / 2))) / (
        2 * jnp.pi * ri_vv
    )
    j_vvi = i2 / (2 * jnp.pi * d_vv * ri_vv)

    zeta = 1 + ((a_vv - 1) * jnp.log((a_vv + 1) / (a_vv - 1)) / (2 * a_vv))

    return zeta * b_vvi * j_vvi * ri_vv


def vv_stress_quench_from_build(
    *,
    z_tf_inside_half,
    dr_tf_inboard,
    r_tf_inboard_mid,
    r_tf_outboard_mid,
    r_tf_inboard_out,
    tfa_first_arc,
    z_plasma_xpoint_upper,
    dz_xpoint_divertor,
    dz_divertor,
    dz_shld_upper,
    dz_vv_upper,
    r_vv_inboard_out,
    dr_vv_outboard,
    dr_tf_outboard,
    dr_tf_shld_gap,
    dr_shld_thermal_outboard,
    dr_shld_vv_gap_outboard,
    len_tf_coil,
    theta1_coil,
    theta1_vv,
    n_tf_coils,
    n_tf_coil_turns,
    a_tf_coil_inboard_steel,
    a_tf_plasma_case,
    a_tf_coil_nose_case,
    dx_tf_side_case_average,
    t_tf_superconductor_quench,
    c_tf_coil,
    dr_vv_shells,
):
    """`.superconducting_tfcoil.vv_stress_quench` from the fields `run` reads.

    Ports `CICCSuperconductingTFCoil.vv_stress_on_quench`
    (`process/models/tfcoil/superconducting.py:1381-1452`) -- the *method*, i.e. the
    geometry prologue that turns twenty-odd build fields into the eighteen arguments of
    the module-level `vv_stress_on_quench` above, and the call itself.

    Two things in the prologue are carried across as written rather than tidied:

    - **`s_rp` is clipped at zero** (`:1445`, PROCESS's own `# TODO: value clipped due
      to #1883`). `jnp.clip` here, a real derivative kink, kept.
    - **`i_op` is `c_tf_coil / n_tf_coil_turns`** under PROCESS's own `# TODO: is this
      the correct current?` (`:1449`). Not second-guessed.

    `tfa_first_arc` is `.tfcoil.tfa[0]`, an array-element read (`naming_convention.md`
    § "Array elements") -- the first arc radius of the D-shaped coil, used both for the
    coil's `rm` and, scaled by the TF/VV plasma-facing radius ratio, for the vessel's.
    """
    h_coil = z_tf_inside_half + (dr_tf_inboard / 2)
    ri_coil = r_tf_inboard_mid
    ro_coil = r_tf_outboard_mid
    # `rm` is measured from the outside edge of the coil, because that is where the
    # radius of the first ellipse is measured from (`:1394-1395`).
    rm_coil = r_tf_inboard_out + tfa_first_arc

    h_vv = (
        z_plasma_xpoint_upper
        + dz_xpoint_divertor
        + dz_divertor
        + dz_shld_upper
        + (dz_vv_upper / 2)
    )
    # `ri`/`ro` for the VV do not consider the shield widths: the shield is assumed to
    # be on the plasma side of the VV (`:1405-1407`).
    ri_vv = r_vv_inboard_out - (dr_vv_outboard / 2)
    ro_vv = (
        r_tf_outboard_mid
        - (dr_tf_outboard / 2)
        - dr_tf_shld_gap
        - dr_shld_thermal_outboard
        - dr_shld_vv_gap_outboard
        - (dr_vv_outboard / 2)
    )

    # The first ellipse of the VV is assumed to be in the same proportion to that of
    # the coil as their plasma-facing radii (`:1417-1419`).
    tf_vv_frac = r_tf_inboard_out / r_vv_inboard_out
    rm_vv = r_vv_inboard_out + (tfa_first_arc * tf_vv_frac)

    return vv_stress_on_quench(
        h_coil=h_coil,
        ri_coil=ri_coil,
        ro_coil=ro_coil,
        rm_coil=rm_coil,
        ccl_length_coil=len_tf_coil,
        theta1_coil=theta1_coil,
        h_vv=h_vv,
        ri_vv=ri_vv,
        ro_vv=ro_vv,
        rm_vv=rm_vv,
        theta1_vv=theta1_vv,
        n_tf_coils=n_tf_coils,
        n_tf_coil_turns=n_tf_coil_turns,
        s_rp=jnp.clip(a_tf_coil_inboard_steel, 0.0, None),
        s_cc=a_tf_plasma_case + a_tf_coil_nose_case + 2.0 * dx_tf_side_case_average,
        taud=t_tf_superconductor_quench,
        i_op=c_tf_coil / n_tf_coil_turns,
        d_vv=dr_vv_shells,
    )


_STRAIN_LIMIT = 0.5e-2
"""The Nb3Sn critical-surface fits' region of applicability,
`process/models/tfcoil/superconducting.py:2911,3018,3053`. PROCESS logs an error and
substitutes `sign(strain) * 0.5e-2`; the substitution is a real value effect and is
ported, the log is not."""


def clip_nb3sn_strain(strain):
    """`sign(strain) * 0.5e-2` outside the fit's range, `strain` inside.

    `superconducting.py:2910-2916` (and the identical blocks at `:3017` and `:3052`).
    PROCESS's `np.sign(strain) * 0.5e-2` is reproduced exactly, including
    `sign(0.0) == 0.0` -- which cannot be reached, because `abs(0.0) > 0.5e-2` is false.
    """
    return jnp.where(
        jnp.abs(strain) > _STRAIN_LIMIT, jnp.sign(strain) * _STRAIN_LIMIT, strain
    )


def cicc_superconductor_properties(
    *,
    j_superconductor_critical,
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
):
    """The shared tail of `tf_cable_in_conduit_superconductor_properties`.

    `superconducting.py:3119-3136`, plus the `j_cables_critical`/
    `c_turn_cables_critical`/`j_crit_str_tf` triple every cable arm except Bi-2212
    repeats verbatim (`:2926-2939` and its four copies).

    Returns
    -------
    :
        `(j_tf_wp_critical, j_crit_str_tf, f_c_tf_turn_operating_critical,
        j_tf_coil_turn, j_superconductor, c_turn_cables_critical)` -- `run`'s own
        write order (`superconducting.py:2725-2742`), with the two that `run` reads off
        `TFSuperconductorLimits` unchanged in between.

    Notes
    -----
    PROCESS's `logger.error` on a non-positive `f_c_tf_turn_operating_critical`
    (`:3138-3153`) has no value effect and is dropped.
    """
    #  Scale for the copper area fraction of the cable
    j_cables_critical = j_superconductor_critical * (1.0 - f_a_tf_turn_cable_copper)

    #  Critical current in all the turn's cables
    c_turn_cables_critical = j_cables_critical * a_tf_turn_cable_space_effective

    # Strand critical current, for costing in $/kAm: superconducting filaments' jc
    # times one minus the strand copper fraction.
    j_crit_str_tf = j_superconductor_critical * (1.0 - f_a_tf_turn_cable_copper)

    # Critical current density in the winding pack. `a_tf_turn` is the area of the
    # entire jacketed conductor with insulation.
    j_tf_wp_critical = c_turn_cables_critical / a_tf_turn

    #  Ratio of operating to critical current
    f_c_tf_turn_operating_critical = c_tf_turn / c_turn_cables_critical

    #  Operating current density
    j_tf_coil_turn = c_tf_turn / a_tf_turn

    #  Actual current density in the superconductor, copper excluded
    j_superconductor = f_c_tf_turn_operating_critical * j_superconductor_critical

    return (
        j_tf_wp_critical,
        j_crit_str_tf,
        f_c_tf_turn_operating_critical,
        j_tf_coil_turn,
        j_superconductor,
        c_turn_cables_critical,
    )


def cicc_superconductor_properties_itersc(
    *,
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    strain,
    temp_tf_coolant_peak_field,
    b_c20max,
    temp_c0max,
):
    """The ITER-Nb3Sn arm (`i_tf_sc_mat == 1`), and `== 4` with its own `(bc20m, tc0m)`.

    `superconducting.py:2905-2939`. Value 4 is the *same* body with `bcritsc`/`tcritsc`
    substituted for the two literals (`:3013-3042`), which is why one function serves
    both and the literals live on the occupant rather than here.

    Returns `(*cicc_superconductor_properties(...), b_c20max, temp_c0max)` -- the two
    critical-surface constants are outputs of this node too
    (`.superconducting_tfcoil.b_tf_superconductor_critical_zero_temp_strain` and
    `.temp_tf_superconductor_critical_zero_field_strain`, `run` `:2733-2739`), because
    the temperature-margin node downstream reads them.
    """
    j_superconductor_critical, _, _ = itersc(
        temp_conductor=temp_tf_coolant_peak_field,
        b_conductor=b_tf_inboard_peak,
        strain=clip_nb3sn_strain(strain),
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )
    return (
        *cicc_superconductor_properties(
            j_superconductor_critical=j_superconductor_critical,
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
        ),
        j_superconductor_critical,
        b_c20max,
        temp_c0max,
    )


def cicc_superconductor_properties_wst_nb3sn(
    *,
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    strain,
    temp_tf_coolant_peak_field,
):
    """The WST-Nb3Sn arm (`i_tf_sc_mat == 5`). `superconducting.py:3046-3082`.

    Identical in shape to the ITER arm, down to the same `(32.97, 16.06)` literals and
    the same strain clip; only the fit differs.
    """
    b_c20max = 32.97
    temp_c0max = 16.06
    j_superconductor_critical, _, _ = western_superconducting_nb3sn(
        temp_conductor=temp_tf_coolant_peak_field,
        b_conductor=b_tf_inboard_peak,
        strain=clip_nb3sn_strain(strain),
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )
    return (
        *cicc_superconductor_properties(
            j_superconductor_critical=j_superconductor_critical,
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
        ),
        j_superconductor_critical,
        b_c20max,
        temp_c0max,
    )


def cicc_superconductor_properties_lubell_nbti(
    *,
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    temp_tf_coolant_peak_field,
):
    """The old-Lubell-NbTi arm (`i_tf_sc_mat == 3`). `superconducting.py:2949-3007`.

    **No strain read at all** -- `jcrit_nbti` has no strain argument, so this occupant
    genuinely reads one field fewer than its Nb3Sn siblings and is the reason
    `i_str_wp` is not a read of *every* member of this family.
    """
    b_c20max = 15.0
    temp_c0max = 9.3
    c0 = 1.0e10
    j_superconductor_critical, _ = jcrit_nbti(
        temp_conductor=temp_tf_coolant_peak_field,
        b_conductor=b_tf_inboard_peak,
        c0=c0,
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )
    return (
        *cicc_superconductor_properties(
            j_superconductor_critical=j_superconductor_critical,
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
        ),
        j_superconductor_critical,
        b_c20max,
        temp_c0max,
    )


def cicc_superconductor_properties_durham_nbti(
    *,
    a_tf_turn_cable_space_effective,
    a_tf_turn,
    b_tf_inboard_peak,
    f_a_tf_turn_cable_copper,
    c_tf_turn,
    strain,
    temp_tf_coolant_peak_field,
    b_crit_upper_nbti,
    t_crit_nbti,
):
    """The Durham Ginzburg-Landau NbTi arm (`i_tf_sc_mat == 7`).
    `superconducting.py:3086-3110`.

    **No strain clip on this arm** -- PROCESS applies the `0.5e-2` substitution only on
    the three Nb3Sn arms (`:2910`, `:3017`, `:3052`) and hands `gl_nbti` the raw strain.
    Transcribed as written.
    """
    j_superconductor_critical, _, _ = gl_nbti(
        temp_conductor=temp_tf_coolant_peak_field,
        b_conductor=b_tf_inboard_peak,
        strain=strain,
        b_c20max=b_crit_upper_nbti,
        t_c0=t_crit_nbti,
    )
    return (
        *cicc_superconductor_properties(
            j_superconductor_critical=j_superconductor_critical,
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
        ),
        j_superconductor_critical,
        b_crit_upper_nbti,
        t_crit_nbti,
    )


_MARGIN_SOLVE_TOL = 1.0e-6
"""`tol`/`rtol` PROCESS passes `scipy.optimize.newton`
(`superconducting.py:1273-1279`)."""


_MARGIN_SOLVE_MAX_ITER = 50
"""`maxiter`, same call."""


def _secant_step(*, p0, q0, p1, q1):
    """One scipy secant step, `scipy/optimize/_zeros_py.py`'s `newton` branch verbatim.

    scipy picks the algebraically-rearranged form whose denominator is the larger of the
    two, which is a genuine value difference in floating point and not a tidy-up -- so
    the branch is reproduced rather than collapsed to one formula. `q1 == q0` (a flat
    secant) makes scipy bisect; that case is reproduced too, guarded so neither arm's
    division poisons the other's tangent.
    """
    flat = q1 == q0
    ratio_01 = jnp.where(flat, 0.0, jnp.where(jnp.abs(q1) > jnp.abs(q0), q0 / q1, 0.0))
    ratio_10 = jnp.where(flat, 0.0, jnp.where(jnp.abs(q1) > jnp.abs(q0), 0.0, q1 / q0))
    secant = jnp.where(
        jnp.abs(q1) > jnp.abs(q0),
        (-ratio_01 * p1 + p0) / (1 - ratio_01),
        (-ratio_10 * p0 + p1) / (1 - ratio_10),
    )
    return jnp.where(flat, (p1 + p0) / 2.0, secant)


def solve_current_sharing_temperature(
    *,
    margin_fn,
    temp_start,
    max_iter=_MARGIN_SOLVE_MAX_ITER,
    tol=_MARGIN_SOLVE_TOL,
    rtol=_MARGIN_SOLVE_TOL,
):
    """The temperature at which `margin_fn(temperature)` vanishes.

    Ports `scipy.optimize.newton`'s **secant** branch as PROCESS calls it
    (`superconducting.py:1265-1281`): `fprime=None` with an explicit
    `x1 = 2 * temp_tf_coolant_peak_field`, so scipy never touches a derivative and the
    iteration is entirely determined by the two starting points, the step rule
    reproduced in `_secant_step`, and `np.isclose(p, p1, rtol, atol=tol)`.

    **Why the iteration is replicated rather than replaced by a better root finder.**
    `solve_duct_diameter` (`models/vacuum/vacuum.py`) took the opposite decision and
    tightened PROCESS's own tolerance, because there PROCESS's `0.01` is a coarse
    heuristic cutoff and the unit is Tier 2 anyway. Here the answer is read by
    constraint 36 and compared against PROCESS's, so the endpoint *is* the quantity
    under test: matching scipy's stopping rule is what makes the comparison a value
    test rather than a tolerance negotiation. The two decisions are the same rule
    applied to different situations -- reproduce what the answer depends on.

    Implemented as a fixed-trip `jax.lax.fori_loop` with a converged flag rather than a
    `while_loop`, because unlike `solve_duct_diameter` this **is** differentiated: the
    harness checks `temp_tf_superconductor_margin`'s gradient against PROCESS's own
    finite difference, and `while_loop` has no reverse rule. Once converged the carry
    freezes, so the extra trips cost accuracy nothing and the tangent is the tangent of
    the last real step.

    PROCESS passes `disp=True`, i.e. scipy *raises* if the 50 iterations run out. A
    traced loop cannot raise; it returns whatever it reached, the same position
    `solve_duct_diameter` records for its own non-convergence log.

    Parameters
    ----------
    margin_fn :
        `temperature -> j_critical(temperature) - j_operating`, i.e. PROCESS's
        `superconductor_current_density_margin` with everything but the temperature
        already bound.
    temp_start :
        `p0`. PROCESS passes `temp_tf_coolant_peak_field`; `p1` is `2 * p0`, scipy's
        `x1`.
    """
    p0 = jnp.asarray(temp_start)
    p1 = 2.0 * p0
    q0 = margin_fn(p0)
    q1 = margin_fn(p1)
    # scipy orders the pair so `p1` carries the smaller residual (`_zeros_py.py`'s
    # "If provided, sort the interval").
    swap = jnp.abs(q1) < jnp.abs(q0)
    p0, p1 = jnp.where(swap, p1, p0), jnp.where(swap, p0, p1)
    q0, q1 = jnp.where(swap, q1, q0), jnp.where(swap, q0, q1)

    def body(_i, carry):
        p0, q0, p1, q1, answer, done = carry
        p = _secant_step(p0=p0, q0=q0, p1=p1, q1=q1)
        # `q1 == q0` is scipy's bisection escape, and it returns immediately -- so it
        # counts as convergence here for exactly the same reason.
        converged = (q1 == q0) | jnp.isclose(p, p1, rtol=rtol, atol=tol)
        answer = jnp.where(done, answer, p)
        done |= converged
        # **Once done, the carry collapses to the flat state `(answer, q, answer, q)`,
        # rather than being frozen by masking the update.** Both keep the value; only
        # this one keeps the *tangent*. A frozen-by-mask carry still evaluates
        # `_secant_step` on the last real pair every remaining trip, and a secant step
        # whose denominator has gone to zero is `inf` -- which `jnp.where` discards in
        # value and multiplies by zero in the JVP, giving `nan`. The flat state has
        # `q0 == q1` by construction, so `_secant_step` takes its bisection arm and
        # returns `(answer + answer) / 2` exactly: finite, and with `answer`'s own
        # derivative. Measured: `i_tf_sc_mat = 4` produced a `nan` gradient under the
        # masked form and the correct one under this.
        next_p0 = jnp.where(done, answer, p1)
        next_p1 = jnp.where(done, answer, p)
        return (
            next_p0,
            q1,
            next_p1,
            jnp.where(done, q1, margin_fn(next_p1)),
            answer,
            done,
        )

    _, _, _, _, answer, _ = jax.lax.fori_loop(
        0, max_iter, body, (p0, q0, p1, q1, p1, jnp.asarray(False))
    )
    return answer


def _temperature_margin(*, margin_fn, temp_tf_coolant_peak_field):
    """`t_zero_margin - temp_tf_coolant_peak_field`. `superconducting.py:1281`."""
    return (
        solve_current_sharing_temperature(
            margin_fn=margin_fn, temp_start=temp_tf_coolant_peak_field
        )
        - temp_tf_coolant_peak_field
    )


def temperature_margin_itersc(
    *,
    j_superconductor,
    b_tf_inboard_peak,
    strain,
    b_c20max,
    temp_c0max,
    temp_tf_coolant_peak_field,
):
    """TF superconductor temperature margin, ITER-Nb3Sn fit (`i_tf_sc_mat` 1 and 4).

    `calculate_superconductor_temperature_margin` (`superconducting.py:1174-1291`) with
    `superconductor_current_density_margin`'s branch 1
    (`process/models/superconductors.py:1259`) as the residual.

    **The strain is *not* clipped here**, unlike in
    `cicc_superconductor_properties_itersc`. PROCESS's `run` reads `str_wp` /
    `str_tf_con_res` afresh at `:2744-2747` and hands the raw value to this function;
    the `0.5e-2` substitution happens inside the *properties* function, on its own local
    copy, and never reaches back. Transcribed as written -- the two functions genuinely
    evaluate the same fit at two different strains when `|strain| > 0.5e-2`, which is a
    defect worth recording rather than smoothing (D4, `superconducting.md`).
    """

    def margin_fn(temperature):
        return (
            itersc(
                temp_conductor=temperature,
                b_conductor=b_tf_inboard_peak,
                strain=strain,
                b_c20max=b_c20max,
                temp_c0max=temp_c0max,
            )[0]
            - j_superconductor
        )

    return _temperature_margin(
        margin_fn=margin_fn, temp_tf_coolant_peak_field=temp_tf_coolant_peak_field
    )


def temperature_margin_wst_nb3sn(
    *,
    j_superconductor,
    b_tf_inboard_peak,
    strain,
    b_c20max,
    temp_c0max,
    temp_tf_coolant_peak_field,
):
    """Temperature margin, WST-Nb3Sn fit (`i_tf_sc_mat == 5`).
    `process/models/superconductors.py:1263-1265`. Same shape as the ITER arm.
    """

    def margin_fn(temperature):
        return (
            western_superconducting_nb3sn(
                temp_conductor=temperature,
                b_conductor=b_tf_inboard_peak,
                strain=strain,
                b_c20max=b_c20max,
                temp_c0max=temp_c0max,
            )[0]
            - j_superconductor
        )

    return _temperature_margin(
        margin_fn=margin_fn, temp_tf_coolant_peak_field=temp_tf_coolant_peak_field
    )


def temperature_margin_lubell_nbti(
    *,
    j_superconductor,
    b_tf_inboard_peak,
    b_c20max,
    temp_c0max,
    temp_tf_coolant_peak_field,
):
    """Temperature margin, old-Lubell-NbTi fit (`i_tf_sc_mat == 3`).

    `process/models/superconductors.py:1260`. **`c0 = 1.0e10` is a literal `run` passes
    (`superconducting.py:1258`), not a read**, and this is the one branch of
    `superconductor_current_density_margin` that consumes it -- which is why `run`
    builds a ten-element `arguments` tuple for material 3 and a nine-element one for
    every other (`:1236-1256`). No strain, for the same reason the properties arm has
    none.
    """

    def margin_fn(temperature):
        return (
            jcrit_nbti(
                temp_conductor=temperature,
                b_conductor=b_tf_inboard_peak,
                c0=1.0e10,
                b_c20max=b_c20max,
                temp_c0max=temp_c0max,
            )[0]
            - j_superconductor
        )

    return _temperature_margin(
        margin_fn=margin_fn, temp_tf_coolant_peak_field=temp_tf_coolant_peak_field
    )


def temperature_margin_durham_nbti(
    *,
    j_superconductor,
    b_tf_inboard_peak,
    strain,
    b_c20max,
    temp_c0max,
    temp_tf_coolant_peak_field,
):
    """Temperature margin, Durham Ginzburg-Landau NbTi fit (`i_tf_sc_mat == 7`).
    `process/models/superconductors.py:1266-1268`.
    """

    def margin_fn(temperature):
        return (
            gl_nbti(
                temp_conductor=temperature,
                b_conductor=b_tf_inboard_peak,
                strain=strain,
                b_c20max=b_c20max,
                t_c0=temp_c0max,
            )[0]
            - j_superconductor
        )

    return _temperature_margin(
        margin_fn=margin_fn, temp_tf_coolant_peak_field=temp_tf_coolant_peak_field
    )


def calculate_vv_stress_on_quench(
    z_tf_inside_half,
    dr_tf_inboard,
    r_tf_inboard_mid,
    r_tf_outboard_mid,
    r_tf_inboard_out,
    tfa,
    z_plasma_xpoint_upper,
    dz_xpoint_divertor,
    dz_divertor,
    dz_shld_upper,
    dz_vv_upper,
    r_vv_inboard_out,
    dr_vv_outboard,
    dr_tf_outboard,
    dr_tf_shld_gap,
    dr_shld_thermal_outboard,
    dr_shld_vv_gap_outboard,
    len_tf_coil,
    theta1_coil,
    theta1_vv,
    n_tf_coils,
    n_tf_coil_turns,
    a_tf_coil_inboard_steel,
    a_tf_plasma_case,
    a_tf_coil_nose_case,
    dx_tf_side_case_average,
    t_tf_superconductor_quench,
    c_tf_coil,
    dr_vv_shells,
):
    """`vv_stress_quench_from_build`'s arm on the whole `tfa` vector -- `tfa[0]` is
    `tf_coil_shape`'s first arc, sliced here rather than at the declaration (see
    `VvStressOnQuench`'s own docstring for why `tfa` must be read whole, not as a
    `FromExactly` element).
    """
    return vv_stress_quench_from_build(
        z_tf_inside_half=z_tf_inside_half,
        dr_tf_inboard=dr_tf_inboard,
        r_tf_inboard_mid=r_tf_inboard_mid,
        r_tf_outboard_mid=r_tf_outboard_mid,
        r_tf_inboard_out=r_tf_inboard_out,
        tfa_first_arc=tfa[0],
        z_plasma_xpoint_upper=z_plasma_xpoint_upper,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_shld_upper=dz_shld_upper,
        dz_vv_upper=dz_vv_upper,
        r_vv_inboard_out=r_vv_inboard_out,
        dr_vv_outboard=dr_vv_outboard,
        dr_tf_outboard=dr_tf_outboard,
        dr_tf_shld_gap=dr_tf_shld_gap,
        dr_shld_thermal_outboard=dr_shld_thermal_outboard,
        dr_shld_vv_gap_outboard=dr_shld_vv_gap_outboard,
        len_tf_coil=len_tf_coil,
        theta1_coil=theta1_coil,
        theta1_vv=theta1_vv,
        n_tf_coils=n_tf_coils,
        n_tf_coil_turns=n_tf_coil_turns,
        a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
        a_tf_plasma_case=a_tf_plasma_case,
        a_tf_coil_nose_case=a_tf_coil_nose_case,
        dx_tf_side_case_average=dx_tf_side_case_average,
        t_tf_superconductor_quench=t_tf_superconductor_quench,
        c_tf_coil=c_tf_coil,
        dr_vv_shells=dr_vv_shells,
    )


def calculate_temperature_margin_with_strain(
    fit,
    j_tf_superconductor,
    b_tf_inboard_peak_with_ripple,
    str_wp,
    b_tf_superconductor_critical_zero_temp_strain,
    temp_tf_superconductor_critical_zero_field_strain,
    tftmp,
):
    """The strained arms' shared temperature-margin fit, doubled onto both
    `.tfcoil.temp_tf_superconductor_margin` and `.tfcoil.temp_margin` -- one number to
    two `VarPath`s, as PROCESS's own duplicate write does (see
    `_TemperatureMarginWithStrain`'s own docstring). `fit` is the material arm's own
    `type(self).fit` `staticmethod` (`temperature_margin_itersc` or
    `temperature_margin_wst_nb3sn`).
    """
    margin = fit(
        j_superconductor=j_tf_superconductor,
        b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
        strain=str_wp,
        b_c20max=b_tf_superconductor_critical_zero_temp_strain,
        temp_c0max=temp_tf_superconductor_critical_zero_field_strain,
        temp_tf_coolant_peak_field=tftmp,
    )
    return margin, margin


def calculate_old_lubell_nbti_temperature_margin(
    j_tf_superconductor,
    b_tf_inboard_peak_with_ripple,
    b_tf_superconductor_critical_zero_temp_strain,
    temp_tf_superconductor_critical_zero_field_strain,
    tftmp,
):
    """`i_tf_sc_mat == 3`'s temperature margin, doubled onto both
    `.tfcoil.temp_tf_superconductor_margin` and `.tfcoil.temp_margin` -- one number to
    two `VarPath`s, as PROCESS's own duplicate write does.
    """
    margin = temperature_margin_lubell_nbti(
        j_superconductor=j_tf_superconductor,
        b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
        b_c20max=b_tf_superconductor_critical_zero_temp_strain,
        temp_c0max=temp_tf_superconductor_critical_zero_field_strain,
        temp_tf_coolant_peak_field=tftmp,
    )
    return margin, margin
