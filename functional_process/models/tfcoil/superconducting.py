"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CICCSuperconductingTFCoil` and the `SuperconductingTFCoil` layer above it.

Audit record: `functional_process/_audit/units/models/tfcoil/superconducting.md`.
The base-class half is `functional_process/models/tfcoil/base.py`; the quench half is
`functional_process/models/tfcoil/quench.py` (read that one for the CoolProp boundary).

**Scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten boundary
reads** (`_audit/tokamak_boundary.md`). In scope and ported here:
`superconducting_tf_wp_geometry`, `superconducting_tf_case_geometry` (split in two, see
below), `tf_wp_currents`, `peak_b_tf_inboard_with_ripple`,
`tf_cable_in_conduit_averaged_turn_geometry`,
`tf_cable_in_conduit_integer_turn_geometry` (2026-08-27, after
`low_aspect_ratio_DEMO`'s silent mis-assembly -- see `CiccIntegerTurnGeometry`),
`tf_cicc_inboard_areas_and_fractions`,
`superconducting_tf_coil_areas_and_masses`, and `run`'s inline `.tfcoil.a_tf_turn`
(`process/models/tfcoil/superconducting.py:2700-2704`).

Deliberately **out** of scope, with reasons:

- `tf_cable_in_conduit_superconductor_properties`, `calculate_superconductor_temperature_margin`
  (`superconducting.py:1174,2806`) -- critical-current physics. Nothing on the boundary
  reads it, and the critical-surface fits themselves are **already ported and shared**:
  `functional_process/models/physics/superconductors.py`, one class per `i_tf_sc_mat`.
  Whoever wires those two functions reuses that module rather than re-porting a fit.
- `stresscl`/`run_and_output_stress`/`vv_stress_on_quench` -- stresses, none on the
  boundary; `stresscl` is `numba.njit` and ~2400 lines of `base.py`.
- `calculate_cable_in_conduit_strand_count`,
  `calculate_cable_in_conduit_superconductor_length` -- write only
  `.superconducting_tfcoil.n_tf_turn_superconducting_cables` /
  `len_tf_coil_superconductor` / `len_tf_superconductor_total`, which nothing on the
  boundary reads.
- every `output_*` method -- reporting.
- the whole `CROCOSuperconductingTFCoil` branch -- selected by
  `.superconducting_tfcoil.i_tf_turn_type == 2` (`process/core/caller.py:307-313`),
  where the default and the reference run take `1` (cable in conduit,
  `superconducting_tf_coil_variables.py:194`).

## Two findings that contradict `tokamak_boundary.md`, recorded not smoothed over

1. **`.tfcoil.c_tf_turn` is a run *input* on `large_tokamak_eval`, not an output of this
   slot.** `tf_cable_in_conduit_averaged_turn_geometry` computes `c_tf_turn` only on its
   first two arms (`superconducting.py:3305-3323`, `i_dx_tf_turn_general_input` or
   `i_dx_tf_turn_cable_space_general_input`); on the third -- both `False`, which is
   both PROCESS's default (`tfcoil_variables.py:108,127`) and the reference file's state
   -- the parameter is returned unchanged (`superconducting.py:3411`) and `run` writes
   it back to the field it came from (`superconducting.py:2372`). The reference input
   sets it explicitly (`tests/regression/input_files/large_tokamak_eval.IN.DAT:371`,
   `c_tf_turn = 85462.675...`) and it is iteration variable 60
   (`process/core/solver/iteration_variables.py:78`). So on this configuration it is an
   optimiser unknown that no node produces, and `CiccAveragedTurnGeometryFromCurrentPerTurn`
   below **reads it and does not own it** -- conditional ownership, the shape
   `models/power/thermal_cryo.py` records for `p_fw_blkt_coolant_pump_mw`. Registering
   it as a boundary `input` is the correct outcome, not a gap.

2. **`.tfcoil.v_tf_coil_dump_quench_kv` does not reach CoolProp**, so the schedule note
   in `tokamak_boundary.md` § this slot ("waits on `next_steps.md` §5's unresolved
   wrapping policy") does not apply to it. See `quench.py`'s module docstring for the
   `file:line` evidence and the measured call surface.

## Switch splits in this file

| PROCESS function | switch | occupants written | UNPORTED |
|---|---|---|---|
| `superconducting_tf_wp_geometry` | `i_tf_wp_geom` | all three (0/1/2) | -- |
| `superconducting_tf_case_geometry`, front/nose | `i_tf_case_geom` | both (0/1) | -- |
| `superconducting_tf_case_geometry`, sidewall | `i_tf_wp_geom` | all three (0/1/2) | -- |
| `peak_b_tf_inboard_with_ripple` | `round(n_tf_coils)` | 16, 18, 20, other | -- |
| `tf_cable_in_conduit_averaged_turn_geometry` | `i_dx_tf_turn_general_input`, `i_dx_tf_turn_cable_space_general_input` | the both-`False` arm | the other two |
| `superconducting_tf_coil_areas_and_masses` | `itart` | both (0/1) | -- |
| `run`'s turn-geometry choice | `i_tf_turns_integer` | both (0 non-integer / 1 integer) | -- |

`i_tf_wp_geom` is `-1` (`UNSET`) by default and `process/core/init.py:977-989` resolves
it before any model runs: `DOUBLE_RECTANGULAR (1)` when `i_tf_turns_integer == 0`,
`RECTANGULAR (0)` when it is `1`. `large_tokamak_eval` sets neither, so the live arm is
`i_tf_wp_geom == 1`.

`round(n_tf_coils)` in `peak_b_tf_inboard_with_ripple` is treated as a switch here even
though `n_tf_coils` is a plain number elsewhere (and stays an ordinary read on every
occupant, because the same formula also uses it continuously): the branches select
different *fit coefficients* **and** different reads -- the `else` arm reads nothing but
`b_tf_inboard_peak_symmetric` -- so the split default applies on its face.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import build, fwbs, superconducting_tfcoil, tfcoil
from process.core import constants

I_TF_SC_MAT_ITER_NB3SN = 1
"""`i_tf_sc_mat`'s ITER-Nb3Sn value and PROCESS's own default
(`process/data_structure/tfcoil_variables.py:246`);
`tests/regression/input_files/large_tokamak_eval.IN.DAT:374` sets the same `1`, so the
material density this port reads is `dcond[0] == 6080.0` (`tfcoil_variables.py:158`).

Same treatment as `models/stellarator/coils/mass.py`'s `CoilsMass`: `dcond` is a real
nine-element `DataStructure` field, so the material lookup is an array-element `VarPath`
(`_audit/naming_convention.md` § "Array elements") rather than an invented mint, and the
index is fixed at class-definition time by the occupant. A different `i_tf_sc_mat` needs
a sibling class overriding only that one `FromExactly`.
"""

_RIPPLE_FIT_COEFFICIENTS = {
    16: (0.28101, 1.8481, -0.88159, 0.93834),
    18: (0.29153, 1.81600, -0.84178, 0.90426),
    20: (0.29853, 1.82130, -0.85031, 0.89808),
}
"""M. Kovari's MAGINT fits, `process/models/tfcoil/superconducting.py:1502-1516`.

Keyed on `round(n_tf_coils)`; every other coil count takes the flat 9 % ripple
allowance instead (`superconducting.py:1519`).
"""

_RIPPLE_FLAT_ALLOWANCE = 1.09
"""The `else` arm's ripple factor, `process/models/tfcoil/superconducting.py:1519`."""


# ---------------------------------------------------------------------------
# `superconducting_tf_wp_geometry` -- `superconducting.py:1558-1793`
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `superconducting_tf_case_geometry`, split in two -- `superconducting.py:1795-1956`
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `tf_wp_currents` -- `superconducting.py:1958-1970`
# ---------------------------------------------------------------------------


def tf_wp_currents(*, c_tf_total, n_tf_coils, a_tf_wp_no_insulation):
    """Winding-pack engineering current density (A/m2), floored at 1.

    Ports `tf_wp_currents`, `process/models/tfcoil/superconducting.py:1958-1970`.
    PROCESS takes the whole `DataStructure` and mutates it in place; the arithmetic is
    three reads and one write, promoted here to an ordinary signature.
    `max` -> `jnp.maximum`, since `c_tf_total` is differentiable.
    """
    return jnp.maximum(1.0, c_tf_total / (n_tf_coils * a_tf_wp_no_insulation))


# ---------------------------------------------------------------------------
# `peak_b_tf_inboard_with_ripple` -- `superconducting.py:1454-1556`
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `tf_cable_in_conduit_averaged_turn_geometry` -- `superconducting.py:3238-3420`
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `tf_cicc_inboard_areas_and_fractions` -- `superconducting.py:3599-3670`
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `run`'s inline `.tfcoil.a_tf_turn` -- `superconducting.py:2700-2704`
# ---------------------------------------------------------------------------


def calculate_a_tf_turn(*, c_tf_total, j_tf_wp, n_tf_coils, n_tf_coil_turns):
    """Cross-sectional area per turn (m2). `superconducting.py:2700-2704`, inline in
    `run`."""
    return c_tf_total / (j_tf_wp * n_tf_coils * n_tf_coil_turns)


# ---------------------------------------------------------------------------
# `superconducting_tf_coil_areas_and_masses` -- `superconducting.py:1972-2093`
# ---------------------------------------------------------------------------


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
        `models/stellarator/coils/mass.py`; see `I_TF_SC_MAT_ITER_NB3SN`.

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


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------

_WP_GEOMETRY_OUTPUTS = (
    "r_tf_wp_inboard_inner",
    "r_tf_wp_inboard_outer",
    "r_tf_wp_inboard_centre",
    "dx_tf_wp_toroidal_min",
    "dr_tf_wp_no_insulation",
    "dx_tf_wp_primary_toroidal",
    "dx_tf_wp_secondary_toroidal",
    "dx_tf_wp_toroidal_average",
    "a_tf_wp_with_insulation",
    "a_tf_wp_no_insulation",
    "a_tf_wp_ground_insulation",
)
"""Documentation only -- the declaration order every `SuperconductingTfWpGeometry`
occupant repeats. Kept as a name so the three classes below can be checked against one
list rather than against each other."""


class SuperconductingTfWpGeometry(ExplicitFunction):
    """The family that owns the inboard winding-pack geometry. `i_tf_wp_geom` decides it.

    All three arms read the same seven fields and differ only in the toroidal-thickness
    formulas, so this is one of the cases the binding policy covers explicitly: identical
    reads-sets still get one class per value.
    """


class SuperconductingTfWpGeometryRectangular(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 0` (rectangular)."""

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_rectangular(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class SuperconductingTfWpGeometryDoubleRectangular(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 1` (double rectangular) -- `large_tokamak_eval`'s arm.

    Reached by `process/core/init.py:980-984`: the file sets neither `i_tf_wp_geom` nor
    `i_tf_turns_integer`, so `UNSET` plus `NON_INTEGER` resolves to this.
    """

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_double_rectangular(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class SuperconductingTfWpGeometryTrapezoidal(SuperconductingTfWpGeometry):
    """`i_tf_wp_geom == 2` (trapezoidal)."""

    r_tf_wp_inboard_inner = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_outer = OutputInto(superconducting_tfcoil)
    r_tf_wp_inboard_centre = OutputInto(superconducting_tfcoil)
    dx_tf_wp_toroidal_min = OutputInto(superconducting_tfcoil)
    dr_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_toroidal_average = OutputInto(superconducting_tfcoil)
    a_tf_wp_with_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_no_insulation = OutputInto(superconducting_tfcoil)
    a_tf_wp_ground_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return superconducting_tf_wp_geometry_trapezoidal(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class TfCaseAreas(ExplicitFunction):
    """The family that owns the four TF case areas. `i_tf_case_geom` decides it."""


class TfCaseAreasCircularFront(TfCaseAreas):
    """`i_tf_case_geom == 0` (circular front case) -- the reference arm."""

    a_tf_coil_inboard_case = OutputInto(tfcoil)
    a_tf_coil_outboard_case = OutputInto(tfcoil)
    a_tf_plasma_case = OutputInto(superconducting_tfcoil)
    a_tf_coil_nose_case = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_inboard_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_leg_outboard=From(tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        r_tf_inboard_out=From(build),
        tan_theta_coil=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_inboard_in=From(build),
    ):
        return tf_case_areas_circular_front(
            a_tf_inboard_total=a_tf_inboard_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_leg_outboard=a_tf_leg_outboard,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            r_tf_inboard_out=r_tf_inboard_out,
            tan_theta_coil=tan_theta_coil,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_inboard_in=r_tf_inboard_in,
        )


class TfCaseAreasStraightFront(TfCaseAreas):
    """`i_tf_case_geom == 1` (straight front case). Reads `dr_tf_plasma_case`."""

    a_tf_coil_inboard_case = OutputInto(tfcoil)
    a_tf_coil_outboard_case = OutputInto(tfcoil)
    a_tf_plasma_case = OutputInto(superconducting_tfcoil)
    a_tf_coil_nose_case = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_inboard_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_leg_outboard=From(tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_inboard_in=From(build),
    ):
        return tf_case_areas_straight_front(
            a_tf_inboard_total=a_tf_inboard_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_leg_outboard=a_tf_leg_outboard,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            tan_theta_coil=tan_theta_coil,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            dr_tf_plasma_case=dr_tf_plasma_case,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_inboard_in=r_tf_inboard_in,
        )


class DxTfSideCase(ExplicitFunction):
    """The family that owns the sidewall case thicknesses. `i_tf_wp_geom` decides it."""


class DxTfSideCaseRectangular(DxTfSideCase):
    """`i_tf_wp_geom == 0`."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_side_case_min=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return dx_tf_side_case_rectangular(
            dx_tf_side_case_min=dx_tf_side_case_min,
            tan_theta_coil=tan_theta_coil,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        )


class DxTfSideCaseDoubleRectangular(DxTfSideCase):
    """`i_tf_wp_geom == 1` -- the reference arm."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_side_case_min=From(tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return dx_tf_side_case_double_rectangular(
            dx_tf_side_case_min=dx_tf_side_case_min,
            tan_theta_coil=tan_theta_coil,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        )


class DxTfSideCaseTrapezoidal(DxTfSideCase):
    """`i_tf_wp_geom == 2`: constant thickness, one read."""

    dx_tf_side_case_average = OutputInto(superconducting_tfcoil)
    dx_tf_side_case_peak = OutputInto(tfcoil)

    def __call__(self, dx_tf_side_case_min=From(tfcoil)):
        return dx_tf_side_case_trapezoidal(dx_tf_side_case_min=dx_tf_side_case_min)


class TfWpCurrents(ExplicitFunction):
    """cottax node: `tf_wp_currents`. Owns `.tfcoil.j_tf_wp`.

    The stellarator port's `models/stellarator/namespace.py:227-259` records a long
    argument about whether `.tfcoil.j_tf_wp` needs a `FixedPointFunction` there. On the
    tokamak path it does not, and the reason is local: PROCESS's own body
    (`superconducting.py:1963-1970`) reads `c_tf_total`, `n_tf_coils` and
    `a_tf_wp_no_insulation` and nothing else -- the entering `j_tf_wp` is never
    consulted, so there is no self-reference to cut.
    """

    j_tf_wp = OutputInto(tfcoil)

    def __call__(
        self,
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
    ):
        return tf_wp_currents(
            c_tf_total=c_tf_total,
            n_tf_coils=n_tf_coils,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        )


class PeakBTfInboardWithRipple(ExplicitFunction):
    """The family that owns `.tfcoil.b_tf_inboard_peak_with_ripple`.

    `round(n_tf_coils)` decides it: three MAGINT-fit occupants and one flat-allowance
    fallback that owns **one** output where the others own four.
    """


class _PeakBTfInboardWithRippleKovari(PeakBTfInboardWithRipple):
    """Shared declaration for the three fitted coil counts; `coefficients` differs."""

    coefficients = ()

    tf_fit_t = OutputInto(superconducting_tfcoil)
    tf_fit_z = OutputInto(superconducting_tfcoil)
    f_b_tf_inboard_peak_ripple_symmetric = OutputInto(superconducting_tfcoil)
    b_tf_inboard_peak_with_ripple = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        dx_tf_wp_primary_toroidal=From(tfcoil),
        dr_tf_wp_no_insulation=From(superconducting_tfcoil),
        r_tf_wp_inboard_centre=From(superconducting_tfcoil),
        b_tf_inboard_peak_symmetric=From(tfcoil),
    ):
        return peak_b_tf_inboard_with_ripple_kovari(
            n_tf_coils=n_tf_coils,
            dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
            dr_tf_wp_no_insulation=dr_tf_wp_no_insulation,
            r_tf_wp_inboard_centre=r_tf_wp_inboard_centre,
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
            coefficients=type(self).coefficients,
        )


class PeakBTfInboardWithRipple16Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 16` -- `large_tokamak_eval.IN.DAT:377` sets exactly 16."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[16]


class PeakBTfInboardWithRipple18Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 18`."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[18]


class PeakBTfInboardWithRipple20Coils(_PeakBTfInboardWithRippleKovari):
    """`round(n_tf_coils) == 20`."""

    coefficients = _RIPPLE_FIT_COEFFICIENTS[20]


class PeakBTfInboardWithRippleFlatAllowance(PeakBTfInboardWithRipple):
    """Any other coil count: `1.09 * b_tf_inboard_peak_symmetric`, one read, one output.

    PROCESS returns at `superconducting.py:1519` before `tf_fit_t`, `tf_fit_z` and
    `f_b_tf_inboard_peak_ripple_symmetric` are assigned, so this occupant genuinely owns
    fewer variables than its siblings -- conditional ownership again, and a case a
    static kwarg could not have expressed at all.
    """

    b_tf_inboard_peak_with_ripple = OutputInto(tfcoil)

    def __call__(self, b_tf_inboard_peak_symmetric=From(tfcoil)):
        return peak_b_tf_inboard_with_ripple_flat(
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric
        )


class CiccTurnGeometry(ExplicitFunction):
    """The family that owns the CICC winding-pack turn geometry.

    `.tfcoil.i_tf_turns_integer` decides it first (`run`'s own branch,
    `superconducting.py:2343-2439`): `0` is the averaged sub-family below, `1` is
    `CiccIntegerTurnGeometry`. The arms cannot share a node even in principle, because
    they disagree about **ownership**, not just formulae: the integer arm owns
    `.tfcoil.c_tf_turn` (current over a fixed turn count) where the reference averaged
    arm reads it as an optimiser unknown, and it owns four per-direction conductor and
    cable-space dimensions the averaged arm never writes.
    """


class CiccAveragedTurnGeometry(CiccTurnGeometry):
    """The averaged (`i_tf_turns_integer == 0`) sub-family.

    Two further booleans decide it (`i_dx_tf_turn_general_input`,
    `i_dx_tf_turn_cable_space_general_input`), and the three arms differ in which of
    `.tfcoil.c_tf_turn` / `.tfcoil.dx_tf_turn_general` /
    `.tfcoil.dx_tf_turn_cable_space_general` they read and which they own. That
    ownership difference is why they cannot share one node even in principle.
    """


class CiccAveragedTurnGeometryFromCurrentPerTurn(CiccAveragedTurnGeometry):
    """Both input flags `False` -- PROCESS's default and `large_tokamak_eval`'s arm.

    **Reads `.tfcoil.c_tf_turn`; does not own it.** See the module docstring, finding 1.
    """

    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)
    a_tf_turn_insulation = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    radius_tf_turn_cable_space_corners = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        j_tf_wp=From(tfcoil),
        c_tf_turn=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        layer_ins=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
    ):
        return cicc_averaged_turn_geometry_from_current_per_turn(
            j_tf_wp=j_tf_wp,
            c_tf_turn=c_tf_turn,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            layer_ins=layer_ins,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        )


class CiccIntegerTurnGeometry(CiccTurnGeometry):
    """`i_tf_turns_integer == 1` -- rectangular turns on a fixed layers x pancakes grid.

    **Owns `.tfcoil.c_tf_turn`** where the averaged reference arm reads it: with the
    turn count fixed by `n_tf_wp_layers * n_tf_wp_pancakes`, the current per turn is
    determined, so iteration variable 60 has a producer on this configuration and is
    not a boundary input. It also owns the four per-direction conductor/cable-space
    dimensions (`dr_tf_turn_conduit_full`, `dx_tf_turn_conduit_full_toroidal`,
    `dr_tf_turn_cable_space`, `dx_tf_turn_cable_space`) that only exist when the turn
    is allowed to be rectangular -- conditional ownership, the same reason
    `SuperconductingTfCoilAreasAndMasses` is a family.

    `n_tf_wp_layers` and `n_tf_wp_pancakes` are **reads, not switches**: they enter the
    formulae continuously (as divisors of the winding-pack extents), so they are plain
    boundary inputs of this node, exactly as `n_tf_coils` is elsewhere. What *is* a
    switch is `i_tf_turns_integer` itself, answered by `indat.py`.
    """

    radius_tf_turn_cable_space_corners = OutputInto(superconducting_tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)
    a_tf_turn_insulation = OutputInto(tfcoil)
    c_tf_turn = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dr_tf_turn_conduit_full = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_toroidal = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    dr_tf_turn_cable_space = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space = OutputInto(superconducting_tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_wp_with_insulation=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        n_tf_wp_layers=From(tfcoil),
        dx_tf_wp_toroidal_min=From(superconducting_tfcoil),
        n_tf_wp_pancakes=From(tfcoil),
        c_tf_coil=From(superconducting_tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
    ):
        return cicc_integer_turn_geometry(
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            n_tf_wp_layers=n_tf_wp_layers,
            dx_tf_wp_toroidal_min=dx_tf_wp_toroidal_min,
            n_tf_wp_pancakes=n_tf_wp_pancakes,
            c_tf_coil=c_tf_coil,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        )


class CiccInboardAreasAndFractions(ExplicitFunction):
    """cottax node: `tf_cicc_inboard_areas_and_fractions`. No switch.

    PROCESS logs an error when any of eight outputs comes out non-positive
    (`superconducting.py:2494-2516`); logging only, dropped here.
    """

    a_tf_wp_coolant_channels = OutputInto(tfcoil)
    a_tf_wp_conductor = OutputInto(tfcoil)
    a_tf_wp_extra_void = OutputInto(tfcoil)
    a_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    a_tf_wp_steel = OutputInto(tfcoil)
    a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        n_tf_coil_turns=From(tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        a_tf_turn_insulation=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_wp_ground_insulation=From(superconducting_tfcoil),
    ):
        return tf_cicc_inboard_areas_and_fractions(
            n_tf_coil_turns=n_tf_coil_turns,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_insulation=a_tf_turn_insulation,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_coils=n_tf_coils,
            a_tf_inboard_total=a_tf_inboard_total,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_wp_ground_insulation=a_tf_wp_ground_insulation,
        )


class TfTurnArea(ExplicitFunction):
    """cottax node: `run`'s inline `.tfcoil.a_tf_turn` (`superconducting.py:2700`)."""

    a_tf_turn = OutputInto(tfcoil)

    def __call__(
        self,
        c_tf_total=From(tfcoil),
        j_tf_wp=From(tfcoil),
        n_tf_coils=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
    ):
        return calculate_a_tf_turn(
            c_tf_total=c_tf_total,
            j_tf_wp=j_tf_wp,
            n_tf_coils=n_tf_coils,
            n_tf_coil_turns=n_tf_coil_turns,
        )


class SuperconductingTfCoilAreasAndMasses(ExplicitFunction):
    """The family that owns the superconducting TF coil masses. `itart` decides it."""


class SuperconductingTfCoilAreasAndMassesConventional(
    SuperconductingTfCoilAreasAndMasses
):
    """`itart == 0` (conventional aspect ratio) -- `large_tokamak_eval`'s arm.

    Owns four of the slot's ten boundary reads (`m_tf_coil_case`, `m_tf_coil_copper`,
    `m_tf_coil_superconductor`, `m_tf_coils_total`) and does **not** own `whtcp` or
    `whttflgs`, which only the `itart == 1` arm writes
    (`superconducting.py:2086-2093`).

    `den_tf_sc_material` is `.tfcoil.dcond[0]`, i.e. `i_tf_sc_mat == 1` -- see
    `I_TF_SC_MAT_ITER_NB3SN`.
    """

    m_tf_coil_wp_insulation = OutputInto(tfcoil)
    cplen = OutputInto(tfcoil)
    m_tf_coil_case = OutputInto(tfcoil)
    m_tf_coil_superconductor = OutputInto(tfcoil)
    m_tf_coil_copper = OutputInto(tfcoil)
    m_tf_wp_steel_conduit = OutputInto(tfcoil)
    m_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    m_tf_coil_conductor = OutputInto(tfcoil)
    m_tf_coil = OutputInto(tfcoil)
    m_tf_coils_total = OutputInto(tfcoil)

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[I_TF_SC_MAT_ITER_NB3SN - 1]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return superconducting_tf_coil_areas_and_masses_conventional(
            len_tf_coil=len_tf_coil,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
            den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_inboard=dr_tf_inboard,
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


class SuperconductingTfCoilAreasAndMassesSphericalTokamak(
    SuperconductingTfCoilAreasAndMasses
):
    """`itart == 1` (spherical tokamak) -- the arm both ST regression files take.

    Same twenty reads as the conventional sibling, no more and no fewer, and the same ten
    outputs **plus two**: `.tfcoil.whtcp` and `.tfcoil.whttflgs`
    (`superconducting.py:2086-2093`). That extra pair is the whole reason `itart` is an
    occupant slot here rather than a static kwarg -- **conditional ownership**: a kwarg
    cannot make two `OutputInto`s appear at one value of a switch and vanish at the
    other.

    On a superconducting TART this node is the **sole producer** of `whtcp`/`whttflgs`,
    which `costs.py`'s `c22211`/`c22212` (`models/costs/costs.py:1616-1653`) and
    `hcpb.py`'s TF nuclear heating (`models/blankets/hcpb.py:491-554`) read. PROCESS's
    resistive-centrepost chain writes the same two fields at `i_tf_sup = 0`; that is a
    different occupant of a different slot, not this one.

    **`den_tf_sc_material` is bound to `dcond[0]` here, exactly as the conventional
    sibling binds it, and that is a known gap on the two live ST files.** Both
    `spherical_tokamak_eval.IN.DAT:355` and `st_regression.IN.DAT:827` set
    `i_tf_sc_mat = 9`, whose density is `dcond[8] == 8500.0` and not
    `dcond[0] == 6080.0` (`tfcoil_variables.py:157-170`). The gap is the pre-existing
    `I_TF_SC_MAT_ITER_NB3SN` bake recorded in `_audit/units/models/tfcoil/
    superconducting.md` § "switches touched" (`i_tf_sc_mat`: "`1` only"), not something
    this arm introduces, and closing it means making the slot an `i_tf_sc_mat` family the
    way `COILS_MASS_MATERIAL`/`WINDING_PACK_MATERIAL` already are -- a change to *both*
    arms, deliberately not made here. Neither ST file assembles yet (they refuse
    downstream at `itart_hcpb`), so the wrong density is not reachable by any machine;
    it must be fixed before it is.
    """

    m_tf_coil_wp_insulation = OutputInto(tfcoil)
    cplen = OutputInto(tfcoil)
    m_tf_coil_case = OutputInto(tfcoil)
    m_tf_coil_superconductor = OutputInto(tfcoil)
    m_tf_coil_copper = OutputInto(tfcoil)
    m_tf_wp_steel_conduit = OutputInto(tfcoil)
    m_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    m_tf_coil_conductor = OutputInto(tfcoil)
    m_tf_coil = OutputInto(tfcoil)
    m_tf_coils_total = OutputInto(tfcoil)
    whtcp = OutputInto(tfcoil)
    whttflgs = OutputInto(tfcoil)

    def __call__(
        self,
        len_tf_coil=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        den_tf_coil_case=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_coil_outboard_case=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        den_tf_sc_material=FromExactly(tfcoil.dcond[I_TF_SC_MAT_ITER_NB3SN - 1]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return superconducting_tf_coil_areas_and_masses_spherical_tokamak(
            len_tf_coil=len_tf_coil,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
            den_tf_wp_turn_insulation=den_tf_wp_turn_insulation,
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_inboard=dr_tf_inboard,
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
