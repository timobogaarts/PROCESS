"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CICCSuperconductingTFCoil` and the `SuperconductingTFCoil` layer above it.

Audit record: `functional_process/_audit/units/models/tfcoil/superconducting.md`.
The base-class half is `functional_process/models/tfcoil/base.py`; the quench half is
`functional_process/models/tfcoil/quench.py` (read that one for the CoolProp boundary).

**Scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten boundary
reads** (`_audit/tokamak_boundary.md`). In scope and ported here:
`superconducting_tf_wp_geometry`, `superconducting_tf_case_geometry` (split in two, see
below), `tf_wp_currents`, `peak_b_tf_inboard_with_ripple`,
`tf_cable_in_conduit_averaged_turn_geometry`, `tf_cicc_inboard_areas_and_fractions`,
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
- `tf_cable_in_conduit_integer_turn_geometry` -- the `i_tf_turns_integer == 1` arm; see
  UNPORTED below.
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
| `superconducting_tf_coil_areas_and_masses` | `itart` | `itart == 0` | `itart == 1` |
| `run`'s turn-geometry choice | `i_tf_turns_integer` | `0` (non-integer) | `1` (integer) |

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
    2007. The `itart == 1` arm differs in two places -- the case-mass formula
    (line 1998-2006) and the extra `whtcp`/`whttflgs` split (2086-2093) -- and is a
    separate occupant, unwritten here.

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
    m_tf_coil_wp_insulation = (
        len_tf_coil
        * (a_tf_wp_with_insulation - a_tf_wp_no_insulation)
        * den_tf_wp_turn_insulation
    )

    cplen = (2.0 * z_tf_inside_half) + (2.0 * dr_tf_inboard)

    # The 2.2 factor fits the ITER-FDR 450 t value; CCFE note T&M/PKNIGHT/PROCESS/026.
    m_tf_coil_case = (
        2.2
        * den_tf_coil_case
        * (
            cplen * a_tf_coil_inboard_case
            + (len_tf_coil - cplen) * a_tf_coil_outboard_case
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


class CiccAveragedTurnGeometry(ExplicitFunction):
    """The family that owns the averaged CICC turn geometry.

    Two booleans decide it (`i_dx_tf_turn_general_input`,
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
