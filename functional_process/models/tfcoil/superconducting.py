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
| `..._areas_and_masses` | `itart`, `i_tf_sc_mat` | 2 x 9 = 18 | -- |
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

from abc import abstractmethod

import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.physics.superconductors import (
    gl_nbti,
    itersc,
    jcrit_nbti,
    western_superconducting_nb3sn,
)
from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import (
    build,
    divertor,
    fwbs,
    superconducting_tfcoil,
    tfcoil,
)
from process.core import constants

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
    `run`.
    """
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


# ---------------------------------------------------------------------------
# `vv_stress_on_quench` -- `superconducting.py:1381-1452` (the method's own geometry
# prologue) and `:4869-5153` (`lambda_term`, `_theta_factor_integral`,
# `_inductance_factor`, `vv_stress_on_quench` itself)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `tf_cable_in_conduit_superconductor_properties` -- `superconducting.py:2806-3160`
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# `calculate_superconductor_temperature_margin` -- `superconducting.py:1174-1291`
# ---------------------------------------------------------------------------

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
    """The family that owns the superconducting TF coil masses. **Two** switches decide
    it -- `itart` and `i_tf_sc_mat` -- and the family is their full 2 x 9 product.

    Both axes are forced, for different reasons, and neither can be an
    `eqx.field(static=True)` kwarg:

    * **`itart`** changes the *owned* set: the spherical arm writes `.tfcoil.whtcp` and
      `.tfcoil.whttflgs` (`superconducting.py:2085-2093`) and the conventional arm
      writes neither. A kwarg cannot make an `OutputInto` appear at one value of a
      switch and vanish at the other -- conditional ownership.
    * **`i_tf_sc_mat`** changes one *read*: `den_tf_sc_material` is
      `.tfcoil.dcond[i_tf_sc_mat - 1]`, an array-element `VarPath`
      (`_audit/naming_convention.md` § "Array elements"). A `FromExactly` default is
      fixed when the class body executes, so the element it names is fixed with the
      class -- the same fact that made `stellarator.coils.coils_mass` a family
      (`_audit/next_steps.md` §14.11).

    Eighteen concrete occupants, then. **They are written as two axes rather than as
    eighteen flat classes, and that is the argument rather than an economy.** The defect
    this family closes (`_audit/units/models/tfcoil/superconducting.md`, 2026-08-27) was
    that *both* arms independently spelled the material, as a module constant baked into
    `FromExactly(tfcoil.dcond[0])` -- one switch answered twice, in two places, invisibly
    to `switch_audit`, which walks static fields and never sees a constant folded into a
    default. Eighteen flat classes would restore eighteen places to spell it. Here each
    material's element is spelled **once**, in one `...TfCoilMass` class, and both
    `itart` arms inherit it, so the two arms are structurally incapable of naming
    different materials -- the argument `_superconducting_tf_coil_masses` and
    `calculate_cplen` already make for the shared algebra, applied to the shared switch.

    That is `_audit/next_steps.md` §14.11's own preference ("the shape it wants is
    **nesting** ... rather than a flat product") taken where it is cheap: the nesting is
    in the class hierarchy, not in the model tree, so the slot still holds one node and
    `indat.SC_TF_MASSES` still holds one class per configuration. No lookup node is
    minted, exactly as `models/stellarator/coils/mass.py` decided -- the lookup's input
    is already a real place and its index is static.

    * The **`itart` axis** (`...Conventional`, `...SphericalTokamak`) declares the
      `OutputInto`s and `_masses`, the arm body.
    * The **`i_tf_sc_mat` axis** (`IterNb3snTfCoilMass` ...
      `HazeltonZhaiRebcoTfCoilMass`) declares `__call__`, whose only per-material entry
      is one `FromExactly`.
    * The **eighteen leaves** pair one of each and add nothing. Both axes are abstract
      on their own -- an arm has no `__call__`, a material has no `_masses` -- so the
      two classes that used to bake the switch cannot be instantiated any more, which is
      the strongest form of "the old answer is gone" available.
    """

    @abstractmethod
    def _masses(
        self,
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
        """Run this `itart` arm, given the density its material occupant read.

        The `itart` axis defines it; a material class on its own does not, which is what
        makes a material class abstract in practice. The arms are abstract by
        construction instead -- they define no `__call__`, and cottax's
        `ExplicitFunction.__call__` is an `abstractmethod`, so instantiating one raises
        `TypeError`. Between the two, only the eighteen leaves are usable, which is the
        property this family wants: the two classes that used to bake `i_tf_sc_mat` no
        longer answer it, and cannot.

        Not a port surface: `_params` reads `__call__`'s signature only
        (`ExplicitFunction._signature_of`), so what the graph sees is the material
        class's parameter list.
        """
        raise NotImplementedError(
            f"{type(self).__name__} names an `i_tf_sc_mat` material but no `itart` arm; "
            "a usable occupant pairs one of each -- see `indat.SC_TF_MASSES`."
        )


class SuperconductingTfCoilAreasAndMassesConventional(
    SuperconductingTfCoilAreasAndMasses
):
    """The `itart == 0` (conventional aspect ratio) **arm** -- one of the two axes.

    Abstract. It declares the ten outputs and the arm body, and takes its `__call__` --
    and with it the `.tfcoil.dcond` element it reads -- from whichever `...TfCoilMass`
    material class it is paired with.

    Owns four of the slot's ten boundary reads (`m_tf_coil_case`, `m_tf_coil_copper`,
    `m_tf_coil_superconductor`, `m_tf_coils_total`) and does **not** own `whtcp` or
    `whttflgs`, which only the `itart == 1` arm writes (`superconducting.py:2085-2093`).
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

    def _masses(
        self,
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
        """`superconducting_tf_coil_areas_and_masses_conventional`,
        the arm this class is.
        """
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
    """The `itart == 1` (spherical tokamak) **arm** -- one of the two axes.

    Abstract. It declares the twelve outputs and the arm body, and takes its `__call__`
    -- and with it the `.tfcoil.dcond` element it reads -- from whichever
    `...TfCoilMass` material class it is paired with.

    Same twenty reads as the conventional sibling, no more and no fewer, and the same
    ten outputs **plus two**: `.tfcoil.whtcp` and `.tfcoil.whttflgs`
    (`superconducting.py:2085-2093`). That extra pair is why `itart` is an occupant axis
    here rather than a static kwarg -- **conditional ownership**: a kwarg cannot make
    two `OutputInto`s appear at one value of a switch and vanish at the other.

    On a superconducting TART this node is the **sole producer** of `whtcp`/`whttflgs`,
    which `costs.py`'s `c22211`/`c22212` (`models/costs/costs.py:1616-1653`) and
    `hcpb.py`'s TF nuclear heating (`models/blankets/hcpb.py:491-554`) read. PROCESS's
    resistive-centrepost chain writes the same two fields at `i_tf_sup = 0`; that is a
    different occupant of a different slot, not this one.
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

    def _masses(
        self,
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
        """`superconducting_tf_coil_areas_and_masses_spherical_tokamak`,
        the arm this class is.
        """
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


class IterNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- ITER Nb3Sn.

    Reads `.tfcoil.dcond[0]` as the superconductor density, 6080.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    PROCESS's own default (`tfcoil_variables.py:246`) and the value both
    `large_tokamak_eval.IN.DAT:374` and `large_tokamak_nof.IN.DAT:583` set, so this is
    the material occupant those two reference machines assemble.

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[0]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class Bi2212TfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == BI2212` (2) -- Bi-2212.

    Reads `.tfcoil.dcond[1]` as the superconductor density, 6080.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[1]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class OldLubellNbtiTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == OLD_LUBELL_NBTI` (3) -- old Lubell NbTi.

    Reads `.tfcoil.dcond[2]` as the superconductor density, 6070.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[2]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class UserDefinedNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- user-defined Nb3Sn.

    Reads `.tfcoil.dcond[3]` as the superconductor density, 6080.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[3]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class WstNb3snTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == WST_NB3SN` (5) -- WST Nb3Sn.

    Reads `.tfcoil.dcond[4]` as the superconductor density, 6080.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    `low_aspect_ratio_DEMO.IN.DAT:910`'s value, and a warning about value tests:
    `dcond[4] == dcond[0] == 6080.0`, so this occupant reads the same *number* the baked
    `dcond[0]` used to, from the element the switch actually names. That machine's
    answers therefore do not move -- and the coincidence is exactly why no value test
    could have caught the bake. `_DCOND_POISON` in the case file is the answer to it.

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[4]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class CrocoRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- CroCo REBCO.

    Reads `.tfcoil.dcond[5]` as the superconductor density, 8500.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[5]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class DurhamNbtiTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == DURHAM_NBTI` (7) -- Durham Ginzburg-Landau NbTi.

    Reads `.tfcoil.dcond[6]` as the superconductor density, 6070.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[6]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class DurhamRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == DURHAM_REBCO` (8) -- Durham Ginzburg-Landau REBCO.

    Reads `.tfcoil.dcond[7]` as the superconductor density, 8500.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[7]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class HazeltonZhaiRebcoTfCoilMass(SuperconductingTfCoilAreasAndMasses):
    """`i_tf_sc_mat == HAZELTON_ZHAI_REBCO` (9) -- Hazelton-Zhai REBCO.

    Reads `.tfcoil.dcond[8]` as the superconductor density, 8500.0 kg/m3
    (`tfcoil_variables.py:157-170`).

    `spherical_tokamak_eval.IN.DAT:355` and `st_regression.IN.DAT:827`'s value, and the
    one this family exists to get right: `dcond[8] == 8500.0` against the
    `dcond[0] == 6080.0` both arms used to bake -- a 40 % superconductor-mass error.

    **Ported here even though the stellarator refuses the same value.**
    `indat.UNPORTED["i_tf_sc_mat", 9]` refuses `HAZELTON_ZHAI_REBCO` for
    `stellarator.coils.winding_pack_intersect_inputs`, because `jcrit_from_material`
    (`process/models/stellarator/coils/coils.py:52-160`) handles 1..8 and then raises:
    there is no critical-surface arm to port. **This slot asks a different question.**
    `superconducting_tf_coil_areas_and_masses` (`process/models/tfcoil/
    superconducting.py:2024-2036`) uses the material for exactly one thing, the density
    `dcond[i_tf_sc_mat - 1]`. No dispatch, no critical surface, and `dcond[8]` is a real
    populated element of a nine-long table (`tfcoil_variables.py:157-170`). So value 9 is
    portable *here* and refused *there*, and the two facts do not contradict: the
    refusal is about a model this node does not use.

    The `i_tf_sc_mat` **axis**, one of the family's two: abstract on its own,
    since `_masses` is the `itart` arm's. Paired with either arm it gives a
    concrete occupant, and this one `FromExactly` is the entire difference from
    its eight siblings.
    """

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
        den_tf_sc_material=FromExactly(tfcoil.dcond[8]),
        a_tf_turn_steel=From(tfcoil),
        den_steel=From(fwbs),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return self._masses(
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


class IterNb3snSuperconductingTfCoilAreasAndMassesConventional(
    IterNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 1)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, ITER_NB3SN]`.
    """


class IterNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    IterNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 1)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, ITER_NB3SN]`.
    """


class Bi2212SuperconductingTfCoilAreasAndMassesConventional(
    Bi2212TfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 2)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, BI2212]`.
    """


class Bi2212SuperconductingTfCoilAreasAndMassesSphericalTokamak(
    Bi2212TfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 2)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, BI2212]`.
    """


class OldLubellNbtiSuperconductingTfCoilAreasAndMassesConventional(
    OldLubellNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 3)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, OLD_LUBELL_NBTI]`.
    """


class OldLubellNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    OldLubellNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 3)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, OLD_LUBELL_NBTI]`.
    """


class UserDefinedNb3snSuperconductingTfCoilAreasAndMassesConventional(
    UserDefinedNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 4)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, USER_DEFINED_NB3SN]`.
    """


class UserDefinedNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    UserDefinedNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 4)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, USER_DEFINED_NB3SN]`.
    """


class WstNb3snSuperconductingTfCoilAreasAndMassesConventional(
    WstNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 5)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, WST_NB3SN]`.
    """


class WstNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    WstNb3snTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 5)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, WST_NB3SN]`.
    """


class CrocoRebcoSuperconductingTfCoilAreasAndMassesConventional(
    CrocoRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 6)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, CROCO_REBCO]`.
    """


class CrocoRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    CrocoRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 6)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, CROCO_REBCO]`.
    """


class DurhamNbtiSuperconductingTfCoilAreasAndMassesConventional(
    DurhamNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 7)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, DURHAM_NBTI]`.
    """


class DurhamNbtiSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    DurhamNbtiTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 7)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, DURHAM_NBTI]`.
    """


class DurhamRebcoSuperconductingTfCoilAreasAndMassesConventional(
    DurhamRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 8)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, DURHAM_REBCO]`.
    """


class DurhamRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    DurhamRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 8)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, DURHAM_REBCO]`.
    """


class HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesConventional(
    HazeltonZhaiRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesConventional
):
    """`(itart, i_tf_sc_mat) == (0, 9)`:
    `SC_TF_MASSES[CONVENTIONAL_ASPECT_RATIO, HAZELTON_ZHAI_REBCO]`.
    """


class HazeltonZhaiRebcoSuperconductingTfCoilAreasAndMassesSphericalTokamak(
    HazeltonZhaiRebcoTfCoilMass, SuperconductingTfCoilAreasAndMassesSphericalTokamak
):
    """`(itart, i_tf_sc_mat) == (1, 9)`:
    `SC_TF_MASSES[SPHERICAL_TOKAMAK, HAZELTON_ZHAI_REBCO]`.
    """


class VvStressOnQuench(ExplicitFunction):
    """cottax node: `.superconducting_tfcoil.vv_stress_quench`, constraint 65's read.

    Ports `CICCSuperconductingTFCoil.vv_stress_on_quench`
    (`process/models/tfcoil/superconducting.py:1381-1452`) -- the geometry prologue and
    the Itoh surrogate it calls. Unswitched: there is no `i_*` anywhere on this path.

    Twenty-nine reads, of which seventeen come from `.build` -- this is the node that
    ties the TF coil's own quantities to the vessel and shield radial build, which is
    why it reads more of `.build` than every other occupant of this slot put together.

    **`.tfcoil.tfa` is read whole and indexed in the body, where every other
    array-element read in this port is a `FromExactly(area.field[k])`.** PROCESS reads
    `self.data.tfcoil.tfa[0]` (`:1396`), so `FromExactly(tfcoil.tfa[0])` is the literal
    transcription -- and cottax refuses it: `tf_coil_shape` owns the *whole* `tfa`
    vector, and `_check_reads_match_owns` rejects a read that "lies inside" an owned
    variable because reads are matched by equality, so the element would silently become
    a boundary input while the array beside it was produced. The rule that distinguishes
    this from `.tfcoil.dcond[k]`, which stays a `FromExactly`, is **whether the array has
    a producer in the same graph**: `dcond` is a nine-entry constant table nothing owns,
    `tfa` is `tf_coil_shape`'s output. Indexing in the body keeps the real edge.

    **`.divertor.dz_divertor` is this slot's only read outside `.build`, `.tfcoil` and
    `.superconducting_tfcoil`,** and it is the one that makes the vessel height depend
    on the divertor -- an edge the TF chain did not previously carry anywhere.
    """

    vv_stress_quench = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        r_tf_inboard_mid=From(build),
        r_tf_outboard_mid=From(build),
        r_tf_inboard_out=From(build),
        tfa=From(tfcoil),
        z_plasma_xpoint_upper=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_shld_upper=From(build),
        dz_vv_upper=From(build),
        r_vv_inboard_out=From(build),
        dr_vv_outboard=From(build),
        dr_tf_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_shld_vv_gap_outboard=From(build),
        len_tf_coil=From(tfcoil),
        theta1_coil=From(tfcoil),
        theta1_vv=From(tfcoil),
        n_tf_coils=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        c_tf_coil=From(superconducting_tfcoil),
        dr_vv_shells=From(build),
    ):
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


class CiccSuperconductorProperties(ExplicitFunction):
    """The family that owns the CICC critical-current chain -- constraint 33's read.

    `i_tf_sc_mat` decides it. Every occupant owns the same nine variables, in `run`'s
    own write order (`superconducting.py:2725-2742`); they differ in which
    critical-surface fit they call, which `(bc20m, tc0m)` pair they hand it, and --
    genuinely -- in **which fields they read**:

    | `i_tf_sc_mat` | fit and its `(bc20m, tc0m)` | strain and extra reads |
    |---|---|---|
    | 1 ITER Nb3Sn *(live)* | `itersc`, `(32.97, 16.06)` | `str_wp` |
    | 3 old Lubell NbTi | `jcrit_nbti`, `(15.0, 9.3)` | **none** -- no strain |
    | 4 user-defined Nb3Sn | `itersc`, read | `str_wp`, `bcritsc`, `tcritsc` |
    | 5 WST Nb3Sn | `western_superconducting_nb3sn`, `(32.97, 16.06)` | `str_wp` |
    | 7 Durham GL NbTi | `gl_nbti`, read | `str_wp`, `b_crit_upper_nbti`, `t_crit_nbti` |

    Four of the nine values are refused, each for its own measured reason -- see
    `indat.UNPORTED` and `superconducting.md`'s dated section. In one sentence each:
    **2** (Bi-2212) reaches `TFSuperconductorLimits(bc20m=bc20m, ...)` with `bc20m` and
    `tc0m` never assigned on its branch, so PROCESS itself raises `UnboundLocalError`;
    **6**, **8** and **9** are `SuperconductorShape.TAPE`, which the function's own
    first guard (`:2882-2889`) refuses before any arithmetic.

    **`i_str_wp` is the second axis, and only its default arm is written.** The strain
    fed to the fit is `.tfcoil.str_tf_con_res` at `i_str_wp == 0` and `.tfcoil.str_wp`
    at `1` (`superconducting.py:2897-2900`). That is a *read*, so it is a class axis for
    the same reason `i_tf_sc_mat` is one for the mass slot -- a `From` default is fixed
    when the class body runs. `1` is PROCESS's default (`tfcoil_variables.py:508`) and
    **no tracked input file sets the switch at all**, so arm `0` is unreachable; it is
    registered in `UNPORTED` rather than baked, so a file that does set it is refused
    loudly instead of silently getting the other strain.

    **`.tfcoil.str_wp` is a new boundary input of the assembled machine**, and an
    honest one: PROCESS writes it in `run_and_output_stress` (`superconducting.py:2221`)
    from `stresscl`, which is unported. Landing this node makes a dependency visible
    that the graph previously did not express at all.
    """

    j_tf_wp_critical = OutputInto(tfcoil)
    j_crit_str_tf = OutputInto(tfcoil)
    f_c_tf_turn_operating_critical = OutputInto(superconducting_tfcoil)
    j_tf_coil_turn = OutputInto(superconducting_tfcoil)
    j_tf_superconductor = OutputInto(superconducting_tfcoil)
    c_tf_turn_cables_critical = OutputInto(superconducting_tfcoil)
    j_tf_superconductor_critical = OutputInto(superconducting_tfcoil)
    b_tf_superconductor_critical_zero_temp_strain = OutputInto(superconducting_tfcoil)
    temp_tf_superconductor_critical_zero_field_strain = OutputInto(
        superconducting_tfcoil
    )


class IterNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 1` -- `large_tokamak_eval.IN.DAT:374`'s own arm.

    `superconducting.py:2905-2939`. The `(32.97, 16.06)` pair is a literal on this
    branch, so it is a literal here and not a read.
    """

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
    ):
        return cicc_superconductor_properties_itersc(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
            b_c20max=32.97,
            temp_c0max=16.06,
        )


class UserDefinedNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 4` -- the ITER fit with `(bcritsc, tcritsc)` read from input.

    `superconducting.py:3011-3042`. Two reads its sibling arm 1 does not declare.
    """

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
        bcritsc=From(tfcoil),
        tcritsc=From(tfcoil),
    ):
        return cicc_superconductor_properties_itersc(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
            b_c20max=bcritsc,
            temp_c0max=tcritsc,
        )


class WstNb3snCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 5` -- `low_aspect_ratio_DEMO.IN.DAT:910`'s arm.

    `superconducting.py:3046-3082`. Same reads and same literals as arm 1; only the fit
    differs.
    """

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
    ):
        return cicc_superconductor_properties_wst_nb3sn(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
        )


class OldLubellNbtiCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 3` -- and the arm that reads **no strain**.

    `superconducting.py:2949-3007`. `jcrit_nbti` has no strain argument, so this
    occupant declares one read fewer than its four siblings. That is the concrete reason
    `i_str_wp` cannot be answered once for the whole family: on this arm it is not a
    question at all.
    """

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        tftmp=From(tfcoil),
    ):
        return cicc_superconductor_properties_lubell_nbti(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            temp_tf_coolant_peak_field=tftmp,
        )


class DurhamNbtiCiccSuperconductorProperties(CiccSuperconductorProperties):
    """`i_tf_sc_mat == 7` -- Durham Ginzburg-Landau NbTi.

    `superconducting.py:3086-3110`. Reads `(b_crit_upper_nbti, t_crit_nbti)` off
    `.tfcoil` and, unlike the three Nb3Sn arms, does **not** clip the strain.

    Ported here even though the *temperature-margin* slot refuses this same value: the
    two are different functions and the refusal is specific to the other one (PROCESS's
    own residual leaves the reals there -- see `TfSuperconductorTemperatureMargin`).
    """

    def __call__(
        self,
        a_tf_turn_cable_space_effective=From(superconducting_tfcoil),
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        c_tf_turn=From(tfcoil),
        str_wp=From(tfcoil),
        tftmp=From(tfcoil),
        b_crit_upper_nbti=From(tfcoil),
        t_crit_nbti=From(tfcoil),
    ):
        return cicc_superconductor_properties_durham_nbti(
            a_tf_turn_cable_space_effective=a_tf_turn_cable_space_effective,
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            c_tf_turn=c_tf_turn,
            strain=str_wp,
            temp_tf_coolant_peak_field=tftmp,
            b_crit_upper_nbti=b_crit_upper_nbti,
            t_crit_nbti=t_crit_nbti,
        )


class TfSuperconductorTemperatureMargin(ExplicitFunction):
    """The family that owns the TF temperature margin -- constraint 36's read.

    Ports `calculate_superconductor_temperature_margin`
    (`superconducting.py:1174-1291`) as `run` calls it (`:2749-2761`). `i_tf_sc_mat`
    decides it, and every occupant owns the **same two** variables --
    `.tfcoil.temp_tf_superconductor_margin` (`run`'s own assignment) and
    `.tfcoil.temp_margin` (written inside the function, `:1279`) -- which hold the same
    number.

    This is the port's second genuine internal solve after
    `models/vacuum/vacuum.py`'s duct diameter, and the first one whose answer a
    constraint reads: `scipy.optimize.newton`'s secant branch, replicated step for step
    in `solve_current_sharing_temperature`. Read that function's docstring for why the
    iteration is replicated rather than improved on.

    **Two `i_tf_sc_mat` values that `CiccSuperconductorProperties` ports are refused
    here**, and the asymmetry is measured, not conservative:

    - **2** (Bi-2212) is refused in both, but for a *different* reason in this one:
      `calculate_superconductor_temperature_margin` short-circuits it to
      `temp_tf_superconductor_margin = 0.0` and never writes `.tfcoil.temp_margin`
      (`:1231-1233`) -- conditional ownership, i.e. a genuinely different node, on an
      arm whose sibling function cannot run at all.
    - **7** (Durham GL NbTi) is refused because **PROCESS's own residual leaves the real
      numbers.** `gl_nbti` raises a negative base to a fractional power while the secant
      search probes above `t_c0`, and Python returns a `complex`; measured on a
      `b_tf_inboard_peak = 8.0` point, `optimize.newton` converges and PROCESS returns
      `0.4561454861673191+1.2475645615451133e-12j` -- a complex temperature margin. At
      `b_tf_inboard_peak = 12.5` the same call instead dies with `TypeError: '<=' not
      supported between instances of 'complex' and 'float'`. There is no real-valued
      PROCESS answer to agree with, so there is nothing to port: a JAX float64 body
      returns `nan`, which would be *more* correct than the reference, and this harness
      exists to measure agreement rather than to improve on PROCESS quietly.
    """

    temp_tf_superconductor_margin = OutputInto(tfcoil)
    temp_margin = OutputInto(tfcoil)


class _TemperatureMarginWithStrain(TfSuperconductorTemperatureMargin):
    """The strained arms' shared declaration; the `fit` attribute picks the fit.

    `run` re-reads the strain at `:2744-2747` and hands the **unclipped** value here,
    where `CiccSuperconductorProperties` clips it inside its own body -- see
    `temperature_margin_itersc`'s docstring, and defect D4.
    """

    fit = staticmethod(temperature_margin_itersc)

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        str_wp=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        margin = type(self).fit(
            j_superconductor=j_tf_superconductor,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            strain=str_wp,
            b_c20max=b_tf_superconductor_critical_zero_temp_strain,
            temp_c0max=temp_tf_superconductor_critical_zero_field_strain,
            temp_tf_coolant_peak_field=tftmp,
        )
        return margin, margin


class IterNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 1` *(live)*. `process/models/superconductors.py:1259`."""

    fit = staticmethod(temperature_margin_itersc)


class UserDefinedNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 4`. The same `itersc` residual as arm 1
    (`process/models/superconductors.py:1261`); the `(bc20m, tc0m)` it uses are the
    properties node's outputs, so the user-defined pair costs no extra read here.
    """

    fit = staticmethod(temperature_margin_itersc)


class WstNb3snTfSuperconductorTemperatureMargin(_TemperatureMarginWithStrain):
    """`i_tf_sc_mat == 5`. `process/models/superconductors.py:1263-1265`."""

    fit = staticmethod(temperature_margin_wst_nb3sn)


class OldLubellNbtiTfSuperconductorTemperatureMargin(TfSuperconductorTemperatureMargin):
    """`i_tf_sc_mat == 3` -- one read fewer, and one literal more.

    `superconductor_current_density_margin`'s branch 3 is the only one that consumes
    `c0`, which `run` passes as the literal `1.0e10` (`superconducting.py:1258`); and
    `jcrit_nbti` takes no strain, so this occupant does not read one.
    """

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        margin = temperature_margin_lubell_nbti(
            j_superconductor=j_tf_superconductor,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            b_c20max=b_tf_superconductor_critical_zero_temp_strain,
            temp_c0max=temp_tf_superconductor_critical_zero_field_strain,
            temp_tf_coolant_peak_field=tftmp,
        )
        return margin, margin
