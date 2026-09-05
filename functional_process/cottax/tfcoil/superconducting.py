"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CICCSuperconductingTFCoil` and the `SuperconductingTFCoil` layer above it.

Audit record: `functional_process/_audit/units/models/tfcoil/superconducting.md`.
The base-class half is `functional_process/cottax/tfcoil/base.py`; the quench half is
`functional_process/cottax/tfcoil/quench.py` (read that one for the CoolProp boundary).

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

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    divertor,
    fwbs,
    superconducting_tfcoil,
    tfcoil,
)
from functional_process.models.tfcoil.superconducting import (
    calculate_a_tf_turn,
    calculate_old_lubell_nbti_temperature_margin,
    calculate_temperature_margin_with_strain,
    calculate_vv_stress_on_quench,
    cicc_averaged_turn_geometry_from_current_per_turn,
    cicc_integer_turn_geometry,
    cicc_superconductor_properties_durham_nbti,
    cicc_superconductor_properties_itersc,
    cicc_superconductor_properties_lubell_nbti,
    cicc_superconductor_properties_wst_nb3sn,
    dx_tf_side_case_double_rectangular,
    dx_tf_side_case_rectangular,
    dx_tf_side_case_trapezoidal,
    peak_b_tf_inboard_with_ripple_flat,
    peak_b_tf_inboard_with_ripple_kovari,
    solve_current_sharing_temperature,  # noqa: F401 -- re-exported for pfcoil/superconductor.py
    superconducting_tf_coil_areas_and_masses_conventional,
    superconducting_tf_coil_areas_and_masses_spherical_tokamak,
    superconducting_tf_wp_geometry_double_rectangular,
    superconducting_tf_wp_geometry_rectangular,
    superconducting_tf_wp_geometry_trapezoidal,
    temperature_margin_itersc,
    temperature_margin_lubell_nbti,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
    temperature_margin_wst_nb3sn,
    tf_case_areas_circular_front,
    tf_case_areas_straight_front,
    tf_cicc_inboard_areas_and_fractions,
    tf_wp_currents,
    vv_stress_on_quench,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
    vv_stress_quench_from_build,  # noqa: F401 -- re-exported for tests/.../test_superconducting.py
)

_RIPPLE_FIT_COEFFICIENTS = {
    16: (0.28101, 1.8481, -0.88159, 0.93834),
    18: (0.29153, 1.81600, -0.84178, 0.90426),
    20: (0.29853, 1.82130, -0.85031, 0.89808),
}
"""M. Kovari's MAGINT fits, `process/models/tfcoil/superconducting.py:1502-1516`.

Keyed on `round(n_tf_coils)`; every other coil count takes the flat 9 % ripple
allowance instead (`superconducting.py:1519`).
"""


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
            coefficients=self.coefficients,
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
        return calculate_vv_stress_on_quench(
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
        return calculate_temperature_margin_with_strain(
            self.fit,
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            str_wp,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            tftmp,
        )


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
        return calculate_old_lubell_nbti_temperature_margin(
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            tftmp,
        )
