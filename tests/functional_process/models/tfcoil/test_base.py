"""Harness cases for the ported device-agnostic TF coil layer
(`functional_process/cottax/tfcoil/base.py`).

Every legacy sample below is the **input** half of a case in PROCESS's own
`tests/unit/models/tfcoil/test_tfcoil.py`, lifted verbatim; the expected half is not
transcribed, because `Tier1Contract` calls the PROCESS reference itself and diffs the
two. That is a free, already-validated oracle (`_audit/test_harness.md` § Tier 1
sampling) and it removes a whole class of transcription error.

**The reference adapters close the `data` back-door on purpose.** Three of the ported
functions were extracted from bodies that read `self.data` *behind* an identically-named
parameter, and the adapters set those fields to the same values they pass. That is not
bookkeeping: `tf_global_geometry`'s minimum-thickness clamp reads
`data.tfcoil.n_tf_coils` (`process/models/tfcoil/base.py:334,339`) while the same
function takes `n_tf_coils` as an argument, and PROCESS's own unit test drives the two
apart -- it passes `n_tf_coils = 12` while `data.tfcoil.n_tf_coils` keeps its default
`16`, so the clamp is evaluated for a 16-coil machine. In the real pipeline
`run_base_tf` passes `self.data.tfcoil.n_tf_coils`, so the two always agree and the port
is right to carry one read; the adapter makes them agree here too. Recorded as defect
**D4** in `base.md`.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.tfcoil.base import (
    calculate_r_b_tf_inboard_peak,
    calculate_tf_global_geometry_circular_case,
    calculate_tf_global_geometry_straight_case,
    circumference,
    dr_tf_plasma_case_from_fraction,
    dr_tf_plasma_case_from_input,
    dx_tf_side_case_min_from_fraction,
    generic_tf_coil_area_and_masses,
    tf_coil_self_inductance_d_shape,
    tf_coil_self_inductance_picture_frame,
    tf_coil_shape_inner_d_shape_double_null,
    tf_coil_shape_inner_d_shape_single_null,
    tf_coil_shape_inner_picture_frame_tart,
    tf_current,
    tf_stored_magnetic_energy,
)
from process.core.model import DataStructure
from process.models.tfcoil.base import TFCoil
from process.models.tfcoil.superconducting import CICCSuperconductingTFCoil

# `i_tf_case_geom` / `i_f_dr_tf_plasma_case` / `tfc_sidewall_is_fraction` values, spelled
# out so an adapter reads as the arm it selects rather than as three bare literals.
_CIRCULAR_CASE = 0
_STRAIGHT_CASE = 1


def _tfcoil():
    """A `CICCSuperconductingTFCoil` with a fresh `DataStructure` attached.

    `TFCoil.__init__` only sets `self.outfile`, and the concrete subclass is used rather
    than the abstract base so `tf_coil_shape_inner` (an instance method that calls
    `self.circumference`) can be reached the same way `run_base_tf` reaches it.
    """
    model = CICCSuperconductingTFCoil()
    model.data = DataStructure()
    return model


# ---------------------------------------------------------------------------
# `circumference`
# ---------------------------------------------------------------------------


class TestCircumference(Tier1Contract):
    """`TFCoil.circumference` -> `circumference`, unchanged.

    Samples are PROCESS's own `test_circumference` parametrisation verbatim
    (`tests/unit/models/tfcoil/test_tfcoil.py:348-353`): one from the John D. Cook blog
    post the routine header cites, one from the 2018 baseline.
    """

    audit_record = "models/tfcoil/base.md"
    reference = staticmethod(TFCoil.circumference)
    ported = circumference

    samples = [
        legacy_sample("circumference-johndcook", aaa=2.667950e9, bbb=6.782819e8),
        legacy_sample(
            "circumference-baseline2018",
            aaa=4.7186039761812131,
            bbb=3.6192586838709673,
        ),
    ]

    fuzz_bounds = {"aaa": (0.5, 12.0), "bbb": (0.5, 12.0)}


# ---------------------------------------------------------------------------
# `tf_global_geometry`, three ways
# ---------------------------------------------------------------------------


def _reference_global_geometry(i_tf_case_geom):
    """The nine unswitched outputs of `tf_global_geometry`, for one case geometry.

    `dr_tf_inboard` and `dr_tf_nose_case` are held at PROCESS's own test values rather
    than being arguments: neither reaches any of the nine outputs (they feed only
    `dr_tf_plasma_case` and `dx_tf_side_case_min`, which the two contracts below own),
    so passing them would declare reads the ported node does not have.
    """

    def reference(
        n_tf_coils,
        r_tf_inboard_out,
        r_tf_inboard_in,
        r_tf_outboard_mid,
        dr_tf_outboard,
    ):
        data = DataStructure()
        data.tfcoil.n_tf_coils = n_tf_coils
        result = TFCoil.tf_global_geometry(
            i_tf_case_geom,
            False,  # i_f_dr_tf_plasma_case -- irrelevant to the nine
            0.0,  # f_dr_tf_plasma_case
            False,  # tfc_sidewall_is_fraction
            0.0,  # casths_fraction
            n_tf_coils,
            0.5,  # dr_tf_inboard -- see docstring
            0.1,  # dr_tf_nose_case
            r_tf_inboard_out,
            r_tf_inboard_in,
            r_tf_outboard_mid,
            dr_tf_outboard,
            data,
        )
        return (
            result.rad_tf_coil_inboard_toroidal_half,
            result.tan_theta_coil,
            result.a_tf_inboard_total,
            result.r_tf_outboard_in,
            result.r_tf_outboard_out,
            result.dx_tf_inboard_out_toroidal,
            result.a_tf_leg_outboard,
            result.dr_tf_full_midplane,
            result.dr_tf_internal_midplane,
        )

    return reference


class TestTfGlobalGeometryCircularCase(Tier1Contract):
    """`i_tf_case_geom == 0`. Sample: `test_tf_global_geometry` case 1's inputs."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_global_geometry(_CIRCULAR_CASE)
    ported = calculate_tf_global_geometry_circular_case

    samples = [
        legacy_sample(
            "tf_global_geometry-circular",
            n_tf_coils=16,
            r_tf_inboard_out=2.0,
            r_tf_inboard_in=1.5,
            r_tf_outboard_mid=5.0,
            dr_tf_outboard=0.3,
        ),
    ]

    fuzz_bounds = {
        "n_tf_coils": (8.0, 24.0),
        "r_tf_inboard_out": (1.5, 5.0),
        "r_tf_inboard_in": (0.5, 1.4),
        "r_tf_outboard_mid": (6.0, 20.0),
        "dr_tf_outboard": (0.2, 1.5),
    }


class TestTfGlobalGeometryStraightCase(Tier1Contract):
    """`i_tf_case_geom == 1`. Sample: `test_tf_global_geometry` case 2's inputs."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_global_geometry(_STRAIGHT_CASE)
    ported = calculate_tf_global_geometry_straight_case

    samples = [
        legacy_sample(
            "tf_global_geometry-straight",
            n_tf_coils=12,
            r_tf_inboard_out=1.8,
            r_tf_inboard_in=1.4,
            r_tf_outboard_mid=4.5,
            dr_tf_outboard=0.25,
        ),
    ]

    fuzz_bounds = {
        "n_tf_coils": (8.0, 24.0),
        "r_tf_inboard_out": (1.5, 5.0),
        "r_tf_inboard_in": (0.5, 1.4),
        "r_tf_outboard_mid": (6.0, 20.0),
        "dr_tf_outboard": (0.2, 1.5),
    }


def _reference_dr_tf_plasma_case_from_input(
    dr_tf_plasma_case, r_tf_inboard_in, dr_tf_inboard, n_tf_coils
):
    """`tf_global_geometry`'s `i_f_dr_tf_plasma_case == False` arm, through `data`.

    Both back-door reads are set to the values the port carries as arguments -- see the
    module docstring for why `n_tf_coils` in particular matters.
    """
    data = DataStructure()
    data.tfcoil.dr_tf_plasma_case = dr_tf_plasma_case
    data.tfcoil.n_tf_coils = n_tf_coils
    return TFCoil.tf_global_geometry(
        _CIRCULAR_CASE,
        False,
        0.0,
        False,
        0.0,
        n_tf_coils,
        dr_tf_inboard,
        0.1,
        2.0,
        r_tf_inboard_in,
        5.0,
        0.3,
        data,
    ).dr_tf_plasma_case


def _reference_dr_tf_plasma_case_from_fraction(
    f_dr_tf_plasma_case, dr_tf_inboard, r_tf_inboard_in, n_tf_coils
):
    """`tf_global_geometry`'s `i_f_dr_tf_plasma_case == True` arm, through `data`."""
    data = DataStructure()
    data.tfcoil.n_tf_coils = n_tf_coils
    return TFCoil.tf_global_geometry(
        _CIRCULAR_CASE,
        True,
        f_dr_tf_plasma_case,
        False,
        0.0,
        n_tf_coils,
        dr_tf_inboard,
        0.1,
        2.0,
        r_tf_inboard_in,
        5.0,
        0.3,
        data,
    ).dr_tf_plasma_case


def _reference_dx_tf_side_case_min_from_fraction(
    casths_fraction, r_tf_inboard_in, dr_tf_nose_case, n_tf_coils
):
    """`tf_global_geometry`'s `tfc_sidewall_is_fraction == True` arm."""
    data = DataStructure()
    data.tfcoil.n_tf_coils = n_tf_coils
    return TFCoil.tf_global_geometry(
        _CIRCULAR_CASE,
        False,
        0.0,
        True,
        casths_fraction,
        n_tf_coils,
        0.5,
        dr_tf_nose_case,
        2.0,
        r_tf_inboard_in,
        5.0,
        0.3,
        data,
    ).dx_tf_side_case_min


class TestDrTfPlasmaCaseFromInput(Tier1Contract):
    """The `FixedPointFunction` arm's body: `max(entering value, geometric minimum)`.

    The legacy sample is `test_tf_global_geometry` case 1's inputs, whose entering
    `dr_tf_plasma_case` is the field default `0.0`
    (`process/data_structure/tfcoil_variables.py:77`) -- exactly the reference run's
    state, and a point where the clamp binds, so the gradient with respect to the
    entering value is `0` there.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_dr_tf_plasma_case_from_input
    ported = dr_tf_plasma_case_from_input

    samples = [
        legacy_sample(
            "dr_tf_plasma_case-clamp-binds",
            dr_tf_plasma_case=0.0,
            r_tf_inboard_in=1.5,
            dr_tf_inboard=0.5,
            n_tf_coils=16,
        ),
        legacy_sample(
            "dr_tf_plasma_case-input-wins",
            dr_tf_plasma_case=0.2,
            r_tf_inboard_in=1.5,
            dr_tf_inboard=0.5,
            n_tf_coils=16,
        ),
    ]

    fuzz_bounds = {
        "dr_tf_plasma_case": (0.0, 0.3),
        "r_tf_inboard_in": (0.5, 3.0),
        "dr_tf_inboard": (0.3, 1.5),
        "n_tf_coils": (8.0, 24.0),
    }


class TestDrTfPlasmaCaseFromFraction(Tier1Contract):
    """The `i_f_dr_tf_plasma_case == True` arm.

    Sample: `test_tf_global_geometry` case 2.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_dr_tf_plasma_case_from_fraction
    ported = dr_tf_plasma_case_from_fraction

    samples = [
        legacy_sample(
            "dr_tf_plasma_case-fraction",
            f_dr_tf_plasma_case=0.1,
            dr_tf_inboard=0.4,
            r_tf_inboard_in=1.4,
            n_tf_coils=12,
        ),
    ]

    fuzz_bounds = {
        "f_dr_tf_plasma_case": (0.01, 0.3),
        "dr_tf_inboard": (0.3, 1.5),
        "r_tf_inboard_in": (0.5, 3.0),
        "n_tf_coils": (8.0, 24.0),
    }


class TestDxTfSideCaseMinFromFraction(Tier1Contract):
    """The `tfc_sidewall_is_fraction == True` arm.

    Sample: `test_tf_global_geometry` case 2.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_dx_tf_side_case_min_from_fraction
    ported = dx_tf_side_case_min_from_fraction

    samples = [
        legacy_sample(
            "dx_tf_side_case_min-fraction",
            casths_fraction=0.05,
            r_tf_inboard_in=1.4,
            dr_tf_nose_case=0.05,
            n_tf_coils=12,
        ),
    ]

    fuzz_bounds = {
        "casths_fraction": (0.01, 0.2),
        "r_tf_inboard_in": (0.5, 3.0),
        "dr_tf_nose_case": (0.05, 0.8),
        "n_tf_coils": (8.0, 24.0),
    }


# ---------------------------------------------------------------------------
# `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak`
# ---------------------------------------------------------------------------


def _reference_r_b_tf_inboard_peak(
    r_tf_inboard_out, dr_tf_plasma_case, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
):
    """`process/models/tfcoil/base.py:166-171`, transcribed.

    There is no PROCESS function to call: the expression is written inline in
    `run_base_tf` between two model calls, so a `data`-mediated adapter would have to
    run the whole of `run_base_tf` to reach it. Copied rather than re-derived, the same
    treatment `test_structure.py::_reference_intercoil_mass_scaling_reference` gives
    `st_strc`'s unstored `msupstr`.
    """
    return (
        r_tf_inboard_out
        - dr_tf_plasma_case
        - dx_tf_wp_insulation
        - dx_tf_wp_insertion_gap
    )


class TestRBTfInboardPeak(Tier1Contract):
    """`.tfcoil.r_b_tf_inboard_peak`. Sample: 2018-baseline build numbers."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_r_b_tf_inboard_peak
    ported = calculate_r_b_tf_inboard_peak

    samples = [
        legacy_sample(
            "r_b_tf_inboard_peak-baseline2018",
            r_tf_inboard_out=4.20194118510911,
            dr_tf_plasma_case=0.060000000000000012,
            dx_tf_wp_insulation=0.0080000000000000019,
            dx_tf_wp_insertion_gap=0.01,
        ),
    ]

    fuzz_bounds = {
        "r_tf_inboard_out": (1.5, 6.0),
        "dr_tf_plasma_case": (0.02, 0.3),
        "dx_tf_wp_insulation": (0.002, 0.05),
        "dx_tf_wp_insertion_gap": (0.002, 0.05),
    }


# ---------------------------------------------------------------------------
# `tf_current`
# ---------------------------------------------------------------------------


class TestTfCurrent(Tier1Contract):
    """`TFCoil.tf_current` -> `tf_current`, unchanged.

    Samples are PROCESS's own `test_tf_current` parametrisation's inputs
    (`tests/unit/models/tfcoil/test_tfcoil.py:290-318`).
    """

    audit_record = "models/tfcoil/base.md"
    reference = staticmethod(TFCoil.tf_current)
    ported = tf_current

    samples = [
        legacy_sample(
            "tf_current-16coils",
            n_tf_coils=16,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            r_b_tf_inboard_peak=2.5,
            a_tf_inboard_total=0.8,
        ),
        legacy_sample(
            "tf_current-12coils",
            n_tf_coils=12,
            b_plasma_toroidal_on_axis=3.0,
            rmajor=5.0,
            r_b_tf_inboard_peak=1.8,
            a_tf_inboard_total=0.5,
        ),
    ]

    fuzz_bounds = {
        "n_tf_coils": (8.0, 24.0),
        "b_plasma_toroidal_on_axis": (2.0, 12.0),
        "rmajor": (4.0, 12.0),
        "r_b_tf_inboard_peak": (1.0, 5.0),
        "a_tf_inboard_total": (0.4, 30.0),
    }


# ---------------------------------------------------------------------------
# `tf_coil_shape_inner`
# ---------------------------------------------------------------------------


def _reference_shape_inner_single_null(
    r_tf_inboard_out,
    rmajor,
    rminor,
    r_tf_outboard_in,
    z_tf_inside_half,
    z_tf_top,
    dr_tf_inboard,
):
    """`tf_coil_shape_inner` at `i_tf_shape = 1`, `itart = 0`, `i_single_null = 1`.

    `r_cp_top`, `r_tf_outboard_mid` and `r_tf_inboard_mid` are `0.0`: the D-shape /
    non-TART branch (`process/models/tfcoil/base.py:498-526`) never reads them, and
    passing them as arguments would declare reads the ported occupant does not have.
    """
    return _tfcoil().tf_coil_shape_inner(
        i_tf_shape=1,
        itart=0,
        i_single_null=1,
        r_tf_inboard_out=r_tf_inboard_out,
        r_cp_top=0.0,
        rmajor=rmajor,
        rminor=rminor,
        r_tf_outboard_in=r_tf_outboard_in,
        z_tf_inside_half=z_tf_inside_half,
        z_tf_top=z_tf_top,
        dr_tf_inboard=dr_tf_inboard,
        dr_tf_outboard=0.0,
        r_tf_outboard_mid=0.0,
        r_tf_inboard_mid=0.0,
    )


def _reference_shape_inner_double_null(
    r_tf_inboard_out,
    rmajor,
    rminor,
    r_tf_outboard_in,
    z_tf_inside_half,
    dr_tf_inboard,
):
    """The same, at `i_single_null = 0`. `z_tf_top` is not read on this arm."""
    return _tfcoil().tf_coil_shape_inner(
        i_tf_shape=1,
        itart=0,
        i_single_null=0,
        r_tf_inboard_out=r_tf_inboard_out,
        r_cp_top=0.0,
        rmajor=rmajor,
        rminor=rminor,
        r_tf_outboard_in=r_tf_outboard_in,
        z_tf_inside_half=z_tf_inside_half,
        z_tf_top=0.0,
        dr_tf_inboard=dr_tf_inboard,
        dr_tf_outboard=0.0,
        r_tf_outboard_mid=0.0,
        r_tf_inboard_mid=0.0,
    )


_SHAPE_SAMPLE = {
    "r_tf_inboard_out": 4.20194118510911,
    "rmajor": 8.8931664516129036,
    "rminor": 2.8830645161290323,
    "r_tf_outboard_in": 15.915405859443332,
    "z_tf_inside_half": 9.0730900215620327,
    "dr_tf_inboard": 1.208,
}
"""Reconstructed from the 2018-baseline arc coordinates PROCESS's own
`test_tf_coil_self_inductance` carries
(`tests/unit/models/tfcoil/test_tfcoil.py:600-640`).

`tf_coil_shape_inner` has no unit test of its own, so the point is built backwards out
of the one place a real converged coil shape is written down: `r_tf_arc[0]` and
`r_tf_arc[2]` are `r_tf_inboard_out` and `r_tf_outboard_in` directly,
`r_tf_arc[1] = rmajor - 0.2 * rminor` fixes the pair above (`rminor` taken from the same
baseline, `rmajor` solved for), `z_tf_arc[3] = -z_tf_inside_half`, and
`z_tf_arc[1] = z_tf_top - dr_tf_inboard` gives `z_tf_top` for the single-null case.
Provenance is PROCESS's own numbers, one algebraic step removed -- stated here rather
than presented as a directly-lifted sample.
"""


class TestTfCoilShapeDShapeSingleNull(Tier1Contract):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 1` -- the reference arm."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_shape_inner_single_null
    ported = tf_coil_shape_inner_d_shape_single_null

    samples = [
        legacy_sample(
            "tf_coil_shape-d-single-null",
            z_tf_top=8.7641467096774191,
            **_SHAPE_SAMPLE,
        ),
    ]


class TestTfCoilShapeDShapeDoubleNull(Tier1Contract):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 0`."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_shape_inner_double_null
    ported = tf_coil_shape_inner_d_shape_double_null

    samples = [legacy_sample("tf_coil_shape-d-double-null", **_SHAPE_SAMPLE)]


def _reference_shape_inner_picture_frame_tart(
    r_cp_top,
    r_tf_outboard_in,
    z_tf_inside_half,
    z_tf_top,
    dr_tf_inboard,
    r_tf_outboard_mid,
):
    """`tf_coil_shape_inner` at `i_tf_shape = 2`, `itart = 1`.

    Five arguments are pinned at `0.0` and each pin is a claim the reference checks:
    `r_tf_inboard_out` and `r_tf_inboard_mid` are read only on the `itart == 0`
    sub-branches (`process/models/tfcoil/base.py:554`, `:572`), `rmajor`/`rminor` only
    on the D-shape branches, and `dr_tf_outboard` only on the D-shape/`itart == 1` one.
    `i_single_null` is passed as `0` -- the picture frame does not consult it at all, so
    either value would do, and `0` is what both ST files set. A port that secretly read
    any of the five would disagree by value here.
    """
    return _tfcoil().tf_coil_shape_inner(
        i_tf_shape=2,
        itart=1,
        i_single_null=0,
        r_tf_inboard_out=0.0,
        r_cp_top=r_cp_top,
        rmajor=0.0,
        rminor=0.0,
        r_tf_outboard_in=r_tf_outboard_in,
        z_tf_inside_half=z_tf_inside_half,
        z_tf_top=z_tf_top,
        dr_tf_inboard=dr_tf_inboard,
        dr_tf_outboard=0.0,
        r_tf_outboard_mid=r_tf_outboard_mid,
        r_tf_inboard_mid=0.0,
    )


class TestTfCoilShapePictureFrameTart(Tier1Contract):
    """`i_tf_shape == 2`, `itart == 1` -- both ST regression files' arm.
    Added 2026-08-27, ST frontier wave 4.

    The legacy point is `spherical_tokamak_eval.IN.DAT` run through PROCESS's own
    `init_process` + `PlasmaGeometry.run()` + `Build.run()` -- one pass, not a converged
    solve, since no converged reference for this file exists yet. That is enough to make
    every number here PROCESS's own rather than reconstructed:
    `r_cp_top = r_tf_inboard_out = 1.333916508197074` m (the `i_tf_sup == 1` fall-through
    at `process/models/build.py:1813`, **not** the `i_r_cp_top = 2` fraction the file
    sets at `:78` -- that branch is `i_tf_sup != 1` only, so `f_r_cp = 1.4` is dead on
    this run), `z_tf_inside_half = 11.735` m, `z_tf_top = 12.635` m,
    `r_tf_outboard_mid = 10.274594873354488` m (the same ripple-limited radius the
    picture-frame ripple wave landed on, arrived at independently here),
    `dr_tf_inboard = 0.9` m (the file's literal, `:345`), and
    `r_tf_outboard_in = r_tf_outboard_mid - 0.5 * dr_tf_outboard = 9.824594873354488` m
    with `dr_tf_outboard = 0.9` (`f_dr_tf_outboard_inboard = 1.0`, `:85`).

    `tfa`/`tfb` come back as exact zeros from both sides -- see the ported function's
    docstring; that is the branch never assigning them, faithfully reproduced.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_shape_inner_picture_frame_tart
    ported = tf_coil_shape_inner_picture_frame_tart

    samples = [
        legacy_sample(
            "spherical_tokamak_eval-first-pass",
            r_cp_top=1.333916508197074,
            r_tf_outboard_in=9.824594873354488,
            z_tf_inside_half=11.735,
            z_tf_top=12.635,
            dr_tf_inboard=0.9,
            r_tf_outboard_mid=10.274594873354488,
        ),
    ]

    fuzz_bounds = {
        "r_cp_top": (0.5, 3.0),
        "r_tf_outboard_in": (5.0, 15.0),
        "z_tf_inside_half": (3.0, 15.0),
        "z_tf_top": (3.0, 16.0),
        "dr_tf_inboard": (0.2, 2.0),
        "r_tf_outboard_mid": (5.0, 16.0),
    }


# ---------------------------------------------------------------------------
# `tf_coil_self_inductance`
# ---------------------------------------------------------------------------


def _reference_self_inductance_d_shape(dr_tf_inboard, r_tf_arc, z_tf_arc):
    """`tf_coil_self_inductance` at `itart = 0`, `i_tf_shape = 1`.

    The four arguments the D-shape branch never reads are passed as `0.0` because the
    source is `numba.njit`-compiled and cannot take `None` -- PROCESS's own test says
    exactly that (`tests/unit/models/tfcoil/test_tfcoil.py:597-599`), which is
    independent confirmation that this occupant's three reads are the whole set.
    """
    return TFCoil.tf_coil_self_inductance(
        dr_tf_inboard=dr_tf_inboard,
        r_tf_arc=np.asarray(r_tf_arc, dtype=float),
        z_tf_arc=np.asarray(z_tf_arc, dtype=float),
        itart=0,
        i_tf_shape=1,
        z_tf_inside_half=0.0,
        dr_tf_outboard=0.0,
        r_tf_outboard_mid=0.0,
        r_tf_inboard_mid=0.0,
    )


def _reference_self_inductance_picture_frame(
    z_tf_inside_half, dr_tf_outboard, r_tf_outboard_mid, r_tf_inboard_mid
):
    """`tf_coil_self_inductance`'s `else`: the closed form."""
    return TFCoil.tf_coil_self_inductance(
        dr_tf_inboard=1.208,
        r_tf_arc=np.zeros(3),
        z_tf_arc=np.zeros(3),
        itart=0,
        i_tf_shape=0,
        z_tf_inside_half=z_tf_inside_half,
        dr_tf_outboard=dr_tf_outboard,
        r_tf_outboard_mid=r_tf_outboard_mid,
        r_tf_inboard_mid=r_tf_inboard_mid,
    )


class TestTfCoilSelfInductanceDShape(Tier1Contract):
    """The 100-interval numerical integration, `lax.scan`-ed.

    Both samples are `test_tf_coil_self_inductance`'s D-shape cases
    (`tests/unit/models/tfcoil/test_tfcoil.py:580-638`), generated from
    `baseline_2018_IN.DAT`. They differ only in the entering `ind_tf_coil`, which the
    function never reads -- so they are the same point twice, kept as two because the
    duplication is PROCESS's own and dropping one would silently disagree with the
    source it was lifted from.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_self_inductance_d_shape
    ported = tf_coil_self_inductance_d_shape

    samples = [
        legacy_sample(
            "tf_coil_self_inductance-baseline2018",
            dr_tf_inboard=1.208,
            r_tf_arc=np.array([
                4.20194118510911,
                8.316545161290323,
                15.915405859443332,
                8.316545161290323,
                4.20194118510911,
            ]),
            z_tf_arc=np.array([
                4.5336880258064509,
                7.5561467096774191,
                0.0,
                -9.0730900215620327,
                -5.4438540129372193,
            ]),
        ),
    ]


class TestTfCoilSelfInductancePictureFrame(Tier1Contract):
    """The closed-form arm. Sample: `test_tf_coil_self_inductance`'s third case."""

    audit_record = "models/tfcoil/base.md"
    reference = _reference_self_inductance_picture_frame
    ported = tf_coil_self_inductance_picture_frame

    samples = [
        legacy_sample(
            "tf_coil_self_inductance-picture-frame",
            z_tf_inside_half=9.0730900215620327,
            dr_tf_outboard=1.208,
            r_tf_outboard_mid=16.519405859443332,
            r_tf_inboard_mid=3.5979411851091103,
        ),
    ]

    fuzz_bounds = {
        "z_tf_inside_half": (4.0, 14.0),
        "dr_tf_outboard": (0.3, 2.0),
        "r_tf_outboard_mid": (10.0, 22.0),
        "r_tf_inboard_mid": (1.5, 6.0),
    }


# ---------------------------------------------------------------------------
# `tf_stored_magnetic_energy`
# ---------------------------------------------------------------------------


class TestTfStoredMagneticEnergy(Tier1Contract):
    """`TFCoil.tf_stored_magnetic_energy` -> the same, unchanged.

    Samples are PROCESS's own `test_tf_stored_magnetic_energy` parametrisation's inputs
    (`tests/unit/models/tfcoil/test_tfcoil.py:1969`), including its two degenerate rows
    (`ind_tf_coil == 0` and `c_tf_total == 0`), which are kept precisely because they
    are where a mis-signed or mis-scaled port still looks right in value.
    """

    audit_record = "models/tfcoil/base.md"
    reference = staticmethod(TFCoil.tf_stored_magnetic_energy)
    ported = tf_stored_magnetic_energy

    samples = [
        legacy_sample("stored-energy-1", ind_tf_coil=1.0, c_tf_total=2.0, n_tf_coils=3),
        legacy_sample("stored-energy-2", ind_tf_coil=0.5, c_tf_total=4.0, n_tf_coils=2),
        legacy_sample("stored-energy-3", ind_tf_coil=2.0, c_tf_total=5.0, n_tf_coils=4),
        legacy_sample("stored-energy-4", ind_tf_coil=0.0, c_tf_total=5.0, n_tf_coils=1),
        legacy_sample("stored-energy-5", ind_tf_coil=1.0, c_tf_total=0.0, n_tf_coils=10),
    ]

    fuzz_bounds = {
        "ind_tf_coil": (1e-6, 1e-5),
        "c_tf_total": (1e7, 5e8),
        "n_tf_coils": (8.0, 24.0),
    }


# ---------------------------------------------------------------------------
# `generic_tf_coil_area_and_masses`
# ---------------------------------------------------------------------------


def _reference_generic_area_and_masses(
    r_tf_inboard_out,
    r_tf_inboard_in,
    rad_tf_coil_inboard_toroidal_half,
    tan_theta_coil,
    len_tf_coil,
    r_tf_inboard_mid,
    r_tf_outboard_mid,
):
    """`generic_tf_coil_area_and_masses` through a `DataStructure`.

    Every field the source reads is set from an argument of the port's signature and
    nothing else is touched, which is the audit's "no hidden `data` read" claim as an
    executable check rather than an assertion.
    """
    model = _tfcoil()
    model.data.build.r_tf_inboard_out = r_tf_inboard_out
    model.data.build.r_tf_inboard_in = r_tf_inboard_in
    model.data.build.r_tf_inboard_mid = r_tf_inboard_mid
    model.data.build.r_tf_outboard_mid = r_tf_outboard_mid
    model.data.tfcoil.len_tf_coil = len_tf_coil
    model.data.superconducting_tfcoil.rad_tf_coil_inboard_toroidal_half = (
        rad_tf_coil_inboard_toroidal_half
    )
    model.data.superconducting_tfcoil.tan_theta_coil = tan_theta_coil

    model.generic_tf_coil_area_and_masses()
    return (
        model.data.tfcoil.tfocrn,
        model.data.tfcoil.tficrn,
        model.data.tfcoil.tfcryoarea,
    )


class TestGenericTfCoilAreaAndMasses(Tier1Contract):
    """Owns `.tfcoil.tfcryoarea`, one of the slot's ten boundary reads.

    Sample: `test_generic_tf_coil_area_and_masses`'s single case
    (`tests/unit/models/tfcoil/test_tfcoil.py:713-727`), from `baseline_2018_IN.DAT`.
    """

    audit_record = "models/tfcoil/base.md"
    reference = _reference_generic_area_and_masses
    ported = generic_tf_coil_area_and_masses

    samples = [
        legacy_sample(
            "generic-areas-baseline2018",
            r_tf_inboard_out=4.20194118510911,
            r_tf_inboard_in=2.9939411851091102,
            rad_tf_coil_inboard_toroidal_half=0.19634954084936207,
            tan_theta_coil=0.19891236737965801,
            len_tf_coil=50.483843027201402,
            r_tf_inboard_mid=3.5979411851091103,
            r_tf_outboard_mid=16.519405859443332,
        ),
    ]

    fuzz_bounds = {
        "r_tf_inboard_out": (1.5, 6.0),
        "r_tf_inboard_in": (1.0, 4.0),
        "rad_tf_coil_inboard_toroidal_half": (0.13, 0.4),
        "tan_theta_coil": (0.13, 0.42),
        "len_tf_coil": (20.0, 80.0),
        "r_tf_inboard_mid": (1.5, 6.0),
        "r_tf_outboard_mid": (10.0, 22.0),
    }
