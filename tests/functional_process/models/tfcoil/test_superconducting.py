"""Harness cases for the ported superconducting / CICC TF coil layer
(`functional_process/models/tfcoil/superconducting.py`).

Same convention as `test_base.py`: every legacy sample is the **input** half of a case
in PROCESS's own `tests/unit/models/tfcoil/test_sctfcoil.py`, and the reference adapter
calls PROCESS through whatever surface it actually offers -- a `@staticmethod` directly
where there is one, a `DataStructure`-mediated instance call where there is not. The
adapters set exactly the fields their function reads, so "the port declares the right
reads" is executed rather than asserted.

Where one PROCESS function was split into two nodes (`superconducting_tf_case_geometry`
-> `TfCaseAreas` + `DxTfSideCase`), the adapter calls the whole PROCESS function and
slices the half the ported occupant owns. The arguments belonging to the *other* half
are held at the sample's own baseline values, so the slice is a projection and not a
different point.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.tfcoil.superconducting import (
    _RIPPLE_FIT_COEFFICIENTS,
    calculate_a_tf_turn,
    cicc_averaged_turn_geometry_from_current_per_turn,
    cicc_integer_turn_geometry,
    dx_tf_side_case_double_rectangular,
    dx_tf_side_case_rectangular,
    dx_tf_side_case_trapezoidal,
    peak_b_tf_inboard_with_ripple_flat,
    peak_b_tf_inboard_with_ripple_kovari,
    superconducting_tf_coil_areas_and_masses_conventional,
    superconducting_tf_wp_geometry_double_rectangular,
    superconducting_tf_wp_geometry_rectangular,
    superconducting_tf_wp_geometry_trapezoidal,
    tf_case_areas_circular_front,
    tf_case_areas_straight_front,
    tf_cicc_inboard_areas_and_fractions,
    tf_wp_currents,
)
from process.core.model import DataStructure
from process.models.tfcoil.superconducting import CICCSuperconductingTFCoil

_RECTANGULAR = 0
_DOUBLE_RECTANGULAR = 1
_TRAPEZOIDAL = 2
_CIRCULAR_CASE = 0
_STRAIGHT_CASE = 1


def _sctfcoil():
    """A `CICCSuperconductingTFCoil` with a fresh `DataStructure` attached."""
    model = CICCSuperconductingTFCoil()
    model.data = DataStructure()
    return model


# ---------------------------------------------------------------------------
# `superconducting_tf_wp_geometry`
# ---------------------------------------------------------------------------


def _reference_wp_geometry(i_tf_wp_geom):
    """`superconducting_tf_wp_geometry` for one `i_tf_wp_geom`, as a flat tuple.

    PROCESS returns a `TFWPGeometry` dataclass; the fields are unpacked in the port's
    own return order (which is the dataclass's own field order,
    `superconducting.py:94-122`).
    """

    def reference(
        r_tf_inboard_in,
        dr_tf_nose_case,
        dr_tf_wp_with_insulation,
        tan_theta_coil,
        dx_tf_side_case_min,
        dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap,
    ):
        result = CICCSuperconductingTFCoil.superconducting_tf_wp_geometry(
            i_tf_wp_geom=i_tf_wp_geom,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
            tan_theta_coil=tan_theta_coil,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )
        return (
            result.r_tf_wp_inboard_inner,
            result.r_tf_wp_inboard_outer,
            result.r_tf_wp_inboard_centre,
            result.dx_tf_wp_toroidal_min,
            result.dr_tf_wp_no_insulation,
            result.dx_tf_wp_primary_toroidal,
            result.dx_tf_wp_secondary_toroidal,
            result.dx_tf_wp_toroidal_average,
            result.a_tf_wp_with_insulation,
            result.a_tf_wp_no_insulation,
            result.a_tf_wp_ground_insulation,
        )

    return reference


_WP_GEOMETRY_SAMPLE = {
    "r_tf_inboard_in": 2.9939411851091102,
    "dr_tf_nose_case": 0.52465000000000006,
    "dr_tf_wp_with_insulation": 0.54261087836601019,
    "tan_theta_coil": 0.19891236737965801,
    "dx_tf_side_case_min": 0.05000000000000001,
    "dx_tf_wp_insulation": 0.0080000000000000019,
    "dx_tf_wp_insertion_gap": 0.01,
}
"""`test_superconducting_tf_wp_geometry`'s inputs, shared by all three of its cases
(`tests/unit/models/tfcoil/test_sctfcoil.py:699-785`) -- they differ only in
`i_tf_wp_geom`, which is what makes them three occupants rather than three points."""

_WP_GEOMETRY_FUZZ = {
    "r_tf_inboard_in": (1.5, 5.0),
    "dr_tf_nose_case": (0.1, 1.0),
    "dr_tf_wp_with_insulation": (0.2, 1.0),
    "tan_theta_coil": (0.13, 0.42),
    "dx_tf_side_case_min": (0.01, 0.2),
    "dx_tf_wp_insulation": (0.002, 0.03),
    "dx_tf_wp_insertion_gap": (0.002, 0.03),
}


class TestSuperconductingTfWpGeometryRectangular(Tier1Contract):
    """`i_tf_wp_geom == 0`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_wp_geometry(_RECTANGULAR)
    ported = superconducting_tf_wp_geometry_rectangular

    samples = [legacy_sample("wp-geometry-rectangular", **_WP_GEOMETRY_SAMPLE)]
    fuzz_bounds = _WP_GEOMETRY_FUZZ


class TestSuperconductingTfWpGeometryDoubleRectangular(Tier1Contract):
    """`i_tf_wp_geom == 1` -- `large_tokamak_eval`'s arm."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_wp_geometry(_DOUBLE_RECTANGULAR)
    ported = superconducting_tf_wp_geometry_double_rectangular

    samples = [legacy_sample("wp-geometry-double-rectangular", **_WP_GEOMETRY_SAMPLE)]
    fuzz_bounds = _WP_GEOMETRY_FUZZ


class TestSuperconductingTfWpGeometryTrapezoidal(Tier1Contract):
    """`i_tf_wp_geom == 2`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_wp_geometry(_TRAPEZOIDAL)
    ported = superconducting_tf_wp_geometry_trapezoidal

    samples = [legacy_sample("wp-geometry-trapezoidal", **_WP_GEOMETRY_SAMPLE)]
    fuzz_bounds = _WP_GEOMETRY_FUZZ


# ---------------------------------------------------------------------------
# `superconducting_tf_case_geometry`, split in two
# ---------------------------------------------------------------------------

_CASE_BASELINE = {
    "a_tf_inboard_total": 27.308689677971632,
    "n_tf_coils": 16,
    "a_tf_wp_with_insulation": 0.70527618095271016,
    "a_tf_leg_outboard": 1.9805354702921749,
    "rad_tf_coil_inboard_toroidal_half": 0.19634954084936207,
    "tan_theta_coil": 0.19891236737965801,
    "r_tf_wp_inboard_outer": 4.06120206347512,
    "r_tf_wp_inboard_inner": 3.5185911851091101,
    "r_tf_inboard_in": 2.9939411851091102,
}
"""`test_superconducting_tf_case_geometry`'s single case
(`tests/unit/models/tfcoil/test_sctfcoil.py:900-921`), less the four arguments the two
halves do not share."""

_R_TF_INBOARD_OUT = 4.20194118510911
_DR_TF_PLASMA_CASE = 0.060000000000000012
_DX_TF_SIDE_CASE_MIN = 0.05000000000000001
_DR_TF_WP_WITH_INSULATION = 0.54261087836601019


def _reference_case_areas(i_tf_case_geom):
    """The first four returns of `superconducting_tf_case_geometry`."""

    def reference(**kwargs):
        return CICCSuperconductingTFCoil.superconducting_tf_case_geometry(
            i_tf_wp_geom=_RECTANGULAR,  # only reaches the sidewall half, sliced off
            i_tf_case_geom=i_tf_case_geom,
            r_tf_inboard_out=kwargs.pop("r_tf_inboard_out", _R_TF_INBOARD_OUT),
            dr_tf_plasma_case=kwargs.pop("dr_tf_plasma_case", _DR_TF_PLASMA_CASE),
            dx_tf_side_case_min=_DX_TF_SIDE_CASE_MIN,
            dr_tf_wp_with_insulation=_DR_TF_WP_WITH_INSULATION,
            **kwargs,
        )[:4]

    return reference


def _reference_side_case(i_tf_wp_geom):
    """The last two returns of `superconducting_tf_case_geometry`."""

    def reference(
        dx_tf_side_case_min, tan_theta_coil=None, dr_tf_wp_with_insulation=None
    ):
        return CICCSuperconductingTFCoil.superconducting_tf_case_geometry(
            i_tf_wp_geom=i_tf_wp_geom,
            i_tf_case_geom=_CIRCULAR_CASE,
            dx_tf_side_case_min=dx_tf_side_case_min,
            dr_tf_wp_with_insulation=(
                _DR_TF_WP_WITH_INSULATION
                if dr_tf_wp_with_insulation is None
                else dr_tf_wp_with_insulation
            ),
            r_tf_inboard_out=_R_TF_INBOARD_OUT,
            dr_tf_plasma_case=_DR_TF_PLASMA_CASE,
            **{
                **_CASE_BASELINE,
                **({} if tan_theta_coil is None else {"tan_theta_coil": tan_theta_coil}),
            },
        )[4:6]

    return reference


class TestTfCaseAreasCircularFront(Tier1Contract):
    """`i_tf_case_geom == 0` -- the reference arm. Does not read `dr_tf_plasma_case`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_case_areas(_CIRCULAR_CASE)
    ported = tf_case_areas_circular_front

    samples = [
        legacy_sample(
            "case-areas-circular",
            r_tf_inboard_out=_R_TF_INBOARD_OUT,
            **_CASE_BASELINE,
        ),
    ]


class TestTfCaseAreasStraightFront(Tier1Contract):
    """`i_tf_case_geom == 1`. Does not read `r_tf_inboard_out`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_case_areas(_STRAIGHT_CASE)
    ported = tf_case_areas_straight_front

    samples = [
        legacy_sample(
            "case-areas-straight",
            dr_tf_plasma_case=_DR_TF_PLASMA_CASE,
            **_CASE_BASELINE,
        ),
    ]


class TestDxTfSideCaseRectangular(Tier1Contract):
    """`i_tf_wp_geom == 0`: sidewall average and peak thickness."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_side_case(_RECTANGULAR)
    ported = dx_tf_side_case_rectangular

    samples = [
        legacy_sample(
            "dx-side-case-rectangular",
            dx_tf_side_case_min=_DX_TF_SIDE_CASE_MIN,
            tan_theta_coil=0.19891236737965801,
            dr_tf_wp_with_insulation=_DR_TF_WP_WITH_INSULATION,
        ),
    ]

    fuzz_bounds = {
        "dx_tf_side_case_min": (0.01, 0.2),
        "tan_theta_coil": (0.13, 0.42),
        "dr_tf_wp_with_insulation": (0.2, 1.0),
    }


class TestDxTfSideCaseDoubleRectangular(Tier1Contract):
    """`i_tf_wp_geom == 1` -- the reference arm."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_side_case(_DOUBLE_RECTANGULAR)
    ported = dx_tf_side_case_double_rectangular

    samples = [
        legacy_sample(
            "dx-side-case-double-rectangular",
            dx_tf_side_case_min=_DX_TF_SIDE_CASE_MIN,
            tan_theta_coil=0.19891236737965801,
            dr_tf_wp_with_insulation=_DR_TF_WP_WITH_INSULATION,
        ),
    ]

    fuzz_bounds = {
        "dx_tf_side_case_min": (0.01, 0.2),
        "tan_theta_coil": (0.13, 0.42),
        "dr_tf_wp_with_insulation": (0.2, 1.0),
    }


class TestDxTfSideCaseTrapezoidal(Tier1Contract):
    """`i_tf_wp_geom == 2`: one read, and peak == average."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_side_case(_TRAPEZOIDAL)
    ported = dx_tf_side_case_trapezoidal

    samples = [
        legacy_sample(
            "dx-side-case-trapezoidal", dx_tf_side_case_min=_DX_TF_SIDE_CASE_MIN
        )
    ]

    fuzz_bounds = {"dx_tf_side_case_min": (0.01, 0.2)}


# ---------------------------------------------------------------------------
# `tf_wp_currents`
# ---------------------------------------------------------------------------


def _reference_wp_currents(c_tf_total, n_tf_coils, a_tf_wp_no_insulation):
    """`tf_wp_currents` through a `DataStructure`.

    PROCESS's version takes the whole structure and writes `.tfcoil.j_tf_wp` in place;
    the entering `j_tf_wp` is deliberately left at its `0.0` default here, because the
    claim the port makes is that it is never read (see `TfWpCurrents`' docstring), and
    PROCESS's own two test cases -- identical except for that entering value, identical
    in expected output -- are the evidence.
    """
    model = _sctfcoil()
    model.data.tfcoil.c_tf_total = c_tf_total
    model.data.tfcoil.n_tf_coils = n_tf_coils
    model.data.superconducting_tfcoil.a_tf_wp_no_insulation = a_tf_wp_no_insulation
    CICCSuperconductingTFCoil.tf_wp_currents(model.data)
    return model.data.tfcoil.j_tf_wp


class TestTfWpCurrents(Tier1Contract):
    """Owns `.tfcoil.j_tf_wp`. Sample: `test_tf_wp_currents`'s first case."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_wp_currents
    ported = tf_wp_currents

    samples = [
        legacy_sample(
            "wp-currents-imode",
            c_tf_total=256500000.00000003,
            n_tf_coils=16,
            a_tf_wp_no_insulation=0.60510952642236249,
        ),
    ]

    fuzz_bounds = {
        "c_tf_total": (5e7, 5e8),
        "n_tf_coils": (8.0, 24.0),
        "a_tf_wp_no_insulation": (0.1, 2.0),
    }


# ---------------------------------------------------------------------------
# `peak_b_tf_inboard_with_ripple`
# ---------------------------------------------------------------------------


def _reference_peak_b_kovari(
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dr_tf_wp_no_insulation,
    r_tf_wp_inboard_centre,
    b_tf_inboard_peak_symmetric,
):
    """The fitted arm, with the three side-effect fields read back off `data`.

    PROCESS returns only the peak field and stores `tf_fit_t`, `tf_fit_z` and
    `f_b_tf_inboard_peak_ripple_symmetric` on `data.superconducting_tfcoil`
    (`superconducting.py:1531,1539,1549`); the port returns all four, so the adapter
    collects them in the same order.
    """
    model = _sctfcoil()
    peak = model.peak_b_tf_inboard_with_ripple(
        n_tf_coils=n_tf_coils,
        dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
        dr_tf_wp_no_insulation=dr_tf_wp_no_insulation,
        r_tf_wp_inboard_centre=r_tf_wp_inboard_centre,
        b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
    )
    d = model.data.superconducting_tfcoil
    return (
        d.tf_fit_t,
        d.tf_fit_z,
        d.f_b_tf_inboard_peak_ripple_symmetric,
        peak,
    )


def _reference_peak_b_flat(b_tf_inboard_peak_symmetric):
    """The `else` arm, at a coil count outside {16, 18, 20}."""
    return _sctfcoil().peak_b_tf_inboard_with_ripple(
        n_tf_coils=17,
        dx_tf_wp_primary_toroidal=1.3,
        dr_tf_wp_no_insulation=0.5,
        r_tf_wp_inboard_centre=3.79,
        b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
    )


def _ported_peak_b_16(**kwargs):
    """`peak_b_tf_inboard_with_ripple_kovari` with the 16-coil fit bound.

    A thin binding, not a second implementation: `PeakBTfInboardWithRipple16Coils`
    binds the same tuple at the node level, so what is checked here is exactly what the
    node computes.
    """
    return peak_b_tf_inboard_with_ripple_kovari(
        coefficients=_RIPPLE_FIT_COEFFICIENTS[16], **kwargs
    )


class TestPeakBTfInboardWithRipple16Coils(Tier1Contract):
    """`round(n_tf_coils) == 16`. Sample: `test_peak_tf_with_ripple`'s first case."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_peak_b_kovari
    ported = _ported_peak_b_16

    samples = [
        legacy_sample(
            "peak-b-ripple-16coils",
            n_tf_coils=16,
            dx_tf_wp_primary_toroidal=1.299782604942499,
            dr_tf_wp_no_insulation=0.50661087836601015,
            r_tf_wp_inboard_centre=3.789896624292115,
            b_tf_inboard_peak_symmetric=11.717722779177526,
        ),
    ]


class TestPeakBTfInboardWithRippleFlatAllowance(Tier1Contract):
    """The flat 9 % arm: one read, one output."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_peak_b_flat
    ported = peak_b_tf_inboard_with_ripple_flat

    samples = [
        legacy_sample("peak-b-ripple-flat", b_tf_inboard_peak_symmetric=11.7),
    ]

    fuzz_bounds = {"b_tf_inboard_peak_symmetric": (4.0, 20.0)}


# ---------------------------------------------------------------------------
# `tf_cable_in_conduit_averaged_turn_geometry`
# ---------------------------------------------------------------------------


def _reference_cicc_averaged_turn_geometry(
    j_tf_wp,
    c_tf_turn,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    layer_ins,
    a_tf_wp_no_insulation,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_space_extra_void,
):
    """The both-flags-`False` arm, as a tuple in the port's return order.

    `dx_tf_turn_general` and `dx_tf_turn_cable_space_general` are passed at PROCESS's
    own test values but are *not* arguments here: on this arm the first is overwritten
    from `sqrt(a_tf_turn)` before use and the second is never read. `c_tf_turn` is an
    argument because it is read; it is not in the returned tuple because the port does
    not own it (see the port module's finding 1).
    """
    result = CICCSuperconductingTFCoil.tf_cable_in_conduit_averaged_turn_geometry(
        j_tf_wp=j_tf_wp,
        dx_tf_turn_steel=dx_tf_turn_steel,
        dx_tf_turn_insulation=dx_tf_turn_insulation,
        dx_tf_turn_general=0.049532469413859428,
        c_tf_turn=c_tf_turn,
        i_dx_tf_turn_general_input=False,
        i_dx_tf_turn_cable_space_general_input=False,
        dx_tf_turn_cable_space_general=0.0,
        layer_ins=layer_ins,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
    )
    return (
        result.a_tf_turn_cable_space_no_void,
        result.a_tf_turn_steel,
        result.a_tf_turn_insulation,
        result.n_tf_coil_turns,
        result.dx_tf_turn_general,
        result.dr_tf_turn,
        result.dx_tf_turn,
        result.dx_tf_turn_conduit_full_average,
        result.radius_tf_turn_cable_space_corners,
        result.dx_tf_turn_cable_space_average,
        result.a_tf_turn_cable_space_effective,
        result.f_a_tf_turn_cable_space_cooling,
    )


class TestCiccAveragedTurnGeometryFromCurrentPerTurn(Tier1Contract):
    """The reference arm of the turn geometry. Owns `.tfcoil.n_tf_coil_turns`.

    Sample: `test_tf_cable_in_conduit_averaged_turn_geometry`'s second case
    (`tests/unit/models/tfcoil/test_sctfcoil.py:1289-1313`), the one with both input
    flags `False`, from the I-mode input file. `dia_tf_turn_coolant_channel` and
    `f_a_tf_turn_cable_space_extra_void` are the values that test hardcodes at the call
    site (`test_sctfcoil.py:1390-1391`).
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_cicc_averaged_turn_geometry
    ported = cicc_averaged_turn_geometry_from_current_per_turn

    samples = [
        legacy_sample(
            "cicc-averaged-turn-imode",
            j_tf_wp=26493137.688284047,
            c_tf_turn=65000.0,
            dx_tf_turn_steel=0.0080000000000000019,
            dx_tf_turn_insulation=0.00080000000000000004,
            layer_ins=0.0,
            a_tf_wp_no_insulation=0.60510952642236249,
            dia_tf_turn_coolant_channel=0.004,
            f_a_tf_turn_cable_space_extra_void=0.3,
        ),
    ]

    fuzz_bounds = {
        "j_tf_wp": (1e7, 4e7),
        "c_tf_turn": (4e4, 1e5),
        "dx_tf_turn_steel": (0.004, 0.012),
        "dx_tf_turn_insulation": (5e-4, 2e-3),
        "layer_ins": (0.0, 1e-3),
        "a_tf_wp_no_insulation": (0.3, 1.2),
        "dia_tf_turn_coolant_channel": (0.002, 0.008),
        "f_a_tf_turn_cable_space_extra_void": (0.2, 0.45),
    }


# ---------------------------------------------------------------------------
# `tf_cable_in_conduit_integer_turn_geometry`
# ---------------------------------------------------------------------------


def _reference_cicc_integer_turn_geometry(
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
    """The `i_tf_turns_integer == 1` arm, as a tuple in the port's return order.

    PROCESS's staticmethod takes `data` for its two `.tfcoil` reads
    (`superconducting.py:3536-3546`), so the adapter sets exactly those two fields on a
    fresh `DataStructure` -- the port's claim that they are reads of this node is
    thereby executed, not asserted. `n_tf_wp_layers`/`n_tf_wp_pancakes` are passed as
    floats deliberately: PROCESS's own body divides and multiplies by them
    continuously, and the fuzz/gradient samples exercise exactly that continuity.
    """
    model = _sctfcoil()
    model.data.tfcoil.dia_tf_turn_coolant_channel = dia_tf_turn_coolant_channel
    model.data.tfcoil.f_a_tf_turn_cable_space_extra_void = (
        f_a_tf_turn_cable_space_extra_void
    )
    result = model.tf_cable_in_conduit_integer_turn_geometry(
        dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        n_tf_wp_layers=n_tf_wp_layers,
        dx_tf_wp_toroidal_min=dx_tf_wp_toroidal_min,
        n_tf_wp_pancakes=n_tf_wp_pancakes,
        c_tf_coil=c_tf_coil,
        dx_tf_turn_steel=dx_tf_turn_steel,
        dx_tf_turn_insulation=dx_tf_turn_insulation,
        data=model.data,
    )
    return (
        result.radius_tf_turn_cable_space_corners,
        result.dr_tf_turn,
        result.dx_tf_turn,
        result.a_tf_turn_cable_space_no_void,
        result.a_tf_turn_steel,
        result.a_tf_turn_insulation,
        result.c_tf_turn,
        result.n_tf_coil_turns,
        result.dr_tf_turn_conduit_full,
        result.dx_tf_turn_conduit_full_toroidal,
        result.dx_tf_turn_conduit_full_average,
        result.dr_tf_turn_cable_space,
        result.dx_tf_turn_cable_space,
        result.dx_tf_turn_cable_space_average,
        result.a_tf_turn_cable_space_effective,
        result.f_a_tf_turn_cable_space_cooling,
        result.dx_tf_turn_general,
    )


class TestCiccIntegerTurnGeometry(Tier1Contract):
    """The integer arm. Owns `.tfcoil.c_tf_turn` -- the averaged arm reads it.

    Sample: `test_tf_cable_in_conduit_integer_turn_geometry`'s first case
    (`tests/unit/models/tfcoil/test_sctfcoil.py:1052-1085`, from the retired
    baseline-2018 file, `n_tf_wp_layers = 10` x `n_tf_wp_pancakes = 20`), with the two
    `data`-mediated reads at their `DataStructure` defaults
    (`tfcoil_variables.py:175,717`), which is what that test's fixture holds. The fuzz
    bounds put `dia_tf_turn_coolant_channel` up to `low_aspect_ratio_DEMO`'s `0.010`
    and the void fraction down to its `0.300`, so the file that motivated this occupant
    is inside the sampled box.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_cicc_integer_turn_geometry
    ported = cicc_integer_turn_geometry

    samples = [
        legacy_sample(
            "cicc-integer-turn-baseline2018",
            dr_tf_wp_with_insulation=0.54261087836601019,
            dx_tf_wp_insulation=0.0080000000000000019,
            dx_tf_wp_insertion_gap=0.01,
            n_tf_wp_layers=10.0,
            dx_tf_wp_toroidal_min=1.299782604942499,
            n_tf_wp_pancakes=20.0,
            c_tf_coil=14805350.287500001,
            dx_tf_turn_steel=0.0080000000000000002,
            dx_tf_turn_insulation=0.002,
            dia_tf_turn_coolant_channel=0.005,
            f_a_tf_turn_cable_space_extra_void=0.4,
        ),
    ]

    fuzz_bounds = {
        # The box is chosen so the cable space stays positive at every corner
        # (worst case: dr_tf_wp 0.45, insulation+gap maximal, 12 layers, 8 mm steel
        # -> dr_tf_turn_cable_space ~ 0.011 m): PROCESS's `np.sqrt` returns NaN on a
        # negative cable space where the port's `safe_sqrt` clamps, and that error
        # path is PROCESS-logging territory, not a point to fuzz across.
        "dr_tf_wp_with_insulation": (0.45, 1.2),
        "dx_tf_wp_insulation": (0.004, 0.010),
        "dx_tf_wp_insertion_gap": (0.005, 0.015),
        "n_tf_wp_layers": (4.0, 12.0),
        "dx_tf_wp_toroidal_min": (0.9, 2.0),
        "n_tf_wp_pancakes": (8.0, 24.0),
        "c_tf_coil": (5e6, 3e7),
        "dx_tf_turn_steel": (0.004, 0.008),
        "dx_tf_turn_insulation": (5e-4, 2e-3),
        "dia_tf_turn_coolant_channel": (0.002, 0.012),
        "f_a_tf_turn_cable_space_extra_void": (0.2, 0.45),
    }


# ---------------------------------------------------------------------------
# `tf_cicc_inboard_areas_and_fractions`
# ---------------------------------------------------------------------------


def _reference_cicc_inboard_areas(**kwargs):
    """`tf_cicc_inboard_areas_and_fractions`, as a flat tuple."""
    result = CICCSuperconductingTFCoil.tf_cicc_inboard_areas_and_fractions(**kwargs)
    return (
        result.a_tf_wp_coolant_channels,
        result.a_tf_wp_conductor,
        result.a_tf_wp_extra_void,
        result.a_tf_coil_wp_turn_insulation,
        result.a_tf_wp_steel,
        result.a_tf_coil_inboard_steel,
        result.f_a_tf_coil_inboard_steel,
        result.a_tf_coil_inboard_insulation,
        result.f_a_tf_coil_inboard_insulation,
    )


class TestTfCiccInboardAreasAndFractions(Tier1Contract):
    """No switch; ten reads, nine outputs.

    PROCESS has no unit test for this function, so the sample is assembled from the
    2018-baseline numbers its neighbours' tests carry -- `n_tf_coil_turns`,
    `a_tf_turn_steel`, `a_tf_turn_cable_space_no_void`,
    `f_a_tf_turn_cable_space_extra_void`, `a_tf_coil_inboard_case`, `n_tf_coils` and
    `a_tf_inboard_total` from `test_superconducting_tf_coil_area_and_masses` /
    `test_superconducting_tf_case_geometry`, `a_tf_wp_ground_insulation` from
    `test_superconducting_tf_wp_geometry`, and `a_tf_turn_insulation` from
    `a_tf_coil_wp_turn_insulation / n_tf_coil_turns` in the first of those. Stated
    rather than presented as one lifted case.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_cicc_inboard_areas
    ported = tf_cicc_inboard_areas_and_fractions

    samples = [
        legacy_sample(
            "cicc-inboard-areas-baseline2018",
            n_tf_coil_turns=200.0,
            dia_tf_turn_coolant_channel=0.005,
            a_tf_turn_cable_space_no_void=0.001293323051622732,
            f_a_tf_turn_cable_space_extra_void=0.30000000000000004,
            a_tf_turn_insulation=0.00043940087233490438,
            a_tf_turn_steel=0.0014685061538103825,
            n_tf_coils=16.0,
            a_tf_inboard_total=27.308689677971632,
            a_tf_coil_inboard_case=1.0015169239205168,
            a_tf_wp_ground_insulation=0.028582295732936136,
        ),
    ]

    fuzz_bounds = {
        "n_tf_coil_turns": (50.0, 400.0),
        "dia_tf_turn_coolant_channel": (0.002, 0.008),
        "a_tf_turn_cable_space_no_void": (5e-4, 3e-3),
        "f_a_tf_turn_cable_space_extra_void": (0.2, 0.45),
        "a_tf_turn_insulation": (1e-4, 1e-3),
        "a_tf_turn_steel": (5e-4, 3e-3),
        "n_tf_coils": (8.0, 24.0),
        "a_tf_inboard_total": (5.0, 40.0),
        "a_tf_coil_inboard_case": (0.3, 3.0),
        "a_tf_wp_ground_insulation": (0.005, 0.1),
    }


# ---------------------------------------------------------------------------
# `run`'s inline `.tfcoil.a_tf_turn`
# ---------------------------------------------------------------------------


def _reference_a_tf_turn(c_tf_total, j_tf_wp, n_tf_coils, n_tf_coil_turns):
    """`process/models/tfcoil/superconducting.py:2700-2704`, transcribed.

    Inline in `run`, so there is no callable to bind -- same treatment as
    `test_base.py::_reference_r_b_tf_inboard_peak`.
    """
    return c_tf_total / (j_tf_wp * n_tf_coils * n_tf_coil_turns)


class TestTfTurnArea(Tier1Contract):
    """`.tfcoil.a_tf_turn`, the cross-section per turn."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_a_tf_turn
    ported = calculate_a_tf_turn

    samples = [
        legacy_sample(
            "a_tf_turn-baseline2018",
            c_tf_total=256500000.00000003,
            j_tf_wp=26493137.688284047,
            n_tf_coils=16.0,
            n_tf_coil_turns=200.0,
        ),
    ]

    fuzz_bounds = {
        "c_tf_total": (5e7, 5e8),
        "j_tf_wp": (1e7, 4e7),
        "n_tf_coils": (8.0, 24.0),
        "n_tf_coil_turns": (50.0, 400.0),
    }


# ---------------------------------------------------------------------------
# `superconducting_tf_coil_areas_and_masses`
# ---------------------------------------------------------------------------


def _reference_sc_areas_and_masses(
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
    """`superconducting_tf_coil_areas_and_masses` at `itart == 0`, through `data`.

    `den_tf_sc_material` is written into every slot of `dcond` rather than into one, so
    the adapter does not have to also carry `i_tf_sc_mat`: the port takes the density
    already indexed (the `models/stellarator/coils/mass.py` convention), and this is the
    way to hand PROCESS the same number without a second, redundant argument.
    """
    model = _sctfcoil()
    model.data.physics.itart = 0
    model.data.fwbs.den_steel = den_steel
    model.data.build.z_tf_inside_half = z_tf_inside_half
    model.data.build.dr_tf_inboard = dr_tf_inboard

    t = model.data.tfcoil
    t.len_tf_coil = len_tf_coil
    t.den_tf_wp_turn_insulation = den_tf_wp_turn_insulation
    t.den_tf_coil_case = den_tf_coil_case
    t.a_tf_coil_inboard_case = a_tf_coil_inboard_case
    t.a_tf_coil_outboard_case = a_tf_coil_outboard_case
    t.n_tf_coil_turns = n_tf_coil_turns
    t.a_tf_turn_cable_space_no_void = a_tf_turn_cable_space_no_void
    t.f_a_tf_turn_cable_space_extra_void = f_a_tf_turn_cable_space_extra_void
    t.f_a_tf_turn_cable_copper = f_a_tf_turn_cable_copper
    t.a_tf_wp_coolant_channels = a_tf_wp_coolant_channels
    t.a_tf_turn_steel = a_tf_turn_steel
    t.a_tf_coil_wp_turn_insulation = a_tf_coil_wp_turn_insulation
    t.n_tf_coils = n_tf_coils
    t.i_tf_sc_mat = 1
    t.dcond = np.full(9, den_tf_sc_material, dtype=float)

    d = model.data.superconducting_tfcoil
    d.a_tf_wp_with_insulation = a_tf_wp_with_insulation
    d.a_tf_wp_no_insulation = a_tf_wp_no_insulation

    model.superconducting_tf_coil_areas_and_masses()
    return (
        t.m_tf_coil_wp_insulation,
        t.cplen,
        t.m_tf_coil_case,
        t.m_tf_coil_superconductor,
        t.m_tf_coil_copper,
        t.m_tf_wp_steel_conduit,
        t.m_tf_coil_wp_turn_insulation,
        t.m_tf_coil_conductor,
        t.m_tf_coil,
        t.m_tf_coils_total,
    )


class TestSuperconductingTfCoilAreasAndMassesConventional(Tier1Contract):
    """Owns four of the slot's ten boundary reads.

    Sample: `test_superconducting_tf_coil_area_and_masses`'s first case
    (`tests/unit/models/tfcoil/test_sctfcoil.py:1740-1830`), from
    `baseline_2018_IN.DAT`. That case sets `i_tf_sc_mat = 5`, whose `dcond` entry is
    `6080.0` -- the same density `i_tf_sc_mat = 1` selects, which is why one number
    stands in for both here (`process/data_structure/tfcoil_variables.py:157-170`: the
    nine densities are `6080` for entries 1, 2, 4, 5 and `6070`/`8500` elsewhere).
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_areas_and_masses
    ported = superconducting_tf_coil_areas_and_masses_conventional

    samples = [
        legacy_sample(
            "sc-masses-baseline2018",
            len_tf_coil=50.483843027201402,
            a_tf_wp_with_insulation=0.70527618095271016,
            a_tf_wp_no_insulation=0.64024601555360383,
            den_tf_wp_turn_insulation=1800.0,
            z_tf_inside_half=9.0730900215620327,
            dr_tf_inboard=1.208,
            den_tf_coil_case=8000.0,
            a_tf_coil_inboard_case=1.0015169239205168,
            a_tf_coil_outboard_case=1.2752592893394648,
            n_tf_coil_turns=200.0,
            a_tf_turn_cable_space_no_void=0.001293323051622732,
            f_a_tf_turn_cable_space_extra_void=0.30000000000000004,
            f_a_tf_turn_cable_copper=0.80884,
            a_tf_wp_coolant_channels=0.015707963267948974,
            den_tf_sc_material=6080.0,
            a_tf_turn_steel=0.0014685061538103825,
            den_steel=7800.0,
            a_tf_coil_wp_turn_insulation=0.087880174466980876,
            n_tf_coils=16.0,
        ),
    ]

    fuzz_bounds = {
        "len_tf_coil": (20.0, 80.0),
        "a_tf_wp_with_insulation": (0.3, 1.5),
        "a_tf_wp_no_insulation": (0.2, 1.2),
        "den_tf_wp_turn_insulation": (1500.0, 2200.0),
        "z_tf_inside_half": (4.0, 14.0),
        "dr_tf_inboard": (0.5, 2.0),
        "den_tf_coil_case": (7000.0, 9000.0),
        "a_tf_coil_inboard_case": (0.3, 3.0),
        "a_tf_coil_outboard_case": (0.3, 3.0),
        "n_tf_coil_turns": (50.0, 400.0),
        "a_tf_turn_cable_space_no_void": (5e-4, 3e-3),
        "f_a_tf_turn_cable_space_extra_void": (0.2, 0.45),
        "f_a_tf_turn_cable_copper": (0.5, 0.9),
        "a_tf_wp_coolant_channels": (0.005, 0.05),
        "den_tf_sc_material": (6000.0, 8600.0),
        "a_tf_turn_steel": (5e-4, 3e-3),
        "den_steel": (7000.0, 8200.0),
        "a_tf_coil_wp_turn_insulation": (0.02, 0.2),
        "n_tf_coils": (8.0, 24.0),
    }
