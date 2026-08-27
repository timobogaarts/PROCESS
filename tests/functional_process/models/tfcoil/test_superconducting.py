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

import dataclasses

import numpy as np
import pytest

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.indat import (
    CICC_SUPERCONDUCTOR_PROPERTIES,
    TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN,
    UNPORTED,
    machine_from_indat,
)
from functional_process.models.tfcoil.superconducting import (
    _RIPPLE_FIT_COEFFICIENTS,
    calculate_a_tf_turn,
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
    superconducting_tf_coil_areas_and_masses_conventional,
    superconducting_tf_coil_areas_and_masses_spherical_tokamak,
    superconducting_tf_wp_geometry_double_rectangular,
    superconducting_tf_wp_geometry_rectangular,
    superconducting_tf_wp_geometry_trapezoidal,
    temperature_margin_itersc,
    temperature_margin_lubell_nbti,
    temperature_margin_wst_nb3sn,
    tf_case_areas_circular_front,
    tf_case_areas_straight_front,
    tf_cicc_inboard_areas_and_fractions,
    tf_wp_currents,
    vv_stress_on_quench,
    vv_stress_quench_from_build,
)
from process.core.model import DataStructure
from process.models.tfcoil.superconducting import (
    CICCSuperconductingTFCoil,
)
from process.models.tfcoil.superconducting import (
    vv_stress_on_quench as process_vv_stress_on_quench,
)
from tests.functional_process.test_machine import TOKAMAK_BASELINE_INDAT

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

_DCOND_POISON = -1.0e9
"""Seeded into every `.tfcoil.dcond` element the occupant under test does not bind.

`i_tf_sc_mat`'s only effect in this function is which element of the nine-long density
table `m_tf_coil_superconductor` is scaled by (`process/models/tfcoil/
superconducting.py:2024-2036`, the sole `dcond` read) -- and **four of the nine elements
hold 6080.0** (`tfcoil_variables.py:157-170`). So an adapter that filled `dcond`
uniformly, as this one did until 2026-08-27, could not tell `dcond[0]` from `dcond[4]`,
and neither could one that left the table at its defaults. That is exactly the hole the
two occupants fell into: both baked `dcond[0]` for every value of the switch and no
value test noticed.

Poisoning every element the occupant does not name turns a wrong-element read into a
mass that is wrong in sign as well as magnitude, at every sample. Same technique and
same reason as `models/pfcoil/test_masses.py`'s `_DCOND_POISON`
(`_audit/next_steps.md` §14.11, the `CoilsMass` lesson turned into an assertion), and it
is safe here for a stronger reason than there: this function reads `dcond` exactly once.
"""


def _run_reference_sc_areas_and_masses(
    i_tf_sc_mat,
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

    The leading argument is not an input of the chain but the **identity of the occupant
    under test**: which `i_tf_sc_mat` PROCESS is driven at, and therefore which single
    `dcond` element `den_tf_sc_material` is planted in. Every other element is
    `_DCOND_POISON`. The port takes the density already indexed (the
    `models/stellarator/coils/mass.py` convention), so the ported side is the same pure
    function for all nine materials and the whole discrimination is on the reference
    side -- `models/pfcoil/test_masses.py`'s shape exactly. The wrappers below bind it;
    the harness never sees it.
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
    t.i_tf_sc_mat = i_tf_sc_mat
    t.dcond = np.full(9, _DCOND_POISON, dtype=float)
    t.dcond[i_tf_sc_mat - 1] = den_tf_sc_material

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


def _reference_sc_areas_and_masses(**inputs):
    """`i_tf_sc_mat = 1` (ITER Nb3Sn), density from `dcond[0]`.

    `large_tokamak_eval.IN.DAT:374` and `large_tokamak_nof.IN.DAT:583`'s configuration,
    and `IterNb3snSuperconductingTfCoilAreasAndMassesConventional`'s binding.
    """
    return _run_reference_sc_areas_and_masses(1, **inputs)


def _reference_sc_areas_and_masses_wst_nb3sn(**inputs):
    """`i_tf_sc_mat = 5` (WST Nb3Sn), density from `dcond[4]`.

    `low_aspect_ratio_DEMO.IN.DAT:910`'s configuration, and
    `WstNb3snSuperconductingTfCoilAreasAndMassesConventional`'s binding.
    """
    return _run_reference_sc_areas_and_masses(5, **inputs)


class TestSuperconductingTfCoilAreasAndMassesConventional(Tier1Contract):
    """Owns four of the slot's ten boundary reads. `(itart, i_tf_sc_mat) == (0, 1)`.

    Sample: `test_superconducting_tf_coil_area_and_masses`'s first case
    (`tests/unit/models/tfcoil/test_sctfcoil.py:1740-1830`), from
    `baseline_2018_IN.DAT`. That case sets `i_tf_sc_mat = 5`, whose `dcond` entry is
    `6080.0` -- the same density `i_tf_sc_mat = 1` selects, which is why one number
    stands in for both here (`process/data_structure/tfcoil_variables.py:157-170`: the
    nine densities are `6080` for entries 1, 2, 4, 5 and `6070`/`8500` elsewhere).

    That coincidence is also why `TestSuperconductingTfCoilAreasAndMassesConventional
    WstNb3sn` exists rather than being redundant with this: the two arms of the material
    axis are told apart by *which element is poisoned*, not by the density's value.
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


def _run_reference_sc_areas_and_masses_spherical_tokamak(
    i_tf_sc_mat,
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
    """`superconducting_tf_coil_areas_and_masses` at `itart == 1`, through `data`.

    The conventional arm's adapter with `itart = 1`, plus `whtcp`/`whttflgs` sliced out
    of `data` -- the two fields only this arm writes. Both are **poisoned with NaN before
    the call**, so "PROCESS wrote them here" is executed rather than assumed: were the
    branch not taken, the comparison would see NaN and fail, instead of silently agreeing
    on a leftover default of zero.

    `i_tf_sc_mat` leads for the same reason as in the conventional adapter: it names the
    occupant under test, `den_tf_sc_material` is planted in that element of `dcond` alone
    and every other element is `_DCOND_POISON`.
    """
    model = _sctfcoil()
    model.data.physics.itart = 1
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
    t.i_tf_sc_mat = i_tf_sc_mat
    t.dcond = np.full(9, _DCOND_POISON, dtype=float)
    t.dcond[i_tf_sc_mat - 1] = den_tf_sc_material
    t.whtcp = np.nan
    t.whttflgs = np.nan

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
        t.whtcp,
        t.whttflgs,
    )


def _reference_sc_areas_and_masses_spherical_tokamak(**inputs):
    """`(itart, i_tf_sc_mat) == (1, 1)`, density from `dcond[0]`.

    The conventional reference's own material with `itart` flipped:
    `IterNb3snSuperconductingTfCoilAreasAndMassesSphericalTokamak`'s binding.
    """
    return _run_reference_sc_areas_and_masses_spherical_tokamak(1, **inputs)


def _reference_sc_areas_and_masses_st_hazelton_zhai_rebco(**inputs):
    """`(itart, i_tf_sc_mat) == (1, 9)`, density from `dcond[8]`.

    `spherical_tokamak_eval.IN.DAT:355` and `st_regression.IN.DAT:827`'s configuration
    -- the value both tracked ST files actually set, and the one this whole family
    exists to stop being answered as `dcond[0]`.
    """
    return _run_reference_sc_areas_and_masses_spherical_tokamak(9, **inputs)


class TestSuperconductingTfCoilAreasAndMassesSphericalTokamak(Tier1Contract):
    """The `itart == 1` sibling: the same ten outputs, plus `whtcp` and `whttflgs`.

    Two samples, and the pair is the point.

    `sc-masses-st-baseline2018` is the conventional case's own point
    (`tests/unit/models/tfcoil/test_sctfcoil.py:1740-1830`, `baseline_2018_IN.DAT`) with
    `itart` flipped and nothing else touched. Holding every number fixed and moving only
    the switch is what makes the two arms comparable: any disagreement between them is
    the branch and nothing else.

    `sc-masses-st-shortleg` moves the geometry to where this arm actually lives -- an ST
    whose `len_tf_coil` is the *outboard* length alone and so is comparable to `cplen`
    rather than more than twice it (`18.5` against `11.0`, where the baseline point is
    `50.5` against `20.6`), with the twelve coils and the REBCO density (`dcond[8] ==
    8500.0`) a spherical tokamak actually runs. `dr_tf_inboard = 0.9` and
    `n_tf_coils = 12` are
    `tests/regression/input_files/spherical_tokamak_eval.IN.DAT:345,362`; the areas are
    the baseline case's, since PROCESS publishes no unit-test point for this arm, so the
    provenance is "PROCESS's own numbers, one algebraic step removed" in the sense
    `test_base.py` already uses. Both samples separate this arm's `tfleng_sph = cplen +
    len_tf_coil` denominator (`superconducting.py:2087`) from any other candidate; the
    second one says the agreement is not an artefact of conventional proportions.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_areas_and_masses_spherical_tokamak
    ported = superconducting_tf_coil_areas_and_masses_spherical_tokamak

    samples = [
        legacy_sample(
            "sc-masses-st-baseline2018",
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
        legacy_sample(
            "sc-masses-st-shortleg",
            len_tf_coil=18.5,
            a_tf_wp_with_insulation=0.70527618095271016,
            a_tf_wp_no_insulation=0.64024601555360383,
            den_tf_wp_turn_insulation=1800.0,
            z_tf_inside_half=4.6,
            dr_tf_inboard=0.9,
            den_tf_coil_case=8000.0,
            a_tf_coil_inboard_case=1.0015169239205168,
            a_tf_coil_outboard_case=1.2752592893394648,
            n_tf_coil_turns=200.0,
            a_tf_turn_cable_space_no_void=0.001293323051622732,
            f_a_tf_turn_cable_space_extra_void=0.30000000000000004,
            f_a_tf_turn_cable_copper=0.80884,
            a_tf_wp_coolant_channels=0.015707963267948974,
            den_tf_sc_material=8500.0,
            a_tf_turn_steel=0.0014685061538103825,
            den_steel=7800.0,
            a_tf_coil_wp_turn_insulation=0.087880174466980876,
            n_tf_coils=12.0,
        ),
    ]

    fuzz_bounds = {
        "len_tf_coil": (10.0, 80.0),
        "a_tf_wp_with_insulation": (0.3, 1.5),
        "a_tf_wp_no_insulation": (0.2, 1.2),
        "den_tf_wp_turn_insulation": (1500.0, 2200.0),
        "z_tf_inside_half": (3.0, 14.0),
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


class TestSuperconductingTfCoilAreasAndMassesConventionalWstNb3sn(Tier1Contract):
    """`(itart, i_tf_sc_mat) == (0, 5)` -- `low_aspect_ratio_DEMO`'s occupant.

    **The material axis, discriminated.** The ported side is the same pure function as
    `TestSuperconductingTfCoilAreasAndMassesConventional`'s -- the occupants differ in
    one `FromExactly`, `.tfcoil.dcond[4]` against `.tfcoil.dcond[0]`, and nothing else --
    so the whole difference is on the reference side: PROCESS runs at `i_tf_sc_mat = 5`,
    `den_tf_sc_material` is planted in `dcond[4]` alone, and every other element is
    `_DCOND_POISON`.

    Why it is not redundant with the case above, given that `dcond[4] == dcond[0] ==
    6080.0`: **that equality is the reason it is needed.** Until 2026-08-27 both
    `itart` occupants read `dcond[0]` at every value of the switch, and
    `low_aspect_ratio_DEMO` (`IN.DAT:910`, `i_tf_sc_mat = 5`) assembled the wrong element
    and got the right number anyway. No value test on this machine could have seen that.
    This one fails unless PROCESS's read *moves with the switch*, which is the claim the
    two occupants make and the claim the old baked constant broke.

    Same converged `baseline_2018` point as the reference case: the material does not
    change the function's domain, only which array slot one scalar comes from.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_areas_and_masses_wst_nb3sn
    ported = superconducting_tf_coil_areas_and_masses_conventional

    samples = [
        legacy_sample(
            "sc-masses-baseline2018-wst-nb3sn",
            **TestSuperconductingTfCoilAreasAndMassesConventional.samples[0].kwargs,
        ),
    ]

    fuzz_bounds = TestSuperconductingTfCoilAreasAndMassesConventional.fuzz_bounds


_ST_SHORTLEG = TestSuperconductingTfCoilAreasAndMassesSphericalTokamak.samples[1].kwargs
"""`sc-masses-st-shortleg`'s point, reused verbatim by the material-axis case below.

Named rather than re-typed so the two cases cannot drift apart in anything but the
`dcond` element they are driven at, which is the only thing they are meant to differ in.
"""


class TestSuperconductingTfCoilAreasAndMassesStHazeltonZhaiRebco(Tier1Contract):
    """`(itart, i_tf_sc_mat) == (1, 9)` -- both tracked ST files' occupant.

    **The value the family was built for.** `spherical_tokamak_eval.IN.DAT:355` and
    `st_regression.IN.DAT:827` both set `i_tf_sc_mat = 9`, whose density is
    `dcond[8] == 8500.0`; both arms used to bake `dcond[0] == 6080.0`, a 40 %
    superconductor-mass error waiting for either file to assemble. This case is that
    claim executed: PROCESS is driven at `i_tf_sc_mat = 9`, `8500.0` sits in `dcond[8]`
    alone, and the other eight elements are `_DCOND_POISON`.

    It is also the executable half of the **portability** argument. `indat.UNPORTED`
    refuses `i_tf_sc_mat = 9` for the stellarator's `winding_pack_intersect_inputs`,
    because `jcrit_from_material` has no branch 9. This function never calls it: the
    material selects one element of a density table, and `dcond[8]` is populated. If that
    reading were wrong, PROCESS would raise here rather than agree -- so a passing case
    is evidence for it, not an assumption about it.

    Sample: `sc-masses-st-shortleg`, whose `den_tf_sc_material = 8500.0` is already
    `dcond[8]`'s true value, so the point is the ST geometry *and* the ST material
    together rather than one grafted onto the other. `whtcp`/`whttflgs` stay NaN-poisoned
    before the call, so the `itart == 1` branch is still an executed check.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_areas_and_masses_st_hazelton_zhai_rebco
    ported = superconducting_tf_coil_areas_and_masses_spherical_tokamak

    samples = [
        legacy_sample(
            "sc-masses-st-shortleg-hazelton-zhai",
            **_ST_SHORTLEG,
        ),
    ]

    fuzz_bounds = TestSuperconductingTfCoilAreasAndMassesSphericalTokamak.fuzz_bounds


# ---------------------------------------------------------------------------
# `vv_stress_on_quench` -- constraint 65's producer
# ---------------------------------------------------------------------------

_VV_POISON = np.nan
"""Written into every `.build`/`.tfcoil`/`.superconducting_tfcoil` field the *method*
under test is not being given, before PROCESS is called.

`vv_stress_on_quench` is an instance method reading twenty-nine fields off `data` with
no arguments at all, so "the port declares the right reads" cannot be checked by
argument count -- it has to be checked by making an undeclared read visibly poison the
answer. NaN is the right poison here rather than `_DCOND_POISON`'s large negative,
because every one of these fields enters a length or an area and a wrong-but-finite one
would be indistinguishable from a rounding difference."""


def _reference_vv_stress_from_build(**inputs):
    """`CICCSuperconductingTFCoil.vv_stress_on_quench`, through a NaN-poisoned `data`.

    Every `.build` field is set to NaN first, then the twenty-nine the port declares are
    written; `.tfcoil.tfa` is a four-element array of which only element 0 is given a
    value, the other three staying NaN. So a port that read a field it does not declare,
    or the wrong `tfa` element, returns NaN and the value test fails rather than
    quietly agreeing.
    """
    model = _sctfcoil()
    for field in dataclasses.fields(model.data.build):
        if isinstance(getattr(model.data.build, field.name), float):
            setattr(model.data.build, field.name, _VV_POISON)

    b, t = model.data.build, model.data.tfcoil
    d = model.data.superconducting_tfcoil
    b.z_tf_inside_half = inputs["z_tf_inside_half"]
    b.dr_tf_inboard = inputs["dr_tf_inboard"]
    b.r_tf_inboard_mid = inputs["r_tf_inboard_mid"]
    b.r_tf_outboard_mid = inputs["r_tf_outboard_mid"]
    b.r_tf_inboard_out = inputs["r_tf_inboard_out"]
    b.z_plasma_xpoint_upper = inputs["z_plasma_xpoint_upper"]
    b.dz_xpoint_divertor = inputs["dz_xpoint_divertor"]
    b.dz_shld_upper = inputs["dz_shld_upper"]
    b.dz_vv_upper = inputs["dz_vv_upper"]
    b.r_vv_inboard_out = inputs["r_vv_inboard_out"]
    b.dr_vv_outboard = inputs["dr_vv_outboard"]
    b.dr_tf_outboard = inputs["dr_tf_outboard"]
    b.dr_tf_shld_gap = inputs["dr_tf_shld_gap"]
    b.dr_shld_thermal_outboard = inputs["dr_shld_thermal_outboard"]
    b.dr_shld_vv_gap_outboard = inputs["dr_shld_vv_gap_outboard"]
    b.dr_vv_shells = inputs["dr_vv_shells"]

    model.data.divertor.dz_divertor = inputs["dz_divertor"]

    t.tfa = np.full(4, _VV_POISON, dtype=float)
    t.tfa[0] = inputs["tfa_first_arc"]
    t.len_tf_coil = inputs["len_tf_coil"]
    t.theta1_coil = inputs["theta1_coil"]
    t.theta1_vv = inputs["theta1_vv"]
    t.n_tf_coils = inputs["n_tf_coils"]
    t.n_tf_coil_turns = inputs["n_tf_coil_turns"]
    t.t_tf_superconductor_quench = inputs["t_tf_superconductor_quench"]

    d.a_tf_coil_inboard_steel = inputs["a_tf_coil_inboard_steel"]
    d.a_tf_plasma_case = inputs["a_tf_plasma_case"]
    d.a_tf_coil_nose_case = inputs["a_tf_coil_nose_case"]
    d.dx_tf_side_case_average = inputs["dx_tf_side_case_average"]
    d.c_tf_coil = inputs["c_tf_coil"]

    model.vv_stress_on_quench()
    return d.vv_stress_quench


def _reference_vv_stress_core(**inputs):
    """The module-level `vv_stress_on_quench`, with PROCESS's own capitalised names.

    A pure function, so no `data` and no poison: this case exists to pin the Itoh
    surrogate itself (the theta-factor integral, the `lambda` branch, the two
    inductances) independently of the prologue that feeds it. Both halves are tested,
    because only one of them is a transcription of arithmetic and the other is a
    transcription of *reads*.
    """
    renamed = {"h_coil": "H_coil", "h_vv": "H_vv"}
    return process_vv_stress_on_quench(**{
        renamed.get(name, name): value for name, value in inputs.items()
    })


_VV_CORE_POINT = {
    "h_coil": 9.5,
    "ri_coil": 3.0,
    "ro_coil": 13.0,
    "rm_coil": 4.0,
    "ccl_length_coil": 50.0,
    "theta1_coil": 45.0,
    "h_vv": 8.0,
    "ri_vv": 4.0,
    "ro_vv": 11.0,
    "rm_vv": 5.0,
    "theta1_vv": 1.0,
    "n_tf_coils": 16.0,
    "n_tf_coil_turns": 200.0,
    "s_rp": 0.5,
    "s_cc": 0.6,
    "taud": 20.0,
    "i_op": 7.0e4,
    "d_vv": 0.12,
}
"""A large-tokamak-scale coil and vessel. `theta1_coil = 45.0` and `theta1_vv = 1.0`
are `tfcoil_variables.py`'s own defaults and `large_tokamak_eval.IN.DAT`'s live values;
the geometry is that file's build to one significant figure."""


class TestVvStressOnQuenchCore(Tier1Contract):
    """The Itoh surrogate. Eighteen arguments, one output, no `data`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_vv_stress_core
    ported = vv_stress_on_quench

    samples = [legacy_sample("vv-stress-large-tokamak", **_VV_CORE_POINT)]

    fuzz_bounds = {
        # The box keeps `ro > rm > ri` and `H` positive for both structures, which is
        # what `_inductance_factor`'s aspect ratio and `_theta_factor_integral`'s
        # `kappa` need to stay finite. `theta1_vv` stays well away from the
        # `cos + sin - 1 == 0` pole at 0 and 90 degrees.
        "h_coil": (7.0, 12.0),
        "ri_coil": (2.5, 3.5),
        "ro_coil": (11.0, 15.0),
        "rm_coil": (3.6, 5.0),
        "ccl_length_coil": (40.0, 60.0),
        "theta1_coil": (35.0, 55.0),
        "h_vv": (6.0, 10.0),
        "ri_vv": (3.5, 4.5),
        "ro_vv": (9.5, 12.5),
        "rm_vv": (4.6, 6.0),
        "theta1_vv": (0.5, 3.0),
        "n_tf_coils": (12.0, 20.0),
        "n_tf_coil_turns": (100.0, 300.0),
        "s_rp": (0.2, 0.9),
        "s_cc": (0.3, 1.0),
        "taud": (10.0, 30.0),
        "i_op": (4.0e4, 1.0e5),
        "d_vv": (0.06, 0.2),
    }


class TestVvStressQuenchFromBuild(Tier1Contract):
    """The whole method: the geometry prologue plus the surrogate, off a poisoned `data`.

    This is the case that checks the **reads**, and it is why `_VV_POISON` exists. The
    port turns twenty-nine `DataStructure` fields into the eighteen arguments above; an
    undeclared read, or `tfa[1]` in place of `tfa[0]`, produces NaN here rather than a
    plausible number.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_vv_stress_from_build
    ported = vv_stress_quench_from_build

    samples = [
        legacy_sample(
            "vv-stress-build-large-tokamak",
            z_tf_inside_half=8.5,
            dr_tf_inboard=1.2,
            r_tf_inboard_mid=3.0,
            r_tf_outboard_mid=13.0,
            r_tf_inboard_out=3.6,
            tfa_first_arc=1.5,
            z_plasma_xpoint_upper=5.6,
            dz_xpoint_divertor=0.6,
            dz_divertor=0.62,
            dz_shld_upper=0.6,
            dz_vv_upper=0.3,
            r_vv_inboard_out=4.2,
            dr_vv_outboard=0.3,
            dr_tf_outboard=1.2,
            dr_tf_shld_gap=0.05,
            dr_shld_thermal_outboard=0.05,
            dr_shld_vv_gap_outboard=0.163,
            len_tf_coil=50.0,
            theta1_coil=45.0,
            theta1_vv=1.0,
            n_tf_coils=16.0,
            n_tf_coil_turns=200.0,
            a_tf_coil_inboard_steel=0.5,
            a_tf_plasma_case=0.2,
            a_tf_coil_nose_case=0.35,
            dx_tf_side_case_average=0.03,
            t_tf_superconductor_quench=17.9728,
            c_tf_coil=1.4e7,
            dr_vv_shells=0.12,
        )
    ]

    fuzz_bounds = {
        # Same shape constraints as the core case, expressed in build fields: the
        # vessel's `ro_vv` is `r_tf_outboard_mid` minus five subtractions, so the
        # outboard radius is kept large and the five gaps small enough that it stays
        # above `rm_vv`.
        "z_tf_inside_half": (7.0, 10.0),
        "dr_tf_inboard": (0.9, 1.5),
        "r_tf_inboard_mid": (2.6, 3.4),
        "r_tf_outboard_mid": (12.0, 15.0),
        "r_tf_inboard_out": (3.3, 4.0),
        "tfa_first_arc": (1.0, 2.0),
        "z_plasma_xpoint_upper": (4.5, 6.5),
        "dz_xpoint_divertor": (0.4, 0.9),
        "dz_divertor": (0.4, 0.9),
        "dz_shld_upper": (0.4, 0.9),
        "dz_vv_upper": (0.2, 0.5),
        "r_vv_inboard_out": (4.0, 4.6),
        "dr_vv_outboard": (0.2, 0.4),
        "dr_tf_outboard": (0.9, 1.5),
        "dr_tf_shld_gap": (0.02, 0.1),
        "dr_shld_thermal_outboard": (0.02, 0.1),
        "dr_shld_vv_gap_outboard": (0.1, 0.25),
        "len_tf_coil": (40.0, 60.0),
        "theta1_coil": (35.0, 55.0),
        "theta1_vv": (0.5, 3.0),
        "n_tf_coils": (12.0, 20.0),
        "n_tf_coil_turns": (100.0, 300.0),
        "a_tf_coil_inboard_steel": (0.2, 0.9),
        "a_tf_plasma_case": (0.1, 0.4),
        "a_tf_coil_nose_case": (0.2, 0.6),
        "dx_tf_side_case_average": (0.01, 0.06),
        "t_tf_superconductor_quench": (10.0, 30.0),
        "c_tf_coil": (8.0e6, 2.0e7),
        "dr_vv_shells": (0.06, 0.2),
    }


# ---------------------------------------------------------------------------
# `tf_cable_in_conduit_superconductor_properties` -- constraint 33's producer
# ---------------------------------------------------------------------------

_SC_PROPERTIES_UNREAD = (
    "a_tf_turn_cable_space",
    "f_a_tf_turn_cable_space_cooling",
    "j_tf_wp",
    "f_strain_scale",
    "bcritsc",
    "tcritsc",
)
"""PROCESS arguments the arm under test does not consult, defaulted to NaN.

`tf_cable_in_conduit_superconductor_properties` takes fourteen arguments and no arm uses
all of them. The first four here are read **only** by the Bi-2212 branch
(`superconducting.py:2957-2968`), which is refused; `bcritsc`/`tcritsc` are read only by
arm 4, which supplies its own and so overrides the NaN. Defaulting the rest to NaN turns
"the port declares fewer reads than PROCESS's signature" into an executed claim: a port
that used any of them would return NaN.
"""


def _run_reference_sc_properties(i_tf_sc_mat, strain, **inputs):
    """The critical-current chain as a tuple in the port's return order.

    `i_tf_sc_mat` is the identity of the occupant under test, not an input; `strain` is
    planted in `.tfcoil.str_wp` with `i_str_wp = 1`, which is the only arm registered.
    `.tfcoil.str_tf_con_res` is left at NaN, so an occupant that read the *other* strain
    field would return NaN -- the `i_str_wp` axis is checked, not assumed.

    `.tfcoil.j_crit_str_tf` is a side-effect write rather than a return
    (`superconducting.py:2937`), so it is read back off `data` and spliced into the
    tuple at the position the port returns it.
    """
    model = _sctfcoil()
    t = model.data.tfcoil
    t.i_str_wp = 1
    t.str_wp = strain
    t.str_tf_con_res = np.nan
    t.j_crit_str_tf = np.nan
    t.b_crit_upper_nbti = inputs.pop("b_crit_upper_nbti", np.nan)
    t.t_crit_nbti = inputs.pop("t_crit_nbti", np.nan)

    result = CICCSuperconductingTFCoil.tf_cable_in_conduit_superconductor_properties(
        i_tf_superconductor=i_tf_sc_mat,
        data=model.data,
        **{**dict.fromkeys(_SC_PROPERTIES_UNREAD, np.nan), **inputs},
    )
    return (
        result.j_tf_wp_critical,
        t.j_crit_str_tf,
        result.f_c_tf_turn_operating_critical,
        result.j_tf_coil_turn,
        result.j_superconductor,
        result.c_turn_cables_critical,
        result.j_superconductor_critical,
        result.bc20m,
        result.tc0m,
    )


def _reference_sc_properties_iter_nb3sn(**inputs):
    """`i_tf_sc_mat = 1` -- `large_tokamak_eval.IN.DAT:374`'s arm."""
    return _run_reference_sc_properties(1, inputs.pop("strain"), **inputs)


def _ported_sc_properties_iter_nb3sn(**inputs):
    """Arm 1's occupant binding: `(32.97, 16.06)` as literals, not arguments.

    The pure function takes the pair because arm 4 reads it; arm 1 does not, and the
    literals are bound here rather than sampled so that the case checks the occupant's
    own binding and not merely the shared body.
    """
    return cicc_superconductor_properties_itersc(
        b_c20max=32.97, temp_c0max=16.06, **inputs
    )


def _reference_sc_properties_user_defined_nb3sn(*, bcritsc, tcritsc, **inputs):
    """`i_tf_sc_mat = 4` -- the ITER fit with the two constants read from input.

    The port's `b_c20max`/`temp_c0max` arguments are PROCESS's `bcritsc`/`tcritsc`
    reads, so the adapter takes the PROCESS spelling and hands the port its own.
    """
    return _run_reference_sc_properties(
        4, inputs.pop("strain"), bcritsc=bcritsc, tcritsc=tcritsc, **inputs
    )


def _ported_sc_properties_user_defined_nb3sn(*, bcritsc, tcritsc, **inputs):
    """`cicc_superconductor_properties_itersc` under arm 4's parameter names."""
    return cicc_superconductor_properties_itersc(
        b_c20max=bcritsc, temp_c0max=tcritsc, **inputs
    )


def _reference_sc_properties_wst_nb3sn(**inputs):
    """`i_tf_sc_mat = 5` -- `low_aspect_ratio_DEMO.IN.DAT:910`'s arm."""
    return _run_reference_sc_properties(5, inputs.pop("strain"), **inputs)


def _reference_sc_properties_lubell_nbti(**inputs):
    """`i_tf_sc_mat = 3` -- the arm with no strain read at all.

    `.tfcoil.str_wp` is NaN here as well as `.str_tf_con_res`: this arm reads neither,
    and poisoning both is what says so.
    """
    return _run_reference_sc_properties(3, np.nan, **inputs)


def _reference_sc_properties_durham_nbti(**inputs):
    """`i_tf_sc_mat = 7` -- the Durham GL NbTi arm.

    Ported here and refused for the temperature margin; see
    `TfSuperconductorTemperatureMargin`.
    """
    return _run_reference_sc_properties(7, inputs.pop("strain"), **inputs)


_SC_PROPERTIES_POINT = {
    "a_tf_turn_cable_space_effective": 1.1e-3,
    "a_tf_turn": 2.4e-3,
    "b_tf_inboard_peak": 8.0,
    "f_a_tf_turn_cable_copper": 0.69,
    "c_tf_turn": 85462.675,
    "temp_tf_coolant_peak_field": 4.75,
}
"""`large_tokamak_eval.IN.DAT`'s `c_tf_turn` (`:371`), `f_a_tf_turn_cable_copper`
(`tfcoil_variables.py:196`) and `tftmp` (`IN.DAT:378`), with a turn geometry of that
machine's order. The field is `8.0` T rather than the machine's ~12.5 T so the same
point can be shared with the two NbTi arms, whose fits are only defined well below
their upper critical field."""

_SC_PROPERTIES_BOUNDS = {
    # `1 - f_a_tf_turn_cable_copper` scales the critical current, so the copper fraction
    # is kept away from 1; the field stays below the NbTi arms' `b_c20max` so every
    # sample is inside every fit's own range.
    "a_tf_turn_cable_space_effective": (5.0e-4, 2.0e-3),
    "a_tf_turn": (1.5e-3, 4.0e-3),
    "b_tf_inboard_peak": (4.0, 10.0),
    "f_a_tf_turn_cable_copper": (0.5, 0.85),
    "c_tf_turn": (4.0e4, 1.2e5),
    "temp_tf_coolant_peak_field": (4.2, 5.5),
    "strain": (-0.004, -0.001),
}


class TestCiccSuperconductorPropertiesIterNb3sn(Tier1Contract):
    """`i_tf_sc_mat == 1` -- the reference machine's arm, and constraint 33's read."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_properties_iter_nb3sn
    ported = _ported_sc_properties_iter_nb3sn

    samples = [
        legacy_sample(
            "cicc-sc-properties-iter-nb3sn", strain=-0.003, **_SC_PROPERTIES_POINT
        )
    ]

    fuzz_bounds = _SC_PROPERTIES_BOUNDS


class TestCiccSuperconductorPropertiesWstNb3sn(Tier1Contract):
    """`i_tf_sc_mat == 5` -- `low_aspect_ratio_DEMO`'s arm. Literal `(32.97, 16.06)`."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_properties_wst_nb3sn
    ported = cicc_superconductor_properties_wst_nb3sn

    samples = [
        legacy_sample(
            "cicc-sc-properties-wst-nb3sn", strain=-0.003, **_SC_PROPERTIES_POINT
        )
    ]

    fuzz_bounds = _SC_PROPERTIES_BOUNDS


class TestCiccSuperconductorPropertiesUserDefinedNb3sn(Tier1Contract):
    """`i_tf_sc_mat == 4` -- two reads its sibling arm 1 turns into literals.

    Driven at a `(bcritsc, tcritsc)` that is **not** arm 1's `(32.97, 16.06)`, so an
    occupant that used the literals instead of the reads disagrees.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_properties_user_defined_nb3sn
    ported = _ported_sc_properties_user_defined_nb3sn

    samples = [
        legacy_sample(
            "cicc-sc-properties-user-defined-nb3sn",
            strain=-0.003,
            bcritsc=24.0,
            tcritsc=16.0,
            **_SC_PROPERTIES_POINT,
        )
    ]

    fuzz_bounds = {
        **_SC_PROPERTIES_BOUNDS,
        "bcritsc": (20.0, 28.0),
        "tcritsc": (14.0, 17.0),
    }


class TestCiccSuperconductorPropertiesOldLubellNbti(Tier1Contract):
    """`i_tf_sc_mat == 3` -- and the executed check that this arm reads no strain.

    Both strain fields are NaN in the adapter, so agreement is only possible for a port
    that does not read either.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_properties_lubell_nbti
    ported = cicc_superconductor_properties_lubell_nbti

    samples = [legacy_sample("cicc-sc-properties-lubell-nbti", **_SC_PROPERTIES_POINT)]

    fuzz_bounds = {
        name: bounds
        for name, bounds in _SC_PROPERTIES_BOUNDS.items()
        if name != "strain"
    }


class TestCiccSuperconductorPropertiesDurhamNbti(Tier1Contract):
    """`i_tf_sc_mat == 7` -- ported here, refused for the temperature margin.

    The asymmetry is the point of having this case: the critical-surface evaluation at a
    fixed temperature is exact, and only the *upward search* in
    `calculate_superconductor_temperature_margin` leaves the reals. A passing case here
    is what makes that a scoped refusal rather than a blanket one.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_sc_properties_durham_nbti
    ported = cicc_superconductor_properties_durham_nbti

    samples = [
        legacy_sample(
            "cicc-sc-properties-durham-nbti",
            strain=-0.003,
            b_crit_upper_nbti=14.86,
            t_crit_nbti=9.04,
            **_SC_PROPERTIES_POINT,
        )
    ]

    fuzz_bounds = {
        **_SC_PROPERTIES_BOUNDS,
        # Below `b_crit_upper_nbti`'s own value throughout, so `gl_nbti` stays on the
        # real branch at every corner.
        "b_tf_inboard_peak": (3.0, 7.0),
        "b_crit_upper_nbti": (13.0, 16.0),
        "t_crit_nbti": (8.5, 9.5),
    }


# ---------------------------------------------------------------------------
# `calculate_superconductor_temperature_margin` -- constraint 36's producer
# ---------------------------------------------------------------------------

_MARGIN_UNREAD = ("dr_tf_hts_tape", "dx_tf_hts_tape_rebco", "dx_tf_hts_tape_total")
"""The three `.superconducting_tfcoil` tape dimensions
`calculate_superconductor_temperature_margin` packs into `arguments` on **every** arm
(`superconducting.py:1236-1256`) and that only branch 9 of
`superconductor_current_density_margin` consumes. NaN-poisoned, because the four ported
arms declare none of them and that claim is worth executing: PROCESS passing a value is
not the same as PROCESS reading it."""


def _run_reference_temperature_margin(i_tf_sc_mat, **inputs):
    """`calculate_superconductor_temperature_margin`, returned as `(margin, margin)`.

    The port owns two variables holding the same number --
    `.tfcoil.temp_tf_superconductor_margin` (`run`'s assignment) and
    `.tfcoil.temp_margin` (the function's own side-effect write, `:1279`) -- so the
    reference returns the pair
    the same way, with the second read back off `data`. A port that returned only one, or
    that wrote the side effect from a different quantity, disagrees.
    """
    model = _sctfcoil()
    d = model.data.superconducting_tfcoil
    for name in _MARGIN_UNREAD:
        setattr(d, name, np.nan)
    model.data.tfcoil.temp_margin = np.nan

    margin = CICCSuperconductingTFCoil.calculate_superconductor_temperature_margin(
        i_tf_superconductor=i_tf_sc_mat,
        c0=1.0e10,
        data=model.data,
        **inputs,
    )
    return (margin, model.data.tfcoil.temp_margin)


def _reference_margin_iter_nb3sn(*, b_c20max, temp_c0max, **inputs):
    """`i_tf_sc_mat = 1` *(live)*."""
    return _run_reference_temperature_margin(
        1, bc20m=b_c20max, tc0m=temp_c0max, **inputs
    )


def _reference_margin_wst_nb3sn(*, b_c20max, temp_c0max, **inputs):
    """`i_tf_sc_mat = 5`."""
    return _run_reference_temperature_margin(
        5, bc20m=b_c20max, tc0m=temp_c0max, **inputs
    )


def _reference_margin_lubell_nbti(*, b_c20max, temp_c0max, **inputs):
    """`i_tf_sc_mat = 3` -- no strain, and `c0` is the literal `run` passes.

    `strain` is still a *parameter* of PROCESS's function, so the adapter has to supply
    one; NaN says the branch does not consume it, which is the same claim the port makes
    by not taking the argument.
    """
    return _run_reference_temperature_margin(
        3, bc20m=b_c20max, tc0m=temp_c0max, strain=np.nan, **inputs
    )


def _ported_margin_pair(fit):
    """`fit(...) -> (margin, margin)`, the pair the occupant owns."""

    def ported(**inputs):
        margin = fit(**inputs)
        return (margin, margin)

    return ported


_MARGIN_POINT = {
    "j_superconductor": 6.0e8,
    "b_tf_inboard_peak": 8.0,
    "temp_tf_coolant_peak_field": 4.75,
}
"""A superconductor current density and field of `large_tokamak_eval`'s order, at that
file's own `tftmp`. `j_superconductor` is the properties node's own output on the arm
above, so the two cases are consecutive links of one chain rather than two unrelated
points."""


class TestTemperatureMarginIterNb3sn(Tier1Contract):
    """`i_tf_sc_mat == 1` -- the live arm, and the port's first *differentiated* solve.

    The gradient half of this case is the one that matters. `solve_current_sharing_
    temperature` is a fifty-trip `fori_loop` whose carry collapses to a flat state once
    converged; if that collapse were done by masking instead, the tangent would be `nan`
    and the value test would still pass. Measured on `i_tf_sc_mat = 4` before the fix,
    and the reason the loop is written the way it is.
    """

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_margin_iter_nb3sn
    ported = _ported_margin_pair(temperature_margin_itersc)

    samples = [
        legacy_sample(
            "temp-margin-iter-nb3sn",
            strain=-0.003,
            b_c20max=32.97,
            temp_c0max=16.06,
            **_MARGIN_POINT,
        )
    ]

    fuzz_bounds = {
        # The box keeps the current-sharing temperature strictly between `tftmp` and
        # `temp_c0max`, i.e. a real root the secant can reach: too high a
        # `j_superconductor` and the margin goes negative, which PROCESS logs and this
        # harness has no reference for.
        "j_superconductor": (2.0e8, 9.0e8),
        "b_tf_inboard_peak": (5.0, 11.0),
        "strain": (-0.004, -0.001),
        "b_c20max": (30.0, 35.0),
        "temp_c0max": (15.0, 17.0),
        "temp_tf_coolant_peak_field": (4.2, 5.5),
    }


class TestTemperatureMarginWstNb3sn(Tier1Contract):
    """`i_tf_sc_mat == 5` -- `low_aspect_ratio_DEMO`'s arm."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_margin_wst_nb3sn
    ported = _ported_margin_pair(temperature_margin_wst_nb3sn)

    samples = [
        legacy_sample(
            "temp-margin-wst-nb3sn",
            strain=-0.003,
            b_c20max=32.97,
            temp_c0max=16.06,
            **_MARGIN_POINT,
        )
    ]

    fuzz_bounds = TestTemperatureMarginIterNb3sn.fuzz_bounds


class TestTemperatureMarginOldLubellNbti(Tier1Contract):
    """`i_tf_sc_mat == 3` -- no strain read, and `c0 = 1.0e10` as a literal."""

    audit_record = "models/tfcoil/superconducting.md"
    reference = _reference_margin_lubell_nbti
    ported = _ported_margin_pair(temperature_margin_lubell_nbti)

    samples = [
        legacy_sample(
            "temp-margin-lubell-nbti",
            b_c20max=15.0,
            temp_c0max=9.3,
            **{**_MARGIN_POINT, "j_superconductor": 1.2e9},
        )
    ]

    fuzz_bounds = {
        "j_superconductor": (6.0e8, 2.0e9),
        "b_tf_inboard_peak": (3.0, 7.0),
        "b_c20max": (14.0, 16.0),
        "temp_c0max": (9.0, 9.6),
        "temp_tf_coolant_peak_field": (4.2, 5.5),
    }


# ---------------------------------------------------------------------------
# The two composite-key slots: total, and refused end to end
# ---------------------------------------------------------------------------


def test_the_two_superconductor_slots_are_total():
    """Every `(i_str_wp, i_tf_sc_mat)` pair either has an occupant or a recorded reason.

    `test_machine.py::test_a_refused_value_says_why` cannot reach these two keys -- their
    *value* is a pair, so no IN.DAT line selects one, and they are in
    `DERIVED_UNPORTED_KEYS`. This is what that skip trades against: totality over the
    full 2 x 9 product for both slots, checked against the same `UNPORTED` dict, so a
    value can neither be silently absent nor silently carry two answers.
    """
    for field, registry in (
        ("i_str_wp_i_tf_sc_mat_cicc_sc_properties", CICC_SUPERCONDUCTOR_PROPERTIES),
        ("i_str_wp_i_tf_sc_mat_temp_margin", TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN),
    ):
        for i_str_wp in (0, 1):
            for material in range(1, 10):
                key = (i_str_wp, material)
                ported = key in registry
                refused = (field, key) in UNPORTED
                assert ported != refused, f"{field} {key}: " + (
                    "has both an occupant and an UNPORTED reason"
                    if ported
                    else "has neither an occupant nor an UNPORTED reason"
                )


def test_i_str_wp_zero_is_refused_end_to_end(tmp_path):
    """A machine that sets `i_str_wp = 0` stops, and the message says which strain.

    The point of registering the unwritten arm rather than baking `.tfcoil.str_wp` into
    every occupant: `0` is a real PROCESS branch that reads the *other* field, and a port
    that silently kept reading `str_wp` would produce a wrong critical current with no
    signal at all. This is that signal, executed.
    """
    indat = tmp_path / "TOK.DAT"
    indat.write_text(
        "".join(
            f"{f} = {v if isinstance(v, str) else int(v)}\n"
            for f, v in {**TOKAMAK_BASELINE_INDAT, "i_str_wp": 0}.items()
        )
    )
    with pytest.raises(NotImplementedError, match="i_str_wp_i_tf_sc_mat"):
        machine_from_indat(str(indat))
