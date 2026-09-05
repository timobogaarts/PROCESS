"""Harness cases for the ported CroCo TF coil layer
(`functional_process/cottax/tfcoil/croco.py`).

Same convention as `test_superconducting.py`: each legacy sample is an input point taken
from PROCESS's own `tests/unit/models/test_superconductors.py` where one exists, and
otherwise from a self-consistent CroCo turn built by running PROCESS's own chain forward
from `spherical_tokamak_eval.IN.DAT`'s tape and cable thicknesses -- so every sample is a
point PROCESS itself can be at, not a set of independently plausible numbers.

**Three of the seven adapters slice PROCESS's return**, and the slice is the port's
subject rather than a convenience: `run` overwrites those outputs before anything reads
them (`croco.py`'s module docstring tabulates all five dead writes), so the ported
function does not compute them and the reference is projected onto what does survive.
The arguments belonging to the dropped half are held at the sample's own values, so the
projection is a slice and not a different point.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.indat import (
    CROCO_SUPERCONDUCTOR_PROPERTIES,
    CROCO_TEMPERATURE_MARGIN,
    CROCO_TURN_GEOMETRY,
    UNPORTED,
    machine_from_indat,
)
from functional_process.cottax.tfcoil.croco import (
    croco_averaged_turn_geometry_from_current_per_turn,
    croco_cable_geometry,
    croco_cable_space_properties,
    croco_inboard_areas_and_fractions,
    croco_superconductor_properties_hijc_rebco,
    croco_turn_cable_space_cooling_fraction,
    croco_turn_cable_space_extra_void,
    temperature_margin_hijc_rebco,
)
from functional_process.cottax.tfcoil.namespace import (
    CiccSuperconductingTfCoil,
    CrocoSuperconductingTfCoil,
    SuperconductingTfCoil,
)
from process.core.model import DataStructure
from process.models.superconductors import (
    SuperconductorModel,
    calculate_croco_cable_geometry,
)
from process.models.tfcoil.superconducting import CROCOSuperconductingTFCoil

_HAZELTON_ZHAI_REBCO = 9


def _croco():
    """A `CROCOSuperconductingTFCoil` with a fresh `DataStructure` attached."""
    model = CROCOSuperconductingTFCoil()
    model.data = DataStructure()
    return model


# ---------------------------------------------------------------------------
# `tf_croco_averaged_turn_geometry`
# ---------------------------------------------------------------------------


def _reference_croco_averaged_turn_geometry(
    j_tf_wp,
    c_tf_turn,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    layer_ins,
    a_tf_wp_no_insulation,
):
    """The both-flags-`False` arm, sliced to the seven outputs the port owns.

    `a_tf_turn_cable_space_no_void` and `a_tf_turn_steel` are dropped: PROCESS returns
    the first straight back off `data` (`superconducting.py:4379`) and computes the
    second from it (`:4374-4376`), and `run` overwrites both from
    `tf_turn_croco_cable_space_properties` before any reader
    (`:3849`, `:3855`). `c_tf_turn` is an argument because it is read; it is not in the
    tuple because on this arm PROCESS returns it unchanged.
    """
    result = _croco().tf_croco_averaged_turn_geometry(
        j_tf_wp=j_tf_wp,
        dx_tf_turn_steel=dx_tf_turn_steel,
        dx_tf_turn_insulation=dx_tf_turn_insulation,
        dx_tf_turn_general=0.0,
        c_tf_turn=c_tf_turn,
        i_dx_tf_turn_general_input=False,
        i_dx_tf_turn_cable_space_general_input=False,
        dx_tf_turn_cable_space_general=0.0,
        layer_ins=layer_ins,
        a_tf_wp_no_insulation=a_tf_wp_no_insulation,
    )
    return (
        result.a_tf_turn_insulation,
        result.n_tf_coil_turns,
        result.dx_tf_turn_general,
        result.dr_tf_turn,
        result.dx_tf_turn,
        result.dx_tf_turn_conduit_full_average,
        result.dx_tf_turn_cable_space_average,
    )


class TestCrocoAveragedTurnGeometryFromCurrentPerTurn(Tier1Contract):
    """`i_dx_tf_turn_general_input == i_dx_tf_turn_cable_space_general_input == False`,
    which is PROCESS's default and both tracked CroCo files' arm.
    """

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_averaged_turn_geometry
    ported = croco_averaged_turn_geometry_from_current_per_turn

    samples = [
        legacy_sample(
            "croco-averaged-turn-geometry",
            j_tf_wp=26493137.688284047,
            c_tf_turn=85462.674970907982,
            dx_tf_turn_steel=8.0e-3,
            dx_tf_turn_insulation=8.0e-4,
            layer_ins=0.0,
            a_tf_wp_no_insulation=0.60510952642236249,
        ),
    ]

    fuzz_bounds = {
        "j_tf_wp": (1e7, 4e7),
        "c_tf_turn": (4e4, 1.5e5),
        "dx_tf_turn_steel": (4e-3, 1.2e-2),
        "dx_tf_turn_insulation": (4e-4, 2e-3),
        "layer_ins": (0.0, 2e-3),
        "a_tf_wp_no_insulation": (0.3, 1.2),
    }


# ---------------------------------------------------------------------------
# `tf_turn_croco_cable_space_properties`
# ---------------------------------------------------------------------------


def _reference_croco_cable_space_properties(
    dx_tf_turn_conduit_full_average, dx_tf_turn_steel
):
    """Sliced to four of five: `f_a_tf_turn_cable_space_cooling` is overwritten by
    `run`'s inline block (`superconducting.py:3948`) before any reader, and is an area
    on this line where its replacement is a fraction.
    """
    result = CROCOSuperconductingTFCoil.tf_turn_croco_cable_space_properties(
        dx_tf_turn_conduit_full_average=dx_tf_turn_conduit_full_average,
        dx_tf_turn_steel=dx_tf_turn_steel,
    )
    return (
        result.dia_tf_turn_croco_cable,
        result.a_tf_turn_cable_space_no_void,
        result.a_tf_turn_cable_space_effective,
        result.a_tf_turn_steel,
    )


class TestCrocoCableSpaceProperties(Tier1Contract):
    """Seven circles in a `3d x 3d` square. No switch."""

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_cable_space_properties
    ported = croco_cable_space_properties

    samples = [
        legacy_sample(
            "croco-cable-space",
            dx_tf_turn_conduit_full_average=0.055166861225577248,
            dx_tf_turn_steel=8.0e-3,
        ),
    ]

    fuzz_bounds = {
        "dx_tf_turn_conduit_full_average": (0.03, 0.09),
        "dx_tf_turn_steel": (4e-3, 1.0e-2),
    }


# ---------------------------------------------------------------------------
# `calculate_croco_cable_geometry`
# ---------------------------------------------------------------------------


def _reference_croco_cable_geometry(
    dia_croco_strand,
    dx_croco_strand_copper,
    dx_hts_tape_rebco,
    dx_hts_tape_copper,
    dx_hts_tape_hastelloy,
):
    """PROCESS's dataclass unpacked in `run`'s own write order
    (`superconducting.py:3866-3888`), which is also the dataclass's field order.
    """
    result = calculate_croco_cable_geometry(
        dia_croco_strand=dia_croco_strand,
        dx_croco_strand_copper=dx_croco_strand_copper,
        dx_hts_tape_rebco=dx_hts_tape_rebco,
        dx_hts_tape_copper=dx_hts_tape_copper,
        dx_hts_tape_hastelloy=dx_hts_tape_hastelloy,
    )
    return (
        result.dia_croco_strand_tape_region,
        result.n_croco_strand_hts_tapes,
        result.a_croco_strand_copper_total,
        result.a_croco_strand_hastelloy,
        result.a_croco_strand_solder,
        result.a_croco_strand_rebco,
        result.a_croco_strand,
        result.dr_hts_tape,
        result.dx_hts_tape_total,
        result.dx_croco_strand_tape_stack,
    )


class TestCrocoCableGeometry(Tier1Contract):
    """One CroCo strand: a copper tube around a soldered stack of REBCO tapes.

    **Tier 1, with the gradient checks structurally excused**, and the excuse is the one
    `test_cs_fatigue.py::TestNCycle` already records: PROCESS's finite difference is not
    a derivative here. `n_croco_strand_hts_tapes` is `np.floor(stack / tape_thickness)`
    (`process/models/superconductors.py:1157-1159`), so four of the ten outputs are
    piecewise constant in every input, and at `epsfcn = 1e-3` PROCESS's perturbation
    crosses **whole tape steps**: on the 10 mm sample below the stack holds 959 tapes and
    one perturbation of `dia_croco_strand` moves the count by about one, which PROCESS
    reports as a slope of `1e5` per metre where the true derivative is `0`. That is a
    property of PROCESS's discretisation, not of this port's arithmetic, so
    `static_argnames` names every argument and `diff_argnames` comes out empty --
    `TestNCycle`'s hammer, for `TestNCycle`'s reason.

    **The excuse is a precaution, and it is only that.** Measured at the CroCo strand
    `spherical_tokamak_eval.IN.DAT` actually describes -- 30.88 tapes, comfortably
    mid-step -- `jax.jacfwd` of the port agrees with a central finite difference of
    PROCESS's own function to between **1.0e-10 and 4.0e-9 relative** across all five
    inputs and all ten outputs. `test_croco_cable_geometry_gradient_within_one_step`
    below is that measurement, kept as a test rather than as a claim in prose.

    Legacy samples are `test_calculate_croco_cable_geometry`'s two parametrised cases
    verbatim, plus the strand the ST input files actually describe.
    """

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_cable_geometry
    ported = croco_cable_geometry

    static_argnames = (
        "dia_croco_strand",
        "dx_croco_strand_copper",
        "dx_hts_tape_rebco",
        "dx_hts_tape_copper",
        "dx_hts_tape_hastelloy",
    )

    samples = [
        legacy_sample(
            "croco-cable-10mm",
            dia_croco_strand=0.010,
            dx_croco_strand_copper=0.001,
            dx_hts_tape_rebco=1e-6,
            dx_hts_tape_copper=2e-6,
            dx_hts_tape_hastelloy=3e-6,
        ),
        legacy_sample(
            "croco-cable-baseline",
            dia_croco_strand=0.0054,
            dx_croco_strand_copper=0.0005,
            dx_hts_tape_rebco=1e-6,
            dx_hts_tape_copper=2e-6,
            dx_hts_tape_hastelloy=3e-6,
        ),
        legacy_sample(
            # `spherical_tokamak_eval.IN.DAT:73-76`'s tape and tube, on the strand
            # diameter the sampled turn above produces.
            "croco-cable-spherical-tokamak",
            dia_croco_strand=0.013055620408525749,
            dx_croco_strand_copper=2.0e-3,
            dx_hts_tape_rebco=1.0e-6,
            dx_hts_tape_copper=2.0e-4,
            dx_hts_tape_hastelloy=1e-5,
        ),
    ]

    fuzz_bounds = {
        "dia_croco_strand": (8e-3, 1.6e-2),
        "dx_croco_strand_copper": (1.0e-3, 2.4e-3),
        "dx_hts_tape_rebco": (8e-7, 1.2e-6),
        "dx_hts_tape_copper": (1.8e-4, 2.2e-4),
        "dx_hts_tape_hastelloy": (9e-6, 1.1e-5),
    }


# ---------------------------------------------------------------------------
# `run`'s literal `f_a_tf_turn_cable_space_extra_void = 0.0`
# ---------------------------------------------------------------------------


def test_croco_turn_cable_space_extra_void_is_zero():
    """`superconducting.py:3895`, one literal assignment, transcribed.

    Not a `Tier1Contract`: there is no PROCESS callable to bind and no input to sample.
    The check that matters is that the *node* owns the field, which
    `test_boundary.py`/`test_machine.py` make; this only pins the constant.
    """
    assert croco_turn_cable_space_extra_void() == 0.0


# ---------------------------------------------------------------------------
# `tf_croco_inboard_areas_and_fractions`
# ---------------------------------------------------------------------------


def _reference_croco_inboard_areas_and_fractions(
    a_tf_turn_cable_space_no_void,
    n_tf_coil_turns,
    f_a_tf_turn_cable_space_extra_void,
    a_tf_turn_insulation,
    a_tf_turn_steel,
    a_tf_coil_inboard_case,
    n_tf_coils,
    a_tf_inboard_total,
    a_tf_wp_ground_insulation,
    a_tf_croco_strand,
):
    """PROCESS's `SuperconTFAreasFractions`, unpacked whole -- no slice here."""
    result = CROCOSuperconductingTFCoil.tf_croco_inboard_areas_and_fractions(
        a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
        n_tf_coil_turns=n_tf_coil_turns,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        a_tf_turn_insulation=a_tf_turn_insulation,
        a_tf_turn_steel=a_tf_turn_steel,
        a_tf_coil_inboard_case=a_tf_coil_inboard_case,
        n_tf_coils=n_tf_coils,
        a_tf_inboard_total=a_tf_inboard_total,
        a_tf_wp_ground_insulation=a_tf_wp_ground_insulation,
        a_tf_croco_strand=a_tf_croco_strand,
    )
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


class TestCrocoInboardAreasAndFractions(Tier1Contract):
    """The nine inboard areas and fractions. No switch."""

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_inboard_areas_and_fractions
    ported = croco_inboard_areas_and_fractions

    samples = [
        legacy_sample(
            "croco-inboard-areas",
            a_tf_turn_cable_space_no_void=1.2044932391216682e-3,
            n_tf_coil_turns=187.6247107030811,
            # PROCESS sets this to exactly zero three statements earlier.
            f_a_tf_turn_cable_space_extra_void=0.0,
            a_tf_turn_insulation=1.9235290952697042e-4,
            a_tf_turn_steel=1.8384107608783318e-3,
            a_tf_coil_inboard_case=1.0015,
            n_tf_coils=12.0,
            a_tf_inboard_total=27.308,
            a_tf_wp_ground_insulation=0.028,
            a_tf_croco_strand=1.3387e-4,
        ),
    ]

    fuzz_bounds = {
        "a_tf_turn_cable_space_no_void": (5e-4, 3e-3),
        "n_tf_coil_turns": (50.0, 400.0),
        "f_a_tf_turn_cable_space_extra_void": (0.0, 0.1),
        "a_tf_turn_insulation": (5e-5, 5e-4),
        "a_tf_turn_steel": (5e-4, 4e-3),
        "a_tf_coil_inboard_case": (0.3, 3.0),
        "n_tf_coils": (8.0, 24.0),
        "a_tf_inboard_total": (5.0, 40.0),
        "a_tf_wp_ground_insulation": (0.005, 0.1),
        "a_tf_croco_strand": (5e-5, 3e-4),
    }


# ---------------------------------------------------------------------------
# `run`'s inline cooling fraction
# ---------------------------------------------------------------------------


def _reference_croco_cooling_fraction(a_tf_turn_cable_space_no_void, a_tf_croco_strand):
    """`process/models/tfcoil/superconducting.py:3948-3955`, transcribed.

    Inline in `run`, so there is no callable to bind -- the same treatment
    `test_superconducting.py::_reference_a_tf_turn` gives `run`'s other inline
    statement. `a_tf_turn_croco_copper_bar` is `a_tf_croco_strand` (`:3930`), which is
    why the second argument appears twice below and only once in the signature.
    """
    n_croco_strands_turn = 6
    return (
        a_tf_turn_cable_space_no_void
        - ((n_croco_strands_turn * a_tf_croco_strand) - a_tf_croco_strand)
    ) / a_tf_turn_cable_space_no_void


class TestCrocoTurnCableSpaceCoolingFraction(Tier1Contract):
    """The one line of `run`'s inline copper block any computation reads."""

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_cooling_fraction
    ported = croco_turn_cable_space_cooling_fraction

    samples = [
        legacy_sample(
            "croco-cooling-fraction",
            a_tf_turn_cable_space_no_void=1.2044932391216682e-3,
            a_tf_croco_strand=1.3387e-4,
        ),
    ]

    fuzz_bounds = {
        "a_tf_turn_cable_space_no_void": (5e-4, 3e-3),
        "a_tf_croco_strand": (5e-5, 1.5e-4),
    }


# ---------------------------------------------------------------------------
# `tf_croco_superconductor_properties`
# ---------------------------------------------------------------------------


def _reference_croco_superconductor_properties(
    a_tf_turn,
    b_tf_inboard_peak,
    cur_tf_turn,
    temp_tf_peak,
    dr_tf_hts_tape,
    dx_tf_hts_tape_rebco,
    dx_tf_hts_tape_total,
    a_tf_croco_strand,
):
    """`i_tf_sc_mat == 9`, with `.tfcoil.j_crit_str_tf` read back off `data`.

    PROCESS returns a `TFSuperconductorLimits` and stores `j_crit_str_tf` and
    `temp_margin` on `data.tfcoil` (`superconducting.py:4508`, `:4546`). The port owns
    the first and not the second: `run` overwrites `temp_margin` from
    `calculate_superconductor_temperature_margin` (`:1278`) before any reader, so the
    `current_sharing_rebco` solve behind it is dead on this path and is not ported.

    `i_str_wp` is set on `data` because the function reads it to pick a strain field
    (`:4443-4446`); the strain itself reaches nothing on this arm, `hijc_rebco` having
    no strain argument.
    """
    model = _croco()
    model.data.tfcoil.i_str_wp = 1
    model.data.tfcoil.str_wp = 0.0
    result = model.tf_croco_superconductor_properties(
        a_tf_turn=a_tf_turn,
        b_tf_inboard_peak=b_tf_inboard_peak,
        cur_tf_turn=cur_tf_turn,
        temp_tf_peak=temp_tf_peak,
        i_tf_superconductor=_HAZELTON_ZHAI_REBCO,
        dr_tf_hts_tape=dr_tf_hts_tape,
        dx_tf_hts_tape_rebco=dx_tf_hts_tape_rebco,
        dx_tf_hts_tape_total=dx_tf_hts_tape_total,
        a_tf_croco_strand=a_tf_croco_strand,
    )
    return (
        result.j_tf_wp_critical,
        model.data.tfcoil.j_crit_str_tf,
        result.f_c_tf_turn_operating_critical,
        result.j_tf_coil_turn,
        result.j_superconductor,
        result.c_turn_cables_critical,
        result.c_turn_cables_critical,
        result.j_superconductor_critical,
        result.bc20m,
        result.tc0m,
    )


class TestHazeltonZhaiRebcoCrocoSuperconductorProperties(Tier1Contract):
    """`i_tf_sc_mat == 9` *(live on both tracked ST files)*.

    `bc20m`/`tc0m` are literals on this branch, so they are constants of the comparison
    rather than sampled arguments -- the same shape
    `TestIterNb3snCiccSuperconductorProperties` has for `(32.97, 16.06)`.
    """

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_croco_superconductor_properties
    ported = croco_superconductor_properties_hijc_rebco

    samples = [
        legacy_sample(
            "croco-sc-properties-rebco9",
            a_tf_turn=3.2258669147172787e-3,
            b_tf_inboard_peak=11.717722779177526,
            cur_tf_turn=85462.674970907982,
            temp_tf_peak=4.75,
            dr_tf_hts_tape=6.2886208094437651e-3,
            dx_tf_hts_tape_rebco=1.0e-6,
            dx_tf_hts_tape_total=2.11e-4,
            a_tf_croco_strand=1.3387e-4,
        ),
    ]

    fuzz_bounds = {
        "a_tf_turn": (1e-3, 8e-3),
        "b_tf_inboard_peak": (4.0, 20.0),
        "cur_tf_turn": (4e4, 1.5e5),
        "temp_tf_peak": (4.0, 20.0),
        "dr_tf_hts_tape": (3e-3, 8e-3),
        "dx_tf_hts_tape_rebco": (8e-7, 1.2e-6),
        "dx_tf_hts_tape_total": (1e-4, 3e-4),
        "a_tf_croco_strand": (5e-5, 3e-4),
    }


# ---------------------------------------------------------------------------
# `calculate_superconductor_temperature_margin`, tape arm 9
# ---------------------------------------------------------------------------


def _reference_temperature_margin_hijc_rebco(
    j_superconductor,
    b_tf_inboard_peak,
    b_c20max,
    temp_c0max,
    dr_hts_tape,
    dx_hts_tape_rebco,
    dx_hts_tape_total,
    temp_tf_coolant_peak_field,
):
    """Branch 9 of `superconductor_current_density_margin`, driven by PROCESS's own
    `scipy.optimize.newton` secant search.

    The three tape dimensions reach the residual off `data.superconducting_tfcoil`
    (`superconducting.py:1245-1253`), not as arguments, so the adapter sets them there.
    `strain` is passed and unused: `hijc_rebco` takes none.
    """
    model = _croco()
    d = model.data.superconducting_tfcoil
    d.dr_tf_hts_tape = dr_hts_tape
    d.dx_tf_hts_tape_rebco = dx_hts_tape_rebco
    d.dx_tf_hts_tape_total = dx_hts_tape_total
    return model.calculate_superconductor_temperature_margin(
        i_tf_superconductor=_HAZELTON_ZHAI_REBCO,
        j_superconductor=j_superconductor,
        b_tf_inboard_peak=b_tf_inboard_peak,
        strain=0.0,
        bc20m=b_c20max,
        tc0m=temp_c0max,
        c0=1.0e10,
        temp_tf_coolant_peak_field=temp_tf_coolant_peak_field,
        data=model.data,
    )


class TestHazeltonZhaiRebcoCrocoTemperatureMargin(Tier1Contract):
    """`i_tf_sc_mat == 9` -- constraint 36's read on a CroCo machine.

    The port's replicated secant search (`solve_current_sharing_temperature`) against
    `scipy.optimize.newton`'s, on a residual that takes three tape dimensions and no
    strain. `b_c20max`/`temp_c0max` come from the properties node above, so they are
    sampled rather than fixed.
    """

    audit_record = "models/tfcoil/croco.md"
    reference = _reference_temperature_margin_hijc_rebco
    ported = temperature_margin_hijc_rebco

    samples = [
        legacy_sample(
            "croco-temp-margin-rebco9",
            j_superconductor=1.0e8,
            b_tf_inboard_peak=11.717722779177526,
            b_c20max=138.0,
            temp_c0max=92.0,
            dr_hts_tape=6.2886208094437651e-3,
            dx_hts_tape_rebco=1.0e-6,
            dx_hts_tape_total=2.11e-4,
            temp_tf_coolant_peak_field=4.75,
        ),
    ]

    fuzz_bounds = {
        "j_superconductor": (5e7, 2e8),
        "b_tf_inboard_peak": (8.0, 16.0),
        "b_c20max": (130.0, 146.0),
        "temp_c0max": (88.0, 96.0),
        "dr_hts_tape": (5e-3, 8e-3),
        "dx_hts_tape_rebco": (8e-7, 1.2e-6),
        "dx_hts_tape_total": (1.8e-4, 2.4e-4),
        "temp_tf_coolant_peak_field": (4.0, 6.0),
    }


# ---------------------------------------------------------------------------
# The registries, and what a CroCo machine assembles to
# ---------------------------------------------------------------------------


def test_croco_and_cicc_registries_partition_the_material_switch():
    """Every `i_tf_sc_mat` is an occupant of exactly one of the two properties slots.

    The two PROCESS functions guard on `SuperconductorShape` in their first four lines
    and take opposite answers, so a material with an occupant in both would mean one of
    the two registries had been written against a branch that raises. Checked rather
    than asserted in a docstring, because the CICC side's `_SC_TAPE_REASON` and the CroCo
    side's `_SC_CABLE_REASON` are two statements of one partition.
    """
    from functional_process.cottax.indat import CICC_SUPERCONDUCTOR_PROPERTIES

    cicc = {mat for _, mat in CICC_SUPERCONDUCTOR_PROPERTIES}
    croco = {mat for _, mat in CROCO_SUPERCONDUCTOR_PROPERTIES}
    assert not (cicc & croco)
    assert all(mat.sc_shape.name == "CABLE" for mat in cicc), (
        "a tape material has a cable-in-conduit occupant"
    )
    assert all(mat.sc_shape.name == "TAPE" for mat in croco), (
        "a cable material has a CroCo occupant"
    )


@pytest.mark.parametrize(
    "slot",
    [
        "i_str_wp_i_tf_sc_mat_croco_sc_properties",
        "i_str_wp_i_tf_sc_mat_croco_temp_margin",
    ],
)
def test_every_unwritten_croco_material_is_refused_with_a_reason(slot):
    """Every `(i_str_wp, i_tf_sc_mat)` pair has an occupant **or** a recorded reason.

    `test_machine.py::test_a_refused_value_says_why` cannot reach these two keys -- their
    *value* is a pair, so no IN.DAT line selects one, and they are in
    `DERIVED_UNPORTED_KEYS`. This is what that skip trades against, and it is the same
    trade `test_superconducting.py::test_the_two_superconductor_slots_are_total` makes
    for the cable-in-conduit pair: totality over the full 2 x 9 product, so a value can
    neither be silently absent nor silently carry two answers, and `_slot_occupant`'s
    "not a known value" branch stays reachable only for a genuine typo.
    """
    registry = (
        CROCO_SUPERCONDUCTOR_PROPERTIES
        if slot.endswith("sc_properties")
        else CROCO_TEMPERATURE_MARGIN
    )
    for i_str_wp in (0, 1):
        for mat in SuperconductorModel:
            key = (i_str_wp, mat)
            ported = key in registry
            refused = (slot, key) in UNPORTED
            assert ported != refused, f"{slot} {key}: " + (
                "has both an occupant and an UNPORTED reason"
                if ported
                else "has neither an occupant nor an UNPORTED reason"
            )


def test_a_croco_machine_refuses_an_unwritten_tape_material(tmp_path):
    """A CroCo machine asking for `i_tf_sc_mat = 8` stops, and says which slot.

    The end-to-end half of the trade above, and the check that the *CroCo* registries are
    the ones a CroCo machine consults: before this wave the same file was refused by the
    cable-in-conduit slot's `_SC_TAPE_REASON`, which was catching the two ST files by
    accident (`indat._refuse_unported_switch`'s docstring records that history).
    """
    from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE

    text = pathlib.Path(TOKAMAK_INPUT_FILE).read_text()
    text = "\n".join(
        "i_tf_sc_mat = 8" if line.startswith("i_tf_sc_mat") else line
        for line in text.splitlines()
    )
    indat = tmp_path / "croco8.IN.DAT"
    indat.write_text(text + "\ni_tf_turn_type = 2\n")

    with pytest.raises(NotImplementedError, match="i_str_wp_i_tf_sc_mat_croco"):
        machine_from_indat(str(indat))


def test_croco_turn_geometry_refuses_integer_turns_quoting_process():
    """Arm `1` is PROCESS's own `ProcessValueError`, not this port's gap, and the
    recorded reason has to say so -- otherwise the next reader budgets for a port that
    cannot be written.
    """
    assert 1 not in CROCO_TURN_GEOMETRY
    assert "PROCESS refuses it" in UNPORTED["croco_turn_geometry_arm", 1]


def test_croco_namespace_is_a_sibling_of_the_cable_in_conduit_one():
    """Both fill the same slot, so they must share a base -- and the base must carry the
    slots `caller.py` runs identically on either turn.
    """
    assert issubclass(CrocoSuperconductingTfCoil, SuperconductingTfCoil)
    assert issubclass(CiccSuperconductingTfCoil, SuperconductingTfCoil)
    assert not issubclass(CrocoSuperconductingTfCoil, CiccSuperconductingTfCoil)


def test_a_croco_machine_assembles_with_the_croco_namespace(tmp_path):
    """`i_tf_turn_type = 2` builds `CrocoSuperconductingTfCoil`, and nothing else moves.

    The measurement `_refuse_unported_switch` was added for, run the other way round:
    until 2026-08-29 a CroCo input assembled **silently as cable-in-conduit**, and the
    point of the refusal was that the namespace should change with the switch. It does
    now, so this checks the change rather than the refusal.

    Built from the reference tokamak rather than from an ST file because the two tracked
    ST files still do not assemble -- they refuse on the PF coil system and on
    `i_tf_stress_model`, neither of which is anything to do with the turn.
    """
    from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE

    text = pathlib.Path(TOKAMAK_INPUT_FILE).read_text()
    text = "\n".join(
        "i_tf_sc_mat = 9" if line.startswith("i_tf_sc_mat") else line
        for line in text.splitlines()
    )
    croco = tmp_path / "croco.IN.DAT"
    croco.write_text(text + "\ni_tf_turn_type = 2\n")

    machine = machine_from_indat(str(croco))
    coil = machine.tokamak.cicc_superconducting_tf_coil
    assert isinstance(coil, CrocoSuperconductingTfCoil)
    assert coil.croco_superconductor_properties is not None

    cicc = machine_from_indat(TOKAMAK_INPUT_FILE).tokamak.cicc_superconducting_tf_coil
    assert isinstance(cicc, CiccSuperconductingTfCoil)
    # The shared half really is shared: same occupant class in every base-class slot.
    shared = {f.name for f in SuperconductingTfCoil.__dataclass_fields__.values()}
    for name in shared:
        if name in {
            "superconducting_tf_coil_areas_and_masses",
            "tf_superconductor_temperature_margin",
        }:
            continue  # keyed on `i_tf_sc_mat`, which this file also changes
        assert type(getattr(coil, name)) is type(getattr(cicc, name)), name


def test_croco_nodes_own_the_tape_stack(tmp_path):
    """The CroCo nodes produce the `*croco*`/`*hts_tape*` fields, not read them.

    `next_steps.md` §18.5 counted nineteen such fields with no producer anywhere in the
    port; this checks the ones `run` actually computes now have one, which is the
    property `missing_producers_tokamak.txt` exists to protect.
    """
    from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE
    from functional_process.cottax.indat import graph_for

    text = pathlib.Path(TOKAMAK_INPUT_FILE).read_text()
    text = "\n".join(
        "i_tf_sc_mat = 9" if line.startswith("i_tf_sc_mat") else line
        for line in text.splitlines()
    )
    croco = tmp_path / "croco.IN.DAT"
    croco.write_text(text + "\ni_tf_turn_type = 2\n")

    graph = graph_for(machine_from_indat(str(croco)))
    owned = {str(var) for var in graph.owners}
    for field in (
        ".superconducting_tfcoil.a_tf_croco_strand",
        ".superconducting_tfcoil.a_tf_croco_strand_copper_total",
        ".superconducting_tfcoil.a_tf_croco_strand_hastelloy",
        ".superconducting_tfcoil.a_tf_croco_strand_rebco",
        ".superconducting_tfcoil.a_tf_croco_strand_solder",
        ".superconducting_tfcoil.dia_tf_croco_strand_tape_region",
        ".superconducting_tfcoil.dia_tf_turn_croco_cable",
        ".superconducting_tfcoil.dr_tf_hts_tape",
        ".superconducting_tfcoil.dx_tf_croco_strand_tape_stack",
        ".superconducting_tfcoil.dx_tf_hts_tape_total",
        ".superconducting_tfcoil.n_tf_croco_strand_hts_tapes",
        # PROCESS writes this one as a literal zero; owning it is what keeps it off the
        # boundary as a read of a coincidence.
        ".tfcoil.f_a_tf_turn_cable_space_extra_void",
    ):
        assert any(str(v) == f"VarPath({field})" for v in graph.owners), field
    assert owned  # the set is non-empty, so the assertion above is meaningful


def test_the_two_tracked_spherical_tokamaks_assemble():
    """Both ST files build a machine and a graph -- the CroCo cluster's whole point.

    **This assertion has been strengthened twice and the history is the reason it is
    written this way.** When the CroCo wave landed it could only check the *content* of
    a refusal, because five PF dimensions and `i_tf_stress_model` still blocked
    assembly; the previous version of this test asserted the files still raised and
    listed the three switch names that had left the message. Both later blockers closed
    (`_audit/units/models/tfcoil/stress.md`, the `extended_plane_strain` section), so
    that version would now fail for the best possible reason -- which is exactly what a
    test written around a refusal does when the refusal expires. Asserting that they
    assemble is the claim that cannot rot in that direction.
    """
    from functional_process.cottax.indat import graph_for

    for name in ("spherical_tokamak_eval", "st_regression"):
        machine = machine_from_indat(f"tests/regression/input_files/{name}.IN.DAT")
        graph = graph_for(machine)
        assert graph.nodes, name


def test_hijc_rebco_is_the_only_new_material_fit():
    """A guard on the claim `croco.md` makes: the CroCo package needed exactly one
    critical-surface fit the port did not already have.
    """
    from functional_process.models.physics import superconductors as ported

    assert hasattr(ported, "hijc_rebco")
    # `current_sharing_rebco` is deliberately absent -- see `croco.py`'s module
    # docstring: its only consumer on the CroCo path is a dead `.tfcoil.temp_margin`.
    assert not hasattr(ported, "current_sharing_rebco")


def test_croco_cable_geometry_tape_count_is_a_floor():
    """`n_croco_strand_hts_tapes` is `floor`, so it is flat between steps.

    Pinned because it is why `TestCrocoCableGeometry` excuses its gradient checks -- a
    reader who later "fixes" the floor to a smooth count would change values, not just
    gradients, and would silently invalidate that excuse's justification.
    """
    tapes = croco_cable_geometry(
        dia_croco_strand=0.010,
        dx_croco_strand_copper=0.001,
        dx_hts_tape_rebco=1e-6,
        dx_hts_tape_copper=2e-6,
        dx_hts_tape_hastelloy=3e-6,
    )[1]
    assert float(tapes) == np.floor(float(tapes))


def test_croco_cable_geometry_gradient_within_one_step():
    """`jacfwd` of the port against a central difference of PROCESS's own function.

    The measurement `TestCrocoCableGeometry`'s docstring cites, kept executable. The
    contract above excuses its gradient checks because PROCESS's `epsfcn = 1e-3`
    perturbation crosses whole tape steps; here the step is `1e-6` relative, which at the
    ST strand's 30.88 tapes stays inside one, and the two agree to better than `1e-8`
    relative on every one of the fifty derivative components.

    Written by hand rather than through `Tier1Contract` because the whole point is a
    *different* perturbation size from the harness's, which is what `epsfcn` fixes.
    """
    base = {
        "dia_croco_strand": 0.013055620408525749,
        "dx_croco_strand_copper": 2.0e-3,
        "dx_hts_tape_rebco": 1.0e-6,
        "dx_hts_tape_copper": 2.0e-4,
        "dx_hts_tape_hastelloy": 1e-5,
    }
    reference = _reference_croco_cable_geometry

    # Mid-step, which is the precondition the measurement is only meaningful under.
    tapes = reference(**base)[1]
    stack, total = reference(**base)[9], reference(**base)[8]
    assert 0.2 < (stack / total - tapes) < 0.95

    def ported_vector(**kwargs):
        return jnp.stack([
            jnp.asarray(v, dtype=float) for v in croco_cable_geometry(**kwargs)
        ])

    for name, value in base.items():
        step = 1e-6 * abs(value)
        jac = jax.jacfwd(
            lambda v, name=name: ported_vector(**{**base, name: v}),
        )(value)
        plus = np.asarray(reference(**{**base, name: value + step}), dtype=float)
        minus = np.asarray(reference(**{**base, name: value - step}), dtype=float)
        finite_difference = (plus - minus) / (2.0 * step)
        for i, (got, want) in enumerate(
            zip(np.asarray(jac, dtype=float), finite_difference, strict=True)
        ):
            scale = max(abs(got), abs(want), 1e-300)
            assert abs(got - want) / scale < 1e-8, f"d(output[{i}])/d({name})"
