"""Harness cases for the ported L-H/L-I transition power thresholds.

The 21 individual `calculate_*` legacy samples below are lifted verbatim from
`tests/unit/models/physics/test_l_h_transition.py`'s own parametrised tests -- a free,
already-validated oracle per `_audit/test_harness.md` § Tier 1 sampling, the same
provenance `test_confinement_time.py` established for its 48 scaling laws.
`PlasmaConfinementTransition`'s own static methods are called directly as the
reference -- these formulas take no `self.data` access at all, so no adapter is needed.

The six `*LHThresholdPower` cottax node classes (the Martin family) are not
`Tier1Contract`-diffed on their own: each is a thin wrapper (unit-convert
`nd_plasma_electron_line` to `dnla20`, then call the already-tested pure function), so
what needs checking is that the wrapping is correct, not the formula underneath it --
same reasoning `test_confinement_time.py` applies to `Iss04ConfinementTime`/
`IterIpb98y2ConfinementTime`. `test_martin_family_occupants_match_process` checks each
occupant reproduces its pure function's own value at a shared point; the
`test_*_reads` functions check the declared `.inputs` are exactly what the audit record
says each arm reads (in particular, that only the aspect-corrected trio reads
`.physics.aspect`).
"""

import pytest

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.l_h_transition import (
    Martin08AspectLowerLHThresholdPower,
    Martin08AspectNominalLHThresholdPower,
    Martin08AspectUpperLHThresholdPower,
    Martin08LowerLHThresholdPower,
    Martin08NominalLHThresholdPower,
    Martin08UpperLHThresholdPower,
    calculate_hubbard2012_lower,
    calculate_hubbard2012_nominal,
    calculate_hubbard2012_upper,
    calculate_hubbard2017,
    calculate_iter1996_lower,
    calculate_iter1996_nominal,
    calculate_iter1996_upper,
    calculate_martin08_aspect_lower,
    calculate_martin08_aspect_nominal,
    calculate_martin08_aspect_upper,
    calculate_martin08_lower,
    calculate_martin08_nominal,
    calculate_martin08_upper,
    calculate_snipes1997_iter,
    calculate_snipes1997_kappa,
    calculate_snipes2000_closed_divertor_lower,
    calculate_snipes2000_closed_divertor_nominal,
    calculate_snipes2000_closed_divertor_upper,
    calculate_snipes2000_lower,
    calculate_snipes2000_nominal,
    calculate_snipes2000_upper,
)
from process.models.physics.l_h_transition import PlasmaConfinementTransition

_AUDIT_RECORD = "models/physics/l_h_transition.md"


class TestIter1996Nominal(Tier1Contract):
    """`calculate_iter1996_nominal` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_iter1996_nominal)
    ported = calculate_iter1996_nominal

    samples = [
        legacy_sample(
            "iter1996_nominal-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
    }


class TestIter1996Upper(Tier1Contract):
    """`calculate_iter1996_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_iter1996_upper)
    ported = calculate_iter1996_upper

    samples = [
        legacy_sample(
            "iter1996_upper-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
    }


class TestIter1996Lower(Tier1Contract):
    """`calculate_iter1996_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_iter1996_lower)
    ported = calculate_iter1996_lower

    samples = [
        legacy_sample(
            "iter1996_lower-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
    }


class TestSnipes1997Iter(Tier1Contract):
    """`calculate_snipes1997_iter` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_snipes1997_iter)
    ported = calculate_snipes1997_iter

    samples = [
        legacy_sample(
            "snipes1997_iter-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
    }


class TestSnipes1997Kappa(Tier1Contract):
    """`calculate_snipes1997_kappa` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_snipes1997_kappa)
    ported = calculate_snipes1997_kappa

    samples = [
        legacy_sample(
            "snipes1997_kappa-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            kappa=1.8,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "kappa": (1.0, 2.2),
    }


class TestMartin08Nominal(Tier1Contract):
    """`calculate_martin08_nominal` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_martin08_nominal)
    ported = calculate_martin08_nominal

    samples = [
        legacy_sample(
            "martin08_nominal-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestMartin08Upper(Tier1Contract):
    """`calculate_martin08_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_martin08_upper)
    ported = calculate_martin08_upper

    samples = [
        legacy_sample(
            "martin08_upper-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestMartin08Lower(Tier1Contract):
    """`calculate_martin08_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_martin08_lower)
    ported = calculate_martin08_lower

    samples = [
        legacy_sample(
            "martin08_lower-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000Nominal(Tier1Contract):
    """`calculate_snipes2000_nominal` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_snipes2000_nominal)
    ported = calculate_snipes2000_nominal

    samples = [
        legacy_sample(
            "snipes2000_nominal-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            rminor=2.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000Upper(Tier1Contract):
    """`calculate_snipes2000_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_snipes2000_upper)
    ported = calculate_snipes2000_upper

    samples = [
        legacy_sample(
            "snipes2000_upper-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            rminor=2.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000Lower(Tier1Contract):
    """`calculate_snipes2000_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_snipes2000_lower)
    ported = calculate_snipes2000_lower

    samples = [
        legacy_sample(
            "snipes2000_lower-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            rminor=2.0,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000ClosedDivertorNominal(Tier1Contract):
    """`calculate_snipes2000_closed_divertor_nominal` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(
        PlasmaConfinementTransition.calculate_snipes2000_closed_divertor_nominal
    )
    ported = calculate_snipes2000_closed_divertor_nominal

    samples = [
        legacy_sample(
            "snipes2000_closed_divertor_nominal-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000ClosedDivertorUpper(Tier1Contract):
    """`calculate_snipes2000_closed_divertor_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(
        PlasmaConfinementTransition.calculate_snipes2000_closed_divertor_upper
    )
    ported = calculate_snipes2000_closed_divertor_upper

    samples = [
        legacy_sample(
            "snipes2000_closed_divertor_upper-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestSnipes2000ClosedDivertorLower(Tier1Contract):
    """`calculate_snipes2000_closed_divertor_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(
        PlasmaConfinementTransition.calculate_snipes2000_closed_divertor_lower
    )
    ported = calculate_snipes2000_closed_divertor_lower

    samples = [
        legacy_sample(
            "snipes2000_closed_divertor_lower-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            rmajor=6.2,
            m_ions_total_amu=2.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "m_ions_total_amu": (2.0, 3.0),
    }


class TestHubbard2012Nominal(Tier1Contract):
    """`calculate_hubbard2012_nominal` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_hubbard2012_nominal)
    ported = calculate_hubbard2012_nominal

    samples = [
        legacy_sample(
            "hubbard2012_nominal-process-test", plasma_current=1.0e6, dnla20=1.0
        ),
    ]

    fuzz_bounds = {
        "plasma_current": (1.0e6, 2.0e7),
        "dnla20": (0.05, 2.0),
    }


class TestHubbard2012Upper(Tier1Contract):
    """`calculate_hubbard2012_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_hubbard2012_upper)
    ported = calculate_hubbard2012_upper

    samples = [
        legacy_sample(
            "hubbard2012_upper-process-test", plasma_current=1.0e6, dnla20=1.0
        ),
    ]

    fuzz_bounds = {
        "plasma_current": (1.0e6, 2.0e7),
        "dnla20": (0.05, 2.0),
    }


class TestHubbard2012Lower(Tier1Contract):
    """`calculate_hubbard2012_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_hubbard2012_lower)
    ported = calculate_hubbard2012_lower

    samples = [
        legacy_sample(
            "hubbard2012_lower-process-test", plasma_current=1.0e6, dnla20=1.0
        ),
    ]

    fuzz_bounds = {
        "plasma_current": (1.0e6, 2.0e7),
        "dnla20": (0.05, 2.0),
    }


class TestHubbard2017(Tier1Contract):
    """`calculate_hubbard2017` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_hubbard2017)
    ported = calculate_hubbard2017

    samples = [
        legacy_sample(
            "hubbard2017-process-test",
            dnla20=1.0,
            a_plasma_surface=100.0,
            b_plasma_toroidal_on_axis=5.0,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "a_plasma_surface": (50.0, 2000.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
    }


class TestMartin08AspectNominal(Tier1Contract):
    """`calculate_martin08_aspect_nominal` -> the same, unchanged.

    This is the reference arm: `i_l_h_threshold = 19` is PROCESS's own default
    (`physics_variables.py:1234`) and `large_tokamak_eval.IN.DAT` never overrides it.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(
        PlasmaConfinementTransition.calculate_martin08_aspect_nominal
    )
    ported = calculate_martin08_aspect_nominal

    samples = [
        legacy_sample(
            "martin08_aspect_nominal-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
            aspect=2.5,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
        "aspect": (1.5, 4.0),
    }


class TestMartin08AspectUpper(Tier1Contract):
    """`calculate_martin08_aspect_upper` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_martin08_aspect_upper)
    ported = calculate_martin08_aspect_upper

    samples = [
        legacy_sample(
            "martin08_aspect_upper-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
            aspect=2.5,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
        "aspect": (1.5, 4.0),
    }


class TestMartin08AspectLower(Tier1Contract):
    """`calculate_martin08_aspect_lower` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaConfinementTransition.calculate_martin08_aspect_lower)
    ported = calculate_martin08_aspect_lower

    samples = [
        legacy_sample(
            "martin08_aspect_lower-process-test",
            dnla20=1.0,
            b_plasma_toroidal_on_axis=5.0,
            a_plasma_surface=100.0,
            m_ions_total_amu=2.0,
            aspect=2.5,
        ),
    ]

    fuzz_bounds = {
        "dnla20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "a_plasma_surface": (50.0, 2000.0),
        "m_ions_total_amu": (2.0, 3.0),
        "aspect": (1.5, 4.0),
    }


# ---------------------------------------------------------------------------
# The Martin-family cottax node occupants -- structural + wrapping checks, not
# Tier1Contract (see the module docstring).
# ---------------------------------------------------------------------------

_MARTIN_FAMILY_POINT = {
    "nd_plasma_electron_line": 1.0e20,
    "b_plasma_toroidal_on_axis": 5.68,
    "a_plasma_surface": 1500.0,
    "m_ions_total_amu": 2.5,
    "aspect": 2.907,
}


def _dnla20(point):
    return 1.0e-20 * point["nd_plasma_electron_line"]


def test_non_aspect_arms_do_not_read_aspect():
    """`Martin08{Nominal,Upper,Lower}` read exactly the reads-set validated as a
    subset of the aspect-corrected trio's -- in particular, none of them reads
    `.physics.aspect`, which only the correction factor needs.
    """
    for cls in (
        Martin08NominalLHThresholdPower,
        Martin08UpperLHThresholdPower,
        Martin08LowerLHThresholdPower,
    ):
        reads = {i.var.path_str() for i in cls().inputs}
        assert reads == {
            ".physics.nd_plasma_electron_line",
            ".physics.b_plasma_toroidal_on_axis",
            ".physics.a_plasma_surface",
            ".physics.m_ions_total_amu",
        }, f"{cls.__name__} reads {reads}"


def test_aspect_corrected_arms_read_one_more_field():
    """`Martin08Aspect{Nominal,Upper,Lower}` read exactly the non-aspect trio's reads
    plus `.physics.aspect` -- one read more, nothing else, matching
    `confinement_time.md`'s `PlasmaPowerLoss` siblings' shape.
    """
    for cls in (
        Martin08AspectNominalLHThresholdPower,
        Martin08AspectUpperLHThresholdPower,
        Martin08AspectLowerLHThresholdPower,
    ):
        reads = {i.var.path_str() for i in cls().inputs}
        assert reads == {
            ".physics.nd_plasma_electron_line",
            ".physics.b_plasma_toroidal_on_axis",
            ".physics.a_plasma_surface",
            ".physics.m_ions_total_amu",
            ".physics.aspect",
        }, f"{cls.__name__} reads {reads}"


def test_every_martin_occupant_owns_p_l_h_threshold_mw():
    """All six occupants own the same field -- genuine alternatives for one slot."""
    for cls in (
        Martin08NominalLHThresholdPower,
        Martin08UpperLHThresholdPower,
        Martin08LowerLHThresholdPower,
        Martin08AspectNominalLHThresholdPower,
        Martin08AspectUpperLHThresholdPower,
        Martin08AspectLowerLHThresholdPower,
    ):
        outputs = {o.var.path_str() for o in cls().outputs}
        assert outputs == {".physics.p_l_h_threshold_mw"}, f"{cls.__name__}: {outputs}"


def test_martin_family_occupants_match_process():
    """Each occupant, called with the reads it declares, reproduces the pure function
    it wraps (and therefore PROCESS's own `calculate_*` at the same point).
    """
    point = _MARTIN_FAMILY_POINT
    dnla20 = _dnla20(point)
    non_aspect_kwargs = {
        "nd_plasma_electron_line": point["nd_plasma_electron_line"],
        "b_plasma_toroidal_on_axis": point["b_plasma_toroidal_on_axis"],
        "a_plasma_surface": point["a_plasma_surface"],
        "m_ions_total_amu": point["m_ions_total_amu"],
    }
    aspect_kwargs = dict(non_aspect_kwargs, aspect=point["aspect"])

    cases = [
        (
            Martin08NominalLHThresholdPower,
            non_aspect_kwargs,
            calculate_martin08_nominal(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
            ),
        ),
        (
            Martin08UpperLHThresholdPower,
            non_aspect_kwargs,
            calculate_martin08_upper(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
            ),
        ),
        (
            Martin08LowerLHThresholdPower,
            non_aspect_kwargs,
            calculate_martin08_lower(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
            ),
        ),
        (
            Martin08AspectNominalLHThresholdPower,
            aspect_kwargs,
            calculate_martin08_aspect_nominal(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
                point["aspect"],
            ),
        ),
        (
            Martin08AspectUpperLHThresholdPower,
            aspect_kwargs,
            calculate_martin08_aspect_upper(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
                point["aspect"],
            ),
        ),
        (
            Martin08AspectLowerLHThresholdPower,
            aspect_kwargs,
            calculate_martin08_aspect_lower(
                dnla20,
                point["b_plasma_toroidal_on_axis"],
                point["a_plasma_surface"],
                point["m_ions_total_amu"],
                point["aspect"],
            ),
        ),
    ]

    for cls, kwargs, expected in cases:
        actual = cls()(**kwargs)
        assert float(actual) == pytest.approx(expected, rel=1e-14, abs=0.0), (
            f"{cls.__name__}: {actual} != {expected}"
        )
