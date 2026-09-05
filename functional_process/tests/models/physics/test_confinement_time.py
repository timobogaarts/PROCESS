"""Harness cases for the ported plasma confinement time scalings (registry unit #10).

The 48 individual `<name>_confinement_time` legacy samples below are lifted verbatim
from `tests/unit/models/physics/test_confinement_time.py`'s own
`test_confinement_time` parametrisation (all-ones inputs, PROCESS-computed expected
values) -- a free, already-validated oracle per `_audit/test_harness.md` § Tier 1
sampling, and the same "legacy" provenance `test_fusion_reactions.py` and
`test_radiation_power.py` already established for this repo. `PlasmaConfinementTime`'s
own static method is called directly as the reference -- these formulas take no
`self.data` access at all, so no adapter is needed.

`TestConfinementTime` (the composite `calculate_confinement_time`) and
`TestCalculateDoubleAndTripleProduct` build a real `DataStructure`, set every
field the source method reads implicitly off `self.data` (see `confinement_time.md`'s
data footprint table), and call the bound method through it -- same "close the data
backdoor" technique used throughout this harness.
"""

import pytest

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.confinement_time import (
    Iss04ConfinementTime,
    IterIpb98y2ConfinementTime,
    PlasmaPowerLossIgnitedCoreRadiation,
    PlasmaPowerLossNonIgnitedCoreRadiation,
    calculate_confinement_time,
    calculate_double_and_triple_product,
    calculate_iter_physics_basis_elongation,
    christiansen_confinement_time,
    ds03_confinement_time,
    goldston_confinement_time,
    gyro_reduced_bohm_confinement_time,
    hubbard_lower_confinement_time,
    hubbard_nominal_confinement_time,
    hubbard_upper_confinement_time,
    iss04_stellarator_confinement_time,
    iss95_stellarator_confinement_time,
    iter_89_0_confinement_time,
    iter_89p_confinement_time,
    iter_93h_confinement_time,
    iter_96p_confinement_time,
    iter_h90_p_amended_confinement_time,
    iter_h90_p_confinement_time,
    iter_h97p_confinement_time,
    iter_h97p_elmy_confinement_time,
    iter_ipb98y1_confinement_time,
    iter_ipb98y2_confinement_time,
    iter_ipb98y3_confinement_time,
    iter_ipb98y4_confinement_time,
    iter_ipb98y_confinement_time,
    iter_pb98py_confinement_time,
    itpa20_confinement_time,
    itpa20_il_confinement_time,
    jaeri_confinement_time,
    kaye_big_confinement_time,
    kaye_confinement_time,
    kaye_goldston_confinement_time,
    lackner_gottardi_confinement_time,
    lackner_gottardi_stellarator_confinement_time,
    lang_high_density_confinement_time,
    menard_nstx_confinement_time,
    menard_nstx_petty08_hybrid_confinement_time,
    merezhkin_muhkovatov_confinement_time,
    mirnov_confinement_time,
    murari_confinement_time,
    ncst_confinement_time,
    neo_alcator_confinement_time,
    neo_kaye_confinement_time,
    nstx_gyro_bohm_confinement_time,
    paz_soldan_nt_confinement_time,
    petty08_confinement_time,
    rebut_lallia_confinement_time,
    riedel_h_confinement_time,
    riedel_l_confinement_time,
    shimomura_confinement_time,
    sudo_et_al_confinement_time,
    t10_confinement_time,
    valovic_elmy_confinement_time,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.physics.confinement_time import PlasmaConfinementTime
from process.models.physics.plasma_geometry import PlasmaGeom


class TestNeoAlcatorConfinementTime(Tier1Contract):
    """`neo_alcator_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.neo_alcator_confinement_time)
    ported = neo_alcator_confinement_time

    samples = [
        legacy_sample(
            "neo_alcator-all-ones", dene20=1.0, rminor=1.0, rmajor=1.0, qstar=1.0
        ),
    ]

    fuzz_bounds = {
        "dene20": (0.05, 2.0),
        "rminor": (0.5, 5.0),
        "rmajor": (2.0, 20.0),
        "qstar": (1.5, 8.0),
    }


class TestMirnovConfinementTime(Tier1Contract):
    """`mirnov_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.mirnov_confinement_time)
    ported = mirnov_confinement_time

    samples = [
        legacy_sample("mirnov-all-ones", rminor=1.0, kappa95=1.0, cur_plasma_ma=1.0),
    ]

    fuzz_bounds = {
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "cur_plasma_ma": (1.0, 20.0),
    }


class TestMerezhkinMuhkovatovConfinementTime(Tier1Contract):
    """`merezhkin_muhkovatov_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.merezhkin_muhkovatov_confinement_time)
    ported = merezhkin_muhkovatov_confinement_time

    samples = [
        legacy_sample(
            "merezhkin_muhkovatov-all-ones",
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            qstar=1.0,
            nd_plasma_electron_line_20=1.0,
            afuel=1.0,
            ten=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "qstar": (1.5, 8.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "afuel": (2.0, 3.0),
        "ten": (2.0, 30.0),
    }


class TestShimomuraConfinementTime(Tier1Contract):
    """`shimomura_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.shimomura_confinement_time)
    ported = shimomura_confinement_time

    samples = [
        legacy_sample(
            "shimomura-all-ones",
            rmajor=1.0,
            rminor=1.0,
            b_plasma_toroidal_on_axis=1.0,
            kappa95=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "kappa95": (1.0, 2.2),
        "afuel": (2.0, 3.0),
    }


class TestKayeGoldstonConfinementTime(Tier1Contract):
    """`kaye_goldston_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.kaye_goldston_confinement_time)
    ported = kaye_goldston_confinement_time

    samples = [
        legacy_sample(
            "kaye_goldston-all-ones",
            kappa95=1.0,
            cur_plasma_ma=1.0,
            n20=1.0,
            rmajor=1.0,
            afuel=1.0,
            b_plasma_toroidal_on_axis=1.0,
            rminor=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "kappa95": (1.0, 2.2),
        "cur_plasma_ma": (1.0, 20.0),
        "n20": (0.05, 2.0),
        "rmajor": (2.0, 20.0),
        "afuel": (2.0, 3.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rminor": (0.5, 5.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestIter89pConfinementTime(Tier1Contract):
    """`iter_89p_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_89p_confinement_time)
    ported = iter_89p_confinement_time

    samples = [
        legacy_sample(
            "iter_89p-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.0, 2.5),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestIter890ConfinementTime(Tier1Contract):
    """`iter_89_0_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_89_0_confinement_time)
    ported = iter_89_0_confinement_time

    samples = [
        legacy_sample(
            "iter_89_0-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.0, 2.5),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestRebutLalliaConfinementTime(Tier1Contract):
    """`rebut_lallia_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.rebut_lallia_confinement_time)
    ported = rebut_lallia_confinement_time

    samples = [
        legacy_sample(
            "rebut_lallia-all-ones",
            rminor=1.0,
            rmajor=1.0,
            kappa=1.0,
            afuel=1.0,
            cur_plasma_ma=1.0,
            zeff=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "rminor": (0.5, 5.0),
        "rmajor": (2.0, 20.0),
        "kappa": (1.0, 2.5),
        "afuel": (2.0, 3.0),
        "cur_plasma_ma": (1.0, 20.0),
        "zeff": (1.0, 3.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestGoldstonConfinementTime(Tier1Contract):
    """`goldston_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.goldston_confinement_time)
    ported = goldston_confinement_time

    samples = [
        legacy_sample(
            "goldston-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestT10ConfinementTime(Tier1Contract):
    """`t10_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.t10_confinement_time)
    ported = t10_confinement_time

    samples = [
        legacy_sample(
            "t10-all-ones",
            nd_plasma_electron_line_20=1.0,
            rmajor=1.0,
            qstar=1.0,
            b_plasma_toroidal_on_axis=1.0,
            rminor=1.0,
            kappa95=1.0,
            p_plasma_loss_mw=1.0,
            zeff=1.0,
            cur_plasma_ma=1.0,
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "rmajor": (2.0, 20.0),
        "qstar": (1.5, 8.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "zeff": (1.0, 3.0),
        "cur_plasma_ma": (1.0, 20.0),
    }


class TestJaeriConfinementTime(Tier1Contract):
    """`jaeri_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.jaeri_confinement_time)
    ported = jaeri_confinement_time

    samples = [
        legacy_sample(
            "jaeri-all-ones",
            kappa95=1.0,
            rminor=1.0,
            afuel=1.0,
            n20=1.0,
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            rmajor=1.0,
            qstar=1.0,
            zeff=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "kappa95": (1.0, 2.2),
        "rminor": (0.5, 5.0),
        "afuel": (2.0, 3.0),
        "n20": (0.05, 2.0),
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "qstar": (1.5, 8.0),
        "zeff": (1.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestKayeBigConfinementTime(Tier1Contract):
    """`kaye_big_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.kaye_big_confinement_time)
    ported = kaye_big_confinement_time

    samples = [
        legacy_sample(
            "kaye_big-all-ones",
            rmajor=1.0,
            rminor=1.0,
            b_plasma_toroidal_on_axis=1.0,
            kappa95=1.0,
            cur_plasma_ma=1.0,
            n20=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "kappa95": (1.0, 2.2),
        "cur_plasma_ma": (1.0, 20.0),
        "n20": (0.05, 2.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestIterH90PConfinementTime(Tier1Contract):
    """`iter_h90_p_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_h90_p_confinement_time)
    ported = iter_h90_p_confinement_time

    samples = [
        legacy_sample(
            "iter_h90_p-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.0, 2.5),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestRiedelLConfinementTime(Tier1Contract):
    """`riedel_l_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.riedel_l_confinement_time)
    ported = riedel_l_confinement_time

    samples = [
        legacy_sample(
            "riedel_l-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestChristiansenConfinementTime(Tier1Contract):
    """`christiansen_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.christiansen_confinement_time)
    ported = christiansen_confinement_time

    samples = [
        legacy_sample(
            "christiansen-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "afuel": (2.0, 3.0),
    }


class TestLacknerGottardiConfinementTime(Tier1Contract):
    """`lackner_gottardi_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.lackner_gottardi_confinement_time)
    ported = lackner_gottardi_confinement_time

    samples = [
        legacy_sample(
            "lackner_gottardi-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestNeoKayeConfinementTime(Tier1Contract):
    """`neo_kaye_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.neo_kaye_confinement_time)
    ported = neo_kaye_confinement_time

    samples = [
        legacy_sample(
            "neo_kaye-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestRiedelHConfinementTime(Tier1Contract):
    """`riedel_h_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.riedel_h_confinement_time)
    ported = riedel_h_confinement_time

    samples = [
        legacy_sample(
            "riedel_h-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa95=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa95": (1.0, 2.2),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestIterH90PAmendedConfinementTime(Tier1Contract):
    """`iter_h90_p_amended_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_h90_p_amended_confinement_time)
    ported = iter_h90_p_amended_confinement_time

    samples = [
        legacy_sample(
            "iter_h90_p_amended-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            afuel=1.0,
            rmajor=1.0,
            p_plasma_loss_mw=1.0,
            kappa=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "afuel": (2.0, 3.0),
        "rmajor": (2.0, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "kappa": (1.0, 2.5),
    }


class TestSudoEtAlConfinementTime(Tier1Contract):
    """`sudo_et_al_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.sudo_et_al_confinement_time)
    ported = sudo_et_al_confinement_time

    samples = [
        legacy_sample(
            "sudo_et_al-all-ones",
            rmajor=1.0,
            rminor=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestGyroReducedBohmConfinementTime(Tier1Contract):
    """`gyro_reduced_bohm_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.gyro_reduced_bohm_confinement_time)
    ported = gyro_reduced_bohm_confinement_time

    samples = [
        legacy_sample(
            "gyro_reduced_bohm-all-ones",
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_20=1.0,
            p_plasma_loss_mw=1.0,
            rminor=1.0,
            rmajor=1.0,
        ),
    ]

    fuzz_bounds = {
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rminor": (0.5, 5.0),
        "rmajor": (2.0, 20.0),
    }


class TestLacknerGottardiStellaratorConfinementTime(Tier1Contract):
    """`lackner_gottardi_stellarator_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(
        PlasmaConfinementTime.lackner_gottardi_stellarator_confinement_time
    )
    ported = lackner_gottardi_stellarator_confinement_time

    samples = [
        legacy_sample(
            "lackner_gottardi_stellarator-all-ones",
            rmajor=1.0,
            rminor=1.0,
            nd_plasma_electron_line_20=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            q=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "q": (2.0, 8.0),
    }


class TestIter93hConfinementTime(Tier1Contract):
    """`iter_93h_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_93h_confinement_time)
    ported = iter_93h_confinement_time

    samples = [
        legacy_sample(
            "iter_93h-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            afuel=1.0,
            rmajor=1.0,
            nd_plasma_electron_line_20=1.0,
            aspect=1.0,
            kappa=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "afuel": (2.0, 3.0),
        "rmajor": (2.0, 20.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "aspect": (1.5, 4.0),
        "kappa": (1.0, 2.5),
    }


class TestIterH97pConfinementTime(Tier1Contract):
    """`iter_h97p_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_h97p_confinement_time)
    ported = iter_h97p_confinement_time

    samples = [
        legacy_sample(
            "iter_h97p-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            nd_plasma_electron_line_19=1.0,
            rmajor=1.0,
            aspect=1.0,
            kappa=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "rmajor": (2.0, 20.0),
        "aspect": (1.5, 4.0),
        "kappa": (1.0, 2.5),
        "afuel": (2.0, 3.0),
    }


class TestIterH97pElmyConfinementTime(Tier1Contract):
    """`iter_h97p_elmy_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_h97p_elmy_confinement_time)
    ported = iter_h97p_elmy_confinement_time

    samples = [
        legacy_sample(
            "iter_h97p_elmy-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            nd_plasma_electron_line_19=1.0,
            rmajor=1.0,
            aspect=1.0,
            kappa=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "rmajor": (2.0, 20.0),
        "aspect": (1.5, 4.0),
        "kappa": (1.0, 2.5),
        "afuel": (2.0, 3.0),
    }


class TestIter96pConfinementTime(Tier1Contract):
    """`iter_96p_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_96p_confinement_time)
    ported = iter_96p_confinement_time

    samples = [
        legacy_sample(
            "iter_96p-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            kappa95=1.0,
            rmajor=1.0,
            aspect=1.0,
            nd_plasma_electron_line_19=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "kappa95": (1.0, 2.2),
        "rmajor": (2.0, 20.0),
        "aspect": (1.5, 4.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestValovicElmyConfinementTime(Tier1Contract):
    """`valovic_elmy_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.valovic_elmy_confinement_time)
    ported = valovic_elmy_confinement_time

    samples = [
        legacy_sample(
            "valovic_elmy-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            afuel=1.0,
            rmajor=1.0,
            rminor=1.0,
            kappa=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "afuel": (2.0, 3.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.0, 2.5),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestKayeConfinementTime(Tier1Contract):
    """`kaye_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.kaye_confinement_time)
    ported = kaye_confinement_time

    samples = [
        legacy_sample(
            "kaye-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            kappa=1.0,
            rmajor=1.0,
            aspect=1.0,
            nd_plasma_electron_line_19=1.0,
            afuel=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "kappa": (1.0, 2.5),
        "rmajor": (2.0, 20.0),
        "aspect": (1.5, 4.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "afuel": (2.0, 3.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestIterPb98pyConfinementTime(Tier1Contract):
    """`iter_pb98py_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_pb98py_confinement_time)
    ported = iter_pb98py_confinement_time

    samples = [
        legacy_sample(
            "iter_pb98py-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIterIpb98yConfinementTime(Tier1Contract):
    """`iter_ipb98y_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_ipb98y_confinement_time)
    ported = iter_ipb98y_confinement_time

    samples = [
        legacy_sample(
            "iter_ipb98y-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIterIpb98y1ConfinementTime(Tier1Contract):
    """`iter_ipb98y1_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_ipb98y1_confinement_time)
    ported = iter_ipb98y1_confinement_time

    samples = [
        legacy_sample(
            "iter_ipb98y1-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIterIpb98y2ConfinementTime(Tier1Contract):
    """`iter_ipb98y2_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_ipb98y2_confinement_time)
    ported = iter_ipb98y2_confinement_time

    samples = [
        legacy_sample(
            "iter_ipb98y2-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIterIpb98y3ConfinementTime(Tier1Contract):
    """`iter_ipb98y3_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_ipb98y3_confinement_time)
    ported = iter_ipb98y3_confinement_time

    samples = [
        legacy_sample(
            "iter_ipb98y3-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIterIpb98y4ConfinementTime(Tier1Contract):
    """`iter_ipb98y4_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iter_ipb98y4_confinement_time)
    ported = iter_ipb98y4_confinement_time

    samples = [
        legacy_sample(
            "iter_ipb98y4-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestIss95StellaratorConfinementTime(Tier1Contract):
    """`iss95_stellarator_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iss95_stellarator_confinement_time)
    ported = iss95_stellarator_confinement_time

    samples = [
        legacy_sample(
            "iss95_stellarator-all-ones",
            rminor=1.0,
            rmajor=1.0,
            nd_plasma_electron_line_19=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            iotabar=1.0,
        ),
    ]

    fuzz_bounds = {
        "rminor": (0.5, 5.0),
        "rmajor": (2.0, 20.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "iotabar": (0.3, 3.0),
    }


class TestIss04StellaratorConfinementTime(Tier1Contract):
    """`iss04_stellarator_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.iss04_stellarator_confinement_time)
    ported = iss04_stellarator_confinement_time

    samples = [
        legacy_sample(
            "iss04_stellarator-all-ones",
            rminor=1.0,
            rmajor=1.0,
            nd_plasma_electron_line_19=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            iotabar=1.0,
        ),
    ]

    fuzz_bounds = {
        "rminor": (0.5, 5.0),
        "rmajor": (2.0, 20.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "iotabar": (0.3, 3.0),
    }


class TestDs03ConfinementTime(Tier1Contract):
    """`ds03_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.ds03_confinement_time)
    ported = ds03_confinement_time

    samples = [
        legacy_sample(
            "ds03-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa95=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa95": (1.0, 2.2),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestMurariConfinementTime(Tier1Contract):
    """`murari_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.murari_confinement_time)
    ported = murari_confinement_time

    samples = [
        legacy_sample(
            "murari-all-ones",
            cur_plasma_ma=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            nd_plasma_electron_line_19=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestPetty08ConfinementTime(Tier1Contract):
    """`petty08_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.petty08_confinement_time)
    ported = petty08_confinement_time

    samples = [
        legacy_sample(
            "petty08-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
    }


class TestLangHighDensityConfinementTime(Tier1Contract):
    """`lang_high_density_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.lang_high_density_confinement_time)
    ported = lang_high_density_confinement_time

    samples = [
        legacy_sample(
            "lang_high_density-all-ones",
            plasma_current=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            rminor=1.0,
            q=1.0,
            qstar=1.0,
            aspect=1.0,
            afuel=1.0,
            kappa_ipb=1.0,
        ),
    ]

    fuzz_bounds = {
        "plasma_current": (1000000.0, 20000000.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line": (1e19, 1e21),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "q": (2.0, 8.0),
        "qstar": (1.5, 8.0),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
        "kappa_ipb": (1.0, 2.5),
    }


class TestHubbardNominalConfinementTime(Tier1Contract):
    """`hubbard_nominal_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.hubbard_nominal_confinement_time)
    ported = hubbard_nominal_confinement_time

    samples = [
        legacy_sample(
            "hubbard_nominal-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_20=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestHubbardLowerConfinementTime(Tier1Contract):
    """`hubbard_lower_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.hubbard_lower_confinement_time)
    ported = hubbard_lower_confinement_time

    samples = [
        legacy_sample(
            "hubbard_lower-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_20=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestHubbardUpperConfinementTime(Tier1Contract):
    """`hubbard_upper_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.hubbard_upper_confinement_time)
    ported = hubbard_upper_confinement_time

    samples = [
        legacy_sample(
            "hubbard_upper-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_20=1.0,
            p_plasma_loss_mw=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
    }


class TestMenardNstxConfinementTime(Tier1Contract):
    """`menard_nstx_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.menard_nstx_confinement_time)
    ported = menard_nstx_confinement_time

    samples = [
        legacy_sample(
            "menard_nstx-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestMenardNstxPetty08HybridConfinementTime(Tier1Contract):
    """`menard_nstx_petty08_hybrid_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(
        PlasmaConfinementTime.menard_nstx_petty08_hybrid_confinement_time
    )
    ported = menard_nstx_petty08_hybrid_confinement_time

    samples = [
        legacy_sample(
            "menard_nstx_petty08_hybrid-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            kappa_ipb=1.0,
            aspect=1.0,
            afuel=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "kappa_ipb": (1.0, 2.5),
        "aspect": (1.5, 4.0),
        "afuel": (2.0, 3.0),
    }


class TestNstxGyroBohmConfinementTime(Tier1Contract):
    """`nstx_gyro_bohm_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.nstx_gyro_bohm_confinement_time)
    ported = nstx_gyro_bohm_confinement_time

    samples = [
        legacy_sample(
            "nstx_gyro_bohm-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            nd_plasma_electron_line_20=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "nd_plasma_electron_line_20": (0.05, 2.0),
    }


class TestItpa20ConfinementTime(Tier1Contract):
    """`itpa20_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.itpa20_confinement_time)
    ported = itpa20_confinement_time

    samples = [
        legacy_sample(
            "itpa20-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            nd_plasma_electron_line_19=1.0,
            p_plasma_loss_mw=1.0,
            rmajor=1.0,
            triang=1.0,
            kappa_ipb=1.0,
            eps=1.0,
            aion=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "rmajor": (2.0, 20.0),
        "triang": (0.0, 0.6),
        "kappa_ipb": (1.0, 2.5),
        "eps": (0.1, 0.5),
        "aion": (2.0, 3.0),
    }


class TestItpa20IlConfinementTime(Tier1Contract):
    """`itpa20_il_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.itpa20_il_confinement_time)
    ported = itpa20_il_confinement_time

    samples = [
        legacy_sample(
            "itpa20_il-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            nd_plasma_electron_line_19=1.0,
            aion=1.0,
            rmajor=1.0,
            triang=1.0,
            kappa_ipb=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
        "aion": (2.0, 3.0),
        "rmajor": (2.0, 20.0),
        "triang": (0.0, 0.6),
        "kappa_ipb": (1.0, 2.5),
    }


class TestNcstConfinementTime(Tier1Contract):
    """`ncst_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.ncst_confinement_time)
    ported = ncst_confinement_time

    samples = [
        legacy_sample(
            "ncst-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            nd_plasma_electron_line_19=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
    }


class TestPazSoldanNtConfinementTime(Tier1Contract):
    """`paz_soldan_nt_confinement_time` -> the same, unchanged.

    Sample is PROCESS's own `test_confinement_time.py::test_confinement_time`'s
    all-ones case for this scaling law, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.paz_soldan_nt_confinement_time)
    ported = paz_soldan_nt_confinement_time

    samples = [
        legacy_sample(
            "paz_soldan_nt-all-ones",
            cur_plasma_ma=1.0,
            b_plasma_toroidal_on_axis=1.0,
            p_plasma_loss_mw=1.0,
            nd_plasma_electron_line_19=1.0,
        ),
    ]

    fuzz_bounds = {
        "cur_plasma_ma": (1.0, 20.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "p_plasma_loss_mw": (10.0, 1000.0),
        "nd_plasma_electron_line_19": (0.5, 20.0),
    }


class TestIterPhysicsBasisElongation(Tier1Contract):
    """`calculate_iter_physics_basis_elongation` -> `PlasmaGeom.calculate_iter_
    physics_basis_elongation`, unchanged. Not this file's own method -- see
    `confinement_time.md`'s "calls into other models".
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaGeom.calculate_iter_physics_basis_elongation)
    ported = calculate_iter_physics_basis_elongation

    samples = [
        legacy_sample("reference-point", vol_plasma=2426.25, rmajor=8.0, rminor=2.5),
    ]

    fuzz_bounds = {
        "vol_plasma": (100.0, 5000.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
    }


class TestCalculateDoubleAndTripleProduct(Tier1Contract):
    """`calculate_double_and_triple_product` -> the same, unchanged.

    Sample is `test_calculate_double_and_triple_product`'s case verbatim -- genuinely
    legacy.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = staticmethod(PlasmaConfinementTime.calculate_double_and_triple_product)
    ported = calculate_double_and_triple_product

    samples = [
        legacy_sample(
            "double-triple-product-reference",
            nd_plasma_electrons_vol_avg=7.5e19,
            temp_plasma_electrons_vol_avg_kev=3.402116961408892,
            t_energy_confinement=12.569,
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "temp_plasma_electrons_vol_avg_kev": (1.0, 40.0),
        "t_energy_confinement": (0.1, 20.0),
    }


def _reference_calculate_confinement_time(**kwargs):
    """Call PROCESS's `PlasmaConfinementTime.calculate_confinement_time` through the
    port's signature, closing the `self.data` back door.

    Sets every field the source method reads implicitly mid-body (see
    `confinement_time.md`'s data footprint table) onto a real `DataStructure`, then
    calls the bound method with its own 25 explicit arguments. Returns the same
    9-tuple the port does: `ConfinementTimeData`'s 7 fields, plus `kappa_ipb` and
    `t_energy_confinement_beta` read back off `data.physics` (the two implicit writes).
    """
    data = DataStructure()
    data.physics.i_rad_loss = kwargs["i_rad_loss"]
    data.physics.pden_plasma_rad_mw = kwargs["pden_plasma_rad_mw"]
    data.physics.pden_plasma_sync_mw = kwargs["pden_plasma_sync_mw"]
    data.physics.p_plasma_inner_rad_mw = kwargs["p_plasma_inner_rad_mw"]
    data.physics.triang = kwargs["triang"]
    data.physics.m_ions_total_amu = kwargs["m_ions_total_amu"]
    data.physics.e_plasma_beta = kwargs["e_plasma_beta"]
    data.physics.tauee_in = kwargs["tauee_in"]
    data.physics.f_p_alpha_plasma_deposited = kwargs["f_p_alpha_plasma_deposited"]
    data.physics.p_plasma_ohmic_mw = kwargs["p_plasma_ohmic_mw"]

    pct = PlasmaConfinementTime()
    pct.data = data

    confinement_time_data = pct.calculate_confinement_time(
        m_fuel_amu=kwargs["m_fuel_amu"],
        p_alpha_total_mw=kwargs["p_alpha_total_mw"],
        aspect=kwargs["aspect"],
        b_plasma_toroidal_on_axis=kwargs["b_plasma_toroidal_on_axis"],
        nd_plasma_electrons_vol_avg=kwargs["nd_plasma_electrons_vol_avg"],
        nd_plasma_electron_line=kwargs["nd_plasma_electron_line"],
        eps=kwargs["eps"],
        hfact=kwargs["hfact"],
        i_confinement_time=kwargs["i_confinement_time"],
        i_plasma_ignited=kwargs["i_plasma_ignited"],
        kappa=kwargs["kappa"],
        kappa95=kwargs["kappa95"],
        p_non_alpha_charged_mw=kwargs["p_non_alpha_charged_mw"],
        p_hcd_injected_total_mw=kwargs["p_hcd_injected_total_mw"],
        plasma_current=kwargs["plasma_current"],
        pden_plasma_core_rad_mw=kwargs["pden_plasma_core_rad_mw"],
        rmajor=kwargs["rmajor"],
        rminor=kwargs["rminor"],
        temp_plasma_electron_density_weighted_kev=kwargs[
            "temp_plasma_electron_density_weighted_kev"
        ],
        q95=kwargs["q95"],
        qstar=kwargs["qstar"],
        vol_plasma=kwargs["vol_plasma"],
        zeff=kwargs["zeff"],
        eden_plasma_electrons_thermal_vol_avg=kwargs[
            "eden_plasma_electrons_thermal_vol_avg"
        ],
        eden_plasma_ions_thermal_vol_avg=kwargs["eden_plasma_ions_thermal_vol_avg"],
    )
    return (
        confinement_time_data.pden_electron_transport_loss_mw,
        confinement_time_data.pden_ion_transport_loss_mw,
        confinement_time_data.t_electron_energy_confinement,
        confinement_time_data.t_ion_energy_confinement,
        confinement_time_data.t_plasma_energy_confinement,
        confinement_time_data.p_plasma_loss_mw,
        confinement_time_data.hstar,
        data.physics.kappa_ipb,
        data.physics.t_energy_confinement_beta,
    )


# One realistic, physically-consistent operating point (not a converged PROCESS run --
# `converged_sample` is not implemented, see `_harness/sampling.py` -- but plausible
# stellarator-scale magnitudes, same spirit as `fusion_reactions.md`'s legacy point:
# `rmajor`/`vol_plasma` match `test_fusion_reactions.py`'s own reference point).
# Radiated power densities are kept small relative to `p_plasma_loss_mw` so every
# `i_rad_loss` arm stays comfortably clear of the `max(..., 1e-3)` floor.
_BASE = {
    "m_fuel_amu": 2.5,
    "p_alpha_total_mw": 400.0,
    "aspect": 3.2,
    "b_plasma_toroidal_on_axis": 5.0,
    "nd_plasma_electrons_vol_avg": 8.0e19,
    "nd_plasma_electron_line": 9.0e19,
    "eps": 1.0 / 3.2,
    "hfact": 1.0,
    "i_plasma_ignited": 0,
    "kappa": 1.7,
    "kappa95": 1.6,
    "p_non_alpha_charged_mw": 20.0,
    "p_hcd_injected_total_mw": 50.0,
    "plasma_current": 15.0e6,
    "pden_plasma_core_rad_mw": 0.02,
    "rmajor": 8.0,
    "rminor": 2.5,
    "temp_plasma_electron_density_weighted_kev": 12.0,
    "q95": 3.0,
    "qstar": 3.5,
    "vol_plasma": 2426.25,
    "zeff": 1.8,
    "eden_plasma_electrons_thermal_vol_avg": 1.5e5,
    "eden_plasma_ions_thermal_vol_avg": 1.5e5,
    "f_p_alpha_plasma_deposited": 0.95,
    "p_plasma_ohmic_mw": 0.5,
    "i_rad_loss": 2,  # NO_RADIATION by default; overridden per sample below
    "pden_plasma_rad_mw": 0.05,
    "pden_plasma_sync_mw": 0.02,
    "p_plasma_inner_rad_mw": 5.0,
    "triang": 0.4,
    "m_ions_total_amu": 2.5,
    "e_plasma_beta": 5.0e8,
    "tauee_in": 3.0,
}


def _point(i_confinement_time, **overrides):
    kwargs = dict(_BASE, i_confinement_time=i_confinement_time)
    kwargs.update(overrides)
    return kwargs


class TestConfinementTime(Tier1Contract):
    """`calculate_confinement_time` -> the composite dispatcher, all 9 outputs.

    Legacy-style points at one realistic operating point (see `_BASE`), swept over a
    representative subset of `i_confinement_time` (ohmic, L-mode, min-of-two, an
    IPB98-family H-mode law that reads `kappa_ipb`, a stellarator law that reads
    `q95` as `iotabar`, and the NSTX-Petty08 hybrid's continuous blend) and over all
    three `i_rad_loss` arms -- not exhaustive over all 51 scaling laws (each is
    already independently legacy-tested above against PROCESS's own static method;
    this class checks the *dispatch and implicit-read promotion*, not the formulas
    again). `fuzz_bounds` holds `i_confinement_time`/`i_rad_loss` fixed at one
    combination (`ISS04_STELLARATOR`, `CORE_ONLY`) and sweeps the continuous physics
    arguments, so the gradient check still exercises `kappa_ipb`'s
    `calculate_iter_physics_basis_elongation` composition and the radiation-loss
    subtraction term.
    """

    audit_record = "models/physics/confinement_time.md"
    reference = _reference_calculate_confinement_time
    ported = calculate_confinement_time

    static_argnames = ("i_confinement_time", "i_plasma_ignited", "i_rad_loss")

    samples = [
        legacy_sample(
            "sudo-et-al-no-radiation",
            **_point(21, i_rad_loss=2),
        ),
        legacy_sample(
            "neo-alcator-full-radiation",
            **_point(1, i_rad_loss=0),
        ),
        legacy_sample(
            "mirnov-core-only-radiation-ignited",
            **_point(2, i_rad_loss=1, i_plasma_ignited=1),
        ),
        legacy_sample(
            "minimum-iter-89p-89o-no-radiation",
            **_point(14, i_rad_loss=2),
        ),
        legacy_sample(
            "ipb98y2-core-only-radiation",
            **_point(34, i_rad_loss=1),
        ),
        legacy_sample(
            "iss04-stellarator-full-radiation",
            **_point(38, i_rad_loss=0),
        ),
        legacy_sample(
            "menard-nstx-petty08-hybrid-blend-region",
            **_point(47, i_rad_loss=2, aspect=2.0, eps=0.5),
        ),
        legacy_sample(
            "menard-nstx-petty08-hybrid-nstx-only",
            **_point(47, i_rad_loss=2, aspect=1.5, eps=1.0 / 1.5),
        ),
    ]

    fuzz_fixed = {
        "i_confinement_time": 38,  # ISS04_STELLARATOR
        "i_rad_loss": 1,  # CORE_ONLY
        "i_plasma_ignited": 0,
        "tauee_in": 3.0,
    }
    fuzz_bounds = {
        "m_fuel_amu": (2.0, 3.0),
        "p_alpha_total_mw": (50.0, 800.0),
        "aspect": (1.5, 4.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "nd_plasma_electron_line": (1.0e19, 2.0e20),
        "eps": (0.1, 0.5),
        "hfact": (0.5, 1.5),
        "kappa": (1.0, 2.5),
        "kappa95": (1.0, 2.2),
        "p_non_alpha_charged_mw": (1.0, 100.0),
        "p_hcd_injected_total_mw": (1.0, 200.0),
        "plasma_current": (1.0e6, 2.0e7),
        "pden_plasma_core_rad_mw": (1.0e-3, 0.05),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "temp_plasma_electron_density_weighted_kev": (2.0, 30.0),
        "q95": (0.3, 3.0),  # bound to iotabar-like range for the ISS04 branch
        "qstar": (1.5, 8.0),
        "vol_plasma": (100.0, 5000.0),
        "zeff": (1.0, 3.0),
        "eden_plasma_electrons_thermal_vol_avg": (1.0e4, 5.0e5),
        "eden_plasma_ions_thermal_vol_avg": (1.0e4, 5.0e5),
        "f_p_alpha_plasma_deposited": (0.8, 1.0),
        "p_plasma_ohmic_mw": (0.01, 5.0),
        "pden_plasma_rad_mw": (1.0e-3, 0.05),
        "pden_plasma_sync_mw": (1.0e-3, 0.05),
        "p_plasma_inner_rad_mw": (0.1, 20.0),
        "triang": (0.0, 0.6),
        "m_ions_total_amu": (2.0, 3.0),
        "e_plasma_beta": (1.0e7, 1.0e9),
    }


def test_confinement_time_invalid_switches_raise():
    """`TITAN_REMOVED` and an out-of-range `i_confinement_time` both raise, matching
    PROCESS's own `ProcessValueError` behaviour, in the port's plain `ValueError`.

    Also covers `USER_INPUT` (0) -- a confirmed real PROCESS bug, not merely an
    out-of-range value; see `confinement_time.md`'s "A dead branch: USER_INPUT".

    Not run through `Tier1Contract`: both `i_confinement_time` and the raise itself are
    switch-driven (static, never traced -- `_audit/naming_convention.md` section
    "switches are not ports"), so this isn't the traced-domain-error case
    `reference_domain_errors` exists for (that's for a *data-dependent* raise on a
    continuous argument, which a traced function cannot reproduce and must return
    non-finite for instead). Here PROCESS and the port raise the identical exception
    for the identical (non-traced) reason, so a direct `pytest.raises` is simpler and
    more honest than routing it through the value-agreement machinery.
    """
    user_input = _point(0)  # ConfinementTimeModel.USER_INPUT -- dead code in PROCESS
    with pytest.raises(ProcessValueError, match="Illegal value"):
        _reference_calculate_confinement_time(**user_input)
    with pytest.raises(ValueError, match="USER_INPUT is dead code"):
        calculate_confinement_time(**user_input)

    kwargs = _point(25)  # ConfinementTimeModel.TITAN_REMOVED
    with pytest.raises(ProcessValueError, match="Scaling removed"):
        _reference_calculate_confinement_time(**kwargs)
    with pytest.raises(ValueError, match="Scaling removed"):
        calculate_confinement_time(**kwargs)

    bad = _point(999)
    with pytest.raises(ProcessValueError, match="Illegal value"):
        _reference_calculate_confinement_time(**bad)
    with pytest.raises(ValueError, match="Illegal value"):
        calculate_confinement_time(**bad)


def test_the_law_decides_which_variable_feeds_it_not_the_device():
    """The `q95`/`iotabar` binding follows the **scaling law**, not the device.

    This is the same regression the old `StellaratorConfinementTime` test pinned, and it
    is now asserted on the objects the graph actually holds. PROCESS's
    `calculate_confinement_time` names its 20th positional parameter `q95`
    (`process/models/physics/confinement_time.py:79`) and the tokamak caller does pass
    `.physics.q95` -- but the stellarator caller passes `.stellarator.iotabar` into that
    same slot (`process/models/stellarator/stellarator.py:2312`), where ISS04 consumes it
    as `iotabar**0.41`.

    **A subclass existed only to rebind that one read, and splitting the switch deleted
    it.** `iss04_stellarator_confinement_time`'s own parameter *is* `iotabar`, so its
    occupant reads `.stellarator.iotabar` because that is what its law takes;
    `iter_ipb98y2_confinement_time` takes `cur_plasma_ma` and never asks for either. The
    read follows from the law, which is why `CONFINEMENT_SCALING` is keyed on
    `i_confinement_time` and not on `istell`.
    """
    iss04 = [i.var.path_str() for i in Iss04ConfinementTime().inputs]
    ipb98 = [i.var.path_str() for i in IterIpb98y2ConfinementTime().inputs]

    assert ".stellarator.iotabar" in iss04
    assert ".physics.q95" not in iss04
    # The tokamak's law asks for neither: its own parameters are the plasma current and
    # the IPB elongation, so the question the subclass existed to answer is not asked.
    assert ".stellarator.iotabar" not in ipb98
    assert ".physics.q95" not in ipb98

    # Same output either way -- one field, whichever law owns it -- which is what makes
    # them genuine alternatives for one slot.
    assert Iss04ConfinementTime().outputs == IterIpb98y2ConfinementTime().outputs


def test_ignition_switch_decides_exactly_one_read_of_the_power_loss_head():
    """The two `PlasmaPowerLoss` occupants differ by one read, and it is the right one.

    `_audit/tokamak_boundary.md` § "What blocked the real file" predicted this exactly:
    *"the NON_IGNITED core-only head is the same formula with one extra term, so the arm
    needs exactly one new class reading exactly one extra variable,
    `.current_drive.p_hcd_injected_total_mw`"*. This pins that the prediction is what was
    built -- one read more, nothing else, and the same owned field either way, which is
    what makes them alternatives for one slot rather than two different nodes.
    """
    ignited = [i.var.path_str() for i in PlasmaPowerLossIgnitedCoreRadiation().inputs]
    non_ignited = [
        i.var.path_str() for i in PlasmaPowerLossNonIgnitedCoreRadiation().inputs
    ]

    assert set(non_ignited) - set(ignited) == {".current_drive.p_hcd_injected_total_mw"}
    assert set(ignited) - set(non_ignited) == set()

    # Neither arm reads the full-radiation density: `CORE_ONLY` subtracts
    # `pden_plasma_core_rad_mw`, and a dead argument passed as zero never reaches a port.
    assert ".physics.pden_plasma_rad_mw" not in ignited + non_ignited

    assert (
        PlasmaPowerLossIgnitedCoreRadiation().outputs
        == PlasmaPowerLossNonIgnitedCoreRadiation().outputs
    )


@pytest.mark.parametrize("i_plasma_ignited", [0, 1])
def test_power_loss_occupants_match_process(i_plasma_ignited):
    """Each `PlasmaPowerLoss` occupant reproduces PROCESS's own `p_plasma_loss_mw`.

    `plasma_power_loss_mw` is an extraction from the head of
    `calculate_confinement_time` and PROCESS has no function of that shape, so the
    occupants cannot be `Tier1Contract`-diffed on their own (their own docstrings say
    so). What *can* be checked directly is that the node, called with the reads it
    declares, returns the number PROCESS's composite computes at the same point -- which
    is `_reference_calculate_confinement_time`'s 6th output. That closes the gap the
    extraction opened at the one boundary PROCESS does expose.
    """
    kwargs = _point(34, i_rad_loss=1, i_plasma_ignited=i_plasma_ignited)
    expected = _reference_calculate_confinement_time(**kwargs)[5]

    shared = {
        "f_p_alpha_plasma_deposited": kwargs["f_p_alpha_plasma_deposited"],
        "p_alpha_total_mw": kwargs["p_alpha_total_mw"],
        "p_non_alpha_charged_mw": kwargs["p_non_alpha_charged_mw"],
        "p_plasma_ohmic_mw": kwargs["p_plasma_ohmic_mw"],
        "pden_plasma_core_rad_mw": kwargs["pden_plasma_core_rad_mw"],
        "vol_plasma": kwargs["vol_plasma"],
    }
    if i_plasma_ignited == 0:
        actual = PlasmaPowerLossNonIgnitedCoreRadiation()(
            p_hcd_injected_total_mw=kwargs["p_hcd_injected_total_mw"], **shared
        )
    else:
        actual = PlasmaPowerLossIgnitedCoreRadiation()(**shared)

    assert float(actual) == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_the_non_ignited_arm_is_not_the_ignited_one():
    """The extra term is worth a number, not just a read -- otherwise the split is free.

    A structural test that two classes declare different ports says nothing about
    whether the distinction matters. At `_BASE`'s operating point the injected heating
    is 50 MW of a few hundred, so the two arms differ by exactly that, and reading the
    ignited arm's answer on a non-ignited run would be wrong by it.
    """
    kwargs = _point(34, i_rad_loss=1)
    shared = {
        "f_p_alpha_plasma_deposited": kwargs["f_p_alpha_plasma_deposited"],
        "p_alpha_total_mw": kwargs["p_alpha_total_mw"],
        "p_non_alpha_charged_mw": kwargs["p_non_alpha_charged_mw"],
        "p_plasma_ohmic_mw": kwargs["p_plasma_ohmic_mw"],
        "pden_plasma_core_rad_mw": kwargs["pden_plasma_core_rad_mw"],
        "vol_plasma": kwargs["vol_plasma"],
    }
    ignited = float(PlasmaPowerLossIgnitedCoreRadiation()(**shared))
    non_ignited = float(
        PlasmaPowerLossNonIgnitedCoreRadiation()(
            p_hcd_injected_total_mw=kwargs["p_hcd_injected_total_mw"], **shared
        )
    )
    assert non_ignited - ignited == pytest.approx(
        kwargs["p_hcd_injected_total_mw"], rel=1e-14
    )
