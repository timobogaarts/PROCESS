"""Harness case for the ported subset of `process/models/cryostat.py`
(`.tokamak.cryostat`).

Audit record: `functional_process/_audit/units/models/cryostat.md`. Two units, both
tier-1 and both halves of the one occupant `.tokamak.cryostat` holds:
`calculate_r_cryostat_inboard` and, since 2026-08-30,
`calculate_cryostat_vertical_clearances`. **Not** the stellarator's cryostat
(`process/models/stellarator/stellarator.py:1282-1330`, already ported).
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.cryostat import (
    calculate_cryostat_vertical_clearances,
    calculate_r_cryostat_inboard,
)
from process.core.model import DataStructure
from process.models.cryostat import Cryostat


def _reference_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat):
    """Call PROCESS's `Cryostat.external_cryo_geometry` through the port's signature.

    Every field downstream of `r_cryostat_inboard` in `external_cryo_geometry` is left
    at its `DataStructure` default -- harmless (no division by an unset zero occurs
    downstream on the default state) and irrelevant, since only `r_cryostat_inboard`
    itself is read back.
    """
    data = DataStructure()
    data.pf_coil.r_pf_coil_outer = np.asarray(r_pf_coil_outer, dtype=float)
    data.fwbs.dr_pf_cryostat = dr_pf_cryostat

    c = Cryostat()
    c.data = data
    c.external_cryo_geometry()
    return c.data.fwbs.r_cryostat_inboard


class TestCalculateRCryostatInboard(Tier1Contract):
    """`calculate_r_cryostat_inboard` -> `Cryostat.external_cryo_geometry`'s first
    line.
    """

    audit_record = "models/cryostat.md"
    reference = _reference_r_cryostat_inboard
    ported = calculate_r_cryostat_inboard

    # tests/unit/models/test_cryostat.py::test_external_cryo_geometry, verbatim
    # (generated from large_tokamak_eval.IN.DAT).
    samples = [
        legacy_sample(
            "large-tokamak-legacy",
            r_pf_coil_outer=np.array([
                6.1290994712971543,
                6.2110624909068086,
                17.305470903073743,
                17.305470903073743,
                15.620546715016166,
                15.620546715016166,
                2.5506597842255361,
                10.666666666666666,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
            dr_pf_cryostat=0.5,
        ),
    ]

    fuzz_bounds = {
        "r_pf_coil_outer": (np.zeros(22), np.full(22, 25.0)),
        "dr_pf_cryostat": (0.1, 2.0),
    }


def _reference_cryostat_vertical_clearances(
    f_z_cryostat, r_cryostat_inboard, z_pf_coil_upper, z_tf_inside_half, dr_tf_inboard
):
    """Call PROCESS's `Cryostat.external_cryo_geometry` through the port's signature.

    The port's chain starts from `r_cryostat_inboard`, which PROCESS computes rather
    than accepts, so the reference is driven to that value the only way the method
    allows: one PF coil at `r_cryostat_inboard` and a zero clearance. That is exact
    rather than approximate -- `np.max` of a one-element array is that element and
    `x + 0.0 == x` for every finite double -- so nothing is smuggled in between the
    port's argument and PROCESS's own first line.

    Everything downstream of `dz_tf_cryostat` in the method (`vol_cryostat_internal`,
    `vol_cryostat`, `dewmkg`) is left at its `DataStructure` default and never read
    back; it is UNPORTED and outside this contract.
    """
    data = DataStructure()
    data.pf_coil.r_pf_coil_outer = np.array([r_cryostat_inboard], dtype=float)
    data.fwbs.dr_pf_cryostat = 0.0
    data.build.f_z_cryostat = f_z_cryostat
    data.pf_coil.z_pf_coil_upper = np.asarray(z_pf_coil_upper, dtype=float)
    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dr_tf_inboard = dr_tf_inboard

    c = Cryostat()
    c.data = data
    c.external_cryo_geometry()
    return (
        c.data.blanket.dz_pf_cryostat,
        c.data.fwbs.z_cryostat_half_inside,
        c.data.buildings.dz_tf_cryostat,
    )


class TestCalculateCryostatVerticalClearances(Tier1Contract):
    """`calculate_cryostat_vertical_clearances` ->
    `Cryostat.external_cryo_geometry`'s vertical chain (`cryostat.py:43-60`).

    Added 2026-08-30 with the `.buildings.dz_tf_cryostat` producer -- see the port
    function's docstring for why an `InputVariable` PROCESS overwrites every pass is a
    missing producer and not a genuine input.
    """

    audit_record = "models/cryostat.md"
    reference = _reference_cryostat_vertical_clearances
    ported = calculate_cryostat_vertical_clearances

    # tests/unit/models/test_cryostat.py::test_external_cryo_geometry, verbatim -- the
    # same generated-from-large_tokamak_eval point `TestCalculateRCryostatInboard` uses,
    # whose four remaining fields this contract is what finally exercises.
    # `r_cryostat_inboard` is that case's `expected_r_cryostat_inboard`, because the
    # legacy point's stored `r_cryostat_inboard = 0` is the pre-call value the method
    # overwrites.
    samples = [
        legacy_sample(
            "large-tokamak-legacy",
            f_z_cryostat=4.2679999999999998,
            r_cryostat_inboard=17.805470903073743,
            z_pf_coil_upper=np.array([
                9.9154920004377978,
                -11.249338850841614,
                3.2350365669570316,
                -3.2350365669570316,
                7.8723998771612473,
                -7.8723998771612473,
                7.9363954477147454,
                4.9333333333333336,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
            z_tf_inside_half=8.8182171641274945,
            dr_tf_inboard=0.92672586247397692,
        ),
    ]

    fuzz_bounds = {
        "f_z_cryostat": (2.0, 10.0),
        "r_cryostat_inboard": (5.0, 25.0),
        "z_pf_coil_upper": (np.full(22, -15.0), np.full(22, 15.0)),
        "z_tf_inside_half": (2.0, 15.0),
        "dr_tf_inboard": (0.1, 3.0),
    }
