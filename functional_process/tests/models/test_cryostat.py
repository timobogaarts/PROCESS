"""Harness cases for `process/models/cryostat.py` (`.tokamak.cryostat`).

Audit record: `functional_process/_audit/units/models/cryostat.md`. Two tier-1
contracts: `calculate_r_cryostat_inboard`, the method's first line, and
`calculate_external_cryo_geometry`, the whole of it (added 2026-08-30 with the other
five fields). The first is not subsumed by the second -- PROCESS's first line is a
genuine sub-expression of the rest, the port calls it as one, and it has its own legacy
point. **Not** the stellarator's cryostat
(`process/models/stellarator/stellarator.py:1282-1330`, already ported).

Both reference adapters drive PROCESS's `external_cryo_geometry` through a bound
`DataStructure` and read the fields back. That is not ceremony: `external_cryo_geometry`
is an instance method with no arguments at all, so the adapter *is* the whole of the
`In`/`Out` binding the port makes structural, and writing it is where the port's read
set gets checked against PROCESS's rather than asserted.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.cryostat import (
    calculate_external_cryo_geometry,
    calculate_r_cryostat_inboard,
)
from process.core.model import DataStructure
from process.models.cryostat import Cryostat

R_PF_COIL_OUTER = np.array([
    6.1290994712971543,
    6.2110624909068086,
    17.305470903073743,
    17.305470903073743,
    15.620546715016166,
    15.620546715016166,
    2.5506597842255361,
    10.666666666666666,
    *([0.0] * 14),
])
"""`.pf_coil.r_pf_coil_outer` at `large_tokamak_eval.IN.DAT`'s converged point, from
`tests/unit/models/test_cryostat.py::test_external_cryo_geometry`. Shared by both
contracts because it is the same legacy point; the trailing fourteen zeros are the
unused tail of the `NGC2`-wide array, kept so the array has the shape PROCESS gives it.
"""


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
            r_pf_coil_outer=R_PF_COIL_OUTER,
            dr_pf_cryostat=0.5,
        ),
    ]

    fuzz_bounds = {
        "r_pf_coil_outer": (np.zeros(22), np.full(22, 25.0)),
        "dr_pf_cryostat": (0.1, 2.0),
    }


def _reference_external_cryo_geometry(
    r_pf_coil_outer,
    dr_pf_cryostat,
    f_z_cryostat,
    z_pf_coil_upper,
    z_tf_inside_half,
    dr_tf_inboard,
    dr_cryostat,
    vol_vv,
    den_steel,
):
    """`Cryostat.external_cryo_geometry` whole, through the port's signature.

    Nine writes onto a fresh `DataStructure` and seven reads back, in the port's return
    order. Every field the method reads is set from the sample, so no default can stand
    in for a read the port declares -- which is the check this adapter exists to make.
    """
    data = DataStructure()
    data.pf_coil.r_pf_coil_outer = np.asarray(r_pf_coil_outer, dtype=float)
    data.pf_coil.z_pf_coil_upper = np.asarray(z_pf_coil_upper, dtype=float)
    data.fwbs.dr_pf_cryostat = dr_pf_cryostat
    data.fwbs.vol_vv = vol_vv
    data.fwbs.den_steel = den_steel
    data.build.f_z_cryostat = f_z_cryostat
    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dr_tf_inboard = dr_tf_inboard
    data.build.dr_cryostat = dr_cryostat

    c = Cryostat()
    c.data = data
    c.external_cryo_geometry()
    return (
        data.fwbs.r_cryostat_inboard,
        data.blanket.dz_pf_cryostat,
        data.fwbs.z_cryostat_half_inside,
        data.buildings.dz_tf_cryostat,
        data.fwbs.vol_cryostat_internal,
        data.fwbs.vol_cryostat,
        data.fwbs.dewmkg,
    )


class TestCalculateExternalCryoGeometry(Tier1Contract):
    """`calculate_external_cryo_geometry` -> `Cryostat.external_cryo_geometry`."""

    audit_record = "models/cryostat.md"
    reference = _reference_external_cryo_geometry
    ported = calculate_external_cryo_geometry

    # tests/unit/models/test_cryostat.py::test_external_cryo_geometry, verbatim
    # (generated from large_tokamak_eval.IN.DAT) -- the same legacy point the
    # first-line contract above uses, with the six further fields it also carries.
    samples = [
        legacy_sample(
            "large-tokamak-legacy",
            r_pf_coil_outer=R_PF_COIL_OUTER,
            dr_pf_cryostat=0.5,
            f_z_cryostat=4.2679999999999998,
            z_pf_coil_upper=np.array([
                9.9154920004377978,
                -11.249338850841614,
                3.2350365669570316,
                -3.2350365669570316,
                7.8723998771612473,
                -7.8723998771612473,
                7.9363954477147454,
                4.9333333333333336,
                *([0.0] * 14),
            ]),
            z_tf_inside_half=8.8182171641274945,
            dr_tf_inboard=0.92672586247397692,
            dr_cryostat=0.15000000000000002,
            vol_vv=1016.2876250857248,
            den_steel=7800.0,
        ),
    ]

    fuzz_bounds = {
        "r_pf_coil_outer": (np.zeros(22), np.full(22, 25.0)),
        "dr_pf_cryostat": (0.1, 2.0),
        "f_z_cryostat": (2.0, 6.0),
        # Kept non-negative: `z_cryostat_half_inside` is `max(z_pf_coil_upper) + ...`,
        # and an all-negative draw would put the cryostat lid below the midplane, which
        # is not a domain either side rejects but is not a machine either.
        "z_pf_coil_upper": (np.zeros(22), np.full(22, 12.0)),
        "z_tf_inside_half": (5.0, 12.0),
        "dr_tf_inboard": (0.4, 1.5),
        "dr_cryostat": (0.05, 0.4),
        "vol_vv": (400.0, 2000.0),
        "den_steel": (7000.0, 8500.0),
    }
