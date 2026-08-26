"""Harness case for the ported subset of `process/models/cryostat.py`
(`.tokamak.cryostat`).

Audit record: `functional_process/_audit/units/models/cryostat.md`. One unit:
`calculate_r_cryostat_inboard`, tier-1, the sole occupant. **Not** the stellarator's
cryostat (`process/models/stellarator/stellarator.py:1282-1330`, already ported).
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.cryostat import calculate_r_cryostat_inboard
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
