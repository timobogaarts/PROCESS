"""Harness cases for `process/models/cryostat.py` (`.tokamak.cryostat`).

Two tier-1 contracts: `calculate_r_cryostat_inboard`, the method's first line, and
`calculate_external_cryo_geometry`, the whole of it. The first is not subsumed by the
second -- PROCESS's first line is a genuine sub-expression of the rest, the port calls
it as one, and it has its own legacy point. **Not** the stellarator's cryostat
(`process/models/stellarator/stellarator.py:1282-1330`, already ported).

Both reference adapters drive PROCESS's `external_cryo_geometry` through a bound
`DataStructure` and read the fields back. That is not ceremony: `external_cryo_geometry`
is an instance method with no arguments at all, so the adapter *is* the whole of the
`In`/`Out` binding the port makes structural, and writing it is where the port's read
set gets checked against PROCESS's rather than asserted.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.process_reference import process_reference
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.cryostat import (
    calculate_external_cryo_geometry,
    calculate_r_cryostat_inboard,
)
from process.core.model import DataStructure
from process.models.cryostat import Cryostat


def _cryostat():
    """A `Cryostat` instance with a fresh `DataStructure` attached."""
    model = Cryostat()
    model.data = DataStructure()
    return model


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


_reference_r_cryostat_inboard = process_reference(
    _cryostat, "external_cryo_geometry", "fwbs.r_cryostat_inboard"
)
"""Every field downstream of `r_cryostat_inboard` in `external_cryo_geometry` is left
at its `DataStructure` default -- harmless (no division by an unset zero occurs
downstream on the default state) and irrelevant, since only `r_cryostat_inboard`
itself is read back."""


class TestCalculateRCryostatInboard(Tier1Contract):
    """`calculate_r_cryostat_inboard` -> `Cryostat.external_cryo_geometry`'s first
    line.
    """

    audit_record = "models/cryostat.md"
    reference = _reference_r_cryostat_inboard
    ported = calculate_r_cryostat_inboard

    # tests/unit/models/test_cryostat.py::test_external_cryo_geometry, verbatim
    # (generated from large_tokamak_eval.IN.DAT).
    samples = FROM_FILE

    fuzz_bounds = {
        "r_pf_coil_outer": (np.zeros(22), np.full(22, 25.0)),
        "dr_pf_cryostat": (0.1, 2.0),
    }


_reference_external_cryo_geometry = process_reference(
    _cryostat,
    "external_cryo_geometry",
    (
        "fwbs.r_cryostat_inboard",
        "blanket.dz_pf_cryostat",
        "fwbs.z_cryostat_half_inside",
        "buildings.dz_tf_cryostat",
        "fwbs.vol_cryostat_internal",
        "fwbs.vol_cryostat",
        "fwbs.dewmkg",
    ),
)
"""`Cryostat.external_cryo_geometry` whole, through the port's signature.

Nine pokes onto a fresh `DataStructure` and seven reads back, in the port's return
order. Every field the method reads is set from the sample, so no default can stand in
for a read the port declares -- which is the check this adapter exists to make."""


class TestCalculateExternalCryoGeometry(Tier1Contract):
    """`calculate_external_cryo_geometry` -> `Cryostat.external_cryo_geometry`."""

    audit_record = "models/cryostat.md"
    reference = _reference_external_cryo_geometry
    ported = calculate_external_cryo_geometry

    # tests/unit/models/test_cryostat.py::test_external_cryo_geometry, verbatim
    # (generated from large_tokamak_eval.IN.DAT) -- the same legacy point the
    # first-line contract above uses, with the six further fields it also carries.
    samples = FROM_FILE

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
