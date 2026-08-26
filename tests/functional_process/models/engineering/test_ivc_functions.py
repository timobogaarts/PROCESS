"""Harness cases for the shared toroidal-shell helpers ported from
`process/models/engineering/ivc_functions.py` (not a numbered registry unit — see
`functional_process/_audit/units/models/engineering/ivc_functions.md`).

Both functions are already pure in `process/`, so the PROCESS reference is called
directly with no `DataStructure` adapter needed.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.engineering.ivc_functions import eshellarea, eshellvol
from process.models.engineering.ivc_functions import eshellarea as _reference_eshellarea
from process.models.engineering.ivc_functions import eshellvol as _reference_eshellvol


class TestEshellarea(Tier1Contract):
    """`eshellarea` -> the same, unchanged."""

    audit_record = "models/engineering/ivc_functions.md"
    reference = staticmethod(_reference_eshellarea)
    ported = eshellarea

    samples = [
        legacy_sample(
            "eshellarea-plausible-geometry",
            rshell=8.0,
            rmini=3.9166666666666665,
            rmino=4.616666666666667,
            zminor=7.503275248730414,
        ),
    ]

    fuzz_bounds = {
        "rshell": (2.0, 20.0),
        "rmini": (0.5, 8.0),
        "rmino": (0.5, 8.0),
        "zminor": (0.5, 15.0),
    }


class TestEshellvol(Tier1Contract):
    """`eshellvol` -> the same, unchanged.

    Legacy sample borrowed from
    `tests/unit/models/test_vacuum.py::test_elliptical_vessel_volumes`'s
    `EllipticalVesselVolumes` point, reduced to the `(r_1, r_2, r_3, ...)` tuple that
    `VacuumVessel.calculate_elliptical_vessel_volumes` derives from it before calling
    `eshellvol` -- see `ivc_functions.md` § sample provenance.
    """

    audit_record = "models/engineering/ivc_functions.md"
    reference = staticmethod(_reference_eshellvol)
    ported = eshellvol

    samples = [
        legacy_sample(
            "eshellvol-elliptical-vessel-legacy",
            rshell=8.0 - 2.6666666666666665 * 0.5,
            rmini=(8.0 - 2.6666666666666665 * 0.5) - 4.083333333333334,
            rmino=12.716666666666667 - (8.0 - 2.6666666666666665 * 0.5),
            zminor=7.5032752487304135,
            drin=0.30000000000000004,
            drout=0.30000000000000004,
            dz=(0.30000000000000004 + 0.30000000000000004) / 2,
        ),
    ]

    fuzz_bounds = {
        "rshell": (2.0, 20.0),
        "rmini": (0.5, 8.0),
        "rmino": (0.5, 8.0),
        "zminor": (0.5, 15.0),
        "drin": (0.05, 1.0),
        "drout": (0.05, 1.0),
        "dz": (0.05, 1.0),
    }
