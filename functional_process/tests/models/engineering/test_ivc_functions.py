"""Harness cases for the shared toroidal-shell helpers ported from
`process/models/engineering/ivc_functions.py` (not a numbered registry unit — see
`functional_process/_audit/units/models/engineering/ivc_functions.md`).

All four functions are already pure in `process/`, so the PROCESS reference is called
directly with no `DataStructure` adapter needed.

2026-08-27 (the D-shaped wave): `dshellarea`/`dshellvol` joined, for the five slots the
two spherical-tokamak input files select on the D-shaped arm.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.models.engineering.ivc_functions import (
    dshellarea,
    dshellvol,
    eshellarea,
    eshellvol,
)
from process.models.engineering.ivc_functions import dshellarea as _reference_dshellarea
from process.models.engineering.ivc_functions import dshellvol as _reference_dshellvol
from process.models.engineering.ivc_functions import eshellarea as _reference_eshellarea
from process.models.engineering.ivc_functions import eshellvol as _reference_eshellvol


class TestEshellarea(Tier1Contract):
    """`eshellarea` -> the same, unchanged."""

    audit_record = "models/engineering/ivc_functions.md"
    reference = staticmethod(_reference_eshellarea)
    ported = eshellarea

    samples = FROM_FILE

    fuzz = True


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

    samples = FROM_FILE

    fuzz = True


class TestDshellarea(Tier1Contract):
    """`dshellarea` -> the same, unchanged.

    No legacy sample: `grep -rl dshellarea tests/unit` is empty, and unlike `eshellvol`
    there is no unit test of a caller whose intermediate tuple could be reduced to one
    (`tests/unit/models/test_vacuum.py` only covers the elliptical vessel). The fuzz box
    below is therefore this contract's whole oracle -- which is enough, because both
    sides are the same closed-form expression and the harness checks gradients too.

    `rminor` is the *width across the shell*, not a plasma minor radius, so its box runs
    wider than `eshellarea`'s `rmini`/`rmino`: on the D-shaped arm callers pass
    `(rmajor + rminor + gap) - r1`, roughly twice a plasma minor radius plus the gaps.
    """

    audit_record = "models/engineering/ivc_functions.md"
    reference = staticmethod(_reference_dshellarea)
    ported = dshellarea

    fuzz_bounds = {
        "rmajor": (1.0, 20.0),
        "rminor": (1.0, 16.0),
        "zminor": (0.5, 15.0),
    }


class TestDshellvol(Tier1Contract):
    """`dshellvol` -> the same, unchanged.

    `drin` is bounded well below the smallest `rmajor` in the box: the inboard term is
    `rmajor**2 - (rmajor - drin)**2`, which turns negative-volume nonsense once `drin`
    exceeds `rmajor`. PROCESS has no guard, so both sides would agree on the nonsense and
    the contract would still pass -- the bound is here to keep the sampled points
    physical rather than to hide a disagreement.
    """

    audit_record = "models/engineering/ivc_functions.md"
    reference = staticmethod(_reference_dshellvol)
    ported = dshellvol

    fuzz_bounds = {
        "rmajor": (1.0, 20.0),
        "rminor": (1.0, 16.0),
        "zminor": (0.5, 15.0),
        "drin": (0.05, 0.9),
        "drout": (0.05, 1.0),
        "dz": (0.05, 1.0),
    }
