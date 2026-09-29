"""Mutual and self inductances of the PF coils, the CS and the plasma."""

from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.models.pfcoil.inductance import (
    calculate_pf_cs_plasma_inductances,  # noqa: F401 -- re-exported for tests
    calculate_pf_cs_plasma_inductances_at_reference_width,
    calculate_pf_plasma_inductances_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_plasma_inductances_no_central_solenoid_bound,
    calculate_pf_plasma_inductances_no_central_solenoid_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_solenoid_self_inductance,  # noqa: F401 -- re-exported for tests
)

NPLAS = 1
"""`nplas`, a literal `1` in `induct` (`pfcoil.py:1734`) -- the plasma is one filament,
at `(rmajor, 0)`.
"""


# cottax node: `.tokamak.pf_coil.inductance`.
PFCoilInductance = pf_coil.function(
    calculate_pf_cs_plasma_inductances_at_reference_width,
    "ind_pf_cs_plasma_mutual",
    reads={
        "rmajor": physics,
        "ind_plasma": physics,
        "dr_cs": build,
    },
)


# cottax node: `.tokamak.pf_coil.inductance`, the `iohcl = 0` occupant.
PFCoilInductanceNoCentralSolenoid = pf_coil.function(
    calculate_pf_plasma_inductances_no_central_solenoid_bound,
    "ind_pf_cs_plasma_mutual",
    reads={
        "rmajor": physics,
        "ind_plasma": physics,
    },
)
