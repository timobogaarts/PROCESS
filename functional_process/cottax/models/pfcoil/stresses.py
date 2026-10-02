"""The Central Solenoid's stress state.

`ohcalc`'s superconducting-coil stress block.
"""

from cottax.cx import V

from functional_process.cottax.paths import pf_coil
from functional_process.models.pfcoil.stresses import (
    _ellipe,  # noqa: F401 -- re-exported for tests
    _ellipk,  # noqa: F401 -- re-exported for tests
    calculate_cs_hoop_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_radial_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_self_peak_midplane_axial_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_stresses,  # noqa: F401 -- re-exported for tests
    calculate_cs_stresses_from_full_width_current,
    calculate_tresca_stress,  # noqa: F401 -- re-exported for tests
    calculate_von_mises_stress,  # noqa: F401 -- re-exported for tests
)

# cottax node: `.tokamak.cs_coil.stresses`.
CSCoilStresses = pf_coil.function(
    calculate_cs_stresses_from_full_width_current,
    "stress_hoop_cs_inner",
    "stress_z_cs_self_peak_midplane",
    "forc_z_cs_self_peak_midplane",
    "stress_radial_cs_peak",
    "stress_radial_cs_inner",
    "stress_shear_cs_peak",
    "stress_mises_cs_peak",
    reads={
        "f_poisson_cs_structure": V.tfcoil.poisson_steel,
    },
)
