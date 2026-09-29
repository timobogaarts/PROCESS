"""Pure-functional port of `Stellarator.st_strc` (chunk 1D of unit #1)."""

from functional_process.cottax.paths import (
    structure,
)
from functional_process.models.safe_math import (
    safe_pow,  # noqa: F401
)
from functional_process.models.stellarator.structure import (
    calculate_intercoil_mass_scaling_reference,  # noqa: F401
    calculate_structure_masses,
)

# cottax node: `calculate_structure_masses`, unchanged, with its ports declared.
StructureMasses = structure.function(
    calculate_structure_masses,
    "aintmass",
    "clgsmass",
    "coldmass",
)
