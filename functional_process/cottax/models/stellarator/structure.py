"""Pure-functional port of `Stellarator.st_strc` (chunk 1D of unit #1)."""

from functional_process.cottax.paths import (
    fwbs,
    physics,
    stellarator,
    stellarator_config,
    structure,
    tfcoil,
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
    reads={
        "stella_config_coilsurface": stellarator_config,
        "f_st_rmajor": stellarator,
        "r_coil_minor": stellarator,
        "stella_config_coil_rminor": stellarator_config,
        "dx_tf_inboard_out_toroidal": tfcoil,
        "len_tf_coil": tfcoil,
        "n_tf_coils": tfcoil,
        "b_plasma_toroidal_on_axis": physics,
        "den_steel": fwbs,
        "m_tf_coils_total": tfcoil,
        "dewmkg": fwbs,
    },
)
