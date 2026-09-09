"""Pure-functional port of `Stellarator.st_strc` (chunk 1D of unit #1)."""

from cottax.interfaces.pytree_namespace_module import (
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    fwbs,
    physics,
    stellarator,
    stellarator_config,
    structure,
    tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.safe_math import (
    safe_pow,  # noqa: F401
)
from functional_process.models.stellarator.structure import (
    calculate_intercoil_mass_scaling_reference,  # noqa: F401
    calculate_structure_masses,
)


class StructureMasses(WrapsFunction):
    """cottax node: `calculate_structure_masses`, unchanged, with its ports declared."""

    fn = calculate_structure_masses

    stella_config_coilsurface = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    r_coil_minor = From(stellarator)
    stella_config_coil_rminor = From(stellarator_config)
    dx_tf_inboard_out_toroidal = From(tfcoil)
    len_tf_coil = From(tfcoil)
    n_tf_coils = From(tfcoil)
    b_plasma_toroidal_on_axis = From(physics)
    den_steel = From(fwbs)
    m_tf_coils_total = From(tfcoil)
    dewmkg = From(fwbs)

    aintmass = OutputInto(structure)
    clgsmass = OutputInto(structure)
    coldmass = OutputInto(structure)
