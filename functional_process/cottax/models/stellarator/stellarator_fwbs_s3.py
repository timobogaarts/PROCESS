"""Pure-functional port of `st_fwbs`'s S3 fragment (`stellarator.py:1030-1043`)."""

from cottax.interfaces.pytree_namespace_module import (
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    divertor,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.stellarator.stellarator_fwbs_s3 import (
    calculate_divertor_plate_mass,
)


class DivertorPlateMass(WrapsFunction):
    """cottax node: `calculate_divertor_plate_mass`, unchanged, ports declared."""

    fn = calculate_divertor_plate_mass

    a_div_surface_total = From(divertor)
    den_div_structure = From(divertor)
    f_vol_div_coolant = From(divertor)
    dx_div_plate = From(divertor)

    m_div_plate = OutputInto(divertor)
