"""Pure-functional port of the one real computation inside `Physics.outplas`
(`process/models/physics/physics.py`).
"""

from functional_process.cottax.paths import physics
from functional_process.models.physics.dimensionless_parameters import (
    calculate_dimensionless_plasma_parameters,
)


# cottax node: `calculate_dimensionless_plasma_parameters`, ports declared.
def _dimensionless_plasma_parameters(
    dlamie,
    vol_plasma,
    rmajor,
    b_plasma_toroidal_on_axis,
    eps,
    nd_plasma_electron_line,
    kappa,
    e_plasma_beta,
    plasma_current,
    m_ions_total_amu,
):
    return calculate_dimensionless_plasma_parameters(
        dlamie,
        vol_plasma,
        rmajor,
        b_plasma_toroidal_on_axis,
        eps,
        nd_plasma_electron_line,
        kappa,
        e_plasma_beta,
        plasma_current,
        m_ions_total_amu,
    )


DimensionlessPlasmaParameters = physics.function(
    _dimensionless_plasma_parameters, "nu_star", "rho_star", "beta_mcdonald"
)
