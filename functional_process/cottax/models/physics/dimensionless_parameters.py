"""Pure-functional port of the one real computation inside `Physics.outplas`
(`process/models/physics/physics.py`).
"""

from functional_process.cottax.paths import physics
from functional_process.models.physics.dimensionless_parameters import (
    calculate_dimensionless_plasma_parameters,
)


# cottax node: `calculate_dimensionless_plasma_parameters`, ports declared.
DimensionlessPlasmaParameters = physics.function(
    calculate_dimensionless_plasma_parameters, "nu_star", "rho_star", "beta_mcdonald"
)
