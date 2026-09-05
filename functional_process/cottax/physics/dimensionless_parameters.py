"""Pure-functional port of the one real computation inside
`Physics.outplas` (`process/models/physics/physics.py`).

Registry unit #9, chunk C. Audit record:
`functional_process/_audit/units/models/physics/dimensionless_parameters.md` -- read it
first for why `outplas` (1095 source lines) reduces to a single 3-output, 9-input pure
function."""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import physics
from functional_process.models.physics.dimensionless_parameters import (
    calculate_dimensionless_plasma_parameters,
)


class DimensionlessPlasmaParameters(ExplicitFunction):
    """cottax node: `calculate_dimensionless_plasma_parameters`, ports declared."""

    nu_star = OutputInto(physics)
    rho_star = OutputInto(physics)
    beta_mcdonald = OutputInto(physics)

    def __call__(
        self,
        dlamie=From(physics),
        vol_plasma=From(physics),
        rmajor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        eps=From(physics),
        nd_plasma_electron_line=From(physics),
        kappa=From(physics),
        e_plasma_beta=From(physics),
        plasma_current=From(physics),
        m_ions_total_amu=From(physics),
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
