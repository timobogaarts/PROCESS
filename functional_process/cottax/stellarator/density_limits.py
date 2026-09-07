"""Pure-functional port of `process/models/stellarator/density_limits.py`."""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.cottax.paths import (
    physics,
    stellarator,
)
from functional_process.models.stellarator.density_limits import (
    calculate_ecrh_density_limit,  # noqa: F401
    calculate_ecrh_density_limit_parabolic,
    calculate_sudo_density_limit,
)


class SudoDensityLimit(ExplicitFunction):
    """cottax node: `calculate_sudo_density_limit`, unchanged, ports declared."""

    nd_plasma_electrons_max = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        p_plasma_loss_mw=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_electron_line=From(physics),
    ):
        return calculate_sudo_density_limit(
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            rmajor,
            rminor,
            nd_plasma_electrons_vol_avg,
            nd_plasma_electron_line,
        )


class EcrhDensityLimit(ExplicitFunction):
    """cottax node: `calculate_ecrh_density_limit_parabolic`, ports declared."""

    dlimit_ecrh = OutputInto(stellarator)
    bt_max_ecrh = OutputInto(stellarator)

    def __call__(
        self,
        max_gyrotron_frequency=From(stellarator),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_ecrh_density_limit_parabolic(
            max_gyrotron_frequency, b_plasma_toroidal_on_axis
        )
