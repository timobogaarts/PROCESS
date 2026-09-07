"""Pure-functional port of `st_div` (registry unit #4, `divertor.py`)."""

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
    divertor,
    first_wall,
    fwbs,
    physics,
    stellarator,
)
from functional_process.models.stellarator.divertor import (
    calculate_divertor,
)


class Divertor(ExplicitFunction):
    """cottax node: `calculate_divertor`, unchanged, ports declared."""

    pflux_div_heat_load_mw = OutputInto(divertor)
    a_div_surface_total = OutputInto(divertor)
    f_ster_div_single = OutputInto(fwbs)

    def __call__(
        self,
        flpitch=From(stellarator),
        rmajor=From(physics),
        p_plasma_separatrix_mw=From(physics),
        anginc=From(divertor),
        xpertin=From(divertor),
        tdiv=From(divertor),
        m_fuel_amu=From(physics),
        bmn=From(stellarator),
        shear=From(stellarator),
        n_res=From(stellarator),
        f_w=From(stellarator),
        m_res=From(stellarator),
        fdivwet=From(stellarator),
        f_asym=From(stellarator),
        a_fw_total=From(first_wall),
    ):
        return calculate_divertor(
            flpitch,
            rmajor,
            p_plasma_separatrix_mw,
            anginc,
            xpertin,
            tdiv,
            m_fuel_amu,
            bmn,
            shear,
            n_res,
            f_w,
            m_res,
            fdivwet,
            f_asym,
            a_fw_total,
        )
