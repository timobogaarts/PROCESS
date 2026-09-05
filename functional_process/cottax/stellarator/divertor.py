"""Pure-functional port of `st_div` (registry unit #4, `divertor.py`).

Audit record: `functional_process/_audit/units/models/stellarator/divertor.md`. `st_div`
is a single 239-line module with one computational function and one purely-reporting
`output()` (its arguments are already-computed locals, no further computation) -- the
whole file is tier-1, self-contained, and has no internal solve or switches, so it is
ported in full.

`Divertor` below is the `cottax` node (`ExplicitFunction`, see `_audit/schema.md`'s
"cottax node" section). `.divertor.a_div_surface_total` is the field chunk 1E2's audit
(`stellarator_E2_fwbs_neutronics.md`) found `st_fwbs` falling back to a hardcoded `50.0`
for on its first call, because `st_fwbs` runs before `st_div` in the pipeline
(`stellarator.py`'s `run()`) -- confirmed here as the real, unconditional producer of
that field. That first-call fallback is a call-*order* problem in `st_fwbs`/`run()`, not
anything about this file: `st_div` itself has no missing input and no internal state.
"""

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
