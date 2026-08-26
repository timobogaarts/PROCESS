"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial, the minimal closure for `.fwbs.r_cryostat_inboard`.

Audit record: `functional_process/_audit/units/models/cryostat.md`. **Not** the
stellarator's cryostat, which is `process/models/stellarator/stellarator.py:1282-1330`
(unit #1 chunk S5, already ported, already a slot of `.stellarator.fwbs`) -- two
different models of two different devices' cryostats.

`tokamak_boundary.md`'s `.tokamak.cryostat` slot has exactly one read:
`.fwbs.r_cryostat_inboard`, consumed by `.buildings.sizing`. `Cryostat.
external_cryo_geometry` (`process/models/cryostat.py:25-85`) computes six fields in a
straight-line sequence (`r_cryostat_inboard` -> `dz_pf_cryostat` -> `z_cryostat_half_
inside` -> `dz_tf_cryostat`/`vol_cryostat_internal` -> `vol_cryostat` -> `dewmkg`); only
the first is on this pass's boundary and nothing downstream of it is needed to produce
it, so per the wave-1 scope discipline ("port the minimal closure of functions that
produces your slot's listed output variables") only that first line is ported. The
other five fields are UNPORTED -- see `cryostat.md`.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import fwbs, pf_coil


def calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat):
    """Cryostat inboard radius (m): furthest PF coil's outer radius plus clearance.

    Ports `Cryostat.external_cryo_geometry`, `process/models/cryostat.py:39-41`,
    unchanged (`np.max` -> `jnp.max`).

    Parameters
    ----------
    r_pf_coil_outer :
        Outer radius of each PF coil (m), fixed-size array.
        `.pf_coil.r_pf_coil_outer`.
    dr_pf_cryostat :
        Clearance between the outermost PF coil and the cryostat (m).
        `.fwbs.dr_pf_cryostat`.

    Returns
    -------
    :
        Cryostat inboard radius (m).
    """
    return jnp.max(r_pf_coil_outer) + dr_pf_cryostat


class Cryostat(ExplicitFunction):
    """cottax node: `.tokamak.cryostat`. Owns `.fwbs.r_cryostat_inboard` only -- see
    module docstring for why the other five fields `external_cryo_geometry` computes
    are UNPORTED.
    """

    r_cryostat_inboard = OutputInto(fwbs)

    def __call__(
        self,
        r_pf_coil_outer=From(pf_coil),
        dr_pf_cryostat=From(fwbs),
    ):
        return calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat)
