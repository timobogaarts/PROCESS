"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial, the minimal closure for `.fwbs.r_cryostat_inboard`.

Audit record: `functional_process/_audit/units/models/cryostat.md`. **Not** the
stellarator's cryostat, which is `process/models/stellarator/stellarator.py:1282-1330`
(unit #1 chunk S5, already ported, already a slot of `.stellarator.fwbs`) -- two
different models of two different devices' cryostats.

`Cryostat.external_cryo_geometry` (`process/models/cryostat.py:25-85`) computes **seven**
fields in one straight-line sequence (`r_cryostat_inboard` -> `dz_pf_cryostat` ->
`z_cryostat_half_inside` -> `dz_tf_cryostat`/`vol_cryostat_internal` -> `vol_cryostat`
-> `dewmkg`). **All seven are ported now; only the first was until 2026-08-30.** (Six is
what this docstring and `cryostat.md` both said, counting the arrow *stages* above --
`dz_tf_cryostat` and `vol_cryostat_internal` share one. The field count is seven, and it
is the field count that a node's outputs have to match.)

The original scope was correct on its own terms and wrong on the measure that did not
exist yet. `tokamak_boundary.md` gave this slot exactly one listed output --
`.fwbs.r_cryostat_inboard`, read by `.buildings.sizing` -- and the wave-1 discipline
("port the minimal closure of functions that produces your slot's listed output
variables") stops there, because nothing downstream of that first line is needed to
produce it. What that walk could not see is a field with a reader *outside* this
graph: `.fwbs.dewmkg` is read by `structure.py`'s `StructureMasses`
(`.structure.coldmass`, and it is 14.4 million kg of it) and `.buildings.dz_tf_cryostat`
by the site-buildings sizing, and both sat frozen at `0.0` while PROCESS recomputed
them every pass. `boundary.unproduced_but_computed` is what named them, and this module
is two of its rows.

The whole method is one node, as it is one method: every field is a plain function of
the ones above it and there is no branch anywhere in it, so splitting it would mint
intermediate places for nothing.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cryostat import (
    calculate_external_cryo_geometry,
    calculate_r_cryostat_inboard,  # noqa: F401
)
from functional_process.paths import blanket, build, buildings, fwbs, pf_coil


class Cryostat(ExplicitFunction):
    """cottax node: `.tokamak.cryostat`, owning all seven fields
    `external_cryo_geometry` writes -- `.fwbs.r_cryostat_inboard` only until 2026-08-30.

    The three areas the outputs land in are PROCESS's own scattering, not a choice made
    here: the lid clearance goes to `.blanket`, the TF clearance to `.buildings`, and
    the rest to `.fwbs`, all from one method.
    """

    r_cryostat_inboard = OutputInto(fwbs)
    dz_pf_cryostat = OutputInto(blanket)
    z_cryostat_half_inside = OutputInto(fwbs)
    dz_tf_cryostat = OutputInto(buildings)
    vol_cryostat_internal = OutputInto(fwbs)
    vol_cryostat = OutputInto(fwbs)
    dewmkg = OutputInto(fwbs)

    def __call__(
        self,
        r_pf_coil_outer=From(pf_coil),
        dr_pf_cryostat=From(fwbs),
        f_z_cryostat=From(build),
        z_pf_coil_upper=From(pf_coil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        dr_cryostat=From(build),
        vol_vv=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_external_cryo_geometry(
            r_pf_coil_outer,
            dr_pf_cryostat,
            f_z_cryostat,
            z_pf_coil_upper,
            z_tf_inside_half,
            dr_tf_inboard,
            dr_cryostat,
            vol_vv,
            den_steel,
        )
