"""Pure-functional port of `Stellarator.sc_tf_coil_nuclear_heating_iter90` (chunk 1F).

Audit record:
`functional_process/_audit/units/models/stellarator/tf_nuclear_heating.md`.
Ports only the SUPERCONDUCTING branch of the source's `i_tf_sup` switch -- per
`core/solver/switches.md`'s `i_tf_sup` entry (split, high confidence, three independent
data points including this unit) and `naming_convention.md`'s "switches are not ports":
the resistive branch takes no inputs and always returns ten zeros, so it is not a
computation to port, it is the absence of this node in the graph once `i_tf_sup` selects
a non-superconducting coil (see the record's open questions).

`ishmat` (source: "stainless steel coil casing is assumed") is hardcoded to the stainless
column of `coef`/`decay`; the unused tungsten column is dropped entirely rather than kept
as a dead second index, per the record's note.

`ScTfCoilNuclearHeating` below is the `cottax` node. Its output `VarPath`s are
best-effort, not existing PROCESS storage: the source never writes 8 of its 10 return
values to `self.data` (see the record's "cottax node" section) -- `.fwbs.*` is inferred
from the two fields that *are* stored elsewhere in `st_fwbs`, flagged there for whoever
audits 1E1/1E2 to confirm.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    costs,
    fwbs,
    physics,
    tfcoil,
)
from functional_process.models.stellarator.tf_nuclear_heating import (
    calculate_sc_tf_coil_nuclear_heating,
)


class ScTfCoilNuclearHeating(ExplicitFunction):
    """cottax node: `calculate_sc_tf_coil_nuclear_heating`, unchanged, ports declared."""

    coilhtmx = OutputInto(fwbs)
    dpacop = OutputInto(fwbs)
    htheci = OutputInto(fwbs)
    flu_tf_neutron_fast_peak = OutputInto(fwbs)
    pheci = OutputInto(fwbs)
    pheco = OutputInto(fwbs)
    ptfiwp = OutputInto(fwbs)
    ptfowp = OutputInto(fwbs)
    raddose = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        dr_shld_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_tf_wp_with_insulation=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        pflux_fw_neutron_mw=From(physics),
        tfsai=From(tfcoil),
        tfsao=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        f_t_plant_available=From(costs),
        life_plant=From(costs),
    ):
        return calculate_sc_tf_coil_nuclear_heating(
            dr_shld_inboard,
            dr_fw_inboard,
            dr_blkt_inboard,
            dr_shld_outboard,
            dr_fw_outboard,
            dr_blkt_outboard,
            dr_tf_wp_with_insulation,
            dx_tf_wp_insulation,
            pflux_fw_neutron_mw,
            tfsai,
            tfsao,
            dr_tf_plasma_case,
            f_t_plant_available,
            life_plant,
        )
