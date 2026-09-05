"""Pure-functional port of `process/models/stellarator/heating.py` (registry unit #5).

Audit record: `functional_process/_audit/units/models/stellarator/heating.md`. Only the
`isthtr` branches that are self-contained (call no other model) are ported: ECRH
(`isthtr == 1`) and lower-hybrid (`isthtr == 2`). The NBI branch (`isthtr == 3`) calls
`stellarator.current_drive.culnbi()` -- a not-yet-audited model -- so it stays
audit-only per the record.

`calculate_ecrh_heating`/`calculate_lowhyb_heating` are **mutually exclusive
alternatives**: both write the same four downstream fields
(`p_hcd_injected_ions_mw`, `p_hcd_injected_electrons_mw`,
`eta_hcd_primary_injector_wall_plug`, `p_hcd_electric_total_mw`), because `isthtr`
selects exactly one of the three source branches to run -- never more than one node from
{`EcrhHeating`, `LowhybHeating`, an eventual NBI node} belongs in an assembled graph at
once. This is the same "switch picks which node exists" shape as `i_tf_sup` in
`tf_nuclear_heating.py`, not a naming collision to fix.

`calculate_beam_current`/`calculate_fusion_gain` (the source's common tail, after the
`isthtr` branch) read `p_hcd_beam_injected_total_mw`/`p_beam_orbit_loss_mw`, which only
the NBI branch ever assigns -- for ECRH/lowhyb, the source relies on these fields already
being (or defaulting to) `0.0` on `data`, an implicit-io dependency on external
initialisation rather than a value this file produces. Ported as explicit arguments
(caller supplies `0.0` for the non-beam case) rather than silently modelled as always
zero -- see the record's open questions.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    current_drive,
    heat_transport,
    physics,
)
from functional_process.models.stellarator.heating import (
    calculate_beam_current,
    calculate_ecrh_heating,
    calculate_fusion_gain,
    calculate_injected_power_total,
    calculate_lowhyb_heating,
)


class EcrhHeating(ExplicitFunction):
    """cottax node: `calculate_ecrh_heating`, unchanged, ports declared.

    See module docstring -- mutually exclusive with `LowhybHeating` (and an eventual
    NBI node): `isthtr` selects at most one at graph-assembly time.
    """

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return calculate_ecrh_heating(
            p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug
        )


class LowhybHeating(ExplicitFunction):
    """cottax node: `calculate_lowhyb_heating`, unchanged, ports declared.

    See module docstring -- mutually exclusive with `EcrhHeating`.
    """

    p_hcd_lowhyb_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_lowhyb_injector_wall_plug=From(current_drive),
    ):
        return calculate_lowhyb_heating(
            p_hcd_primary_extra_heat_mw, eta_lowhyb_injector_wall_plug
        )


class InjectedPowerTotal(ExplicitFunction):
    """cottax node: `calculate_injected_power_total`, unchanged, ports declared."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_injected_electrons_mw=From(current_drive),
        p_hcd_injected_ions_mw=From(current_drive),
    ):
        return calculate_injected_power_total(
            p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw
        )


class BeamCurrent(ExplicitFunction):
    """cottax node: `calculate_beam_current`, unchanged, ports declared.

    Not yet registered in `total_process.py`: for ECRH/lower-hybrid, its
    `p_hcd_beam_injected_total_mw` input has no producing node in the current graph
    (only the not-yet-ported NBI branch would write it) -- see module docstring and the
    audit record's open questions.
    """

    c_beam_total = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_beam_injected_total_mw=From(current_drive),
        e_beam_kev=From(current_drive),
    ):
        return calculate_beam_current(p_hcd_beam_injected_total_mw, e_beam_kev)


class FusionGain(ExplicitFunction):
    """cottax node: `calculate_fusion_gain`, unchanged, ports declared.

    Not yet registered in `total_process.py` -- same reason as `BeamCurrent`.
    """

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return calculate_fusion_gain(
            p_fusion_total_mw,
            p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw,
        )
