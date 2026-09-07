"""The availability subsystem's namespace."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.availability.availability import Avail, CplifeAvail
from functional_process.cottax.power.electric_production import (
    PlantElectricProductionReactor,
    PowerProfilesOverTime,
)


class Availability(ModelNamespace):
    """Plant availability and component lifetimes."""

    electric_production: PowerProfilesOverTime | PlantElectricProductionReactor = (
        dataclasses.field(kw_only=True)
    )
    """Net electric power over the pulse cycle (`.costs.ireactor`, default 1)."""

    # `PowerProfilesOverTime`/`PlantElectricProductionReactor` are the two arms of the
    # `.costs.ireactor` slot below, not unswitched members -- see that slot.
    # `availability.py` (unit #17). `Stellarator.run()`'s solve-time branch calls
    # `self.availability.avail()` directly (`stellarator.py:175`), bypassing
    # `.costs.i_plant_availability`'s dispatch entirely -- so `Avail` (not `Avail2`/
    # `AvailSt`) is the node actually exercised at solve time regardless of that
    # switch's value, and belongs in the unswitched part, not behind a slot. Its
    # `.costs.cplife` self-loop is resolved the same way as `plasma_composition`'s
    # `first_call`/`thermal_cryo.py`'s six fields above: `CplifeAvail`
    # (`FixedPointFunction`) owns `.costs.cplife` alone; `Avail` (`ExplicitFunction`)
    # owns every other output, reading `cplife` as a plain `FromExactly`.
    # `CpLifetimeSuperconducting`/`CpLifetimeResistive` are deliberately NOT registered:
    # `CplifeAvail.step` duplicates their `i_tf_sup` dispatch inline instead of calling
    # them (see `CplifeAvail`'s own docstring) precisely so only one node ever owns
    # `.costs.cplife` -- registering both pairs together would conflict.
    # `WardTaylorAvailability` is NOT registered either: PROCESS's own default
    # `.costs.i_plant_availability = 2` (MORRIS, `cost_variables.py:408`) means `avail()`
    # 's internal `WARD_TAYLOR` branch (`i_plant_availability == 1`) never fires, so
    # `.costs.f_t_plant_available` has no producer under the default configuration --
    # unconditional registration would reproduce the `EcrhDensityLimit` bug class
    # (computing a value the default configuration never computes), and it cannot be a
    # `Switch` either (no counterpart node exists for any other value, so
    # `check_arms_are_exclusive` would reject a one-real-arm pairing, same as
    # `i_vacuum_pumping`/`i_cost_model`). `ibkt_life=0`/`itart=0` match
    # `cost_variables.py:416`/`physics_variables.py:994`'s defaults.
    avail: Avail = dataclasses.field(kw_only=True)
    """Component lifetimes and the pulse-cycle factors -- one occupant per
    `.costs.ibkt_life` value.
    """
    cplife_avail: CplifeAvail | None = dataclasses.field(kw_only=True)
    """The centrepost lifetime, or **nothing at all** -- `.physics.itart` and
    `.tfcoil.i_tf_sup` jointly.
    """
