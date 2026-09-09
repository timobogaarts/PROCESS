"""Pure-functional port of `PlasmaInductance`, `process/models/physics/physics.py`
(lines 4702-5150).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.paths import physics, times
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.physics.plasma_inductance import (
    calculate_internal_inductance_menard,
    calculate_internal_inductance_wesson,
    calculate_normalised_internal_inductance_iter_3,
    calculate_volt_second_requirements,
    internal_inductance_norm_scalings,
    internal_inductance_norm_wesson,
)

__all__ = [
    "calculate_internal_inductance_menard",
    "calculate_internal_inductance_wesson",
    "calculate_normalised_internal_inductance_iter_3",
]


class PlasmaInternalInductanceScalings(WrapsFunction):
    """cottax node: `.tokamak.plasma_inductance.scalings`."""

    fn = internal_inductance_norm_scalings

    alphaj = From(physics)
    kappa = From(physics)
    b_plasma_surface_poloidal_average = From(physics)
    plasma_current = From(physics)
    vol_plasma = From(physics)
    rmajor = From(physics)

    ind_plasma_internal_norm_wesson = OutputInto(physics)
    ind_plasma_internal_norm_menard = OutputInto(physics)
    ind_plasma_internal_norm_iter_3 = OutputInto(physics)


class PlasmaInternalInductanceNormWesson(WrapsFunction):
    """cottax node: `.tokamak.plasma_inductance.internal_inductance_norm`."""

    fn = internal_inductance_norm_wesson

    ind_plasma_internal_norm_wesson = From(physics)

    ind_plasma_internal_norm = OutputInto(physics)


class PlasmaVoltSecondRequirements(WrapsFunction):
    """cottax node: `.tokamak.plasma_inductance.volt_seconds`."""

    fn = calculate_volt_second_requirements

    csawth = From(physics)
    eps = From(physics)
    f_c_plasma_inductive = From(physics)
    ejima_coeff = From(physics)
    kappa = From(physics)
    rmajor = From(physics)
    res_plasma = From(physics)
    plasma_current = From(physics)
    t_plant_pulse_fusion_ramp = From(times)
    t_plant_pulse_burn = From(times)
    ind_plasma_internal_norm = From(physics)

    vs_plasma_internal = OutputInto(physics)
    ind_plasma = OutputInto(physics)
    vs_plasma_burn_required = OutputInto(physics)
    vs_plasma_ramp_required = OutputInto(physics)
    vs_plasma_ind_ramp = OutputInto(physics)
    vs_plasma_res_ramp = OutputInto(physics)
    vs_plasma_total_required = OutputInto(physics)
    v_plasma_loop_burn = OutputInto(physics)


class TokamakPlasmaInductance(ModelNamespace):
    """`.tokamak.plasma_inductance` -- three slots, one of them switched."""

    scalings: PlasmaInternalInductanceScalings = PlasmaInternalInductanceScalings()
    """Unconditional -- PROCESS evaluates all three scalings before any switch is read
    (`physics.py:4721-4736`).
    """

    internal_inductance_norm: PlasmaInternalInductanceNormWesson | None = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.i_ind_plasma_internal_norm` -- `1` (Wesson) is written; `0`
    (USER_INPUT) is **no node at all** (`physics.py:4760` selects the field from itself,
    so the field is a run input and the slot is empty); `2` (Menard) is UNPORTED.
    """

    volt_seconds: PlasmaVoltSecondRequirements = PlasmaVoltSecondRequirements()
    """Unconditional -- `calculate_volt_second_requirements` has no branch at all."""
