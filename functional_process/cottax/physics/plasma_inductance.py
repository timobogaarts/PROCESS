"""Pure-functional port of `PlasmaInductance`, `process/models/physics/physics.py`
(lines 4702-5150).

Audit record: `functional_process/_audit/units/models/physics/plasma_inductance.md`.

**Why this is its own module rather than part of `models/physics/physics.py`.**
`PlasmaInductance` is a `Model` class of its own -- `Physics.__init__` takes it by
injection (`physics.py:207`, `self.inductance = plasma_inductance`) and the model tree
already has a `.tokamak.plasma_inductance` slot for it -- but it happens to live in the
same 5000-line source file as `Physics`. Mirroring that file layout would put this port
inside `functional_process/cottax/physics/physics.py`, which is a shared file. The unit
of the port is the *model*, not the source file it was parked in, so it gets its own
module, its own record and its own case. Recorded here because it is the first place in
this port where the mirror-path rule is deliberately not followed.

Scope: the minimal closure for `.physics.vs_plasma_ramp_required`, which
`functional_process/cottax/pfcoil/currents.py::CSFluxSwing` reads
(`process/models/pfcoil.py:624`). That closure is `calculate_volt_second_requirements`
plus whatever fixes `ind_plasma_internal_norm`, which on the reference arm is the Wesson
scaling. `calculate_volt_second_requirements` returns eight values from one expression
tree, so all eight are owned rather than pruned to the one the boundary asks for -- the
same call `structure.md` makes for `Structure.structure`'s five-way tuple.

**`.physics.e_plasma_magnetic_stored` is not here.** `0.5 * ind_plasma *
plasma_current**2` (`physics.py:952-954`) reads `ind_plasma`, which this module owns,
but it is written by `Physics.physics()`, not by `PlasmaInductance`, so it belongs to
`.tokamak.physics`. UNPORTED here.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.paths import physics, times
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


class PlasmaInternalInductanceScalings(ExplicitFunction):
    """cottax node: `.tokamak.plasma_inductance.scalings`.

    Ports the first three statements of `PlasmaInductance.run()`
    (`process/models/physics/physics.py:4721-4736`), which evaluate all three
    normalised-internal-inductance scalings unconditionally, before any switch is read.
    One node, not three: PROCESS computes all three on every pass whatever the switch
    says, so splitting them would invent a structure the source does not have -- and
    they are stored fields in their own right, reported to the mfile.
    """

    ind_plasma_internal_norm_wesson = OutputInto(physics)
    ind_plasma_internal_norm_menard = OutputInto(physics)
    ind_plasma_internal_norm_iter_3 = OutputInto(physics)

    def __call__(
        self,
        alphaj=From(physics),
        kappa=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
        plasma_current=From(physics),
        vol_plasma=From(physics),
        rmajor=From(physics),
    ):
        return internal_inductance_norm_scalings(
            alphaj,
            kappa,
            b_plasma_surface_poloidal_average,
            plasma_current,
            vol_plasma,
            rmajor,
        )


class PlasmaInternalInductanceNormWesson(ExplicitFunction):
    """cottax node: `.tokamak.plasma_inductance.internal_inductance_norm`.

    Occupant for `i_ind_plasma_internal_norm = 1` (`IndInternalNormModel.WESSON`), the
    value set on `large_tokamak_eval.IN.DAT:311`. Owns
    `.physics.ind_plasma_internal_norm` and reads the Wesson scaling and nothing else --
    `get_ind_internal_norm_value` (`physics.py:4752-4764`) builds a three-entry dict and
    indexes it, so the *reads* of the two arms it does not take are real reads in
    PROCESS and invented edges in a port.

    **The `USER_INPUT` arm is not a node at all.** `IndInternalNormModel.USER_INPUT`
    maps `.physics.ind_plasma_internal_norm` to itself (`physics.py:4760`), so an
    occupant for it would read the `VarPath` it owns -- not a `FixedPointFunction` case
    but the *absence* of a node: on that arm the field is a run input with no
    producer, and the graph simply has nothing in this slot. Recorded in
    `plasma_inductance.md`; `MENARD` is an ordinary UNPORTED sibling occupant.
    """

    ind_plasma_internal_norm = OutputInto(physics)

    def __call__(self, ind_plasma_internal_norm_wesson=From(physics)):
        return internal_inductance_norm_wesson(ind_plasma_internal_norm_wesson)


class PlasmaVoltSecondRequirements(ExplicitFunction):
    """cottax node: `.tokamak.plasma_inductance.volt_seconds`.

    Owns all eight outputs of `calculate_volt_second_requirements`, including
    `.physics.vs_plasma_ramp_required` -- the boundary read that
    `models/pfcoil/currents.py::CSFluxSwing` declares -- and `.physics.ind_plasma`,
    which `models/pfcoil/inductance.py::PFCoilInductance` declares.

    No switch: the function has no branch at all. Every value in it is a continuous
    input.
    """

    vs_plasma_internal = OutputInto(physics)
    ind_plasma = OutputInto(physics)
    vs_plasma_burn_required = OutputInto(physics)
    vs_plasma_ramp_required = OutputInto(physics)
    vs_plasma_ind_ramp = OutputInto(physics)
    vs_plasma_res_ramp = OutputInto(physics)
    vs_plasma_total_required = OutputInto(physics)
    v_plasma_loop_burn = OutputInto(physics)

    def __call__(
        self,
        csawth=From(physics),
        eps=From(physics),
        f_c_plasma_inductive=From(physics),
        ejima_coeff=From(physics),
        kappa=From(physics),
        rmajor=From(physics),
        res_plasma=From(physics),
        plasma_current=From(physics),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        ind_plasma_internal_norm=From(physics),
    ):
        return calculate_volt_second_requirements(
            csawth=csawth,
            eps=eps,
            f_c_plasma_inductive=f_c_plasma_inductive,
            ejima_coeff=ejima_coeff,
            kappa=kappa,
            rmajor=rmajor,
            res_plasma=res_plasma,
            plasma_current=plasma_current,
            t_plant_pulse_fusion_ramp=t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn=t_plant_pulse_burn,
            ind_plasma_internal_norm=ind_plasma_internal_norm,
        )


class TokamakPlasmaInductance(ModelNamespace):
    """`.tokamak.plasma_inductance` -- three slots, one of them switched.

    The slot names are the ones each node's own docstring already claims. The Wesson
    internal-inductance occupant here supersedes `plasma_current.py`'s
    `WessonInternalInductance` (see `TokamakPlasmaCurrent`'s docstring): both own
    `.physics.ind_plasma_internal_norm`, one graph admits one producer, and
    `plasma_current.md` open question 1 rules for this module the moment the
    volt-second work exists -- which `volt_seconds` below is.
    """

    scalings: PlasmaInternalInductanceScalings = PlasmaInternalInductanceScalings()
    """Unconditional -- PROCESS evaluates all three scalings before any switch is read
    (`physics.py:4721-4736`)."""

    internal_inductance_norm: PlasmaInternalInductanceNormWesson | None = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.i_ind_plasma_internal_norm` -- `1` (Wesson) is written; `0`
    (USER_INPUT) is **no node at all** (`physics.py:4760` selects the field from
    itself, so the field is a run input and the slot is empty); `2` (Menard) is
    UNPORTED."""

    volt_seconds: PlasmaVoltSecondRequirements = PlasmaVoltSecondRequirements()
    """Unconditional -- `calculate_volt_second_requirements` has no branch at all."""
