"""Pure-functional port of the tokamak plasma-current chain.

Audit record: `functional_process/_audit/units/models/physics/plasma_current.md` —
read it first, especially "## the chain is not one file" and "## the cycle that is not
live here".

**Source spans two PROCESS files, and that is deliberate.** The unit's nominal source is
`process/models/physics/plasma_current.py` (`PlasmaCurrent`), but `PlasmaCurrent.run()`
is empty (`plasma_current.py:75-76`): the model is *called* from
`Physics.run()`, and the three quantities immediately downstream of the current — the
cylindrical safety factor, the current-profile index and the normalised internal
inductance — are computed by functions PROCESS files in
`process/models/physics/physics.py`. They are one chain:

    plasma_current  ->  qstar  ->  alphaj  ->  ind_plasma_internal_norm

Each arrow is a straight data dependency (`physics.py:286`, `:303`, `:330-343`, `:356`
into `PlasmaInductance.run`, `physics.py:4712-4750`), and every one of the three
switches involved (`i_plasma_current`, `i_alphaj`, `i_ind_plasma_internal_norm`) is one
of `_audit/tokamak_scope.md`'s 17 new topology decisions. Splitting the chain across two
port files would have put a switch family's occupants on one side of a file boundary and
their sole consumer on the other; it is ported here in one piece, with `file:line`
attribution into `physics.py` on every function that came from there. **If a later pass
gives `physics.py`'s `PlasmaInductance` its own unit, `WessonInternalInductance` and
`calculate_internal_inductance_wesson` move there wholesale** — flagged, not decided.

**Scope of the first pass: the `large_tokamak_eval.IN.DAT` reference arm.** That file
sets `i_plasma_current = 4` (IPDG89, line 288), `i_alphaj = 1` (Wesson, line 275) and
`i_ind_plasma_internal_norm = 1` (Wesson, line 311). One occupant class per value, per
`next_steps.md` §14.2 — no `i_*` integer appears as a kwarg or inside any body here.

**The ST closing wave (2026-08-29) added `i_plasma_current = 9`** (FIESTA ST), the value
both `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` set. It is the second
occupant of `PlasmaCurrentScaling` and the first evidence that the family's arms differ
in *read set* rather than in constants: FIESTA reads the separatrix `kappa`/`triang`
where IPDG89 reads the 95%-surface pair.

**Not ported, and why:**

- Seven of the nine `i_plasma_current` values. `1` (Peng analytic fit), `2` (Peng
  divertor / TART), `3` (simple ITER cylindrical), `5`/`6` (Todd I/II), `7`
  (Connor-Hastie), `8` (Sauter) — none live on any tracked input,
  and each reads a different set (`len_plasma_poloidal` for 1; `aspect`/`kappa`/`triang`
  for 2; `alphaj`/`alphap`/`pres_plasma_thermal_on_axis` for 7; `kappa`/`triang` for
  8). Their `fq` coefficient functions are two-to-ten-line pure `@staticmethod`s
  (`plasma_current.py:690-1016`) and each becomes an occupant the day a machine selects
  it; the audit record's "switches touched" table carries each one's reads.
  **Value 2 is structurally different from the rest** — it does not go through
  `calculate_cyclindrical_plasma_current` at all (`plasma_current.py:322-330, 392`), so
  its occupant would not share this file's `fq * cylindrical` shape.
- `i_alphaj == 0` (`USER_INPUT`) and `i_ind_plasma_internal_norm == 0` (`USER_INPUT`)
  are **empty slots**, not unported models: `physics.py:338` is literally
  `self.data.physics.alphaj = self.data.physics.alphaj` and `physics.py:4760` selects
  `.physics.ind_plasma_internal_norm` from itself. Under those values the field is a
  boundary input with no producer, exactly the `i_pulsed_plant == 0` shape
  (`tokamak_scope.md`). No class here, and none needed.
- `i_ind_plasma_internal_norm == 2` (`MENARD`) — `calculate_internal_inductance_menard`
  (`physics.py`, reads `.physics.kappa`), not live here. UNPORTED.
- `.physics.alphaj_wesson`, `.physics.ind_plasma_internal_norm_wesson`,
  `.ind_plasma_internal_norm_menard`, `.ind_plasma_internal_norm_iter_3` and the nine
  `.physics.c_plasma_*` fields — **reporting-only**. Measured: no reader anywhere in
  `process/` outside `output()`/`output_volt_second_information()` and
  `core/io/plot/summary.py`. Not carried, same call `plasma_geometry.py`'s port made for
  `PlasmaGeom.output()`. The consequence worth stating: `WessonCurrentProfileIndex` owns
  `.physics.alphaj` **directly** rather than owning `alphaj_wesson` and then copying it,
  so the switched family has one owned `VarPath` and no intermediate.
  `ind_plasma_internal_norm_iter_3` is the one whose omission removes a real edge — it
  reads `.physics.b_plasma_surface_poloidal_average`, `.plasma_current`, `.vol_plasma`
  and `.rmajor` (`physics.py:4731-4736`) purely to print a number.
- `.physics.b_plasma_surface_poloidal_average` — **already owned**, by
  `SurfaceAveragedPoloidalFieldAmperes` in
  `functional_process/cottax/physics/physics.py` (the `i_plasma_current != 2` arm). Not
  duplicated here. The `i_plasma_current == 2` arm computes it from `plascar_bpol`
  (`plasma_current.py:625-688`, this file's nominal source) and stays UNPORTED with the
  rest of value 2.
- `plasma_current_MA` — **not a `DataStructure` field.** `physics_variables.py` has
  `plasma_current` (line 1156) and nothing else; `plasma_current_MA` exists only as an
  MFILE label written inline as `plasma_current / 1.0e6` (`plasma_current.py:100-104`).
  Nothing to own.
- `PlasmaDiamagneticCurrent` (`plasma_current.py:1058-1175`) — same PROCESS file, but a
  different chain (`.current_drive.f_c_plasma_diamagnetic*`, keyed on
  `i_diamagnetic_current`), and not on the route to `.physics.plasma_current`. Out of
  this pass's scope.

**The compound Sauter predicate is not re-derived here.** `plasma_geometry.py`'s
`PlasmaGeometryArm` family is the `i_plasma_current == 8 or i_plasma_shape == SAUTER`
disjunction and owns it (that unit's record, open question 2). This file's occupants are
keyed on `i_plasma_current` alone; the day `Sauter` (8) gets an occupant here, the
factory is what has to make both answers agree.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_current import (
    calculate_current_coefficient_fiesta,
    calculate_current_coefficient_ipdg89,
    calculate_current_profile_index_wesson,
    calculate_cyclindrical_plasma_current,
    calculate_cylindrical_safety_factor,
    calculate_internal_inductance_wesson,
    calculate_plasma_current_fiesta,
    calculate_plasma_current_ipdg89,
)

__all__ = [
    "calculate_current_coefficient_fiesta",
    "calculate_current_coefficient_ipdg89",
    "calculate_cyclindrical_plasma_current",
]


class PlasmaCurrentScaling(ExplicitFunction):
    """The family that owns `.physics.plasma_current` under `i_plasma_current`.

    One occupant per value this port supports -- **this pass ports only `IPDG89_SCALING`
    (4)**, the value `large_tokamak_eval.IN.DAT:288` sets. The other eight are UNPORTED;
    the module docstring and the audit record's "switches touched" table carry each
    one's reads and the reason.

    Ragged in a way worth naming: value `2` (`PENG_DIVERTOR_SCALING`) does not multiply
    `calculate_cyclindrical_plasma_current` at all and reads `aspect`/`kappa`/`triang`
    instead of the 95%-surface pair, and value `7` (`CONNOR_HASTIE_MODEL`) reads
    `.physics.alphaj` -- which this file's `CurrentProfileIndexScaling` *owns*. At
    `i_plasma_current == 7` the chain closes into a genuine three-node SCC; see the
    audit record's "the cycle that is not live here". Nothing in this pass is cyclic.
    """


class Ipdg89PlasmaCurrent(PlasmaCurrentScaling):
    """`i_plasma_current == IPDG89_SCALING` (4) -- the arm `large_tokamak_eval` takes.

    Reads the 95%-flux-surface shaping pair (`kappa95`/`triang95`, produced by
    `plasma_geometry.py`'s `Ipdg89XPointPlasmaShape`) and nothing from the separatrix
    pair; `.physics.q95` is iteration variable 18 and a boundary input to this node.
    """

    plasma_current = OutputInto(physics)

    def __call__(
        self,
        eps=From(physics),
        kappa95=From(physics),
        triang95=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        q95=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_plasma_current_ipdg89(
            eps=eps,
            kappa95=kappa95,
            triang95=triang95,
            rminor=rminor,
            rmajor=rmajor,
            q95=q95,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        )


class FiestaStPlasmaCurrent(PlasmaCurrentScaling):
    """`i_plasma_current == FIESTA_ST_SCALING` (9) -- the arm both tracked spherical
    tokamaks take (`spherical_tokamak_eval.IN.DAT:288`, `st_regression.IN.DAT`).

    **A different read set from `Ipdg89PlasmaCurrent`, not a different constant.** This
    arm reads the *separatrix* shaping pair `.physics.kappa`/`.physics.triang`; the
    IPDG89 arm reads the 95%-flux-surface pair and nothing from the separatrix. The
    remaining five reads are shared. That is the whole reason the family exists, and it
    is the second occupant to demonstrate it.

    `.physics.q95` is iteration variable 18 and `.physics.rmajor` iteration variable 3;
    both are boundary inputs to this node, as in the IPDG89 arm.
    """

    plasma_current = OutputInto(physics)

    def __call__(
        self,
        eps=From(physics),
        kappa=From(physics),
        triang=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        q95=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_plasma_current_fiesta(
            eps=eps,
            kappa=kappa,
            triang=triang,
            rminor=rminor,
            rmajor=rmajor,
            q95=q95,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        )


class PlasmaCylindricalSafetyFactor(ExplicitFunction):
    """cottax node: `qstar`, ports declared.

    Unconditional -- no family, matching `physics.py:303`'s unswitched call. Its readers
    outside this chain are `models/physics/density_limit.py:82`,
    `models/physics/confinement_time.py:232/251/340` and
    `models/stellarator/stellarator.py:2313`.
    """

    qstar = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        plasma_current=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        kappa95=From(physics),
        triang95=From(physics),
    ):
        return calculate_cylindrical_safety_factor(
            rmajor=rmajor,
            rminor=rminor,
            plasma_current=plasma_current,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            kappa95=kappa95,
            triang95=triang95,
        )


class CurrentProfileIndexScaling(ExplicitFunction):
    """The family that owns `.physics.alphaj` under `i_alphaj` (`physics.py:334-348`).

    Two values, and only one of them is a node: `USER_INPUT` (0) is
    `self.data.physics.alphaj = self.data.physics.alphaj` (`physics.py:338`), an
    **empty slot** under which `.physics.alphaj` is a boundary input with no producer.
    `WESSON` (1) is below.
    """


class WessonCurrentProfileIndex(CurrentProfileIndexScaling):
    """`i_alphaj == WESSON` (1) -- `large_tokamak_eval.IN.DAT:275`.

    Owns `.physics.alphaj` outright. PROCESS routes the same number through
    `.physics.alphaj_wesson` first (`physics.py:330`, then `:343`); that field is
    reporting-only, so the copy is not reproduced and there is no second owned
    `VarPath`.
    """

    alphaj = OutputInto(physics)

    def __call__(self, qstar=From(physics), q0=From(physics)):
        return calculate_current_profile_index_wesson(qstar, q0)


class NormalisedInternalInductanceScaling(ExplicitFunction):
    """The family that owns `.physics.ind_plasma_internal_norm` under
    `i_ind_plasma_internal_norm` (`physics.py:4738-4764`).

    Three values. `USER_INPUT` (0) selects the field from itself
    (`physics.py:4760`) -- an **empty slot**, the field being a boundary input.
    `MENARD` (2) is `calculate_internal_inductance_menard(kappa)`, UNPORTED (not live).
    `WESSON` (1) is below.

    **`PlasmaInductance.run()` computes all three scalings unconditionally**
    (`physics.py:4721-4736`) and only then selects one, so the PROCESS body reads
    `alphaj`, `kappa`, `b_plasma_surface_poloidal_average`, `plasma_current`,
    `vol_plasma` and `rmajor` on every run -- five of those six only to fill reporting
    fields. This family's live occupant reads exactly one.
    """


class WessonInternalInductance(NormalisedInternalInductanceScaling):
    """`i_ind_plasma_internal_norm == WESSON` (1) -- `large_tokamak_eval.IN.DAT:311`.

    Readers of what it owns, outside this chain:
    `models/physics/plasma_geometry.py:354` (the `i_plasma_geometry == 9` arm),
    `models/physics/bootstrap_current.py:154/162` and `models/pfcoil.py:451/570`.
    """

    ind_plasma_internal_norm = OutputInto(physics)

    def __call__(self, alphaj=From(physics)):
        return calculate_internal_inductance_wesson(alphaj)


class TokamakPlasmaCurrent(ModelNamespace):
    """`.tokamak.plasma_current` -- the plasma-current chain, three slots.

    The audit record's registration instructions name a fourth slot
    (`internal_inductance: NormalisedInternalInductanceScaling`), written before
    `plasma_inductance.py` existed. Its own open question 1 already gave the rule that
    resolves the collision: *"if the slot is built for real ... then
    `WessonInternalInductance` and `calculate_internal_inductance_wesson` should move
    there and this record should lose its step 4."* The `.tokamak.plasma_inductance`
    slot **is** built for real (`PlasmaVoltSecondRequirements`, which the PF coil
    package reads), and its `PlasmaInternalInductanceNormWesson` owns
    `.physics.ind_plasma_internal_norm` -- two producers of one `VarPath` cannot share a
    graph, so this namespace has three slots and `WessonInternalInductance` above stays
    an unregistered sibling of the occupant that superseded it.
    """

    plasma_current: PlasmaCurrentScaling = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_current` -- nine values, one occupant.
    `4` (IPDG89, `large_tokamak_eval.IN.DAT:288`) is written; the other eight are
    UNPORTED with per-value reasons in `indat.py`."""

    cylindrical_safety_factor: PlasmaCylindricalSafetyFactor = (
        PlasmaCylindricalSafetyFactor()
    )
    """Unconditional -- PROCESS computes `qstar` outside every switch
    (`physics.py:303`)."""

    current_profile_index: CurrentProfileIndexScaling | None = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_alphaj` -- `1` (Wesson) is written; `0` (USER_INPUT) is an **empty
    slot** (`physics.py:338` assigns the field to itself), under which
    `.physics.alphaj` is a boundary input with no producer."""
