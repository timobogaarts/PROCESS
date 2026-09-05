"""Pure-functional port of the tokamak bootstrap-current chain.

Audit record: `functional_process/_audit/units/models/physics/bootstrap_current.md` --
read it first, especially "## the chain is not one file" and "## the invented
`triang` edge".

**Scope of this pass: the `large_tokamak_eval.IN.DAT` reference arm.** That file sets
`i_bootstrap_current = 4` (Sauter, line 286) and leaves `i_diamagnetic_current` and
`i_pfirsch_schluter_current` at their `physics_variables.py` defaults of `0`
(lines 856, 895; the `OUT.DAT` confirms `i_diamagnetic_current = 0` at line 970). One
occupant class per switch value, per `next_steps.md` §14.2 -- no `i_*` integer appears as
a kwarg or inside any body here. The other thirteen `i_bootstrap_current` values, the two
non-zero `i_diamagnetic_current` values and the one non-zero `i_pfirsch_schluter_current`
value were UNPORTED; the audit record's "switches touched" tables carry each one's reads
and reason.

**The ST closing wave (2026-08-29) added the two SCENE fits.**
`i_diamagnetic_current = 2` (`SceneDiamagneticCurrent`) and
`i_pfirsch_schluter_current = 1` (`ScenePfirschSchluterCurrent`) are what both
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` set, and they are the first
two occupants that make `.current_drive.f_c_plasma_diamagnetic` and
`.current_drive.f_c_plasma_pfirsch_schluter` non-zero anywhere in this port -- so they
are also the first live test of `PlasmaCurrentFractions`' three-term sum. Only
`i_diamagnetic_current == 1` (Hender) remains UNPORTED in the two families.

**Source spans two PROCESS files, and that is deliberate.** The unit's nominal source is
`process/models/physics/bootstrap_current.py` (`PlasmaBootstrapCurrent`,
`SauterBootstrapCurrent`), but the *bookkeeping* that turns a bootstrap fraction into the
auxiliary and inductive fractions is written inline in `Physics.run()`,
**`process/models/physics/physics.py:543-588`**, immediately after
`self.plasma_bootstrap_current.run()`. They are one chain:

    f_c_plasma_bootstrap_sauter  ->  f_c_plasma_bootstrap (capped)
                                       |
                                       +-> f_c_plasma_internal
                                             |
                                             +-> f_c_plasma_auxiliary
                                                 f_c_plasma_inductive

`.physics.f_c_plasma_auxiliary` is a declared boundary read of
`functional_process/cottax/physics/current_drive.py`'s `HcdPrimaryInjectedPower`, and
`physics.py:585-588` is its **only** producer anywhere in `process/`. Leaving it on the
boundary while porting the thing it is computed from would have been the invented-edge
defect this project exists to remove, so the six-statement block is ported here with
`file:line` attribution on every function that came from `physics.py`. **If a later pass
gives `physics.py`'s current-fraction bookkeeping its own unit,
`PlasmaCurrentFractions` and `calculate_plasma_current_fractions` move there wholesale**
-- flagged, not decided. The oracle weakness this creates is real and is stated in the
test module: `Physics.run()` has no callable sub-shell, so that one contract's reference
is a second reading of the source rather than PROCESS's own execution, and it is
anchored separately on `large_tokamak_eval`'s recorded MFILE numbers.

**The whole `i_bootstrap_current` family is computed by PROCESS on every run.**
`PlasmaBootstrapCurrent.run` (`bootstrap_current.py:81-262`) evaluates all fourteen
scalings into fourteen `.current_drive.f_c_plasma_bootstrap_*` fields and only then
indexes the family by the switch (`get_bootstrap_current_fraction_value`, `:264-298`).
`_audit/tokamak_boundary.md`'s note on this slot reads that as "one node producing the
family plus an index, not an occupant per arm". **This port takes the other reading**,
because §14.2's binding policy binds: the thirteen unselected scalings are dead work
whose only consumers are `output()` and the MFILE, and computing them would give this
node the union of fourteen arms' reads -- twenty-odd `.physics` fields against the
seventeen the Sauter arm actually needs, including `alphaj`, `alphap`,
`beta_thermal_poloidal_vol_avg`, `ind_plasma_internal_norm`, `kappa`, `eps` and
`nd_plasma_electron_max_array[6]`, none of which the live arm touches. The disagreement
is recorded in the audit record rather than smoothed over.

**Not ported, and why:**

- Thirteen of the fourteen `i_bootstrap_current` values -- see the audit record.
  Value `0` (`USER_INPUT`) is an **empty slot** rather than an unported model:
  `get_bootstrap_current_fraction_value` selects `.current_drive.f_c_plasma_bootstrap`
  from itself (`bootstrap_current.py:283`) and `physics.py:549-551` exempts it from the
  cap, so under it the field is a boundary input with no producer.
- The thirteen `.current_drive.f_c_plasma_bootstrap_*` sibling fields and
  `.current_drive.bscf_gi_i`/`bscf_gi_ii` -- **reporting-only**. Measured: no reader
  anywhere in `process/` outside `PlasmaBootstrapCurrent.output`
  (`bootstrap_current.py:1271-1445`) and `core/io/plot/summary.py:8934-8946`.
  `.current_drive.f_c_plasma_bootstrap_sauter` *is* carried, because it is the live arm's
  own value and PROCESS stores it (`bootstrap_current.py:138`) before the cap is applied.
- `.physics.err242` / `.physics.err243` -- reporting-only flags, read only at
  `bootstrap_current.py:1406` and `:1410`, both inside `output()`. Same call
  `plasma_current.py`'s port made for `.physics.alphaj_wesson`.
- `.current_drive.f_c_plasma_pfirsch_schluter_scene` (`physics.py:534`) and
  `.current_drive.f_c_plasma_diamagnetic_hender`/`_scene`
  (`plasma_current.py:1068-1079`) -- computed unconditionally by PROCESS, selected only
  at `i_pfirsch_schluter_current == 1` / `i_diamagnetic_current in (1, 2)`, and otherwise
  read only by `output()`. Reporting-only on this arm.
- `_trapped_particle_fraction_sauter`'s `fit = 1` (Sauter 2002) and `fit = 2` (Sauter
  2016) branches (`bootstrap_current.py:2509-2527`). `fit` is a **method-choice static
  kwarg with no `DataStructure` field behind it** and no call site anywhere passes it, so
  the port drops the parameter entirely and carries the ASTRA branch alone. See "## the
  invented `triang` edge" -- this is what makes `.physics.triang` unread by the live
  chain.
- `PlasmaBootstrapCurrent.output` and `SauterBootstrapCurrent.run`/`output` -- reporting,
  and the latter two are empty (`bootstrap_current.py:1450-1454`).
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.stated import StatesValues
from functional_process.paths import current_drive, physics
from functional_process.models.physics.bootstrap_current import (
    _beta_poloidal_sauter,
    _beta_poloidal_total_sauter,
    _calculate_l31_32_coefficient,
    _calculate_l31_coefficient,
    _calculate_l34_alpha_31_coefficient,
    _electron_collisionality_sauter,
    _ion_collisionality_sauter,
    _trapped_particle_fraction_sauter,
    bootstrap_fraction_sauter,
    calculate_plasma_current_fractions,
    diamagnetic_fraction_scene,
    enforce_bootstrap_current_fraction_max,
    ps_fraction_scene,
    sauter_bootstrap_current_fraction,
)

__all__ = [
    "_beta_poloidal_sauter",
    "_beta_poloidal_total_sauter",
    "_calculate_l31_32_coefficient",
    "_calculate_l31_coefficient",
    "_calculate_l34_alpha_31_coefficient",
    "_electron_collisionality_sauter",
    "_ion_collisionality_sauter",
    "_trapped_particle_fraction_sauter",
    "bootstrap_fraction_sauter",
    "enforce_bootstrap_current_fraction_max",
]


class BootstrapCurrentFractionScaling(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_bootstrap` under
    `i_bootstrap_current` (`bootstrap_current.py:250-262`, `:264-298`).

    Fourteen values. **This pass ports only `SAUTER` (4)**, the value
    `large_tokamak_eval.IN.DAT:286` sets; `USER_INPUT` (0) is an **empty slot** under
    which the field is a boundary input with no producer, and the other twelve are
    UNPORTED (audit record's "switches touched").

    Each occupant owns the *capped* fraction as well as its own scaling value, because
    the cap at `physics.py:546-556` is exempted for `USER_INPUT` alone -- so the cap is
    per-occupant behaviour, and putting it in a node of its own would answer
    `i_bootstrap_current` twice (`model_tree_design.md` §8 step 4d).

    Ragged in a way worth naming: value `4` is the only arm that integrates over the
    plasma profiles at all. The other thirteen are closed-form expressions in a dozen
    volume-averaged scalars, so an occupant for any of them will not share this one's
    profile reads.
    """


class SauterBootstrapCurrentFraction(BootstrapCurrentFractionScaling):
    """`i_bootstrap_current == SAUTER` (4) -- the arm `large_tokamak_eval` takes.

    Seventeen reads plus one static shape, against the twenty-odd `.physics` fields
    PROCESS's `PlasmaBootstrapCurrent.run` touches to fill the whole fourteen-member
    family (module docstring). `.physics.triang` is not read -- see
    `_trapped_particle_fraction_sauter` and the audit record's "## the invented `triang`
    edge".

    Owns three `VarPath`s:

    - `.current_drive.f_c_plasma_bootstrap_sauter` -- the scaling's own value, scaled by
      `cboot` (`bootstrap_current.py:141-143`). PROCESS stores it; it is the field the
      MFILE reports.
    - `.physics.j_plasma_bootstrap_sauter_profile` -- the bootstrap current density
      profile (`bootstrap_current.py:139`). No reader outside `output()`; declared
      anyway, so the graph shows what the source computes, following `ProfileGrid`'s
      precedent in `profiles.py`. `Graph.prune` drops it.
    - `.current_drive.f_c_plasma_bootstrap` -- the selected, capped fraction
      (`bootstrap_current.py:255-257` then `physics.py:552-555`).
    """

    n_plasma_profile_elements: int = eqx.field(static=True)

    f_c_plasma_bootstrap_sauter = OutputInto(current_drive)
    j_plasma_bootstrap_sauter_profile = OutputInto(physics)
    f_c_plasma_bootstrap = OutputInto(current_drive)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        a_plasma_poloidal=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        q0=From(physics),
        q95=From(physics),
        m_ions_total_amu=From(physics),
        f_plasma_fuel_helium3=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        plasma_current=From(physics),
        cboot=From(current_drive),
        f_c_plasma_bootstrap_max=From(current_drive),
    ):
        return sauter_bootstrap_current_fraction(
            n_plasma_profile_elements=self.n_plasma_profile_elements,
            radius_plasma_profile_norm=radius_plasma_profile_norm,
            nd_plasma_electron_profile=nd_plasma_electron_profile,
            temp_plasma_electron_profile_kev=temp_plasma_electron_profile_kev,
            a_plasma_poloidal=a_plasma_poloidal,
            rminor=rminor,
            rmajor=rmajor,
            nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
            q0=q0,
            q95=q95,
            m_ions_total_amu=m_ions_total_amu,
            f_plasma_fuel_helium3=f_plasma_fuel_helium3,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            plasma_current=plasma_current,
            cboot=cboot,
            f_c_plasma_bootstrap_max=f_c_plasma_bootstrap_max,
        )


class PlasmaDiamagneticCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_diamagnetic` under
    `i_diamagnetic_current` (`plasma_current.py:1081-1094`).

    Three values: `NONE` (0, below), `HENDER_ST_FIT` (1) and `SCENE_FIT` (2, below --
    added by the ST closing wave, 2026-08-29, for the two tracked spherical tokamaks).
    `HENDER_ST_FIT` reads only `.physics.beta_total_vol_avg` and stays UNPORTED: it is
    not live on any tracked input, and it is a genuinely smaller read set than SCENE's,
    so it is its own occupant rather than a constant in this one.
    """


class NoDiamagneticCurrent(PlasmaDiamagneticCurrentFraction, StatesValues):
    """`i_diamagnetic_current == NONE` (0) -- the default, and this input's value.

    **A node that computes nothing, and that is the finding rather than an accident.**
    (It reads its own output's statement and nothing else -- `models/stated.py` derives
    that read, so the claim is unchanged from when it read literally nothing.) PROCESS
    never assigns `.current_drive.f_c_plasma_diamagnetic` on this arm
    (`plasma_current.py:1081-1094` has no `else`), the field is not settable from
    `IN.DAT` (`process/core/input.py` has no entry for it), and no other model writes it
    -- so it holds its `current_drive_variables.py:77` default of `0.0` for the whole
    run. Declaring the zero is what keeps a computed quantity off the boundary, exactly
    as `current_drive.py`'s `NoSecondaryHcd` argues: `_audit/tokamak_boundary.md` §"The
    twelve that are simply inputs" is explicit that the boundary is for variables PROCESS
    computes *nowhere*, not for ones a switch happened to skip.

    The zero is **stated** and not a literal in the body: a constant this node's readers
    multiply by is one XLA deletes the readers of (`models/stated.py`,
    `_audit/optimise_design.md` §28, §34).
    """

    f_c_plasma_diamagnetic = OutputInto(current_drive)
    """The diamagnetic current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_diamagnetic`."""


class SceneDiamagneticCurrent(PlasmaDiamagneticCurrentFraction):
    """`i_diamagnetic_current == SCENE_FIT` (2) -- both tracked spherical tokamaks.

    Ports `PlasmaDiamagneticCurrent.diamagnetic_fraction_scene`
    (`process/models/physics/plasma_current.py:1158-1179`, `@nb.njit` dropped) and the
    `SCENE_FIT` limb of the selection at `:1088-1094`.

    **Owns `.current_drive.f_c_plasma_diamagnetic` directly, not
    `f_c_plasma_diamagnetic_scene` and then a copy.** PROCESS computes *both* fits
    unconditionally (`:1068-1079`) and then selects; the Hender and SCENE sibling fields
    are reporting-only, measured (this module's docstring, and
    `bootstrap_current.md` "## data footprint"). One owned `VarPath`, no intermediate --
    the same call `WessonCurrentProfileIndex` made for `.physics.alphaj_wesson`.

    The `q95 / q0` quotient is unguarded, exactly as PROCESS writes it: at `q0 == 0` the
    value is `inf` and the tangent is not finite. `q0` is a plasma input (`2.0` on
    `spherical_tokamak_eval.IN.DAT:287`) and PROCESS neither guards nor clamps it; the
    zero-boundary probe therefore steps aside on its own, because zeroing `q0` makes the
    *value* non-finite too.
    """

    f_c_plasma_diamagnetic = OutputInto(current_drive)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        q95=From(physics),
        q0=From(physics),
    ):
        return diamagnetic_fraction_scene(beta=beta_total_vol_avg, q95=q95, q0=q0)


class PlasmaPfirschSchluterCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_pfirsch_schluter` under
    `i_pfirsch_schluter_current` (`physics.py:538-541`).

    Two values: `0` (below) and `1`, which copies
    `.current_drive.f_c_plasma_pfirsch_schluter_scene` -- `ps_fraction_scene(
    beta_total_vol_avg)`, `physics.py:534-536`. Value `1` is `ScenePfirschSchluterCurrent`
    below, added by the ST closing wave (2026-08-29); it is not live on
    `large_tokamak_eval`, whose MFILE records the enforced fraction as `0.0`
    (line 14427) while the SCENE value it would have copied is `-2.9e-3` (line 6900).
    """


class NoPfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction, StatesValues):
    """`i_pfirsch_schluter_current == 0` -- the default, and this input's value.

    Same shape and same argument as `NoDiamagneticCurrent`: `physics.py:538-541` has no
    `else`, the field is not an `IN.DAT` variable, nothing else writes it, and it holds
    its `current_drive_variables.py:283` default of `0.0`. The zero is **stated** for the
    reason `NoDiamagneticCurrent`'s is.
    """

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)
    """The Pfirsch-Schluter current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_pfirsch_schluter`."""


class ScenePfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction):
    """`i_pfirsch_schluter_current == 1` -- both tracked spherical tokamaks.

    Ports the module-level `ps_fraction_scene`
    (`process/models/physics/physics.py:161-179`, `@nb.jit` dropped) and the `== 1` limb
    of `physics.py:538-541`.

    Owns `.current_drive.f_c_plasma_pfirsch_schluter` directly rather than
    `.f_c_plasma_pfirsch_schluter_scene` and a copy, for the reason
    `SceneDiamagneticCurrent` gives: the `_scene` sibling is reporting-only, measured.

    **The fraction is negative** (`-0.09 * beta`), and that is PROCESS's, not a sign
    slip: the Pfirsch-Schlüter current opposes the plasma current, and
    `PlasmaCurrentFractions` sums it into `.current_drive.f_c_plasma_internal` with the
    bootstrap and diamagnetic fractions as written.
    """

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)

    def __call__(self, beta_total_vol_avg=From(physics)):
        return ps_fraction_scene(beta=beta_total_vol_avg)


class PlasmaCurrentFractions(ExplicitFunction):
    """cottax node: `calculate_plasma_current_fractions`, ports declared.

    Unconditional -- `physics.py:558-588` runs on every device path with no switch on it.

    `.physics.f_c_plasma_non_inductive` is iteration variable 44
    (`core/solver/iteration_variables.py:72`) and a boundary input to this node;
    `large_tokamak_eval.IN.DAT:283` seeds it at `0.4242184436680697`.

    Owns the three fields `physics.py` writes here.
    `.current_drive.f_c_plasma_internal` has no reader outside this block and
    `output()`, but it is the clamped intermediate the other two are defined against, so
    it is declared rather than inlined; `Graph.prune` drops it if nothing wants it.
    """

    f_c_plasma_internal = OutputInto(current_drive)
    f_c_plasma_auxiliary = OutputInto(physics)
    f_c_plasma_inductive = OutputInto(physics)

    def __call__(
        self,
        f_c_plasma_bootstrap=From(current_drive),
        f_c_plasma_diamagnetic=From(current_drive),
        f_c_plasma_pfirsch_schluter=From(current_drive),
        f_c_plasma_non_inductive=From(physics),
    ):
        return calculate_plasma_current_fractions(
            f_c_plasma_bootstrap=f_c_plasma_bootstrap,
            f_c_plasma_diamagnetic=f_c_plasma_diamagnetic,
            f_c_plasma_pfirsch_schluter=f_c_plasma_pfirsch_schluter,
            f_c_plasma_non_inductive=f_c_plasma_non_inductive,
        )
