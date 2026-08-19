---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/physics/plasma_profiles.py`, whole file (430 lines, 7 methods on one
class `PlasmaProfile(Model)`). Registry unit #12; scope was recorded as
`PlasmaProfile.run()`, which is the whole file — `run()` calls `parameterise_plasma()`,
which reaches every other method.

### Scope correction: `process/models/physics/profiles.py` is a missing registry unit

`PlasmaProfile` holds two injected sub-models, `self.neprofile` / `self.teprofile`
(constructed in `process/main.py:674-676` as `NeProfile()` / `TeProfile()`), and **calls
`.run()` on both, in both branches** (lines 120-121, 196-197). Those classes live in
`process/models/physics/profiles.py` — 558 LOC, `Profile(Model, ABC)` +
`NeProfile` + `TeProfile` + two `IntEnum`s — and that file **appears nowhere in
`unit_registry.md`**.

This is the same class of scoping miss as `coils/`, `rether`, and
`fusion_reactions.py`/`radiation_power.py`: the original scope grep looked for
`self.<attr>.<method>` on `Stellarator`'s *own* injected sub-models, and `profiles.py` is
reached one level deeper — injected into `PlasmaProfile`, which is itself injected into
`Stellarator`. `profiles.py` also imports `PlasmaDensityLimit` from
`physics/density_limit.py`, a third level.

**Consequence for this unit**: `PlasmaProfile.run()` is *not* portable end-to-end without
`profiles.py`, because both branches call `neprofile.run()`/`teprofile.run()` for effect
and then read four `data.physics` fields those calls write. It is, however, portable in
pieces — see "proposed signatures": everything downstream of the profile arrays is pure
arithmetic and takes those arrays as explicit arguments. The true unit #12 is
`plasma_profiles.py` + `profiles.py` = **988 LOC**, not 430.

**Registry action**: add `physics/profiles.py` as a new unit (`NeProfile.run()`,
`TeProfile.run()`, and their `set_physics_variables`/`calculate_profile_y`/`ncore`/`tcore`
callees). Not audited here — out of this record's scope, flagged so the next dispatch has
it.

## data footprint

`self.neprofile.*` / `self.teprofile.*` are object attributes, not `data` paths; they are
listed with their attribute name since they are the unit's real inputs and have no
`VarPath`.

### `parameterise_plasma` (the entry point)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.f_temp_plasma_ion_electron` | read | explicit-arg | |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | |
| `.physics.temp_plasma_ion_vol_avg_kev` | **read+write** | conditional-ownership-by-data | written **only if** `f_temp_plasma_ion_electron > 0`; otherwise the input value is used as-is. See open questions — this is a *new* classification, distinct from `conditional-ownership-by-run-config` |
| `.physics.i_plasma_pedestal` | read | switch | selects the branch; see "switches touched" |

### `parabolic_parameterisation` (`i_plasma_pedestal == 0`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.radius_plasma_pedestal_temp_norm` | read+write | input-validation-reset | one of 7 fields conditionally reset to L-mode values with a `logger.error`; **not part of the pure core**, see open questions |
| `.physics.radius_plasma_pedestal_density_norm` | read+write | input-validation-reset | ″ |
| `.physics.temp_plasma_pedestal_kev` | read+write | input-validation-reset | ″ |
| `.physics.temp_plasma_separatrix_kev` | read+write | input-validation-reset | ″ |
| `.physics.nd_plasma_pedestal_electron` | read+write | input-validation-reset | ″ |
| `.physics.nd_plasma_separatrix_electron` | read+write | input-validation-reset | ″ |
| `.physics.tbeta` | read+write | input-validation-reset | ″ |
| `.physics.alphan` | read | explicit-arg | |
| `.physics.alphat` | read | explicit-arg | |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | |
| `.physics.nd_plasma_ions_total_vol_avg` | read | explicit-arg | |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | |
| `.physics.temp_plasma_ion_vol_avg_kev` | read | explicit-arg | set by `parameterise_plasma` above |
| `.physics.f_temp_plasma_electron_density_vol_avg` | write | local-intermediate | written L126, read back L156/L160 in the same straight line |
| `.physics.nd_plasma_electron_line` | write | explicit-arg | |
| `.physics.temp_plasma_electron_line_avg_kev` | write | explicit-arg | |
| `.physics.temp_plasma_electron_density_weighted_kev` | write | explicit-arg | |
| `.physics.temp_plasma_ion_density_weighted_kev` | write | explicit-arg | |
| `.physics.temp_plasma_electron_on_axis_kev` | write | **redundant-duplicate-write** | see below |
| `.physics.temp_plasma_ion_on_axis_kev` | write | **redundant-duplicate-write** | ″ |
| `.physics.nd_plasma_electron_on_axis` | write | **redundant-duplicate-write** | ″ |
| `.physics.nd_plasma_ions_on_axis` | write | **redundant-duplicate-write** | ″ |

**The four `redundant-duplicate-write` rows, verified rather than assumed.** All four are
already written by `teprofile.run()`/`neprofile.run()` forty lines earlier in the *same
method* (`profiles.py:531-558` `TeProfile.set_physics_variables`, `profiles.py:332-358`
`NeProfile.set_physics_variables`), and in the parabolic branch the two formulas are
algebraically identical:

| field | written by profile object (parabolic branch) | rewritten here |
|---|---|---|
| `temp_plasma_electron_on_axis_kev` | `te_vol * (1 + alphat)` | `te_vol * (1 + alphat)` — *identical* |
| `temp_plasma_ion_on_axis_kev` | `ti_vol / te_vol * te_on_axis` | `ti_vol * (1 + alphat)` — equal after substitution |
| `nd_plasma_electron_on_axis` | `nd_vol * (1 + alphan)` | `nd_vol * (1 + alphan)` — *identical* |
| `nd_plasma_ions_on_axis` | `nd_ion_vol / nd_vol * nd_e_on_axis` | `nd_ion_vol * (1 + alphan)` — equal after substitution |

Substituting: `ti_vol/te_vol * (te_vol*(1+αt)) ≡ ti_vol*(1+αt)`, and likewise for
density. **Algebraically identical, but not bitwise** — the profile object's form does a
divide-then-multiply the rewrite does not. Well inside tier-1's `rtol=1e-12`, but it means
"the two writes agree" is a statement about real arithmetic, not about floats, and a port
that keeps only one of them will differ from PROCESS in the last bits depending on which.
**Recommendation: keep the profile object's write** (it is the one that also covers the
pedestal branch) and drop the rewrite, per the schema's rule for this classification.

**This also resolves the obvious cross-branch worry.** `calculate_profile_factors` reads
all four on-axis fields, and `pedestal_parameterisation` writes none of them — so the
question was whether the pedestal path reads stale values. It does not: `NeProfile`/
`TeProfile.set_physics_variables` write all four in *both* branches (with a different
formula per branch, `ncore`/`tcore` for the pedestal case). No staleness. This is the one
finding here that most needed checking and it came back clean.

### `pedestal_parameterisation` (`i_plasma_pedestal != 0`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `neprofile.profile_x` / `.profile_y` / `.profile_dx` / `.profile_integ` | read | explicit-arg | object attributes, no `VarPath`; arrays of `n_plasma_profile_elements` |
| `teprofile.profile_y` / `.profile_integ` | read | explicit-arg | ″ |
| `.physics.temp_plasma_ion_vol_avg_kev` | read | explicit-arg | |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | |
| `.physics.nd_plasma_separatrix_electron` | read | explicit-arg | |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | |
| `.physics.temp_plasma_electron_density_weighted_kev` | write | local-intermediate | written L217, read back L221/L227 |
| `.physics.temp_plasma_ion_density_weighted_kev` | write | explicit-arg | |
| `.physics.f_temp_plasma_electron_density_vol_avg` | write | explicit-arg | |
| `.physics.nd_plasma_electron_line` | write | explicit-arg | straight copy of `neprofile.profile_integ` |
| `.physics.temp_plasma_electron_line_avg_kev` | write | explicit-arg | straight copy of `teprofile.profile_integ` |
| **`.divertor.prn1`** | write | explicit-arg | **cross-area write** — the only one in this file; `max(0.01, nd_sep/nd_vol)`, guarding a later division |

### `calculate_profile_factors` (both branches)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `neprofile.profile_y`, `teprofile.profile_y` | read | explicit-arg | arrays |
| `.physics.nd_plasma_electron_on_axis` | read | explicit-arg | from the profile objects, both branches |
| `.physics.temp_plasma_electron_on_axis_kev` | read | explicit-arg | ″ |
| `.physics.nd_plasma_ions_on_axis` | read | explicit-arg | ″ |
| `.physics.temp_plasma_ion_on_axis_kev` | read | explicit-arg | ″ |
| `.physics.nd_plasma_ions_total_vol_avg` | read | explicit-arg | |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | |
| `.physics.nd_plasma_fuel_ions_vol_avg` | read | explicit-arg | |
| `.physics.f_temp_plasma_ion_electron` | read | explicit-arg | |
| `.physics.temp_plasma_electron_density_weighted_kev` | read | explicit-arg | written by whichever branch ran |
| `.physics.temp_plasma_ion_density_weighted_kev` | read | explicit-arg | ″ |
| `.physics.alphan`, `.physics.alphat` | read | explicit-arg | |
| `.physics.alphaj` | read | explicit-arg | |
| `.physics.plasma_current` | read | explicit-arg | |
| `.physics.a_plasma_poloidal` | read | explicit-arg | |
| `.physics.pres_plasma_thermal_on_axis` | write | explicit-arg | |
| `.physics.pres_plasma_electron_profile` | write | explicit-arg | **array-valued** |
| `.physics.pres_plasma_ion_total_profile` | write | explicit-arg | **array-valued**, local-intermediate for the next row |
| `.physics.pres_plasma_thermal_total_profile` | write | explicit-arg | **array-valued**, sum of the two above |
| `.physics.pres_plasma_fuel_profile` | write | explicit-arg | **array-valued** |
| `.physics.alphap` | write | explicit-arg | `alphan + alphat` |
| `.physics.pres_plasma_thermal_vol_avg` | write | explicit-arg | |
| `.physics.j_plasma_on_axis` | write | explicit-arg | uses `sp.special.beta` |

### `calculate_parabolic_profile_factors` (parabolic branch only)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_pedestal` | read | switch | **re-checked inside**, though only ever called from the parabolic branch — dead guard, see open questions |
| `.physics.alphat`, `.physics.alphan` | read | explicit-arg | three-way branch each: `>1`, `(0,1]`, `<=0` → raises |
| `.physics.temp_plasma_electron_on_axis_kev` | read | explicit-arg | |
| `.physics.nd_plasma_electron_on_axis` | read | explicit-arg | |
| `.physics.rminor` | read | explicit-arg | |
| `.physics.gradient_length_te` | write | explicit-arg | |
| `.physics.gradient_length_ne` | write | explicit-arg | |

## proposed signature(s)

Five pure units. Everything except the two profile-object `.run()` calls and the
input-validation reset is straight-line arithmetic over scalars and profile arrays.

```python
def calculate_ion_vol_avg_temperature(
    f_temp_plasma_ion_electron,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """`parameterise_plasma`'s conditional write, as a total function.

    Takes the incumbent `temp_plasma_ion_vol_avg_kev` as an argument precisely because
    PROCESS leaves it alone when `f_temp_plasma_ion_electron <= 0` -- a `jnp.where` needs
    both arms, so the field is both a read and a write. -> temp_plasma_ion_vol_avg_kev
    """


def calculate_parabolic_profile_values(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """`parabolic_parameterisation`'s arithmetic tail (L126-181), minus the four
    redundant-duplicate on-axis writes.
    -> (f_temp_plasma_electron_density_vol_avg, nd_plasma_electron_line,
        temp_plasma_electron_line_avg_kev, temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev)
    """


def calculate_pedestal_profile_values(
    rho,
    dens,
    temp,
    profile_dx,
    nd_profile_integ,
    temp_profile_integ,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
):
    """`pedestal_parameterisation` (L205-247). `rho`/`dens`/`temp` are the profile
    arrays; Simpson's rule is reimplemented in `jnp` (see JAX-difficulty flags).
    -> (temp_plasma_electron_density_weighted_kev, temp_plasma_ion_density_weighted_kev,
        f_temp_plasma_electron_density_vol_avg, nd_plasma_electron_line,
        temp_plasma_electron_line_avg_kev, prn1)
    """


def calculate_profile_factors(
    ne_profile_y,
    te_profile_y,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_ions_on_axis,
    temp_plasma_ion_on_axis_kev,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_temp_plasma_ion_electron,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    alphan,
    alphat,
    alphaj,
    plasma_current,
    a_plasma_poloidal,
):
    """L259-324. Four of the eight returns are arrays of `n_plasma_profile_elements`.
    -> (pres_plasma_thermal_on_axis, pres_plasma_electron_profile,
        pres_plasma_ion_total_profile, pres_plasma_thermal_total_profile,
        pres_plasma_fuel_profile, alphap, pres_plasma_thermal_vol_avg, j_plasma_on_axis)
    """


def calculate_parabolic_gradient_lengths(
    alphat,
    alphan,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_electron_on_axis,
    rminor,
):
    """L342-430, with the dead `i_plasma_pedestal` guard dropped.
    Raises on negative alphat/alphan in PROCESS; the port returns non-finite instead
    (declare `reference_domain_errors = (ProcessValueError,)` on the contract).
    -> (gradient_length_te, gradient_length_ne)
    """
```

## cottax node

**`calculate_parabolic_profile_values` and `calculate_ion_vol_avg_temperature` now have
nodes, and both are registered** — `ParabolicProfileValues` (under the
`.physics.i_plasma_pedestal == 0` arm) and `IonVolAvgTemperature` (in `COMMON`, since
PROCESS writes that field in `parameterise_plasma` *before* the branch). This reverses
the deferral recorded here, and the reason is worth keeping:

> The deferral rested on open question 1 below ("`i_plasma_pedestal` holds two different
> switch roles"). That question was **already settled in practice** by the time it was
> read again: `total_process.py`'s value-0 arm co-locates
> `EcrhDensityLimit(i_plasma_pedestal=0)` with the parabolic profile nodes, so there is
> exactly one place the value is written and nothing left to reconcile. The record was
> stale, not wrong.

Leaving them unwritten had a cost nobody had measured. `.physics.temp_plasma_ion_vol_avg_kev`,
`.physics.temp_plasma_electron_density_weighted_kev` and
`.physics.temp_plasma_ion_density_weighted_kev` were **boundary inputs**, so ion
temperature was structurally disconnected from `.physics.temp_plasma_electron_vol_avg_kev`
— which is iteration variable 4. Every *value* in the graph was still right; every
*derivative* with respect to that variable was ~0 where PROCESS's is O(1), and
`FusionRates`' `sigmav_dt_average` (which goes roughly as `T²`) had a relative sensitivity
of `2.4e-16`. Found by `_audit/optimise_design.md` §10.5a's Jacobian comparison, which is
the only check in this project that could have found it. Registering the two nodes fixed
every affected cell and added +7 MDA-harness agreements with 0 new disagreements.

**`IonVolAvgTemperature` is a `FixedPointFunction`, not an `ExplicitFunction`**, and that
is the interesting part. `calculate_ion_vol_avg_temperature` takes the incumbent value as
an argument (faithfully — PROCESS writes the field only when
`f_temp_plasma_ion_electron > 0`), so the field is a read *and* a write of the node and
`to_graph` refuses it. The fixed-point shape carries the conditional-ownership-by-data
classification in both directions: with the ratio positive the residual has derivative
`-1` and the problem is well-posed; with it non-positive `g` is the exact identity, the
residual is structurally zero, and `functional_process.sand.degenerate_fixed_points`
detects that by differentiation and drops the problem — reverting the field to a boundary
input, which *is* PROCESS's "use the input value" semantics, recovered from structure.

**`calculate_pedestal_profile_values` still has no node.** Under `i_plasma_pedestal == 1`
these five fields therefore have no producer at all, which is the honest state of the port
(that arm needs `profiles.py`'s profile arrays) rather than a silent fallback to the
parabolic formula.

`calculate_profile_factors` and `calculate_parabolic_gradient_lengths` have no such
problem — `calculate_profile_factors` runs in both branches, and
`calculate_parabolic_gradient_lengths` is parabolic-only with no counterpart arm.

**The seven-field L-mode reset now has a node too: `LModeProfileReset`**, registered under
the same `.physics.i_plasma_pedestal == 0` arm. This closes open question 2 below, and the
answer is the opposite of what that question proposed: the reset is **not**
graph-assembly-time input coercion for `configuration.py` to perform, it is an ordinary
node, because PROCESS applies it inside the pipeline and only on the parabolic arm.

Two things make it a node rather than a special case:

- **Its post-condition is unconditional.** PROCESS's guard is *"if any of the seven differs
  from its L-mode value, set all seven to their L-mode values"*, so when the guard is false
  every field already holds what the body would assign. The seven fields hold
  `(1, 1, 0, 0, 0, 0, 2)` on exit either way; the guard decides only whether `logger.error`
  fires, which is a diagnostic and is not ported. `TestLModeProfileReset` fuzzes all seven
  arguments against PROCESS to turn that from a reading into a measurement, with one legacy
  sample on each side of the guard.
- **It declares no `Input`**, for the same reason: the result does not depend on the
  incoming values, so declaring the seven fields as reads as well as writes would state a
  seven-way self-loop the computation does not have. `lmode_profile_reset` keeps the seven
  arguments (defaulted to the L-mode values) purely so the independence is testable.

**What leaving it unwritten cost, measured on the reference stellarator run.** Four of the
seven fields are read by the graph — `radius_plasma_pedestal_density_norm`,
`nd_plasma_pedestal_electron` and `nd_plasma_separatrix_electron` by
`profiles.DensityProfile`, `tbeta` by `SynchrotronRadiationPower` — and all four were
unowned boundary inputs. A **warm** solve seeded from PROCESS's converged `DataStructure`
therefore got the reset for free (those fields are already `0` there) while a **cold** one
seeded from the input file carried `nd_plasma_pedestal_electron = 4e19` and
`nd_plasma_separatrix_electron = 3e19` into `DensityProfile`, whose single formula is the
pedestal one and only degenerates to the parabolic profile once they are zero. The two
were solving *different problems*: SAND finished at `objf` 1.217757 warm and 1.215038 cold,
a 0.22 % gap that `_audit/next_steps.md` §11.10 recorded as *"either several local minima
or a direction flat enough that two starts stop in different places"*. With this node the
cold solve lands on the warm one's answer to nine digits (both `1.217757336`, every
iteration variable agreeing to 6–7 digits) and the median distance to PROCESS's converged
`x` falls from `1.4e-02` to `8.6e-03`. Same defect class as `_audit/optimise_design.md`
§10.5a/§10.5c — a missing producer every value test passes — found by a third method again:
neither a gradient nor a consumer, but two starting points disagreeing.

## tier signal

**Tier 1** for all five signatures: no internal iteration, no `scipy.optimize`, no call
into another model. `sp.integrate.simpson` is a fixed quadrature rule, not a solver.

`PlasmaProfile.run()` *as a whole* is **tier 3** — it composes the five above with two
external `Model.run()` calls (`neprofile`, `teprofile`), which is what the scope
correction above is about.

## switches touched

- **`.physics.i_plasma_pedestal`** (values 0 = `PARABOLIC_PROFILE`, non-0 =
  `PEDESTAL_PROFILE`, via `PlasmaProfileShapeType`) — **topology-changing here.** The
  parabolic arm runs three methods (`parabolic_parameterisation`,
  `calculate_profile_factors`, `calculate_parabolic_profile_factors`); the pedestal arm
  runs two (`pedestal_parameterisation`, `calculate_profile_factors`). Different node
  sets, and the arms' reads-sets differ substantially (the pedestal arm needs the profile
  arrays and `nd_plasma_separatrix_electron`; the parabolic arm needs neither).
  **Split.** Also branched on inside `profiles.py`'s `NeProfile`/`TeProfile`
  `set_physics_variables` and `calculate_profile_y` — so the switch spans both files of
  the corrected unit.

  **This is the same switch `total_process.py` currently resolves as a static kwarg**,
  `EcrhDensityLimit(i_plasma_pedestal=0)`. That call is correct *in `density_limits.py`*,
  where `i_plasma_pedestal != 0` has no formula at all. But one switch now holds **both**
  roles across two units — topology-changing here, static-kwarg there — which neither
  `naming_convention.md` nor `configuration.py` anticipated. See open questions.

  Not yet added to `core/solver/switches.md`: it was not in the original 10, and it
  deserves a full site list (it is read in at least `plasma_profiles.py`, `profiles.py`,
  `density_limits.py`) rather than a stub. **Owed.**

## calls into other models

- `self.teprofile.run()` — `TeProfile` (`physics/profiles.py`), lines 120, 196.
- `self.neprofile.run()` — `NeProfile` (`physics/profiles.py`), lines 121, 197.

Both called **for effect**: the return value is unused, and the caller then reads four
`data.physics` fields and six object attributes the call wrote. This is the
`implicit-io-via-callee` pattern at the file level rather than within one function.

## JAX-difficulty flags

- `sp.special.gamma` (L139-141, 147-149) — **minor**. `jax.scipy.special.gamma` exists and
  differentiates; verified in `process_port`.
- `sp.special.beta` (L321) — **workaround-known**. No `jax.scipy.special.beta`, but
  `exp(gammaln(a) + gammaln(b) - gammaln(a+b))` reproduces it to ~2e-15 relative (verified;
  well inside tier-1's `rtol=1e-12`) and differentiates.
- `sp.integrate.simpson` (L213-214, and `profiles.py:110`) — **workaround-known**.
  Not in JAX, but Simpson's rule over a uniform grid is a few lines of `jnp` and is
  exactly differentiable. **Care needed**: `scipy`'s `simpson` has a specific
  even-interval-count fallback rule, and `n_plasma_profile_elements` defaults to 201
  (200 intervals, even), so the plain composite rule applies — but the port must match
  `scipy`'s choice, not assume it. This is the one place a naive reimplementation would
  silently disagree.
- `logger.error` + the 7-field reset (L101-117) — **blocker for that block only**, and it
  should not be ported: it is input validation, not physics (see open questions).
- `ProcessValueError` on negative `alphat`/`alphan` (L380, L418) — **minor**, standard
  `reference_domain_errors` handling.
- `profile_y[rho_index] = ...` in `profiles.py:203-209` — **workaround-known**, needs
  `.at[].set()`. Flagged for the `profiles.py` unit, not this one.

## open questions

1. **[RESOLVED IN PRACTICE — see "cottax node" above.]** **`i_plasma_pedestal` holds two
   different switch roles across two units.** Topology-
   changing in `plasma_profiles.py`/`profiles.py`; a static kwarg on one node in
   `density_limits.py`. `configuration.py`'s `TOPOLOGY_SWITCHES` is a flat list of
   independent choices with no way to say "when this switch's arm is selected, that
   node's static kwarg must agree". Today nothing checks it, so a graph could be assembled
   with the pedestal arm *and* `EcrhDensityLimit(i_plasma_pedestal=0)`. **Needs a
   decision before either arm gets a node** — probably: a switch declared in
   `TOPOLOGY_SWITCHES` also supplies its value to any node that takes it as a static
   kwarg, so there is one source of truth. Not implemented.
2. **[RESOLVED — see "cottax node" above. It became `LModeProfileReset`, an ordinary node
   on the `i_plasma_pedestal == 0` arm, not configuration-time coercion.]**
   **What should the input-validation reset become?** L92-117 conditionally overwrites
   seven input fields and logs an error when a parabolic run is given pedestal-shaped
   inputs. It is not physics and cannot be traced (data-dependent logging). The natural
   home is graph-assembly-time validation of the configuration, alongside the switch that
   selects the parabolic arm — but that makes `configuration.py` responsible for input
   coercion, which it currently is not. Same family as `preset_config.py`'s open question.
3. **Which of the two duplicate on-axis writes should the port keep?** Recommended above:
   the profile object's, since it covers both branches. But that write lives in
   `profiles.py`, which is unaudited and unported — so until that unit lands, a port of
   `parabolic_parameterisation` that drops the rewrite has *no* producer for those four
   fields. Sequencing constraint, not a design question.
4. **`calculate_parabolic_profile_factors` re-checks `i_plasma_pedestal` (L342-345) even
   though it is only ever called from the parabolic branch (L77).** Dead guard as far as
   this file goes. Dropping it is proposed above; worth one grep for other callers before
   acting — `grep -rn "calculate_parabolic_profile_factors" process` was not run.
5. **`.divertor.prn1` is this file's only cross-area write**, and it is written only in
   the pedestal branch. The parabolic branch leaves it at its input value (the comment at
   L240-241 says so explicitly). That is another `conditional-ownership` case, this time
   split across switch arms rather than by a data value — the pedestal node owns
   `.divertor.prn1`, the parabolic node does not. Representable, just worth stating.
6. **`n_plasma_profile_elements` is a shape, not a value.** It sets the length of every
   profile array (default 201). In a traced port it is a static shape parameter; it is
   read from `data.physics` in `profiles.py:62-64`. Needs the `Out.static`-like treatment
   or to be fixed at graph-assembly time. Deferred to the `profiles.py` unit.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

1 square root in this file has been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
