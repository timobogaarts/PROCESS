# Splitting the physics from the graph declarations

Status: **step 1 in progress**, started 2026-09-05. Agreed 2026-09-03/05.

Progress on step 1 (bodies extracted, declaration becomes a name):

- [x] `models/physics/l_h_transition.py` — 6 Martin08 arms. Bitwise gate passed on
  `large_tokamak_eval` (the reference arm, `i_l_h_threshold = 19`), `large_tokamak_nof`
  and `st_regression`.
- [x] `models/pfcoil/**` — 21 across seven files: currents (6), fields (4), geometry (4),
  inductance (2), masses (1), stresses (1), superconductor (3), including the slicing
  group (`c_pf_cs_coil_*[:CS_INDEX + 1]`, `n_pf_coil_turns[CS_INDEX]`,
  `c_pf_cs_coils_peak_ma[CS_INDEX]`). `geometry.py`'s private `_placed` instance method
  is gone, replaced by module-level functions. The `None`-binding arguments
  (`rref=None`, `r_cs_middle=None`, `PFCoilSizesNoCentralSolenoid`'s four) stay --
  declaration-level configuration.
- [x] `models/stellarator/coils/calculate.py` — 5 bodies (`CoilCoilToroidalGap`,
  `Bi2212`/`UserDefinedNb3sn`/`DurhamNbti` `WindingPackIntersectInputs`,
  `WindingPackTotalSizePost`) plus `models/stellarator/plasma_physics.py`'s
  `StellaratorBetaAndStoredEnergy` (same shape, found alongside). The five
  bare-module-function `jcrit_*` arguments (`IterNb3sn`/`OldLubellNbti`/`WstNb3sn`/
  `CrocoRebco`/`DurhamRebco`) are left as-is -- function-valued, not computed. Bitwise
  gate passed on `stellarator_helias` and `large_tokamak_nof`.
- [x] `models/tfcoil/**` — 3 bodies (`croco.py`'s `HazeltonZhaiRebcoCrocoTemperatureMargin`,
  `superconducting.py`'s `_TemperatureMarginWithStrain` and
  `OldLubellNbtiTfSuperconductorTemperatureMargin`) plus 3 computed-argument sites
  (`quench.py`'s `jnp.asarray` wrap, `superconducting.py`'s `type(self).coefficients`
  and `tfa[0]` slice). `superconducting.py`'s `IterNb3snCiccSuperconductorProperties`
  (`b_c20max=32.97`) is left as-is -- declaration-level configuration, not computation.
  Bitwise gate passed on `stellarator_helias` and `large_tokamak_nof`.
- [x] `models/physics/**` — 10: `bootstrap_current` (Sauter), `composition`
  (`CalculateEffectiveChargeIonisationProfiles`), `confinement_time`
  (`ConfinementScalingInputs`), `fusion_reactions` (`FusionRates`), `plasma_inductance`
  (2), `radiation_power` (`ImpurityRadiationTotals`), `scrape_off_layer` (3).
- [x] `models/power/**` — `thermal_cryo.py`'s 3 `BinOp`s plus
  `electric_production.py`'s `ResistiveCentrepostLiquidBreeder`, whose argument list
  composed two other function calls. Its three sibling arms have the same shape and are
  **not** done.
- [x] the rest: `cs_fatigue.py`, `structure.py`, `vacuum/vacuum.py` (2 --
  `VacuumPumpingSimple`, `DuctFeasibilityConditions`).

**Step 1 gate, run once over the merged result rather than per commit** (2026-09-05):
`tests/functional_process tests/unit` 7713 passed / 8162 skipped, and the **full
seven-configuration cold matrix is bitwise identical to `reference_cold_matrix.txt` on
all twelve rows**. Census over the whole package: **380 pure delegation** (from 343),
35 pure-plus-fields, and the non-thin categories down from 57 to **10**.

**What is deliberately left**, and it is not oversight:

- 22 `thin-but-computed-args`, nearly all binding a bare constant, `None` or an enum
  member. That is the null-arm-of-a-switch pattern and it becomes a field with a default
  under the `fn = <function>` interface below, not an extraction.
- 7 multi-statement (2), 1 multi-statement (3), 1 tuple, 1 `expr:BinOp` — the residue,
  including `currents.py`'s `CSCurrentDensityPulseStart` (a bare product) and
  `scrape_off_layer.py`'s `Mast2014SOLPowerDecayLength2`, whose own docstring argues the
  unit conversion belongs at the call site. Each needs a judgement rather than the
  mechanical treatment.
- `models/physics/composition.py:808` `PlasmaCompositionNonIgnited` builds
  `functools.partial(plasma_composition_non_ignited, f_nd_beam_electron=...)` where
  `f_nd_beam_electron` is a **live port**. Investigated rather than changed: the partial
  is built inside `__call__` from that call's own traced argument and consumed in one
  Python frame, so it never becomes a persistently-compared field the way the
  identity-keyed-closure defect at `a4468d65` did — it is not a stale-constant bake-in.
  It is still closure-shaped and wants a deliberate decision when the declaration
  interface is built.

## The target layout

`functional_process/` becomes shaped like `process/` — all the physics functions, mirroring
the original's structure — with everything cottax-dependent confined to one subtree:

```
functional_process/
    <the pure functions, mirroring process/>
    cottax/      the graph declarations, mirroring the functions by stem
    tests/       moved in from tests/functional_process/
    _audit/
```

**Why this shape and not a `formulas/` subtree**: `models/` currently holds *declarations*,
which collides with what "models" means in PROCESS itself. Putting the functions at the top
level and naming the cottax subtree for what it contains removes the collision.

**The win is one enforceable sentence**: *nothing outside `functional_process/cottax/`
imports cottax.* One test, and the physics half is provably standalone — usable as
PROCESS's physics in pure JAX with no graph machinery. `~/openmdao_process` already imports
those bodies and would be the first consumer, so the boundary gets exercised rather than
merely asserted.

## The census — 77 % is already pure delegation, not 87 %

AST walk over every declaration with `From()`-style ports. The script now lives at
`_audit/declaration_census.py` (it used to be quoted as living in a scratchpad, which does
not survive a session); re-run it rather than trusting this table after any change.

**The first census's `thin` was a proxy and it was too generous.** It accepted any single
`return f(...)`, without looking at the argument list — so `return f(1e-20 * n, ...)` counted
as a thin delegator although the arithmetic is still *in* the declaration, and the deferred
`fn = <function>` interface below could not express it. Tightened to "one return, of one
call, every argument a bare parameter or one of the declaration's own fields":

| shape of `__call__` | n |
|---|---|
| **pure delegation** (every argument a bare port) | **343** |
| **pure delegation + own fields** (`self.i_pf_conductor`) | **16** |
| thin body, **computed arguments** | **31** |
| multi-statement (2) | 35 |
| multi-statement (3 / 4 / 6) | 10 |
| single `BinOp` expression | 6 |
| tuple, bare-name | 6 |
| **total** | **447** |

So the work is **57 bodies to extract** (the number this file has always carried, and it is
right) **plus 31 argument lists to look at** — the second group is new and was hidden by the
proxy. It is not 31 more extractions: reading them, most are *declaration-level
configuration* rather than computation, and they split three ways.

- **~20 bind a constant, an enum member or `None`** — `p_hcd_injected_total_mw=0.0`,
  `i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY`, `r_cs_middle=None`. These are the
  null-arm-of-a-switch pattern; under `fn = <function>` they become fields with defaults,
  which is where they belong anyway.
- **~8 slice an array port** — `c_pf_cs_coil_pulse_start_ma[:CS_INDEX + 1]`,
  `n_pf_coil_turns[CS_INDEX]`, `tfa[0]`. Real computation, and all in `models/pfcoil/**`
  and `models/tfcoil/**`. These need extraction like the 57.
- **three are worth a second look on their own account**, not because of this split:
  `PlasmaCompositionNonIgnited` (`models/physics/composition.py:808`) passes
  `functools.partial(plasma_composition_non_ignited, f_nd_beam_electron=...)` — a closure
  over a **port value**, which is the pattern the array ban removed everywhere else;
  `TfCoilQuenchHeatCurrentDensity` wraps two of its own fields in `jnp.asarray` at the call;
  `PlantElectricProductionResistiveCentrepostLiquidBreeder` calls two other functions inside
  its argument list.

Clusters for step 1 are unchanged: `models/pfcoil/**` (currents, fields, geometry,
inductance, masses, stresses, superconductor — and it owns most of the slicing group too)
and `models/physics/l_h_transition.py` (six near-identical Martin08 variants).

## The plan

1. **Extract the 57.** Inline physics moves to module-level functions, one commit per file
   or cluster. Pure refactor, so **bitwise identity is the gate** and the cold matrix is the
   instrument that already proves it.
2. **Move the files.** Functions to the top level mirroring `process/`, declarations to
   `functional_process/cottax/` by matching stem. Mechanical once step 1 is done.
3. **Move `tests/functional_process/` to `functional_process/tests/`.**
4. **Update `unit_registry.md`** — it names paths explicitly and is enforced by
   `tests/functional_process/test_registry_coverage.py`, so all 88 rows move. The test tells
   you immediately if one is missed.
5. **Enforce the boundary**: a test asserting nothing outside `functional_process/cottax/`
   imports cottax. Without it the first person in a hurry imports `From` into a physics file
   and the separation is over.

**What the audit records and tests mirror**: the *functions*, not the declarations — the
records are about physics fidelity against PROCESS, not about graph wiring.

## Later, not now

Cache PROCESS's reference answers so `functional_process` runs with no base PROCESS
installed. Noted 2026-09-05; deliberately out of scope for this split.

## The declaration interface — considered, deliberately deferred

The 87 % thin-delegator result invites making the declaration *name* its implementation
rather than contain it:

```python
class GreenwaldDensityLimit(ExplicitFunction):
    fn = calculate_greenwald_density_limit
    nd_plasma_electron_greenwald_max = OutputInto(physics)
    plasma_current = From(physics)
    rminor = From(physics, parameter="minor_radius")  # only where names differ
```

called by **keyword**, with `__check_init__` comparing declared field names against
`inspect.signature(fn)` and refusing a mismatch — so the opacity objection dissolves,
because there is no order to get wrong and the error names the offending parameter.

**Not now**: the steps above deliver the reusable-physics product with no upstream change,
and this needs an `interfaces/pytree_namespace_module.py` change plus 447 declarations
rewritten. It is also close to cottax's own `AbstractImplementation` direction, so a second
spelling now risks having to unify them later. Steps 1–2 make it *smaller* when it happens.

**Rejected, so they are not re-proposed:**

- **Positional `fn(*args)`** — opaque; parameter names invisible, order silent. No reason to
  accept that when a construction-time check costs one method.
- **Signature-as-declaration with the body removed** — leaves two signatures to keep in
  sync; the opacity relocated, not removed.
- **Deriving ports from the signature** — inverts the dependency, so renaming a parameter in
  a physics function would silently change the graph.

**Implementation caution if the alias is ever built.** OpenMDAO has exactly this mechanism
(`primal_name`) and its implementation carries a silent-failure bug: `get_function_deps`
returns `wrt` in primal names then filters against OpenMDAO names, so nothing matches and
the Jacobian comes back **all zero with no warning** — it surfaced as 15 totals of exactly
0.0 while values stayed bitwise correct. Runnable case in `~/openmdao_process`. Test the
alias; do not merely support it.
