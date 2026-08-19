---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/physics/profiles.py`, whole file (558 lines): `PlasmaProfileShapeType`
(`IntEnum`, L22-38), `Profile` (abstract base, L41-112), `DensityProfilePedestalType`
(`IntEnum`, L115-137), `NeProfile` (L140-358), `TeProfile` (L361-558). Registry unit #21,
added by `plasma_profiles.md`'s scope correction with stated scope `NeProfile.run()`,
`TeProfile.run()`, `Profile.*`, `set_physics_variables`, `calculate_profile_y`,
`ncore`/`tcore` — in practice the whole file, since `run()` on either subclass reaches
every method except one.

**One method sits outside that stated scope and is ported anyway:
`NeProfile.set_pedestal_and_separatrix_values` (L286-331).** Neither `NeProfile.run()`
nor `TeProfile.run()` call it — confirmed by reading both bodies (L145-159, L366-381) —
and its only call site anywhere in `process/` is `physics.py:365-368`:

```python
if (
    PlasmaProfileShapeType(self.data.physics.i_plasma_pedestal)
    == PlasmaProfileShapeType.PEDESTAL_PROFILE
):
    self.plasma_profile.neprofile.set_pedestal_and_separatrix_values()
```

inside `PhysicsCalculations.physics()` — **unit #22, tokamak**, not `CLAUDE.md`'s declared
stellarator scope. `grep -n "neprofile\|teprofile\|set_pedestal_and_separatrix_values"
process/models/stellarator/stellarator.py` returns nothing; the only stellarator call into
the profile machinery is `self.plasma_profile.run()` (`stellarator.py:1913`), which reaches
`plasma_profiles.py` (unit #12) and, through it, everything *this* record scopes as
in-scope except `set_pedestal_and_separatrix_values`. So `calculate_greenwald_density_
fractions`/`calculate_pedestal_separatrix_densities` and their two nodes are ported code
the stellarator pipeline never reaches at runtime. This mirrors unit #20's whole-file
treatment of `ImpurityRadiation` (port the pure core because splitting it is more work than
porting it), but unlike that case the extra code here is reachable *only* from a unit
explicitly outside scope. Not un-ported — the cost of leaving it out was no lower than the
cost of porting it, and unit #22 will want it verbatim — but the registry should record
that unit #21 has a stellarator-scoped 11/12 and a tokamak-only 1/12, rather than reading
as uniformly in-scope. See open question 5.

## data footprint

Grouped by source method. `NeProfile`/`TeProfile` are injected into `PlasmaProfile`
(`process/main.py:674-676`, `self.ne_profile = NeProfile()`, ditto `TeProfile`) and every
value below not on `self.data` is a `Profile` object attribute (`profile_x`, `profile_y`,
`profile_dx`, `profile_integ`) — no `VarPath`, since PROCESS never stores them.

### `Profile.run` / `.normalise_profile_x` / `.calculate_profile_dx` (L59-95, both subclasses, unconditional)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.n_plasma_profile_elements` | read | shape, not value | sizes `profile_x`/`profile_y` (default 201); same `Out.static` treatment as `plasma_profiles.md` open question 6 |

No `data` write. `profile_x`/`profile_y`/`profile_dx` are object attributes only.

### `Profile.integrate_profile_y` (L103-112)

No `data` access at all — pure arithmetic over `self.profile_y`, `self.profile_x`,
`self.profile_dx`. `profile_dx` is read but not a real dependency: `sp.integrate.simpson`
ignores `dx=` whenever `x=` is also given, the same finding unit #20 made independently at
its own two call sites into this function's sibling helper.

### `NeProfile.calculate_profile_y` (L161-211)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_pedestal` | read | switch — **dead inside this function**, see "switches touched" | selects a branch whose result is unconditionally overwritten below it |

All other parameters (`rho`, `radius_plasma_pedestal_density_norm`, `n0`, `nped`, `nsep`,
`alphan`) are plain call arguments, passed explicitly by both callers (`NeProfile.run` and
six sites in `current_drive.py`, out of scope). Writes `self.profile_y` in place; the
Python function itself returns `None` (see open question 3).

### `NeProfile.ncore` (L213-284) — pure `@staticmethod`, no `data` access.

### `NeProfile.set_pedestal_and_separatrix_values` (L286-331) — out of stated scope, see § source

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_nd_plasma_pedestal_separatrix` | read | switch | selects arm; see "switches touched" — **a second switch, not previously in `switches.md`** |
| `.physics.nd_plasma_pedestal_electron` | read (`USER_INPUT`) / write (`GREENWALD_FRACTION`) | conditional-ownership-by-run-config | same shape as `build.py`'s `dr_blkt_inboard` |
| `.physics.nd_plasma_separatrix_electron` | read/write, ″ | ″ | ″ |
| `.physics.f_nd_plasma_pedestal_greenwald` | write (`USER_INPUT`) / read (`GREENWALD_FRACTION`) | conditional-ownership-by-run-config | exact inverse of the row above |
| `.physics.f_nd_plasma_separatrix_greenwald` | write/read, ″ | ″ | ″ |
| `.physics.plasma_current` | read | explicit-arg | both arms, via `calculate_greenwald_density_limit` |
| `.physics.rminor` | read | explicit-arg | both arms, ″ |

### `NeProfile.set_physics_variables` (L333-358)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_pedestal` | read | switch | genuinely topology-changing here — different formula per arm, see "switches touched" |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | both arms (parabolic: direct; pedestal: as `nav` into `ncore`; both: divisor in the unconditional ion line) |
| `.physics.alphan` | read | explicit-arg | both arms |
| `.physics.radius_plasma_pedestal_density_norm` | read | explicit-arg | pedestal arm only |
| `.physics.nd_plasma_pedestal_electron` | read | explicit-arg | pedestal arm only |
| `.physics.nd_plasma_separatrix_electron` | read | explicit-arg | pedestal arm only |
| `.physics.nd_plasma_ions_total_vol_avg` | read | explicit-arg | both arms, ion line |
| `.physics.nd_plasma_electron_on_axis` | write | **redundant-duplicate-write**, parabolic arm only — see below | |
| `.physics.nd_plasma_ions_on_axis` | write | **redundant-duplicate-write**, both arms (the ion line is unconditional) — see below | |

### `TeProfile.calculate_profile_y` (L383-456)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_pedestal` | read | switch | genuinely selects two different formulas — unlike `NeProfile`'s twin, the parabolic branch has an explicit `return` (L433) |

Same argument shape as `NeProfile.calculate_profile_y`; writes `self.profile_y`, and the
function returns `None` on both branches (parabolic: bare `return`; pedestal: falls off
the end — see open question 3).

### `TeProfile.tcore` (L458-529) — pure `@staticmethod`, no `data` access.

### `TeProfile.set_physics_variables` (L531-558)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_pedestal` | read | switch | |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | both arms (parabolic: direct; pedestal: as `tav` into `tcore`; both: divisor in the ion line) |
| `.physics.alphat` | read | explicit-arg | both arms |
| `.physics.radius_plasma_pedestal_temp_norm` | read | explicit-arg | pedestal arm only |
| `.physics.temp_plasma_pedestal_kev` | read | explicit-arg | pedestal arm only |
| `.physics.temp_plasma_separatrix_kev` | read | explicit-arg | pedestal arm only |
| `.physics.tbeta` | read | explicit-arg | pedestal arm only |
| `.physics.temp_plasma_ion_vol_avg_kev` | read | explicit-arg | both arms, ion line |
| `.physics.temp_plasma_electron_on_axis_kev` | write | **redundant-duplicate-write**, parabolic arm only | |
| `.physics.temp_plasma_ion_on_axis_kev` | write | **redundant-duplicate-write**, both arms | |

### The `redundant-duplicate-write`s, re-verified directly against `plasma_profiles.py`

Unit #12's record claims "`profiles.py` owns the four on-axis fields unit #12 was
redundantly rewriting" (`nd_plasma_electron_on_axis`, `nd_plasma_ions_on_axis`,
`temp_plasma_electron_on_axis_kev`, `temp_plasma_ion_on_axis_kev`), written again forty
lines later in `plasma_profiles.parabolic_parameterisation`. **The claim holds** — read
directly against both sources here, not re-derived from unit #12's table:

| field | `profiles.py` (this unit, parabolic arm) | `plasma_profiles.py` rewrite |
|---|---|---|
| `nd_plasma_electron_on_axis` | `nd_vol * (1 + alphan)` (L339-342) | `nd_vol * (1 + alphan)` — *identical* |
| `nd_plasma_ions_on_axis` | `nd_ion_vol / nd_vol * nd_e_on_axis` (L354-358) | `nd_ion_vol * (1 + alphan)` — equal after substitution |
| `temp_plasma_electron_on_axis_kev` | `te_vol * (1 + alphat)` (L537-540) | `te_vol * (1 + alphat)` — *identical* |
| `temp_plasma_ion_on_axis_kev` | `ti_vol / te_vol * te_on_axis` (L554-558) | `ti_vol * (1 + alphat)` — equal after substitution |

Both are only ever written in the **parabolic** arm by `plasma_profiles.py`'s rewrite (the
pedestal branch there writes none of the four) — but `profiles.py`'s own writes are
unconditional (the ion-line write runs regardless of `i_plasma_pedestal`; the on-axis
formula differs by arm but always runs). So `profiles.py` is the sole, unconditional
producer of all four fields, and the redundancy is entirely on `plasma_profiles.py`'s side,
confined to its parabolic arm — consistent with unit #12's recommendation to keep this
unit's write and drop that rewrite. The port acts on this: `ParabolicOnAxisDensities` /
`ParabolicOnAxisTemperatures` / `PedestalOnAxisDensities` / `PedestalOnAxisTemperatures`
each declare the relevant pair as `Output`s and are what unit #12's dropped rewrite now
has as its producer.

### Minted `VarPath`s (no PROCESS storage — object attributes promoted to graph edges)

| VarPath | source attribute | producer node | justification |
|---|---|---|---|
| `.physics.radius_plasma_profile_norm` | `Profile.profile_x` | `ProfileGrid` | **reused**, not newly minted — unit #20's `radiation_power.md` already minted this exact spelling from `neprofile.profile_x` and recorded "Producer is unit #21." This lands that producer and closes the dangling edge. |
| `.physics.dradius_plasma_profile_norm` | `Profile.profile_dx` | `ProfileGrid` | new mint, sibling of the row above. No traced consumer (see JAX-difficulty flags) — declared because the source computes and stores it, pruned by `Graph.prune` in practice. |
| `.physics.nd_plasma_electron_profile` | `neprofile.profile_y` | `DensityProfile` | **reused** — already minted by `plasma_profiles.ProfileFactors` (unit #12), which reads it and had no producer. This unit is that producer; unit #12's largest dangling edge closes here. |
| `.physics.temp_plasma_electron_profile_kev` | `teprofile.profile_y` | `ParabolicTemperatureProfile` / `PedestalTemperatureProfile` (switch arms) | **reused**, same story as the row above. |
| `.physics.nd_plasma_electron_profile_integral` | `neprofile.profile_integ` | `NeProfileIntegral` | new mint. Consumed by `plasma_profiles.pedestal_parameterisation` (pedestal-only consumer; the node itself runs in both arms). |
| `.physics.temp_plasma_electron_profile_integral_kev` | `teprofile.profile_integ` | `TeProfileIntegral` | new mint, sibling of the row above. |

**All six re-verified in the MDA-harness triage (`_audit/next_steps.md` §8.1), against
`process/models/physics/profiles.py` directly rather than against each other.** Every one
is an instance attribute of a `Profile` object, not a `DataStructure` field:
`profile_x`/`profile_y` are created in `Profile.run` (`profiles.py:61,64`), `profile_dx`
in `calculate_profile_dx` (`profiles.py:93`), `profile_integ` in `integrate_profile_y`
(`profiles.py:110`). `process/data_structure/physics_variables.py` has no field of any of
these six names (grepped; it does carry many other `*_profile` arrays, and
`nd_plasma_electron_*` / `temp_plasma_electron_*` scalars — none of them these). So all six
are class (a) in that triage: genuine mints with no PROCESS counterpart, correct as-is,
and unverifiable by the harness by construction.

One relationship worth recording, because it looks like a duplicate and is not. In the
**pedestal** arm PROCESS *does* store the two `profile_integ` values, at real fields:
`plasma_profiles.py:234` writes `neprofile.profile_integ` to
`.physics.nd_plasma_electron_line` and `plasma_profiles.py:236-238` writes
`teprofile.profile_integ` to `.physics.temp_plasma_electron_line_avg_kev`. That is not a
second name for the mint — it is the pass-through this port already models
(`plasma_profiles.calculate_pedestal_profile_values` takes `ne_profile_integ`/
`te_profile_integ` and returns them under the real names). In the **parabolic** arm
(`i_plasma_pedestal == 0`, which is what the MDA harness's run uses) those same two real
fields are instead the closed-form gamma expressions
(`plasma_profiles.py:134-152`), computed without touching `profile_integ` at all, while
`neprofile.run()`/`teprofile.run()` still compute and discard it. Collapsing the mint onto
`.physics.nd_plasma_electron_line` would therefore put two producers on one field.

Three independent units (#12, #20, #21) now agree on the same spelling for the two profile
arrays and the radius grid — the naming convention's "port the existing name, don't invent"
default has nothing to invent here, but where an attribute genuinely has no PROCESS name at
all, this convergence across units audited independently is the closest thing available to
confirming the mint was the right one.

## proposed signature(s)

Thirteen pure functions (one more than nodes — `ncore`/`tcore` are helpers with no node of
their own, folded into the two on-axis functions that call them, matching unit #12's
treatment of its own helpers). The port already exists and is the authoritative copy of
every signature (`_audit/schema.md`'s rule: once ported, the `.py` file is more current
than a duplicate written back here); listed for cross-reference only.

- `calculate_profile_grid(n_plasma_profile_elements)` — `Profile.run`+`normalise_profile_x`
  +`calculate_profile_dx` (L59-95) -> `(profile_x, profile_dx)`.
- `integrate_profile_y(profile_y, profile_x)` — `Profile.integrate_profile_y` (L103-112),
  delegates to `plasma_profiles._simpson` -> `profile_integ`.
- `calculate_density_profile(rho, radius_plasma_pedestal_density_norm,
  nd_plasma_electron_on_axis, nd_plasma_pedestal_electron, nd_plasma_separatrix_electron,
  alphan)` — `NeProfile.calculate_profile_y` (L161-211), **all of it, no switch arm** ->
  `profile_y`.
- `calculate_parabolic_temperature_profile(rho, temp_plasma_electron_on_axis_kev, alphat)`
  — `TeProfile.calculate_profile_y`'s live parabolic branch (L425-433) -> `profile_y`.
- `calculate_pedestal_temperature_profile(rho, radius_plasma_pedestal_temp_norm,
  temp_plasma_electron_on_axis_kev, temp_plasma_pedestal_kev, temp_plasma_separatrix_kev,
  alphat, tbeta)` — `TeProfile.calculate_profile_y`'s pedestal branch (L435-456) ->
  `profile_y`, raises `ProcessValueError` in source, non-finite in the port.
- `ncore(radius_plasma_pedestal_density_norm, nd_plasma_pedestal_electron,
  nd_plasma_separatrix_electron, nd_plasma_electrons_vol_avg, alphan)` — `NeProfile.ncore`,
  name kept (L213-284) -> `nd_plasma_electron_on_axis`.
- `tcore(radius_plasma_pedestal_temp_norm, temp_plasma_pedestal_kev,
  temp_plasma_separatrix_kev, temp_plasma_electron_vol_avg_kev, alphat, tbeta)` —
  `TeProfile.tcore`, name kept (L458-529) -> `temp_plasma_electron_on_axis_kev`.
- `calculate_parabolic_on_axis_densities(nd_plasma_electrons_vol_avg,
  nd_plasma_ions_total_vol_avg, alphan)` — `NeProfile.set_physics_variables`'s parabolic
  arm + unconditional ion line (L339-342, L354-358) -> `(nd_plasma_electron_on_axis,
  nd_plasma_ions_on_axis)`.
- `calculate_pedestal_on_axis_densities(radius_plasma_pedestal_density_norm,
  nd_plasma_pedestal_electron, nd_plasma_separatrix_electron, nd_plasma_electrons_vol_avg,
  nd_plasma_ions_total_vol_avg, alphan)` — pedestal arm, via `ncore` (L343-358) -> same
  tuple.
- `calculate_parabolic_on_axis_temperatures(temp_plasma_electron_vol_avg_kev,
  temp_plasma_ion_vol_avg_kev, alphat)` — `TeProfile.set_physics_variables`'s parabolic arm
  + ion line (L537-540, L554-558) -> `(temp_plasma_electron_on_axis_kev,
  temp_plasma_ion_on_axis_kev)`.
- `calculate_pedestal_on_axis_temperatures(radius_plasma_pedestal_temp_norm,
  temp_plasma_pedestal_kev, temp_plasma_separatrix_kev, temp_plasma_electron_vol_avg_kev,
  temp_plasma_ion_vol_avg_kev, alphat, tbeta)` — pedestal arm, via `tcore` (L541-558) ->
  same tuple.
- `calculate_greenwald_density_fractions(nd_plasma_pedestal_electron,
  nd_plasma_separatrix_electron, plasma_current, rminor)` —
  `set_pedestal_and_separatrix_values`'s `USER_INPUT` arm (L294-313) ->
  `(f_nd_plasma_pedestal_greenwald, f_nd_plasma_separatrix_greenwald)`.
- `calculate_pedestal_separatrix_densities(f_nd_plasma_pedestal_greenwald,
  f_nd_plasma_separatrix_greenwald, plasma_current, rminor)` — `GREENWALD_FRACTION` arm
  (L314-331), PROCESS's default -> `(nd_plasma_pedestal_electron,
  nd_plasma_separatrix_electron)`.

`_greenwald_limit(plasma_current, rminor)`, a private helper, inlines
`PlasmaDensityLimit.calculate_greenwald_density_limit` — see "calls into other models".

## cottax node

Twelve, all `ExplicitFunction`s, in `profiles.py`. **None are registered in
`total_process.py` yet** — this record assesses readiness, it does not change
registration.

**Ready for `total_process.COMMON`, switch-independent:**
- `ProfileGrid` — no `Input`s at all (its only argument is the static
  `n_plasma_profile_elements`); mints both outputs.
- `NeProfileIntegral`, `TeProfileIntegral` — run in both `i_plasma_pedestal` arms.
- `DensityProfile` — genuinely switch-independent (see "switches touched"), not an
  `Alternative`. The caveat this entry used to carry (its parabolic-configuration
  correctness resting on an input coercion nothing performed) is **closed**:
  `plasma_profiles.LModeProfileReset` is that coercion, registered on the
  `i_plasma_pedestal == 0` arm, and it owns the three pedestal fields this node reads.

**Blocked on `i_plasma_pedestal`'s two-role reconciliation (unit #12 open question 1,
`configuration.py`'s missing "switch also supplies its static-kwarg value" mechanism):**
`Alternative` pairs sharing an output, one per switch value —
`ParabolicTemperatureProfile`/`PedestalTemperatureProfile`,
`ParabolicOnAxisDensities`/`PedestalOnAxisDensities`,
`ParabolicOnAxisTemperatures`/`PedestalOnAxisTemperatures`.

**Blocked on a second, nested switch with no `configuration.py` mechanism yet
(`i_nd_plasma_pedestal_separatrix`, itself gated by `i_plasma_pedestal ==
PEDESTAL_PROFILE`, and reached only from tokamak-only `physics.py`) — see open questions 2
and 5:** `GreenwaldDensityFractions`/`PedestalSeparatrixDensities`.

`ncore`/`tcore` get no node — helpers with no PROCESS storage of their own, called from
inside `PedestalOnAxisDensities`/`PedestalOnAxisTemperatures`'s `fn`, exactly as unit #12
folds its own helpers.

## tier signal

**Tier 1**, all thirteen signatures. No `scipy.optimize`, no internal iteration, no call
into another `Model`'s `run()`. `set_pedestal_and_separatrix_values`'s call into
`physics/density_limit.py` is a `@staticmethod` one-liner with no `self`/`data` access —
inlined rather than iterated, so it stays tier 1 too (see "calls into other models").

## switches touched

- **`.physics.i_plasma_pedestal`** (`PlasmaProfileShapeType`, 0 = `PARABOLIC_PROFILE`,
  non-0 = `PEDESTAL_PROFILE`) — **mixed role within this single file, more finely split
  than unit #12's record states.** Unit #12 (`plasma_profiles.md` § switches touched)
  describes it as topology-changing "here" (`profiles.py`) in one blanket sentence: "the
  parabolic arm runs three methods... the pedestal arm runs two... different node sets."
  That is **true for `TeProfile`** (`calculate_profile_y` and `set_physics_variables` both
  have two live, differently-shaped formulas) but **not true for `NeProfile.
  calculate_profile_y`**, whose parabolic branch never survives to be returned — see
  `DensityProfile`'s entry above and the dead-code finding under "proposed signatures".
  The corrected, node-level split (the answer this record was asked to settle):

  | node | arm |
  |---|---|
  | `ProfileGrid` | common — switch not read |
  | `NeProfileIntegral` | common — switch not read |
  | `TeProfileIntegral` | common — switch not read |
  | `DensityProfile` | common — switch read but dead; one formula covers both configurations |
  | `ParabolicTemperatureProfile` | parabolic (`i_plasma_pedestal == 0`) only |
  | `ParabolicOnAxisDensities` | parabolic only |
  | `ParabolicOnAxisTemperatures` | parabolic only |
  | `PedestalTemperatureProfile` | pedestal (`i_plasma_pedestal != 0`) only |
  | `PedestalOnAxisDensities` | pedestal only |
  | `PedestalOnAxisTemperatures` | pedestal only |
  | `GreenwaldDensityFractions` | neither arm of *this* switch — see `i_nd_plasma_pedestal_separatrix` below |
  | `PedestalSeparatrixDensities` | ″ |

  4 common, 3 parabolic-only, 3 pedestal-only, 2 gated by the different, nested switch.
  Still **split** by the schema's default — the reads-sets genuinely differ for 6 of the
  12 — but "arms run different method sets" is not uniform across the file, and a
  configuration mechanism keyed only on "which arm does this node belong to" needs a
  `common` bucket in addition to the two `Alternative` arms, or `DensityProfile` has
  nowhere to go. Also branched on in `plasma_profiles.py`, `density_limits.py` (as a
  static kwarg) — see `plasma_profiles.md` for that cross-file entanglement, not
  duplicated here.

- **`.physics.i_nd_plasma_pedestal_separatrix`** (`DensityProfilePedestalType`, 0 =
  `USER_INPUT`, 1 = `GREENWALD_FRACTION`, PROCESS's default) — **new, not previously in
  `switches.md` or any audit record.** Read only in `set_pedestal_and_separatrix_values`
  (L290-291). **Split** — the two arms are exact inverses of each other (fractions-from-
  densities vs. densities-from-fractions), not a shared-reads-set formula choice, so they
  cannot be a static kwarg on one node either. **Nested**: the method that reads it is
  itself called only under `i_plasma_pedestal == PEDESTAL_PROFILE` (`physics.py:365-368`),
  so this switch's very existence as a live choice is conditional on the first switch's
  value — the same "nested switches" shape `next_steps.md` § 1 already flags for
  `irefprop` under `i_blkt_coolant_type == WATER`, and `configuration.TOPOLOGY_SWITCHES`'s
  flat-tuple shape has no way to express either instance yet. This is the second occurrence
  of that gap, not a new kind of problem. **Owed a `switches.md` entry.**

## calls into other models

- `PlasmaDensityLimit.calculate_greenwald_density_limit(c_plasma, rminor)` —
  `process/models/physics/density_limit.py:365-395`, called twice from
  `set_pedestal_and_separatrix_values` (L298-303, L308-313, and the `GREENWALD_FRACTION`
  arm's L320-324, L326-331). A pure `@staticmethod` one-liner
  (`1.0e14 * c_plasma / (pi * rminor**2)`) with no `self`/`data` access — confirmed by
  reading it directly. The port inlines it as `_greenwald_limit` rather than importing
  `physics/density_limit.py` (not itself an audited unit — only
  `models/stellarator/density_limits.py` is), which keeps `calculate_greenwald_density_
  fractions`/`calculate_pedestal_separatrix_densities` tier 1 and their signatures equal to
  the real reads-set. **This duplicates the formula.** Flagged, not fixed: when
  `physics/density_limit.py` is eventually audited, this should become a graph edge from
  that unit's own Greenwald node rather than staying a second copy — same shape as unit
  #20's flag about `impurity_radiation.py`'s eventual audit.
- Nothing else. `ncore`/`tcore` are local staticmethods; every other read is a plain
  `data`/object-attribute access.

## JAX-difficulty flags

- **`profile_y[rho_index] = ...` / `profile_y[~rho_index] = ...`** (source L203/L209,
  L443/L449) — flagged in `plasma_profiles.md` and `next_steps.md` as needing
  `.at[].set()`. **The landed port does not use `.at[].set()`.** Every one of
  `calculate_density_profile`, `calculate_parabolic_temperature_profile`,
  `calculate_pedestal_temperature_profile` instead computes both the "inside" and "edge"
  branch as whole-array expressions and combines them with a single `jnp.where(inside,
  core, edge)`. This is a **different resolution than predicted**, not a partial one — no
  scatter-update appears anywhere in the file — and it sidesteps a real correctness trap
  `.at[].set()` would not have: the source's two-statement sequential overwrite
  (`profile_y[mask] = ...` then `profile_y[~mask] = ...`) is exactly what makes
  `NeProfile.calculate_profile_y`'s dead parabolic branch dead, and a literal `.at[mask].
  set()` / `.at[~mask].set()` translation would have reproduced that fragility rather than
  making it visible. **Workaround-known, resolved**, worth recording as the concrete
  example next_steps.md's flag should point to.
- **`n_plasma_profile_elements` as a static shape, not a value** — flagged in both prior
  records. Landed exactly as predicted: `ProfileGrid.n_plasma_profile_elements: int =
  eqx.field(static=True)`, no `Input` for it, not differentiated.
- **Three independent `0**x` / division-by-zero derivative traps**, one per profile
  function, each guarded the same way (`jnp.where` selecting a safe base before
  exponentiating, substituting the true value back after) — **workaround-known, minor**:
  - `calculate_density_profile`: `(rho/rped)**2` base goes negative for `rho > rped`
    (unreached by the source's mask); `(1-rho)/(1-rped)` at `rped == 1`.
  - `calculate_parabolic_temperature_profile`: `(1-rho**2)**alphat` at `rho == 1`, exactly
    the endpoint the harness's `_RHO_INTERIOR` sample routes around for the *reference*
    finite difference (PROCESS's own FD steps past `rho = 1` into a domain where `numpy`
    itself returns NaN — not a port disagreement).
  - `calculate_pedestal_temperature_profile`: `(rho/rped)**tbeta` at `rho == 0`, the
    `(1-ratio_pow)**alphat` base for `rho > rped`, and `(1-rho)/(1-rped)` at `rped == 1`.
- **`ProcessValueError("Negative temperature in plasma profile")`** (source L455-456) —
  **workaround-known**. The port poisons the *whole* returned array with `jnp.nan` rather
  than only the offending points, matching the source's exception discarding the entire
  profile rather than partially computing it. `reference_domain_errors =
  (ProcessValueError,)` on `TestPedestalTemperatureProfile`.
- **`sp.special.beta`** (`tcore`, L527) — **workaround-known**, reuses
  `plasma_profiles._beta` (the `gammaln` identity already verified to ~2e-15 relative in
  unit #12).
- **`sp.integrate.simpson`** (`Profile.integrate_profile_y`, L110-112) —
  **workaround-known**, reuses `plasma_profiles._simpson`. Correctness-critical, not
  cosmetic: `TestIntegrateProfileY`'s `"non-uniform-grid"` sample exercises the exact case
  (`x=` given, non-uniform spacing) where a naive uniform-shortcut reimplementation would
  disagree with `scipy` in value as well as derivative — the bug class unit #12's gradient
  check caught. No such reimplementation exists here because the helper is imported, not
  rewritten.
- **`ncore`'s negative-value floor** (source L275-283) — **workaround-known, and is model,
  not error-handling.** `jnp.where(value < 0.0, NCORE_FLOOR, value)` reproduces both the
  substituted value and its derivative (identically zero in the floored region, matching
  what PROCESS's own finite difference reports there too — verified by `TestNcore`'s
  `"floored-negative"` sample).
- **`tcore` has no analogous floor.** Unlike `ncore`, the source never checks `tcore`'s
  output for negativity — a negative or otherwise unphysical on-axis temperature only
  becomes visible one call later, when `calculate_pedestal_temperature_profile`'s
  min-check catches it (or doesn't, if the negative value happens not to propagate to a
  negative point in the profile). Not a JAX-traceability problem — a genuine asymmetry
  between the two `*core` functions, ported faithfully, worth a second look by whoever
  owns the physics (see open question 6).

## open questions

1. **`i_plasma_pedestal`'s two-role problem (unit #12 open question 1) is confirmed, and
   this record supplies the node-level split needed to act on it.** See "switches
   touched" for the table: 4 nodes are switch-independent, 3 are parabolic-arm-only, 3 are
   pedestal-arm-only. Nothing here resolves the reconciliation with
   `density_limits.EcrhDensityLimit`'s static-kwarg use of the same field — still needs
   the "a switch also supplies its value to any node declaring it as a static kwarg"
   mechanism `next_steps.md` § 1 proposes.

2. **`i_nd_plasma_pedestal_separatrix` is a newly-found nested switch with no
   `switches.md` entry and no `configuration.py` mechanism for "this switch only exists
   under that other switch's value."** Same shape as the already-known `irefprop` /
   `i_blkt_coolant_type` nesting; this is the second occurrence, not a new kind of gap. Its
   two arms (`GreenwaldDensityFractions`, `PedestalSeparatrixDensities`) are exact
   inverses of each other rather than alternative formulas for the same output, which
   should make the eventual `Alternative` declaration mechanical once the nesting problem
   has a general answer.

3. **`calculate_profile_y` returns `None` on both `NeProfile` and `TeProfile`, and six call
   sites in `current_drive.py` (`culecd`, and two more call pairs — grep found `dlocal =
   1.0e-20 * ...calculate_profile_y(...)` and `tlocal = ...calculate_profile_y(...)`
   assigned directly) use that return value as a number.** Read directly:
   `NeProfile.calculate_profile_y` has no `return` statement anywhere in its body;
   `TeProfile.calculate_profile_y` returns `None` on both its branches (bare `return` in
   the parabolic arm, falls off the end in the pedestal arm). `1.0e-20 * None` and
   `tlocal = None` used arithmetically downstream both raise `TypeError` in real Python —
   this reads as a genuine latent bug, not a port artifact. **Not chased further**:
   `current_drive.py`'s `culecd`/`lhrad`-family methods are not called anywhere in
   `stellarator.py` (confirmed by grep) and are not a registered unit, so whether this
   code path is ever actually reached in a real tokamak run (and has therefore either never
   fired or always crashed) is current_drive.py's audit's question, not this one's. The
   port necessarily returns the array rather than `None` — any pure function has to — so
   this is a case where matching PROCESS's actual behaviour (as opposed to its evident
   intent) would mean deliberately reproducing a `TypeError`, which the port does not do.
   Flagged for whoever eventually audits `current_drive.py`.

4. **[RESOLVED — `LModeProfileReset` is that reset, registered on the
   `i_plasma_pedestal == 0` arm; see `plasma_profiles.md`'s "cottax node" section for the
   measured effect. The closing paragraph below is superseded: the reset turned out to be
   an ordinary node, not graph-assembly-time coercion, and until it existed a cold
   parabolic run really did get a pedestal-shaped density profile — it cost the SAND cold
   solve a 0.22 % objective gap against the warm one.]**
   **`DensityProfile`'s correctness in the parabolic configuration still depends on the
   unimplemented L-mode input-validation reset (`plasma_profiles.md` open question 2).**
   Verified algebraically here (not just cited): with `rped = 1`, `nped = nsep = 0` (what
   `plasma_profiles.parabolic_parameterisation`'s reset supplies, confirmed by reading
   `plasma_profiles.py:91-116` directly — it fires whenever any of the seven pedestal
   fields differ from their L-mode values, which they do under PROCESS's own defaults:
   `nd_plasma_pedestal_electron` defaults to `4.0e19`, not `0.0`), the pedestal formula
   `nped + (n0-nped)*(1-(rho/rped)**2)**alphan` degenerates term-by-term to exactly `n0 *
   (1 - rho**2) ** alphan` — the parabolic formula. Without that reset having already run,
   `DensityProfile` would compute the pedestal-shaped answer for a parabolic
   configuration silently. `DensityProfile` itself has nothing to fix; the reset is graph-
   assembly-time input coercion nothing currently performs.

5. **Scope**: `set_pedestal_and_separatrix_values` and its two functions/nodes are ported
   and tested but reachable only from `physics.py` (unit #22, tokamak) — never from
   `stellarator.py`. Registry action proposed: either note unit #21 as spanning one
   tokamak-only method, or record it as pre-emptively covering part of unit #22's future
   scope (precedent: unit #20 did the same for `impurity_radiation.py`'s `ImpurityRadiation`
   half, which *is* called by the stellarator path — the difference here is that this
   method is not). Not resolved here; a registry-level call, not an audit-level one.

6. **`tcore` has no floor while `ncore` does** (see JAX-difficulty flags). Whether this
   asymmetry is intentional (the physical failure mode for a runaway pedestal temperature
   is supposed to surface as `calculate_pedestal_temperature_profile`'s domain error rather
   than being silently floored) or an oversight is not something this audit can settle —
   flagged for physics review, not fixed.

7. **The density profile has no negative-value guard anywhere**, unlike the temperature
   profile's explicit check-and-raise. Not verified here whether a physically reachable
   input combination can drive `calculate_density_profile`'s output negative (the `core`
   term is `nped + (n0-nped)*shape` with `shape` clamped into `[0, 1]`-ish territory by the
   `jnp.where` guards, which bounds it whenever `n0 >= nped >= 0`, but no argument bound is
   enforced by this file). If PROCESS never checks density the way it checks temperature,
   that is presumably deliberate, but it was not independently confirmed.
