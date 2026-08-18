# Next steps

Snapshot as of the wave that consolidated §4c's 4-agent dispatch (`buildings.py`,
`vacuum.py`, `availability.py`, `costs/costs.py`+`costs_2015.py`) into
`total_process.py`/`unit_registry.md`. `unit_registry.md` remains the authoritative
per-unit status — this file is a priority-ordered punch list, not a second source of
truth; update it as items close rather than letting it drift the way a status doc always
tends to.

Suite at this snapshot: `$PY -m pytest functional_process` → **1499 passed, 1147
skipped** at the harness's default (`--fp-fuzz=1`, gradient checks gated off by default
— see §1c), **2637 passed, 9 skipped with `--fp-gradients`**, both green **except one
pre-existing, out-of-scope failure**:
`test_registry_coverage.py::test_every_record_file_is_in_the_registry` fails on
`functional_process/models/power_A_tf_coil_power.md`, an audit record the concurrent
`power.py` agent (running in parallel with this wave, per this file's own standing
instruction not to touch `models/power*`) landed mid-session; registering it in
`unit_registry.md` is that agent's own consolidation, not this one's. Default graph
(`total_process.GRAPH`) is now **58 nodes** (was 55), still exactly one SCC
(`Divertor`/`AFwTotalWithPowerflow`, only under `ipowerflow != 0`). `tests/unit`/
`~/jaxgraph` parity not re-checked this pass (no `process/`/`~/jaxgraph` changes made);
see the previous snapshot's line for those counts (846 / 307).

## 0. Closed since the last snapshot

- **§4c's 4-agent dispatch wave is consolidated.** All four units audited/ported —
  `buildings.py` (unit #15), `vacuum.py` (unit #16), `availability.py` (unit #17),
  `costs/costs.py`+`costs/costs_2015.py` (unit #18) — see `unit_registry.md`'s rows
  #15-18 for full per-unit detail. §4c itself (below) is kept for the historical
  dispatch-table record, marked `[CLOSED, consolidated]`.
  - **Registered**: `buildings.py`'s `TfCoilEnvelope` (unconditional, `COMMON`) and
    `Bldgs`/`BldgsSizes` under a new genuine topology `Switch`,
    `.buildings.i_bldgs_size` (default `0`/`ITER_1992` → `Bldgs`; `1`/`CHAPMAN_2024` →
    `BldgsSizes(i_hcd_primary=5)`) — the two arms collide on
    `.buildings.a_plant_floor_effective`/`.volnucb`, satisfying
    `check_arms_are_exclusive`. `vacuum.py`'s `VacuumOld` (unconditional, `COMMON`,
    matching PROCESS's own default `.vacuum.i_vacuum_pumping = "old"`). Default graph:
    55 -> **58 nodes**, still exactly one SCC.
  - **Two candidate switches investigated and found genuinely NOT wireable under the
    current `Switch`/`Alternative` mechanism** — new, general finding, not specific to
    either unit: `.vacuum.i_vacuum_pumping` (`VacuumOld`/`VacuumPumpingSimple` own
    completely disjoint output sets — no field in common at all, unlike
    `i_bldgs_size`'s working pair) and `.costs.i_cost_model` (the currently-ported
    subsets of `costs.py`/`costs_2015.py` share no output either, and `costs_2015.py`
    has zero `cottax` nodes written yet regardless). Both fail
    `Switch.check_arms_are_exclusive` — confirmed against
    `test_configuration.py::test_non_exclusive_arms_are_rejected`'s exact scenario, not
    just reasoned about. Registering either arm unconditionally instead was also
    rejected: PROCESS's own defaults (`i_vacuum_pumping = "old"`,
    `i_cost_model = 1`/`KOVARI_2014`) mean the *non-default* arm's nodes would compute
    values PROCESS's default configuration never reaches — the same bug class already
    found and fixed for `EcrhDensityLimit`. `VacuumPumpingSimple` and `costs.py`'s 23
    leaf cost nodes are left ported-but-unregistered. See § 1 (new sub-bullet) for this
    as a standing structural gap, not just a per-unit judgment call.
  - **`availability.py`'s bypass mechanism confirmed, but doesn't rescue registration.**
    `Stellarator.run()`'s solve-time branch calls `self.availability.avail()` directly
    (`stellarator.py:175`), bypassing `.costs.i_plant_availability`'s dispatch entirely
    (confirmed by reading the call site, matches the source's own `# TODO` comment) — so
    only `avail()`'s nodes are ever relevant during the solve; `avail_2`/`avail_st` are
    output/reporting-only, same treatment as `coils/output.py`. **But this doesn't make
    `avail()`'s own top-level node registrable**: `Avail` (and, independently,
    `Avail2`/`AvailSt`) each fail `cottax` node construction outright —
    `to_graph(Avail(...))` raises `ValueError: reads ['.costs.cplife'], which it also
    owns` directly from `cottax`'s own `__check_init__`, a genuine self-loop (`Avail`
    both reads and writes `.costs.cplife`, modelling PROCESS's real
    conditional-ownership: `.costs.cplife` is only *recomputed* when `.physics.itart ==
    1`, otherwise passed through unchanged). This is the **third and fourth confirmed
    instance** of the missing-`Cut` gap § 5 already tracks, found by direct
    construction, not audit reasoning. None of `availability.py`'s 6 written nodes are
    registered this pass.
  - **Three real PROCESS bugs found, all reproduced faithfully, not fixed**:
    `buildings.py`'s `calculate_bldgs_sizes` divides `life_plant` by itself in its
    hot-cell `hcomp_req_supply` term (always `1.0`, never actually scales with
    lifetime, unlike two sibling calculations right after it); `availability.py`'s
    `avail_2`/`avail_st` use `+` for a cross term where `avail()`'s own formula uses
    `-` in the total-availability calculation (caught by the harness's own checks), and
    both unconditionally divide by `.costs.f_t_plant_available`, which can legitimately
    be exactly `0.0` (a real, fuzz-reproduced `ZeroDivisionError` risk, mostly via
    `avail_st`'s maintenance-cycle model); `costs.py`'s `acc223` computes its
    neutral-beam cost (`c2233`) only when `ifueltyp == 1`, silently never otherwise —
    found in not-yet-ported code during audit, not itself part of the 23 ported nodes.
- **§4b's 5-agent dispatch wave is consolidated.** All five units audited/ported —
  `hcpb.py` (unit #13), `superconductors.py` (unit #22), `impurity_radiation.py`
  (unit #23), `physics.py`'s 8 in-scope methods (unit #9, split into 3 chunks —
  `physics_A_pure_formulas.py`/`_B_composition.py`/`_C_outplas.py`, see
  `unit_registry.md`'s new chunk table), `confinement_time.py`+`exhaust.py`
  (units #10/#11) — are now registered (where registration was warranted) in
  `total_process.py` and reflected in `unit_registry.md`. 11 new nodes landed in
  `total_process.COMMON`: `IonElectronEquilibration`, `AuxiliaryPhysicsQuantities`,
  `TotalPlasmaHeatingPower`, `ElectronThermalEnergy`, `IonThermalEnergy`,
  `FastAlphaBeta(i_beta_fast_alpha=1)`, `CalculateEffectiveChargeIonisationProfiles`,
  `DimensionlessPlasmaParameters`,
  `ConfinementTime(i_confinement_time=34, i_rad_loss=1, i_plasma_ignited=0)`,
  `DoubleAndTripleProduct`, `RadiationFraction`. Default graph: 44 -> **55 nodes**,
  still exactly one SCC.
  - **Deliberately left unregistered**: `superconductors.py`'s 8 functions and
    `impurity_radiation.py`'s 2 functions (no node written at all — every real call
    site's arguments are locals inside a not-yet-wired unit, same reasoning
    `coils.py`'s own unwrapped functions already established); `hcpb.py`'s 3 nodes
    (written, but reachable only from `blanket_neutronics()`, itself only called under
    `.fwbs.blktmodel == 1` — S2 of the `st_fwbs` synthesis, §3 below — and one of the
    three, `NuclearHeatingMagnets`, would conflict with `ScTfCoilNuclearHeating`'s
    existing ownership of `.fwbs.p_tf_nuclear_heat_mw` if registered unconditionally);
    `confinement_time.py`'s `IterPhysicsBasisElongation` (a duplicate producer of
    `.physics.kappa_ipb`, already owned by `ConfinementTime` itself); `physics_B_
    composition.py`'s `plasma_composition` (a genuine `.physics.first_call` self-loop
    — see the new §5 entry below).
  - **A real defect was found and fixed during consolidation, not by the porting
    agent**: `confinement_time.py`'s `ConfinementTime` node had `i_confinement_time`/
    `i_rad_loss` as plain (non-`eqx.field(static=True)`) annotated fields and
    `i_plasma_ignited` as an ordinary `Input` — none of which survive `jax.jit`/
    `jacfwd` tracing, since `calculate_confinement_time`'s dispatch does real Python
    `if`/`elif` branching on all three. Fixed to `eqx.field(static=True)` throughout;
    verified directly with `eqx.filter_jit` + `jax.jacfwd` post-fix.
  - **Four real PROCESS bugs found this wave, all reproduced faithfully rather than
    fixed** (per this project's standing policy — see `radiation_power.md`'s
    precedent): `stellarator.py`'s `blanket_neutronics()` calls
    `self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with **zero**
    arguments, though both are `@staticmethod`s requiring 2/7 keyword arguments —
    would `TypeError` the moment a `blktmodel == 1` run reaches this code, untested by
    any existing PROCESS test. **This directly unblocks, and immediately re-blocks,
    `st_fwbs`'s S2** (§3) — whoever picks up S2 next hits this bug on the very first
    line `blanket_neutronics()` executes past the geometry setup, and needs a plan for
    it (pass the same explicit kwargs `run()`'s own call sites use, or report upstream)
    before S2's own port can proceed. `coils.py:136` calls `jcrit_rebco(t_helium,
    b_max, 0)` — 3 positional args against a 2-argument signature — would `TypeError`
    if `i_tf_sc_mat == 6` ever executed. `calculate_confinement_time`'s
    `KAYE_GOLDSTON` branch (`i_confinement_time == 5`) calls its scaling law with
    positionally-scrambled arguments (6 of 8 land on the wrong physical quantity).
    `calculate_confinement_time`'s `USER_INPUT` (`i_confinement_time == 0`) always
    raises (a genuine `if`/`if` instead of `if`/`elif` bug) and `PAZ_SOLDAN_NT`
    (`i_confinement_time == 51`) is permanently shadowed by `NCST`'s identical enum
    value — both dead branches, reproduced as such.
  - **Registry bookkeeping fix**: 4 of the wave's 8 new/updated rows initially used
    `**reviewed**` in `unit_registry.md` where the record's own frontmatter still said
    `draft` (`physics_A_pure_formulas.md`, `physics_C_outplas.md`, `confinement_time.md`,
    `exhaust.md`) — all four are fully tier-1 and fully ported, the same shape 1D/1F
    were bumped to `reviewed` for, but bumping a record's own frontmatter is outside
    this consolidation pass's editing boundary (the 5 units' own files). Left as
    `draft` in the registry to match; a quick follow-up pass can bump both together.
- **`i_plasma_pedestal`'s two-role blocker is resolved and implemented.** It is a real
  `Switch` in `total_process.TOPOLOGY_SWITCHES` now. The mechanism next_steps previously
  proposed generically ("a Switch also supplies its value to any node declaring it as a
  static kwarg") turned out not to need new machinery for this instance: `EcrhDensityLimit
  (i_plasma_pedestal=0)` is simply one of the `value == 0` arm's declarations, so there is
  only one place `i_plasma_pedestal=0` is ever written, next to the switch value that
  requires it. **This surfaced a real correctness bug in the previous wiring**:
  `EcrhDensityLimit` was unconditionally in `COMMON`, but PROCESS's own default is
  `i_plasma_pedestal = 1` (`physics_variables.py:889`), and PROCESS's own
  `st_d_limit_ecrh` has no formula at all for that value (`density_limits.py:146-150`,
  the `else` arm only logs an error and leaves `dlimit_ecrh`/`bt_max_ecrh` undefined). So
  the default `GRAPH` was previously claiming to produce a value PROCESS's own default
  configuration never actually computes. Fixed by moving `EcrhDensityLimit` into the
  switch's `value == 0` arm — the default `GRAPH` now correctly omits it.
- **Unit #21 (`physics/profiles.py`) is fully audited and ported** — 12/12 functions,
  all tier-1, tests passing incl. gradients. Its per-arm classification is what unblocked
  `i_plasma_pedestal` above: 4 nodes common to both arms, 3 parabolic-only, 3
  pedestal-only, and 2 gated by a **different, newly-found nested switch**
  (`i_nd_plasma_pedestal_separatrix`) that turn out to be unreachable from the
  stellarator pipeline at all (ported, not registered — see unit #21's row). This
  **corrects** unit #12's original claim that both `NeProfile` and `TeProfile` branch on
  `i_plasma_pedestal` — true for `TeProfile`, but `NeProfile.calculate_profile_y`'s check
  is dead code (no `return` after the parabolic assignment), so `DensityProfile` is one
  common node, not two arms.
- **Unit #19 (`physics/fusion_reactions.py`) is ported**: `FusionRates`/`SetFusionPowers`,
  both in `COMMON`. `beam_fusion()` stays audit-only for a **new kind of blocker**, not
  entanglement or traceability alone: its `scipy.integrate.quad` call is non-JAX-traceable
  *and*, measured directly, only accurate to ~1e-6 relative even in PROCESS's own hands —
  four orders outside tier-1's tolerance regardless of how the integral is re-approximated.
  Worth a schema note for "PROCESS's own answer has a real accuracy ceiling", a category
  `test_harness.md` doesn't yet name.
- **Unit #20 (`physics/radiation_power.py`) is ported**: all 3 functions, all in
  `COMMON`. `ImpurityRadiationTotals` needed a static `imp_indices` kwarg (which impurity
  species exist — a graph-assembly-time fact, same move as a topology switch); assembled
  as all 14 species for the default graph, since H/He are always recomputed and species
  2-13 are held non-zero by iteration variables 125-136's bounds in the reference
  configuration this scope targets. Not generalized to other configurations — flagged,
  not resolved (radiation_power.md § open questions 2).
- **`coils/coils.py`'s `intersect` is ported** — self-contained tier-2, the first real
  `Tier2Contract` exercise. PROCESS's fixed-iteration finite-difference Newton loop (with
  a data-dependent early `break`, no faithful JAX translation) was replaced with
  `optimistix.Bisection` over the curves' guaranteed-valid overlap plus exact
  `jax.grad`-based Newton corrections (exact because `jnp.interp` is piecewise-linear).
  `winding_pack_total_size` (unit #9) is now unblocked on `intersect` specifically — its
  remaining blocker is `jcrit_from_material` → unit #22 (`superconductors.py`, new, see
  below), plus a design step of its own (minting VarPaths for its locals).
- **`st_fwbs`'s three chunks (1E1/1E2/1E3) are synthesized into real boundaries** — see
  `stellarator_E_fwbs_synthesis.md` and § 3. This was "the largest single blocker"; two of
  its six real sub-computations (S1, S5) are portable now, a third (S4) needs no further
  audit, and the two switch-discovery discrepancies the three chunks independently
  flagged are resolved (one was a false alarm, one — `first_call_stfwbs` — is confirmed
  as genuine cross-call state, requiring an `SCC`+`FixedPoint` node, not an ordinary one).
- **The transitive-closure sweep over constructor injection is done.** Six real scope
  gaps found beyond the four already on record, cross-checked against every registry row
  rather than found by accident: `physics/superconductors.py` (new unit #22, a *third*-
  level miss reached from inside an already-known-but-unported unit's own body — the
  first instance of that shape), a correction to unit #20's `impurity_radiation.py` note
  (two of its functions are reached by a second caller unit #20 didn't check — unit #9's
  `physics.py` methods, not just unit #20's own path — now unit #23), a trivial one-method
  addition to unit #21's scope (`physics/density_limit.py::calculate_greenwald_density_
  limit`), a **level-1 miss unit #9 (`physics.py`) should have caught**
  (`PlasmaBeta.fast_alpha_beta`, missed because `Stellarator.__init__` aliases the
  injected `plasma_beta` sub-model to `self.beta`, not the name the original grep
  checked), and method-list corrections to units #14 (`power.py`, +755 LOC of same-file
  callees) and #17 (`availability.py`, +2 whole alternate implementations behind a new
  switch, `i_plant_availability`). The sweep reports itself converged for every frontier
  it walked; two ported units' internals and 4 of `superconductors.py`'s 7 functions
  weren't re-swept (see the fork's full report), which is fine — those are inside
  already-scoped or already-excluded material, not open frontier.
- **Harness default-run compile time fixed** — unrelated to unit porting, prompted by
  `pytest functional_process` taking ~5 minutes at the harness's prior fuzz default,
  unworkable for routine iteration. See § 1c for the two structural fixes
  (`Tier1Contract`/`Tier2Contract` base-class changes, so every future unit gets them for
  free) and their measured effect.

## 1. Variant dispatch — the mechanism holds up under a second real case

`i_plasma_pedestal` (§ 0) is the second instance (`ipowerflow` was the first) where a
switch changes more than a formula — here, whether a static-kwarg precondition on an
*unrelated* node is even satisfiable in the default configuration. Both cases were
resolved without adding to `configuration.py`'s machinery: the general proposal from the
last snapshot ("a Switch also supplies its value to a node's static kwarg") is now
**withdrawn as unneeded** — every case seen so far reduces to "put the kwarg-carrying
instantiation inside the `Alternative` that requires it," which needs no new abstraction.
Revisit only if a *third* instance can't be expressed that way.

Still open, unchanged from last snapshot:

- **`blkttype` is three values over two arms** (`blkttype in {1, 2}` vs. `3`), which
  `Alternative.value` — one integer per arm — does not express. **Now has a concrete
  landing site**: `st_fwbs`'s S2/S4 sub-computations (§ 3) are exactly where `blkttype`'s
  arms live, and S2 is blocked on unit #13 anyway — decide the multi-value-per-arm
  question when S2 is actually audited, not before.
- **Nested switches**, now with a **second instance**: `i_nd_plasma_pedestal_separatrix`
  (§ 0), nested under `i_plasma_pedestal == PEDESTAL_PROFILE`, joins `irefprop` (nested
  under `i_blkt_coolant_type == WATER`). Both instances happen to be moot for this
  scope so far — `i_nd_plasma_pedestal_separatrix`'s gated nodes are unreachable from
  the stellarator pipeline at all, and `irefprop` is entangled with the CoolProp question
  (§ 5) anyway — but two independent instances of the same structural gap is worth a
  real decision once a third shows up on a *reachable* node.
- **New this wave: "reads-set genuinely differs, kept static anyway" now has three
  independent instances, not zero.** `i_confinement_time`/`i_rad_loss`
  (`confinement_time.md`) and `i_plasma_ignited` (both `confinement_time.md`'s and
  `physics_B_composition.md`'s independent uses of it) are all cases where
  `traceability_policy.md`'s own default ("reads-set differs -> split") technically
  applies but was **not followed** — each port kept the switch as a plain static kwarg
  on one composite node instead of splitting into per-value `Alternative`s, because the
  differing part of the body is small relative to a large shared body (2-6 lines inside
  a 48-branch dispatcher or a 328-line function). This is the same policy-deviation
  shape both porting agents independently flagged, not a mistake — but it is now a
  *pattern*, not an isolated judgment call, and the mechanism this section already
  withdrew ("a Switch also supplies its value to a static kwarg") was about
  topology-changing switches specifically, not this: the open question here is whether
  `traceability_policy.md`'s split-by-default rule itself needs a size/entanglement-aware
  exception (e.g. "split only if the differing body exceeds N lines or M% of the
  function"), not whether any one instance was handled correctly. `i_tf_sc_mat`
  (`superconductors.md`, confirmed split-worthy, 8 genuinely different reads-sets, no
  shared body to speak of — the *opposite* shape, one function per branch, no large
  common body) is a useful contrast case sitting right next to these three. Worth
  resolving as a real policy decision before a fourth instance shows up and each new
  unit re-derives the same judgment call independently.
- **New this wave: a switch can be genuinely mutually exclusive by PROCESS's own
  `if`/`elif` and still fail `Switch.check_arms_are_exclusive`, because that check only
  accepts colliding *output ownership* as proof of exclusivity, and two real instances
  now don't have it.** `.vacuum.i_vacuum_pumping` (`VacuumOld`/`VacuumPumpingSimple`
  own completely disjoint fields) and `.costs.i_cost_model` (the currently-ported
  subsets of `costs.py`/`costs_2015.py` share no output, though the unported
  `coe`/`concost` accumulation would) both hit this — see `unit_registry.md`'s rows
  #16/#18. Distinct from the `blkttype`/nested-switch gaps above: those are about one
  arm needing more than one integer value, or a switch nested inside another arm; this
  one is about the exclusivity *proof* itself being too narrow — a switch whose arms
  are exclusive by construction (an `if`/`elif` on the same field, not by name overlap)
  but happen to touch disjoint parts of `data`. The two rejected workarounds
  (unconditional-in-`COMMON` for one arm; loosening `check_arms_are_exclusive` to
  accept a caller's assertion instead of proving it) both have real downsides — the
  first reproduces the `EcrhDensityLimit` bug class when the registered arm isn't the
  configuration's actual value, the second removes the safety net
  `test_non_exclusive_arms_are_rejected` exists to provide. Worth a real design pass
  once a third instance turns up (candidates: `blktmodel`'s own remaining arms once S2
  is audited, § 3) rather than resolved ad hoc per unit.

## 1b. Harness: the gradient error bar fix — reconfirmed, not re-diagnosed

Unchanged from last snapshot; re-verified as still holding by re-running
`--fp-gradients` over the whole suite. (The pass counts recorded at the time —
2027-vs-1615 — are superseded by § 1c's fuzz-default and test-split changes; see the
top-of-file snapshot line for current numbers. The finding itself, that the gradient
check still separates a real bug from noise at `gradient_safety = 25`, is unaffected by
either change.) `test_harness_sensitivity.py`'s pinned regression (re-injecting the
`simpson` bug) still fails the gradient check as designed — the loosening has not been
blunted.

## 1c. Harness: default-run compile time fixed

Two structural fixes in `_harness/contracts.py`, both base-class changes so every future
unit inherits them automatically — not something a unit's own port needs to do anything
for.

- **Gradient checks fully gated, not just the finite-difference comparison.**
  `Tier1Contract.test_outputs_finite` used to differentiate every argument on *every*
  run — even without `--fp-gradients` — to catch a `jnp.where` leaking NaN through an
  untaken branch. Split: `test_outputs_finite` is now eager, value-only, always on; the
  differentiation moved to a new `test_gradient_finite`, marked `pytest.mark.gradient`
  alongside `test_gradient_agreement` — both skip unless `--fp-gradients` is passed. A
  default run now compiles zero autodiff graphs.
- **Batched multi-argument `jacfwd`.** `_jacobian(sample, name)` (one `jax.jacfwd` trace
  — one XLA compile — per differentiable argument, in a Python loop) became
  `_jacobians(sample)` (one `jax.jacfwd(f, argnums=(...))` trace over every argument at
  once). Measured 2.7x on `FusionRates`'s 11 arguments: 3.88s → 1.42s. `_jacobian` kept
  as a one-line wrapper for `test_harness_sensitivity.py`'s direct single-argument calls.
- **`Tier2Contract.ported` now `eqx.filter_jit`-wrapped once, at class-definition time.**
  `intersect`'s `optx.root_find` closes over its curve-defining arrays as free variables
  inside its own internally-traced `lax.while_loop`; unjitted, that embeds each sample's
  concrete values as literal constants in the compiled program, so a *different* sample
  of the *same shape* is a genuinely different program — every call recompiled from
  scratch, never getting cheaper (measured: four same-shape calls, 0.44/0.29/0.28/0.28s,
  no warm-up effect at all). Wrapping once as a class attribute — not inside a test
  method, where a fresh wrapper per pytest item would defeat its own cache — fixed it:
  0.24s once, ~0s for the rest. `residual` in `test_coils.py` was already hand-wrapped
  this way; `ported` wasn't, and every future Tier2 unit now gets the fix for free.

**Net effect**: default `pytest functional_process` (already down to `--fp-fuzz=1` from
the prior default of 8) went from **39.2s → ~13s**. `test_coils.py` (the one Tier2 unit)
went **5.90s → 3.05s**; a single isolated Tier2 test node went **~6s → ~2.2s** — the
complaint that triggered this. `--fp-gradients` (the full opt-in check, unchanged in
scope) still takes **~50s**, now only paid when actually asked for.

## 2. Review pass (yours, not mechanical)

- **`coilcurrent`** — resolved, no action needed (unchanged from last snapshot).
- **`preset_config.py`** (unit #8) — unchanged, still a proposal not a decision. Now a
  **fourth** instance of "this node always/only produces literals" alongside unit #6's
  device-preset literals and chunk 1D's constants — worth the same single policy
  decision, still not made.
- **`build.py` open questions** — unchanged, `dz_shld_upper` under `blktmodel <= 0` still
  open.
- **`neoclassics.py`** — unchanged, `.neoclassics.iota`/`.er` producer still unlocated.
- **New this wave**: unit #21's `set_pedestal_and_separatrix_values` is ported code
  reachable *only* from a unit explicitly outside this scope (unit #22, tokamak) — worth
  a registry-level policy on whether "port because splitting is more work than porting,
  but the caller is out of scope" (now two instances: this one and `ImpurityRadiation`'s
  whole-file treatment) should be a standing practice or reconsidered case-by-case.
- **New this wave**: `fusion_reactions.py`'s `calculate_profile_y` return-value bug
  (`profiles.py` returns `None` on both classes; 6 call sites in `current_drive.py` use
  the return value arithmetically) — not reachable from the stellarator pipeline, flagged
  for `current_drive.py`'s eventual audit, not chased further here.
- **Everything carried over and still unreviewed**: the hidden double-call pattern in
  `power_at_ignition_point` (`st_phys`'s is now understood, see § 3's S3), the
  constraint-91 unconditional-call discrepancy, the `.fwbs.fwclfr` possibly-dead-code
  flag, and (new) `radiation_power.md`'s open question 4 — the two `combine_radiation_
  powers` callers disagree on whether to clip at zero (`stellarator.py` does, `physics.py`
  doesn't).

## 3. Consolidation — `st_fwbs` synthesis done, one new re-chunking to act on

`stellarator_E_fwbs_synthesis.md` replaces the even-thirds 1E1/1E2/1E3 cuts with six real
sub-computations. Concretely, in priority order:

- **S1** `fw_blanket_shield_geometry_setup` (515-605) and **S5** `cryostat_and_vv_geometry`
  (1282-1330) are tier-1 and self-contained — **portable now**, same standing practice as
  any other self-contained tier-1 chunk (port as part of finishing the audit, not queued).
- **S4** `blanket_shield_fw_coolant_mass` (1045-1274 excl. S3) is tier-1 with **no further
  audit needed** — only blocked on S2/S3's signatures existing to call into.
- **S3** `divertor_mass_and_first_call_seed` (1030-1043) is the hard one: a genuine
  two-node SCC with `Divertor` (unit #4), bootstrapped by a hardcoded `50.0` on the true
  first call. This is **not** an ordinary node — it needs `Blocking` + a `FixedPoint`/
  `Square` driver over the `{st_fwbs, Divertor}` pair, the same shape as the `ipowerflow`
  SCC already found between `AFwTotalWithPowerflow` and `Divertor`. **First concrete
  candidate for § 5's "run `Blocking` over the real graph"** — small, well-understood, and
  would exercise the SCC-driver machinery for the first time on ported code rather than
  the current graph-structure-only check.
- **S2** `blanket_shield_tf_nuclear_power` (422-480+608-1030) is the real
  `blktmodel`×`ipowerflow` dispatch (3 live arms, tier-3 in two) — **unit #13
  (`hcpb.py`) landed this wave, so S2's own audit/port can now start**, but two things
  land in the very next auditor's lap immediately, both found while consolidating
  `hcpb.py`'s port, neither fixed:
  - **A live call-site bug, first thing S2's `blktmodel == 1` arm hits.**
    `blanket_neutronics()` (the function S2's `blktmodel == 1` arm *is*) calls
    `self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with **zero**
    arguments, but both are `@staticmethod`s requiring 2/7 keyword arguments — would
    `TypeError` the moment this path actually executes. Not exercised by any existing
    test. Two candidate fixes exist (pass the same explicit kwargs `CCFE_HCPB.run()`'s
    own call sites already use, since every value is on `self.data` by that point; or
    report upstream as a genuine PROCESS bug) — not resolved, S2's auditor's call.
  - **A real output-ownership conflict, needed for `total_process.py` registration.**
    `hcpb.py`'s `NuclearHeatingMagnets` writes `.fwbs.p_tf_nuclear_heat_mw`; so does
    `ScTfCoilNuclearHeating` (chunk 1F, unconditional in `COMMON` already) — confirmed
    against `stellarator.py` that `blanket_neutronics()` itself calls *both*
    `nuclear_heating_magnets()` and `sc_tf_coil_nuclear_heating_iter90()`, discarding
    the latter's own `p_tf_nuclear_heat_mw`-shaped output at that specific call site,
    while a *different* call site elsewhere in `st_fwbs` keeps it — i.e. these are
    genuinely alternative producers gated by `blktmodel`, not a redundant pair. S2's
    node design needs to resolve this (most likely a `Switch` on `blktmodel`, possibly
    entangled with `i_tf_sup` — see § 1's `blkttype` note, same landing site) before
    `hcpb.py`'s 3 already-ported nodes (or `ScTfCoilNuclearHeating`) can be
    conditionally registered instead of unconditionally as today.
  
  Still the sole blocker on two of the four open switch questions (§ 1's `blkttype`,
  plus `blktmodel`'s own remaining arm details) — now blocked only on S2's own audit,
  not on `hcpb.py`.
- **S6** `st_fwbs_output` (1331-1682) is a reporting shell, out of scope, no action.

`st_phys` (chunk 1B) — unchanged, still recommended tier-3 composition of ~13 sub-calls,
not yet acted on. Its Picard-iteration mechanism is understood; what's missing is the
same kind of node-shape decision S3 above needs, at a larger scale.

## 4. Remaining audit dispatches (updated — §4b's and §4c's waves closed, see §0)

Registry rows still `pending` or otherwise open, dependency-annotated. §4b's and §4c's
ten units (13, 22, 23, 9, 10, 11 from §4b; 15, 16, 17, 18 from §4c) are all now
audited and ported; what's left is the orchestration layer, `power.py` (a separate,
concurrently-running consolidation, not this file's to sequence), and the structural
blockers § 3/§ 5 already describe:

- `power.py` (unit #14, corrected method list, +755 LOC) — **being audited right now by
  a separate, concurrently-running agent**, per this file's own standing instruction
  (§4c's dispatch note) not to touch `models/power*` or wait on it. Its own
  consolidation (registration, registry/next_steps updates) is that agent's job, not a
  future wave of this file's.
- `stellarator.py` chunks 1A, 1B, 1C, 1G — still `draft`, not yet `reviewed`. 1E1/1E2/1E3
  are superseded by the synthesis (§ 3), not separately actionable anymore.
- S1/S5 (§ 3) — ready to port now, no audit blocker, just execution.
- `st_fwbs`'s S2 (§ 3) — unblocked on `hcpb.py` (landed §4b), now blocked only on its own
  audit (and needs a plan for the `blanket_neutronics()` zero-arg bug and the
  `p_tf_nuclear_heat_mw` ownership conflict, both § 3).
- **New from §4c's wave, not yet acted on**: the `Cut`-machinery blocker (§ 5) now has
  four confirmed instances (`plasma_composition`'s `first_call`, `st_fwbs`'s S3,
  `Avail`/`Avail2`'s `.costs.cplife`, `AvailSt`'s `.costs.cplife`) — building it is now
  the single highest-leverage piece of structural work left, since it unblocks
  `availability.py`'s entire top-level registration in addition to the two
  orchestration-layer cases. The `Switch.check_arms_are_exclusive`
  disjoint-output gap (§ 1, new sub-bullet) is smaller but affects two units already
  (`i_vacuum_pumping`, `i_cost_model`) and would need a real design decision (not just
  more porting) whenever a third instance turns up on a node that's otherwise ready to
  register.

With §4b/§4c closed, the balance-of-plant "wide but shallow" units are exhausted — what
remains is either the orchestration layer (sequential-only, see §4b's own reasoning for
why it was never parallelized) or genuine structural/machinery work (§ 5), not another
independent-files parallel wave. A future dispatch should pick one of: (a) build `Cut`
and resolve the four self-loops at once (single-threaded, architectural), (b) `st_fwbs`'s
S1/S5 (mechanical, no blocker, small), or (c) start on `stellarator.py`'s own chunks
(1A/1C first — 1B/`st_phys` is the largest and most entangled, better attempted once the
smaller orchestration chunks establish the pattern).

**Closed this snapshot** (§4c's wave, 4/4 units): `buildings.py` (unit #15), `vacuum.py`
(unit #16), `availability.py` (unit #17), `costs/costs.py` + `costs/costs_2015.py` (unit
#18) — see § 0 for the consolidated summary and `unit_registry.md` for the per-unit
detail. §4c itself is kept above for the historical dispatch-table record, not as an
active punch-list item.

**Closed previously** (§4b's wave, 5/5 units): `physics/superconductors.py` (unit
#22), `physics/impurity_radiation.py` (unit #23), `blankets/hcpb.py` (unit #13),
`physics/physics.py`'s 8 in-scope methods (unit #9, 3 chunks), `physics/confinement_
time.py` + `physics/exhaust.py` (units #10/#11) — see § 0's previous-snapshot entry
and `unit_registry.md` for the per-unit detail. §4b itself is kept below for the
historical dispatch-table record, not as an active punch-list item.

## 4b. [CLOSED, consolidated — see § 0] Dispatch — 5 parallel agents, priority-ordered

`hcpb.py` is the single highest-value target right now: it's the sole blocker on
`st_fwbs`'s S2 (§ 3) and, transitively, on two of the four open switch questions (§ 1).
The other four slots go to units that are either independently blocking
(`superconductors.py`, `impurity_radiation.py`) or simply large and completely untouched
(`physics.py`, plus the shared subset's two remaining small files). None of the five
reads or writes another's target file, so all five can run fully in parallel — the only
shared state any of them could collide on is `unit_registry.md`/`next_steps.md`/
`total_process.py`, and none of them should touch those (see Boundary, below).

| # | agent | target | why this slot | writes |
|---|---|---|---|---|
| 1 | `hcpb.py` | `blankets/hcpb.py`, 3 in-scope methods (`nuclear_heating_blanket`/`_magnets`/`_shield`) | **keystone** — sole blocker on `st_fwbs` S2 and 2/4 open switch questions | `models/blankets/hcpb.md` (+ `.py`/`test_*.py` if self-contained tier-1/tier-2) |
| 2 | `superconductors.py` | `physics/superconductors.py`, 7 material models behind `i_tf_sc_mat` (~603 of 1289 LOC) | blocks `jcrit_from_material` (unit #10) → `winding_pack_total_size` (unit #9) → `st_coil` | `models/physics/superconductors.md` (+ port whatever is self-contained; expect tier-2 leaning, `scipy.optimize`-based, per the registry's existing note) |
| 3 | `impurity_radiation.py` | `physics/impurity_radiation.py`, L379-755 model half + `calculate_average_charge_at_temp`/`element2index` | closes `ImpurityRadiationTotals`'s remaining gap and unit #9's dependency on it | `models/physics/impurity_radiation.md` (+ `.py`/`test_*.py`) |
| 4 | `physics.py` | `physics/physics.py`'s 8 in-scope methods (`plasma_composition`, `calaculate_stored_thermal_energy` [sic], `calculate_effective_charge_ionisation_profiles`, `calculate_total_plasma_heating_power`, `outplas`, `phyaux`, `rether`, `PlasmaBeta.fast_alpha_beta`) | central physics file, still fully `pending`, the largest remaining unaudited unit | `models/physics/physics.md` — **expect this needs chunking**, the same way `stellarator.py` did at a comparable size; instruct the agent explicitly to split into per-method or per-group records rather than force one flat file if any method turns out entangled or large enough to warrant it |
| 5 | `confinement_time.py` + `exhaust.py` | `physics/confinement_time.py` (2 methods), `physics/exhaust.py` (1 method) | small, in-scope-only method lists — bundled since individually tiny, not because they're related | `models/physics/confinement_time.md`, `models/physics/exhaust.md` (+ ports) |

**Boundary, same discipline as the last wave**: each agent may write its own audit
record(s), port file(s), and test file(s) — nothing else. `total_process.py`
registration, `unit_registry.md`/`next_steps.md` bookkeeping, and any new
`Switch`/`Alternative` wiring (`i_tf_sc_mat` for #2, plus whatever #1/#4 turn up) stay
reserved for the consolidation pass after all five return — this is what avoided
merge conflicts last time and the reasoning hasn't changed.

**Agents must not run the full suite (`pytest functional_process` with no path) —
only their own new test file(s).** Confirmed this wave: an agent that only writes new
files under its own scope *cannot* have broken anything elsewhere — there is nothing to
regression-check outside what it just wrote. Running the whole suite anyway costs several
times its own test file's runtime for zero additional signal, and it also reliably
surfaces `test_registry_coverage.py::test_record_frontmatter_agrees_with_registry`
failures caused by *sibling* agents' still-`pending` registry rows (registry bookkeeping
is explicitly not the agent's job, see Boundary above) — a false alarm the agent then has
to reason about and explain away in its report, for a check it was never going to be able
to fix. Scope every verification command to the agent's own path, e.g.
`pytest functional_process/models/blankets/` or
`pytest functional_process/models/physics/test_confinement_time.py
functional_process/models/physics/test_exhaust.py` — never the bare directory root.

**Agents do not need to chase `ruff` clean.** This is still proof-of-concept work — a
passing test suite is the actual bar; lint noise is not. Every unit ported so far carries
the same baseline `INP001`/`B008`/`PLC2701`/`D102` findings (namespace-package layout,
cottax's `Input(...)`/`Output(...)`-as-default idiom, numpy-docstring convention, private
imports) inherent to how this project is structured, not per-unit regressions — each
agent so far has independently re-confirmed this by diffing its own `ruff check` output
against an already-merged sibling file's, which is itself wasted effort once one agent has
established the baseline. Don't ask agents to run that comparison, and don't block a
report on lint findings that match the existing pattern; only flag a *new kind* of
finding, not the expected recurring ones.

**Deliberately not in this wave** — sequenced after, not parallel, with the reason each
was held back:

- **`st_fwbs`'s S1/S5** — fully audited already, no blocker, "just execution" per § 3.
  Small enough to fold into the consolidation pass directly rather than spend an agent
  slot on mechanical work with no audit judgment left to make.
- **`st_fwbs`'s S2** — waits on agent #1 above by construction; dispatching it now would
  mean auditing against an `hcpb.py` signature that doesn't exist yet.
- **`st_fwbs`'s S3 (the SCC)** — deliberately *not* a fork task. It's the first real
  exercise of `Blocking` + `FixedPoint` on ported code (§ 5), not another audit-and-port
  — that's an architectural first-use and deserves a focused, single-threaded pass, not
  a parallel agent working from a template that doesn't exist yet.
- **`stellarator.py` chunks 1A/1B(`st_phys`)/1C/1G** — the orchestration layer itself.
  `st_phys` has real internal Picard-iteration structure (§ 0); these chunks are exactly
  where the *next* SCC is most likely to surface, and a careless parallel split here
  (two agents both deriving node shapes from the same not-yet-settled orchestration
  logic, or one agent's node shape breaking another's assumed dependency) is the actual
  failure mode this section exists to avoid. Sequence these one at a time, or as a
  carefully-scoped pair once S2/S3's shapes are settled — not blindly parallelized like
  the five above.
- **`buildings.py`, `vacuum.py`, `availability.py`, `costs.py`/`costs_2015.py`** —
  untouched balance-of-plant units, no blockers, genuinely independent of everything
  above and of each other. Good candidates for a second 4-5-agent wave once this one
  lands and gets consolidated — left out of this wave only to keep its size and review
  load manageable, not because of any real dependency. **This is now § 4c.**

## 4c. [CLOSED, consolidated — see § 0] Proposed next dispatch (not yet launched)

§4b's own closing note already named the obvious next candidates; they're still exactly
right now that §4b has landed and been consolidated. A short proposal, not full agent
prompts (per this section's usual scope):

- **4 slots, same "independent, no shared file" discipline as §4b**: `buildings.py`
  (`Buildings.run()`, unit #15), `vacuum.py` (`Vacuum.run()`, unit #16),
  `availability.py` (`Availability.run()` + `avail_2`/`avail_st` behind
  `i_plant_availability`, unit #17), `costs/costs.py` + `costs/costs_2015.py`
  (gated by `i_cost_model`, unit #18) — all untouched, all `pending`, all genuinely
  independent of each other and of everything §4b touched.
  - `availability.py`'s slot should explicitly ask the agent to check whether
    `avail_st`'s `physics.itart == 1` requirement is truly dead on the stellarator
    pipeline — `unit_registry.md`'s row already calls this "a strong hypothesis, not
    structurally proven," and this wave separately found (`hcpb.md`) that `itart` is
    an ordinary, ungated `InputVariable` a stellarator IN.DAT *could* set to 1, which
    weakens (without refuting) the "provably dead" framing — worth a real check, not
    another assumption stacked on the first one.
  - `costs.py`'s slot is the first real test of the "two candidate units until
    `i_cost_model` is resolved" framing (`unit_registry.md` row #18) — expect it to
    surface a genuine topology-changing `Switch` (1990/2015/custom cost models are
    fully disjoint subgraphs, not a shared-body-with-a-branch case like this wave's
    `i_confinement_time`/`i_plasma_ignited` instances), which would make it the third
    real `TOPOLOGY_SWITCHES` entry after `isthtr`/`ipowerflow`/`i_plasma_pedestal`.
  - Not proposing `power.py` (unit #14) for this same wave despite also being
    `pending`: it's markedly larger (~755+ LOC across 6 methods and their callees)
    than the other four combined, and mixing one large unit into an otherwise-uniform
    small-unit wave risks the large one becoming the pacing item for a consolidation
    pass that would otherwise be as quick as this one. Better as its own slot, or
    paired with just one of the four above, in whichever wave picks it up.
- **Still deliberately not proposed for parallel dispatch**: `st_fwbs`'s S2 (needs its
  own audit, not blocked anymore, but has two known landmines — § 3 — worth a focused
  single pass rather than a parallel slot); `st_fwbs`'s S3 and `stellarator.py`'s
  orchestration chunks (1A/1B/1C/1G) — unchanged reasoning from §4b's own "deliberately
  not in this wave" list, restated there, not repeated here.

## 5. Structural work

- **Run `Blocking`/SCC over the real graph — updated result.** Default configuration is
  now **58 nodes** (was 55, before that 44, before that 32), still decomposing into
  exactly **one genuine SCC** (`Divertor`/`AFwTotalWithPowerflow`, present only when
  `ipowerflow != 0`). This is a tracked empirical finding, not a test of the rewrite's
  actual thesis (`CLAUDE.md`'s case for the rewrite is structural — making the graph
  explicit, not a bet on how much of it is cyclic).

- **Two genuinely different shapes have turned up under "cyclic-looking," not one — this
  was conflated in earlier snapshots of this section and is corrected here.** Both are
  real findings; they need different treatment, and only one of them needs anything not
  already in `cottax` today.

  **Shape A — ordinary cross-node cycles.** Two or more already-valid, separately-owning
  nodes whose dependencies happen to close a loop. `Graph` builds these with no error at
  all; `Blocking`/`strongly_connected_components` finds them exactly as designed. There
  is **no blocker** to registering these today — doing so requires no `Cut`, no minted
  copy, no driver decision, nothing beyond declaring the ordinary `Input` that names the
  real `VarPath` the other node owns. Confirmed instances:
  - `Divertor`/`AFwTotalWithPowerflow` — already registered, `ipowerflow != 0` only.
  - `Divertor`/`st_fwbs` S3's `DivertorPlateMass` — **registered.** Confirmed by directly
    checking `divertor.py`'s own `Input`s that `Divertor` depends on nothing `st_fwbs`
    produces, so this is a plain one-directional edge, not a cycle at all — PROCESS's own
    staleness (`st_fwbs` runs before `st_div`, so it reads the *previous* `run()`'s value)
    was purely a call-order artifact of its imperative code. Default graph: 58 → **61
    nodes** (S1/S5 also registered alongside it, portable-now per § 3), **still exactly
    one SCC** — confirms the edge really is acyclic, not just assumed to be.

  **Shape B — genuine single-node self-loops.** One `NodalDeclaration` whose own `Output`
  and `Input` name the identical `VarPath` — the value assumed and the value produced at
  once. `~/jaxgraph/CLAUDE.md`'s stated invariant: *"a node may not read what it owns...
  so a cycle is always at least two nodes."* This is not a style preference `cottax`
  enforces loosely — it is a hard construction error, confirmed empirically this wave:
  `to_graph(Avail(...))` raises `ValueError: reads ['.costs.cplife'], which it also
  owns` directly from `cottax.spec`'s `__check_init__`. **These cannot be represented as
  a plain node at all, regardless of whether anyone intends to drive them yet** — that is
  the one place a decision is actually forced now, and the decision is `cottax.interfaces
  .pytree_namespace_module.FixedPointFunction` (already implemented, confirmed by direct
  read), not a new primitive. Declaring `step(...)` instead of `__call__` mints the cut
  internally (`mint_key_cond`/`prefix_path`): the body reads the real `VarPath` and
  writes a minted `^cond.<var>` copy; a separate `FixedPoint` `DeclaredNode` (no body)
  reads that copy and owns the real `VarPath`. This is a **structural admission
  requirement, not a solver choice** — the resulting `FixedPoint` problem node is
  perfectly valid to sit undriven in the graph (`Graph.declared`/`declared_outside_cycles`
  exist for exactly this), same as Shape A's undriven SCCs. Confirmed instances, all
  currently unregistered (some not even written as nodes) because of this gap alone —
  every one of them is otherwise fully ported, tested, and correct:
  - `physics_B_composition.py`'s `plasma_composition` — `.physics.first_call`, read to
    pick a branch, written to `0` on the bootstrap branch (`d(first_call_next)/
    d(first_call) == 1` on the pass-through branch, caught by `test_gradient_agreement`).
    No node was written at all (the outcome was known in advance from `Avail`'s
    precedent).
  - `availability.py`'s `Avail`/`Avail2` (`.costs.cplife`, via a `cplife`/`cplife_in`
    pair modelling PROCESS's real conditional ownership under `.physics.itart == 1`) and
    `AvailSt` (a bare `cplife` pair) — this is what produced the hard construction error
    above; blocks all of `availability.py`'s top-level registration (`unit_registry.md`
    row #17), not just one node.
  - `power.py`'s `component_thermal_powers` — `.power.delta_eta`, read (feeding
    `plant_thermal_efficiency`) before being computed later in the same call. Same shape
    as the `power_at_ignition_point`/`st_phys` worked example in `test_harness.md`.
  - `st_phys`'s `.physics.beta_fast_alpha` and `.physics.beta_beam` — both read at
    `stellarator.py:1931-1932` before being (re)computed later in the same `st_phys`
    call. Found this wave; broadens what was previously reported as "one confirmed
    channel" (`b_plasma_surface_poloidal_average`, `power_at_ignition_point`'s known
    2-call Picard trick) to at least three.

  **Action: rewrite the Shape B functions above as `FixedPointFunction` declarations.**
  This is also doable now, for the same reason Shape A's registration is — minting the
  cut is structural, not a decision about how (or whether) to ever drive the result.
  Doing this unblocks `plasma_composition`'s and all three `availability.py` units'
  registration at once, plus `i_tf_sup`'s already-ready `CpLifetimeSuperconducting`/
  `CpLifetimeResistive` pair (blocked only because its sole consumer, `Avail`, can't be
  registered yet — see `unit_registry.md`'s `i_tf_sup` row).

  **What stays deferred, deliberately, either shape**: assigning an actual `Drive`
  step/solver algorithm to any of these — Shape A's SCCs or Shape B's `FixedPoint`
  problem nodes — is a separate decision from representing them, held back until there's
  a total plan for how the whole graph gets cut/driven/nested into an MDA (per the
  session's own explicit call on this). For Shape A, closing a cycle with a driver later
  is `rewrites.Cut` + `FixedPointCut`/`RootFindCut`/`ResidualCut` on the *already-built*
  graph — a different, later mechanism from `FixedPointFunction`'s built-in cut, which
  only applies at declaration time for Shape B. Do not reach for `rewrites.Cut` to
  represent Shape A today; there is nothing to cut, the graph is already valid.

  **On coverage**: only one genuine cross-*subsystem* cycle (Shape A, `Divertor`/
  `AFwTotalWithPowerflow`) has turned up across ~60 audited nodes spanning physics,
  build, divertor, coils, heating, confinement, radiation, buildings, vacuum,
  availability, costs, and power. Every Shape B instance found is a *single PROCESS
  function* referencing its own earlier/later self, not evidence of broader
  cross-subsystem coupling waiting to be found by splitting those functions further —
  splitting `component_thermal_powers` (for example) would very plausibly just turn one
  self-referencing node into two mutually-referencing ones representing the *same* local
  loop, not reveal a new one. Read as real, moderately strong evidence for the
  acyclic-heavy thesis, not as an artifact to second-guess by hunting for more splits.
- **CoolProp / non-traceable-call policy** — still only flagged, unchanged from last
  snapshot.
- **Tolerance policy for tier-4 comparison** — still explicitly deferred.
- **New this wave**: a schema note is worth adding for `beam_fusion`'s blocker shape (§ 0)
  — "PROCESS's own reference answer has a bounded accuracy ceiling independent of
  approximation method" is a third category alongside "opaque external call" and
  "PROCESS's answer isn't ground truth for a non-converged tier-2 loop," not yet named in
  `test_harness.md`.
