# Next steps

Snapshot as of the wave that registered 9 already-ported-but-unregistered units into
`total_process.py`: `stellarator_B_st_phys.py` (chunk 1B), `stellarator_C_geometry.py`
(chunk 1C), `stellarator_fwbs_s2.py` (S2), `power_A_tf_coil_power.py`/
`power_B_thermal_cryo.py`/`power_C_electric_production.py` (unit #14, chunks A/B/C),
`availability.py` (unit #17), `physics_B_composition.py` (chunk B), and `coils/
calculate.py`'s `WindingPackJTfWp` (unit #9). `unit_registry.md` remains the
authoritative per-unit status — this file is a priority-ordered punch list, not a second
source of truth; update it as items close rather than letting it drift the way a status
doc always tends to.

**Update, much later same session — read § 8 first if you are picking this file up
fresh.** A separate arc of work (constraint/objective porting completed to full
coverage, then a real block-by-block MDA-vs-PROCESS comparison harness built and run
for the first time) landed after everything above and is the most current state of the
project. § 8 is the entry point for that; everything below this paragraph is the
snapshot this section's header describes, now one layer further back in history.

**Update, later same session**: `first_call` was removed entirely (see § 5's tally) —
`NextFirstCall` no longer exists, dropping the default graph from 94 to **92 nodes**, 11
SCCs to **10** (one fewer `FixedPointFunction` self-loop pair). This also finally broke
`test_configuration.py::test_ipowerflow_decides_whether_the_graph_has_a_cycle`'s
`coupled.cycles[0]`-index assumption described below (removing `NextFirstCall` changed
which cycle sorts first) — **now fixed**, the same way this section already recommended:
it searches `coupled.cycles`/`uncoupled.cycles` for the specific
`{Divertor, AFwTotalWithPowerflow}` set rather than indexing `[0]` or asserting global
`is_acyclic` (which is `False` in both configurations now, given the unconditional
`FixedPointFunction` self-loops). `$PY -m pytest functional_process` is fully green
again: 2766 passed, 2243 skipped, 0 failed (`--fp-fuzz=1` default).

Suite at the snapshot below (superseded by the update just above, kept for the historical
trail): `$PY -m pytest functional_process` → **2769 passed, 2243
skipped, 1 failed** at the harness's default (`--fp-fuzz=1`, gradient checks gated off by
default — see §1c), **5003 passed, 9 skipped, 1 failed with `--fp-gradients`**. The one
failure, same in both runs,
**not fixed this pass, out of this pass's editable-file boundary**:
`test_configuration.py::test_ipowerflow_decides_whether_the_graph_has_a_cycle` asserts
`coupled.cycles[0]` (hardcoded index) equals exactly `{Divertor, AFwTotalWithPowerflow}`
— an assumption that predates this session's own Shape-B wave (which added 6 new
`FixedPointFunction` self-loop pairs already, before this pass) and this pass's own 3 more
(`NextFirstCall`, `CplifeAvail`, `WindingPackJTfWp`) plus 1 new genuine cross-node cycle
(see below): `coupled.cycles` now has 11 entries and `Divertor`/`AFwTotalWithPowerflow` is
no longer reliably first. The fix is a two-line change to that one test (search
`coupled.cycles` for the specific `{Divertor, AFwTotalWithPowerflow}` set rather than
indexing `[0]`) but `test_configuration.py` is not in this pass's editable-file list, so it
is reported here instead of changed. `test_registry_coverage.py` is fully green this
snapshot (it was not, at the start of this pass — see below).

Default graph (`total_process.GRAPH`) was **94 nodes** (was 61), **11 SCCs** at this
snapshot, now 92/10 after `first_call`'s removal (see update above): 9 (now 8) are the
structurally-inherent 2-node `[node, ^problem[node]]` pairs every `FixedPointFunction`
mints (not new cross-subsystem coupling — see § 5), and 2 are genuine cross-node Shape-A
cycles — `Divertor`/`AFwTotalWithPowerflow` (unchanged, `ipowerflow != 0` only) and a
**newly found second instance**, `DensityProfile → FusionRates → PlasmaComposition →
PedestalOnAxisDensities → DensityProfile`, surfaced only once `PlasmaComposition` joined
the full graph this pass (a real density/composition/fusion-rate/pedestal feedback loop,
not an artefact — see § 5's own update). `tests/unit`/`~/jaxgraph` parity not re-checked
this pass (no `process/`/`~/jaxgraph` changes made); see the previous snapshot's line for
those counts (846 / 307).

## 0. Closed since the last snapshot

- **9-unit registration/consolidation pass is done.** Everything landed but not yet
  registered as of the previous snapshot — `stellarator_B_st_phys.py` (chunk 1B),
  `stellarator_C_geometry.py` (chunk 1C), `stellarator_fwbs_s2.py` (S2), unit #14's three
  `power_*.py` chunks, `availability.py`'s `Avail`/`CplifeAvail`, `physics_B_
  composition.py`'s `NextFirstCall`/`PlasmaComposition`, and `coils/calculate.py`'s
  `WindingPackJTfWp` — is now registered in `total_process.py`, or explicitly and
  individually justified as not-yet-registerable. See `unit_registry.md`'s per-row detail
  (rows 1B/1C/S2/#14/#17, chunk B) and its "Ported so far" final bullet for the
  consolidated write-up; not repeated in full here.
  - **Two new real `Switch`es wired**: `.tfcoil.i_tf_sup` (`power_A_tf_coil_power.py`,
    `TfPowerResistive`/`TfPowerSuperconducting`) and a synthetic joint
    `.fwbs.blktmodel,.heat_transport.ipowerflow` (`stellarator_fwbs_s2.py`, 2/3 arms —
    see § 3's own update and the switches table). `i_tf_sup`'s new pairing is
    independent of `availability.md`'s own long-standing `i_tf_sup` candidate
    (`CpLifetimeSuperconducting`/`CpLifetimeResistive`, still unregistered) — two
    `Switch` objects sharing one `path` coexist fine, confirmed directly, no new
    `configuration.py` machinery needed.
  - **Two more `EcrhDensityLimit`-class registration bugs found and fixed**:
    `ScTfCoilNuclearHeating` (unconditionally in `COMMON`, correct only for the new
    switch's non-default arm) and `BlktmodelBlanketThickness` (unconditionally in
    `COMMON` despite PROCESS's own default, `blktmodel = 0`, requiring it not exist at
    all — found incidentally while checking precedent for `DefaultAspectRatio`). Neither
    was this pass's own unit; both were existing `total_process.py` entries from earlier
    waves, now corrected.
  - **A second confirmed cross-node Shape-A cycle**, found once `PlasmaComposition`
    joined the full graph: `DensityProfile`/`FusionRates`/`PlasmaComposition`/
    `PedestalOnAxisDensities`. See § 5.
  - **A second wave of Shape-B self-loops found, deliberately not resolved this pass**:
    `PlantThermalEfficiency`/`PlantThermalEfficiency2`, `Cryo`/`CryoLoads`,
    `PlantElectricProduction` — all in the concurrently-landed `power_*.py` chunks, all
    confirmed by direct `to_graph` construction. See § 5's own update.
  - **Two concurrently-running, unrelated sessions' work found mid-pass, deliberately
    not integrated**: `coils/coils.py`'s new `jcrit_from_material` per-branch nodes (no
    consumer anywhere in the graph, minted `VarPath`s only) and `coils/calculate.py`'s
    `intersect` wiring (`next_steps.md` §7's own pending item, landed mid-session,
    replacing `WindingPackTotalSize` with `WindingPackIntersectInputs` +
    `Intersect`/`WindingPackTotalSizePost` — not independently audited by this pass, left
    for the next one; see § 7).
  - **`test_registry_coverage.py` is fully green again**: 6 orphan `.md` records (S1+S5,
    S2, S3, and the 3 `power_*.py` chunks) added to the registry, plus a genuine
    pre-existing gap fixed (`stellarator_fwbs_s1_s5.md` had no frontmatter block at all —
    added, matching its sibling records' schema, no computational content touched) and
    one stale status bump (`stellarator_C_geometry.md`'s registry row said `draft`, its
    own frontmatter already said `reviewed`).
  - **One pre-existing test now fails that did not before**, not fixed (outside this
    pass's editable-file boundary): `test_configuration.py::
    test_ipowerflow_decides_whether_the_graph_has_a_cycle`'s `coupled.cycles[0]`
    index-based assertion is stale against the now-11-SCC graph. See the top-of-file
    snapshot line for the exact fix needed.
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
  is audited, § 3) rather than resolved ad hoc per unit. **Update: S2 itself turned out
  not to be a third instance** — `ExponentialAttenuationBlanketShieldPower`/
  `DetailedPowerflowBlanketShieldPower` (the two ported `blktmodel != 1` arms) *do*
  share 4 outputs, so `check_arms_are_exclusive` passes cleanly; S2 is registered as a
  real `Switch` this snapshot (a synthetic joint `path`, see `unit_registry.md`'s
  switches table — a different, unrelated wrinkle from this gap).
- **New this snapshot: a genuinely distinct third gap — two *different* real switch
  values that select the *identical* node set.** `.tfcoil.i_tf_sup`
  (`power_A_tf_coil_power.py`, unit #14 chunk A): `Power.tfpwr` dispatches on
  `i_tf_sup != 1` only, so values `0` (resistive copper) and `2` (aluminium) both run
  `calculate_tf_power_resistive`, the identical node. Declaring both as ordinary
  `Alternative`s (two values, one node set) fails `test_configuration.py::
  test_arms_select_different_node_sets` directly (confirmed, not assumed) — that test's
  own stated philosophy is exactly right: a switch value that doesn't change which nodes
  exist belongs in the *other* `naming_convention.md` category (a static kwarg), not a
  `Switch` value. Resolved pragmatically this snapshot by declaring value `2` `unported`,
  pointing at value `0`'s identical result, rather than duplicating the `Alternative` —
  works, and is honest (nothing is actually missing, `.tfcoil.i_tf_sup == 0` gives the
  exact same graph), but is a workaround, not a structural answer: `Switch`/`Alternative`
  has no way to say "these two literal values are the same arm" directly, only "declared"
  vs. "unported." Distinct from both gaps above — those are about arms not proving
  exclusivity by output; this is about two arms proving *too much* exclusivity (identical
  output, not just overlapping). Worth a real design pass (e.g. letting one `Alternative`
  declare a tuple of `value`s) once a second instance turns up.

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
  now **94 nodes** (was 61, before that 58, 55, 44, 32), decomposing into **11 SCCs
  total** — 9 are the structurally-inherent 2-node `[node, ^problem[node]]` pairs every
  registered `FixedPointFunction` mints (`NextFirstCall`, `DeltaEtaStep`, `EtaTurbineStep`,
  `EtathLiqStep`, `TempTurbineCoolantInStep`, `PFwDivHeatDepositedMwStep`,
  `PFwBlktCoolantPumpMwStep`, `CplifeAvail`, `WindingPackJTfWp` — not new cross-subsystem
  coupling, see Shape B below), and **2 are genuine cross-node Shape-A SCCs**:
  `Divertor`/`AFwTotalWithPowerflow` (unchanged) and a **newly found second instance**,
  `DensityProfile`/`FusionRates`/`PlasmaComposition`/`PedestalOnAxisDensities` (see Shape
  A's own update below). This is a tracked empirical finding, not a test of the rewrite's
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
  - `DensityProfile`/`FusionRates`/`PlasmaComposition`/`PedestalOnAxisDensities` — **new
    this snapshot**, a genuine 4-node cycle, the second confirmed cross-subsystem
    instance. Surfaced only once `physics_B_composition.py`'s `PlasmaComposition` joined
    the full graph (registering it alone, or `to_graph`-checking it against any proper
    subset of the other three, would not have shown this — it needed the whole assembled
    `GRAPH`). The edges: `PedestalOnAxisDensities` → `DensityProfile`
    (`nd_plasma_electron_on_axis`) → `FusionRates` (`nd_plasma_electron_profile`) →
    `PlasmaComposition` (`proton_rate_density`) → `PedestalOnAxisDensities`
    (`nd_plasma_ions_total_vol_avg`). A real physics feedback loop (density profile ↔
    plasma composition ↔ fusion reaction rates ↔ pedestal on-axis density), not an
    artefact of how the port split functions — registered as-is (Shape A needs no `Cut`),
    not driven. This **corrects** the "On coverage" paragraph below, which had reported
    only one genuine cross-subsystem cycle across ~60 nodes; make that "two, across 94"
    going forward.

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

  **On coverage — updated, corrected.** Previously reported as "only one genuine
  cross-*subsystem* cycle across ~60 audited nodes"; **now two, across 94** — see Shape
  A's new `DensityProfile`/`FusionRates`/`PlasmaComposition`/`PedestalOnAxisDensities`
  entry above, found this snapshot once `PlasmaComposition` joined the full graph. Every
  Shape B instance found (both waves, see the "Shape B conversion wave" subsection below
  for the second) is still a *single PROCESS function* referencing its own earlier/later
  self, not evidence of broader cross-subsystem coupling waiting to be found by splitting
  those functions further — that half of the finding stands. But the Shape-A half no
  longer reads as "basically none": two real cross-subsystem cycles in 94 nodes is not
  strong evidence either way yet, and the density/composition/fusion-rate/pedestal loop
  in particular is a genuine, not-especially-rare shape (plasma-state variables feeding
  back into geometry/profile choices) — worth treating as "expect a few more, not zero"
  rather than re-asserting the acyclic-heavy thesis on the strength of one further data
  point.
- **CoolProp / non-traceable-call policy** — still only flagged, unchanged from last
  snapshot.
- **Tolerance policy for tier-4 comparison** — still explicitly deferred.
- **New this wave**: a schema note is worth adding for `beam_fusion`'s blocker shape (§ 0)
  — "PROCESS's own reference answer has a bounded accuracy ceiling independent of
  approximation method" is a third category alongside "opaque external call" and
  "PROCESS's answer isn't ground truth for a non-converged tier-2 loop," not yet named in
  `test_harness.md`.
- **Shape B conversion wave: 3 of 5 originally-known instances resolved with
  `FixedPointFunction`, one turned out to be a 4th ordering artifact and was un-resolved
  again, one Shape A false-alarm caught before it was built, two new instances surfaced
  along the way.**
  - Resolved (split into a tiny `FixedPointFunction` for the self-referential piece +
    an ordinary node for everything else, `to_graph()`-verified):
    `winding_pack_total_size`'s `j_tf_wp` (`WindingPackJTfWp`, degenerate/identity fixed
    point off the Bi-2212 material branch, confirmed with `jax.grad` rather than
    asserted).
  - **`plasma_composition`'s `first_call` — un-resolved.** This wave's first pass split
    it the same way (`NextFirstCall`/`FixedPointFunction`), and `to_graph()` did verify
    the split assembled. A later pass asked *why* `first_call` exists at all rather than
    just how to represent it, and found the answer PROCESS bootstraps toward
    (`.physics.f_temp_plasma_electron_density_vol_avg`, from `plasma_profiles.py`) has no
    dependency back on `plasma_composition` — same shape as `Divertor` and
    `beta_fast_alpha`/`beta_beam` below, a 4th confirmed ordering artifact, not a cycle.
    `NextFirstCall` has been deleted; `plasma_composition` now always uses the real value
    directly, and `first_call`/`alphan`/`alphat` are not ported at all. See
    `physics_B_composition.md`'s "the `first_call` self-loop" section for the full
    account. **The lesson this leaves for the remaining Shape B instances below**:
    "representable via `FixedPointFunction`" and "`to_graph()` assembles" are necessary
    but not sufficient — check the *real* producer of what a self-loop's other branch
    resolves to before concluding the loop is genuine, the same discipline that caught
    `beta_fast_alpha`/`beta_beam` as Shape A.
  - `st_phys`'s `beta_fast_alpha`/`beta_beam` turned out to be **Shape A, not Shape
    B** — caught by the task's own instruction to re-verify rather than trust the
    original "read-before-write staleness" framing. `beta_fast_alpha`'s sole owner
    (`FastAlphaBeta`, already registered) has no dependency back on anything computed
    later in `st_phys`; `to_graph([FastAlphaBeta(...), StellaratorBetaAndRhoStar()])`
    assembles cleanly. `beta_beam`'s sole owner is `beam_fusion`, unported (see the
    accuracy-ceiling blocker above), same shape once it lands. No code needed for
    either — an ordinary edge to register later, third confirmed instance of "don't
    assume a cycle without checking" alongside `Divertor` (`first_call` above is the
    fourth).
  - **Two new instances found, deliberately not fixed yet, for two different
    reasons**:
    - `plasma_composition`'s `.impurity_radiation.f_nd_impurity_electron_array` —
      reads indices `2:13`, writes indices `0:1`, same array `VarPath`. **Update, this
      snapshot: resolved, correcting the paragraph this replaces** (which described
      this as "worked around... not declared as an `Output` on `PlasmaComposition` at
      all" — no longer accurate). Per-index `VarPath`s
      (`s.impurity_radiation.f_nd_impurity_electron_array[i]`, a `SequenceKey`
      component — the exact mechanism this paragraph previously said didn't exist for
      a `slice`, but does for a plain `int` index) turn out to be sufficient: indices
      2-13 are twelve ordinary `Input`s, indices 0/1 (`f_nd_impurity_electron_array_h`/
      `_he`) are two ordinary `Output`s, no slice-addressing needed at all since every
      index is handled individually. `PlasmaComposition` is now registered in
      `total_process.COMMON` with this shape; `test_plasma_composition_owns_h_and_he_
      fractions` confirms it directly. No open `cottax` capability question remains
      for *this* instance — flagged only in case a future array self-reference isn't
      as cleanly index-separable as this one turned out to be.
    - `st_coil`'s `len_tf_coil` — on inspection, **not actually a Shape B blocker at
      all**. `st_coil` has no `cottax` node of its own (it's a plain composed
      function, deliberately — see `calculate.md`); the port already resolved
      PROCESS's own call-order staleness by giving the early (stale) read its own
      parameter name, `len_tf_coil_stale`, distinct from the freshly-computed
      `len_tf_coil` used everywhere else in the function. No node anywhere currently
      owns `.tfcoil.len_tf_coil` as an `Output` (grepped, confirmed empty) — so
      there's no ownership conflict to hit, only an unowned external input, same as
      many others in the graph. Nothing to fix here; flagged only so it doesn't get
      re-discovered and mistaken for an eighth Shape B case.
  - **A second wave, 5 more instances, found by this snapshot's registration pass —
    3 resolved (already landed before this pass started), 5 new ones found, still
    unresolved.** Landed (not this pass's own work, found already done when this pass
    began): `availability.py`'s `Avail`/`Avail2` (`CplifeAvail`) and `AvailSt`
    (`CplifeAvailSt`) — see unit #17's row, now registered (`Avail`+`CplifeAvail`) this
    pass. `power_B_thermal_cryo.py`'s `component_thermal_powers` turned out to have
    **six**, not one (`delta_eta` plus `eta_turbine`/`etath_liq`/
    `temp_turbine_coolant_in`/`p_fw_div_heat_deposited_mw`/
    `p_fw_blkt_coolant_pump_mw`) — all six already split into their own
    `FixedPointFunction`s before this pass began, now registered alongside
    `ComponentThermalPowers`. **New, found by this pass, deliberately left
    unregistered**: `power_B_thermal_cryo.py`'s `PlantThermalEfficiency`/
    `PlantThermalEfficiency2` (the raw `ExplicitFunction`s the `EtaTurbineStep`/
    `EtathLiqStep`/`TempTurbineCoolantInStep` splits above were extracted from — each
    is *itself* still a self-loop on its own, confirmed by direct `to_graph`
    construction, superseded rather than fixed) and `Cryo`/`CryoLoads` (`.fwbs.qnuc`,
    plus `.power.qss`/`qac`/`qcl`/`qmisc` for `CryoLoads`, which calls `calculate_cryo`
    internally under its own guard); `power_C_electric_production.py`'s
    `PlantElectricProduction` (`.heat_transport.p_plant_electric_gross_mw`/
    `p_plant_electric_recirc_mw`/`p_plant_electric_net_mw`, `.power.p_turbine_loss_mw`/
    `f_p_plant_electric_recirc`). All confirmed the same way as every instance above —
    direct `to_graph` construction, not audit reasoning — and all are otherwise fully
    ported and tested. No `FixedPointFunction` split written for any of the five yet;
    same recipe as above would apply, just not done this pass (this pass's own
    boundary was registration, not further porting).

## 6. Constraints, objective, iteration variables — not a separate layer, just a thin
   selection over models already being ported

Prompted by a direct discussion of `CLAUDE.md`'s own mapping table, worth stating
explicitly here so it survives past this session: **constraints and the objective are
not architecturally special to `cottax`.** The only thing that makes them look like a
separate layer is that PROCESS bundles them with the *solver*, not that they need new
graph machinery.

- **Constraints** (`process/core/solver/constraints.py`) are either an ordinary
  `Compare(place, pairs=[(model_output, stored_bound)])` node over outputs that are
  already (or will be) ordinary ported model nodes, or — for a constraint that just
  thresholds one `data` field against a bound — not even a node, a bare residual read.
  No new primitive needed. **Likely register unconditionally**, not behind a `Switch`:
  PROCESS doesn't gate a constraint's *computability* on anything, only whether the
  solver bothers enforcing it (`numerics.icc`, the active-constraint list) — that's an
  `Optimise`-problem-assembly-time decision (`Combine` folding the wanted subset
  together), not a graph-existence one. Only 3 are audited so far (91 in full, 17/24
  briefly) — none ported as `Compare` nodes yet. That's the actual remaining work:
  ordinary audit-and-port, `Tier1Contract`-style, same as every other unit.
- **The objective** (`numerics.i_figure_merit`'s branch selection) is not a node at
  all — *"a per-run selection of which existing output is 'wanted', same as cottax's
  refusal to have an `OutputNode`"* (`CLAUDE.md`). Whatever `rmajor`/`coe`/etc. a given
  `i_figure_merit` value picks is already a real output of an already-ported model.
  Selecting it is a `Graph.prune`-style query run when assembling an `Optimise`
  problem, not something `total_process.py`/`configuration.py` needs to represent at
  assembly time the way topology switches do. `total_process.py` picking one example
  as an illustrative `prune` call is optional demonstration, not required structure.
- **Iteration variables** (`ITERATION_VARIABLES[id]`/`numerics.ixc`) are the same
  shape again: *"The integer ID... is exactly a `VarPath`; the ID itself is throwaway
  indirection once names are structural"* (`CLAUDE.md`). Not a model, not new
  computation — a designation ("this `VarPath` is free, not derived") on values that
  already exist once their producer models are ported. Falls out for free once
  constraints + models exist; no separate porting effort.

**Net**: the only genuinely open porting work in this whole area is the handful of
constraint bodies. The objective and iteration-variable pieces need no new code at
all, just a query/designation once the constraint and model layers are there. This
significantly narrows what was previously described (earlier in this file, before this
session's discussion) as "a real porting pass, smaller, different in kind, not yet
started" — it's smaller than that phrasing implied.

## 7. A third pattern, distinct from Shape A/B — raised, then resolved by a sharper
   test than "does it iterate"

Raised by direct challenge: every `Tier2Contract` unit ported so far solves its own
internal iteration **eagerly, inside a plain JAX function** — `optx.root_find(...)`
(`coils.py`'s `intersect`) or a hand-rolled `jax.lax.while_loop` Newton scheme
(`vacuum.py`'s `solve_duct_diameter`/`VacuumOld`) — rather than as a `cottax.interfaces.
pytree_namespace_module.ImplicitFunction`. First framing (wrong, corrected below): this
looked like the same gap the Shape B wave (§ 5) had just closed, just with `RootFind`
instead of `FixedPoint`.

**It isn't, and the reason clarifies the actual rule.** A `RootFind` `DeclaredNode` (what
`ImplicitFunction` mints) has **no body** — it produces no value at all until a `Drive`
step runs some algorithm against it. `FixedPointFunction`'s Shape B conversions cost
nothing precisely because `step`'s "next" value was *already* a complete, fully
determined computation with no external solver needed — the only problem was the
self-referential *naming*. `intersect` is different in kind: there is no value without
actually running an iterative algorithm, so converting it to a declared, undriven
`ImplicitFunction` wouldn't just add structure, it would remove `intersect`'s current
ability to be called and produce an answer — which both `Tier2Contract`'s own tests and
`winding_pack_total_size`'s still-eager body genuinely need today.

**The right test turns out not to be "does it iterate," but "does anything else in the
graph need to read or write something inside that iteration's own state."**
`intersect`'s unknowns (`wp_width_r`, `lhs`, `rhs`) are fully encapsulated inside
`winding_pack_total_size`'s own private computation — nothing else in the graph needs
them independently, now or plausibly ever. That makes it structurally a numerical
primitive (no different in kind from calling `jnp.linalg.solve` inside a larger pure
function), not a coupled subsystem — declaring it would be structure with no consumer,
pure cost, no benefit. Same conclusion for `solve_duct_diameter`/`solve_duct_geometry`
inside `VacuumOld`. The genuine Shape B cases (`first_call`, `cplife`, `delta_eta`, and
`power_B_thermal_cryo.py`'s five) were different precisely because their `VarPath`s
*are* real, externally-relevant `DataStructure` fields — that external relevance is what
actually earns the `FixedPointFunction`/`ImplicitFunction` treatment, not internal
iteration by itself.

**Conclusion: `intersect`/`solve_duct_diameter` are fine as they are — no follow-up pass
needed for these two.** The actionable form of this finding, for future tier-2 units:
before assuming a unit needs `ImplicitFunction`, check whether any *other* node actually
needs its internal unknowns — if not, an eager JAX function is the correct, not merely

**Update, this snapshot: `intersect` was converted anyway, mid-session, by a separate,
concurrently-running agent** (not this pass's own work — found only because it changed
`coils/calculate.py`'s public names out from under this pass's own registration attempt,
breaking an import). `coils.py` now has `Intersect(ImplicitFunction)` (a `RootFind`
`DeclaredNode`), and `coils/calculate.py`'s `WindingPackTotalSize` was split into
`WindingPackIntersectInputs` (mints the `(wp_width_r, lhs, rhs)` curves `Intersect`
reads) + `Intersect` + `WindingPackTotalSizePost` (reads `Intersect`'s
`.stellarator.wp_width_r_min` as an ordinary `Input`). Its own docstring gives a
**different** justification than this section's own test ("does anything else need the
internal unknowns" — still, on its own terms, "no" here): making the root-find's solver
algorithm "a first-class, swappable `Drive` choice, not something hardcoded inside one
node's body." That is a real, distinct consideration this section didn't weigh — worth
reconciling explicitly in a future pass (does "swappable `Drive` choice" alone justify
`ImplicitFunction` even with no external consumer, updating this section's own
conclusion, or was this conversion done for a different reason not yet written down) —
not resolved here. **Not audited or registered by this pass**: `coils.py` is explicitly
outside this pass's editable/relied-upon boundary, the split's own tests do pass
(`pytest functional_process/models/stellarator/coils/test_calculate.py` → 67 passed, 28
skipped, checked directly), but this pass did not independently verify the design the
way it verified everything it registered. `WindingPackJTfWp` (unaffected — still calls
the unchanged `winding_pack_total_size` pure function directly) is registered;
`WindingPackIntersectInputs`/`Intersect`/`WindingPackTotalSizePost` are not. Left for the
next consolidation pass, alongside `coils.py`'s new `jcrit_from_material` nodes (see
`unit_registry.md`'s "Ported so far").
expedient, shape.

## 8. The MDA harness — built, run for the first time, and iterated on this session

**Read this section first if picking the file up fresh — then § 9, which is one layer
more recent still.** Everything above is real history, not stale, but this section and
§ 9 are where a new session should orient from. § 9 closes the single blocker
`_audit/optimise_design.md` names (`.costs.coe` had no producer) and adds a third
`Alternative` state to `configuration.py`; it also supersedes recommended-next-step 5
below in part.

### What exists now that didn't at the start of this session

- **All ~82 PROCESS constraints are ported** (`functional_process/core/solver/
  constraints.py`/`.md`/`test_constraints.py`), not just the 5 stellarator-specific
  ones from the earlier snapshot. Only 2 are genuinely excluded (50/52, IFE-only —
  `.ife.*` is an entirely unbuilt subsystem, a real "good reason," not an oversight).
  `unit_registry.md`'s Constraints section is split into the original 5-row
  "stellarator-specific" table plus a new ~75-row "general" table.
- **All 16 `i_figure_merit` objective branches are ported**
  (`functional_process/core/solver/objectives.py`/`.md`/`test_objectives.py`) — a
  device-agnostic, no-`istell`-anywhere function; per `§6`'s own reasoning this needed
  no node, just the pure functions plus a lookup.
- **`functional_process/core/solver/drivers.py`: `PicardDriver`** — a generic
  `AbstractDriver` answering `cottax.problem.FixedPoint`, built locally in this repo
  (not `~/jaxgraph`) because unlike the `Feasibility`/`to_graph` gaps below, a Picard
  driver is exactly the kind of swappable solver choice `AbstractDriver` exists to let
  a *caller* supply, not core graph machinery. Mirrors `cottax.drivers.NewtonDriver`'s
  contract exactly (needs a `start`, same error shape). `AbstractDriver`'s own
  docstring already named this pairing ("a Newton drives `RootFind`, a Picard
  `FixedPoint`") — nothing in `cottax` had implemented the second half until now.
- **`functional_process/mda.py`** — turns `total_process.GRAPH` (or any
  `graph_for(configuration)`) into something actually *runnable*. Two raw cross-node
  cycles had no declared problem at all (`Divertor`/`AFwTotalWithPowerflow`, gated on
  `ipowerflow != 0`; `DensityProfile`/`FusionRates`/`PedestalOnAxisDensities`/
  `PlasmaComposition`, ungated) — `Drive` refuses to run a block with zero declared
  problems. `mda.CUTS` names the one variable per cycle that closes it (found via
  `Graph.closing_readers` + an empirical single-cut acyclicity check, not guessed):
  `.physics.proton_rate_density` and `.fwbs.f_ster_div_single`. `driven_graph()`
  applies `cottax.rewrites.FixedPointCut` to both (skipping the second cleanly when a
  given graph's configuration doesn't have that cycle at all — `ipowerflow == 0`).
  `default_drivers()` then assigns `NewtonDriver`/`PicardDriver` to every SCC
  automatically, by problem type. **`schedule()` now builds a runnable `Schedule` for
  the entire graph** — every one of the (now) 11 SCCs (8 structural self-loops +
  `Intersect`'s `RootFind` + the 2 cuts) is driven. First time the whole registered
  graph has been executable end to end, not just individually-tested pieces.
  `functional_process/test_mda.py` pins the structural claims.
- **A real cottax core-library fix, upstream in `~/jaxgraph`**: `to_graph()`/
  `node_and_names` (both `interfaces/{flat,pytree}_namespace_module.py`) claimed in
  their own error message to accept a bare `NodeDefinition` but had no code path that
  did — discovered via `vacuum.py`'s `DuctFeasibility` (a bare `Feasibility` built
  directly from `problem.py`, not a `NodalDeclaration`, so it has no class-derived
  name). Fixed: both now accept a `{name: NodeDefinition}` mapping alongside the
  existing forms. Not `Feasibility`-specific — the same gap would hit a bare `RootFind`
  built directly instead of through `ImplicitFunction`. `~/jaxgraph`'s suite: 396
  passed, 2 skipped, unchanged. `vacuum.py`/`test_vacuum.py` updated to use the new
  form instead of a hand-built `Graph(path_map(...))`.
- **`functional_process/mda_harness.py`** — the actual point of all of the above:
  `converged_data(input_file)` runs PROCESS's own `SingleRun` in-process on
  `tests/regression/input_files/stellarator_helias.IN.DAT` to convergence and returns
  the live `DataStructure` (no MFile round-trip — `cottax.tools.pytree.get_at` reads
  `VarPath`s off it directly, including `SequenceKey`-indexed array fields, no
  workaround needed). `compare(graph, data)` seeds `mda`'s `Schedule` from that same
  run's own values (boundary inputs *and* every driven block's starting guess — we are
  checking whether the graph reproduces an answer PROCESS already found, not solving
  cold) and diffs every value the schedule produces against `data`'s own value at the
  same field. `functional_process/run_mda_harness.py` is the runnable entry point.
  **`total_process.GRAPH`'s bare default configuration did not match this file**
  (`i_plasma_pedestal`: default `1`, this run `0`) — `graph_for(Configuration({".physics.
  i_plasma_pedestal": 0}))` is what the harness actually checks.
- **`functional_process/mda_constraint_harness.py`** — the same idea applied to
  constraints/objectives instead of nodes: every ported `constraint_N`/
  `objective_metric_N` called with this same converged run's real field values,
  diffed against PROCESS's own `ConstraintManager`/`objective_function` evaluation for
  the same run. Checks something the existing per-unit `Tier1Contract` tests
  structurally cannot: whether a port function is right *given a real, internally
  self-consistent set of simultaneous field values*, not just hand-built samples that
  could accidentally combine values that would never co-occur.

### Current harness numbers (last verified this session, independently reproduced)

`run_mda_harness.py`: **227 agreements, 11 disagreements (0 in driven/cyclic blocks —
every `NewtonDriver`/`PicardDriver`-driven block now reproduces PROCESS exactly),
64 unverifiable, 3 ungrounded inputs, 13 errors.** `mda_constraint_harness.py`:
**66/66 evaluable constraints agree, 15/16 objectives agree**, every skip traced to a
legitimate cause (switch precondition, genuine `0/0` for a net-current-free
stellarator's PF-coil-adjacent fields, one harness parameter-resolution limitation on
constraint 76's array-indexed argument) — no port defect found in this layer at all.
Full `functional_process` suite: **3414 passed, 0 failed**. `total_process.GRAPH`:
**97 nodes**.

### Bugs found by the harness and fixed this session

- **`i_confinement_time`**: `total_process.py` hardcoded `34` (`ITER_IPB98Y2`, a
  *tokamak* H-mode scaling law — just PROCESS's own bare default, inherited
  uncritically) → corrected to `38` (`ISS04_STELLARATOR`, already fully ported,
  just never wired at registration). This alone fixed a catastrophic cascade
  (`DoubleAndTripleProduct.ntau` was `0.0` vs. PROCESS's `5.37e20`; several `inf`
  values downstream from dividing by the resulting ~0 confinement time).
- **`.physics.profile_x` / `ProfileGrid`'s `.physics.radius_plasma_profile_norm`**:
  two independently-minted, numerically-identical quantities from two different port
  units, never wired together. `fusion_reactions.py`'s `FusionRates` now reads the
  established name instead of minting a duplicate.
- **`i_thermal_electric_conversion`**: hardcoded `0` (`CCFE_HCPB_VALUE`) across 4 node
  registrations (`ComponentThermalPowers`, `DeltaEtaStep`, `EtaTurbineStep`,
  `TempTurbineCoolantInStep`) vs. this run's real `2` (`USER_INPUT`) — checked first
  that `USER_INPUT` needs no additional wiring (it is an identity pass-through, same
  input set as every other branch) before flipping the default, same discipline as
  `i_confinement_time`. This is also what had made `EtaTurbineStep`/`DeltaEtaStep`
  (the first two real `PicardDriver`-driven blocks with real physics) look like they
  might be driver-convergence problems — they were not: feeding the corrected switch
  value reproduces PROCESS to full floating-point precision in one deterministic
  Picard step, `PicardDriver` fully vindicated.
- **`Build.z_tf_inside_half` — a genuine dual-ownership conflict *in PROCESS itself*,
  not a formula bug.** Two independent real PROCESS producers exist (`st_build`'s
  formula in `build.py`, `st_coil`'s in `coils/calculate.py`), and which one survives
  into the converged `DataStructure` depends on call order: `stellarator.py`'s
  `run()` calls them in *opposite* order depending on the `output` flag, and every
  real run ends with an `output=True` report pass where `st_coil` overwrites last.
  This port's `Build` node had correctly ported `st_build`'s formula — just not the
  one whose value survives. Fixed by moving ownership: `st_coil`'s formula extracted
  into a new node, `ZTfInsideHalf` (`coils/calculate.py`), which now owns
  `.build.z_tf_inside_half`; `Build` no longer declares that `Output`. (The fix's own
  first draft had a real bug — a 1-tuple return where `cottax`'s single-`Output`
  binding convention wants the bare value — caught only by the end-to-end harness,
  not by the new unit's own `Tier1Contract`/assembly tests, which never actually run
  the node through `_run_acyclic`. Worth remembering as a general fact about this
  harness's coverage: it catches wiring-binding bugs unit tests structurally can't.)
- **Harness-side fix, not a port bug**: `DetailedPowerflowBlanketShieldPower.
  f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard` are the node's own documented
  "best-effort" outputs — PROCESS never actually writes them to `data` in this arm
  (confirmed in the class's own docstring), so `expected=0.0` was
  `DataStructure()`'s bare uninitialised default, not a real PROCESS answer.
  `mda_harness.py`'s `KNOWN_UNVERIFIABLE_OUTPUTS` now excludes exactly these two
  `VarPath`s from comparison (the node's other 14 real outputs stay in scope).

### Still open, precisely diagnosed, not yet fixed

- **`ConfinementTime`/`DoubleAndTripleProduct`**: **CLOSED — fixed in § 8.3.** The
  diagnosis below stood up exactly; the fix turned out not to be a one-liner (the
  binding is device-dependent and `Input`s are class-level, so it needed a subclass
  under a new `.stellarator.istell` `Switch`). It is
  neither an ISS04 formula discrepancy nor `i_plasma_ignited`: `ConfinementTime`'s
  `q95` port is bound to `.physics.q95` (`confinement_time.py:2027`) while PROCESS's
  stellarator call site passes `.stellarator.iotabar` into that same positional slot
  (`process/models/stellarator/stellarator.py:2312`). This run has `q95 = 1.03`,
  `iotabar = 1.0`, and ISS04 goes as `iotabar**0.41`
  (`process/models/physics/confinement_time.py:3379-3386`) — `1.03**0.41 =
  1.0121928428817748`, which *is* the reported `rel_diff` to every digit. Rebinding and
  re-running the ported function on the converged run's own field values reproduces all
  nine `ConfinementTime` outputs to 12 significant figures.
- **`AuxiliaryPhysicsQuantities.fusrat`**: **closed — reclassified, not a bug** (§ 8.2).
  `phyaux`'s stellarator caller unpacks this one return value into a bare local
  `_fusrat` (`process/models/stellarator/stellarator.py:2383`), so `expected=0.0` was
  `physics_variables.py:1730`'s uninitialised default, not a PROCESS answer. Now in
  `mda_harness.KNOWN_UNVERIFIABLE_OUTPUTS`, same treatment as
  `.fwbs.f_a_fw_coolant_inboard`/`_outboard`.
- **`VacuumOld`**: **explained, not benign-floating-point — see § 8.2.** It is a
  deliberate, already-documented solver-tolerance difference (this port solves the duct
  diameter to `tol=1e-10`, PROCESS stops at a 1% relative step,
  `process/models/vacuum.py:469-477`), and the two off fields are one cause, not two.
  Recorded in `mda_harness.EXPLAINED_DISAGREEMENTS`; deliberately *not* suppressed.
- **`i_p_coolant_pumping`**: **closed** — corrected from `2` to `1` at all four
  registration sites (§ 8.2), and now checked automatically rather than by luck.
- **The 3 "ungrounded inputs" and 13 "errors"** the harness reports: **now triaged
  individually — see § 8.1 below, which supersedes this bullet.** The guess recorded
  here ("mostly structural coverage gaps ... rather than bugs") was right in aggregate
  (15 of 16 are correct mints) but wrong in one specific and findable place:
  `.tfcoil.den_tf_sc_material` had a real `DataStructure` field behind it all along,
  `.tfcoil.dcond[i_tf_sc_mat - 1]`, and is now fixed at the source. Three others had a
  plausible-looking real field one namespace away that would have compared against
  `0.0` — recorded in § 8.1 so they are not "fixed" wrongly later.
- **The 2 minted islands excluded from the harness entirely**
  (`DuctDiameterRootFind`; the `Intersect`/`WindingPackIntersectInputs`/
  `WindingPackTotalSizePost` SCC) — confirmed correct exclusions, not gaps to close:
  every `VarPath` either touches is minted, PROCESS never stores them, there is
  nothing to compare against. See `mda_harness.py`'s own docstring and
  `EXCLUDED_NODE_NAMES`'s comment for the full account.

### The validation-chain question for `Intersect`/`DuctDiameterRootFind`/`DuctFeasibility`

Investigated directly (not assumed) this session: both `Intersect` and
`DuctDiameterRootFind` are validated via `Tier2Contract`, not `Tier1Contract` — **no
direct value-agreement check against PROCESS's own reported number exists, by
construction**, because PROCESS's own algorithm (`intersect`'s fixed 100-iteration
cap; `_newton_method_duct_diameter`'s loose 0.01 relative-step tolerance) stops before
the true root, so PROCESS's own answer is not ground truth for either. What is
actually checked: the port's answer, plugged back into the real defining equation
(`intersect_residual`/`duct_diameter_residual`), is small in an absolute sense and no
worse than PROCESS's own residual — and the *declared* node reproduces the *eager*
port function's own answer, including at least one real PROCESS-derived legacy sample
point each. **The accurate claim is "reproduces PROCESS's own formula, solved more
tightly than PROCESS's own loose iteration" — not "matches PROCESS's own reported
number."** Worth stating this distinction explicitly anywhere this validation gets
cited, since it does different work than the whole-graph harness's node comparisons.
`DuctFeasibility` has no PROCESS equivalent at all — `solve_duct_geometry` is one
specific heuristic (shrink by 10% until it fits), not "any feasible point" — and
remains unvalidated by design pending an `Optimise`+real-objective wrapper and a
constrained-optimization driver, neither built (see the earlier-session discussion
this file doesn't repeat here).

### Recommended next steps, in order

1. ~~**`i_p_coolant_pumping`**: apply the same fix pattern as `i_thermal_electric_
   conversion` — check what the real value needs, flip the default if safe.~~
   **Done — § 8.2**, together with a third instance of the same defect
   (`i_plasma_ignited`) and a systemic check that ends the class.
2. ~~**`AuxiliaryPhysicsQuantities.fusrat`** and **`ConfinementTime`'s residual 1.2%**:
   both look like real, tractable bugs now that the catastrophic masking cause is
   gone — worth dedicated investigation, likely small individually.~~ **Both
   diagnosed — § 8.2.** `fusrat` was not a bug (reclassified). `ConfinementTime`'s 1.2%
   *is* a real bug with an exactly-confirmed cause, but the fix is one line in
   `models/physics/confinement_time.py:2027` (bind `q95` to `.stellarator.iotabar`),
   outside that pass's editable-file boundary. **Successor item, and the highest-value
   single line left in this section: make that rebinding.** Note it is a device-mode
   (`istell`) question, so decide deliberately whether it belongs in the node
   (a stellarator-mode `ConfinementTime`) or in `total_process.py` (which would need a
   rebind `GraphOp` that `cottax` does not have — `Cut` mints a *new* name and cannot
   point a port at an existing variable, checked directly in `rewrites.py:38-77`).
3. ~~**Triage the 13 "errors"/3 "ungrounded inputs"** one at a time, `constraints.md`-
   style, to separate "genuinely unmodeled" from "should have a real `VarPath` but
   doesn't yet."~~ **Done — § 8.1.** Successor item: the two `KNOWN_MINT_VALUES`
   follow-ups § 8.1 ends with (both `mda_harness.py` edits), plus the judgement call on
   whether to ground rows 1/2 and thereby stop excluding the winding-pack SCC.
4. **Re-run `run_mda_harness.py`/`mda_constraint_harness.py` after each fix** — both
   are cheap (~2 min) and have already caught real bugs their own authoring tests
   missed (see `Build`/`ZTfInsideHalf`'s 1-tuple bug above); treat them as a
   standing regression check for this whole area, not a one-off.
5. Longer-term, not started: building a real `Optimise`/constrained-optimization
   driver so `DuctFeasibility` (and, eventually, a real `Optimise` problem wired from
   the now-fully-ported constraints/objective layer) can be checked against something,
   and wiring constraints/objective into an actual `Optimise` `DeclaredNode` in
   `total_process.py` at all (still not done — both layers are fully ported and
   independently validated, but nothing assembles them into a solvable problem yet).
   **Partly superseded by § 9**: the specific gap `optimise_design.md` named as *its*
   blocker — "this run's own objective is not computable", i.e. `.costs.coe` had no
   producer anywhere in the ported graph — is closed. `i_figure_merit = 6`'s objective
   now evaluates from the graph and agrees with PROCESS's converged run to
   `rel_diff = 1.704e-06`. What remains of this item is the driver and the assembly,
   not the objective.

### 8.1 Triage of the 3 "ungrounded inputs" and 13 "errors" — done, one at a time

This closes recommended-next-step 3 above ("Triage the 13 errors/3 ungrounded inputs one
at a time, `constraints.md`-style, to separate 'genuinely unmodeled' from 'should have a
real `VarPath` but doesn't yet'"). All 16 were checked **individually against
`process/`**, not batched — including the six `.physics.*profile*` entries, which do turn
out to share one root cause but were verified separately rather than assumed to.

Recall what the two columns mean in `mda_harness.compare`: an **ungrounded input** is an
unowned (boundary) `VarPath`, or a driven block's unknown, that `_ground_truth` cannot
resolve to a `DataStructure` field (`mda_harness.py`, the `driven.unowned_inputs` loop);
an **error** is an *owned output* whose `VarPath` has no field either (the
`for var, owner in driven.owners.items()` loop). Same underlying condition, opposite side
of the edge.

Classification key: **(a)** genuine mint, no PROCESS counterpart, correct as-is;
**(b)** duplicate of an existing real field under a different minted name — a bug;
**(c)** should map to a real field, wrong/renamed spelling — a bug; **(d)** genuinely
unmodeled / out of scope.

| # | `VarPath` | column | class | evidence (`file:line`) |
|---|---|---|---|---|
| 1 | `.tfcoil.a_tf_wp_with_insulation` | ungrounded | **(a)** | Python local in `winding_pack_total_size`, `process/models/stellarator/coils/calculate.py:496-499`; the source's own comment on `:496` says "(not global)". Produced in this port by `WindingPackTotalSizePost` (`functional_process/models/stellarator/coils/calculate.py:1136-1137`) — it reads as *ungrounded* only because `mda_harness.EXCLUDED_NODE_NAMES` deletes that node's whole SCC. **Near-miss:** a field of exactly this name exists at `.superconducting_tfcoil.*` (`process/data_structure/superconducting_tf_coil_variables.py:35`) but is written only by the tokamak resistive TF model (`process/models/tfcoil/resistive.py:310`), never in a stellarator run — rebinding there would compare against `DataStructure()`'s bare `0.0`. |
| 2 | `.tfcoil.a_tf_wp_no_insulation` | ungrounded | **(a)** | Same, `process/models/stellarator/coils/calculate.py:500`; same `.superconducting_tfcoil` near-miss (`superconducting_tf_coil_variables.py:40`, written at `process/models/tfcoil/resistive.py:334`). |
| 3 | `.tfcoil.den_tf_sc_material` | ungrounded | **(c) — FIXED** | No such name anywhere in `process/` (grepped). The real read is `data.tfcoil.dcond[data.tfcoil.i_tf_sc_mat - 1]` at `process/models/stellarator/coils/mass.py:88`; `dcond` is a real nine-entry field (`process/data_structure/tfcoil_variables.py:157-170`). `i_tf_sc_mat` is 1 both by PROCESS default (`tfcoil_variables.py:246`) and in this run's input (`tests/regression/input_files/stellarator_helias.IN.DAT:235`), so the element is `dcond[0] == 6080.0`. **Fixed at the source:** `functional_process/models/stellarator/coils/mass.py`'s `CoilsMass` now reads `.tfcoil.dcond[0]` — an array-element `VarPath` exactly as `naming_convention.md` § "Array elements" prescribes and as `physics/radiation_power.py:619-660` already binds `.impurity_radiation.f_nd_impurity_electron_array[0..13]`. |
| 4 | `.first_wall.a_fw_total_unadjusted` | error | **(a)** | Not even a local: `st_build` assigns the unadjusted area *to the real field* `data.first_wall.a_fw_total` (`process/models/stellarator/build.py:166-168`) and then overwrites the same field in place in both `ipowerflow` arms (`build.py:170-181`). Nothing holds the unadjusted number at the end of the call. **Near-miss:** `process/data_structure/first_wall_variables.py:10` declares `a_fw_total_full_coverage`, documented as "First wall total surface area with no holes or ports" — but nothing in `process/` ever assigns it (only reads, all in the tokamak `process/models/fw.py:65,78,223,229,277,283`), so it stays `0.0` here. |
| 5 | `.physics.radius_plasma_profile_norm` | error | **(a)** | `Profile.profile_x`, an instance attribute created in `Profile.run` (`process/models/physics/profiles.py:61`) and normalised at `:84`. No field of this name in `process/data_structure/physics_variables.py`. This is the *established* name — it is what `.physics.profile_x` was corrected **to** in the earlier fix `mda_harness.KNOWN_MINT_VALUES`' docstring records; it is not itself a duplicate. |
| 6 | `.physics.dradius_plasma_profile_norm` | error | **(a)** | `Profile.profile_dx`, set in `Profile.calculate_profile_dx` (`process/models/physics/profiles.py:93`). No field. Also has no traced consumer (`scipy` ignores `dx=` whenever `x=` is given) — declared only because the source computes it. |
| 7 | `.physics.nd_plasma_electron_profile` | error | **(a)** | `neprofile.profile_y`, created in `Profile.run` (`process/models/physics/profiles.py:64`), filled by `NeProfile.calculate_profile_y` (`:192-210`). No field (`physics_variables.py` has `nd_plasma_electron_line`/`_on_axis`/`_max_array`, none of them a rho-profile). |
| 8 | `.physics.temp_plasma_electron_profile_kev` | error | **(a)** | `teprofile.profile_y`, same mechanism, filled by `TeProfile.calculate_profile_y`. No field. |
| 9 | `.physics.nd_plasma_electron_profile_integral` | error | **(a)** | `neprofile.profile_integ`, set in `Profile.integrate_profile_y` (`process/models/physics/profiles.py:110`). **Checked hardest of the six, because it looks like a (b):** PROCESS *does* store this value at the real field `.physics.nd_plasma_electron_line` — but only in the **pedestal** arm (`process/models/physics/plasma_profiles.py:234`). In the **parabolic** arm, which is what this harness's run uses (`i_plasma_pedestal == 0`), that same field is instead the closed-form gamma expression (`plasma_profiles.py:136-142`), computed without touching `profile_integ`, while `neprofile.run()` still computes and discards it. Collapsing the mint onto `nd_plasma_electron_line` would put two producers on one field. The pedestal-arm identity is already modelled correctly as a pass-through (`functional_process/models/physics/plasma_profiles.py:208-209,270-271`). |
| 10 | `.physics.temp_plasma_electron_profile_integral_kev` | error | **(a)** | `teprofile.profile_integ`, exact sibling of row 9: stored as `.physics.temp_plasma_electron_line_avg_kev` in the pedestal arm (`plasma_profiles.py:236-238`), closed-form in the parabolic arm (`plasma_profiles.py:144-150`). |
| 11 | `.neoclassics.chi_process_e` | error | **(a)** | `chi_PROCESS_e`, assigned at `process/models/stellarator/neoclassics.py:396`, returned as the 22nd tuple element (`:426`), unpacked into a local at `process/models/stellarator/stellarator.py:2426`, and from there reaching only `st_phys_output` (`:2439`) which prints it (`:2512-2513`). `process/data_structure/neoclassics_variables.py` has no `chi_*` field at all. |
| 12 | `.impurity_radiation.pden_impurity_rad_total_mw` | error | **(a)** | Instance attribute of `ImpurityRadiation`, initialised at `process/models/physics/impurity_radiation.py:667`, assigned at `:737`, read back off the instance at `process/models/physics/radiation_power.py:107,132`. No `pden_impurity_*` field in `process/data_structure/impurity_radiation_variables.py`. (It *is* reconstructible: `radiation_power.py:132` gives `pden_plasma_rad_mw = this + pden_plasma_sync_mw`, both real and written unclipped at `process/models/stellarator/stellarator.py:2147,2151`.) |
| 13 | `.impurity_radiation.pden_impurity_core_rad_total_mw` | error | **(a)** | Same, `impurity_radiation.py:668` / `radiation_power.py:107,128`. **Not** reconstructible the way row 12 is: `process/models/stellarator/stellarator.py:2153-2155` clips `.physics.pden_plasma_core_rad_mw` at 0 after writing it. |
| 14 | `.stellarator.coilcurrent` | error | **(a)** | Local at `process/models/stellarator/coils/calculate.py:46`, returned bare from `calculate_current` (`:378`); no field in `process/data_structure/stellarator_variables.py`. Exactly recoverable from a real field though — `calculate.py:276` writes `data.tfcoil.c_tf_total = n_tf_coils * coilcurrent * 1e6`, and this port's `coils/quench.py:201` already inverts it. |
| 15 | `.stellarator.dlimit_ecrh` | error | **(a)** | `st_d_limit_ecrh` returns it (`process/models/stellarator/density_limits.py:152`); both callers keep it local — `st_density_limits` as `ne0_max_ECRH` (`:40`), `min`-clamped (`:46`) and passed to `output()` (`:50`); `power_at_ignition_point` as `ne0_max` (`:191`), used only on a deep-copied proxy `DataStructure` (`:185,198-206`) that is discarded. |
| 16 | `.stellarator.bt_max_ecrh` | error | **(a)** | Same call sites, as `bt_ecrh` (`density_limits.py:40,47,50`) / `bt_ecrh_max` (`:191,204-206`). |

**Result: 15 × (a), 1 × (c), 0 × (b), 0 × (d).** The single real bug is row 3. Every other
one of the sixteen is a correct mint that the harness structurally cannot check — which is
the answer §8 asked for, and it is a better answer than "mostly structural coverage gaps"
because three of them (rows 1/2/4) had a plausible-looking real field one namespace away
that a less careful pass would have bound to and got `0.0` from.

Per-unit records updated to match the code and to carry this evidence:
`models/stellarator/coils/mass.md` (the fix, plus its open question 1 marked resolved),
`models/stellarator/coils/calculate.md` (rows 1/2/14),
`models/stellarator/build.md` (row 4), `models/physics/profiles.md` (rows 5-10),
`models/stellarator/neoclassics.md` (row 11),
`models/physics/radiation_power.md` (rows 12/13),
`models/stellarator/density_limits.md` (rows 15/16).

#### Measured effect of the row-3 fix

Measured by toggling **only** `models/stellarator/coils/mass.py` against an otherwise
identical tree, because `mda_harness.py`/`total_process.py` were being edited concurrently
by another agent during this session and a plain before/after would have attributed their
changes to this one:

| | control (`mass.py` at `HEAD`) | with the fix |
|---|---|---|
| agreements | 227 | 227 |
| disagreements | 10 | 10 |
| unverifiable | 65 | 65 |
| **ungrounded inputs** | **3** | **2** |
| errors | 13 | 13 |

Exactly one column moves and nothing regresses. Note honestly what the fix does *not* buy:
`CoilsMass`'s eight outputs stay `unverifiable`, because the same node still reads rows 1
and 2, which are genuine mints. What it does buy is fidelity — before the fix
`den_tf_sc_material` was seeded with `compare`'s `0.0` placeholder, so
`m_tf_coil_superconductor` was identically zero in every schedule run; it now carries the
real 6080 kg/m³. `$PY -m pytest functional_process -q`: 3414 passed, 2783 skipped, unchanged.

#### Two harness-side follow-ups, deliberately *not* made here

`mda_harness.py` is owned by another agent this session, so these are recorded rather than
applied. Both are `KNOWN_MINT_VALUES` entries — exactly the mechanism that dict's docstring
says it is kept (as an empty dict) for. Each turns one row above from an unscored mint into
a real comparison:

1. `".stellarator.coilcurrent": lambda d: d.tfcoil.c_tf_total / (d.tfcoil.n_tf_coils * 1e6)`
   — the inverse of `process/models/stellarator/coils/calculate.py:276`, already
   implemented in this port at `models/stellarator/coils/quench.py:201`. Moves row 14 out of
   `errors` and scores `CoilCurrent`.
2. `".impurity_radiation.pden_impurity_rad_total_mw": lambda d: d.physics.pden_plasma_rad_mw - d.physics.pden_plasma_sync_mw`
   — the inverse of `process/models/physics/radiation_power.py:132`, valid because both
   fields are written unclipped (`process/models/stellarator/stellarator.py:2147,2151`).
   Moves row 12 out of `errors`. **Do not** add the `core` sibling (row 13): its field is
   clipped at 0 (`stellarator.py:2153-2155`).

Rows 1 and 2 are *also* reconstructible —
`a_tf_wp_no_insulation == .tfcoil.dx_tf_wp_primary_toroidal * .tfcoil.dr_tf_wp_with_insulation`
and
`a_tf_wp_with_insulation == (.tfcoil.dr_tf_wp_with_insulation + 2*.tfcoil.dx_tf_wp_insulation) * (.tfcoil.dx_tf_wp_primary_toroidal + 2*.tfcoil.dx_tf_wp_insulation)`,
both read off `process/models/stellarator/coils/calculate.py:483-501`, all three inputs real
fields. Grounding those two would additionally unblock `CoilsMass`/`MaxForceDensity` and
their descendants out of the `unverifiable` column, which is where most of the 65 sit.
Flagged as the highest-value of the three, and as the one that most needs a second opinion
first, since it hands real seed values to an SCC the harness currently excludes wholesale.

### 8.2 The systemic static-switch audit — the whole defect class, closed at once

**Concurrency note**: this pass ran alongside § 8.1's, which was editing
`functional_process/models/**` while this one held only `total_process.py` and
`mda_harness.py`. The numbers below were measured before § 8.1's
`.tfcoil.den_tf_sc_material` grounding landed, so the ungrounded/unverifiable/error
counts here are this pass's own before/after, not the merged state. A final run made
once § 8.1's change had landed reads **227 agreements, 10 disagreements (0 driven), 65
unverifiable, 2 ungrounded inputs, 13 errors, 34 switch kwargs checked / 0 mismatched** --
i.e. identical to this pass's "after" column except for `ungrounded inputs` 3 -> 2, which
is § 8.1's `.tfcoil.den_tf_sc_material` grounding, not anything here. The switch-audit and
disagreement figures are unaffected by that change.

#### The check

Four bugs found in this project so far are one defect: a `total_process.py` registration
carrying a hardcoded static switch kwarg copied from the corresponding
`process/data_structure/*_variables.py` bare Python default rather than from the run
being modelled (`i_confinement_time`, `i_thermal_electric_conversion`,
`i_p_coolant_pumping`, `i_plasma_ignited`). Nothing checked any of them; each was found
only when a downstream value happened to diverge loudly enough to notice, so a wrong
switch that moved no compared output would never have been found at all.

`mda_harness.switch_audit(graph, data)` now checks every one of them on every harness
run, and `ComparisonReport.summary()` reports the counts alongside the existing ones.
**By introspection, not source parsing**: the kwargs are `eqx.field(static=True)`
attributes on the declaration instances the assembled graph actually holds (reachable as
`CallableNode.fn.__self__`; the walk is generic so `FixedPointFunction`/problem nodes are
covered too), so what is checked is what the graph carries, not what `total_process.py`'s
text says. The kwarg name is resolved to a `DataStructure` field by scanning every area
for an attribute of exactly that name — which works because PROCESS's own naming scheme
makes field names globally unique, and is *checked* rather than assumed (a name found in
two areas is reported unresolved, never silently resolved to whichever area came first).

Three-way classification, the same discipline the harness already applies to
ungrounded/unverifiable/errors:

- **checked** — resolved to a real field and compared. 34 of the 35 static kwargs on the
  default stellarator graph.
- **not data-backed** — declared in `STATIC_KWARGS_WITHOUT_BACKING_FIELD` with a reason.
  Exactly one: `ImpurityRadiationTotals.imp_indices`.
- **unresolved** — neither. Reported, never silently dropped. Currently zero.

One alias was needed and is declared explicitly in `STATIC_KWARG_ALIASES`:
`PlasmaComposition.is_ignited` is a `bool` standing for `i_plasma_ignited == IGNITED`
(`physics_B_composition.py:134-136`, `physics_variables.py:45-49`), which name-based
resolution cannot recover. That is the only case; the other 33 resolve by name alone.

A useful property, confirmed directly: run against a bare `DataStructure()` (PROCESS's
own defaults) instead of a converged run, the audit flags exactly the two *deliberate*
deviations already documented in `total_process.py` (`i_confinement_time = 38`,
`i_thermal_electric_conversion = 2`) plus the requested `i_plasma_pedestal` — i.e. it
reads as a diff against PROCESS's defaults, which is precisely the axis the bug class
lives on.

#### What it found, and what was done

7 mismatches across 2 switches, no others:

| node | kwarg | was | this run | source |
|---|---|---|---|---|
| `PlasmaComposition` | `is_ignited` | `False` | `True` | `stellarator_helias.IN.DAT:126` |
| `ConfinementTime` | `i_plasma_ignited` | `0` | `1` | same |
| `HeatingAndRadiationPower` | `i_plasma_ignited` | `0` | `1` | same |
| `ComponentThermalPowers` | `i_p_coolant_pumping` | `2` | `1` | `stellarator_helias.IN.DAT:198` |
| `DeltaEtaStep` | `i_p_coolant_pumping` | `2` | `1` | same |
| `PFwDivHeatDepositedMwStep` | `i_p_coolant_pumping` | `2` | `1` | same |
| `PFwBlktCoolantPumpMwStep` | `i_p_coolant_pumping` | `2` | `1` | same |

(§ 8's earlier "5 sites" for `i_p_coolant_pumping` was one too many — there are four.)

All seven corrected. Each flip was checked first for a hole in the target arm, the same
discipline the `i_thermal_electric_conversion` fix used, and in every case the correct
arm reads a strict *subset* of the wrong arm's inputs:

- `physics_B_composition.py:219-222` — IGNITED sets `nd_beam_ions = 0`; NON_IGNITED needs
  `nd_plasma_electrons_vol_avg * f_nd_beam_electron`.
- `confinement_time.py:1333-1334` and `stellarator_B_st_phys.py:273-274` — NON_IGNITED
  *adds* `p_hcd_injected_total_mw`; IGNITED omits the term.
- `power_B_thermal_cryo.py:206-211` and `:308-310` — both are conditional-ownership
  pass-throughs, and `FRACTION_OF_HEAT` selects the *recompute* side of both, out of
  operands every affected node already takes as `Input`s.

After the fixes: **0 mismatches, 34 checked, 1 not data-backed, 0 unresolved.**

#### The `i_plasma_ignited` hypothesis: correct as a registration bug, wrong as the cause

The strong prior going in was that `i_plasma_ignited` was the sole cause of the residual
1.219e-02 on `ConfinementTime`/`DoubleAndTripleProduct`, via
`confinement_time.py:1333-1334`. **It is not, and the fix changed those numbers by
exactly zero** — `t_energy_confinement` is bit-identical before and after. Reason,
measured on the converged run: `.current_drive.p_hcd_injected_total_mw == 0.0`, so the
term the two arms differ by is numerically inert in this run. The registration was
genuinely wrong and is genuinely fixed; it was simply never this symptom's cause. Worth
remembering as a general caution — "this switch is wrong *and* would produce a difference
of about this size" is two claims, and this project's harness can check the first
directly but not the second.

The real cause is § 8's `q95`/`iotabar` binding, above; see that bullet. It was found by
reading PROCESS's own stellarator call site rather than by reasoning about magnitudes, and
confirmed two independent ways: `1.03**0.41 == 1.0121928428817748` matches the reported
`rel_diff` digit for digit, and re-running the ported function with `q95` bound to
`iotabar` on the converged run's own field values reproduces **all nine** `ConfinementTime`
outputs to 12 significant figures. `confinement_time.py`'s own class docstring had already
flagged the binding as a device-mode question and predicted that "a stellarator-mode
instantiation of this class would rebind the `q95` input" — it is now measured, not
predicted.

#### Two reclassifications, both confirmed against `process/` directly

- **`.physics.fusrat`** → `mda_harness.KNOWN_UNVERIFIABLE_OUTPUTS`. `phyaux` returns seven
  values and the stellarator caller assigns six to `self.data.physics.*` fields but
  unpacks the third into a bare local `_fusrat`
  (`process/models/stellarator/stellarator.py:2383`); the tokamak caller
  (`process/models/physics/physics.py:961`) does store it. So the field is real but
  unwritten on this pipeline, and `expected=0.0` was `physics_variables.py:1730`'s
  declared default. Exactly the `.fwbs.f_a_fw_coolant_*` case.
- **`VacuumOld`** — § 8 called this "almost certainly benign floating-point path
  differences." It is better than that: it is a *deliberate, already-documented*
  algorithmic difference. This port solves the duct-diameter equation to a relative-step
  tolerance of `1e-10` (`functional_process/models/vacuum.py:250`, whose docstring at
  262-271 states and justifies the deviation); PROCESS stops the same Newton iteration at
  `dd <= 0.01` (`process/models/vacuum.py:469-477`), a bound ~34x *looser* than the
  observed difference — so PROCESS's number is not ground truth for this field, the same
  Tier2 framing this section already records for `Intersect`/`DuctDiameterRootFind`. The
  two off fields are one cause, not two: `dlscal ∝ d**1.4`
  (`process/models/vacuum.py:424`), and `1.4 × 2.941e-04 = 4.117e-04` against a measured
  `4.118e-04`. Recorded in `mda_harness.EXPLAINED_DISAGREEMENTS` — **documentation only,
  deliberately not wired into `compare()`**, since a per-field tolerance would mask a
  future real regression on the same field.

#### Report change: every disagreement, not just each node's worst

`summary()` kept the per-node "worst offenders" block and gained a full `all
disagreements` listing. This was not cosmetic: the per-node summary had hidden
`AuxiliaryPhysicsQuantities`'s second off variable
(`.physics.f_t_alpha_energy_confinement`) from a reader who had only the summary, and a
node's largest `rel_diff` is not reliably its most diagnostic one — here the *hidden* one
shared the `ConfinementTime` root cause while the *reported* one (`fusrat`) was not a bug
at all.

#### Numbers

| | before | after |
|---|---|---|
| agreements | 227 | 227 |
| disagreements | 11 | 10 |
| … in driven blocks | 0 | 0 |
| unverifiable | 64 | 65 |
| ungrounded inputs | 3 | 3 |
| errors | 13 | 13 |
| static switch kwargs checked / mismatched | — (no check existed) | 34 / **0** |
| `pytest functional_process -q` | 3414 passed, 2783 skipped | 3414 passed, 2783 skipped |

The only count that moved is `fusrat`'s reclassification (disagreements 11 → 10,
unverifiable 64 → 65). **The seven switch corrections moved no compared value** — and
that is the point of the check, not a disappointment: `i_plasma_ignited`'s term is inert
at `p_hcd_injected_total_mw == 0`, and `i_p_coolant_pumping`'s two pass-throughs had been
sitting on their *identity* arm, so those nodes were agreeing with PROCESS only because
the harness seeds them from PROCESS's own converged value. On the corrected arm they
recompute from raw inputs and still agree — the same count, but a real check where there
had been a tautology. Four bugs of this class were found by luck before this run; the
class is now checked on every run.


### 8.3 The `q95`/`iotabar` binding bug — the residual 1.2%, closed

**Read this before §8's "Still open" list, which it supersedes in part.** §8 recorded a
residual ~1.2% disagreement on `t_energy_confinement`/`ntau` as "not yet traced to a
specific cause", with a wrong `i_plasma_ignited` registration as the leading hypothesis.
That hypothesis was wrong. `i_plasma_ignited` *was* mis-registered (§8.2 fixed it) but
changed the numbers by exactly zero: `.current_drive.p_hcd_injected_total_mw == 0.0` on
this run, so the only term the two arms differ by is inert.

**The real cause is a wrong input binding, not a formula.** PROCESS's
`calculate_confinement_time` names its 20th positional parameter `q95`
(`process/models/physics/confinement_time.py:79`) and its *tokamak* caller does pass
`.physics.q95` — but the stellarator caller passes `self.data.stellarator.iotabar` into
that same slot (`process/models/stellarator/stellarator.py:2312`), and
`iss04_stellarator_confinement_time` consumes it as `iotabar**0.41`. The port's
`ConfinementTime` bound `.physics.q95`, so the ISS04 law was fed a safety factor.

Confirmed arithmetically, not inferred: `q95 = 1.03`, `iotabar = 1.0`,
`1.03**0.41 = 1.0121928428817748` — the reported `rel_diff` of `1.219e-02` to every
digit, with `1.205e-02` on `.physics.f_t_alpha_energy_confinement` downstream (it scales
as `1/tau`).

**Fix**: `StellaratorConfinementTime` (`models/physics/confinement_time.py`), a subclass
rebinding exactly that one read, registered as the `value=6` arm of a new
`.stellarator.istell` `Switch` in `total_process.py`. Three points worth carrying:

- **A subclass is the unit of rebinding, structurally.** `Input` is a class-level
  `__call__` parameter default (`cottax/interfaces/pytree_namespace_module.py:113-126`),
  so no `eqx.field(static=True)` on an instance can vary a read. `NodalDeclaration.name`
  is `type(self).__name__` (`ibid.:190-194`), so the arm gets its own node name free.
- **The signature is derived, not restated.** `_rebound_signature` copies the base
  signature and replaces one default, so the arm cannot drift if `ConfinementTime` ever
  gains or rebinds a parameter — and a renamed parameter raises at import instead of
  silently rebinding nothing. Restating all 36 parameters would have reintroduced exactly
  the class of bug being fixed.
- **A `Switch`, not a hardcoded binding**, because changing which node produces a read
  changes an edge — `configuration.py`'s own criterion for a topology switch. `default=0`
  is PROCESS's own (`stellarator_variables.py:46`), so **every stellarator consumer must
  pass `Configuration({".stellarator.istell": 6})`**; `run_mda_harness.py` does. `istell`
  in 1..5 is declared `unported` (the `preset_config.py` machine-preset tables, still
  open in §2), so a wrong device config fails loudly instead of assembling silently.

**The general lesson**: a positional parameter named from one device's vocabulary is a
standing trap here. No `Tier1Contract` in `test_confinement_time.py` could ever have
caught this — every case passes `q95` positionally to the pure function, and the pure
function is correct. Only the *binding* was wrong, and bindings live in node
declarations, which per-unit tests do not exercise. This is the third distinct bug class
(after `ZTfInsideHalf`'s 1-tuple return and §8.2's static-switch defaults) that the
end-to-end harness catches and unit tests structurally cannot. Any other `calculate_*`
with device-dependent call sites deserves the same check.

### 8.4 Harness state after §8.1–8.3

| | §8 baseline | now |
|---|---|---|
| agreements | 227 | **237** |
| disagreements | 11 (0 driven) | **2** (0 driven) |
| unverifiable | 64 | 65 |
| ungrounded inputs | 3 | 2 |
| errors | 13 | 11 |
| static switch kwargs | *unchecked* | 34 checked, 0 mismatched |
| `pytest functional_process` | 3414 passed | 3418 passed |

**Both remaining disagreements are one documented, deliberate cause** — `VacuumOld`'s
`dlscal` (4.118e-04) and `dia_vv_vacuum_ducts` (2.941e-04), where this port solves the
duct diameter to `tol=1e-10` and PROCESS stops at a 1% relative step
(`models/vacuum.py:250` and its docstring; `process/models/vacuum.py:469-477`). One
cause, not two: `dlscal ∝ d^1.4`, and `1.4 × 2.941e-04 = 4.117e-04`. PROCESS's own
number is not ground truth here, so this is recorded in `EXPLAINED_DISAGREEMENTS` and
deliberately *not* wired into `compare()` — a future real regression on the same field
must still show.

The +10 agreements come from §8.1's two `KNOWN_MINT_VALUES` reconstructions
(`.stellarator.coilcurrent` from `calculate.py:276`;
`.impurity_radiation.pden_impurity_rad_total_mw` from `radiation_power.py:132`). The
`pden_impurity_core_rad_total_mw` sibling deliberately gets none — PROCESS clips
`.physics.pden_plasma_core_rad_mw` at zero after storing it
(`process/models/stellarator/stellarator.py:2153-2155`), so it is not recoverable back
through the sum.

## 9. `.costs.coe` has a producer — the `Optimise` layer's stated blocker, closed

**Read this after §8; it is the layer §8's harness was built to make checkable, applied
to the one field the objective needs.** `_audit/optimise_design.md` names exactly one
blocker for wiring a real `Optimise` problem: "this run's own objective is not
computable". `tests/regression/input_files/stellarator_helias.IN.DAT:229` sets
`i_figure_merit = 6` (`FiguresOfMerit.COST_OF_ELECTRICITY` ->
`objective_metric_6(coe) = coe / 100.0`, `core/solver/objectives.py:88-90`, ported since
§8), and until this pass nothing in the ported graph produced `.costs.coe`. Now
`CostOfElectricity` does.

### What was ported

Registry unit #18's `costs.py` went from 23/43 methods (leaf accounts, none registered)
to **41/43 methods, 44 pure functions, 44 nodes, 43 registered**. The two left out are
`run` (the call-order dispatcher a `Graph` replaces — its two *computations*,
`.costs.cdirt` and `.costs.concost`, are ported as their own nodes) and `output`
(reporting). `costs.md`'s new § "coverage map for `.costs.coe`" derives the chain and its
headline result, which is worth repeating because it decided the size of this pass:
**`.costs.coe` depends on every computational method in the file.** The accumulation is a
plain sum of all nine Account-22 sub-totals and all six top-level accounts, none of which
PROCESS ever skips, so "port only what `coe` needs" and "port everything except
`run`/`output`" are the same instruction here. That was checked (by walking
`costs.py:2990` backwards) before any code was written, not assumed.

Three of the first wave's own conclusions are **corrected**, each in place in `costs.md`:

- **`acc2222`'s "dynamic-length loop", recorded as "the one real structural JAX blocker
  in the whole file", is not one.** The bound is `.pf_coil.n_cs_pf_coils`, which is
  neither an iteration variable nor a scan variable (`grep` over
  `process/core/solver/iteration_variables.py` and `process/core/scan.py`: no match), so
  it is constant for a whole solve and belongs in `naming_convention.md`'s static-kwarg
  category — exactly `ImpurityRadiationTotals.imp_indices`'s case. Neither
  `lax.fori_loop` nor padding is needed; the loop unrolls at trace time. There is now no
  structural JAX blocker anywhere in `costs.py`. Worth generalising: "dynamic-length
  loop" is the wrong first question. The right one is *whether the length can change
  between two evaluations of one assembled graph*, which for PROCESS is answered by the
  iteration-variable and scan-variable lists, not by reading the loop.
- **`i_cost_model` "is not wireable as a `Switch`"** — it is; see below.
- **`unit_registry.md`'s framing that `costs.py`'s `Costs` model "never even runs by
  default", so its nodes must stay unregistered.** True of PROCESS's bare default
  (`cost_variables.py:327` -> `1`, `KOVARI_2014`), and irrelevant to the run this
  project validates against: `stellarator_helias.IN.DAT:248` sets `i_cost_model = 0`
  explicitly, with the file's own comment "0: 1990 cost module, the 2015 does not work
  yet for stellarators". Reasoning from the bare `*_variables.py` default rather than
  from the reference IN.DAT is the same habit §8.2's static-switch audit found four
  separate instances of; this is a fifth, at the level of *whether a node exists* rather
  than what value a kwarg takes. **No systemic check catches this one** — `switch_audit`
  checks the kwargs a registered node carries, not whether the right nodes are
  registered. That gap is real and is not closed by this pass.

### The registration decision — a new third `Alternative` state

`.costs.i_cost_model` is a textbook topology switch (resolved once in
`process/main.py`'s `Models.costs` `@property`, lines 745-764, which picks a whole
`Model` instance before any model runs; never read inside either cost file). The trap is
that **`GRAPH = graph_for()` is evaluated at import time from the bare default
configuration**, and the default arm (`1`, `KOVARI_2014`) has no ported nodes at all —
`costs_2015.py` still has zero. Declaring it `unported` would raise `NotImplementedError`
on `import functional_process.total_process`, breaking the package and every one of
`test_configuration.py`'s per-switch parametrised assemblies besides.

Three readings were weighed; the argument is kept in full in `total_process.py`'s own
comment block on the switch, and summarised here:

1. **Register the 43 nodes unconditionally in `COMMON`.** Rejected. This is *worse* than
   the `EcrhDensityLimit`/`WardTaylorAvailability` bug class this file already tracks,
   not an instance of it: those computed a value the default configuration never
   computes, whereas PROCESS's default *does* compute `.costs.coe` — by the 2015 model.
   An unconditional registration would put a **different number in the same field**.
2. **Change what `graph_for()`'s bare default means** — either `GRAPH =
   graph_for(REFERENCE_CONFIGURATION)`, or making `GRAPH` lazy so that asking for the
   bare default is allowed to fail. Structurally this is the most honest answer:
   PROCESS's bare-default configuration is *not* fully ported, and asking for it
   arguably should fail loudly. **Deliberately not taken here, and flagged as the user's
   call.** It changes a shared contract (`Switch.default`'s "a silent IN.DAT reproduces
   PROCESS's own defaults", pinned by
   `test_configuration.py::test_default_configuration_matches_process_defaults`) and
   would make `test_configuration.py`, `test_mda.py`, `mda.py`'s default argument and
   `render_xdsm.py` all carry a configuration. Note the tension it would resolve is
   already visible: `run_mda_harness.py` overrides three switches (`i_plasma_pedestal`,
   `istell`, and now `i_cost_model`) because the bare default matches no real run, and
   `total_process.py`'s own docstring already concedes the bare-default graph is
   "device-incoherent". A future pass that wants to take option 2 should do it as its
   own change, with that whole tension in view — not as a side effect of porting a cost
   model.
3. **An arm that assembles as empty** — taken. `Alternative` gains a third state,
   `unproduced`, alongside `declarations` and `unported`. It contributes no nodes and
   does not raise, so under `KOVARI_2014` `.costs.coe`/`.costs.concost` simply have no
   producer and any consumer surfaces as an unowned (boundary) input rather than being
   silently handed the 1990 model's formula. `check_arms_are_exclusive` skips it the same
   way it skips `unported` arms (a single-ported-arm switch has no pair to check), and
   `Alternative.__post_init__` now *requires* exactly one of the three to be given — an
   arm that declares nothing at all is rejected, so an empty arm can never be confused
   with an oversight.

**The measured cost of option 3 is zero**: the default `GRAPH` is still exactly 97 nodes,
byte-for-byte the graph it was before this arm existed, so nothing that depended on it
moved. The honest cost is conceptual, and worth stating rather than burying —
`unproduced` is a *weaker* guarantee than `unported`. It is only sound when the arm's
outputs have no other producer in the assembled graph, and nothing checks that for you,
because "no producer" is exactly what it is being told to produce.
`test_configuration.py` pins both halves of the claim on the real instance:
`.costs.coe` has an owner under `i_cost_model == 0` and none under `== 1`.

`i_cost_model == 2` (`USER_PROVIDED`) is `unported`, not `unproduced`, and the contrast
is the clearest statement of the distinction: it injects a user-supplied `Model` at
runtime (`process/main.py:766-768`), so there is no PROCESS-side subgraph to port at
all, and a caller asking for it has a model in mind this graph has never seen. Refusing
is right there; assembling empty would not be.

### Harness effect — measured as a controlled A/B, not against §8.4's numbers

`mda_harness.py` was being edited by a concurrently-running agent during this pass (the
`constraint_32_investigation.md` work, which took the winding-pack coil island out of
`EXCLUDED_NODE_NAMES` and added three `KNOWN_MINT_VALUES` entries). A plain before/after
against §8.4's column would have attributed that agent's +59 agreements to this one, so
the numbers below are **control vs. treatment on the same tree**, identical in every
respect except `.costs.i_cost_model` — the same discipline §8.1 used for its own
row-3 fix:

| | control (`i_cost_model = 1`, empty arm) | treatment (`i_cost_model = 0`) |
|---|---|---|
| nodes | 99 | **142** (+43) |
| agreements | 296 | **397** (+101) |
| disagreements | 2 | **14** (+12) |
| … in driven (cyclic) blocks | 0 | **0** |
| unverifiable | 3 | 3 |
| ungrounded inputs | 0 | 0 |
| errors | 21 | 21 |
| static switch kwargs checked / mismatched | 34 / 0 | **51 / 0** (+17) |

`pytest functional_process -q`: **3575 passed, 2909 skipped**, up from 3420/2783 at the
start of this pass, 0 failed. `--fp-gradients --fp-fuzz=3` on `test_costs.py`: 884
passed.

Three things are worth reading off that table rather than just the headline:

- **Nothing lands in `errors`, `ungrounded` or `unverifiable`.** All three columns are
  unchanged. This unit mints no `VarPath`s at all — every one of the 43 nodes' inputs
  and outputs is a real `DataStructure` field — which is why the whole chain scores
  instead of being structurally uncheckable, and it is the first wave of this size where
  that is true.
- **+17 static switch kwargs, 0 mismatched.** Every one of this wave's static values
  (`ife`, `itart`, `ireactor`, `ipnet`, `supercond_cost_model`, `n_cs_pf_coils`,
  `iohcl`, `i_pf_conductor`, `i_pulsed_plant`, `istore`) resolves to a real field and is
  now checked on every harness run. §8.2 built that check for a defect class found four
  times by luck; this is the first wave to be born under it. One of the ten is a
  deliberate departure from a PROCESS default — `iohcl = 0` against
  `build_variables.py:177`'s `1` — and it is exactly the kind that used to go unnoticed.
- **All 12 new disagreements are one already-documented cause, and it is not this
  unit's.** `VacuumOld` solves the vacuum-duct diameter to `tol=1e-10` where PROCESS
  stops at a 1% relative step (`models/vacuum.py:250`,
  `process/models/vacuum.py:469-477`, `mda_harness.EXPLAINED_DISAGREEMENTS`), and
  `.vacuum.dlscal`/`.dia_vv_vacuum_ducts` feed Account 224, which feeds `c22`, `cdirt`,
  `cindrt`/`ccont`, `concost`, and finally `coe`. **Demonstrated, not argued**: fed
  PROCESS's own converged values for those two fields, `calculate_vacuum_system_cost`
  reproduces all seven of PROCESS's Account-224 numbers at *exactly zero* relative
  difference, and every downstream absolute delta is that one `c224` delta pushed
  through the linear accumulation chain —

      delta c224 = delta c22 = delta cdirt = 1.226201741e-02
      delta cindrt = cfind[lsa-1]*(1+cowner)*delta cdirt   (predicted to 11 s.f.)
      delta ccont  = fcontng*(delta cdirt + delta cindrt)  (predicted to 11 s.f.)
      delta concost = delta cdirt + delta cindrt + delta ccont

  `.costs.coe` itself agrees with PROCESS's converged run to `rel_diff = 1.704e-06`
  (`121.49188552` against `121.49167845`), and that residual is entirely the above.
  Deliberately *not* suppressed, per §8.2's own reasoning: a per-field tolerance would
  mask a future real regression. This is also the first time the `VacuumOld` deviation
  has been shown to *propagate* rather than sit in two fields — worth knowing before
  anyone treats it as cosmetic, since it now reaches the objective.

### Findings, none fixed

- **A second `acc223` defect**, alongside the `c2233` one the first wave found: `fkind`
  (the Nth-of-a-kind multiplier) is applied **only** on the `ifueltyp == 1` branch, for
  all three sub-accounts (`process/models/costs/costs.py:1877-1881`, `1899-1903`,
  `1917-1921`) — PROCESS computes each `cNNNN` unconditionally and then applies both
  `(1 - fcdfuel)` *and* `fkind` inside the `if`. Every other account in the file applies
  `fkind` unconditionally, so Account 223 silently escapes it on any run with
  `ifueltyp != 1`. Reproduced as written.
- **The `c2233` defect is more precisely characterised than before, and the refinement
  matters.** The first wave recorded it as "the same history-dependence shape as
  `cdrlife_cal`". It is not. `ifueltyp` is a run-configuration constant, so "never
  assigned in *this* call" implies "never assigned in *any* call of the run", and the
  field can only ever hold `cost_variables.py:165`'s dataclass default `0.0` — making
  the port's `0.0` exact rather than an approximation. `cdrlife_cal`'s gate
  (`life_blkt_fpy < life_plant`) genuinely moves between VMCON iterations, which is what
  makes *that* one a real unrepresentable-state gap. Two superficially identical
  patterns, opposite conclusions, and the discriminator is the same
  iteration-variable/scan-variable question the `acc2222` correction turned on.
- **`.costs.c2234` is a dead field**: a term of `c223` (`costs.py:1976`) written in
  exactly one place in all of `process/` (`costs.py:1968`), to `0.0`, inside the IFE
  *and* `ifueltyp == 1` branch. Folded to the literal rather than declaring an `Output`
  nothing produces.
- **A JAX trap worth carrying forward, caught only by `test_gradient_finite`.** PROCESS
  clamps a negative net electric power to zero before a square root
  (`costs.py:2874-2888`). The obvious port, `jnp.sqrt(jnp.maximum(p, 0.0))`, is
  value-correct and returns `nan` from `jacfwd` on the clamped branch, because `sqrt`
  has an infinite derivative at zero. Fixed with the standard *double* `jnp.where`. The
  general shape — a `jnp.maximum` guard in front of a function with an unbounded
  derivative *at the guard point* — is not specific to this unit, and the value tests
  passed throughout.

### Left undone, with reasons

- **`costs_2015.py` is untouched**: still 2/13 functions ported, still zero nodes. Its
  `run()`-level accumulation (`costs_2015.py:52-102`, eight `calc_*` methods filling a
  100-slot `s_cost` array) is the single change that would turn `i_cost_model` into an
  ordinary two-armed `Switch` with no further machinery, since `.costs.coe`/
  `.costs.concost` are exactly the outputs the two arms share. Out of this pass's scope
  and not needed by any run in this project's stellarator scope.
- **`TfMagnetCostResistive` is written but not registered.** `acc2221`'s two arms share
  no body and read disjoint fields, so they are two nodes (the split default), not one
  node with a static `i_tf_sup` kwarg. Registering both is a duplicate-ownership
  conflict on `.costs.c22211`/`c22212`/`c2221`; pairing them as a `Switch` would require
  that switch to **nest** inside `.costs.i_cost_model`. That is a **third instance of the
  nested-switch gap** §1 tracks (after `irefprop` under `i_blkt_coolant_type` and
  `i_nd_plasma_pedestal_separatrix` under `i_plasma_pedestal`) — and unlike those two,
  **this one is on a reachable node**, which is the condition §1 itself set for taking
  the question seriously ("worth a real decision once a third shows up on a *reachable*
  node"). That condition is now met. `.tfcoil.i_tf_sup == 1` is both PROCESS's default
  and the reference run's, so nothing is currently wrong; what is missing is the ability
  to assemble a resistive-TF cost graph at all.
- **The IFE arms of six methods are not ported** (`ife` static, refused). `.ife.*` has no
  unit in `unit_registry.md` at all, and the 2-D `fwmatm`/`blmatm`/`shmatm` `VarPath`
  question the first wave flagged is deferred, not answered.
- **The split-default deviation count is now six, not three.** §1 records three instances
  of "reads-set genuinely differs, kept static anyway" and asks whether
  `traceability_policy.md` needs a size/entanglement-aware exception. This wave adds
  three more (`supercond_cost_model` on two nodes, `i_pf_conductor`, and `itart` on
  `CostOfElectricity`) — and, unusually, a controlled contrast *inside one file*:
  `acc2221` **was** split (two arms, no shared body, disjoint reads) while `coelc`'s
  `itart` was **not** (15 lines of a 290-line function, ~275 shared). Those two together
  are the clearest evidence this project has of what the missing rule would have to say,
  and they are in the same unit, ported the same day, by the same reasoning. §1's open
  question is now well-posed enough to answer.
- **No systemic check exists for "is the right *node* registered".** `switch_audit`
  checks the static kwargs a registered node carries against the run; nothing checks that
  the set of registered nodes matches the configuration the run describes. The
  `i_cost_model` case above is the fifth instance of a registration derived from a bare
  `*_variables.py` default rather than the modelled run, and the first where the
  consequence was 43 nodes missing rather than one kwarg wrong. A check in the shape of
  `switch_audit` — walk `TOPOLOGY_SWITCHES`, resolve each `path` against the converged
  `DataStructure`, and report where the assembled configuration disagrees — would be
  cheap and would have found this. Not built here (`mda_harness.py` was owned by another
  agent this session); recommended as the direct successor to §8.2.

## 10. The `Optimise` layer — built, and the ladder run end to end

**`_audit/optimise_design.md` §10 is the record; this section is the index and the parts
that belong to *this* file** (harness numbers, the punch list, and the defects found).
§8's recommended-next-step 5 ("building a real `Optimise`/constrained-optimization driver
… and wiring constraints/objective into an actual `Optimise` `DeclaredNode` … still not
done") is closed.

### What now exists

`core/solver/drivers.py` grew **`VmconDriver`** — `pyvmcon` (PROCESS's own SQP) fed one
`jax.jacfwd` of the block's `ConditionMap` instead of `Evaluators.fcnvmc2`'s
1 %-step finite differences. `sand.py` assembles PROCESS's actual problem onto the graph
as SAND, `sand_harness.py`/`run_sand_harness.py` run the three-stage ladder, and
`test_sand.py` adds 16 tests that need no PROCESS run. `mda.default_drivers` grew an
`Optimise` arm that reads the equality/inequality counts off the `Optimise` node itself.

### Harness numbers, controlled per change

| | start of session | +c16/c24 producers | +the x4 fix |
|---|---|---|---|
| `graph_for()` nodes | 142 | 143 | **146** |
| agreements | 397 | 397 | **404** |
| disagreements | 14 | 32 | **32** |
| … in driven (cyclic) blocks | **0** | **0** | **0** |
| unverifiable / ungrounded / errors | 3 / 0 / 21 | 3 / 0 / 21 | 3 / 0 / 21 |
| static switch kwargs / mismatched | 51 / 0 | 55 / 0 | **55 / 0** |
| `pytest functional_process` | 3577 passed | 3581 | **3597 passed, 0 failed** |

The 18 new disagreements are **one cause**, proven arithmetically rather than asserted —
see below.

### Defects found and fixed this session

- **Iteration variable 4 had no path into the physics.** `.physics.temp_plasma_ion_vol_avg_kev`,
  `.physics.temp_plasma_electron_density_weighted_kev` and
  `.physics.temp_plasma_ion_density_weighted_kev` were boundary inputs with no producer,
  so ion temperature was structurally disconnected from the electron temperature the
  optimiser varies. Every *value* was right; every derivative w.r.t. `x4` was ~0 where
  PROCESS's is O(1), and `FusionRates`' `sigmav_dt_average` (roughly `T²`) had a relative
  sensitivity of `2.4e-16`. Both pure functions were already ported — only the nodes were
  missing, deferred by `plasma_profiles.md`'s open question 1, which
  `total_process.py`'s own `i_plasma_pedestal` arm had already answered. Fixed by
  registering `IonVolAvgTemperature` and `ParabolicProfileValues`; +7 agreements, 0 new
  disagreements, and every affected Jacobian cell now agrees with PROCESS to between
  `4e-6` and `1e-3`.
  **This is the defect class §8's harness cannot reach, found by the thing built to reach
  it.**
- **`fast_alpha_beta`'s clamped square root returned `nan` from `jacfwd`.**
  `jnp.sqrt(jnp.maximum(0.0, x))` with the clamp **active** on this run
  (`temp_sum_20 = 0.6449` against `0.65`) — the same trap §9 records for
  `costs.py:2874-2888`, now a second instance. Fixed with the double `jnp.where`;
  value-identical, and it took the SAND Jacobian from 17 non-finite cells to 0. It had
  been latent because nothing downstream of `.physics.beta_fast_alpha` fed a condition
  until constraint 24 got a producer. **Generalise: a gradient defect only becomes
  visible once something downstream reads into a condition, so closing a producer gap
  routinely exposes one.**
- **Seven 1-tuple returns** (`power_B_thermal_cryo.py` ×6, `vacuum.py` ×1) — the
  `ZTfInsideHalf` bug class, third wave. Invisible to `PicardDriver`, fatal to
  `Residualise`. The full sweep is recorded in `power_B_thermal_cryo.md`.

### Two constraints un-blocked, and the registration gaps behind them

`optimise_design.md` §2.4 recorded constraints **16** and **24** as permanently INERT.
Both are now live, which matters because an `Optimise` over 12 of PROCESS's 14 active
constraints is a *different problem* and comparing its answer to PROCESS's would be
meaningless. `sand.constraint_nodes` now raises on any active `icc` entry it cannot
assemble rather than dropping it.

- `PlantElectricProductionReactor` (new, `power_C_electric_production.py`) — the
  `ireactor == 1` arm of `plant_electric_production`, whose five "self-referential"
  fields are dead reads on that arm. `.costs.ireactor` becomes a two-armed topology
  `Switch`. **This also gave `.costs.coe` — the run's own objective — a real dependence
  on the design along the net-electric-power path, where before it read a boundary
  input.**
- `StellaratorBetaAndStoredEnergy` (new, `stellarator_B_st_phys.py`) —
  `StellaratorBetaAndRhoStar` minus the one output (`rho_star`) that actually collided
  with `DimensionlessPlasmaParameters`. Dropping the whole node to resolve that collision
  had cost `.physics.beta_total_vol_avg` and `.physics.e_plasma_beta` their only producer
  as collateral.

**Both are further instances of §9's "no systemic check exists for *is the right node
registered*".** That gap is now four instances deep (`i_cost_model`'s 43 nodes,
`PlantElectricProduction`, `StellaratorBetaAndRhoStar`, and `plasma_profiles.py`'s two)
and every one of them was found by a *downstream consumer* noticing, never by a check.
The recommended shape is unchanged and now clearly worth building: walk the ported units'
node classes, and report every one that is written but registered nowhere, with the
reason recorded beside it.

### A finding about PROCESS itself, not the port

**PROCESS's converged `DataStructure` is internally inconsistent, and the port is the
self-consistent side.** Measured by instrumenting a real solve:
`Buildings.run(output=False)` leaves `.buildings.a_plant_floor_effective = 563075.16` and
`Buildings.run(output=True)` leaves `680433.44` — the *same method*, differing only via
`.build.z_tf_inside_half` (`4.1556` at solve time, `7.3592` in the report pass), which is
§8's `ZTfInsideHalf` dual write. `Stellarator.run(output=True)` then calls
`power.output_plant_electric_powers()` rather than `plant_electric_production()`, so
`.heat_transport.p_plant_electric_base_total_mw` is never recomputed and keeps its
solve-pass value while `a_plant_floor_effective` moves.

The resulting **`+17.604 MW`** offset accounts for *all eighteen* new harness
disagreements, exactly and linearly, through `Acpow` and the cost chain to `.costs.coe`
(`rel = 1.73e-2`). Recorded in `mda_harness.EXPLAINED_DISAGREEMENTS`, deliberately not
suppressed. It is also the largest identified reason the port's optimiser does not land
on PROCESS's `x`.

**Consequence for §8's framing:** the harness's `expected` column is PROCESS's *reported*
state, which for any field the report pass recomputes is not the state the solver used.
That is a third category alongside `KNOWN_UNVERIFIABLE_OUTPUTS` and
`EXPLAINED_DISAGREEMENTS`, and nothing currently detects it in general — a field is only
noticed when a consumer of it disagrees.

### Recommended next steps, in order

1. **`c62`'s Jacobian row.** `.physics.f_t_alpha_energy_confinement`'s value is exact and
   its derivative is wrong by ~5× with a sign flip on `x4`. The only cell in the whole
   Jacobian disagreeing for an unknown reason. Use the same topological
   sensitivity walk that found `x4` (`optimise_design.md` §10.5a).
2. **The cold-start gap.** The port cannot run its own pipeline from a cold input file at
   all — 20 of 24 SAND conditions are `nan` there, because the block reads 330 context
   variables and the cold `DataStructure` has `0.0` in every model-computed field. This
   blocks `optimise_design.md` §5.3's C3, the only check that would license "the port
   reproduces PROCESS's optimisation". It is a graph-completeness question, not an
   `Optimise` one.
3. **The "is the right node registered" check**, per the four instances above.
4. **`costs_2015.py`** and the other items of §9's "left undone" — unchanged.

## 11. Session wrap — verified state, and what to pick up first

**Read this section first tomorrow.** §§ 8-10 are the detailed records written as the work
landed; this is the consolidated state, the measurements that exist only here, and the
priority order. Everything below was independently re-verified on a quiet tree at the end
of the session, not taken from an agent's report.

### 11.1 Verified state

| check | value |
|---|---|
| `pytest functional_process -q` | **3597 passed**, 2909 skipped |
| `run_mda_harness.py` | **404 agreements, 32 disagreements** (0 in driven blocks), 3 unverifiable, **0 ungrounded**, 21 errors, 55 switch kwargs / 0 mismatched |
| `GRAPH` (`REFERENCE_CONFIGURATION`) | 146 nodes; 148 after `driven_graph`'s cuts |
| SAND | 24 conditions x 17 design, **0 non-finite cells** |
| Stage C2 | converges, 47 SQP iterations, feasible, convergence 7.4e-9 (PROCESS's own: 2.4e-7) |

**The 32 disagreements are three causes, not thirty-two.** 18 of them are a single
`+17.604 MW` offset propagating linearly — verified arithmetically, identical to six
decimals across seven fields with the sign correctly flipping on
`p_plant_electric_net_mw` (net = gross - recirc). Its cause is an inconsistency **in
PROCESS**, not the port (see § 10's "A finding about PROCESS itself"). The rest are the
`VacuumOld` tolerance difference (§ 8.4) and its propagation into Account 224 and the cost
chain.

### 11.2 The Stage C timing, corrected — compilation, not the optimiser

§ 10 reports the C2 solve as "7.5 s against PROCESS's 96.8 s". That number has trace and
compilation **inside** the timed region (`run_sand_harness.py` wraps
`solve_schedule(...)`, and `VmconDriver.__call__` builds its `jax.jit` closures inside
that call). Split by timestamping the SQP callback, 47 iterations reproduced:

| | |
|---|---|
| total wall time | 6.69 s |
| **trace + compilation** | **5.63 s — 84 % of total** |
| the remaining 46 iterations | **0.72 s** |
| median per iteration | **14.7 ms** (min 12.7, max 27.1) |

So the honest comparison against PROCESS's 96.8 s is **0.72 s of solving** plus a one-time
5.6 s compile — the port was being undersold, not oversold. Two consequences:

- **The optimiser plumbing is the per-iteration cost, not the graph.** 14.7 ms against a
  0.69 ms Jacobian is ~21x overhead: `pyvmcon`'s line search makes several `evaluate`
  calls per iteration, each crossing the JAX/numpy boundary. That is where to look if
  per-iteration cost ever matters — not at the derivative.
- **`VmconDriver.__call__` rebuilds its `jax.jit` closures on every call**, so each is a
  fresh cache key and every solve pays the 5.6 s again. One solve absorbs it; **a `Scan` of
  N points would recompile N times.** Given § 11.4's measured 46x batching win, this is
  the single change that would matter most for the scan use case, and it is small: hoist
  the jitted callables so they are built once per graph shape.

### 11.3 Boundary inputs — audited, and much smaller than it looked

Full record: **`_audit/boundary_inputs_audit.md`**. Of 379 unowned inputs, only 79 are set
by the input file — but of the remaining 300, **only 12 are real cut edges**: 242 are
genuine inputs (207 never assigned anywhere in `process/`, 35 assigned only to a constant),
and 46 are off-path for a stellarator (`caller.py:272-275` returns after
`stellarator.run()`; `Power.pfpwr` alone accounts for 19).

**`len_tf_coil` does not hide a cross-subsystem cycle** — measured, not inspected: its
producer's inputs are owned by `StellaratorScalingFactors`, unreachable from all four
readers. The only cycle among the twelve is `fusden_alpha_total`, and it lands *inside* an
SCC `mda.CUTS` already cuts. **So closing all twelve leaves "zero SCCs crossing a subsystem
boundary" intact** (§ 11.4).

Two findings from that audit matter more than the cut edges themselves, and **both are
open**:

1. **`ProfileValues.rho` is bound to `.neoclassics.r_eff`, a field PROCESS never writes**
   (default `0.0`); PROCESS passes the literal `0.6` (`neoclassics.py:290`). Measured:
   `dr_densities` is identically zero at rho=0 against `-4.2e19` at rho=0.6. This is a
   **wrong answer**, not a coverage gap — same class as the `q95`/`iotabar` bug (§ 8.3).
2. **`mda_harness.compare` silently drops every array-valued output.**
   `float(np.asarray(got))` sits inside a bare `except: continue` (`mda_harness.py:695-699`),
   *before* any bookkeeping, so **29 of 487 owned variables are counted as neither
   agreement, disagreement, unverifiable, nor error.** That blind spot is why (1) was
   invisible, and it means § 11.1's 404/32 has 29 unexamined variables behind it.

Fix (2) before (1): (2) is what makes (1) verifiable.

### 11.4 Structure — provenance vs derived clustering, measured

Recorded in `_audit/switch_elimination_design.md` § 11. Comparing the port's module layout
(provenance) against `Blocking.scc`:

| grouping grain | real SCCs | crossing a boundary |
|---|---|---|
| subsystem (`stellarator`, `physics`, `costs`) | 3 | **0** |
| source file | 3 | **3** |

Every cycle is contained within one subsystem and spans several files inside it, while
**346 cross-subsystem edges** exist — so subsystems are heavily coupled but *acyclically*,
which is why the MDA needs only 11 driven blocks of 128. **The subsystem is the right grain
for a model group; the file is not.**

Caveat: this measures PROCESS's *filing habits*, since provenance here is the module tree.
Declaring the grouping on physical grounds and finding the same containment would be the
stronger claim.

Also measured (§ 4.4 of `optimise_design.md`): the **driven** MDA schedule — all 9
Newton/Picard blocks, `while_loop`s included — `vmap`s cleanly, **~46x throughput at
B=64**, guarded against dead-code elimination (169 of 573 outputs genuinely vary). The
payoff target is `Scan`, which today re-solves every point from scratch.

### 11.5 Design work recorded today, not yet implemented

**`_audit/switch_elimination_design.md`** — replacing integer `i_*` switches with model
selection. Conclusions: it is already the declared policy (`traceability_policy.md`'s split
default), `COMMON` is *already* the model list (79 declarations; a `Switch` is just
`int -> tuple[declaration]`), and the settled target is **three trees** — model tree
(structure, provenance intrinsic to each model's own module, namespaces as classes/modules
rather than dicts to mirror how `DataStructure` addresses variables), settings tree (same
shape, `None` where absent, schema *derivable* from each model's `eqx.field(static=True)`
declarations), and `env` (traced boundary inputs, already separate). `materialise(models,
settings)` zips the first two and replaces the integer `Configuration` outright.

Two cottax changes are required, both small: `NodalDeclaration.name` must be position-derived
rather than `type(self).__name__`, and `to_graph`/`node_and_names` must accept a namespace of
`NodalDeclaration`s (today the mapping form takes only raw `NodeDefinition`s — it wants
`.owns`). **Hierarchical `NodePath`s already work** in `Graph`/`Blocking`/`topological_order`,
verified directly; only the declaration surface cannot express one. Instance-derived ports
**also already work** (`_params` is an ordinary property) — an earlier claim that this needed
an upstream change was tested and retracted.

Undecided, and worth settling before writing code: whether `graph.under(prefix)` sees through
mint prefixes (`^problem.physics.profiles.x` does not live under `physics`, so subtree swaps
would strand minted nodes).

### 11.6 Priority order

1. **`mda_harness.compare`'s dropped arrays** (§ 11.3.2) — a 29-variable hole in the
   measurement everything else is judged by. Cheap, and it gates the next item.
2. **`ProfileValues.rho` / `r_eff`** (§ 11.3.1) — a known wrong answer.
3. **`c62`'s Jacobian row** (§ 10) — value exact, derivative wrong by ~5x with a sign flip
   on x4; the only cell disagreeing for an unknown reason. The same topological walk that
   found the x4 defect should settle it.
4. **Hoist `VmconDriver`'s jitted callables** (§ 11.2) — small, and the prerequisite for
   batched/scanned solves being worth anything.
5. **The cold-start gap** (§ 10, Stage C3) — 20 of 24 conditions are `nan` at the cold
   point because the cold `DataStructure` has `0.0` in every model-computed field. **The
   port cannot yet run its own pipeline from a cold input file at all**, optimiser or not.
   This is the ceiling on the whole result: today's claim is "the port solves its own
   problem faster and tighter than PROCESS solves PROCESS's, *seeded from PROCESS's
   answer*". "Reproduces PROCESS's optimisation from scratch" stays undemonstrated until
   this closes, and § 11.3's twelve cut edges are part of what a cold start must produce
   for itself.
6. **The two systemic gaps, now at four instances each** — "is the right *node*
   registered" (every instance found by a downstream consumer, never by a check;
   `switch_audit` checks kwargs on nodes that are already present) and PROCESS's
   report-pass/solve-pass inconsistency as an undetected category, alongside
   `KNOWN_UNVERIFIABLE_OUTPUTS`/`EXPLAINED_DISAGREEMENTS`.
7. **The switch-elimination work** (§ 11.5) — scheduled next, per the design doc's own
   order: enum-aware `switch_audit` first so the net that caught five bugs is not lost in
   the act of acting on it.

### 11.7 A recurring lesson, worth keeping

Five confident diagnoses were overturned by measurement this session, several of them mine:
`i_plasma_ignited` as the cause of the 1.2 % confinement gap (it was the `q95`/`iotabar`
binding); `sig_tf_wp_max = 0.0` as the cause of c32's `inf` (it was the coil-island
placeholder feeding `a_tf_wp_no_insulation`, one node further up); "no speed win at this
size" (an unjitted cold timing); "costs is unported" (23 nodes existed); and "instance
fields cannot drive a node's reads-set" (they can). In every case the correction came from
running something, not from reading harder. The `jnp.sqrt(jnp.maximum(0, x))` nan-gradient
trap has now appeared **twice** for the same reason — value-correct, derivative `nan` — and
only the gradient tests see it.
