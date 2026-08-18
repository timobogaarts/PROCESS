# Unit registry — Phase 0, stellarator scope

Scope rule (derived by tracing `Stellarator.run()`'s actual call surface, not assumed):
**all of `process/models/stellarator/**` in full, plus only the specifically-called
methods (and their direct callees) on `Stellarator`'s injected sub-models** — not those
shared files' entirety. Injected sub-models per `Models.__init__` (`process/main.py`):
`availability`, `buildings`, `vacuum`, `costs`, `power`, `plasma_profile`, `hcpb`
(CCFE-HCPB), `current_drive`, `physics`, `neoclassics`, `plasma_beta`,
`plasma_bootstrap`. Of these, `current_drive`/`plasma_beta`/`plasma_bootstrap` are never
called directly by `stellarator.py` (grep for `self.current_drive.`/`self.plasma_beta.`/
`self.plasma_bootstrap.` in `process/models/stellarator/*.py` returns nothing) — they're
reached only indirectly, through `self.physics.*` methods that were themselves
constructed with those sub-models in `main.py`. Not in scope directly; may surface as
transitive callees of the `physics.py` methods below and get their own row then.

Legend: **status** `pending` (not started) / `in-progress` / `draft` (fork finished, not
reviewed) / `reviewed` (you've checked flagged items) / `final`.

## Model-unit files — `process/models/stellarator/**` (full scope)

**Scope correction (found auditing chunk 1A):** the original 8-file, 4972-LOC scope was
wrong — `find process/models/stellarator -name "*.py"` (recursive) was never actually
run; the earlier `wc -l process/models/stellarator/*.py` used a non-recursive glob and
silently missed a whole subpackage, `process/models/stellarator/coils/` (6 files, 1950
LOC: `calculate.py` 593, `coils.py` 303, `forces.py` 133, `mass.py` 146, `output.py` 546,
`quench.py` 228 — `__init__.py` excluded). `st_coil` (imported from `coils/calculate.py`
and called at `stellarator.py:144,160`) is exactly as central to `run()`'s call sequence
as `st_build`/`st_div`/`st_heat`, which were already in scope. True full-scope total:
14 files, **6922 LOC**, not 8/4972. Added as units 9-14 below.

| # | source | LOC | record path | status |
|---|---|---|---|---|
| 1 | `stellarator.py` | 2600 | split into 9 chunks below, one record each — real method boundaries turned out uneven, the original "3-4 chunks" guess was wrong | **in-progress** |
| 2 | `build.py` | 439 | `functional_process/models/stellarator/build.md` | **reviewed — ported** (`build.py`, whole file, 4 tier-1 functions/nodes, tests passing (fuzz only, no PROCESS unit test exists for `st_build`); split `.fwbs.blktmodel` and `.heat_transport.ipowerflow`; found a real `conditional-ownership-by-run-config` case on `.build.dr_blkt_inboard`/`dr_blkt_outboard`, second precedent alongside `.physics.aspect` in 1C) |
| 3 | `density_limits.py` | 290 | `functional_process/models/stellarator/density_limits.md` | draft (confidence: medium — blocked on unit #1 for `power_at_ignition_point`'s signature) |
| 4 | `divertor.py` | 233 | `functional_process/models/stellarator/divertor.md` | **reviewed — ported** (`divertor.py`, whole file, tier-1, self-contained; `Divertor` node registered; confirmed `.divertor.a_div_surface_total` is unconditionally produced here — the `50.0` fallback chunk 1E2 found in `st_fwbs` is a call-order issue in unit #1, not a gap in this unit) |
| 5 | `heating.py` | 235 | `functional_process/models/stellarator/heating.md` | **reviewed — ported** (`heating.py`, `isthtr` branches 1/2 + common tail, 5 tier-1 functions, tests passing; `isthtr==3`/NBI stays audit-only, calls `culnbi()`) |
| 6 | `initialization.py` | 67 | `functional_process/models/stellarator/initialization.md` | **reviewed — mostly not a node** (16 of 19 writes are unconditional stellarator-mode device-preset literals, same shape as chunk 1D's `fncmass`/`gsmass`; only the 3 summed pulse durations are a real computation — ported as `calculate_pulse_durations`/`PulseDurations`, tier-1, tests passing) |
| 7 | `neoclassics.py` | 841 | `functional_process/models/stellarator/neoclassics.md` | **draft — no chunking needed** (13 clean sub-methods, no line-range splits required, one record). Dispatched specifically to hunt for a self-contained tier-2 unit — **none found**: despite the size, `init_neoclassics`/`calc_neoclassics` are straight-line tier-3 pipelines (each sub-method called exactly once, fixed order), not an internal solve. All 13 pure functions ported + harness-tested. 2 are fully scalar-argument (`calculate_profile_values`, `calculate_effective_thermal_diffusivity`, both `ExplicitFunction` nodes registered in `total_process.py`); the other 10 take species-length-4 or quadrature-length-30 **array** arguments and were briefly blocked on `Tier1Contract` assuming every differentiated argument was scalar. **That harness gap is now fixed** (`_harness/contracts.py`'s `_reference_along`/`_jacobian` differentiate component-by-component over an array argument, `_component_label` names failures as `kt[1, 7]`), so these 10 are exercised, not merely translated — no node written for them yet, which is a wiring question, not a testing one. **Two caveats, both open**: `TestCollisionFrequency` and `TestNormalizedCollisionFrequency` each fail `test_gradient_agreement` at one fuzz point under `--fp-gradients` (the default suite is green — the gradient check is opt-in). Both agree to ~8e-10 *relative* on outputs of magnitude ~1e12 and fail only against an error bar that is absolute and truncation-derived, so this reads as a harness-tolerance gap rather than a port bug — but it is **not proven**, and "validated" should not be claimed for these two until it is. Also found: `.neoclassics.iota` and `.neoclassics.er` are read but never written anywhere in this file (external producer not yet located — candidates: `st_geom`/1C for `iota`, unknown for `er`); `iota` is confusingly two different things under one name within this same file (a `data` field vs. a forwarded argument, not asserted equal). |
| 8 | `preset_config.py` | 266 | `functional_process/models/stellarator/preset_config.md` | **reviewed — not portable as a node** (confirms/extends chunk 1C's `istell` device-config-table finding: 5 hardcoded machine presets + reflective `hasattr`/`setattr` copy, real output set only knowable by cross-referencing `StellaratorConfigData`'s fields, not a fixed declarable `outputs` list; `istell==6` does file I/O, confirms 1C's flag from the other call site. Recommends a static per-machine config record, selected at graph-assembly time, same open question as unit #6's device-preset literals — not resolved here) |
| 9 | `coils/calculate.py` | 593 | `functional_process/models/stellarator/coils/calculate.md` | **reviewed — 10/12 functions ported** (`calculate.py`, `ExplicitFunction` nodes, tier-1 tests passing). `st_coil` (the orchestrator, contains it — still called directly from `run()`) and `winding_pack_total_size` are **not** ported: both call into units #10-14 (`winding_pack_total_size` needs `intersect`/`bmax_from_awp`/`jcrit_from_material` from unit #10, a genuine Newton–Raphson root-find — real tier-2, blocked, same shape as `power_at_ignition_point`↔`st_phys`). Found: two locals with no PROCESS storage (`coilcurrent`, `coilcoilgap`) — `coilcurrent` minted `.stellarator.coilcurrent` since it's a real graph edge (feeds a ported node), `coilcoilgap` dropped (reporting-only, like `msupstr`). |
| 10 | `coils/coils.py` | 303 | `functional_process/models/stellarator/coils/coils.md` | **draft — 3/4 functions ported.** `intersect` is now ported, self-contained tier-2, **the first real `Tier2Contract` exercise**: kept PROCESS's defining equation (`interp(x,x1,y1) - interp(x,x2,y2) = 0`) but replaced the fixed-iteration finite-difference Newton loop (data-dependent early `break`, no faithful JAX translation) with `optimistix.Bisection` over the curves' guaranteed-valid overlap, followed by exact `jax.grad`-based Newton corrections (exact because `jnp.interp` is piecewise-linear). Validated against PROCESS's own legacy test case plus 6 curated fixed-seed synthetic curve pairs. No `## cottax node` written yet — its only call site's arguments are locals inside `winding_pack_total_size`'s unported solve loop, not established `VarPath`s; a sketch is left in the record. `jcrit_from_material` **still not ported** — confirmed genuinely blocked, not just deferred: an 8-way switch (`i_tf_sc_mat`, new, see switches table) dispatching into `physics/superconductors.py` (now unit #22, unaudited). `winding_pack_total_size` (unit #9) is **now unblocked on `intersect`** — its remaining blocker is exclusively unit #22, plus a design step of its own (minting real VarPaths for its locals, including `coilcurrent` — `calculate.md` already flags this same need). |
| 11 | `coils/forces.py` | 133 | `functional_process/models/stellarator/coils/forces.md` | **reviewed — ported** (all 7 functions tier-1, tests passing (fuzz only); 2 of 7 got `cottax` nodes, other 5 didn't — see record for why) |
| 12 | `coils/mass.py` | 146 | `functional_process/models/stellarator/coils/mass.md` | **reviewed — ported** (`calculate_coils_mass`, one composed tier-1 function chaining 8 internal steps, tests passing (fuzz only); uses the `local-intermediate` classification throughout, same as chunk 1D) |
| 13 | `coils/output.py` | 546 | `functional_process/models/stellarator/coils/output.md` | **reviewed — confirmed pure reporting shell**, no computation to port (one straight-line function, zero branches, zero writes; several trivial inline arithmetic expressions for display only, same "reporting isn't quite inert" pattern as `density_limits.py`/`stellarator_E3` but none worth extracting — see record) |
| 14 | `coils/quench.py` | 228 | `functional_process/models/stellarator/coils/quench.md` | **reviewed — ported** (`calculate_quench_protection` + 3 helpers, tier-1, tests passing; eliminated the `coilcurrent` parameter that previously blocked a node — verified `coilcurrent = c_tf_total / (n_tf_coils * 1e6)` exactly from `coils/calculate.py:276`, so the unit #9 dependency this unit had is now gone) |

### `stellarator.py` chunks (unit #1, split by method — see `stellarator.py`'s
`grep -n "^    def "` for exact boundaries)

Two units already depend on chunk 1B (`density_limits.md`'s `power_at_ignition_point`,
`constraints.md`'s constraint 91) — 1B is priority.

| # | methods | lines | record path | status |
|---|---|---|---|---|
| 1A | `run`, `output` | 114-190 | `functional_process/models/stellarator/stellarator_A_orchestration.md` | draft, confidence: high — orchestration/call-map only; found a second hidden double-`st_phys`-call pattern and confirmed the constraint-91 "unconditional call" discrepancy from the caller side |
| 1B | `st_phys` | 1886-2456 | `functional_process/models/stellarator/stellarator_B_st_phys.md` | draft, confidence: medium-high — unblocks `density_limits.md`'s `power_at_ignition_point`; found the mechanism behind its hardcoded "call twice": a genuine stale-read-before-write on `.physics.b_plasma_surface_poloidal_average`, i.e. `power_at_ignition_point`'s two calls are literally 2 steps of Picard iteration, not an arbitrary constant. Two new shared-file units needed (added below: `fusion_reactions.py`, `radiation_power.py`); `rether` added to unit 9's method list |
| 1C | `st_new_config`, `st_geom` | 191-319 | `functional_process/models/stellarator/stellarator_C_geometry.md` | **reviewed — ported and registered this pass** (bumped to match the record's own frontmatter, which already said `reviewed` — a bookkeeping-only fix, not new audit work). Found a real conditional-ownership pattern on `physics.aspect` (owned by this node only when *not* an active iteration variable) and a second, distinct role for `istell` (device-config table selection, data-table-shaped not switch-shaped) beyond the pipeline split already in `switches.md`. `DefaultAspectRatio`/`StellaratorScalingFactors`/`StellaratorPlasmaGeometry` all registered unconditionally in `total_process.COMMON` — the bare `NumericsData.ixc` dataclass default (`[0, 0, ...]`, no active iteration variables) makes `1 not in ixc` true, so `DefaultAspectRatio` belongs in the default `GRAPH`, matching every other topology decision's use of PROCESS's own bare defaults. |
| 1D | `st_strc` | 320-421 | `functional_process/models/stellarator/stellarator_D_structure.md` | **reviewed — ported** (`stellarator_D_structure.py`, tier-1 tests passing) |
| 1E1 | `blanket_neutronics` + `st_fwbs` part 1 | 422-880 | `functional_process/models/stellarator/stellarator_E1_fwbs_setup.md` | draft, confidence: medium — cut falls mid-computation; **superseded by the synthesis below**, kept for its per-chunk detail. Its `.fwbs.f_p_blkt_multiplication` stale-read worry is a **false alarm**: traced to `core/input.py:391`, an ordinary `InputVariable`, unrelated to call history — see synthesis. `.fwbs.p_div_rad_total_mw` read-but-never-assigned is **confirmed real**: a copy-paste duplicate-write leaves it permanently at its `0.0` default whenever `blktmodel != 1 & ipowerflow == 1` — deterministic, not merely stale. |
| 1E2 | `st_fwbs` part 2 | 881-1280 | `functional_process/models/stellarator/stellarator_E2_fwbs_neutronics.md` | draft, confidence: medium-high — **superseded by the synthesis below**. Its `first_call_stfwbs` finding is confirmed as the real (and only) cross-call mechanism in this method. |
| 1E3 | `st_fwbs` part 3 | 1281-1682 | `functional_process/models/stellarator/stellarator_E3_fwbs_shield_divertor.md` | draft, confidence: high — **superseded by the synthesis below**. Both `i_tf_sup` corrections stand. |
| — | **`st_fwbs` synthesis** (reads 1E1+1E2+1E3 together, per `next_steps.md` §3) | 422-1682 | `functional_process/models/stellarator/stellarator_E_fwbs_synthesis.md` | draft — **the real semantic boundaries, replacing the even-thirds line cuts**: **S1** `fw_blanket_shield_geometry_setup` (515-605, tier-1, portable now); **S2** `blanket_shield_tf_nuclear_power` (422-480+608-1030, the real `blktmodel`×`ipowerflow` dispatch, 3 live arms, tier-3 in two — blocked on unit #13 `hcpb.py`); **S3** `divertor_mass_and_first_call_seed` (1030-1043, tiny but hardest — a genuine two-node SCC with `Divertor`, unit #4, needs `Blocking`+`FixedPoint`, not an ordinary node — `first_call_stfwbs` is real call-order state: `st_fwbs` runs before `st_div` every iteration, so every call after the first reads `.divertor.a_div_surface_total` as written by the *previous* `run()`, bootstrapped by a hardcoded `50.0` only on the true first call); **S4** `blanket_shield_fw_coolant_mass` (1045-1274 excl. S3, tier-1, blocked only on S2/S3's signatures settling — where `blktmodel`'s and `blkttype`'s mass arms live); **S5** `cryostat_and_vv_geometry` (1282-1330, self-contained, portable now); **S6** `st_fwbs_output` (1331-1682, reporting shell, out of scope). Net: **S1 and S5 are portable today**; S4 needs no further audit, just S2/S3's signatures; S2 is blocked on unit #13; S3 needs the SCC/FixedPoint node shape, not more auditing. The `coilhtmx`-etc. locals are confirmed ordinary same-function `In`/`Out` values (S2 produces them in one arm, S6 consumes under the identical guard — no unassigned-reference risk, just 700 unrelated lines between). |
| S1+S5 | `fw_blanket_shield_geometry_setup` + `cryostat_and_vv_geometry` | 515-605, 1282-1330 | `functional_process/models/stellarator/stellarator_fwbs_s1_s5.md` | **reviewed — ported and registered** (`FwBlanketShieldGeometry`, `CryostatAndVvGeometry`, both tier-1, unconditional in `total_process.COMMON`; frontmatter was missing entirely on this record, added this pass to unblock `test_registry_coverage.py`, no computational content changed) |
| S2 | `blanket_shield_tf_nuclear_power` | 422-480, 608-1030 | `functional_process/models/stellarator/stellarator_fwbs_s2.md` | **draft — 2/3 arms ported, both registered this pass.** `ExponentialAttenuationBlanketShieldPower` (`blktmodel != 1 & ipowerflow == 0`) and `DetailedPowerflowBlanketShieldPower` (`blktmodel != 1 & ipowerflow == 1`, PROCESS's real default) wired as a joint `Switch` (`total_process.TOPOLOGY_SWITCHES`, `path=".fwbs.blktmodel,.heat_transport.ipowerflow"`, a synthetic composite key since the real dispatch is two switches nested together, not one) — arm 3 (`blktmodel == 1`) stays `unported`, blocked on unit #13's live call-site bug (`blanket_neutronics()`'s zero-argument `hcpb.py` calls). **A real registration bug found and fixed this pass**: `ScTfCoilNuclearHeating` (chunk 1F) was unconditionally in `COMMON`, but its formula is only correct for arm 2 (`ipowerflow == 0`) — PROCESS's own default configuration (`blktmodel = 0`, `ipowerflow = 1`) lands in arm 3, which computes its own, different `.fwbs.p_tf_nuclear_heat_mw` (`DetailedPowerflowBlanketShieldPower`'s `pnucsi + pnucso - pnucshldi - pnucshldo`, `i_tf_sup == SUPERCONDUCTING` only). Unconditional placement was computing the wrong formula for the default `GRAPH`, the same bug class already fixed once for `EcrhDensityLimit`/`BlktmodelBlanketThickness` (below). `ScTfCoilNuclearHeating` moved into the switch's arm-2 `Alternative`. |
| S3 | `divertor_mass_and_first_call_seed` | 1030-1043 | `functional_process/models/stellarator/stellarator_fwbs_s3.md` | draft — ported as `DivertorPlateMass`, already registered in `total_process.COMMON` (see unit #4's row and `next_steps.md` §5 Shape A: confirmed an ordinary acyclic edge onto `Divertor`, not a genuine SCC, so no `Blocking`/`FixedPoint` was actually needed here despite the synthesis row's original expectation) |
| 1F | `sc_tf_coil_nuclear_heating_iter90` | 1683-1885 | `functional_process/models/stellarator/stellarator_F_tf_nuclear_heating.md` | **reviewed — ported** (`stellarator_F_tf_nuclear_heating.py`, SC branch only, tier-1 tests passing) |
| 1G | `st_phys_output` | 2457-2600 | `functional_process/models/stellarator/stellarator_G_output.md` | draft — confirmed purely reporting (no computation invoked, unlike density_limits.py's `output()`), high confidence, no proposed signature (out of port scope) |

`st_fwbs` (481-1682, 1200 lines) has no clean top-level section markers — E1/E2/E3's line
cuts are even thirds, not semantic boundaries. Each of those three records must say
explicitly if a computation spans its chunk's boundary rather than silently treating the
cut as a real seam.

## Model-unit files — scoped subset of shared files

Only the methods listed are in scope; the record must say so explicitly (schema's
"scope reason if partial").

| # | source | methods in scope | record path | status |
|---|---|---|---|---|
| 9 | `physics/physics.py` | `plasma_composition`, `calaculate_stored_thermal_energy` [sic — matches source spelling], `calculate_effective_charge_ionisation_profiles`, `calculate_total_plasma_heating_power`, `outplas`, `phyaux`, `rether` [added — found in chunk 1B's audit; imported as a bare function from this file by `stellarator.py`, missed by the original `self.physics.<method>` grep since it's not called through the `self.physics.` attribute], `PlasmaBeta.fast_alpha_beta` [added — scope sweep: called directly as `self.beta.fast_alpha_beta(...)` at `stellarator.py:2079`; missed by the original grep because `Stellarator.__init__` aliases the injected `plasma_beta` sub-model to `self.beta` (`stellarator.py:107`), not the name the grep checked — a level-1 miss, not a level-2 one like the others in this list. `self.bootstrap.*`/`self.current_drive.*` confirmed genuinely unreached (no aliasing issue there, just not called).] | split into 3 chunks, see table below | **ported (8/8 methods)** — see chunk table |
### `physics.py` chunks (unit #9, chunked by *tier characteristic*, not position — the 8
in-scope methods scatter across a 6931-line file and two classes, `Physics`/`PlasmaBeta`)

| chunk | methods | record path | status |
|---|---|---|---|
| A | `rether`, `phyaux`, `calculate_total_plasma_heating_power`, `calaculate_stored_thermal_energy` [sic], `PlasmaBeta.fast_alpha_beta` | `functional_process/models/physics/physics_A_pure_formulas.md` | **draft — ported**, tests passing (fully tier-1 and fully self-contained, same shape 1D/1F were bumped to `reviewed` for — left `draft` here only because bumping a record's own frontmatter is outside this consolidation pass's editing boundary, not because the work is less complete; a quick follow-up pass can bump it. (all 5 already fully pure in the source — zero `self.data` access in any body — no translation judgment left, only `if`/`min`/`max` -> `jnp.where`/`jnp.minimum`/`jnp.maximum`. 6 `ExplicitFunction` nodes (`calaculate_stored_thermal_energy` bound twice, electron/ion), all registered in `total_process.COMMON`. `i_beta_fast_alpha` kept as a static kwarg on `FastAlphaBeta` — both branches read the same 6 variables, textbook "identical reads-set" exception, not a `Switch`. Two `jnp.where`s needed a denominator guard against a genuine `0/0` on the *unselected* branch (`phyaux`'s `t_alpha_confinement`/`burnup`, `fast_alpha_beta`'s `fact2`) — flagged as a named recurring category, not yet in `traceability_policy.md`.) |
| B | `plasma_composition`, `calculate_effective_charge_ionisation_profiles` | `functional_process/models/physics/physics_B_composition.md` | **draft — 2/2 get nodes, both registered this pass.** `.physics.first_call`'s self-loop is now resolved (`next_steps.md` §5's Shape-B recipe, applied this session): split across `NextFirstCall` (`FixedPointFunction`, owns `.physics.first_call` alone) and `PlasmaComposition` (`ExplicitFunction`, every other output, `first_call` read as a plain `Input`) — both registered unconditionally in `total_process.COMMON`, `is_ignited=False` matching `physics_variables.py:881`'s default. **A second Shape-B self-loop was found and resolved without any `Cut`/`FixedPoint` machinery at all**: `.impurity_radiation.f_nd_impurity_electron_array` is read at indices 2-13 and written at indices 0/1 by the same function — genuinely disjoint ranges, so per-index `VarPath`s (`s.impurity_radiation.f_nd_impurity_electron_array[i]`, a `SequenceKey` component) make this an ordinary node, not a self-reference; `CalculateEffectiveChargeIonisationProfiles`'s own signature was updated to match (per-index inputs) and reconfirmed to `to_graph` correctly together with `PlasmaComposition`, now a real edge on indices 0/1 where there was previously an unproduced boundary input. **Registering `PlasmaComposition` surfaced a new, genuine cross-node Shape-A cycle**, found only once the full graph was assembled: `DensityProfile → FusionRates → PlasmaComposition → PedestalOnAxisDensities → DensityProfile` (`nd_plasma_electron_profile`/`proton_rate_density`/`nd_plasma_ions_total_vol_avg`/`nd_plasma_electron_on_axis`) — a real physics feedback loop (density profile ↔ composition ↔ fusion rates ↔ pedestal), the second confirmed cross-subsystem SCC in this graph (see `next_steps.md` §5, which had previously found only one). Not driven, per this project's standing policy of representing cycles before deciding how to drive them. Also not ported: the `znfuel < 0` domain check (PROCESS raises; the quantity stays a well-defined-but-unphysical finite float either way, so "return non-finite" doesn't apply cleanly — flagged, not resolved). `i_plasma_ignited` kept as a static `bool` despite differing reads-sets (`traceability_policy.md`'s "split" default technically applies) — a deliberate, flagged policy deviation: the differing branch is 2 lines inside an otherwise-shared 328-line function. |
| C | `outplas` | `functional_process/models/physics/physics_C_outplas.md` | **draft — ported.** 1095-line reporting method reduces to exactly 3 writes / 1 real computation (`nu_star`, `rho_star`, `beta_mcdonald`) — confirmed by an exhaustive `self.data.* =` grep, not sampling. 1 node, `DimensionlessPlasmaParameters`, registered in `total_process.COMMON`. Same `reviewed`-eligible-but-left-`draft` note as chunk A applies. |

| 10 | `physics/confinement_time.py` | `calculate_confinement_time`, `calculate_double_and_triple_product` | `functional_process/models/physics/confinement_time.md` | **draft — ported** (48 transitive scaling-law statics + the 2 in-scope methods + 1 out-of-file one-liner (`plasma_geometry.py::calculate_iter_physics_basis_elongation`), tier-1, tests passing incl. gradients. 3 cottax nodes: `ConfinementTime`, `DoubleAndTripleProduct`, `IterPhysicsBasisElongation` — only the first two are registered in `total_process.py`; `IterPhysicsBasisElongation` is a duplicate producer of `.physics.kappa_ipb` (already owned by `ConfinementTime` itself) and is deliberately left unregistered, not a bug in the port. **A real defect was found and fixed during this consolidation pass**: the record's own "cottax node" sketch and "switches touched" section disagreed on whether `i_plasma_ignited` is a static field or an `Input`, and the shipped `.py` had `i_plasma_ignited` as an `Input` while `i_confinement_time`/`i_rad_loss` were plain (non-`eqx.field(static=True)`) annotated fields — none of the three were actually static under `eqx.Module`'s default (dynamic pytree leaf) semantics, which would raise under any `jax.jit`/`jacfwd` a driver traces this node inside (`calculate_confinement_time`'s dispatch does real Python `if`/`elif` branching on all three). Fixed minimally in `confinement_time.py`: all three now `eqx.field(static=True)`, `i_plasma_ignited` moved out of `__call__`'s `Input` list. Verified directly with `eqx.filter_jit` + `jax.jacfwd` post-fix. Three real PROCESS bugs found and reproduced faithfully, not fixed: `KAYE_GOLDSTON`'s scrambled positional call, `USER_INPUT`'s always-raises dead branch, `PAZ_SOLDAN_NT`'s dead-by-enum-aliasing branch — see the record's dedicated sections.) |
| 11 | `physics/exhaust.py` | `calculate_radiation_fraction` | `functional_process/models/physics/exhaust.md` | **draft — ported** (tier-1, self-contained, `RadiationFraction` node registered in `total_process.py`; NaN-safe `jnp.where` division guard verified against `test_gradient_finite`) |
| 12 | `physics/plasma_profiles.py` | `PlasmaProfile.run()` (= whole file; `run()` reaches every method) | `functional_process/models/physics/plasma_profiles.md` | **draft — 5/5 arithmetic functions ported, all tier-1, tests passing incl. gradients**. `PlasmaProfile.run()` is *not* portable end-to-end from this file: both branches call `neprofile.run()`/`teprofile.run()` for effect, which belong to unit #21 (`physics/profiles.py`, now itself ported — see below). Ported: `calculate_ion_vol_avg_temperature`, `calculate_parabolic_profile_values`, `calculate_pedestal_profile_values`, `calculate_profile_factors`, `calculate_parabolic_gradient_lengths`. **`i_plasma_pedestal`'s two-role blocker is resolved** — see switches table and `total_process.TOPOLOGY_SWITCHES`. `ProfileFactors` is in `COMMON` (unconditional — verified by unit #21's audit that both `i_plasma_pedestal` arms feed it identically). `ParabolicGradientLengths` is now registered under the switch's `value == 0` arm (no pedestal-arm counterpart exists in PROCESS itself, so it is not an `Alternative` — `.physics.gradient_length_te`/`_ne` are simply unproduced when `i_plasma_pedestal == 1`, PROCESS's own default). Found: the four on-axis writes in `parabolic_parameterisation` are `redundant-duplicate-write`s of `NeProfile`/`TeProfile.set_physics_variables` (verified algebraically identical, and independently re-verified by unit #21's own audit); two minted VarPaths for the profile arrays (no PROCESS storage, same as `.stellarator.coilcurrent`); `.divertor.prn1` is the file's only cross-area write and is pedestal-branch-only. **A real port bug was caught by the gradient check** — see the `_simpson` docstring. |
| 21 | `physics/profiles.py` | `NeProfile.run()`, `TeProfile.run()`, `Profile.*`, `set_physics_variables`, `calculate_profile_y`, `ncore`/`tcore` | `functional_process/models/physics/profiles.md` | **draft — 12/12 functions ported**, all tier-1, tests passing incl. gradients (558 LOC source, injected into `PlasmaProfile` in `main.py:674-676`). **Resolves unit #12's `i_plasma_pedestal` blocker**: per-arm classification (12 node classes) — common to both arms: `ProfileGrid`, `NeProfileIntegral`, `TeProfileIntegral`, `DensityProfile` (this *corrects* unit #12's blanket claim that both `NeProfile` and `TeProfile` branch on the switch — `NeProfile.calculate_profile_y`'s check is dead code, no `return` after the parabolic assignment, so the pedestal formula unconditionally overwrites it; verified algebraically and by a dedicated test); parabolic-only (`i_plasma_pedestal == 0`): `ParabolicTemperatureProfile`, `ParabolicOnAxisDensities`, `ParabolicOnAxisTemperatures`; pedestal-only (`!= 0`): `PedestalTemperatureProfile`, `PedestalOnAxisDensities`, `PedestalOnAxisTemperatures`; gated by a **different, newly-found nested switch** `i_nd_plasma_pedestal_separatrix` (see switches table): `GreenwaldDensityFractions`, `PedestalSeparatrixDensities`. All registered in `total_process.py` except the last two — confirmed reachable **only** from `physics.py` (unit #22, tokamak, out of stellarator scope), never from `stellarator.py`; ported anyway since splitting the file was more work than porting it (same call as unit #20 made for `ImpurityRadiation`), but left out of the graph as genuinely unreachable on this pipeline. Confirms unit #12's other claim: this file owns the four on-axis fields unit #12 was redundantly rewriting. Minted 6 VarPaths total (3 new, 3 reused, closing unit #12's largest dangling edge). JAX note: landed as `jnp.where` branch selection throughout, not `.at[].set()` as `next_steps.md` predicted — avoids reproducing `NeProfile`'s sequential-overwrite fragility. Also imports `physics/density_limit.py::calculate_greenwald_density_limit` (2-line pure `@staticmethod`) — the file's only reached method; not yet given its own row, see below. |
| 22 | `physics/superconductors.py` | `itersc`, `bi2212`, `jcrit_nbti`, `western_superconducting_nb3sn`, `jcrit_rebco`, `gl_nbti`, `gl_rebco` (~603 of 1289 LOC) | `functional_process/models/physics/superconductors.md` | **draft — 7/7 (+1 shared helper, `bottura_scaling`) ported, tier-1, tests passing incl. gradients**. **Corrects the registry's own "tier-2 leaning" expectation**: the file's only `scipy.optimize` call (`current_sharing_rebco`) is out of scope and calls `jcrit_rebco`, not the reverse — all 8 in-scope functions are ordinary closed-form algebra, `jnp.where` for every data-dependent branch, domain guards verified finite under `jax.jacfwd` at every branch boundary. **No cottax node written for any of the 8** — same reasoning as `coils.py`'s own three unwrapped functions: every real call site's arguments are locals inside `jcrit_from_material` (`coils/coils.py`, unit #10, still unported), not established `VarPath`s; a per-branch node sketch is left in the record for whoever ports `jcrit_from_material`. `i_tf_sc_mat` (`.tfcoil.i_tf_sc_mat`) formally characterized here as genuinely `split` (8 branches, reads-sets differ per branch — see the table below) — first full recording of this switch. **Real PROCESS bug found, not fixed**: `coils.py:136` calls `jcrit_rebco(t_helium, b_max, 0)` with 3 positional args, but `jcrit_rebco` takes exactly 2 — would `TypeError` if `i_tf_sc_mat == 6` ever executed; not reproduced or fixed here (out of file boundary), flagged for whoever ports `jcrit_from_material`. |
| 23 | `physics/impurity_radiation.py` | `ImpurityRadiation`, `create_f_rad_core_profile`, `calculate_impurity_radiation_power_density` (L379-755); **also** `calculate_average_charge_at_temp`/`_calculate_average_charge_at_temp_compiled` (L408-511, `@njit`) and `element2index` (L605-627) | `functional_process/models/physics/impurity_radiation.md` | **draft — 2/2 remaining functions ported** (`calculate_average_charge_at_temp`, `element2index`; the rest of the requested range was already ported by unit #20 into `radiation_power.py` — cross-referenced here, not duplicated, verified accurate against that record). No cottax node written — both are consumed inside unit #9 chunk B's (`physics_B_composition.py`) per-species/per-profile-point loops, never returning a single `VarPath` on their own; node ownership belongs to that chunk, which now imports and uses `calculate_average_charge_at_temp` directly (`element2index`'s results are resolved to Python-literal indices at port-write time, not called at trace time — see `physics_B_composition.py`'s own docstring). `element2index` is a graph-assembly-time lookup over a compile-time-constant label array, never traced. |
| 13 | `blankets/hcpb.py` | `nuclear_heating_blanket`, `nuclear_heating_magnets`, `nuclear_heating_shield` | `functional_process/models/blankets/hcpb.md` | **draft — 3/3 ported, tier-1, tests passing.** 3 `ExplicitFunction` nodes written (`NuclearHeatingBlanket`, `NuclearHeatingShield`, `NuclearHeatingMagnets`) but **deliberately not registered in `total_process.py`**: all three are reachable only from `stellarator.py`'s `blanket_neutronics()`, itself only called under `.fwbs.blktmodel == 1` (S2 of the `st_fwbs` synthesis — see the `stellarator.py` chunks table below) — a dispatch not yet built. Registering them now would also be a genuine graph error, not just premature: `NuclearHeatingMagnets` writes `.fwbs.p_tf_nuclear_heat_mw`, which `ScTfCoilNuclearHeating` (chunk 1F, already unconditional in `COMMON`) also writes — confirmed against `stellarator.py` (`blanket_neutronics()` calls `nuclear_heating_magnets()` *and* `sc_tf_coil_nuclear_heating_iter90()`, discarding the latter's own `p_tf_nuclear_heat_mw`-shaped output at that call site, while a *different* call site of `sc_tf_coil_nuclear_heating_iter90()` elsewhere in `st_fwbs` keeps it) that these are genuinely alternative producers gated by `blktmodel`, not a redundant pair — resolving this is S2's own `Switch` design, not a registration-pass decision. **A live call-site bug found, not fixed, directly relevant to whoever builds S2 next**: `blanket_neutronics()` calls `self.hcpb.nuclear_heating_blanket()` and `self.hcpb.nuclear_heating_shield()` with **zero arguments** (lines 440, 458), but both are `@staticmethod`s requiring 2 and 7 keyword arguments respectively — would `TypeError` the moment a `blktmodel == 1` stellarator run actually executes this path; not exercised by any existing test (`tests/unit/models/stellarator/test_stellarator.py` only ever sets `blktmodel=0`). Whoever picks up S2 will hit this immediately, since `blanket_neutronics()` is the first thing that arm calls. |
| 14 | `power.py` | `tfpwr` (→ `tfpwcall` → `tfcpwr`, ~341 LOC), `component_thermal_powers` (→ `plant_thermal_efficiency`, ~138 LOC), `calculate_cryo_loads` (→ `cryo`, ~81 LOC), `acpow`, `plant_electric_production` (→ `power_profiles_over_time`, ~195 LOC), `output_plant_electric_powers` | split into 3 chunks below, see table | **ported (3/3 chunks), consolidated and registered this pass** — see chunk table |

### `power.py` chunks (unit #14, chunked by sub-system, landed via a separate,
concurrently-running agent; registered and consolidated into `total_process.py` this pass)

| chunk | methods | record path | status |
|---|---|---|---|
| A | `tfpwr` (→ `tfpwcall` → `tfcpwr`) | `functional_process/models/power_A_tf_coil_power.md` | **draft — ported, registered.** `TfPowerResistive`/`TfPowerSuperconducting`, 2 `ExplicitFunction` nodes. The record's own note that this "looked like `vacuum.py`'s disjoint-output `i_vacuum_pumping` shape" is **corrected here**: both nodes share `.heat_transport.p_tf_electric_supplies_mw`, satisfying `check_arms_are_exclusive` — a real `.tfcoil.i_tf_sup` `Switch` (default `1`, `tfcoil_variables.py:261`) is now wired in `total_process.TOPOLOGY_SWITCHES`. `Power.tfpwr` dispatches on `i_tf_sup != 1` only, so values `0` (resistive copper) and `2` (aluminium) select the identical node — `2` is declared `unported` (pointing at `0`'s identical result) rather than duplicated, since a duplicate-node-set `Alternative` fails this project's own `test_arms_select_different_node_sets`. |
| B | `component_thermal_powers` (→ `plant_thermal_efficiency`, `plant_thermal_efficiency_2`), `calculate_cryo_loads` (→ `cryo`) | `functional_process/models/power_B_thermal_cryo.md` | **draft — ported, partially registered.** Largest chunk: 5 ported functions plus 6 `FixedPointFunction` splits landed this session for `calculate_component_thermal_powers`'s six genuine single-node self-loops (`.power.delta_eta`, `.heat_transport.eta_turbine`/`etath_liq`/`temp_turbine_coolant_in`, `.heat_transport.p_fw_div_heat_deposited_mw`, `.primary_pumping.p_fw_blkt_coolant_pump_mw` — `DeltaEtaStep`/`EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep`/`PFwDivHeatDepositedMwStep`/`PFwBlktCoolantPumpMwStep`), matching `next_steps.md` §5's Shape-B recipe. `ComponentThermalPowers` plus all six `FixedPointFunction`s are **registered** in `total_process.COMMON`. **Four more genuine self-loops found and confirmed this consolidation pass, not yet resolved**: `PlantThermalEfficiency`/`PlantThermalEfficiency2` (the raw, un-split functions `EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep` extract from) each independently own *and* read `eta_turbine`/`temp_turbine_coolant_in` (resp. `etath_liq`/`temp_turbine_coolant_in`) — `to_graph(PlantThermalEfficiency(...))` raises directly; and `Cryo`/`CryoLoads` (`.fwbs.qnuc`, plus `.power.qss`/`qac`/`qcl`/`qmisc` for `CryoLoads`) — confirmed the same way. All four left **ported but unregistered**, a second wave of the same Shape-B gap. **Real bug found, not fixed**: `power.py:2038` compares against a non-existent `ElectricConversionModelTypes` enum member (`SUPERCRITICAL_CO2_CYCLE`, the real name is `SUPERCRITICAL_CO2_BRAYTON_CYCLE`) — PROCESS itself raises `AttributeError` for `i_thermal_electric_conversion == 4`, unconditionally, before any physics runs; the port uses the correct member name, so this one value cannot be diffed against PROCESS's reference (PROCESS crashes there). |
| C | `acpow`, `plant_electric_production` (→ `power_profiles_over_time`) | `functional_process/models/power_C_electric_production.md` | **draft — ported, partially registered.** `Acpow`/`PowerProfilesOverTime` registered in `total_process.COMMON`. `PlantElectricProduction` (the real PROCESS caller of `power_profiles_over_time`, and the node with the rest of the plant electric-production outputs) is **not registered**: found this pass, it owns *and* reads five fields (`p_plant_electric_gross_mw`, `p_turbine_loss_mw`, `p_plant_electric_recirc_mw`, `p_plant_electric_net_mw`, `f_p_plant_electric_recirc`) — a third genuine self-loop alongside chunk B's four, confirmed directly via `to_graph`. `PowerProfilesOverTime`'s whole output set is a strict subset of `PlantElectricProduction`'s (real PROCESS call order: `plant_electric_production` calls `power_profiles_over_time` internally), so it stands in on its own for now. |
| 15 | `buildings.py` | `Buildings.run()` | `functional_process/models/buildings.md` | **reviewed — ported and registered.** All 3 functions tier-1 (`calculate_tf_coil_envelope`, `calculate_bldgs`, `calculate_bldgs_sizes`, plus `calculate_shield_height` extracted from the `bldgs` call site), 4 nodes (`TfCoilEnvelope`, `Bldgs`, `BldgsSizes`; `calculate_shield_height` stays inlined in `Bldgs.__call__`, not its own node). `TfCoilEnvelope` registered unconditionally in `COMMON` (its outputs feed both branches). **New genuine topology `Switch`**: `.buildings.i_bldgs_size` (`BuildingsModel`, default `0`/`ITER_1992`) — confirmed the two arms collide on `.buildings.a_plant_floor_effective`/`.volnucb` (both branches' real building-floor-area and nuclear-building-volume outputs), satisfying `check_arms_are_exclusive`; wired into `total_process.TOPOLOGY_SWITCHES` with `Bldgs` under `value=0` and `BldgsSizes(i_hcd_primary=5)` under `value=1` (`i_hcd_primary`'s own PROCESS default, kept as a static kwarg — an enum lookup that can't be traced, same shape as `EcrhDensityLimit`). **Real bug found, not fixed**: `calculate_bldgs_sizes`'s inboard/outboard hot-cell `hcomp_req_supply` divides `life_plant` by itself (`_safe_ratio(life_plant, life_plant)`), always evaluating to exactly `1.0` rather than scaling by a per-component replacement lifetime, unlike the divertor/centre-post calculations two sections later in the same function, which correctly divide by `life_div_fpy`/`cplife`. |
| 16 | `vacuum.py` | `Vacuum.run()` | `functional_process/models/vacuum.md` | **reviewed — ported, partially registered.** 3 units: `calculate_vacuum_pumping_simple` (tier-1, `VacuumPumpingSimple`), `solve_duct_diameter`/`duct_diameter_residual` (tier-2 internal Newton solve, no standalone node — internal helper of the next unit), `calculate_vacuum_pumping_old` (tier-2, the full `"old"` ETR duct-sizing model, `VacuumOld` — takes only raw `.build.*`/`.physics.*` fields, no blocking arguments). **`VacuumOld` registered unconditionally in `COMMON`**, matching PROCESS's own default (`.vacuum.i_vacuum_pumping = "old"`, `vacuum_variables.py:18`). **`.vacuum.i_vacuum_pumping` investigated as a candidate topology `Switch` and rejected**: unlike `i_bldgs_size`, `VacuumPumpingSimple` and `VacuumOld` own *completely disjoint* output sets (`n_iter_vacuum_pumps` vs. `n_vac_pumps_high`/`n_vv_vacuum_ducts`/`dlscal`/`m_vv_vacuum_duct_shield`/`dia_vv_vacuum_ducts` — no field in common), so `Switch.check_arms_are_exclusive` would raise (confirmed against `test_configuration.py::test_non_exclusive_arms_are_rejected`'s exact scenario) — this project's own switch machinery requires colliding ownership to prove two nodes are genuine alternatives, and these aren't provably that by output alone. `VacuumPumpingSimple` is left **ported but unregistered**; registering it unconditionally alongside `VacuumOld` would compute `n_iter_vacuum_pumps` even though PROCESS's own default configuration never reaches that branch. See `total_process.TOPOLOGY_SWITCHES`'s docstring for the same finding recorded next to `i_bldgs_size`'s working case, and `next_steps.md` §1 for this as a new instance of the "arms without a shared name" structural gap (alongside `blkttype`). **Also noted, not blocking**: `VacuumOld` is an `ExplicitFunction` despite being tier-2 (its Newton solve is fully self-contained inside a `jax.lax.while_loop`, no unknowns exposed to the graph) — `schema.md`'s stated convention is `ImplicitFunction`/`FixedPointFunction` for a self-contained tier-2 unit; not changed here (would be a real redesign, not a registration decision), flagged for whoever next touches this file. **Real findings, not bugs to fix**: PROCESS's own Newton loop verifiably stops ~0.16% short of its true root (not actionable); `pend`/`pstart`'s ratio is algebraically always `log(100)`, making two PROCESS inputs dead reads (already dropped from the port's signature). `VacuumVessel` (the file's second class) confirmed unreachable on the stellarator pipeline, no action needed. |
| 17 | `availability.py` | `Availability.run()`, dispatches on new switch `i_plant_availability` (see switches table) to one of: `avail()` (original scope), `avail_2()` (Morris, ~172 LOC), `avail_st()` (ST/spherical-tokamak, ~308 LOC) | `functional_process/models/availability.md` | **reviewed — ported, `Avail` now registered.** All 18 functions tier-1. **Bypass confirmed**: `Stellarator.run()`'s solve-time branch (`stellarator.py:175`) calls `self.availability.avail(output=False)` **directly, every solver iteration**, bypassing `.costs.i_plant_availability`'s dispatch entirely — only `Avail`'s nodes are ever relevant at solve time; `Avail2`/`AvailSt` are output/reporting-only, same treatment as `coils/output.py`, and stay unregistered. **The `.costs.cplife` self-loop that previously blocked `Avail`'s registration entirely is now resolved** (`next_steps.md` §5's Shape-B recipe, landed this session before this consolidation pass): `CplifeAvail`/`CplifeAvailSt` (`FixedPointFunction`s) isolate the self-reference, each duplicating the `.tfcoil.i_tf_sup` SC/resistive dispatch *inline* as a static `if` rather than consuming `CpLifetimeSuperconducting`/`CpLifetimeResistive` directly (so only one node ever owns `.costs.cplife`); `Avail`/`Avail2`/`AvailSt` themselves became ordinary `ExplicitFunction`s over every *other* output, reading `cplife` as a plain `Input`. **This pass registers `Avail(ibkt_life=0, itart=0)` + `CplifeAvail(i_tf_sup=1, itart=0)` unconditionally in `total_process.COMMON`** (matching the bypass: `Avail` is exercised regardless of `i_plant_availability`'s value, so it does not belong behind a `Switch`). **Deliberately still not registered**: `CpLifetimeSuperconducting`/`CpLifetimeResistive` (their sole would-be consumer, `CplifeAvail`, duplicates their dispatch inline instead of reading their `Output`s, specifically to avoid a `.costs.cplife` double-ownership conflict with `CplifeAvail` — registering the pair alongside `CplifeAvail` would reintroduce exactly that conflict); `WardTaylorAvailability` (PROCESS's own default `.costs.i_plant_availability = 2`/MORRIS means `avail()`'s internal `WARD_TAYLOR` branch, `i_plant_availability == 1`, never fires, so `.costs.f_t_plant_available` has no producer under the default configuration — unconditional registration would reproduce the `EcrhDensityLimit` bug class, and it cannot be `Switch`-wired either, no counterpart node exists for any other value); `Avail2`/`AvailSt`/`CplifeAvailSt` (unreachable during solve per the bypass, output-only). **Two real PROCESS bugs found, not fixed**: `avail_2`/`avail_st`'s total-availability formula uses `+` for a cross term where `avail()`'s own formula uses `-` (caught by the harness's own gradient/value checks, not by inspection); `avail_2`/`avail_st` unconditionally divide by `.costs.f_t_plant_available`, which can legitimately be exactly `0.0` — a real, fuzz-reproduced `ZeroDivisionError` risk on physically plausible inputs, mostly via `avail_st`'s maintenance-cycle model. **Also corrected**: `.physics.itart == 1` is not actually forbidden for a stellarator input (the registry's prior "provably dead" hypothesis for `avail_st` was too strong) — but this turns out moot given the bypass finding above: `avail_st` is unreachable during the solve regardless of `itart`. |
| 18 | `costs/costs.py`, `costs/costs_2015.py` | `Costs.run()`/`.output()` or `Costs2015.run()` — **which one is itself gated by `i_cost_model`, a switch; treat as two candidate units until that's resolved** | `functional_process/models/costs/costs.md`, `functional_process/models/costs/costs_2015.md` | **draft — ported (leaf pieces only), none registered.** `.costs.i_cost_model` (`CostModels`, default `1`/`KOVARI_2014`) confirmed as a genuine topology `Switch` in principle: it is resolved in exactly one place, `process/main.py`'s `Models.costs` `@property`, selecting a whole `Model` instance before any model runs, and is never read inside either `costs.py` or `costs_2015.py` itself. 23/43 `costs.py` methods ported (tier-1, 23 `ExplicitFunction` nodes — `StructuresCost`, `DivertorCost`, `VacuumSystemCost`, etc.), all genuine leaf sub-cost accumulators (each owns its own distinct `.costs.c2xx` field(s), no ownership conflicts among them). 2/13 `costs_2015.py` methods ported (`calculate_building_costs`, `calculate_land_costs`) but **no `cottax` node written for either** — plain functions only. **`i_cost_model` is NOT wired as a `Switch` this pass, despite looking like a strong candidate**: the two files' *full* models are genuinely disjoint subgraphs sharing only `.costs.coe`/`.costs.concost` (per `costs.md`'s own finding), but neither of those two shared fields is among what's actually ported on either side yet (only leaf sub-costs, not the top-level accumulation) — so the currently-ported subsets share **no** output at all, and `costs_2015.py` additionally has zero nodes to pair against regardless. `Switch.check_arms_are_exclusive` would reject this exactly as it does for `i_vacuum_pumping` (see unit #16's row). **Also not registered unconditionally**: PROCESS's own default is `i_cost_model = 1` (KOVARI_2014/`costs_2015.py`), so `costs.py`'s `Costs` model never even runs by default — putting its 23 leaf nodes unconditionally in `COMMON` would reproduce the same "computes a value the default configuration never computes" bug class as `EcrhDensityLimit`/`WardTaylorAvailability` above. All 23 nodes left **ported but unregistered**; revisit once either file's top-level `coe`/`concost` accumulation is ported, which is where the real cross-arm overlap lives. **Real bug found, not fixed**: `acc223`'s neutral-beam cost (`c2233`) is nested inside `if ifueltyp == 1` and never computed otherwise. |
| 19 | `physics/fusion_reactions.py` | `FusionReactionRate` (`.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`), `beam_fusion()`, `set_fusion_powers()` | `functional_process/models/physics/fusion_reactions.md` | **draft — ported**: `FusionRates` (fuses all three in-scope `FusionReactionRate` methods into one node, `ExplicitFunction`, ready for `Tier1Contract`) and `SetFusionPowers`, both registered in `total_process.COMMON`, tests passing incl. gradients. `beam_fusion()`/`beam_reaction_rate_coefficient()` stay audit-only — not blocked by entanglement but by a **new category**: the `scipy.integrate.quad` call is both non-JAX-traceable and, measured directly, only accurate to ~1e-6 relative even in PROCESS's own hands (fixed-quadrature replacements plateau at the same disagreement), four orders outside tier-1's tolerance. Scope check: the registry's stated method list matched exactly what `st_phys` (chunk 1B) calls, no correction needed. Open gap: `SetFusionPowers`'s input `.physics.p_beam_alpha_mw` has no producer node until `beam_fusion` is resolved. |
| 20 | `physics/radiation_power.py` | `calculate_radiation_powers()` | `functional_process/models/physics/radiation_power.md` | **draft — 3/3 functions ported**: `SynchrotronRadiationPower`, `ImpurityRadiationTotals`, `PlasmaRadiationPowers`, all registered in `total_process.COMMON` (`ImpurityRadiationTotals` needs a static `imp_indices` kwarg — assembled as all 14 species for the default graph, see that node's docstring and switches table). Found the missing `impurity_radiation.py` unit (now #23, above) and, independently, the same file via unit #21's own audit. |

## Constraints

**Updated — broadened to the full general-constraint set.** Every
`@ConstraintManager.register_constraint`-decorated function in
`process/core/solver/constraints.py` (~82 of them) is now audited and ported except 50
and 52 (IFE-only, `.ife.*` subsystem entirely unbuilt — see
`functional_process/core/solver/constraints.md`'s "Constraints considered and
excluded" section). Originally scoped to five stellarator-*specific* constraints
(bodies that read `.stellarator.*`/`istell`); broadened in a later pass to the
remaining ~77 general physics/engineering constraints (density limit, current-drive
power, TF coil stress, etc.) — these are not irrelevant to a real stellarator run
(most are almost certainly active via a stellarator `IN.DAT`'s `numerics.icc`), they
just weren't stellarator-*specific* in the narrower sense the first pass looked for.

### Stellarator-specific (bodies that read `.stellarator.*`/`istell`)

| id | note | record | status |
|---|---|---|---|
| 91 | the actual stellarator-specific constraint (`data.stellarator.powerht_constraint`/`powerscaling_constraint`); docstring literally says "stellarators only". Full audit in place, including a strong-evidence candidate iteration-variable pairing (`te0_ecrh_achievable`, ID 169). | `functional_process/core/solver/constraints.md` | **reviewed** (ported, tested) |
| 17 | general constraint (plasma radiation fraction limit) — **not** stellarator-specific, just has an `istell`-gated adjustment branch. Full audit in place: bare residual read, no hole-in-MDA, one open PROCESS-native `# TODO` reproduced as-is (not fixed). | `functional_process/core/solver/constraints.md` | **reviewed** (ported, tested) |
| 24 | beta upper limit — same shape as 17, a general constraint with an `istell`-gated branch (`istell != 0` silently overrides `i_beta_component` to always force the total-beta limit — a real PROCESS finding, documented not fixed). Direct iteration-variable match: `beta_total_vol_avg` (ID 5). | `functional_process/core/solver/constraints.md` | **reviewed** (ported, tested) |
| 82 | toroidal consistency of the stellarator build (`toroidalgap >= dx_tf_inboard_out_toroidal`) — genuinely stellarator-specific, no switch. Both operands already minted by `coils/calculate.py`'s ported `CoilCoilToroidalGap`/`CoilToroidalThickness`. No hole-in-MDA, no open questions. | `functional_process/core/solver/constraints.md` | **reviewed** (ported, tested) |
| 83 | radial consistency of the stellarator build (`available_radial_space >= required_radial_space`) — genuinely stellarator-specific, no switch. Both operands already minted by `stellarator/build.py`'s ported `Build` node. No hole-in-MDA, no open questions. | `functional_process/core/solver/constraints.md` | **reviewed** (ported, tested) |

This "general constraint + embedded istell branch" (17/24) is a third pattern beyond
"shared switch site" and "stellarator-only constraint" (82/83/91) — worth its own note
wherever `switches.md`'s `istell` entry lands.

### General (everything else PROCESS registers)

All ported this session, `Tier1Contract`-tested against PROCESS's real registered
functions via `ConstraintManager`, records in `functional_process/core/solver/
constraints.md`. See that file for full source/data-footprint/hole-in-MDA detail per
constraint — this table is a compact index, not a substitute.

| id | note | status |
|---|---|---|
| 1 | relationship between beta, temperature and density — the sole `Compare`-shaped constraint found (calls a re-derived `calculate_plasma_beta`); never active on a real stellarator run (`Stellarator.run()` overwrites the field directly) | **reviewed** |
| 2 | global power balance (total) | **reviewed** |
| 3 | global power balance for ions | **reviewed** |
| 4 | global power balance for electrons | **reviewed** |
| 5 | electron density upper limit | **reviewed** |
| 6 | epsilon beta-poloidal upper limit | **reviewed** |
| 7 | hot beam ion density consistency — shares `beta_beam`'s already-flagged conditional-producer gap (`beam_fusion` unported) | **reviewed** |
| 8 | neutron wall load upper limit | **reviewed** |
| 9 | fusion power upper limit | **reviewed** |
| 11 | radial build consistency (equality) — first equality constraint audited (needed `eq`) | **reviewed** |
| 12 | volt-second capability lower limit | **reviewed** |
| 13 | burn time lower limit | **reviewed** |
| 14 | NBI e-decay lengths to plasma centre (equality) | **reviewed** |
| 15 | L-H power threshold limit (H-mode enforcement) | **reviewed** |
| 16 | net electric power lower limit | **reviewed** |
| 18 | divertor heat load upper limit | **reviewed** |
| 19 | MVA (power) upper limit, resistive TF coil set | **reviewed** |
| 20 | neutral beam tangency radius upper limit | **reviewed** |
| 21 | minor radius lower limit | **reviewed** |
| 22 | L-H power threshold limit (enforce L-mode) | **reviewed** |
| 23 | conducting shell radius / rminor upper limit | **reviewed** |
| 25 | peak toroidal field upper limit | **reviewed** |
| 26 | Central Solenoid current density upper limit at end-of-flattop | **reviewed** |
| 27 | Central Solenoid current density upper limit at beginning-of-pulse | **reviewed** |
| 28 | fusion gain (big Q) lower limit | **reviewed** |
| 29 | inboard major radius consistency | **reviewed** |
| 30 | injection power upper limit | **reviewed** |
| 31 | TF coil stress/current-density limits (SCTF) -- a real finding — structurally vacuous for stellarators (see combined 31/32/33 entry — tokamak TF-coil model never runs when `istell != 0`) | **reviewed** |
| 34 | TF coil dump voltage upper limit | **reviewed** |
| 35 | TF coil J_wp upper limit for quench protection | **reviewed** |
| 36 | TF coil superconductor temperature margin lower limit | **reviewed** |
| 37 | current drive gamma upper limit | **reviewed** |
| 39 | first wall temperature upper limit | **reviewed** |
| 40 | auxiliary power lower limit | **reviewed** |
| 41 | plasma current ramp-up time lower limit | **reviewed** |
| 42 | cycle time lower limit | **reviewed** |
| 43 | average centrepost temperature consistency (TART) | **reviewed** |
| 44 | centrepost temperature upper limit (TART) | **reviewed** |
| 45 | edge safety factor lower limit (TART) | **reviewed** |
| 46 | I_p / I_rod upper limit (TART) | **reviewed** |
| 48 | poloidal beta upper limit | **reviewed** |
| 50 | IFE repetition rate upper limit — **not ported** | **reviewed** |
| 51 | startup flux equality | **reviewed** |
| 52 | IFE tritium breeding ratio lower limit — **not ported** | **reviewed** |
| 53 | fast neutron fluence on TF coil, upper limit | **reviewed** |
| 54 | peak TF coil nuclear heating, upper limit — real extraction gap — producer is inline arithmetic inside `Stellarator.st_fwbs`, never its own node | **reviewed** |
| 56 | Pₛₑₚ / R₀ upper limit | **reviewed** |
| 59 | neutral beam shine-through fraction upper limit | **reviewed** |
| 60 | Central Solenoid s/c temperature margin lower limit | **reviewed** |
| 61 | plant availability lower limit | **reviewed** |
| 62 | alpha-particle / energy confinement time ratio lower limit | **reviewed** |
| 63 | high-vacuum pump count upper limit (`i_vacuum_pumping = simple`) — reconsidered under the broadened scope — now ported (was excluded under the earlier stellarator-only pass, see below) | **reviewed** |
| 64 | plasma effective charge (Zeff) upper limit | **reviewed** |
| 65 | vacuum vessel stress on TF coil quench upper limit | **reviewed** |
| 66 | upper limit on rate of change of poloidal field energy | **reviewed** |
| 67 | simple upper limit on radiation wall load — real naming trap: `pflux_fw_rad_max_mw` is the value, `pflux_fw_rad_max` (no `_mw`) is the bound | **reviewed** |
| 68 | upper limit on Psep scaling (PsepBt / q95*A*R0) | **reviewed** |
| 72 | upper limit on Central Solenoid Tresca yield stress | **reviewed** |
| 73 | lower limit, separatrix power >= L-H threshold + auxiliary power | **reviewed** |
| 74 | upper limit on TF coil quench temperature (CroCo HTS only) | **reviewed** |
| 75 | upper limit on TF coil current / copper area (CroCo HTS only) | **reviewed** |
| 76 | upper limit, Eich critical separatrix density model — PROCESS's own source has an unresolved `# TODO` about a stray write; not reproduced as a write here | **reviewed** |
| 77 | maximum TF coil current per turn upper limit | **reviewed** |
| 78 | Reinke criterion, divertor impurity fraction lower limit | **reviewed** |
| 79 | maximum central solenoid (CS) field — real stale-copy-paste unit-tag bug (`"A/turn"`, actually a field in T) — reproduced faithfully | **reviewed** |
| 80 | lower limit on power crossing the separatrix | **reviewed** |
| 81 | lower limit ensuring central density exceeds pedestal density | **reviewed** |
| 84 | lower limit of plasma beta | **reviewed** |
| 85 | equality constraint for centrepost (CP) lifetime | **reviewed** |
| 86 | upper limit on TF winding-pack turn edge length | **reviewed** |
| 87 | TF coil cryogenic power upper limit | **reviewed** |
| 88 | TF coil vertical strain upper limit | **reviewed** |
| 89 | CS coil current / copper area upper limit | **reviewed** |
| 90 | CS coil stress load cycles lower limit — real PROCESS side-effect write (`.cs_fatigue.n_cycle_min`); handled via a local-only override, flagged as an open node-ownership question | **reviewed** |
| 92 | D/T/He3 fuel fraction consistency | **reviewed** |

**Excluded** (not ported, see `constraints.md`'s own section for the full reasoning):
50 and 52 (IFE-only, `.ife.*` subsystem entirely unbuilt anywhere in this codebase).

## Objective function

`process/core/solver/objectives.py`'s single `objective_function` — no stellarator/
`istell` special-casing anywhere (checked both by docstring and by grepping the
function body for `.stellarator.`/`istell` reads). Ported as sixteen standalone
`objective_metric_<id>` pure functions (one per `FiguresOfMerit` value) plus an
`OBJECTIVE_METRICS` lookup dict, deliberately not as one node or one traced dispatcher
— see `functional_process/core/solver/objectives.py`'s module docstring and
`_audit/next_steps.md` §6 for why branch selection is an assembly-time query, not a
port. All sixteen are `Tier1Contract`-tested against PROCESS's real `objective_function`
(150 passed, `--fp-gradients`). Two real PROCESS docstring inaccuracies found (not code
bugs, see `objectives.md`): ids 16 and 19 are listed identically as "major radius/burn
time" in `objective_function`'s inline docstring but compute genuinely different
formulas; id 15's precondition docstring is imprecise about which
`i_plant_availability` values are valid. Several branches' fields have no `Output`
producer yet in this codebase (`rmajor`, `b_plasma_toroidal_on_axis`,
`p_plant_electric_net_mw`, `pf_power.srcktpm`, the `costs.py`/`times.py` fields) — see
`objectives.md`'s hole-in-MDA table; `rmajor`/`b_plasma_toroidal_on_axis` are plausibly
genuine free design/iteration variables rather than porting gaps, not confirmed this
pass.

| unit | files | record | status |
|---|---|---|---|
| objective function | `functional_process/core/solver/objectives.py`/`test_objectives.py` | `functional_process/core/solver/objectives.md` | **reviewed** (ported, tested) |

## Switches

All switches read directly inside `process/models/stellarator/stellarator.py`, found by
grepping `self.data.<area>.i_*` (mechanical, not yet a per-branch reads-set diff — that's
the pilot's job for these).

| switch | area | record | status |
|---|---|---|---|
| `istell` | stellarator | `functional_process/core/solver/switches.md` | **draft — pilot** (master pipeline switch — tokamak/stellarator/IFE split, not expected to need a "split decision" in the formula sense, it's the top-level pipeline selector) |
| `i_tf_sup` | tfcoil | `total_process.TOPOLOGY_SWITCHES` | **resolved and implemented — now a real `Switch`, wired via `power_A_tf_coil_power.py` (unit #14 chunk A), not `availability.py`.** `TfPowerResistive`/`TfPowerSuperconducting` share `.heat_transport.p_tf_electric_supplies_mw`, satisfying `check_arms_are_exclusive`; wired with `default=1` (`tfcoil_variables.py:261`, SUPERCONDUCTING). `Power.tfpwr` dispatches on `i_tf_sup != 1` only, so value `2` (aluminium) is declared `unported` pointing at value `0`'s identical resistive result, rather than duplicated (a duplicate-node-set `Alternative` fails `test_arms_select_different_node_sets`). **`availability.md`'s own candidate pairing is still NOT wired**: `CpLifetimeSuperconducting`/`CpLifetimeResistive` (both own `.costs.cplife`) remain unregistered even though their blocker is gone (`Avail` is registrable now, via `CplifeAvail` — see unit #17's row) — `CplifeAvail.step` duplicates the `i_tf_sup` dispatch *inline* instead of consuming this pair specifically to keep `.costs.cplife` single-owned; registering `CpLifetimeSuperconducting`/`CpLifetimeResistive` alongside `CplifeAvail` would conflict. Two independent `Switch` objects sharing this one `path` coexist fine in `TOPOLOGY_SWITCHES` (`Switch.path` is a lookup key, not a unique identity — `configuration.py`'s own docstring), so wiring this second pairing later, once/if `.costs.cplife`'s ownership question is revisited, needs no new machinery. Also still used as a plain traced (non-static) argument elsewhere (`buildings.py`'s `BldgsSizes`, `jnp.where`-selected) — this switch is genuinely inconsistent in treatment across ported units, not yet resolved as one policy. |
| `i_blkt_coolant_type` | fwbs | ″ | **draft — pilot** |
| `i_thermal_electric_conversion` | fwbs | ″ | **resolved as static kwarg, reads-set genuinely differs but not split** (`power_B_thermal_cryo.md`, unit #14 chunk B): `calculate_plant_thermal_efficiency`'s 5 branches each touch a different subset of fields, textbook "split" case, **not followed** — kept static, same policy deviation as `i_confinement_time`/`i_plasma_ignited`. Default `0` (`CCFE_HCPB_VALUE`, `fwbs_variables.py:264`) — under this default the branch computes `eta_turbine = 0.411` unconditionally (real formula, not a pass-through). **Real PROCESS bug found, not fixed**: `power.py:2038` compares against a non-existent enum member (`SUPERCRITICAL_CO2_CYCLE`), so PROCESS itself crashes with `AttributeError` for `i_thermal_electric_conversion == 4` before any physics runs; the port uses the correct member name, so this value can't be diffed against a PROCESS reference. |
| `i_plasma_ignited` | physics | ″ | **draft — reads-set evidence in from two units.** `confinement_time.md`: differs by exactly one term (`p_hcd_injected_total_mw`, added only when `NON_IGNITED`) — kept static on `ConfinementTime`. `physics_B_composition.md`: `plasma_composition`'s two branches genuinely differ (`f_nd_beam_electron`/`nd_plasma_electrons_vol_avg` vs. nothing) — `traceability_policy.md`'s default ("split") technically applies in both cases but was **not followed**, kept as a static `bool`/`int` in both ports instead (2 lines deep inside an otherwise-shared body in each case). Flagged as a deliberate policy deviation in both records — see `next_steps.md` §1 for the open question this raises about a size/entanglement-aware exception to the split default. |
| `i_beta_fast_alpha` | physics | ″ | **draft — pilot, resolved as static kwarg.** `physics_A_pure_formulas.md`: both `fast_alpha_beta` branches read the identical 6 variables, differing only in 2 coefficients and a `sqrt` guard — textbook "identical reads-set" exception, kept as `FastAlphaBeta(i_beta_fast_alpha=...)`, not a `Switch`. Default `1` (`physics_variables.py:875`). |
| `i_pflux_fw_neutron` | physics | ″ | **resolved as static kwarg** (`stellarator_B_st_phys.md`, chunk 1B): a 3-way dispatch jointly with `.heat_transport.ipowerflow` inside `calculate_neutron_wall_load`/`calculate_radiated_wall_load_and_fraction` (`NeutronWallLoad`/`RadiatedWallLoadAndFraction`). Default `1` (`physics_variables.py:1006`) — at this value both functions take their first branch unconditionally, making `ipowerflow`'s value inert for the actual result (still required as a field). Registered unconditionally in `COMMON` with `i_pflux_fw_neutron=1, ipowerflow=1`. |
| `i_confinement_time` | physics | ″ | **draft — reads-set evidence in.** `confinement_time.md`: genuinely differs per value (each of the 48 scaling laws takes a different argument subset) — textbook "split" case by `naming_convention.md`'s own rule, **not followed**: kept as one composite dispatcher/node (`ConfinementTime(i_confinement_time=...)`, static kwarg) matching PROCESS's own granularity, since 51 separate `Alternative` nodes for one dispatcher is a real, unresolved design question (see `next_steps.md` §1), not obviously better. Default `34` (IPB98(y,2), `physics_variables.py:962`). |
| `i_rad_loss` | physics | none yet | **new, found by `confinement_time.md`'s audit.** 3-way (`FULL_RADIATION`/`CORE_ONLY`/`NO_RADIATION`), reads-set differs per value — same "genuinely differs but kept static, not split" treatment as `i_confinement_time`, on the same `ConfinementTime` node. Default `1` (`physics_variables.py:954`). |
| `i_p_coolant_pumping` | fwbs | ″ | **resolved as static kwarg** (`power_B_thermal_cryo.md`, unit #14 chunk B): gates several conditional-ownership pass-throughs inside `calculate_component_thermal_powers` (`.primary_pumping.p_fw_blkt_coolant_pump_mw`, `.heat_transport.p_fw_div_heat_deposited_mw`, both now their own `FixedPointFunction`s, `PFwBlktCoolantPumpMwStep`/`PFwDivHeatDepositedMwStep`) plus ordinary formula branches on `ComponentThermalPowers`/`DeltaEtaStep`. Default `2` (`MECHANICAL`, `fwbs_variables.py:249`). |
| `i_cost_model` | costs | ″ | **confirmed genuine topology switch in principle, but not wireable yet.** `costs.md`: resolved in exactly one place, `process/main.py`'s `Models.costs` `@property`, selecting a whole `Model` instance (`costs.py`/`costs_2015.py`/user-provided) before any model runs; never read inside either file. The two files' *full* models are genuinely disjoint subgraphs sharing only `.costs.coe`/`.costs.concost`. **Not wired into `total_process.TOPOLOGY_SWITCHES` this pass**: neither shared field is among what's actually ported yet on either side (23 leaf sub-costs in `costs.py`, 0 nodes at all in `costs_2015.py`), so `Switch.check_arms_are_exclusive` would reject it exactly as it does for `i_vacuum_pumping` (see unit #16/#18's rows). Default `1` (`KOVARI_2014`/`costs_2015.py`, `cost_variables.py:327`). |
| `i_plasma_pedestal` | physics | `total_process.TOPOLOGY_SWITCHES` | **resolved and implemented** — genuinely topology-changing (`plasma_profiles.py`/`profiles.py`, see units #12/#21), *and* the value-0 requirement `density_limits.EcrhDensityLimit` already enforced as a static kwarg is not independent of it: PROCESS's own `st_d_limit_ecrh` has no formula for `i_plasma_pedestal != 0` (`density_limits.py:146-150`, the `else` arm only logs an error). Both facts are captured by one `Switch` — `EcrhDensityLimit(i_plasma_pedestal=0)` lives inside the `value == 0` `Alternative`'s declarations, so there is one place the two ever agree, no separate cross-check mechanism needed. Default `1` (`physics_variables.py:889`) — note this means `EcrhDensityLimit`'s outputs are **not produced** in PROCESS's own default `GRAPH`, correcting the previous unconditional-in-`COMMON` placement, which was wrong for the default configuration. |
| `i_nd_plasma_pedestal_separatrix` | physics | none yet | **new, found by unit #21's audit** — nested under `i_plasma_pedestal == PEDESTAL_PROFILE`, gates `NeProfile.set_pedestal_and_separatrix_values` (`GreenwaldDensityFractions`/`PedestalSeparatrixDensities`). Moot for this scope: that method is reachable only from `physics.py` (unit #22, tokamak), never from `stellarator.py` — ported but not registered in the stellarator graph. Second instance of the "nested switch" gap `irefprop` already flagged. |
| `i_tf_sc_mat` | tfcoil | none yet | **draft — reads-set evidence now in, from unit #22's own audit** (`superconductors.md`, completing `coils.md`'s provisional finding). 8-way dispatch inside `coils/coils.py::jcrit_from_material` (unit #10) into `physics/superconductors.py` (unit #22): branch 4 alone reads `.tfcoil.b_crit_sc`/`t_crit_sc`, branch 7 alone reads `.tfcoil.b_crit_upper_nbti`/`t_crit_nbti`, branches 1/3/5/8 use fixed literals, branches 2/6 use neither — reads-sets genuinely differ. **Split decision: split, confidence: high** — one node per `i_tf_sc_mat` value once unit #10 mints real `VarPath`s for `jcrit_from_material`'s locals (`b_max`, `t_helium`, per-branch `bc20m`/`tc0m`); a sketch is left in `superconductors.py`. A second, inconsistent dispatcher (`superconductor_current_density_margin`, out of scope) maps a different, disagreeing subset of the same `SuperconductorModel` enum from PF-coil/tokamak-TF-coil callers — evidence `i_tf_sc_mat`'s valid range is caller-dependent, not a bug in this unit. Still blocks unit #10/#22's own wiring, transitively unit #9's `winding_pack_total_size`. **Real PROCESS bug found**: `coils.py:136` calls `jcrit_rebco(t_helium, b_max, 0)` — 3 positional args against a 2-argument signature; would `TypeError` if `i_tf_sc_mat == 6` executed. Not fixed (out of `superconductors.py`'s file boundary). |
| `i_plant_availability` | availability | none yet | **resolved as moot for this scope, not a `Switch`.** `Availability.run()`'s own dispatch (which honours this switch, reaching `avail`/`avail_2`/`avail_st`) is only called once, post-solve, from `Stellarator.output()`. `Stellarator.run()`'s solve-time branch (`stellarator.py:175`) calls `self.availability.avail(output=False)` **directly**, bypassing this switch entirely — confirmed by reading the call site, matches the source's own `# TODO: should availability.run be called rather than availability.avail?`. So for the solve-time graph this switch never actually gates anything: only `avail()`'s nodes are reachable regardless of its value. Not wired as a `Switch` — see unit #17's row for why `avail()`'s own nodes still can't be registered (unrelated self-loop on `.costs.cplife`). Default `2` (`MORRIS`, `cost_variables.py:408`) — note `avail()` is still what runs at solve time even under this default, since the bypass ignores the switch's value. |
| `i_bldgs_size` | buildings | `total_process.TOPOLOGY_SWITCHES` | **resolved and implemented** — `buildings.md`: genuinely topology-changing, `Bldgs`/`BldgsSizes` are two disjoint building-size estimation models that collide on `.buildings.a_plant_floor_effective`/`.volnucb` (satisfying `check_arms_are_exclusive`). Wired: `value=0` (`ITER_1992`, PROCESS's own default, `buildings_variables.py:206`) → `Bldgs`; `value=1` (`CHAPMAN_2024`) → `BldgsSizes(i_hcd_primary=5)` (`.current_drive.i_hcd_primary`'s own default, kept static — an enum lookup, not traceable). |
| `i_vacuum_pumping` | vacuum | none — investigated, not wireable | **new, found this wave — investigated as a candidate `Switch`, rejected.** String-valued (`"old"`/`"simple"`, default `"old"`, `vacuum_variables.py:18`), dispatches `Vacuum.run()` to `VacuumOld`/`VacuumPumpingSimple`. Unlike `i_bldgs_size`, the two arms own **completely disjoint** output sets — no field in common at all — so `Switch.check_arms_are_exclusive` rejects the pairing (verified directly, see `total_process.py`'s `TOPOLOGY_SWITCHES` docstring and unit #16's row). `VacuumOld` (the default) registered unconditionally in `COMMON` instead; `VacuumPumpingSimple` stays ported-but-unregistered. First confirmed instance, alongside `i_cost_model` (unit #18), of "genuinely mutually exclusive per PROCESS's own `if`/`elif`, but not provably so by output ownership" — a real, so-far-unresolved gap in this project's switch machinery, distinct from the already-tracked `blkttype`/nested-switch gaps in `next_steps.md` §1. |
| `.fwbs.blktmodel` (joint with `.heat_transport.ipowerflow`) | fwbs | `total_process.TOPOLOGY_SWITCHES` | **resolved and implemented — 2/3 arms.** S2's real dispatch (`stellarator_fwbs_s2.md`, `stellarator_E_fwbs_synthesis.md`) is `if blktmodel == 1: ... else: if ipowerflow == 0: ... else: ...` — a genuine *joint* two-switch dispatch, not a single `blktmodel` enum: `blktmodel != 1` splits further by `ipowerflow`. Modelled as one `Switch` with a synthetic composite `path` (`".fwbs.blktmodel,.heat_transport.ipowerflow"`, not a real single `.area.field` — `configuration.py`'s own `path` is a lookup key, not a `VarPath`, so this is within its stated contract) and an arm-index `value` (0/1/2, documented per-`Alternative`), rather than two nested binary `Switch`es, since the two `!= 1` arms are genuinely mutually exclusive with each other in the `check_arms_are_exclusive` sense (share 4 outputs) while `blktmodel == 1` vs. `!= 1` is a different axis. `value=0` (blktmodel==1) stays `unported` — blocked on unit #13's live call-site bug. `value=1` (blktmodel!=1, ipowerflow==0) → `ExponentialAttenuationBlanketShieldPower` + `ScTfCoilNuclearHeating`. `value=2` (blktmodel!=1, ipowerflow==1, **PROCESS's real default**) → `DetailedPowerflowBlanketShieldPower`. Default `2`. **Found and fixed a real registration bug while wiring this**: `ScTfCoilNuclearHeating` was unconditionally in `COMMON`, correct only for `value=1` — PROCESS's actual default lands in `value=2`, which computes a *different* `.fwbs.p_tf_nuclear_heat_mw` formula; moved into `value=1`'s `Alternative`. |
| `.physics.itart` | physics | none yet | **new, found by `hcpb.md`'s audit** — read by `nuclear_heating_shield` (1 branch) and `nuclear_heating_magnets` (2 more), all 2-way. Reads-sets genuinely differ per branch (nominally calls for split), **not split**: PROCESS's own staticmethod signatures already unify both branches under one plain `int` parameter, and the port's job is to reproduce that shape, not redesign it — kept as an ordinary traced `jnp.where` argument in all three ported functions (not even a static kwarg, since `itart` is a genuine `InputVariable` a stellarator IN.DAT could set either way, not fixed by device mode — `stellarator.py` itself never reads `.physics.itart`, confirmed by grep, so nothing forces it to a particular value on this pipeline). **Not topology-changing** on current evidence; recommendation is keep-static/keep-traced, not a `Switch`. Split decision: keep-static (as an ordinary argument, not even a switch-shaped one). Confidence: medium (no counter-evidence found, but not exhaustively checked against every stellarator input file). |

## Pilot batch (this dispatch)

3 fork agents, one per row marked "draft — pilot" above:
1. Model-unit #3 (`density_limits.py`) — smallest, self-contained, stellarator-specific:
   representative of the common case.
2. Constraint 17 — the only stellarator-specific constraint, tests the hole-in-MDA
   judgment call end to end.
3. Switch batch (all 10 switches above) — tests the site-enumeration + reads-set-diff
   workflow at once, since these are cheap individually.

Purpose: validate the schema and naming convention against real material before
committing to units 1-2, 4-18 (the remaining ~18 units).

## Ported so far

- Unit #3's two **tier-1** functions: `st_sudo_density_limit` and `st_d_limit_ecrh`,
  ported in `functional_process/models/stellarator/density_limits.py`, validated by
  `test_density_limits.py`. Its record stays `draft` — the port covers the tier-1 half,
  while `power_at_ignition_point` (tier 2) is still blocked on unit #1's `st_phys`
  (now unblocked, see 1B — not yet ported).
- Chunk **1D** (`calculate_structure_masses`,
  `calculate_intercoil_mass_scaling_reference`) and chunk **1F**
  (`calculate_sc_tf_coil_nuclear_heating`, SC branch) — both fully tier-1, both records
  bumped to `reviewed` since a passing test suite is stronger evidence than an
  unreviewed audit claim. Each record now also carries a `## cottax node` section
  (`CallableNode` wrap of the ported `fn`), per `schema.md`'s new template section —
  see `_harness/varpath.py` for the `.area.field` string -> `VarPath` helper the wrap
  uses.
- **This snapshot's wave**: unit #21 (`physics/profiles.py`, 12/12 functions, 4 nodes
  unconditional + 6 gated by `i_plasma_pedestal`), unit #19 (`physics/fusion_reactions.py`,
  `FusionRates`/`SetFusionPowers`), unit #20 (`physics/radiation_power.py`,
  `SynchrotronRadiationPower`/`ImpurityRadiationTotals`/`PlasmaRadiationPowers`), and
  unit #10's `intersect` (`coils/coils.py`, self-contained tier-2, `Tier2Contract`'s
  first real exercise, no node wrap yet — see that row). **`i_plasma_pedestal` is now a
  real `Switch`** in `total_process.TOPOLOGY_SWITCHES`, resolving unit #12's/#21's
  two-roles blocker; `EcrhDensityLimit` moved out of `COMMON` into that switch's
  `value == 0` arm (PROCESS's own default is `value == 1`, so the default `GRAPH` no
  longer produces `dlimit_ecrh`/`bt_max_ecrh` — a correctness fix, not a regression: see
  switches table). Default graph is now 44 nodes (was 32), still exactly one SCC
  (`Divertor`/`AFwTotalWithPowerflow`, only under `ipowerflow != 0`).
- **This snapshot's wave** (5-agent dispatch, `next_steps.md` §4b, now consolidated):
  unit #13 (`blankets/hcpb.py`, 3/3 functions ported, 3 nodes written but **not**
  registered — see that row and `total_process.py`'s docstring for the
  `p_tf_nuclear_heat_mw` ownership conflict with chunk 1F's `ScTfCoilNuclearHeating`);
  unit #22 (`physics/superconductors.py`, 7/7 + 1 shared helper ported, no nodes yet —
  blocked on unit #10's own wiring); unit #23 (`physics/impurity_radiation.py`, 2/2
  remaining functions ported, no nodes — consumed by unit #9 chunk B); unit #9
  (`physics/physics.py`, 8/8 in-scope methods ported across 3 chunks — see the new
  chunk table above — 8 nodes registered, `plasma_composition` deliberately not
  node-wrapped, a genuine `.physics.first_call` self-loop); units #10/#11
  (`physics/confinement_time.py`, `physics/exhaust.py`, both fully ported, 3 nodes
  registered — `IterPhysicsBasisElongation` written but deliberately not registered,
  duplicate producer of `.physics.kappa_ipb`). **A real defect was found and fixed
  during this consolidation pass**: `confinement_time.py`'s `ConfinementTime` node had
  `i_confinement_time`/`i_rad_loss` as plain (non-static) annotated fields and
  `i_plasma_ignited` as an ordinary `Input`, none of which survive `jax.jit`/`jacfwd`
  tracing given `calculate_confinement_time`'s internal Python `if`/`elif` dispatch on
  all three — fixed to `eqx.field(static=True)` throughout, verified directly with
  `eqx.filter_jit` + `jax.jacfwd`. Default graph is now **55 nodes** (was 44), still
  exactly one SCC (`Divertor`/`AFwTotalWithPowerflow`, only under `ipowerflow != 0`).
  Four real PROCESS bugs found this wave, all reproduced faithfully rather than fixed:
  `blanket_neutronics()`'s two zero-argument `hcpb` calls (unit #13), `jcrit_from_material`'s
  3-positional-arg call into a 2-argument `jcrit_rebco` (unit #22), and
  `calculate_confinement_time`'s `KAYE_GOLDSTON` scrambled positional call plus its
  `USER_INPUT`/`PAZ_SOLDAN_NT` dead branches (unit #10) — see each unit's row for detail.
- **This snapshot's wave** (4-agent dispatch, `next_steps.md` §4c, now consolidated):
  `buildings.py` (unit #15, 3/3 functions ported, 3 nodes, **all registered** — new
  `i_bldgs_size` topology `Switch`); `vacuum.py` (unit #16, 3 units ported, `VacuumOld`
  registered unconditionally in `COMMON`, `VacuumPumpingSimple` ported but unregistered —
  `i_vacuum_pumping` investigated and rejected as a `Switch`, disjoint-output arms);
  `availability.py` (unit #17, 18/18 functions ported, 6 nodes written, **none
  registered** — `Avail`/`Avail2`/`AvailSt` each fail `cottax` node construction outright
  with a genuine self-loop on `.costs.cplife`, confirmed directly, not merely audited;
  this corrects the dispatch's own suggested "register `avail`'s nodes in `COMMON`
  unconditionally" framing, which turned out to describe a real bypass mechanism
  (confirmed: `Stellarator.run()` calls `self.availability.avail()` directly, bypassing
  `i_plant_availability`) but not a registrable node); `costs/costs.py` +
  `costs/costs_2015.py` (unit #18, 23/43 + 2/13 methods ported, 23 nodes written in
  `costs.py`, none registered — `i_cost_model` confirmed a genuine topology switch in
  principle but not wireable yet, since the currently-ported subsets share no output).
  Default graph: 55 -> **58 nodes** (`TfCoilEnvelope`, `VacuumOld` unconditional;
  `Bldgs`/`BldgsSizes` under the new switch), still exactly one SCC
  (`Divertor`/`AFwTotalWithPowerflow`, only under `ipowerflow != 0`). **Two new
  structural findings, not resolved**: (1) a *third and fourth* instance of the
  missing-`Cut` self-loop gap `next_steps.md` §5 tracks (`Avail`/`Avail2`'s
  `cplife`/`cplife_in` pair, `AvailSt`'s bare `cplife`), found by directly attempting
  `to_graph` construction rather than by audit reasoning alone — `cottax` itself raises
  `ValueError: reads ['.costs.cplife'], which it also owns`; (2) a new
  "genuinely-exclusive-by-PROCESS's-own-`if`/`elif`-but-not-provably-so-by-output-
  ownership" gap in `Switch.check_arms_are_exclusive`, hit by both `i_vacuum_pumping`
  (`VacuumOld`/`VacuumPumpingSimple` share no output at all) and `i_cost_model` (the
  currently-ported subsets of `costs.py`/`costs_2015.py` share no output either, though
  the *full* models would). Registering unconditionally in either case was rejected too:
  it would reproduce the `EcrhDensityLimit`-class bug of computing a value PROCESS's own
  default configuration never computes (`WardTaylorAvailability` under
  `i_plant_availability`'s non-default `WARD_TAYLOR` value; `costs.py`'s 23 leaf nodes
  under `i_cost_model`'s non-default `PROCESS_1990` value). Three real PROCESS bugs found
  this wave, all reproduced faithfully rather than fixed: `calculate_bldgs_sizes`'s
  self-divided `hcomp_req_supply` (unit #15); `avail_2`/`avail_st`'s `+`-for-`-` sign
  error in the total-availability cross term, plus their unconditional divide-by-
  `f_t_plant_available` (a real, fuzz-reproduced `ZeroDivisionError` risk, unit #17);
  `acc223`'s neutral-beam cost (`c2233`) computed only when `ifueltyp == 1` (unit #18,
  in not-yet-ported code, found during audit).

Porting a unit's tier-1 functions is not the same as its audit reaching `reviewed` in
general (see density_limits.py, still `draft` — a unit with a still-open tier-2 half
doesn't get bumped), but a unit that is *entirely* tier-1 and fully ported, as 1D/1F are,
does move to `reviewed`.

**Standing practice going forward**: a chunk confirmed tier-1 *and* self-contained (not
flagged as spanning an unresolved boundary, e.g. 1E1-1E3) should be ported — function +
`ExplicitFunction` node (real code, `cottax.interfaces.pytree_namespace_module`, see
`schema.md`) + harness test class — as part of finishing that chunk's audit, not queued
for a later pass. A self-contained tier-2 chunk (internal solve, calls no other
not-yet-ported unit) gets the same treatment using `ImplicitFunction`/`FixedPointFunction`
instead of `ExplicitFunction`. Entangled or tier-3 chunks (or a tier-2 chunk that calls
into another unported unit, e.g. `power_at_ignition_point` → `st_phys`) stay audit-only
until their boundary/dependency is resolved.

Every ported node is registered in `functional_process/total_process.py` — running
`python -m functional_process.render_xdsm` draws the current graph to
`functional_process/xdsm.html` for visual inspection. Both are meant to be kept current as
units land, not rebuilt from scratch each time.

**There is no longer a single graph.** A node whose existence depends on a
topology-changing switch is registered as an `Alternative` under that switch in
`total_process.TOPOLOGY_SWITCHES`, not in `COMMON`; `graph_for(configuration)` assembles
the arms one configuration selects, and `GRAPH` is what PROCESS's own switch defaults
produce. See `functional_process/configuration.py` for why assembly time is the only
correct place to resolve a switch (no switch in PROCESS is ever an iteration variable or a
scan variable, so none can change between two evaluations of one graph). A ported arm that
is not reachable from any configuration is dead code, and `test_configuration.py` fails on
it — which is the state `LowhybHeating` and `AFwTotalNoPowerflow` were in before this
existed.

- **This snapshot's wave (registration/consolidation pass over 9 already-ported-but-
  unregistered units).** `stellarator_B_st_phys.py` (chunk 1B, 7/8 nodes: `TotalField`,
  `PoloidalFieldFromRotationalTransform`, `FusionPowerTotalsMw`,
  `NeutronWallLoad(i_pflux_fw_neutron=1, ipowerflow=1)`,
  `HeatingAndRadiationPower(i_plasma_ignited=0)`,
  `RadiatedWallLoadAndFraction(i_pflux_fw_neutron=1, ipowerflow=1)`,
  `ThermalEnergyTotals` — `StellaratorBetaAndRhoStar` left out, its `rho_star` is
  algebraically identical to `DimensionlessPlasmaParameters`'s, a real
  redundant-duplicate-write in PROCESS itself); `stellarator_C_geometry.py` (chunk 1C,
  3/3: `DefaultAspectRatio`, `StellaratorScalingFactors`, `StellaratorPlasmaGeometry`,
  bumped to `reviewed`); `stellarator_fwbs_s2.py` (S2, 2/3 arms, new joint
  `.fwbs.blktmodel,.heat_transport.ipowerflow` `Switch` — see switches table); unit #9's
  `coils/calculate.py` (`WindingPackJTfWp` only — `WindingPackTotalSize` was replaced
  mid-session by a concurrently-running agent's `intersect` wiring,
  `WindingPackIntersectInputs`/`coils.py`'s `Intersect`/`WindingPackTotalSizePost`, not
  independently audited or registered this pass, see `next_steps.md` §7);
  `power_A_tf_coil_power.py`/`power_B_thermal_cryo.py`/`power_C_electric_production.py`
  (unit #14, chunks A/B/C — see the new chunk table under row #14: `TfPowerResistive`/
  `TfPowerSuperconducting` under a new `.tfcoil.i_tf_sup` `Switch`;
  `ComponentThermalPowers` + 6 `FixedPointFunction`s; `Acpow`/`PowerProfilesOverTime`);
  `availability.py` (unit #17, `Avail` + `CplifeAvail` now registered, its `.costs.cplife`
  self-loop resolved via `next_steps.md` §5's recipe); `physics_B_composition.py` (chunk
  B, `NextFirstCall` + `PlasmaComposition`, both Shape-B self-loops this chunk had
  resolved — see row B's own entry for the second one, a per-index-`VarPath` fix needing
  no `Cut`/`FixedPoint` at all). Default graph: 61 → **94 nodes**.
  - **New genuine cross-node Shape-A cycle found, second confirmed instance**:
    `DensityProfile → FusionRates → PlasmaComposition → PedestalOnAxisDensities →
    DensityProfile`, surfaced only once `PlasmaComposition` joined the full graph — a
    real density/composition/fusion-rate/pedestal feedback loop, not an artefact.
    `next_steps.md` §5's "only one genuine cross-subsystem cycle across ~60 nodes"
    finding is now superseded — two are confirmed across 94.
  - **Two more registration bugs found and fixed this pass, same class as
    `EcrhDensityLimit`'s earlier fix** (a node unconditionally in `COMMON` whose formula
    is wrong for PROCESS's own default configuration): `ScTfCoilNuclearHeating` (moved
    from unconditional `COMMON` into the new joint switch's `value=1` arm — PROCESS's
    real default lands in `value=2`, a different formula) and `BlktmodelBlanketThickness`
    (removed from `COMMON` entirely — PROCESS's own default `blktmodel = 0` means this
    node's own docstring says it must not be instantiated at all; found while checking
    the precedent for `DefaultAspectRatio`'s own conditional-ownership treatment).
  - **A second wave of Shape-B self-loops found, not yet resolved** (beyond the ones this
    session's own Shape-B wave already fixed): `PlantThermalEfficiency`/
    `PlantThermalEfficiency2` (`power_B_thermal_cryo.py`, each independently owns *and*
    reads `eta_turbine`/`temp_turbine_coolant_in` or `etath_liq`/
    `temp_turbine_coolant_in` — superseded for graph purposes by the `*Step`
    `FixedPointFunction`s extracted from them, not usable standalone); `Cryo`/`CryoLoads`
    (`power_B_thermal_cryo.py`, `.fwbs.qnuc` plus, for `CryoLoads`, `.power.qss`/`qac`/
    `qcl`/`qmisc`); `PlantElectricProduction` (`power_C_electric_production.py`,
    `.heat_transport.p_plant_electric_gross_mw`/`p_plant_electric_recirc_mw`/
    `p_plant_electric_net_mw`, `.power.p_turbine_loss_mw`/`f_p_plant_electric_recirc`).
    All confirmed by direct `to_graph` construction, not audit reasoning alone. Left
    ported-but-unregistered.
  - Concurrent, unrelated sessions landed mid-pass and were deliberately left alone or
    only lightly touched: `coils/coils.py`'s `jcrit_from_material` (8 new per-branch
    `ExplicitFunction` nodes, `.tfcoil.t_helium`/`b_max`/`j_crit_sc` all minted, no
    consumer anywhere in the graph yet — not registered) and `coils/calculate.py`'s
    `intersect` wiring (see above).
