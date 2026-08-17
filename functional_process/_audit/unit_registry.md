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
| 10 | `coils/coils.py` | 303 | `functional_process/models/stellarator/coils/coils.md` | **reviewed — 2/4 functions ported** (`j_crit_cable_from_fraction`, `bmax_from_awp`, tier-1, tests passing). `jcrit_from_material` and `intersect` not ported — `intersect` is the Newton-Raphson root-find `winding_pack_total_size` (unit #9) needs, real tier-2, not self-contained the way it's currently called (needs unit #9's solve context). |
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
| 1C | `st_new_config`, `st_geom` | 191-319 | `functional_process/models/stellarator/stellarator_C_geometry.md` | draft, confidence: medium-high — found a real conditional-ownership pattern on `physics.aspect` (owned by this node only when *not* an active iteration variable) and a second, distinct role for `istell` (device-config table selection, data-table-shaped not switch-shaped) beyond the pipeline split already in `switches.md` |
| 1D | `st_strc` | 320-421 | `functional_process/models/stellarator/stellarator_D_structure.md` | **reviewed — ported** (`stellarator_D_structure.py`, tier-1 tests passing) |
| 1E1 | `blanket_neutronics` + `st_fwbs` part 1 | 422-880 | `functional_process/models/stellarator/stellarator_E1_fwbs_setup.md` | draft, confidence: medium — cut falls mid-computation (locals carry into 1E2, consistent with 1E3's independent finding of the same pattern). Found two more undiscovered switches (`.heat_transport.ipowerflow`, `.fwbs.blktmodel` — 1E2 independently found the same two plus `.fwbs.blkttype`) and a likely latent source bug: `.fwbs.p_div_rad_total_mw` is read (line 792) but never assigned in this branch — `.fwbs.p_fw_hcd_rad_total_mw` is written twice with the identical formula nearby instead (redundant-duplicate-write), suggesting a copy-paste error where the divertor computation should have been. Also flagged `.fwbs.f_p_blkt_multiplication` as a possible stale-cross-call read (only written when `blktmodel==1`, read when `blktmodel!=1`) — 1E2's independently-found `first_call_stfwbs` mechanism may be the actual explanation, not yet reconciled between the two records. |
| 1E2 | `st_fwbs` part 2 | 881-1280 | `functional_process/models/stellarator/stellarator_E2_fwbs_neutronics.md` | draft (confidence: medium-high; found a cross-call stateful dependency — `first_call_stfwbs`/divertor surface area — and 3 new switches: `.heat_transport.ipowerflow`, `.fwbs.blktmodel`, `.fwbs.blkttype`) |
| 1E3 | `st_fwbs` part 3 | 1281-1682 | `functional_process/models/stellarator/stellarator_E3_fwbs_shield_divertor.md` | draft, confidence: high — 88% of this chunk (1331-1682) is a reporting-only `if output:` block; real computation is only lines 1282-1330 (cryostat/VV geometry). Corrects both `i_tf_sup` sites cited in `switches.md` — neither (lines 1022, 1724) actually falls in this chunk, both are in 1E2/1F respectively. Found `sc_tf_coil_nuclear_heating`-derived locals (coilhtmx etc.) surviving from chunk 1E1 all the way to this chunk's output block, skipping 1E2 — confirms 1E1/1E2/1E3 must be synthesized as one function set, not ported independently. |
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
| 9 | `physics/physics.py` | `plasma_composition`, `calaculate_stored_thermal_energy` [sic — matches source spelling], `calculate_effective_charge_ionisation_profiles`, `calculate_total_plasma_heating_power`, `outplas`, `phyaux`, `rether` [added — found in chunk 1B's audit; imported as a bare function from this file by `stellarator.py`, missed by the original `self.physics.<method>` grep since it's not called through the `self.physics.` attribute] | `functional_process/models/physics/physics.md` | pending |
| 10 | `physics/confinement_time.py` | `calculate_confinement_time`, `calculate_double_and_triple_product` | `functional_process/models/physics/confinement_time.md` | pending |
| 11 | `physics/exhaust.py` | `calculate_radiation_fraction` | `functional_process/models/physics/exhaust.md` | pending |
| 12 | `physics/plasma_profiles.py` | `PlasmaProfile.run()` (= whole file; `run()` reaches every method) | `functional_process/models/physics/plasma_profiles.md` | **draft — 5/5 arithmetic functions ported, all tier-1, tests passing incl. gradients**. `PlasmaProfile.run()` is *not* portable end-to-end from this file: both branches call `neprofile.run()`/`teprofile.run()` for effect, which belong to **unit #21 (`physics/profiles.py`), a scope correction added below**. Ported: `calculate_ion_vol_avg_temperature`, `calculate_parabolic_profile_values`, `calculate_pedestal_profile_values`, `calculate_profile_factors`, `calculate_parabolic_gradient_lengths`. Only `ProfileFactors` is registered as a node — the two branch arms and `ParabolicGradientLengths` are blocked on reconciling `i_plasma_pedestal`'s two roles (topology switch here, static kwarg in `density_limits.py`). Found: the four on-axis writes in `parabolic_parameterisation` are `redundant-duplicate-write`s of `NeProfile`/`TeProfile.set_physics_variables` (verified algebraically identical); two minted VarPaths for the profile arrays (no PROCESS storage, same as `.stellarator.coilcurrent`); `.divertor.prn1` is the file's only cross-area write and is pedestal-branch-only. **A real port bug was caught by the gradient check** — see the `_simpson` docstring. |
| 21 | `physics/profiles.py` | `NeProfile.run()`, `TeProfile.run()`, `Profile.*`, `set_physics_variables`, `calculate_profile_y`, `ncore`/`tcore` | `functional_process/models/physics/profiles.md` | pending — **new, found auditing unit #12**: 558 LOC, injected into `PlasmaProfile` in `main.py:674-676` and called for effect by both of its branches. Missed by the original scoping grep for the same reason as `coils/`, `rether` and units #19/#20 — it is reached one level deeper than `Stellarator`'s own injected sub-models. Unit #12 cannot be composed end-to-end without it, and it owns the four on-axis fields unit #12 was redundantly rewriting. Also imports `physics/density_limit.py`, a third level. |
| 13 | `blankets/hcpb.py` | `nuclear_heating_blanket`, `nuclear_heating_magnets`, `nuclear_heating_shield` | `functional_process/models/blankets/hcpb.md` | pending |
| 14 | `power.py` | `tfpwr`, `component_thermal_powers`, `calculate_cryo_loads`, `acpow`, `plant_electric_production`, `output_plant_electric_powers` | `functional_process/models/power.md` | pending |
| 15 | `buildings.py` | `Buildings.run()` | `functional_process/models/buildings.md` | pending |
| 16 | `vacuum.py` | `Vacuum.run()` | `functional_process/models/vacuum.md` | pending |
| 17 | `availability.py` | `Availability.run()`, `.avail()` | `functional_process/models/availability.md` | pending |
| 18 | `costs/costs.py`, `costs/costs_2015.py` | `Costs.run()`/`.output()` or `Costs2015.run()` — **which one is itself gated by `i_cost_model`, a switch; treat as two candidate units until that's resolved** | `functional_process/models/costs/costs.md`, `costs_2015.md` | pending |
| 19 | `physics/fusion_reactions.py` | `FusionReactionRate` (`.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`), `beam_fusion()`, `set_fusion_powers()` | `functional_process/models/physics/fusion_reactions.md` | pending — **new, found in chunk 1B's audit**: `stellarator.py`'s `st_phys()` calls these via a bare `import ... as reactions` module alias, not a `self.<attr>.<method>` pattern, so the original scoping grep missed it entirely (same class of miss as the `coils/` and `rether` findings above) |
| 20 | `physics/radiation_power.py` | `calculate_radiation_powers()` | `functional_process/models/physics/radiation_power.md` | pending — **new, found in chunk 1B's audit**, same bare-import miss as unit 19 |

## Constraints

| id | note | record | status |
|---|---|---|---|
| 91 | **corrected from this row's original label "17"** — the actual stellarator-specific constraint (`data.stellarator.powerht_constraint`/`powerscaling_constraint`); docstring literally says "stellarators only". Full audit in place, including a strong-evidence candidate iteration-variable pairing (`te0_ecrh_achievable`, ID 169). | `functional_process/core/solver/constraints.md` | draft |
| 17 | general constraint (plasma radiation fraction limit) — **not** stellarator-specific, just has an `istell`-gated adjustment branch. Recorded briefly for context alongside 91, not fully audited. | `functional_process/core/solver/constraints.md` | draft (brief only) |

Also noted, not yet a full unit: constraint 24 (beta limit, `constraints.py:803-806`) has
the same shape as constraint 17 — a general constraint with an `istell`-gated branch, not
a separate stellarator-only constraint. This "general constraint + embedded istell
branch" is a third pattern beyond "shared switch site" and "stellarator-only constraint" —
worth its own note wherever `switches.md`'s `istell` entry lands.

## Switches

All switches read directly inside `process/models/stellarator/stellarator.py`, found by
grepping `self.data.<area>.i_*` (mechanical, not yet a per-branch reads-set diff — that's
the pilot's job for these).

| switch | area | record | status |
|---|---|---|---|
| `istell` | stellarator | `functional_process/core/solver/switches.md` | **draft — pilot** (master pipeline switch — tokamak/stellarator/IFE split, not expected to need a "split decision" in the formula sense, it's the top-level pipeline selector) |
| `i_tf_sup` | tfcoil | ″ | **draft — pilot** |
| `i_blkt_coolant_type` | fwbs | ″ | **draft — pilot** |
| `i_thermal_electric_conversion` | fwbs | ″ | **draft — pilot** |
| `i_plasma_ignited` | physics | ″ | **draft — pilot** |
| `i_beta_fast_alpha` | physics | ″ | **draft — pilot** |
| `i_pflux_fw_neutron` | physics | ″ | **draft — pilot** |
| `i_confinement_time` | physics | ″ | **draft — pilot** |
| `i_p_coolant_pumping` | fwbs | ″ | **draft — pilot** |
| `i_cost_model` | costs | ″ | **draft — pilot** (decides unit #18 above) |

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
