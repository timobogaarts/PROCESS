# Next steps

Snapshot as of the stellarator-native + `coils/` audit wave landing (see
`unit_registry.md` for authoritative per-unit status — this file is a priority-ordered
punch list, not a second source of truth; update it as items close rather than letting it
drift the way a status doc always tends to).

## 1. Blocking — fix before dispatching more audit forks

**`_harness/contracts.py`'s `Tier1Contract` only differentiates scalar arguments.**
Found by `neoclassics.py`'s audit: 10 of its 12 ported functions take a species-array
(length 4) or quadrature-grid (length 30+) argument, and `_jacobian`/`_reference_along`
(`contracts.py`) both do `float(sample.kwargs[name])`, which fails or silently
misvalidates on an array argument. Those 10 functions are ported (faithful translations)
but **not test-validated** — real, not hypothetical: `fusion_reactions.py` and
`radiation_power.py` (both still `pending`, both likely array-shaped given their physics)
will hit the same wall immediately. Fix once, centrally, before dispatching either of
those two units — otherwise every array-argument function ported in the meantime repeats
neoclassics.py's "ported but unvalidated" gap. Shape of the fix: `_jacobian`/
`_reference_along` need to `jacfwd`/perturb over a whole array argument at once (or
elementwise, accumulating a Jacobian block) rather than assuming `float(...)` succeeds.

## 2. Review pass (yours, not mechanical)

Findings flagged across this audit wave that need a human call, roughly in the order
they'd block downstream work:

- **`preset_config.py`** (unit #8): confirmed not representable as a `cottax` node at all
  — its real output set is only knowable by cross-referencing a runtime `hasattr`/
  `setattr` reflection loop against `StellaratorConfigData`'s fields, with silent drops
  on mismatch (a possible latent bug, not fixed). Recommends replacing the 5 hardcoded
  machine-preset dicts with static, fully-enumerated per-machine config records selected
  at graph-assembly time. Same open question as unit #6's (`initialization.py`)
  device-preset literals and chunk 1D's `fncmass`/`gsmass` constants — three independent
  instances of "this node always/only produces literals, not a computation" now on
  record. Worth a single policy decision, not three separate ones.
- **`total_process.py`'s two switch-alternative exclusions** — `LowhybHeating`
  (`isthtr==2`) and `AFwTotalNoPowerflow` (`ipowerflow==0`) are ported and tested but not
  wired into the demo graph, because they collide on ownership with the default-value
  alternative that is wired in. This is the concrete, now-unavoidable instance of
  `CLAUDE.md`'s open "variant dispatch has no clean node-graph shape" question — worth
  deciding the real pattern (a `Graph`-per-config selected at assembly time? a
  `Rewire`/`Insert` keyed by switch value?) now that there are two live examples to
  design against, rather than deferring further.
- **`coils/calculate.py`'s invented `.stellarator.coilcurrent` VarPath** — minted because
  it's a real cross-function graph edge with no existing `data` storage. `quench.py`'s
  fork independently verified the same value is recoverable as
  `c_tf_total / (n_tf_coils * 1e6)` and used that instead of reading the minted path —
  i.e. two different forks solved the same wiring problem two different ways. Reconcile:
  either `CoilCurrent`'s node should feed `QuenchProtection` directly via the minted
  path, or the minted path should be dropped in favour of the algebraic relationship
  quench.py used. Not urgent (both are individually correct), but leaving both live
  invites the two to silently diverge if either formula changes.
- **`build.py` open questions**: which `blktmodel`/`ipowerflow` combination is actually
  PROCESS's default case (assumed, not verified, when picking `ipowerflow=1` for
  `total_process.py` above — true for the switch's *default*, not necessarily the
  combination `build.py`'s own audit meant), and what `.build.dz_shld_upper` should be
  when `blktmodel <= 0` (no symmetric "external input" story the way the blanket
  thicknesses have one).
- **`neoclassics.py`**: `.neoclassics.iota` and `.neoclassics.er` are read but never
  written anywhere in the file; `iota` is confusingly two different things under one name
  within the same file (a `data` field vs. a forwarded argument, not asserted equal).
  Needs a source read beyond this file's scope to resolve.
- **Everything carried over from the previous wave, still unreviewed**: the two hidden
  double-call patterns (`power_at_ignition_point`, `st_phys`'s `output=True` path), the
  constraint-91 unconditional-call discrepancy, the `dlimit_ecrh`/`p_div_rad_total_mw`
  likely latent bugs, the `.fwbs.fwclfr` possibly-dead-code flag. None of this wave's
  findings supersede them.

## 3. Consolidation (mechanical, no new audit)

- **`switches.md`** — still owes `blktmodel`, `blkttype`, `ipowerflow`, `irefprop`
  (all found across 1E1/1E2/1E3/build.py, several times independently), and `istell`'s
  second role (device-config table selection, found in 1C and confirmed by
  `preset_config.py`). This has been the top pending item for two waves running; it's
  pure write-up of evidence already collected, not new investigation — the reason to do
  it now rather than later is that both new switch-alternative conflicts in
  `total_process.py` (§2 above) are instances of switches this file is supposed to
  track.
- **Synthesize `st_fwbs`'s real function boundaries** (chunks 1E1/1E2/1E3 of unit #1) —
  three independent chunk audits now confirm locals (the `sc_tf_coil_nuclear_heating`
  outputs, `first_call_stfwbs`/divertor-area state) span all three chunks. No chunk can
  be ported as written; this needs one synthesis pass reading 1E1+1E2+1E3 together before
  any of `st_fwbs` is portable.
- **`st_phys`** (chunk 1B) — recommended tier-3 composition of ~13 sub-calls rather than
  one 570-line signature; not yet acted on. Blocks `power_at_ignition_point`'s tier-2
  port (the mechanism is understood — 2 steps of Picard iteration on
  `b_plasma_surface_poloidal_average` — but there's still no signature to drive).

## 4. Remaining audit dispatches (unblocked, straightforward to fork out)

Registry rows still `pending`, none blocking each other:

- `physics/physics.py` (rest of scoped methods), `physics/confinement_time.py`,
  `physics/exhaust.py`, `physics/plasma_profiles.py` — **wait for §1's harness fix**
  before dispatching `confinement_time.py` in particular (scaling-law family, likely to
  be array/switch-heavy) and `fusion_reactions.py`/`radiation_power.py` below.
- `blankets/hcpb.py`, `power.py`, `buildings.py`, `vacuum.py`, `availability.py`,
  `costs/costs.py` + `costs/costs_2015.py` (two candidate units, gated by `i_cost_model`,
  already known to be "resolved above this file" per `switches.md`'s existing entry).
- `physics/fusion_reactions.py`, `physics/radiation_power.py` — found via chunk 1B's
  bare-import miss; **wait for §1's harness fix**, high odds of the same array-argument
  gap given the physics (rate coefficients, reaction cross-sections).
- `coils/coils.py`'s remaining 2 of 4 functions (`jcrit_from_material`, `intersect`) —
  `intersect` is the Newton-Raphson root-find `winding_pack_total_size` (unit #9) needs;
  porting it as a self-contained tier-2 unit here is what would then unblock
  `winding_pack_total_size` and, eventually, `st_coil` itself.
- `stellarator.py` chunks 1A, 1C, 1E1, 1E2, 1E3, 1G — audited (`draft`) but not yet
  `reviewed`; folding a "port what's tier-1 and self-contained" pass into their review
  is the same standing practice as everything else, just not yet applied to unit #1
  itself (1D and 1F already got it).

## 5. Structural work (after §1-4, not before)

Nothing here should start until the audit is far enough along that a real `Blocking`
run over the accumulated `total_process.py` graph would say something true:

- Run `cottax.Blocking`/SCC detection over the real (not demo) graph once enough of it
  exists to make the answer meaningful — this is the actual test of this whole rewrite's
  central hypothesis, that most of PROCESS's stellarator pipeline is not genuinely
  cyclic once dependencies are explicit.
- Decide the variant-dispatch pattern for real (§2's `total_process.py` note) instead of
  picking a default value ad hoc per switch, once there are enough live examples to
  generalise from.
- CoolProp / non-traceable-call policy — still only flagged (`st_geom`'s `istell==6`
  file I/O, `density_limits.py`'s CoolProp branch), never resolved.
- Tolerance policy for tier-4 comparison against PROCESS's own not-really-converged
  reference — still explicitly deferred in `test_harness.md`.
