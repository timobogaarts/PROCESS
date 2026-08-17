# Next steps

Snapshot as of the variant-dispatch wave. `unit_registry.md` remains the authoritative
per-unit status — this file is a priority-ordered punch list, not a second source of
truth; update it as items close rather than letting it drift the way a status doc always
tends to.

Suite at this snapshot: `$PY -m pytest functional_process` → **1123 passed, 521 skipped**
(and `--fp-gradients` clean on the newly ported unit).

## 0. Closed since the last snapshot

- **The `Tier1Contract` array-argument gap is fixed.** `_harness/contracts.py`'s
  `_reference_along`/`_jacobian` now perturb one flat component of an array argument at a
  time on the PROCESS side and take one `jacfwd` per argument on the port side, with
  `_component_label` naming failures as `temperatures[2]` / `kt[1, 7]`. `neoclassics.py`'s
  10 array-argument functions are validated, not just ported. **Nothing is blocked on the
  harness any more** — including `fusion_reactions.py` and `radiation_power.py`, which
  were gated on this.
- **Variant dispatch has a mechanism** (`configuration.py`, `test_configuration.py`) —
  see § 1.
- **`switches.md`'s five owed entries are written** (`blktmodel`, `blkttype`,
  `ipowerflow`, `irefprop`, `istell`'s second role), plus the general result that
  licensed the dispatch design.
- **The `coilcurrent` double-solution is reconciled** — see § 2.
- **Unit #12 (`physics/plasma_profiles.py`) is audited and ported** — 5 tier-1 functions,
  all gradient-clean. Three findings that outlive the unit:
  - **`physics/profiles.py` (558 LOC) was missing from the registry entirely**, now added
    as unit #21. Same scoping miss as `coils/`/`rether`/#19/#20: reached one level deeper
    than `Stellarator`'s own injected sub-models. Unit #12 cannot be composed end-to-end
    without it. **This is the fourth instance — the scoping grep's blind spot is
    systematic, and a transitive-closure sweep over constructor injection would be worth
    more than finding the fifth one by hand** (see § 4).
  - **The gradient check caught a real port bug.** `scipy.integrate.simpson(y, x=rho)`
    uses the general *non-uniform* rule whenever `x` is passed, even on a uniform grid. A
    uniform-shortcut implementation gives identical values and *wrong derivatives with
    respect to every `x[i]`* — off by factors of 2-30 here. No value comparison at any
    tolerance could have seen it. This is the concrete payoff `test_harness.md` predicted
    when it argued for differentiating against PROCESS's own finite difference.
  - **`i_plasma_pedestal` holds two switch roles at once** — topology-changing in
    `plasma_profiles.py`/`profiles.py`, static kwarg in `density_limits.py`. Blocks three
    of the unit's five nodes. See § 1.

## 1. Variant dispatch — decided and implemented, with two things still open

`naming_convention.md` § "Switches are not ports" had already fixed the *policy* (a
topology-changing switch "is consumed by the Python code that assembles the `Graph`");
what was missing was a mechanism, so the two ported-and-tested alternatives
(`LowhybHeating`, `AFwTotalNoPowerflow`) were dead code no graph selected.

**The decision, and why it is not a preference.** No switch in PROCESS is ever an
iteration variable or a scan variable (`grep` over `iteration_variables.py` and `scan.py`
returns nothing for `i_*`, `istell`, `blktmodel`, `blkttype`, `ipowerflow`, `irefprop`).
A switch is therefore constant across every evaluation of one assembled graph, carries no
derivative, and cannot appear on an edge. Graph-per-configuration is the only faithful
representation; `Rewire`/`Insert` would be machinery for a choice that is already final
before the first evaluation, and a fused node branching internally would invent edges
(reading both `eta_ecrh_*` and `eta_lowhyb_*` when only one is live) and put a
non-differentiable integer on a port.

**The result that settles it against the fused-node reading**: `ipowerflow`'s two arms
differ in *reads*, not just formula — `AFwTotalWithPowerflow` reads
`.fwbs.f_ster_div_single`, which `Divertor` owns, while `Divertor` reads
`.first_wall.a_fw_total`, which both arms own. So `ipowerflow != 0` has a genuine two-node
SCC and `ipowerflow == 0` is acyclic. **A switch decides whether the graph is a DAG.** No
single node could express that. Pinned in `test_configuration.py`.

Still open:

- **One switch can hold both roles at once.** `i_plasma_pedestal` is topology-changing in
  `plasma_profiles.py`/`profiles.py` (its arms run different method sets with different
  reads) *and* a static kwarg on `density_limits.EcrhDensityLimit`, where it genuinely has
  no second formula. Both calls are individually right. But `TOPOLOGY_SWITCHES` has no way
  to say "when this switch's arm is selected, that node's static kwarg must agree", so a
  graph could today be assembled with the pedestal arm *and*
  `EcrhDensityLimit(i_plasma_pedestal=0)`. **Now blocking**: three of unit #12's five
  nodes are unregistered waiting on it. Proposed shape — a `Switch` also supplies its
  value to any node declaring it as a static kwarg, so there is one source of truth — but
  not implemented, and it is the first thing to decide next.
- **`blkttype` is three values over two arms** (`blkttype in {1, 2}` vs. `3`), which
  `Alternative.value` — one integer per arm — does not express. Not yet forced, since the
  site is inside the unresolved `st_fwbs` boundary and nothing is ported from it. The fix
  is either several alternatives sharing declarations or a predicate-keyed arm; decide
  when the first such arm is actually ported, not before.
- **Nested switches.** `irefprop` is only reached under
  `i_blkt_coolant_type == WATER`, so its arms are conditional on another switch's value.
  `TOPOLOGY_SWITCHES` is currently a flat tuple with independent choices. Also not yet
  forced, and entangled with the CoolProp question (§ 4) rather than being purely
  structural.

## 1b. Harness: the gradient error bar was too tight — diagnosed and fixed

Surfaced by running `--fp-gradients` over the whole suite (which had not been done since
the array-argument fix landed). `neoclassics.py`'s `TestCollisionFrequency` and
`TestNormalizedCollisionFrequency` each failed at one fuzz point.

**First reading was wrong** and is recorded here so it is not repeated: the guess was
"the bar has no relative floor". It already had a round-off floor
(`finite_difference.py`), so that diagnosis was incomplete.

**What the measurement showed.** Refining the step proves the ports are *correct* —
PROCESS's own difference converges to `jacfwd` as `h -> 0`, agreeing to 3e-11 relative at
`epsfcn = 1e-4` and degrading again below that (the classic round-off V). PROCESS's
default `epsfcn = 1e-3` sits where truncation and cancellation are comparable, so the bar,
not the port, was at fault. The two failures had *different* causes:

| contract | x | truncation | round-off | actual error | dominant term |
|---|---|---|---|---|---|
| `TestCollisionFrequency` | 2.0e-15 | 56.3 | 17.4 | 1342 | round-off, implied **77 ULPs** |
| `TestNormalizedCollisionFrequency` | — | 1.92e6 | ~0 | 3.43e7 | truncation, higher-order |

Both needed ~1.8x more headroom than `gradient_safety = 10` allowed — the same factor from
unrelated causes, which is why the fix addressed the terms rather than either test.

**Fix, in two parts.** `REFERENCE_EVALUATION_ULPS = 64` in the round-off term: the old
`eps * |f|` assumed the reference was evaluated to a single rounding, which is simply
wrong for a chain of tens-to-hundreds of flops, and wrong in the direction that fails
correct ports. And `gradient_safety` 10 -> 25, covering the measured worst case with ~40%
margin.

**The loosening is pinned from the other side.** `test_harness_sensitivity.py`
reintroduces the `scipy.integrate.simpson` bug this harness really caught (unit #12) and
asserts that values still agree at machine precision *and* the gradient check still
fails it, with the injected error orders of magnitude outside the bar. If that module
ever goes green on the second assertion, the check has been blunted past usefulness.

## 2. Review pass (yours, not mechanical)

- **`coilcurrent` — resolved, no action needed, recorded here so it is not re-opened.**
  `grep -rn coilcurrent process/data_structure/` returns nothing, so
  `.stellarator.coilcurrent` is genuinely storage-less and the minted path is a real edge
  (`CoilCurrent` → `CoilsSummaryVariables`) with no alternative. `quench.py`'s
  `c_tf_total / (n_tf_coils * 1e6)` sits *downstream* of that, off a field
  `CoilsSummaryVariables` owns. The two are a chain, not two answers to one question. Keep
  both; the invariant `c_tf_total = n_tf_coils * coilcurrent * 1e6`
  (`coils/calculate.py:266`) is what ties them, and it is documented at
  `coils/quench.py:121-126`.
- **`preset_config.py`** (unit #8): confirmed not representable as a `cottax` node at all
  — its real output set is only knowable by cross-referencing a runtime `hasattr`/
  `setattr` reflection loop against `StellaratorConfigData`'s fields, with silent drops
  on mismatch (a possible latent bug, not fixed). Recommends replacing the 5 hardcoded
  machine-preset dicts with static, fully-enumerated per-machine config records selected
  at graph-assembly time. Same open question as unit #6's (`initialization.py`)
  device-preset literals and chunk 1D's `fncmass`/`gsmass` constants — three independent
  instances of "this node always/only produces literals, not a computation" now on
  record. **Worth a single policy decision, not three separate ones.** Now that
  `configuration.py` exists, the natural shape is a per-machine record selected the same
  way an arm is, but that is a proposal, not a decision.
- **`build.py` open questions**: which `blktmodel`/`ipowerflow` combination is actually
  PROCESS's default case, and what `.build.dz_shld_upper` should be when `blktmodel <= 0`
  (no symmetric "external input" story the way the blanket thicknesses have one).
  *Partly answered since*: both defaults are now cited from `data_structure` in
  `TOPOLOGY_SWITCHES` (`blktmodel = 0`, `ipowerflow = 1`), so the default *combination*
  is settled; what `dz_shld_upper` should be in that combination is not.
- **`neoclassics.py`**: `.neoclassics.iota` and `.neoclassics.er` are read but never
  written anywhere in the file; `iota` is confusingly two different things under one name
  within the same file (a `data` field vs. a forwarded argument, not asserted equal).
  Needs a source read beyond this file's scope to resolve.
- **Everything carried over from the previous wave, still unreviewed**: the two hidden
  double-call patterns (`power_at_ignition_point`, `st_phys`'s `output=True` path), the
  constraint-91 unconditional-call discrepancy, the `dlimit_ecrh`/`p_div_rad_total_mw`
  likely latent bugs, the `.fwbs.fwclfr` possibly-dead-code flag.

## 3. Consolidation (mechanical, no new audit)

- **Synthesize `st_fwbs`'s real function boundaries** (chunks 1E1/1E2/1E3 of unit #1) —
  three independent chunk audits confirm locals (the `sc_tf_coil_nuclear_heating`
  outputs, `first_call_stfwbs`/divertor-area state) span all three chunks. No chunk can
  be ported as written; this needs one synthesis pass reading 1E1+1E2+1E3 together before
  any of `st_fwbs` is portable. **Now the largest single blocker**: it also holds up
  `blktmodel`'s and `blkttype`'s arms, i.e. two of the four open switch questions.
- **`st_phys`** (chunk 1B) — recommended tier-3 composition of ~13 sub-calls rather than
  one 570-line signature; not yet acted on. Blocks `power_at_ignition_point`'s tier-2
  port (the mechanism is understood — 2 steps of Picard iteration on
  `b_plasma_surface_poloidal_average` — but there's still no signature to drive). This is
  also what would give `Tier2Contract` its first real exercise.

## 4. Remaining audit dispatches (all unblocked)

Registry rows still `pending`, none blocking each other, and none now waiting on the
harness:

- **`physics/profiles.py` (unit #21)** — the direct continuation of unit #12, and what
  turns five ported functions into a composable `PlasmaProfile.run()`. It also owns the
  four on-axis fields #12 was redundantly rewriting, so it closes that sequencing
  constraint. Flagged JAX work already known: `profile_y[rho_index] = ...` needs
  `.at[].set()`, and `n_plasma_profile_elements` is a static shape, not a value.
- **A transitive-closure sweep over constructor injection.** Four units have now been
  found by accident rather than by the scoping rule (`coils/`, `rether`, #19/#20, #21).
  The rule greps `self.<attr>.<method>` on `Stellarator`'s own sub-models; every miss has
  been one level deeper — a sub-model's sub-model, or a bare module import. Walking
  `Models.__init__`'s constructor graph transitively, plus bare-import call sites, would
  settle the scope once instead of one surprise per audit. Cheap, mechanical, and it
  should happen before more units are dispatched against a scope known to be incomplete.
- `physics/fusion_reactions.py`, `physics/radiation_power.py` — **were** gated on the
  array-argument fix; now free, and both are what chunk 1B needs. Note both take
  `PlasmaProfile` as an opaque object, so unit #21 should land first or their
  back-doors cannot be closed either.
- `physics/physics.py` (rest of scoped methods), `physics/confinement_time.py`,
  `physics/exhaust.py`.
- `blankets/hcpb.py`, `power.py`, `buildings.py`, `vacuum.py`, `availability.py`,
  `costs/costs.py` + `costs/costs_2015.py` (two candidate units gated by `i_cost_model` —
  and now a clean fit for `TOPOLOGY_SWITCHES`, since `switches.md` already records that
  PROCESS resolves this one outside the pipeline in `Models.costs`, exactly the pattern
  `configuration.py` generalises).
- `coils/coils.py`'s remaining 2 of 4 functions (`jcrit_from_material`, `intersect`) —
  `intersect` is the Newton-Raphson root-find `winding_pack_total_size` (unit #9) needs;
  porting it as a self-contained tier-2 unit is what would unblock
  `winding_pack_total_size` and, eventually, `st_coil`.
- `stellarator.py` chunks 1A, 1C, 1E1, 1E2, 1E3, 1G — audited (`draft`) but not yet
  `reviewed`.

## 5. Structural work

- **Run `Blocking`/SCC over the real graph.** Partly started, and already saying something
  true: the default configuration's 32 nodes decompose into `[25 run] → [2 solve] →
  [5 run]`, i.e. exactly one genuine SCC, and that SCC exists only in one arm of one
  switch. The central hypothesis — that most of PROCESS's stellarator pipeline is not
  genuinely cyclic once dependencies are explicit — is so far **supported**, but on a
  graph whose producers are mostly still unported, so the couplings that would create
  cycles largely aren't in it yet. Re-run and re-read this as § 4 lands; it is not yet a
  result.
- **CoolProp / non-traceable-call policy** — still only flagged, never resolved, and now
  with a sharper statement of the problem: `switches.md`'s `irefprop` entry shows the
  CoolProp call is reached through two levels of switch nesting
  (`i_blkt_coolant_type == WATER`, then `irefprop`), and that `irefprop`'s two arms have
  an *identical* reads-set differing only in traceability. So the decision is not "split
  or keep-static" but "does a node get to be non-differentiable, and if so how is that
  declared". Same question as `st_geom`'s `istell == 6` file I/O and `density_limits.py`'s
  CoolProp branch.
- **Tolerance policy for tier-4 comparison** against PROCESS's own not-really-converged
  reference — still explicitly deferred in `test_harness.md`.
