# Next steps

**What this file is.** A current-state reference and a priority-ordered punch list for
the `functional_process` port. It is deliberately *not* a changelog: sections record what
is true now and what is open, not the order in which it became true.
`unit_registry.md` remains the authoritative per-unit status.

**Section numbers are frozen.** They are cited from other `_audit/*.md` records, from the
per-unit `.md` records, and from live `.py` docstrings (`total_process.py` cites §5,
`mda_harness.py` §8, `sand.py` §6, among others). A section whose material is closed is
emptied to a stub that says where the live material went; nothing is ever renumbered.

**Where to start:** §16 (the current state and priority order, 2026-08-29), then §5 (the
structural vocabulary — Shape A / Shape B — that the code itself cites). §13's priority
order is superseded by §16.6, and §11's by §13; the measurements in both stay live.

**The `Verified state` table below is stale in more than one place** — it says 159 nodes
/ 348 unowned inputs where the live graph is 156 / 320 (`model_tree_design.md` step 4c
removed three cost nodes for a subsystem a stellarator does not have), and its suite
count is behind by the whole 2026-08-27 wave day. **§16.1 carries the current
measurements**, taken against `HEAD`; re-measure rather than cite this table, and see
§13.1 before trusting any gate measured against `cottax`'s working tree.

## Verified state

| check | value |
|---|---|
| `pytest functional_process -q` | **3730 passed**, 3344 skipped, 0 failed (the 6 added are `test_mda_harness.py`; 3724 re-confirmed 2026-08-25 before it landed. Previously: re-measured 2026-08-20 — a `git worktree` at `aca0fb6d` also measures 3724, so the enum conversion of `model_tree_design.md` §8 step 1 added and moved nothing) |
| `cd ~/jaxgraph && pytest` | **738 passed**, 3 skipped (re-measured 2026-08-25; the recorded 482 was stale by ~250 tests — cottax has moved through `fdc4f5e` since, and the port's own numbers are unmoved by all of it) |
| `run_mda_harness.py` | **499 agreements** (23 array-valued, **73 both-sides-exactly-zero**), **34 disagreements** (0 in driven blocks, 34 acyclic), 3 unverifiable, **0 ungrounded**, 21 errors |
| … accounting | **557 owned variables walked, 0 unaccounted**; 61 switch kwargs checked / 0 mismatched / 3 not data-backed / **0 unresolved** |
| `GRAPH` (`REFERENCE_CONFIGURATION`) | **159 nodes**; **139 blocks, 14 driven**; **348 unowned inputs** (re-measured 2026-08-25 against `Blocking.scc(driven_graph(GRAPH))`; the 138/349 carried here since `LModeProfileReset` were stale by one, and the model-tree conversion did not move them — before == after) |
| **MDA from a cold `IN.DAT`** | **137 blocks, 0 failures**, 1 non-finite (`.physics.nu_star`, `nan` in PROCESS too, read by nothing) |
| SAND | **21 conditions × 14 design**, **0 non-finite cells**; drive 124 nodes (re-measured 2026-08-27; the 30 × 23 carried here is from before ten `FixedPoint`s dissolved into ordinary nodes) |
| SAND C2 (seeded from PROCESS's answer) | **326 SQP iterations**, conv `8.8e-11`, `objf 1.217757338` |
| **SAND C3 (cold start)** | **258 iterations**, conv `8.0e-09`, `objf 1.217757452` — **the same point as C2 to six digits**; §11.11 |
| … and why those are not 42/100 | `max_iter` is 500 for Stage C now, not `VmconDriver`'s PROCESS-inherited 100. The count grew with round 2's switch-to-slot conversion (22 unknowns / 16 equalities → 14 / 8), measured against a `git archive` of the pre-round-2 tree at 131; the cottax version and every condition scale are ruled out. `optimise_design.md` §12. |
| MDF C2 / C3 | **stale — see §16.1b.** Recorded here as 129 it/converged and 200 it/not converged; measured 2026-08-29 at **200/not converged** and **60/not converged**, and identically so at `87ee1285`, so the move is the wave day's |
| MDF | 15 conditions × 8 design (`icc` × `ixc`), Jacobian compared to PROCESS's **unreduced** |
| PROCESS itself, same problem | 46 VMCON iterations, **94 s**, conv `2.40e-07` |

**A gap this exposed, cheap to close.** SAND has **no Stage B0** — no check of its own
`jax.jacfwd` Jacobian against a central difference of its own condition map. Adding one is
what found the `c24` kink above, in 2 of 690 cells, and nothing else in the repo looks for
this failure. Note it is a *different* class from the audited one: §9/§10.5b/§11.8 are
about `nan` derivatives **at exactly zero**, where this is a finite derivative on one side
and an unbounded one on the other, `1.9e-09` from the switch. How many other clamped roots
the solve is sitting on is unknown and unlooked-for.

**Model tree — `model_tree_design.md` §8 step 4 is DONE** (2026-08-25). The ten topology
switches are ten typed **slots**; `machine_from_indat` is the only place an `i_*` integer
is read; `configuration.py` (`Switch`/`Alternative`/`Configuration`/`build_graph`) and
`TOPOLOGY_SWITCHES` are deleted, 505 lines of it. **All 159 nodes are in the tree now** —
the 59 that were bare class-named tuples at the root have slot names, 43 of them under
`.costs.*`, which is a module for the first time. `test_configuration.py` becomes
`test_machine.py`: exclusivity is by construction (one slot, one occupant) so those tests
are gone, and what replaced them compares occupants **by ports rather than by node
names** — step 3 made node identity the slot, so a name comparison would now pass
vacuously for every single-node slot.

Every gate identical, and **measured against `HEAD` rather than against this file's
recorded numbers**: 159 nodes, 139 blocks / 14 driven, 348 unowned inputs, the per-node
(inputs, outputs, type) sha unchanged; MDA harness 499/34/3/0 · 557/0 · 61/0/3/0; SAND C2
and C3 **byte-identical** (31 it / `objf 1.217757347`, 99 it / `1.217757378`). The C2/C3
figures recorded in the table above (42 it, 100 it) were stale by several intervening
changes — comparing against them would have reported a failure that was not one, which is
the same trap as §11.6 item 6's.

The **grouped DSM moved, correctly**: `costs` is a 43-node subsystem instead of 43
root-level singletons, taking the picture from 7 groups / 57 cross-group edges / 1-of-7
contiguous to **8 / 144 / 0-of-8**. §11.4's provenance-against-structure measurement now
includes the cost model, and says no subsystem is contiguous in the run order.

**Model tree — §8 step 3 is DONE** (2026-08-25). `COMMON` is a
`Machine()` **instance** of eleven `ModelNamespace` classes, so a node is named by the
snake_case slot path that reaches it (`.stellarator.coils.coil_current`) and the class
name is gone from the name: identity is the *place*. `configuration.ROOT` and the
`{ROOT: subtree}` idiom are deleted — an instance is a composition with no name of its
own, so nothing has to be stripped. Switch arms are still bare tuples named by class at
the root; step 4 places them. Every gate identical, measured before-and-after with one
script rather than against this table: 159 nodes, 139 blocks / 14 driven, 348 unowned
inputs, and a sha of every node's `(inputs, outputs, type)` unchanged. Exactly two name
literals existed in the port and both needed re-pinning
(`mda_harness.EXCLUDED_NODE_NAMES`, `test_configuration.py`'s `divertor_cycle`) — the
prediction that the first would survive is corrected in `path_refactor.md` §B.5, and it
is worth reading: **`in`-matching buys tolerance to a prefix changing, not to the matched
substring being replaced**, and the failure was silent.

**Two renderer defects fixed alongside it, both invisible to every gate** (2026-08-25).

- **The coloured DSM was destroyed by step 3 and reported success.**
  `visualization/grouping.py` read a node's group off its leading `DictKey`s; step 3 made
  slots `GetAttrKey`s, so every machine node fell to `UNGROUPED`, `grouped` printed
  `1 group(s) at depth 1` where there are six, exited 0 and wrote both files. Now counts
  both kinds, and asks `group_of(..., among=graph.nodes)` for what the kind was a proxy
  for — a name minted over a node unmints to a node, one minted over a variable does not,
  which keeps `^problem.fwbs.f_ster_div_single` from inventing an `fwbs` subsystem.
  Restored: **7 groups, 57 cross-group edges, 1/7 contiguous in run order**.
- **`dsm_sand.html` had never rendered at all**, despite this file claiming the command
  writes it. `cottax`'s `to_ragraph` did
  `formatter = RaGraphFormat() if formatter is None else formatter` — a caller's
  formatter *replaced* the flat-namespace adaptation instead of being its `base`, which
  is what `MINTED_NODE_SUFFIX` exists for (`^cond.y2` is both a `Compare` and what it
  produces). Passing `xDSMFormatterFlat` therefore disarmed the disambiguation and
  `_check_names` refused a well-formed graph. Now composed. **Uncommitted in
  `~/jaxgraph`** — that tree had other live work in it; see the session note.

Both were found by *looking at the output*, not by a check, and neither could have been:
one exits 0 with a wrong picture, the other never ran. That is the same shape as the
`atol` floor above — a guard's absence is only visible when something is wrong.

**Declaration surface — Part A is DONE** (2026-08-20). Every read and write is declared
`From(area)` / `OutputInto(area)` (or `FromExactly`/`Output` for the 43 array-element
escapes, now lambda-free recorder chains), all 36 files converted and each proven by
`path_refactor.md` §A.4's side-by-side port-identity checker — identical `VarPath`s, in
order, per declaration. The 36 body renames landed with pure functions and sample keys
untouched; the out-of-census `path_of` lambda sites went through cottax's
`resolve(area.field, VarPath)`. The flat endpoint holds:
`grep -rn "lambda s:" functional_process --include='*.py'` → **0**, and the suite and
MDA harness are byte-identical (see the Verified state table). The record is
`path_refactor.md`'s header; the node-name half (Part B) proceeds as
`model_tree_design.md` §8 steps 3–4.

**Closed, and the answer is not what either side of it expected.**
`_audit/closed/x109_pinning_verification.md` settled the feasibility question — the pinned point
is genuinely better (`max|eq| 2.1e-12`, no inequality violated) and the multiplier
hypothesis is refuted by five orders of magnitude (`Σ|λ_eq| = 1.22`, not `6.3e+04`).
`_audit/closed/x109_hypotheses.md` then settled *why*, and the cause is a **kink in the model**:
at every converged point the design sits on `(Te + Ti)/20 == 0.65`, the threshold of
`fast_alpha_beta`'s clamped square root (`pure_formulas.py:342-348`) — `1.9e-09`
from it at the free optimum — where `c24` rises like `2√h` on one side and linearly on the
other. AD reports one side. Of 690 Jacobian cells exactly two disagree with a central
difference of the port's own condition map, both in the `c24` row, and `c24` alone drifts
like `h^0.52` where every other condition drifts like `h^2.00`. **The SQP is stopping
correctly against an incorrect linear model of one constraint** — not terminating
prematurely, and not descending a wrong objective (the `objf` row is correct to `7.7e-10`
against the same central difference). PROCESS is unaffected, and the reason is worth stating as a
mechanism rather than a coincidence: its answer sits `5.8e-03` below the switch, and its
finite difference (`epsfcn = 1.0e-3`, `numerics.py:595`) is **six orders of magnitude wider
than the feature**. A coarse finite difference is a low-pass filter on the derivative — it
cannot resolve a kink narrower than its own step, so it returns a chord *across* it where
AD returns the exact one-sided slope. The exact derivative is the correct answer to a
question an SQP is not asking: it wants a model valid over a finite step. **Here the
approximate gradient is the more useful one precisely because it is approximate.**
(`_audit/closed/x109_hypotheses.md` says `epsfcn = 0.01`; the value is `1.0e-3`, matching
`_harness/finite_difference.py`'s `PROCESS_EPSFCN`. The argument is unchanged.)

XDSM/DSM of the assembled SAND graph: `python -m functional_process.render_xdsm sand`
writes `xdsm_sand.html`/`dsm_sand.html` (self-contained, pan/zoom). The bare form renders
the declared model graph instead.

`run_mda_harness.py` is disk-cached (`mda_harness.CACHE_DIR`, keyed on the input files'
bytes plus a `(path, size, mtime_ns)` fingerprint of the `process/` tree;
`FP_HARNESS_NO_CACHE=1` disables it). A full run went from **1m56s to 17s** with
identical numbers, so it is cheap enough to be a standing regression check after every
change — treat it as one.

## 0. Closed since the last snapshot

*Stub.* This section was a chronological log of closed items and has been emptied; the
document no longer keeps a changelog. Per-unit status lives in `unit_registry.md`; the
structural findings that came out of those waves live in §5 (Shape A / Shape B), §8.1
(the mint/error triage), §8.2 (the static-switch audit) and §9 (the `Alternative` states).

## 1. Variant dispatch — the mechanism holds up under a second real case

The general proposal from an early snapshot ("a `Switch` also supplies its value to a
node's static kwarg") is **withdrawn as unneeded** — every topology case seen so far
reduces to "put the kwarg-carrying instantiation inside the `Alternative` that requires
it," which needs no new abstraction. Revisit only if a case turns up that can't be
expressed that way.

Four real gaps in `Switch`/`Alternative` remain open. None blocks anything today; each is
recorded here so the next instance is recognised as an instance rather than re-derived.

**All four (plus the joint-switch synthetic path and §9's nested-switch instance) have a
designed resolution: `_audit/model_tree_design.md`**, which replaces the
`Switch`/`Alternative` mechanism outright with a typed model tree and an IN.DAT factory —
see its §2 for the gap-by-gap mapping. They stay listed here as *open* until that design
lands (its §8 steps 3–4); do not patch the old mechanism for a new instance of any of
them in the meantime.

- **One arm, several literal values.** `blkttype` is three values over two arms
  (`blkttype in {1, 2}` vs. `3`), which `Alternative.value` — one integer per arm — does
  not express. Landing site: `st_fwbs`'s S2 (§3), which is where `blkttype`'s arms live.
- **Nested switches**, three instances, and the third is on a *reachable* node, which is
  the condition this section itself set for taking the question seriously.
  `i_nd_plasma_pedestal_separatrix` (under `i_plasma_pedestal == PEDESTAL_PROFILE`) and
  `irefprop` (under `i_blkt_coolant_type == WATER`) are both moot for this scope; the
  third, `costs.py`'s `TfMagnetCostResistive` (`i_tf_sup` nested under
  `.costs.i_cost_model`, §9's "left undone"), is not. Nothing is currently *wrong* —
  `.tfcoil.i_tf_sup == 1` is both PROCESS's default and the reference run's — but a
  resistive-TF cost graph cannot be assembled at all.
- **"Reads-set genuinely differs, kept static anyway" — now six instances, not three.**
  `i_confinement_time`/`i_rad_loss` (`confinement_time.md`), `i_plasma_ignited`
  (`confinement_time.md` and `composition.md` independently),
  `supercond_cost_model` (two nodes), `i_pf_conductor`, and `itart` on `CostOfElectricity`
  (§9) are all cases where `traceability_policy.md`'s default ("reads-set differs →
  split") technically applies but was **not followed**, because the differing part of the
  body is small relative to a large shared body. `i_tf_sc_mat` (`superconductors.md`, 8
  genuinely different reads-sets, no shared body) is the contrast case, and `costs.py`
  supplies a controlled contrast *inside one file*: `acc2221` **was** split (two arms, no
  shared body, disjoint reads) while `coelc`'s `itart` was **not** (15 lines of a 290-line
  function). The open question is whether `traceability_policy.md`'s split-by-default rule
  needs a size/entanglement-aware exception — see that file, which now records the
  deviations. It is well-posed enough to answer.
- **Exclusivity proved only by colliding output ownership.**
  `Switch.check_arms_are_exclusive` accepts colliding *output ownership* as its only proof
  of exclusivity, and two real instances are exclusive by PROCESS's own `if`/`elif` without
  having it: `.vacuum.i_vacuum_pumping` (`VacuumOld`/`VacuumPumpingSimple` own disjoint
  fields) and `.costs.i_cost_model` (the ported subsets share no output). Both rejected
  workarounds have real downsides — unconditional-in-`COMMON` for one arm reproduces the
  `EcrhDensityLimit` bug class when the registered arm isn't the run's actual value, and
  loosening the check to accept a caller's assertion removes the safety net
  `test_non_exclusive_arms_are_rejected` exists to provide.
- **Two different values selecting the *identical* node set.** `.tfcoil.i_tf_sup` values
  `0` (resistive copper) and `2` (aluminium) both run `calculate_tf_power_resistive`
  (`Power.tfpwr` dispatches on `i_tf_sup != 1` only), and declaring both as ordinary
  `Alternative`s fails `test_configuration.py::test_arms_select_different_node_sets` —
  correctly, since a value that doesn't change which nodes exist belongs in
  `naming_convention.md`'s static-kwarg category. Worked around by declaring value `2`
  `unported` and pointing at value `0`; honest, but `Alternative` still has no way to say
  "these two literal values are the same arm" (e.g. a tuple of `value`s).

## 1b. Harness: the gradient error bar fix — reconfirmed, not re-diagnosed

*Stub.* The finding is live and unchanged: the per-point Richardson error bar still
separates a real bug from noise at `gradient_safety = 25`, and
`test_harness_sensitivity.py`'s pinned regression (re-injecting the `simpson` bug) still
fails the gradient check as designed. The design itself is in `test_harness.md`.

## 1c. Harness: default-run compile time fixed

Three base-class changes in `_harness/contracts.py`, so every unit inherits them: gradient
checks are fully gated behind `--fp-gradients` (`test_outputs_finite` is eager and
value-only; `test_gradient_finite` carries the differentiation), `_jacobians(sample)` takes
one batched `jax.jacfwd(f, argnums=(...))` trace instead of one per argument, and
`Tier2Contract.ported` is `eqx.filter_jit`-wrapped once at class-definition time (not per
test method, where a fresh wrapper would defeat its own cache).

## 2. Review pass (yours, not mechanical)

Judgement calls that no test will make for you. All still open:

- ~~**`preset_config.py`** (unit #8) — a **fourth** instance of "this node always/only
  produces literals".~~ **Closed** for this unit by `preset_config.md`: the 34
  `stella_config_*` scalars get one zero-input producer node, because nothing else in
  PROCESS ever writes them and leaving them unowned is what made the graph unrunnable
  cold. Arms 1-5 wired 2026-08-30, closing `helias_5b.IN.DAT`.
  **The policy question itself is not closed**, and the record says why the answer does
  not simply transfer: unit #6's 16 device-preset literals are stellarator-mode
  *overrides of ordinary input fields* (`.physics.q95 = 1.03`, `.build.dr_cs = 0.0`), so a
  producer node for them would claim ownership of fields an `IN.DAT` also sets — which is
  not the situation here. Unit #6 and chunk 1D's constants still need the decision.
- ~~**`build.py`** — `dz_shld_upper` under `blktmodel <= 0`.~~ **Closed** by
  `_audit/units/models/build.md` § "resolves `next_steps.md` §2". Under `blktmodel > 0`,
  `process/models/build.py:1650-1662` produces `dr_blkt_inboard`/`dr_blkt_outboard`/
  `dz_shld_upper`, and `models/stellarator/build.py::BlktmodelBlanketThickness` **already
  ports that block verbatim** — the two source files are line-for-line identical there.
  Under `blktmodel <= 0` all three are plain run inputs, so no occupant is needed and none
  is missing: it is `conditional-ownership-by-run-config`, the same shape as
  `.build.dz_xpoint_divertor` and `.tfcoil.dx_tf_side_case_min`, and not an unlocated
  producer. Nothing to decide.
- **`neoclassics.py`** — `.neoclassics.iota`/`.er` producer still unlocated.
- **Ported code whose only caller is out of scope** — unit #21's
  `set_pedestal_and_separatrix_values` (reachable only from tokamak unit #22) and
  `ImpurityRadiation`'s whole-file treatment. Worth a registry-level policy on whether
  "port because splitting is more work than porting" is standing practice.
- ~~**`fusion_reactions.py`'s `calculate_profile_y` return-value bug**~~ **Closed** by
  `_audit/units/models/physics/current_drive.md` § "A live PROCESS bug in two sibling
  arms", which is that audit. The flag was right and the consequence is sharper than
  "6 call sites use the return value arithmetically": exactly **two** of `i_hcd_primary`'s
  thirteen values reach them, and both are therefore **unreachable in PROCESS itself** —
  `6` (`CULHAM_LOWER_HYBRID`) raises `TypeError` at `current_drive.py:1498` via
  `cullhy → lhrad → lheval`, and `7` (`CULHAM_ELECTRON_CYCLOTRON`) at `:815` via
  `culecd`. Recorded, not fixed: repairing it is a physics decision for the PROCESS
  maintainers, not a porting one. `indat.py`'s `UNPORTED` carries both values with that
  reason, which makes this the first pair of refusals in the port that are **not** about
  what this port has failed to write.
- **Carried over, still unreviewed**: the constraint-91 unconditional-call discrepancy,
  the `.fwbs.fwclfr` possibly-dead-code flag, and `radiation_power.md`'s open question 4
  (the two `combine_radiation_powers` callers disagree on whether to clip at zero —
  `stellarator.py` does, `physics.py` doesn't).

## 3. Consolidation — `st_fwbs` synthesis done, one new re-chunking to act on

`stellarator_E_fwbs_synthesis.md` replaces the even-thirds 1E1/1E2/1E3 cuts with six real
sub-computations. S1 (`fw_blanket_shield_geometry_setup`) and S5
(`cryostat_and_vv_geometry`) are ported and registered; S4
(`blanket_shield_fw_coolant_mass`) is ported as `stellarator_fwbs_s4.{py,md}`
(`BlanketComponentMasses` under a synthetic joint `Switch` on
`.fwbs.blktmodel,.fwbs.blkttype`, plus `ShieldMass` in `COMMON`); S6 (`st_fwbs_output`) is
a reporting shell, out of scope. Two remain:

- **S2** `blanket_shield_tf_nuclear_power` (422-480 + 608-1030) — the real
  `blktmodel`×`ipowerflow` dispatch, 3 live arms, tier-3 in two. Two landmines wait for
  its auditor, both found while consolidating `hcpb.py`, neither fixed: (a)
  `blanket_neutronics()` calls `self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_
  shield()` with **zero** arguments although both are `@staticmethod`s requiring 2/7
  keyword arguments — a `TypeError` the moment that path executes, exercised by no test;
  (b) a real output-ownership conflict on `.fwbs.p_tf_nuclear_heat_mw`, written by
  `hcpb.py`'s `NuclearHeatingMagnets` *and* by chunk 1F's `ScTfCoilNuclearHeating`
  (unconditional in `COMMON`), which `blanket_neutronics()` calls both of, discarding one.
  S2 is also the landing site for §1's `blkttype` gap.
- **S3** `divertor_mass_and_first_call_seed` (1030-1043) — a genuine two-node SCC with
  `Divertor` (unit #4), bootstrapped by a hardcoded `50.0` on the true first call.

`st_phys` (chunk 1B) — unchanged, still a recommended tier-3 composition of ~13 sub-calls;
its Picard mechanism is understood, the node-shape decision is not made.

## 4. Remaining audit dispatches (updated — §4b's and §4c's waves closed, see §0)

The wide-but-shallow balance-of-plant units are exhausted. What is left is not another
parallel wave:

- `stellarator.py` chunks 1A, 1B, 1C, 1G — the orchestration layer, still `draft`. Where
  the next SCC is most likely to surface, and the reason these are sequenced one at a time
  rather than parallelised: two agents deriving node shapes from the same unsettled
  orchestration logic is the failure mode this section exists to avoid.
- `st_fwbs`'s S2 and S3 (§3) — S2 needs a focused single pass because of its two
  landmines; S3 is a first-use of `Blocking` + a driver on ported code, not another
  audit-and-port.
- `costs_2015.py` (§9's "left undone") — the one change that would turn `i_cost_model`
  into an ordinary two-armed `Switch`.

## 4b. [CLOSED, consolidated — see § 0] Dispatch — 5 parallel agents, priority-ordered

*Stub.* The five-agent dispatch table this section held is gone — every unit in it landed;
see `unit_registry.md`. The two conventions it established are still standing
practice and are the only part worth keeping: an agent writes **only** its own audit
record, port and test files (registration, registry/`next_steps` bookkeeping and any new
`Switch` wiring are the consolidation pass's job), and an agent runs **only its own test
files**, never the bare `pytest functional_process`, since an agent that writes only new
files under its own scope cannot have broken anything else.

## 4c. [CLOSED, consolidated — see § 0] Proposed next dispatch (not yet launched)

*Stub.* Landed; see `unit_registry.md` for the per-unit status of `buildings.py`,
`vacuum.py`, `availability.py` and `costs.py`/`costs_2015.py`. **The prediction
`costs.py`/`costs.md` cite this section for was correct**: `i_cost_model` is a genuine
topology-changing `Switch` (the 1990/2015/custom cost models are disjoint subgraphs, not a
shared body with a branch), and it is now a real `TOPOLOGY_SWITCHES` entry — see §9 for
how its default arm had to be represented.

## 5. Structural work

**This section is the vocabulary the rest of the project cites — Shape A and Shape B.**
Keep the two definitions intact; the counts are current-state and move.

- **`Blocking`/SCC over the real graph.** `REFERENCE_CONFIGURATION` assembles **149
  nodes** (151 once `driven_graph` applies its cuts), decomposing into **131 blocks, 12 of
  them driven**. The driven set is: the structurally-inherent 2-node
  `[node, ^problem[node]]` pairs every registered `FixedPointFunction` mints,
  `Intersect`'s `RootFind`, and the two genuine cross-node Shape-A cycles that `mda.CUTS`
  closes. This is a tracked empirical finding, not a test of the rewrite's thesis
  (`CLAUDE.md`'s case is structural — making the graph explicit, not a bet on how much of
  it is cyclic).

- **Two genuinely different shapes turn up under "cyclic-looking," not one.** Both are
  real findings; they need different treatment, and only one of them needs anything not
  already in `cottax` today.

  **Shape A — ordinary cross-node cycles.** Two or more already-valid, separately-owning
  nodes whose dependencies happen to close a loop. `Graph` builds these with no error at
  all; `Blocking`/`strongly_connected_components` finds them exactly as designed. There
  is **no blocker** to registering these today — doing so requires no `Cut`, no minted
  copy, no driver decision, nothing beyond declaring the ordinary `Input` that names the
  real `VarPath` the other node owns. Confirmed instances:
  - `Divertor`/`AFwTotalWithPowerflow` — `ipowerflow != 0` only; closed by cutting
    `.fwbs.f_ster_div_single`.
  - The density/fusion/pedestal/composition loop — a real physics feedback loop
    (density profile ↔ plasma composition ↔ fusion reaction rates ↔ pedestal on-axis
    density), not an artefact of how the port split functions. Originally a 4-node cycle
    closed by one cut (`.physics.proton_rate_density`); now a **6-node** cycle needing
    **two** cuts, because `FusionTotalsNoBeam` gave `.physics.fusden_total`/
    `.fusden_alpha_total`/`.p_dt_total_mw` their first producers and opened a parallel
    edge `FusionRates → FusionTotalsNoBeam → PlasmaComposition`. Which second cut was
    **measured, not chosen**: of the 42 variables owned inside the enlarged cycle,
    `.physics.fusden_alpha_total` is the only one that works paired with
    `proton_rate_density`, and no single variable works alone. `driven_graph` groups a
    cycle's cuts into **one** `FixedPointCut`, so the block declares one `FixedPoint` over
    both unknowns rather than two problems `Blocking` would refuse. `test_mda.py` asserts
    both sufficiency *and* minimality.
  - `Divertor`/`st_fwbs` S3's `DivertorPlateMass` — checked and found **not** to be a
    cycle at all: PROCESS's own staleness (`st_fwbs` runs before `st_div`, so it reads the
    previous `run()`'s value) was a call-order artefact, and the edge is one-directional.

  **Shape B — genuine single-node self-loops.** One `NodalDeclaration` whose own `Output`
  and `Input` name the identical `VarPath` — the value assumed and the value produced at
  once. `~/jaxgraph/CLAUDE.md`'s stated invariant: *"a node may not read what it owns...
  so a cycle is always at least two nodes."* This is not a style preference `cottax`
  enforces loosely — it is a hard construction error: `to_graph(Avail(...))` raises
  `ValueError: reads ['.costs.cplife'], which it also owns` directly from `cottax.spec`'s
  `__check_init__`. **These cannot be represented as a plain node at all, regardless of
  whether anyone intends to drive them yet** — and the answer is
  `cottax.interfaces.pytree_namespace_module.FixedPointFunction`, not a new primitive.
  Declaring `step(...)` instead of `__call__` mints the cut internally
  (`mint_key_cond`/`prefix_path`): the body reads the real `VarPath` and writes a minted
  `^cond.<var>` copy; a separate `FixedPoint` `DeclaredNode` (no body) reads that copy and
  owns the real `VarPath`. This is a **structural admission requirement, not a solver
  choice** — the resulting `FixedPoint` node is perfectly valid sitting undriven in the
  graph, same as Shape A's undriven SCCs.

  Converted and registered: `availability.py`'s `CplifeAvail`/`CplifeAvailSt`
  (`.costs.cplife`), `thermal_cryo.py`'s six (`delta_eta`, `eta_turbine`,
  `etath_liq`, `temp_turbine_coolant_in`, `p_fw_div_heat_deposited_mw`,
  `p_fw_blkt_coolant_pump_mw`), and `winding_pack_total_size`'s `j_tf_wp`
  (`WindingPackJTfWp`, a degenerate/identity fixed point off the Bi-2212 branch, confirmed
  with `jax.grad` rather than asserted).

  Written but still unconverted, all otherwise fully ported and tested, all confirmed by
  direct `to_graph` construction rather than by audit reasoning:
  `thermal_cryo.py`'s `PlantThermalEfficiency`/`PlantThermalEfficiency2` and
  `Cryo`/`CryoLoads` (`.fwbs.qnuc`, plus `.power.qss`/`qac`/`qcl`/`qmisc`);
  `electric_production.py`'s `PlantElectricProduction`.

  **The discipline that matters more than either shape**: "representable via
  `FixedPointFunction`" and "`to_graph()` assembles" are necessary but not sufficient —
  check the *real* producer of what a self-loop's other branch resolves to before
  concluding the loop is genuine. Four apparent self-loops dissolved that way
  (`plasma_composition`'s `first_call`, whose bootstrap target has no dependency back on it,
  so `NextFirstCall` was deleted and `first_call`/`alphan`/`alphat` are not ported at all —
  see `composition.md`; `st_phys`'s `beta_fast_alpha` and `beta_beam`, both Shape
  A; and `Divertor`/`DivertorPlateMass`). Two more needed nothing: `plasma_composition`'s
  `.impurity_radiation.f_nd_impurity_electron_array` (reads indices 2-13, writes 0-1 —
  per-index `VarPath`s with a `SequenceKey` component are sufficient, no slice addressing
  needed) and `st_coil`'s `len_tf_coil` (no node owns it; the stale read has its own
  parameter name, `len_tf_coil_stale`).

  **Representing a cycle and choosing its driver stay separate decisions.** For the
  registered graph nothing is deferred — `mda.schedule()` drives every block. For anything
  newly written: closing a Shape A cycle with a driver is `rewrites.Cut` +
  `FixedPointCut`/`RootFindCut`/`ResidualCut` on the *already-built* graph, a different and
  later mechanism from `FixedPointFunction`'s built-in cut, which applies only at
  declaration time for Shape B. Do not reach for `rewrites.Cut` to represent Shape A; there
  is nothing to cut, the graph is already valid.

  **On coverage.** Two genuine cross-subsystem cycles across the whole registered graph,
  a count that has held steady while the graph itself has roughly tripled. Every Shape B
  instance found is a *single PROCESS function* referencing its own earlier/later self, not
  evidence of broader coupling waiting to be found by splitting those functions further. But
  the density/composition/fusion-rate/pedestal loop is a genuine and not-especially-rare
  shape, and it *grew* (4 nodes → 6) as more producers landed — expect a few more rather
  than zero, and expect existing cycles to enlarge as boundary inputs get producers.

- **CoolProp / non-traceable-call policy** — still only flagged, unchanged.
- **Tolerance policy for tier-4 comparison** — still explicitly deferred.

## 6. Constraints, objective, iteration variables — not a separate layer, just a thin
   selection over models already being ported

**Constraints and the objective are not architecturally special to `cottax`.** The only
thing that makes them look like a separate layer is that PROCESS bundles them with the
*solver*, not that they need new graph machinery. This framing is cited by
`core/solver/objectives.py`, `core/solver/constraints.py` and `sand.py`, and it held up
when the `Optimise` layer was actually built (§10, `optimise_design.md` §10).

- **Constraints** (`process/core/solver/constraints.py`) are either an ordinary
  `Compare(place, pairs=[(model_output, stored_bound)])` node over outputs that are
  already ordinary ported model nodes, or — for a constraint that just thresholds one
  `data` field against a bound — not even a node, a bare residual read. No new primitive
  needed. **Registered unconditionally**, not behind a `Switch`: PROCESS doesn't gate a
  constraint's *computability* on anything, only whether the solver bothers enforcing it
  (`numerics.icc`), which is an `Optimise`-problem-assembly-time decision, not a
  graph-existence one.
- **The objective** (`numerics.i_figure_merit`'s branch selection) is not a node at all —
  *"a per-run selection of which existing output is 'wanted', same as cottax's refusal to
  have an `OutputNode`"* (`CLAUDE.md`). Selecting it is a `Graph.prune`-style query run
  when assembling an `Optimise` problem.
- **Iteration variables** (`ITERATION_VARIABLES[id]`/`numerics.ixc`) are the same shape
  again: a designation ("this `VarPath` is free, not derived") on values that already
  exist once their producer models are ported. No separate porting effort.

**Status.** All ~82 constraints and all 16 `i_figure_merit` objective branches are ported
(`core/solver/{constraints,objectives}.py`); only 50/52 are excluded, IFE-only, `.ife.*`
being an entirely unbuilt subsystem. So the *porting* work this section describes is
done — an earlier version of this section closed by saying the only open work was "a
handful of constraint bodies", which is no longer true in either direction. What the
`Optimise` build (§10) actually found was that the section's own prediction was right
about the shape and wrong about the difficulty: nothing needed a new primitive, and every
real obstacle was a **missing producer** for a field a constraint reads (§10's
"is the right node registered" gap), not anything about constraints as such.

### 6.1 Which constraints are cycles the graph is holding open — measured, and it is one

The question: is a PROCESS constraint sometimes not a *requirement* at all, but a
consistency equation closing a loop the graph could close itself? Asked structurally on
the reference run: for each active `icc` entry, which of its reads have producers, and are
those producers already connected to each other. A constraint that compares a computed
quantity to an **input bound** adds no edge and cannot be a closure; one that equates two
computed quantities on the *same dependency chain* is the return edge of a cycle.

**Result: 1 of 14, and PROCESS labels it itself.**

- **`c2`, global power balance, is a held-open cycle.** Eight of its ten reads are
  produced, and **20 ordered pairs of those producers are already connected** — every
  heating and radiation term flows into `StellaratorConfinementTime`, which produces the
  two transport-loss terms `c2` compares them against. Wiring "losses = heating" in
  closes the loop. `stellarator_helias.IN.DAT:16` says so in its own comment:
  `icc = 2  *Global power balance (consistency equation)`.
- **Nine cannot be closures at all**: exactly one produced read against an input bound
  (`.constraints.*`, `.tfcoil.*_max`, `.divertor.*_max`).
- **Four compare two produced quantities and are still genuine limits** — `c17`, `c82`,
  `c35` (connected producers) and `c83` (not connected). All `leq`/`geq`: *"the gap must
  be at least the coil thickness"*, *"available radial space ≥ required"*. Two computed
  quantities related by an inequality is a feasibility requirement, not a closure —
  nothing says they must be equal.

**And the variable that closes `c2` is the narrowest in the run.** `x10 = hfact` has
**one** reader and reaches only constraints `[2, 62]`; `rmajor` and
`b_plasma_toroidal_on_axis` each reach **all 14**. So `(c2, hfact)` is very nearly a
separable block: a `RootFind` on `hfact` zeroing `c2` would lift one unknown and one
equality out of the optimiser, leaving `c62` as an ordinary inequality on what remains.
That is §12.1's trade stated concretely — *"free a coupling variable and let the optimiser
close the loop → SAND; keep the producer and drive the cycle → MDF"* — and **neither
architecture does the latter today**: SAND and MDF both hand all 8 to the optimiser.

**What would decide it is the bound, not the physics.** A `RootFind` does not respect
`boundl`/`boundu`, so lifting `c2` out is only equivalent while `hfact`'s root is interior.
This `IN.DAT` sets no bound on `x10` at all (it bounds 1, 2, 3, 4, 6 only), so `hfact`
carries the defaults `[0.1, 3.0]` from a start of `1.0`, and §12.3 measured all 8
iteration variables interior on this run. Cited, not re-measured here.

### 6.2 `c16` is a `geq` counted as an equality, and nothing checks that

`n_equality_constraints = 2` is set **by hand in the input file**
(`stellarator_helias.IN.DAT:12`), and `process/core/init.py:1285-1293` never compares it
against the constraint bodies — it takes the number and computes
`n_inequality = total - n_equality`. The split is positional and user-declared, exactly as
`CLAUDE.md` says, and on this run the two disagree: position 1 is `icc = 16`,
`*Net electric power lower limit`, whose body is `return geq(p_plant_electric_net_mw,
p_plant_electric_net_required_mw)`. VMCON therefore receives it as `h(x) = 0` — the plant
is pinned **exactly** at the required net power instead of being allowed to exceed it.

That may well be intended (a design study minimising cost has no reason to overbuild), and
it is not a port defect: the port reproduces PROCESS faithfully. It is worth writing down
because the intent is expressed as *a count and an ordering*, checked by nothing, and a
constraint's own `eq`/`leq`/`geq` is the one place that intent could have been declared and
verified. A cheap check — "every constraint inside `n_equality_constraints` returns `eq`" —
would either confirm the choice is deliberate or find the next one that is not.

## 7. A third pattern, distinct from Shape A/B — raised, then resolved by a sharper
   test than "does it iterate"

Every `Tier2Contract` unit solves its own internal iteration **eagerly, inside a plain JAX
function** — `optx.root_find(...)` (`coils.py`'s `intersect`) or a hand-rolled
`jax.lax.while_loop` Newton scheme (`vacuum.py`'s `solve_duct_diameter`/`VacuumOld`).
First framing (wrong): this is the same gap Shape B (§5) closed, with `RootFind` instead
of `FixedPoint`.

**It isn't, and the reason clarifies the actual rule.** A `RootFind` `DeclaredNode` (what
`ImplicitFunction` mints) has **no body** — it produces no value at all until a `Drive`
step runs some algorithm against it. `FixedPointFunction`'s Shape B conversions cost
nothing precisely because `step`'s "next" value was *already* a complete, fully determined
computation with no external solver needed; the only problem was the self-referential
*naming*. An eager root-find is different in kind: there is no value without running an
iterative algorithm, so a declared, undriven `ImplicitFunction` doesn't just add
structure, it removes the ability to call the thing and get an answer.

**The right test turns out not to be "does it iterate," but "does anything else in the
graph need to read or write something inside that iteration's own state."** `intersect`'s
unknowns (`wp_width_r`, `lhs`, `rhs`) are fully encapsulated inside
`winding_pack_total_size`'s own private computation; nothing else in the graph needs them.
That makes it structurally a numerical primitive (no different in kind from calling
`jnp.linalg.solve` inside a larger pure function), not a coupled subsystem. Same for
`solve_duct_diameter`/`solve_duct_geometry` inside `VacuumOld`. The genuine Shape B cases
were different precisely because their `VarPath`s *are* real, externally-relevant
`DataStructure` fields — that external relevance is what earns the
`FixedPointFunction`/`ImplicitFunction` treatment, not internal iteration by itself. For a
future tier-2 unit, that is still the question to ask first: **if no other node needs the
internal unknowns, an eager JAX function is the correct, not merely expedient, shape.**

**But this section's own verdict — "no follow-up pass needed for these two" — has been
overtaken, and is withdrawn.** Both units were converted anyway: `coils.py` now has
`Intersect(ImplicitFunction)` with `WindingPackTotalSize` split into
`WindingPackIntersectInputs` + `Intersect` + `WindingPackTotalSizePost`, and `vacuum.py`
has `DuctDiameterRootFind` (see `models/vacuum.md`, which says so explicitly). The
justification given was **different from this section's test**: making the root-find's
solver algorithm "a first-class, swappable `Drive` choice, not something hardcoded inside
one node's body." That is a real, distinct consideration this section did not weigh, and
the reconciliation is **open**: does "swappable `Drive` choice" alone justify
`ImplicitFunction` with no external consumer? If yes, this section's test is a sufficient
condition, not a necessary one, and should say so. Note the cost is not zero — a declared
root-find's answer is not directly comparable to PROCESS's own (see §8's validation-chain
subsection).

## 8. The MDA harness — built, run for the first time, and iterated on this session

The layer that catches what per-unit tests structurally cannot. What exists:

- **`mda.py`** — turns `total_process.GRAPH` (or any `graph_for(configuration)`) into
  something runnable. `mda.CUTS` names the variables that close each raw cross-node cycle
  (found via `Graph.closing_readers` plus an empirical acyclicity check, not guessed);
  `driven_graph()` applies `cottax.rewrites.FixedPointCut` per cycle, skipping cleanly a
  cycle a given configuration doesn't have; `default_drivers()` assigns
  `NewtonDriver`/`PicardDriver`/`VmconDriver` by problem type; `schedule()` builds a
  runnable `Schedule` for the whole graph.
- **`core/solver/drivers.py`** — `PicardDriver` (a generic `AbstractDriver` answering
  `cottax.problem.FixedPoint`; `AbstractDriver`'s own docstring named the pairing, nothing
  in `cottax` had implemented it) and `VmconDriver` (§10).
- **`mda_harness.py`** — `converged_data(input_file)` runs PROCESS's own `SingleRun`
  in-process on `tests/regression/input_files/stellarator_helias.IN.DAT` and returns the
  live `DataStructure` (no MFile round-trip; `cottax.tools.pytree.get_at` reads `VarPath`s
  off it directly, `SequenceKey`-indexed array fields included). `compare(graph, data)`
  seeds `mda`'s `Schedule` from that same run's values — boundary inputs *and* every
  driven block's starting guess, because the question is whether the graph reproduces an
  answer PROCESS already found, not whether it solves cold — and diffs every value the
  schedule produces. `run_mda_harness.py` is the entry point.
- **`mda_constraint_harness.py`** — the same idea for constraints/objectives: every ported
  `constraint_N`/`objective_metric_N` called with the converged run's real field values,
  diffed against PROCESS's own `ConstraintManager`/`objective_function`. Checks whether a
  port is right *given a real, internally self-consistent set of simultaneous values*, not
  just hand-built samples that could combine values which would never co-occur.
- **A cottax core fix, upstream in `~/jaxgraph`**: `to_graph()`/`node_and_names` claimed in
  their own error message to accept a bare `NodeDefinition` but had no code path that did
  — found via `vacuum.py`'s `DuctFeasibility` (a bare `Feasibility`, so no class-derived
  name). Both now accept a `{name: NodeDefinition}` mapping.

**The bug classes this harness catches and unit tests structurally cannot** — this is the
durable content of the section, and each is a real instance, not a hypothetical:

1. **Wiring/binding bugs.** `ZTfInsideHalf`'s 1-tuple return (a single-`Output` node
   returning `(x,)`), caught only by running the node through `_run_acyclic`. Seven more of
   the same class turned up later (`thermal_cryo.py` ×6, `vacuum.py` ×1) —
   invisible to `PicardDriver`, fatal to `Residualise`.
2. **Static switch kwargs copied from a `*_variables.py` default instead of the run being
   modelled** — four instances found by luck before the class was closed by §8.2's
   `switch_audit`.
3. **A wrong `Input` binding on a correct pure function** — §8.3's `q95`/`iotabar`.
4. **A missing *producer*, where every value is right and only a derivative is wrong** —
   §10's iteration variable 4, and §11's `c62`. Three instances of this class now.
5. **A dual-ownership conflict in PROCESS itself.** `.build.z_tf_inside_half` has two real
   PROCESS producers (`st_build`'s formula, `st_coil`'s), and which survives depends on
   call order: `stellarator.py`'s `run()` calls them in opposite order depending on the
   `output` flag, and every real run ends with an `output=True` report pass where `st_coil`
   overwrites last. Generalised in §10's "A finding about PROCESS itself".

**Still open from this section's original list:**

- **The 2 minted islands originally excluded from the harness** — `DuctDiameterRootFind`
  is still excluded and correct as such (every `VarPath` it touches is minted, PROCESS
  stores none of them). The `Intersect`/`WindingPackIntersectInputs`/
  `WindingPackTotalSizePost` island is **no longer excluded** — see
  `optimise_design.md` §5.2, which grounded `a_tf_wp_no_insulation`/`a_tf_wp_with_
  insulation` through `KNOWN_MINT_VALUES`.
- **`VacuumOld`'s two disagreements — explained, deliberately not suppressed.** This port
  solves the duct-diameter equation to `tol=1e-10` (`models/vacuum.py:250`, whose docstring
  states and justifies the deviation); PROCESS stops the same Newton iteration at a 1%
  relative step (`process/models/vacuum.py:469-477`), a bound far looser than the observed
  difference — so PROCESS's number is not ground truth here. The two off fields are one
  cause, not two: `dlscal ∝ d**1.4`. Recorded in `mda_harness.EXPLAINED_DISAGREEMENTS`;
  **not** wired into `compare()`, since a per-field tolerance would mask a future real
  regression. It **propagates**: through Account 224 into `c22`, `cdirt`, `cindrt`/`ccont`,
  `concost` and finally `.costs.coe`, so it is not cosmetic.
- Everything else this section listed — `i_confinement_time`, `i_thermal_electric_
  conversion`, `i_p_coolant_pumping`, `i_plasma_ignited`, `AuxiliaryPhysicsQuantities.
  fusrat`, `ConfinementTime`'s residual 1.2%, the 3 ungrounded inputs and 13 errors — is
  **CLOSED**; see §8.1 (the triage), §8.2 (the switch audit and the two reclassifications)
  and §8.3 (the `q95`/`iotabar` binding).

### The validation-chain question for `Intersect`/`DuctDiameterRootFind`/`DuctFeasibility`

Investigated directly (not assumed): both `Intersect` and `DuctDiameterRootFind` are
validated via `Tier2Contract`, not `Tier1Contract` — **no direct value-agreement check
against PROCESS's own reported number exists, by construction**, because PROCESS's own
algorithm (`intersect`'s fixed 100-iteration cap; `_newton_method_duct_diameter`'s loose
0.01 relative-step tolerance) stops before the true root, so PROCESS's own answer is not
ground truth for either. What is actually checked: the port's answer, plugged back into
the real defining equation (`intersect_residual`/`duct_diameter_residual`), is small in an
absolute sense and no worse than PROCESS's own residual — and the *declared* node
reproduces the *eager* port function's own answer, including at least one real
PROCESS-derived legacy sample point each. **The accurate claim is "reproduces PROCESS's
own formula, solved more tightly than PROCESS's own loose iteration" — not "matches
PROCESS's own reported number."** State that distinction wherever this validation is
cited; it does different work from the whole-graph harness's node comparisons.
`DuctFeasibility` has no PROCESS equivalent at all — `solve_duct_geometry` is one specific
heuristic (shrink by 10% until it fits), not "any feasible point" — and remains
unvalidated by design, pending an `Optimise`+real-objective wrapper and a constrained
driver.

### 8.1 Triage of the 3 "ungrounded inputs" and 13 "errors" — done, one at a time

All 16 were checked **individually against `process/`**, not batched — including the six
`.physics.*profile*` entries, which do turn out to share one root cause but were verified
separately rather than assumed to. **The table below is a standing reference**, cited from
eight other records: it is the evidence that a given `VarPath` is a genuine mint with no
PROCESS counterpart, and the warning about the near-miss field a later pass would
otherwise bind to and get `0.0` from.

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

**All three follow-ups this triage proposed have since landed** as
`mda_harness.KNOWN_MINT_VALUES` entries, which is exactly what that dict exists for:
`.stellarator.coilcurrent` (the inverse of `coils/calculate.py:276`, row 14),
`.impurity_radiation.pden_impurity_rad_total_mw` (the inverse of `radiation_power.py:132`,
row 12), and rows 1/2's winding-pack areas (`a_tf_wp_no_insulation ==
.tfcoil.dx_tf_wp_primary_toroidal * .tfcoil.dr_tf_wp_with_insulation`, and its
insulation-inclusive sibling, both read off `coils/calculate.py:483-501`) — the last of
which is what took the winding-pack coil island out of `EXCLUDED_NODE_NAMES` entirely
(`optimise_design.md` §5.2). **Row 13 deliberately gets none**: PROCESS clips
`.physics.pden_plasma_core_rad_mw` at zero after storing it
(`process/models/stellarator/stellarator.py:2153-2155`), so it is not recoverable back
through the sum. Row 3 (`.tfcoil.den_tf_sc_material` → `.tfcoil.dcond[0]`) was fixed at
the source, in `models/stellarator/coils/mass.py`.

### 8.2 The systemic static-switch audit — the whole defect class, closed at once

Four bugs found in this project were one defect: a `total_process.py` registration
carrying a hardcoded static switch kwarg copied from the corresponding
`process/data_structure/*_variables.py` bare Python default rather than from the run being
modelled (`i_confinement_time`, `i_thermal_electric_conversion`, `i_p_coolant_pumping`,
`i_plasma_ignited`). Nothing checked any of them; each was found only when a downstream
value diverged loudly enough to notice, so a wrong switch that moved no compared output
would never have been found at all.

`mda_harness.switch_audit(graph, data)` now checks every one on every harness run.
**By introspection, not source parsing**: the kwargs are `eqx.field(static=True)`
attributes on the declaration instances the assembled graph actually holds (reachable as
`CallableNode.fn.__self__`; the walk is generic, so `FixedPointFunction`/problem nodes are
covered too), so what is checked is what the graph carries, not what `total_process.py`'s
text says. A kwarg name is resolved to a `DataStructure` field by scanning every area for
an attribute of exactly that name — which works because PROCESS's naming scheme makes field
names globally unique, and is *checked*, not assumed (a name found in two areas is reported
unresolved, never silently resolved to whichever area came first). ~~One alias is needed
and is declared in `STATIC_KWARG_ALIASES`~~ — **the alias is gone**:
`model_tree_design.md` §8 step 1 deleted `PlasmaComposition.is_ignited` (the field is now
`i_plasma_ignited: PlasmaIgnitionModel`, resolving by name like every other switch) and
`STATIC_KWARG_ALIASES` with it. The same step enum-typed every model-selection kwarg the
audit walks, so a mismatch now prints member names beside the integers, and a new
always-empty `not_enum_typed` list guards against a bare-integer registration returning.
The 61 / 0 / 3 / 0 totals were unmoved by all of it.

Three-way classification, the same discipline the harness applies to
ungrounded/unverifiable/errors: **checked** (resolved to a real field and compared);
**not data-backed** (declared in `STATIC_KWARGS_WITHOUT_BACKING_FIELD` with a reason —
`ImpurityRadiationTotals.imp_indices` and `ProfileValues.rho`); **unresolved** (neither —
reported, never silently dropped; currently zero). Run against a bare `DataStructure()`
instead of a converged run, the audit flags exactly the *deliberate* deviations
`total_process.py` documents, i.e. it reads as a diff against PROCESS's defaults, which is
precisely the axis the bug class lives on.

**Two cautions worth keeping.** The seven corrections this audit made moved **no compared
value** — the differing terms were numerically inert on this run, or sat on an identity arm
where the node was agreeing with PROCESS only because the harness seeds it from PROCESS's
own converged value. Same count, but a real check where there had been a tautology. And
generally: "this switch is wrong *and* would produce a difference of about this size" is
**two** claims; this harness checks the first directly and the second not at all. The
strong prior that `i_plasma_ignited` caused the 1.2% confinement gap was right about the
bug and wrong about the cause (§8.3).

### 8.3 The `q95`/`iotabar` binding bug — the residual 1.2%, closed

**The cause was a wrong input binding, not a formula.** PROCESS's
`calculate_confinement_time` names its 20th positional parameter `q95`
(`process/models/physics/confinement_time.py:79`) and its *tokamak* caller does pass
`.physics.q95` — but the stellarator caller passes `self.data.stellarator.iotabar` into
that same slot (`process/models/stellarator/stellarator.py:2312`), and
`iss04_stellarator_confinement_time` consumes it as `iotabar**0.41`. The port's
`ConfinementTime` bound `.physics.q95`, so the ISS04 law was fed a safety factor.
Confirmed arithmetically, not inferred: `q95 = 1.03`, `iotabar = 1.0`,
`1.03**0.41 = 1.0121928428817748` — the reported `rel_diff` of `1.219e-02` to every digit.

**Fix**: `StellaratorConfinementTime` (`models/physics/confinement_time.py`), a subclass
rebinding exactly that one read, registered as the `value=6` arm of a `.stellarator.istell`
`Switch`. **A subclass is the unit of rebinding, structurally** — `Input` is a class-level
`__call__` parameter default, so no `eqx.field(static=True)` on an instance can vary a
read, and `NodalDeclaration.name` is `type(self).__name__`, so the arm gets its own node
name free. **The signature is derived, not restated**: `_rebound_signature` copies the base
signature and replaces one default, so the arm cannot drift if `ConfinementTime` gains or
rebinds a parameter, and a renamed parameter raises at import instead of silently rebinding
nothing (restating all 36 parameters would have reintroduced exactly the class of bug being
fixed). And a **`Switch`, not a hardcoded binding**, because changing which node produces a
read changes an edge — so every stellarator consumer must pass `.stellarator.istell = 6`,
which `REFERENCE_CONFIGURATION` does, and `istell` in 1..5 is `unported` so a wrong device
config fails loudly.

**The general lesson**: *a positional parameter named from one device's vocabulary is a
standing trap here.* No `Tier1Contract` could have caught this — every case passes `q95`
positionally to the pure function, and the pure function is correct. Only the *binding* was
wrong, and bindings live in node declarations, which per-unit tests do not exercise. Any
other `calculate_*` with device-dependent call sites deserves the same check.

### 8.4 Harness state after §8.1–8.3

*Stub.* A point-in-time table, superseded. Current numbers are in this file's
"Verified state" block at the top and in §11.1.

## 9. `.costs.coe` has a producer — the `Optimise` layer's stated blocker, closed

`optimise_design.md` named exactly one blocker for wiring a real `Optimise` problem: "this
run's own objective is not computable". `tests/regression/input_files/
stellarator_helias.IN.DAT:229` sets `i_figure_merit = 6`
(`FiguresOfMerit.COST_OF_ELECTRICITY` → `objective_metric_6(coe) = coe / 100.0`), and
nothing in the ported graph produced `.costs.coe`. `CostOfElectricity` does.

Unit #18's `costs.py` is **41/43 methods, 44 pure functions, 44 nodes, 43 registered**; the
two left out are `run` (the call-order dispatcher a `Graph` replaces — its two
*computations*, `.costs.cdirt` and `.costs.concost`, are ported as their own nodes) and
`output`. **`.costs.coe` depends on every computational method in the file** — the
accumulation is a plain sum of all nine Account-22 sub-totals and all six top-level
accounts, none of which PROCESS ever skips — so "port only what `coe` needs" and "port
everything except `run`/`output`" were the same instruction. Checked by walking
`costs.py:2990` backwards before any code was written.

Two general corrections came out of it, both recorded in `costs.md`:

- **`acc2222`'s "dynamic-length loop" is not a JAX blocker.** The bound is
  `.pf_coil.n_cs_pf_coils`, which is neither an iteration variable nor a scan variable, so
  it is constant for a whole solve and belongs in `naming_convention.md`'s static-kwarg
  category; the loop unrolls at trace time. **Generalise: "is this a dynamic-length loop"
  is the wrong first question. The right one is whether the length can change between two
  evaluations of one assembled graph** — which for PROCESS is answered by the
  iteration-variable and scan-variable lists, not by reading the loop.
- **"`Costs` never even runs by default, so its nodes must stay unregistered."** True of
  PROCESS's bare default (`KOVARI_2014`) and irrelevant to the run this project validates
  against, which sets `i_cost_model = 0` explicitly. Reasoning from the bare
  `*_variables.py` default rather than from the reference IN.DAT is the same habit §8.2
  found four instances of — this is a fifth, at the level of *whether a node exists*.

### The registration decision — a new third `Alternative` state

`.costs.i_cost_model` is a textbook topology switch (resolved once in `process/main.py`'s
`Models.costs` `@property`, which picks a whole `Model` instance before any model runs;
never read inside either cost file). The trap is that **`GRAPH = graph_for()` is evaluated
at import time from the bare default configuration**, and the default arm (`1`,
`KOVARI_2014`) has no ported nodes at all — `costs_2015.py` still has zero. Declaring it
`unported` would raise `NotImplementedError` on `import functional_process.total_process`,
breaking the package and every one of `test_configuration.py`'s per-switch parametrised
assemblies besides.

Three readings were weighed; the argument is kept in full in `total_process.py`'s own
comment block on the switch:

1. **Register the 43 nodes unconditionally in `COMMON`.** Rejected — and this is *worse*
   than the `EcrhDensityLimit`/`WardTaylorAvailability` bug class, not an instance of it:
   those computed a value the default configuration never computes, whereas PROCESS's
   default *does* compute `.costs.coe`, by the 2015 model. An unconditional registration
   would put a **different number in the same field**.
2. **Change what `graph_for()`'s bare default means** — `GRAPH =
   graph_for(REFERENCE_CONFIGURATION)`, or a lazy `GRAPH` that is allowed to fail.
   Structurally the most honest answer, since PROCESS's bare-default configuration is not
   fully ported. **Deliberately not taken, and flagged as the user's call** — it changes a
   shared contract (`Switch.default`'s "a silent IN.DAT reproduces PROCESS's own defaults",
   pinned by `test_default_configuration_matches_process_defaults`) and would make
   `test_configuration.py`, `test_mda.py`, `mda.py`'s default argument and `render_xdsm.py`
   all carry a configuration. **The tension is open and visible**:
   `REFERENCE_CONFIGURATION` exists precisely because the bare default matches no real run,
   and `total_process.py`'s docstring concedes the bare-default graph is
   "device-incoherent". A future pass that wants this should do it as its own change.
3. **An arm that assembles as empty** — taken. `Alternative` gains a third state,
   `unproduced`, alongside `declarations` and `unported`. It contributes no nodes and does
   not raise, so under `KOVARI_2014` `.costs.coe`/`.costs.concost` simply have no producer
   and any consumer surfaces as an unowned (boundary) input rather than being silently
   handed the 1990 model's formula. `check_arms_are_exclusive` skips it the way it skips
   `unported` arms, and `Alternative.__post_init__` now *requires* exactly one of the three
   to be given, so an arm that declares nothing can never be confused with an oversight.

The measured cost of option 3 was zero — the default `GRAPH` was byte-for-byte the graph it
had been. The honest cost is conceptual, and worth stating rather than burying:
**`unproduced` is a *weaker* guarantee than `unported`.** It is only sound when the arm's
outputs have no other producer in the assembled graph, and nothing checks that for you,
because "no producer" is exactly what it is being told to produce. `test_configuration.py`
pins both halves on the real instance: `.costs.coe` has an owner under `i_cost_model == 0`
and none under `== 1`.

`i_cost_model == 2` (`USER_PROVIDED`) is `unported`, not `unproduced`, and the contrast is
the clearest statement of the distinction: it injects a user-supplied `Model` at runtime,
so there is no PROCESS-side subgraph to port at all, and a caller asking for it has a model
in mind this graph has never seen. Refusing is right there; assembling empty would not be.

### Findings, none fixed

- **A second `acc223` defect** alongside the `c2233` one: `fkind` (the Nth-of-a-kind
  multiplier) is applied **only** on the `ifueltyp == 1` branch, for all three sub-accounts
  (`process/models/costs/costs.py:1877-1881`, `1899-1903`, `1917-1921`). Every other account
  applies `fkind` unconditionally, so Account 223 silently escapes it on any run with
  `ifueltyp != 1`. Reproduced as written.
- **The `c2233` defect is not the same shape as `cdrlife_cal`.** `ifueltyp` is a
  run-configuration constant, so "never assigned in *this* call" implies "never assigned in
  *any* call of the run", and the field can only hold its dataclass default `0.0` — making
  the port's `0.0` exact rather than an approximation. `cdrlife_cal`'s gate
  (`life_blkt_fpy < life_plant`) genuinely moves between VMCON iterations, which is what
  makes *that* one a real unrepresentable-state gap. Two superficially identical patterns,
  opposite conclusions, and the discriminator is the same iteration-variable/scan-variable
  question the `acc2222` correction turned on.
- **`.costs.c2234` is a dead field**: written in exactly one place in all of `process/`, to
  `0.0`, inside the IFE *and* `ifueltyp == 1` branch. Folded to the literal.
- **A JAX trap, caught only by `test_gradient_finite`**: PROCESS clamps a negative net
  electric power to zero before a square root (`costs.py:2874-2888`), and the obvious port
  `jnp.sqrt(jnp.maximum(p, 0.0))` is value-correct while returning `nan` from `jacfwd` on
  the clamped branch. Fixed with the standard *double* `jnp.where`. **A `jnp.maximum` guard
  in front of a function with an unbounded derivative *at the guard point*** is the general
  shape; every value test passed throughout, and it has since appeared again
  (`fast_alpha_beta`, §10).

### Left undone, with reasons

- **`costs_2015.py` is untouched**: still 2/13 functions ported, zero nodes. Its
  `run()`-level accumulation (`costs_2015.py:52-102`, eight `calc_*` methods filling a
  100-slot `s_cost` array) is the single change that would turn `i_cost_model` into an
  ordinary two-armed `Switch` with no further machinery, since `.costs.coe`/`.costs.concost`
  are exactly the outputs the two arms share.
- **`TfMagnetCostResistive` is written but not registered.** `acc2221`'s two arms share no
  body and read disjoint fields, so they are two nodes (the split default), not one node
  with a static `i_tf_sup` kwarg. Registering both is a duplicate-ownership conflict on
  `.costs.c22211`/`c22212`/`c2221`; pairing them as a `Switch` would require that switch to
  **nest** inside `.costs.i_cost_model` — §1's nested-switch gap, third instance and the
  first on a reachable node.
- **The IFE arms of six methods are not ported** (`ife` static, refused). `.ife.*` has no
  unit in `unit_registry.md` at all, and the 2-D `fwmatm`/`blmatm`/`shmatm` `VarPath`
  question is deferred, not answered.
- **The split-default deviation count is six, not three** — see §1's third bullet, and
  `traceability_policy.md`, which now records the deviations.
- **No systemic check exists for "is the right *node* registered".** `switch_audit` checks
  the static kwargs a registered node carries; nothing checks that the set of registered
  nodes matches the configuration the run describes. See §11.6.

## 10. The `Optimise` layer — built, and the ladder run end to end

**`_audit/optimise_design.md` §10 is the record; this section is the index and the parts
that belong to *this* file.** `core/solver/drivers.py` grew **`VmconDriver`** — `pyvmcon`
(PROCESS's own SQP) fed one `jax.jacfwd` of the block's `ConditionMap` instead of
`Evaluators.fcnvmc2`'s 1%-step finite differences. `sand.py` assembles PROCESS's actual
problem onto the graph as SAND, `sand_harness.py`/`run_sand_harness.py` run the three-stage
ladder, `test_sand.py` adds tests that need no PROCESS run, and `mda.default_drivers` grew
an `Optimise` arm that reads the equality/inequality counts off the `Optimise` node itself.

### Defects found and fixed

- **Iteration variable 4 had no path into the physics.**
  `.physics.temp_plasma_ion_vol_avg_kev`, `.physics.temp_plasma_electron_density_weighted_
  kev` and `.physics.temp_plasma_ion_density_weighted_kev` were boundary inputs with no
  producer, so ion temperature was structurally disconnected from the electron temperature
  the optimiser varies. Every *value* was right; every derivative w.r.t. `x4` was ~0 where
  PROCESS's is O(1), and `FusionRates`' `sigmav_dt_average` (roughly `T²`) had a relative
  sensitivity of `2.4e-16`. Both pure functions were already ported — only the nodes were
  missing. Fixed by registering `IonVolAvgTemperature` and `ParabolicProfileValues`.
  **This is the defect class §8's harness cannot reach, found by the thing built to reach
  it**, and it is now the *first* of three instances (§11.6).
- **`fast_alpha_beta`'s clamped square root returned `nan` from `jacfwd`.**
  `jnp.sqrt(jnp.maximum(0.0, x))` with the clamp **active** on this run — the same trap §9
  records for `costs.py:2874-2888`. Fixed with the double `jnp.where`; value-identical, and
  it took the SAND Jacobian from 17 non-finite cells to 0. It had been latent because
  nothing downstream of `.physics.beta_fast_alpha` fed a condition until constraint 24 got a
  producer. **Generalise: a gradient defect only becomes visible once something downstream
  reads into a condition, so closing a producer gap routinely exposes one.**
- **Seven 1-tuple returns** (`thermal_cryo.py` ×6, `vacuum.py` ×1) — the
  `ZTfInsideHalf` bug class, third wave. Invisible to `PicardDriver`, fatal to
  `Residualise`. The full sweep is in `thermal_cryo.md`.

### Two constraints un-blocked, and the registration gaps behind them

`optimise_design.md` §2.4 recorded constraints **16** and **24** as permanently INERT. Both
are live, which matters because an `Optimise` over 12 of PROCESS's 14 active constraints is
a *different problem* and comparing its answer to PROCESS's would be meaningless.
`sand.constraint_nodes` now raises on any active `icc` entry it cannot assemble rather than
dropping it.

- `PlantElectricProductionReactor` (`electric_production.py`) — the `ireactor == 1`
  arm of `plant_electric_production`, whose five "self-referential" fields are dead reads on
  that arm. `.costs.ireactor` becomes a two-armed topology `Switch`. **This also gave
  `.costs.coe` — the run's own objective — a real dependence on the design along the
  net-electric-power path, where before it read a boundary input.**
- `StellaratorBetaAndStoredEnergy` (`plasma_physics.py`) —
  `StellaratorBetaAndRhoStar` minus the one output (`rho_star`) that actually collided with
  `DimensionlessPlasmaParameters`. Dropping the whole node to resolve that collision had
  cost `.physics.beta_total_vol_avg` and `.physics.e_plasma_beta` their only producer as
  collateral.

Both are further instances of §9's "no systemic check exists for *is the right node
registered*" (§11.6).

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

The resulting **`+17.604 MW`** offset accounts for eighteen harness disagreements exactly
and linearly, through `Acpow` and the cost chain to `.costs.coe`. Recorded in
`mda_harness.EXPLAINED_DISAGREEMENTS`, deliberately not suppressed. It is also the largest
identified reason the port's optimiser does not land on PROCESS's `x`.

**Consequence for §8's framing:** the harness's `expected` column is PROCESS's *reported*
state, which for any field the report pass recomputes is not the state the solver used.
That is a third category alongside `KNOWN_UNVERIFIABLE_OUTPUTS` and
`EXPLAINED_DISAGREEMENTS`, and **nothing currently detects it in general** — a field is
noticed only when a consumer of it disagrees. See `test_harness.md`, which now names it.

## 11. Session wrap — verified state, and what to pick up first

**Read this section first.** The consolidated state and the priority order.

### 11.1 Verified state

See this file's **Verified state** table at the top — it is the same measurement and is
kept in one place rather than two.

**The 34 disagreements are two causes, not thirty-four**, and all 34 are in acyclic blocks
(every driven block reproduces PROCESS exactly). Eighteen are the single `+17.604 MW`
offset propagating linearly — verified arithmetically, identical to six decimals across
seven fields, with the sign correctly flipping on `p_plant_electric_net_mw` (net = gross −
recirc) — whose cause is an inconsistency **in PROCESS**, not the port (§10, "A finding
about PROCESS itself"). The rest are the `VacuumOld` tolerance difference (§8) and its
propagation into Account 224 and the cost chain.

**The measurement itself is now accounted for.** `ComparisonReport` carries
`owned_total`/`unaccounted` and enforces that every owned variable lands in exactly one
bucket: 550 walked, 0 unaccounted. Array-valued outputs are compared elementwise and
reported as one `Disagreement` carrying the worst element (`shape`/`index`/`n_off`);
non-comparable pairs become explicit `errors` instead of vanishing.

### 11.2 The Stage C timing, corrected — compilation, not the optimiser

The headline Stage C2 time has trace and compilation **inside** the timed region
(`run_sand_harness.py` wraps `solve_schedule(...)`, and `VmconDriver.__call__` builds its
`jax.jit` closures inside that call). Split by timestamping the SQP callback, the great
majority of the wall time is one-time trace + compilation and the per-iteration cost is
milliseconds. Two consequences, both still live:

- **The optimiser plumbing is the per-iteration cost, not the graph.** `pyvmcon`'s line
  search makes several `evaluate` calls per iteration, each crossing the JAX/numpy
  boundary, which is roughly an order of magnitude more than the Jacobian itself. That is
  where to look if per-iteration cost ever matters — not at the derivative.
- **`VmconDriver.__call__` rebuilds its `jax.jit` closures on every call**, so each is a
  fresh cache key and every solve repays the compile. One solve absorbs it; **a `Scan` of N
  points would recompile N times.** Given the measured batching win (§11.4), this is the
  single change that would matter most for the scan use case, and it is small: hoist the
  jitted callables so they are built once per graph shape. **Open.**

### 11.3 Boundary inputs — audited, and much smaller than it looked

Full record: **`_audit/boundary_inputs_audit.md`**, which is maintained there, not here —
including the split of the 375 unowned inputs into genuine inputs, off-path-for-a-
stellarator fields, and real cut edges. Two structural points from that audit belong here:

- **`len_tf_coil` does not hide a cross-subsystem cycle** — measured, not inspected: its
  producer's inputs are owned by `StellaratorScalingFactors`, unreachable from all four
  readers. Closing the remaining cut edges therefore does not create new cross-subsystem
  SCCs.
- **Closing a boundary input can enlarge an existing cycle**, and did:
  `FusionTotalsNoBeam` giving `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`
  their first producers took the density/fusion cycle from 4 nodes to 6 and from one cut to
  two (§5). Expect that, rather than expecting each closure to be inert.

The audit's two headline findings are **both closed**: `ProfileValues.rho` (see §11.6) and
`mda_harness.compare`'s dropped arrays (§11.1).

### 11.4 Structure — provenance vs derived clustering, measured

Recorded in `_audit/switch_elimination_design.md` §11. Comparing the port's module layout
(provenance) against `Blocking.scc`: **every cycle is contained within one subsystem**
(`stellarator`, `physics`, `costs`) **and spans several files inside it**, while
cross-subsystem edges are numerous — so subsystems are heavily coupled but *acyclically*,
which is why only 12 of 131 blocks need driving. **The subsystem is the right grain for a
model group; the file is not.**

Caveat: this measures PROCESS's *filing habits*, since provenance here is the module tree.
Declaring the grouping on physical grounds and finding the same containment would be the
stronger claim.

Also measured (`optimise_design.md` §4.4): the **driven** MDA schedule — Newton/Picard
blocks, `while_loop`s included — `vmap`s cleanly, with a large throughput win at B=64,
guarded against dead-code elimination. The payoff target is `Scan`, which today re-solves
every point from scratch and would recompile every point besides (§11.2).

### 11.5 Design work recorded today, not yet implemented

**The settled design is `_audit/model_tree_design.md`**, which supersedes
`switch_elimination_design.md`'s three-tree/`materialise` shape (that file's banner maps
which of its sections stay live — the measurements do, the design does not). One tree,
not three: `eqx.Module` namespaces with typed snake_case slots, occupants carrying their
own settings as enum-typed `eqx.field(static=True)` fields, `Machine()` defaults =
PROCESS defaults, and a single `machine_from_indat` factory as the only place an `i_*`
integer is ever read. `Switch`/`Alternative`/`Configuration` are deleted by its §8
step 4; the migration order, per-step gates, and the one small cottax change
(`ModelNamespace` in `node_and_names`, minting `GetAttrKey` slot names so a node name
is a working address into the machine tree — no rendering change needed) are in its
§§3 and 8.

**The cottax hierarchical-`NodePath` half has landed** (`~/jaxgraph` `789df8b`), so
nothing upstream blocks step 3. §12's `free` remains the companion question — the tree
is how a selection is *spelled*, `free` is what one *means* — but is no longer framed as
a prerequisite: the two share only the boundary postcondition
(`model_tree_design.md` §6).

### 11.6 Priority order

1. **[CLOSED — see 11.8] The cold-start gap.** It was the ceiling on the whole result
   and it is not there any more: the port runs its own pipeline from a cold `IN.DAT`,
   and both SAND and MDF now solve from one.
2. **[CLOSED — see §11.11, and the conclusion is not the one this item predicted.]**
   ~~The `x109` disagreement, and the L-mode reset behind it.~~ The L-mode reset was a
   real missing producer and is now `plasma_profiles.LModeProfileReset` — the **eighth**
   instance of that class. What it closed is the **warm/cold** split (§11.10's "several
   local minima?"), not `x109`, which it moves by 0.05 %. `x109` and `x56` are both
   **flat directions** of the port's own problem: pinned at PROCESS's value and re-solved,
   `x109` gives a *lower* objective than the port's free optimum (−0.017 %) and `x56`
   costs +0.004 % to move 5.6 %. The inference "all four solver/start combinations agree,
   therefore a model difference" is **withdrawn**.
3. **`free`, and what a design variable means** — §12. This is the item that grew out of
   `x109` being a flat direction nobody could see, and it subsumes the switch work (§11.7
   below is *not* the reason to do it; §12.1's postcondition is).
4. **Hoist `VmconDriver`'s jitted callables** (§11.2) — small, and the prerequisite for
   batched or scanned solves being worth anything.
5. **The "is the right *node* registered" check** — now **five** instances
   (`i_cost_model`'s 43 nodes, `PlantElectricProduction`, `StellaratorBetaAndRhoStar`,
   `plasma_profiles.py`'s two, and `FusionTotalsNoBeam`), **every one found by a downstream
   consumer, never by a check**. `switch_audit` checks the kwargs of nodes that are already
   present. The recommended shape is unchanged and now clearly worth building: walk the
   ported units' node classes and report every one that is written but registered nowhere,
   with the reason recorded beside it.
6. **[CLOSED.]** ~~A third measurement hole, of the same family as the dropped arrays:
   `mda_harness.compare`'s `atol=1e-9` makes any field whose natural magnitude is below
   that vacuously agree.~~ **Closed by `atol=0.0`** — the comparison is now purely
   relative. Two measurements settled it, and they had to be taken in that order.

   **On the code as it stands, the floor is inert.** Of 499 agreements, **0** depend on
   it: every one still agrees at `atol=0.0`, and the three fields this item named as
   "not actually checked by anything" (`.neoclassics.temperatures`/`dr_temperatures` at
   ~1e-15 J, `.physics.sigmav_dt_average` at 6.4e-23) agree **bit-exactly**, to every
   digit. Taken alone that reads as "the trap was never sprung, remove it for free".

   **That reading is wrong, and reintroducing the bug proves it.** The floor was inert
   only because the one wrong answer it was hiding had been fixed by another route
   (`ProfileValues.rho`, closed below). Set `rho` back to `0.0` — the old `r_eff`
   binding's value — and re-run the harness:

   | field | port | PROCESS | `atol=1e-9` | `atol=0.0` |
   |---|---|---|---|---|
   | `.neoclassics.densities` | 2.357e+20 | 2.016e+20 | DIFFER | DIFFER |
   | `.neoclassics.dr_densities` | -0.0 | -6.104e+19 | DIFFER | DIFFER |
   | `.neoclassics.temperatures` | 1.9997e-15 | 1.1705e-15 | **AGREE** | DIFFER |
   | `.neoclassics.dr_temperatures` | -0.0 | -1.215e-15 | **AGREE** | DIFFER |

   The two Joule-valued fields are **71 % and 100 % wrong** and the old floor reports
   both as agreements. So `atol=0.0` is not a tidy-up: it converts a demonstrated,
   previously-invisible wrong answer into a reported disagreement, and it doubles what
   the harness sees of that specific bug (2 disagreements → 4). `boundary_inputs_audit.md`
   §7.1 and `test_harness.md` were right about the mechanism; this item was wrong only in
   its present tense, and only because the bug behind it had since been fixed.

   **The vacuous category that is *also* there is larger and different: 73 of the 499
   agreements are zero on *both* sides**, port and PROCESS alike
   (`.physics.beta_fast_alpha`, `.current_drive.c_beam_total`, 40-odd `.costs.c22*`
   accounts, ...). They are real agreements and stay counted, but they say *"this path is
   switched off in this configuration"*, not *"the port reproduces PROCESS"* — no
   arithmetic in the node was exercised. Now reported as
   `ComparisonReport.trivial_agreements` and printed beside the array count
   (`agreements: 499 (of which array-valued: 23, both-sides-exactly-zero: 73)`), so the
   headline number cannot be read as more coverage than it is. Policy and the `rho=0.0`
   numbers above are pinned by `test_mda_harness.py`.

   **What this leaves open, stated so it is not read as closed:** those 73 are ~15 % of
   the agreement count and the port's evidence for each is nil. Whether each *should* be
   zero on this run is a separate question, unasked — the reference `IN.DAT` is one
   stellarator configuration, and a second configuration that switches some of those
   paths on is the only thing that would actually exercise them.
7. **PROCESS's report-pass/solve-pass inconsistency as an undetected category** (§10) —
   nothing detects it in general; a field is noticed only when a consumer disagrees.
8. **The switch-elimination work** (§11.5) — per the design doc's own order: enum-aware
   `switch_audit` first, so the net that caught five bugs is not lost in the act of acting
   on it.
9. **[CLOSED] `boundary_inputs_audit.md` §7** — all seven items are done; that
   file's §9 records what they bought.

**Closed, and how** — kept here because each closed item names a defect class that will
recur:

- **`mda_harness.compare`'s dropped arrays.** The hole was **25 variables, not the 29
  previously estimated** — 21 arrays were silently *agreeing*, 4 silently *disagreeing*.
  Closed by elementwise array comparison plus the `owned_total`/`unaccounted` accounting
  invariant (§11.1).
- **`ProfileValues.rho`.** Now `eqx.field(static=True, default=0.6)` — PROCESS's own literal
  at `neoclassics.py:290` — instead of `Input(lambda s: s.neoclassics.r_eff)`, a field
  PROCESS never assigns; registered in `mda_harness.STATIC_KWARGS_WITHOUT_BACKING_FIELD`.
  `.neoclassics.densities`/`dr_densities` went from disagreeing (`dr_densities` was `-0.0`
  against `-6.1e19`) to agreeing.
- **`design_scale`'s missing floor.** `1 / x_start` conditioning tested `flat_start != 0.0`
  — exact zero only — where PROCESS's own `check_iteration_variable` rejects
  `abs(value) <= 1e-12`. `.power.qac` is exactly `0.0` on a seeded env and `-3.8e-27`
  after a solve, so **restarting one solve from another's answer** handed VMCON a scale of
  `-2.6e+26` and its QP died. Reachable only by restarting, never from a cold start, which
  is why every run passed until one was tried. Now floored at PROCESS's own threshold,
  with `scale = 1` rather than PROCESS's hard error, because SAND legitimately owns
  coupling unknowns that converge to ~0 — unscalable, not ill-posed. Seven tests pin it.
  A **second** argument for §12's dimensional scaling: a start-dependent scale conditions
  the same problem differently warm and cold.
- **`c62`'s Jacobian row**, "the only cell in the whole Jacobian disagreeing for an unknown
  reason". Diagnosed and closed, **and the cause was not local to `c62`**:
  `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw` had no producer, so
  `t_alpha_confinement = nd_alphas / fusden_alpha_total` had a frozen denominator and its
  temperature derivative was structurally absent. `FusionTotalsNoBeam`
  (`plasma_physics.py`, the `else` arm of `stellarator.py:2002-2054`, three
  identities) gives them producers. Measured on the `c62` row: x4 `5.10e+00*` → `5.42e-04`,
  x6 `6.66e+00*` → `3.99e-05`, x109 `1.41e-01*` → `3.25e-07`, with c2/c8/c17/c18/c67 also
  improving. **The only starred cells left in the whole Jacobian are `objf` and `c16` (the
  report-pass/solve-pass inconsistency) and `c24 x3` (a division by PROCESS's own exact
  zero).** This is the **third** instance of the same defect class as iteration variable 4
  (`optimise_design.md` §10.5a): *a missing producer that every value test passes and only a
  gradient sees.*

### 11.7 A recurring lesson, worth keeping

Confident diagnoses that measurement overturned: `i_plasma_ignited` as the cause of the
1.2% confinement gap (it was the `q95`/`iotabar` binding); `sig_tf_wp_max = 0.0` as the
cause of c32's `inf` (it was the coil-island placeholder feeding `a_tf_wp_no_insulation`,
one node further up); "no speed win at this size" (an unjitted cold timing); "costs is
unported" (23 nodes existed); "instance fields cannot drive a node's reads-set" (they can);
and `c62`'s row as a local problem (its cause was three fields with no producer, one node
further up — the same *shape* as the c32 correction). In every case the correction came
from running something, not from reading harder.

Two traps have now appeared three times each and should be assumed present until checked:
**`jnp.sqrt(jnp.maximum(0, x))`** — value-correct, derivative `nan`, visible only to a
gradient test — and **a missing producer** — every value test passes, and only a gradient,
or a consumer far downstream, ever notices. The missing producer has now been seen
**eight** times, and §11.11 adds a third way of noticing it: **two different starting
points converging to different answers**, because a warm seed taken from PROCESS's own
`DataStructure` silently supplies the value the absent node would have produced and a cold
seed does not. Any solve that is warm-started off PROCESS is blind to this whole class.

A third entry for the list of confident diagnoses measurement overturned: **"`x109` is a
model difference because all four solver/start combinations agree"** — they agree because
the direction is flat, and at PROCESS's own `x109` the port is feasible with a *better*
objective than at its own converged answer (§11.11). Solver agreement is evidence about the
optimiser, not about the model.

A fourth, and this one caught *this document's own author mid-correction*. Measuring
`atol`'s effect on the current code showed it carrying **zero** agreements, which reads
cleanly as "the trap was never sprung" — and §11.6 item 6 was briefly rewritten to say so.
It is false. The floor was inert only because the single wrong answer it had been hiding
was fixed by an unrelated change; reintroducing that bug shows the floor concealing a 71 %
and a 100 % error. **A guard that currently catches nothing is not thereby shown to be
unnecessary — it may be that nothing is currently wrong.** The measurement that settles a
guard's worth is taken with the defect *present*, and the cheapest way to get one is to
reintroduce a defect the project has already fixed and still has written down. This
project's closed-items list is, in that sense, a test corpus nobody has been using as one.

The companion rule survives the correction and is worth keeping in its own right: **a fix
that changes no measurement has not been shown to have fixed anything.** A per-field
relative floor — item 6's proposed fix — would have passed every gate, moved not one
number, and closed the item, all without either establishing that a hole existed or that
it had been shut.

### 11.8 The cold start, closed — and what it took

**The port runs its own pipeline from a cold `IN.DAT`, and both architectures now solve
from one.** This was §11.6's item 1 and the stated ceiling on the whole result. It closed
in four distinct pieces, and the shape of them is the point: only one was an optimiser
problem.

| | before | after |
|---|---|---|
| MDA schedule, cold | `nan` before any driver ran | **137 blocks, 0 failures** |
| SAND, cold (C3) | **0** SQP steps | **91 steps, conv 1.86e-09** |
| MDF, cold | did not exist | **127 steps, conv 4.5e-09** |
| SAND, warm (C2) | 62 steps | 62 steps (unchanged) |

**1. A missing model, not a missing edge.** `load_stellarator_config` was unported, so
every `.stellarator_config.*` was `0.0` and `.tfcoil.n_tf_coils` was `0` cold — the first
division by it poisoned everything downstream. Ported as `StellaratorMachineConfig`
(unit #8), which owns 34 fields and holds the parsed JSON as a static payload. The rest of
`st_new_config`'s arithmetic turned out to be **already ported** in chunk 1C.

**2. A driver, not a producer.** `Intersect`'s unknown is seeded from
`.tfcoil.dr_tf_wp_with_insulation`, a model-computed field that is `0.0` cold — and,
measured, the `Intersect` residual is **exactly flat** (`-8329.4857`) everywhere below
`x ~ 0.1`, so the derivative there is zero and Newton cannot move at all. Not a bad guess,
a *dead* one. `SeededNewtonDriver` falls back to a guess derived from the block's own
`ConditionMap.context` — and the guess is **PROCESS's own**,
`(r_coil_minor / 10) ** 2` from `winding_pack_pre_intersect`. The fallback fires only when
the seeded value is unusable, so warm runs are unchanged bit for bit. A test now pins that
every `RootFind` has a fallback; it immediately caught a second one
(`DuctDiameterRootFind`, given PROCESS's own `1e-6`).

**3. A harness bug that made "cold" mean "impossible".** `run_sand_harness._seed`'s
docstring said coupling unknowns come from an MDA run — "exactly what an MDF architecture
would hand iteration 0" — and the code tried `ground_truth(base, var)` **first**. Every
coupling unknown *has* a `DataStructure` field holding the dataclass default `0.0`, so the
lookup always succeeded and the fallback was never reached. Twelve of twenty-three unknowns
started at exactly zero: net electric power `-1.9e6` MW, `coe = 1.0e25`. Fixed, and the
rule now differs by stage because the stages mean different things — C2 is *start where
PROCESS ended*, C3 is *start from the input file*, which carries design variables only.

**4. The `nan`s were a symptom of (3).** 46 non-finite Jacobian cells in exactly two rows
(`objf`, `c16`), from `x ** p` with `0 < p < 1` at `x == 0` — value `0`, derivative `+inf`,
`inf * 0 = nan` — at `buildings.py:282` and three sites in `costs.py`, reachable only
because the cryogenic loads were seeded to zero. **Fourth instance** of the
unbounded-derivative-at-a-boundary class (§9's `sqrt(maximum(...))`, §10.5b's
`fast_alpha_beta`).

**Two hypotheses were refuted by measurement, and one of them would have made things
worse.** Recomputing `residual_condition_scales` at the cold point — the obvious fix —
takes the condition number from `1.14e23` to `6.71e36`, because `1/|u|` carries no
information when `u == 0`. And the "genuinely non-convex at a cold design" theory was
wrong: with a consistent start the conditioning is `2.87e4`, indistinguishable from the
warm point's `2.08e4`. **No trust region, Hessian safeguard or homotopy was needed.**

`VmconDriver` now refuses a non-finite problem at its boundary, naming the offending
conditions — the port's missing analogue of PROCESS's own `constraint_eqns` guard
(`constraints.py:1997-2002`). Without it a `nan` row reaches `cvxpy` and returns as *"the
problem seems to be non-convex"*, which points at the Hessian when the fault is in the
constraint matrix. That message cost hours; it now costs a line.

### 11.9 Two architectures, and what each is for

`mdf.py`/`run_mdf_harness.py` now sit beside `sand.py`. **PROCESS is MDF**:
`Caller.call_models` converges the whole pipeline inside every evaluation and VMCON sees
only `ixc`. So MDF's design vector *is* `ixc` (8), its conditions *are* `icc` + the figure
of merit (15), and **its Jacobian is compared to PROCESS's finite differences with no
reduction at all** — where SAND needs an equilibrated Schur complement off a raw condition
number of `2.4e29`. That makes MDF the like-for-like comparison.

- **The gradient through the inner solve is correct**, established rather than assumed:
  forward-mode AD through 12 `lax.while_loop`s and a root find, against a central
  difference of the same map — worst disagreement `3.3e-06`, no cell above `1e-5`.
  Reverse mode is unavailable at any price (`while_loop` has a JVP and no transpose rule).
- **~98 % of solve time is `pyvmcon`/`cvxpy`, not the model or the derivative.** One MDF
  condition evaluation — a whole converged MDA — is **0.53 ms** against PROCESS's
  `call_models` at 43.6 ms; its Jacobian **0.71 ms** against `fcnvmc2`'s 741 ms
  (**82x** and **1038x**).
- **What cottax lacks is one field's type**: `Drive.body` is a `Graph`, run acyclically,
  where MDF needs a `Step`. Made locally as an `MdfConditionMap`; the unmodified
  `VmconDriver` cannot tell the difference, which is the measure of how narrow the gap is.

**SAND keeps two things MDF does not**: it exposes the coupling to the solver, so an
ill-posed coupling shows up as a singular equality block instead of hiding inside a Picard
loop (in MDF an identity fixed point silently returns its seed); and it has no inner
tolerance to interact with the outer one. MDF's honest weakness is that its conditions are
defined only to the inner solve's tolerance and its `while_loop` trip counts are piecewise
in `x`. Measured consequence here is small (inner residuals `<= 1.2e-8` relative), but
tightening the inner solvers makes the outer solve **worse** — measured, not explained.

### 11.10 A second SQP, and what it settled

`SlsqpDriver` (`core/solver/drivers.py`) answers `Optimise` on exactly the problem
`VmconDriver` receives — both build it through the shared `scaled_problem`, so the
comparison is controlled: same scaling, same bounds, same `jax.jacfwd` Jacobian, only the
solver differs.

| | SAND warm | SAND cold | MDF warm | MDF cold |
|---|---|---|---|---|
| VMCON | **62** | **91** | oscillates, no certificate | **127** |
| SLSQP | 107 | **73** | **185, `max|eq| 2.6e-12`** | **39** |

**The two VMCON/SAND cells are superseded**: a later re-run of the whole ladder gives
**42** warm and **88** cold (see the Verified-state table, which carries the current
figures). Only those two were re-measured, so the rest of this table is left as recorded
rather than silently mixed with numbers from a different state of the tree. The
comparison the table exists to make — VMCON against SLSQP under identical scaling — is
between cells measured together, so re-measuring one solver alone would break it. Redo
the row as a row, or not at all.

**Cold, on SAND, the two solvers land on the same point to five or six digits** (x56
31.5688 vs 31.5685; x109 0.0299361445 vs 0.0299361385). Two independently written SQPs,
different QP solvers, different line searches, from a cold start. **What that licenses is
narrower than this paragraph used to claim — see §11.11.** Solver agreement establishes
that the *stationary point* is well determined; in a nearly-flat valley that is exactly
what one gets, and it does not distinguish a model difference from flatness. Measured:
both `x109` and `x56` are flat, and at PROCESS's own `x109` the port is feasible with a
*better* objective than at its own answer. Everything else is within 3 %.

**The "within 0.5 %" figure this paragraph used to carry was wrong** — corrected against a
re-run of the whole ladder rather than restated. The measured cold-start distances to
PROCESS, all eight: `x3` 0.28 %, `x2` 0.45 %, `x10` 0.61 %, `x4` 0.89 %, `x6` 1.95 %,
`x59` 2.67 %, `x56` 10.6 %, `x109` 10.9 %. Only two are inside 0.5 %. The shape of the
claim survives — a long tail of near-agreement and two outliers — but the tail is 3 %
wide, not 0.5 %.

**Warm and cold do not land on the same point** — **[SUPERSEDED; see §11.11. They do
now, and the cause was neither multi-modality nor flatness but one missing node.]** C2
finished at `objf` 1.217757 and C3 at 1.215038, 0.22 % apart, with `x2` 4.7093 against
4.6827; both converged and both feasible, which correctly ruled out a tolerance artefact
and was then over-read as "several local minima or a flat direction". The cause was
`plasma_profiles.LModeProfileReset`: the warm seed came from a `DataStructure` where
PROCESS had already applied the L-mode reset and the cold seed did not, so the two stages
were solving different problems. The discriminator suggested here ("restart C3 from C2's
answer") was run, and it is a **trap** — `VmconDriver`'s unfloored `1/x_start` scaling
makes any restart from a solved point diverge, which is how that driver defect was found.
See §11.11.

**The objective itself differs from PROCESS's by 1.7 % at PROCESS's own converged point**
(Stage A: port `1.235974`, PROCESS `1.214917`), so port and PROCESS `objf` *values* are
not comparable and should not be read as a convergence gap. It is a cost-of-electricity
model difference, in the same family as `x109` and separate from the optimisation.

**Do not switch VMCON to CLARABEL.** `optimise_design.md` §4.2 records that PROCESS uses it
and the port does not, which reads like a discrepancy to fix. Measured: warm C2 converges in
62 iterations under OSQP and runs to `max_iter` **without converging** under CLARABEL, and
at the cold point both fail. Recorded so the next reader does not "fix" it.

### 11.11 `x56` and `x109` — both diagnosed, and neither is a model difference

**The one-line answer: `x56` is a flat direction of the port's own constrained problem;
`x109` is not.** The port's answer is the point where the design first reaches
`(Te + Ti)/20 == 0.65` — the kink in `fast_alpha_beta`'s clamped square root, which the
port stops on to `1.9e-09` — and the solvers are held there by a `c24` Jacobian row that is
not the derivative of the `c24` they evaluate (`_audit/closed/x109_hypotheses.md`). Treating the
two variables as one thing is what sent two investigations down the same wrong road. **The
eighth missing producer that everyone was looking for is real but is the cause of something
else entirely.** Everything below was run against
`tests/regression/input_files/stellarator_helias.IN.DAT` off one cached PROCESS run.

#### The missing producer, found and ported — but it is not `x109`'s cause

`LModeProfileReset` (`models/physics/plasma_profiles.py`) owns the seven fields
`PlasmaProfile.parabolic_parameterisation` resets to L-mode values
(`process/models/physics/plasma_profiles.py:92-117`), registered under the
`.physics.i_plasma_pedestal == 0` arm. **Eighth instance of the missing-producer class**
(§11.7), and the first found by a *third* method — not a gradient, not a downstream
consumer, but **two starting points disagreeing**.

Its post-condition is unconditional, which is why it can be a node with no `Input` and no
guard: PROCESS's `if` is *"if any of the seven differs from its L-mode value, set all seven
to their L-mode values"*, so when the guard is false every field already holds what the
body would assign. The guard governs only a `logger.error`.
`TestLModeProfileReset` fuzzes all seven arguments against PROCESS and pins that.

**What it fixes is the warm/cold split, not `x109`.** Four of the seven fields are read by
the graph (`profiles.DensityProfile` reads three, `SynchrotronRadiationPower` reads
`tbeta`) and all four were unowned. So a **warm** solve seeded from PROCESS's converged
`DataStructure` got the reset for free — those fields are already `0` there — and a
**cold** one carried the input file's `nd_plasma_pedestal_electron = 4e19` /
`nd_plasma_separatrix_electron = 3e19` into `DensityProfile`, whose single formula is the
pedestal one and only degenerates to the parabolic profile once they are zero. C2 and C3
were solving **different problems**:

| | before | after |
|---|---|---|
| SAND C2 `objf` | 1.217757336 | 1.217757336 |
| SAND C3 `objf` | **1.215038106** | **1.217757336** |
| C3 `x2` | 4.682670 | 4.709285 (C2: 4.709284) |
| median distance to PROCESS's `x`, cold | 1.42e-02 | 8.62e-03 |

**§11.10's "the port's problem has either several local minima or a direction flat enough
that the two starts stop in different places" is answered: neither.** It was one missing
node. The controlled A/B is direct — zero `nd_plasma_pedestal_electron`/
`nd_plasma_separatrix_electron` on the cold `DataStructure`, change nothing else, and the
cold solve moves onto the warm one's answer to nine digits. Tightening the tolerance had
already ruled out a stopping artefact: at `tolerance = 1e-13` the two still converged to
`1.217757336` and `1.215037642`, both with `conv ~1e-11` and feasible.

The effect on `x109` is **0.05 %** (`1.088e-01` → `1.083e-01`) and on `x56` **0.5 %**
(`1.062e-01` → `1.008e-01`). So the leading hypothesis in §11.6 item 2 was a real defect
and *not* the explanation it was proposed as. This should have been visible without
measuring — C2 seeds from a `DataStructure` where the reset has already fired, and C2's
`x109` was 10.8 % off — which is a small lesson of its own: §11.7's list is about
diagnoses that reading overturned, and this is one reading could have narrowed for free.

#### `x56` and `x109`: measure the *cost* of agreeing with PROCESS, not the distance

The decisive experiment for both is the same and it is cheap: **pin the variable at
PROCESS's own value, re-solve the other seven, and read off what it costs the objective.**
A flat direction is free to move along; a genuine model disagreement is not. Run on the
corrected graph, `tolerance = 1e-11`, continuation from the free optimum
(`objf 1.217757336`):

| pinned | value | `objf` | vs free | `conv` | `max\|eq\|` | feasible |
|---|---|---|---|---|---|---|
| — | — | 1.217757336 | — | 8.6e-12 | 1.2e-13 | yes |
| `x56` | 33.540284 (52 % of the way to PROCESS's 35.32) | 1.217805177 | **+0.0039 %** | 9.4e-12 | 5.2e-13 | yes |
| `x109` | 0.033590406 (**PROCESS's exact value**) | 1.217549493 | **−0.0171 %** | 9.4e-12 | 3.3e-09 | yes |

Read those two rows carefully:

- **`x56` costs 39 parts per million to move 5.6 %.** The direction is flat. `x56`'s
  solver- and start-dependence (§11.10's 3.9 %–10.6 % spread, and MDF C3 landing at 36.20
  where SAND lands at 31.76 — on *opposite sides* of PROCESS's 35.32) is what a flat
  direction looks like, not a wrong formula. It is interior to its bounds `[1, 50]` at
  every landing.
- **`x109` at PROCESS's own value is feasible with a `objf` 0.017 % *lower* than the port's
  free optimum** — re-verified to `max|eq| 2.1e-12` with no inequality violated. A
  constrained minimum cannot be improved by adding a constraint, so the port's converged
  answer is **not** the global optimum of the port's own problem. The landscape is
  confirmed multi-modal — a barrier of height `1.2e-05` at `x109 ≈ 0.0300` and a strictly
  better feasible region beyond `0.031`, reaching `objf 1.21757404` at `0.0320` with
  `max|eq| 4.0e-14` — but **the port's answer is not the other minimum of it**. Released
  *inside* the better region, both VMCON and SLSQP walk back out of it uphill; with
  `x109 >= 0.031` imposed they converge **on that bound**. A correct local method does not
  do that, which is what points at the Jacobian and not at the landscape. `x109` is
  interior to `[0.0001, 0.4]`. See `_audit/closed/x109_hypotheses.md`.

**§11.10's inference from "all four solver/start combinations agree to five or six digits"
is therefore wrong**, and the reason is worth keeping: two independent SQPs agreeing tells
you the *stationary point* is well determined, which in a nearly-flat valley is exactly
what you get and says nothing about whether the model is right. The claim
*"`x109` is a model difference, not an optimisation artefact"* is withdrawn. The
measurement that separates the two is the pinning cost, not solver agreement.

#### What actually determines the landing point

At the free optimum the active set is `c2`, `c16` (the two equalities), the fifteen SAND
residual equalities, and exactly four inequalities — `c24` (beta limit), `c83` (place for
blanket), `c62` (thermal He) and `c35` (quench). Twenty-three unknowns against twenty-one
binding conditions leaves **two** free directions, and the objective's own gradient in
those directions is what would fix them, and **it is not what does**.

The account this section used to give — that the `objf` row of the Stage B Jacobian
disagrees with PROCESS's by 18–34 % in every column (§10.5c), and that a 20–30 % gradient
error along a flat valley moves the landing point by ~10 % — has been **measured and is
wrong**. Against a central difference of the port's *own* condition map the `objf` row is
correct to `7.7e-10`; §10.5c's 18–34 % is a *port-vs-PROCESS* difference, not an error in
the port's own descent direction. What fixes the landing point is one row of the
**constraint** Jacobian: `c24`, at a point where its function is not differentiable (see
this file's opening summary, and `_audit/closed/x109_hypotheses.md` §3). Of 690 cells exactly the
two `c24` cells disagree with that same central difference, and `c24` alone drifts like
`h^0.52` along the null direction where every other condition drifts like `h^2.00`. The
active-set reading above was re-measured and stands: same four inequalities, `A` is 21×23
of rank 21, LICQ holds.

**Architecture is excluded as a cause.** MDF warm converges at `x109 = 0.0299518328`
against SAND's `0.0299518330` — nine digits — so exposing the coupling to the optimiser is
not it. That also supersedes §11.10's "MDF C3 landing at 36.20 where SAND lands at 31.76":
warm, the two give `x56` 31.7606 / 31.7570.

§10.9 item 3 (pin `.heat_transport.p_plant_electric_base_total_mw` to PROCESS's own 89.461
and re-solve) is **no longer the controlled test for `x109`** — it tests the `objf`/`c16`
difference, which is real and is a separate question. **The cheaper substitute was tried and is not a substitute**: dropping `c16`
from the problem entirely does not converge and is not informative — `c16` is the net
electric power lower limit, so without it the objective runs away (`objf` 0.656, `x3` and
`x56` both pinned at their upper bounds). Recorded so it is not retried.

#### Two defects in the driver, found on the way

- **`VmconDriver`'s `1/x_start` design scaling has no floor.** The guard is
  `np.divide(1.0, x, where=x != 0.0)` — exact zero only. `.power.qac` is an unknown that is
  *identically* `0.0` on any seeded env and `-3.8e-27` after a solve, so **restarting a
  solve from its own answer** hands VMCON a scale of `-2.6e26` for that column and the
  QP is destroyed: from C2's converged point (`conv 9.9e-10`) a restart wanders to
  `objf 1.2228` and fails in 19 iterations. This is a trap for exactly the check §11.10
  recommended ("restart C3 from C2's answer"), which is how it was found. A relative floor
  (scale `1.0` when `|x_start|` is below, say, `1e-12` times the vector's median magnitude)
  would fix it. **Open.**
- **`pyvmcon`'s reported convergence parameter is not a restartable property of the
  point.** It is computed from the QP step, which uses the accumulated BFGS Hessian; the
  same point re-entered with `B = I` reports `6.9e-04` where the run that stopped there
  reported `9.9e-10`. Do not read a small `conv` as "this point is a KKT point to that
  tolerance" — read `max|eq|` and the worst inequality, which are properties of the point.
  C2 at `tolerance = 1e-8` stops with `max|eq| = 2.7e-06`; at `1e-13` it reaches `9.2e-08`.

#### Cost of the new node, stated

MDA harness **492 → 499 agreements**, 34 disagreements unchanged, 550 → 557 owned
variables, 0 unaccounted. `pytest functional_process -q`: **3697 passed, 0 failed** (the
seven new tests are `TestLModeProfileReset`'s). **One regression, in the solver and not in
the model:** at the harness's default `tolerance = 1e-8` the SAND cold solve now runs the
full 100 iterations and stalls at `conv 1.7e-06` instead of converging in 88, and MDF's
cold solve runs 200 without converging instead of 127 with. Both land on the right point
(SAND C3 agrees with C2 to six digits); the corrected cold problem is simply harder for
the SQP than the incorrect one was. Fixing that is solver work, and the `1/x_start` floor
above is the first thing to try.

## 12. `free`, alternatives, and what a design variable means — brainstormed, to crystallize

**Status: a brainstorm, not a design.** Recorded so tomorrow starts from here rather than
from scratch. Nothing in this section is implemented, and the three parts are one
mechanism seen from three angles rather than three projects.

The thing that provoked it: `x109` cost a day. Two solvers agreed to six digits, the
value sat 10.9 % from PROCESS, and the whole apparatus reported that as a discrepancy to
hunt. It was a **flat direction** — a coordinate this problem does not determine — and
nothing in the port could say so, because an iteration variable here is an integer with a
value and no other properties. Meaning is the fix, and `free` is where meaning would come
from.

### 12.1 `free(graph, paths)` — drop the producer, promote to unknown

The operation PROCESS spells `numerics.ixc`. Today the port applies it by *conditionally
registering* nodes (`DefaultAspectRatio` carries a docstring saying "only instantiate this
node when `1 not in ixc`", enforced by nothing), which means the design vector is a
**precondition for assembling the graph**. `free` inverts that: assemble everything once,
then apply the design vector as a rewrite. "Which models does this design vector kill?"
becomes a query instead of something you had to know in advance.

**Two cases, and conflating them is how you lose track.** Measured on the reference run:
all 8 iteration variables are *inputs* PROCESS promotes to unknowns; `.physics.aspect` is
the rare one that has a producer.

- **unowned path** → promote input to unknown. Graph unchanged.
- **owned path** → drop the producer *and* promote. One fewer equation.

**It is not `Residualise`.** `Residualise` keeps the equation as a condition (unknown +
condition, square preserved); `free` discards it (one net degree of freedom for the
objective). `aspect = stella_config_aspect_ref` is a *default* and discarding it is
right; a coupling variable's equation is *physics* and discarding it under-determines the
system. Which is a pleasant collapse: **that distinction is SAND vs MDF.** Free a coupling
variable and let the optimiser close the loop → SAND. Keep the producer and drive the
cycle → MDF. The port hand-builds both today; they would become one parameterised rewrite.

**Preconditions.**

1. **Sole ownership.** The producer must own that path alone, or dropping it orphans its
   siblings — the missing-producer class, **eight instances**. Refuse, or make the caller
   say what happens to each sibling.
2. **Cycle membership.** Inside an SCC, refuse and say `Residualise` was meant.
3. **Bounds.** Part of the *selection*, not of the variable: `boundl`/`boundu` are
   per-ID and this run's `IN.DAT` overrides five of eight.
4. **A seed**, with provenance. This is exactly where `run_sand_harness._seed` went wrong
   (§11.8 item 3): 12 of 23 unknowns started at `0.0`.
5. **`prune` afterwards**, for ancestors that existed only to feed the dropped producer —
   and `wanted` must include **reporting outputs**, or `OUT.DAT` gets pruned away. That
   set is currently declared nowhere.
6. **Existence.** A typo'd path fails; it does not silently free nothing.

**The postcondition is the whole point.** `unowned_inputs` must not grow. Every new entry
is a variable that just became a frozen constant with a structurally-zero derivative,
which is the defect this project has found eight times and never by a check.

### 12.2 Alternatives are keyed on output — nearly

*(Resolution designed: `_audit/model_tree_design.md` — exclusivity becomes
by-construction (one slot, one occupant) and this section's real content, the
partial-overlap hazard and "the check belongs on consumers", becomes that design's §6
boundary postcondition. Kept in full below because the reasoning is what that section
implements.)*

`Switch.check_arms_are_exclusive` already accepts colliding output ownership as its only
proof of exclusivity, so "these nodes cannot coexist" is detected exactly that way today.
**Collision proves exclusivity but does not define it**: two real cases are exclusive by
PROCESS's own `if`/`elif` with *disjoint* outputs — `.vacuum.i_vacuum_pumping` and
`.costs.i_cost_model` (§1). So the mechanism needs collision *plus* a way to declare
exclusivity without it.

**Partial overlap is the hazard.** A owns `{x, y}`, B owns `{x}`: they collide on `x`, so
they are detected as alternatives, and choosing B leaves `y` with no producer — silently a
boundary input. So the check belongs on **consumers, not producers**: after selecting an
arm, does every remaining read still have an owner or a declared input? That is §12.1's
postcondition again, and it is why these are one mechanism.

**Do not** require arms to have equal output sets. `i_cost_model`'s arms genuinely compute
different things, and forcing a common set means inventing fields that exist only to
satisfy a check.

### 12.3 Iteration variables need meaning, and the port already computes it

PROCESS's own users feel this: the reference `IN.DAT` annotates **every** `ixc` by hand in
a comment (`ixc = 109 * f_nd_alpha_thermal_electron: thermal alpha density / electron
density`), which is unchecked, unparsed prose. Measured on this run: all 8 are **interior
to their bounds**, so no bound is active and every gradient is balanced by constraint
multipliers alone.

Ranked by value/cost:

1. **Address by path, never by integer.** The ID is an `IN.DAT`-boundary concern. Nearly
   free, and it removes `x109` from every report.
2. **Sensitivity and multipliers, printed every solve.** Reduced gradient per variable,
   multiplier per binding constraint, distance to each bound. The `jax.jacfwd` Jacobian is
   already built; a near-zero column **is** a flat direction, detected rather than
   discovered a day later. This is the item that would have prevented §11.11 entirely.
3. **Dimensional scaling instead of `1/x_start`.** Start-dependent conditioning means the
   same problem is conditioned differently warm and cold, and it is what produced the
   `-2.6e+26` in §11.6's closed list. A characteristic magnitude per quantity type is
   stable and **derivable from the name** — `standards.md`'s
   `<type>_<system>_<description>_<units>` already encodes the dimension.
4. **Separate physical design variables from f-value slacks.** 28 of 83 iteration
   variables start with `f`; some are genuine physics (`f_nd_alpha_thermal_electron` is an
   alpha-ash fraction), others are PROCESS's f-value idiom — slacks paired with an
   inequality to make it an equality. Those are not design freedom and reporting them
   beside real variables is noise. Needs checking per variable: the prefix does not
   distinguish them.
5. **Graph-derived role** — which conditions a variable can reach, which nodes read it.
   Straight from the graph, and it is what the `IN.DAT` comment gestures at in prose.
6. **Bounds with provenance.** `boundu(10) = 1.2` on `hfact` is a physics judgement
   nobody wrote down.

**These are properties of the selection, not of the variable** — which is what `free`
returns. So a `DesignVariable(path, bounds, scale, dimension, role, seed)` is `free`'s
natural return type, and (2) is what you print about the set it returns: the same object,
before and after the solve.

## 13. Priority order, 2026-08-25 — what to pick up next

Written at the end of a session that closed §11.6's `atol` item, `model_tree_design.md`
§8 steps 3–4d, the three-mirror layout and the chunk-letter renames. The **Verified state**
table at the head of this file is stale where it says 159 nodes / 348 unowned inputs; the
live graph is **156 / 320**, and the split in §13.2 may move file layout but must not move
either number. Re-measure the whole table on a settled tree before trusting it.

### 13.1 The blocker: `cottax` is mid-refactor and the gates are not trustworthy against it

`~/jaxgraph`'s working tree is being changed by a concurrent session. Two symptoms hit
this port today, both on a **clean** `PROCESS` tree (verified by stashing):

- `cottax.interfaces.pytree_namespace_module.Root` was deleted outright — gone from the
  whole package, no replacement — which `functional_process/paths.py` imported. Nothing in
  the port could import at all. Resolved here by reimplementing `_Root` locally: it was
  never really cottax's, since the area list comes from PROCESS's own `DataStructure` and
  every name it accepts or rejects is a PROCESS fact. It still returns cottax's `Area`, so
  nothing downstream can tell.
- A condition-map arity changed: `run_mda_harness` raises `TypeError: condition map of
  ['.physics.profiles.ion_vol_avg_temperature'] takes 1 unknown(s) ..., got 0`, failing 9
  tests and erroring 4. **RESOLVED 2026-08-26**, and it was not an arity change but a
  *contract* change, which is why guessing would have gone wrong: `AbstractDriver.__call__`
  now takes `(conditions, data: Mapping[type[DriverIn], tuple])` instead of
  `(conditions, start)`, and `Drive.role_data` fills that mapping by walking the driver's
  own `requires`. All four of this port's drivers still declared `requires = ()`, so they
  were handed `{}`, `ravel_pytree` flattened it to nothing, and the condition map was
  called with zero unknowns. The port's half is: `requires = (Start,)` on every driver,
  `cottax.rewrites.Initialise` on every problem (`driven_graph`, and
  `sand.optimise_graph` for the `Optimise` -- `_check_roles_agree` refuses to *join* two
  problems that disagree about their driver-data kinds), and every seeding site writing
  the unknown's `^guess.*` port instead of its own name. **No cottax change was needed.**

**Until that settles, every measurement must pin `cottax` at its own `HEAD`**, on both
sides of every comparison:

```bash
SP=<scratch>; (cd ~/jaxgraph && git archive HEAD src/cottax) | tar -x -C $SP
PYTHONPATH=$SP/src $PY -m pytest tests/functional_process -q     # 3770 passed, 3347 skipped
```

Steps 4c and 4d were both measured this way. A run that reports 9 failures is a missing
`PYTHONPATH`, not a regression. **Reconciling with the new `cottax` API is the first
thing to do**, because until then no gate in this file means what it says.

**Done, 2026-08-26, against `cottax` `ef093ba` (`~/jaxgraph` working tree dirty at the
time, so measured through the pin above).** Whole suite green: **3783 passed, 3347
skipped, 0 failed, 0 errors** — of the +26 against the recorded 3757, **13 are the nine
failures and four errors above** and 13 are a concurrent session's new
`visualization/test_grouping.py` cases, not this work. Every other gate re-measured and
**unmoved**: `GRAPH` still 156 nodes with an identical per-node
`(name, type, inputs, outputs)` sha, `driven_graph` still 158 nodes / 136 blocks / 14
driven; MDA harness identical at 485 agreements / 34 disagreements / 3 unverifiable /
**0 ungrounded** / 543 owned walked / 0 unaccounted; SAND C2 and C3 **byte-identical**
(31 it / `objf 1.217757347`, 99 it / `1.217757378`) and MDF C2/C3 likewise (129 / 200).

Two rows of the **Verified state** table are invalidated by this and want re-measuring
with the rest of it (§13 head): the harness row's `499 agreements ... 557 owned ... 61
switch kwargs` is stale — the *unmodified* tree on pre-refactor `cottax` (`05be0c5`) also
measures **485 / 543 / 57**, so the drift predates this work and is not caused by it —
and the `pytest` row's 3730. The `GRAPH` row's 159/348 was already known stale (156/320).

**The boundary grew, by design: 320 → 338.** `Initialise` mints one `^guess.<place>` per
driven unknown and those are ordinary unowned inputs — 18 of them, one per unknown across
the 14 driven problems (`cryo_q_loads_step` owns 4, `proton_rate_density.cycle` 2, the
rest 1 each). §13.7's `check_boundary` pin should therefore be generated against **338**,
or against 320 with minted `^guess.*` excluded; that is a real choice, not bookkeeping.

**Nesting/MDF was NOT adopted, and the reason is a fact about `cottax`, not a choice.**
At `ef093ba` `Drive`'s fields are `(subgraph, driver)`, `ConditionMap.body` is a `Graph`
run by `_run_acyclic`, and `Schedule.steps` builds `Drive(sub, driver)` without reading
`blocking.inner` — so `mdf.py`'s account of the gap is still exactly true and
`MdfConditionMap` stays. The fix *is* written, in `~/jaxgraph`'s **uncommitted** working
tree (`Drive` gains `problem` and `body : Step`; `Schedule.steps` descends), and porting
against another session's unlanded work is what the pin above exists to prevent.

The handoff needs no memo, because the port already has a tripwire:
`test_mdf.py::test_cottax_cannot_run_that_nesting`, whose own docstring says *"the day it
stops being true is the day `mdf.MdfConditionMap` should be deleted in favour of a real
nested `Drive`. A failure here is good news, not a regression."* Run against the live
editable `cottax` today it **already fails** (`DID NOT RAISE ValueError`) while the whole
suite is green against the pin — so the signal is live right now, and the sequence when
that lands is: `nested_blocking()` → `schedule_for` → delete `MdfConditionMap`, with
cottax's `tests/test_evaluate.py::test_a_solve_nested_in_a_solve_runs` as the worked
example.

**One latent defect this surfaced, worth more than the port.** `sand.degenerate_fixed_points`
read a problem's `.reads` to build its probe. After `Initialise` that includes the `Start`
port, so the probe raised `KeyError` on a `^guess.*` with no value — and the function's
bare `except Exception: # undetectable is not degenerate` swallowed it and reported
**nothing** degenerate. The two identity fixed points of §13.6 (`eta_turbine_step`,
`cplife_avail`) then reached `reduce_jacobian` as exactly-zero rows of `J_RY`, i.e. a
singular equality block, and SAND died in `np.linalg.solve`. Fixed by asking
`cottax.problem.conditions_of` for the conditions rather than taking every read. The
lesson is the bare `except`: it converted a structural change into a silent wrong answer,
and the only reason it was caught is that the *next* stage happened to be a matrix
inversion that could not fail quietly. §13.6's two identity fixed points are now
load-bearing evidence, not a curiosity.

### 13.2 In flight — `total_process.py`'s three-way split

2178 lines: 1114 in fifteen `ModelNamespace` classes, 1064 in imports (188 names over 246
lines), registries, `UNPORTED` and the factory. Only `StellaratorProcess` (83 lines, 7
slots) is about the whole machine; `Costs` is 211 lines naming 40 nodes that all live in
`models/costs/costs.py`, and 20 stellarator modules are imported solely to be named in
stellarator slots.

- **Subsystem namespaces → their packages.** A subsystem's namespace belongs with its
  models.
- **The switch layer → `functional_process/indat.py`.** The factory is not part of the
  tree; it is an adapter to PROCESS's legacy input format. If the tree carries no switches
  and `machine_from_indat` is the only place an `i_*` integer is read, then everything
  switch-shaped — the ten registries, `UNPORTED`, the arm functions, `REFERENCE_MACHINE` —
  is PROCESS's input *encoding*, not the machine. Registries go here and **not** with the
  subsystems, which would re-scatter what steps 4b–4d consolidated.
- **`total_process.py` keeps** `StellaratorProcess` and `graph_for`, ~150 lines.

Node names are slot paths, so nothing here may move one: 156 nodes and the per-node
`(name, type, sorted inputs, sorted outputs)` sha are the gate.

### 13.3 The namespaces are naming scopes, not encapsulation boundaries

Each depth-1 group's interface, measured on the 156-node graph (`internal` = reads whose
producer is in the same group, `imported` = from another group, `boundary` = unowned,
`exported` = this group's outputs read by another):

| group | nodes | internal | imported | boundary | exported |
|---|---|---|---|---|---|
| stellarator | 54 | 192 | 32 | 173 | 122 |
| costs | 40 | 49 | 69 | 164 | **0** |
| physics | 33 | 94 | 25 | 123 | 40 |
| power | 20 | 39 | 47 | 40 | 25 |
| availability | 4 | 3 | 22 | 23 | 13 |
| vacuum | 3 | 2 | 9 | 20 | 5 |
| buildings | 2 | 4 | 14 | 30 | 13 |

**Grouping related physics together is the goal; encapsulation is not.** `stellarator`
and `physics` have 6x and 3.8x more internal than imported edges; `costs` has 0.7x and
exports nothing at all — a pure sink, which is why the dependency ordering puts it last.
That is not a defect and not a scorecard for whether a namespace deserves to exist: a cost
model is downstream of everything by nature. It is recorded because it says what a
namespace here *is*, and heads off reading one as a component and then being surprised
that all its inputs are foreign.

### 13.4 Grouping depth — [DONE 2026-08-26] the grain is the tree's, and the headline was wrong twice

**Closed, and not as this section proposed.** `depth=None` is now the default and means
*the namespace the node lives in*; `depth` survives only as a zoom for a picture, and it
saturates (3 and 99 are one grouping). More importantly the *measure* changed: `crosses`
was "spans more than one group", which at the tree's grain reports a subsystem's internals
as coupling between subsystems. It now asks containment — `BlockGrouping.container` is the
longest common prefix, and a block crosses only when nothing contains it. `top_of` keys
colour on the subsystem so a subtree reads as one thing, the ribbon nests one lane per
level instead of truncating, and `cross_subsystem_edges` is reported beside
`cross_group_edges` (at `depth=1` they are equal and both are 144, which is the recorded
figure — the finer number does not replace the coarser one).

**The fact, which needs no knob:** of 14 multi-node SCCs in the declared graph, 13 are
inside a single namespace and **one** spans `physics`, `physics.profiles` and
`physics.profiles.parameterisation`. **Nothing crosses a subsystem boundary anywhere.**
Both previous headlines were wrong: "0 crossing blocks" at depth 1 hid the loop entirely,
and "1 crossing block" at depth 2 called a subtree's internals a subsystem crossing.
`render_xdsm.grouped` prints both kinds, since printing only crossings would have made the
one interesting loop invisible at the moment the measure got sharper.

*Superseded below; kept because the measurements are still the record of why.*

`depth` is plumbed through `grouped()`, `group_of`, `grouping_report` and
`dependency_group_sequence` and has never been called with anything but `1`.

| depth | groups | blocks crossing a group boundary |
|---|---|---|
| 1 | 7 | **0** |
| 2 | 11 | **1** |
| 3 | 12 | 1 |

At depth 2 the crossing block is the density/composition/fusion-rate loop straddling
`physics` and `physics.profiles`:

```
.physics.fusion_power_totals_mw, .physics.fusion_totals_no_beam,
.physics.profiles.parameterisation.parabolic_on_axis_densities,
.physics.profiles.density_profile, .physics.fusion_rates,
.physics.plasma_composition, ^problem.physics.proton_rate_density.cycle
```

Depth 1 hides it by swallowing the whole thing inside `physics`. Depth 2 should be the
report's default for exactly that reason. Two cautions: the group axis is a weaker claim
at depth 2 (more groups fall inside one condensed component, so the tie-break carries more
weight), and `group_of`'s `among` guard and `_cut_owner` fallback were written and tested
at depth 1 only — the minted node above is *in* the crossing block, so it is precisely the
case that exercises the fallback at depth > 1. Pin it.

Then the picture per subsystem, which `structure_order`'s own docstring already
anticipates ("a block's interior is its own blocking and draws its own picture") and
nothing does. `stellarator` at 54 nodes and `costs` at 40 are past the size where one
156-node matrix is readable. This only becomes well-posed *after* §13.2: a per-subsystem
picture is a real object only if the subsystem is one.

### 13.5 The switch survey's remaining bands

`switch_kwarg_survey.md` found 32 slots hardcoding a PROCESS switch as a static kwarg, 59
`(slot, switch)` pairs over 26 switches. Band (a)'s five live incoherences are closed
(steps 4c, 4d). What remains:

- **(b) 23 slots whose branches differ in declared reads — 79 invented edges, 23% of the
  345 declared reads on switch-carrying slots.** This is the correctness band: a node
  branching internally declares the union of both arms' reads and so claims dependencies
  the run does not have, which is exactly what `machine_from_indat`'s own docstring says
  was rejected for switched slots. Sub-banded (b1) 10 slots where conversion changes block
  structure, (b2) 8 with large fan-in, (b3) 4 small.
- **(c) 9 slots, reads identical.** Tidying; zero invented edges.
- **(d) 3 high-arity families** — `i_confinement_time` (49 reachable values),
  `i_tf_sc_mat` (9), and the `i_thermal_electric_conversion` x `i_blanket_type` x
  `secondary_cycle_liq` triple shared by 4 slots. One occupant per value is the wrong
  answer here; the rule is an occupant per value *this port supports*, everything else in
  `UNPORTED`.

Also still unwritten, and named in the survey's §7: **no declared read may be dead at the
value the slot holds.** Note it would *not* have caught band (a)'s four — at
`i_tf_sup = 0` every stale kwarg still made its node's reads live, just live for the wrong
arm — so it complements `test_no_slot_contradicts_a_factory_switch` rather than
subsuming it.

### 13.6 Two structural findings from the survey, neither acted on

- **A switch is inventing a cycle.** `.tfcoil.j_tf_wp` is dead at `i_tf_sc_mat = 1` and is
  machine-checked to be the *sole* back-edge closing the 4-node coils SCC. Remove it and
  the driven block collapses to 2. This is the port's central thesis showing up as a
  measurement, and it is the most valuable single item in §13.5's band (b).
- **[One closed 2026-08-26.]** `cryo_q_nuc`'s fixed point is deleted, not fixed: splitting
  `inuclear` into occupants showed the fixed point was an artefact of the switch, since one
  arm never reads the incumbent and the other *is* "qnuc is input", i.e. an empty slot.
  Driven blocks 14 → 13. `eta_turbine_step` — the one that reaches `.costs.coe` — is
  untouched and still driven; it is the remaining instance below.

- **Two `FixedPoint`s are the identity map** on the reference machine: `cplife_avail`
  (`itart = 0`, six of seven declared reads dead) and `eta_turbine_step`. The second owns
  `.heat_transport.eta_turbine`, which reaches `.costs.coe`, this run's objective. A fixed
  point that determines nothing is being driven anyway.

### 13.7 `check_boundary` — [DONE 2026-08-26] built, categorised, pinned at 338

**Built** as `functional_process/boundary.py`, with a pin at
`functional_process/reference_boundary.txt` (generated by
`$PY -m functional_process.boundary --write`, never hand-edited) and
`tests/functional_process/test_boundary.py` asserting equality, so a boundary that *grew*
fails naming the orphans and their readers, and one that *shrank* fails asking for the pin
to be regenerated.

**The pin carries two categories, and that is load-bearing.** `input` is read from the
`DataStructure` — growth there is the lost-producer defect. `guess` is a `^guess.<place>`
`Start` port, one per driven unknown, minted mechanically by `Initialise` — growth there is
a new problem, i.e. structure. Without the split the two move the same total in opposite
directions and cancel, which would make the number useless for exactly the programme it is
meant to serve: the end state is a purely functional graph whose boundary holds the real
inputs and nothing else. **Today: 320 input + 18 guess = 338**, the declared graph being
the 320.

*Original text follows.*

`model_tree_design.md` §8 step 5's `check_boundary` **did not exist**; step 4c found
this when it expected a pin to trip and there was none. The boundary is now 320, not 348 —
step 4c's cost-slot removal *shrank* it (3 added, 31 removed; category (d) is empty in the
cost model). Generating the pin against 320 rather than 348 is strictly better, and it is
the precondition `free`'s design (§12.1) also wants.

### 13.8 The input format — `indat_to_python`, and the 109 numbers

`stellarator_helias.IN.DAT`'s 168 assignments are 22 `icc`/`ixc` ID lines, 15
`boundl`/`boundu`, 22 integer switches, and **109 float-valued inputs**. The tree covers
the switches. The 109 still arrive because `SingleRun.__init__` runs `init_process` and
populates the whole `DataStructure` (`sand_harness.py:117` states this), from which the
graph reads anything no node owns — the 320 of §13.7.

So "author the machine in Python" is half an input format today: correct models, no
numbers. The cheap first move is an `indat_to_python(file) -> str` emitter next to
`machine_from_indat` in the new `indat.py` — same parse, prints the literal instead of
building it, testable by emit/exec/compare against `REFERENCE_MACHINE`. It forces the
question of what the Python surface must cover, which surfaces the numeric gap
concretely. The two halves of PROCESS's file format then live in one module, in both
directions.

Note the split the format should keep, which `IN.DAT` conflates: the **machine** (a tree
of choices) and the **study** (objective, unknowns, conditions — `icc`/`ixc`, where the
integer-ID indirection is worst and where structural names already exist).

### 13.9 The tokamak — sized by hand here, then measured: `_audit/tokamak_scope.md`

**Measured 2026-08-26, and this section's estimate held.** `machine_survey.py` classifies
an input file's integers against the tree without building anything;
`_audit/tokamak_scope.md` is the result and supersedes the table below. On
`large_tokamak_eval`: 33 integers = 6 not topology, 3 the factory already dispatches on, 7
pinned as static kwargs (**4 of which the file contradicts**), **17 new** — against the
"~16" estimated here by hand. Two corrections to what follows. The device is the
*top-level class* (`StellaratorProcess`), not a slot inside one, so a tokamak is a sibling
class and no existing node name moves. And `i_confinement_time = 34` is **already ported**
(`ITER_IPB98Y2`, `confinement_time.py:809`) — the tokamak arm is refused for a
*combination* reason that a tokamak device class dissolves, not for missing physics. The
four contradicted pins are the first deliverable; `_audit/tokamak_scope.md` §"The order
this implies" carries the sequence, including the per-switch CoolProp flag §5's policy
still wants.

*Original sizing follows.*

Structurally the tree already supports it: a `device: Stellarator | Tokamak` slot is the
mechanism steps 4b–4d bought, and `istell` already picks. The blocker is models.
`switches_from_indat` over the tokamak regression inputs, against the ten the factory
reads and the switches the tree hardcodes:

| input | switch-shaped ints | factory reads | tree hardcodes | unknown |
|---|---|---|---|---|
| `large_tokamak_eval` | 27 | 3 | 5 | 19 |
| `low_aspect_ratio_DEMO` | 33 | 3 | 5 | 25 |
| `spherical_tokamak_eval` | 54 | 5 | 4 | 45 |

Net of noise (`icc`/`ixc` are array-parse artefacts; `i_process_run_mode`,
`i_figure_merit`, `lsa`, `ifueltyp` are run control and cost inputs, not topology):
**~16 genuinely new topology decisions for a conventional large tokamak, ~40 for a
spherical one.** Do the conventional case first.

Two consequences for ordering. `itart` is hardcoded `CONVENTIONAL_ASPECT_RATIO` and
`spherical_tokamak_eval` sets it, so §13.5's band (b) is on the tokamak critical path, not
parallel tidying. And the five switches `large_tokamak_eval` shares with the Helias run
are exactly the ones already hardcoded, so converting those *is* the first tokamak
deliverable rather than a prerequisite to it.

The bounded next step is to adopt `large_tokamak_eval.IN.DAT` as a **second reference run**
and let the boundary check report what is missing, instead of estimating from
`unit_registry.md`. `cost_boundary_inputs.md`'s category (d) rows already carry the
producer `file:line` for every PF/CS/structure variable a tokamak must restore.

### 13.10 Smaller open items

- **§6.2's cheap check, still unwritten**: every constraint inside `n_equality_constraints`
  returns `eq`. `c16` is written `geq` and sits inside `n_equality_constraints = 2`, which
  `process/core/init.py:1285-1293` takes from the input file and never checks.
- **The converged `DataStructure`, not the IN.DAT, is the right oracle** for switch
  checks. All 26 hardcoded switches agree with PROCESS's converged state; six differ from
  its bare default, and one of those — `iohcl = 0` — is set by neither the file nor the
  factory (it comes from `st_init`), so **no input-file-based test can ever see it**.
- **Six chunk-lettered `.md` records remain** under `_audit/units/models/stellarator/`
  (`stellarator_A_orchestration`, `stellarator_E{1,2,3}_*`, `stellarator_E_fwbs_synthesis`,
  `stellarator_G_output`). They have no module behind them, so there was nothing to rename
  them alongside; they resolve when those units are ported.
- **The four `stellarator_fwbs_s*.py`** keep their chunk letters until §3's S1–S6
  re-chunking lands, because those names will move again on their own.
- **`.costs.coe`'s `rel_diff = 1.733e-02`** is the report-pass
  `z_tf_inside_half`/`a_plant_floor_effective` inconsistency, a `+17.604 MW` offset
  propagated linearly and confirmed on `.heat_transport.tlvpmw`. Two records still say
  otherwise and want correcting: `models/costs/costs.md`'s open question 7 blames
  `VacuumOld` (which contributes `1.195e-04`, ~140x below the real cause), and
  `unit_registry.md`:113 still records `1.704e-06`.

### 13.11 A lesson from this session, in the shape of §11.7's

**A grep that finds nothing is not proof.** The chunk-letter rename had to update
references that a line-based grep could not see: three were line-wrapped *mid-stem*
(`` `power_B_thermal_\ncryo` ``), six cited abbreviated letters (`physics_A/B/C.md`,
`stellarator_D`) rather than full stems, and two passages named the old filenames *as
examples of chunk letters*, so a mechanical substitution would have silently destroyed
their point. Flatten whitespace before searching, search for the abbreviation as well as
the stem, and read the surrounding sentence before substituting.

The same session's companion: **a number carried in a document is not a measurement.**
`CLAUDE.md` recorded `pytest functional_process` at ~1390 passed + ~640 skipped against an
actual 3752 + 3347, and `~/jaxgraph` at 307 against 740 — both stale by more than double,
in a file whose own instruction is to suspect the env when a number moves.


## 14. State, 2026-08-26 — the switch conversion, the driver refactor, and the tokamak

**Read §13 first for the priority order; this section is what moved and what is now open.**
Every number here was measured against `~/jaxgraph` pinned at **`b7c5572`** ("driver is now
declarable and first-class graph citizen") via `git archive HEAD src/cottax`, not against a
number written anywhere.

### 14.1 Verified state

| check | value |
|---|---|
| `pytest tests/functional_process` | **3810 passed**, 3349 skipped, 0 failed |
| `GRAPH` | **159 nodes**, **316** declared boundary |
| `driven_graph(GRAPH)` | 140 blocks, **13 driven** (was 14) |
| boundary, categorised | 316 input + 17 guess = **333** (`boundary.py`, pinned) |
| MDA harness | 484 agreements / 34 disagreements / 3 unverifiable / 0 ungrounded / 24 errors; **545 owned walked, 0 unaccounted** |
| static switch kwargs | **20 switches over 28 slots, 49 pairs** (was 25 / 31 / 56) |
| tokamak machine | **100 nodes, 314 boundary**; 239 shared with the stellarator, **75 tokamak-only** |

The MDA harness's `errors` count includes the three port-owned intermediates that PROCESS
has no field for (`.physics.t_electron_confinement`, `nd_plasma_electron_line_19`,
`cur_plasma_ma`) — the stated, accepted cost of decomposing finer than PROCESS does.

### 14.2 The policy, restated by the user and now binding

**No switch is a static kwarg, whatever its reads.** The earlier position — that
`switch_kwarg_survey.md` band (c) (branches with identical reads) was the legitimate case
for keeping one — is **withdrawn**. A switch value selects an occupant, full stop.

`test_occupants_of_one_slot_differ` enforced the old policy by asserting occupants differ
**by ports**, which makes an identical-reads split impossible by construction. It now
asserts each value selects a **distinct occupant class**. That is weaker, and the gap is
named in its docstring: nothing now catches a family whose occupants are identical in
behaviour too, and nothing cheaply can. `i_tf_sup == 2` (PROCESS runs the byte-identical
branch to `== 0`) is still handled by refusing it rather than registering it twice, and
that refusal is now the only thing standing there.

### 14.3 Closed: the confinement family

`.physics.confinement_time` was one node carrying three switches and declaring **32
reads**; it is six slots and the scaling occupant declares **6**. Two of the 32 were dead
at this machine's own values — `.current_drive.p_hcd_injected_total_mw` (not read when
ignited) and `.physics.pden_plasma_rad_mw` (not read under core-only radiation) — the
first inventing a `.current_drive -> .physics` subsystem edge no run makes.

`head` / `law` / `tail` were extracted as pure functions first and proved **bit-exact**
against the composite on all nine outputs, off the `1e-3` clamp, before anything touched
the graph; the unit stays green under `--fp-gradients` (561 passed), which is what proves
an extraction preserved derivatives and not just values.

**`StellaratorConfinementTime` and the `ConfinementTime` composite node are deleted** (217
lines), along with `_rebound_signature`, which existed only to build the subclass's rebound
signature. The subclass existed to rebind one read — PROCESS hands ISS04 the rotational
transform through a parameter its own source calls `q95` — and with one class per law that
is not a rebinding: `iss04_stellarator_confinement_time`'s own parameter *is* `iotabar`. The
read follows from the law, so `CONFINEMENT_TIME` keyed on `istell` had nothing left to
decide. Its test is re-targeted onto the live occupants, not deleted.

**`calculate_confinement_time` stays and is not dead**: it is the composite PROCESS itself
has, and `TestConfinementTime` diffs it against `PlasmaConfinementTime.calculate_confinement_time`
sample by sample. That is the boundary the port can compare at. Any split finer than
PROCESS's own has no 1:1 reference, which is the trade this decomposition makes.

### 14.4 Closed: `inuclear`, `i_pulsed_plant`

- **`inuclear` removed a driven block.** Its arms read disjoint variables and the
  "otherwise" arm is PROCESS's own *"if inuclear = 1: qnuc is input"* — so the computed arm
  is an ordinary node reading `p_tf_nuclear_heat_mw` and the input arm is an **empty slot**
  (`CryoQNuc | None`). The `FixedPointFunction` that existed only because one body both
  read and owned `.fwbs.qnuc` is gone, with its minted `^cond` copy and its `^guess` port.
  What `sand.degenerate_fixed_points` used to recover at runtime by differentiating a
  residual, the tree now states.
- **`i_pulsed_plant` was two dead reads.** At `== 0` Account 225.3 is identically zero, so
  the node declared a `.heat_transport -> .costs` edge and a `.costs.fkind` edge no run
  makes. The unpulsed occupant reads **nothing**. `istore` remains a kwarg on the pulsed
  occupant **pending §14.2** — its two ported values differ only in a literal, and under the
  binding policy it too must become occupants.

Of `_audit/tokamak_scope.md`'s four contradicted switches, **three are closed**; only
`i_p_coolant_pumping` (5 slots, two of them 28- and 31-read nodes) remains.

### 14.5 `i_tf_sc_mat` — done, reverted, and now unblocked

The conversion works and does what §13.6 predicted: only Bi-2212 reads `.tfcoil.j_tf_wp`,
so the ITER Nb3Sn occupant does not declare it and **the four-node coils SCC collapses** to
the inherent `Intersect`/`^problem` pair. It was **reverted**, because it broke four MDF
tests for a reason worth more than the conversion:

> **The root-find's starting guess was living on the invented edge.** `ROOT_FIND_SEEDS`
> derives `wp_width_r_min`'s guess from `.stellarator.r_coil_minor` read out of *the
> block's context* — and `r_coil_minor` was only in that context because `j_tf_wp` dragged
> `WindingPackIntersectInputs` into the block. Remove the fake cycle and the seed loses its
> source.

At the time this was blocked: cottax's `Output` refuses a raw `VarPath`, so a node could not
own the minted `^guess.*` name. **`b7c5572` dissolves it** — `rewrites.Supply` exists for
exactly this, its docstring saying *"`Assign` opens `^guess.u` as a boundary input, and this
is how a low-fi model's output becomes the start instead."* `winding_pack_pre_intersect`
already computes the guess and discards it. Redo: own it, `Supply` it onto the `Start`,
delete the `ROOT_FIND_SEEDS` entry (whose own docstring warned `switch_audit` could not
check its material branch). **This is the highest-value item outstanding.**

### 14.6 The driver refactor — `Initialise` is gone

`cottax.rewrites.Initialise` no longer exists; `Assign(problem, driver)` retypes a problem
into a `Driven` and *mints* the ports the algorithm's own `requires` names. The port
follows, and cottax's refusals drove the shape:

- **`cut_graph` split from `driven_graph`.** Cutting a cycle is structure; assigning a
  driver is an algorithm choice. `Combine` refuses to join two problems carrying a driver
  (*"combining two problems discards the algorithm answering each"*), and SAND joins every
  `FixedPoint` into one `Optimise` — so it builds on the cut graph and assigns after. The
  seam was always there; the refusal made it visible.
- `assign_drivers` / `reassign_drivers` — attach vs. swap, kept apart because `Assign`
  deliberately will not overwrite.
- `default_drivers` takes a **`Graph`**, not a `Blocking`: the choice happens before there
  is a blocking, and never needed one.
- **MDF builds two graphs, not two driver maps.**
- `starts_for` returns `()` for an undriven problem — legitimate now, where under
  `Initialise` a missing port could only be a bug.
- `Driven` forwards only the graph-facing surface, so `sand_shape` says
  `node.problem.equalities`: *a driven node has a problem, it is not one.*

Two bugs the refactor surfaced: `mdf.guess_ports` was asking the **undriven** graph for
start ports, so every port fell to `0.0` and each inner solve began at exactly the cold
point `prime()` exists to escape; and `MdfConditionMap` needed the new **`roles`** field.

**`roles` closes `optimise_design.md` §8 upstream** — the condition map now carries what
each condition *is*, parallel to `conditions`. `VmconDriver.n_equality`/`n_inequality` can
therefore be retired in favour of reading roles off the map. **Open, not done.**

### 14.7 New instruments

- **`functional_process/boundary.py`** + `reference_boundary.txt` (generated;
  `$PY -m functional_process.boundary --write`) + `tests/functional_process/test_boundary.py`.
  `check_boundary` refuses a read with no producer, naming each orphan **and its readers**.
  The pin carries **two categories** — `input` (read from the `DataStructure`; growth is the
  lost-producer defect) and `guess` (a `Start` port; moves only when a driver does) —
  because without the split, landing a producer and declaring a problem move one total in
  opposite directions and cancel.
- **The swap contract** (`test_machine.py`, pinned in `reference_swaps.txt`):
  `boundary.orphaned_by(base, swapped)` asks the question of **consumers, not producers**,
  which is §12.2's rule implemented at last. First run found **six slots, thirty-seven
  reads** — every multi-arm slot in the tree has a partial overlap. Pinned rather than
  asserted away, because a read served by a `DataStructure` default and a read served by
  nothing look identical from inside the graph. They stop looking identical when the
  boundary is declared rather than implied, which is where this port is going.
- **`functional_process/machine_survey.py`** + `tests`: classifies any `IN.DAT`'s integers
  against the tree — factory / pinned / new / not-topology — every column measured, with a
  deliberately weak CoolProp *neighbourhood* flag.

### 14.8 The tokamak — it assembles

`_audit/tokamak_call_surface.md` (traced with `sys.setprofile`, intersected against the
Helias run) and `_audit/units/models/physics/plasma_geometry.md` (unit #24) and
`_audit/tokamak_boundary.md`. `TokamakProcess` is a **sibling class** of
`StellaratorProcess`, `Tokamak` has 25 slots all `| None`, and `istell == 0` builds one.

**The 75 tokamak-only boundary reads: 58 are an empty slot's work list** (11 of 25 slots),
4 a shared subsystem's gap, 12 permanent boundary, 1 already declared empty. Biggest:
`ccfe_hcpb` 16, `cicc_superconducting_tf_coil` 10, `physics` 8, `build` 6, `plasma_geom` 5.
The stellarator machine is unmoved — per-node sha `6d61c802…` identical before and after.

Three findings: **CoolProp is one reached module** (`tfcoil/quench.py`), not six —
`engineering/pumping.py` does not import it and three others' CoolProp arms are dormant
behind this run's switch values (2,077 lines that go live on another input); **three files
are reached with no `.run()` in `caller.py`** (inheritance and plain imports), the tokamak
analogue of the missed `coils/` subpackage; and `ecrh_density_limit` sat unconditionally in
the shared parabolic arm, so a parabolic *tokamak* would have got a stellarator-only node —
the `EcrhDensityLimit` bug class a third time, first time caused by a **device** rather than
a switch value.

`large_tokamak_eval.IN.DAT` **does not assemble as written**: it leaves `i_plasma_ignited`
at PROCESS's default `0` and only the IGNITED arm is written. The refusal is correct and was
left in place.

### 14.9 What is open, in order

1. ~~**`i_tf_sc_mat` via `Supply`** (§14.5)~~ — **done**, and the claim it was sold on
   was wrong. It does *not* delete a driven block: the four-node coils SCC collapses to
   **two** nodes, `Intersect` and its own `^problem`, which is the **inherent** pair every
   declared problem has. The driven-block count is unchanged at **13** before and after.
   What the conversion actually bought is what §14.5's body says — a `j_tf_wp` edge that
   no run makes stops being declared — and that is worth having on its own; the boundary
   moved 316 → 311 inputs and 17 → 16 guesses with it. Corrected here rather than in
   §14.5, whose body never made the claim.
2. ~~**`i_p_coolant_pumping`** — the last contradicted switch, 5 slots.~~ **done**, and
   it was cashed in rather than tidied away: assembling the first tokamak with the real
   value (`3`, against the `1` hardcoded at all five sites) made `power`'s
   `p_fw_blkt_coolant_pump_mw_step` and `.tokamak.ccfe_hcpb.pumping_power` both claim
   `.primary_pumping.p_fw_blkt_coolant_pump_mw`, and cottax refused the graph by name.
   **A pinned switch that disagrees with a real input file is a latent dual-ownership
   bug**, and this is the first time one of the four was demonstrated to be one.
   `machine_survey`'s contradiction list for `large_tokamak_eval.IN.DAT` is now empty.
3. **The remaining 20 switch kwargs**, per §14.2's binding policy, `istore` included.
4. **Retire `VmconDriver.n_equality`/`n_inequality`** in favour of `ConditionMap.roles`.
5. **Port tokamak models** against `tokamak_boundary.md`'s 58, `pfcoil.py` +
   `tfcoil/superconducting.py` first (6,264 lines, 40% of the unported surface).
6. `indat_to_python` and the 109 numeric inputs (§13.8) — untouched.

### 14.10 Two process lessons

**A pin regenerated without reading its diff is not a pin.** Rewriting a namespace class
silently dropped `DoubleAndTripleProduct`, orphaning `.physics.ntau`/`.nTtau` — and
`check_boundary`, built the same morning, would have caught it except that the pin was
regenerated over the evidence. What caught it was the MDA harness moving 485 → 483 and a
refusal to accept a two-variable discrepancy, then diffing owned sets against a `git
worktree` at `HEAD`. Every regeneration since prints its diff first.

**The working tree is not the commit.** `Blocking.nest` and a runnable nested `Schedule`
were read out of `~/jaxgraph`'s *working tree* and reported as landed; at the commit,
`Drive` had no `body` and `Schedule.steps` never read `blocking.inner`. §13.1's pin
discipline exists for exactly this, and skipping it produced a confident wrong answer.

### 14.11 §14.2's policy, carried across the tree — twenty slots, and what it cost the graph

**Re-measured first, not read off §14.2's table.** `mda_harness.switch_audit`'s walk over
the *assembled* graph (`eqx.field(static=True)` per declaration instance) on both machines
found **48 switch-kind kwargs over 19 switches on 30 slots** (stellarator) and **42 over 17**
(tokamak) — not §14.1's "20 switches over 28 slots, 49 pairs", because the tokamak wave had
landed since. **Six remain on each machine, over three switches and two slots.**

#### Converted

| slot | switch(es) | occupants | reads dropped on the reference machine |
|---|---|---|---|
| `costs.{first_wall,blanket,shield,power_injection,auxiliary_component_cooling,fuel_processing}_cost` | `ife` | the six classes, unchanged; `("ife", INERTIAL_CONFINEMENT)` in `UNPORTED` | 0 — the refusal moved from seven bodies to one assembly-time answer |
| `costs.cost_of_electricity` | `ife`, `ipnet`, `ireactor`, `itart` | `CostOfElectricity{ConventionalAspectRatio,SphericalTokamak}` | `.costs.cplife_cal`, `.cpstcst`, `.cplife` |
| `costs.tf_magnet_cost_superconducting` | `supercond_cost_model` | `...{PerKg,PerKam}` | `.costs.sc_mat_cost_0`, `.tfcoil.j_crit_str_0`, `.j_crit_str_tf` |
| `costs.energy_storage_cost` | `istore` | `EnergyStorageCostPulsedElectrowattOption{1,2}` | 0 — literal-only, split under §14.2 as decided |
| `physics.fast_alpha_beta` | `i_beta_fast_alpha` | `FastAlphaBeta{IterPhysicsRules,Ward}` | 0 — identical reads, split on §4's extensibility argument |
| `physics.plasma_composition` | `i_plasma_ignited` | `PlasmaComposition{Ignited,NonIgnited}` | `.physics.f_nd_beam_electron` |
| `physics.profiles.parameterisation.ecrh_density_limit` | `i_plasma_pedestal` | none — the kwarg is **deleted**, the container is the answer | 0 |
| `stellarator.{neutron_wall_load,radiated_wall_load_and_fraction}` | `i_pflux_fw_neutron`, `ipowerflow` | three arms each, one `_wall_load_arm` dispatch for both slots | `.fwbs.fhole`, `.first_wall.a_fw_total`, `.f_a_fw_outboard_hcd`, `.f_ster_div_single` (x2 slots) |
| `stellarator.heating_and_radiation_power` | `i_plasma_ignited` | `HeatingAndRadiationPower{Ignited,NonIgnited}` | `.current_drive.p_hcd_injected_total_mw` |
| `stellarator.coils.coils_mass` | `i_tf_sc_mat` (**a module constant**) | eight, one per material, keyed by `WINDING_PACK_MATERIAL`'s own key | 0 — but see below |
| `power.acpow` | `i_pf_energy_storage_source` | `Acpow{Line,MotorGeneratorFlywheel}`; value `3` in `UNPORTED` | `.heat_transport.fmgdmw` |
| `power.eta_turbine` | `i_thermal_electric_conversion`, `i_blanket_type` | four + **an empty arm** | the whole node: `.heat_transport.eta_turbine` is an input here |
| `power.etath_liq` | `secondary_cycle_liq` | one + an empty arm | — |
| `power.temp_turbine_coolant_in` | the three above | two + an empty arm | `.fwbs.temp_blkt_coolant_out` |
| `power.p_fw_div_heat_deposited_mw` | `i_p_coolant_pumping` | one + an empty arm | — |
| `power.p_fw_blkt_coolant_pump_mw` | `i_p_coolant_pumping` | one (the slot was already `\| None`) | — |
| `power.cryo_q_loads` | `i_tf_sup`, `i_pf_conductor` | two + an empty arm | — |
| `power.cryo_loads` | `i_tf_sup`, `i_pf_conductor` | two; aluminium refused at `power.tf_power` | `.tfcoil.p_cp_resistive`, `.p_tf_leg_resistive`, `.p_tf_joints_resistive`, `.fwbs.pnuc_cp_tf` |
| `availability.electric_production` | `itart`, `i_tf_sup`, `i_blkt_dual_coolant`, `i_p_coolant_pumping` | five arms (`ireactor` folded in) | `.tfcoil.p_cp_coolant_pump_elec`, `.heat_transport.etath_liq`, `.power.p_blkt_liquid_breeder_heat_deposited_mw` |
| `availability.avail` | `ibkt_life`, `itart` | `Avail{NeutronFluence,DisplacementsPerAtom}`; **`itart` deleted, not split** | `.costs.life_dpa`, `.physics.p_fusion_total_mw`, `.costs.cplife` |
| `availability.cplife_avail` | `i_tf_sup`, `itart` | `CplifeAvail{Superconducting,Resistive}` + **an empty arm** | the whole node |

#### The structural result: ten `FixedPoint`s were the switch, not the model

`switch_kwarg_survey.md` §4.7 predicted two of these and measured seven more as
"self-read dead on the live arm". Splitting the switch settles all ten the same way, and
the two shapes are worth separating:

* **Five became empty slots.** `calculate_plant_thermal_efficiency`'s `USER_INPUT` arm is
  `return eta_turbine`; `calculate_cplife_next`'s `itart != 1` arm is `return cplife`;
  `calculate_p_fw_div_heat_deposited_mw`'s and `calculate_p_fw_blkt_coolant_pump_mw`'s
  pass-throughs, and `CryoQLoads` outside PROCESS's guard, are the same sentence. A body
  whose whole content is `return x` is not a fixed point; it is **"x is an input"**, and
  the tree spells that as absence — `inuclear`'s shape (§14.4), now applied seven times.
* **Five became ordinary `ExplicitFunction`s**, because on the computing arm nothing reads
  what it owns.

Measured: **`GRAPH` 159 → 150 nodes**, and the boundary **327 → 303 = 297 input + 6 guess**
(from 311 + 16). The **ten guesses** are the ten unknowns that stopped being driven. The
input half is **+2 −16**: the two additions are `.costs.cplife` and
`.heat_transport.eta_turbine`, the two fields §4.7 named as *driven while determining
nothing* — one of them the quantity `.costs.coe`, this run's objective, depends on. They
are inputs now because PROCESS takes them as inputs on this configuration, said outright.

#### `CoilsMass` — a switch no instrument could see

`models/stellarator/coils/mass.py` answered `i_tf_sc_mat` with a **module constant**
(`I_TF_SC_MAT_ITER_NB3SN = 1`) baked into a `FromExactly(tfcoil.dcond[0])` default.
`switch_audit` walks `eqx.field(static=True)` and nothing else, so it was invisible to it,
to `test_no_slot_contradicts_a_factory_switch`, and to `machine_survey`. The node next door
has been an eight-occupant family since §14.5, so an `i_tf_sc_mat = 5` machine assembled
`WstNb3snWindingPackIntersectInputs` beside a coil-mass node still reading `dcond[0]` — band
(a)'s incoherence, one layer below where anything was looking. It is eight occupants now,
keyed on `WINDING_PACK_MATERIAL`'s own key.

**What would catch the next one.** `switch_audit`'s premise is that a switch reaches a node
as a static field; a constant folded into a `FromExactly` default reaches it as a *port*
instead. Two cheap checks, neither written here:

1. **A cross-slot read check.** For every switch the factory resolves into an occupant,
   assemble at each value and assert that no node's **declared reads** stay fixed while the
   occupant changes — `CoilsMass` read `dcond[0]` at all eight values of a switch that
   changed its sibling's class. This is `test_no_slot_contradicts_a_factory_switch`'s
   question asked of ports rather than of fields, and it needs no new machinery:
   `orphaned_by` already walks reads across a swap.
2. **A source-level constant check.** Any module-level `int` whose name matches a
   `DataStructure` switch field (case-insensitively, `I_TF_SC_MAT_ITER_NB3SN` ->
   `i_tf_sc_mat`) is a switch answered outside `indat.py`. Crude, and it would have caught
   this one; it is the only such constant in the tree today, so the check would start green.

#### Not converted, and why

`power.component_thermal_powers` and `power.delta_eta_step` still carry
`i_p_coolant_pumping`, `i_blkt_dual_coolant` and `i_thermal_electric_conversion`. Both were
**cut down** rather than left alone:

* `ComponentThermalPowers` lost `i_blanket_type` and `secondary_cycle_liq` **and seven
  reads** — it recomputed six of `calculate_component_thermal_powers`'s twenty-seven
  outputs and discarded them, which is `switch_kwarg_survey.md` §6's "reads dead at every
  value" residue. It is no longer wired into four fixed points it does not consume.
* Every remaining value is **threaded from the file**, so no slot can contradict it and
  `machine_survey`'s `pinned` column for `large_tokamak_eval.IN.DAT` is **empty**.

What is left is a genuine 2 x 3 x 2 product of occupants over a 26-read signature — twelve
classes for `ComponentThermalPowers` and eight for `DeltaEtaStep`, removing **two** further
reads on the reference machine (`.primary_pumping.p_fw_blkt_coolant_pump_mw` on the
`FRACTION_OF_HEAT` arm and `.fwbs.f_nuc_pow_bz_liq` on the single-coolant one). The
arithmetic is worth stating because it is the first place in this wave where the policy's
cost clearly exceeds what it buys, and because the shape it wants is **nesting** —
`model_tree_design.md` §2's sub-slot — rather than a flat product. Left for a pass that
decides that, with the arm analysis above so it is mechanical when it happens.

#### `.ife.ife` as a port — still there, deliberately

Four cost nodes read `.ife.ife` as an ordinary declared port and multiply by
`jnp.where(ife == 1, 0.0, ...)` (`switch_kwarg_survey.md` §4.11's ten sites, six switches).
With `ife == 1` now refused at assembly those `where`s are provably inert, so the reads
could go. They are not static kwargs and the brief's rule — *a switch read arithmetically
in a formula stays an ordinary declared read* — covers them; recorded here as the obvious
next increment rather than taken.

## 15. State, 2026-08-27 — waves 2/3 registered, the PF coil block driven

Consolidation round 2 (`_audit/consolidation_round_2.md`), single agent, serial.
Section numbers above are frozen; this appends.

### 15.1 What landed

The eleven items the round-2 brief listed are registered, each per its record's
registration instructions: `.tokamak.plasma_current` (3 slots — the record's fourth,
`internal_inductance`, was superseded by `plasma_inductance`'s occupant under that
record's own open question 1; both owned `.physics.ind_plasma_internal_norm` and one
graph admits one producer), `.tokamak.bootstrap_current` plus the three new `Tokamak`
slots (`diamagnetic_current`, `pfirsch_schluter_current`, `current_fractions`),
`.tokamak.l_h_transition` (single node, live arm 19), `.tokamak.density_limit` (3),
`.tokamak.scrape_off_layer` (8, flat), `.tokamak.cs_coil` (3) + `.tokamak.pf_coil`
(10, incl. `inductance`) behind one joint predicate (`indat._pf_coil_system_arm` — one
predicate, thirteen slots, resolved once), `.tokamak.plasma_inductance` (3),
`TokamakPulse.burn_time`, `.tokamak.shield` (2, D-arm on the existing
`_fw_blkt_vv_shape_arm` key), and nothing for `water_use`/`cs_fatigue` (scoping
records; registry rows already existed). `Tokamak` is twenty-eight slots, twenty-six
filled, two empty.

Registration created **two new raw cycles**, both cut in `mda.CUTS` by measurement:

* volt-seconds/burn-time (2 nodes): both edges are sufficient single cuts; the one
  chosen, `.times.t_plant_pulse_burn`, is the value PROCESS carries across a pass
  (`physics.py:4882-4884`'s own comment).
* the PF coil ring (5 nodes): eleven candidates, exactly two sufficient single cuts
  (`c_pf_cs_coils_peak_ma`, `f_j_cs_start_end_flat_top`), **neither chosen** — the cut
  is the pair PROCESS itself seeds on `first_call` (`pfcoil.py:605-608`),
  `ind_pf_cs_plasma_mutual` + `n_pf_coil_turns`, each necessary given the other. One
  `FixedPoint` over two unknowns, driven Picard; the RootFind on the
  `n_pf_coil_turns` residual is a recorded later upgrade, deliberately not done.
  `test_mda.py` pins both measurements and both tie-breaks.

### 15.2 Headline numbers (cottax pinned at f0bf9bb)

* Tokamak MDA harness (`--machine`): **597 agreements (48 array, 40 both-zero) / 16
  disagreements / 0 unverifiable / 0 ungrounded / 20 errors; 633 owned walked, 0
  unaccounted; switch kwargs 8 checked, 0 mismatched** — against 500/14/0/0/20 (534
  walked) at round start. The 14 pre-existing disagreements are unchanged (the
  explained `VacuumOld` pair and the 1.3e-6 cost chain); the two new ones are one
  cause, `.pf_coil.n_pf_coil_turns`'s dead array tail (port writes structural zeros at
  indices 8-21 where PROCESS's converged state still holds the `100.0` `first_call`
  seed; live indices agree) — documented in `mda_harness.EXPLAINED_DISAGREEMENTS`,
  not filtered.
* Stellarator MDA harness: byte-identical to the round-start baseline
  (472/34/3/0, errors 25, switches 7/0). The stellarator pin regenerates identically.
* Tokamak boundary: 328 inputs + 8 guesses → **349 inputs + 11 guesses**. Ten rows
  closed (`plasma_current`, `alphaj`, `f_c_plasma_auxiliary`, `vol_shld_total`,
  `t_plant_pulse_burn`, five `.pf_coil.*` extents/masses — including the three
  cross-wave reads `Structure`/`Cryostat` were waiting on); thirty-one opened, every
  one a newly registered node's own declared input named in advance by its record.
* `machine_survey` on `large_tokamak_eval`: `unknown` 10 → **3** (`i_beta_component`,
  `i_plant_availability`, `i_shld_primary_heat`); `factory` 17 → 24; `pinned` stays 0.

### 15.3 Two policy items, now standing

* **Structural integers moved by the solve.** `noh` (`pfcoil/inductance.py`, a `ceil`
  of converged CS geometry, pinned 30) and `n_cycle` (`cs_fatigue.py`, a trip count)
  are integers no input decides and no gradient check can treat as smooth: the policy
  is *pin as a module constant / static argument, excuse the gradient check for the
  discrete output with the reason recorded, never tune a tolerance around a staircase*.
  `cs_fatigue.md` open question 1 carries the DECIDED-DEFERRED application (eager
  `lax.while_loop`, Tier-1 values, gradients structurally excused).
* **`KNOWN_UNVERIFIABLE_OUTPUTS` is device-gated.** Entries carry the device root they
  apply on (`mda_harness.device_root`), resolved off the graph itself rather than from
  a caller argument — a stellarator-specific unverifiable output must not silently
  excuse the same-named tokamak value.

### 15.4 PlasmaComposition

The `i_plasma_ignited` occupant split (`PlasmaCompositionIgnited` /
`PlasmaCompositionNonIgnited`, factory-threaded) landed with the switch-kwarg
conversion (commit 1db889f6) before this round started; round 2 verified it rather
than re-fixing it. The tokamak's remaining `not data-backed: 1` in `switch_audit` is
`imp_indices`, a declared kind-(c) non-switch — the old hardcoded-IGNITED defect no
longer surfaces in any audit column.

## 16. State, 2026-08-29 — the wave day banked, and the tokamak SAND ladder runs

Consolidation round 3 (`_audit/consolidation_round_3.md`) §§1-2, single agent, serial.
Section numbers above are frozen; this appends.

### 16.1 What the wave day bought, measured on the merged tree

Round 3 §1's two producer branches were merged before the session ended
(`aea56b7b`, `87ee1285`) and the acceptance battery it prescribed was not run. It is
run here, against `HEAD` rather than against any recorded number:

| gate | value |
|---|---|
| `pytest tests/functional_process` | **5614 passed**, 4857 skipped, 0 failed (~2 min). The 3730 in the Verified-state table is stale by the whole wave day; 5474 of these were already green at `87ee1285`, and this session added 140 |
| Tokamak warm MDA (`--machine`) | **221 nodes; 648 agreements** (49 array, 42 both-zero) / **16 disagreements** / 0 unverifiable / **0 ungrounded** / 20 errors; **684 owned walked, 0 unaccounted**; switch kwargs 10 checked, 0 mismatched, 3 not data-backed |
| Stellarator warm MDA | **150 nodes; 472/34/3/0**, 25 errors, 534 walked, switches 7/0/3/0 — byte-identical to round 2's baseline |
| Stellarator SAND | C2 **326 it**, C3 **258 it** — byte-identical to §11.11's pins |
| Tokamak SAND | **runs end to end for the first time**; §16.3 |

The 16 tokamak disagreements are unchanged in membership (the explained `VacuumOld`
pair, the 1.3e-6 cost chain, `n_pf_coil_turns`' dead array tail). The CS/physics
commit reported 635 agreements; 648 is the same measurement after the TF half is
merged alongside it, which is the ordinary reason a branch's number and the trunk's
differ.

### 16.1b MDF's stellarator solve has regressed, and not in this session

`run_mdf_harness` (stellarator) now reports **C2 200 SQP iterations, not converged**
and **C3 60 iterations, not converged**, against the Verified-state table's *129,
converged* / *200, not converged*. C2 stopping at exactly its cap while no longer
converging is the shape of a real regression, not a re-tuning.

**It is not this session's**, and that was measured rather than argued: a `git worktree`
at `87ee1285` — the session-start commit, before anything here — produces the same two
numbers, and an instrumented run shows the MDF path **never reaches the `None` seed at
all** (`NONE-SEEDED VARIABLES SEEN: none`), so §16.2's change is inert for it. The move
therefore belongs somewhere in the 2026-08-27 wave day, whose own brief flagged the
L-mode profile reset for c81 as something that "may legitimately move stellarator
numbers; analyse, don't suppress". SAND's stellarator numbers did **not** move
(326/258, byte-identical), which narrows it: the graph the two harnesses share is
unchanged, so the difference is in what MDF does with it.

Folded into round 3 §4's MDF item rather than opened as a separate thread — that item
already owns the cap question ("is MDF C3's 200-and-not-converged the same cap artefact
SAND's was?"), and the answer now has to cover C2 as well.

**[RESOLVED 2026-08-29 — `_audit/optimise_design.md` §14.** It was a cap, exactly as
§12.2's SAND "oscillation" was: C2 converges at **523** iterations (`conv 7.400e-09`,
`objf 1.217758052`), on the same point SAND's C2 reaches to `1.2e-07 .. 1.5e-05` per
`ixc`, and `MAX_ITER = 200` stopped it two-thirds of the way through. The mover is
`1db889f6` (round 2), not the wave day — every tree from it to `HEAD` and every cottax
from `b7c5572` to today is byte-identical on these numbers — and it *improved* C2, whose
pre-round-2 ancestor stops at 144 without reaching the optimum. C3's 60 is **not** a cap:
`pyvmcon` raises `QSPSolverException`, and raising `MAX_ITER` does not move it. The
recorded 129 is from a configuration that cannot be rebuilt on either axis and is not a
target. `MAX_ITER` is now 800 and the report says which of the three ways a solve ended.
The Verified-state row above should read **C2 523, converged / C3 60, the QP went
infeasible**.]**

### 16.2 `.tfcoil.sig_tf_cs_bucked`: the seed for a value PROCESS never writes

The tokamak SAND harness died at `run_sand_harness.py:239` with
`ValueError: None is not a valid value for jnp.array`, the moment the TF stress
producers landed and constraint 72 acquired a reader. **One** context variable had no
ground truth: `stresscl` assigns `sig_tf_cs_bucked` only at `i_tf_bucking >= 2`
(`process/models/tfcoil/base.py:3235`), so on `large_tokamak_eval` it is `None` in
PROCESS's own **converged** `DataStructure` — §11.5 recorded the `None`, and
`_audit/units/models/tfcoil/superconducting.md` § "a dead read, not a gap" recorded
why it is not a missing producer: `sand._bind` declares an `In` for every non-switch
parameter of a constraint whether or not the statically selected arm consumes it.

**The policy: `ground_truth` seeds `nan` for a field PROCESS never writes**, in the one
place every seeding path already goes through, and `nan` rather than `0.0` *because the
deadness is the claim being made*. A plausible number lets a read that is not actually
dead be believed; `nan` propagates into the condition and the pre-solve probe already
stops on a non-finite one and names it — so the claim is re-checked on every run rather
than asserted once in a table. Same shape as §15.3's device-gated
`KNOWN_UNVERIFIABLE_OUTPUTS`, one layer down.

**And the measurement corrected the claim.** Python's builtin `max(x, nan)` returns `x`
(`nan > x` is False), so the alarm is loud through arithmetic and **mute through
`max`/`min`** — which is exactly c72's *other* arm. Nothing depends on it (a machine
taking that arm has `i_tf_bucking >= 2`, PROCESS writes the field, the sentinel is never
reached) but "seed `nan` and it will show" is the obvious thing to believe and it is not
unconditionally true. `test_sand.py` pins both halves.

**The structural repair is measured and deliberately not done.** Having `_bind` drop the
reads the bound arm cannot reach is the honest fix; the cost of it is that it moves
edges. Measured by `make_jaxpr` invar-liveness (`switch_kwarg_survey.md` §1 method 3):
**12 dead declared reads on the tokamak** (c2, c5, c24 ×3, c72, c68 ×6) and **6 on the
stellarator** (c2 ×3, c24 ×3), of which four are produced *inside* the drive. Dropping
those can change `constraints_outside_block`, hence which constraints an `Optimise`
omits, hence the pinned Stage C numbers. That is a wave with its own before/after, not
a seeding rule.

### 16.3 The tokamak ladder, end to end

`python -m functional_process.run_sand_harness --machine`, 164 drive nodes, 9 unknowns,
30 conditions, 390 context, 9 equalities / 20 inequalities, 82 schedule steps:

- **Stage A**: 20 of 23 comparable conditions exact (rel < 1e-9), c72 itself at
  `1.2e-15`. Every constraint the wave day produced — 26, 27, 33, 35, 36, 65, 81, 68 —
  agrees.
- **Stage B**: **0 non-finite cells**, 61.9 s compile, **2.2 ms** per jitted Jacobian
  against PROCESS's 0.30 s for the same one by finite differences. Four rows sit outside
  the FD's own error bar at `x4` (`objf`, c8, c16, c62) and three at `x6`.
- **Stage C2**: **0 SQP iterations**, landing on PROCESS's `x` to `1.4e-16` / `0.0`.
  That is the §11.6 shape, not convergence: **c68 (+4.9%) and c72 (+55%) are violated at
  PROCESS's own converged answer**, so the first QP is infeasible and `VmconDriver`
  returns the start. Both values are PROCESS's, reproduced to 1e-15 — an evaluation run
  does not enforce inequalities, and now that c72 has a producer the port can say so.
- **Stage C3 (cold)**: **1 of 30 conditions non-finite**, down from 7 of 25 in §11.6.
  Six of the seven were closed by the cold producers and this wave.

**The survivor is single-caused.** c65 is `nan` because
`.superconducting_tfcoil.vv_stress_quench` is, and after a *cold MDA pass* exactly one
of `vv_stress_on_quench`'s 29 inputs is still zero: **`.build.r_vv_inboard_out`, a
boundary input with no producer** (`vv_stress_quench_from_build`'s
`tf_vv_frac = r_tf_inboard_out / r_vv_inboard_out`). Substituting PROCESS's `4.3186`
for that one input alone makes the node return a finite `4.33e7`. It is
`cold_boundary.md`'s pattern exactly, a fifth cold root, visible only now that c65 has
a producer — and it is the same shape as §4's outstanding `.build.r_sh_inboard_out`.
**Cold tokamak SAND C3 is one producer away.**

### 16.4 Three consolidation items, each turned into a check

Round 3 §2's list, done — and in each case the list itself was the defect, so what
landed is a guard rather than a corrected list.

- **`DERIVED_UNPORTED_KEYS` had five stale entries, not two.** The brief named
  `itart_hcpb` and `nuclear_heating_renormalisation_arm`; measuring found
  `n_divertors`, `fw_blkt_vv_shape_arm` and `i_tf_shape_build` too. A skip whose
  `UNPORTED` refusal has since been ported defers a case that no longer exists, and
  **nothing can notice**: the parametrisation is over `UNPORTED`, so a stale entry never
  runs and never fails. Now `test_the_skip_list_holds_no_entry_that_is_now_ported`.
- **`machine_survey` said "the port has never read it" for all three `unknown` rows on
  `large_tokamak_eval`, and it is false for every one.** Two (`i_beta_component`,
  `i_plant_availability`) are bound by the constraint/objective layer as static kwargs —
  read, in a layer the machine tree does not contain. The third,
  `.heat_transport.i_shld_primary_heat`, is declared as an ordinary `In`: a **declared
  port carrying a switch integer**, which is `switch_kwarg_survey.md` §0's second defect
  arriving from a second direction, i.e. work rather than absence. The verdict stays
  `unknown` (§15.2's counts are cited elsewhere); only the sentence changed, and it is
  derived from the two live sources rather than listed.
- **`SLOTS` could not reach the tokamak at all.** It swaps into `REFERENCE_MACHINE`,
  which is a *stellarator*, so no tokamak slot had anywhere to go and the occupant
  meta-tests skipped the group in silence. The brief asked for "the tokamak TF
  registries"; the measurement made the ask bigger — **58 registries, 135 arms**, most
  of the tokamak. All now assemble against a tokamak base with the slot **derived** from
  the machine (by occupant type, then by slot name) rather than written as a lambda per
  slot, so a future wave's registry is covered the day it lands, and
  `test_no_slot_registry_is_covered_by_nothing` fails until a new one is placed. Two
  arms exist only jointly — `CentrepostNeutronicsAbsent` owns the same
  `.fwbs.p_cp_shield_nuclear_heat_mw` the ST renormalisation arms do, so swapping either
  alone is a duplicate producer — which cottax's refusal named and the case now states.
  **The swap-orphan pin (`reference_swaps.txt`) stays on the stellarator base**; a
  tokamak half is a separate regeneration, deferred out loud rather than half-done.

### 16.5 Registry debt, closed

`pfcoil/stresses.py` and `pfcoil/superconductor.py` had neither a record nor a row —
the CS/physics wave was asked to leave `unit_registry.md` alone while sibling agents
held it, so both modules' docstrings said a row was owed and the material went into
`pfcoil/fields.md`. Both records now exist at their mirrored paths (rows #53, #54), the
material moved unchanged, and the modules and their test cases point at their own
records. Eighteen more rows gained a wave-day line naming the dated sections in their
records, and **row 13 (`blankets/hcpb.py`) was corrected**: it still read "3
`ExplicitFunction` nodes written and **deliberately not registered**", which the
centrepost wave had made false.

### 16.6 Four wave-day findings the round-3 brief asked this section to carry

Each has a full account in its unit's record; these are the one-paragraph forms, here
because a finding that only exists inside one unit's record is not findable from the
outside.

- **The driver benchmark** (`optimise_design.md` §13): the port is **9-12x PROCESS
  end-to-end and 181x per iteration**, and the residual scaling is load-bearing for
  pyvmcon while being actively harmful to SLSQP. §16.3's 2.2 ms jitted Jacobian against
  PROCESS's 0.30 s finite-difference one is the same result seen on the tokamak.
- **The cold path closed** (`cold_boundary.md`): four cold producers took the cold
  tokamak MDA from **11 unproduced roots to 0**. What is left is one root *below* the
  MDA, found only once c65 had a producer — §16.3's `.build.r_vv_inboard_out`.
- **`.physics.f_p_div_lower` has no producer, and that is the right answer**
  (`divertor.md`, `models/divertor.py`). The double-null Wade arm is the one arm in that
  wave that adds a *read* rather than changing a constant, and the field it reads is
  written nowhere in `process/` outside the input parser (declared at
  `physics_variables.py:740`, scan variable 51, read in four places). It is an input in
  PROCESS and an input here. **Stubbing it to `1.0` would silently pick the lower
  divertor and hide the `max`** — both ST files set it to `0.5` explicitly.
- **A PROCESS half-edit, reproduced** (`blanket_library.md`): at `n_divertors == 2`
  the `if` covers only the blanket *surface* assignment, so PROCESS removes **two**
  divertors' solid angle from the surface and **one** from the volume, with no comment
  anywhere justifying the asymmetry. `models/fw.py`'s analogous arm is symmetric — the
  two functions look like the same edit and only one was made completely. Reproduced
  exactly per `traceability_policy.md`, and executed by a contract whose reference is
  PROCESS's own bound function, so a "fix" would fail the test.
- **One ulp of `arcsin`, amplified to 1e-8** (`hcpb.md`): at the trapezoid's last panel
  `1 - rho_maj**2 sin(phy_cp)**2` is analytically zero and numerically either sign;
  PROCESS clamps with `max(., 0)` and takes a square root. `np.arcsin` and `jnp.arcsin`
  differ by one ulp on the FNSF point, which lands the radicand at `+1.1e-16` for numpy
  and `-2.2e-16` for jax, so one of twenty terms differs in the eighth digit. Measured
  over 4000 fuzz points: 686 disagree at all, worst `4.8e-10` relative. **The port's `0`
  is the correct value and PROCESS's `1.05e-8` is the spurious one**, which is why the
  tolerance is set with the reason attached rather than tuned until green.

### 16.7 What is next, in order

1. **A producer for `.build.r_vv_inboard_out`** — one boundary zero, and cold tokamak
   SAND C3 runs (§16.3). `.build.r_sh_inboard_out` (§4) is the same shape and probably
   the same wave.
2. **The ST closing wave** (round 3 §3). Re-surveyed 2026-08-29 and unchanged: both
   `spherical_tokamak_eval` and `st_regression` need exactly four unported switch
   values — `i_plasma_current = 9` (FIESTA), `i_diamagnetic_current = 2`,
   `i_pfirsch_schluter_current = 1`, `i_tf_sc_mat = 9` (REBCO). The first spherical
   tokamak the port runs is still one wave away.
3. **The MDF benchmark, closure hoisting and the cap re-test** (round 3 §4), which
   carries the SAND-vs-MDF architecture decision — and now also owes an explanation for
   §16.1b's C2 regression, which is on `main` and predates this session.
4. **`_bind`'s dead reads** (§16.2), whose blast radius is now measured.

**Superseded by §16.10**, which is this session's final ordering.

### 16.8 The cold tokamak SAND solve, and what stood in front of it

**The cold tokamak solves.** From `large_tokamak_eval.IN.DAT`'s own values, the
fsolve-analogue -- the problem PROCESS actually solves in evaluation mode, its two
equalities with all 23 inequalities omitted -- converges in **7 SQP iterations**,
`conv 2.68e-17`, landing on PROCESS's converged answer to **3.4e-12** and **3.1e-12**
on `temp_plasma_electron_vol_avg_kev` and `nd_plasma_electrons_vol_avg`. §11.6 ran that
analogue warm (1 iteration, `conv 2.2e-13`); this is the same analogue from the cold
start, which is what every cold producer since `cold_boundary.md` was being bought for.
Shape: 130 drive nodes, 9 unknowns, 10 conditions, 17 inputs seeded from the cold MDA.

Two things had to be fixed to get there, and the second is the more interesting.

**1. `.build.r_vv_inboard_out` had no producer** -- §16.3's single-caused blocker,
closed by `models/build.py`'s `VacuumVesselAndShieldRadiiTfOutsideCs`
(`_audit/units/models/build.md`, 2026-08-29). It also closed
`.build.r_sh_inboard_out`, `consolidation_round_3.md` §4's last item, from the same
three lines of PROCESS.

**2. `_seed` was handing the solve cold `DataStructure` defaults for the SCC cuts.**
`run_sand_harness._seed` treats "coupling" as *drive unknowns not in the design set*,
and the `^hat.*` **cuts are not unknowns**: `mda.CUTS` mints them to open each SCC and
they enter as schedule *inputs*. They therefore fell through to `ground_truth(base,
...)`, which unmints them to the real field and reads the cold structure's dataclass
default. The cold tokamak solve was being started at
`.pf_coil.n_pf_coil_turns = 0`, `.pf_coil.ind_pf_cs_plasma_mutual = 0` and
`.times.t_plant_pulse_burn = 1000` (`times_variables.py`'s default) while a completed
cold MDA env sat beside it holding `3814.9`, `132.7` and `144099`. That is exactly the
disease `_seed`'s own docstring diagnoses for unknowns, one mint further out; the fix is
one clause and it took the run from `0` to **17 inputs seeded from the MDA env**.

**The symptom is the part worth remembering: the harness's pre-solve probe passed.**
It reported all 30 conditions finite while the solve went non-finite on `objf`, `c13`
and `c16` at *evaluation zero*. The two build their context differently -- the probe
reads `fallback` first (so it used the MDA values), the solve runs the schedule on
`_inputs_only` of the seeded env (so it used the defaults). **A probe that seeds
differently from the solve it is probing can only report on itself**, and it will report
health. Diffing the two contexts is what found this: 3 entries of 390 differed, and all
three were cuts.

### 16.9 The full 25-constraint tokamak problem is infeasible, and now says so

With the inequalities in, C2 and C3 both return their start after **0 SQP iterations**.
That is not convergence: `pyvmcon`'s first QP has no feasible point. Measured at the
cold start, by evaluating the condition map and its `jacfwd` there:

| condition | value | gradient row |
|---|---|---|
| `c72` (CS Tresca yield) | **+2.73e+02** | **identically zero** |
| `c68` (Pdiv/R metric) | +3.84e-02 | 2.6e-02 |
| `c16` (net electric) | +1.95e-01 | 5.8e-02 |
| `c2` (global power balance) | +6.10e-03 | 4.1e-02 |

**A condition that is violated *and* constant cannot be fixed by any step**, so every QP
containing it is infeasible however good the rest of the problem is. c72 is that
condition, at both points: `+0.553` at PROCESS's own converged answer and `+273` cold.
The values are PROCESS's own, reproduced to `1.2e-15` (§16.3's Stage A), and PROCESS
never notices because evaluation mode does not enforce inequalities.

Eleven more rows are identically zero at the cold point (`c5`, `c8`, `c25`-`c27`,
`c33`-`c36`, `c65`, `c81`). That is structural rather than wrong: `large_tokamak_eval`
has `ixc = [4, 6]`, so the design is two plasma quantities, and the engineering
constraints on coil stress, current density and quench simply do not depend on them.

`run_sand_harness._why_no_step` now **measures** this instead of printing the old
"either VMCON converged or its QP failed -- the condition values above say which". It
costs one `jacfwd` compile on a path where the solve has already failed, and it is the
SAND-side counterpart to `run_mdf_harness._why_it_stopped` (§14).

*And the check's first draft was wrong in a way worth recording*, because it is the
same class of mistake as the probe above. "Away from satisfaction" is **not**
`abs(value) > tol`: an inequality's normalised residual is *negative when satisfied*, so
the first version reported twelve happily-satisfied constraints as the reason the QP had
no feasible point -- a confident, precise, wrong answer, printed by a run that otherwise
looked healthy. The sign convention *is* the check, so the equality/inequality split is
now read off the problem node rather than guessed from the values. It was caught by
reading the output rather than by any test, which is §11's standing lesson about
renderers arriving again in the solver.

*And the fix's own first draft was wrong the same way a third time*, which is why this
is recorded at length rather than tidied. Reading the split off the problem node and
testing `condition in definition.equalities` classifies **every** condition as the
objective: the two sides name the same nine equalities and compare equal to none of
them -- measured, **0 of 30**. The check then reported, confidently, that nothing was
stuck. `VmconDriver.n_equality`'s docstring already says how to do it and why (a driver
"cannot ask" which condition is which; `Drive.conditions` is the problem node's `reads`
and `Optimise.inputs` is `(objective, *equalities, *inequalities)`, so **counts recover
the split**) -- which is how the driver itself slices `values` at `[0]`, `[1 : 1 + meq]`
and `[1 + meq :]`. By position it reports one stuck condition, `c72` at `+2.73e+02`,
matching the standalone measurement exactly. Three drafts, three silent-agreement
failures, one line of real output: the whole §16.8-§16.9 sequence is the same lesson
about checks that agree with themselves.

### 16.10 What is next, in order

1. **Cold start, everywhere.** The cold tokamak result above is one file, one
   architecture, one problem shape. What is owed is the matrix: **SAND and MDF x cold
   and warm x every reference configuration** (`stellarator_helias`,
   `large_tokamak_eval`, `large_tokamak_nof`, `low_aspect_ratio_DEMO`, and the two
   spherical tokamaks once §16.11 lands). The pieces all exist -- `_seed`'s cut fix,
   `run_mdf_harness`'s `MAX_ITER` 800 and `_why_it_stopped`, `_why_no_step` -- so this
   is a harness loop and a table, not new modelling. Two things it will settle that
   nothing else can: whether the cut-seeding bug was also costing MDF, and whether c72's
   infeasibility is peculiar to `large_tokamak_eval`'s two-variable design or general.
2. **Why the port needs so many more iterations than PROCESS.** PROCESS solves this
   problem in **46** VMCON iterations; the port's SAND C2 takes 326 and its MDF C2 takes
   523. For SAND that gap is explained and measured (§13.7): 27 iterations are the
   tolerance (`1e-8` against PROCESS's `epsvmc = 1e-6` -- 299 at PROCESS's own), and the
   remaining ~250 are the formulation, since SAND hands the SQP 14 unknowns and 8
   equalities where PROCESS has 8 and 2. **For MDF it is not explained**, and that is a
   real hole: MDF's outer problem *is* PROCESS's (8 design, 2 equalities, 12
   inequalities), `VmconDriver.scaled` reproduces PROCESS's own `1/x0` conditioning,
   every row is one of PROCESS's normalised residuals, and both call the same `pyvmcon`
   -- yet it is 512 against 46 at equal tolerance, *starting from PROCESS's own answer*.
   Three candidates, each one controlled run: PROCESS differentiates at
   `epsfcn = 0.01`, a 1 % perturbation that smooths the problem the port sees exactly
   (the same low-pass argument that resolved x109); PROCESS pins its QP to
   `cvxpy.CLARABEL` where `VmconDriver` takes pyvmcon's default; and §10.4's 17.604 MW
   self-consistency defect means PROCESS converges a *different*, self-inconsistent
   function that may simply be easier. The port's convergence parameter is also
   non-monotone -- it touches `1e-4` at iteration 8 and then wanders four hundred more
   before an abrupt endgame, which is a QP-conditioning signature more than a model one.
   **Note the framing §13.8 already fixed**: the efficiency case was never fewer
   iterations, it is 13 ms against 2.4 s each (181x), 9.1x end-to-end *despite* 7x the
   count -- and 90 % of the port's 13 ms is now `cvxpy`, not the model.
3. **c72's infeasibility.** Decide whether the port should report a violated-and-constant
   condition as an infeasible problem (it does now), drop it as PROCESS effectively does
   in evaluation mode, or treat it as a finding about the reference design. The third
   reading is the interesting one: **PROCESS's own converged answer violates its own CS
   Tresca limit by 55 %**, and nothing in PROCESS's evaluation-mode run would ever say
   so.
4. **The two remaining ST blockers** (§16.11): the CroCo TF coil `Model` class and the
   PF coil system package. Both are unported model *packages*, not arms.
5. **`_bind`'s dead reads** (§16.2), whose blast radius is measured and which is the one
   structural repair this session deliberately deferred.

### 16.11 The ST closing wave: four arms landed, and the last two blockers are packages

`consolidation_round_3.md` §3 sized this wave as "exactly four unported switch values".
The measured frontier was **six**, for two reasons that are both now closed, and
**neither ST file assembles yet** -- that is this wave's honest headline.

| blocker (identical on `spherical_tokamak_eval` and `st_regression`) | outcome |
|---|---|
| `i_plasma_current == 9` (FIESTA ST) | **arm landed** |
| `i_diamagnetic_current == 2` (SCENE) | **arm landed** |
| `i_pfirsch_schluter_current == 1` (SCENE) | **arm landed** |
| `i_beta_norm_max == 0` (`USER_INPUT`) | **empty `None` slot landed** -- not in the brief |
| `i_tf_turn_type == 2` (CroCo) | **correctly refused** -- a whole unported `Model` class |
| `pf_coil_system_arm` | **still refused** -- ~~four~~ **five** dimensions at once (`-1`, `-2`, `-3`, `-6`, `-7`); the `-2` was hidden by the short-circuit, see §18 |

**The FIESTA defect was handled two ways, and neither was a tolerance.** `triang < 0`
gives `nan` and is reproduced, because PROCESS's own caller raises before reaching it
for every scaling but Sauter. The infinite derivative at `triang == 0` is the canonical
`x**p, 0<p<1` shape `models/safe_math.py` exists for, so the port writes
`safe_pow(triang, 0.060)`: bit-identical for every non-zero base *including* the `nan`,
finite tangent only at zero. No gradient excused, no `boundary.py` entry earned.

**Two survey blind spots, both closed and both worth knowing.**

- `machine_survey` checked `UNPORTED` and **not the registry**, so `_slot_occupant`'s
  `ValueError` path -- no occupant *and* no recorded reason -- printed as "the factory
  dispatches on it". That is how `i_beta_norm_max = 0` survived a re-survey whose whole
  purpose was to count blockers, including the one in §16.1's own priority list.
  `slot_registries()` now derives field -> registry from `indat.py`'s AST.
- **A slot on a *derived* arm index has no name in any `IN.DAT`**, so no column of a
  switch survey can ever reach it -- which is why `pf_coil_system_arm` was invisible to
  the count. `report()` now ends with one real `machine_from_indat` attempt.

**And a silent mis-assembly, the second of its kind.** Before the CroCo refusal existed,
a machine with `i_tf_turn_type = 2` **assembled as cable-in-conduit** -- measured by
appending that line to `large_tokamak_eval.IN.DAT`. The two ST files were only being
caught by a refusal inside `CICC_SUPERCONDUCTOR_PROPERTIES`, a slot a CroCo machine
never reaches. Same shape as `low_aspect_ratio_DEMO`'s integer-turn defect.

The brief's `i_tf_sc_mat = 9` hypothesis does not survive either half: "no branch in
`jcrit_from_material`" is a *stellarator* function, and `pfcoil.py:4851` has a real
`HAZELTON_ZHAI_REBCO` arm. PROCESS can do value 9 on a tokamak, and the tokamak's
blocker was never the superconductor.


## 17. State, 2026-08-30 — the iteration-count gap closed to 1.26x, and what it left open

§16.10 item 2 was "why the port needs 326 (SAND) / 523 (MDF) SQP iterations where
PROCESS needs 46", with three candidates. The answer is in
`_audit/optimise_design.md` §15; this section is the punch list that came out of it.

**The finding, in one line.** `pyvmcon.solve_qsp` defaults to `cvxpy`'s OSQP; PROCESS
overrides it with CLARABEL on every run and `VmconDriver` passed no `qsp_options` at
all, so every SQP subproblem the port ever solved went to a *different solver from
PROCESS's* — first-order ADMM at `1e-5` against interior-point at `1e-9`. Cold, at
PROCESS's own `epsvmc = 1e-6`, fixing that takes SAND from 258 to **58** and MDF from a
failure at 60 to **58**. PROCESS is 46. The gap is 1.26x, both formulations agree on the
count, and neither the tolerance (worth 10–27 iterations) nor the derivative explains
any of it.

### 17.1 The warm regression, which is the price of the fix — pick a policy

C2 (start at PROCESS's converged x) **stops short under CLARABEL in both harnesses**
where it used to converge under OSQP: MDF at 45 on an `infeasible` QP, SAND at 207 on an
`unbounded` one. Both are correct reports of a degenerate subproblem that OSQP was
concealing by returning an unverified direction, so this is information appearing, not
capability lost — but the C2 rows of both harness reports now read "not converged", and
somebody has to decide what the port should do about it. The options, none measured:

- **Bound SAND's coupling unknowns.** `VmconDriver.bounds` leaves an unknown with no
  entry unbounded on both sides, and SAND's `^hat.*`/residual unknowns have no entries.
  An `unbounded` QP status needs a feasible ray to exist; PROCESS's own variables all
  carry `boundl`/`boundu`. This is the most likely single fix for the SAND half and it
  is cheap to test.
- **A better `initial_B`.** §15.4's reading of the warm blow-up (`objf 2.81` by
  iteration 5 of MDF C2) is that `B = I` in scaled coordinates makes the first step
  essentially the negative gradient from a nearly-stationary point. PROCESS carries a
  `set_b(2.0)` retry but guards it with `n_solver_iterations < 2`, so PROCESS's own
  ladder would not fire on these failures — copying it faithfully would not help, and
  inventing a different one is a design decision.
- **Accept it and keep CLARABEL.** Defensible: cold is the only start PROCESS ever
  takes, and C2 exists to test the port against PROCESS's answer, not to be a
  production start.

Do **not** resolve this by reverting to OSQP. The 10x was that.

### 17.2 The one thing today's fix did not move: 11 % on two variables

`worst relative deviation from PROCESS's x` is **1.08e-01 in every converged cell** of
the whole matrix — two formulations x two QP solvers x two starts x two tolerances —
while every cell agrees on `objf` to six digits (1.21775...). It sits on
`f_nd_alpha_thermal_electron` (ID 109, 10.8 %) and `t_tf_superconductor_quench` (ID 56,
9.9 %); `rmajor` agrees to 2.0e-03. A flat objective in two directions is the obvious
reading and it is testable: evaluate PROCESS's objective at the port's x and the port's
at PROCESS's, and see whether the difference is inside either side's convergence
tolerance. If it is, the 11 % is not a defect and the harness should stop reporting it
as one. If it is not, it is a model discrepancy in two named variables, which is a much
smaller search than the port at large. **This is the highest-value single measurement
left in the ladder** and it is maybe an hour.

### 17.3 The residual 58 vs 46

Not explained, and worth much less than it was before: 12 iterations between two SQP
runs on functions that are not bit-identical (§10.4's self-consistency defect is still
open) is ordinary. What has NOT been ruled out, cheapest first:

- **`force_vmcon_inequality_satisfication`.** PROCESS defaults it to `1` and passes
  `additional_convergence=_ineq_cons_satisfied` (`solver.py:255-258`); the port passes
  nothing. It can only make PROCESS *slower*, so it does not explain the gap — but it
  means PROCESS's 46 is a stricter 46 than the port's 58, and the comparison should say
  so. Add it to `VmconDriver` and re-measure; it is a one-field change.
- PROCESS's trajectory is not smooth either: its own `conv` wanders between 1e-2 and
  1e-1 from iteration 5 to 33 before dropping. Nobody has compared the two trajectories
  point by point, and the port's cold trace is recorded in the scratchpad run behind
  §15. That comparison would say whether the 12 iterations are a *different path* or the
  *same path taken more carefully*.

### 17.4 Still owed from §16.10, unchanged

Cold start across the full matrix (all reference configurations, both formulations),
c72's infeasibility, the two ST blockers, `_bind`'s dead reads. §16.10 item 1 is now
cheaper than it was: the harnesses gained two knobs and the cold path works.

**c72's infeasibility (§16.9) is promoted by today's measurement**, because the whole
tokamak SAND ablation — eight cells, both starts, both QP solvers, both tolerances —
returns 0 iterations on an `infeasible` first QP in every one. The tokamak cannot
contribute to any solver or formulation question until that row is resolved, so it is
now blocking rather than merely open.

### 17.5 Verified state

`functional_process/core/solver/drivers.py` gained `qsp_solver` (default `"CLARABEL"`,
PROCESS's) and `epsfcn` (default `None`, exact Jacobian). Commits `60eb752c`,
`92aa3688`, `56167578`. Numbers in `_audit/optimise_design.md` §15; every cell in that
table was measured today, none carried forward.


## 18. The spherical tokamaks' complete blocker list, 2026-08-30

§16.11 left the two ST files "refused, two blockers, both packages" and §16.10 item 4
carried them forward. This section is the **complete** list, obtained by patching past
every refusal at once rather than one at a time -- the discipline
`consolidation_round_3.md` §5 asks for, applied to a whole file instead of to one
switch. **Neither ST file assembles, and no work landed here changes that**; what
landed is a correct count and a refusal message that carries it.

### 18.1 How it was measured

`_slot_occupant`, `_refuse_unported_switch` and `_pf_coil_system_arm` were monkeypatched
in a scratch script to **record** a refusal and continue with an arbitrary stand-in
occupant. Assembly then runs to completion in one pass and every refusal is in the log,
not just the first. The resulting machine is numerically meaningless -- the point is the
frontier, and the stand-in graph is only evidence about *structure*.

Then, on that stand-in machine, `graph_for` / `sand.reference_problem` /
`mdf.assemble` were run with the file's own `ixc`/`icc`/`i_figure_merit` and
`sand.switch_values_for(cold, ...)`, to find what lies **behind** the model blockers.
That half found one more blocker (§18.3) and otherwise came back clean.

### 18.2 The eight model-level blockers, identical on both files

| # | blocker | cluster |
|---|---|---|
| 1 | `i_tf_turn_type == 2` -- the CroCo TF `Model` class | CroCo |
| 2 | `i_str_wp_i_tf_sc_mat_cicc_sc_properties == (1, 9)` -- `HAZELTON_ZHAI_REBCO` tape in the CICC properties slot | CroCo |
| 3 | `i_str_wp_i_tf_sc_mat_temp_margin == (1, 9)` -- the same tape in the temperature-margin slot | CroCo |
| 4 | `pf_coil_system_arm == -1` -- `iohcl = 0`, no central solenoid at all | PF |
| 5 | `pf_coil_system_arm == -2` -- `i_pf_location = (2,3,3,4)`, `n_pf_coils_in_group = (2,2,2,2)` | PF |
| 6 | `pf_coil_system_arm == -3` -- `itart`/`itartpf` ST placement and currents | PF |
| 7 | `pf_coil_system_arm == -6` -- `(i_pf_superconductor, i_cs_superconductor) = (9, 1)` | PF |
| 8 | `pf_coil_system_arm == -7` -- `i_tf_shape = 2` picture frame, `i_r_pf_outside_tf_placement = 1` | PF |

**Two clusters, not eight items.** 1-3 are one package: `i_tf_sc_mat = 9` is a *tape*
superconductor and the CICC properties function refuses a non-`CABLE` shape in its first
four lines, so the two slot refusals are what the CroCo path exists to answer. Porting
CroCo removes all three from the list -- precisely because a CroCo machine never reaches
either CICC slot -- but note it does not *port* those two slots: the CroCo namespace
needs its own superconductor-properties and temperature-margin occupants, keyed on the
same `i_tf_sc_mat`.

**§16.11 recorded four PF dimensions; it is five.** `_pf_coil_system_arm` short-circuits
at `-1` and `-2` was never evaluated. `_pf_coil_system_deviations` (2026-08-30) now
evaluates all seven predicates and `machine_from_indat` raises one error naming every
refused dimension with its recorded reason and the count. This is the third instance of
the same class of error in this file's history (§16.11's own "six, not four"; §16.9's
three self-agreeing drafts), and it is now checked by a test rather than by care.

### 18.3 The ninth blocker, `st_regression` only, and it is not a model

`st_regression.IN.DAT` activates **iteration variable 135**,
`f_nd_impurity_electrons(13)` -- an `ITERATION_VARIABLES` entry with an `array_index`.
`sand.iteration_variable_path` refuses it (`optimise_design.md` §1.3: an `Optimise`
cannot own an array element while a node reads the enclosing array), and **both** SAND
and MDF hit it, at `optimise_graph` and `mdf.assemble` respectively. It is a formulation
blocker, entirely independent of the two model packages, and it would still be there the
day CroCo and the PF system land. `spherical_tokamak_eval` does not have it: its `ixc`
is `[4, 6, 29]`.

Dropping 135 and re-measuring, both formulations assemble -- so it is the *only* thing
of its kind in that file's 14 iteration variables.

### 18.4 What is behind the blockers: nothing else, measured

With stand-ins for all eight (and 135 dropped for `st_regression`):

| | `spherical_tokamak_eval` | `st_regression` |
|---|---|---|
| `graph_for` | 221 nodes | 221 nodes |
| `sand.switch_values_for` | 7 entries | 6 entries |
| `mdf.assemble` | OK -- 240 nodes, 3 design, 19 conditions (3 eq / 15 ineq), 220 inner blocks, 5 driven | OK -- 240 nodes, 13 design, 19 conditions, 218 inner blocks, 5 driven |
| `sand.reference_problem` | OK -- 3 design, 3 equalities | OK -- 13 design, 3 equalities |
| constraints omitted | none | none |

**Every active constraint of both files assembles** (`icc` is 18 long for both) and
**no iteration variable collides with a producer**. The `ValueError: design variable(s)
['.build.dr_bore'] are not boundary input` seen before this session was an artefact of
calling `mdf.assemble` without `graph=`/`switch_values=`; with the machine's own graph it
does not occur. Caveat the stand-ins deserve: the real ST occupants would own different
`VarPath`s from the conventional ones standing in for them, so this is strong evidence
that the solver layer holds no *further* blocker, not proof.

### 18.5 What each package needs, precisely

**CroCo** -- `CROCOSuperconductingTFCoil`, `process/models/tfcoil/superconducting.py:
3773-4865`, **1093 lines**, of which `run` is 489 and `output`/`output_croco_info` are
170 that the port does not write. It calls `run_base_superconducting_tf` (already
ported; the CICC path shares it) and then five pure methods, all with the clean
keyword-in/dataclass-out seam:

| function | lines | note |
|---|---|---|
| `tf_croco_averaged_turn_geometry` | 114 | non-integer turns only |
| `tf_turn_croco_cable_space_properties` | 47 | |
| `tf_croco_inboard_areas_and_fractions` | 63 | |
| `tf_croco_superconductor_properties` | 168 | the `i_tf_sc_mat` dispatch, tape shapes only |
| `croco_voltage` | 30 | quench voltage |
| `superconductors.calculate_croco_cable_geometry` | 80 | module-level, the tape stack |

Plus `superconductors.hijc_rebco` (122 lines), which the port has never called --
`jcrit_rebco` and `gl_rebco` it already uses on the stellarator side, `hijc_rebco` is
what `i_tf_sc_mat = 9` needs and it is the only genuinely new material model in the
package. State: **19** `*croco*`/`*hts_tape*` fields in
`superconducting_tf_coil_variables.py`, none of which any ported node owns or reads.
It refuses integer turn geometry outright (`:3838`) -- both ST files set
`i_tf_turns_integer = 0`, so that arm need not be written. Estimated shape: one new
namespace under `.tokamak.tf_coil` with 5-6 nodes and its own superconductor-properties
slot, and a `safe_*` review of the tape-stack divisions.

**PF coil system**: five dimensions, thirteen nodes, five audit records
(`pfcoil/geometry.md`, `currents.md`, `fields.md`, `masses.md`, `inductance.md`). `-1`
alone deletes the CS from the package (no filaments, `c_cs_flat_top_end = 0`, the
`:626-661` flux-swing arm, index 6 of every coil array unwritten) and `-3` replaces coil
*placement* and *currents* wholesale (`z_tf_inside_half - zref[g]`, `ccls` from
`aspect**1.6`, no `efc` call). `-2` changes every array index. `-6` is the cheapest of
the five and is genuinely small -- `models/pfcoil.py:4851` has a real
`HAZELTON_ZHAI_REBCO` branch calling `superconductors.hijc_rebco`, and the switch's only
effect in the ported closure is which `.tfcoil.dcond` element the masses node reads.
`-7` is one placement formula. **The order that unblocks the most per unit of work is
`-6`, `-7`, `-2`, then the `-1`/`-3` pair, which is the actual package.**

**Iteration variable 135** is `optimise_design.md` §1.3's open design question, not a
model: either the enclosing `f_nd_impurity_electron_array` stops being read as a whole
(so an element can be owned), or `Optimise` learns to own an element of an array it also
reads. Neither is written.

### 18.6 One bug fixed on the way

`sand.reference_problem` called `optimise_graph` **positionally** past
`i_figure_merit`, which put `switch_values` into the `driver` slot and `omit` into
`switch_values`. Any caller passing `switch_values` got
`TypeError: {...} is not an AbstractDriver`. It was invisible because
`sand_harness.mda_env`'s own call has always used keywords and nothing else in the repo
called `reference_problem` at all -- the ST probe was its first caller. Fixed to
keywords.

## 19. State, 2026-08-30 (evening) — the cold-start matrix, and the missing producers

`optimise_design.md` §16 is the full account. This is the punch list.

**The finding in one line.** The port's graph does not produce everything PROCESS
computes: **twenty-two boundary `input` entries on `large_tokamak_nof` are fields
PROCESS writes every pipeline pass**, twenty of them frozen at exactly `0.0`. c72's
infeasibility, which §17.4 promoted to blocking, is the far end of that chain. Three
earlier explanations — a degenerate burn-time loop, a genuinely infeasible cold design,
and a bad coupling seed — were each argued from real measurements and each overturned;
§16.3 records them because the pattern (algebraic argument stated as established, killed
by a cheap measurement) is the reusable lesson.

### 19.1 The work, priority-ordered

1. **One producer remains** — twenty-one of the twenty-two landed on 2026-08-30, in
   waves (`optimise_design.md` §16.9). The pin holds **`.tfcoil.str_wp` alone**;
   ~~`.costs.c2214`, `.costs.c2222`, `.costs.c2252`, `.fwbs.dewmkg`~~ are closed —
   `c2214` and `c2252` by device-decided `Costs` slots once `.tokamak.structure` and
   `Power.pfpwr` were there to read, and **`c2222` last**, which needed something other
   than a producer: `PfMagnetCost` had been written and tested for two waves and was
   refused because it carried `.costs.supercond_cost_model` as a static kwarg over two
   arms with disjoint strand-cost reads, one of which (`.pf_coil.j_crit_str_pf`) had no
   producer either. Splitting the node into a `PfMagnetCostPerKg`/`PerKam` family and
   porting `superconpf`'s PF call closed both at once
   (`_audit/cost_boundary_inputs.md` §13.4). **The lesson worth carrying:** a node whose
   ports are wrong is as effective at holding a row on this list as a node that does not
   exist, and it is harder to see, because the row names the *account* and the defect is
   in the *edges*.
   `.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` are missing on the constraint surface
   only. **The three `tfcoil` rows are one job**: all are outputs of
   `tfcoil/base.py::stresscl`, 1053 lines with 65 parameters, which wants its own
   registry row rather than a slot. Its absence is not cosmetic — constraints 31 and 32
   are active on `large_tokamak_nof` and the port evaluates both as `0 <= max`, dropped
   rather than wrong, and `str_wp = 0` is the peak of the Nb3Sn strain fit feeding 33
   and 36.
   Two triage hints written into the agent briefs were **wrong** and are worth recording
   as such: `.buildings.dz_tf_cryostat` (seed `2.5`, not `0.0`) looked like a genuine
   input and is not — `cryostat.py:58-60` writes it unconditionally before the only live
   reader — and `.physics.dlamie` looked stellarator-only and is the reverse: its only
   writer in `process/` is on the tokamak path, so **PROCESS's own stellarator computes
   with a `dlamie` nothing ever wrote**.

2. **Port `cs_fatigue.ncycle`.** `low_aspect_ratio_DEMO`'s SAND cells stop at **zero**
   iterations on c90, violated with an identically zero row at exactly `+1.000000`.
   Blocks that configuration from any start.
3. ~~**Fix `degenerate_fixed_points`' one-level body** (§16.7).~~ **Done**
   (`optimise_design.md` §17). `graph.ancestors` with the declared problem nodes
   excluded, `owns`/conditions flattened so an array unknown is measurable, and the bare
   `except` narrowed: `sand.fixed_point_residuals` records each block's Jacobian *or* the
   exception that stopped it, and `degenerate_fixed_points` raises rather than reporting
   an unmeasured block healthy. This item's own "five of six" is wrong -- it was one of
   six, and the other five ran and returned numbers for the **wrong function** (the
   burn-time cycle came back as `J = -I` exactly, i.e. flawless). §17.1 has the table.
4. ~~**Fix `mdf.inner_residuals`' array crash** (`mdf.py:750`).~~ **Done**
   (`optimise_design.md` §17). Reduced over the array to the worst element by relative
   gap. Run on all four assembling configurations at both seeds: **the burn-time cycle is
   not converged at the cold start on all three tokamaks** (`3.1e-05` relative on
   `large_tokamak_nof`) and its answer moves when `PicardDriver.max_iter` goes 20 -> 200,
   while nothing anywhere moves warm.
4a. **Re-run §16.1's cold matrix with `PicardDriver(max_iter=200)`** -- the measurement
   item 4 makes possible and did not itself take. MDF's conditions are PROCESS's
   constraints *at a converged MDA*, and on the tokamaks cold they are not; whether that
   is worth anything to the cold rows is unmeasured, and asserting it would be §16.3's
   pattern for the fourth time.
4b. **`^problem.times.t_plant_pulse_burn.cycle` has `cond ~1e13`** on every tokamak at
   every seed -- four to five orders worse than any other block in the port, and the
   first time it has been looked at (§17.2). Its `numpy` rank verdict flips between 506
   and 507 on the threshold and means nothing; the conditioning does. Where an SQP is
   handed this as 507 residual equalities (SAND) that is a candidate explanation for a
   great deal, and it is a candidate, not a finding.
5. **Bound SAND's coupling unknowns.** §17.1's cheapest hypothesis, still unmeasured, and
   now more important than when filed: seeded at PROCESS's *converged answer*, SAND takes
   two steps and walks off a feasible point to c72 `+5.4e-01`. `VmconDriver.bounds`
   leaves an unknown with no entry unbounded on both sides and SAND's `^hat.*`/residual
   unknowns have no entries.
6. **Scale the harness's feasibility report.** §16.1's `max|eq|` column applies an
   absolute `1e-6` to SAND's residual equalities, which are in physical units — on a
   `1e20` variable a relative `1e-6` *is* `1e14`. `condition_scale` exists for this.

### 19.2 Retracted from the record

- **"SLSQP converges `large_tokamak_nof` in 36 iterations"** — it stops infeasible with
  c72 at `+9.0e+01`. The old harness had no feasibility check, so a run that *returned*
  was recorded as a run that *solved*. The inference built on it — that `pyvmcon` was
  refusing something another solver could do — is backwards: `pyvmcon` has no QP
  relaxation and refusing was the more honest report.
- **§17.4's framing of c72 as a solver question.** Correct about severity, wrong about
  cause.
- **§19.1 item 3's "five of six driven blocks are reported healthy without being
  inspected at all"** (from `optimise_design.md` §16.7). One of six, and the mechanism is
  the array `jnp.stack`, not a short body's `KeyError`. The correction makes the defect
  *worse*, not smaller: the other five were inspected with the wrong body and answered
  with a number. See `optimise_design.md` §17.1.

### 19.3 Verified state

`functional_process/models/physics/physics.py` gained `calculate_poloidal_beta` and
`PoloidalBeta`, registered `.tokamak.plasma_beta.poloidal` (`tokamak_namespace.py`).
`boundary.py` gained `computed_by_process` and `unproduced_but_computed`;
`missing_producers_tokamak.txt` pins eighteen; tokamak boundary 361 → 360. Two meta-tests
in `test_boundary.py`. Commit `e76d82d9`.

**The poloidal-beta port did not move the tokamak** — MDF `0/0/3`, SAND `2/2/35`, c72
unchanged at `+3.9e+02`/`+1.3e+02`/`+9.1e+01`. It was worth landing (a real hole,
`constraint_48` had read it unproduced since `batch5.md`) but the claim that it was the
root of the c72 chain is **not supported**, and was made before the `i_pf_current` switch
arm was checked. Recorded as another instance of §16.3's pattern.

The stellarator is unaffected and still solves cold in both formulations at 58 iterations
under CLARABEL against PROCESS's 46.

## 20. `helias_5b` closed — the reference stellarator's sibling, 2026-08-30

Four of the eight `tests/regression/input_files/*.IN.DAT` reference machines did not
assemble. **`helias_5b.IN.DAT` does now, and it had exactly one blocker with nothing
behind it.**

### 20.1 The blocker, and why it was smaller than its refusal said

`istell = 1`, refused by `indat.UNPORTED[("istell", 1)]` with `_ISTELL_PRESET_REASON`:
*"`istell` in 1..5 selects one of five hardcoded machine presets ... only `istell == 6`
(config read from file) is in scope"*. The reason was accurate about scope and misleading
about cost. Unit #8 (`preset_config.md`) had already ported the **copy mechanism** —
`select_stellarator_config_scalars`, generic over any mapping — and
`test_preset_config.py` had already driven all five preset dicts through PROCESS's own
`load_stellarator_config` as reference samples. The five arms were **unreachable, not
unwritten**: nothing in the graph, no node body, no `Out`, no test reference had to
change.

What landed is selection, in two places:

- `preset_config.machine_config_for_istell(istell, config_file)` — PROCESS's
  `match istell` (`process/models/stellarator/preset_config.py:238-257`) and nothing
  else, returning the same `(key, value)` tuple on all six arms, so
  `StellaratorMachineConfig` cannot tell a preset from a file. `STELLARATOR_MACHINE_PRESETS`
  imports the five dicts from `process/` rather than re-typing them (argued in the
  record: transcription buys a non-importing test and pays with a drift mode).
- `indat.DEVICE` gains `1`-`5` as `StellaratorProcess`, and the five `("istell", 1..5)`
  rows leave `UNPORTED`. **`istell` now has no `UNPORTED` rows at any value.** A device
  outside 0..6 is a `ValueError` ("not a known value"), which is what
  `test_machine.test_a_silent_indat_is_still_refused_but_no_longer_on_istell` probes now
  that no `istell` refusal is left to probe with.

### 20.2 What is behind it: nothing, measured

Same method as §18.4, minus the monkeypatching — no stand-in was needed, because assembly
runs clean. On the file's own `ixc`/`icc`/`i_figure_merit`, with
`sand.switch_values_for` over its cold `SingleRun(...).data`:

| | `helias_5b` (`istell = 1`) | `stellarator_helias` (`istell = 6`) |
|---|---|---|
| `machine_from_indat` | `StellaratorProcess` | `StellaratorProcess` |
| `graph_for` | 150 nodes | 150 nodes |
| `ixc` | `[4, 6, 10]` (3) | 8 |
| `icc` | `[2, 11, 16, 84, 24]` (5 = 3 eq / 2 ineq) | 14 (2 eq / 12 ineq) |
| `switch_values` | 5, incl. `istell: 1` | 4, incl. `istell: 6` |
| `mdf.assemble` | OK — 156 nodes, 3 design, 6 conditions, 145 inner blocks, 5 driven | OK — 165 nodes, 8 design, 15 conditions, 154 inner blocks, 5 driven |
| `sand.reference_problem` | OK — 3 design, 3 eq, 2 ineq, 0 degenerate | OK — 8 design, 2 eq, 12 ineq |
| constraints omitted | none | none |

**Every one of `helias_5b`'s five active constraints assembles and none of its three
iteration variables collides with a producer.** `icc = [2, 11, 16, 84, 24]` is a strict
subset-in-kind of the reference stellarator's fourteen, so this was expected; it is
recorded because expecting is not measuring.

**The two graphs have the same node count**, which is the structural claim unit #8's shape
decision makes: a machine config decides values, not topology, so a Helias-5b graph and a
Helias-5 graph are one graph with a different constant folded into one zero-input node.

### 20.3 Not attempted, deliberately

**Solving it.** `helias_5b` assembles and both formulations build; whether it *converges*
cold is a separate measurement and was not run — the reference stellarator's 58-iteration
cold solve took a full PROCESS run to calibrate against, and nothing here claims a number
for this file. `run_mdf_harness.py`/`run_sand_harness.py` accept it now, which is what
makes that measurement possible.

**Value agreement against PROCESS at `istell = 1`.** The unit's tier-1 case already
compares the port to PROCESS's reflective loop on the HELIAS5B dict itself
(`test_preset_config.py::preset-helias5b`), which is the same comparison a run-level check
would make with more machinery around it.

### 20.4 The three that remain

`spherical_tokamak_eval` and `st_regression` are §18's two clusters (CroCo, the PF
system) plus iteration variable 135 for the latter — unchanged, nothing here touches
them. `IFE.IN.DAT` is `ife == 1`, refused by `_refuse_unported_switch` with the
seven-Account-22x reason: inertial confinement is a whole device and the `.ife.*`
subsystem has no unit in `unit_registry.md` at all.

So the reference machines now split **five that assemble** (`stellarator_helias`,
`helias_5b`, `large_tokamak_nof`, `large_tokamak_eval`, `low_aspect_ratio_DEMO`) against
**three that do not** (`spherical_tokamak_eval`, `st_regression`, `IFE`), and the three
remaining are three genuinely unported packages — CroCo, the PF coil system, and IFE —
rather than any further wiring.
**Both §16.7 instruments are repaired and measured** (`optimise_design.md` §17).
`functional_process/sand.py` gained `FixedPointResidual`/`fixed_point_residuals` and
`degenerate_fixed_points` became a filter over them that raises on an unmeasurable block;
`functional_process/mdf.py::inner_residuals` reduces over array unknowns. Six regression
tests (`test_sand.py` four, `test_mdf.py` two), all of which fail against the old code:
the sand ones assert the *value* of a two-node cycle's residual Jacobian, which the
one-level body gets wrong (`-1` for every slope) rather than merely failing to produce.
`tests/functional_process` 5790 → 5796 passed, 5011 skipped; `tests/unit` 846.
## 20. State, 2026-08-30 (late) — the cold start is a stage now, and it found two things

`optimise_design.md` §17 is the full account. This is the punch list.

**In one line.** §16's rule — *a check performed where the seed supplies the answer is not
a check* — is now an instrument: `functional_process/cold_start.py` seeds the port's MDA
from the pre-model `DataStructure` and diffs every owned variable against PROCESS's own
state after one cold pipeline pass, on all four assembling configurations, pinned in
`reference_cold_start.txt` with a required reason per row.

**Cold agreement, first run:** stellarator 453/49, `large_tokamak_nof` 631/79,
`low_aspect_ratio_DEMO` 672/44, `large_tokamak_eval` 688/24 (agree/disagree, plus 2–3
output-pass-only each). `large_tokamak_nof` warm is 682/33; **the 46-row gap is the point
of the stage** and it is two causes.

### 20.1 The work this adds, priority-ordered

1. **`inductance.NOH = 30` is wrong on two of the three tokamaks — a new defect, pinned
   not fixed.** `noh` is `ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] -
   r_pf_coil_inner[CS]))` and every mutual inductance depends on it. Measured: 30/30
   (cold/converged) on `large_tokamak_eval`, **32/27** on `large_tokamak_nof`, **28/27**
   on `low_aspect_ratio_DEMO`. The constant is right on exactly the file it was measured
   on, and **no single constant is right at both designs of either other file**, so this
   is not a matter of picking a better number: it is
   `_audit/units/models/pfcoil/inductance.md`'s "a structural integer that the solve
   moves", with a price on it. Substitution measures the price: `NOH = 32` takes
   `large_tokamak_nof` cold from 631/82 to **662/51**. Whoever answers the convention
   question should expect `test_the_pinned_noh_is_right_on_one_configuration_and_wrong_on_two`
   to fail; that is what it is for.
2. **`stresscl` is now priced, not just named.** §19.1 item 1's last row,
   `.tfcoil.str_wp`, seeds at `0.0` — the *peak* of the Nb3Sn fit — and costs seven
   variables on each of the three tokamaks, worst `.tfcoil.temp_tf_superconductor_margin`
   at `1.58` against PROCESS's `1.24` (+27 %, and **optimistic**). Confirmed by
   substitution: PROCESS's own cold `str_wp` removes exactly those seven and adds none.
   Unchanged in priority, now with a number attached.
3. **The stellarator's 44-row cold chain is closed, and is not work.**
   `.build.z_tf_inside_half` and everything under it is PROCESS's solve-pass /
   report-pass ordering split (`st_coil`-then-`st_build` against the reverse). The port
   computes PROCESS's own output-pass answer to **sixteen digits**; PROCESS is the side
   that is not self-consistent. Same finding as
   `mda_harness.EXPLAINED_DISAGREEMENTS`'s `p_plant_electric_base_total_mw` entry, seen
   from the cold side. Do not chase it.
4. **A caveat for reading any cold row below `1e-4`.** `PicardDriver`'s own tolerance is
   `rtol = atol = 1e-4` and `compare` runs at `1e-6`, so a residue at `1e-6` inside or
   downstream of a `Drive` is the algorithm's stopping criterion showing through.
   `large_tokamak_eval`'s ten remaining rows are exactly that, at `1.1e-06`. If those
   rows are ever wanted as a real check, the driver tolerance has to move first — and it
   cannot today: at `1e-9` the PF block goes non-finite before it converges, which is its
   own item and is **not** investigated here.

### 20.2 What §19's list still owes, unchanged

Items 2–6 of §19.1 (`cs_fatigue.ncycle`, `degenerate_fixed_points`' one-level body,
`mdf.inner_residuals`' array crash, SAND's unbounded coupling unknowns, the feasibility
report's scaling) are untouched by this session.

### 20.3 Verified state

New: `functional_process/cold_start.py`, `functional_process/reference_cold_start.txt`
(215 rows: 4 agreement counts, 4 error counts, 196 disagreements, 11 output-pass-only),
`tests/functional_process/test_cold_start.py` (14 tests, ~59 s on a warm cache).
Changed: `mda_harness.compare` gained `seed=None`; `boundary.computed_by_process` now
delegates to `cold_start.cold_state` so the declaration-side and value-side halves of the
missing-producer question come from one evaluation. No port model changed, and no
existing pin moved.

`tests/functional_process` 5790 → 5804 passed, 5011 skipped. `tests/unit` 846 passed.

## 21. Session close, 2026-08-30 — what landed, what is open, and what not to re-derive

A long day. `optimise_design.md` §16 and §17 carry the detail; this is the handover.

### 21.1 The headline

**The port's graph did not produce everything PROCESS computes.** Twenty-two boundary
`input` entries on `large_tokamak_nof` were fields PROCESS writes every pipeline pass,
twenty of them frozen at exactly `0.0`. Every cold tokamak solve failed on the
consequence. **The count is now zero**, across every configuration and both graph
surfaces, and three tokamaks that took zero SQP steps this morning solve cold.

| | start of day | end of day |
|---|---|---|
| missing producers | 22 (unknown; nothing measured it) | **0** |
| reference machines that assemble | 4 of 8 | **5 of 8**, two more in flight |
| tokamaks solving cold | 0 | **3** |
| `tests/functional_process` | 5654 | **~5832** |

### 21.2 Priority order for tomorrow

1. **Give the burn-time cycle a Newton driver.** Measured today and unambiguous:
   `^problem.times.t_plant_pulse_burn.cycle` is a **507-dimensional** fixed point
   (1 + 22x22 + 22) with condition number `7.6e12`, driven by `PicardDriver` at
   `max_iter = 20` -- the default -- and **it does not converge cold on any tokamak**
   (worst relative gap `3.1e-05` on nof; the answer moves when the cap is raised). MDF's
   correctness claim is that its conditions are PROCESS's constraints at a *converged*
   MDA, so this is a correctness defect, not a performance one.
   Newton on `r(u) = g(u) - u` from the same cold start reaches `||r||inf 5.1e-11` in
   **one step** and machine zero in five. Jitted cost: residual `0.18 ms`, 507x507
   `jacfwd` `5.24 ms` -- **29x a residual, not 507x**, XLA batches the tangent basis. One
   Newton step is ~1.4x Picard's entire unconverged 20-iteration budget.
   The work is plumbing: `SeededNewtonDriver.drives = RootFind`, so this needs either a
   `FixedPoint`-capable Newton driver or the block residualised. Keep the linear solve on
   device (`jnp.linalg.solve` inside the jitted step) -- in the throwaway probe the dense
   `np.linalg.solve` plus host transfer cost ~10x the Jacobian.
2. **The two spherical tokamaks.** §18's eight blockers, in two clusters (CroCo TF,
   PF coil system). Two agents were mid-flight at session close; read their handover
   before restarting, they will have moved the frontier.
3. **`NOH`, the structural integer the solve moves** (`cold_start.py`'s finding).
   `PFCoil.induct` segments the CS by `ceil(...)` of a ratio that changes with the
   design: 30 on `large_tokamak_eval`, 32 on `nof` cold and 27 converged. Substituting
   32 takes nof cold from 631/82 agreements to 662/51. **No constant is right at both
   designs of either file**, so this is a convention question with a measured price, not
   a bug with a fix.
4. **`.tfcoil.str_wp`'s temperature-margin cost is now priced** and the producer landed
   late in the day (`stresscl`, row 55) -- re-run `cold_start` and see whether the seven
   rows it predicted would vanish actually do.
5. **SAND's coupling unknowns are unbounded.** §17.1's cheapest hypothesis, still
   unmeasured, and now more interesting: seeded at PROCESS's *converged answer*, SAND
   takes two steps and walks off a feasible point to c72 `+5.4e-01`.
6. **The harness's feasibility column is wrong for SAND** -- it applies an absolute
   `1e-6` to residual equalities carried in physical units. On a `1e20` variable a
   relative `1e-6` *is* `1e14`. `condition_scale` exists for this.

### 21.3 Measurements taken today that tomorrow should not redo

- **`large_tokamak_eval`'s c72 is immovable, and no producer changes that.** PROCESS's
  own finite differences at its converged point: `largest |d(c72)/dx| = 0.000000e+00`
  over its two design variables, against `1.72` on `large_tokamak_nof`'s twenty. The
  file is inequality-infeasible by construction -- PROCESS solves it with `fsolve` over
  the equalities alone and never examines the inequalities. `stresscl` gives its c72 a
  *correct* value and leaves it just as unreachable.
- **PROCESS's converged stellarator answer is not the global optimum.** PROCESS+SLSQP
  (a new adapter, PROCESS's own models/Jacobian/scaling through SciPy) reaches a
  **feasible** point at `objf 1.16603189` against PROCESS+VMCON's `1.2178` -- ~4 %
  better cost of electricity. Checked: bounds identical to VMCON's, answer inside the
  box, `objf` self-consistent to `1.15e-11`, equality/inequality split correct, worst
  inequality `-3.4e-08`. It hit the 500 cap so it is not converged either, but where it
  stands is feasible and better.
- **PROCESS+SLSQP iteration counts**, same adapter: `large_tokamak_nof` **7** (VMCON 8),
  `low_aspect_ratio_DEMO` **126** (VMCON 16), `large_tokamak_eval` 500 (cap, infeasible
  on the immovable c72), stellarator 500 (cap, feasible).
- **PROCESS's cold state is settled, not unconverged.** `ColdState.drift` runs three
  further Gauss-Seidel passes past `check_agreement`: worst motion exactly `0` on two
  configurations, `7.0e-07` on the stellarator, against smallest reported disagreements
  of `1.2e-04`. Thirty further passes move nothing. Cold disagreements are two fixed
  points of two maps, not one side failing to settle.
- **The stellarator's 44-row cold chain is not a port defect.** `Stellarator.run` orders
  `st_coil`/`st_build` one way for the solve pass and the reverse for output; the port
  matches PROCESS's own *output-pass* answer to sixteen digits. PROCESS is the
  inconsistent side.

### 21.4 Four wrong answers, recorded so they are not re-derived

All four were argued from real measurements and stated before the measurement that would
settle them. §16.3 has the detail; the pattern is the lesson.

1. **"The burn-time loop is degenerate."** Killed by one derivative row: `d(t_burn)/dx`
   is nonzero on 13 of 20 design variables, largest `2.0e+04`. The sixteen-digit
   `vs_required = -vs_available` agreement is a *bookkeeping tautology*, not evidence
   about the rank of the system that produced it.
2. **"The tokamak cold start is genuinely infeasible and the port is the honest one."**
   The port's cold state was consistent with a *smaller model set*. The 983/1039
   agreement that looked like vindication was measured at PROCESS's converged design,
   where `mdf.seed` hands every missing producer the right answer.
3. **"The blocker is model ports"** (on the PF/volt-second chain). That chain agrees to
   6-8 digits at PROCESS's answer. The blocker was elsewhere.
4. **"`beta_poloidal_vol_avg` is the root of the c72 chain."** Landing it did not move
   the matrix at all. The claim was made before checking which `i_pf_current` arm was
   even live.

Also retracted: **"SLSQP converges `large_tokamak_nof` in 36 iterations"** (it stopped
infeasible at c72 `+9.0e+01`; the old harness had no feasibility check, so a run that
*returned* was recorded as a run that *solved*), and **§16.7's own claim** that five
driven blocks raised `KeyError` and were swallowed -- one was undetectable, the other
five *ran on a wrong function* and returned confident numbers.

### 21.5 The ordering rule, which is the transferable part

Verify the two systems are the same problem before comparing how they behave:
**structure -> identity -> values -> derivatives -> behaviour.** Today entered at
behaviour and worked backwards through four wrong answers.

Two corollaries with evidence:

- **A check performed where the seed supplies the answer is not a check.** Stage A and C2
  seed boundary inputs from PROCESS's converged `DataStructure`. `cold_start.py` exists
  because of this.
- **Repair broken instruments before investigating anything.** Three of the four wrong
  answers were behavioural theories built on top of `inner_residuals` (crashed on every
  tokamak) and `degenerate_fixed_points` (reported a 507-dimensional, `1e13`-conditioned
  block as `J = -I` exactly, a flawless fixed point). Both are fixed; both found real
  defects within minutes of working.
