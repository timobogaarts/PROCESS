# Next steps

**What this file is.** A current-state reference and a priority-ordered punch list for
the `functional_process` port. It is deliberately *not* a changelog: sections record what
is true now and what is open, not the order in which it became true.
`unit_registry.md` remains the authoritative per-unit status.

**Section numbers are frozen.** They are cited from other `_audit/*.md` records, from the
per-unit `.md` records, and from live `.py` docstrings (`total_process.py` cites §5,
`mda_harness.py` §8, `sand.py` §6, among others). A section whose material is closed is
emptied to a stub that says where the live material went; nothing is ever renumbered.

**Where to start:** §11 (verified state and priority order), then §5 (the structural
vocabulary — Shape A / Shape B — that the code itself cites).

## Verified state

| check | value |
|---|---|
| `pytest functional_process -q` | **3730 passed**, 3344 skipped, 0 failed (the 6 added are `test_mda_harness.py`; 3724 re-confirmed 2026-08-25 before it landed. Previously: re-measured 2026-08-20 — a `git worktree` at `aca0fb6d` also measures 3724, so the enum conversion of `model_tree_design.md` §8 step 1 added and moved nothing) |
| `cd ~/jaxgraph && pytest` | **738 passed**, 3 skipped (re-measured 2026-08-25; the recorded 482 was stale by ~250 tests — cottax has moved through `fdc4f5e` since, and the port's own numbers are unmoved by all of it) |
| `run_mda_harness.py` | **499 agreements** (23 array-valued, **73 both-sides-exactly-zero**), **34 disagreements** (0 in driven blocks, 34 acyclic), 3 unverifiable, **0 ungrounded**, 21 errors |
| … accounting | **557 owned variables walked, 0 unaccounted**; 61 switch kwargs checked / 0 mismatched / 3 not data-backed / **0 unresolved** |
| `GRAPH` (`REFERENCE_CONFIGURATION`) | **159 nodes**; **139 blocks, 14 driven**; **348 unowned inputs** (re-measured 2026-08-25 against `Blocking.scc(driven_graph(GRAPH))`; the 138/349 carried here since `LModeProfileReset` were stale by one, and the model-tree conversion did not move them — before == after) |
| **MDA from a cold `IN.DAT`** | **137 blocks, 0 failures**, 1 non-finite (`.physics.nu_star`, `nan` in PROCESS too, read by nothing) |
| SAND | 30 conditions × 23 design, **0 non-finite cells**; graph 171 nodes |
| SAND C2 (seeded from PROCESS's answer) | **42 SQP iterations**, conv `9.9e-10`, `objf 1.2177574` |
| **SAND C3 (cold start)** | **100 iterations**, stalls at conv `1.7e-06` (`max\|eq\| 5.5e-10`, feasible) at `objf 1.2177575` — **the same point as C2 to six digits**; §11.11 |
| MDF C2 / C3 | **129 iterations, converged** / **200 iterations, not converged** (was 127/converged before §11.11's node; the cold problem it now solves is a different, correct one) |
| MDF | 15 conditions × 8 design (`icc` × `ixc`), Jacobian compared to PROCESS's **unreduced** |
| PROCESS itself, same problem | 46 VMCON iterations, **94 s**, conv `2.40e-07` |

**A gap this exposed, cheap to close.** SAND has **no Stage B0** — no check of its own
`jax.jacfwd` Jacobian against a central difference of its own condition map. Adding one is
what found the `c24` kink above, in 2 of 690 cells, and nothing else in the repo looks for
this failure. Note it is a *different* class from the audited one: §9/§10.5b/§11.8 are
about `nan` derivatives **at exactly zero**, where this is a finite derivative on one side
and an unbounded one on the other, `1.9e-09` from the switch. How many other clamped roots
the solve is sitting on is unknown and unlooked-for.

**Model tree — `model_tree_design.md` §8 step 3 is DONE** (2026-08-25). `COMMON` is a
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
`_audit/x109_pinning_verification.md` settled the feasibility question — the pinned point
is genuinely better (`max|eq| 2.1e-12`, no inequality violated) and the multiplier
hypothesis is refuted by five orders of magnitude (`Σ|λ_eq| = 1.22`, not `6.3e+04`).
`_audit/x109_hypotheses.md` then settled *why*, and the cause is a **kink in the model**:
at every converged point the design sits on `(Te + Ti)/20 == 0.65`, the threshold of
`fast_alpha_beta`'s clamped square root (`physics_A_pure_formulas.py:342-348`) — `1.9e-09`
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
(`x109_hypotheses.md` says `epsfcn = 0.01`; the value is `1.0e-3`, matching
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
  (`confinement_time.md` and `physics_B_composition.md` independently),
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

- **`preset_config.py`** (unit #8) — a **fourth** instance of "this node always/only
  produces literals", alongside unit #6's device-preset literals and chunk 1D's constants.
  One policy decision would settle all four; it has not been made.
- **`build.py`** — `dz_shld_upper` under `blktmodel <= 0`.
- **`neoclassics.py`** — `.neoclassics.iota`/`.er` producer still unlocated.
- **Ported code whose only caller is out of scope** — unit #21's
  `set_pedestal_and_separatrix_values` (reachable only from tokamak unit #22) and
  `ImpurityRadiation`'s whole-file treatment. Worth a registry-level policy on whether
  "port because splitting is more work than porting" is standing practice.
- **`fusion_reactions.py`'s `calculate_profile_y` return-value bug** (`profiles.py`
  returns `None` on both classes; 6 call sites in `current_drive.py` use the return value
  arithmetically) — not reachable from the stellarator pipeline; flagged for
  `current_drive.py`'s eventual audit.
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
  (`.costs.cplife`), `power_B_thermal_cryo.py`'s six (`delta_eta`, `eta_turbine`,
  `etath_liq`, `temp_turbine_coolant_in`, `p_fw_div_heat_deposited_mw`,
  `p_fw_blkt_coolant_pump_mw`), and `winding_pack_total_size`'s `j_tf_wp`
  (`WindingPackJTfWp`, a degenerate/identity fixed point off the Bi-2212 branch, confirmed
  with `jax.grad` rather than asserted).

  Written but still unconverted, all otherwise fully ported and tested, all confirmed by
  direct `to_graph` construction rather than by audit reasoning:
  `power_B_thermal_cryo.py`'s `PlantThermalEfficiency`/`PlantThermalEfficiency2` and
  `Cryo`/`CryoLoads` (`.fwbs.qnuc`, plus `.power.qss`/`qac`/`qcl`/`qmisc`);
  `power_C_electric_production.py`'s `PlantElectricProduction`.

  **The discipline that matters more than either shape**: "representable via
  `FixedPointFunction`" and "`to_graph()` assembles" are necessary but not sufficient —
  check the *real* producer of what a self-loop's other branch resolves to before
  concluding the loop is genuine. Four apparent self-loops dissolved that way
  (`plasma_composition`'s `first_call`, whose bootstrap target has no dependency back on it,
  so `NextFirstCall` was deleted and `first_call`/`alphan`/`alphat` are not ported at all —
  see `physics_B_composition.md`; `st_phys`'s `beta_fast_alpha` and `beta_beam`, both Shape
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
   the same class turned up later (`power_B_thermal_cryo.py` ×6, `vacuum.py` ×1) —
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
- **Seven 1-tuple returns** (`power_B_thermal_cryo.py` ×6, `vacuum.py` ×1) — the
  `ZTfInsideHalf` bug class, third wave. Invisible to `PicardDriver`, fatal to
  `Residualise`. The full sweep is in `power_B_thermal_cryo.md`.

### Two constraints un-blocked, and the registration gaps behind them

`optimise_design.md` §2.4 recorded constraints **16** and **24** as permanently INERT. Both
are live, which matters because an `Optimise` over 12 of PROCESS's 14 active constraints is
a *different problem* and comparing its answer to PROCESS's would be meaningless.
`sand.constraint_nodes` now raises on any active `icc` entry it cannot assemble rather than
dropping it.

- `PlantElectricProductionReactor` (`power_C_electric_production.py`) — the `ireactor == 1`
  arm of `plant_electric_production`, whose five "self-referential" fields are dead reads on
  that arm. `.costs.ireactor` becomes a two-armed topology `Switch`. **This also gave
  `.costs.coe` — the run's own objective — a real dependence on the design along the
  net-electric-power path, where before it read a boundary input.**
- `StellaratorBetaAndStoredEnergy` (`stellarator_B_st_phys.py`) —
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
  (`stellarator_B_st_phys.py`, the `else` arm of `stellarator.py:2002-2054`, three
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
not the derivative of the `c24` they evaluate (`_audit/x109_hypotheses.md`). Treating the
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
  interior to `[0.0001, 0.4]`. See `_audit/x109_hypotheses.md`.

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
this file's opening summary, and `_audit/x109_hypotheses.md` §3). Of 690 cells exactly the
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
