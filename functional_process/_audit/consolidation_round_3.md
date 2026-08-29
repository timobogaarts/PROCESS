# Consolidation round 3 — the 2026-08-27 wave day's bookkeeping and the ST endgame

Written at the end of the 2026-08-27 orchestration session, for the next session to
execute. Same conventions as `consolidation_round_2.md`: single agent for the
consolidation half, per-unit records are authoritative for detail, measure before
trusting any number written here. Environment: `PY=~/miniconda/envs/process_port/bin/python`
(no "3"), x64 on, cottax pinned per measurement (`git archive HEAD src/cottax` from
`~/jaxgraph`; this session measured everything at `db4f025` — jaxgraph may have moved).
**All subagents launch on the opus model** (user rule, recorded in session memory).

## What landed today (all committed on main; every gate green when merged)

Consolidation round 2 (waves 2/3 registered, PF ring cut Picard, tokamak 203 nodes);
the MDF `_inputs_only`/`restart` seam for cottax's new owned-name guard; cottax gained
`Namespace`/`tree_from_names`/`tree_from_variables` + the `Schedule` owned-name guard
(merged to jaxgraph main, `db4f025`); the cold-boundary audit (`cold_boundary.md`) and
the four cold producers (cold tokamak MDA: 11 roots → 0); `SAND_MAX_ITER` (the
"oscillation" was a cap; stellarator C2/C3 converge, 326/258 it); the driver benchmark
(`optimise_design.md` §13 — port 9–12× PROCESS end-to-end, 181× per iteration; the
residual scaling is load-bearing for pyvmcon, harmful to SLSQP); `large_tokamak_nof`
adopted (602/9 first try); `low_aspect_ratio_DEMO` unlocked + its integer-turn
silent-mis-assembly fixed (601/30); the ST frontier chain: pulse-ramp, ECRH-13, TART
divertor (3-way split), no-precomp radii, picture-frame, TART TF shape (arm ladder
misread fixed), TART TF masses, the `i_tf_sc_mat` 18-leaf family, double-null (7 slots,
one PROCESS half-edit defect found), D-shaped fw/blkt/vv (+`dshellarea`/`dshellvol`),
hcpb centrepost cluster (composite node, renormalisation square total).

## 1. Pending merges (check `git branch` first)

Two producer agents were wrapping when the session ended:
- `worktree-agent-a841026fe5f56877b` — TF half of `optimise_design.md` §11.5's
  missing-producer table (`j_tf_wp_critical`, margin, quench c35 [CoolProp decision
  required — see `quench.md`], VV stress, TF stress chain).
- `worktree-agent-a88cf2e706e09a584` — CS/physics half (CS criticals/margin/shear,
  beta trio, c68, the L-mode profile reset for c81 — may legitimately move stellarator
  numbers; analyse, don't suppress).
If their branches exist: merge (watch `indat.py`; their model files are disjoint from
everything merged), run their unit tests + `test_machine`/`test_registry_coverage`,
then the acceptance battery they may have skipped: §11.5 re-check, cold SAND probe
(`run_sand_harness --machine` — c33/c35 should go finite; if C3 solves, record it),
warm harness both machines, pin diffs read before regenerating, full suite.

## 2. Registry/docs debt (the consolidation half) — **DONE 2026-08-29, `next_steps.md` §16**

Every item below is closed; three of them turned out to be bigger than written (five
stale skip entries, not two; 58 registries uncovered, not the TF group) and each landed
as a check rather than a corrected list. §1's merges were already on `main` when this
was picked up; its acceptance battery is in §16.1, the one thing it found — the
`sig_tf_cs_bucked` `None` — is §16.2, and the four findings this section asked §16 to
carry are §16.6. Kept below as written, for the diff.

- `unit_registry.md`: rows/status updates for every 2026-08-27 wave above (each unit's
  dated section carries the facts; row 13's `itart == 1` note is explicitly stale).
- `next_steps.md`: append §16 (never renumber): the wave-day summary, the benchmark
  headline, the cold-path closure, the `f_p_div_lower` producer-less finding, the
  double-null half-edit defect, the ulp-amplification finding (hcpb.md).
- `test_machine.py` `DERIVED_UNPORTED_KEYS`: `"itart_hcpb"` and
  `"nuclear_heating_renormalisation_arm"` are stale entries, left to avoid concurrent
  collisions — remove.
- `machine_survey`'s "the port has never read it" phrasing misleads for graph-declared
  inputs (integer-turn agent's report has the analysis) — small fix + test.
- `test_machine.py` `SLOTS` does not cover the tokamak TF registries
  (`SC_TF_MASSES`, `SC_TF_WP_GEOMETRY`, `TF_CASE_AREAS`, `CICC_TURN_GEOMETRY`,
  `TF_COIL_SHAPE`, `TF_OUTBOARD_MID/EDGE_RIPPLE`, `TF_INBOARD_RADII`,
  `CENTREPOST_NEUTRONICS`...) — the occupant meta-tests silently skip the whole group.

## 3. The ST closing wave (after §1 merges; territories then free)

Survey-enumerated remainder for BOTH `spherical_tokamak_eval` and `st_regression`:
`i_plasma_current == 9` (FIESTA ST — NaN for negative triangularity, infinite
derivative at zero; port the defect faithfully), `i_diamagnetic_current == 2`,
`i_pfirsch_schluter_current == 1`, `pf_coil_system_arm == -3`, and one remaining
`i_tf_sc_mat = 9` tfcoil site. When they assemble: `machine_survey` (new-integer check)
then the warm harness — the first spherical tokamak the port ever runs is the
headline measurement.

## 4. Queued measurements/decisions

- **MDF benchmark + closure hoisting + MDF-C3 cap re-test** (user-approved direction,
  not yet dispatched): same jit discipline as §13's benchmark; test whether MDF C3's
  "200 it, not converged" is the same cap artefact SAND's was; hoist per-solve jit
  closures so marginal solves stop re-paying trace time. This carries the SAND-vs-MDF
  architecture decision — the measured case for MDF is already strong (§13: evaluation
  is ms-scale, so hiding the coupling inside the evaluation stops costing anything).
- Cold SAND C3 tokamak end-to-end once §1's producers land.
- Tokamak MDF assembly (inherits the same producers).
- D-shaped × single-null cells (~20 lines of composition, refusal says exactly this).
- `blanket_library.py`'s private shell helpers filing (ivc record notes it).
- `.build.r_sh_inboard_out` producer (hcpb centrepost's one declared-not-produced read).

## 5. Standing session lessons (operational)

- Agent worktrees cut at the session-start commit, NOT current HEAD: every brief must
  start with a base check + `git merge main`.
- Agents commit on their worktree branch; orchestrator merges; cross-branch *semantic*
  conflicts happen (a test pinning a refusal another wave just ported) — run the
  meta-tests after every merge.
- Frontier probing: refusals name only the first blocker; "five slots" measured as
  seven once and exactly five another time — always re-measure; a diagnostic in-memory
  patch past a refusal cheaply reveals whether the next one is the same cluster.
