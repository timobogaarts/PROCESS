# Consolidation round 2 — waves 2/3 registration + fixes (2026-08-26/27)

Single agent, serial, after the switch-kwarg conversion agent has landed (verify no
other agent is running before you start heavy work; the orchestrator launches you only
when the tree is quiet). Same environment/WSL2 rules as round 1
(`consolidation_brief.md` — read its header). Do not git commit; do not touch ~/jaxgraph.

The per-unit registration detail lives in each unit's audit record ("Registration
instructions" section) — those are authoritative; this brief carries sequence, the
cross-cutting decisions, and gates.

## 1. Register, in dependency order

1. **plasma_current** (`_audit/units/models/physics/plasma_current.md`) →
   `.tokamak.plasma_current` (4 sub-slots; `i_alphaj==0` / `i_ind==0` are EMPTY slots).
2. **bootstrap_current** (`.../bootstrap_current.md`) → `.tokamak.bootstrap_current`
   plus THREE NEW `Tokamak` slots: `diamagnetic_current`, `pfirsch_schluter_current`,
   `current_fractions` (closes `HcdPrimaryInjectedPower`'s `.physics.f_c_plasma_auxiliary`
   read). New boundary inputs: `.current_drive.cboot`, `.current_drive.f_c_plasma_bootstrap_max`,
   `.physics.f_c_plasma_non_inductive`. `i_bootstrap_current==0` is an EMPTY slot.
3. **l_h_transition** (`.../l_h_transition.md`) → `.tokamak.l_h_transition`, live arm 19.
4. **density_limit** (`.../density_limit.md`) → `.tokamak.density_limit`
   (`GreenwaldDensityLimit` unconditional + `EnforcedDensityLimitGreenwald` at 7 +
   `GreenwaldFraction`).
5. **scrape_off_layer** (`.../scrape_off_layer.md`) → `.tokamak.scrape_off_layer`;
   registration shape (flat vs grouped) is your call — prefer whatever the tree's
   existing sub-namespace precedents make most uniform.
6. **pfcoil** (`_audit/units/models/pfcoil/{geometry,currents,fields,masses,inductance}.md`)
   → `.tokamak.cs_coil` (3 sub-slots) + `.tokamak.pf_coil` (10 sub-slots incl.
   `inductance`). This closes `Structure`'s and `Cryostat`'s three boundary reads.
7. **plasma_inductance** (`.../physics/plasma_inductance.md`) →
   `.tokamak.plasma_inductance` (3 sub-slots). `i_ind_plasma_internal_norm==0` is NO
   NODE (field is a run input).
8. **pulse** (`.../pulse.md`) → `burn_time: PulseBurnTime` third slot on `TokamakPulse`.
9. **plasma_fields**: the four new nodes have instance defaults in the existing
   namespace class — verify assembly picks them up; no wiring expected.
10. **shield** (`.../shield.md`) → `.tokamak.shield`; its D-shaped arm joins the
    EXISTING `_fw_blkt_vv_shape_arm` joint key in `indat.py` (do not mint a second).
11. **water_use / cs_fatigue**: nothing to register (scoping records). Registry rows only.

`indat.py`: every UNPORTED entry each record lists; the `i_plasma_current` Sauter
disjunction stays factory-owned (plasma_geometry + plasma_current key off the same arm
function).

## 2. The PF coil SCC — measure the cut, drive it Picard

The 5-file pfcoil registration creates a Shape-A ring:
`PFCoilTimePointCurrents → PFCoilCurrentWaveform → PFCoilSizes → CSFluxSwing →
PFCoilTimePointCurrents`, enlarged by `PFCoilInductance` (reads `n_pf_coil_turns`, read
by `CSFluxSwing`). Add the `mda.CUTS` entry by MEASUREMENT (the `test_mda.py` pattern:
enumerate owned variables with closing readers, test each single cut for sufficiency,
assert minimality; add the tokamak test case). Tie-break by PROCESS's stale-read
semantics: PROCESS's `first_call` seeds `ind_pf_cs_plasma_mutual = 1.0` and
`n_pf_coil_turns = 100.0` (`pfcoil.py:605-608`) — those are the loop-carried values, so
prefer a cut whose seed those literals (or the converged `DataStructure`) can supply.
Driver: FixedPointCut → Picard (DECIDED — validation against PROCESS first; RootFind on
the `n_pf_coil_turns` residual is a recorded later upgrade, do not do it now).

## 3. Fixes decided by the orchestrator

- **`PlasmaComposition` ignition split** (`models/physics/namespace.py:386` hardcodes
  `IGNITED`): make `i_plasma_ignited` select occupants — the NON_IGNITED arm reads a
  superset (adds the beam term; see `process/models/physics/plasma_composition`-area
  code, evidence in the MDA agent's report + `next_steps.md` §14.8). Factory already
  resolves the switch; thread it. The stellarator (ignited) machine's digest must not
  move; the tokamak gains the correct occupant and `switch_audit`'s 1 mismatch → 0.
- **`boundary._main`'s orphan message** names `reference_boundary.txt` even under
  `--machine` — fix the message to name the machine's pin.
- **`ComparisonReport.summary` truncates error listing at 20** — raise sensibly or print
  a "N more" line; do not drop information silently.
- **cs_fatigue `ncycle` decision, record only** (no port): eager `lax.while_loop` is the
  correct shape per §7's no-external-reader test; Tier-1 value agreement legitimate
  (deterministic termination = ground truth); gradient check structurally excused
  (`n_cycle` is a discrete count, piecewise-constant — same class as `noh`). Put this in
  `cs_fatigue.md`'s open question as DECIDED-DEFERRED.

## 4. Docs/registry

- `unit_registry.md`: rows for every new record without one (pulse, shield, water_use,
  cs_fatigue, plasma_fields, plasma_inductance, pfcoil/inductance, l_h_transition,
  density_limit, scrape_off_layer — check which exist already; #37–#42 do); update the
  #37–#42 rows' "not registered" text once registered. Frontmatter/status must
  normalise (the `partial` trap — use `draft`).
- `next_steps.md`: add a §15 (or extend §14) state entry: waves 2/3 summary, the two new
  policy items (structural integers moved by the solve: `noh`, `n_cycle`; the
  device-gated `KNOWN_UNVERIFIABLE_OUTPUTS`), tokamak MDA headline numbers, the
  PlasmaComposition fix. Keep it factual and short; section numbers are frozen —
  append, never renumber.
- `tokamak_boundary.md`: add a dated [UPDATED] header note (the style its §counts
  already use) marking the stale rows: pf_coil zero-reads row, plasma_fields zero-reads
  row, and the two `heat_transport.p_*_coolant_pump_mw` rows hcpb refuted. Do not
  rewrite the original text.

## 5. Gates (serial, in order)

1. Full suite once: `$PY -m pytest tests/functional_process -q` — 0 failed.
2. Stellarator harness: MUST print 484 / 34 / 3 / 0, errors 25, switch mismatches 0,
   digest unmoved from the switch-agent's post-landing state (measure before you start,
   compare after — never against a number in a doc).
3. Tokamak harness (`--machine`): report the full numbers. Expect agreements to RISE
   (new producers) and the switch mismatch to go 1 → 0. Any new disagreement: analyse,
   don't tune. Boundary inputs closed by pfcoil/bootstrap/inductance registrations must
   show up as the tokamak input-boundary SHRINKING — print and read both pin diffs
   before regenerating.
4. `ruff check`/`format` on changed files.

Report: registration summary, the PF cut measurement table + choice, all gate numbers,
pin diffs, tokamak harness delta vs 508/14/0/0/20 with per-variable analysis of any
movement, anything you had to decide beyond this brief.
