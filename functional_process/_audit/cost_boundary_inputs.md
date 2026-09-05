# The ten cost nodes that read nothing the graph computes — input, or missing producer?

**Question**: `GRAPH` has 19 pure-source nodes (read set entirely unowned), ten of them
`.costs.*`. For each of the ~50 variables they read: genuine PROCESS input, or an unported
producer?

**Short answer: all settled, zero are a port-coverage gap.** Of 50 distinct variables: 32
are genuine PROCESS inputs; **16 PROCESS computes, but only on a code path this
configuration (a stellarator) never executes** — mostly the PF-coil/PF-power subsystem,
which a stellarator has none of; 2 PROCESS computes to an unconditional literal `0.0`; 0
are a real gap. Three of the ten nodes (all reading only from that dead PF subsystem) were
subsequently removed from the tree entirely (2026-08-25); one was restored once tokamak
support needed it (2026-08-30).

## The finding worth carrying forward: vacuous agreement, attributed

14 of the harness's 73 "trivial" (both-sides-exactly-zero) agreements trace to **one
cause** — PROCESS's stellarator branch never running its PF-coil/PF-power subsystem at
all. No amount of harness tuning makes these informative; there's no PROCESS behaviour
there to reproduce on a stellarator. Four further non-source nodes read from the same
dead subsystem and are correct-but-vacuous for the same reason
(`.power.acpow`, `.buildings.sizing`, `.availability.electric_production`,
`.power.cryo_q_loads_step`). **Blast radius for closing this on a tokamak, pre-enumerated
so it doesn't need re-deriving**: 19 boundary variables across 6 nodes (12 owned by
`PFCoil`/`CSCoil`, 7 by `Power.pfpwr`) — a whole-subsystem port (`PFCoil` ~3600 lines),
not a node-sized fix.

## One flagged, not-yet-a-bug fragility

`.stellarator.pulse_durations` reads six `.times.*` fields; four of them are written not
by any dataclass default but by `st_init` (unported as formula, left as a literal —
`t_plant_pulse_burn = 3.15576e7`) — and the port's cold `DataStructure` only carries the
right value because its entry point happens to run `init_process` (and therefore
`st_init`) before any model runs. **This would be a real cold-start gap under any future
entry point that seeds from a bare `DataStructure()`** instead — the dataclass default for
`t_plant_pulse_burn` is `1000.0` against `st_init`'s `3.15576e7`, a factor of 31558.
Verified this configuration's cold path does run `init_process`; not fixed structurally.

## What this audit did not determine

Whether any of the 32 "genuine input" verdicts is seeded from the *wrong* field — this
audit checked whether PROCESS writes the field anywhere, not whether the port's binding
points at the field the PROCESS call site actually reads. That is exactly the
`.neoclassics.r_eff` defect class recorded in `boundary_inputs_audit.md` (a field with
zero writes that is simultaneously a genuine input *and* the wrong binding) — not checked
for any of these 50 variables.
