---
kind: model-unit
status: draft
confidence: high
---

## source
`process/models/stellarator/stellarator.py`, lines 114-190: `run()` (114-189) and
`output()` (110-112, one line calling `run(output=True)`). Chunk 1A of unit #1 (see
`../../_audit/unit_registry.md`).

## scope note
This is the orchestrator, not a computation. Its value is the call sequence below —
every other chunk's "calls into other models" section should agree with it.

## call sequence

`output()` is exactly `self.run(output=True)` — not a separate path, a wrapper.

`run(output)` has **two entirely disjoint bodies**, selected by the `output` bool
(lines 125-154 vs. 156-189, with a `return` at line 154 separating them — not an
if/else, an early-return, but the effect is the same: exactly one of the two runs):

**`output=True` branch** (125-154):
```
1. costs.run()
2. costs.output()
3. availability.run(output=True)
4. physics.calculate_effective_charge_ionisation_profiles()
5. physics.outplas()
6. st_heat(self, True, self.data)                 [free fn, stellarator/heating.py]
7. self.st_phys(True)                             [chunk 1B]
8. st_density_limits(self, True, self.data)       [free fn, stellarator/density_limits.py]
9. self.st_phys(False)                            [chunk 1B — SECOND call, see finding below]
10. st_div(self, True, self.data)                 [free fn, stellarator/divertor.py]
11. st_build(self, True, self.data)                [free fn, stellarator/build.py]
12. st_coil(self, True, self.data)                 [free fn, stellarator/coils/calculate.py]
13. self.st_strc(True)                            [chunk 1D]
14. self.st_fwbs(True)                            [chunks 1E1-1E3]
15. power.tfpwr(output=True)
16. buildings.run(output=True)
17. vacuum.run(output=True)
18. power.acpow(output=True)
19. power.output_plant_electric_powers()
→ return. power_at_ignition_point is NOT called in this branch.
```

**`output=False` branch** (156-189, the actual computational path driven by the solver):
```
1. self.st_new_config()                           [chunk 1C]
2. self.st_geom()                                 [chunk 1C]
3. self.st_phys(False)                            [chunk 1B — called ONCE here, contrast with branch above]
4. st_density_limits(self, False, self.data)      [free fn, density_limits.py]
5. st_coil(self, False, self.data)                [free fn, coils/calculate.py]
6. st_build(self, False, self.data)               [free fn, build.py]
7. self.st_strc(False)                            [chunk 1D]
8. self.st_fwbs(False)                            [chunks 1E1-1E3]
9. st_div(self, False, self.data)                 [free fn, divertor.py]
10. power.tfpwr(output=False)
11. power.component_thermal_powers()
12. power.calculate_cryo_loads()
13. buildings.run(output=False)
14. vacuum.run(output=False)
15. power.acpow(output=False)
16. power.plant_electric_production()
17. availability.avail(output=False)              [TODO comment in source questions whether .run() should be called instead]
18. costs.run()
19. power_at_ignition_point(self, max_gyrotron_frequency, te0_ecrh_achievable)  [density_limits.py — see finding below]
20. data.stellarator.first_call = False
```

## cross-check against pilot findings

**Confirmed, not refuted**: `power_at_ignition_point` (step 19 of the `output=False`
branch) is called **unconditionally**, with no check of whether constraint 91 is active
in `icc`. The source comment immediately above the call (lines 178-179) says *"This call
is comparably time consuming.. If the respective constraint equation is not called, do
not set the values"* — stating an intent the code does not implement. This matches
exactly what `constraints.md`'s Constraint 91 record flagged from the other direction
(reading the constraint, not the caller). Both records should be considered confirming
each other, not independent guesses.

**One more thing found while confirming it**: `power_at_ignition_point` is called **only**
in the `output=False` branch — never in `output=True`. So `data.stellarator.
powerht_constraint`/`powerscaling_constraint`, as read by constraint 91 and as shown in
any output/report generated via `output()`, are **stale values from the last
`output=False` run**, not recomputed for reporting. Whether that's intentional
(expensive, no need to recompute for display) or an oversight isn't determinable from
this file alone — flagged, not resolved.

## new finding: a second hidden double-call pattern, in this file's own scope

The `output=True` branch calls `self.st_phys(True)` (step 7) then, two calls later,
`self.st_phys(False)` again (step 9) — with an explicit admission in the source comment
(lines 135-138): *"Change in density limit can result in changed dene? A second call of
st_phys is used to make sure it is consistent. st_phys and density limits should be
integrated to avoid this double call. Problem was probably bigger in the older version."*

This is the same shape as `power_at_ignition_point`'s "second call seems to be necessary
... and is sufficient" pattern already flagged in `density_limits.md` — call the physics
core twice, unverified, no convergence check — but it's a **second, independent instance**
of it, in a different file, closing a different (implicit) loop: `st_phys` ↔
`st_density_limits`. Two independent occurrences of the identical workaround shape is
evidence this is a recurring architectural pattern in the stellarator pipeline, not a
one-off — relevant to `test_harness.md`'s tier-2 design, which should expect more of
these, not treat the `density_limits.py` case as unique.

**Also asymmetric with the computational path**: this double-call only happens in the
`output=True` branch. The `output=False` branch (the one the solver actually iterates on)
calls `st_phys(False)` exactly once (step 3). So the "make sure it's consistent" fix is
applied only when generating output, not during optimisation — the solver never benefits
from it. Worth flagging for whoever eventually designs the `Blocking`/`Drive` structure
around `st_phys`/`st_density_limits`: their real coupling should be resolved once,
structurally, not patched only in the reporting path.

## a general finding: `output: bool` is a topology-changing switch in every sense but name

`traceability_policy.md`'s split-default policy is written in terms of `data.<area>.i_*`
switches, but `run(output: bool)` is functionally identical in kind: it selects between
**two disjoint call sequences with different reads/writes** (compare the two lists
above — e.g. `power_at_ignition_point` appears in one and not the other), exactly the
criterion the policy uses to mandate splitting. It just isn't a `DataStructure` field.
Recommend `traceability_policy.md` generalise "switch" to any parameter — not only a
`data.*` field — whose value changes which nodes run. Concretely for this unit: the pure
port should have two separate top-level functions/graphs (`compute` and `report`), not
one `run(output)` with a branch, matching how `output=True` doesn't even call the same
set of stellarator submodules as `output=False` (contrast steps 6-9 vs. step 3 above, and
note `output=True` never computes `power_at_ignition_point` at all).

## data footprint

Direct reads/writes in this file's own lines (not counting what the ~25 called
methods/functions do internally — that's each callee's own record):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.max_gyrotron_frequency` | read | explicit-arg | passed positionally into `power_at_ignition_point` |
| `.stellarator.te0_ecrh_achievable` | read | explicit-arg | same |
| `.stellarator.powerht_constraint` | write | explicit-arg | see "stale on output" note above — write only happens in `output=False` branch |
| `.stellarator.powerscaling_constraint` | write | explicit-arg | same |
| `.stellarator.first_call` | write | — (new pattern, see below) | set `False` unconditionally at end of `output=False` branch; never read in this file |

## proposed signature(s)

Not a computational unit — no `calculate_*` to propose. The port's shape here is
**graph assembly**, not a function: two `Graph`s (or one `Graph` covering both, pruned
two different ways via `Graph.prune(wanted)` — worth deciding once more chunks exist to
know whether `output=True`'s extra work is genuinely a different graph or a superset/
different-pruning of the same one, given they share most of their node list).

## tier signal
N/A — orchestration/call-map only. Defines topology for tier-3/4 units built from the
pieces it calls; contributes no tier-1/2 function of its own.

## switches touched
None (`data.*` sense). See "general finding" above re: `output: bool` behaving like one
structurally.

## calls into other models
Full list is the call sequence above. Summary of distinct callees: `costs`,
`availability`, `physics` (2 direct methods here, more via chunk 1B), `st_heat`,
`st_density_limits`, `power_at_ignition_point` (density_limits.py), `st_div`, `st_build`,
`st_coil` (coils/calculate.py — **scope gap found and fixed, see registry note above**),
`power`, `buildings`, `vacuum`, plus in-file chunks 1B/1C/1D/1E1-3.

## JAX-difficulty flags
- `needs-lax-cond-or-where` / arguably `needs-two-separate-graphs`: the `output` branch
  split itself — see "general finding" above. Severity: **blocker** for a single unified
  `run()` port; not a blocker if split into two graphs as recommended.
- New second instance of the empirically-tuned double-call pattern (`st_phys` ×2 in the
  `output=True` branch) — severity **blocker**, same class as `density_limits.md`'s
  finding, relevant to `test_harness.md` tier 2.
- `self.first_call_stfwbs` (set in `__init__`, not touched in lines 114-190 but visible in
  the class) and `data.stellarator.first_call` (set here, line 189) are **two separate
  "first call" flags in two different places** (instance attribute vs. `DataStructure`
  field) — neither read within this chunk's scope, so their consumer is presumably
  `st_fwbs` (chunk 1E, given the attribute name) or another chunk. Flag for chunk 1E and
  whoever aggregates: this is a temporal/statefulness dependency across separate top-level
  calls to `run()`, not just within one call — doesn't cleanly fit `implicit-io` (loop-
  local) or `implicit-io-via-callee` (deepcopy-proxy) as currently defined in
  `schema.md`. Recommend the schema/policy review (already pending per the pilot
  retrospective) add a category for this, e.g. `implicit-io-across-calls`, once chunk 1E
  confirms `first_call_stfwbs`'s actual role.

## open questions
1. Is `data.stellarator.powerht_constraint`/`powerscaling_constraint` being stale during
   `output=True` (never recomputed there) intentional or an oversight? Needs a domain
   read, not resolvable from this file.
2. Does `output=True`'s call sequence amount to a strict superset of `output=False`'s (plus
   the missing `power_at_ignition_point`), or does it genuinely diverge (e.g. `st_phys`
   called with a different bool)? Relevant to whether "one graph, two prunings" or "two
   graphs" is the right target shape — revisit once chunk 1B (`st_phys`) shows whether its
   `output` argument changes its own reads/writes too.
3. `first_call_stfwbs`'s actual behaviour — deferred to chunk 1E.
