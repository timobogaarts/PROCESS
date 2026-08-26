---
kind: model-unit
status: draft
confidence: high
---

**Scoping record only — nothing ported (2026-08-26).** Written as part of wave-1's
reachability-first dispatch (`next_steps.md`'s tokamak wave). No `unit_registry.md` row,
no `next_steps.md` edit; this record is the deliverable per the wave brief's outcome (b):
"reached but reporting-only ... write the audit record with the evidence ... and port
nothing".

## reachability

**Reached, but every output is a dead end — confirmed unreachable-beyond-itself, no
action needed**, the same verdict `unit_registry.md` row 16 records for `VacuumVessel` on
the stellarator pipeline, applied here to a whole file rather than one class.

`WaterUse.run()` is entered at `process/core/caller.py:385` (`models.water_use.run()`),
tokamak-only, level 19 of `tokamak_call_surface.md` §A's call order; confirmed entered on
`large_tokamak_eval.IN.DAT` (`tokamak_call_surface.md` §B: `water_use.py`, 265 of 323
lines entered, 7 functions, "unported — no registry row"). `main.py:965` also calls
`self.water_use.output()` (the reporting pass, `run(output=True)`).

`WaterUse`'s outputs — every field `self.data.water_use.*` this file writes:

```
evapratio, volheat, energypervol, volperenergy, evapvol, waterusetower,
wateruserecirc, wateruseonethru
```

```
grep -rn "\.water_use\." process/ --include=*.py | grep -v "process/models/water_use.py" | grep -v "process/data_structure"
```

returns exactly two lines, both call sites (`caller.py:385`, `main.py:965`) — **zero**
reads of any `.water_use.*` field anywhere else in `process/`: not in
`core/solver/constraints.py`, not in `core/solver/objectives.py`, not in `costs/*.py`,
not in any other model. Independently corroborated by
`functional_process/_audit/tokamak_boundary.md:151`, which lists `water_use` among the
fourteen slots attributed **zero** boundary reads in the currently-assembled graph.

**Verdict: outcome (b), reached but reporting-only.** `WaterUse.run()` computes real
physics (evaporation ratios, cooling-tower and cooling-water-body water withdrawal from
`p_plant_primary_heat_mw`/`eta_turbine`), and it is not dead code in the sense of never
executing — but nothing downstream of it, in this codebase, ever reads what it computes.
It is a terminal report, structurally: PROCESS computes and prints "how much water this
plant would use" without feeding that number into any cost, constraint, or objective.

## source

`process/models/water_use.py` (324 lines, full file). Three methods in scope:
`WaterUse.run` (unconditional caller of the two below), `WaterUse.cooling_towers`,
`WaterUse.cooling_water_body` — plus the module-level `CoolingWaterBodyCoeffs` dataclass
(three fixed-coefficient instances, no `self.data` access, pure per-instance methods).
`WaterUse.output` is a one-line `run(output=True)` alias, not separately audited.

No `i_*`/model-selection switch anywhere in the file — the only branching is a Python
`for icool in icools:` loop over three literal coefficient sets (Brady/Webster/Gulliver),
averaged at the end (`evapsum /= len(icools)`), not a device- or config-dependent choice.

## tier signal

Not assigned — no port to validate. Would be **Tier 1** if ever ported (no internal
solver, no CoolProp, no call into another `Model`).

## open questions

None. This file needs no further attention unless a future unit gives one of its outputs
a reader — per the wave brief, "do not port dead code to inflate coverage."
