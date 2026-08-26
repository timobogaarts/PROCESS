---
kind: model-unit
status: draft
confidence: medium
---

**Not a numbered registry unit.** This record documents a *shared helper* module ported
opportunistically because two wave-1 tokamak units (`.tokamak.first_wall`,
`.tokamak.vacuum_vessel`) both call into it — per the wave-1 dispatch's own convention
that a helper needed by two or more units belongs here, not duplicated privately into
each caller. Written by the same pass that ported `models/fw.py` and the `VacuumVessel`
extension to `models/vacuum/vacuum.py`; the orchestrator should fold it into whichever
registry entry (or its own) it decides fits.

## source

`process/models/engineering/ivc_functions.py` (partial — 2 of the file's several
functions). `eshellarea` (lines 99-130) and `eshellvol` (lines 170-247) only.

Not ported here: `dshellarea`/`dshellvol` (the D-shaped counterparts — no caller in
either of this pass's two units, since both reach only the elliptical arm on the
reference input), `calculate_pipe_bend_radius`, `pumping_powers_as_fractions` (neither
needed by the outputs `.tokamak.first_wall`/`.tokamak.vacuum_vessel` are scoped to
produce).

## data footprint

Both functions are already pure in `process/` — no `self.data` access at all, plain
module-level `def`s taking explicit float arguments. Nothing to classify.

| VarPath | read/write | classification | note |
|---|---|---|---|
| *(none)* | — | — | pure math, no `DataStructure` access in either function |

## proposed signature(s)

```python
def eshellarea(rshell, rmini, rmino, zminor) -> tuple[float, float, float]:  # ain, aout, atot
def eshellvol(rshell, rmini, rmino, zminor, drin, drout, dz) -> tuple[float, float, float]:  # vin, vout, vtot
```

Unchanged from source — no signature work needed, mechanical `np.` -> `jnp.` port.

## cottax node

None. Both are leaf pure functions called from inside another unit's node
(`FirstWall`'s elliptical-area occupant, `VacuumVessel`'s elliptical-volume occupant);
neither owns a `VarPath` of its own, so there is nothing here for `From`/`OutputInto` to
wrap. `total_process.py` wiring is the consolidation pass's job, per the wave-1 brief.

## tier signal

**Tier 1.** No iteration, no calls into any other model, no CoolProp, no `self.data`.

**Sample provenance.** No legacy unit test exists for either function directly
(`grep -rl eshellarea|eshellvol tests/unit/` is empty). The legacy sample used here is
borrowed from `tests/unit/models/test_vacuum.py::test_elliptical_vessel_volumes`, which
exercises `VacuumVessel.calculate_elliptical_vessel_volumes` — a thin wrapper around
`eshellvol` with one extra `r_1`/`r_2`/`r_3` derivation in front — so `eshellvol` itself
is exercised with that test's `(r_1, r_2, r_3, dz_vv_half, dr_vv_inboard, dr_vv_outboard,
(dz_vv_upper+dz_vv_lower)/2)` tuple as the legacy point. `eshellarea` has no equivalent
free oracle in `tests/unit/`; its legacy sample is a plausible geometry point (all
scale-consistent, no zero denominators) checked against `process`'s own `eshellarea`
called directly, not against a pre-existing hardcoded expectation.

## switches touched

None.

## calls into other models

None — leaf pure math.

## JAX-difficulty flags

- `eshellarea`/`eshellvol` divide by `rmini`/`rmino` (`elong = zminor / rmini`, etc.).
  Both are geometric half-widths, strictly positive on any physical input in the
  reference run; not flagged `safe_*` since this is an ordinary domain restriction, not
  a `0 < p < 1` power-law zero-derivative trap.

## open questions

- **Where should this record ultimately live?** It is not one of `unit_registry.md`'s
  numbered units. Flagging for the orchestrator to fold it into a numbered entry (most
  naturally alongside whichever of `.tokamak.first_wall`/`.tokamak.vacuum_vessel` is
  registered first) or to give it its own row — not decided here, per the wave-1 brief's
  "report, don't improvise a registry policy."
