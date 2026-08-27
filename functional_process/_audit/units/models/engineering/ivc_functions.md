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

## 2026-08-27 — `dshellarea`/`dshellvol` ported (D-shaped wave)

The open question this record's own module docstring left open — *"add them here, not
privately, the day a D-shaped occupant is written"* — was answered by
`tests/regression/input_files/spherical_tokamak_eval.IN.DAT` and
`st_regression.IN.DAT`, which both set `i_fw_blkt_vv_shape = 1` (`D_SHAPED`) **and**
`itart = 1`. The predicate `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED` is therefore
doubly true on both files, and five slots needed the D-shaped shell formulas at once.

**What was added.** `dshellarea` (ports `process/models/engineering/ivc_functions.py:
133-167`) and `dshellvol` (`:249-306`), both verbatim `np.` → `jnp.`, both beside their
elliptical siblings. The file now holds all four of PROCESS's shell helpers.

**The two pairs are not one pair with a parameter.** A D-shaped shell's inboard section
is a *cylinder*: `ain = 4 * zminor * pi * rmajor` and `vin = 2 * (zminor + dz) * pi *
(rmajor**2 - (rmajor - drin)**2)`, both closed-form, where the elliptical arm's inboard
half is the difference of two ellipse revolutions about a shared `rshell`. Only the
outboard halves correspond, and even those differ in what the ellipse is centred on
(`rmajor`, the outer edge of the cylinder, versus `rshell`). No substitution turns either
pair into the other, which is why this is four functions rather than two.

**Filing consolidation.** `models/shield.py` had carried private `_eshellvol`/`_dshellvol`
copies since wave 1, with an explicit note in their docstrings that the consolidation pass
should lift them here. It did: `shield.py` now imports both from this file and the private
copies are gone. `models/blankets/blanket_library.py`'s `_eshellarea`/`_eshellvol` were
**left alone** — the same note applies to them, but moving two files' filing in one wave
would have mixed the D-shaped work with an unrelated cleanup. That remains open.

**Tests.** `TestDshellarea` and `TestDshellvol` in
`tests/functional_process/models/engineering/test_ivc_functions.py`, both Tier 1, both
diffed directly against `process`'s own callables (already pure, no adapter). Neither has
a legacy sample: `grep -rl 'dshellarea|dshellvol' tests/unit` is empty and no unit test of
a caller carries a reducible intermediate the way `test_vacuum.py` does for `eshellvol`.
The fuzz boxes are the whole oracle, which suffices for a closed-form expression checked
for gradients as well as values. `dshellvol`'s `drin` is capped below the smallest
`rmajor` in its box: the inboard term goes negative past `drin > rmajor` and PROCESS has
no guard, so both sides would agree on the nonsense — the bound keeps draws physical, it
does not hide a disagreement.

Green at `--fp-gradients`. No new boundary input. The § open questions entry about where
this record should ultimately live is unchanged.
