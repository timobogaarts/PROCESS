---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `heating.py` / `test_heating.py`, five tier-1 contracts passing (legacy +
fuzz): `calculate_ecrh_heating`, `calculate_lowhyb_heating`,
`calculate_injected_power_total`, `calculate_beam_current`, `calculate_fusion_gain`. The
`isthtr == 3` (NBI) branch stays audit-only — see below.

## source
`process/models/stellarator/heating.py` (235 lines, full file in scope). Two module-level
functions: `st_heat` (computation, lines 13-140) and `output` (reporting, 143-236).

## data footprint

`st_heat` branches on `isthtr` (`data.stellarator.isthtr`, values 1/2/3), then runs a
common tail (lines 103-137) regardless of which branch ran.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.isthtr` | read | explicit-arg (switch) | selects the whole branch; see "switches touched" |
| `.current_drive.p_hcd_primary_extra_heat_mw` | read | explicit-arg | branches 1, 2, 3 |
| `.current_drive.eta_ecrh_injector_wall_plug` | read | explicit-arg | branch 1 only |
| `.current_drive.eta_lowhyb_injector_wall_plug` | read | explicit-arg | branch 2 only |
| `.current_drive.f_p_beam_orbit_loss` | read | explicit-arg | branch 3 only, out of port scope |
| `.current_drive.p_hcd_ecrh_injected_total_mw` | write | explicit-arg | branch 1 |
| `.current_drive.p_hcd_lowhyb_injected_total_mw` | write | explicit-arg | branch 2 |
| `.current_drive.p_hcd_beam_injected_total_mw` | write (branch 3), **read** (common tail, all branches) | implicit-io | branches 1/2 never write this field but the common tail reads it (`abs(...)>1e-8` guard, then again in the Q denominator) — see open question 1 |
| `.current_drive.p_beam_orbit_loss_mw` | write (branch 3), **read** (common tail, all branches) | implicit-io | same pattern as above, same open question |
| `.current_drive.p_hcd_injected_ions_mw` | write | explicit-arg | all three branches write it (0 for 1/2, computed for 3) |
| `.current_drive.p_hcd_injected_electrons_mw` | write | explicit-arg | all three branches |
| `.current_drive.eta_hcd_primary_injector_wall_plug` | write | explicit-arg | all three branches |
| `.current_drive.e_beam_kev` | read | explicit-arg | common tail |
| `.heat_transport.p_hcd_electric_total_mw` | write | explicit-arg | all three branches |
| `.current_drive.p_hcd_injected_total_mw` | write | explicit-arg | common tail |
| `.current_drive.c_beam_total` | write | explicit-arg | common tail |
| `.physics.p_plasma_ohmic_mw` | read | explicit-arg | common tail |
| `.physics.p_fusion_total_mw` | read | explicit-arg | common tail |
| `.current_drive.big_q_plasma` | write | explicit-arg | common tail |
| `.current_drive.f_p_beam_orbit_loss`, `f_p_beam_injected_ions` (via `culnbi()` return), `f_p_beam_shine_through` | read/write | **implicit-io-via-callee** | branch 3 only, out of port scope (calls `stellarator.current_drive.culnbi()`, a not-yet-audited model) |

No `redundant-duplicate-write` in this file.

## proposed signature(s)

**As ported** — branches 1 and 2 (self-contained), plus the common tail as three
separate functions since (per open question 1) their inputs from the *skipped* branch
differ in kind, not just value:

```python
def calculate_ecrh_heating(
    p_hcd_primary_extra_heat_mw: float, eta_ecrh_injector_wall_plug: float,
) -> tuple[float, float, float, float, float]:
    # (p_hcd_ecrh_injected_total_mw, p_hcd_injected_ions_mw, p_hcd_injected_electrons_mw,
    #  eta_hcd_primary_injector_wall_plug, p_hcd_electric_total_mw)
    ...

def calculate_lowhyb_heating(
    p_hcd_primary_extra_heat_mw: float, eta_lowhyb_injector_wall_plug: float,
) -> tuple[float, float, float, float, float]:
    # (p_hcd_lowhyb_injected_total_mw, p_hcd_injected_ions_mw, p_hcd_injected_electrons_mw,
    #  eta_hcd_primary_injector_wall_plug, p_hcd_electric_total_mw)
    ...

def calculate_injected_power_total(
    p_hcd_injected_electrons_mw: float, p_hcd_injected_ions_mw: float,
) -> float:  # p_hcd_injected_total_mw
    ...

def calculate_beam_current(
    p_hcd_beam_injected_total_mw: float, e_beam_kev: float,
) -> float:  # c_beam_total
    ...

def calculate_fusion_gain(
    p_fusion_total_mw: float, p_hcd_injected_total_mw: float,
    p_beam_orbit_loss_mw: float, p_plasma_ohmic_mw: float,
) -> float:  # big_q_plasma
    ...
```

`isthtr == 3` (NBI) is **not proposed** — blocked on auditing `CurrentDrive.culnbi()`
(not in current registry scope; `current_drive` is one of the sub-models noted in
`unit_registry.md`'s header as "reached only indirectly," never yet given its own row).

## cottax node

**Actually written**, in `heating.py` (`EcrhHeating`, `LowhybHeating`,
`InjectedPowerTotal`, `BeamCurrent`, `FusionGain`, all `ExplicitFunction`s).
`EcrhHeating`/`LowhybHeating` are **not both registered in `total_process.py` at
once** — see the data-footprint table, both write the same four downstream fields, so
only one belongs in an assembled graph, selected by `isthtr` at build time (same shape
as `i_tf_sup` in `stellarator_F_tf_nuclear_heating.md`). `BeamCurrent`/`FusionGain` are
**not yet registered** in `total_process.py` at all: `p_hcd_beam_injected_total_mw`/
`p_beam_orbit_loss_mw` have no producing node once NBI is unported (see open question 1)
— wiring them now would either dangle an unowned input or silently hardcode the "assume
zero" behaviour as a phantom producer, and this record isn't the place to decide which.

## tier signal

- `calculate_ecrh_heating`, `calculate_lowhyb_heating`: **tier 1** — no calls out, no
  internal solve.
- `calculate_injected_power_total`, `calculate_beam_current`, `calculate_fusion_gain`:
  **tier 1** — same, but see open question 1 for what their real upstream producer is
  when `isthtr` isn't 3.
- `isthtr == 3` branch: **not assessed** — calls `culnbi()`, out of scope until that
  model is audited.

## switches touched

- `isthtr` (`.stellarator.isthtr`) — **not previously in `switches.md`** (10-switch pilot
  scope was `stellarator.py` itself; `heating.py` is a separate file). **Split**: the
  three branches write disjoint sets of "total" fields
  (`p_hcd_ecrh_injected_total_mw`/`p_hcd_lowhyb_injected_total_mw`/
  `p_hcd_beam_injected_total_mw`+`p_beam_orbit_loss_mw`) and branch 3 alone calls out to
  another model — reads-sets clearly differ. Recommend adding as its own entry in
  `switches.md`; not added there directly by this record per the dispatch instruction to
  touch only this unit's own row.
- `i_plasma_ignited` (`.physics.i_plasma_ignited`) — read only in `output()` (line 154,
  selects a comment string), not in `st_heat`. Confirms `switches.md`'s existing entry
  lists this file as a site, but adds no new computational evidence — this file's only
  use is reporting-only, consistent with that entry's site (a) shape from
  `stellarator.py`, not a second data point for the split decision.

## calls into other models

- `isthtr == 3` branch calls `stellarator.current_drive.culnbi()` — out of scope (see
  above).
- `output()` calls nothing beyond formatting.

## JAX-difficulty flags

- `calculate_beam_current`'s `abs(...)>1e-8` branch and `calculate_fusion_gain`'s
  `abs(...)<1e-6` branch — `workaround-known` (`needs-lax-cond-or-where`), ported with
  `jnp.where`. Both guards are away from their sample points' fuzz ranges by construction
  (see `test_heating.py`), so no gradient-discontinuity risk was exercised, only the
  value-side branch.
- Genuine `float / float` division by zero (`eta_hcd_primary_injector_wall_plug == 0.0`,
  `e_beam_kev == 0.0`) is **not guarded** in the source (would raise
  `ZeroDivisionError` in real Python, not caught) — not fixing, not adding a guard the
  original doesn't have; a traced port would simply produce `inf`/`nan` instead of
  raising, consistent with the rewrite's general raise-becomes-nonfinite convention.
  `minor`, no evidence PROCESS input files ever hit this in practice.

## open questions

1. **What actually produces `p_hcd_beam_injected_total_mw`/`p_beam_orbit_loss_mw` when
   `isthtr` isn't 3?** The source relies on these fields already holding (or defaulting
   to) `0.0` on `data` — an implicit dependency on external zero-initialisation, not a
   value any branch of this file produces. Not resolvable from this file alone: needs
   either confirmation that `DataStructure`'s dataclass defaults are `0.0` for both
   (making the port's "caller passes 0.0" choice provably faithful) or a decision that
   this is itself a `conditional-ownership-by-run-config` case, symmetric to
   `stellarator_C_geometry.md`'s `.physics.aspect` finding but gated by `isthtr` instead
   of `ixc`.
2. Whether `EcrhHeating`/`LowhybHeating` sharing four output `VarPath`s (only
   discriminated by which "total" field is minted) is a pattern the graph-assembly
   tooling needs to handle generically (a switch-selected node set with partially
   overlapping, not just disjoint, output ports) — `i_tf_sup`'s SC/resistive split is
   simpler (resistive contributes nothing at all). Flagging for whoever designs the
   switch-to-graph-selection mechanism.
