---
kind: model-unit
status: draft
confidence: medium
---

**Partially ported (2026-08-26), see "## ported" below.** Written as part of wave-1's
reachability-first dispatch (`next_steps.md`'s tokamak wave); reachability was checked
against `functional_process/_audit/tokamak_call_surface.md` (the 38-file entered list)
and `process/core/caller.py` before any porting, per that dispatch's instructions. No
`unit_registry.md` row, no `next_steps.md` edit; registration is the consolidation pass's
job.

## reachability

**Reached, with a real (non-reporting) reader.** `Shield.run()` is entered at
`process/core/caller.py:329` (`models.shield.run()`), tokamak-only, level 8 of
`tokamak_call_surface.md` §A's call order; confirmed entered on
`large_tokamak_eval.IN.DAT` (`tokamak_call_surface.md` §B: `shield.py`, 270 of 482 lines
entered, 4 functions, "unported — no registry row").

`Shield`'s outputs:

| VarPath | read outside `shield.py`? |
|---|---|
| `.build.a_shld_inboard_surface` | only `models/stellarator/stellarator.py:580,588` — out of tokamak scope |
| `.build.a_shld_outboard_surface` | only `models/stellarator/stellarator.py:583,591` — out of tokamak scope |
| `.build.a_shld_total_surface` | only `models/stellarator/stellarator.py:577,581,584` — out of tokamak scope |
| `.blanket.vol_shld_inboard` | none found (`grep -rn "\.vol_shld_inboard\b" process/ --include=*.py`) |
| `.blanket.vol_shld_outboard` | none found |
| `.fwbs.vol_shld_total` | **yes** — `process/models/blankets/hcpb.py:310,371,503,1350` |
| `.blanket.dz_shld_half` | none outside `shield.py` itself (write-then-read within the same `run()` call, `local-intermediate`) |

`.fwbs.vol_shld_total` is read by `blankets/hcpb.py` (`CCFE_HCPB`, entered at
`caller.py:345`, tokamak-only *on this run* per `tokamak_call_surface.md` §A row 10,
selected by `.fwbs.i_blanket_type=1` per §"the reference run and what its switches
resolve to"):

- `hcpb.py:310` — `coolvol += self.data.fwbs.vol_shld_total * self.data.fwbs.vfshld`
  (shield coolant volume, real physics, not reporting).
- `hcpb.py:371-375` — `self.data.fwbs.whtshld = self.data.fwbs.vol_shld_total *
  self.data.fwbs.den_steel * (1.0 - self.data.fwbs.vfshld)` — the shield **mass**. Per
  `functional_process/_audit/tokamak_boundary.md:168-169`, `.fwbs.whtshld` in turn feeds
  `.costs.shield_cost` and `.buildings.sizing`.
- `hcpb.py:503` — `self.data.ccfe_hcpb.shield_density = self.data.fwbs.whtshld /
  self.data.fwbs.vol_shld_total` — also real physics.
- `hcpb.py:1350` — reporting (`po.ovarre`).

These four sites are all within `hcpb.py`'s own entered surface (956 of 1663 lines
entered per `tokamak_call_surface.md` §B), i.e. on the traced call path for
`large_tokamak_eval.IN.DAT`, not merely reachable in general. `hcpb.py`'s existing port
(unit #13, `nuclear_heating_blanket`/`magnets`/`shield`) does **not** cover this code —
those three nodes are a different part of the file (`blanket_neutronics()`'s
`.fwbs.blktmodel == 1` arm), deliberately unregistered and blocked on a different
synthesis (`unit_registry.md` row 13). The mass-calculation block that reads
`vol_shld_total` is unaudited and unported.

**Verdict: outcome (a), reached with a real reader.** `.fwbs.vol_shld_total` is the one
output worth a closure; the three `a_shld_*` fields are dead on the tokamak path and are
not ported (below).

## source

`process/models/shield.py` (483 lines, full file in scope). `Shield.__init__` and
`Shield.output`/`output_shld_areas_and_volumes` are structure/reporting, not audited
further (the latter has no `data` writes, unlike `plasma_geometry.md`'s `output()`).

**Six `def`s in audit scope:**

| # | function | lines | shape |
|---|---|---|---|
| 1 | `Shield.run` | 33-140 | stateful shell; `n_divertors` branch + `itart`/`i_fw_blkt_vv_shape` compound branch |
| 2 | `Shield.calculate_shield_half_height` | 142-197 | `@staticmethod`, pure, unifies both `n_divertors` branches in one signature |
| 3 | `Shield.calculate_dshaped_shield_volumes` | 199-267 | `@staticmethod`, pure |
| 4 | `Shield.calculate_dshaped_shield_areas` | 269-324 | `@staticmethod`, pure — **not ported**, no reader (see above) |
| 5 | `Shield.calculate_elliptical_shield_volumes` | 326-385 | `@staticmethod`, pure |
| 6 | `Shield.calculate_elliptical_shield_areas` | 387-435 | `@staticmethod`, pure — **not ported**, no reader |

## the extraction seam

Same shape as `plasma_geometry.md`: functions 2/3/5/6 are already `@staticmethod`s over
plain floats, zero `self.data` access, needing only `np.` -> `jnp.` (via the
`engineering/ivc_functions.py` helpers `eshellvol`/`dshellvol`, themselves pure and
zero-`self.data`). The seam is at `run()`'s boundary: every `data` read/write is in
`run()`, every formula is already outside it.

`run()`'s only non-shell arithmetic is the coverage-factor block (`:120-140`), which has
no PROCESS-function counterpart — inline code applying `fvolsi`/`fvolso` to both the
areas and the volumes and re-summing. This unit ports the volume half only.

## data footprint

Reference run: `large_tokamak_eval.IN.DAT` — `itart=0` (default,
`physics_variables.py:994`), `i_fw_blkt_vv_shape=2`/`ELLIPTICAL_SHAPED` (dataclass
default, `fwbs_variables.py:49`, unset in the IN.DAT — `input.py` range is `(1,2)` so `2`
is the type's own default, not a code default standing in for an unvalidated field),
`n_divertors=1` (resolved at `process/core/init.py:617` from `i_single_null=1`, before
any model in `_call_models_once` runs — **not** an `InputVariable` in its own right,
confirmed by `grep -n "n_divertors" process/core/input.py` returning nothing).

### `run()` — half-height (lines 35-46)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_plasma_xpoint_lower` | read | explicit-arg | |
| `.build.dz_xpoint_divertor` | read | explicit-arg | |
| `.divertor.dz_divertor` | read | explicit-arg | |
| `.divertor.n_divertors` | read | switch | *(live, `=1`)* — branches the formula, per `divertor.md`'s "a switch read to branch selects an occupant" precedent |
| `.build.z_plasma_xpoint_upper` | read | explicit-arg | *(live branch only, `n_divertors != 2`)* |
| `.build.dr_fw_plasma_gap_inboard`, `.dr_fw_plasma_gap_outboard`, `.dr_fw_inboard`, `.dr_fw_outboard`, `.dz_blkt_upper` | read | explicit-arg | *(live branch only)* |
| `.blanket.dz_shld_half` | **write** | explicit-arg | *(live)* sole tokamak producer; read back later in the same `run()` call (`local-intermediate` at the `run()` level, an ordinary graph edge across this unit's own two families) |

### `run()` — shape dispatch and coverage factors (lines 48-140)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | switch | *(live, `=0`)* half of the compound shape switch |
| `.fwbs.i_fw_blkt_vv_shape` | read | switch | *(live, `=2`/`ELLIPTICAL_SHAPED`)* other half |
| `.build.r_shld_inboard_inner`, `.r_shld_outboard_outer`, `.dr_shld_inboard`, `.dr_shld_outboard`, `.dz_shld_upper` | read | explicit-arg | elliptical-arm inputs *(live)* |
| `.physics.rmajor`, `.rminor`, `.triang` | read | explicit-arg | |
| `.build.a_shld_inboard_surface`, `.a_shld_outboard_surface`, `.a_shld_total_surface` | **write** | explicit-arg | **not ported** — no reader on the tokamak path outside `stellarator.py` (see reachability above) |
| `.blanket.vol_shld_inboard`, `.vol_shld_outboard` | **write** | explicit-arg | *(live)* raw value from `calculate_elliptical_shield_volumes`, then **overwritten** by the coverage-factor block two lines later — see the suspected defect below |
| `.fwbs.vol_shld_total` | **write, twice** | explicit-arg / **redundant-duplicate-write**-adjacent | *(live)* the tuple-unpack write (`:107`, from the staticmethod's own raw sum) is immediately discarded; the surviving write is the coverage-corrected `vol_shld_inboard + vol_shld_outboard` two lines later (`:138-140`). Not quite `redundant-duplicate-write` (`schema.md`'s definition: same value written twice) — these are two **different** values, the first thrown away. See "suspected defect" below |
| `.fwbs.fvolsi`, `.fvolso` | read | explicit-arg | coverage factors, applied to volumes (this unit's scope) and areas (not ported) |

## proposed signature(s)

```python
def calculate_shield_half_height_double_null(z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor) -> float
def calculate_shield_half_height_single_null(z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor, z_plasma_xpoint_upper, dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard, dr_fw_inboard, dr_fw_outboard, dz_blkt_upper) -> float
def calculate_elliptical_shield_volumes(r_shld_inboard_inner, r_shld_outboard_outer, rmajor, triang, dr_shld_inboard, rminor, dz_shld_half, dr_shld_outboard, dz_shld_upper) -> tuple[float, float, float]
def calculate_dshaped_shield_volumes(r_shld_inboard_inner, dr_shld_inboard, dr_fw_inboard, dr_fw_plasma_gap_inboard, rminor, dr_fw_plasma_gap_outboard, dr_fw_outboard, dr_blkt_inboard, dr_blkt_outboard, dz_shld_half, dr_shld_outboard, dz_shld_upper) -> tuple[float, float, float]
def apply_shield_volume_coverage_factors(vol_shld_inboard, vol_shld_outboard, fvolsi, fvolso) -> tuple[float, float, float]
def calculate_shield_volumes_elliptical(...) -> tuple[float, float, float]   # composes the two above
```

Verbatim, matching PROCESS's own names, except the two `run()`-only extractions
(`apply_shield_volume_coverage_factors`, new; `calculate_shield_volumes_elliptical`, a
composition).

## cottax nodes

| class | family | owns | reads |
|---|---|---|---|
| `SingleNullShieldHalfHeight` | `ShieldHalfHeight` | `.blanket.dz_shld_half` | `.build.z_plasma_xpoint_lower/upper`, `.build.dz_xpoint_divertor`, `.divertor.dz_divertor`, `.build.dr_fw_plasma_gap_inboard/outboard`, `.build.dr_fw_inboard/outboard`, `.build.dz_blkt_upper` |
| `DoubleNullShieldHalfHeight` | `ShieldHalfHeight` | `.blanket.dz_shld_half` | `.build.z_plasma_xpoint_lower`, `.build.dz_xpoint_divertor`, `.divertor.dz_divertor` |
| `EllipticalShieldVolumes` | `ShieldVolumes` | `.blanket.vol_shld_inboard/outboard`, `.fwbs.vol_shld_total` | `.build.r_shld_inboard_inner/outboard_outer`, `.physics.rmajor/triang/rminor`, `.build.dr_shld_inboard/outboard`, `.blanket.dz_shld_half`, `.build.dz_shld_upper`, `.fwbs.fvolsi/fvolso` |

`.build.a_shld_*` and the D-shaped arm's occupant are not instantiated (no reader / not
live respectively).

## tier signal

**Tier 1 for both in-scope functions.** No `scipy.optimize`, no internal loop, no call
into another `Model`'s method, no CoolProp. `run()` is a dispatch shell over pure
arithmetic; `output()` is genuinely reporting-only (no `data` writes at all, cleaner than
`plasma_geometry.md`'s `output()`).

**Sample provenance is synthetic, not legacy.** No `tests/unit/models/test_shield.py`
exists in `process/`'s own suite (checked: `find tests/unit -iname "*shield*"` returns
nothing) — this file has no PROCESS-side unit test to lift samples from at all, unlike
`plasma_geometry.md`'s `tests/unit/models/physics/test_plasma_geom.py`. Every sample here
is a physically-plausible ITER/DEMO-scale synthetic point (same provenance class as
`plasma_geometry.md`'s `sauter_geometry`/`plasma_poloidal_perimeter`), chosen to keep
`calculate_elliptical_shield_volumes`'s two radii (`r_2`, `r_3`) comfortably positive.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.divertor.n_divertors` | `0`, `1`, `2` (not an `InputVariable`; resolved from `i_single_null` at `init.py`) | `1` | **split** | `calculate_shield_half_height`'s `n_divertors == 2` branch drops five reads entirely (reads-sets differ); same treatment `divertor.md` gave `divwade`'s `n_divertors` branch |
| `.physics.itart` x `.fwbs.i_fw_blkt_vv_shape` | `itart∈{0,1}`, `i_fw_blkt_vv_shape∈{1,2}`, compound `itart==1 or shape==D_SHAPED` | `False` (`itart=0`, `shape=ELLIPTICAL_SHAPED`) | **split** | disjoint reads (`plasma_square`-style asymmetry doesn't apply here, but the two arms read entirely different `.build.*` field sets — compare `calculate_dshaped_shield_volumes`'s args to `calculate_elliptical_shield_volumes`'s). Same joint key `unit_registry.md` records `indat.py::_fw_blkt_vv_shape_arm` as already owning for `blanket_library.md`/`fw.md`/`vacuum.md` — **this file's split should join that key at consolidation, not mint an independent one** |

## calls into other models

None. `run()` calls only `self.<staticmethod>` plus `engineering/ivc_functions.py`'s
module-level `dshellarea`/`dshellvol`/`eshellarea`/`eshellvol` (imported at
`shield.py:9-14`) — pure helpers, not a `Model`.

## JAX-difficulty flags

- No fractional powers, no `min`/`max`, no CoolProp, no in-place mutation. The cleanest
  file audited in this wave — every operation is `+`, `-`, `*`, `/`, integer powers
  (`**2`, `**3`), and one integer-valued switch branch resolved at graph-assembly time.
- `_eshellvol`/`_dshellvol`'s `elong = b / a` terms divide by a shield-thickness-derived
  radius (`rmini`, `rmino`, or `rmajor - drin`); zero or negative values are physically
  invalid builds, not reachable through any switch, and not guarded in PROCESS either —
  same "caller's precondition to hold" situation as `plasma_geometry.md`'s D1, but with
  no confirmed live failure mode found in this pass (unlike D1, not measured/searched for
  a concrete counterexample; flagged as an open question below rather than a confirmed
  defect).

## suspected defects in PROCESS

**D1 — the raw `vol_shld_total` from `calculate_{elliptical,dshaped}_shield_volumes` is
computed then discarded. Confirmed by reading (same class as `plasma_geometry.md`'s
D10).** `Shield.run()` unpacks `(vol_shld_inboard, vol_shld_outboard, vol_shld_total) =
self.calculate_elliptical_shield_volumes(...)` (`:104-108`), then two lines of
coverage-factor scaling later reassigns `self.data.fwbs.vol_shld_total = vol_shld_inboard
+ vol_shld_outboard` (`:138-140`) using the just-rescaled inboard/outboard — the first
`vol_shld_total` value is never read anywhere before being overwritten. Not a value bug
(the surviving value is the physically intended one, coverage-adjusted), but the same
"one write is dead on arrival" shape as D10, and the reason
`apply_shield_volume_coverage_factors` recomputes the total from the *adjusted*
inboard/outboard rather than scaling the raw total by some combined factor — ported
faithfully, ordering preserved.

## open questions

1. **Is `elong = b/a`'s zero/negative-denominator domain reachable through any tracked
   IN.DAT?** Not checked in this pass (no `plasma_angles_arcs`-style measured
   counterexample found or sought) — flagged per the JAX-difficulty note above, not
   resolved.
2. **Which pass should wire the `itart` x `i_fw_blkt_vv_shape` joint key for this file's
   `ShieldVolumes` family?** `EllipticalShieldVolumes` is written and tested against the
   live arm, but its occupant family is not registered against
   `indat.py::_fw_blkt_vv_shape_arm` — that edit is out of this pass's scope per the
   wave's fencing rules (no `indat.py` edits). Whoever consolidates should add this
   file's arm to the existing joint switch rather than minting a fifth independent one.
3. **Should `Shield.calculate_dshaped_shield_areas`/`calculate_elliptical_shield_areas`
   ever be ported?** Currently dead on the tokamak path (only `stellarator.py` reads
   `.build.a_shld_*`). If a future unit gives them a tokamak-side reader, port then, not
   speculatively now — same "don't port dead code to inflate coverage" instruction this
   wave's brief states explicitly.

## ported (2026-08-26)

Port: `functional_process/models/shield.py`. Tests:
`tests/functional_process/models/test_shield.py`.

**Scope: the minimal closure that produces `.fwbs.vol_shld_total`** (the one output with
a real reader — see "reachability" above), plus `.blanket.vol_shld_inboard/outboard`
(produced by the same node) and `.blanket.dz_shld_half` (the upstream half-height, needed
for closure).

Functions ported:

| function | shape | wired to an occupant? |
|---|---|---|
| `calculate_shield_half_height_double_null(z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor)` | `n_divertors == 2` branch | yes — `DoubleNullShieldHalfHeight` (not live, but cheap: both values of this binary switch are fully supported) |
| `calculate_shield_half_height_single_null(...)` | `n_divertors != 2` branch | yes — `SingleNullShieldHalfHeight` (**live**) |
| `calculate_elliptical_shield_volumes(...)` | verbatim port of the staticmethod | via `calculate_shield_volumes_elliptical` → `EllipticalShieldVolumes` (**live**) |
| `calculate_dshaped_shield_volumes(...)` | verbatim port of the staticmethod, for completeness | no — not live, no occupant wired (matches `plasma_geometry.md`'s `sauter_geometry` precedent) |
| `apply_shield_volume_coverage_factors(...)` | new extraction of `run()`'s coverage-factor block (volume half) | via `calculate_shield_volumes_elliptical` |
| `calculate_shield_volumes_elliptical(...)` | new composition of the two above | yes — `EllipticalShieldVolumes` |

**Not ported, and why** (see the module docstring for the same list with more detail):
shield areas (`a_shld_*`, no reader on the tokamak path outside `stellarator.py`); the
D-shaped volume arm's occupant (not live, and shares a joint switch key another pass
already owns — see open question 2); `Shield.output()` (no `data` writes at all).

**Deviations from PROCESS: none.** Both ported branches and both ported staticmethods
are bit-for-bit translations (`np.` -> `jnp.`), including the D1 discard-then-overwrite
of the raw `vol_shld_total` (faithfully reproduced by `apply_shield_volume_coverage_
factors` recomputing the total from the coverage-adjusted inboard/outboard, exactly as
`run()` does, rather than "fixing" it to scale the raw total).

**Testing note on `calculate_shield_volumes_elliptical`'s `dz_shld_half` argument.**
Declared `static_argnames` in its test contract, not because it is a switch, but because
the real `Shield.run()` (this composite's only available oracle for the coverage-factor
wiring) always recomputes `dz_shld_half` internally and never accepts it as a boundary
input — so a `_run_shield`-style adapter cannot hold it fixed while independently
perturbing it for a finite-difference gradient check. `d(output)/d(dz_shld_half)` is
fully covered elsewhere: `calculate_elliptical_shield_volumes`'s own contract diffs
directly against the real `Shield.calculate_elliptical_shield_volumes` staticmethod with
`dz_shld_half` as an ordinary differentiable argument. See the test file's own docstring
for detail; flagged here since it is a real deviation from every other contract in this
unit, not because it hides a gap.
