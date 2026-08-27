---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial — minimal closure, live switch cell only).** `fw.py` / `test_fw.py`:
one occupant, `FirstWall`, composed from five tier-1 pure functions, producing all three
of `tokamak_boundary.md`'s `.tokamak.first_wall` reads (`.first_wall.a_fw_total`,
`.physics.p_fw_alpha_mw`, `.physics.pflux_fw_neutron_mw`) plus `a_fw_inboard`/
`a_fw_outboard` (siblings of `a_fw_total`, read by other already-ported/pending units —
see § scope discipline).

## source

`process/models/fw.py`, 867 lines, 6 entered functions. In scope: `FirstWall.run()`
(44-149, the stateful shell — only its live-configuration lines), `calculate_first_
wall_half_height` (151-199, `@staticmethod`), `calculate_elliptical_first_wall_areas`
(232-284, `@staticmethod`). Out of scope: `calculate_dshaped_first_wall_areas`
(202-230, `@staticmethod`, D-shaped arm, UNPORTED), `apply_first_wall_coverage_factors`
(286-345, `@staticmethod` — ported *partially*, single-null arm only), `set_fw_
geometry` (347-352, writes `.build.dr_fw_{in,out}board` — not on this slot's boundary),
`fw_temp` (354-677, thermal-hydraulics + CoolProp, dormant on this run — see module
docstring), `calculate_total_fw_channels` (679-707, not on this slot's boundary),
`output_fw_*` (709-867, pure reporting).

## data footprint

`FirstWall.run()`'s live-configuration reads/writes only (full table in
`process/models/fw.py:46-149`; everything below already excludes the D-shaped arm, the
`n_divertors == 2` arm, `set_fw_geometry`, `calculate_pipe_bend_radius`, and
`calculate_total_fw_channels`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_plasma_xpoint_lower` | read | explicit-arg | `:47` |
| `.build.dz_xpoint_divertor` | read | explicit-arg | `:48` |
| `.divertor.dz_divertor` | read | explicit-arg | `:49` |
| `.build.dz_blkt_upper` | read | explicit-arg | `:50` |
| `.build.z_plasma_xpoint_upper` | read | explicit-arg | `:51` — only reached on the `n_divertors != 2` (single-null) arm |
| `.build.dz_fw_plasma_gap` | read | explicit-arg | `:52` — same arm |
| `.divertor.n_divertors` | read | switch (branch) | `:53` — see § switches touched; not a parameter of the port |
| `.build.dr_fw_inboard` | read | explicit-arg | `:54` |
| `.build.dr_fw_outboard` | read | explicit-arg | `:55` |
| `.fwbs.dz_fw_half` | write | explicit-arg | `:46`, then read back at `:69,83` (`local-intermediate`, same straight-line call) — kept as a local in the port, not written to `data` (nothing outside `run()` reads it before `set_fw_geometry`/`fw_temp`, both out of scope) |
| `.physics.itart` | read | switch | `:59` — see § switches touched |
| `.fwbs.i_fw_blkt_vv_shape` | read | switch | `:60` — see § switches touched |
| `.physics.rmajor` | read | explicit-arg | `:80` |
| `.physics.rminor` | read | explicit-arg | `:80` |
| `.physics.triang` | read | explicit-arg | `:82` — elliptical arm only |
| `.build.dr_fw_plasma_gap_inboard` | read | explicit-arg | `:70,84` |
| `.build.dr_fw_plasma_gap_outboard` | read | explicit-arg | `:71,85` |
| `.first_wall.a_fw_inboard_full_coverage` | write | local-intermediate | `:76`, read back at `:96` — kept a local in the port (not written to `data`; nothing outside `apply_first_wall_coverage_factors` in `run()` reads it — `hcpb.py:1649` reads the sibling `a_fw_total_full_coverage` but only inside `output()`, a reporting-only read, see § scope discipline) |
| `.first_wall.a_fw_outboard_full_coverage` | write | local-intermediate | same |
| `.fwbs.f_ster_div_single` | read | explicit-arg | `:94` — produced by `.tokamak.divertor`'s `DivertorHeatFluxSplit` (this pass's other unit) |
| `.fwbs.f_a_fw_outboard_hcd` | read | explicit-arg | `:95` |
| `.first_wall.a_fw_inboard` | write | explicit-arg | `:88-91` |
| `.first_wall.a_fw_outboard` | write | explicit-arg | same |
| `.first_wall.a_fw_total` | write | explicit-arg | same |
| `.physics.p_alpha_total_mw` | read | explicit-arg | `:147` |
| `.physics.f_p_alpha_plasma_deposited` | read | explicit-arg | `:148` |
| `.physics.p_fw_alpha_mw` | write | explicit-arg | `:146-149` |
| `.physics.i_pflux_fw_neutron` | read | switch | `:121,131` — see § switches touched |
| `.physics.ffwal` | read | explicit-arg | `:123` — `i_pflux_fw_neutron == 1` arm only |
| `.physics.pflux_plasma_surface_neutron_avg_mw` | read | explicit-arg | `:124` — same arm |
| `.physics.pflux_fw_neutron_mw` | write | explicit-arg | `:121-129` |

## scope discipline

Three things this file computes are dropped, each for a different reason:

- **`fw_temp` and everything it feeds** (thermal-hydraulic peak temperature, coolant
  mass flow, channel-bend radii) — dormant on this run (`i_p_coolant_pumping = 3`, not
  `2`) and produces nothing on `.tokamak.first_wall`'s declared boundary. The only
  CoolProp obstacle in this file lives entirely inside it.
- **`a_fw_{in,out}board_full_coverage`** — computed by `calculate_elliptical_first_
  wall_areas`, consumed only by `apply_first_wall_coverage_factors` within the same
  `run()` call. `a_fw_total_full_coverage` (their sibling, also dropped) is read once,
  by `blankets/hcpb.py:1649-1650`, but only inside `hcpb.py`'s own `output()` — a pure
  reporting read, not a computational dependency — so kept as local intermediates
  rather than `data` writes, matching the precedent `plasma_geometry.md` set for
  `a_plasma_surface_outboard`'s sibling locals.
- **`set_fw_geometry`, `calculate_total_fw_channels`** — write `.build.dr_fw_{in,out}
  board` and `.blanket.n_fw_{in,out}board_channels` respectively; neither is on
  `.tokamak.first_wall`'s declared boundary.
- **`.physics.pflux_fw_rad_mw`/`.constraints.pflux_fw_rad_max_mw`** (`fw.py:131-144`) —
  the `i_pflux_fw_neutron`-gated sibling of `pflux_fw_neutron_mw`, computed
  unconditionally right after it in `run()`. Reads `.physics.a_plasma_surface`
  (divides by it — the test adapter must give it a nonzero value or `run()` crashes on
  an unrelated line before the target outputs are read back, confirmed by running the
  composite test with the `DataStructure` default of `0.0`). Neither field is on
  `.tokamak.first_wall`'s declared boundary; not ported.

## proposed signature(s)

```python
def calculate_first_wall_half_height(
    z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor, dz_blkt_upper,
    z_plasma_xpoint_upper, dz_fw_plasma_gap, dr_fw_inboard, dr_fw_outboard,
) -> float:  # dz_fw_half, n_divertors == 1 baked in

def calculate_elliptical_first_wall_areas(
    rmajor, rminor, triang, dz_fw_half, dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard,
) -> tuple[float, float, float]:  # a_fw_{in,out}board_full_coverage, a_fw_total_full_coverage

def apply_first_wall_coverage_factors(
    f_ster_div_single, f_a_fw_outboard_hcd, a_fw_inboard_full_coverage, a_fw_outboard_full_coverage,
) -> tuple[float, float, float]:  # a_fw_inboard, a_fw_outboard, a_fw_total, n_divertors == 1 baked in

def calculate_p_fw_alpha_mw(p_alpha_total_mw, f_p_alpha_plasma_deposited) -> float:

def calculate_pflux_fw_neutron_mw_ffwal(
    ffwal, pflux_plasma_surface_neutron_avg_mw,
) -> float:  # i_pflux_fw_neutron == 1 baked in
```

## cottax node

`FirstWall(ExplicitFunction)`, in `functional_process/models/fw.py`. Composes all five
functions above; owns `a_fw_inboard`, `a_fw_outboard`, `a_fw_total`, `p_fw_alpha_mw`,
`pflux_fw_neutron_mw`.

## tier signal

**Tier 1**, all five functions. No iteration, no calls into another model's method
(`eshellarea` is a shared leaf helper, not a `Model`), no CoolProp on this path.

**Sample provenance.** No legacy unit test covers any of these five functions directly
— `tests/unit/models/test_fw.py` exercises only `fw_temp` (out of scope). All samples
are fuzz-only, drawn from physically plausible ranges (see `test_fw.py`).

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.divertor.n_divertors` | `1`, `2` | `1` — derived by `process/core/init.py:606-616` from `.physics.i_single_null = 1` (`large_tokamak_eval.IN.DAT:307`), not the field's own default of `2` | **split** (baked, no parameter) | `calculate_first_wall_half_height`'s `if n_divertors == 2: z_top = z_bottom else: z_top = z_plasma_xpoint_upper + dz_fw_plasma_gap` (`fw.py:194-197`) and `apply_first_wall_coverage_factors`'s `if n_divertors == 2: ... else: ...` (`:320-333`) both read genuinely different fields per arm. Same switch, same live value, as `divertor.md`/`structure.md` |
| `.physics.itart` / `.fwbs.i_fw_blkt_vv_shape` (compound) | `itart ∈ {0,1}`, `i_fw_blkt_vv_shape ∈ {D_SHAPED=1, ELLIPTICAL_SHAPED=2}` | `itart = 0` (default), `i_fw_blkt_vv_shape = 2` (default) — condition `itart == 1 or shape == D_SHAPED` is `False` → elliptical arm | **split** (baked, no parameter) | `fw.py:58-86`: D-shaped arm reads no `triang`; elliptical arm does (`plasma_geometry.md`'s "compound Sauter switch" precedent — same disjunctive shape, same resolution: one predicate evaluated once at assembly time). D-shaped arm UNPORTED |
| `.physics.i_pflux_fw_neutron` | `0`, `1` | `1` (PROCESS default, `physics_variables.py:1006`, not set in the reference file) | **split** (baked, no parameter) | `fw.py:121-129`: `== 1` reads `ffwal`, `pflux_plasma_surface_neutron_avg_mw`; `== 0` reads `p_neutron_total_mw`, `.first_wall.a_fw_total` instead (a self-read of this same slot's own output — would be Shape B, `.tokamak.first_wall` reading what it owns, if ever ported; flagging for whoever ports that arm) |

## calls into other models

None on this path. `eshellarea` (`functional_process.models.engineering.
ivc_functions`) is a shared leaf helper, not a `Model`.

## JAX-difficulty flags

None beyond `ivc_functions.md`'s own (division by strictly-positive shell half-widths).
No fractional powers, no CoolProp on this path, no in-place mutation.

## deviations from PROCESS

- **`apply_first_wall_coverage_factors`'s `ProcessValueError` on a non-credible
  outboard area** (`fw.py:337-343`, `a_fw_outboard <= 0.0`) is dropped. A traced
  function cannot raise on a data-dependent condition
  (`_audit/naming_convention.md`'s domain-guard convention is for `nan`-producing
  operations; this one is an ordinary sign check on an otherwise-finite value, so there
  is no non-finite result to assert against either — the port simply returns the value,
  physical or not). Not exercised on the reference operating point.

## open questions

- **`i_pflux_fw_neutron == 0`'s self-read.** That arm reads `.first_wall.a_fw_total`,
  which this same slot's `FirstWall` node owns — a genuine Shape B (`FixedPointFunction`
  candidate) if it is ever ported, not resolved here since it is UNPORTED.
- **Registration.** `FirstWall` is proposed as the sole occupant of `.tokamak.first_wall`
  — no alternative PROCESS `Model` competes for this slot, unlike `.tokamak.divertor`'s
  three-way `i_div_heat_load` dispatch.

## 2026-08-27 — `set_fw_geometry` ported (cold-boundary wave)

`cold_boundary.md` producer 1. The § scope discipline entry above dropping
`set_fw_geometry` ("not on this slot's boundary") was correct for the slot and wrong
for the graph: `.build.dr_fw_inboard`/`.build.dr_fw_outboard` were two of the six cold
boundary zeros, behind 7 of the cold tokamak MDA's 11 non-finite roots (both hcpb
coolant void fractions divide by `dr_fw_inboard`, and `nuclear_heating_magnets`'
`vffwm` does too). Ported as `set_fw_geometry` + `FirstWallGeometry`, a **second node**
occupying a new `Tokamak` slot `first_wall_geometry` — not two more outputs of
`FirstWall`, because `FirstWall` *reads* both fields for the half-height and a node
must not read what it owns.

That read is also a measured PROCESS quirk this split preserves without reproducing:
`run()` computes the half-height from the **entering** `dr_fw_*` (`fw.py:46-56`) and
only then calls `set_fw_geometry` (`:110`), so PROCESS's very first cold pass sees
`0.0` where every later pass — and this graph, which reads the produced value — sees
`2*radius_fw_channel + 2*dr_fw_wall = 0.018`. Both fields are constants of two run
inputs, so the fixed point lands after one PROCESS pass and the port's fresh read *is*
the converged value; no cycle exists or is created (`Blocking.scc` on both reference
machines, measured this wave).

Data footprint delta: reads `.fwbs.radius_fw_channel`, `.fwbs.dr_fw_wall` (both
explicit-arg, both run inputs / dataclass defaults on the reference file); writes
`.build.dr_fw_inboard`, `.build.dr_fw_outboard` (whole writeback — the outboard is the
inboard's value assigned, not recomputed, reproduced by returning the same
intermediate twice). Tier 1, no switch; `test_fw.py::TestSetFwGeometry` diffs the real
instance method through the `data` back-door (legacy point = the two defaults, fuzz
over plausible channel geometry).


## 2026-08-27 — the `n_divertors == 2` arm ported (double-null wave)

`.tokamak.first_wall` is a family of two now: `FirstWall` is the abstract base,
`FirstWallSingleNull` the occupant this slot already had (unchanged in body), and
`FirstWallDoubleNull` the new one. `indat.py`'s `FIRST_WALL` registry gains arm `1` and
the `('first_wall_arm', -1)` refusal is gone.

Two of this file's functions branch on `n_divertors`, and both arms are now written:

- **Half-height (`fw.py:186-197`).** `z_top = z_bottom`, so the result is `z_bottom`
  itself — written reduced, exactly as in `shield.py` and `blankets/blanket_library.py`,
  and exact in floating point rather than only algebraically. **Two fewer reads**:
  `z_plasma_xpoint_upper` and `dz_fw_plasma_gap` are the whole of the `else` arm's
  `z_top`.
- **Coverage factors (`fw.py:320-327`).** Both the inboard and the outboard area lose
  `2 * f_ster_div_single`. **This arm is symmetric** — both assignments sit inside the
  `if` — unlike `blanket_library.py`'s coverage factors, where only the area assignment
  was doubled and the volume was left on the single-divertor factor. Recorded here as
  well as there, because the two look like the same edit and only one of them was made
  completely. PROCESS's `ProcessValueError` guard is dropped on this arm for the same
  reason as on the single-null one (§ deviations).

The composite `calculate_first_wall_outputs_double_null` mirrors
`calculate_first_wall_outputs` with those two parameters absent — the whole reason it is a
second composite rather than one with a traced `n_divertors`.

**`_first_wall_arm` was reordered.** It now asks the shape and `i_pflux_fw_neutron`
first and the divertor count last, because the divertor count is the only one of the
three this port can answer either way; the two refusals are what a rejected file needs
told. Practical effect on the two spherical-tokamak files: they now stop at
`('fw_blkt_vv_shape_arm', 0)` (D-shaped) rather than at the double-null refusal.

**Tests.** `TestCalculateFirstWallHalfHeightDoubleNull`,
`TestApplyFirstWallCoverageFactorsDoubleNull` and
`TestCalculateFirstWallOutputsDoubleNull`, all Tier 1. Both double-null adapters
**poison** `z_plasma_xpoint_upper` and `dz_fw_plasma_gap` with `nan` rather than zeroing
them — the composite can do it too, because `process/models/fw.py` reads both fields in
exactly one place (`:51-52`, the arguments of the half-height call), so a `nan` in
`.build` reaches nothing else in `run()`. Green at `--fp-gradients --fp-fuzz 40`.

**A sample-domain note worth carrying forward.** The single-null contracts' fuzz box has
`z_plasma_xpoint_lower` in `(-10, -1)`, which is not the field's sign:
`process/models/build.py:170-172` assigns it `rminor * kappa`, a magnitude. The
single-null arm survives the wrong sign because it averages `z_bottom` against a positive
`z_top`; the double-null arm *is* `z_bottom`, so a negative draw drives `eshellarea` to
negative areas and PROCESS into its `ProcessValueError`. The new contracts therefore use
`(3.0, 7.0)` and a positive legacy point.

**Pre-existing, unrelated, not touched by this wave.**
`TestCalculateFirstWallOutputs` (single null) fails 36 assertions at `--fp-fuzz 40` —
the negative-half-height draws above, reaching PROCESS's `ProcessValueError` and the
port's `nan`. Verified identical on a clean tree with this wave's changes stashed, so it
is this record's finding rather than this wave's regression. The fix is the fuzz box's
sign, and it belongs to whoever next owns this contract.

No new boundary input.

## 2026-08-27 — the D-shaped double-null cell ported (D-shaped wave)

**This slot's occupant grid is a shape × divertor-count product**, and `.tokamak.first_wall`
is one of only two slots in the port where that is true. `FirstWall.run()`
(`process/models/fw.py:44-149`) branches on `n_divertors` twice — the half-height at
`:46-56` and the coverage factors at `:103-109` — with the shape branch at `:58-86`
sitting between them. The two predicates are independent of each other. But this port
keeps the whole of `run()` as **one** composite node rather than three, so the grid does
not factor:

| | single null | double null |
|---|---|---|
| **elliptical** | `FirstWallSingleNull` | `FirstWallDoubleNull` |
| **D-shaped** | UNPORTED, `('first_wall_arm', -2)` | `FirstWallDShapedDoubleNull` |

`models/vacuum/vacuum.py` pays the same product for the same reason.
`blankets/blanket_library.py` and `models/shield.py` do **not**, because wave 1 had already
split their `run()`s into one slot per branch — so the difference is in the decomposition,
not in PROCESS. Worth carrying forward: *whether a switch pair is a product is decided by
how finely the `run()` was cut, not by the source's branch nesting.*

**What was added.** `calculate_dshaped_first_wall_areas` (ports `fw.py:201-230`, verbatim,
calling the new shared `ivc_functions.dshellarea`), the composite
`calculate_first_wall_outputs_dshaped_double_null`, and the occupant
`FirstWallDShapedDoubleNull`. `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`
select it: `i_single_null = 0`, `itart = 1`, `i_fw_blkt_vv_shape = 1`.

**Reads-set difference.** The D-shaped area formula reads **no `triang`** — its inboard
section is a cylinder at `rmajor - rminor - dr_fw_plasma_gap_inboard`, a radius
triangularity does not enter, where the elliptical arm centres both ellipses at
`rmajor - rminor * triang`. So the new composite has sixteen parameters against the
elliptical single-null composite's nineteen: `triang` on top of the double-null arm's
`z_plasma_xpoint_upper` and `dz_fw_plasma_gap`.

**Why the fourth cell is refused rather than written.** Every ingredient of D-shaped
single null exists and is harness-tested — the D-shaped area function, the single-null
half-height and the single-null coverage factors — so writing it would be one composite
and one class with no new arithmetic. It is refused because no input file in this
repository selects it, which is this wave's reachability-first discipline, and
`('first_wall_arm', -2)` now says exactly that instead of "the D-shaped first wall". A
later wave with a D-shaped single-null file should expect a very short port.

**Refusal ordering.** `_first_wall_arm` asks `i_pflux_fw_neutron` first (it can only ever
be answered one way here), then the one unwritten grid cell, then maps the three written
cells — the "ask the answerable condition last" convention the double-null wave
established, extended to a second unanswerable condition.

**Tests.** `TestCalculateDshapedFirstWallAreas` (bare PROCESS staticmethod, no adapter —
and no `nan` poisoning possible, because PROCESS's own D-shaped signature simply has no
`triang` parameter, which is the stronger form of the claim) and
`TestCalculateFirstWallOutputsDshapedDoubleNull` (real `FirstWall.run()`, adapter poisons
a *third* field, `.physics.triang`, read at `fw.py:82` only). The composite's box is the
spherical-tokamak operating point — `rmajor` near `1.8 × rminor`, both files' `aspect` —
rather than the large-tokamak one the other two composites use, because the D-shaped
inboard radius `rmajor - rminor - gap` goes small in the ST regime and a
conventional-aspect box would never visit it. Green at `--fp-gradients`.

No new boundary input.
