---
kind: model-unit
status: draft
confidence: medium-high
---

## source
`process/models/stellarator/stellarator.py`, lines 881-1280: middle third of `st_fwbs()`
(481-1682). Chunk 1E2 of unit #1 (see `../../_audit/unit_registry.md`). Even-thirds line
cut, not a semantic boundary.

**Boundary note (881, shared with 1E1)**: lines 881-1029 continue a deeply-indented block
(4 levels) whose governing `if`/`for` conditions are established before line 881 — this
chunk cannot be read as self-contained at the top; it inherits an outer branch context
from 1E1's territory. Whoever reviews 1E1+1E2 together needs to confirm what that outer
condition actually is (not re-derived here, out of directive scope).

**Boundary note (1280, shared with 1E3)**: line 1280 ends mid-comment
("External cryostat outboard major radius (m)"), computation continues cleanly into 1E3
starting the next statement — a clean-enough cut, no variable defined in 1E2 is left
dangling, but the cryostat/vacuum-vessel geometry block that starts right after is one
continuous sequence split across 1E2/1E3 arbitrarily.

## data footprint

Grouped by sub-block (writes as primary rows; reads consolidated per block rather than
one row each — ~75 distinct reads in this range, mostly single-use `explicit-arg`
material/geometry constants from `.fwbs`/`.build`/`.divertor`/`.first_wall`).

| VarPath (write) | classification | reads consumed | note |
|---|---|---|---|
| `.heat_transport.p_fw_coolant_pump_mw` | explicit-arg | `.heat_transport.f_p_fw_coolant_pump_total_heat`, block-local `p_fw_*_nuclear_heat_mw`, `psurffwi/o`, `.current_drive.p_beam_orbit_loss_mw` | only written under `i_p_coolant_pumping == FRACTION_OF_HEAT`; `USER_INPUT` branch writes nothing (keeps whatever `data` already holds) — confirms `switches.md`'s existing split decision for `i_p_coolant_pumping` |
| `.heat_transport.p_blkt_coolant_pump_mw` | explicit-arg | same switch, `.fwbs.f_p_blkt_multiplication`, block-local `pnucbzi/o` | same conditionality as above |
| `.fwbs.p_blkt_multiplication_mw` | explicit-arg, then **incremented again** later in this same chunk (line 948, `+=`) | `.fwbs.f_p_blkt_multiplication`, block-local `pnucbzi/o` | two separate contributions summed into one field across ~18 lines — not a `redundant-duplicate-write` (values differ, both intentional), but worth noting as a two-part accumulator so the ported function doesn't accidentally treat the first assignment as final |
| `.fwbs.p_fw_nuclear_heat_total_mw` | explicit-arg | block-local `p_fw_inboard/outboard_nuclear_heat_mw` | |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | explicit-arg | block-local `pnucbzi/o`, `.fwbs.f_p_blkt_multiplication` | |
| `.fwbs.p_shld_nuclear_heat_mw` | explicit-arg | block-local `pnucshldi/o` | |
| `.heat_transport.p_shld_coolant_pump_mw`, `.heat_transport.p_div_coolant_pump_mw` | explicit-arg | `.heat_transport.f_p_shld/div_coolant_pump_total_heat`, `.physics.p_plasma_separatrix_mw`, `.fwbs.p_div_nuclear_heat_total_mw`, `.fwbs.p_div_rad_total_mw` | again only under `i_p_coolant_pumping == FRACTION_OF_HEAT` |
| `.fwbs.p_tf_nuclear_heat_mw` | explicit-arg (switch) | `.tfcoil.i_tf_sup`, block-local `pnucsi/o`, `pnucshldi/o` | SUPERCONDUCTING branch reads 4 quantities; resistive branch reads nothing, writes constant `0.0`. **Independent third confirmation of `switches.md`'s `i_tf_sup` → split decision** (site is `stellarator.py:1022`, exactly matching the pilot's own site enumeration — cross-checked, not re-derived) |
| `.divertor.a_div_surface_total` | **implicit-io — cross-call, not intra-call (see open question 1)** | `self.first_call_stfwbs` (object attribute, not a `data` field) | **only written on the first call** (`50.0` hardcoded fallback); source comment explains why: the real value is computed by a divertor routine ("stdiv") *later* in the pipeline than `st_fwbs` runs, so on the first pass it doesn't exist yet. This is evidence of a genuine pipeline-order dependency / near-circularity between `st_fwbs` and the divertor model, not just a local implicit-io case — see open question 1 |
| `.divertor.m_div_plate` | explicit-arg | `.divertor.a_div_surface_total` (see above), `.divertor.den_div_structure/f_vol_div_coolant/dx_div_plate` | downstream consumer of the cross-call-dependent field above |
| `.fwbs.wtbllipb`, `.fwbs.m_blkt_lithium`, `.fwbs.m_blkt_li2o`, `.fwbs.m_blkt_beryllium`, `.fwbs.m_blkt_steel_total`, `.fwbs.m_blkt_vanadium`, `.fwbs.whtblbreed`, `.fwbs.m_blkt_total`, `.fwbs.f_a_blkt_cooling_channels` | explicit-arg | `.fwbs.blktmodel` (switch, **not in the original 10** — see switches touched), `.fwbs.blkttype` (switch-like, values `{1,2}` vs. else, also not in the original 10), plus ~15 material-fraction/density constants (`fblli`, `fblli2o`, `fblbe`, `fblss`, `fblvd`, `den_steel`, `densbreed`, `fblbreed`, `fblhebmi/mo/pi/po`, `.build.blbuith/blbuoth/blbmith/blbmoth/blbpith/blbpoth`, `.build.dr_blkt_inboard/outboard`, `.fwbs.vol_blkt_inboard/outboard/total`) | `blktmodel == 0` vs. `!= 0` selects two structurally different formulas (different variable sets entirely, not just different constants) — clear split candidate. `blkttype in {1,2}` vs. else similarly picks liquid- vs. solid-breeder formulas with disjoint reads |
| `.fwbs.whtshld`, `.fwbs.wpenshld` | explicit-arg | `.fwbs.vol_shld_total`, `.fwbs.den_steel`, `.fwbs.vfshld` | `wpenshld` is a bare copy of `whtshld` ("penetration shield set = internal shield") |
| `.fwbs.m_fw_total`, local `coolvol` (not a `data` field itself, but feeds `.fwbs.m_fw_blkt_div_coolant_total` below) | explicit-arg (switch) | `.heat_transport.ipowerflow` (switch, **flagged pending in the pilot batch, now with real evidence**), `.first_wall.a_fw_total` or `a_fw_inboard/outboard` depending on branch, `.build.dr_fw_inboard/outboard`, `.fwbs.den_steel`, `.fwbs.fwclfr` | `ipowerflow == 0` reads total first-wall area + a single averaged thickness; `ipowerflow != 0` reads inboard/outboard areas and thicknesses *separately* plus `f_a_fw_coolant_inboard/outboard` (which are themselves computed in 1E1's territory, not re-derived here) — reads-sets clearly differ, this switch should be marked **split**, upgrading it from `switches.md`'s current "pending" |
| `.fwbs.fwclfr` | explicit-arg — but **possibly dead** (none of the 5 existing classifications fit "written, maybe never read," described in note rather than mislabeled) | derived from the same fields just used to compute `coolvol`/`m_fw_total` in the `ipowerflow != 0` branch | only written in the `ipowerflow != 0` branch — comment says "only used by old routines in fispact.f90, safety.f90" (i.e. Fortran modules that may not even exist in this Python codebase anymore) — flag for you: possibly dead output entirely, worth checking before porting rather than assuming it's needed |
| `.fwbs.m_fw_blkt_div_coolant_total` | explicit-arg (switch) | `.fwbs.blktmodel`, `.fwbs.i_blkt_coolant_type` (switch), local `coolvol` | **exact site the pilot switch-batch already found and flagged "unsure, leaning keep-static"** (`stellarator.py:1269-1274`, "same read `coolvol`, only the constant differs: 806.719 vs 1.517") — confirmed by direct read, no new evidence either way on the keep-static question; the pilot's caveat stands (this switch's real value set is water/helium/two water sub-types, this single site may not be representative of the whole switch) |

## proposed signature(s)

This chunk is **not one function** — it's several independent-ish computations sharing an
outer branch context from 1E1. Proposing per logical group rather than one giant
signature; final grouping should be decided once 1E1 is also in hand (these may belong
inside one larger `calculate_blanket_shield_powers`-type function together with 1E1's
setup, or stay separate — not decided here).

```python
def calculate_coolant_pumping_powers(
    i_p_coolant_pumping: int,  # PumpingPowerModelTypes, static
    f_p_fw_coolant_pump_total_heat: float,
    f_p_blkt_coolant_pump_total_heat: float,
    f_p_shld_coolant_pump_total_heat: float,
    f_p_div_coolant_pump_total_heat: float,
    p_fw_inboard_nuclear_heat_mw: float,
    p_fw_outboard_nuclear_heat_mw: float,
    psurffwi: float,
    psurffwo: float,
    p_beam_orbit_loss_mw: float,
    pnucbzi: float,
    pnucbzo: float,
    pnucshldi: float,
    pnucshldo: float,
    p_plasma_separatrix_mw: float,
    p_div_nuclear_heat_total_mw: float,
    p_div_rad_total_mw: float,
) -> tuple[float, float, float, float]:
    # (p_fw_coolant_pump_mw, p_blkt_coolant_pump_mw, p_shld_coolant_pump_mw, p_div_coolant_pump_mw)
    # USER_INPUT branch: returns whatever was already in `data` unchanged — see open
    # question 2, this is a "pass" case that doesn't fit a pure function cleanly.
    ...

def calculate_tf_coil_nuclear_heat(
    i_tf_sup: int,  # TFConductorModel, static
    pnucsi: float, pnucso: float, pnucshldi: float, pnucshldo: float,
) -> float:  # p_tf_nuclear_heat_mw
    ...

def calculate_blanket_mass(
    blktmodel: int, blkttype: int,  # both static
    # ~15 material fractions/densities/volumes, see footprint table
    ...
) -> tuple[float, ...]:  # (m_blkt_lithium or m_blkt_li2o, m_blkt_beryllium, m_blkt_steel_total, m_blkt_vanadium or whtblbreed, m_blkt_total, f_a_blkt_cooling_channels)
    ...

def calculate_fw_blkt_coolant_mass(
    ipowerflow: int, blktmodel: int, i_blkt_coolant_type: int,  # all static
    a_fw_total: float, a_fw_inboard: float, a_fw_outboard: float,
    dr_fw_inboard: float, dr_fw_outboard: float, den_steel: float,
    f_a_fw_coolant_inboard: float, f_a_fw_coolant_outboard: float,
    vol_blkt_total: float, f_a_blkt_cooling_channels: float,
    vol_shld_total: float, vfshld: float,
) -> tuple[float, float]:  # (m_fw_total, m_fw_blkt_div_coolant_total)
    ...
```

`a_div_surface_total`/`m_div_plate` (the cross-call-dependent block) is **not proposed as
a pure function here** — see open question 1, its signature depends on resolving the
pipeline-order dependency first.

## tier signal

**Tier 1 for everything except the divertor cross-call block, which is tier-2-adjacent**
— not because it has an internal solver, but because its correct value depends on
*which call in the run() sequence this is*, which no pure function of its declared inputs
alone can express without also taking "is this the first call" as an explicit input (at
which point it stops being a hidden default and becomes an honest boundary condition —
see open question 1). Nothing else in this chunk calls another model or contains a loop.

## switches touched

- `i_p_coolant_pumping` — confirms existing `switches.md` split decision, two more sites.
- `i_tf_sup` — confirms existing split decision, matches the pilot's own site list exactly.
- `i_blkt_coolant_type` — confirms the exact site the pilot flagged "unsure, leaning
  keep-static"; no new evidence to resolve it either way.
- `.heat_transport.ipowerflow` — **new evidence**: reads-sets differ structurally (total
  vs. separate inboard/outboard geometry), recommend upgrading from "pending" to
  **split** in `switches.md`.
- `.fwbs.blktmodel` — **not in the original 10, found here.** `== 0` vs. `!= 0` selects
  disjoint formulas for blanket mass. Recommend **split**.
- `.fwbs.blkttype` — **not in the original 10, found here.** `in {1,2}` (liquid breeder)
  vs. else (solid breeder, HCPB) selects disjoint material-fraction sets. Recommend
  **split**. Not `i_`-prefixed despite functioning exactly like a switch — same pattern
  `traceability_policy.md` should probably note explicitly (naming convention isn't a
  reliable way to enumerate all switches).

## calls into other models

None. All computation is local arithmetic on already-available `data` fields (plus the
one cross-call self-reference noted above, which isn't a call to another model but is a
call-history dependency worth the same scrutiny).

## JAX-difficulty flags

- `self.first_call_stfwbs` pattern — `blocker` for a direct translation, `workaround-known`
  once reframed as an explicit input (see open question 1). Tag: **cross-call stateful
  flag**, not covered by any existing `traceability_policy.md` category — closest is
  `implicit-io`, but that's defined as intra-call; this is genuinely a different shape
  (depends on invocation history, not on other reads within the same call).
- `raise ProcessValueError("i_p_coolant_pumping = 0 or 1 only for stellarator")` on an
  unreachable-in-practice `else` (line 925-928) — `needs-lax-cond-or-where` in principle,
  but since `i_p_coolant_pumping` is a switch already slated for `split`, this becomes
  moot: the port simply never generates a function body for values other than 0/1, no
  runtime check needed. Flagging so it isn't independently "fixed" with a `lax.cond` when
  the switch-split already makes it unreachable by construction.
- No CoolProp, no dynamic shapes elsewhere in this range.

## open questions

1. **The `first_call_stfwbs` / `a_div_surface_total` pattern is a real pipeline-ordering
   dependency, not just an implicit-io classification question.** `st_fwbs` needs a
   divertor surface area that a different routine ("stdiv", per the source comment)
   computes *later* in `run()`'s sequence — so the first call uses a hardcoded `50.0`
   guess and subsequent calls (within the same `Caller.call_models` idempotence loop, or
   across the outer optimiser's repeated evaluations) presumably use whatever `stdiv` last
   wrote. This is the same shape as the "iterate the whole pipeline until idempotent"
   pattern already flagged in `../../CLAUDE.md`, but localized to a two-node cycle
   (`st_fwbs` ↔ `stdiv`) rather than the whole graph. Worth checking `stdiv`'s location in
   `run()` (out of this chunk's scope) to confirm whether this is a genuine `Cut`-shaped
   fixed point or resolves trivially. Recommend treating `first_call_stfwbs` as an
   explicit boolean input to the ported function (`is_first_evaluation` or equivalent)
   rather than hidden state, at minimum — this makes the current behavior faithfully
   reproducible without deciding the cycle question yet.
2. **The `i_p_coolant_pumping == USER_INPUT` branch (`pass`) doesn't fit the pure-function
   model cleanly** — it means "leave `data` exactly as the caller already had it," which
   for a stateful `Model.run()` is free (nothing happens) but for a pure function means
   the function must either (a) not be called at all when this branch is selected (a
   graph-assembly-level decision, consistent with `naming_convention.md`'s
   switches-are-not-ports stance), or (b) explicitly thread the existing values through
   as pass-through outputs. Same shape as chunk 1F's "degenerate branch, maybe shouldn't
   exist as a node" finding — recommend the same treatment.
3. Whether `.fwbs.p_blkt_multiplication_mw`'s two-part accumulation (written at line 930,
   incremented at line 948) is intentional double-counting-avoidance or worth collapsing
   into one expression when ported — functionally equivalent either way, purely a
   readability question for the pure port.
