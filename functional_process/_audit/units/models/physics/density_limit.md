---
kind: model-unit
status: draft
confidence: medium
---

**Ported (2026-08-26).** `density_limit.py` declares the eight one-liner formulas plus
two small extraction functions and three cottax nodes: `GreenwaldDensityLimit`,
`EnforcedDensityLimitGreenwald`, `GreenwaldFraction`. No `unit_registry.md` row and no
`next_steps.md` edit — registration is the consolidation pass's job; see
"## registration instructions" below.

## not the stellarator unit

`process/models/physics/density_limit.py` (`PlasmaDensityLimit`, singular "limit") is
the **tokamak** density limit and this record's whole subject. It is a different
PROCESS source file, a different physics model, and a different audit record from
`process/models/stellarator/density_limits.py` (`st_sudo_density_limit` and friends,
plural "limits"), already ported as stellarator-scope unit #3
(`_audit/units/models/stellarator/density_limits.md`). Both write
`.physics.nd_plasma_electrons_max` — genuinely the same field, different producers,
switched between at graph-build time by whole-device mode (tokamak vs. stellarator),
not by `i_density_limit` itself. `CLAUDE.md`'s "The current architecture" section names
`PlasmaDensityLimit.run`/`calculate_density_limit` as its worked example of the
pure/impure split this whole project is organised around — this is that exemplar's
port.

## source

`process/models/physics/density_limit.py` (697 lines, one class,
`PlasmaDensityLimit`). In scope: the eight `calculate_*_density_limit` statics
(`:143-440`), `calculate_density_limit` (`:442-616`, the composite that calls all
eight), `get_density_limit_value` (`:111-141`, the selection dispatch), and `run`
(`:67-109`, the stateful shell — three assignments: the array, the enforced scalar, the
Greenwald fraction). `output` (`:618-696`) is a pure reporting shell, out of scope by
the schema's own convention.

**`large_tokamak_eval.IN.DAT` sets `i_density_limit = 7`**
(`tests/regression/input_files/large_tokamak_eval.IN.DAT:289`) — GREENWALD. PROCESS's
own bare `physics_variables.py` default is `8` (ASDEX_NEW, `:863`); the reference file
overrides it, confirmed by reading the `IN.DAT` line directly, not assumed from the
default.

## the eight are a computes-then-selects family

`calculate_density_limit` evaluates all eight formulas unconditionally into
`nd_plasma_electron_max_array[0..7]` (`:526-610`, no `i_density_limit` branch anywhere
in that block) and only then calls `get_density_limit_value(DensityLimitModel(i_density_
limit), nd_plasma_electron_max_array)` (`:614-616`) to pick one element as the enforced
scalar. This is the same shape `bootstrap_current.md` and `l_h_transition.md` record for
their own families, and the wave coordinator's settled policy for it applies here
unchanged (`bootstrap_current.md`'s "## deviations" item 2, restated in the wave-1
brief): **occupant per switch value**, not one node computing the family plus an index.
`_audit/tokamak_boundary.md`'s "zero boundary reads" note on `.density_limit` (the
25-slot table) predates this port and is not a disagreement — it measured the graph as
it stood before any tokamak physics unit existed to read this slot's outputs; the real
readers below are constraint bodies and one sibling unit, not graph nodes, which is
exactly what that table would still show today.

**Measured, not assumed: which array elements have a real reader.** `grep -rn` over
`process/`, excluding `output()`/`data_structure/`/the selection dispatch itself:

| element | model | non-reporting, non-selection readers |
|---|---|---|
| `[0]` ASDEX | — | **none** |
| `[1]` BORRASS_ITER_I | — | **none** |
| `[2]` BORRASS_ITER_II | — | **none** |
| `[3]` JET_EDGE_RADIATION | — | **none** |
| `[4]` JET_SIMPLE | — | **none** |
| `[5]` HUGILL_MURAKAMI | — | **none** |
| `[6]` GREENWALD | — | `density_limit.py:106-109` itself (`f_nd_plasma_greenwald`); `constraints.py:1657` (constraint 76, Eich model); `bootstrap_current.py:245` (Sauter arm's `n_greenwald`, another agent's ported unit — already declares this exact element as a boundary read, see its record's "the family PROCESS computes and this port does not"); `physics.py:1105` (reinke-criterion branch, gated on constraint 78 being active, out of scope) |
| `[7]` ASDEX_NEW | — | **none** |

Seven of eight match `bootstrap_current.md`'s "dead work" finding exactly: no reader
anywhere in `process/` outside `output()` and the selection dispatch that only exists to
pick one of them. **Element 6 (Greenwald) does not** — it has three real readers that
depend on it regardless of `i_density_limit`'s value, because `calculate_density_limit`
computes it unconditionally, not behind any switch check. That asymmetry is the reason
this unit has an unconditional node (`GreenwaldDensityLimit`) alongside its one
switch-selected occupant (`EnforcedDensityLimitGreenwald`), rather than a single
per-arm node the way `bootstrap_current.py`'s Sauter arm gets one.

## why two nodes, not one

Considered and rejected: folding `GreenwaldDensityLimit` and
`EnforcedDensityLimitGreenwald` into one node with two `Output`s (the shape
`bootstrap_current.md`'s "## deviations" item 3 uses for
`.physics.j_plasma_bootstrap_sauter_profile` — a function's second return value, both
kept). For `i_density_limit == 7` the two fields genuinely are the same PROCESS float,
so the merge would be numerically identical to what is ported. Rejected because it
would stop being accurate the moment a second arm is ported: `.physics.
nd_plasma_electron_max_array[6]` must stay present and unchanged no matter which arm is
selected (its three readers do not care about `i_density_limit`), while `.physics.
nd_plasma_electrons_max` must come from whichever arm's own formula is switched on. A
single merged node would have to be un-merged the first time e.g. ASDEX gets wired,
where a bare `EnforcedDensityLimitAsdex` occupant reads its own five inputs and does
not touch the Greenwald array element at all. Two nodes today is not overhead bought
for nothing; it is the shape the graph already needs at two occupants and costs nothing
extra at one.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.plasma_current` | read | explicit-arg | `GreenwaldDensityLimit`'s `c_plasma` |
| `.physics.rminor` | read | explicit-arg | `GreenwaldDensityLimit` |
| `.physics.nd_plasma_electron_max_array[6]` | write | explicit-arg | `GreenwaldDensityLimit`'s sole output; unconditional, see above |
| `.physics.nd_plasma_electron_max_array[6]` | read | explicit-arg | `EnforcedDensityLimitGreenwald`, `GreenwaldFraction` — a real edge, not invented: `get_density_limit_value`'s GREENWALD arm is a bare index into this element (`:139`), and `run()`'s Greenwald-fraction line reads the same element it just had `calculate_density_limit` populate (`:106-109`) |
| `.physics.nd_plasma_electrons_max` | write | explicit-arg | `EnforcedDensityLimitGreenwald`'s sole output; the field constraint 5's body reads (`core/solver/constraints.py:444-476` — a constraint, not a graph node) |
| `.physics.nd_plasma_electron_line` | read | explicit-arg | `GreenwaldFraction`; produced elsewhere in `physics.py`, out of this unit's scope |
| `.physics.f_nd_plasma_greenwald` | write | explicit-arg | `GreenwaldFraction`'s sole output; unconditional (`:106-109`, no `i_density_limit` check) |
| `.physics.i_density_limit` | switch | topology (graph-build time) | selects which occupant class exists at this slot; `7` on the reference arm. Not a `VarPath` on any node — `_audit/naming_convention.md` § "switches are not ports" |
| the seven unwired elements' own inputs (`b_plasma_toroidal_on_axis`, `q95`, `rmajor`, `prn1`, `zeff`, `p_hcd_injected_total_mw`, `qcyl`, `p_plasma_separatrix_mw`) | read (by the unwired formulas only) | explicit-arg | ported and Tier-1-tested as bare functions, no node — see "## UNPORTED" |

## proposed signature(s)

```python
def calculate_greenwald_density_limit(c_plasma: float, rminor: float) -> float: ...
def select_enforced_density_limit_greenwald(nd_plasma_electron_max_array_7: float) -> float: ...
def calculate_greenwald_fraction(
    nd_plasma_electron_line: float, nd_plasma_electron_max_array_7: float
) -> float: ...
```
Plus the seven unwired formulas, each unchanged from its source `@staticmethod`
signature (see `density_limit.py`'s docstrings for exact `file:line` per formula).

## cottax nodes

Written in `functional_process/models/physics/density_limit.py`:
`GreenwaldDensityLimit`, `EnforcedDensityLimitGreenwald`, `GreenwaldFraction`. Not yet
imported into `total_process.py` — consolidation's job, see below.

## tier signal

**Tier 1** for all ten ported functions (eight formulas + the two extraction functions).
No internal iteration anywhere in this unit; `calculate_density_limit` and `run` are
both straight-line.

## switches touched

`i_density_limit` (`.physics`, values 1-8, tokamak-only — the stellarator arm never
reads it). **Split** — the settled computes-then-selects policy (see above): one
occupant class per value. Only `7` (GREENWALD) wired; `1`-`6`, `8` UNPORTED, see below.
Cross-reference `core/solver/switches.md` if/when it gains an `i_density_limit` entry —
none exists yet as of this pass.

## calls into other models

None — `calculate_density_limit` and its eight statics call nothing outside this file.
`run()` calls no other `Model`.

## UNPORTED

| `i_density_limit` value | model | reason |
|---|---|---|
| 1 | ASDEX | formula ported and Tier-1-tested (`calculate_asdex_density_limit`, free oracle from `test_calculate_density_limit`'s `expected_dlimit[0]`); no occupant node — dead work at this switch value on the reference arm (see "measured" table above) |
| 2 | BORRASS_ITER_I | same as above, `expected_dlimit[1]` |
| 3 | BORRASS_ITER_II | same as above, `expected_dlimit[2]` |
| 4 | JET_EDGE_RADIATION | same as above, `expected_dlimit[3]`; formula has a real domain branch (`denom <= 0.0`), ported as `jnp.where`, tested both branches |
| 5 | JET_SIMPLE | same as above, `expected_dlimit[4]` |
| 6 | HUGILL_MURAKAMI | same as above, `expected_dlimit[5]` |
| 8 | ASDEX_NEW | same as above, `expected_dlimit[7]`; also PROCESS's own bare default, which the reference file overrides |

Each is a "trivially cheap validated sibling" per the wave-1 brief: a `@staticmethod`
already free of `self.data` access, diffed directly against PROCESS's own function
(the strongest oracle available, not a transcription), one legacy sample plus fuzz. If
a later pass wires any of them as an occupant (e.g. to answer a different input file),
the function is already ported and tested — only a node class and a
`total_process.py` registration would be needed.

## JAX-difficulty flags

- `calculate_jet_edge_radiation_density_limit`: `needs-lax-cond-or-where`,
  `workaround-known` — source's Python `if denom <= 0.0: return 0.0` on a traced
  quantity, ported as `jnp.where` with the standard safe-denominator guard (matches
  `exhaust.py`'s `calculate_radiation_fraction`).
- Every fractional exponent (`0 < p < 1`) across the eight formulas: `safe_pow`/
  `safe_sqrt` used throughout, per `_audit/next_steps.md` §9. None hit exactly zero at
  any declared sample point (all inputs are strictly positive physical quantities), so
  no `_harness/boundary.py` registration was earned this pass — `test_gradient_finite_
  at_zero` passed clean for every contract.

## open questions

1. **Two-node split is a documented, not-yet-tested seam.** No second `i_density_limit`
   arm is wired yet, so the split described in "why two nodes, not one" is argued from
   reading the source, not exercised by a test with two live occupants at once. Whoever
   wires a second arm next should re-read that section before assuming the shape is
   settled by precedent rather than by argument.
2. **`GreenwaldFraction`'s reference is a transcription**, the weakest oracle in this
   unit (see the test module's `_reference_greenwald_fraction` docstring) — `run()` has
   a callable shell, but reaching this one line through it means also running the other
   seven unrelated formulas and back-deriving a `(plasma_current, rminor)` pair for a
   synthetic array element. The formula is a single division with no branches, so the
   risk this represents is low, but it is not a real PROCESS call the way the other nine
   contracts in this unit are.
3. **`nd_plasma_electron_line`'s own producer is out of this unit's scope** (`physics.py`,
   not yet ported) — `GreenwaldFraction` declares it as an ordinary boundary read.
4. **Fuzz bounds are hand-chosen**, not taken from `bounds_from_iteration_variables`:
   neither `plasma_current`/`rminor` (Greenwald's own inputs) nor any of the seven
   unwired formulas' inputs are declared PROCESS iteration variables under a matching
   name. Same gap several sibling records already carry (e.g.
   `bootstrap_current.md`'s question 5).

## registration instructions (for the consolidation pass)

- Import `GreenwaldDensityLimit`, `EnforcedDensityLimitGreenwald`, `GreenwaldFraction`
  from `functional_process.models.physics.density_limit` into `total_process.py`.
- Slot: `.tokamak.density_limit` (per `_audit/tokamak_boundary.md`'s 25-slot table).
  `GreenwaldDensityLimit` is unconditional within that slot (present for every
  `i_density_limit` value this graph ever wires, not just 7); `EnforcedDensityLimit
  Greenwald` occupies the slot's `i_density_limit == 7` arm specifically. No other arm
  has an occupant yet, so today the slot has exactly one occupant class regardless.
- Add an `i_density_limit` row to `core/solver/switches.md` if/when that file exists in
  this scope (it did not at the time of this pass); values `1`-`6`, `8` go in
  `indat.py`'s `UNPORTED` table with the reasons in "## UNPORTED" above.
- No `unit_registry.md` row added by this pass, per the wave-1 brief's write-scope
  restriction — the consolidation pass adds it.
