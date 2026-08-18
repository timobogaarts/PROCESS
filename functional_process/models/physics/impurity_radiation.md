---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/physics/impurity_radiation.py`, 756 lines total. Registry unit #23,
scoped by `unit_registry.md` row #23 to: the model half at roughly L379-755
(`ImpurityRadiation`, `create_f_rad_core_profile`,
`calculate_impurity_radiation_power_density`), plus
`calculate_average_charge_at_temp`/`_calculate_average_charge_at_temp_compiled`
(L408-511, `@njit`) and `element2index` (L605-627).

### Most of that range is already ported, by unit #20

`radiation_power.py` (registry unit #20) audited `calculate_radiation_powers`, whose
first act constructs `ImpurityRadiation(plasma_profile, data_structure)` and calls
`.calculate_imprad()` — a bare-module-import miss, the same blind spot that found units
#19/#21 too. That unit's record (`radiation_power.md` § "Scope correction") ported the
whole `ImpurityRadiation` class, `create_f_rad_core_profile`, and
`calculate_impurity_radiation_power_density` (L379-405, L513-602, L632-755) into
`functional_process/models/physics/radiation_power.py`, as three cottax nodes
(`SynchrotronRadiationPower`, `ImpurityRadiationTotals`, `PlasmaRadiationPowers`) and two
of the "no node" helper functions. **Not re-audited or re-ported here** — see that
record, which is the authoritative one for that range. This record cross-references it
rather than duplicating it, per this directive's boundary (this file may not edit
`radiation_power.md`/`.py`).

What unit #20's scope explicitly did **not** reach, and what this record actually
covers, is the remainder of the requested range:

| function | in unit #20's port? | in this record? |
|---|---|---|
| `create_f_rad_core_profile` (L379-405) | yes — `radiation_power.py` | cross-referenced only |
| `calculate_average_charge_at_temp` / `_calculate_average_charge_at_temp_compiled` (L408-511) | **no** — `radiation_power.md`: "not on this path at all" | **yes**, ported below |
| `calculate_impurity_radiation_power_density` (L513-602) | yes — `radiation_power.py` | cross-referenced only |
| `element2index` (L605-627) | **no** — not on unit #20's path | **yes**, ported below |
| `ImpurityRadiation` (L632-755) | yes — `radiation_power.py`'s `ImpurityRadiationTotals`/`PlasmaRadiationPowers` | cross-referenced only |

This matches `unit_registry.md` row #23's own note precisely: unit #23 is "no longer
blocking unit #20, which shipped without it — but still needed for
`ImpurityRadiationTotals`'s full closure and for unit #9" — see § open questions 1 below
for exactly what "full closure" does and does not mean here.

### Second caller, the actual reason this unit exists

`calculate_average_charge_at_temp` and `element2index` are reached from
`process/models/physics/physics.py`'s `PhysicsCalculations.plasma_composition()`
(L1166-…) and `.calculate_effective_charge_ionisation_profiles()` (L1749-1781) — both
already-registered unit #9 methods (`physics.py`, audited in parallel with this unit, not
touched here). Call sites:

- `plasma_composition`: `calculate_average_charge_at_temp` at L1246 (`znimp`
  accumulation, one call per impurity species, scalar-shaped), L1363 (effective charge
  `n_charge_plasma_effective_vol_avg`, same shape) and L1482/L1759/L1777 (further
  accumulations, same shape); `element2index` at L1289, L1301, L1337, L1342, L1347,
  L1350 — six calls, resolving `"H_"`, `"He"`, `"C_"`, `"O_"`, `"Fe"`, `"Ar"` to array
  indices used both to *read* and to *write*
  `.impurity_radiation.f_nd_impurity_electron_array`.
- `calculate_effective_charge_ionisation_profiles`: `calculate_average_charge_at_temp`
  at L1759 (`.physics.n_charge_plasma_effective_profile`, one call per profile point) and
  L1777 (`.impurity_radiation.n_charge_impurity_profile`, one call per (species, profile
  point) pair — a double loop, `N_IMPURITIES * n_plasma_profile_elements` calls total).

None of this is scope for the present unit — `physics.py`'s reads/writes and its own
node design belong to unit #9's record — but it is why these two functions are audited
here at all, and it is what "in scope for a different reason than unit #20's own path"
(`unit_registry.md` row #23) means concretely.

### Out of scope

`initialise_imprad`, `init_imp_element`, `read_impurity_file` (L27-376) — one-time
startup I/O that parses 28 `.dat` files under
`process/data/lz_non_corona_14_elements/` into
`.impurity_radiation.{temp_impurity_keV_array, pden_impurity_lz_nd_temp_array,
impurity_arr_zav, impurity_arr_label, impurity_arr_z, m_impurity_amu_array,
impurity_arr_len_tab}`. Its product is a compile-time constant of the graph, not a value
flowing along a runtime edge — same reasoning `radiation_power.md` already gives for the
identical exclusion. Not audited or ported here.

## data footprint

### `calculate_average_charge_at_temp(imp_element_index, temp_electron_kev, data)`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.impurity_radiation.temp_impurity_keV_array` | read | explicit-arg (constant) | `(14, 200)`, one species row selected by `imp_element_index`; file-loaded at startup, never written by a model — same table `calculate_impurity_radiation_power_density` reads |
| `.impurity_radiation.impurity_arr_zav` | read | explicit-arg (constant) | `(14, 200)`, same shape/provenance; **the one table this function reads that `calculate_impurity_radiation_power_density` does not** |
| `.impurity_radiation.impurity_arr_len_tab` | read | **dead** | `(14,)`, all entries 200; used only to index the top of the table (`impurity_arr_len_tab[i] - 1`), but the interpolation itself always uses the full 200-wide row — same dead read `radiation_power.md` open question 3 already found in the sibling function. Dropped from the port's signature |

No writes — a pure function of its three arguments (`imp_element_index` plus the two
table reads), returning its result rather than mutating `data`.

### `element2index(element, data)`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.impurity_radiation.impurity_arr_label` | read | explicit-arg (constant) | `(14,)` string array, populated once by `init_imp_element` from `ImpurityRadiationData.imp_label`'s fixed default order (`"H_"`, `"He"`, `"Be"`, ..., `"W_"`) — never written anywhere else. A lookup over a compile-time constant, not a runtime dependency in any meaningful sense |

No writes. `element` is a plain string literal at every call site (`"H_"`, `"He"`,
`"C_"`, `"O_"`, `"Fe"`, `"Ar"` — the six labels `physics.py` actually looks up), not a
`data` field.

### Off the caller's side (context only — not this unit's footprint)

Both functions' *results* feed into `.impurity_radiation.f_nd_impurity_electron_array`
(written, via `element2index`-resolved indices, at `physics.py:1288-1307`),
`.physics.n_charge_plasma_effective_vol_avg` / `.n_charge_plasma_effective_profile`, and
`.impurity_radiation.n_charge_impurity_profile` — all unit #9's writes, listed here only
so the "calls into other models" section below is concrete, not as part of this unit's
own footprint.

## proposed signature(s)

```python
def calculate_average_charge_at_temp(
    temp_electron_kev, temp_impurity_kev, impurity_arr_zav,
):
    """`impurity_radiation.py:437-510`, for one species, with its table row passed in
    rather than indexed out of `data`. Log-x, linear-y interpolation of <Z>(T_e). Drops
    the dead `np.digitize` block, the dead `len_tab` read, and the boundary clamps
    (redundant with `jnp.interp`'s own clamping — see the port's docstring for the
    argument and the value-agreement test for the confirmation).
    -> n_charge_impurity_average (dimensionless, same shape as temp_electron_kev)
    """

def element2index(element, impurity_arr_label):
    """`impurity_radiation.py:605-627`, with the label array passed in rather than
    indexed out of `data`. A lookup, not a numerical computation.
    -> index (int, 0-13)
    """
```

## cottax node

**Neither function gets a node in this file.** Both are consumed *inside* loops in
`physics.py`'s `plasma_composition`/`calculate_effective_charge_ionisation_profiles`
(unit #9) — `calculate_average_charge_at_temp` is called once per species (accumulated
into a scalar) or once per (species, profile-point) pair (accumulated into
`.impurity_radiation.n_charge_impurity_profile`), never returning a single
`data.<area>.<field>` value on its own; `element2index`'s result is consumed as an array
index, not stored anywhere. Per `schema.md`'s "cottax node" section — "Skip this section
… while open questions about the signature itself are unresolved" — node ownership for
whatever aggregates these calls (e.g. a vmapped `ChargeProfileTable`-shaped node
producing `.impurity_radiation.n_charge_impurity_profile`, or `element2index` resolved
once at graph-assembly time the way `ImpurityRadiationTotals.imp_indices` already is)
belongs to unit #9's own audit, which owns the surrounding loop structure and the
`.physics`/`.impurity_radiation` write sites. This file's job is to close the `data`
back door and hand unit #9 tested, importable pure functions to build that node from —
not to pre-empt its design while unit #9's audit is in progress in parallel.

Both functions are plain module-level functions, imported directly:

```python
from functional_process.models.physics.impurity_radiation import (
    calculate_average_charge_at_temp,
    element2index,
)
```

## tier signal

**Tier 1**, both. `calculate_average_charge_at_temp`: no internal iteration, no
`scipy.optimize`, no call into another model — a single `jnp.interp`, same
classification `radiation_power.md` gives its sibling. `element2index`: a degenerate
tier-1 case with no continuous inputs at all — a pure Python lookup over a compile-time
constant array, no numerics, no iteration, no model call. Both are tested via
`Tier1Contract`; `element2index`'s harness case fuzzes nothing (no `fuzz_bounds` — there
is no continuous domain to draw from) and relies on 14 legacy samples, one per known
species label, which is the entire input space that matters.

## switches touched

None. Neither function reads an `i_*` field or any other switch.

## calls into other models

- Neither function calls another model's method — both are self-contained, straight-line
  (or, for `element2index`, straight-lookup) code over their arguments.
- **Called by** `physics.py`'s `plasma_composition()` and
  `calculate_effective_charge_ionisation_profiles()` (registry unit #9, audited in
  parallel, not touched by this record) — see § source's "Second caller" subsection for
  exact call sites and argument shapes.
- **Not called by** `radiation_power.calculate_radiation_powers` (unit #20) — confirmed
  directly in `radiation_power.md`, which states `calculate_average_charge_at_temp` is
  "not on this path at all" and does not mention `element2index` among its reads.

## JAX-difficulty flags

- `np.interp` in log-x space (`impurity_radiation.py:474-479`) — **minor**. `jnp.interp`
  exists, clamps to the end values exactly as `np.interp` does, and differentiates —
  same conclusion `radiation_power.md` reaches for the sibling function's log-log
  interpolation, one order simpler here since only the *x*-axis is logged.
- Boolean-mask assignment `n_charge_impurity_average[mask] = ...`
  (`impurity_radiation.py:489`, `:505`) — **not carried into the port at all**, rather
  than worked around with `jnp.where`: the assignments are redundant with what
  `jnp.interp`'s own boundary clamping already produces (see the port's docstring for
  the argument, and `TestCalculateAverageChargeAtTemp`'s `argon-varying` sample, whose
  points sit outside the table on both sides, for the empirical confirmation — value
  agreement holds to `MACHINE_PRECISION` with the assignments omitted). Contrast with
  `radiation_power.md`'s JAX-difficulty note on the sibling function's *own* boundary
  clamps, which are **not** redundant (they are the bug — see § open questions 2 below).
- `np.digitize(...)` computing an unused `indices` array
  (`impurity_radiation.py:471-473`) — **not on this path**, dead code, dropped. Also
  flagged in `radiation_power.md` open question 3 for the sibling function; this is the
  second, independent confirmation of the same dead block.
- `numba.njit(cache=True)` on `_calculate_average_charge_at_temp_compiled`
  (`:437`) — **irrelevant to the port**, per this directive's framing: the decorator
  selects a compilation backend, not a computational shape: this port merges the shell
  (`calculate_average_charge_at_temp`, which just unpacks `data`) and the compiled body
  into one plain function, the same "flatten the pure-core/shell pair" move
  `radiation_power.md` made for `calculate_impurity_radiation_power_density`.
- `element2index`'s `.astype(str).tolist().index(...)` — **not traceable, and
  deliberately not traced.** String comparison and Python-level list search have no JAX
  equivalent and none is needed: `impurity_arr_label` is a compile-time constant and
  `element` is always a literal at every call site, so this function is resolved on
  concrete Python values, never inside a `jacfwd`/`jit` trace. Flagged here only so a
  future reader does not mistake the absence of `jnp` in this function's port for an
  oversight.

## open questions

1. **What "full closure" of `ImpurityRadiationTotals` does and does not mean.**
   `unit_registry.md` row #23 says this unit is "still needed for
   `ImpurityRadiationTotals`'s full closure." Checked directly against
   `radiation_power.md`'s own open question 2 (the thing that actually blocks
   `ImpurityRadiationTotals` from `total_process.py`): the blocker is resolving
   `imp_indices` at graph-assembly time, which requires knowing, for the specific
   run configuration, whether `physics.plasma_composition()`'s helium selection
   (`f_plasma_fuel_helium3 == 0` and `f_nd_alpha_thermal_electron == 0`) ever drops
   below the `1e-30` threshold. That computation is `plasma_composition()`'s own body
   (unit #9), which in turn calls `element2index` (to locate `"He"`'s array slot) and
   `calculate_average_charge_at_temp` (for the surrounding `znimp`/effective-charge
   arithmetic in the same method). **This unit supplies the two leaf functions that
   chain needs; it does not itself resolve `imp_indices`.** The graph-assembly-time
   derivation `radiation_power.md` open question 2 describes as "not implemented" is
   still not implemented after this record — porting `plasma_composition()` (unit #9)
   is the next link, and the actual assembly-time resolution (reading the
   post-`plasma_composition` fractions, asserting none is near `1e-30`) is a
   `configuration.py`-level step neither unit performs. Read `unit_registry.md`'s
   phrasing as "removes a leaf blocker on the way to full closure," not "closes it."

2. **Confirms, rather than discovers, a known asymmetry between the two sibling
   functions' boundary clamps.** `radiation_power.md` open question 1 documents that
   `calculate_impurity_radiation_power_density`'s clamps are a real unit-conversion
   bug (assigning `L(Z, Te)` — ~1e-33 — directly into a power-density array — ~1e6 —
   instead of the intended conservative overestimate). This record's equivalent check
   on `calculate_average_charge_at_temp` finds the opposite: its clamps assign
   `impurity_arr_zav[i, 0 or -1]` into `n_charge_impurity_average`, which is exactly
   what `jnp.interp`'s (and `np.interp`'s) own out-of-domain clamping already produces
   — the units match on both sides (`Zav` in, `Zav` out) and the values are identical
   in range and out. No bug here; recorded so a future reader checking both functions
   for the same class of error does not have to re-derive this by hand. Not something
   to "fix" in PROCESS either way, since it produces the same answer.

3. **Not independently verified**: whether `physics.py`'s six `element2index` call
   sites (`"H_"`, `"He"`, `"C_"`, `"O_"`, `"Fe"`, `"Ar"`) are the complete set of labels
   ever looked up anywhere in scope, or whether some other in-scope method (not yet
   audited) looks up a different label. Low-confidence on completeness of that specific
   claim since it rests on a grep of `physics.py` alone, not a full trace of unit #9's
   scope — flagging for whoever finishes unit #9's audit to cross-check.
