---
kind: model-unit
status: draft
confidence: medium-high
---

## source
`process/models/stellarator/stellarator.py`, lines 191-319: `st_new_config()` (191-274),
`st_geom()` (276-318). Chunk 1C of unit #1 (see `../../_audit/unit_registry.md`). Both
methods read in full.

## data footprint

`st_new_config()`:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.istell` | read | explicit-arg | passed to `load_stellarator_config` — see "calls into other models" and the switch note below, **this is a second, distinct role for `istell` beyond the top-level tokamak/stellarator pipeline split already recorded in `switches.md`** |
| `.globals.output_prefix` | read | explicit-arg | used only to build a `Path` for the `istell==6` case (see JAX-difficulty flags) |
| `.numerics.ixc` | read | explicit-arg (but see open question 1) | list-membership test (`1 not in ixc`) — **this is the solver reading its own configuration to decide whether a physics field gets forward-computed at all**, not an ordinary physical input |
| `.physics.aspect` | write, **conditional** | explicit-arg (but see open question 1) | only written when `1 not in ixc`, i.e. only when `aspect` is *not* currently an iteration variable |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.rminor` | write | explicit-arg | `rmajor / aspect`, using whatever `aspect` now holds (post conditional-write above) |
| `.physics.eps` | write | explicit-arg | `1 / aspect` |
| `.tfcoil.n_tf_coils` | write, unconditional | explicit-arg | comment at source: "This overwrites n_tf_coils in input file" — an intentional, visible clobber of a user-settable default, not hidden; noted, not flagged |
| `.stellarator_config.stella_config_coilspermodule` | read | explicit-arg | from the device-config table, see below |
| `.stellarator_config.stella_config_symmetry` | read | explicit-arg | same |
| `.stellarator.f_st_rmajor` | write | explicit-arg | |
| `.stellarator.f_st_rminor` | write | explicit-arg | |
| `.stellarator.f_st_aspect` | write | explicit-arg | |
| `.stellarator.f_st_n_coils` | write | explicit-arg | |
| `.stellarator.f_st_b` | write | explicit-arg | |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | |
| `.stellarator.f_st_coil_aspect` | read | explicit-arg | plain input field, default `1.0` (`stellarator_variables.py:19`), not written anywhere in scope so far — confirmed by grep, not assumed |
| `.stellarator.r_coil_major` | write | explicit-arg | |
| `.stellarator.r_coil_minor` | write | explicit-arg | |
| `.stellarator.f_coil_shape` | write | explicit-arg | |
| `.stellarator_config.stella_config_rmajor_ref` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_rminor_ref` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_aspect_ref` | read | explicit-arg | device-config table (also the conditional default for `physics.aspect`) |
| `.stellarator_config.stella_config_bt_ref` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_coil_rmajor` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_coil_rminor` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_min_plasma_coil_distance` | read | explicit-arg | device-config table |

`st_geom()` — clean, no switches, no calls out:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.f_st_rmajor` | read | explicit-arg | produced by `st_new_config`, ordinary cross-node input, not implicit-io |
| `.stellarator.f_st_rminor` | read | explicit-arg | same |
| `.stellarator_config.stella_config_vol_plasma` | read | explicit-arg | device-config table |
| `.stellarator_config.stella_config_plasma_surface` | read | explicit-arg | device-config table |
| `.physics.rminor` | read | explicit-arg | produced by `st_new_config` |
| `.physics.vol_plasma` | write | explicit-arg | |
| `.physics.a_plasma_surface` | write | explicit-arg | |
| `.physics.a_plasma_poloidal` | write | explicit-arg | |
| `.physics.a_plasma_surface_outboard` | write | explicit-arg | comment: "retained only for obsolescent fispact calculation" and "approximate as for tokamaks" — candidate for a stale-code flag, not resolved here |

No `implicit-io`, `implicit-io-via-callee`, or `redundant-duplicate-write` instances in
either method — every read/write in this chunk is either a plain input, a plain output,
or the one conditional-write case captured in open question 1 (which doesn't cleanly fit
any of the four existing labels, see below, consistent with chunk 1D's independent
finding that the four-way scheme needs a supplementary label for non-risky same-call
patterns — this is a *different* supplementary case than 1D's, see open question 1).

## proposed signature(s)

Device-config loading should **not** be inside the traced computation at all — see "calls
into other models" and JAX-difficulty flags. Proposed split:

```python
# setup-time, outside jit — plain Python, called once per run configuration
def load_stellarator_device_config(
    istell: int, config_file: Path | None
) -> DeviceConfig: ...  # returns a dataclass/dict of the stella_config_* fields, ported from preset_config.py
```

```python
# tier-1, traced
def calculate_stellarator_scaling_factors(
    rmajor: float,
    aspect: float,  # see open question 1 — may or may not be an input here depending on run config
    b_plasma_toroidal_on_axis: float,
    f_st_coil_aspect: float,
    stella_config_coilspermodule: int,
    stella_config_symmetry: int,
    stella_config_rmajor_ref: float,
    stella_config_rminor_ref: float,
    stella_config_aspect_ref: float,
    stella_config_bt_ref: float,
    stella_config_coil_rmajor: float,
    stella_config_coil_rminor: float,
    stella_config_min_plasma_coil_distance: float,
) -> tuple[float, ...]:
    # returns (rminor, eps, n_tf_coils, f_st_rmajor, f_st_rminor, f_st_aspect,
    #          f_st_n_coils, f_st_b, r_coil_major, r_coil_minor, f_coil_shape)
    ...
```

```python
def calculate_stellarator_plasma_geometry(
    f_st_rmajor: float,
    f_st_rminor: float,
    rminor: float,
    stella_config_vol_plasma: float,
    stella_config_plasma_surface: float,
) -> tuple[float, float, float, float]:
    # returns (vol_plasma, a_plasma_surface, a_plasma_poloidal, a_plasma_surface_outboard)
    ...
```

The `aspect` input to the second function is provisional pending open question 1 — if
`aspect` is an active iteration variable it should not be recomputed by this function at
all (must arrive as a boundary input from the optimizer, not be produced here).

## tier signal

**Tier 1** for both methods' actual arithmetic. The device-config lookup itself is
**not tier 1 in the usual sense — it's setup-time configuration, not a traced
computation** (see above); flagging it as tier 1 would be misleading since it shouldn't
be inside the jitted graph at all.

## switches touched

- **`data.stellarator.istell`** — already has an entry in `switches.md` from the pilot
  batch, describing it as the top-level tokamak/stellarator pipeline selector
  (values effectively 0 vs. non-zero). **This chunk found a second, separate role**:
  once inside the stellarator pipeline, `istell`'s specific non-zero value (1-5, or 6 for
  a user-supplied JSON file) selects *which stellarator device configuration*
  (`HELIAS5B`/`HELIAS4`/`HELIAS3`/`W7X30`/`W7X50`/custom-JSON`, all defined in
  `preset_config.py`) to load. This is a genuinely different question from the pipeline
  split, decided by the *same field*. Recommend `switches.md`'s `istell` entry be
  extended to note this dual role rather than treating it as fully covered by the
  pipeline-split description — **not edited here, out of this chunk's directive scope**,
  flagging for whoever next touches that record.
- **Answering this chunk's directive question directly**: the device-config selection
  (`istell` 1-6 inside `st_new_config`) is **data-table-shaped, not a topology-changing
  switch**. Checked directly, not assumed: `load_stellarator_config`
  (`preset_config.py:215-250`) does `match istell: case 1: machine_config = HELIAS5B ...`
  then `for variable_name, variable_value in machine_config.items(): setattr(data.stellarator_config, f"stella_config_{variable_name.lower()}", variable_value) if hasattr(...)`.
  All five hardcoded tables (`HELIAS5B`, `HELIAS4`, `HELIAS3`, `W7X30`, `W7X50`) share the
  same key schema (checked `HELIAS5B`/`HELIAS4` directly; did not byte-for-byte diff all
  five key sets against each other, flagged as a cheap follow-up check rather than
  assumed), and the `hasattr` guard means even the arbitrary-JSON case (`istell==6`) can
  only populate pre-declared `stellarator_config` fields — so the **shape** (which
  `VarPath`s get written) is uniform across all six cases, only the **values** differ.
  This is `keep-static`-shaped in spirit, but more precisely it's the same "resolved
  above the traced computation" pattern already found for `i_cost_model`
  (`Models.costs` in `main.py`) — except simpler: `i_cost_model` picks between
  *different implementations* (`Costs` vs `Costs2015`, each with its own internal logic
  needing its own audit), while this picks between *different constant tables* feeding
  one uniform downstream computation. Worth distinguishing these two sub-cases of
  "resolved above this file" in `traceability_policy.md`/`schema.md` — not done here,
  flagged as a suggestion.

## calls into other models

- `load_stellarator_config(istell, config_file, data)` — module-level function in
  `process/models/stellarator/preset_config.py`, **not a `Model` method**, but takes the
  whole `data: DataStructure` object and writes into `data.stellarator_config.*`
  directly (an `implicit-io-via-callee`-shaped call, except the callee isn't a `Model` —
  the same underlying pattern regardless). `preset_config.py` is registry unit #8,
  currently `pending`. This chunk's record does not depend on that audit landing first
  (the config-loading step is being proposed as non-traced setup code regardless of its
  internal shape), but the full device-config table contents (all `stella_config_*`
  fields, not just the ones read directly here) are only fully enumerable once unit #8 is
  audited.

## JAX-difficulty flags

- **File I/O, `non-traceable-external-call`, severity: minor.** `istell==6` triggers
  `open(config_file).read()` + `json.load` inside `load_stellarator_config`. Not a
  blocker because it belongs in the non-traced setup step proposed above (same treatment
  as parsing IN.DAT) — flagged so whoever ports this doesn't accidentally put it inside a
  jitted path.
- **`conditional-ownership-by-run-config`, severity: needs-a-policy-decision (new tag,
  not yet in `traceability_policy.md` — see open question 1).** The `physics.aspect`
  conditional write is not a data-dependent branch in the usual JAX sense (it doesn't
  branch on a traced *value*); it branches on whether a field is a member of the current
  run's iteration-variable list, which is fixed for the whole run (as static as a
  switch) but determines whether *this node owns this output at all*. That's a
  structural question about the node's port set, not a runtime `lax.cond`/`jnp.where`
  question — see open question 1 for why this doesn't fit the switch-split framework
  either.

## open questions

1. **The `physics.aspect` conditional-ownership pattern — the main finding of this
   chunk.** `if 1 not in self.data.numerics.ixc: self.data.physics.aspect = <config
   default>` means: **when `aspect` is an active iteration variable, this function does
   not touch it at all** (it stays whatever the optimizer/previous state set); **when
   it isn't, this function unconditionally overwrites it every call.** This is the same
   underlying tension flagged generally in `../../CLAUDE.md` ("no model declares its
   read set or write set" / iteration variables as boundary inputs with no forward
   producer) but now a concrete, minimal instance: **whether this node owns
   `physics.aspect` as an output is itself a per-run configuration decision** (does
   `1 ∈ ixc`?), not fixed by the code alone. In cottax terms, a node's `owns` set can't
   depend on a runtime condition — `physics.aspect` would have to be conditionally
   *excluded* from this node's declared outputs whenever it's also declared as an
   iteration variable for that run, decided at the same graph-build time as switches are
   resolved (not inside the traced function). Recommend this become its own named
   pattern in `traceability_policy.md`, since **finding one instance suggests there may
   be more** — this exact `if <id> not in ixc: field = <default>` idiom is a plausible
   thing to grep for across the remaining units (a quick search: this specific pattern,
   `not in self.data.numerics.ixc`, wasn't checked codebase-wide from this chunk;
   flagging as worth a targeted grep before assuming it's rare).
2. Whether the five hardcoded device-config tables in `preset_config.py` (not fully
   audited here, only spot-checked) are *exactly* schema-uniform, or whether some carry
   extra/missing keys relative to others — would firm up "data-table-shaped, not
   switch-shaped" from a spot-check to a confirmed fact. Cheap follow-up, not done here
   to stay in this chunk's line range.
3. `physics.a_plasma_surface_outboard`'s "obsolescent fispact calculation" /
   "approximate as for tokamaks" comments — candidate stale/approximate code, flagged for
   your judgment, not a structural finding.
