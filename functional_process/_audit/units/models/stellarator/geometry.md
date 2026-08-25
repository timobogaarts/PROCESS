---
kind: model-unit
status: reviewed
confidence: high
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

**Update — ported, open question 1 resolved by precedent.** Between this record's
`draft` pass and this update, unit #2 (`build.py`) independently found and ported the
*same* `conditional-ownership-by-run-config` shape for `.build.dr_blkt_inboard`/
`dr_blkt_outboard` (gated on `blktmodel > 0` rather than `ixc` membership, but
structurally identical: a field this chunk owns only under one run configuration, and is
a plain external input under the other). `build.md`/`build.py` resolved it by **splitting
the maybe-owned field's producer into its own tiny node** (`BlktmodelBlanketThickness`,
instantiated only when `blktmodel > 0`) and having the downstream function
(`calculate_build`) take the field as an ordinary explicit arg regardless of source. This
record adopts the identical resolution for `.physics.aspect`:

- `calculate_default_aspect_ratio(stella_config_aspect_ref) -> aspect` — the `1 not in
  ixc` branch, its own tier-1 function/node (`DefaultAspectRatio`). The body is a bare
  passthrough of `stella_config_aspect_ref` (PROCESS's own branch does nothing more than
  that assignment) — ported as a real function anyway, not inlined as a bare graph edge,
  because a `Graph` node is what makes "does this chunk own `.physics.aspect` this run?"
  a structural, inspectable fact rather than a comment. Instantiate this node **only**
  when `1 not in data.numerics.ixc` for the run being assembled.
- `calculate_stellarator_scaling_factors(...)` (below) takes `aspect` as a plain
  explicit arg, sourced from `DefaultAspectRatio`'s output or an external
  iteration-variable input depending on that same run-config fact — it never re-decides
  or re-derives it, matching what `st_new_config` itself does after the branch (reads
  `self.data.physics.aspect` once, whatever value is now there).

This is now a **second confirmed instance** of `conditional-ownership-by-run-config`
(first: `build.md`), which strengthens (but does not by itself settle — still only two
data points) the suspicion recorded in the original open question 1 that the `if <id> not
in ixc: field = <default>` idiom recurs elsewhere; a codebase-wide grep for `not in
self.data.numerics.ixc` remains undone and is still worth doing before assuming this
pattern is now fully catalogued.

**A second, new finding from writing the actual port: `f_st_n_coils` is dead
arithmetic.** `st_new_config` unconditionally overwrites `.tfcoil.n_tf_coils =
stella_config_coilspermodule * stella_config_symmetry`, then a few lines later computes
`f_st_n_coils = n_tf_coils / (stella_config_coilspermodule * stella_config_symmetry)` —
same straight-line function, no branch or call between the two statements that could
make `n_tf_coils` diverge from that exact product. So under the current PROCESS control
flow, **`f_st_n_coils` is always identically `1.0`**. Documented, not fixed, per this
audit's charter: ported faithfully as a genuine division of two arguments (not collapsed
to a literal), both because collapsing it would be a behaviour change outside this
audit's scope, and because a future non-PROCESS caller of `calculate_stellarator_scaling_
factors` could legitimately supply an `n_tf_coils` not freshly derived from the same two
config fields (e.g. once/if `n_tf_coils`'s own conditional-ownership, if it has any, is
untangled elsewhere) — collapsing the formula now would quietly foreclose that. Classified
`local-intermediate` in the data footprint table above (`n_tf_coils` write, then read
back unconditionally in the same straight-line function) — technically correct per
`schema.md`'s definition, but this is the degenerate case that definition's `redundant-
duplicate-write` neighbour was reaching for: not a duplicate *write*, but the *read*
side is provably redundant (always evaluates to the same known constant, `1.0`), which
`local-intermediate` alone doesn't capture. Worth a `schema.md` note for whoever
generalizes these labels next; not resolved here.

## cottax node

Three nodes, all in `geometry.py` (this record's mirrored `.py` file, per
`test_harness.md`'s "a unit is a stem" convention):

- `DefaultAspectRatio` — wraps `calculate_default_aspect_ratio`. Owns `.physics.aspect`.
  **Instantiate only when `1 not in data.numerics.ixc`** for the run being assembled —
  see open question 1's resolution above. This is a graph-assembly-time decision, the
  same kind `naming_convention.md`'s "switches are not ports" already describes for
  ordinary switches, just keyed off iteration-variable membership rather than an `i_*`
  field.
- `StellaratorScalingFactors` — wraps `calculate_stellarator_scaling_factors`. Owns
  `.physics.rminor`, `.physics.eps`, `.tfcoil.n_tf_coils`, and the seven `.stellarator.
  f_st_*`/`r_coil_*`/`f_coil_shape` fields. Reads `.physics.aspect` as a plain `Input` —
  agnostic to whether `DefaultAspectRatio` produced it or it arrived as an external
  iteration-variable value.
- `StellaratorPlasmaGeometry` — wraps `calculate_stellarator_plasma_geometry`. Owns
  `.physics.vol_plasma`, `.physics.a_plasma_surface`, `.physics.a_plasma_poloidal`,
  `.physics.a_plasma_surface_outboard`. No switches, no conditional ownership; a clean
  consumer of `StellaratorScalingFactors`'s outputs (`f_st_rmajor`, `f_st_rminor`,
  `rminor`) plus two device-config fields.

Device-config loading (`load_stellarator_config`/`istell` 1–6) is **not** a node here —
see "switches touched" below and `preset_config.md` (unit #8, reviewed, "not portable as
a node"). Not registered in `functional_process/total_process.py` — out of this chunk's
strict file boundary (a concurrent consolidation pass owns that file); whoever next
touches `total_process.py` should wire in `DefaultAspectRatio` (conditionally),
`StellaratorScalingFactors`, `StellaratorPlasmaGeometry`.

**Downstream consumers already ported and unaffected.** Checked directly (not assumed):
`build.py` (unit #2), `structure.py` (unit #7's sibling chunk 1D), and
`coils/calculate.py` (unit #6, partial) all already read `f_st_rmajor`, `r_coil_minor`,
`r_coil_major`, `f_st_n_coils`, `f_st_b`, `n_tf_coils` as plain `Input`s of their own —
none of them produce these fields, confirming this chunk is their sole producer and that
porting it introduces no duplicate-ownership conflict with already-ported nodes.

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
  the same underlying pattern regardless). `preset_config.py` is registry unit #8.
  **Update: unit #8 has since landed** (`preset_config.md`, `reviewed`) and confirms this
  record's own independent finding below: 5 hardcoded machine presets plus a reflective
  `hasattr`/`setattr` copy, not portable as a cottax node as-is (real output set only
  knowable by cross-referencing `StellaratorConfigData`'s fields, not a fixed declarable
  `outputs` list). This chunk's port does not depend on unit #8's internal shape either
  way — the config-loading step is out of the traced computation regardless — but the two
  records now corroborate each other rather than one being provisional on the other.

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

   **Resolved (this update).** Ported per the `build.py`/unit #2 precedent — see
   "cottax node" above: split into `DefaultAspectRatio` (owns `.physics.aspect`,
   instantiated only when `1 not in ixc`) and `StellaratorScalingFactors` (takes `aspect`
   as a plain `Input` regardless of source). The codebase-wide grep for `not in
   self.data.numerics.ixc` this note asked for is **still not done** — still worth doing
   before assuming these two instances (this chunk, `build.py`) are the only ones.
2. Whether the five hardcoded device-config tables in `preset_config.py` (not fully
   audited here, only spot-checked) are *exactly* schema-uniform, or whether some carry
   extra/missing keys relative to others — would firm up "data-table-shaped, not
   switch-shaped" from a spot-check to a confirmed fact. Cheap follow-up, not done here
   to stay in this chunk's line range. **Still open** — `preset_config.md` (unit #8,
   landed since this record's `draft` pass) does not appear to resolve this specific
   sub-question either (it confirms the `hasattr`/`setattr` reflective-copy shape, not a
   byte-for-byte key diff across all five tables); still a cheap follow-up for whoever
   next touches either record.
3. `physics.a_plasma_surface_outboard`'s "obsolescent fispact calculation" /
   "approximate as for tokamaks" comments — candidate stale/approximate code, flagged for
   your judgment, not a structural finding. **Still open** — ported faithfully as-is
   (`calculate_stellarator_plasma_geometry` computes it exactly as PROCESS does); not
   resolved, not fixed, per this audit's charter.
4. **New (this update).** `f_st_n_coils` is provably always `1.0` under the current
   PROCESS control flow (see the "second, new finding" note above, under "proposed
   signature(s)") — ported faithfully rather than collapsed. Not itself a blocking
   question, but worth someone's judgment on whether this is intentional
   defensive/future-proofing code (in case `n_tf_coils` is someday set independently of
   `coilspermodule * symmetry`) or genuinely dead arithmetic worth flagging upstream to
   PROCESS's maintainers.

## port status

**Ported and harness-tested** (this update). `geometry.py`:
`calculate_default_aspect_ratio`/`DefaultAspectRatio`,
`calculate_stellarator_scaling_factors`/`StellaratorScalingFactors`,
`calculate_stellarator_plasma_geometry`/`StellaratorPlasmaGeometry`. Tests in
`test_geometry.py`, all `Tier1Contract`. Reference adapters for the two
`st_new_config`-derived functions stub `load_stellarator_config` to a no-op via
`unittest.mock.patch` (see that file's module docstring) so a sample's `stella_config_*`
values aren't clobbered by the real device-config table lookup — the same "config loading
is out of scope" decision this record already made, just made mechanically necessary by
needing to call `st_new_config` itself as the oracle.

Verified: `pytest functional_process/models/stellarator/test_geometry.py
-q` (15 passed, 12 skipped — gradient/finite-difference checks are `--fp-gradients`
opt-in) and again with `--fp-gradients --fp-fuzz 50 --fp-fuzz-seed 7` (615 passed, 0
failed) — both value and gradient agreement hold across a wide fuzz sweep, no domain
errors encountered (no `reference_domain_errors` needed for any of the three functions;
all are unconditional arithmetic with no PROCESS-side raise).

Not registered in `functional_process/total_process.py` — out of this chunk's strict file
boundary; see "cottax node" above.
