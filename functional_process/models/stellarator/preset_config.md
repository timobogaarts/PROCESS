---
kind: model-unit
status: reviewed
confidence: high
---

## source
`process/models/stellarator/preset_config.py` (266 lines, full file in scope). Five
machine-preset dict literals (`HELIAS5B`, `HELIAS4`, `HELIAS3`, `W7X30`, `W7X50`, ~25
scalar fields each) plus one function, `load_stellarator_config(istell, config_file,
data)`.

## data footprint

**Confirms and extends chunk 1C's finding** (`stellarator_C_geometry.md`): `istell`
plays a second role here beyond the top-level tokamak/stellarator pipeline split
(`switches.md`) — it selects one of five hardcoded machine-preset tables (`istell` 1-5),
or an externally-loaded JSON file (`istell == 6`). This is a **data-table selector, not
a formula switch**: none of the five branches differ in what they *compute* (nothing is
computed — each is a dict literal), only in *which fixed table* is selected.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.istell` | read | explicit-arg (switch, see above) | `match istell: case 1..6` |
| `.stellarator_config.stella_config_*` | write, ~25 fields, **dynamic** | implicit-io (see open question 1) | see below — not a fixed, statically-known write set |

The write set is **not statically enumerable from this file alone**: the function loops
over `machine_config.items()` (the chosen dict, or the JSON file's parsed contents) and
writes `data.stellarator_config.stella_config_<key.lower()>` **only if
`hasattr(data.stellarator_config, name_on_data_structure)`** — a silent skip for any dict
key that doesn't match an existing `StellaratorConfigData` field name, no error, no log.
The real output set is the *intersection* of whichever dict's keys and
`StellaratorConfigData`'s field names, resolved by reflection at call time, not
determinable by reading this file's source alone (needs
`process/data_structure/stellarator_configuration.py`'s field list cross-referenced
against each dict's keys, one machine at a time).

For `istell == 6`: `config_file` is read from disk (`open(config_file)`, `json.load`) —
real file I/O, a non-traceable external call (see JAX-difficulty flags). Independently
confirms chunk 1C's own note about `istell == 6` doing file I/O — same finding, different
file, not a new one.

## proposed signature(s)

**Not portable as an ordinary `calculate_*` function or `ExplicitFunction` node.** Two
reasons, both structural, not a signature-writing problem:

1. The write set depends on reflection against a *second* file's field list
   (`StellaratorConfigData`), which this function doesn't itself enumerate — a cottax
   node's `outputs` must be a fixed, declared list of `VarPath`s, and this function's
   real output set is only knowable by cross-referencing two sources, and even then is
   conditional on a dict's keys matching (which is exactly the kind of "implicit,
   discoverable only by reading the source" data flow this whole audit exists to
   eliminate, not reproduce).
2. `istell` here selects a *complete named preset* (all ~25 fields at once), not a
   formula — this is `naming_convention.md`'s "switches are not ports" in its purest
   form: a topology/config decision made once, before any node runs, per
   `unit_registry.md`'s existing `i_cost_model` precedent.

**Recommended treatment** (not resolved here — flagging for the design decision):
replace the five dict literals + reflective loop with five **static, fully-enumerated**
config records (e.g. one `dataclass`/`NamedTuple` per machine, with a fixed field list
verified once against `StellaratorConfigData`, not filtered at runtime by `hasattr`),
selected by `istell` at stellarator-mode graph-assembly time — literal default *inputs*
to whichever nodes read `.stellarator_config.*`, not a node of their own. This is the
same recommendation `initialization.md`'s open question 2 makes for its 16 device-preset
literals; the two files should get one shared answer; a `preset_config.md`/
`initialization.md`-only decision would let a stellarator-mode default-input assembly
step be split across two unrelated-looking places for no reason.

Not ported: no pure function exists to write (see above) and no node to register in
`total_process.py`.

## cottax node
None — see "proposed signature(s)" above for why.

## tier signal
**N/A** — config/preset-table selection, not a computation. If forced into the
tier scheme: closer to tier-0 (static data) than tier 1, since there is no arithmetic
at all, only a lookup and a reflective copy.

## switches touched
`istell` — **second role, extending `switches.md`'s existing entry** (data-table
selection, same shape 1C already found; not a new switch, a second confirmed site for
the role 1C described). No local formula branch to add to the reads-set diff — see
`data footprint` above for why a per-branch reads-set table doesn't apply here the way
it does for an ordinary formula switch.

## calls into other models
None. `open`/`json.load` for `istell == 6` is file I/O, not a model call, but shares the
"external, non-traceable" concern — see JAX-difficulty flags.

## JAX-difficulty flags
- **File I/O** (`open(config_file)`, `json.load`) for `istell == 6` — `non-traceable-
  external-call`, `minor` severity (only reachable for a user-supplied custom machine,
  and it's a config-load-time operation, not something any traced computation depends
  on). Confirms chunk 1C's identical flag from the other call site.
- **Reflective `hasattr`/`setattr` loop** — not JAX-traceable in any useful sense (it's
  Python-level metaprogramming over field names, resolved once at config-load time, long
  before any array exists) — `not applicable to the pure rewrite target`, same category
  `density_limits.md` used for `copy.deepcopy`: evidence of *why* the config step needs
  a static replacement, not a pattern to carry into the port.

## open questions
1. **Silent field-name mismatches.** If a preset dict's key doesn't match any
   `StellaratorConfigData` field (typo, renamed field, stale preset), the value is
   dropped with no warning — worth checking, when the static replacement above is
   designed, that all ~25 keys in all 5 presets currently *do* match a real field (a
   mismatch today would be a silent data-loss bug in the original PROCESS source, not
   something this audit fixes, but worth surfacing since the static rewrite would need
   to decide whether to preserve that silence or raise on a mismatch).
2. Same decision as `initialization.md`'s open question 2: where do stellarator-mode
   default/preset literals live in the ported graph — recommend one shared answer for
   both files.
