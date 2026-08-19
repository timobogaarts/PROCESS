---
kind: model-unit
status: reviewed
confidence: high
---

## source
`process/models/stellarator/preset_config.py` (266 lines, full file in scope). Five
machine-preset dict literals (`HELIAS5B`, `HELIAS4`, `HELIAS3`, `W7X30`, `W7X50`, ~35
scalar fields each) plus one function, `load_stellarator_config(istell, config_file,
data)`.

Its **only** caller is `Stellarator.st_new_config()`
(`process/models/stellarator/stellarator.py:191-232`, first statement), which runs before
anything else in the stellarator pipeline. The rest of `st_new_config` — the `1 not in
ixc` aspect-ratio default, `rminor`/`eps`, the `n_tf_coils` overwrite, the seven
`f_st_*`/`r_coil_*`/`f_coil_shape` scaling factors, and nothing else (it ends at
`f_coil_shape`, line 275; `st_geom` is the next method) — is **already ported**, as
`DefaultAspectRatio`/`StellaratorScalingFactors` in
`stellarator_C_geometry.py`/`.md` (chunk 1C of unit #1). This unit is therefore the
last unported piece of `st_new_config`, and porting it is what closes it.

## data footprint

**Confirms and extends chunk 1C's finding** (`stellarator_C_geometry.md`): `istell`
plays a second role here beyond the top-level tokamak/stellarator pipeline split
(`switches.md`) — it selects one of five hardcoded machine-preset tables (`istell` 1-5),
or an externally-loaded JSON file (`istell == 6`). This is a **data-table selector, not
a formula switch**: none of the six branches differ in what they *compute* (nothing is
computed — each is a dict literal or a `json.load`), only in *which fixed table* is
selected.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.istell` | read | explicit-arg (switch, see above) | `match istell: case 1..6` |
| `.globals.output_prefix` | read (at the call site) | explicit-arg (switch-adjacent) | `st_new_config` builds `Path(f"{output_prefix}stella_conf.json")`; only consulted for `istell == 6` |
| `.stellarator_config.stella_config_*` | write, 35 fields | explicit-arg (see "resolved" below) | 34 numeric + `stella_config_name` (a `str`) |

The write set was originally recorded here as **not statically enumerable**, because the
function loops over `machine_config.items()` and writes
`stella_config_<key.lower()>` **only if `hasattr(data.stellarator_config, ...)`** — a
silent skip for any dict key that doesn't match a `StellaratorConfigData` field name, no
error, no log. That is still an accurate description of *PROCESS's* control flow, but it
is **not** an obstacle to declaring the port's outputs, and treating it as one was this
record's original mistake:

- **Resolved by inverting the loop.** PROCESS iterates the config's keys and asks which
  field each one lands on. The port iterates the *fields* — a fixed list,
  `STELLA_CONFIG_SCALAR_FIELDS`, transcribed once from
  `process/data_structure/stellarator_configuration.py` — and asks whether the config
  has a key for each. The two agree exactly on the intersection, which is all either one
  writes, and the inversion makes the write set knowable before the config is even
  chosen. `test_preset_config.py` keeps the transcription honest by reading the reference
  side back off a real `StellaratorConfigData` **by those same field names**, so a field
  renamed or added in `process/` fails the value test rather than drifting.
- **Measured, both directions, on all six configs PROCESS ships** (the five presets and
  `tests/regression/input_files/stellarator_helias.stella_conf.json`): every one of them
  supplies all 35 fields — the "unset field" half of the risk has no live instance. Three
  keys go the other way and are dropped; see the bug below.

## PROCESS bug found (documented, not fixed)

**`stellarator_helias.stella_conf.json` silently loses three of its keys**:
`number_nu_star`, `D11_star_mono_input`, `nu_star_mono_input`. They map to
`stella_config_number_nu_star` / `stella_config_d11_star_mono_input` /
`stella_config_nu_star_mono_input`, none of which is a `StellaratorConfigData` field, so
`hasattr` is false and the `setattr` never happens — no error, no warning. Nor is the
data picked up under any other name: `grep -rn "nu_star_mono\|D11_star_mono\|
number_nu_star" process/` matches **nothing at all**, while `neoclassics.py` computes its
own mono-energetic transport coefficients from a hardcoded `no_roots` grid. So a config
author supplying a precomputed 20-point mono-energetic transport table — which is
plainly what those three keys are — gets it discarded in silence and the run proceeds
with the built-in model.

Reproduced faithfully: `select_stellarator_config_scalars` drops exactly the same three
keys (both sides are compared by `test_preset_config.py`'s value test, which would fail
if either side kept one). `dropped_config_keys` makes the loss *inspectable* without
changing it. This answers open question 1 of this record's previous revision — the five
presets are clean, the reference JSON is not.

## proposed signature(s)

```python
def select_stellarator_config_scalars(machine_config: Mapping) -> tuple[float, ...]:
    # returns one value per entry of STELLA_CONFIG_SCALAR_FIELDS, in that order
    ...
```

Ported as `select_stellarator_config_scalars` in `preset_config.py` (this record's `.py`
sibling), together with two non-traced helpers: `read_stellarator_config_file` (the
`istell == 6` `json.load`) and `dropped_config_keys` (the bug above, made visible).

### The shape decision, argued

Three candidate shapes were on the table. The choice is **(2) with (1) supplying its
payload**, and the reasoning is a split along a line the audit already draws, not a
preference:

1. **`eqx.field(static=True)` graph-assembly facts on each consumer**, the way
   `ImpurityRadiationTotals.imp_indices` carries its index table. *Rejected for the
   values.* Thirty-four scalars read by a dozen different nodes would have to be
   duplicated into a dozen constructor calls in `total_process.py`; the machine being
   designed would exist nowhere as a thing, only as scattered literals, and the XDSM
   would show no reason why `.tfcoil.n_tf_coils` is 50. It also loses the property that
   made this worth doing at all — that the config→design dependency becomes an *edge*.
2. **One node with no inputs owning the 34 outputs.** *Chosen.* See below.
3. **Stay outside the graph and seed `env`** — the status quo. *Rejected, and this is the
   measurement that motivated the whole unit.* Left as boundary inputs, the 34 fields
   have no producer, so a graph stepped from a cold `SingleRun(...).data` reads
   `StellaratorConfigData`'s dataclass default of `0.0` for every one of them.
   `.tfcoil.n_tf_coils` is `stella_config_coilspermodule * stella_config_symmetry`, so it
   is `0`, and the first division by it poisons everything downstream. Measured by
   stepping the MDA schedule block by block from `SingleRun(input_file, "vmcon").data`
   with no PROCESS initialisation applied, on one tree, with and without this node:
   **1 block failing outright and 16 emitting non-finite values (of 135), against 0 and 1
   (of 136) with the node** — the single remaining one is `.physics.nu_star`
   (`DimensionlessPlasmaParameters`), a separate problem with no consumer in the graph.
   "Seed it from a converged run" is
   not a shape, it is the absence of one — it makes the port unable to start from an
   input file, which is exactly what `test_harness.md` § "Not built" lists as the missing
   validation path.

**Why the *selection* is static and the *values* are a node.** The two halves of
`load_stellarator_config` sit on opposite sides of the split rule in
`traceability_policy.md` § Switches:

- *Which* table or file is read changes **only values**, never the reads-set: all six
  branches write the identical 35 fields (measured, see above) and read nothing. By the
  rule's own criterion that is the **static** case, not the split case — so `istell`'s
  device-selection role becomes a static parameter (`machine_config`), and the file read
  it may imply happens once at assembly time, where `configuration.py` already argues
  every switch-shaped decision belongs. This is also what keeps `istell == 6`'s
  `non-traceable-external-call` (`open`/`json.load`) permanently outside any traced body:
  no `pure_callback`, no custom primitive, nothing deferred.
- The *values*, once selected, are ordinary graph variables that other nodes read. They
  are constants for a solve, but so is every boundary input; what distinguishes them is
  that a rule inside PROCESS produces them, and a rule with a producer is a node.

**The `Scan` question, taken seriously rather than waved at.** If a scan could sweep a
`stella_config_*` field, a static payload would be wrong — the sweep would have to
rebuild the graph per point. Two facts settle it, and neither is an assumption:
`process/core/scan.py`'s `ScanVariable` list contains no `stellarator_config` entry
(`configuration.py`'s docstring records the same grep for switches), and PROCESS's `Scan`
re-solves each point from scratch anyway, so even a hypothetical config sweep would get a
freshly assembled graph rather than a mutated one. A static payload therefore costs
nothing that is currently reachable, and re-assembling per point remains available if a
config sweep is ever wanted.

**Why one node and not 34.** They share a single provenance — one file, or one preset
table, chosen by one switch value. Splitting them into 34 source nodes would put 34
identical boxes in the XDSM carrying no information the one box doesn't, and would make
the "which machine is this?" question 34 places instead of one. The unit of selection in
PROCESS is the whole table; the node matches it.

**`stella_config_name` is excluded** from the node and from
`STELLA_CONFIG_SCALAR_FIELDS`. It is a `str`: not a graph variable, not comparable by the
harness's `_as_array`, and read by nothing but the output routines. Noted rather than
silently dropped, since it is the one field of the 35 that does not appear.

## cottax node

**Actually written**, in `preset_config.py`, registered in
`functional_process/total_process.py`:

```python
class StellaratorMachineConfig(ExplicitFunction):
    machine_config: tuple = eqx.field(static=True)

    stella_config_symmetry = Output(
        lambda s: s.stellarator_config.stella_config_symmetry
    )
    ...  # 34 in STELLA_CONFIG_SCALAR_FIELDS order

    def __call__(self):
        return select_stellarator_config_scalars(dict(self.machine_config))
```

**Registered under `TOPOLOGY_SWITCHES`'s `.stellarator.istell` switch, `value=6` arm** —
not in `COMMON`. It is genuinely `istell`-gated: `istell == 0` is a tokamak and has no
machine config at all, so an unconditional registration would make the tokamak graph
produce stellarator device data, which is the `EcrhDensityLimit` registration-bug class
`configuration.py` exists to prevent. `Switch.check_arms_are_exclusive` is satisfied by
the arm's existing `StellaratorConfinementTime`/`ConfinementTime` collision.

Arms 1–5 stay `unported`. That is now a **transcription** gap rather than a design one:
`select_stellarator_config_scalars` is generic over any mapping and
`test_preset_config.py` already drives all five presets through the reference, so wiring
them means adding five payloads to `total_process.py` (and deciding whether to transcribe
the tables into the port or import PROCESS's dicts). Deliberately left out of this pass —
`unit_registry.md`'s scope rule is `istell == 6`, and an `unported` arm refuses loudly
rather than assembling a wrong graph.

**Verified consequences** (`run_mda_harness.py`, same reference run): agreements
**+34** (453 → 487 on the tree this unit landed on; the 34 new outputs, all agreeing,
none disagreeing), disagreements **34 → 34** — unchanged, nothing moved — owned variables
walked **+34** (511 → 545), unaccounted **0**. `n_tf_coils`/`rminor`/`eps` were already
produced by chunk 1C, so their values did not move; what changed is that their *inputs*
now have producers instead of being seeded. `run_sand_harness.py` Stage A and Stage B are
**bit-identical** with and without the node (0 non-finite Jacobian cells, no new starred
cell) — expected, since a constant with no inputs carries no derivative and the seeded
values were the same numbers.

## tier signal
**Tier 1.** No internal iteration, no calls into other models, an explicit signature. Not
"tier 0 / not a tier", as this record previously said — that verdict followed from the
"not portable as a node" one and falls with it. There is no arithmetic, true, but the
harness's tier-1 checks are exactly the right ones: value agreement against PROCESS's own
reflective loop is what validates the static enumeration, on all six configs.

The gradient checks pass vacuously (`static_argnames = ("machine_config",)`, so
`diff_argnames` is empty). That is honest rather than a gap: the node has no inputs, so
there is no derivative to get wrong — the *reason* this unit is safe to freeze into
constants is the same reason there is nothing to differentiate.

## switches touched
`istell` — **second role, extending `switches.md`'s existing entry** (data-table
selection, same shape chunk 1C found; not a new switch, a second confirmed site for the
role 1C described). **Split decision: keep-static**, with the reads-set evidence
`traceability_policy.md` demands — all six arms read nothing and write the same 35
fields, so the branches differ only in value, which is the rule's stated exception rather
than a size-based deviation of the kind that section's own "six recorded instances"
paragraph is about. Resolved at graph-assembly time
(`total_process.py`'s `.stellarator.istell` switch), the `i_cost_model` /
`Models.costs` precedent.

## calls into other models
None. `open`/`json.load` for `istell == 6` is file I/O, not a model call — see
JAX-difficulty flags.

## JAX-difficulty flags
- **File I/O** (`open(config_file)`, `json.load`) for `istell == 6` —
  `non-traceable-external-call`, **resolved, not deferred**: `read_stellarator_config_file`
  runs at graph-assembly time and hands the node a static payload, so nothing traced ever
  touches the filesystem. No `pure_callback` or custom primitive is needed, and none
  should be added.
- **Reflective `hasattr`/`setattr` loop** — not JAX-traceable in any useful sense.
  **Resolved by replacement**: `STELLA_CONFIG_SCALAR_FIELDS` + the inverted loop, checked
  against PROCESS's reflective original by the unit's own value test on all six configs.
- **Integer-valued fields** (`stella_config_symmetry`, `stella_config_coilspermodule`)
  become float64 arrays in the graph, as every other port's integers do. `minor`: they are
  only ever multiplied (`n_tf_coils = coilspermodule * symmetry = 50`), never used as an
  index or a loop bound.

## open questions
1. **Should arms 1–5 be wired, and if so where do the five preset tables live** — in the
   port (transcribed, so the port has no dependency on `process/`'s data) or imported
   from `process.models.stellarator.preset_config` (no duplication, but the port's data
   and the reference's data become the same object, which weakens the value test to a
   tautology for those five samples)? The test module currently imports them, which is
   fine *for a test*; a registration would be a different call. Not needed for the
   `istell == 6` scope.
2. **`initialization.md`'s open question 2 is now answered by precedent, but not yet
   applied there.** Unit #6's 16 device-preset literals are the same shape as this unit's
   34 config scalars — "constants nothing produces" — and the answer this record reaches
   (give them one producer node rather than leaving them as unowned boundary inputs) is
   directly transferable. Whether it *should* transfer is a live question rather than a
   formality: those 16 are stellarator-mode overrides of ordinary input fields
   (`.physics.q95 = 1.03`, `.build.dr_cs = 0.0`), so a producer node for them would claim
   ownership of fields an `IN.DAT` also sets, which is not the situation here (nothing but
   this file ever writes a `stella_config_*`). Left to whoever picks up unit #6.
3. **A config key that matches no field is still dropped silently by the port**, because
   PROCESS drops it silently. `dropped_config_keys` exists so the loss is *findable*, but
   nothing calls it in anger. If the port ever becomes the primary path rather than the
   validated mirror, this should become a loud refusal at assembly time — a decision that
   belongs with whoever declares the port authoritative, not with this audit.
