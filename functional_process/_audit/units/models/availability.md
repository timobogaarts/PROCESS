---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `availability.py` / `test_availability.py`, 19 tier-1 contracts passing
(legacy + fuzz, value + `--fp-gradients`): the shared leaf helpers
(`calculate_dpa_per_fpy`, `calculate_divertor_lifetime`,
`calculate_cp_lifetime_superconducting`/`_resistive`, `calculate_u_unplanned_magnets`/
`_divertor`/`_fwbs`/`_bop`/`_hcd`/`_vacuum`, `calculate_blanket_lifetime_fpy_avail`/
`_simple`, `calculate_u_planned`, `calculate_ward_taylor_availability`) and all three
top-level branches (`calculate_avail`, `calculate_avail_2`, `calculate_avail_st`).
`calculate_redun_vac` is ported but deliberately has no harness contract (plain Python,
not `jnp`-traceable — see "JAX-difficulty flags").

**Node split done this wave (`next_steps.md` §5, Shape B).** `Avail`/`Avail2`/`AvailSt`
used to each read and own `.costs.cplife` in one node, which `to_graph` refused outright
(`ValueError: reads ['.costs.cplife'], which it also owns`). Each is now split into a
tiny `CplifeAvail`/`CplifeAvailSt` `FixedPointFunction` (owns `.costs.cplife`) plus an
ordinary `ExplicitFunction` for the branch's other outputs (reads it). `to_graph`
succeeds on every one of the five resulting node classes, standalone and combined — see
"cottax node" below for the exact shape and `test_availability.py`'s "Graph assembly"
section for the tests that prove it. **Still not registered in `total_process.py`** —
that stays a later, separate consolidation step, per this wave's scope.

## source

`process/models/availability.py`, full file (1593 lines) in scope: `Availability.run()`
(the switch dispatch), `avail()` (119-409), `avail_2()` (410-581) plus its six helper
methods `calc_u_planned`/`calc_u_unplanned_magnets`/`_divertor`/`_fwbs`/`_bop`/`_hcd`/
`_vacuum` (582-1227), `avail_st()` (1229-1536), and the two shared helpers
`cp_lifetime()`/`divertor_lifetime()` (1537-1593). `output()` bodies (the `if output:`
blocks writing to `process_output`) are not ported, per every prior unit's convention.

## The `itart`/`avail_st` reachability question — findings

`unit_registry.md`'s row #17 calls `avail_st`'s dead-on-the-stellarator-path status "a
strong hypothesis, not structurally proven" because it hard-requires `physics.itart == 1`
(checked in `Availability.run()`, lines 100-109 — raises `ProcessValueError` otherwise).
This wave's `hcpb.py` port separately found `itart` is an ordinary, ungated
`InputVariable`. Traced properly, both halves of that hypothesis turn out to need
correcting, in different directions:

1. **`itart == 1` is not refuted by anything in PROCESS's own input validation.**
   Searched `process/core/init.py` (the post-parse consistency-check pass,
   `check_input_error`/`set_device_type` and everything between), `process/core/input.py`
   (`itart`'s `InputVariable` declaration: `InputVariable("physics", int, choices=[0,
   1])`, no `istell`-conditioned constraint), and every stellarator-specific module
   (`process/models/stellarator/*.py`, `stellarator/preset_config.py`'s
   `load_stellarator_config`) for any `istell`/`itart` cross-check. **None exists.**
   `init.py` has a comparable cross-check for a *different* pair (`istell == 0 and
   ConfinementTimeModel(...).mode == STELLARATOR` raises "Stellarator confinement time
   scaling cannot be used for a tokamak", lines 1217-1224) — proving PROCESS *does* have
   the mechanism for this kind of check where its authors thought to add one. No
   equivalent exists for `istell`/`itart`. A stellarator `IN.DAT` setting `itart = 1` is
   syntactically and semantically legal input as far as PROCESS's own validation is
   concerned — physically nonsensical (a stellarator has no centrepost), but not
   rejected.

2. **But `itart == 1` is the wrong question — the real gate is which call path reaches
   `Availability.run()`'s switch dispatch at all**, and this is where the registry's
   framing needs to be sharpened, not just weakened. `process/core/caller.py`'s
   `Caller._call_models_once` (the function actually driving every solver iteration)
   dispatches on `istell` at the very top (lines 272-275): `if istell != 0:
   self.models.stellarator.run(); return`. Availability's own `.run()` — the function
   with the `i_plant_availability` switch — is only reached in the **tokamak** branch of
   `_call_models_once` (line 382, `self.models.availability.run()`), which a stellarator
   run's early `return` never reaches. `Stellarator.run(output=False)` (the path executed
   on *every* solver iteration) instead calls `self.availability.avail(output=False)`
   **directly** (`stellarator.py` line 175) — bypassing `Availability.run()`'s switch
   entirely. The source itself flags this as unresolved: `# TODO: should availability.run
   be called rather than availability.avail?` (lines 173-174). **The practical
   consequence: for a stellarator run, `i_plant_availability` has no effect at all during
   the solve loop — `avail()` runs unconditionally, regardless of the switch's value.**
   `avail_2`/`avail_st` genuinely never execute during optimisation on the stellarator
   pipeline, independent of `itart`.
3. **`avail_st` (and `avail_2`) *are* reachable once, at final output.**
   `Stellarator.output()` → `Stellarator.run(output=True)` calls
   `self.availability.run(output=True)` (line 128) — the **full** switch dispatch, not
   the direct `avail()` call. This only runs once, after the solve has already converged
   using `avail()` throughout (`main.py`'s `Models.write()`, called post-solve for the
   final `OUT.DAT`). So: a stellarator `IN.DAT` with `i_plant_availability = 3` and
   `itart = 1` — a legal, if physically odd, combination — reaches `avail_st` exactly
   once, in the final report-writing pass, having never influenced the actual
   optimisation. `i_plant_availability = 2` similarly reaches `avail_2` at the same point,
   with no `itart` requirement at all (only `avail_st` demands `itart == 1`; `avail_2` has
   no such gate).
4. **Same pattern, independently, on the IFE path**: `process/models/ife.py`'s
   `IFE.run()` (lines 101-104) calls `self.availability.avail(output=output)` — literally
   the same code shape and the same `# TODO: should availability.run be called rather
   than availability.avail?` comment, copy-pasted. Unlike stellarator, IFE has **no**
   corresponding "call `Availability.run()` at final output" path (`ife.py` never calls
   `Availability.run()`), so `avail_2`/`avail_st` are provably 100% unreachable for IFE —
   supporting evidence that this `avail()`-bypass shape is a real, repeated pattern in
   PROCESS's device-type dispatch, not a one-off.

**Conclusion**: `avail_st` is not "provably dead on the stellarator path" — it is
reachable, but only in the output-writing pass, never during optimisation, and only under
an unusual (not forbidden) switch combination nobody appears to test
(`tests/unit/models/stellarator/test_stellarator.py` was checked; it does not exercise
`i_plant_availability` at all). Both `avail_2` and `avail_st` are ported in full rather
than left audit-only, since (a) they are reachable and self-contained (no calls into
unported models), and (b) leaving genuinely-callable code unaudited would be a worse gap
than the extra port cost. Whoever designs `total_process.py`'s `Switch` wiring for
`i_plant_availability` should treat "reached only at output time, not during the solve"
as a real property of this switch's stellarator-relevant slice, not an oversight to
paper over — a graph built purely for the *solve* would not need `Avail2`/`AvailSt`
wired in for a stellarator configuration at all; a graph reproducing PROCESS's *complete*
behaviour (including final-report values) would.

## `i_plant_availability`'s dispatch shape

The three branches **do not share one clean interchangeable output-`VarPath` set** — the
shape is a partial overlap, not a triple alternative with identical ports:

| output | `avail` | `avail_2` | `avail_st` |
|---|---|---|---|
| `.fwbs.life_blkt_fpy` | yes | yes | yes |
| `.costs.life_div_fpy` | yes | yes | yes |
| `.costs.life_hcd_fpy` | yes | yes | yes |
| `.costs.cplife` | yes (conditional on `itart`, see below) | yes (conditional on `itart`) | yes (unconditional — see below) |
| `.costs.f_t_plant_available` | yes, but **only when `i_plant_availability == 1`** — see below | yes, always | yes, always |
| `.costs.cpfact` | yes | yes | yes |
| `.costs.bktcycles` | **yes** | no | no |
| `.costs.t_plant_operational_total_yrs` | **no** (field not touched at all) | yes | yes |
| `.costs.redun_vac` | **no** | yes | yes |

`avail` is a **strict subset** of what `avail_2`/`avail_st` produce, missing
`t_plant_operational_total_yrs`/`redun_vac` entirely (no operational-time or
redundant-pump concept in the 1999 model) and uniquely owning `bktcycles` (a "number of
fusion cycles to reach allowable DPA" figure the other two models don't compute at all).
`avail_2` and `avail_st` are much closer to each other, but not identical: `avail_st`
additionally reads `.costs.u_unplanned_cp` (a centrepost term with no counterpart in
`avail_2`) and computes several centrepost-only locals with no `VarPath`
(`maint_cycle`/`n_cycles_main`/`n_centre_cols`) that `avail_2` has no equivalent of.

**Within `avail()` itself there is a second, cleaner switch that fits the clean shape
exactly**: `.costs.f_t_plant_available` for `i_plant_availability == 0` (USER_INPUT) has
**no producer at all** — the source never touches it on that branch, simply leaving the
input value in place. This is precisely cottax's "no `InputNode`": a boundary variable is
one nothing owns. The port makes this structural by splitting `avail()` into
`calculate_avail` (the common tail, taking `f_t_plant_available` as a plain input) and
`calculate_ward_taylor_availability` (the `i_plant_availability == 1`-only producer of
that one slot) — see "cottax node" below. This *is* the clean "switch selects which
producer, if any, of one slot" shape the dispatch documentation asked to identify; the
outer `avail`/`avail_2`/`avail_st` split is not that shape, and forcing it into one would
misrepresent the source.

**Recommendation for `total_process.py`'s eventual wiring**: `i_plant_availability`
should be a **topology-changing switch** selecting one of `{Avail, Avail2, AvailSt}` as a
whole (three mutually exclusive alternative producers of the *union* of their outputs,
with `avail`'s missing `t_plant_operational_total_yrs`/`redun_vac` and
`avail_2`/`avail_st`'s missing `bktcycles` needing an explicit "not produced by this
alternative" policy — the same open question `heating.md`'s open question 2 already
flagged for `EcrhHeating`/`LowhybHeating`'s partially-overlapping ports, generalised to a
third case with a genuinely different, not just smaller, output set). The
`f_t_plant_available` ownership split inside `avail()` is a second, independent switch
(`i_plant_availability == 0` vs `1`) nested inside the `avail`/USER_INPUT-or-WARD_TAYLOR
alternative, not flattened into the outer three-way choice.

## data footprint

Grouped by function. `implicit-io`/`conditional-ownership-by-run-config` entries are the
interesting ones; straightforward `explicit-arg` reads/writes are collapsed into the
function signatures documented in "proposed signature(s)" rather than repeated here.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.costs.i_plant_availability` | read | explicit-arg (switch) | `Availability.run()`'s dispatch; not read inside `avail`/`avail_2`/`avail_st` themselves except once more inside `avail()` (see below) |
| `.costs.i_plant_availability` | read (again) | explicit-arg (switch) | inside `avail()`, a *second* read selects USER_INPUT (0) vs WARD_TAYLOR (1) — see "dispatch shape" above; ported as two separate node alternatives, not folded into the outer switch |
| `.physics.itart` | read | explicit-arg (switch), gates ownership | `avail()`/`calc_u_planned` (`avail_2`'s helper): `.costs.cplife` is only *computed* when `itart == 1` — `conditional-ownership-by-run-config`, same shape as `stellarator_C_geometry.md`'s `.physics.aspect` finding. Ported by threading a `cplife_in` passthrough argument (see "proposed signatures"), not by resolving the ownership question in this port |
| `.physics.itart` | read (second site) | explicit-arg (switch) | separately gates the later *lifetime-adjustment* step in all three branches (`if itart == 1 and cplife < life_plant: cplife = min(...)`) — a **different** gate from the ownership one above: `avail_st` computes `cplife` **unconditionally** (line 1289, no `itart` check) and only this second gate is `itart`-conditioned there. The two `itart` reads are not interchangeable and are documented separately in the port's docstrings |
| `.tfcoil.i_tf_sup` | read | explicit-arg (switch) | `cp_lifetime()`'s SC-vs-resistive branch; both branches nontrivial (unlike `i_tf_sup` in `stellarator_F_tf_nuclear_heating.md`, where the alternative was all-zero) — split into two separate node alternatives, `calculate_cp_lifetime_superconducting`/`_resistive` |
| `.costs.ibkt_life` | read | explicit-arg (switch) | blanket-lifetime model choice, appears in three places (`avail`'s 4-way version, `calc_u_planned`'s and `avail_st`'s identical 2-way version) — kept as a static kwarg (Python `if`), not split into node alternatives (see "switches touched") |
| `.fwbs.life_fw_fpy` | read | explicit-arg, data-dependent (not a switch) | `avail()`'s blanket-lifetime block branches on `life_fw_fpy < 0.0001` — this **is** a genuine data-dependent condition (a continuous field, not a switch), ported with `jnp.where`, not a static branch. Not read at all by `calc_u_planned`/`avail_st`'s copy of the formula (equivalent to it being permanently the "unset" case there) |
| `.ife.ife` | read | explicit-arg (switch), gates the whole lifetime block | `avail`/`avail_2`/`avail_st` each gate their entire lifetime-computation section on `ife.ife != 1`. **Out of scope for this port** (stellarator-only per `test_harness.md`'s Scope note; `ife.ife` is never touched by `Stellarator.run()`) — `calculate_avail`/`_2`/`_st` implement only the `ife.ife != 1` path |
| `.costs.f_t_plant_available` | read (input value, `i_plant_availability == 0`) / write (`== 1`, or always in `avail_2`/`avail_st`) | conditional-ownership-by-run-config | see "dispatch shape" — the cleanest single finding in this record |
| `.costs.cplife` | read/write, multiply-conditional | conditional-ownership-by-run-config | see the two `itart`-gate rows above |
| `.costs.bktcycles` | write | explicit-arg | `avail()` only; not produced by `avail_2`/`avail_st` at all |
| `.costs.t_plant_operational_total_yrs` | write | explicit-arg | `avail_2`/`avail_st` only; not produced by `avail()` — retains whatever `data` already held if `avail()` was the selected branch |
| `.costs.redun_vac` | write | explicit-arg, **static downstream** | `avail_2`/`avail_st` compute it (`math.floor(...)`) then immediately use it as a Python `range()` bound in `calc_u_unplanned_vacuum` — see "JAX-difficulty flags" |
| `.vacuum.n_vac_pumps_high` | read | explicit-arg, **static** | genuinely `int`-typed (`vacuum_variables.py: n_vac_pumps_high: int = 0`), feeds the same `range()` bound |
| `.costs.u_planned`, local `u_unplanned`, `n_cycles_main`, `n_centre_cols`, `maint_cycle` | n/a | **not PROCESS storage** | kept as local Python variables in the source, never written to `data` (confirmed: no `self.data.costs.u_planned = ...`/etc. anywhere in `availability.py`) — the ported functions still return them (useful for direct unit testing) but no cottax `Output`/`VarPath` binds them, matching `stellarator_F_tf_nuclear_heating.md`'s precedent for a function returning more than `data` stores |
| `.costs.u_unplanned_cp` | read | explicit-arg | `avail_st` only, added straight into its unplanned-unavailability sum; **not computed by this unit** — produced elsewhere (a centrepost structural/neutronics model, out of registry scope), so it's a genuine boundary input for the `AvailSt` node as ported |
| `.costs.redun_vacp`, `.divertor.pflux_div_heat_load_mw`, `.costs.adivflnc`, `.costs.abktflnc`, `.physics.pflux_fw_neutron_mw`, `.costs.life_dpa`, `.physics.p_fusion_total_mw`, `.costs.life_plant`, `.costs.num_rh_systems`, `.tfcoil.temp_*_margin_min`, `.costs.conf_mag`, `.tfcoil.temp_margin`, `.costs.div_*`, `.costs.fwbs_*`, `.times.t_plant_pulse_total`, `.times.t_plant_pulse_burn`, `.costs.tmain`, `.costs.t_div_replace_yrs`, `.costs.t_blkt_replace_yrs`, `.costs.tcomrepl`, `.costs.uu*`, `.fwbs.neut_flux_cp`, `.constraints.flu_tf_neutron_fast_max`, `.costs.cpstflnc` | read | explicit-arg | ordinary parameters, one per function as documented in each `calculate_*`'s docstring in `availability.py` |

No `redundant-duplicate-write` found in this file.

## proposed signature(s)

Eighteen functions, in `functional_process/models/availability.py`; see each one's own
docstring for the full parameter list (repeating the ~15-28-argument signatures here
would just be the file's table of contents, not new information):

- `calculate_dpa_per_fpy`, `calculate_divertor_lifetime`,
  `calculate_cp_lifetime_superconducting`, `calculate_cp_lifetime_resistive` — leaf
  helpers shared by two or three branches.
- `calculate_u_unplanned_magnets`, `_divertor`, `_fwbs`, `_bop`, `_hcd`, `_vacuum` —
  the six unplanned-unavailability terms, shared by `avail_2`/`avail_st`.
- `calculate_redun_vac` — plain Python, not `jnp` (see JAX-difficulty flags).
- `calculate_blanket_lifetime_fpy_avail` (avail's own 4-way version, with the
  `pflux_fw_neutron_mw == 0` guard — see PROCESS-bug note below) and
  `calculate_blanket_lifetime_fpy_simple` (the 2-way version shared by `calc_u_planned`/
  `avail_st`, **without** that guard).
- `calculate_u_planned` — `avail_2`'s planned-unavailability model.
- `calculate_ward_taylor_availability` — `avail()`'s `i_plant_availability == 1`-only
  producer of `.costs.f_t_plant_available`.
- `calculate_avail`, `calculate_avail_2`, `calculate_avail_st` — the three top-level
  branches, composed from the above.

## cottax node

Written in `availability.py`. Per the rule that a node must own at least one variable
(`~/jaxgraph/CLAUDE.md`), only functions whose *entire* return tuple maps onto real
`data` storage get a node: `CpLifetimeSuperconducting`/`CpLifetimeResistive` (mutually
exclusive alternatives for `.costs.cplife`, same shape as `i_tf_sup` in
`stellarator_F_tf_nuclear_heating.py`), `WardTaylorAvailability` (exists only when
`i_plant_availability == 1`), and `Avail`/`Avail2`/`AvailSt` (one node per branch,
matching PROCESS's own granularity — nothing outside `Availability` calls
`divertor_lifetime`/`calc_u_planned`/etc. independently, so atomising further would
misrepresent the actual call graph). The six `calculate_u_unplanned_*` helpers,
`calculate_u_planned`, and the two blanket-lifetime functions are tested directly
(tier-1 contracts) but have **no standalone node** — their outputs (`u_unplanned_magnets`,
`u_planned`, etc.) have no `VarPath`, only the composite branch nodes' *aggregated*
outputs do. `calculate_redun_vac` has no node at all (see JAX-difficulty flags).

`Avail`/`Avail2`/`AvailSt` declare `ibkt_life`/`itart` (and, for `Avail2`/`AvailSt`,
`n_vac_pumps_high`/`redun_vac`) as `eqx.field(static=True)` class fields, following
`EcrhDensityLimit`'s precedent (`stellarator/density_limits.py`) for a switch resolved
as a static kwarg rather than a graph `From` read.

**`.costs.cplife` is split out of `Avail`/`Avail2`/`AvailSt` entirely — resolved this
wave, previously blocking (`next_steps.md` §5, Shape B).** Each of the three branch
node classes used to declare `.costs.cplife` as *both* a `From` read and an `OutputInto` write
(`Avail`/`Avail2` via a `cplife`/`cplife_in` pair; `AvailSt` via a single bare `cplife`
pair) — a genuine single-node self-loop, since real `avail()`/`avail_2()`/`avail_st()`
each read-then-(conditionally-or-unconditionally)-rewrite `.costs.cplife` within one
call. `to_graph(Avail(...))` raised `ValueError: reads ['.costs.cplife'], which it also
owns` directly from `cottax.spec`'s `__check_init__` — not representable as a plain
node at all, independent of any driving decision. Fixed per `next_steps.md` §5's
Action, using `cottax.interfaces.pytree_namespace_module.FixedPointFunction` (already
implemented; no new cottax primitive needed):

- **`CplifeAvail`** — a `FixedPointFunction`, shared by `Avail` and `Avail2` (their
  `itart == 1` cplife-adjustment formula is identical, confirmed by direct comparison of
  `calculate_avail`'s and `calculate_avail_2`'s `itart == 1` blocks). `step` ->
  `calculate_cplife_next`: `itart != 1` returns the current `.costs.cplife` value
  unchanged (a real, if trivial, fixed point — `avail()`/`avail_2()` never touch the
  field on that branch at all); `itart == 1` recomputes from scratch via whichever
  `calculate_cp_lifetime_superconducting`/`_resistive` alternative a **static**
  `i_tf_sup` field selects, then applies the lifetime adjustment
  (`calculate_cplife_lifetime_adjustment`, a new free-standing duplicate of the formula
  already inlined three times in `calculate_avail`/`_2`/`_st` — kept a duplicate rather
  than a shared extraction so those three functions, and their existing harness
  contracts, stay byte-for-byte untouched). `to_graph(CplifeAvail(...))` alone is
  **cyclic by construction** (`is_acyclic is False`) — the body genuinely reads the real
  `.costs.cplife` on the pass-through branch, same shape as
  `physics_B_composition.py`'s `plasma_composition`/`first_call` precedent.
- **`CplifeAvailSt`** — a separate `FixedPointFunction` (not shared with `CplifeAvail`):
  `avail_st()` computes `.costs.cplife` **unconditionally** (the two `itart` gates are
  not the same gate reused — see the data-footprint table above), so its `step` ->
  `calculate_cplife_avail_st_next` never reads a previous `.costs.cplife` value at all,
  only the genuine recompute inputs. `to_graph(CplifeAvailSt(...))` alone is
  **acyclic** (`is_acyclic is True`) — a degenerate `FixedPoint` that converges in
  exactly one iteration regardless of its starting guess, still correctly represented as
  a `FixedPointFunction` for the structural reason given below, not because it needs
  iterating.
- **`Avail`/`Avail2`/`AvailSt` themselves** are now ordinary `ExplicitFunction`s over
  each branch's *other* outputs only — `.costs.cplife` is no longer one of their
  declared `Output`s. `Avail`/`Avail2` read it back as a plain current-value `From` read;
  inspection of `calculate_avail`'s/`calculate_avail_2`'s bodies shows this value is
  **provably inert** for every output either node still declares (`cplife`/`cplife_in`
  feed only the now-discarded `cplife_mod` return slot) — kept as a real `From` read anyway
  for documentation fidelity, not because the graph needs it. **`AvailSt` cannot do the
  same** and does *not* read `.costs.cplife` at all: its `shortest_lifetime` (hence
  `maint_cycle`/`u_planned`/`t_plant_operational_total_yrs`/every unplanned-
  unavailability term/`f_t_plant_available`/every `*_mod` output it declares) needs the
  *pre*-adjustment cplife value, which is a different number from the *post*-adjustment
  value `CplifeAvailSt` owns whenever `itart == 1` and the adjustment actually applies
  (`cplife / f_t_plant_available != cplife` in general) — so `AvailSt` instead
  recomputes that same pre-adjustment value inline via `calculate_cp_lifetime_
  superconducting`/`_resistive` (a **new** static `i_tf_sup` field on `AvailSt` itself),
  matching what `test_availability.py::TestAvailSt`'s own `ported` adapter already did
  before this split.
- **Registering `CpLifetimeSuperconducting`/`CpLifetimeResistive` alongside
  `CplifeAvail`/`CplifeAvailSt` is an open question, deliberately not resolved here**
  (see "open questions" below) — both `i_tf_sup`'s branch inside `CplifeAvail`/
  `CplifeAvailSt` and the standalone `CpLifetimeSuperconducting`/`CpLifetimeResistive`
  nodes independently want to own `.costs.cplife`, and only one producer of one
  `VarPath` may exist in any graph that registers both together. `CplifeAvail`/
  `CplifeAvailSt` duplicate the two-line SC/resistive dispatch inline instead of
  consuming those two nodes' outputs, sidestepping the conflict rather than resolving
  it — `total_process.py`'s eventual wiring is where that resolution belongs.

`to_graph` succeeds on every one of the five node classes above (standalone, and for one
representative pair each — `CplifeAvail`+`Avail`, `CplifeAvailSt`+`AvailSt` — combined
with no ownership conflict), confirmed by `test_availability.py`'s "Graph assembly"
section. **None of the five is registered in `total_process.py`** — that remains a
later, separate consolidation step; this split only makes the nodes representable.

## tier signal

**All 21 functions are tier 1** — explicit, no internal iteration, no `scipy.optimize`/
`fsolve`, no calls into any other `Model`. `avail`/`avail_2`/`avail_st` are compositions
of already-ported tier-1 functions with no solver introduced at the composition level
(same treatment as `heating.md`'s common-tail functions, kept tier-1 rather than deferred
to the not-yet-built tier-3 machinery). `calculate_cplife_lifetime_adjustment`/
`calculate_cplife_next`/`calculate_cplife_avail_st_next` (the new Shape B split's step
functions) are the same shape — explicit compositions of already-ported tier-1
functions, no new solver.

## switches touched

- `.costs.i_plant_availability` (values 0-3) — **split**, into three whole-branch
  alternatives (`Avail`/`Avail2`/`AvailSt`) **plus** a nested split inside `avail()`
  itself (0 vs 1, for `.costs.f_t_plant_available`'s producer) — see "dispatch shape".
- `.physics.itart` — **kept static** (`eqx.field(static=True)`), not split into node
  alternatives, despite gating real ownership questions (see the two data-footprint
  rows above). Reason: unlike `i_tf_sup`/`i_plant_availability`, splitting `itart` would
  require the *rest* of `calculate_avail`/`_2`/`_st` (six-plus outputs, not one slot) to
  exist as two near-duplicate node sets differing only in whether `cplife` is threaded
  through or recomputed — a much bigger duplication cost for a switch that, on the
  stellarator pipeline specifically, is itself an unusual, untested input (see the
  reachability finding above). Flagging for whoever designs the general
  `Switch`/`Alternative` mechanism: this is a case where "keep-static" was chosen over
  "split" primarily because splitting's cost was disproportionate to the switch's
  actual traffic on this pipeline, not because the two behaviours are actually the same
  (they aren't — see the conditional-ownership rows).
- `.costs.ibkt_life` — **kept static**. Both branches produce one scalar
  (`life_blkt_fpy`) via different formulas reading disjoint input subsets; a Python `if`
  costs nothing here that a node split would improve, unlike `i_tf_sup`'s case where the
  two branches read from genuinely different subsystems (superconducting-magnet vs.
  resistive-magnet fields) that a reader benefits from seeing as separate nodes.
- `.tfcoil.i_tf_sup` — **split**, at two different granularities now (see "cottax
  node"): as the top-level `CpLifetimeSuperconducting`/`CpLifetimeResistive` alternative
  pair, *and*, independently, as a static field duplicated inline inside
  `CplifeAvail`/`CplifeAvailSt`'s own recompute (necessary, not redundant — see "cottax
  node" for why those two can't simply consume the top-level pair's output).
- `.ife.ife` — **resolved above this file, for this port's scope**: never touched by
  `Stellarator.run()`, so effectively fixed at its default (0) on the stellarator
  pipeline. Not implemented as a graph switch at all here (see the `ife.ife` data-footprint
  row); a tokamak/IFE port would need to.

## calls into other models

None. `avail`, `avail_2`, `avail_st` and every helper call only other methods of
`Availability` itself — confirmed by reading the full 1593-line file; the only
cross-model reference is `.costs.u_unplanned_cp`, a plain data read with no method call
attached (see data footprint).

## JAX-difficulty flags

- **`calculate_redun_vac` is fundamentally non-traceable — `blocker`, by design, not
  worked around.** It computes `math.floor(n_vac_pumps_high * redun_vacp / 100.0 +
  0.5)`, and its result becomes a Python `range()` bound inside
  `calculate_u_unplanned_vacuum` (the cryopump-redundancy combinatorial sum,
  `math.comb(total_pumps, n)` for `n` in `range(redun_vac + 1, total_pumps + 1)`). Under
  `jax.jacfwd`, `math.floor`/`range()` on a traced value raises immediately (a tracer has
  no concrete `int()`/comparison). This is not a corner the port can smooth over: the
  quantities involved (`n_vac_pumps_high: int`, `redun_vac: int`) are genuinely
  integer-typed in PROCESS's own dataclasses, so the "problem" is real, not an artifact
  of the port's arithmetic. **Resolution**: `calculate_redun_vac` stays plain Python (no
  `jnp`), has **no harness contract** (would fail `test_gradient_finite`/
  `test_gradient_agreement` outright, and the harness's `Tier1Contract` assumes `ported`
  is `jacfwd`-able), and is instead exercised indirectly — every `TestAvail2`/`TestAvailSt`
  sample's `redun_vac` value is computed by calling it directly, so a value regression
  there would surface downstream. `calculate_u_unplanned_vacuum` itself declares
  `n_vac_pumps_high`/`redun_vac` `static_argnames`, consistent with `switches are not
  ports` generalised to "a value that becomes another function's loop bound is not a
  port either." This is the same class of gap `~/jaxgraph/CLAUDE.md`'s "not everything
  is JAX-traceable" difficulty already names for CoolProp — a structural boundary of the
  tracing model, not a bug.
- **`jnp.where`-selected divisions guarded against NaN-gradient leakage —
  `workaround-known`.** Several branches select between two formulas at a boundary where
  the *unselected* one's division is exactly zero (`calc_u_unplanned_magnets`'s
  `start_of_risk - tmargmin`, `calc_u_unplanned_divertor`/`_fwbs`'s `n` at the
  `below`/`above` edges, `cp_lifetime`'s SC branch at `neut_flux_cp == 0`,
  `calculate_blanket_lifetime_fpy_avail`'s `pflux_fw_neutron_mw == 0` sub-case). Each is
  ported with the standard double-`jnp.where` pattern (substitute a safe denominator
  inside the unselected branch, then `jnp.where` on the value) so `test_gradient_finite`
  passes rather than leaking a `0/0` NaN through the chain rule — this is exactly the
  failure mode `heating.md`'s worked example (`stop_gradient`, 10 failing gradient tests)
  demonstrated the harness catches.
- **`math.ceil`/`math.floor`/`round` on continuous physical quantities —
  `minor`.** `calc_u_planned`'s `n = ceil(lifetime_longest/lifetime_shortest) - 1`,
  `avail_st`'s `n_centre_cols = ceil(n_cycles_main)`, `calc_u_unplanned_bop`'s
  `bop_num_failures = ceil(...)`, `calc_u_unplanned_vacuum`'s `n_shutdown = round(...)` —
  ported with `jnp.ceil`/`jnp.round` (0 gradient almost everywhere, which *is* the
  correct derivative away from an integer boundary; PROCESS's own finite-difference
  reference sees the same discontinuity). Sample points are chosen away from these
  integer boundaries (matching `heating.md`'s guard-band precedent), so no gradient-kink
  failures were observed; a fuzz draw landing exactly on a boundary would be a shared
  false-positive risk for the reference too, not specific to this port.
- **Unguarded `abktflnc / pflux_fw_neutron_mw` in `calc_u_planned`/`avail_st`'s
  blanket-lifetime formula — see PROCESS-bug note below (not a JAX-specific issue, kept
  as-is).**

## PROCESS bugs found (documented, not fixed)

1. **`avail_2`/`avail_st`'s total-availability formula uses a different sign than
   `avail()`'s.** `avail()`'s WARD_TAYLOR block (line 250-252) computes
   `f_t_plant_available = 1.0 - (uplanned + uutot - uplanned * uutot)` — the standard
   inclusion-exclusion combination for "neither planned nor unplanned downtime happens"
   (`P(A ∪ B) = P(A) + P(B) - P(A)P(B)`). `avail_2` (line 474-476) and `avail_st` (line
   1362-1364) both instead compute `max(1.0 - (u_planned + u_unplanned + u_planned *
   u_unplanned), 0.0)` — **`+`, not `-`**, for the cross term. This is not a typo found
   by inspection alone — it was caught by the harness itself: an early draft of the port
   used `-` uniformly (copying `avail()`'s formula into `calculate_avail_2`/
   `calculate_avail_st` by analogy) and `TestAvail2`'s legacy sample (adapted directly
   from `tests/unit/models/test_availability.py::test_avail_2`, which mocks the six
   `calc_u_unplanned_*` returns to fixed values `0.02..0.07` and `calc_u_planned` to
   `0.01`) failed value agreement by exactly `2 × u_planned × u_unplanned` (`0.7227`
   ported vs `0.7173` PROCESS) once cross-checked directly against the real
   `Availability.avail_2()`. Fixed in the port to match PROCESS's actual (inconsistent)
   behaviour — **not** changed to the mathematically-standard `-` form, since the port's
   job is to reproduce PROCESS, not correct it. Worth a maintainer's attention: either
   `avail()`'s formula or `avail_2`/`avail_st`'s is wrong relative to the other, and
   nothing in the source comments explains the discrepancy.
2. **`calc_u_planned`/`avail_st`'s blanket-lifetime formula (`ibkt_life == 0` branch)
   divides by `pflux_fw_neutron_mw` with no zero-guard**, unlike `avail()`'s own
   near-identical block, which explicitly special-cases `pflux_fw_neutron_mw == 0.0`
   (returning `life_plant` instead of dividing) in the corresponding sub-branch
   (`avail()` lines 161-175, ternary at 167). `calc_u_planned` (lines 624-628) and
   `avail_st` (lines 1275-1279) both perform the division unconditionally. If
   `pflux_fw_neutron_mw` legitimately reaches exactly `0.0` with `ibkt_life == 0`
   (plausible if a neutron-wall-load calculation underflows, or simply at an early,
   unconverged iterate), real PROCESS would raise `ZeroDivisionError` from `avail_2`/
   `avail_st` but not from `avail()`. The port reproduces this asymmetry faithfully
   (`calculate_blanket_lifetime_fpy_avail` guards, `calculate_blanket_lifetime_fpy_simple`
   does not) rather than silently protecting the unguarded copies — a traced port
   naturally turns the would-be crash into a non-finite value instead of raising, which
   is the harness's standard convention for domain edges (`reference_domain_errors`),
   not a behaviour change.
3. **`avail_2`/`avail_st`'s "Modify lifetimes" step unconditionally divides by
   `.costs.f_t_plant_available`, which is only clamped, not bounded away from zero.**
   `f_t_plant_available = max(1.0 - (u_planned + u_unplanned + u_planned *
   u_unplanned), 0.0)` (lines 474-476/1362-1364) can legitimately equal exactly `0.0`
   whenever the summed unplanned+planned unavailability reaches or exceeds 1 — not a
   rare or contrived corner: fuzzing this port (bounded scans of the real iteration-
   variable-adjacent inputs, no adversarial search) reproduced a genuine PROCESS
   `ZeroDivisionError` from `avail_st` on a sizeable minority of draws once
   `avail_st`'s own `shortest_lifetime`/`tmain` maintenance-cycle ratio was allowed to
   get small (`u_planned = tmain / (shortest_lifetime + tmain)` climbs toward 1 fast
   as any one of blanket/divertor/centrepost lifetime approaches `tmain`). PROCESS's
   real Python-float division raises there; the port's `jnp` division instead
   saturates to the *finite* value `life_plant` (`min(life_blkt_fpy / 0.0, life_plant)`
   is `min(inf, life_plant) = life_plant` under IEEE-754, never `inf`) — a genuinely
   different, and arguably more defensible, limiting behaviour, not a bug in the port.
   The harness's samples were deliberately bounded (lifetimes kept an order of
   magnitude above `tmain`, `t_plant_pulse_total` kept large relative to
   `div_nref`/`fwbs_nref`'s implied cycle count) to stay off this edge — see
   `test_availability.py`'s `TestAvail2`/`TestAvailSt` fuzz-bound comments, which record
   the specific ratios that triggered it during porting. Not fixed (this record's job is
   to reproduce PROCESS, not harden it), but worth a maintainer's attention: a solver
   exploring near this boundary would hit a real crash in today's PROCESS, not a graceful
   degradation.

## open questions

1. **The `avail`/`avail_2`/`avail_st` output-set mismatch (`bktcycles` vs.
   `t_plant_operational_total_yrs`/`redun_vac`) has no resolution here** — see "dispatch
   shape". Whoever designs `total_process.py`'s `i_plant_availability` `Switch` needs a
   policy for a slot one alternative doesn't produce at all (not just computes
   differently), which is a step beyond `heating.md`'s open question 2 (partially
   overlapping ports, but every branch there wrote the same four "total" fields).
2. **Whether `itart` should eventually be split into node alternatives anyway.** This
   record chose "keep static" primarily on a traffic argument (unusual, untested on the
   stellarator pipeline) rather than a structural one — a tokamak-facing port of this
   same file, where `itart == 1` (spherical tokamak) is a completely ordinary, heavily
   exercised configuration, might reasonably make the opposite call. Not resolved here
   since this port's scope is stellarator-only.
3. **No PROCESS unit test exercises `i_plant_availability` for a stellarator `IN.DAT`**
   (checked `tests/unit/models/stellarator/test_stellarator.py`) — the reachability
   argument above is derived from reading the call graph, not from an existing test that
   demonstrates `avail_st`/`avail_2` actually running in a real stellarator solve. A
   `tests/regression` case with `istell != 0`, `i_plant_availability = 3`, `itart = 1`
   would be the strongest possible confirmation and does not appear to exist.
4. **Whether `CpLifetimeSuperconducting`/`CpLifetimeResistive` should ever be registered
   alongside `CplifeAvail`/`CplifeAvailSt`, and if so how, is unresolved** (new this
   wave — see "cottax node"). As written, both pairs independently want to own
   `.costs.cplife`; `CplifeAvail`/`CplifeAvailSt` sidestep the conflict by duplicating
   `i_tf_sup`'s SC/resistive dispatch inline rather than consuming the standalone pair's
   output, which leaves `CpLifetimeSuperconducting`/`CpLifetimeResistive` valid but with
   no consumer in this file's own graph. Whoever designs `total_process.py`'s wiring
   needs a policy here: keep both pairs and pick one per registration, delete the
   now-unconsumed standalone pair, or find a cottax modelling convention (not present in
   `~/jaxgraph` today) for "one node's *intermediate* value, not its persisted one, feeds
   another node's self-reference" that would let them compose instead of conflict.
5. **Whether `AvailSt`'s duplicated SC/resistive recompute (inline, alongside
   `CplifeAvailSt`'s own copy) is worth deduplicating once a driver is assigned.** Two
   independent evaluations of the same two-line formula per graph run is cheap arithmetic
   but a real duplication; a rewrite pass after `total_process.py`'s wiring is designed
   (e.g. `Cut`/`rewrites`-style sharing, or reconsidering whether `AvailSt`'s
   pre-adjustment need is better modelled as its own minted intermediate) might remove it.
   Not attempted here — this wave's job was making the nodes representable, not
   optimising the resulting graph.
