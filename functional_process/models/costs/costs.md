---
kind: model-unit
status: draft
confidence: high
---

**The `i_cost_model` finding (this dispatch's main ask): confirmed genuine topology
`Switch`, disjoint subgraphs.** `.costs.i_cost_model` is read in exactly one place in the
whole codebase that matters for graph assembly: `process/main.py`'s `Models.costs`
`@property` (lines 745-764), which picks a whole `Model` instance
(`Costs()`/`Costs2015()`/a user-provided custom model) **before any model runs** and
hands that single instance to every caller (`stellarator.py`'s `self.costs =
costs` constructor injection, `Caller._call_models_once`'s `self.models.costs.run()`).
Neither `costs.py` nor `costs_2015.py` reads `i_cost_model` internally (confirmed by
grep — zero hits in either file) and `stellarator.py` never branches on it either — it
only ever calls `self.costs.run()`/`.output()` on whatever was already injected. This is
*exactly* the precedent `_audit/schema.md`'s own switch-record template names
("resolved above this file... see `i_cost_model` / `Models.costs` in `process/main.py`
for the precedent") — the schema was written anticipating this instance.

**And the two arms are genuinely disjoint, not a shared-body-with-a-branch case** — the
`next_steps.md` §4c prediction is correct. `costs.py` writes 114 distinct `.costs.*`
fields (`grep -oP 'self\.data\.costs\.\w+(?=\s*=[^=])' costs.py | sort -u | wc -l`);
`costs_2015.py` writes only 4 distinct `.costs_2015.*` scalar fields plus the shared
100-slot `s_cost`/`s_cref`/`s_k`/`s_kref`/`s_cost_factor` arrays (`grep` for
`self.data.costs_2015.\w+ =` returns only `total_costs`, `mean_electric_output`,
`annual_electric_output`, `maintenance`; everything else is `s_*[i] =`, indexed writes).
**The only two `VarPath`s both files write are `.costs.coe` and `.costs.concost`** —
PROCESS's own two "final" outputs (cost of electricity, constructed cost — the latter is
literally commented `# Save as concost, the variable used as a Figure of Merit` in
`costs_2015.py`'s `run()`). That is exactly the shape `configuration.py`'s
`Switch.check_arms_are_exclusive` requires of a real `Alternative` pair (own at least one
output in common, or they are not alternatives at all — they could coexist). Confirms
this is the third real `TOPOLOGY_SWITCHES` entry after `isthtr`/`ipowerflow`/
`i_plasma_pedestal`. **Not wired into `total_process.TOPOLOGY_SWITCHES` here** — reserved
for the consolidation pass, per this dispatch's boundary.

**Ported: 23 of 43 methods** in `costs.py` (`costs.py` in this directory,
`test_costs.py`), all tier-1, all self-contained (no calls into any other `Model`, no
internal iteration, no `scipy`). See "method inventory" below for the full 43-row table
(23 ported, 20 audit-only) and their reasons. **Three real findings, none fixed**: a
genuine PROCESS bug in `acc223` (§ open questions #1), a history-dependence gap the port
cannot reproduce (`cdrlife_cal`/`cplife_cal`, § open questions #2), and a dynamic-length
loop in `acc2222` that blocks porting it under this project's current JAX-difficulty
conventions (§ open questions #3).

## source

`process/models/costs/costs.py`, 3027 lines, registry unit #18 (shared with
`costs_2015.py`, see `costs_2015.md`). Full method list: `__init__`, `run`, `output`,
then 40 `accNNNN`-named account methods plus `coelc` and `convert_fpy_to_calendar`.
**Transitive-closure check**: no method in this file calls any other `Model`'s method,
`scipy`, or CoolProp (`grep -nP '^\s+self\.\w+\.\w+\(' costs.py` matching anything but
`self.accNNNN(`/`self.coelc(`/`self.convert_fpy_to_calendar(` returns nothing; `grep -n
"scipy\|fsolve\|optimize"` returns nothing). The only loop anywhere in the file is
`acc2222`'s `for i in range(self.data.pf_coil.n_cs_pf_coils)` — a **dynamic-length**
loop (see JAX-difficulty flags), the sole structural blocker in the whole file.

### method inventory (43 methods)

| method | account | ported? | note |
|---|---|---|---|
| `convert_fpy_to_calendar` | — | **yes** | branch-free besides 3 independent thresholds |
| `acc21` | 21 | **yes** | structures/buildings, one `ireactor` branch (`c213`) |
| `acc2211` | 221.1 | no | first wall; `ife.ife` branch reads 2-D `ife.fwmatm[i,j]` arrays — deferred, see open questions |
| `acc2212` | 221.2 | no | blanket; same `ife.fwmatm`/`ife.blmatm` 2-D-array shape as `acc2211` |
| `acc2213` | 221.3 | no | shield; same `ife.shmatm` 2-D-array shape |
| `acc2214` | 221.4 | **yes** | reactor structure, branch-free |
| `acc2215` | 221.5 | **yes** | divertor, trivial `ife.ife` zero-branch (no 2-D arrays) |
| `acc221` | 221 (total) | no | sum of the five above; not worth porting with 3/5 unported |
| `acc2221` | 222.1 | no | TF magnets; `i_tf_sup`/`itart`/`ifueltyp` 3-way nested branching, `supercond_cost_model` sub-branch — left for a future pass, no blocker found, just out of this pass's bounded scope |
| `acc2222` | 222.2 | no | PF magnets; **dynamic-length** `for i in range(n_cs_pf_coils)` loop — structural JAX blocker, see open questions #3 |
| `acc2223` | 222.3 | **yes** | vacuum vessel assembly, branch-free |
| `acc222` | 222 (total) | no | sum incl. unported `acc2221`/`acc2222`; also has its own `ife.ife==1: c222=0` branch |
| `acc2231`/`acc2232`/`acc2233` (inside `acc223`) | 223 | no | power injection; **a real PROCESS bug found**, see open questions #1 — deferred pending that bug's resolution, entangled with unported `current_drive.py` fields anyway |
| `acc224` | 224 | **yes** | vacuum system, one `VacuumPumpType` branch |
| `acc2251` | 225.1 | **yes** | TF coil power conditioning, `i_tf_sup` branches (x2) |
| `acc2252` | 225.2 | **yes** | PF coil power conditioning, one guarded-division branch |
| `acc2253` | 225.3 | no | energy storage/thermal storage; `pulse.i_pulsed_plant`-gated, `istore`-switched (4-way, raises `ProcessValueError` on an illegal value) — left for a future pass, no blocker found beyond size/time budget |
| `acc225` | 225 (total) | no | sum incl. unported `acc2253`; also `ife.ife==1: c225=0` branch |
| `acc2261` | 2261 | **yes** | reactor cooling, branch-free |
| `acc2262` | 2262 | no | auxiliary component cooling — not read this pass, time budget |
| `acc2263` | 2263 | no | cryogenic system — not read this pass, time budget |
| `acc226` | 226 (total) | no | sum incl. unported `acc2262`/`acc2263` |
| `acc2271` | 2271 | **yes** | fuelling system, branch-free |
| `acc2272` | 2272 | no | fuel processing; `ife.ife` branch **also writes `.physics.wtgpd`** (a cross-area write feeding `acc2272`'s own next line), not a plain `explicit-arg` shape — deferred |
| `acc2273` | 2273 | no | atmospheric recovery; `f_plasma_fuel_tritium` threshold branch, straightforward but not read in enough depth this pass — time budget |
| `acc2274` | 2274 | **yes** | nuclear building ventilation, branch-free |
| `acc227` | 227 (total) | no | sum incl. unported `acc2272`/`acc2273` |
| `acc228` | 228 | **yes** | instrumentation and control, branch-free |
| `acc229` | 229 | **yes** | maintenance equipment, branch-free |
| `acc23` | 23 | **yes** | turbine plant equipment; PROCESS only computes it when `ireactor==1`, see open questions #4 |
| `acc241`..`acc245` | 241-245 | **yes** (all 5) | electric plant equipment sub-accounts, all branch-free |
| `acc24` | 24 (total) | **yes** | sum of the five above — all 5 inputs are ported, so the total is too |
| `acc25` | 25 | **yes** | misc plant equipment, branch-free |
| `acc26` | 26 | **yes** | heat rejection, one `ireactor` branch (formula selection, not a default-zero case like `acc23`) |
| `acc9` | 9 | **yes** | indirect cost/contingency, branch-free besides `cfind[lsa-1]` |
| `coelc` | — | no | cost of electricity, ~290 lines, several branches (`istore`, `ireactor`, NaN-guard clamping) — large, out of this pass's bounded scope, no structural blocker found |
| `run` | — | no (orchestration) | the call-order dispatcher itself; not a computation to port — cottax's graph replaces it |
| `output` | — | no | pure reporting, calls `self.run()` first then only `ovarre`/`oheadr` calls — same "reporting isn't quite inert but nothing to extract" shape as `coils/output.py` |

**23/43 ported. 20 audit-only**, none blocked by entanglement with an unported *unit* in
the graph-topology sense (no method in this file calls another `Model`) — the 20 are
either genuinely out of this pass's bounded time budget (most of them: straightforward,
same shape as the 23 that *are* ported, just not gotten to), or hit one of three real
structural issues (2-D `ife.*` arrays in `acc2211`/`acc2212`/`acc2213`; the dynamic-length
loop in `acc2222`; the live bug in `acc223`).

## data footprint

Full per-argument tables are in each ported function's own docstring in `costs.py`
(right above its `def`) — not duplicated here, per this project's practice of keeping the
signature and its documentation next to each other rather than in two places that can
drift. Every read across all 23 functions is `explicit-arg` (plain parameter, no
mid-function re-read, no branching on the same field it also writes) **except**:

- `convert_fpy_to_calendar`'s `cdrlife_cal`/`cplife_cal` — `implicit-io`, not
  `local-intermediate`: PROCESS only writes these on the "fast" branch of their own
  threshold and otherwise leaves them at whatever a *previous solver iteration* left
  (the threshold's own inputs, `life_blkt_fpy`/`life_plant`, are physics quantities that
  can genuinely change between VMCON iterations within one run — unlike `ireactor`/
  `i_blkt_coolant_type`/`lsa`, which are fixed run-configuration constants). A pure
  function has no notion of "the value before this call", so the port returns `0.0` on
  the untaken branch instead — a real, documented behavioural difference from PROCESS,
  not a silent normalisation. See open questions #2.
- `acc23`'s `c23` (`calculate_turbine_plant_equipment_cost`) — same shape, but here the
  gating value (`ireactor`) genuinely is a fixed run-configuration constant, so the
  port's `0.0` default is safe (not history-dependent) — see open questions #4 for why
  this is flagged as an assumption anyway, cross-referenced against `cdrlife_cal` so the
  two similar-looking cases aren't conflated.

No `redundant-duplicate-write`s and no `implicit-io-via-callee` found in any of the 23
(no `copy.deepcopy`, no calls into another model).

## proposed signature(s)

See `costs.py` — 23 `calculate_*` functions (one `convert_*`, matching PROCESS's own
verb-named method), each documented in place. Not repeated here.

## cottax node

23 `ExplicitFunction` nodes, one per function, written in `costs.py` immediately after
each function — every input/output is an ordinary `VarPath`, no minted names, no
switches-as-ports beyond the plain static-looking device/config values already discussed
under "switches touched". `ElectricPlantEquipmentCost` reads the other five Account-24x
nodes' own outputs (`c241`..`c245`), matching `Costs.acc24`'s own call order.

## tier signal

All 23: **tier 1**. No internal iteration, no calls into other models, no `scipy`. Ported
as part of finishing this pass's bounded scope, per `unit_registry.md`'s "standing
practice going forward".

## switches touched

None of these are topology-changing in the graph-assembly sense (unlike `i_cost_model`
itself, this file's headline finding) — all are ordinary data-dependent branches inside
already-existing PROCESS function signatures, kept as plain traced arguments (`jnp.where`)
per the `itart` precedent (`hcpb.md`'s switches-touched section: a device/config flag the
source's own code does not itself split into separate functions is not this project's
business to split either).

- **`lsa`** (`.costs.lsa`, 1-4) — indexes a `cmlsa` scaling-factor table, *different
  literal values per account* (not a shared constant — each `accNNNN` method hardcodes
  its own 4-element list inline). Not a switch in the topology sense: selects a
  coefficient, never which formula runs. `static_argnames` in every contract that reads
  it (array indexing, not differentiable in any physically meaningful sense).
- **`ireactor`** (`.costs.ireactor`, 0/1) — gates `acc21`'s `c213` (default-zero when 0,
  see open questions #4), `acc23`'s whole computation (same default-zero shape), and
  `acc26`'s formula selection (a genuine 2-way formula, not a default-zero). Kept static.
- **`ife`** (`.ife.ife`, 0/1) — gates `acc2215`'s divertor cost to zero for IFE devices
  (IFE has no divertor plates in this model). Kept static. The stellarator pipeline this
  project scopes to always has `ife.ife != 1` (whole-device mode is decided by `istell`,
  not `ife.ife`, per `switches.md`'s `istell` row), so this branch is exercised only by
  the fuzz/legacy test points, not by any real stellarator run — noted, not resolved.
- **`ifueltyp`** (`.costs.ifueltyp`, 0/1/2) — gates `acc2215`'s split between capital cost
  and fuel cost. Kept static.
- **`i_tf_sup`** (`.tfcoil.i_tf_sup`, `TFConductorModel`, 0/1/2) — gates two independent
  2-way formula choices in `acc2251` (breakers, bussing). Compared against the plain int
  `1` rather than importing the enum (keeps this port free of a `process.*` import at
  module scope — see JAX-difficulty flags). Kept static.
- **`i_vacuum_pump_type`** (`.vacuum.i_vacuum_pump_type`, `VacuumPumpType`, 0/1) — gates
  `acc224`'s pump unit-cost formula. Same enum-avoidance treatment. Kept static.
- **`i_blkt_coolant_type`** (`.fwbs.i_blkt_coolant_type`, 1/2) — indexes 2-element
  `uchts`/`ucturb` tables in `acc2261`/`acc23`. Array index, not a formula switch. Kept
  static.

None of these seven appear in `_audit/core/solver/switches.md` under this file yet (out
of this file's boundary to add — reserved for the consolidation pass, same as
`i_cost_model` itself).

## calls into other models

None. Confirmed by grep and by direct read of all 23 ported bodies (see "source").

## JAX-difficulty flags

- **`acc2222`'s dynamic-length loop**, `minor`→`workaround-known` in principle but
  **not exercised here**: `for i in range(self.data.pf_coil.n_cs_pf_coils)` sums a
  per-PF-coil winding length, where `n_cs_pf_coils` is a run-time count, not a
  compile-time constant. `jax.lax.fori_loop`/a fixed-size-padded-array-plus-mask are the
  two standard workarounds; neither is applied here since `acc2222` itself is out of
  this pass's ported set (see method inventory). Flagged as the one real structural
  blocker in the file, for whoever picks up `acc2222`/`acc222` next.
- **Enum comparisons against plain ints** (`i_tf_sup == 1`, `i_vacuum_pump_type == 1`)
  rather than `TFConductorModel.SUPERCONDUCTING`/`VacuumPumpType.COMPOUND_CRYOPUMP` —
  `minor`, `workaround-known`. Keeps `costs.py` (the port) free of any `process.*` import
  at module scope, which otherwise would be the first port file in this project to import
  a `process` enum for its *value* rather than calling a `process` reference function
  from a *test* file only. The values themselves are `IntEnum` members (verified against
  `process/models/tfcoil/base.py`/`process/data_structure/vacuum_variables.py`), so the
  plain-int comparison is exact, not approximate.
- **`acc2252`'s guarded division** (`srcktpm / pfckts` only when `pfckts != 0`) —
  `minor`, `workaround-known`, same guarded-denominator `jnp.where` pattern already
  documented in `hcpb.md`'s `vv_density`.
- **Every `cmlsa`/`cfind`/`uchts`/`ucturb` table lookup** (`array[index - 1]`) —
  `minor`, `workaround-known`: ordinary JAX dynamic/static indexing, no special
  handling needed (verified these compile fine under `jax.jacfwd` — the index arguments
  are `static_argnames`, excluded from differentiation, so there is no question of
  differentiating *through* an index).
- No CoolProp, no `scipy`, no data-dependent early exit anywhere in the 23 ported bodies.

## open questions

1. **A real PROCESS bug in `acc223`** (Account 223, power injection — not ported, see
   method inventory). Structure:
   ```
   if ife.ife != 1:
       c2231 = ...                    # 223.1 ECH
       if ifueltyp == 1: c2231 = ...
       if i_hcd_primary != 2: c2232 = ...(LH)   else: c2232 = ...(ICH)   # 223.2
       if ifueltyp == 1:
           c2232 = ...
           c2233 = ...                # 223.3 Neutral Beam -- nested INSIDE this if!
       if ifueltyp == 1:
           c2233 = ...
   ```
   **`c2233` (neutral beam cost) is only ever computed when `ifueltyp == 1`** — its
   assignment is nested inside the `if ifueltyp == 1:` block that also finishes `c2232`,
   not inside its own top-level `#  Account 223.3` section the comment claims it starts.
   Whenever `ifueltyp != 1` (the common case — `ifueltyp` defaults to something other
   than 1 in most configurations), `c2233` is never assigned at all in this call and
   stays at whatever it held from a previous iteration (same history-dependence shape as
   `cdrlife_cal`, open question #2) or its dataclass default. **Not fixed** (out of this
   file's scope decision to leave `acc223` audit-only this pass; documented so whoever
   ports it next sees it immediately rather than re-deriving it). Confirmed by direct
   read of `costs.py:1867-1978`, not by running PROCESS (no existing PROCESS test
   exercises `ifueltyp != 1` for this specific method per a quick check of
   `tests/unit/models/test_costs_1990.py::test_acc223`'s fixture — not chased further).
2. **`cdrlife_cal`/`cplife_cal`'s history dependence** (`convert_fpy_to_calendar`, see
   data footprint) — the port cannot reproduce PROCESS's "leave it at whatever a
   previous solver iteration wrote" behaviour, because a pure function has no notion of
   a previous call's state. Chose `0.0` (the dataclass default) for the untaken branch,
   which is correct on iteration 1 of any run but is a genuine, provable divergence from
   PROCESS's own behaviour from iteration 2 onward whenever the branch's own condition
   flips between iterations. Not resolved — this is a structural gap in what a pure
   function *can* represent, not a bug in this port; flagged for whoever designs the
   `Cut`/state-carrying machinery `next_steps.md` §5 already tracks for the unrelated but
   structurally similar `.physics.first_call` self-loop.
3. **`acc2222`'s dynamic-length loop** — see JAX-difficulty flags. The one real
   structural blocker found in the whole file; everything else audit-only this pass is a
   time-budget decision, not a blocker.
4. **`acc23`'s and `acc21`'s `c213`'s default-zero-on-`ireactor==0`** assumption — unlike
   `cdrlife_cal` (open question #2), `ireactor` is a run-configuration constant set once
   from IN.DAT and never touched again during a solve (confirmed: not in
   `process/core/solver/iteration_variables.py`, not in `process/core/scan.py`, same
   evidence `configuration.py`'s own module docstring already uses for every topology
   switch), so the port's `0.0` default is safe — flagged only so a reader does not
   conflate this case with `cdrlife_cal`'s genuine one.
5. **20 audit-only methods, most with no blocker beyond this pass's time budget** — see
   the method inventory table's "note" column. A follow-up pass should be able to port
   most of `acc241`-adjacent accounts (`acc2262`/`acc2263`/`acc2273`) quickly; `acc2211`/
   `acc2212`/`acc2213` need a decision on how to represent PROCESS's 2-D `ife.fwmatm`/
   `blmatm`/`shmatm` arrays as `VarPath`s (mechanical, not a design blocker); `acc2221`/
   `acc2253`/`coelc` are larger and were simply not reached this pass.
