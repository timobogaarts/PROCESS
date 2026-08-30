---
kind: model-unit
status: reviewed
confidence: high
---

**Second porting wave (`.costs.coe`): 41/43 methods ported, 44 nodes written, 43
registered.** The first wave ported 23 self-contained leaf accounts and left
`.costs.coe` -- the `i_figure_merit == 6` objective
(`tests/regression/input_files/stellarator_helias.IN.DAT:229`,
`functional_process/core/solver/objectives.py:88-90`) -- without a producer anywhere in
the ported graph, which `_audit/optimise_design.md` names as its single blocker. This
wave ports the whole transitive chain (§ "coverage map for `.costs.coe`" below) and
registers it in `total_process.COSTS_1990`, under a new
`.costs.i_cost_model` `Switch`. The two methods left unported are `run` (the call-order
dispatcher, which a `Graph` replaces -- its two *computations*, `cdirt` and `concost`,
are ported as their own nodes) and `output` (reporting only).

Three of the first wave's own conclusions are **corrected** by this wave, each in place
below: `acc2222`'s "dynamic-length loop is a structural JAX blocker" (open question 3 --
the bound is a run-configuration constant, so it is a static kwarg and the loop unrolls);
"`i_cost_model` is not wireable as a `Switch`" (§ `i_cost_model` -- it is, now that this
arm owns `.costs.coe`/`.costs.concost` themselves); and `unit_registry.md`'s reasoning
that `costs.py` "never even runs by default" so its nodes must stay unregistered (true of
PROCESS's bare default, `cost_variables.py:327`; irrelevant to the run this project
validates against, which sets `i_cost_model = 0` explicitly at
`stellarator_helias.IN.DAT:248` with the comment "0: 1990 cost module, the 2015 does not
work yet for stellarators").

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
`i_plasma_pedestal`. ~~**Not wired into `total_process.TOPOLOGY_SWITCHES` here** —
reserved for the consolidation pass, per this dispatch's boundary.~~ **Now wired**, by
the second wave, but *not* as the two-real-arm pairing this paragraph anticipated:
`costs_2015.py` still has zero `cottax` nodes, so its arm (`value == 1`, PROCESS's own
default) is declared `unproduced` -- a new third `Alternative` state that assembles as
empty instead of raising, added to `configuration.py` for exactly this case. The
alternative readings were weighed and rejected; the full argument, including why an
`unported` default arm would break `import functional_process.total_process` outright,
is in `total_process.py`'s own comment block on the switch and in `_audit/next_steps.md`
§9. The claim above that the two arms *would* satisfy `check_arms_are_exclusive` once
both were ported still stands and is now half-demonstrated: this arm owns
`.costs.coe`/`.costs.concost`, so the moment `costs_2015.py` grows a `run()`-level node
the pairing becomes a real two-armed switch with no further machinery.

**Ported: 41 of 43 methods** in `costs.py` (`costs.py` in this directory,
`test_costs.py`) -- 23 in the first wave, 18 in the second -- plus the two computations
`Costs.run()` performs inline (`cdirt`, `concost`), giving 44 pure functions and 44
`cottax` nodes. All tier-1, all self-contained (no calls into any other `Model`, no
internal iteration, no `scipy`). 43 of the 44 nodes are registered; only
`TfMagnetCostResistive` is not (see the coverage map). See "method inventory" below for
the full 43-row table and the per-method reasons. **Findings, none fixed**: two genuine
PROCESS defects in `acc223` (§ open questions #1), a history-dependence gap the port
cannot reproduce (`cdrlife_cal`/`cplife_cal`, § open questions #2), and one dead field
(`.costs.c2234`, § open questions #6). Open question #3 (`acc2222`'s "dynamic-length
loop") is **resolved and was wrong**: the bound is a run-configuration constant, so it
is a static kwarg and the loop unrolls at trace time.

## source

`process/models/costs/costs.py`, 3027 lines, registry unit #18 (shared with
`costs_2015.py`, see `costs_2015.md`). Full method list: `__init__`, `run`, `output`,
then 40 `accNNNN`-named account methods plus `coelc` and `convert_fpy_to_calendar`.
**Transitive-closure check**: no method in this file calls any other `Model`'s method,
`scipy`, or CoolProp (`grep -nP '^\s+self\.\w+\.\w+\(' costs.py` matching anything but
`self.accNNNN(`/`self.coelc(`/`self.convert_fpy_to_calendar(` returns nothing; `grep -n
"scipy\|fsolve\|optimize"` returns nothing). The only loops anywhere in the file are
`acc2222`'s two `for i in range(...)` loops over the PF coil count -- **not**
dynamic-length, as the first wave recorded: `.pf_coil.n_cs_pf_coils` and `.build.iohcl`
are neither iteration variables nor scan variables (`grep -n n_cs_pf_coils
process/core/solver/iteration_variables.py process/core/scan.py` -> no match), so both
are graph-assembly-time constants and the loops unroll at trace time. There is no
structural JAX blocker anywhere in this file.

### method inventory (43 methods)

`wave` is 1 (the first, leaf-account wave) or 2 (this one, the `.costs.coe` chain).
`registered` says whether the node is in `total_process.COSTS_1990`.

| method | account | wave | registered | note |
|---|---|---|---|---|
| `convert_fpy_to_calendar` | — | 1 | yes | branch-free besides 3 independent thresholds |
| `acc21` | 21 | 1 | yes | structures/buildings, one `ireactor` branch (`c213`) |
| `acc2211` | 221.1 | **2** | yes | first wall; `ife` is a **static** kwarg and only `ife != 1` is ported -- the IFE arm reads the 2-D `.ife.fwmatm`, a different reads-set over an unbuilt subsystem |
| `acc2212` | 221.2 | **2** | yes | blanket; same static-`ife` treatment (`.ife.blmatm`). `c22128` is deliberately not an output -- written only in the IFE arm and not a term of `c2212` in either |
| `acc2213` | 221.3 | **2** | yes | shield; same static-`ife` treatment (`.ife.shmatm`) |
| `acc2214` | 221.4 | 1 | yes | reactor structure, branch-free |
| `acc2215` | 221.5 | 1 | yes | divertor, trivial `ife.ife` zero-branch (no 2-D arrays), kept traced |
| `acc221` | 221 (total) | **2** | yes | sum of the five above; ported as its own node with PROCESS's five sub-calls neutralised in the test wrapper |
| `acc2221` | 222.1 | **2** | **SC arm only** | **split into two functions/nodes**, per the split default: `TfMagnetCostSuperconducting` (`i_tf_sup == 1`) and `TfMagnetCostResistive` (`!= 1`) share no body and read disjoint fields. Only the SC one is registered -- see the coverage map |
| `acc2222` | 222.2 | **2** | yes | PF magnets; the "dynamic-length loop" turned out to be static (see § source). **Split on `supercond_cost_model` 2026-08-30** into `PfMagnetCostPerKg`/`PerKam` over `_pf_magnet_cost`; `n_cs_pf_coils`/`iohcl`/`i_pf_conductor` remain static kwargs. Registered on a tokamak only (`None` on a stellarator) |
| `acc2223` | 222.3 | 1 | yes | vacuum vessel assembly, branch-free |
| `acc222` | 222 (total) | **2** | yes | sum; its `ife == 1` arm is a bare `c222 = 0`, so `ife` stays *traced* here (unlike 2211-2213) |
| `acc223` | 223 | **2** | yes | power injection; **two real PROCESS defects reproduced**, see open questions #1. Static `ife` (four-way IFE driver dispatch unported); `i_hcd_primary` traced |
| `acc224` | 224 | 1 | yes | vacuum system, one `VacuumPumpType` branch |
| `acc2251` | 225.1 | 1 | yes | TF coil power conditioning, `i_tf_sup` branches (x2), traced |
| `acc2252` | 225.2 | 1 | yes | PF coil power conditioning, one guarded-division branch |
| `acc2253` | 225.3 | **2** | yes | energy storage; static `i_pulsed_plant`/`istore`. `istore == 3` refused (a third reads-set: `p_plant_primary_heat_mw`/`t_plant_pulse_no_burn`/`dtstor`); options 1/2 are literal sums |
| `acc225` | 225 (total) | **2** | yes | sum; traced `ife`, same shape as `acc222` |
| `acc2261` | 2261 | 1 | yes | reactor cooling, branch-free |
| `acc2262` | 2262 | **2** | yes | auxiliary component cooling; static `ife` -- the IFE arm *adds* terms reading `.ife.tdspmw`/`.tfacmw` |
| `acc2263` | 2263 | **2** | yes | cryogenic system, branch-free |
| `acc226` | 226 (total) | **2** | yes | pure sum in the source (does not call its sub-accounts) |
| `acc2271` | 2271 | 1 | yes | fuelling system, branch-free |
| `acc2272` | 2272 | **2** | yes | fuel processing; static `ife`. **The only method in the file that writes outside `.costs.*`** -- it owns `.physics.wtgpd`, and nothing else in `process/` writes that field |
| `acc2273` | 2273 | **2** | yes | atmospheric recovery; the tritium threshold is a plain traced `jnp.where` (same reads both sides) |
| `acc2274` | 2274 | 1 | yes | nuclear building ventilation, branch-free |
| `acc227` | 227 (total) | **2** | yes | pure sum in the source |
| `acc228` | 228 | 1 | yes | instrumentation and control, branch-free |
| `acc229` | 229 | 1 | yes | maintenance equipment, branch-free |
| `acc23` | 23 | 1 | yes | turbine plant equipment; PROCESS only computes it when `ireactor==1`, see open questions #4 |
| `acc241`..`acc245` | 241-245 | 1 | yes (all 5) | electric plant equipment sub-accounts, all branch-free |
| `acc24` | 24 (total) | 1 | yes | sum of the five above |
| `acc25` | 25 | 1 | yes | misc plant equipment, branch-free |
| `acc26` | 26 | 1 | yes | heat rejection, one `ireactor` branch |
| `acc9` | 9 | 1 | yes | indirect cost/contingency, branch-free besides `cfind[lsa-1]` |
| `acc22` | 22 (total) | **2** | yes | sum of 221-229 plus `crctcore`; sub-calls neutralised in the test wrapper |
| `coelc` | — | **2** | yes | **cost of electricity -- `.costs.coe` itself.** ~290 lines, static `ife`/`itart`/`ireactor`/`ipnet`, traced `ifueltyp` |
| `run` | — | **n/a (orchestration)** | — | the call-order dispatcher; a `Graph` replaces it. Its two *computations* are ported as `TotalPlantDirectCost` (`.costs.cdirt`) and `ConstructedCost` (`.costs.concost`) |
| `output` | — | **n/a (reporting)** | — | calls `self.run()` then only `ovarre`/`oheadr` -- the same "reporting isn't quite inert but nothing to extract" shape as `coils/output.py` |

**41/43 methods ported (23 + 18), 44 pure functions, 44 nodes, 43 registered.** The two
unported methods are orchestration and reporting; nothing is left blocked.

### coverage map for `.costs.coe`

Derived before any code was written, by walking the chain backwards from
`process/models/costs/costs.py:2990`. It is worth stating the result plainly because it
is the reason this wave is as large as it is:

    coe = coecap + coefuelt + coeoam + coedecom                        (coelc, :2990)
      coecap   <- capcost <- moneyint <- concost                       (:2735-2757)
      coefuelt = coefwbl + coediv + coecdr + coecp + coefuel + coewst  (:2982)
                 <- fwallcst (acc2211), blkcst (acc2212), divcst (acc2215),
                    cpstcst (acc2221), cdcost (acc223), life_blkt/cdrlife_cal/
                    life_div/cplife_cal (convert_fpy_to_calendar), wtgpd (acc2272)
      coedecom <- concost                                              (:2960-2970)
    concost = cdirt + cindrt + ccont                                   (run, :77-79)
      cindrt, ccont <- acc9 <- cdirt
    cdirt   = c21 + c22 + c23 + c24 + c25 + c26                        (run, :64-71)
      c22   = c221+c222+c223+c224+c225+c226+c227+c228+c229             (acc22, :923-933)
        c221 <- acc2211 acc2212 acc2213 acc2214 acc2215
        c222 <- acc2221 acc2222 acc2223
        c225 <- acc2251 acc2252 acc2253
        c226 <- acc2261 acc2262 acc2263
        c227 <- acc2271 acc2272 acc2273 acc2274

**`.costs.coe` therefore depends on every computational method in the file.** There is no
smaller sufficient subset: the accumulation is a plain sum of all nine Account-22
sub-totals and all six top-level accounts, none of which PROCESS ever skips. "Port only
what `coe` needs" and "port everything except `run`/`output`" are the same instruction
here, and that was checked rather than assumed -- the 18 second-wave methods are exactly
the 20 first-wave audit-only entries minus `run` and `output`.

Two nodes are **written but deliberately not registered**:

- `TfMagnetCostResistive` (`acc2221`'s `i_tf_sup != 1` arm). Registering it alongside
  `TfMagnetCostSuperconducting` would be a duplicate-ownership conflict on
  `.costs.c22211`/`c22212`/`c2221`; pairing them as a real `Switch` would require that
  switch to nest inside `.costs.i_cost_model`, and nested switches are a still-open gap
  (`next_steps.md` §1, now with a third instance). `.tfcoil.i_tf_sup == 1` is both
  PROCESS's own default (`tfcoil_variables.py:261`) and the reference run's value, so
  the registered arm is correct under either -- this is a missing *generalisation*, not
  a wrong registration.
- `calculate_tf_magnet_cost_resistive`'s sibling is the only such case. Everything else
  written by either wave is now in `total_process.COSTS_1990`.

## data footprint

Full per-argument tables are in each ported function's own docstring in `costs.py`
(right above its `def`) — not duplicated here, per this project's practice of keeping the
signature and its documentation next to each other rather than in two places that can
drift. Every read across all 44 functions is `explicit-arg` (plain parameter, no
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

- `acc223`'s `c2233` -- looks like the same shape but is **not** history-dependent in
  practice, and this is the one place where saying so precisely matters. PROCESS never
  assigns it unless `ifueltyp == 1` (open questions #1), so on any other run it holds
  `cost_variables.py:165`'s dataclass default `0.0` -- and because `ifueltyp` is a
  run-configuration constant, "never assigned in this call" implies "never assigned in
  any call of the run". The port's `0.0` is therefore *exact* there, unlike
  `cdrlife_cal`'s, whose gate can flip between iterations. Confirmed on the converged
  reference run: `.costs.c2233 == 0.0` with `ifueltyp == 0`.
- `coelc`'s `cpstcst`/`cplife_cal`/`cplife` -- read unconditionally by the ported node
  even though PROCESS only reads them under `itart == 1`. A deliberate deviation from
  the split default, argued at the function's docstring and in "switches touched" below.

No `redundant-duplicate-write`s and no `implicit-io-via-callee` found in any of the 44
(no `copy.deepcopy`, no calls into another model). One cross-area write:
`acc2272` owns `.physics.wtgpd` (see the method inventory).

## proposed signature(s)

See `costs.py` — 44 functions (43 `calculate_*` plus `convert_fpy_to_calendar`,
matching PROCESS's own verb-named method), each documented in place. Not repeated here.

## cottax node

44 `ExplicitFunction` nodes, one per function -- the first wave's 23 written
immediately after their functions, the second wave's 21 in a block at the end of the
file. Every input/output is an ordinary `VarPath`; **no minted names anywhere in this
unit**, which is why the whole chain scores in the MDA harness rather than landing in
`errors`/`unverifiable`. The accumulator nodes (`ReactorCost`, `MagnetsCost`,
`PowerConditioningCost`, `HeatTransportSystemCost`, `FuelHandlingCost`,
`FusionPowerIslandCost`, `TotalPlantDirectCost`, `ConstructedCost`,
`ElectricPlantEquipmentCost`) read their sub-accounts' own outputs, so the graph
reproduces `Costs.run()`'s call order structurally instead of by hand.

43 of the 44 are registered, in `total_process.COSTS_1990`, as the
`.costs.i_cost_model == 0` arm of a new `Switch`. `TfMagnetCostResistive` is the one
exception -- see the coverage map.

## tier signal

All 44: **tier 1**. No internal iteration, no calls into other models, no `scipy`.
Every one has a `Tier1Contract` in `test_costs.py`, checked against PROCESS's own method
(for the six accumulators PROCESS calls its sub-accounts before summing, the reference
wrapper neutralises those sub-calls with instance-level no-ops so the check exercises the
accumulation and nothing else -- the sub-accounts have their own contracts alongside).

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

**Second wave, all genuinely `static`** (`eqx.field(static=True)` on the node, so
`mda_harness.switch_audit` checks each against the modelled run on every harness run --
17 new checked kwargs, 0 mismatched). Each is listed with *why* it is static rather than
traced, because the split default (`traceability_policy.md`) says a differing reads-set
means split, and three of these deviate from it deliberately:

- **`ife`** (`.ife.ife`) on `FirstWallCost`/`BlanketCost`/`ShieldCost`/
  `PowerInjectionCost`/`AuxiliaryComponentCoolingCost`/`FuelProcessingCost`/
  `CostOfElectricity` — split, only `ife != 1` implemented, the other arm raises. Each
  of these IFE arms reads fields no non-IFE arm does (2-D `.ife.fwmatm`/`blmatm`/
  `shmatm`; the `.ife.cdriv*`/`dcdrv*`/`edrive` driver-cost tables; `.ife.tdspmw`/
  `tfacmw`; `.ife.gain`/`fburn`/`reprat`/`uctarg`), and `.ife.*` has no unit in
  `unit_registry.md` at all. **Note this is a different treatment from the first wave's**
  `acc2215`/`acc222`/`acc225`, where the IFE arm is a bare zero with no new reads and
  `ife` is a plain traced `jnp.where` — the criterion is the reads-set, not the field.
- **`supercond_cost_model`** (`.costs.supercond_cost_model`, default `0`,
  `cost_variables.py:552`) on `TfMagnetCostSuperconducting`/`PfMagnetCost` — ~~**deviates
  from the split default**: the two arms differ by three scalar reads inside an otherwise
  shared body (~100 and ~200 lines respectively). Fourth and fifth instance of the
  size-aware exception `next_steps.md` §1 tracks.~~ **Both split, and the exception is
  withdrawn on both.** Account 222.1 split first (`next_steps.md` §14.2,
  `switch_kwarg_survey.md` §3); Account 222.2 followed on 2026-08-30 into
  `PfMagnetCostPerKg`/`PerKam` over a shared `_pf_magnet_cost`, with
  `indat.PF_MAGNET_COST` as the registry. The size-aware argument was wrong here for a
  reason it is not usually wrong: the four reads it kept were not merely dead, one of
  them (`.pf_coil.j_crit_str_pf`) had **no producer at all**, so the static kwarg was
  the only thing keeping a whole account off the graph. §13.2 of
  `cost_boundary_inputs.md` is the refusal it caused. The split is bigger than 222.1's
  because `acc2222` branches on the switch **twice** — once per PF coil and once for the
  central solenoid — with the copper/conduit/winding arithmetic interleaved, so the arms
  hand `_pf_magnet_cost` two strand costs where `_tf_magnet_cost_superconducting` takes
  one.
- **`n_cs_pf_coils`/`iohcl`** (`.pf_coil.n_cs_pf_coils`, `.build.iohcl`) on
  `PfMagnetCost` — loop bounds. Static because they must be: `range()` needs a Python
  int. Legitimate because neither is an iteration or scan variable, so both are
  graph-assembly-time facts, exactly `ImpurityRadiationTotals.imp_indices`'s case.
  `iohcl = 0` at registration is the one deliberate departure from a PROCESS default in
  this unit (`build_variables.py:177` says `1`); the reference stellarator has no central
  solenoid and `switch_audit` confirms it. (Since 2026-08-30 the stellarator has no
  occupant at all here — the slot is `None` on that device — and the tokamak's occupant
  takes `iohcl = PRESENT` and `n_cs_pf_coils = 7`, both pinned by
  `_pf_coil_system_deviations` before a tokamak finishes assembling.)
- **`i_pf_conductor`** (`.pf_coil.i_pf_conductor`, default `0`) on `PfMagnetCost` —
  ~~same size-aware exception as `supercond_cost_model`.~~ **Still static, and now for a
  stronger reason than size**: unlike `supercond_cost_model` it does not choose between
  the two arms, it branches *inside* both of them, selecting the conduit cost and the
  copper fraction and zeroing the superconductor strand cost. Splitting it would be a
  2x2 product of occupants over one 25-read signature, and `_pf_coil_system_deviations`'
  `-5` refuses the resistive value before a tokamak assembles anyway.
- **`i_pulsed_plant`/`istore`** (`.pulse.i_pulsed_plant`, `.pulse.istore`) on
  `EnergyStorageCost` — split: `istore == 3` reads three fields options 1/2 do not, and
  is refused. Options 1/2 add no reads at all (literal sums), so they stay in one body.
- **`itart`** (`.physics.itart`) on `CostOfElectricity` — **deviates from the split
  default**, sixth instance: the centrepost block is 15 lines of a ~290-line function
  and both arms are implemented in one body, so the node reads
  `.costs.cpstcst`/`cplife_cal`/`cplife` even at `itart == 0`. Contrast
  `TfMagnetCostSuperconducting`/`_resistive` in this same file, which *were* split
  because they share no body at all -- the two cases together are the clearest statement
  this project has of what the missing size/entanglement-aware rule would have to say.
- **`ireactor`/`ipnet`** (`.costs.ireactor`, `.costs.ipnet`) on `CostOfElectricity` —
  not switches but *preconditions*: `Costs.run()` calls `coelc()` only when
  `ireactor == 1 and ipnet == 0` (`costs.py:82-83`), so the node's `__check_init__`
  refuses any other pair rather than producing a `.costs.coe` PROCESS would have left
  untouched. Same move as `EcrhDensityLimit(i_plasma_pedestal=0)`.

None of these appear in `_audit/units/core/solver/switches.md` under this file yet (out
of this file's boundary to add — reserved for the consolidation pass). `i_cost_model`
itself now has a row in `unit_registry.md`'s switches table.

## calls into other models

None. Confirmed by grep and by direct read of all 44 ported bodies (see "source").

## JAX-difficulty flags

- ~~**`acc2222`'s dynamic-length loop**, the one real structural blocker in the
  file.~~ **Withdrawn -- the premise was wrong.** `for i in
  range(self.data.pf_coil.n_cs_pf_coils)` is not a run-time count in any sense that
  matters here: `n_cs_pf_coils` is not an iteration variable
  (`process/core/solver/iteration_variables.py`, no match) and not a scan variable
  (`process/core/scan.py`, no match), so it is constant for a whole solve and belongs in
  `naming_convention.md`'s static-kwarg category. Neither `lax.fori_loop` nor padding is
  needed: the loop unrolls at trace time, exactly as `ImpurityRadiationTotals`'s
  `imp_indices` tuple already does for "which impurity species exist". `.build.iohcl`
  (the second loop's bound, via `npf`) gets the same treatment. **There is now no
  structural JAX blocker anywhere in `costs.py`.**
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
- **`coelc`'s clamped square root** — `needs-lax-cond-or-where`, and a real trap.
  PROCESS zeroes `sqrt(p_plant_electric_net_mw / 1200)` when the power is negative
  (`costs.py:2874-2888`). The obvious port, `jnp.sqrt(jnp.maximum(p, 0.0))`, is
  value-correct and returns `nan` from `jacfwd` on the clamped branch, because `sqrt`
  has an infinite derivative at zero. Caught by `test_gradient_finite` on the
  `negative-net-electric-power-clamps-to-zero` sample, not by inspection; fixed with the
  standard *double* `jnp.where` (the inner one keeps the untaken branch's argument away
  from zero, the outer one discards its value). PROCESS's own finite difference there is
  identically zero, which is what the fixed form differentiates to. Worth carrying
  forward as a general note: every `jnp.maximum`-guard in front of a function with an
  unbounded derivative at the guard point has this shape.
- No CoolProp, no `scipy`, no data-dependent early exit anywhere in the 44 ported bodies.

## cold-start finding: `acc2221`'s `nan` is a missing producer, not a costs defect

Found by the MDA cold-start catalogue — `TfMagnetCostSuperconducting` was one of only two
blocks (of 134) that ran and emitted a non-finite value from a cold `DataStructure`,
producing `nan` for `.costs.c22211` and `.costs.c2221`. **The costs port is faithful and
was not changed.** The root cause is upstream and already on the punch list.

**The arithmetic.** `costs.py:1300` (port) / `costs.py:1505` (PROCESS):

```python
costtfcu = uccu * m_tf_coil_copper / (len_tf_coil * n_tf_coil_turns)
```

and the same denominator one line up in `costtfsc` (`supercond_cost_model == 0` arm,
which is the active one for `stellarator_helias.IN.DAT`). At the cold point
`.tfcoil.len_tf_coil == 0.0`, so the denominator is `0.0 * 136.6 == 0.0`; the numerators
are `600.0 * 0.0` and `75.0 * 0.0`, because `.tfcoil.m_tf_coil_superconductor` and
`.tfcoil.m_tf_coil_copper` are `CoilsMass` outputs and every one of `coils/mass.py`'s
mass expressions is directly proportional to `len_tf_coil`. So both are `0.0 / 0.0`.
`ctfconpm` becomes `nan`, and `c22211 = fkind * 1e-6 * ctfconpm * winding_length` is
`nan * 0.0 = nan` (`winding_length` is also zero), hence `c2221 = nan`. `c22212`,
`c22213` are a clean `0.0`; `c22214`/`c22215` read `.structure.aintmass`/`clgsmass`,
which do not depend on `len_tf_coil`, and stay finite and correct.

**PROCESS does the same thing, only louder.** `Costs.acc2221`'s superconducting arm is
character-equivalent, and calling the reference wrapper
(`test_costs._reference_tf_magnet_cost_superconducting`) with the cold argument tuple
gives `ZeroDivisionError: float division by zero` at `costs.py:1489`. The difference is
dtype, not logic: PROCESS's `ucsc` is a plain Python list, so the division is Python
float arithmetic and raises; the port's `jnp.asarray(ucsc)[...]` makes it array
arithmetic, which returns `nan`. This is the harness's standard domain-guard shape
(`_harness/contracts.py`'s `reference_domain_errors`) and the port is on the correct side
of it — a traced function cannot raise on a data-dependent condition. Note the port
*also* raises `ZeroDivisionError` if handed plain Python floats at this point; inside the
graph its inputs are always `jnp` arrays, so it yields `nan` there.

**Root cause: `.tfcoil.len_tf_coil` has no producer.** It is an *unowned input* of both
the full and the driven graph — no ported node owns it — while PROCESS writes it at
`process/models/stellarator/coils/calculate.py:87`. Four ported nodes read it
(`StructureMasses`, `PlasmaFacingCoilArea`, `CoilsMass`, `TfMagnetCostSuperconducting`).
Cold value `0.0`, PROCESS-converged value `40.8655 m`; with the converged value the port
reproduces PROCESS exactly (`c22211 = 566.001`, `c2221 = 989.525`). This is the
already-documented "missing producer" case, deliberately left open because giving
`len_tf_coil` an owner would silently switch `PlasmaFacingCoilArea` from PROCESS's stale
read to the fresh value — a decision, not a port. See
`_audit/boundary_inputs_audit.md` §4c (c1) and §7 item 4, `coils/calculate.md`'s "real
PROCESS bugs found" finding 2, and `_audit/next_steps.md:274`. **Fixing that fixes this;
nothing in `costs.py` needs to change.**

**Value and gradient** (`jax.jacfwd` over
`m_tf_coil_superconductor`/`len_tf_coil`/`m_tf_coil_copper`/`m_tf_coil_case`): at the
cold point both the value and the Jacobian rows for `c22211`/`c2221` are `nan`; at the
converged point the value and the full Jacobian are finite. This is *not* the
`jnp.sqrt(jnp.maximum(0, x))` shape — there is no value-correct-but-gradient-`nan`
asymmetry to miss, both go wrong together.

**Recommendation: leave `costs.py` alone.** A guarded denominator
(`jnp.where(len_tf_coil * n_tf_coil_turns > 0, ..., 0.0)`) would be value-identical
wherever PROCESS is defined — PROCESS raises when the denominator is zero — but it is the
wrong fix twice over. It would return a *fabricated* `0.0` capital cost at the cold
point instead of the `nan` that correctly signals "an input you need has not been
produced yet", masking exactly the missing-producer signal the catalogue exists to
surface; and unlike the dead-sink `nu_star` case, `.costs.c2221` is read by `MagnetsCost`
and propagates up the cost tree into `coe`, the run's objective — so a plausible-looking
zero here would silently corrupt the figure of merit, where a `nan` cannot. (The single
`jnp.where` would also still leave a `nan` gradient, needing the double-`where` of
`coelc` above, for a branch that should never be taken in the first place.)

A note on the inactive arm: `supercond_cost_model != 0` computes
`sc_mat_cost_0[i] * j_crit_str_0[i] / j_crit_str_tf`, and `.tfcoil.j_crit_str_tf` is
`0.0` at the cold point *and* at PROCESS's converged point (it is another unowned input,
never written in the stellarator pipeline). That arm would be `inf` rather than `nan`.
It is unreached for this input file (`supercond_cost_model == 0`, bound as a static
switch kwarg), so it is not the finding above — but it is the same class of problem and
will need the same upstream answer if a tokamak input ever selects it.

## open questions

1. **Two real PROCESS defects in `acc223`** (Account 223, power injection). **Now
   ported, reproducing both faithfully** rather than fixing them, per this project's
   standing policy (`radiation_power.md`'s precedent). Structure
   (`process/models/costs/costs.py:1866-1922`):
   ```
   if ife.ife != 1:
       c2231 = ...                    # 223.1 ECH
       if ifueltyp == 1: c2231 = (1-fcdfuel)*c2231; c2231 = fkind*c2231
       if i_hcd_primary != 2: c2232 = ...(LH)   else: c2232 = ...(ICH)   # 223.2
       if ifueltyp == 1:
           c2232 = (1-fcdfuel)*c2232; c2232 = fkind*c2232
           c2233 = ...                # 223.3 Neutral Beam -- nested INSIDE this if!
       if ifueltyp == 1:
           c2233 = (1-fcdfuel)*c2233; c2233 = fkind*c2233
   ```
   (a) **`c2233` (neutral beam cost) is only ever computed when `ifueltyp == 1`** — its
   assignment (`costs.py:1909-1915`) is nested inside the `if ifueltyp == 1:` block that
   also finishes `c2232`, not inside its own top-level `#  Account 223.3` section the
   comment claims it starts. Whenever `ifueltyp != 1` the field is never assigned in
   this call. **Refinement on the first wave's write-up**: it recorded this as "the same
   history-dependence shape as `cdrlife_cal`". It is not, and the difference is what
   makes it portable exactly. `ifueltyp` is a run-configuration constant (not an
   iteration variable, not a scan variable), so "never assigned in *this* call" implies
   "never assigned in *any* call of the run" — the field can only ever hold
   `cost_variables.py:165`'s dataclass default `0.0`. `cdrlife_cal`'s gate, by contrast,
   is `life_blkt_fpy < life_plant`, whose left side genuinely moves between VMCON
   iterations. The port returns `0.0` on that branch, which is therefore exact, not an
   approximation. Confirmed on the converged reference run: `.costs.c2233 == 0.0` with
   `.costs.ifueltyp == 0`.
   (b) **`fkind` is applied only on the `ifueltyp == 1` branch**, for all three
   sub-accounts (`costs.py:1877-1881`, `1899-1903`, `1917-1921`): PROCESS computes
   `cNNNN` unconditionally and then applies both `(1 - fcdfuel)` *and* the
   Nth-of-a-kind multiplier inside the `if`. So Account 223 silently escapes `fkind` on
   every run with `ifueltyp != 1`, unlike every other account in the file, which applies
   it unconditionally. New this wave, found while porting; reproduced as written.
2. **`cdrlife_cal`/`cplife_cal`'s history dependence** (`convert_fpy_to_calendar`, see
   data footprint) — the port cannot reproduce PROCESS's "leave it at whatever a
   previous solver iteration wrote" behaviour, because a pure function has no notion of
   a previous call's state. Chose `0.0` (the dataclass default) for the untaken branch,
   which is correct on iteration 1 of any run but is a genuine, provable divergence from
   PROCESS's own behaviour from iteration 2 onward whenever the branch's own condition
   flips between iterations. Not resolved — this is a structural gap in what a pure
   function *can* represent, not a bug in this port; flagged for whoever designs the
   `Cut`/state-carrying machinery `next_steps.md` §5 already tracks for the unrelated but
   structurally similar `.physics.first_call` self-loop. **Inert on the reference run**:
   `life_blkt_fpy = 25.98 < life_plant = 40`, so the fast branch is taken and
   `cdrlife_cal` is genuinely written; `itart == 0`, so `cplife_cal` is `0.0` in both
   PROCESS and the port.
3. ~~**`acc2222`'s dynamic-length loop** — the one real structural blocker found in the
   whole file.~~ **RESOLVED, and the finding was wrong.** See JAX-difficulty flags: the
   loop bound is a run-configuration constant, so it is a static kwarg and the loop
   unrolls at trace time. `acc2222` is ported and registered. There is no structural JAX
   blocker anywhere in `costs.py`.
4. **`acc23`'s and `acc21`'s `c213`'s default-zero-on-`ireactor==0`** assumption — unlike
   `cdrlife_cal` (open question #2), `ireactor` is a run-configuration constant set once
   from IN.DAT and never touched again during a solve (confirmed: not in
   `process/core/solver/iteration_variables.py`, not in `process/core/scan.py`, same
   evidence `configuration.py`'s own module docstring already uses for every topology
   switch), so the port's `0.0` default is safe — flagged only so a reader does not
   conflate this case with `cdrlife_cal`'s genuine one. `CostOfElectricity` makes the
   same constancy argument load-bearing rather than merely reassuring: it *refuses* to
   be constructed unless `ireactor == 1 and ipnet == 0`.
5. ~~**20 audit-only methods**~~ — **closed**: 18 of the 20 are ported this wave; the
   remaining two are `run` (orchestration) and `output` (reporting), neither of which is
   a computation. The IFE arms of six of them are *not* ported (static `ife`, refused),
   which is the honest residue of that item: `.ife.*` has no unit in `unit_registry.md`
   at all, and the 2-D `fwmatm`/`blmatm`/`shmatm` `VarPath` question the first wave
   flagged is deferred, not answered — it will be whoever ports the IFE device mode's to
   make, and nothing in this project's stellarator scope needs it.
6. **`.costs.c2234` is a dead field.** It is a term of `c223` (`costs.py:1976`) but is
   written in exactly one place in all of `process/`: `costs.py:1968`, inside the IFE
   *and* `ifueltyp == 1` branch, where it is set to `0.0` (grepped, no other writer).
   So it is `0.0` on every possible run, and the port folds it to the literal rather
   than declaring an `Output` nothing produces. New this wave; not fixed.
7. **The residual `.costs.coe` disagreement is `VacuumOld`'s, not this unit's.** The MDA
   harness reports 12 disagreements across the cost chain at `rel_diff` between 1.7e-06
   and 4.1e-04, and all twelve are one already-documented cause: this port solves the
   vacuum-duct diameter to `tol=1e-10` where PROCESS stops at a 1% relative step
   (`models/vacuum.py:250`, `process/models/vacuum.py:469-477`,
   `mda_harness.EXPLAINED_DISAGREEMENTS`), and `.vacuum.dlscal`/`dia_vv_vacuum_ducts`
   feed Account 224. Demonstrated, not argued: fed PROCESS's own converged values for
   those two fields, `calculate_vacuum_system_cost` reproduces all seven of PROCESS's
   Account-224 numbers at **exactly zero** relative difference, and every downstream
   absolute delta is that one `c224` delta pushed through the (linear) accumulation
   chain — `delta c224 = delta c22 = delta cdirt = 1.226201741e-02`, `delta cindrt =
   cfind[lsa-1]*(1+cowner)*delta cdirt` and `delta ccont = fcontng*(delta cdirt + delta
   cindrt)` both predicted to 11 significant figures, `delta concost` their sum. Nothing
   in this unit is off; deliberately not suppressed, so a future real regression on the
   same fields still shows.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

17 fractional power laws and 1 square root in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
