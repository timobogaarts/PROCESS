# Tokamak call surface — the scope rule, traced

**What this file is.** The conventional-tokamak analogue of `unit_registry.md`'s opening
scope rule. That rule was *"derived by tracing `Stellarator.run()`'s actual call surface,
not assumed"*, and its own header records that an earlier non-recursive glob silently
missed a whole subpackage (`models/stellarator/coils/`, 6 files, 1950 LOC). This file does
not repeat that: **every model below was reached by executing `Caller._call_models_once`
under a profiler**, not by reading `caller.py` and guessing.

It extends `tokamak_scope.md`, which counted the *switch* decisions a tokamak adds. This
one counts the *code* a tokamak reaches. Written by a tracing pass only — no `.py`,
`next_steps.md`, `unit_registry.md` or `tokamak_scope.md` was touched
(`next_steps.md` §4b: registration and bookkeeping are a consolidation pass's job).

## The scope rule

> **All of `process/core/caller.py::_call_models_once`'s reachable call graph with
> `istell == 0` and `ife == 0` — 38 files under `process/models/**`, entered through 31
> top-level `Model` calls plus the sub-models `Models.__init__` injects into them.** Not
> whole files: 338 distinct functions of the ~1000 those 38 files define. `models/ife.py`,
> `models/stellarator/**`, `models/tfcoil/resistive.py`, `models/blankets/dcll.py`,
> `models/costs/costs_2015.py`, `models/engineering/pumping.py`,
> `models/engineering/materials.py` and the whole of `models/geometry/**` are **outside**
> it on this reference run.

`models/geometry/**` (11 files, 1537 LOC) is the trap this rule is written to avoid in the
other direction: it lives under `process/models/`, a whole-directory glob would sweep it
in, and it is imported **only** by `process/core/io/plot/summary.py` — plotting, never the
solve. It was reached zero times.

## Regenerating

There is no `functional_process` module for this yet. The measurement is a `sys.setprofile`
hook over one `_call_models_once`:

```python
sr = SingleRun("large_tokamak_eval.IN.DAT")
sr.initialise()
load_iteration_variables(sr.data)  # else xcm is all-NaN and the call raises
xc = sr.data.numerics.xcm[: int(sr.data.numerics.n_iteration_variables)]
# profile hook: record (f_code.co_filename, f_code.co_qualname) for every "call" event
# whose filename is under process/models/, and the models-frame caller of every call into
# process/core/coolprop_interface.py
Caller(sr.models, sr.data)._call_models_once(xc)
```

Use `co_qualname`, not `co_name`: with `co_name` the three distinct `run` methods in
`physics.py` (`Physics`, `PlasmaBeta`, `PlasmaInductance`) collapse into one entry and the
file's function count reads 9 instead of 11. That error was made and corrected in this
pass; every number below is the `co_qualname` version.

## The reference run and what its switches resolve to

`tests/regression/input_files/large_tokamak_eval.IN.DAT`, read through `SingleRun.initialise()`.
Values below were **read out of the assembled `DataStructure`**, not off the input file —
several of the decisive ones are not in the file at all and come from
`process/data_structure/*_variables.py` defaults:

| switch | value | source | selects |
|---|---|---|---|
| `.stellarator.istell` | 0 | default (`stellarator_variables.py:46`) | tokamak pipeline |
| `.ife.ife` | 0 | default (`ife_variables.py:253`) | not IFE |
| `.tfcoil.i_tf_sup` | 1 | default (`tfcoil_variables.py:261`) | `SUPERCONDUCTING` |
| `.superconducting_tfcoil.i_tf_turn_type` | 1 | default (`superconducting_tf_coil_variables.py:194`) | `CABLE_IN_CONDUIT` |
| `.fwbs.i_blanket_type` | 1 | default (`fwbs_variables.py:70`) | `CCFE_HCPB` |
| `.physics.itart` | 0 | default (`physics_variables.py:994`) | not a tight-aspect-ratio machine |
| `.costs.i_cost_model` | 0 | IN.DAT:112 | 1990 `Costs` |
| `.fwbs.i_p_coolant_pumping` | 3 | IN.DAT:172 | `MECHANICAL_WITH_PRESSURE_DROP` |
| `.heat_transport.ipowerflow` | 0 | IN.DAT:185 | — |
| `.costs.i_plant_availability` | 0 | IN.DAT:113 | `avail()` (the `else` arm, `availability.py:116`) |
| `.physics.i_plasma_pedestal` | 1 | default | pedestal parameterisation |
| `.times.pulsetimings` | 0 | default (`times_variables.py:12`) | — |
| `.vacuum.i_vacuum_pumping` | `"old"` | default | `VacuumOld` |
| `.buildings.i_bldgs_size` | 0 | default | `Bldgs` |
| `.physics.i_confinement_time` | 34 | IN.DAT:300 | `ITER_IPB98Y2` |
| `.tfcoil.n_tf_coils` | 16 | IN.DAT | — |
| `.numerics.i_figure_merit` | 7 | IN.DAT | — |

Four of these (`i_tf_sup`, `i_tf_turn_type`, `i_blanket_type`, `itart`) decide *four of the
seven conditionals in `_call_models_once`* and none of them is written in the input file.
Anyone re-deriving this scope from the IN.DAT alone will get the wrong branches.

## A. The call surface, in call order

Level 0 is `Caller._call_models_once` itself. Level 1 is a sub-model `Models.__init__`
injected into a level-0 model and which that model `.run()`s. `models.<name>` is the
`Models` attribute; the class and file are what the trace recorded.

| # | call | `file:line` | model | scope |
|---|---|---|---|---|
| 0 | `models.plasma_geom.run()` | `caller.py:284` | `physics/plasma_geometry.py::PlasmaGeom` | tokamak arm of a shared file |
| 1 | `models.build.run()` | `caller.py:288` | `build.py::Build` | tokamak-only |
| 2 | `models.physics.run()` | `caller.py:290` | `physics/physics.py::Physics` | tokamak arm of a shared file |
| 2.1 | `self.inductance.run()` | `physics.py:356` | `physics/physics.py::PlasmaInductance` | tokamak-only |
| 2.2 | `self.plasma_profile.run()` | `physics.py:370` | `physics/plasma_profiles.py::PlasmaProfile` | shared |
| 2.2a | → `TeProfile.run()`, `NeProfile.run()` | via `plasma_profiles.py` | `physics/profiles.py` | shared |
| 2.3 | `self.beta.run()` | `physics.py:429` | `physics/physics.py::PlasmaBeta` | tokamak-only |
| 2.4 | `self.dia_current.run()` | `physics.py:527` | `physics/plasma_current.py::PlasmaDiamagneticCurrent` | tokamak-only |
| 2.5 | `self.plasma_bootstrap_current.run()` | `physics.py:543` | `physics/bootstrap_current.py::PlasmaBootstrapCurrent` | tokamak-only |
| 2.6 | `self.plasma_transition.run()` | `physics.py:788` | `physics/l_h_transition.py::PlasmaConfinementTransition` | tokamak-only |
| 2.7 | `self.scrape_off_layer.run()` | `physics.py:832` | `physics/scrape_off_layer.py::ScrapeOffLayer` | tokamak-only |
| 2.8 | `self.density_limit.run()` | `physics.py:870` | `physics/density_limit.py::PlasmaDensityLimit` | tokamak-only |
| 3 | `models.cicc_sctfcoil.run()` | `caller.py:306` | `tfcoil/superconducting.py::CICCSuperconductingTFCoil` | tokamak-only |
| 4 | `models.pfcoil.run()` | `caller.py:319` | `pfcoil.py::PFCoil` | tokamak-only |
| 5 | `models.pulse.run()` | `caller.py:322` | `pulse.py::Pulse` | tokamak-only |
| 6 | `models.divertor.run()` | `caller.py:324` | `divertor.py::Divertor` | tokamak-only |
| 7 | `models.fw.run()` | `caller.py:327` | `fw.py::FirstWall` | tokamak-only |
| 8 | `models.shield.run()` | `caller.py:329` | `shield.py::Shield` | tokamak-only |
| 9 | `models.vacuum_vessel.run()` | `caller.py:331` | `vacuum.py::VacuumVessel` | tokamak-only |
| 10 | `models.ccfe_hcpb.run()` | `caller.py:345` | `blankets/hcpb.py::CCFE_HCPB` | tokamak-only *on this run* |
| 11 | `models.cryostat.run()` | `caller.py:351` | `cryostat.py::Cryostat` | tokamak-only |
| 12 | `models.structure.run()` | `caller.py:354` | `structure.py::Structure` | tokamak-only |
| 13 | `models.power.run()` | `caller.py:364` | `power.py::Power` | shared |
| 14 | `models.vacuum.run()` | `caller.py:367` | `vacuum.py::Vacuum` | shared |
| 15 | `models.buildings.run()` | `caller.py:370` | `buildings.py::Buildings` | shared |
| 16 | `models.power.acpow(output=False)` | `caller.py:376` | `power.py::Power` | shared |
| 17 | `models.power.plant_electric_production()` | `caller.py:379` | `power.py::Power` | shared |
| 18 | `models.availability.run()` | `caller.py:382` | `availability.py::Availability` | shared |
| 19 | `models.water_use.run()` | `caller.py:385` | `water_use.py::WaterUse` | tokamak-only |
| 20 | `models.costs.run()` | `caller.py:395` | `costs/costs.py::Costs` | shared |

`models.costs` is not an attribute — it is the `@property` at `main.py:744-764`, which
resolves `.costs.i_cost_model` to a whole `Model` instance before anything runs. At `0` it
returns `self._costs_1990`. This is the one place a `_call_models_once` branch is taken
outside `caller.py` itself.

**Reached without an explicit `.run()` in `caller.py`** — three files enter the surface
only through injection or inheritance, and a `caller.py`-only reading misses all three:

- `blankets/blanket_library.py` (14 functions, 822 lines) — `models.blanket_library` is
  constructed at `main.py:678` and **never called**; the file is reached because
  `CCFE_HCPB(OutboardBlanket, InboardBlanket)` (`hcpb.py:25`) inherits from
  `BlanketLibrary` (`blanket_library.py:56`).
- `tfcoil/base.py` (8 functions, 753 lines) — `models.tfcoil.run()` at `caller.py:361` is
  **not** called here (`itart == 0`), but `TFCoil`'s methods run anyway via
  `CICCSuperconductingTFCoil → SuperconductingTFCoil → TFCoil`.
- `engineering/ivc_functions.py` (3 functions) — plain module functions imported by
  `fw.py:14-16`, `shield.py:12-13`, `vacuum.py:13`.

Likewise `pfcoil.py`'s `CSCoil` (11 of the file's 24 entered functions) is reached through
the injected `PFCoil(cs_fatigue=..., cs_coil=...)` at `main.py:652`, and
`cs_fatigue.py::CsFatigue.ncycle` through `pfcoil.py:3492`.

## B. Every file reached — LOC, port status, CoolProp

`LOC` is the whole file. `entered LOC` is the union of the source-line spans of the
functions actually entered — the honest scope number, and still an upper bound, since an
entered function's span includes branches inside it that did not run. `fns` is entered
functions by `co_qualname`. `∩ stell` is how many of those are also entered by the
stellarator reference run (see §C). Status is per `unit_registry.md`.

| file | LOC | entered LOC | fns | ∩ stell | port status (`unit_registry.md`) | CoolProp |
|---|---|---|---|---|---|---|
| `costs/costs.py` | 3027 | 2170 | 42 | 42 | **ported** — unit #18, 41/43 methods, 43 nodes registered | |
| `power.py` | 2907 | 2134 | 17 | 11 | **partly** — unit #14 A/B/C ported; `Power.pfpwr` + 4 `_pf_loss_*` are new | |
| `physics/physics.py` | 6931 | 1681 | 11 | 3 | **partly** — unit #9 ported 8 methods, only 3 of them on this path | |
| `physics/confinement_time.py` | 4185 | 1070 | 3 | 2 | **ported** — unit #10; the tokamak arm `iter_ipb98y2` is already written | |
| `physics/fusion_reactions.py` | 1597 | 781 | 12 | 12 | **ported** — unit #19 | |
| `vacuum.py` | 994 | 568 | 7 | 4 | **partly** — unit #16 ported `Vacuum`; `VacuumVessel` unported | |
| `physics/plasma_geometry.py` | 1229 | 549 | 7 | 1 | **partly** — only `calculate_iter_physics_basis_elongation` is ported | |
| `physics/density_limit.py` | 696 | 531 | 11 | 0 | **partly** — only `calculate_greenwald_density_limit` (unit #21's note) | |
| `physics/profiles.py` | 558 | 440 | 13 | 10 | **ported** — unit #21, 12/12 | |
| `superconductors.py` | 1289 | 402 | 7 | 2 | **partly** — unit #22 ported 7+1; `superconductor_current_density_margin` explicitly out of scope | |
| `buildings.py` | 1536 | 368 | 2 | 2 | **ported** — unit #15 | |
| `availability.py` | 1593 | 351 | 3 | 2 | **ported** — unit #17 | |
| `physics/impurity_radiation.py` | 755 | 274 | 10 | 10 | **ported** — units #20/#23 | |
| `physics/radiation_power.py` | 243 | 213 | 2 | 2 | **ported** — unit #20 | |
| `physics/plasma_profiles.py` | 430 | 178 | 4 | 3 | **partly** — unit #12; `calculate_pedestal_profile_values` ported, no node | |
| `physics/exhaust.py` | 220 | 126 | 4 | 1 | **partly** — unit #11 ported `calculate_radiation_fraction` only | |
| `blankets/hcpb.py` | 1663 | 956 | 7 | 0 | **partly** — unit #13 ported 3 of the 7 entered, all unregistered | |
| `pfcoil.py` | 5285 | 3525 | 24 | 0 | **unported** — no registry row | |
| `tfcoil/superconducting.py` | 5153 | 2457 | 19 | 0 | **unported** — no registry row | |
| `build.py` | 2360 | 2306 | 6 | 0 | **unported** — no registry row | |
| `physics/bootstrap_current.py` | 2529 | 1228 | 14 | 0 | **unported** — no registry row | |
| `physics/l_h_transition.py` | 1476 | 1124 | 23 | 0 | **unported** — no registry row | |
| `blankets/blanket_library.py` | 3828 | 822 | 14 | 0 | **unported** — no registry row | |
| `tfcoil/base.py` | 4670 | 753 | 8 | 0 | **unported** — no registry row | |
| `physics/current_drive.py` | 2996 | 737 | 5 | 0 | **unported** — no registry row | |
| `tfcoil/quench.py` | 551 | 436 | 11 | 0 | **unported** — no registry row | **YES** |
| `physics/plasma_current.py` | 1175 | 296 | 4 | 0 | **unported** — no registry row | |
| `fw.py` | 866 | 299 | 6 | 0 | **unported** — no registry row | no (see §D) |
| `shield.py` | 482 | 270 | 4 | 0 | **unported** — no registry row | |
| `divertor.py` | 494 | 262 | 5 | 0 | **unported** — no registry row | |
| `water_use.py` | 323 | 265 | 7 | 0 | **unported** — no registry row | |
| `pulse.py` | 316 | 236 | 12 | 3 | **unported** — no registry row | |
| `physics/scrape_off_layer.py` | 328 | 226 | 5 | 0 | **unported** — no registry row | |
| `structure.py` | 231 | 200 | 2 | 0 | **unported** — `models/structure.py` is *not* unit 1D (that is `stellarator/structure.py`) | |
| `engineering/ivc_functions.py` | 306 | 128 | 3 | 0 | **unported** — no registry row | |
| `cs_fatigue.py` | 262 | 93 | 1 | 0 | **unported** — no registry row | |
| `cryostat.py` | 132 | 69 | 2 | 0 | **unported** — `models/cryostat.py` is *not* S5 (that is `stellarator.py:1282-1330`) | |
| `physics/plasma_fields.py` | 268 | 67 | 1 | 0 | **unported** — no registry row | |

### Totals

| | files | entered LOC | entered fns |
|---|---|---|---|
| **ported** (every entered function covered) | 8 | 5 667 | 87 |
| **partly ported** | 9 | 7 125 | 75 |
| **unported** | 21 | 15 799 | 176 |
| **total reached** | **38** | **28 591** | **338** |

Whole-file LOC of those 38 files is 63 884 — the number to quote only if the intent is
"how much source a tokamak's files contain", not "how much a tokamak runs". The entered-LOC
figure is 45 % of it.

`.costs.i_cost_model = 0` matters for the totals: `costs/costs.py`'s 2170 entered lines are
the single largest already-ported block, and they are ported **only** for arm `0`.
`costs_2015.py` (arm `1`, `unproduced`) is 2/13 functions and was reached zero times here.

## C. Device-agnostic or tokamak-specific — measured, not assumed

The measurement supplied for this task was taken on the *assembled stellarator graph*
(`costs` 40 nodes, `power` 20, `availability` 4, `vacuum` 3, `buildings` 2 touch no
`.stellarator*` data; `physics` 31 of 33 shared). That is a port-side, node-side number. To
cross-check it on the PROCESS side, the same profile hook was run a second time on
`tests/regression/input_files/stellarator_helias.IN.DAT` (`istell = 6`, reaching
`Stellarator.run()` at `caller.py:273`; `st_phys`, `st_fwbs`, `st_geom`, `st_strc`,
`st_new_config` all confirmed entered, so it is a genuine full pass) and the two function
sets were intersected.

**Result: 16 of the tokamak's 38 files are also entered by the stellarator run; 22 are
tokamak-only. 110 of the tokamak's 338 entered functions are shared, 8 122 of its 28 591
entered lines. The tokamak-new surface is 228 functions / 20 469 lines.**

This corroborates the supplied node-side measurement and sharpens it in three places:

1. **`costs/costs.py` is bit-identically shared.** Both devices enter the *same 42
   functions*; the tok-only and stell-only sets are both empty. So the cost nodes
   `model_tree_design.md` §8 step 4c deleted are a **port-side** pruning, not a PROCESS-side
   device difference — `tokamak_scope.md`'s warning that "`costs` will grow again" is
   correct, and the growth is purely restoring nodes for functions PROCESS was calling all
   along.
2. **`power.py` gains exactly one subsystem, not twenty.** Shared: 11 functions / 1522
   lines. Tokamak-new: `Power.run`, `Power.pfpwr` and its four `_pf_loss_*` helpers — the
   PF-coil power supply, which a stellarator has no PF coils to need. Nothing else in
   `power.py` is device-tied.
3. **`buildings.py`, `impurity_radiation.py`, `radiation_power.py`, `fusion_reactions.py`
   are 100 % shared** (empty diff both ways). `availability.py` differs by one frame only —
   the tokamak enters `Availability.run` (the real `i_plant_availability` dispatch) where
   the stellarator bypasses it (`stellarator.py:175` calls `avail()` directly, unit #17's
   "bypass" finding). At `i_plant_availability = 0` that dispatch lands in the `else` arm at
   `availability.py:116` — **`avail()`, the same arm the stellarator bypass reaches, and the
   arm already ported and registered as `Avail`/`CplifeAvail`**.

Only four reached files read `.stellarator`/`istell` at all, and in every case the
tokamak takes the `istell == 0` arm:

| file | sites |
|---|---|
| `physics/physics.py` | `:1835`, `:1844`, `:2171`, `:2196`, `:2779`, `:2784`, `:2823`, `:3131`, `:4575` |
| `physics/plasma_geometry.py` | `:523`, `:538`, `:606` |
| `physics/plasma_current.py` | `:82`, `:150`, and a read of `.stellarator.iotabar` at `:196` |
| `physics/confinement_time.py` | `:1331`, `:1388` — **output only** (`output_confinement_comparison`), not on the solve path |

That is the exact analogue of the supplied "`physics` is 31 of 33 shared" finding, and it
confirms `.physics.confinement_time.model` as one of the two device-tied physics nodes: the
only substantive `istell` split in `confinement_time.py` is which scaling law is selected.

Two registry predictions are confirmed live by this trace, both now actionable:

- Unit #21 left `GreenwaldDensityFractions`/`PedestalSeparatrixDensities` **ported but
  unregistered**, on the stated ground that `NeProfile.set_pedestal_and_separatrix_values`
  is *"reachable only from `physics.py` (unit #22, tokamak), never from `stellarator.py`"*.
  The tokamak trace enters it, from `physics.py:368`. The prediction was right and those two
  nodes have a graph to go into.
- Unit #16 recorded `VacuumVessel` as *"confirmed unreachable on the stellarator pipeline,
  no action needed"*. It is reached on the tokamak pipeline (`caller.py:331`), 3 functions.

One registry **path inaccuracy**, found incidentally and not fixed here: unit #22 is filed
as `physics/superconductors.py`; the file is `process/models/superconductors.py`. The
methods listed are correct.

## D. CoolProp — one module, and it is not the six

`next_steps.md` §5's wrapping policy is still unresolved, so a node blocked this way is
*scheduled* differently from one merely unwritten. Measured by recording the
`process/models/**` frame that calls into `process/core/coolprop_interface.py`:

> **Exactly one module reaches CoolProp on this reference run: `process/models/tfcoil/quench.py`,
> via `_quench_integrand_at_temperature` (`quench.py:332` and `:334`), called from
> `_quench_integrals` (`:392`) ← `calculate_quench_protection_current_density` (`:539`) ←
> `SuperconductingTFCoil.quench_heat_protection_current_density`
> (`superconducting.py:1366`). 450 CoolProp calls in a single `_call_models_once`.**
>
> It is **unported** and has no registry row. 436 entered lines, 11 entered functions,
> 1.5 % of the tokamak's entered surface — but it sits on the TF-coil quench chain, which
> constraints 34/35/36/74/75 all read.

The task brief named six CoolProp-bound modules. That list is a **weak** signal in exactly
the sense `tokamak_scope.md` §Traceability says it is — "some module in the neighbourhood
reaches CoolProp" — and on this run it over-counts by five. Precisely:

- `models/engineering/pumping.py` does **not** import `FluidProperties` at all. The only
  occurrence of the string "CoolProp" in it is a docstring at `:25` explaining that
  `CoolantType.full_name` must match a CoolProp fluid name. It was also reached zero times.
  **Five** modules import `FluidProperties`, not six: `fw.py:9`, `tfcoil/quench.py:9`,
  `stellarator/stellarator.py:15`, `blankets/blanket_library.py:10`, `blankets/hcpb.py:11`.
- `stellarator/stellarator.py` is out of scope by construction.
- `fw.py`, `blanket_library.py` and `hcpb.py` are all **reached**, but none of them reaches
  CoolProp here, because every one of their CoolProp sites is behind
  `.fwbs.i_p_coolant_pumping == MECHANICAL` (2) and this run sets **3**
  (`MECHANICAL_WITH_PRESSURE_DROP`). The dispatch is `hcpb.py:801-846`; the `MECHANICAL`
  arm at `:843` calls `primary_coolant_properties` (`blanket_library.py:691`, CoolProp at
  `:729/750/765`) and `thermo_hydraulic_model` (reaching `fw.fw_temp`, `fw.py:354`, CoolProp
  at `:425/432`, and `coolant_pumping_power`, `blanket_library.py:3440`, CoolProp at
  `:3507/3518`). None of `fw_temp`, `primary_coolant_properties`, `thermo_hydraulic_model`
  or `coolant_pumping_power` appears in the trace.
- `hcpb.py:794`'s remaining site is behind `.fwbs.i_blkt_coolant_type == CoolantType.WATER`
  (2); this run has `1` (`HELIUM`).

So: **CoolProp-bound-and-actually-reached is 1 module / 436 lines. CoolProp-bound-and-in-
scope-but-dormant is 3 more modules / 2 077 entered lines, live the moment
`i_p_coolant_pumping` moves to 2 or `i_blkt_coolant_type` to 2.** Both numbers should be
carried; quoting only the first understates what a second tokamak input file costs.

The stellarator reference run reaches CoolProp **zero** times, so this is genuinely a new
obstacle for the tokamak, not an inherited one.

## E. The 17 new topology decisions — where each one lives

`tokamak_scope.md` enumerated 17 decisions the port has never read. Below is the model each
is the site of, by grepping `.<switch>` reads across the reached surface. Read-count per
file in parentheses; the run's value is from §A's table or from the assembled `DataStructure`.

| # | decision | value here | site(s) — model file | that file's status |
|---|---|---|---|---|
| 1 | `i_plasma_current` | 4 | `physics/plasma_current.py` (4), `physics/physics.py` (2), `physics/plasma_geometry.py` (1) | unported / partly / partly |
| 2 | `i_plasma_geometry` | 0 | `physics/plasma_geometry.py` (15), `physics/plasma_current.py` (1) | partly / unported |
| 3 | `i_single_null` | 1 | `build.py` (4), `tfcoil/base.py` (1), `divertor.py` (1) | all unported |
| 4 | `i_alphaj` | 1 | `physics/physics.py` (3), `physics/plasma_current.py` (1) | partly / unported |
| 5 | `i_ind_plasma_internal_norm` | 1 | `physics/physics.py` (4) — `PlasmaInductance.get_ind_internal_norm_value`, entered | partly |
| 6 | `i_bootstrap_current` | 4 | `physics/bootstrap_current.py` (4), `physics/physics.py` (1) | unported / partly |
| 7 | `i_beta_component` | 1 | `physics/physics.py` (8) — `PlasmaBeta.get_beta_norm_max_value`, entered | partly |
| 8 | `i_density_limit` | 7 | `physics/density_limit.py` (5) | partly (1 of 11 fns) |
| 9 | `i_div_heat_load` | 2 | `divertor.py` (5) | unported |
| 10 | `i_hcd_primary` | 10 | `physics/current_drive.py` (14), `buildings.py` (4), `costs/costs.py` (1), `build.py` (1) | unported / ported / ported / unported |
| 11 | `i_pf_superconductor` | 3 | `pfcoil.py` (6), `costs/costs.py` (4) | unported / ported |
| 12 | `i_cs_superconductor` | 1 | `pfcoil.py` (8), `costs/costs.py` (4) | unported / ported |
| 13 | `n_pf_coil_groups` (a count) | 4 | `pfcoil.py` (26), `power.py` (1) | unported / partly |
| 14 | `i_shld_primary_heat` | 1 | `power.py` (4) | partly |
| 15 | `pulsetimings` | 0 | `physics/physics.py` (`:476`) — the **only** read in all of `process/models/**` | partly |
| 16 | `i_plant_availability` | 0 | `availability.py` (6) | **ported** — see below |
| 17 | `n_tf_coils` (a count) | 16 | `tfcoil/superconducting.py` (17), `buildings.py` (8), `tfcoil/base.py` (6), `power.py` (6), `costs/costs.py` (6); also `tfcoil/resistive.py` (30) — **not reached** | mixed |

Three things this table shows that the switch count alone could not:

- **Decision 16 is already occupied at this run's value.** `i_plant_availability = 0` falls
  through `availability.py:100-116` to `avail()`, which unit #17 has ported and registered
  as `Avail(ibkt_life=0, itart=0)` + `CplifeAvail(i_tf_sup=1, itart=0)` — and `itart = 0`
  is exactly what this reference run has. This is the same shape as `tokamak_scope.md`'s
  `i_confinement_time = 34` finding: *the refusal is about the absent device, not about
  absent physics*. Whether the other values (1 `WARD_TAYLOR`, 2 `MORRIS`, 3 `ST`) need arms
  is a different question — unit #17 records `WardTaylorAvailability` as written and
  deliberately unregistered, and `Avail2`/`AvailSt` as output-only.
- **Decisions 15 and 5 and 7 all land inside `physics.py`.** Together with 1, 4 and 6's
  `physics.py` sites, six of the seventeen are read inside the one 6931-line file of which
  only 3 functions are ported. `physics.py` is the single largest blocker in this scope by
  decision count, not just by LOC.
- **Decisions 11, 12 and 13 all land inside `pfcoil.py`** (5285 LOC, 3525 entered, 24
  functions, zero ported, zero shared with the stellarator). This is the biggest wholly-new
  model in the scope and it is the site of three decisions plus the `power.pfpwr` chain that
  hangs off it.

Six of the seventeen (`i_hcd_primary`, `i_pf_superconductor`, `i_cs_superconductor`,
`n_pf_coil_groups`, `i_shld_primary_heat`, `n_tf_coils`) have at least one read inside an
already-ported file (`costs/costs.py`, `buildings.py`, `power.py`) — meaning the port has
already reproduced the *consuming* formula and will need only the switch, not the model.

## F. What tracing could not settle, and what it could

Stated deliberately, because `next_steps.md` §13.11 records a number-that-was-not-measured
appearing as though it had been. **Twice.**

### Settled by measurement

- **Call-order state.** `_call_models_once` was run three times in one process and the
  entered-file and entered-function sets diffed pairwise. **No difference at either
  granularity.** So no `first_call`-style flag opens or closes a branch of the tokamak
  surface between passes — unlike `st_fwbs`'s `first_call_stfwbs` on the stellarator side
  (unit #1 chunk 1E2). This does *not* mean no cross-call state exists, only that none of
  it changes which code runs.
- **Every `_call_models_once` conditional.** All seven were evaluated by running them, not
  by reading them; §A's table and the switch table above record which arm each took.
- **Which arm every switch above selects.** Read out of the assembled `DataStructure`.

### Not settled

- **Only one point in the design space was traced.** `xc` is the initial iteration-variable
  vector after `load_iteration_variables`, not a converged solution, and every count here is
  from that one point. Data-dependent branches *inside* an entered function (a `jnp.where`-
  shaped `if` on a physical quantity, a `while` that iterates a different number of times)
  can differ elsewhere. The entered-LOC figure absorbs this — it counts a whole function's
  span once the function is entered — but the *function* counts do not, and a converged-point
  trace could add functions. It was not run.
- **Only one input file.** Every "conditional, not called" below is conditional *in general*
  and merely unselected *here*. A second tokamak IN.DAT with `i_tf_sup = 0`, `i_blanket_type
  = 5`, `itart = 1` or `i_p_coolant_pumping = 2` reaches a materially different surface, and
  the last of those changes the CoolProp answer (§D). The following are **conditional, not
  called**, with the condition:

  | not called | condition | file that stays out of scope |
  |---|---|---|
  | `models.stellarator.run()` (`caller.py:273`) | `.stellarator.istell != 0` | `models/stellarator/**` |
  | `models.ife.run()` (`caller.py:279`) | `.ife.ife != 0` | `models/ife.py` |
  | `models.copper_tf_coil.run()` (`caller.py:296`) | `i_tf_sup == 0` | `tfcoil/resistive.py` |
  | `models.croco_sctfcoil.run()` (`caller.py:313`) | `i_tf_sup == 1` **and** `i_tf_turn_type == 2` | `CROCOSuperconductingTFCoil` half of `superconducting.py` |
  | `models.aluminium_tf_coil.run()` (`caller.py:316`) | `i_tf_sup == 2` | `tfcoil/resistive.py` |
  | `models.dcll.run()` (`caller.py:349`) | `i_blanket_type == 5` | `blankets/dcll.py` |
  | `models.tfcoil.run()` (`caller.py:361`) | `itart == 1` **and** `i_tf_sup != 1` | — (`tfcoil/base.py` is reached anyway, by inheritance) |
  | `Costs2015.run()` (`main.py:756`) | `i_cost_model == 1` | `costs/costs_2015.py` |
  | `hcpb`/`blanket_library`/`fw` CoolProp arms | `i_p_coolant_pumping == 2`, or `i_blkt_coolant_type == 2` | — (files reached, arms dormant) |

- **The output path was not traced.** `Models.write` (`main.py:840-970`) calls
  `physics_detailed.output()` (`main.py:885`) and `current_drive.output()` (`main.py:888`),
  **neither of which appears anywhere in `_call_models_once`** — `DetailedPhysics` is
  reached only at the final point. Whether the port needs it is a scope decision this file
  does not make; it is flagged so nobody concludes from `caller.py` that
  `models.physics_detailed` is dead.
- **The solve wrapper was not traced.** `Caller.call_models` (`caller.py:100-133`) runs the
  whole surface **up to 10 times** per optimiser evaluation, and `objective_function` /
  `constraints.constraint_eqns` (`caller.py:103-104`) sit outside `_call_models_once`
  entirely. Both are already fully ported (all ~82 constraints, all 16 objective branches,
  `unit_registry.md`'s own tables), so nothing is missing — but their reads are *not* in the
  38-file surface measured here, and a boundary-input enumeration must add them.
- **Read/write sets are not derived here.** This is a *call* surface. `CLAUDE.md`'s standing
  difficulty — "no model declares its read set or write set" — is untouched by tracing which
  functions run. `check_boundary` on an assembled `Tokamak` machine is what
  `tokamak_scope.md` §"The order this implies" step 4 asks for, and it still needs the
  device class that step 3 builds.
- **`bootstrap_current.py`, `l_h_transition.py` and `density_limit.py` compute every
  variant, then select.** The trace enters 14, 23 and 11 functions in them respectively even
  though `i_bootstrap_current = 4`, `i_density_limit = 7` and one L-H threshold is wanted —
  because `get_bootstrap_current_fraction_value`, `l_h_threshold_power` and
  `get_density_limit_value` evaluate the whole family and index it. So for these three,
  the switch is **not** a topology switch at all: it selects a *value* from a computed
  vector, and the honest cottax shape is one node producing the family plus an index, not
  an `Alternative` per arm. That is a design finding this trace produced and it is *not*
  in `unit_registry.md`'s switches table.

## G. What this implies for order

Not a plan — `next_steps.md` owns the priority order. Three facts from the measurement that
bear on it:

1. **The already-ported 5 667 entered lines are all balance-of-plant plus the shared physics
   core**, and they arrive with zero new work beyond re-registration:
   `costs/costs.py` (2170, arm `0`), `confinement_time.py` (1070, arm `34` already written),
   `fusion_reactions.py` (781), `profiles.py` (440), `buildings.py` (368), `availability.py`
   (351, arm `0` already registered), `impurity_radiation.py` (274), `radiation_power.py`
   (213). That corroborates `tokamak_scope.md`'s "balance of plant is already there".
2. **Two files carry 40 % of the unported surface**: `pfcoil.py` (3525) and
   `tfcoil/superconducting.py` (2457), 6 264 lines together and the site of three of the 17
   decisions. Neither has a registry row, neither is shared with the stellarator, and
   `power.pfpwr` and half of `costs/costs.py`'s restored accounts hang off the first.
3. **`tfcoil/quench.py` is the only thing in this scope that `next_steps.md` §5's unresolved
   CoolProp policy actually blocks** — 436 lines, and it is on the quench-constraint chain.
   Everything else in the scope is merely unwritten, which is a different schedule.
