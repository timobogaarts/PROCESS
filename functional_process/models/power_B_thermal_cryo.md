---
kind: model-unit
status: draft
confidence: high
---

**Ported (5 units).** `power_B_thermal_cryo.py` / `test_power_B_thermal_cryo.py`:
`calculate_plant_thermal_efficiency`, `calculate_plant_thermal_efficiency_2`,
`calculate_component_thermal_powers`, `calculate_cryo`, `calculate_cryo_loads` --
all tier-1, all tests passing (value + gradient, `--fp-gradients`). Two genuine
PROCESS bugs found while building the harness (not fixed in `process/`, see below).

**Node-level split: all six `ComponentThermalPowers` self-loops are now
representable.** `calculate_component_thermal_powers` (the pure function) is
unchanged -- same signature, same values, all its original tests pass unmodified. At
the node level, each of the six self-references (`.power.delta_eta`,
`.heat_transport.eta_turbine`, `.heat_transport.etath_liq`,
`.heat_transport.temp_turbine_coolant_in`,
`.heat_transport.p_fw_div_heat_deposited_mw`,
`.primary_pumping.p_fw_blkt_coolant_pump_mw` -- see § "The `delta_eta` self-loop" and
§ "The five remaining self-loops" below) is now cut into its own tiny
`cottax.interfaces.pytree_namespace_module.FixedPointFunction`
(`DeltaEtaStep`, `EtaTurbineStep`, `EtathLiqStep`, `TempTurbineCoolantInStep`,
`PFwDivHeatDepositedMwStep`, `PFwBlktCoolantPumpMwStep`); `ComponentThermalPowers`
(the `ExplicitFunction`) keeps every other output and reads all six as plain,
current-value `Input`s. Six small pure helpers were extracted from
`calculate_component_thermal_powers`'s body -- verbatim, not reimplemented --
(`calculate_p_fw_blkt_coolant_pump_mw`, `calculate_p_fw_blkt_heat_deposited_mw`,
`calculate_p_shld_heat_deposited_mw`, `calculate_p_div_heat_deposited_mw`,
`calculate_delta_eta`, `calculate_p_fw_heat_deposited_mw`,
`calculate_p_fw_div_heat_deposited_mw`) so every `FixedPointFunction`'s `step` and the
main function share one source of the same logic; the two remaining splits
(`EtaTurbineStep`, `EtathLiqStep`, `TempTurbineCoolantInStep`) instead reuse the
already-shared `calculate_plant_thermal_efficiency`/`calculate_plant_thermal_efficiency_2`
directly, with an unused placeholder for whichever of their two return elements the
split doesn't need (confirmed by inspection that the entering value of that unused
parameter never affects the element being isolated -- see each class's own
docstring). `to_graph(DeltaEtaStep(...))`, `to_graph(EtaTurbineStep(...))`,
`to_graph(EtathLiqStep(...))`, `to_graph(TempTurbineCoolantInStep(...))`,
`to_graph(PFwDivHeatDepositedMwStep(...))` and
`to_graph(PFwBlktCoolantPumpMwStep(...))` all build cleanly (confirmed directly, see
"cottax node" below); **`to_graph(ComponentThermalPowers(...))` now also builds
cleanly** -- it no longer raises at all, since none of its six former self-loop
fields is declared as an `Output` any more.

**Findings**: `delta_eta`'s fixed point is numerically degenerate in a uniform way
(`d(delta_eta_next)/d(delta_eta) == 0` exactly, for every switch combination -- see
§ "The `delta_eta` self-loop"). The other five are **not** uniformly degenerate --
each is a *piecewise* identity/zero split, exactly `1` (pass-through) on the switch
values PROCESS leaves the field untouched and exactly `0` on the switch values that
recompute it from other inputs, confirmed by `jax.grad` per switch combination, never
a value strictly between `0` and `1`. See § "The five remaining self-loops" below for
the full per-switch table.

`ComponentThermalPowers` and the six `FixedPointFunction`s are **not** registered
together in any graph by this task -- registration into `total_process.py` is a
separate, later consolidation step.

## source

`process/models/power.py` (registry unit #14), chunk B of 3 (see
`power_A_tf_coil_power.md`/`power_C_electric_production.md` for the other two).

- `Power.component_thermal_powers` (814-1036) -- the plant thermal-power balance.
  Calls `plant_thermal_efficiency`/`plant_thermal_efficiency_2` internally.
- `Power.plant_thermal_efficiency` (1935-2071), `Power.plant_thermal_efficiency_2`
  (2073-2116) -- thermal-to-electric conversion efficiency correlations.
- `Power.calculate_cryo_loads` (1037-1118), `Power.cryo` (1773-1852) -- cryogenic
  heat loads and electric power demand.

## data footprint

Full parameter lists are in each ported function's docstring
(`power_B_thermal_cryo.py`); this table covers only the findings that shaped the
signatures, not every plain `explicit-arg` read (there are ~30 in
`component_thermal_powers` alone).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.power.delta_eta` | read (in `plant_thermal_efficiency`), write (in `component_thermal_powers`, later in the same call) | **implicit-io** | see "The `delta_eta` self-loop" below |
| `.primary_pumping.p_fw_blkt_coolant_pump_mw` | read/write, conditional on `.fwbs.i_p_coolant_pumping` | **conditional-ownership-by-run-config** | owned by `component_thermal_powers` for `USER_INPUT`/`FRACTION_OF_HEAT`; produced by `process/models/blankets/hcpb.py`/`blanket_library.py` (unit #13, not yet ported) for `MECHANICAL`/`MECHANICAL_WITH_PRESSURE_DROP` -- confirmed by grep, no other producer |
| `.heat_transport.p_fw_div_heat_deposited_mw` | read/write, conditional on the *same* switch but a *different* partition (`!= MECHANICAL_WITH_PRESSURE_DROP`, vs. `not in {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}` above) | **conditional-ownership-by-run-config** | only other producer in `process/` is `models/ife.py` (IFE, out of scope) |
| `.heat_transport.temp_turbine_coolant_in` | write, both in `plant_thermal_efficiency` (2 of 5 branches) and `plant_thermal_efficiency_2` (1 of 2 valid branches) | **implicit-io** | see "Write-ordering on `temp_turbine_coolant_in`" below |
| `.power.p_blkt_liquid_breeder_heat_deposited_mw` | write, conditional on `.fwbs.i_blkt_dual_coolant in {1, 2}` | conditional-ownership, but empirically inert | no other producer anywhere in `process/` (confirmed by grep) and its dataclass default is `0.0` -- ported as an unconditional `0.0` in the `else` arm, same convention as `helpow_cryal` below |
| `.fwbs.qnuc` | read/write, conditional on `inuclear == 0 and i_tf_sup == 1` | **conditional-ownership-by-run-config** | "Issue #511: if inuclear = 1: qnuc is input" -- an ordinary `InputVariable` otherwise |
| `.power.qss`/`.power.qac`/`.power.qcl`/`.power.qmisc` | write, only if `cryo()` is called at all (`calculate_cryo_loads`'s outer guard) | conditional call, not a per-field switch | see "`calculate_cryo_loads`'s outer guard" below |
| `.heat_transport.helpow_cryal` | write, conditional on `.tfcoil.i_tf_sup == 2` | conditional-ownership, but empirically inert | no other producer anywhere in `process/` (confirmed by grep), dataclass default `0.0` -- ported as unconditional `0.0` in the `else` arm |

## The `delta_eta` self-loop

`component_thermal_powers` calls `plant_thermal_efficiency` (line 964) **before**
computing `f_p_div_primary_heat`/`delta_eta` (lines 1009-1014) in the same call --
but `plant_thermal_efficiency`'s `CCFE_HCPB_VALUE_WITH_DIVERTOR` and
`STEAM_RANKINE_CYCLE` branches read `.power.delta_eta` (lines 1987, 2029). So for
those two switch values, `plant_thermal_efficiency` reads **this call's `delta_eta`
input as whatever the *previous* call to `component_thermal_powers` left there** --
a genuine same-field stale read, structurally identical to the worked example
`_audit/test_harness.md` documents for `power_at_ignition_point`/`st_phys`
(`.physics.b_plasma_surface_poloidal_average`), except here the two "calls" are two
separate invocations of `component_thermal_powers` across PROCESS's own
`Caller.call_models` idempotence loop (up to 10 full-pipeline passes), not two
explicit calls inside one method.

**Node-level resolution (this pass): cut, not decided-and-deferred any more.**
`calculate_component_thermal_powers` (the pure function) is unchanged -- it still
takes `delta_eta` as an ordinary parameter (the entering value) and returns a
freshly-computed `delta_eta` in its return tuple, on the same conceptual `VarPath`.
At the node level, `power_B_thermal_cryo.py` now declares this self-reference as a
`cottax.interfaces.pytree_namespace_module.FixedPointFunction`, `DeltaEtaStep`: its
`step` reads the real `.power.delta_eta` and (via the `FixedPointFunction` base's
built-in mint) writes a `^cond.power.delta_eta` copy; the paired `FixedPoint` problem
node reads that copy and owns the real `.power.delta_eta`. Confirmed directly:
`to_graph(DeltaEtaStep(...))` builds; `to_graph(ComponentThermalPowers(...))` raised
`ValueError: reads ['.power.delta_eta', ...], which it also owns` before this split
existed, and (once the five remaining splits below landed in the same pass) no longer
raises at all. Only reachable when `i_thermal_electric_conversion` selects
`CCFE_HCPB_VALUE_WITH_DIVERTOR` or `STEAM_RANKINE_CYCLE`; for the other three values
`delta_eta` is written but never read back by `plant_thermal_efficiency`, so the loop
is syntactically present but inert for those switch values.

**A driver for the resulting `FixedPoint` problem is still not assigned** (deliberately
-- `_audit/next_steps.md` § 5's "What stays deferred" applies to this cut exactly as
to every other Shape B instance; representing the self-reference is a structural
requirement, driving it is a separate later decision). If one ever is, this task found
the closure is trivial: `d(delta_eta_next)/d(delta_eta) == 0` **exactly** (confirmed
by `jax.grad`, not an approximation) -- the two `plant_thermal_efficiency` branches
that read `delta_eta` only affect `eta_turbine`, and nothing `calculate_delta_eta`
reads is derived from `eta_turbine`. So any fixed-point iteration here converges in
exactly one step from any starting value, regardless of algorithm. See
`calculate_delta_eta`'s and `DeltaEtaStep`'s docstrings in `power_B_thermal_cryo.py`,
and `test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta` in the test file.

## The five remaining self-loops

Five more genuine single-node self-loops in `calculate_component_thermal_powers`,
found while resolving `delta_eta` above and resolved in this same pass. Each is read
as an ordinary parameter (the entering value) and written again, on the same
`VarPath`, later in the same call -- same shape as `delta_eta`, same
`FixedPointFunction` cut, same "confirmed by inspection + `jax.grad`, not assumed"
verification. Unlike `delta_eta`, **none of these five is uniformly degenerate** --
each is a piecewise identity/zero split across the switch value(s) that gate it,
confirmed per-combination, not asserted from the shape alone.

| `VarPath` | node | entering value's role | gradient (confirmed by `jax.grad`) |
|---|---|---|---|
| `.heat_transport.eta_turbine` | `EtaTurbineStep` | read/written by `calculate_plant_thermal_efficiency` (line 964-ish, `power.py`) | `1` (identity) on the pass-through sub-branches (`USER_INPUT`; `CCFE_HCPB_VALUE`/`CCFE_HCPB_VALUE_WITH_DIVERTOR` with `i_blanket_type != CCFE_HCPB`; the `i_thermal_electric_conversion` default arm); `0` on the sub-branches that overwrite it (`CCFE_HCPB_VALUE`/`CCFE_HCPB_VALUE_WITH_DIVERTOR` with `i_blanket_type == CCFE_HCPB`, `STEAM_RANKINE_CYCLE` with a matching blanket, `SUPERCRITICAL_CO2_BRAYTON_CYCLE`) |
| `.heat_transport.etath_liq` | `EtathLiqStep` | read/written by `calculate_plant_thermal_efficiency_2` | `1` for `secondary_cycle_liq == 2` (plain pass-through); `0` for `== 4` (recomputed from `outlet_temp_liq` alone) |
| `.heat_transport.temp_turbine_coolant_in` | `TempTurbineCoolantInStep` | read/written by *both* `calculate_plant_thermal_efficiency` and `calculate_plant_thermal_efficiency_2`, in sequence -- see "Write-ordering" below | `1` only when *both* stages pass it through unchanged (`i_thermal_electric_conversion` selects `CCFE_HCPB_VALUE`/`CCFE_HCPB_VALUE_WITH_DIVERTOR`/`USER_INPUT`/default **and** `secondary_cycle_liq == 2`); `0` for every other combination |
| `.heat_transport.p_fw_div_heat_deposited_mw` | `PFwDivHeatDepositedMwStep` | conditional-ownership pass-through, `i_p_coolant_pumping` (line 955-961, `power.py`) | `1` for `MECHANICAL_WITH_PRESSURE_DROP` (pass-through, owned elsewhere -- `models/ife.py`); `0` for every other value (recomputed from `p_fw_heat_deposited_mw + p_div_heat_deposited_mw`) |
| `.primary_pumping.p_fw_blkt_coolant_pump_mw` | `PFwBlktCoolantPumpMwStep` | conditional-ownership pass-through, `i_p_coolant_pumping` (line 815-820, `power.py`) | `1` for `MECHANICAL`/`MECHANICAL_WITH_PRESSURE_DROP` (pass-through, owned elsewhere -- `models/blankets/hcpb.py`/`blanket_library.py`, unit #13); `0` for `USER_INPUT`/`FRACTION_OF_HEAT` (recomputed from `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw`) |

**Verified against the full function, not just the extracted helpers.** For two full
switch-combination sweeps (all pass-through and all overwrite, at the
`calculate_component_thermal_powers` level, not just each `step` in isolation), each
`FixedPointFunction.step`'s output was checked to match the corresponding element of
`calculate_component_thermal_powers`'s return tuple exactly (`==`, not `approx`) --
see each `test_*_step_matches_calculate_component_thermal_powers` test.

**Reuse over reimplementation, same discipline `DeltaEtaStep` established.**
`PFwDivHeatDepositedMwStep`/`PFwBlktCoolantPumpMwStep` share the extracted helpers
`calculate_p_fw_heat_deposited_mw`/`calculate_p_fw_div_heat_deposited_mw`/
`calculate_p_div_heat_deposited_mw`/`calculate_p_fw_blkt_coolant_pump_mw` with
`calculate_component_thermal_powers` itself (the first two are new this pass, the
latter two already existed for `DeltaEtaStep`). `EtaTurbineStep`/`EtathLiqStep`/
`TempTurbineCoolantInStep` instead reuse `calculate_plant_thermal_efficiency`/
`calculate_plant_thermal_efficiency_2` directly, passing an unused `0.0` placeholder
for whichever parameter of the *other* return element that split doesn't need
(`temp_turbine_coolant_in` for the first two; `eta_turbine`/`etath_liq`/`delta_eta`
for the third) -- confirmed by inspecting every branch of both functions that the
entering value of the placeholder parameter never influences the element being
isolated, so `0.0` never leaks into a real output. This was the cheaper path for
these three: extracting a parallel eta_turbine-only/etath_liq-only/
temp_turbine_coolant_in-only helper would have meant duplicating the same five-way
(`calculate_plant_thermal_efficiency`) or two-way (`calculate_plant_thermal_efficiency_2`)
branch structure a second time, whereas the shared function already *is* that logic.

**All five self-contained, not wired to `ComponentThermalPowers`'s own outputs** --
same "read the same raw inputs twice, fan out, don't create a two-node cycle" pattern
`DeltaEtaStep`'s docstring already argues for.

## Write-ordering on `temp_turbine_coolant_in`

`component_thermal_powers` calls `plant_thermal_efficiency` then
`plant_thermal_efficiency_2`, in that order. Both may write
`.heat_transport.temp_turbine_coolant_in` (from different source temperatures --
`temp_blkt_coolant_out` vs. `outlet_temp_liq`). If both branches happen to write it
in one call, only `plant_thermal_efficiency_2`'s value survives -- an order-dependent
overwrite, not a bug PROCESS guards against, but worth knowing before assuming the
field tracks whichever efficiency correlation "owns" it.
`calculate_component_thermal_powers` reproduces this by threading
`temp_turbine_coolant_in` through both calls in the same order.

## `calculate_cryo_loads`'s outer guard

`Power.calculate_cryo_loads` only calls `self.cryo(...)` at all when `i_tf_sup == 1`
or `i_pf_conductor == SUPERCONDUCTING` -- a whole-function conditional call, not a
per-field `jnp.where` (same shape as `vacuum.py`'s branch dispatch, see
`vacuum.md`). When neither holds, `.power.qss`/`qac`/`qcl`/`qmisc`/`.fwbs.qnuc` are
left completely untouched by this PROCESS method (not even read), so
`calculate_cryo_loads` takes `qss`/`qac`/`qcl`/`qmisc` as ordinary pass-through
parameters (same treatment `qnuc` already needs for `calculate_cryo` itself).
`helpow`/`p_cryo_plant_electric_mw` are *not* pass-throughs -- PROCESS unconditionally
(re-)initialises both to `0.0` at the top of the method regardless of the guard.

## Real findings (documented, not fixed)

1. **`ElectricConversionModelTypes.SUPERCRITICAL_CO2_CYCLE` does not exist**
   (`power.py:2038`) -- the real enum member is `SUPERCRITICAL_CO2_BRAYTON_CYCLE`
   (value 4). Calling `Power.plant_thermal_efficiency` with
   `i_thermal_electric_conversion == 4` therefore raises `AttributeError`
   unconditionally, before any physics runs -- **this branch of PROCESS's own model
   has never been reachable**. Confirmed directly:
   ```
   >>> ElectricConversionModelTypes.SUPERCRITICAL_CO2_CYCLE
   AttributeError: type object 'ElectricConversionModelTypes' has no attribute
   'SUPERCRITICAL_CO2_CYCLE'
   ```
   The port uses the correct member name (the intent is unambiguous from the
   branch's own body, a working supercritical-CO2 correlation identical in shape to
   `plant_thermal_efficiency_2`'s `secondary_cycle_liq == 4` branch), but this means
   the harness **cannot** sample `i_thermal_electric_conversion == 4` and diff
   against PROCESS -- excluded from `test_power_B_thermal_cryo.py`'s fuzz combos, see
   open questions.
2. **`logger.log(f"{'i_blanket_type is not equal to 1.'}")` crashes** at
   `power.py:2033` (and the equivalent lines in the `CCFE_HCPB_VALUE`/
   `STEAM_RANKINE_CYCLE` branches, ~1973/2002) -- `logging.Logger.log` requires
   `(level, msg, *args)`; called with one positional argument, Python raises
   `TypeError: Logger.log() missing 1 required positional argument: 'msg'`. Found
   directly by this port's own fuzzing (a `Sample` with `i_blanket_type ==
   BlktModelTypes.DCLL` and `i_thermal_electric_conversion ==
   CCFE_HCPB_VALUE_WITH_DIVERTOR` crashed `test_value_agreement`'s reference call,
   not the port). **Any input where `i_blanket_type != CCFE_HCPB` reaches this log
   line crashes PROCESS outright**, for all three switch values that guard on
   `i_blanket_type` (`CCFE_HCPB_VALUE`, `CCFE_HCPB_VALUE_WITH_DIVERTOR`,
   `STEAM_RANKINE_CYCLE`) -- i.e. any DCLL-blanket run (`i_blanket_type = 5`, the only
   other declared `BlktModelTypes` value) using one of those three
   `i_thermal_electric_conversion` settings cannot complete a call to
   `plant_thermal_efficiency`. This port's "pass `eta_turbine` through unchanged"
   behaviour on that branch is the obviously-intended fallback (same shape as the
   branch's own `if/else` structure), but cannot be diffed against a PROCESS
   reference that never returns -- excluded from the harness, see open questions.
3. **`i_thermal_electric_conversion` is used with two different reads-sets in two
   different methods.** `plant_thermal_efficiency` treats it as a genuine 5-way
   switch; `component_thermal_powers` (line 977) treats the same field as a
   **binary** switch (`CCFE_HCPB_VALUE` vs. everything else). Not a bug -- the two
   methods are answering different questions (which efficiency correlation to use,
   vs. whether divertor heat counts as primary or secondary) -- but worth knowing
   before assuming one switch implies one reads-set across a whole file (see also
   chunk A's `res_tf_leg == 0.0` / `i_tf_sup == 1` finding, a similar "two
   independent uses of a related concept" shape).

## proposed signature(s)

All five actually written in `power_B_thermal_cryo.py`:
`calculate_plant_thermal_efficiency`, `calculate_plant_thermal_efficiency_2`,
`calculate_component_thermal_powers`, `calculate_cryo`, `calculate_cryo_loads` --
see each function's own docstring for its full parameter list and return tuple.

## cottax node

`PlantThermalEfficiency`, `PlantThermalEfficiency2`, `ComponentThermalPowers`,
`Cryo`, `CryoLoads` -- all `ExplicitFunction`, actually written in
`power_B_thermal_cryo.py`. Every switch (`i_thermal_electric_conversion`,
`i_blanket_type`, `secondary_cycle_liq`, `i_p_coolant_pumping`,
`i_blkt_dual_coolant`, `i_tf_sup`, `i_pf_conductor`, `inuclear`) is a plain
`eqx.field(static=True)` on the relevant node, per `naming_convention.md` and the
`FastAlphaBeta` precedent (`physics_A_pure_formulas.py`) -- none is wrapped in
a read, since `cottax`'s `NodalDeclaration.__check_init__` requires every `__call__`
parameter to carry a `From(...)`/`FromExactly(...)` default, so a static config value
cannot be an ordinary parameter at all on this surface.

**`DeltaEtaStep`, added this pass**: a `FixedPointFunction` (not an `ExplicitFunction`)
isolating `.power.delta_eta`'s self-reference out of `ComponentThermalPowers`. Static
fields: `i_p_coolant_pumping`, `i_blkt_dual_coolant`, `i_thermal_electric_conversion`
(the three switches its own computation needs; `i_blanket_type`/`secondary_cycle_liq`
are not, since those only affect `eta_turbine`/`etath_liq`, not `delta_eta`). Its
`step` deliberately reads the same *raw*, externally-owned inputs
`calculate_component_thermal_powers` itself reads (via the four shared helper
functions), rather than `ComponentThermalPowers`'s own `Output`s for the same
intermediate quantities (`p_fw_blkt_heat_deposited_mw`, `p_shld_heat_deposited_mw`,
`p_div_heat_deposited_mw`) -- reading those `Output`s instead would recreate a
two-node cycle (`ComponentThermalPowers` -> `DeltaEtaStep` -> `ComponentThermalPowers`)
rather than cutting one, exactly the risk `_audit/next_steps.md` § 5 names: *"splitting
`component_thermal_powers` would very plausibly just turn one self-referencing node
into two mutually-referencing ones representing the same local loop, not reveal a new
one."* Confirmed directly: `to_graph(DeltaEtaStep(i_p_coolant_pumping=..., ...))`
builds a two-node graph (`['DeltaEtaStep']`, `^problem['DeltaEtaStep']`) for every
switch combination this chunk supports (`test_delta_eta_step_to_graph_builds`,
parametrised over all four).

**`EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep`/`PFwDivHeatDepositedMwStep`
/`PFwBlktCoolantPumpMwStep`, added this pass**: five more `FixedPointFunction`s,
same shape and same "read the same raw inputs, don't recreate a two-node cycle"
discipline as `DeltaEtaStep`. Static fields: `EtaTurbineStep`
(`i_thermal_electric_conversion`, `i_blanket_type`); `EtathLiqStep`
(`secondary_cycle_liq`); `TempTurbineCoolantInStep` (all three:
`i_thermal_electric_conversion`, `i_blanket_type`, `secondary_cycle_liq`, since both
stages it composes are switch-gated); `PFwDivHeatDepositedMwStep`/
`PFwBlktCoolantPumpMwStep` (`i_p_coolant_pumping` only -- each reads a different
partition of the same switch, see `calculate_component_thermal_powers`'s own
docstring). `EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep` reuse
`calculate_plant_thermal_efficiency`/`calculate_plant_thermal_efficiency_2` directly
with an unused placeholder for the return element they don't need (see § "The five
remaining self-loops" for why this is safe); `PFwDivHeatDepositedMwStep` uses the two
newly extracted helpers `calculate_p_fw_heat_deposited_mw`/
`calculate_p_fw_div_heat_deposited_mw` plus the already-shared
`calculate_p_div_heat_deposited_mw`; `PFwBlktCoolantPumpMwStep` reuses the
already-shared `calculate_p_fw_blkt_coolant_pump_mw` directly, unmodified. Confirmed
directly: `to_graph(...)` builds a two-node graph for each, for every switch
combination this chunk supports (`test_eta_turbine_step_to_graph_builds`,
`test_etath_liq_step_to_graph_builds`,
`test_temp_turbine_coolant_in_step_to_graph_builds`,
`test_p_fw_div_heat_deposited_mw_step_to_graph_builds`,
`test_p_fw_blkt_coolant_pump_mw_step_to_graph_builds`).

**`ComponentThermalPowers`, adjusted**: no longer declares any of the six
self-referencing fields as `Output`s (ownership moved to the six `FixedPointFunction`s'
`FixedPoint` problem nodes); still reads all six as plain `Input`s (the current
values), and its `__call__` still calls the unmodified `calculate_component_thermal_powers`
and forwards every other element of its return tuple, dropped by named unpacking (the
six dropped indices -- `0`, `15`, `16`, `17`, `18`, `23` -- are not contiguous, so a
slice no longer suffices the way it did for `delta_eta` alone). **`to_graph(ComponentThermalPowers(...))`
no longer raises at all** (`test_component_thermal_powers_to_graph_builds_cleanly`) --
this was the actual point of this pass: before it, the five fields beyond `delta_eta`
(`.heat_transport.eta_turbine`/`etath_liq`/`temp_turbine_coolant_in`,
`.heat_transport.p_fw_div_heat_deposited_mw`,
`.primary_pumping.p_fw_blkt_coolant_pump_mw`) were each a pre-existing self-loop of
the same shape (`Output`'s `where` equals one `Input`'s `where`), confirmed by direct
`to_graph` probing while building the `delta_eta` split, and flagged there for
whoever did this pass next.

## tier signal

All five **tier 1**: no internal iteration anywhere in this chunk.
`cryo`/`calculate_cryo_loads` read as though they might be an iterative solve from
their names, but both are single straight-line evaluations.

## switches touched

- `.fwbs.i_thermal_electric_conversion` (0-4) -- **formula-changing, different
  reads/writes-sets per value** in `plant_thermal_efficiency` (5-way); **binary** in
  `component_thermal_powers` (`CCFE_HCPB_VALUE` vs. else, see real finding #3). Ported
  as a static field, plain `if`/`elif`, not `jnp.where` (unlike `vacuum.py`'s
  precedent) -- the branches' reads-sets are too different to share one evaluated
  expression cheaply, and value `4` cannot even be tested against PROCESS (real
  finding #1).
- `.fwbs.i_blanket_type` (1 `CCFE_HCPB` / 5 `DCLL`) -- formula-changing inside three
  of `plant_thermal_efficiency`'s branches; the non-`CCFE_HCPB` arm crashes PROCESS
  outright (real finding #2), so this switch's "other" value is a live crash trigger
  in current PROCESS, not merely an unhandled formula.
- `.fwbs.secondary_cycle_liq` (2 user input / 4 supercritical CO2; anything else
  raises `ProcessValueError`) -- static, plain `if`/`raise`.
- `.fwbs.i_p_coolant_pumping` (`PumpingPowerModelTypes`, 4 values) -- two different
  partitions used in `component_thermal_powers` (see conditional-ownership findings
  above) -- static.
- `.fwbs.i_blkt_dual_coolant` (0/1/2) -- static, three formula variants.
- `.tfcoil.i_tf_sup` (0/1/2), `.pf_coil.i_pf_conductor` (`PFConductorModel`),
  `.fwbs.inuclear` (0/1) -- static, gate `cryo`/`calculate_cryo_loads`'s branches.

None of these are yet in `_audit/core/solver/switches.md` (out of this fork's edit
boundary) -- flagging for the coordinating session, same as chunk A's `i_tf_sup`.

## calls into other models

None. `component_thermal_powers` reads several `.fwbs.*`/`.current_drive.*`/
`.physics.*` fields whose values originate in other, not-yet-ported units (blanket
neutronics, current drive, plasma physics), but it does not call any of their
methods -- ordinary `Input` edges, not entanglement, per this task's distinction
between "reads a field" and "calls a model."

## JAX-difficulty flags

None found in this chunk beyond the switch-resolution choices already covered above
(all handled with static Python control flow, not `jnp.where`, so no
unguarded-branch-NaN risk of the kind chunk A's `res_tf_leg == 0.0` singularity or
`physics_A_pure_formulas.md`'s denominator guards describe).

## open questions

1. **`i_thermal_electric_conversion == 4` has no automated value-agreement
   coverage** -- PROCESS's own reference crashes there (real finding #1). Not
   resolvable without either fixing `process/power.py` (out of scope, "document,
   don't fix") or adding a `reference_domain_errors`-style carve-out to
   `Tier1Contract` for reference *crashes* rather than domain rejections (a
   different failure mode than what that mechanism was built for -- flagging for
   whoever next revises `_harness/contracts.py`, not resolved here).
2. **`i_blanket_type != CCFE_HCPB` inside `plant_thermal_efficiency`'s
   `CCFE_HCPB_VALUE`/`CCFE_HCPB_VALUE_WITH_DIVERTOR`/`STEAM_RANKINE_CYCLE` branches
   has no automated coverage either**, for the same reason (real finding #2) --
   PROCESS's reference raises `TypeError` from inside its own logging call, not a
   `ProcessValueError`/domain rejection. Same open question as #1: a genuine
   PROCESS-crashes-here case, not a porting gap.
3. **[Resolved, both passes] All six `ComponentThermalPowers` self-loops are now
   represented as `FixedPointFunction`s** (`DeltaEtaStep`, `EtaTurbineStep`,
   `EtathLiqStep`, `TempTurbineCoolantInStep`, `PFwDivHeatDepositedMwStep`,
   `PFwBlktCoolantPumpMwStep`) **and every one's closure behaviour is checked, not
   assumed**: `delta_eta`'s is uniformly degenerate (`d(delta_eta_next)/d(delta_eta)
   == 0` exactly, every switch combination); the other five are piecewise
   identity/zero, per switch combination -- see § "The five remaining self-loops"
   for the full table. `to_graph(ComponentThermalPowers(...))` no longer raises at
   all (`test_component_thermal_powers_to_graph_builds_cleanly`). **What remains
   open**: (a) no driver is actually assigned to any of the six resulting
   `FixedPoint` problem nodes (deliberately deferred, per `_audit/next_steps.md` § 5's
   "What stays deferred"); (b) `ComponentThermalPowers` and the six
   `FixedPointFunction`s are not registered together in one graph by this task
   (deferred to `total_process.py`'s later consolidation pass); and (c) whether
   PROCESS's own `Caller.call_models` re-run loop actually converges each field to
   the same fixed point its node's `FixedPoint` problem would (given each gradient is
   exactly `0` or exactly `1`, trivially yes in exact arithmetic for every case except
   possibly a chain of several `1`-gradient pass-throughs compounding across
   `call_models`'s outer iterations, but not cross-checked against a PROCESS run
   here).


## Update: six 1-tuple returns fixed

`DeltaEtaStep`, `EtaTurbineStep`, `EtathLiqStep`, `TempTurbineCoolantInStep`,
`PFwDivHeatDepositedMwStep` and `PFwBlktCoolantPumpMwStep` each ended their `step` with
`return (x,)` while declaring a **single** `Output`. `cottax.evaluate._bind`'s
`len(owns) == 1` branch takes the whole return as that one output, so the env held
`(value,)` instead of `value`.

Invisible to `PicardDriver` -- `ravel_pytree` flattens a 1-tuple happily and the
comparison happens on the flat vector -- and **fatal to `rewrites.Residualise`**, whose
minted `Compare` body is `operator.sub(condition, unknown)` and raises
`TypeError: unsupported operand type(s) for -: 'tuple' and 'ArrayImpl'`. So this was a
prerequisite for the `Optimise` layer, not a tidy-up: see
`_audit/optimise_design.md` §6.4a and §10.3.

Same bug class as `ZTfInsideHalf`'s (`next_steps.md` §8), now three instances. A sweep of
`grep -n 'return (.*,)$' functional_process/models/` found one more genuine case
(`VacuumPumpingSimple`, `vacuum.py:180`, also fixed) and two false positives in
`AbstractDriver.__call__` implementations, where a tuple **is** the contract.


## Update: the `Cryo`/`CryoLoads` self-loops, cut — the unit is now fully registered

`boundary_inputs_audit.md` §7 item 7. `Cryo` and `CryoLoads` were ported and
deliberately unregistered because each reads and owns `.fwbs.qnuc` (and `CryoLoads`
also `.power.qss`/`qac`/`qcl`/`qmisc`), the Shape-B construction error
`next_steps.md` §5 defines. Both are conditional-ownership pass-throughs, described
in "`calculate_cryo_loads`'s outer guard" above; that section is the design input for
this split and is unchanged.

**The split written, and why this shape.** `calculate_cryo_loads` keeps its signature
and its Tier-1 contract against real PROCESS, but its body is now assembled from four
pieces, so the node-level split cannot drift from the function that contract
validates:

| piece | node | owns | condition |
|---|---|---|---|
| `calculate_cryo_qnuc` | `CryoQNucStep` (`FixedPointFunction`) | `.fwbs.qnuc` | `inuclear == 0 and i_tf_sup == 1` |
| `calculate_cryo_q_loads` | `CryoQLoadsStep` (`FixedPointFunction`) | `.power.qss`/`qac`/`qcl`/`qmisc` | `cryo_is_active` — `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` |
| `calculate_cryo_plant_loads` | `CryoLoads` (`ExplicitFunction`) | `.heat_transport.helpow`/`p_cryo_plant_electric_mw`/`helpow_cryal`, `.tfcoil.cryo_cool_req` | unconditional |
| `calculate_helpow` | — (shared helper) | — | — |

`Cryo` itself stays **unregistered**, in exactly the position
`PlantThermalEfficiency`/`PlantThermalEfficiency2` hold: the raw un-split node for a
function `to_graph` will not build, kept because `calculate_cryo` is still what the
Tier-1 contract compares against `Power.cryo`.

**Two fixed points, not one, and that is the load-bearing decision.** The five `q*`
fields are written by one PROCESS function but carry **two different** ownership
conditions. A single `FixedPointFunction` owning all five would, whenever
`inuclear == 1` with `i_tf_sup == 1`, produce a residual block that is the identity on
the `qnuc` row and constant on the other four — one structurally-zero row in an
otherwise well-posed block. `sand.degenerate_fixed_points` drops a problem only when
the residual vanishes *entirely*, so it could not drop that, and the SAND equality
block would be silently rank-deficient. Splitting by condition keeps each problem
uniformly degenerate or uniformly well-posed, which is the property that detector
needs. Under `REFERENCE_CONFIGURATION` (`i_tf_sup = 1`, `inuclear = 0`,
`i_pf_conductor = 0`) both are well-posed and neither is dropped —
`run_sand_harness.py` lists both under `residualised`, and both SAND residuals are
exactly `0.0` at PROCESS's converged point.

`CryoQLoadsStep` reads `.fwbs.qnuc` (the real path `CryoQNucStep`'s `FixedPoint` owns)
rather than recomputing it, because `qmisc = 0.45 * (qss + qnuc + qac + qcl)` uses the
value `Power.cryo` has just written. That is an ordinary edge, not a cycle:
`CryoQNucStep` reads only `.fwbs.p_tf_nuclear_heat_mw` and its own minted copy.
`DeltaEtaStep`'s "rebuild from raw inputs rather than read another node's `Output`"
rule exists to avoid recreating a two-node cycle; there is no cycle to avoid here.

**One rearrangement inside `calculate_cryo_loads`, stated because it looks like a
change and is not**: `calculate_cryo_qnuc` is now called *outside* the superconducting
guard, where PROCESS's `qnuc` write sits inside it. The inner condition is
`inuclear == 0 and i_tf_sup == 1`, and `i_tf_sup == 1` already implies the outer guard,
so the two are identical on every input.

**Prerequisite closed with it**: `.tfcoil.tfcryoarea` had no producer anywhere in the
graph. It is now `coils/calculate.py`'s `TfCryoArea`, carved out of `st_coil`'s inline
geometry block exactly as `ZTfInsideHalf` was — see `calculate.md`. Without it,
registering these nodes would have traded two boundary inputs for one new one.

**Measured.** `pytest functional_process` 3619 → 3643 passed, 0 failed (24 new tests,
20 of them node-level: `to_graph` assembly on all five switch arms, the ownership
partition, composition-equals-`calculate_cryo_loads` on all five arms, and the two
fixed points' Jacobians as exactly `0` or exactly `I`). MDA harness: **438 → 453
agreements, and nothing else in the report moved at all** — 34 disagreements (0 in
driven blocks), 3 unverifiable, 0 ungrounded, 21 errors, all byte-identical before and
after; owned variables walked 496 → 511 (10 real fields plus the 5 minted `^cond`
copies), unaccounted still 0; static switch kwargs 55 → 61, 0 mismatched. Graph: 149 →
155 nodes, 12 → 14 driven blocks (the two new `[node, ^problem[node]]` pairs), boundary
inputs 375 → 379.

**The boundary-input count went *up*, and that is the honest result.** Two edges closed
(`.heat_transport.helpow`, `.heat_transport.p_cryo_plant_electric_mw`) and six genuine
inputs appeared behind them: `.tfcoil.eff_tf_cryo` and `.tfcoil.temp_cp_coolant_inlet`
(read unconditionally, both ordinary `*_variables.py` inputs), and
`.tfcoil.p_cp_resistive`/`p_tf_leg_resistive`/`p_tf_joints_resistive`/`.fwbs.pnuc_cp_tf`,
which are read **only** on the `i_tf_sup == 2` cryogenic-aluminium branch this
configuration never takes. Same shape as `stellarator_fwbs_s4.py`'s 6-for-5 trade
(`boundary_inputs_audit.md` §7 item 3), where the same prediction was also wrong in the
same direction.

**What it bought, which is a gradient, not a value.** Every one of the 15 new outputs
agrees with PROCESS to the harness's tolerance, so no *value* moved. The Jacobian did:
`run_sand_harness.py` Stage B, `objf` row (`i_figure_merit = 6`, net electric power) and
`c16` row, before → after —

```
objf  x2 1.90e-01* -> 1.26e-02*   x3 2.36e-01* -> 1.03e-01*   x4 1.66e-01* -> 9.47e-02*
      x6 1.66e-01* -> 9.48e-02*   x56 1.97e-02* -> 1.97e-02*  x59 3.38e-01* -> 4.82e-02*
      x109 2.27e-01* -> 1.51e-01*
c16   x2 4.78e-01* -> 2.18e-02    x3 4.15e-02* -> 1.51e-02*   x4 4.96e-02* -> 3.09e-05
      x6 4.94e-02* -> 1.80e-09    x59 4.81e-01* -> 4.07e-03*  x109 4.95e-02* -> 9.07e-12
```

Four of `c16`'s six starred cells lost their star; no cell anywhere got worse, no new
row starred, still 0 non-finite cells. **Attributed by measurement, not inference**: a
run with `CryoLoads` + `TfCryoArea` registered but the two `*Step` nodes withheld
reproduces the *old* Jacobian bit-for-bit. So the gain comes entirely from
`.power.q*`/`.fwbs.qnuc` acquiring producers — with them frozen as boundary constants,
`helpow` (and therefore `p_cryo_plant_electric_mw`, and therefore net electric power)
had **zero** sensitivity to `coldmass`, `c_tf_turn`, `n_tf_coils`, `tfcryoarea`,
`p_tf_nuclear_heat_mw` and `t_plant_pulse_plasma_present`, all of which depend on
iteration variables. Fourth instance of the iteration-variable-4 defect class: every
value test passes and only a gradient sees it.

**Left open, and it is not in Stage A or B**: `run_sand_harness.py` Stage C2 went from
34 SQP iterations (converging to `conv = 1.4e-09`) to the full 100 without formally
converging, oscillating around `objf ≈ 1.2179`. The same withheld-`*Step` run pins this
to the two new fixed points as well. Its final point is *closer* to PROCESS on five of
the eight iteration variables (x2 1.85e-03 → 9.15e-05, x3 3.33e-03 → 7.21e-04, x56
1.61e-01 → 3.29e-02, x59 4.56e-02 → 8.37e-03), so the SQP is now solving a more
accurate problem rather than a broken one — but the five new design unknowns are
O(1e4) magnitudes entering an unscaled design vector alongside 1e20 and 3e-02, and
`sand.residual_condition_scales` scales only the *conditions*, not the unknowns. That
is a `sand.py` scaling question, not a modelling one, and it is left for whoever owns
that layer.
