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

**Not resolved as a graph-wiring decision here** (out of one audit record's scope,
per `_audit/naming_convention.md`'s treatment of run-config-dependent ownership) --
`calculate_component_thermal_powers` takes `delta_eta` as an ordinary `Input` (the
entering value) and returns a freshly-computed `delta_eta` as an `Output`, on the
same `VarPath`. Wiring this into a real graph needs either a `Blocking`/`FixedPoint`
around this self-loop, or an explicit decision to accept one round of staleness
(matching PROCESS's own current behaviour) -- flagged for whoever does that wiring,
not decided here. Only reachable when `i_thermal_electric_conversion` selects
`CCFE_HCPB_VALUE_WITH_DIVERTOR` or `STEAM_RANKINE_CYCLE`; for the other three values
`delta_eta` is written but never read back by `plant_thermal_efficiency`, so the loop
is inert.

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
`Input`, since `cottax`'s `NodalDeclaration.__check_init__` requires every `__call__`
parameter to carry an `Input(...)` default, so a static config value cannot be an
ordinary parameter at all on this surface.

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
3. **The `delta_eta` self-loop's actual closure behaviour is not verified here** --
   this record documents the stale-read shape and the pure port's in/out signature
   faithfully reproduces it, but whether it actually converges over PROCESS's
   `Caller.call_models` re-run loop (and to what) is not checked, matching
   `power_at_ignition_point`'s precedent (documented, not solved, until a `Blocking`/
   `FixedPoint` wiring pass exists to drive it for real).
