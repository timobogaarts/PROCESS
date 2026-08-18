---
kind: model-unit
status: draft
confidence: high
---

**Ported (3 units).** `power_C_electric_production.py` /
`test_power_C_electric_production.py`: `calculate_acpow`, `power_profiles_over_time`,
`calculate_plant_electric_production` -- all tier-1, all tests passing (value +
gradient, `--fp-gradients`).

## source

`process/models/power.py` (registry unit #14), chunk C of 3 (see
`power_A_tf_coil_power.md`/`power_B_thermal_cryo.md` for the other two).

- `Power.acpow` (696-813) -- AC power requirements. Output section (755-812)
  excluded from scope.
- `Power.power_profiles_over_time` (2632-2825, already a `@staticmethod`) -- spreads
  steady-state powers over a fixed pulse-phase profile.
- `Power.plant_electric_production` (1631-1772) -- composes the above with the
  overall plant electric balance. `output_power_profiles_over_time` (2827+, purely
  `po.o*` calls after `power_profiles_over_time` returns) is out of scope, same as
  `output_plant_electric_powers` below.
- `Power.output_plant_electric_powers` (1430-1630) -- **audit-only, not ported**:
  confirmed by grep (`grep -n "self\.data\.\w*\.\w* *=" ` over the method's line
  range returns nothing) to be a pure reporting shell, no computation, same
  convention as `vacuum.py`'s `_write_to_outfile`/`_vacuum_simple_output`.

## data footprint

Full parameter lists are in each ported function's docstring
(`power_C_electric_production.py`).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.power.p_cp_coolant_pump_elec_mw` | write, conditional on `itart == 1 and i_tf_sup == 0` | conditional-ownership, but empirically inert in the else arm | `0.0` when not owned -- no other producer confirmed by grep |
| `.heat_transport.p_plant_electric_gross_mw`, `.power.p_turbine_loss_mw`, `.heat_transport.p_plant_electric_recirc_mw`, `.heat_transport.p_plant_electric_net_mw`, `.heat_transport.f_p_plant_electric_recirc` | write, conditional on `.costs.ireactor == 1` | **conditional-ownership-by-run-config** | `.costs.ireactor` belongs to unit #18 (`costs.py`, not yet ported per the registry, but read here as an ordinary `Input`, not a call into unit #18) -- see below |
| `.heat_transport.p_plant_electric_gross_mw`, `.heat_transport.p_plant_electric_net_mw` | read, unconditionally, by `power_profiles_over_time` | **implicit-io** (downstream of the above) | `plant_electric_production` calls `power_profiles_over_time` regardless of `ireactor`, so if `ireactor != 1` these two are whatever they entered the call with (stale/default), not values this call computed -- ported faithfully, not resolved, same shape as `power_B_thermal_cryo.md`'s `.power.delta_eta` |

**`ireactor`'s pass-through is not resolved here as a graph-wiring decision** --
same policy as chunk B's `delta_eta` self-loop and `p_fw_blkt_coolant_pump_mw`
conditional ownership: `calculate_plant_electric_production` takes the five fields as
ordinary `Input`s (entering values) and only overwrites them internally when
`ireactor == 1`, exactly mirroring PROCESS's own conditional write. Whether a real
graph should drive this with a `Blocking`/`FixedPoint`, or simply route unit #18's
`ireactor` output into this node once it exists, is a later wiring question.

No `implicit-io-via-callee` or `redundant-duplicate-write` in this chunk.

## `power_profiles_over_time`'s fixed-length time axis

`Power.power_profiles_over_time` takes a `pulse_timings: PulseTimings` argument
(`process/models/pulse.py`) rather than reading `self.data` directly -- it was
already a clean `@staticmethod` before this port touched it. `PulseTimings` itself is
a plain dataclass wrapping the six `.times.t_plant_pulse_*` fields (confirmed: no
`self.data` access anywhere in `pulse.py`, just `@property` arithmetic over its own
six stored floats) -- not reused as an object in the port; its two properties this
function needs (`total_pulse_cumulative`, a 7-tuple of cumulative phase-boundary
times; `n_pulse_points_total = len(total_pulse_cumulative)`, always exactly `7`
regardless of the six durations' *values*) are inlined directly.
`n_pulse_points_total` being a true compile-time constant (never data-dependent in
*length*, only in the boundary *values*) is what makes every array in this chunk a
static shape `(7,)` -- there is no dynamic-shape JAX difficulty here at all, unlike
several other chunks in this codebase.

Verified by hand against `Power.power_profiles_over_time` directly (constructing a
`PulseTimings`, comparing all 13 outputs) before writing the harness -- exact
agreement, including through the diagnostic `logger.error` path (dropped from the
port, side-effect only, see the function's docstring).

## Real findings (documented, not fixed)

None found in this chunk (unlike chunks A and B, which each found a genuine
PROCESS bug). `acpow`/`power_profiles_over_time`/`plant_electric_production` are
comparatively clean straight-line arithmetic with no enum-name typos, broken log
calls, or singular derivatives encountered while fuzzing.

## proposed signature(s)

All three actually written in `power_C_electric_production.py`: `calculate_acpow`,
`power_profiles_over_time`, `calculate_plant_electric_production` -- see each
function's own docstring for its full parameter list and return tuple.

## cottax node

`Acpow`, `PowerProfilesOverTime`, `PlantElectricProduction` -- all
`ExplicitFunction`, actually written in `power_C_electric_production.py`. Switches
(`i_pf_energy_storage_source`, `itart`, `i_tf_sup`, `ireactor`,
`i_blkt_dual_coolant`, `i_p_coolant_pumping`) are `eqx.field(static=True)`, same
convention as chunks A/B.

## tier signal

All three **tier 1**: no internal iteration, no calls into any other model.

## switches touched

- `.pf_power.i_pf_energy_storage_source` (`== 2`) -- static, two mutually exclusive
  additive terms (`peakmva` vs. `fmgdmw`) in `acpow`.
- `.physics.itart` (0/1) + `.tfcoil.i_tf_sup` (0/1/2, `== 0` here) -- static,
  jointly gate `p_cp_coolant_pump_elec_mw`'s ownership in
  `plant_electric_production`. Note this is a *third*, independent use of
  `i_tf_sup` in this registry unit alongside chunk A's `tfpwr` dispatch and chunk
  B's `cryo`/`calculate_cryo_loads` branches -- each with its own reads-set, same
  "multiple independent uses of one switch" pattern chunk A's real finding #2
  already flagged.
- `.costs.ireactor` (`== 1`) -- static, conditional-ownership (see data footprint
  above). Belongs to unit #18 (`costs.py`), read here as a plain field.
- `.fwbs.i_blkt_dual_coolant` (`> 0`) + `.fwbs.i_p_coolant_pumping`
  (`== MECHANICAL`) -- static, jointly select `plant_electric_production`'s
  alternate gross-electric-power formula (splits `p_plant_primary_heat_mw` into a
  liquid-breeder-heat term at `etath_liq` and the rest at `eta_turbine`, vs. one
  term at `eta_turbine`). Same fields chunk B's `component_thermal_powers` also
  reads, consistent usage here.

None of these are yet in `_audit/core/solver/switches.md` (out of this fork's edit
boundary) -- flagging for the coordinating session, same as chunks A/B.

## calls into other models

None. `plant_electric_production` reads `.costs.ireactor` (unit #18, not yet ported)
and `.buildings.a_plant_floor_effective` (unit #15, already ported per the concurrent
consolidation pass -- confirmed by reading `functional_process/models/buildings.py`
directly, which declares `a_plant_floor_effective` as one of its outputs) as ordinary
data fields, not by calling either unit's methods.

## JAX-difficulty flags

None. Every branch in this chunk is resolved with static Python control flow (per
the switches above), and `power_profiles_over_time`'s fixed-length arrays (see
above) avoid any dynamic-shape concern. No singular derivatives or unguarded
`jnp.where` branches found while fuzzing (contrast chunk A's `res_tf_leg == 0.0`).

## open questions

None outstanding for this chunk -- both switch-heavy composed functions
(`calculate_component_thermal_powers` in chunk B and `calculate_plant_electric_
production` here) were fuzzed across every combination of their static switches
listed above, and every sample (including the two legacy `baseline_2018_IN.DAT`
points) passed value and gradient agreement.
