---
kind: model-unit
status: draft
confidence: high
---

**Ported and registered (1 node, 11 owned fields).** `models/power/pf_coil_power.py` /
`tests/functional_process/models/power/test_pf_coil_power.py`:
`calculate_pf_coil_power_supplies` (tier-1) with three module-private helpers
mirroring PROCESS's own `_pf_loss_*` staticmethods. One cottax node,
`PfCoilPowerSupplies`, bound at `.power.pf_coil_power` — filled on a tokamak and `None`
on a stellarator.

## source

`process/models/power.py` (registry unit #14), chunk D of 4 — added 2026-08-30, after
chunks A (`tf_coil_power.md`), B (`thermal_cryo.md`) and C
(`electric_production.md`). The same split rationale: `power.py`'s methods span
largely-independent sub-domains of one file, and this is the PF-coil one.

- `Power.pfpwr` (`:300-604`) — the MVA, power and energy requirements of the PF coil
  power supply system. The output section (`:606-694`) is excluded from scope, as in
  every other chunk.
- `Power._pf_loss_storage_j` (`:99-120`), `_pf_loss_power_supply_j` (`:122-176`),
  `_pf_loss_busbar_j` (`:178-222`), `_pf_loss_interval_total_j` (`:224-299`) — the
  four helpers `pfpwr`'s dissipation block calls, one per loss mechanism (M. Kovari,
  "PF power supplies accounting 2", issue #972).

`Power.pfpwr`'s only callers are `Power.run` (`:54,81`), which a stellarator never
enters — `stellarator.py:114-186` calls `tfpwr`, `component_thermal_powers`,
`calculate_cryo_loads`, `acpow` and `plant_electric_production` directly and never
`run`. This is the one subsystem of `power.py` that is not device-agnostic.

## why it was ported: four measured missing producers

`boundary.unproduced_but_computed` on
`tests/regression/input_files/large_tokamak_nof.IN.DAT` listed four boundary `input`
entries that PROCESS *writes* every pipeline pass:

| VarPath | PROCESS computes | port had | consumer already ported |
|---|---|---|---|
| `.pf_power.srcktpm` | 1113.0075 kW | `0.0` | `electric_production.py::calculate_acpow`; `costs.py` acc2252; `objectives.py::objective_metric_4` |
| `.pf_power.ensxpfm` | 17038.228 MJ | `0.0` | `costs.py` acc2252 (`ucpfdr1`); `costs_2015.py:1055` |
| `.heat_transport.peakmva` | 134.98773 MVA | `0.0` | `calculate_acpow`; `costs.py:2139` (`ucpfps`) |
| `.pf_coil.p_pf_electric_supplies_mw` | 4.8813983 MW | `0.0` | `calculate_plant_electric_production`, `power_profiles_over_time` |

Every consumer was ported and reading the zero. Nothing failed, and that is the point:
Stage A and C2 seed boundary inputs from PROCESS's **converged** `DataStructure`, so
each of those nodes was handed the right value at the one place the harness looks
hardest. Only a cold start exposes the hole.

`_audit/cost_boundary_inputs.md` §6 had recorded seven of this node's eleven fields as
"`Power.pfpwr`, unported", settled as category (d), and §7 said outright that there was
"nothing to register". Both are now updated in place; the reasoning there was not wrong,
it was drawn from the *stellarator's* converged state, where all seven genuinely are
`0.0` because PROCESS never computes them either.

## data footprint

### reads (16)

| VarPath | classification | note |
|---|---|---|
| `.physics.rmajor` | explicit-arg | bus length, `pfbusl = 8*R + 140` |
| `.physics.p_plasma_ohmic_mw` | explicit-arg | the ohmic supply's own wall-plug loss |
| `.pf_coil.c_pf_coil_turn_peak_input` | explicit-arg | bus sizing, burn current, per-circuit MVA rating |
| `.pf_coil.rhopfbus`, `.pf_coil.rho_pf_coil` | explicit-arg | bus and conductor resistivity |
| `.pf_coil.r_pf_coil_middle` | explicit-arg | coil resistance |
| `.pf_coil.j_pf_coil_wp_peak` | explicit-arg | coil resistance |
| `.pf_coil.f_a_pf_coil_void` | explicit-arg | coil resistance |
| `.pf_coil.c_pf_cs_coils_peak_ma`, `.pf_coil.c_pf_cs_coil_pulse_end_ma` | explicit-arg | coil resistance; burn current per turn |
| `.pf_coil.n_pf_coil_turns` | explicit-arg | coil resistance and the resistive flat-top powers |
| `.pf_coil.c_pf_coil_turn` | explicit-arg | the `NGC2 x 6` waveform — the whole inductive and dissipative block |
| `.pf_coil.ind_pf_cs_plasma_mutual` | explicit-arg | the 8x8 mutual inductance matrix |
| `.pf_power.f_p_pf_energy_store_loss`, `.pf_power.f_p_pf_psu_loss` | explicit-arg | loss fractions |
| `.pf_coil.etapsu` | explicit-arg | ohmic-heating supply efficiency |
| `.times.t_plant_pulse_coil_precharge`, `..._plasma_current_ramp_up`, `..._fusion_ramp`, `..._burn`, `..._plasma_current_ramp_down` | explicit-arg | the five phases `PulseTimings.pf_active_cumulative` spans; the dwell is deliberately absent |

Five of these were **new** boundary inputs on the tokamak (`.pf_coil.etapsu`,
`.pf_coil.rho_pf_coil`, `.pf_coil.rhopfbus`, `.pf_power.f_p_pf_energy_store_loss`,
`.pf_power.f_p_pf_psu_loss`), which is why the tokamak boundary went 360 -> 361 in the
same commit that took four rows off it. All five are genuine `IN.DAT` inputs —
`computed_by_process` does not list any of them — which is the discrimination
`unproduced_but_computed` exists to make and the reason the input *count* alone is not
the measure.

### writes (11) — all owned

| VarPath | source line | note |
|---|---|---|
| `.pf_power.srcktpm` | `:352,411` | summed peak resistive power in the circuits (kW) |
| `.pf_power.poloidalpower` | `:509-515` | length 5 (one per interval), with PROCESS's `9.9e9` short-interval sentinel reproduced |
| `.pf_power.ensxpfm` | `:562` | peak stored poloidal energy (MJ) |
| `.pf_power.peakpoloidalpower` | `:564-566` | peak `|dE/dt|` (MW) |
| `.heat_transport.peakmva` | `:569` | `max(powpfr + powpfi, powpfr2)` |
| `.pf_power.vpfskv` | `:571` | literal `20.0` |
| `.pf_power.pfckts` | `:572-574` | `(n_circuits - 2) + 6`, i.e. 12 |
| `.pf_power.spfbusl` | `:575` | `pfbusl * pfckts` |
| `.pf_power.acptmax` | `:576,589-593` | mean of the circuit peak currents (kA) |
| `.pf_power.spsmva` | `:577,581-587` | summed power-supply MVA |
| `.pf_coil.p_pf_electric_supplies_mw` | `:598-604` | ohmic wall-plug loss + mean PF dissipation |

Seven of the eleven were not asked for by any missing producer; they are owned because
they come out of the same three blocks and a node that computed them and threw them away
would be a subset of its own source for no reason. `.pf_power.pfckts` in particular is
read by `costs.py:2167` alongside `srcktpm`.

`pfbuspwr` (`:353,410`) is **not** owned: it is accumulated and then read only by
`po.ovarre` in the output section. Dropped as a dead term, the same convention chunk A
used for `tfreacmw`.

## JAX notes

**No dynamic shapes and no `lax` control flow.** The four loop bounds `pfpwr` uses —
`n_pf_coil_groups` (+1 for the CS), `n_pf_coils_in_group[group]`,
`n_pf_cs_plasma_circuits`, `PulseTimings.n_pf_active_points_total` — are all
graph-assembly data on the reference topology, so every loop is an ordinary Python
`for` that unrolls at trace time. `models/pfcoil/__init__.py` records why that is
legitimate (`_audit/naming_convention.md` § "Switches are not ports"); this module
re-exports the same constants rather than re-deriving them, adding only
`COILS_IN_GROUP_WITH_CS` and `GROUP_CIRCUIT_INDEX` for the CS's own one-coil group
(`pfcoil.py:155`, `power.py:342-344`).

Two data-dependent branches become `jnp.where`:

- `if dt_pulse_phase_s <= 0: return 0` in `_pf_loss_interval_total_j` (`:271-272`).
- the `> 1 s` guards on `poloidalpower` (`:504-515`) and on the flat-top denominator
  (`:545-556`).

The two divisions in the second bullet are additionally guarded
(`jnp.where(dt == 0, 1, dt)`) so that the *discarded* arm cannot produce an `inf` for the
JVP to multiply by zero — the same `nan` mechanism
`solve_current_sharing_temperature`'s carry-collapse comment describes. The first bullet
needs no such guard: none of the three loss terms divides by `dt` (the busbar term
multiplies by it), so the untaken arm is finite on its own.

## agreement

Measured against a converged `large_tokamak_nof` run (`SingleRun` + one
`Evaluators.fcnvmc1`), all eleven outputs agree with PROCESS at `rtol=1e-12, atol=0` —
including the length-5 `poloidalpower` array, whose third entry is exactly `0.0` on both
sides because the waveform has the same currents at the ends of that interval.

## open questions

1. **`.costs.c2252` still has no producer**, and it is the account this node's outputs
   feed. `model_tree_design.md` §8 step 4c deleted Accounts 221.4, 222.2 and 225.2 from
   the port's cost tree because a stellarator has no reactor structure, no PF coils and
   no PF coil power conditioning. A tokamak has all three, and now has the inputs 225.2
   needs. `cost_boundary_inputs.md` category (d) carries the producer `file:line` for
   every one; restoring them is the obvious next step and is not taken here.
2. **`.pf_power.maxpoloidalpower`** (`pf_power_variables.py:44`, default `1000.0`) is a
   bound, not an output, and constraint 66 compares `peakpoloidalpower` against it. That
   constraint is not in the reference run's `icc`; if it is ever activated, this node is
   already its producer.
3. **The stellarator's `None`** is currently decided by `device is TokamakProcess` in
   `indat`, which is the only place in that factory that branches on the device rather
   than on a switch. It is honest and it is one line, but if a second such slot appears
   it wants a named predicate rather than a second copy of the test.

## 2026-08-31 -- `pfpwr`'s nested loops are array expressions

The module docstring's "every loop here is an ordinary Python `for` that unrolls at trace
time" was accurate and, measured, expensive: 6,828 jaxpr equations on `large_tokamak_nof`
from a routine whose loops run seven and eight times. The trip counts are small; the
nesting is not. `pfpwr`'s inductive block is four deep (`group`, `coil`, `circuit`,
`point`) and its innermost body is a single `+=`, which unrolled costs about nine
equations -- two or three scalar gathers, an arithmetic operation, an add -- of which one
is arithmetic. 336 iterations became ~2,800 equations; `_pf_loss_power_supply_j`'s
`5 x 7 x 8` became ~2,520.

Rewritten as arrays, with no change of algorithm:

- `_pf_loss_power_supply_j` -- `sum_j M_ij dI_j` is `coupling @ delta` and the outer sum
  over `i` is `jnp.sum`. `dI_j` does not depend on `i`, so PROCESS recomputes it once per
  pair; it is one vector subtraction here.
- `_pf_loss_busbar_j` -- one gather of the representative circuits, then `jnp.sum`.
- The inductive block -- `vpfij` is the outer product `coupling * d_current / delktim`,
  `inductxcurrent` is `coupling @ c_circuit`, and `vpfi`, `powpfi`, `poloidalenergy`,
  `powpfr`, `powpfr2` are four `jnp.sum`s over the coil or circuit axis.
- `spsmva` / `acptmax` -- `jnp.sum` over the circuit axis.

`pf_coil_power.py`'s share of `large_tokamak_nof` fell from 6,828 equations to 598.

### Reductions reassociated by vectorisation

**Every one of the sums above was an ordered Python accumulation and is now an XLA
reduce, so the last bits move.** Measured against the previous implementation over 20
random samples on both topologies, the worst disagreement in any of the eleven returned
values is **37 ulp** (`poloidalpower`, which is a difference of two stored energies
divided by an interval, so it amplifies whatever the energies disagree by); the rest are
at most 8 ulp. That is ~1e-14 relative -- three orders below the `1e-9` these values are
compared at anywhere in the harness, and the tier-1 contracts pass unchanged. A reader
diffing this unit against PROCESS should not expect bit-identity from it.
