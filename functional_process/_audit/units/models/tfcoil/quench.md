---
kind: model-unit
status: draft
confidence: high
---

**Not registered** — same note as `base.md`.

## source

`process/models/tfcoil/quench.py` (551 lines, all of it except
`_build_cumulative_quench_integral`) plus
`process/models/tfcoil/superconducting.py:1298-1379`
(`CICCSuperconductingTFCoil.quench_heat_protection_current_density`), which is the only
caller of `calculate_quench_protection_current_density` in the tokamak model chain.

`_build_cumulative_quench_integral` (`quench.py:403-471`) is **out of scope and dead for
the solver**: its only callers are `process/core/io/plot/summary.py:15205,15215`, i.e.
the plotting pass. Measured with `grep -rn "_build_cumulative_quench_integral" process/`.

## The headline finding

**`.tfcoil.v_tf_coil_dump_quench_kv` does not reach CoolProp.**
`_audit/tokamak_boundary.md` § `.tokamak.cicc_superconducting_tf_coil` schedules the
whole slot behind the unresolved CoolProp wrapping policy on the ground that
*"`.tfcoil.v_tf_coil_dump_quench_kv` above is on that chain"*. The `file:line` evidence
that it is not:

```python
# process/models/tfcoil/superconducting.py:1357-1360
v_tf_dump_voltage_peak = (
    2.0e0 * e_tf_coil_magnetic_stored / (t_tf_quench_dump * c_tf_turn)
)
# process/models/tfcoil/superconducting.py:1362-1377
j_tf_wp_quench_protection_max = (
    a_tf_turn_cable_space
    / a_tf_turn
    * quench.calculate_quench_protection_current_density(...)  # the only CoolProp path
)
return j_tf_wp_quench_protection_max, v_tf_dump_voltage_peak
```

and `superconducting.py:2793-2795` divides the second return by `1.0e3`. Three reads,
no material property. The two returns of one PROCESS function are two independent
computations; the boundary table attributed the CoolProp dependency to the function
rather than to the return, which is one level too coarse. **`v_tf_coil_dump_quench_kv`
is ported in full, with no seam and no approximation.**

`.tfcoil.j_tf_wp_quench_heat_max` is the return that does reach helium, and it is **not
on this slot's boundary list at all** — nothing in the currently-assembled tokamak graph
reads it.

## The CoolProp call surface, measured

Probe: monkeypatch `process.core.coolprop_interface.PropsSI`, call
`CICCSuperconductingTFCoil.quench_heat_protection_current_density` once with
`large_tokamak_eval.IN.DAT` values plus `tfcoil_variables.py` defaults
(`c_tf_turn = 85462.675` `IN.DAT:371`, `t_tf_quench_dump = 17.9728` `IN.DAT:376`,
`tftmp = 4.75` `IN.DAT:378`, `flu_tf_neutron_fast_max = 1e22` `IN.DAT:387`,
`f_a_tf_turn_cable_copper = 0.69` `:196`, `temp_tf_conductor_quench_max = 150.0` `:688`,
`rrr_tf_cu = 100.0` `:895`, `t_tf_quench_detection = 3.0` `:892`).

| | |
|---|---|
| fluid | `He`, only |
| output properties | `C` (isobaric specific heat, J/kg/K) x 75 and `D` (density, kg/m3) x 75 — nothing else |
| state variables | `("T", T, "P", P)`, always |
| pressure | `600000.0` Pa exactly, hardcoded at `quench.py:301,382,448`, source comment *"no plans to make input"* |
| temperatures | the 75 Gauss-Legendre nodes of `[tftmp, temp_tf_conductor_quench_max] = [4.75, 150.0]` K, i.e. `4.7868372` … `149.9631628` K |
| calls, first evaluation | **150** `PropsSI` |
| calls, every later evaluation | **0** |
| feeds | `ihe_integrand = cp_He * rho_He / nu_Cu` (`quench.py:334`), integrated over T, weighted by `f_a_cable_space_helium`, into `sqrt(factor * f_cu_cable * total_integral)` (`:551`) |

**The zero is the operative number.** `process/core/coolprop_interface.py` memoises each
property on its `(T, P, fluid)` tuple with `functools.cache`, and the quadrature grid
depends only on `tftmp` and `temp_tf_conductor_quench_max` — **neither is written by any
model** (`grep -rn "tftmp\s*=\|temp_tf_conductor_quench_max\s*=" process/` finds only the
read at `superconducting.py:2785` and the `data_structure` defaults) and **neither is an
iteration variable** (`process/core/solver/iteration_variables.py` lists neither; the
only quench unknown is `t_tf_superconductor_quench`, ID 56, which enters through
`1/(0.5*tau + t_detect)` **outside** the integrals, `quench.py:546`). Confirmed
empirically: perturbing `t_tf_quench_dump` and `b_tf_inboard_peak` and re-calling adds
**0** `PropsSI` calls.

So on this run the helium property surface is **a frozen table of 150 constants**, not a
function the optimiser moves along.

`tokamak_boundary.md` states *"450 CoolProp calls per `_call_models_once`"*. The measured
figure at the `PropsSI` level is 150, once. The two are reconcilable if the 450 counted
Python-level `FluidProperties` accesses (75 constructions + 75 `cp` + 75 `density` = 225)
across two model instances, or predates the `@cache`; either way the number that matters
for a wrapping decision is the one above.

### What this implies for the three options the dispatch brief named

- **No JAX-traceable fit is needed** for `large_tokamak_eval`. A 75-entry constant table
  is already exact and already costs one CoolProp round-trip for the whole run.
- **No interpolation table is needed** either, for the same reason — there is nothing to
  interpolate *between*, because the evaluation points never move.
- **No callback is needed**, and a `pure_callback` would be strictly worse: it would
  reintroduce a host round-trip inside a `jit` region to fetch numbers that are constant.
- A fit or a 1-D interpolation in `T` (at fixed `P`) becomes necessary **only if** a
  future configuration makes `tftmp` or `temp_tf_conductor_quench_max` an unknown or an
  output. `P` never needs to vary while `quench.py:301` is a literal.

The port is written so that either resolution drops straight in: the helium properties
enter `quench_integrands_at_temperature`, `quench_integrals` and
`calculate_quench_protection_current_density` as ordinary array arguments, and
`quench_quadrature_temperatures(temp_he_peak=…, temp_quench_max=…)` returns exactly the
list of states to ask CoolProp about.

## data footprint

`quench_heat_protection_current_density` (`superconducting.py:1298-1379`) is a
`@staticmethod` with no `data` access at all — its thirteen arguments are bound by `run`
at `:2775-2791`.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.e_tf_coil_magnetic_stored` | read | explicit-arg | *(live)* dump-voltage half |
| `.tfcoil.t_tf_superconductor_quench` | read | explicit-arg | *(live)* both halves; iteration variable 56 |
| `.tfcoil.c_tf_turn` | read | explicit-arg | *(live)* dump-voltage half |
| `.tfcoil.a_tf_turn_cable_space_no_void` | read | explicit-arg | current-density half |
| `.tfcoil.a_tf_turn` | read | explicit-arg | current-density half |
| `.tfcoil.f_a_tf_turn_cable_copper` | read | explicit-arg | current-density half |
| `.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling` | read | explicit-arg | current-density half |
| `.tfcoil.tftmp` | read | explicit-arg | current-density half; **sets the CoolProp grid** |
| `.tfcoil.temp_tf_conductor_quench_max` | read | explicit-arg | current-density half; **sets the CoolProp grid** |
| `.tfcoil.b_tf_inboard_peak_with_ripple` | read | explicit-arg | current-density half |
| `.tfcoil.rrr_tf_cu` | read | explicit-arg | current-density half |
| `.tfcoil.t_tf_quench_detection` | read | explicit-arg | current-density half |
| `.constraints.flu_tf_neutron_fast_max` | read | explicit-arg | current-density half |
| `.tfcoil.j_tf_wp_quench_heat_max` | write | explicit-arg | **not on the boundary**; needs the seam resolved |
| `.tfcoil.v_tf_coil_dump_quench_kv` | write | explicit-arg | **boundary read #10**; ported, no seam |

Three reads and one write for the voltage; twelve reads for the current density. Ported
as two nodes' worth of work, of which one node is written.

## proposed signature(s)

Shipped in `functional_process/models/tfcoil/quench.py`:

```python
copper_specific_heat_capacity(temperature)
copper_rrr_resistivity(temperature, rrr)
copper_irradiation_resistivity(fluence)
copper_magneto_resistivity(resistivity, field)
copper_electrical_resistivity(temperature, field, rrr, fluence)
nb3sn_specific_heat_capacity(temperature)

quench_quadrature_temperatures(*, temp_he_peak, temp_quench_max)        # the seam's states
quench_integrands_at_temperature(*, temperature, field, rrr, fluence,
                                 den_helium, cp_helium)                 # the seam
quench_integrals(*, temp_he_peak, temp_quench_max, field, rrr, fluence,
                 den_helium_at_nodes, cp_helium_at_nodes)
calculate_quench_protection_current_density(*, tau_discharge, b_peak, f_a_cable_copper,
        f_a_cable_space_helium, temp_he_peak, temp_quench_max, cu_rrr,
        t_quench_detection, fluence, den_helium_at_nodes, cp_helium_at_nodes)

tf_dump_voltage_peak(*, e_tf_coil_magnetic_stored, t_tf_quench_dump, c_tf_turn)
v_tf_coil_dump_quench_kv(*, e_tf_coil_magnetic_stored, t_tf_superconductor_quench,
        c_tf_turn)
```

## cottax node

**One**: `TfCoilDumpQuenchVoltage`, owning `.tfcoil.v_tf_coil_dump_quench_kv`.

**No node is declared for `.tfcoil.j_tf_wp_quench_heat_max`, deliberately.**
`den_helium_at_nodes`/`cp_helium_at_nodes` are not `DataStructure` fields and have no
`VarPath`; declaring the node would mean minting two, which is exactly the policy call
`traceability_policy.md` § "Non-traceable external calls" defers and which the dispatch
brief told this unit not to improvise. The pure function is written and diffed against
PROCESS; only the binding is open.

## tier signal

**Tier 1.** No solve anywhere: `_quench_integrals` is a fixed 75-point Gauss-Legendre
quadrature with no convergence test, and the dump voltage is one division. The
quadrature nodes and weights come from the same `np.polynomial.legendre.leggauss(75)`
call PROCESS makes at import (`quench.py:293`), so the two quadratures are identical by
construction rather than by agreement.

## switches touched

None. `quench.py` has no `i_*` branch. Two *data-dependent* branches exist and are
`jnp.where`d — see JAX-difficulty flags.

## calls into other models

`quench_heat_protection_current_density` calls
`process.models.tfcoil.quench.calculate_quench_protection_current_density`, which calls
`process.core.coolprop_interface.FluidProperties.of` — the only cross-module call, and
the seam.

## JAX-difficulty flags

- `non-traceable-external-call` — **the one in the tokamak scope**, and after
  measurement it is **workaround-known, not blocker**: `FluidProperties.of("He",
  temperature=T, pressure=6e5)` at 75 fixed temperatures, two properties each, constant
  for the run. Isolated behind a pure argument seam; see § "The CoolProp call surface".
- `needs-lax-cond-or-where` — **minor**. `_copper_magneto_resistivity`'s `if field >
  1e-2` (`quench.py:184`) branches on a traced field; ported as `jnp.where` with the
  `log10` argument guarded by substitution, since below the cut-off it would be zero or
  negative and would poison the untaken branch's tangent. The fluence clip
  (`quench.py:533-537`) is `jnp.clip` — a real derivative kink, kept, not dropped; inert
  at the reference run's `1e22`.
- `x ** p` at zero — **workaround-known**. `sqrt(factor * f_cu_cable * total_integral)`
  (`:551`) uses `safe_math.safe_sqrt`: `f_a_cable_copper == 0` makes the radicand zero
  with a finite value and an infinite derivative.
- Python loop over quadrature nodes (`quench.py:388-398`) — vectorised.

## defects found

- **D3.** `quench.py:18` sets `COPPER_DENSITY = 8960.0`; `process/core/constants.py:289`
  sets `DEN_COPPER = 8900.0`. The quench integrand uses the first, the TF mass chain
  (`superconducting.py:2042`) the second — a 0.67 % disagreement about the density of
  copper inside one coil model. Ported faithfully, each formula keeping its own constant.
- PROCESS's `logger.warning` on an out-of-range fluence (`:534`) and on a
  non-Nb3Sn material (`superconducting.py:2769`) have no value effect and are dropped.

## open questions

1. **What `VarPath`s should the helium properties get, if any?** Options, in the order
   this record would rank them given the measurement: (a) no `VarPath` at all — bind the
   constant table into the node the way `_RIPPLE_FIT_COEFFICIENTS` is bound, since the
   values are constants of the run; (b) mint
   `^he.tfcoil.den_quench_nodes`/`^he.tfcoil.cp_quench_nodes` and give them a producer
   node that is honest about being non-traceable; (c) a fit. **(a) is a real option only
   because `tftmp` and `temp_tf_conductor_quench_max` are inputs**, and the choice should
   be recorded as depending on that, so it is revisited if either ever becomes an
   unknown.
2. **The port's derivative with respect to `temp_he_peak`/`temp_quench_max` is not
   defined until (1) is answered**, because moving either moves the states the property
   table was built at. The harness contracts mark both `static_argnames` for exactly that
   reason (`tests/functional_process/models/tfcoil/test_quench.py`). Every other argument
   of the hotspot criterion **is** differentiated and does agree with PROCESS's own
   finite difference.

## 2026-08-27 — the CoolProp policy call, made; `j_tf_wp_quench_heat_max` bound

Open question 1 above is **closed in favour of its own option (a)**, and open question 2
is closed with it. `.tfcoil.j_tf_wp_quench_heat_max` — constraint 35's read, and one of
the seven rows `_audit/optimise_design.md` §11.5 measured on the TF side — now has a
node, `TfCoilQuenchHeatCurrentDensity`, and the CoolProp dependency is **inside** the
assembled tokamak graph for the first time.

### What was decided, and on what evidence

The decision is the one this record already ranked first, taken for the reason it already
gave rather than a new one:

> **The helium property table is a constant of the run.** The quadrature grid is a pure
> function of `.tfcoil.tftmp` and `.tfcoil.temp_tf_conductor_quench_max`; neither is
> written by any model (only site: the read at `superconducting.py:2785`) and neither is
> an iteration variable. So the 150 numbers cannot move during a solve, a scan point or
> a gradient sweep, and evaluating them once at graph-assembly time is exact rather than
> approximate.

Concretely: `helium_properties_at_quench_nodes` (in the port module) is called by
`indat._quench_helium_table` while the machine is being assembled, and the resulting two
75-tuples are carried on the occupant as `eqx.field(static=True)`, alongside the two
temperatures themselves. **Nothing is wrapped, fitted, interpolated or called back.** No
CoolProp call happens inside any traced or differentiated region, and the numbers the
port integrates are bit-for-bit the ones PROCESS's own body asks `PropsSI` for.

The precedent is `models/stellarator/preset_config.py`'s `StellaratorMachineConfig`,
whose `machine_config: tuple = eqx.field(static=True)` carries a parsed `stella_conf.json`
by exactly this argument — *"which machine is being designed is fixed for a whole solve,
so it is graph-assembly-time information"*. The two facts a table needs to be static are
the same two: it is per-machine, and nothing in the solve can move it.

The two options not taken, and why, in one line each:

- **`jax.pure_callback`** would reintroduce a host round-trip inside a `jit` region to
  fetch numbers that are constant. Strictly worse than a constant, at every call.
- **A fit or a 1-D interpolation in `T`** would introduce an approximation where an
  exact table is already available at zero marginal cost. It becomes the right answer
  only if a future configuration makes either temperature an unknown or a model output —
  which is precisely the condition the guard below tests for.

### The guard is the load-bearing half, and it is executable

A static table is correct exactly while the temperatures it was built at cannot move.
That is a claim about the run, so it is **checked at assembly time and not assumed**:
`indat._quench_helium_table` scans the run's `ixc` against `ITERATION_VARIABLES` and
raises `NotImplementedError` if either `tftmp` or `temp_tf_conductor_quench_max` is an
unknown, naming the resolution (give the properties a producer — options (b) or (c)) and
saying in so many words that relaxing the check is not it.

Without that, this would be the same defect shape as the `dcond[0]` bake
`superconducting.md` records for the mass slot: a value folded into a class where
`switch_audit` cannot see it. With it, the failure mode is a refused machine rather than
a stale table.

It is visible in the harness as well as refused in the factory. The two temperatures are
named for their `DataStructure` fields (`tftmp`, `temp_tf_conductor_quench_max`) rather
than for the pure function's parameters (`temp_he_peak`, `temp_quench_max`) **so that
`mda_harness.switch_audit` resolves them by name and value-checks the frozen numbers
against the converged run on every harness pass**. `mda_harness.STATIC_KWARG_KINDS` gains
a fifth kind for them, `FROZEN_INPUT` — a real `DataStructure` input the graph carries
statically — and the two arrays are `ASSEMBLY_PAYLOAD` with the reason recorded in
`STATIC_KWARGS_WITHOUT_BACKING_FIELD`. The report line moves from *"static switch kwargs
checked: 8 … not data-backed: 1, unresolved: 4"* with four entries under
**MUST BE EMPTY** to *"checked: 10 … not data-backed: 3, unresolved: 0"* with none.

### Open question 2, answered

The port's derivative with respect to the two temperatures is **structurally zero, and
that is now a stated property rather than an undefined one**: they are not reads of the
node, so the graph carries no edge from either and `jax` differentiates neither. The
harness case's `static_argnames` and the occupant's static fields now say the same thing
for the same reason, which is what the earlier note asked for. Every other argument of
the hotspot criterion is differentiated and agrees with PROCESS's own finite difference.

### The node

`TfCoilQuenchHeatCurrentDensity`, nine reads, one output:

| read | from |
|---|---|
| `a_tf_turn_cable_space_no_void`, `a_tf_turn`, `t_tf_superconductor_quench`, `b_tf_inboard_peak_with_ripple`, `f_a_tf_turn_cable_copper`, `rrr_tf_cu`, `t_tf_quench_detection` | `.tfcoil` |
| `f_a_tf_turn_cable_space_cooling` | `.superconducting_tfcoil` |
| `flu_tf_neutron_fast_max` | `.constraints` |

`j_tf_wp_quench_heat_max` (the pure function) is `superconducting.py:1362-1377` — one
cable-to-turn area ratio over `calculate_quench_protection_current_density`, which this
record's § "proposed signature(s)" already listed and which was already diffed against
PROCESS.

### Validation and acceptance

`TestJTfWpQuenchHeatMax` (`tests/functional_process/models/tfcoil/test_quench.py`) takes
PROCESS's `quench_heat_protection_current_density`'s **first** return, where its sibling
`TestTfDumpVoltagePeak` takes the second — between them the two cases now cover the whole
PROCESS function. The ported side calls the *shipped* `helium_properties_at_quench_nodes`,
not a test-local re-derivation, so the resolved seam is what is under test.
tfcoil case files: **275 passed** plain, **550** with `--fp-gradients`.

Measured on `large_tokamak_eval.IN.DAT`:

| | before | after |
|---|---|---|
| §11.5 missing-producer rows | 15 | **11** (`j_tf_wp_quench_heat_max` among the four closed) |
| SAND cold probe (C3) | 2 of 24 conditions non-finite: `c33` `inf`, `c35` `inf` | 1 of 26: `c65` `nan` (see `superconducting.md`) |
| Stage B row for `c35` | `0.00e+00` | `0.00e+00` (unchanged, and now on a produced value) |
| warm MDA harness | 611 agree / 16 disagree / 20 errors | 624 / 16 / 20 — the disagreement list is byte-identical |

**`c35` was `inf` at the cold seed and is finite now**, which is the whole point of the
row: the constraint divided by a boundary constant that PROCESS's own solve moves off
zero and the port had frozen at the `DataStructure` default.

### 2026-08-31 — the wrapper is vendored, and the import is gone

`helium_properties_at_quench_nodes` no longer imports `process`. The wrapper it needs is
now `functional_process/fluid_properties.py`, a **byte-for-byte copy** of
`process/core/coolprop_interface.py` (checked as such by
`test_fluid_properties.py::test_vendored_source_is_verbatim`, not just asserted here),
sitting beside `vocabulary/` rather than inside it because a CoolProp wrapper is
behaviour, not a declaration — §23.2's rule was written for constants, enums and tables
and is applied to it anyway.

**Nothing about this unit's numerics changed, and that is the claim being made.** Both
copies call the same `PropsSI` in the same installed `CoolProp` 8.0.0, so
`tests/functional_process/test_fluid_properties.py` asserts `==`, not `approx`, over the
range this chain actually queries: helium at `QUENCH_HELIUM_PRESSURE_PA = 6.0e5` Pa, all
four `(tftmp, temp_tf_conductor_quench_max)` intervals the seven regression inputs
produce — `(4.2, 4.5, 4.75, 20.0)` × `150.0`, i.e. **300 quadrature states**, the shipped
table itself — plus a denser 61-point 4–200 K sweep and all nine properties, not only the
`D`/`C` pair this unit uses. The CoolProp policy call recorded above (option (a), the
static table, with `indat._quench_helium_table`'s refusal as its guard) is **untouched**:
this cut an import, not a seam. `CLAUDE.md`'s "CoolProp is not JAX-traceable" stays open.

**The lazy import stays, for a new reason as well as the old one.** It was deferred so
that importing this module needed no `process`; `import CoolProp` costs **~3 s**
(measured), and only a tokamak assembly ever wants a helium table, so it stays deferred
now that `process` is not the issue. `test_process_free_import.py::
test_importing_the_model_layer_does_not_load_coolprop` pins that: sweeping all 117
modules must leave `CoolProp` out of `sys.modules`.
