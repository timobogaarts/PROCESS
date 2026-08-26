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
